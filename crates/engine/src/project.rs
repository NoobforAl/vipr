//! The shared contract: every on-disk sidecar of a result folder, and the JSON the app exchanges
//! with `llm-worker --mode translate|compare`.
//!
//! Agents code against these types in parallel, so change them only through the integration owner.
//! All times are seconds from the start of the media. Every file carries `version` (now 1). Fields
//! added later must use `serde(default)` so older files keep loading, and unknown status or flag
//! values read as `Unknown` instead of failing, so an older app can still open a newer file.
//!
//! Save and load sidecars with [`crate::fsio::save_json`] and [`crate::fsio::load_json`]. Only
//! [`ProjectMeta`] has its own helpers, since each owner decides how its own file fails.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::Result;
use crate::fsio;
use crate::profiles::{GlossaryTerm, Profile};

/// Written to every sidecar and worker file.
pub const VERSION: u32 = 1;

pub(crate) fn version() -> u32 {
    VERSION
}

// ------------------------------------------------------------------------- locations

/// Where one recording's results live: `<dir>/<stem>.*`.
///
/// `stem` is the real one the pipeline chose, which may carry a ` (ext)` suffix (see
/// `pipeline::result_stem`), so never rebuild it from the media path.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProjectPaths {
    pub dir: PathBuf,
    pub stem: String,
}

impl ProjectPaths {
    pub fn new(dir: impl Into<PathBuf>, stem: impl Into<String>) -> ProjectPaths {
        ProjectPaths {
            dir: dir.into(),
            stem: stem.into(),
        }
    }

    fn file(&self, suffix: &str) -> PathBuf {
        self.dir.join(format!("{}.{suffix}", self.stem))
    }

    /// `<stem>.json`: the canonical `Transcript`.
    pub fn transcript(&self) -> PathBuf {
        self.file("json")
    }
    /// `<stem>.project.json`: [`ProjectMeta`].
    pub fn project(&self) -> PathBuf {
        self.file("project.json")
    }
    /// `<stem>.glossary.txt`: the merged glossary passed through `--glossary`.
    pub fn glossary(&self) -> PathBuf {
        self.file("glossary.txt")
    }
    /// `<stem>.<lang>.translation.json`: [`TranslationFile`].
    pub fn translation(&self, lang: &str) -> PathBuf {
        self.file(&format!("{lang}.translation.json"))
    }
    /// `<stem>.<lang>.<ext>`: translated output, where `ext` is `txt`, `srt` or `vtt`.
    pub fn translated(&self, lang: &str, ext: &str) -> PathBuf {
        self.file(&format!("{lang}.{ext}"))
    }
    /// `<stem>.<lang>.ckpt.jsonl`: translation checkpoint, one [`TranslationCue`] per line.
    pub fn checkpoint(&self, lang: &str) -> PathBuf {
        self.file(&format!("{lang}.ckpt.jsonl"))
    }
    /// `<stem>.cuts.json`: `cuts::CutsFile`.
    pub fn cuts(&self) -> PathBuf {
        self.file("cuts.json")
    }
    /// `<stem>.chapters.json`: [`ChaptersFile`].
    pub fn chapters(&self) -> PathBuf {
        self.file("chapters.json")
    }
    /// `<stem>.marks.json`: `marks::MarksFile`.
    pub fn marks(&self) -> PathBuf {
        self.file("marks.json")
    }
    /// `<stem>.import-<slug>.json`: [`ImportFile`].
    pub fn import(&self, slug: &str) -> PathBuf {
        self.file(&format!("import-{slug}.json"))
    }
    /// `<stem>.compare-<slug>.json`: [`CompareFile`].
    pub fn compare(&self, slug: &str) -> PathBuf {
        self.file(&format!("compare-{slug}.json"))
    }
    /// `<stem>.reviewed.<ext>`: saved compare result, where `ext` is `srt`, `vtt` or `txt`.
    pub fn reviewed(&self, ext: &str) -> PathBuf {
        self.file(&format!("reviewed.{ext}"))
    }
}

// ------------------------------------------------------------------------- <stem>.project.json

/// Links a result folder back to its media, since `Transcript.source` is only a file name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectMeta {
    #[serde(default = "version")]
    pub version: u32,
    pub media_path: PathBuf,
    pub stem: String,
}

impl ProjectMeta {
    pub fn new(media_path: impl Into<PathBuf>, stem: impl Into<String>) -> ProjectMeta {
        ProjectMeta {
            version: VERSION,
            media_path: media_path.into(),
            stem: stem.into(),
        }
    }

    pub fn load(path: &Path) -> Result<ProjectMeta> {
        fsio::load_json(path)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        fsio::save_json(path, self)
    }
}

// ------------------------------------------------------------------------- shared pieces

/// One timed piece of text. `id` is unique within its list, 1-based and stable; for ViPr's own
/// subtitles it is the cue's position in `subtitles::build_cues`, plus one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimedText {
    pub id: u64,
    pub start: f64,
    pub end: f64,
    /// Lines joined with `\n`.
    pub text: String,
}

// ------------------------------------------------------------------------- translation

/// Whether a cue really holds a translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CueStatus {
    /// `text` is a translation that passed the guards.
    Translated,
    /// `text` is the source language: every retry failed. Saved SRT shows `[untranslated] …`.
    #[default]
    Untranslated,
    /// The user edited `text` by hand. Never overwritten by a re-run.
    Edited,
    /// Written by a newer version.
    #[serde(other)]
    Unknown,
}

/// A soft warning on a cue: shown to the user, never a reason to reject it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CueFlag {
    NumberMismatch,
    GlossaryMismatch,
    /// Written by a newer version.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslationCue {
    pub id: u64,
    pub start: f64,
    pub end: f64,
    /// The text it was translated from.
    pub source: String,
    /// The translation, or a copy of `source` when `status` is `Untranslated`. Empty on a translated
    /// cue when its sentence's translation had too few words to reach it, which leaves that text on a
    /// neighbour. Never display markers such as U+200F or `[untranslated]`: writers add those.
    pub text: String,
    #[serde(default)]
    pub status: CueStatus,
    #[serde(default)]
    pub flags: Vec<CueFlag>,
}

/// `<stem>.<lang>.translation.json`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslationFile {
    #[serde(default = "version")]
    pub version: u32,
    /// Target language code, as in `lang::LANGUAGES`.
    pub target: String,
    /// [`source_hash`] of the cues this was made from. A mismatch means the source has changed
    /// since, and the file is stale.
    pub source_hash: String,
    /// Registry id of the model.
    #[serde(default)]
    pub model: String,
    /// The profile as it was when this ran, so later edits to it don't change this file's meaning.
    #[serde(default)]
    pub profile: Option<Profile>,
    pub cues: Vec<TranslationCue>,
}

/// Fingerprint of the source cues: changes when any id, time or text changes. Times are rounded to
/// milliseconds so a JSON round trip can't alter the hash.
pub fn source_hash(cues: &[TimedText]) -> String {
    let mut h = Sha256::new();
    for c in cues {
        let line = format!(
            "{}|{}|{}|{}\n",
            c.id,
            (c.start * 1000.0).round() as i64,
            (c.end * 1000.0).round() as i64,
            c.text
        );
        h.update(line.as_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

// ------------------------------------------------------------------------- import

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubFormat {
    Srt,
    Vtt,
    Ass,
    /// Written by a newer version.
    #[serde(other)]
    Unknown,
}

/// A part of the subtitle file that could not be used. Parsing never fails because of one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportWarning {
    /// 1-based line in the decoded text, when known.
    #[serde(default)]
    pub line: Option<u32>,
    pub message: String,
}

/// `<stem>.import-<slug>.json`. Text is untrusted: see `sanitize` and `lang::clean_separators`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportFile {
    #[serde(default = "version")]
    pub version: u32,
    /// Filesystem-safe, unique within the result folder.
    pub slug: String,
    /// File name it came from. `None` when pasted.
    #[serde(default)]
    pub source_name: Option<String>,
    pub format: SubFormat,
    /// `encoding_rs` label, such as `UTF-8` or `windows-1256`.
    pub encoding: String,
    /// Seconds added to the times in `cues` to match the media. `cues` keep the file's own times.
    #[serde(default)]
    pub offset: f64,
    #[serde(default)]
    pub warnings: Vec<ImportWarning>,
    pub cues: Vec<TimedText>,
}

// ------------------------------------------------------------------------- compare

/// Which text a compare column holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CompareSource {
    /// ViPr's own transcript.
    Transcript,
    /// A translation, by language code.
    Translation { lang: String },
    /// An import, by slug.
    Import { slug: String },
}

/// What the AI picked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AiChoice {
    A,
    B,
    /// Use `AiPick::merged`.
    Merged,
}

/// How a row's AI pick came about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AiStatus {
    /// Not judged yet.
    #[default]
    Pending,
    /// A and B are the same text. `choice` is `A`; the LLM was not asked.
    Identical,
    /// Only one side has text. `choice` is that side; the LLM was not asked.
    OneSided,
    /// The LLM picked, and the pick passed `compare::check_merged` where that applies.
    Picked,
    /// The LLM's answer was unusable or its merge was rejected. `choice` is `None`.
    Failed,
    /// Written by a newer version.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AiPick {
    #[serde(default)]
    pub choice: Option<AiChoice>,
    /// Set only when `choice` is `Merged`.
    #[serde(default)]
    pub merged: Option<String>,
    /// At most 120 characters of untrusted model output: show as plain text, labelled "AI note".
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub status: AiStatus,
}

/// What the user decided. Overrides the AI pick.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "choice", rename_all = "snake_case")]
pub enum UserPick {
    A,
    B,
    /// Accept the AI's merged text.
    Merged,
    /// Typed by hand.
    Edited {
        text: String,
    },
}

/// One moment in the compare table. A side is `None` when it has no text here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompareRow {
    pub id: u64,
    pub start: f64,
    pub end: f64,
    #[serde(default)]
    pub a: Option<String>,
    #[serde(default)]
    pub b: Option<String>,
    #[serde(default)]
    pub ai: AiPick,
    #[serde(default)]
    pub user: Option<UserPick>,
}

impl CompareRow {
    /// The text this row saves as: the user's pick, else the AI's. For A or B this always copies
    /// the row's own text and never the model's. `None` means there is nothing to save yet.
    pub fn resolved_text(&self) -> Option<&str> {
        match &self.user {
            Some(UserPick::A) => self.a.as_deref(),
            Some(UserPick::B) => self.b.as_deref(),
            Some(UserPick::Merged) => self.ai.merged.as_deref(),
            Some(UserPick::Edited { text }) => Some(text),
            None => match self.ai.choice? {
                AiChoice::A => self.a.as_deref(),
                AiChoice::B => self.b.as_deref(),
                AiChoice::Merged => self.ai.merged.as_deref(),
            },
        }
    }
}

/// `<stem>.compare-<slug>.json`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompareFile {
    #[serde(default = "version")]
    pub version: u32,
    pub slug: String,
    pub a: CompareSource,
    pub b: CompareSource,
    /// Registry id of the model.
    #[serde(default)]
    pub model: String,
    /// Seconds applied to B's times to line it up with A (see `subalign::best_offset`).
    #[serde(default)]
    pub offset: f64,
    pub rows: Vec<CompareRow>,
}

// ------------------------------------------------------------------------- llm-worker: translate

/// One topic of a recording. It lasts until the next chapter starts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chapter {
    pub start: f64,
    pub title: String,
}

/// `llm-worker --mode chapters --input <this> --out <ChaptersOutput>`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChaptersInput {
    #[serde(default = "version")]
    pub version: u32,
    /// Length of the recording in seconds.
    pub duration: f64,
    pub lines: Vec<TimedText>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ChaptersStats {
    #[serde(default)]
    pub windows: usize,
    /// Windows that gave nothing usable. Their topics are missing from the result.
    #[serde(default)]
    pub failed_windows: usize,
    /// Chapters proposed before the rules and the merge.
    #[serde(default)]
    pub candidates: usize,
    #[serde(default)]
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChaptersOutput {
    #[serde(default = "version")]
    pub version: u32,
    pub chapters: Vec<Chapter>,
    #[serde(default)]
    pub stats: ChaptersStats,
}

/// What the player reads, next to the recording's other results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChaptersFile {
    #[serde(default = "version")]
    pub version: u32,
    /// [`source_hash`] of the lines these came from, to tell when the transcript has moved on.
    pub source_hash: String,
    #[serde(default)]
    pub model: String,
    pub chapters: Vec<Chapter>,
}

/// `llm-worker --mode translate --input <this> --out <TranslateOutput>`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslateInput {
    #[serde(default = "version")]
    pub version: u32,
    /// Language code of the cues. Empty = unknown.
    #[serde(default)]
    pub source_lang: String,
    /// Target language code.
    pub target: String,
    pub cues: Vec<TimedText>,
    /// Added to the user message as background. Fixed guards still apply.
    #[serde(default)]
    pub profile_prompt: Option<String>,
    /// The user's terms. One with a target must be translated that way, and one without stays exactly
    /// as written. Mismatches are flags only.
    #[serde(default)]
    pub glossary: Vec<GlossaryTerm>,
    /// Names and jargon from the shared glossary file, which also holds ordinary words such as "merge".
    /// These only help the model and are never required.
    #[serde(default)]
    pub hints: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct TranslateStats {
    #[serde(default)]
    pub translated: usize,
    #[serde(default)]
    pub untranslated: usize,
    #[serde(default)]
    pub flagged: usize,
    /// Cues taken from the checkpoint instead of the model.
    #[serde(default)]
    pub resumed: usize,
    #[serde(default)]
    pub elapsed_seconds: f64,
}

/// Same cues and order as the input. The app wraps this into a [`TranslationFile`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslateOutput {
    #[serde(default = "version")]
    pub version: u32,
    pub target: String,
    pub cues: Vec<TranslationCue>,
    #[serde(default)]
    pub stats: TranslateStats,
}

// ------------------------------------------------------------------------- llm-worker: compare

/// A row the LLM has to judge. The app sends only rows with text on both sides that differ
/// (`compare::needs_llm`); every field here is untrusted and must go into the prompt JSON-encoded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompareInRow {
    pub id: u64,
    pub a: String,
    pub b: String,
}

/// `llm-worker --mode compare --input <this> --out <CompareOutput>`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompareInput {
    #[serde(default = "version")]
    pub version: u32,
    pub rows: Vec<CompareInRow>,
    #[serde(default)]
    pub profile_prompt: Option<String>,
    #[serde(default)]
    pub glossary: Vec<GlossaryTerm>,
    /// See [`TranslateInput::hints`].
    #[serde(default)]
    pub hints: Vec<String>,
}

/// One judged row. Choices name the real sides: the worker undoes its id-parity swap before writing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompareOutRow {
    pub id: u64,
    pub ai: AiPick,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CompareStats {
    #[serde(default)]
    pub picked: usize,
    #[serde(default)]
    pub failed: usize,
    #[serde(default)]
    pub elapsed_seconds: f64,
}

/// One entry per input row. The app merges these into the [`CompareFile`] by id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompareOutput {
    #[serde(default = "version")]
    pub version: u32,
    pub rows: Vec<CompareOutRow>,
    #[serde(default)]
    pub stats: CompareStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(a: Option<&str>, b: Option<&str>) -> CompareRow {
        CompareRow {
            id: 1,
            start: 0.0,
            end: 1.0,
            a: a.map(str::to_string),
            b: b.map(str::to_string),
            ai: AiPick::default(),
            user: None,
        }
    }

    #[test]
    fn paths_keep_the_real_stem() {
        let p = ProjectPaths::new("out", "meeting (mp3)");
        let name = |path: PathBuf| path.file_name().unwrap().to_string_lossy().into_owned();
        assert_eq!(name(p.transcript()), "meeting (mp3).json");
        assert_eq!(name(p.project()), "meeting (mp3).project.json");
        assert_eq!(name(p.glossary()), "meeting (mp3).glossary.txt");
        assert_eq!(
            name(p.translation("fa")),
            "meeting (mp3).fa.translation.json"
        );
        assert_eq!(name(p.translated("fa", "srt")), "meeting (mp3).fa.srt");
        assert_eq!(name(p.checkpoint("fa")), "meeting (mp3).fa.ckpt.jsonl");
        assert_eq!(name(p.marks()), "meeting (mp3).marks.json");
        assert_eq!(name(p.import("x")), "meeting (mp3).import-x.json");
        assert_eq!(name(p.compare("x")), "meeting (mp3).compare-x.json");
        assert_eq!(name(p.reviewed("vtt")), "meeting (mp3).reviewed.vtt");
    }

    #[test]
    fn user_pick_beats_the_ai_and_a_or_b_copy_our_own_text() {
        let mut r = row(Some("left"), Some("right"));
        assert_eq!(r.resolved_text(), None);
        r.ai = AiPick {
            choice: Some(AiChoice::B),
            merged: Some("model text".into()),
            reason: String::new(),
            status: AiStatus::Picked,
        };
        assert_eq!(r.resolved_text(), Some("right"));
        r.user = Some(UserPick::A);
        assert_eq!(r.resolved_text(), Some("left"));
        r.user = Some(UserPick::Merged);
        assert_eq!(r.resolved_text(), Some("model text"));
        r.user = Some(UserPick::Edited { text: "own".into() });
        assert_eq!(r.resolved_text(), Some("own"));
    }

    #[test]
    fn hash_changes_with_the_source_but_not_with_tiny_float_noise() {
        let cue = |text: &str, start: f64| TimedText {
            id: 1,
            start,
            end: 2.0,
            text: text.into(),
        };
        let base = source_hash(&[cue("a", 1.0)]);
        assert_eq!(base, source_hash(&[cue("a", 1.000_000_1)]));
        assert_ne!(base, source_hash(&[cue("b", 1.0)]));
        assert_ne!(base, source_hash(&[cue("a", 1.5)]));
        assert_eq!(base.len(), 64);
    }

    #[test]
    fn minimal_and_future_files_still_load() {
        let t: TranslationFile = serde_json::from_str(
            r#"{"target":"fa","source_hash":"x","cues":[
                {"id":1,"start":0,"end":1,"source":"a","text":"b","status":"from_the_future",
                 "flags":["number_mismatch","new_flag"],"added_later":true}]}"#,
        )
        .unwrap();
        assert_eq!(t.version, VERSION);
        assert_eq!(t.cues[0].status, CueStatus::Unknown);
        assert_eq!(t.cues[0].flags, [CueFlag::NumberMismatch, CueFlag::Unknown]);

        let c: CompareFile = serde_json::from_str(
            r#"{"slug":"s","a":{"kind":"transcript"},"b":{"kind":"import","slug":"i"},
                "rows":[{"id":1,"start":0,"end":1,"a":"x"}]}"#,
        )
        .unwrap();
        assert_eq!(c.rows[0].ai.status, AiStatus::Pending);
        assert_eq!(c.rows[0].b, None);
        assert_eq!(c.b, CompareSource::Import { slug: "i".into() });
    }

    #[test]
    fn every_file_round_trips_with_a_version() {
        let compare = CompareFile {
            version: VERSION,
            slug: "s".into(),
            a: CompareSource::Transcript,
            b: CompareSource::Translation { lang: "fa".into() },
            model: "m".into(),
            offset: 1.5,
            rows: vec![CompareRow {
                user: Some(UserPick::Edited { text: "t".into() }),
                ..row(Some("a"), None)
            }],
        };
        let json = serde_json::to_string(&compare).unwrap();
        assert!(json.contains("\"version\":1"));
        assert_eq!(serde_json::from_str::<CompareFile>(&json).unwrap(), compare);

        let import = ImportFile {
            version: VERSION,
            slug: "s".into(),
            source_name: None,
            format: SubFormat::Ass,
            encoding: "windows-1256".into(),
            offset: -2.0,
            warnings: vec![ImportWarning {
                line: Some(3),
                message: "bad time".into(),
            }],
            cues: vec![],
        };
        let json = serde_json::to_string(&import).unwrap();
        assert_eq!(serde_json::from_str::<ImportFile>(&json).unwrap(), import);
    }

    #[test]
    fn project_meta_saves_and_loads() {
        let dir = std::env::temp_dir().join(format!("vipr-project-{}", std::process::id()));
        let path = ProjectPaths::new(&dir, "a").project();
        let meta = ProjectMeta::new("C:/media/a.mp4", "a");
        meta.save(&path).unwrap();
        let back = ProjectMeta::load(&path).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(back, meta);
    }
}
