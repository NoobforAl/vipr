//! The subtitle tracks of a result folder: the original and any `<stem>.<lang>.translation.json`.
//!
//! All of this text is untrusted. `clean_separators` runs on it here, once, before any widget sees
//! it, and the display-only `isolate_rtl` form is kept apart from the plain text.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use engine::chapters::transcript_lines;
use engine::diff::{DiffKind, DiffSpan, diff_words};
use engine::lang::{clean_separators, isolate_rtl, language};
use engine::project::{Chapter, ChaptersFile, ProjectPaths, TranslationFile, source_hash};
use engine::subtitles::{SubtitleOptions, build_cues};
use engine::transcript::Transcript;

#[derive(Debug, Clone, PartialEq)]
pub struct SubCue {
    pub id: u64,
    pub start: f64,
    pub end: f64,
    /// Safe for a widget.
    pub text: String,
    /// `text` for the video overlay. Display only: never save this.
    pub overlay: String,
    /// How many lines `text` needs in the timeline panel. Worked out once, since the panel asks for
    /// every cue on every redraw.
    pub panel_lines: u16,
    /// The AI cleanup changed the utterance this cue belongs to.
    pub ai_edited: bool,
    /// What it changed. Only on the utterance's first cue: an utterance splits into cues differently before and
    /// after the cleanup, so only the whole utterance can be compared.
    pub ai_diff: Option<Arc<AiDiff>>,
}

#[derive(Debug, PartialEq)]
pub struct AiDiff {
    /// Safe for a widget.
    pub spans: Vec<DiffSpan>,
    /// How many lines `spans` need in the timeline panel.
    pub panel_lines: u16,
}

/// What the AI cleanup changed, `None` when no word changed.
fn ai_diff(raw: &str, text: &str) -> Option<AiDiff> {
    let spans = diff_words(&clean_separators(raw), &clean_separators(text));
    if spans.iter().all(|s| s.kind == DiffKind::Same) {
        return None;
    }
    let chars: usize = spans.iter().map(|s| s.text.chars().count() + 1).sum();
    let panel_lines = chars.div_ceil(PANEL_LINE_CHARS).max(1) as u16;
    Some(AiDiff { spans, panel_lines })
}

/// Characters on one line of the timeline panel, on the low side so text is rarely cut off.
const PANEL_LINE_CHARS: usize = 32;

impl SubCue {
    fn new(id: u64, start: f64, end: f64, raw: &str) -> SubCue {
        let text = clean_separators(raw).trim().to_string();
        let overlay = isolate_rtl(&text);
        let panel_lines = text.chars().count().div_ceil(PANEL_LINE_CHARS).max(1) as u16;
        SubCue {
            id,
            start,
            end,
            text,
            overlay,
            panel_lines,
            ai_edited: false,
            ai_diff: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    pub label: String,
    /// The translation's language code. `None` for the original.
    pub code: Option<String>,
    /// Sorted by start, with nothing to show left out.
    pub cues: Vec<SubCue>,
}

impl Track {
    fn new(label: String, mut cues: Vec<SubCue>) -> Track {
        cues.retain(|c| !c.text.is_empty() && c.end > c.start);
        cues.sort_by(|a, b| a.start.total_cmp(&b.start));
        Track {
            label,
            code: None,
            cues,
        }
    }

    /// The cue on screen at `time`.
    pub fn cue_at(&self, time: f64) -> Option<&SubCue> {
        self.cue_index_at(time).map(|i| &self.cues[i])
    }

    /// When cues overlap, the one that started last wins.
    pub fn cue_index_at(&self, time: f64) -> Option<usize> {
        /// How far back to look for a long cue under shorter ones.
        const OVERLAP_LOOKBACK: usize = 8;
        let after = self.cues.partition_point(|c| c.start <= time);
        (after.saturating_sub(OVERLAP_LOOKBACK)..after)
            .rev()
            .find(|&i| time < self.cues[i].end)
    }

    /// Where "previous line" goes: the start of this line, or of the one before if this line has
    /// only just begun, so that pressing twice keeps moving back.
    pub fn prev_start(&self, time: f64) -> Option<f64> {
        /// Further into a line than this, "previous" restarts it.
        const RESTART_AFTER: f64 = 1.5;
        let after = self.cues.partition_point(|c| c.start <= time);
        let current = self.cues.get(after.checked_sub(1)?)?;
        if time - current.start > RESTART_AFTER || after < 2 {
            Some(current.start)
        } else {
            Some(self.cues[after - 2].start)
        }
    }

    /// Where "next line" goes.
    pub fn next_start(&self, time: f64) -> Option<f64> {
        let after = self.cues.partition_point(|c| c.start <= time);
        self.cues.get(after).map(|c| c.start)
    }
}

/// The label of the transcript's own track.
const ORIGINAL: &str = "Original";

/// Each utterance's share of a track, joined into one text. A cue belongs to the last utterance that started by then. An
/// utterance no cue reached gets an empty text. `starts` must be sorted, as a transcript's are.
fn texts_per_utterance(starts: &[f64], cues: &[SubCue]) -> Vec<String> {
    let mut texts = vec![String::new(); starts.len()];
    for cue in cues {
        let at = starts
            .partition_point(|s| *s <= cue.start)
            .saturating_sub(1);
        if let Some(text) = texts.get_mut(at) {
            if !text.is_empty() {
                text.push(' ');
            }
            let flat = cue.text.replace('\n', " ");
            // the mark a subtitle uses for a new speaker: here each speaker already has a row
            text.push_str(flat.strip_prefix("- ").unwrap_or(&flat));
        }
    }
    texts
}

/// One saved translation, ready to show next to a transcript.
#[derive(Debug, Clone, PartialEq)]
pub struct TranslatedLines {
    /// The language's code, as in the translation's file name.
    pub code: String,
    /// The language's name.
    pub label: String,
    /// Safe for a widget. One for each utterance of the transcript.
    pub texts: Vec<String>,
}

/// The languages a result has translations in, as `(code, name)`. Reads the folder only, so it is quick.
pub fn translation_languages(paths: &ProjectPaths) -> Vec<(String, String)> {
    let names = std::fs::read_dir(&paths.dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok());
    translation_langs(names, &paths.stem)
        .into_iter()
        .map(|code| {
            let name =
                language(&code).map_or_else(|| clean_separators(&code), |l| l.name.to_string());
            (code, name)
        })
        .collect()
}

/// All saved translations of a result, for a transcript whose utterances start at `starts`. Touches the disk, so call it
/// off the UI thread.
pub fn translated_lines(paths: &ProjectPaths, starts: &[f64]) -> Vec<TranslatedLines> {
    let names = std::fs::read_dir(&paths.dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok());
    translation_langs(names, &paths.stem)
        .into_iter()
        .filter_map(|code| {
            let file: TranslationFile = engine::fsio::load_json(&paths.translation(&code)).ok()?;
            let track = translation_track(&code, &file);
            Some(TranslatedLines {
                texts: texts_per_utterance(starts, &track.cues),
                label: track.label,
                code,
            })
        })
        .collect()
}

fn original_track(t: &Transcript) -> Track {
    let mut diffs: HashMap<usize, Arc<AiDiff>> = HashMap::new();
    if t.refined {
        for u in &t.utterances {
            if let Some(diff) = ai_diff(&u.raw_text, &u.text) {
                diffs.insert(u.id, Arc::new(diff));
            }
        }
    }
    let mut shown: HashSet<usize> = HashSet::new();
    let cues = build_cues(t, &SubtitleOptions::default())
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut cue = SubCue::new(i as u64 + 1, c.start, c.end, &c.text);
            if let Some(diff) = diffs.get(&c.utterance) {
                cue.ai_edited = true;
                // a cue with nothing to show is left out of the track, so it can't carry the difference
                if !cue.text.is_empty() && cue.end > cue.start && shown.insert(c.utterance) {
                    cue.ai_diff = Some(diff.clone());
                }
            }
            cue
        })
        .collect();
    Track::new(ORIGINAL.to_string(), cues)
}

fn translation_track(lang: &str, file: &TranslationFile) -> Track {
    let label = language(lang).map_or_else(|| clean_separators(lang), |l| l.name.to_string());
    let cues = file
        .cues
        .iter()
        .map(|c| SubCue::new(c.id, c.start, c.end, &c.text))
        .collect();
    Track {
        code: Some(lang.to_string()),
        ..Track::new(label, cues)
    }
}

/// The language codes of the translation files among `names`, sorted. The stem may hold dots; a
/// language code doesn't.
fn translation_langs(names: impl Iterator<Item = String>, stem: &str) -> Vec<String> {
    let prefix = format!("{stem}.");
    let mut langs: Vec<String> = names
        .filter_map(|name| {
            let lang = name
                .strip_prefix(&prefix)?
                .strip_suffix(".translation.json")?;
            (!lang.is_empty() && !lang.contains('.')).then(|| lang.to_string())
        })
        .collect();
    langs.sort();
    langs
}

#[derive(Debug, Clone, Default)]
pub struct TrackSet {
    pub tracks: Vec<Track>,
    /// Safe for a widget. Empty until they've been worked out.
    pub chapters: Vec<Chapter>,
    /// The transcript has changed since the chapters were worked out.
    pub chapters_stale: bool,
    /// For the user: what couldn't be read.
    pub problems: Vec<String>,
}

fn load_chapters(paths: &ProjectPaths, t: &Transcript, set: &mut TrackSet) {
    let path = paths.chapters();
    if !path.is_file() {
        return;
    }
    let Ok(file) = engine::fsio::load_json::<ChaptersFile>(&path) else {
        set.problems
            .push("The chapters couldn't be read.".to_string());
        return;
    };
    set.chapters_stale = file.source_hash != source_hash(&transcript_lines(t));
    set.chapters = file
        .chapters
        .into_iter()
        .map(|c| Chapter {
            start: c.start,
            title: clean_separators(&c.title).replace('\n', " "),
        })
        .collect();
}

/// Touches the disk, so call it off the UI thread.
pub fn load_tracks(paths: &ProjectPaths) -> TrackSet {
    let mut set = TrackSet::default();
    match Transcript::load(&paths.transcript()) {
        Ok(t) => {
            set.tracks.push(original_track(&t));
            load_chapters(paths, &t, &mut set);
        }
        Err(_) => set
            .problems
            .push("The original subtitles couldn't be read.".to_string()),
    }
    let names = std::fs::read_dir(&paths.dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok());
    for lang in translation_langs(names, &paths.stem) {
        match engine::fsio::load_json::<TranslationFile>(&paths.translation(&lang)) {
            Ok(file) => set.tracks.push(translation_track(&lang, &file)),
            Err(_) => set
                .problems
                .push(format!("The {lang} translation couldn't be read.")),
        }
    }
    set
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::project::{CueStatus, TranslationCue};

    fn cue(id: u64, start: f64, end: f64, text: &str) -> SubCue {
        SubCue::new(id, start, end, text)
    }

    fn track(cues: Vec<SubCue>) -> Track {
        Track::new("T".to_string(), cues)
    }

    #[test]
    fn cue_lookup_respects_times_and_gaps() {
        let t = track(vec![cue(1, 1.0, 2.0, "a"), cue(2, 3.0, 4.0, "b")]);
        assert_eq!(t.cue_at(0.5), None);
        assert_eq!(t.cue_at(1.0).unwrap().id, 1);
        assert_eq!(t.cue_at(2.0), None, "the end is exclusive");
        assert_eq!(t.cue_at(2.5), None);
        assert_eq!(t.cue_at(3.5).unwrap().id, 2);
        assert_eq!(t.cue_at(99.0), None);
        assert_eq!(track(vec![]).cue_at(1.0), None);
    }

    #[test]
    fn previous_restarts_the_line_then_moves_back_and_next_moves_on() {
        let t = track(vec![
            cue(1, 10.0, 20.0, "a"),
            cue(2, 30.0, 40.0, "b"),
            cue(3, 50.0, 60.0, "c"),
        ]);
        assert_eq!(t.prev_start(5.0), None, "before the first line");
        assert_eq!(t.prev_start(35.0), Some(30.0), "well into a line");
        assert_eq!(t.prev_start(30.5), Some(10.0), "only just started");
        assert_eq!(t.prev_start(10.2), Some(10.0), "nothing before the first");
        assert_eq!(t.prev_start(45.0), Some(30.0), "between lines");
        assert_eq!(t.next_start(5.0), Some(10.0));
        assert_eq!(t.next_start(30.0), Some(50.0));
        assert_eq!(t.next_start(55.0), None);
        assert_eq!(track(vec![]).prev_start(1.0), None);
    }

    fn utterance(id: usize, start: f64, raw: &str, text: &str) -> engine::transcript::Utterance {
        engine::transcript::Utterance {
            id,
            start,
            end: start + 20.0,
            speaker: 0,
            raw_text: raw.to_string(),
            text: text.to_string(),
            words: vec![],
        }
    }

    fn transcript(refined: bool, utterances: Vec<engine::transcript::Utterance>) -> Transcript {
        let mut t: Transcript = serde_json::from_value(serde_json::json!({
            "version": 1, "source": "", "language": "", "audio_seconds": 0.0, "speakers": [], "utterances": []
        }))
        .unwrap();
        t.refined = refined;
        t.utterances = utterances;
        t
    }

    #[test]
    fn the_difference_sits_on_the_first_cue_of_a_changed_utterance_only() {
        let long = "The first sentence is here and it is long enough. ".repeat(4);
        let t = transcript(
            true,
            vec![
                utterance(1, 0.0, &long, &long),
                utterance(2, 30.0, &format!("helo {long}"), &format!("hello {long}")),
            ],
        );
        let track = original_track(&t);
        let second: Vec<&SubCue> = track.cues.iter().filter(|c| c.start >= 30.0).collect();
        assert!(second.len() > 1, "the utterance has to split for this test");
        assert!(second.iter().all(|c| c.ai_edited));
        assert!(second[0].ai_diff.is_some());
        assert!(second[1..].iter().all(|c| c.ai_diff.is_none()));
        let mut first = track.cues.iter().filter(|c| c.start < 30.0);
        assert!(first.all(|c| !c.ai_edited && c.ai_diff.is_none()));
    }

    #[test]
    fn nothing_is_marked_before_the_cleanup_ran_or_when_only_spacing_changed() {
        let not_refined = transcript(false, vec![utterance(1, 0.0, "helo", "hello")]);
        assert!(!original_track(&not_refined).cues[0].ai_edited);
        let spacing = transcript(true, vec![utterance(1, 0.0, "a  b", "a b")]);
        assert!(!original_track(&spacing).cues[0].ai_edited);
    }

    #[test]
    fn chapters_load_clean_and_say_when_the_transcript_moved_on() {
        let dir = std::env::temp_dir().join(format!("vipr-chapters-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let paths = ProjectPaths::new(&dir, "clip");
        let mut t = transcript(true, vec![utterance(1, 0.0, "hello", "hello")]);
        t.save(&paths.transcript()).unwrap();
        let file = ChaptersFile {
            version: 1,
            source_hash: source_hash(&transcript_lines(&t)),
            model: String::new(),
            chapters: vec![Chapter {
                start: 0.0,
                title: "a\u{2029}b".to_string(),
            }],
        };
        engine::fsio::save_json(&paths.chapters(), &file).unwrap();

        let set = load_tracks(&paths);
        assert_eq!(set.chapters[0].title, "a b");
        assert!(!set.chapters_stale);

        t.utterances[0].text = "changed".to_string();
        t.save(&paths.transcript()).unwrap();
        assert!(load_tracks(&paths).chapters_stale);

        std::fs::write(paths.chapters(), "{ not json").unwrap();
        let broken = load_tracks(&paths);
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(broken.chapters.is_empty());
        assert_eq!(broken.problems.len(), 1);
    }

    #[test]
    fn a_translation_is_shared_out_to_the_utterances_by_time() {
        let cues = [
            cue(1, 5.0, 6.0, "eins"),
            cue(2, 7.0, 8.0, "zwei\ndrei"),
            cue(3, 20.0, 21.0, "vier"),
        ];
        // nothing reaches the utterance at 10 s, and a cue before the first start joins the first
        assert_eq!(
            texts_per_utterance(&[5.0, 10.0, 15.0], &cues),
            ["eins zwei drei", "", "vier"]
        );
        assert_eq!(
            texts_per_utterance(&[6.0], &[cue(1, 5.0, 5.5, "früh")]),
            ["früh"]
        );
        assert_eq!(texts_per_utterance(&[], &cues), Vec::<String>::new());
    }

    #[test]
    fn the_subtitle_mark_for_a_new_speaker_is_left_out() {
        let cues = [cue(1, 0.0, 1.0, "- بله."), cue(2, 1.0, 2.0, "a - b")];
        assert_eq!(texts_per_utterance(&[0.0], &cues), ["بله. a - b"]);
    }

    #[test]
    fn only_translations_are_offered_with_the_lines_of_each() {
        let dir = std::env::temp_dir().join(format!("vipr-lines-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let paths = ProjectPaths::new(&dir, "clip");
        transcript(true, vec![utterance(1, 0.0, "hello", "hello")])
            .save(&paths.transcript())
            .unwrap();
        let file = TranslationFile {
            version: 1,
            target: "fa".to_string(),
            source_hash: String::new(),
            model: String::new(),
            profile: None,
            cues: vec![TranslationCue {
                id: 1,
                start: 0.0,
                end: 1.0,
                source: "hello".to_string(),
                text: "سلام".to_string(),
                status: CueStatus::Translated,
                flags: vec![],
            }],
        };
        engine::fsio::save_json(&paths.translation("fa"), &file).unwrap();

        let lines = translated_lines(&paths, &[0.0]);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].label, "Persian");
        assert_eq!(lines[0].code, "fa");
        assert_eq!(lines[0].texts, ["سلام"]);
    }

    #[test]
    fn the_difference_is_cleaned_for_a_widget() {
        let diff = ai_diff("abc\u{2029}سلام", "abd\u{2029}سلام").unwrap();
        assert!(diff.spans.iter().all(|s| !s.text.contains('\u{2029}')));
        assert_eq!(diff.panel_lines, 1);
        assert_eq!(ai_diff("same", "same"), None);
    }

    #[test]
    fn panel_lines_grow_with_the_text() {
        assert_eq!(cue(1, 0.0, 1.0, "short").panel_lines, 1);
        assert_eq!(cue(2, 0.0, 1.0, &"x".repeat(65)).panel_lines, 3);
        // counted in characters, not bytes
        assert_eq!(cue(3, 0.0, 1.0, &"س".repeat(30)).panel_lines, 1);
    }

    #[test]
    fn overlapping_cues_prefer_the_later_start_then_fall_back() {
        let t = track(vec![cue(1, 0.0, 10.0, "long"), cue(2, 2.0, 3.0, "short")]);
        assert_eq!(t.cue_at(2.5).unwrap().id, 2);
        assert_eq!(t.cue_at(5.0).unwrap().id, 1);
    }

    #[test]
    fn unsorted_empty_and_backwards_cues_are_handled() {
        let t = track(vec![
            cue(2, 5.0, 6.0, "b"),
            cue(3, 7.0, 8.0, "  \n "),
            cue(4, 9.0, 8.0, "backwards"),
            cue(1, 1.0, 2.0, "a"),
        ]);
        let ids: Vec<u64> = t.cues.iter().map(|c| c.id).collect();
        assert_eq!(ids, [1, 2]);
    }

    #[test]
    fn a_translated_cue_with_no_text_shows_nothing() {
        let piece = |id, text: &str| TranslationCue {
            id,
            start: id as f64,
            end: id as f64 + 1.0,
            source: "source".to_string(),
            text: text.to_string(),
            status: CueStatus::Translated,
            flags: vec![],
        };
        let file = TranslationFile {
            version: 1,
            target: "fa".to_string(),
            source_hash: String::new(),
            model: String::new(),
            profile: None,
            cues: vec![piece(1, ""), piece(2, "جالبه")],
        };
        let t = translation_track("fa", &file);
        assert_eq!(t.cues.len(), 1);
        assert_eq!(t.cue_at(1.5), None);
        assert_eq!(t.cue_at(2.5).unwrap().id, 2);
    }

    #[test]
    fn translation_files_are_found_by_exact_stem() {
        let names = [
            "talk.fa.translation.json",
            "talk.pt-BR.translation.json",
            "talk.json",
            "talk.fa.srt",
            "talk.fa.ckpt.jsonl",
            "talk.v2.fa.translation.json", // belongs to the stem "talk.v2"
            "other.de.translation.json",
            "talk..translation.json",
        ]
        .map(String::from);
        assert_eq!(
            translation_langs(names.clone().into_iter(), "talk"),
            ["fa", "pt-BR"]
        );
        assert_eq!(translation_langs(names.into_iter(), "talk.v2"), ["fa"]);
    }

    #[test]
    fn translated_text_is_cleaned_and_isolated_only_for_the_overlay() {
        // U+2029 between text of differing direction panics in cosmic-text
        let c = cue(1, 0.0, 1.0, "abc\u{2029}سلام");
        assert_eq!(c.text, "abc\nسلام");
        assert!(!c.text.contains('\u{2067}'));
        assert!(c.overlay.contains('\u{2067}'), "{:?}", c.overlay);
        assert!(c.overlay.starts_with("abc\n"));
        // left to right text is unchanged
        assert_eq!(cue(2, 0.0, 1.0, "hello").overlay, "hello");
    }

    #[test]
    fn translation_track_keeps_cue_ids_and_names_the_language() {
        let file = TranslationFile {
            version: 1,
            target: "whatever the file claims".to_string(),
            source_hash: String::new(),
            model: String::new(),
            profile: None,
            cues: vec![TranslationCue {
                id: 7,
                start: 1.0,
                end: 2.0,
                source: "hi".to_string(),
                text: "سلام".to_string(),
                status: CueStatus::Translated,
                flags: vec![],
            }],
        };
        let t = translation_track("fa", &file);
        assert_eq!(t.label, "Persian");
        assert_eq!(t.cues[0].id, 7);
        assert_eq!(translation_track("xx-unknown", &file).label, "xx-unknown");
    }

    #[test]
    fn load_reads_what_exists_and_reports_the_rest() {
        let dir = std::env::temp_dir().join(format!("vipr-subs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let paths = ProjectPaths::new(&dir, "clip");
        std::fs::write(paths.translation("de"), "{ not json").unwrap();
        let file = TranslationFile {
            version: 1,
            target: "fa".to_string(),
            source_hash: String::new(),
            model: String::new(),
            profile: None,
            cues: vec![],
        };
        engine::fsio::save_json(&paths.translation("fa"), &file).unwrap();

        let set = load_tracks(&paths);
        std::fs::remove_dir_all(&dir).unwrap();
        // no transcript, one bad translation, one good
        assert_eq!(set.tracks.len(), 1);
        assert_eq!(set.tracks[0].label, "Persian");
        assert_eq!(set.problems.len(), 2);
    }
}
