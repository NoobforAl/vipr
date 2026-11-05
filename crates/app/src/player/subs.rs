//! The subtitle tracks of a result folder: the original and any `<stem>.<lang>.translation.json`.
//!
//! All of this text is untrusted. `clean_separators` runs on it here, once, before any widget sees
//! it, and the display-only `isolate_rtl` form is kept apart from the plain text.

use engine::lang::{clean_separators, isolate_rtl, language};
use engine::project::{ProjectPaths, TranslationFile};
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
}

impl SubCue {
    fn new(id: u64, start: f64, end: f64, raw: &str) -> SubCue {
        let text = clean_separators(raw).trim().to_string();
        let overlay = isolate_rtl(&text);
        SubCue {
            id,
            start,
            end,
            text,
            overlay,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    pub label: String,
    /// Sorted by start, with nothing to show left out.
    pub cues: Vec<SubCue>,
}

impl Track {
    fn new(label: String, mut cues: Vec<SubCue>) -> Track {
        cues.retain(|c| !c.text.is_empty() && c.end > c.start);
        cues.sort_by(|a, b| a.start.total_cmp(&b.start));
        Track { label, cues }
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
}

fn original_track(t: &Transcript) -> Track {
    let cues = build_cues(t, &SubtitleOptions::default())
        .iter()
        .enumerate()
        .map(|(i, c)| SubCue::new(i as u64 + 1, c.start, c.end, &c.text))
        .collect();
    Track::new("Original".to_string(), cues)
}

fn translation_track(lang: &str, file: &TranslationFile) -> Track {
    let label = language(lang).map_or_else(|| clean_separators(lang), |l| l.name.to_string());
    let cues = file
        .cues
        .iter()
        .map(|c| SubCue::new(c.id, c.start, c.end, &c.text))
        .collect();
    Track::new(label, cues)
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
    /// For the user: what couldn't be read.
    pub problems: Vec<String>,
}

/// Touches the disk, so call it off the UI thread.
pub fn load_tracks(paths: &ProjectPaths) -> TrackSet {
    let mut set = TrackSet::default();
    match Transcript::load(&paths.transcript()) {
        Ok(t) => set.tracks.push(original_track(&t)),
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
