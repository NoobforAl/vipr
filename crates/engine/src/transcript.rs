//! Canonical transcript: the one JSON file per recording that every export is rendered from.
//!
//! `AsrResult` (from asr-worker) + speaker turns (from diar-worker) -> `Transcript` (utterances with
//! words and speakers). The LLM step may replace utterance text / speaker later; `.txt`, `.srt` and
//! `.vtt` are rendered from the result, so exporting again never needs Whisper to run again.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{Error, Result, io_err};

// ---------------------------------------------------------------- inputs from the workers

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrWord {
    pub text: String,
    pub start: f64,
    pub end: f64,
    #[serde(default)]
    pub prob: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    #[serde(default)]
    pub no_speech_prob: f32,
    #[serde(default)]
    pub words: Vec<AsrWord>,
}

/// Result file written by `asr-worker`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrResult {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub language: String,
    pub audio_seconds: f64,
    #[serde(default)]
    pub elapsed_seconds: f64,
    pub segments: Vec<AsrSegment>,
}

impl AsrResult {
    pub fn load(path: &Path) -> Result<AsrResult> {
        let text = std::fs::read_to_string(path).map_err(io_err(path))?;
        serde_json::from_str(&text).map_err(|e| Error::Registry {
            path: path.display().to_string(),
            msg: e.to_string(),
        })
    }
}

/// One speaker turn from `diar-worker` (0-based speaker, sorted by start).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SpeakerTurn {
    pub start: f64,
    pub end: f64,
    pub speaker: usize,
}

// ---------------------------------------------------------------- canonical transcript

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Utterance {
    /// 1-based, stable; the LLM stage refers to utterances by this id.
    pub id: usize,
    pub start: f64,
    pub end: f64,
    pub speaker: usize,
    /// What Whisper heard (words joined).
    pub raw_text: String,
    /// Final text: equals `raw_text` until the LLM cleanup replaces it.
    pub text: String,
    pub words: Vec<Word>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Speaker {
    pub id: usize,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Transcript {
    pub version: u32,
    /// Source file name, e.g. `2026-10-21 17-51-13.mp4`.
    pub source: String,
    pub language: String,
    pub audio_seconds: f64,
    #[serde(default)]
    pub asr_model: Option<String>,
    pub speakers: Vec<Speaker>,
    pub utterances: Vec<Utterance>,
    /// True once the LLM cleanup changed text / speakers.
    #[serde(default)]
    pub refined: bool,
}

/// Longest silence inside one speaker's turn before it is split into a new utterance.
const MAX_GAP_IN_UTTERANCE: f64 = 2.0;
/// An utterance this long (in words) is split at the next sentence end.
const SOFT_MAX_WORDS: usize = 90;

fn ends_sentence(s: &str) -> bool {
    s.trim_end_matches(['"', '\'', ')', '\u{201d}'])
        .ends_with(['.', '?', '!'])
}

/// Speaker whose turn overlaps `[s, e]` most; the nearest turn when none overlaps. `turns` sorted by start.
fn speaker_for(turns: &[SpeakerTurn], s: f64, e: f64) -> usize {
    let first = turns.partition_point(|t| t.end < s);
    let mut best: Option<(f64, usize)> = None;
    for t in &turns[first..] {
        if t.start > e {
            break;
        }
        let overlap = e.min(t.end) - s.max(t.start);
        if best.is_none_or(|(o, _)| overlap > o) {
            best = Some((overlap, t.speaker));
        }
    }
    if let Some((_, sp)) = best {
        return sp;
    }
    // inside a gap between turns: take the closer neighbour
    let mid = (s + e) / 2.0;
    let before = turns[..first].last();
    let after = turns.get(first);
    match (before, after) {
        (Some(b), Some(a)) => {
            if mid - b.end <= a.start - mid {
                b.speaker
            } else {
                a.speaker
            }
        }
        (Some(b), None) => b.speaker,
        (None, Some(a)) => a.speaker,
        (None, None) => 0,
    }
}

/// Diarization boundaries are a little jittery: a turn often starts or ends a word or two too early or late,
/// which splits one sentence between two speakers ("Yeah, I" | "mean, how does it work?"). When the speaker
/// changes in the middle of a sentence, give the whole sentence to the speaker who talks most of it.
fn smooth_speakers(words: &mut [(Word, usize)]) {
    let n = words.len();
    let mut i = 1;
    while i < n {
        let changed = words[i].1 != words[i - 1].1;
        let mid_sentence = !ends_sentence(&words[i - 1].0.text);
        let close = words[i].0.start - words[i - 1].0.end < 1.5;
        if changed && mid_sentence && close {
            let mut s = i - 1;
            while s > 0 && !ends_sentence(&words[s - 1].0.text) {
                s -= 1;
            }
            let mut e = i;
            while e + 1 < n && !ends_sentence(&words[e].0.text) {
                e += 1;
            }
            if e - s < 50 {
                let mut by_speaker: Vec<(usize, f64)> = Vec::new();
                for (w, sp) in &words[s..=e] {
                    let d = (w.end - w.start).max(0.02);
                    match by_speaker.iter_mut().find(|(x, _)| x == sp) {
                        Some(slot) => slot.1 += d,
                        None => by_speaker.push((*sp, d)),
                    }
                }
                if let Some((winner, _)) = by_speaker
                    .iter()
                    .copied()
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                {
                    for item in &mut words[s..=e] {
                        item.1 = winner;
                    }
                }
            }
            i = e + 1;
            continue;
        }
        i += 1;
    }
}

impl Transcript {
    pub const VERSION: u32 = 1;

    /// Merge Whisper words with speaker turns. `turns = None` (no diarization) puts everything on speaker 0.
    /// `names` are display names by speaker index; missing ones become "Speaker N".
    pub fn build(
        source: &str,
        asr: &AsrResult,
        turns: Option<&[SpeakerTurn]>,
        names: &[String],
    ) -> Transcript {
        let mut turns: Vec<SpeakerTurn> = turns.map(|t| t.to_vec()).unwrap_or_default();
        turns.sort_by(|a, b| a.start.total_cmp(&b.start));

        // flat word list with a speaker each; a segment without word timings becomes one word
        let mut flat: Vec<(Word, usize)> = Vec::new();
        for seg in &asr.segments {
            if seg.words.is_empty() {
                let sp = if turns.is_empty() {
                    0
                } else {
                    speaker_for(&turns, seg.start, seg.end)
                };
                flat.push((
                    Word {
                        text: seg.text.trim().to_string(),
                        start: seg.start,
                        end: seg.end,
                    },
                    sp,
                ));
                continue;
            }
            for w in &seg.words {
                let sp = if turns.is_empty() {
                    0
                } else {
                    speaker_for(&turns, w.start, w.end.max(w.start))
                };
                flat.push((
                    Word {
                        text: w.text.clone(),
                        start: w.start,
                        end: w.end,
                    },
                    sp,
                ));
            }
        }

        smooth_speakers(&mut flat);

        let mut utterances: Vec<Utterance> = Vec::new();
        let mut cur: Vec<Word> = Vec::new();
        let mut cur_speaker = 0usize;
        let flush = |cur: &mut Vec<Word>, speaker: usize, out: &mut Vec<Utterance>| {
            if cur.is_empty() {
                return;
            }
            let text = cur
                .iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            out.push(Utterance {
                id: out.len() + 1,
                start: cur.first().map_or(0.0, |w| w.start),
                end: cur.last().map_or(0.0, |w| w.end),
                speaker,
                raw_text: text.clone(),
                text,
                words: std::mem::take(cur),
            });
        };
        for (w, sp) in flat {
            let split = match cur.last() {
                None => false,
                Some(last) => {
                    sp != cur_speaker
                        || w.start - last.end > MAX_GAP_IN_UTTERANCE
                        || (cur.len() >= SOFT_MAX_WORDS && ends_sentence(&last.text))
                }
            };
            if split {
                flush(&mut cur, cur_speaker, &mut utterances);
            }
            if cur.is_empty() {
                cur_speaker = sp;
            }
            cur.push(w);
        }
        flush(&mut cur, cur_speaker, &mut utterances);

        let count = utterances
            .iter()
            .map(|u| u.speaker + 1)
            .max()
            .unwrap_or(1)
            .max(names.len())
            .max(1);
        let speakers = (0..count)
            .map(|id| Speaker {
                id,
                name: names
                    .get(id)
                    .filter(|n| !n.trim().is_empty())
                    .cloned()
                    .unwrap_or_else(|| format!("Speaker {}", id + 1)),
            })
            .collect();

        Transcript {
            version: Self::VERSION,
            source: source.to_string(),
            language: asr.language.clone(),
            audio_seconds: asr.audio_seconds,
            asr_model: asr.model.clone(),
            speakers,
            utterances,
            refined: false,
        }
    }

    pub fn speaker_name(&self, id: usize) -> String {
        self.speakers
            .iter()
            .find(|s| s.id == id)
            .map(|s| s.name.clone())
            .unwrap_or_else(|| format!("Speaker {}", id + 1))
    }

    /// Apply LLM results: `(utterance id, new text, new speaker)`. Unknown ids are ignored.
    pub fn apply_refined(&mut self, refined: &[(usize, String, Option<usize>)]) {
        for (id, text, speaker) in refined {
            if let Some(u) = self.utterances.iter_mut().find(|u| u.id == *id) {
                u.text = text.clone();
                if let Some(sp) = speaker {
                    u.speaker = *sp;
                }
            }
        }
        self.refined = true;
    }

    pub fn rename_speaker(&mut self, id: usize, name: &str) {
        if let Some(s) = self.speakers.iter_mut().find(|s| s.id == id) {
            s.name = name.trim().to_string();
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(io_err(dir))?;
        }
        let json = serde_json::to_string_pretty(self).map_err(|e| Error::Registry {
            path: path.display().to_string(),
            msg: e.to_string(),
        })?;
        std::fs::write(path, json).map_err(io_err(path))
    }

    pub fn load(path: &Path) -> Result<Transcript> {
        let text = std::fs::read_to_string(path).map_err(io_err(path))?;
        serde_json::from_str(&text).map_err(|e| Error::Registry {
            path: path.display().to_string(),
            msg: e.to_string(),
        })
    }

    /// Plain text: `[HH:MM:SS] Name: text`, one line per utterance (same layout as the Python pipeline).
    pub fn to_txt(&self) -> String {
        let mut out = String::new();
        for u in &self.utterances {
            out.push_str(&format!(
                "[{}] {}: {}\n",
                clock(u.start),
                self.speaker_name(u.speaker),
                u.text
            ));
        }
        out
    }
}

/// `HH:MM:SS`
pub fn clock(seconds: f64) -> String {
    let s = seconds.max(0.0) as u64;
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asr(words: &[(&str, f64, f64)]) -> AsrResult {
        AsrResult {
            model: None,
            language: "en".into(),
            audio_seconds: 60.0,
            elapsed_seconds: 1.0,
            segments: vec![AsrSegment {
                start: words[0].1,
                end: words.last().unwrap().2,
                text: words.iter().map(|w| w.0).collect::<Vec<_>>().join(" "),
                no_speech_prob: 0.0,
                words: words
                    .iter()
                    .map(|&(t, s, e)| AsrWord {
                        text: t.into(),
                        start: s,
                        end: e,
                        prob: 1.0,
                    })
                    .collect(),
            }],
        }
    }

    #[test]
    fn splits_utterances_on_speaker_change() {
        let a = asr(&[
            ("hello", 0.0, 0.4),
            ("there.", 0.5, 0.9),
            ("hi", 1.2, 1.4),
            ("yes.", 1.5, 1.8),
        ]);
        let turns = [
            SpeakerTurn {
                start: 0.0,
                end: 1.0,
                speaker: 0,
            },
            SpeakerTurn {
                start: 1.1,
                end: 2.0,
                speaker: 1,
            },
        ];
        let t = Transcript::build("a.mp4", &a, Some(&turns), &["Eric".into()]);
        assert_eq!(t.utterances.len(), 2);
        assert_eq!(t.utterances[0].text, "hello there.");
        assert_eq!(t.utterances[1].speaker, 1);
        assert_eq!(t.speaker_name(0), "Eric");
        assert_eq!(t.speaker_name(1), "Speaker 2");
        assert!(
            t.to_txt()
                .starts_with("[00:00:00] Eric: hello there.\n[00:00:01] Speaker 2: hi yes.\n")
        );
    }

    #[test]
    fn sentence_is_not_split_between_speakers() {
        // "Yeah, I" was heard just before the turn change; the sentence belongs to speaker 1
        let a = asr(&[
            ("Give", 0.0, 0.3),
            ("me", 0.3, 0.5),
            ("a", 0.5, 0.6),
            ("second.", 0.6, 1.2),
            ("Yeah,", 1.5, 1.8),
            ("I", 1.85, 1.95),
            ("mean,", 2.0, 2.4),
            ("how", 2.4, 2.6),
            ("does", 2.6, 2.9),
            ("it", 2.9, 3.0),
            ("work?", 3.0, 3.6),
            ("It", 3.9, 4.0),
            ("does.", 4.0, 4.4),
        ]);
        let turns = [
            SpeakerTurn {
                start: 0.0,
                end: 1.3,
                speaker: 0,
            },
            SpeakerTurn {
                start: 1.3,
                end: 1.97,
                speaker: 1,
            },
            SpeakerTurn {
                start: 1.97,
                end: 3.7,
                speaker: 0,
            },
            SpeakerTurn {
                start: 3.7,
                end: 4.5,
                speaker: 1,
            },
        ];
        let t = Transcript::build("a.mp4", &a, Some(&turns), &[]);
        let texts: Vec<(usize, &str)> = t
            .utterances
            .iter()
            .map(|u| (u.speaker, u.text.as_str()))
            .collect();
        assert_eq!(
            texts,
            vec![
                (0, "Give me a second. Yeah, I mean, how does it work?"),
                (1, "It does.")
            ]
        );
    }

    #[test]
    fn long_silence_starts_a_new_utterance_without_diarization() {
        let a = asr(&[("one", 0.0, 0.3), ("two", 5.0, 5.3)]);
        let t = Transcript::build("a.mp4", &a, None, &[]);
        assert_eq!(t.utterances.len(), 2);
        assert!(t.utterances.iter().all(|u| u.speaker == 0));
    }

    #[test]
    fn word_in_a_gap_takes_the_nearest_turn() {
        let turns = [
            SpeakerTurn {
                start: 0.0,
                end: 1.0,
                speaker: 0,
            },
            SpeakerTurn {
                start: 3.0,
                end: 4.0,
                speaker: 1,
            },
        ];
        assert_eq!(speaker_for(&turns, 1.1, 1.3), 0);
        assert_eq!(speaker_for(&turns, 2.7, 2.9), 1);
        assert_eq!(speaker_for(&turns, 0.9, 1.4), 0); // overlap wins
    }

    #[test]
    fn refined_text_and_round_trip() {
        let a = asr(&[("helo", 0.0, 0.4), ("world", 0.5, 0.9)]);
        let mut t = Transcript::build("a.mp4", &a, None, &[]);
        t.apply_refined(&[
            (1, "Hello, world.".into(), None),
            (99, "ignored".into(), None),
        ]);
        assert_eq!(t.utterances[0].raw_text, "helo world");
        assert_eq!(t.utterances[0].text, "Hello, world.");
        assert!(t.refined);
        let dir = std::env::temp_dir().join("mt_transcript_test");
        let p = dir.join("t.json");
        t.save(&p).unwrap();
        assert_eq!(Transcript::load(&p).unwrap(), t);
    }
}
