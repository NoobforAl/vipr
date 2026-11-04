//! Subtitle cues from a transcript, and the `.srt` / `.vtt` writers.
//!
//! Cues are built from word timings (not from whole speaker turns, which are far too long for a
//! screen). Limits are configurable; the defaults follow common broadcast guidance
//! (42 characters per line, 2 lines, 1 to 7 seconds, about 20 characters per second).

use crate::transcript::{Transcript, Utterance};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeakerStyle {
    /// No speaker marks.
    None,
    /// `- ` at the start of the first cue after a speaker change.
    Dash,
    /// `Name: ` at the start of the first cue after a speaker change (and of the very first cue).
    Name,
}

#[derive(Debug, Clone, Copy)]
pub struct SubtitleOptions {
    pub max_chars_per_line: usize,
    pub max_lines: usize,
    pub min_duration: f64,
    pub max_duration: f64,
    /// Reading speed limit; cues that are too fast are lengthened when there is room.
    pub max_cps: f64,
    /// Silence longer than this always ends a cue.
    pub max_gap: f64,
    pub speaker_style: SpeakerStyle,
}

impl Default for SubtitleOptions {
    fn default() -> Self {
        SubtitleOptions {
            max_chars_per_line: 42,
            max_lines: 2,
            min_duration: 1.0,
            max_duration: 7.0,
            max_cps: 20.0,
            max_gap: 0.8,
            speaker_style: SpeakerStyle::Dash,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Cue {
    pub start: f64,
    pub end: f64,
    /// Lines joined with `\n`.
    pub text: String,
    pub speaker: usize,
}

#[derive(Clone)]
struct TimedWord {
    text: String,
    start: f64,
    end: f64,
}

/// Words of an utterance with timings. When the final text differs from what Whisper heard (LLM cleanup),
/// the new words are spread over the old timings in proportion, so the cue timing stays valid.
fn timed_words(u: &Utterance) -> Vec<TimedWord> {
    if u.text == u.raw_text && !u.words.is_empty() {
        return u
            .words
            .iter()
            .map(|w| TimedWord {
                text: w.text.clone(),
                start: w.start,
                end: w.end,
            })
            .collect();
    }
    let tokens: Vec<&str> = u.text.split_whitespace().collect();
    if tokens.is_empty() {
        return Vec::new();
    }
    if u.words.is_empty() {
        let step = (u.end - u.start).max(0.0) / tokens.len() as f64;
        return tokens
            .iter()
            .enumerate()
            .map(|(i, t)| TimedWord {
                text: t.to_string(),
                start: u.start + step * i as f64,
                end: u.start + step * (i + 1) as f64,
            })
            .collect();
    }
    let n_old = u.words.len();
    let n_new = tokens.len();
    tokens
        .iter()
        .enumerate()
        .map(|(i, t)| {
            // map new word i to the old word at the same relative position
            let a = (i * n_old / n_new).min(n_old - 1);
            let b = (((i + 1) * n_old).div_ceil(n_new)).clamp(a + 1, n_old) - 1;
            TimedWord {
                text: t.to_string(),
                start: u.words[a].start,
                end: u.words[b].end.max(u.words[a].start),
            }
        })
        .collect()
}

fn ends_clause(s: &str) -> bool {
    s.ends_with([',', ';', ':', '.', '?', '!'])
}

/// Break `text` into at most `max_lines` lines of `width` characters, balanced when two lines are needed.
fn wrap(text: &str, width: usize, max_lines: usize) -> String {
    if text.chars().count() <= width || max_lines < 2 {
        return text.to_string();
    }
    let words: Vec<&str> = text.split(' ').collect();
    // choose the split that minimises the longer line
    let mut best = (usize::MAX, 1usize);
    for i in 1..words.len() {
        let a: usize = words[..i].join(" ").chars().count();
        let b: usize = words[i..].join(" ").chars().count();
        let worst = a.max(b);
        if worst < best.0 {
            best = (worst, i);
        }
    }
    format!(
        "{}\n{}",
        words[..best.1].join(" "),
        words[best.1..].join(" ")
    )
}

/// Build cues for the whole transcript.
pub fn build_cues(t: &Transcript, opts: &SubtitleOptions) -> Vec<Cue> {
    let line_budget = opts.max_chars_per_line * opts.max_lines;
    let mut cues: Vec<Cue> = Vec::new();
    let mut last_speaker: Option<usize> = None;

    for u in &t.utterances {
        let words = timed_words(u);
        let mut i = 0;
        while i < words.len() {
            // greedily take words for one cue; reserve room for a speaker mark on the first cue of a turn
            let first_of_turn = i == 0 && last_speaker != Some(u.speaker);
            let mark = match (first_of_turn, opts.speaker_style) {
                (true, SpeakerStyle::Dash) if last_speaker.is_some() => "- ".to_string(),
                (true, SpeakerStyle::Name) => format!("{}: ", t.speaker_name(u.speaker)),
                _ => String::new(),
            };
            let mut text = mark.clone();
            let start = words[i].start;
            let mut end = words[i].end;
            let mut taken = 0;
            while i + taken < words.len() {
                let w = &words[i + taken];
                let add = if taken == 0 {
                    w.text.clone()
                } else {
                    format!(" {}", w.text)
                };
                let too_long = text.chars().count() + add.chars().count() > line_budget;
                let too_slow = taken > 0 && w.end - start > opts.max_duration;
                let gap = taken > 0 && w.start - end > opts.max_gap;
                // after a sentence end, start a fresh cue once there is enough text to read
                let sentence_break = taken > 0
                    && ends_clause(&words[i + taken - 1].text)
                    && words[i + taken - 1].text.ends_with(['.', '?', '!'])
                    && text.chars().count() >= opts.max_chars_per_line / 2;
                if taken > 0 && (too_long || too_slow || gap || sentence_break) {
                    break;
                }
                text.push_str(&add);
                end = w.end.max(end);
                taken += 1;
            }
            i += taken.max(1);
            cues.push(Cue {
                start,
                end,
                text: wrap(&text, opts.max_chars_per_line, opts.max_lines),
                speaker: u.speaker,
            });
        }
        last_speaker = Some(u.speaker);
    }

    fix_timing(&mut cues, opts);
    cues
}

/// Enforce minimum duration and reading speed where there is room, and remove any overlap.
fn fix_timing(cues: &mut [Cue], opts: &SubtitleOptions) {
    const GAP: f64 = 0.04; // keep a tiny gap between cues
    for i in 0..cues.len() {
        let next_start = cues.get(i + 1).map(|c| c.start);
        let chars = cues[i].text.chars().filter(|c| *c != '\n').count() as f64;
        let wanted = opts.min_duration.max(chars / opts.max_cps);
        let mut end = cues[i].end.max(cues[i].start + wanted);
        if let Some(ns) = next_start {
            end = end.min((ns - GAP).max(cues[i].start + 0.2));
        }
        cues[i].end = end.max(cues[i].start + 0.1);
        if let Some(ns) = next_start
            && cues[i].end > ns
        {
            cues[i].end = ns.max(cues[i].start + 0.05);
        }
    }
}

fn stamp(seconds: f64, comma: bool) -> String {
    let ms_total = (seconds.max(0.0) * 1000.0).round() as u64;
    let (h, m, s, ms) = (
        ms_total / 3_600_000,
        (ms_total / 60_000) % 60,
        (ms_total / 1000) % 60,
        ms_total % 1000,
    );
    format!(
        "{h:02}:{m:02}:{s:02}{}{ms:03}",
        if comma { ',' } else { '.' }
    )
}

pub fn to_srt(cues: &[Cue]) -> String {
    let mut out = String::new();
    for (n, c) in cues.iter().enumerate() {
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            n + 1,
            stamp(c.start, true),
            stamp(c.end, true),
            c.text
        ));
    }
    out
}

pub fn to_vtt(cues: &[Cue]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for c in cues {
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            stamp(c.start, false),
            stamp(c.end, false),
            c.text
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::{AsrResult, AsrSegment, AsrWord, SpeakerTurn};

    /// A long monologue plus a second speaker, with realistic word timings.
    fn sample() -> Transcript {
        let text = "So the plan for today is to look at the webhook that receives every WhatsApp message and then \
                    forwards it to the API, which we deployed behind Cloudflare last week. Can you check that the \
                    logs show the request before we merge anything into main? Yes I will do that right after this call";
        let mut words = Vec::new();
        let mut t = 0.0;
        for w in text.split_whitespace() {
            let len = 0.12 + 0.03 * w.len() as f64;
            words.push(AsrWord {
                text: w.into(),
                start: t,
                end: t + len,
                prob: 1.0,
            });
            t += len + 0.04;
        }
        let end = words.last().unwrap().end;
        let asr = AsrResult {
            model: None,
            language: "en".into(),
            audio_seconds: end,
            elapsed_seconds: 1.0,
            segments: vec![AsrSegment {
                start: 0.0,
                end,
                text: text.into(),
                no_speech_prob: 0.0,
                words,
            }],
        };
        let turns = [
            SpeakerTurn {
                start: 0.0,
                end: end - 3.0,
                speaker: 0,
            },
            SpeakerTurn {
                start: end - 3.0,
                end,
                speaker: 1,
            },
        ];
        Transcript::build("a.mp4", &asr, Some(&turns), &[])
    }

    #[test]
    fn cues_respect_the_limits() {
        let opts = SubtitleOptions::default();
        let cues = build_cues(&sample(), &opts);
        assert!(cues.len() >= 4, "expected several cues, got {}", cues.len());
        for (i, c) in cues.iter().enumerate() {
            let lines: Vec<&str> = c.text.lines().collect();
            assert!(
                lines.len() <= opts.max_lines,
                "cue {i}: too many lines: {:?}",
                c.text
            );
            for l in &lines {
                assert!(
                    l.chars().count() <= opts.max_chars_per_line + 2,
                    "cue {i}: line too long ({}): {l:?}",
                    l.chars().count()
                );
            }
            assert!(c.end > c.start, "cue {i}: empty duration");
            assert!(
                c.end - c.start <= opts.max_duration + 1.0,
                "cue {i}: too long {:.1}s",
                c.end - c.start
            );
            if let Some(n) = cues.get(i + 1) {
                assert!(c.end <= n.start + 1e-9, "cue {i} overlaps the next one");
            }
        }
    }

    #[test]
    fn dash_marks_the_speaker_change_only() {
        let cues = build_cues(&sample(), &SubtitleOptions::default());
        assert!(!cues[0].text.starts_with("- "));
        let dashes = cues.iter().filter(|c| c.text.starts_with("- ")).count();
        assert_eq!(dashes, 1, "exactly one speaker change: {cues:#?}");
    }

    #[test]
    fn name_style_prefixes_first_cue_and_changes() {
        let opts = SubtitleOptions {
            speaker_style: SpeakerStyle::Name,
            ..Default::default()
        };
        let cues = build_cues(&sample(), &opts);
        assert!(cues[0].text.starts_with("Speaker 1: "));
        assert_eq!(
            cues.iter()
                .filter(|c| c.text.starts_with("Speaker "))
                .count(),
            2
        );
    }

    #[test]
    fn refined_text_keeps_valid_timing() {
        let mut t = sample();
        let u = &mut t.utterances[0];
        u.text = "So, the plan for today: look at the webhook.".into();
        let cues = build_cues(&t, &SubtitleOptions::default());
        assert!(cues[0].text.starts_with("So, the plan"));
        assert!(cues.windows(2).all(|w| w[0].end <= w[1].start + 1e-9));
    }

    #[test]
    fn writers_use_the_right_time_format() {
        let cues = vec![Cue {
            start: 3661.5,
            end: 3663.25,
            text: "Hello\nthere".into(),
            speaker: 0,
        }];
        assert_eq!(
            to_srt(&cues),
            "1\n01:01:01,500 --> 01:01:03,250\nHello\nthere\n\n"
        );
        assert_eq!(
            to_vtt(&cues),
            "WEBVTT\n\n01:01:01.500 --> 01:01:03.250\nHello\nthere\n\n"
        );
    }

    #[test]
    fn wrap_balances_two_lines() {
        assert_eq!(wrap("aaaa bbbb cccc dddd", 10, 2), "aaaa bbbb\ncccc dddd");
        assert_eq!(wrap("short", 42, 2), "short");
    }
}
