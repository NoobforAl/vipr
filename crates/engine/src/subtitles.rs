//! Subtitle cues from a transcript, and the `.srt` / `.vtt` writers.
//!
//! Cues are built from word timings (not from whole speaker turns, which are far too long for a
//! screen). Limits are configurable; the defaults follow common broadcast guidance
//! (42 characters per line, 2 lines, 1 to 7 seconds, about 20 characters per second).

use unicode_bidi::{BidiClass, bidi_class};
use unicode_segmentation::UnicodeSegmentation;

use crate::lang::language;
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

/// True when a word ends a sentence, in the scripts that have their own full stop or question mark.
/// `،` and `؛` are a comma and a semicolon, so they don't count.
pub(crate) fn ends_sentence(s: &str) -> bool {
    s.ends_with(['.', '?', '!', '。', '！', '？', '؟'])
}

/// Columns a grapheme takes: 2 for wide CJK and emoji, 0 for invisible format characters, else 1.
fn grapheme_width(g: &str) -> usize {
    let is_wide = |c: char| {
        matches!(c as u32,
            0x1100..=0x115F
            | 0x2E80..=0x303E
            | 0x3041..=0x33FF
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xA000..=0xA4CF
            | 0xA960..=0xA97F
            | 0xAC00..=0xD7A3
            | 0xF900..=0xFAFF
            | 0xFE30..=0xFE6F
            | 0xFF00..=0xFF60
            | 0xFFE0..=0xFFE6
            | 0x1F300..=0x1F64F
            | 0x1F680..=0x1F6FF
            | 0x1F900..=0x1F9FF
            | 0x20000..=0x3FFFD)
    };
    let is_invisible = |c: char| {
        matches!(c as u32,
            0x200B..=0x200F | 0x202A..=0x202E | 0x2060..=0x2069 | 0xFEFF)
    };
    if g.chars().all(is_invisible) {
        0
    } else if g.chars().any(is_wide) {
        2
    } else {
        1
    }
}

/// Columns a string takes on screen, counted by grapheme so that combining marks and Persian or
/// Arabic vowel signs don't add width.
fn display_width(s: &str) -> usize {
    s.graphemes(true).map(grapheme_width).sum()
}

/// Fill lines up to `limit` columns, breaking between graphemes, at the last space on the line when
/// `at_spaces` is set and there is one. Whatever is left when `max_lines` is reached goes on the
/// last line: subtitle text is never dropped.
fn fill_lines(text: &str, limit: usize, max_lines: usize, at_spaces: bool) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut used = 0;
    for g in text.graphemes(true) {
        let width = grapheme_width(g);
        if used + width > limit && !current.is_empty() && lines.len() + 1 < max_lines {
            if g == " " {
                lines.push(std::mem::take(&mut current));
                used = 0;
                continue;
            }
            match current.rfind(' ').filter(|_| at_spaces) {
                Some(space) => {
                    let tail = current.split_off(space);
                    lines.push(std::mem::replace(
                        &mut current,
                        tail.trim_start_matches(' ').to_string(),
                    ));
                    used = display_width(&current);
                }
                None => {
                    lines.push(std::mem::take(&mut current));
                    used = 0;
                }
            }
        }
        current.push_str(g);
        used += width;
    }
    lines.push(current);
    lines
        .iter()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

/// Break `text` into at most `max_lines` lines of `width` columns, balanced when two lines are
/// needed. Wide characters count as 2. A word too long for a line breaks at a grapheme boundary.
/// Text too long for `max_lines` overflows the lines rather than losing words.
pub fn wrap(text: &str, width: usize, max_lines: usize) -> String {
    if display_width(text) <= width || max_lines < 2 {
        return text.to_string();
    }
    let width = width.max(1);
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let total = display_width(&flat);
    if total <= width {
        return flat;
    }

    // choose the two-line split at a space that minimises the longer line
    let words: Vec<&str> = flat.split(' ').collect();
    let widths: Vec<usize> = words.iter().map(|w| display_width(w)).collect();
    let mut left = 0;
    let mut best: Option<(usize, usize)> = None;
    for i in 1..words.len() {
        left += widths[i - 1] + usize::from(i > 1);
        let worst = left.max(total - left - 1);
        if best.is_none_or(|(w, _)| worst < w) {
            best = Some((worst, i));
        }
    }
    if let Some((worst, i)) = best
        && worst <= width
    {
        return format!("{}\n{}", words[..i].join(" "), words[i..].join(" "));
    }

    // no such split: fill evenly over the lines needed, then as tight as the width allows, and only
    // then break inside a word that is too long to fit on a line anywhere
    let lines_needed = total.div_ceil(width).clamp(2, max_lines);
    let even = total.div_ceil(lines_needed);
    for (limit, at_spaces) in [(even, true), (width, true), (even, false), (width, false)] {
        let lines = fill_lines(&flat, limit, max_lines, at_spaces);
        if lines.iter().all(|l| display_width(l) <= width) {
            return lines.join(
                "
",
            );
        }
    }
    // too long for any layout: balanced beats one short line and one huge one
    fill_lines(&flat, even, max_lines, true).join(
        "
",
    )
}

/// Prefix each line that has RTL text with U+200F, for a file in an RTL target language. A plain
/// text player then starts the line at the right and puts `؟` at the correct end, even when the line
/// begins with a Latin word. Lines with no RTL text stay as they are (an `[untranslated]` English
/// line would otherwise get its full stop on the wrong side), and so does everything for an LTR
/// target.
///
/// This is for the writer step only: a cue's stored `text` never holds these marks.
pub fn mark_rtl_text(text: &str, lang: &str) -> String {
    if !language(lang).is_some_and(|l| l.rtl) {
        return text.to_string();
    }
    let lines: Vec<String> = text
        .split('\n')
        .map(|line| {
            let has_rtl = line
                .chars()
                .any(|c| matches!(bidi_class(c), BidiClass::R | BidiClass::AL) && c != '\u{200F}');
            if has_rtl && !line.starts_with('\u{200F}') {
                format!("\u{200F}{line}")
            } else {
                line.to_string()
            }
        })
        .collect();
    lines.join("\n")
}

/// Copies of `cues` for passing to [`to_srt`] or [`to_vtt`] when writing a file for `lang`. See
/// [`mark_rtl_text`].
pub fn mark_rtl(cues: &[Cue], lang: &str) -> Vec<Cue> {
    cues.iter()
        .map(|c| Cue {
            text: mark_rtl_text(&c.text, lang),
            ..c.clone()
        })
        .collect()
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
                let too_long = display_width(&text) + display_width(&add) > line_budget;
                let too_slow = taken > 0 && w.end - start > opts.max_duration;
                let gap = taken > 0 && w.start - end > opts.max_gap;
                // after a sentence end, start a fresh cue once there is enough text to read
                let sentence_break = taken > 0
                    && ends_sentence(&words[i + taken - 1].text)
                    && display_width(&text) >= opts.max_chars_per_line / 2;
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

    fn check_wrap(text: &str, width: usize, lines: usize) -> String {
        let out = wrap(text, width, lines);
        assert!(!out.ends_with('\n'), "trailing newline: {out:?}");
        assert!(!out.starts_with('\n'), "leading newline: {out:?}");
        assert!(!out.contains("\n\n"), "empty line: {out:?}");
        assert!(out.split('\n').count() <= lines, "too many lines: {out:?}");
        let squeeze = |s: &str| s.split_whitespace().collect::<String>();
        assert_eq!(squeeze(&out), squeeze(text), "text changed");
        out
    }

    #[test]
    fn long_spaceless_text_has_no_trailing_newline() {
        let long = "a".repeat(50);
        let out = check_wrap(&long, 42, 2);
        assert_eq!(out.split('\n').count(), 2);
        assert!(out.split('\n').all(|l| l.chars().count() <= 42));
        // balanced, not 42 and 8
        assert_eq!(out, format!("{}\n{}", "a".repeat(25), "a".repeat(25)));
    }

    #[test]
    fn wide_characters_count_double() {
        let cjk = "这是一个没有空格的很长的句子用来测试换行功能是否正常工作";
        assert_eq!(cjk.chars().count(), 28);
        let out = check_wrap(cjk, 42, 2);
        let lines: Vec<&str> = out.split('\n').collect();
        assert_eq!(lines.len(), 2);
        assert!(lines.iter().all(|l| display_width(l) <= 42), "{out}");
        // 21 characters is exactly one full line
        assert_eq!(wrap(&"字".repeat(21), 42, 2), "字".repeat(21));
        assert_eq!(display_width("한글 カナ ａ"), 12);
        assert_eq!(display_width("abc"), 3);
    }

    #[test]
    fn graphemes_are_never_split() {
        // base letters with combining marks: one column each, and each must stay whole
        let text = "e\u{301}".repeat(30);
        let out = check_wrap(&text, 20, 2);
        for line in out.split('\n') {
            assert!(!line.starts_with('\u{301}'), "split a grapheme: {line:?}");
            assert_eq!(line.chars().count() % 2, 0);
        }
        let flags = "🇮🇷".repeat(30);
        let out = check_wrap(&flags, 20, 2);
        assert!(out.split('\n').all(|l| l.chars().count() % 2 == 0), "{out}");
    }

    #[test]
    fn zero_width_characters_take_no_room() {
        assert_eq!(display_width("می\u{200C}روم"), 5);
        assert_eq!(display_width("\u{200F}abc"), 3);
    }

    #[test]
    fn a_long_word_among_short_ones_still_fits() {
        let out = check_wrap(&format!("see {} now", "x".repeat(50)), 42, 2);
        assert!(out.split('\n').all(|l| l.chars().count() <= 42), "{out}");
    }

    #[test]
    fn text_over_the_budget_is_kept() {
        let long = "word ".repeat(40);
        let out = check_wrap(long.trim(), 42, 2);
        assert_eq!(out.split('\n').count(), 2);
        let spaceless = "字".repeat(100);
        check_wrap(&spaceless, 42, 2);
    }

    #[test]
    fn more_than_two_lines_are_used_when_allowed() {
        let long = "word ".repeat(20);
        let out = check_wrap(long.trim(), 42, 3);
        assert_eq!(out.split('\n').count(), 3);
        assert!(out.split('\n').all(|l| l.chars().count() <= 42), "{out}");
    }

    #[test]
    fn existing_line_breaks_are_reflowed_when_too_long() {
        assert_eq!(
            check_wrap("aaaa bbbb\ncccc dddd", 10, 2),
            "aaaa bbbb\ncccc dddd"
        );
        assert_eq!(wrap("a\nb", 42, 2), "a\nb");
    }

    #[test]
    fn narrow_limits_do_not_panic() {
        assert_eq!(wrap("", 0, 2), "");
        check_wrap("abc def", 0, 2);
        check_wrap("字字字", 1, 2);
        assert_eq!(wrap("too long for one", 5, 1), "too long for one");
    }

    #[test]
    fn sentence_ends_in_other_scripts() {
        for w in ["end.", "why?", "wow!", "結束。", "嗎？", "好！", "چرا؟"] {
            assert!(ends_sentence(w), "{w}");
        }
        for w in ["word", "a,", "و،", "ف؛", "12"] {
            assert!(!ends_sentence(w), "{w}");
        }
    }

    #[test]
    fn a_persian_question_mark_starts_a_new_cue() {
        let words = "این یک جمله نسبتا طولانی برای آزمایش است؟ جمله بعدی اینجا شروع می شود";
        let mut t = sample();
        let u = &mut t.utterances[0];
        u.text = words.into();
        u.raw_text = words.into();
        u.words = words
            .split_whitespace()
            .enumerate()
            .map(|(i, w)| crate::transcript::Word {
                text: w.into(),
                start: i as f64 * 0.3,
                end: i as f64 * 0.3 + 0.25,
            })
            .collect();
        let cues = build_cues(&t, &SubtitleOptions::default());
        assert!(cues[0].text.ends_with('؟'), "{cues:#?}");
    }

    #[test]
    fn rlm_marks_each_rtl_line_for_rtl_targets_only() {
        assert_eq!(
            mark_rtl_text("سلام\nAPI نسخه\nplain english.", "fa"),
            "\u{200F}سلام\n\u{200F}API نسخه\nplain english."
        );
        assert_eq!(mark_rtl_text("שלום", "he-IL"), "\u{200F}שלום");
        assert_eq!(mark_rtl_text("سلام", "en"), "سلام");
        assert_eq!(mark_rtl_text("سلام", "xx"), "سلام");
        assert_eq!(mark_rtl_text("\u{200F}سلام", "fa"), "\u{200F}سلام");
        assert_eq!(mark_rtl_text("\u{200F}abc", "fa"), "\u{200F}abc");
    }

    #[test]
    fn rlm_reaches_the_writers_but_not_the_cues() {
        let cues = vec![Cue {
            start: 1.0,
            end: 2.0,
            text: "سلام\nدنیا".into(),
            speaker: 0,
        }];
        let marked = mark_rtl(&cues, "fa");
        assert_eq!(
            to_srt(&marked),
            "1\n00:00:01,000 --> 00:00:02,000\n\u{200F}سلام\n\u{200F}دنیا\n\n"
        );
        assert!(to_vtt(&marked).contains("\u{200F}دنیا"));
        assert!(!to_srt(&cues).contains('\u{200F}'));
        assert_eq!(mark_rtl(&cues, "de"), cues);
    }
}
