//! Reading subtitle files: decoding bytes to text, then tolerant SRT, VTT and ASS parsing.
//!
//! Cue text keeps its words and line breaks but loses formatting markup (`<i>`, `{\an8}`, ASS
//! overrides), since those would make two identical lines look different in a compare and would be
//! read out by the translator. Text is otherwise untrusted: see `sanitize` and
//! `lang::clean_separators`.

use chardetng::{EncodingDetector, Iso2022JpDetection, Utf8Detection};
use encoding_rs::{Encoding, REPLACEMENT, UTF_8, UTF_16BE, UTF_16LE};

use crate::project::{ImportWarning, SubFormat, TimedText};

/// Longest markup tag this looks for, in characters. Keeps the scan linear on hostile input.
const MAX_TAG: usize = 256;

/// Bytes turned into text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decoded {
    pub text: String,
    /// `encoding_rs` label, such as `UTF-8` or `windows-1256`.
    pub encoding: String,
    /// True when some bytes didn't fit the encoding and were replaced. The preview should say so.
    pub lossy: bool,
}

/// A parsed subtitle file. Bad cues become warnings, so parsing itself cannot fail.
#[derive(Debug, Clone, PartialEq)]
pub struct Parsed {
    pub format: SubFormat,
    /// Ids count from 1 in file order. Text is as written in the file (still untrusted), minus
    /// formatting markup.
    pub cues: Vec<TimedText>,
    pub warnings: Vec<ImportWarning>,
}

fn decode_with(encoding: &'static Encoding, bytes: &[u8]) -> Decoded {
    let (text, lossy) = encoding.decode_without_bom_handling(bytes);
    Decoded {
        text: text.into_owned(),
        encoding: encoding.name().to_string(),
        lossy,
    }
}

/// UTF-16 without a BOM still has a NUL beside every ASCII character, and every subtitle file is
/// full of ASCII timestamps. `chardetng` doesn't look for this, and it would also pass as UTF-8.
fn bomless_utf16(bytes: &[u8]) -> Option<&'static Encoding> {
    if bytes.len() < 4 || !bytes.len().is_multiple_of(2) {
        return None;
    }
    let units = bytes.len() / 2;
    let nul_at = |offset: usize| {
        bytes
            .iter()
            .skip(offset)
            .step_by(2)
            .filter(|b| **b == 0)
            .count()
    };
    let (even, odd) = (nul_at(0), nul_at(1));
    if odd * 4 >= units && even * 100 <= units {
        Some(UTF_16LE)
    } else if even * 4 >= units && odd * 100 <= units {
        Some(UTF_16BE)
    } else {
        None
    }
}

/// BOM, then strict UTF-8, then `chardetng` + `encoding_rs`.
pub fn decode(bytes: &[u8]) -> Decoded {
    if let Some((encoding, bom)) = Encoding::for_bom(bytes) {
        return decode_with(encoding, &bytes[bom..]);
    }
    if let Some(encoding) = bomless_utf16(bytes) {
        let decoded = decode_with(encoding, bytes);
        if !decoded.lossy {
            return decoded;
        }
    }
    if std::str::from_utf8(bytes).is_ok() {
        return decode_with(UTF_8, bytes);
    }
    let mut detector = EncodingDetector::new(Iso2022JpDetection::Deny);
    detector.feed(bytes, true);
    decode_with(detector.guess(None, Utf8Detection::Deny), bytes)
}

/// Decode with an encoding the user picked in the import preview. `None` for an unknown label.
pub fn decode_as(bytes: &[u8], label: &str) -> Option<Decoded> {
    let encoding = Encoding::for_label(label.trim().as_bytes())?;
    if encoding == REPLACEMENT {
        return None; // a label for encodings that are refused on purpose
    }
    let (text, lossy) = encoding.decode_with_bom_removal(bytes);
    Some(Decoded {
        text: text.into_owned(),
        encoding: encoding.name().to_string(),
        lossy,
    })
}

/// Parse decoded text. The format is detected from the content.
pub fn parse(text: &str) -> Parsed {
    let text = text.strip_prefix('\u{FEFF}').unwrap_or(text);
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let lines: Vec<&str> = normalized.split('\n').collect();
    let first = lines
        .iter()
        .map(|l| l.trim())
        .find(|l| !l.is_empty())
        .unwrap_or("");

    let has_arrow = lines.iter().any(|l| l.contains("-->"));
    let ass_like = first.eq_ignore_ascii_case("[script info]")
        || (!has_arrow && lines.iter().any(|l| has_prefix_ci(l.trim(), "dialogue:")));

    let mut out = if first.starts_with("WEBVTT") {
        parse_timed_lines(&lines, SubFormat::Vtt)
    } else if ass_like {
        parse_ass(&lines)
    } else if has_arrow {
        parse_timed_lines(&lines, SubFormat::Srt)
    } else {
        Parsed {
            format: SubFormat::Unknown,
            cues: Vec::new(),
            warnings: Vec::new(),
        }
    };
    if out.cues.is_empty() {
        out.warnings.push(warning(None, "No subtitle cues found."));
    }
    out
}

fn warning(line: Option<usize>, message: &str) -> ImportWarning {
    ImportWarning {
        line: line.and_then(|n| u32::try_from(n + 1).ok()),
        message: message.to_string(),
    }
}

fn has_prefix_ci(line: &str, prefix: &str) -> bool {
    line.get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
}

// ------------------------------------------------------------------------- times

/// `HH:MM:SS,mmm`, `HH:MM:SS.mmm`, `MM:SS.mmm`, `H:MM:SS.cc`, `HH:MM:SS:mmm` or `HH:MM:SS`.
fn parse_time(s: &str) -> Option<f64> {
    let (main, frac) = match s.rfind([',', '.']) {
        Some(i) => (&s[..i], Some(&s[i + 1..])),
        None => (s, None),
    };
    let parts: Vec<&str> = main.split(':').collect();
    let (clock, frac) = match (parts.len(), frac) {
        (2 | 3, _) => (&parts[..], frac),
        (4, None) => (&parts[..3], Some(parts[3])),
        _ => return None,
    };
    let mut numbers = clock.iter().map(|p| number(p));
    let (hours, minutes, seconds) = if clock.len() == 3 {
        (numbers.next()??, numbers.next()??, numbers.next()??)
    } else {
        (0, numbers.next()??, numbers.next()??)
    };
    if seconds >= 60 || (clock.len() == 3 && minutes >= 60) {
        return None;
    }
    let fraction = match frac {
        Some(f) => number(f)? as f64 / 10f64.powi(f.len() as i32),
        None => 0.0,
    };
    Some((hours * 3600 + minutes * 60 + seconds) as f64 + fraction)
}

fn number(s: &str) -> Option<u64> {
    if s.is_empty() || s.len() > 9 || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// A line meant as a timing line, even if its times turn out to be broken. Text such as
/// `go --> there` is not one, so it stays text.
fn looks_like_timing(line: &str) -> bool {
    let Some((left, _)) = line.split_once("-->") else {
        return false;
    };
    let left = left.trim().trim_end_matches('-').trim_end();
    let only_time_chars = left
        .chars()
        .all(|c| c.is_ascii_digit() || matches!(c, ':' | ',' | '.' | ' ' | '-'));
    // `00:00:xx,000` is a damaged time, while `10:30 meeting` is a sentence
    let damaged_time = left.starts_with(|c: char| c.is_ascii_digit())
        && left.contains(':')
        && !left.contains(char::is_whitespace);
    left.chars().any(|c| c.is_ascii_digit()) && (only_time_chars || damaged_time)
}

/// Start and end of a timing line. Anything after the end time (VTT settings, SRT coordinates) is
/// ignored.
fn parse_timing(line: &str) -> Option<(f64, f64)> {
    let (left, right) = line.split_once("-->")?;
    let start = parse_time(left.trim().trim_end_matches('-').trim_end())?;
    let end = parse_time(right.split_whitespace().next()?)?;
    Some((start, end))
}

// ------------------------------------------------------------------------- SRT and VTT

struct OpenCue {
    start: f64,
    end: f64,
    line: usize,
    text: Vec<String>,
}

fn close_cue(open: &mut Option<OpenCue>, out: &mut Parsed) {
    let Some(cue) = open.take() else {
        return;
    };
    if cue.text.is_empty() {
        out.warnings
            .push(warning(Some(cue.line), "Cue has no text and was skipped."));
        return;
    }
    out.cues.push(TimedText {
        id: out.cues.len() as u64 + 1,
        start: cue.start,
        end: cue.end,
        text: cue.text.join("\n"),
    });
}

/// SRT and VTT share a shape: optional id line, timing line, text lines, blank line. Written as a
/// line scanner, not by blocks, so a missing blank line can't swallow the next cue.
fn parse_timed_lines(lines: &[&str], format: SubFormat) -> Parsed {
    let vtt = format == SubFormat::Vtt;
    let mut out = Parsed {
        format,
        cues: Vec::new(),
        warnings: Vec::new(),
    };
    let mut open: Option<OpenCue> = None;
    let mut skipping = false;

    for (i, raw) in lines.iter().enumerate() {
        let line = raw.trim();
        if looks_like_timing(line) {
            close_cue(&mut open, &mut out);
            skipping = false;
            match parse_timing(line) {
                Some((start, end)) if end > start => {
                    open = Some(OpenCue {
                        start,
                        end,
                        line: i,
                        text: Vec::new(),
                    });
                }
                Some(_) => {
                    skipping = true;
                    out.warnings.push(warning(
                        Some(i),
                        "Cue ends before it starts and was skipped.",
                    ));
                }
                None => {
                    skipping = true;
                    out.warnings
                        .push(warning(Some(i), "Unreadable time; cue skipped."));
                }
            }
            continue;
        }
        if line.is_empty() {
            close_cue(&mut open, &mut out);
            skipping = false;
            continue;
        }
        if skipping {
            continue;
        }
        let next_is_timing = lines
            .get(i + 1)
            .is_some_and(|l| looks_like_timing(l.trim()));
        if let Some(cue) = open.as_mut() {
            // a cue number right before the next timing line, where the blank line went missing
            if next_is_timing && line.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            let mut text = strip_markup(line);
            if vtt {
                text = decode_entities(&text);
            }
            let text = text.trim();
            if !text.is_empty() {
                cue.text.push(text.to_string());
            }
        } else if next_is_timing {
            // a cue number or VTT cue id
        } else {
            skipping = true;
            let header = vtt
                && ["WEBVTT", "NOTE", "STYLE", "REGION"]
                    .iter()
                    .any(|h| line.starts_with(h));
            if !header {
                out.warnings
                    .push(warning(Some(i), "Text outside any cue was ignored."));
            }
        }
    }
    close_cue(&mut open, &mut out);
    out
}

/// Removes `<i>`, `<font ...>`, VTT tags and `{\an8}`. Only names known to be markup, so that
/// `a <b and c> d` or `<3` stay text.
fn strip_markup(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(pos) = rest.find(['<', '{']) {
        out.push_str(&rest[..pos]);
        let tail = &rest[pos..];
        let close = if tail.starts_with('<') {
            find_within(tail, '>').filter(|&end| is_markup_tag(&tail[1..end]))
        } else if tail.starts_with("{\\") {
            find_within(tail, '}')
        } else {
            None
        };
        match close {
            Some(end) => rest = &tail[end + 1..],
            None => {
                out.push_str(&tail[..1]);
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn find_within(s: &str, wanted: char) -> Option<usize> {
    s.char_indices()
        .take(MAX_TAG)
        .find(|(_, c)| *c == wanted)
        .map(|(i, _)| i)
}

fn is_markup_tag(inner: &str) -> bool {
    let inner = inner.strip_prefix('/').unwrap_or(inner);
    // VTT karaoke timestamp: <00:00:01.000>
    if inner.contains(':') && parse_time(inner).is_some() {
        return true;
    }
    let (head, attrs) = inner.split_once(char::is_whitespace).unwrap_or((inner, ""));
    let name = head.split('.').next().unwrap_or(head).to_ascii_lowercase();
    match name.as_str() {
        "i" | "b" | "u" | "s" | "c" | "ruby" | "rt" => attrs.is_empty(),
        "font" => attrs.is_empty() || attrs.contains('='),
        "v" | "lang" => true,
        _ => false,
    }
}

/// The character references WebVTT defines by name. One pass, so `&amp;lt;` stays `&lt;`.
fn decode_entities(line: &str) -> String {
    const ENTITIES: [(&str, &str); 6] = [
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&nbsp;", " "),
        ("&lrm;", "\u{200E}"),
        ("&rlm;", "\u{200F}"),
    ];
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(pos) = rest.find('&') {
        out.push_str(&rest[..pos]);
        let tail = &rest[pos..];
        match ENTITIES.iter().find(|(from, _)| tail.starts_with(from)) {
            Some((from, to)) => {
                out.push_str(to);
                rest = &tail[from.len()..];
            }
            None => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

// ------------------------------------------------------------------------- ASS and SSA

/// Field positions from a `Format:` line. Text is always the last field, so commas in it survive.
struct AssFields {
    count: usize,
    start: usize,
    end: usize,
    text: usize,
}

impl AssFields {
    fn standard() -> AssFields {
        // Layer (or Marked), Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        AssFields {
            count: 10,
            start: 1,
            end: 2,
            text: 9,
        }
    }

    fn from_format(spec: &str) -> Option<AssFields> {
        let names: Vec<String> = spec
            .split(',')
            .map(|n| n.trim().to_ascii_lowercase())
            .collect();
        let at = |wanted: &str| names.iter().position(|n| n == wanted);
        Some(AssFields {
            count: names.len(),
            start: at("start")?,
            end: at("end")?,
            text: at("text")?,
        })
    }
}

fn parse_ass(lines: &[&str]) -> Parsed {
    let mut out = Parsed {
        format: SubFormat::Ass,
        cues: Vec::new(),
        warnings: Vec::new(),
    };
    let mut fields = AssFields::standard();
    let mut in_events = false;

    for (i, raw) in lines.iter().enumerate() {
        let line = raw.trim();
        if line.starts_with('[') && line.ends_with(']') {
            in_events = line.eq_ignore_ascii_case("[events]");
        } else if in_events && has_prefix_ci(line, "format:") {
            match AssFields::from_format(&line["format:".len()..]) {
                Some(parsed) => fields = parsed,
                None => out.warnings.push(warning(
                    Some(i),
                    "Events format is missing Start, End or Text; using the standard one.",
                )),
            }
        } else if has_prefix_ci(line, "dialogue:") {
            ass_dialogue(&line["dialogue:".len()..], i, &fields, &mut out);
        }
    }
    out
}

fn ass_dialogue(body: &str, line: usize, fields: &AssFields, out: &mut Parsed) {
    let values: Vec<&str> = body.splitn(fields.count, ',').collect();
    let (Some(start), Some(end), Some(text)) = (
        values.get(fields.start),
        values.get(fields.end),
        values.get(fields.text),
    ) else {
        out.warnings.push(warning(
            Some(line),
            "Dialogue line has too few fields; skipped.",
        ));
        return;
    };
    let (Some(start), Some(end)) = (parse_time(start.trim()), parse_time(end.trim())) else {
        out.warnings
            .push(warning(Some(line), "Unreadable time; cue skipped."));
        return;
    };
    if end <= start {
        out.warnings.push(warning(
            Some(line),
            "Cue ends before it starts and was skipped.",
        ));
        return;
    }
    let text = ass_text(text);
    if text.is_empty() {
        out.warnings
            .push(warning(Some(line), "Cue has no text and was skipped."));
        return;
    }
    out.cues.push(TimedText {
        id: out.cues.len() as u64 + 1,
        start,
        end,
        text,
    });
}

/// Removes `{...}` override blocks and vector drawings, and turns `\N`, `\n` and `\h` into text.
fn ass_text(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut drawing = false;
    let mut rest = raw;
    while let Some(c) = rest.chars().next() {
        if c == '{'
            && let Some(close) = rest.find('}')
        {
            if let Some(level) = drawing_level(&rest[1..close]) {
                drawing = level > 0;
            }
            rest = &rest[close + 1..];
            continue;
        }
        if c == '\\' {
            let replacement = match rest.as_bytes().get(1) {
                Some(b'N' | b'n') => Some('\n'),
                Some(b'h') => Some(' '),
                _ => None,
            };
            if let Some(r) = replacement {
                if !drawing {
                    out.push(r);
                }
                rest = &rest[2..];
                continue;
            }
        }
        if !drawing {
            out.push(c);
        }
        rest = &rest[c.len_utf8()..];
    }
    let lines: Vec<&str> = out
        .split('\n')
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    lines.join("\n")
}

/// The last `\p<level>` in an override block. Text after a non-zero level is drawing commands.
fn drawing_level(block: &str) -> Option<u32> {
    block
        .match_indices("\\p")
        .filter_map(|(i, m)| block[i + m.len()..].chars().next()?.to_digit(10))
        .last()
}

#[cfg(test)]
mod tests {
    use super::*;
    use encoding_rs::{BIG5, SHIFT_JIS, WINDOWS_1251, WINDOWS_1256};

    fn texts(p: &Parsed) -> Vec<&str> {
        p.cues.iter().map(|c| c.text.as_str()).collect()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    const PERSIAN_SRT: &str = "1\n00:00:01,000 --> 00:00:03,000\nسلام دوستان، امروز درباره‌ي برنامه صحبت مي‌كنيم\n\n\
        2\n00:00:04,000 --> 00:00:06,000\nاين نسخه جديد پروژه است و چند گزارش تازه دارد\n\n\
        3\n00:00:07,000 --> 00:00:09,000\nلطفا پيام‌ها را بخوانيد و به ما بگوييد چه فكر مي‌كنيد\n\n";

    #[test]
    fn decode_honours_every_bom() {
        let utf8 = [b"\xEF\xBB\xBF".as_slice(), "سلام".as_bytes()].concat();
        let d = decode(&utf8);
        assert_eq!((d.text.as_str(), d.encoding.as_str()), ("سلام", "UTF-8"));

        let le: Vec<u8> = [0xFF, 0xFE]
            .into_iter()
            .chain("hi سلام".encode_utf16().flat_map(u16::to_le_bytes))
            .collect();
        let d = decode(&le);
        assert_eq!(
            (d.text.as_str(), d.encoding.as_str()),
            ("hi سلام", "UTF-16LE")
        );

        let be: Vec<u8> = [0xFE, 0xFF]
            .into_iter()
            .chain("hi سلام".encode_utf16().flat_map(u16::to_be_bytes))
            .collect();
        let d = decode(&be);
        assert_eq!(
            (d.text.as_str(), d.encoding.as_str()),
            ("hi سلام", "UTF-16BE")
        );
        assert!(!d.lossy);
    }

    #[test]
    fn decode_finds_utf16_without_a_bom() {
        let srt = "1\n00:00:01,000 --> 00:00:02,000\nسلام\n";
        let le: Vec<u8> = srt.encode_utf16().flat_map(u16::to_le_bytes).collect();
        let d = decode(&le);
        assert_eq!((d.text.as_str(), d.encoding.as_str()), (srt, "UTF-16LE"));
        let be: Vec<u8> = srt.encode_utf16().flat_map(u16::to_be_bytes).collect();
        let d = decode(&be);
        assert_eq!((d.text.as_str(), d.encoding.as_str()), (srt, "UTF-16BE"));
    }

    #[test]
    fn decode_prefers_strict_utf8() {
        let d = decode("سلام abc".as_bytes());
        assert_eq!(d.encoding, "UTF-8");
        assert!(!d.lossy);
        assert_eq!(decode(b"").encoding, "UTF-8");
        assert_eq!(decode(b"abc").text, "abc");
    }

    #[test]
    fn decode_detects_windows_1256_persian() {
        let (bytes, _, unmappable) = WINDOWS_1256.encode(PERSIAN_SRT);
        assert!(!unmappable, "fixture must fit windows-1256");
        let d = decode(&bytes);
        assert_eq!(d.encoding, "windows-1256");
        assert_eq!(d.text, PERSIAN_SRT);
        assert!(!d.lossy);
    }

    #[test]
    fn decode_detects_other_legacy_encodings() {
        let russian = "1\n00:00:01,000 --> 00:00:03,000\nПривет всем, сегодня мы поговорим о новом проекте и планах на будущее\n";
        let (bytes, _, _) = WINDOWS_1251.encode(russian);
        assert_eq!(decode(&bytes).encoding, "windows-1251");
        let japanese =
            "1\n00:00:01,000 --> 00:00:03,000\nこんにちは、今日は新しい計画について話しましょう\n";
        let (bytes, _, _) = SHIFT_JIS.encode(japanese);
        assert_eq!(decode(&bytes).encoding, "Shift_JIS");
    }

    #[test]
    fn manual_override() {
        let (bytes, _, _) = WINDOWS_1256.encode(PERSIAN_SRT);
        let d = decode_as(&bytes, "windows-1256").unwrap();
        assert_eq!(d.text, PERSIAN_SRT);
        assert_eq!(d.encoding, "windows-1256");
        // labels are the WHATWG ones, so common aliases work
        assert_eq!(
            decode_as(&bytes, " cp1256 ").unwrap().encoding,
            "windows-1256"
        );
        assert_eq!(decode_as(&bytes, "no-such-encoding"), None);
        assert_eq!(decode_as(&bytes, "iso-2022-kr"), None);
        assert_eq!(decode_as(&bytes, ""), None);

        // a wrong pick shows up as lossy, and a matching BOM isn't left in the text
        assert!(decode_as(&bytes, "utf-8").unwrap().lossy);
        let d = decode_as(b"\xEF\xBB\xBFabc", "utf-8").unwrap();
        assert_eq!(d.text, "abc");
        let (big5, _, _) = BIG5.encode("繁體字幕檔案內容");
        assert_eq!(decode_as(&big5, "big5").unwrap().text, "繁體字幕檔案內容");
    }

    #[test]
    fn lossy_bytes_are_reported() {
        let d = decode(b"\xEF\xBB\xBFabc\xFF");
        assert!(d.lossy);
        assert!(d.text.contains('\u{FFFD}'));
    }

    #[test]
    fn srt_basic_with_crlf_and_bom() {
        let text = "\u{FEFF}1\r\n00:00:01,500 --> 00:00:03,250\r\nHello\r\nthere\r\n\r\n\
                    2\r\n01:02:03,004 --> 01:02:04,000\r\nنسخه\r\n";
        let p = parse(text);
        assert_eq!(p.format, SubFormat::Srt);
        assert_eq!(p.warnings, []);
        assert_eq!(texts(&p), ["Hello\nthere", "نسخه"]);
        assert_eq!(p.cues[0].id, 1);
        assert_eq!(p.cues[1].id, 2);
        assert!(close(p.cues[0].start, 1.5));
        assert!(close(p.cues[0].end, 3.25));
        assert!(close(p.cues[1].start, 3723.004));
    }

    #[test]
    fn lone_cr_line_ends_work() {
        let p = parse(
            "1\r00:00:01,000 --> 00:00:02,000\rHello\r\r2\r00:00:03,000 --> 00:00:04,000\rBye\r",
        );
        assert_eq!(texts(&p), ["Hello", "Bye"]);
    }

    #[test]
    fn srt_tolerates_common_damage() {
        // no numbers, missing blank line, dot decimals, odd arrow spacing, coordinates
        let p = parse(
            "00:00:01.000-->00:00:02.000\nOne\n\
             2\n\
             00:00:03,000 --> 00:00:04,000 X1:10 X2:20\nTwo\n\n\n\n\
             00:00:05,5 ---> 00:00:06\nThree\n",
        );
        assert_eq!(p.warnings, []);
        assert_eq!(texts(&p), ["One", "Two", "Three"]);
        assert!(close(p.cues[2].start, 5.5));
        assert!(close(p.cues[2].end, 6.0));
        assert_eq!(p.cues.iter().map(|c| c.id).collect::<Vec<_>>(), [1, 2, 3]);
    }

    #[test]
    fn bad_cues_become_warnings_and_ids_stay_contiguous() {
        let p = parse(
            "1\n00:00:01,000 --> 00:00:02,000\nGood\n\n\
             2\n00:00:xx,000 --> 00:00:04,000\nBad time\n\n\
             3\n00:00:09,000 --> 00:00:05,000\nBackwards\n\n\
             4\n00:00:06,000 --> 00:00:07,000\n\n\
             stray credit line\n\n\
             5\n00:00:08,000 --> 00:00:09,000\nAlso good\n",
        );
        assert_eq!(texts(&p), ["Good", "Also good"]);
        assert_eq!(p.cues[1].id, 2);
        let lines: Vec<Option<u32>> = p.warnings.iter().map(|w| w.line).collect();
        assert_eq!(lines, [Some(6), Some(10), Some(14), Some(16)]);
    }

    #[test]
    fn number_inside_text_is_kept_unless_a_timing_follows() {
        let p = parse(
            "00:00:01,000 --> 00:00:02,000\n1984\nwas good\n\n00:00:03,000 --> 00:00:04,000\n2049\n",
        );
        assert_eq!(texts(&p), ["1984\nwas good", "2049"]);
    }

    #[test]
    fn an_arrow_in_text_stays_text() {
        let p = parse("00:00:01,000 --> 00:00:02,000\ngo --> there\nand <-- back\n");
        assert_eq!(p.warnings, []);
        assert_eq!(texts(&p), ["go --> there\nand <-- back"]);
    }

    #[test]
    fn huge_or_malformed_times_do_not_panic() {
        for s in [
            "99999999999999999999:00:00,000 --> 00:00:01,000\nx",
            "00:00:00,99999999999 --> 00:00:01,000\nx",
            "١:٠٠:٠٠ --> 00:00:01,000\nx",
            "-00:00:01,000 --> 00:00:02,000\nx",
            ":: --> ::\nx",
            "1 -->\nx",
            "-->\nx",
        ] {
            let p = parse(s);
            assert_eq!(p.cues, [], "{s:?}");
            assert!(!p.warnings.is_empty(), "{s:?}");
        }
    }

    #[test]
    fn times_in_every_spelling() {
        for (s, want) in [
            ("00:00:01,000", Some(1.0)),
            ("00:00:01.5", Some(1.5)),
            ("1:02:03.04", Some(3723.04)),
            ("01:02.500", Some(62.5)),
            ("90:00.000", Some(5400.0)),
            ("00:00:01:250", Some(1.25)),
            ("00:00:07", Some(7.0)),
            ("00:60:00", None),
            ("00:00:60", None),
            ("5", None),
            ("", None),
            ("a:b:c", None),
        ] {
            assert_eq!(parse_time(s), want, "{s:?}");
        }
    }

    #[test]
    fn markup_goes_but_lookalikes_stay() {
        for (input, want) in [
            ("<i>Hello</i> <b>there</b>", "Hello there"),
            ("<I>Hello</I>", "Hello"),
            ("<font color=\"#ff0000\">red</font>", "red"),
            ("{\\an8}top", "top"),
            ("{\\an8}<i>top</i>", "top"),
            ("<c.yellow.bg_blue>x</c>", "x"),
            ("<v Bob>Hi</v>", "Hi"),
            ("<lang en>Hi</lang>", "Hi"),
            ("one <00:00:01.000>two", "one two"),
            ("<ruby>漢<rt>kan</rt></ruby>", "漢kan"),
            ("a < b", "a < b"),
            ("a <b and c> d", "a <b and c> d"),
            ("I <3 you", "I <3 you"),
            ("<i", "<i"),
            ("{not markup}", "{not markup}"),
            ("<|im_end|>", "<|im_end|>"), // sanitize's job, and it needs to see it
        ] {
            assert_eq!(strip_markup(input), want, "{input:?}");
        }
    }

    #[test]
    fn vtt_full_file() {
        let text = "WEBVTT - some title\nKind: captions\n\n\
                    NOTE a comment\nover two lines\n\n\
                    STYLE\n::cue { color: red }\n\n\
                    REGION\nid:fred\n\n\
                    intro-cue\n00:01.000 --> 00:02.500 line:90% align:start\n<v Bob>Tom &amp; Jerry &lt;3</v>\n\n\
                    00:00:03.000 --> 00:00:04.000\n&amp;lt; stays one level\nsecond line\n";
        let p = parse(text);
        assert_eq!(p.format, SubFormat::Vtt);
        assert_eq!(p.warnings, []);
        assert_eq!(
            texts(&p),
            ["Tom & Jerry <3", "&lt; stays one level\nsecond line"]
        );
        assert!(close(p.cues[0].start, 1.0));
        assert!(close(p.cues[0].end, 2.5));
    }

    #[test]
    fn vtt_header_directly_followed_by_a_cue() {
        let p = parse("WEBVTT\n00:01.000 --> 00:02.000\nHi\n");
        assert_eq!(texts(&p), ["Hi"]);
    }

    const ASS: &str = "[Script Info]\nTitle: x, y\nScriptType: v4.00+\n\n\
        [V4+ Styles]\nFormat: Name, Fontname, Fontsize\nStyle: Default,Arial,20\n\n\
        [Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\
        Comment: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,hidden\n\
        Dialogue: 0,0:00:01.50,0:00:03.25,Default,Bob,0,0,0,,Hello, world, again\n\
        Dialogue: 0,0:00:04.00,0:00:05.00,Default,,0,0,0,,{\\i1}First{\\i0}\\Nsecond\\hpart\n\
        Dialogue: 0,0:00:06.00,0:00:07.00,Default,,0,0,0,,{\\p1}m 0 0 l 10 10{\\p0}\n\
        Dialogue: 0,0:00:08.00,0:00:09.00,Default,,0,0,0,,{\\pos(1,2)}Kept\n\
        Dialogue: broken\n\
        Dialogue: 0,bad,0:00:09.00,Default,,0,0,0,,Bad time\n";

    #[test]
    fn ass_keeps_commas_and_drops_overrides() {
        let p = parse(ASS);
        assert_eq!(p.format, SubFormat::Ass);
        assert_eq!(
            texts(&p),
            ["Hello, world, again", "First\nsecond part", "Kept"]
        );
        assert!(close(p.cues[0].start, 1.5));
        assert!(close(p.cues[0].end, 3.25));
        assert_eq!(p.cues.iter().map(|c| c.id).collect::<Vec<_>>(), [1, 2, 3]);
        assert_eq!(p.warnings.len(), 3, "{:?}", p.warnings);
    }

    #[test]
    fn ass_follows_the_format_line() {
        let p = parse("[Events]\nFormat: Start, End, Text\nDialogue: 0:00:01.00,0:00:02.00,a, b\n");
        assert_eq!(texts(&p), ["a, b"]);

        // a styles Format line must not be taken for the events one
        let p = parse(
            "[Script Info]\n[V4 Styles]\nFormat: Name, Start\n[Events]\n\
             Dialogue: Marked=0,0:00:01.00,0:00:02.00,Default,NTP,0,0,0,,old ssa, text\n",
        );
        assert_eq!(texts(&p), ["old ssa, text"]);
    }

    #[test]
    fn ass_literal_braces_and_backslashes() {
        assert_eq!(ass_text("a {b"), "a {b");
        assert_eq!(ass_text("a\\b"), "a\\b");
        assert_eq!(ass_text("end\\"), "end\\");
        assert_eq!(ass_text("{\\p1}m 0 0{\\p0}after"), "after");
        assert_eq!(ass_text("سلام{\\b1}، دنیا"), "سلام، دنیا");
    }

    #[test]
    fn format_detection() {
        assert_eq!(parse("WEBVTT\n").format, SubFormat::Vtt);
        assert_eq!(parse("[Script Info]\n").format, SubFormat::Ass);
        assert_eq!(
            parse("Dialogue: 0,0:00:01.00,0:00:02.00,D,,0,0,0,,x").format,
            SubFormat::Ass
        );
        // a line that only looks like ASS, in a file with real SRT timing
        let p = parse("00:00:01,000 --> 00:00:02,000\nDialogue: nice\n");
        assert_eq!(p.format, SubFormat::Srt);
        assert_eq!(texts(&p), ["Dialogue: nice"]);
        // pasted text with no timing at all
        let p = parse("just some text\nand more\n");
        assert_eq!(p.format, SubFormat::Unknown);
        assert_eq!(p.cues, []);
        assert_eq!(p.warnings.len(), 1);
        let p = parse("");
        assert_eq!(p.format, SubFormat::Unknown);
        assert_eq!(p.warnings.len(), 1);
    }

    #[test]
    fn decode_then_parse_a_legacy_file() {
        let (bytes, _, _) = WINDOWS_1256.encode(PERSIAN_SRT);
        let p = parse(&decode(&bytes).text);
        assert_eq!(p.cues.len(), 3);
        assert!(p.cues[1].text.starts_with("اين نسخه"));
    }

    #[test]
    fn hostile_input_does_not_panic_or_stall() {
        let started = std::time::Instant::now();
        for s in [
            "<".repeat(200_000),
            "{\\".repeat(100_000),
            "-->".repeat(50_000),
            "\n".repeat(200_000),
            format!("00:00:01,000 --> 00:00:02,000\n{}", "<i".repeat(100_000)),
            "Dialogue:".repeat(20_000),
            "\u{0}\u{1C}\u{2029}".repeat(10_000),
        ] {
            parse(&s);
        }
        assert!(started.elapsed().as_secs() < 30);
    }
}
