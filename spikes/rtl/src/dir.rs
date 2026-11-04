//! Prototype of the `dir_align` helper the plan wants for panels and tables.

use iced::widget::text::Alignment;
use unicode_bidi::{BidiClass, bidi_class};

/// Direction of the first strong character (Unicode bidi rules P2 and P3): `Some(true)` for RTL,
/// `Some(false)` for LTR, `None` when the first paragraph has no strong character.
pub fn first_strong_rtl(text: &str) -> Option<bool> {
    let mut isolates = 0u32;
    for c in text.chars() {
        match bidi_class(c) {
            BidiClass::B => return None,
            BidiClass::LRI | BidiClass::RLI | BidiClass::FSI => isolates += 1,
            BidiClass::PDI => isolates = isolates.saturating_sub(1),
            BidiClass::L if isolates == 0 => return Some(false),
            BidiClass::R | BidiClass::AL if isolates == 0 => return Some(true),
            _ => {}
        }
    }
    None
}

/// Alignment for a line of content text. Use with `.width(Fill)`, or the text box shrinks to the
/// text and there is nothing to align in.
pub fn dir_align(text: &str) -> Alignment {
    match first_strong_rtl(text) {
        Some(true) => Alignment::Right,
        _ => Alignment::Left,
    }
}

/// Text for a `text` widget that really honours Right or Center on RTL lines, one line or wrapped.
///
/// iced 0.14 draws a one-line RTL paragraph with `align_x` Left/Right/Center at the wrong place, or
/// off the widget entirely (spikes/rtl section 5). LRM makes the line's base direction LTR, so iced
/// never takes that path, and RLI..PDI keeps the content RTL. Display only: never put this in a
/// text_input or editor value, or in a saved file.
pub fn isolate_rtl(text: &str) -> String {
    let lines: Vec<String> = text
        .split('\n')
        .map(|line| match first_strong_rtl(line) {
            Some(true) => format!("\u{200E}\u{2067}{line}\u{2069}"),
            _ => line.to_string(),
        })
        .collect();
    lines.join("\n")
}

/// cosmic-text panics on one line holding bidi paragraphs of different direction, which is what
/// U+2029, U+0085 and U+001C..U+001E produce (shape.rs:960). Turn them into plain line breaks
/// before any untrusted text reaches a widget.
pub fn clean_separators(text: &str) -> String {
    text.chars()
        .map(|c| {
            if matches!(c, '\u{2029}' | '\u{85}' | '\u{1C}'..='\u{1E}') {
                '\n'
            } else {
                c
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use unicode_bidi::BidiInfo;

    #[test]
    fn isolate_makes_the_base_ltr_per_line() {
        let out = isolate_rtl("سلام\nabc\n\u{200F}API نسخه");
        for line in out.split('\n') {
            assert!(!shaper_rtl(line), "{line:?}");
        }
        assert!(out.contains("\nabc\n"));
    }

    #[test]
    fn separators_leave_one_paragraph_per_line() {
        let out = clean_separators("abc\u{2029}سلام\u{85}x\u{1C}y");
        for line in out.split('\n') {
            assert_eq!(BidiInfo::new(line, None).paragraphs.len(), 1);
        }
    }

    /// What cosmic-text uses for the line's base direction (shape.rs: `bidi.paragraphs[0]`).
    fn shaper_rtl(text: &str) -> bool {
        BidiInfo::new(text, None)
            .paragraphs
            .first()
            .is_some_and(|p| p.level.is_rtl())
    }

    #[test]
    fn matches_the_shaper_base_direction() {
        for s in [
            "",
            "   ",
            "123",
            "(5) !?",
            "۱۲۳",
            "سلام",
            "שלום",
            "Hello",
            "API نسخه",
            "نسخه 5G را API",
            "5G نسخه",
            "(API) نسخه",
            "(نسخه) API",
            "\u{200F}API نسخه",
            "\u{200E}سلام",
            "\u{2067}abc\u{2069} سلام",
            "\u{2066}سلام\u{2069} abc",
            "؟abc",
            "،abc",
            "日本語 سلام",
        ] {
            assert_eq!(first_strong_rtl(s).unwrap_or(false), shaper_rtl(s), "{s:?}");
        }
    }

    #[test]
    fn alignment() {
        assert_eq!(dir_align("نسخه 5G را API"), Alignment::Right);
        // digits are not strong, but the G in 5G is: this line's base direction is LTR
        assert_eq!(dir_align("5G نسخه"), Alignment::Left);
        assert_eq!(dir_align("12 نسخه"), Alignment::Right);
        assert_eq!(dir_align("؟"), Alignment::Right); // U+061F is class AL, so it counts as strong
        assert_eq!(dir_align("API نسخه"), Alignment::Left);
        assert_eq!(dir_align("\u{200F}API نسخه"), Alignment::Right);
        assert_eq!(dir_align("12:30"), Alignment::Left);
    }
}
