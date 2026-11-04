//! Languages and direction-aware display of content text.
//!
//! Findings from the RTL spike (`spikes/rtl`) that shape these helpers:
//! - `dir_align` alone makes short one-line RTL text vanish in iced 0.14, so subtitles use
//!   `isolate_rtl` for display, and panels use a right-aligned container with Default-aligned text.
//! - U+2029, U+0085 and U+001C..U+001E between text of differing direction panic in cosmic-text, so
//!   `clean_separators` must run on untrusted text before any widget gets it.

use unicode_bidi::{BidiClass, bidi_class};

/// One language the user can pick as a translation target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Language {
    /// BCP 47 code: used in file names and worker input.
    pub code: &'static str,
    /// English name, used in the UI and in prompts.
    pub name: &'static str,
    pub rtl: bool,
}

/// Writing systems the translation guards can tell apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Script {
    Latin,
    Arabic,
    Hebrew,
    Cyrillic,
    Greek,
    Devanagari,
    Han,
    Kana,
    Hangul,
    Thai,
}

/// Where a line of content text goes in its panel. Engine has no iced, so the app converts this
/// with `out::content_align`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAlign {
    Left,
    Right,
}

const fn ltr(code: &'static str, name: &'static str) -> Language {
    Language {
        code,
        name,
        rtl: false,
    }
}

const fn rtl(code: &'static str, name: &'static str) -> Language {
    Language {
        code,
        name,
        rtl: true,
    }
}

/// All supported target languages, sorted by name for the picker.
pub const LANGUAGES: &[Language] = &[
    ltr("af", "Afrikaans"),
    ltr("sq", "Albanian"),
    ltr("am", "Amharic"),
    rtl("ar", "Arabic"),
    ltr("hy", "Armenian"),
    ltr("az", "Azerbaijani"),
    ltr("eu", "Basque"),
    ltr("be", "Belarusian"),
    ltr("bn", "Bengali"),
    ltr("bs", "Bosnian"),
    ltr("bg", "Bulgarian"),
    ltr("my", "Burmese"),
    ltr("ca", "Catalan"),
    ltr("zh-Hans", "Chinese (Simplified)"),
    ltr("zh-Hant", "Chinese (Traditional)"),
    ltr("hr", "Croatian"),
    ltr("cs", "Czech"),
    ltr("da", "Danish"),
    rtl("dv", "Dhivehi"),
    ltr("nl", "Dutch"),
    ltr("en", "English"),
    ltr("et", "Estonian"),
    ltr("fil", "Filipino"),
    ltr("fi", "Finnish"),
    ltr("fr", "French"),
    ltr("gl", "Galician"),
    ltr("ka", "Georgian"),
    ltr("de", "German"),
    ltr("el", "Greek"),
    ltr("gu", "Gujarati"),
    rtl("he", "Hebrew"),
    ltr("hi", "Hindi"),
    ltr("hu", "Hungarian"),
    ltr("is", "Icelandic"),
    ltr("id", "Indonesian"),
    ltr("ga", "Irish"),
    ltr("it", "Italian"),
    ltr("ja", "Japanese"),
    ltr("kn", "Kannada"),
    ltr("kk", "Kazakh"),
    ltr("km", "Khmer"),
    ltr("ko", "Korean"),
    rtl("ckb", "Kurdish (Sorani)"),
    ltr("ky", "Kyrgyz"),
    ltr("lo", "Lao"),
    ltr("lv", "Latvian"),
    ltr("lt", "Lithuanian"),
    ltr("mk", "Macedonian"),
    ltr("ms", "Malay"),
    ltr("ml", "Malayalam"),
    ltr("mt", "Maltese"),
    ltr("mr", "Marathi"),
    ltr("mn", "Mongolian"),
    ltr("ne", "Nepali"),
    ltr("no", "Norwegian"),
    rtl("ps", "Pashto"),
    rtl("fa", "Persian"),
    ltr("pl", "Polish"),
    ltr("pt", "Portuguese"),
    ltr("pa", "Punjabi"),
    ltr("ro", "Romanian"),
    ltr("ru", "Russian"),
    ltr("sr", "Serbian"),
    rtl("sd", "Sindhi"),
    ltr("si", "Sinhala"),
    ltr("sk", "Slovak"),
    ltr("sl", "Slovenian"),
    ltr("es", "Spanish"),
    ltr("sw", "Swahili"),
    ltr("sv", "Swedish"),
    ltr("tg", "Tajik"),
    ltr("ta", "Tamil"),
    ltr("te", "Telugu"),
    ltr("th", "Thai"),
    ltr("tr", "Turkish"),
    ltr("uk", "Ukrainian"),
    rtl("ur", "Urdu"),
    rtl("ug", "Uyghur"),
    ltr("uz", "Uzbek"),
    ltr("vi", "Vietnamese"),
    ltr("cy", "Welsh"),
    rtl("yi", "Yiddish"),
];

/// Case-insensitive lookup by code. A regional or script code falls back to its language, so
/// `fa-IR` finds `fa` and `zh-CN` finds the first Chinese entry.
pub fn language(code: &str) -> Option<&'static Language> {
    let code = code.trim().replace('_', "-");
    if let Some(found) = LANGUAGES
        .iter()
        .find(|l| l.code.eq_ignore_ascii_case(&code))
    {
        return Some(found);
    }
    let primary = code.split('-').next()?;
    if primary.is_empty() {
        return None;
    }
    LANGUAGES
        .iter()
        .find(|l| primary_subtag(l.code).eq_ignore_ascii_case(primary))
}

fn primary_subtag(code: &str) -> &str {
    code.split('-').next().unwrap_or(code)
}

/// Scripts a correct translation into `code` may be written in. Empty when unknown, which turns the
/// wrong-script guard off.
pub fn expected_scripts(code: &str) -> &'static [Script] {
    let Some(lang) = language(code) else {
        return &[];
    };
    match primary_subtag(lang.code) {
        "af" | "az" | "bs" | "ca" | "cs" | "cy" | "da" | "de" | "en" | "es" | "et" | "eu"
        | "fi" | "fil" | "fr" | "ga" | "gl" | "hr" | "hu" | "id" | "is" | "it" | "lt" | "lv"
        | "ms" | "mt" | "nl" | "no" | "pl" | "pt" | "ro" | "sk" | "sl" | "sq" | "sv" | "sw"
        | "tr" | "uz" | "vi" => &[Script::Latin],
        "ar" | "ckb" | "fa" | "ps" | "sd" | "ug" | "ur" => &[Script::Arabic],
        "he" | "yi" => &[Script::Hebrew],
        "be" | "bg" | "kk" | "ky" | "mk" | "mn" | "ru" | "tg" | "uk" => &[Script::Cyrillic],
        "sr" => &[Script::Cyrillic, Script::Latin],
        "el" => &[Script::Greek],
        "hi" | "mr" | "ne" => &[Script::Devanagari],
        "zh" => &[Script::Han],
        "ja" => &[Script::Han, Script::Kana],
        "ko" => &[Script::Hangul],
        "th" => &[Script::Thai],
        // written in a script `Script` has no value for
        _ => &[],
    }
}

/// The script a letter belongs to. `None` for digits, punctuation and scripts `Script` doesn't know.
pub fn script_of(c: char) -> Option<Script> {
    if !c.is_alphabetic() {
        return None;
    }
    match c as u32 {
        0x41..=0x5A
        | 0x61..=0x7A
        | 0xC0..=0x24F
        | 0x1E00..=0x1EFF
        | 0x2C60..=0x2C7F
        | 0xA720..=0xA7FF
        | 0xFF21..=0xFF3A
        | 0xFF41..=0xFF5A => Some(Script::Latin),
        0x370..=0x3FF | 0x1F00..=0x1FFF => Some(Script::Greek),
        0x400..=0x52F | 0x1C80..=0x1C8F | 0x2DE0..=0x2DFF | 0xA640..=0xA69F => {
            Some(Script::Cyrillic)
        }
        0x590..=0x5FF | 0xFB1D..=0xFB4F => Some(Script::Hebrew),
        0x600..=0x6FF | 0x750..=0x77F | 0x8A0..=0x8FF | 0xFB50..=0xFDFF | 0xFE70..=0xFEFC => {
            Some(Script::Arabic)
        }
        0x900..=0x97F | 0xA8E0..=0xA8FF => Some(Script::Devanagari),
        0xE00..=0xE7F => Some(Script::Thai),
        0x1100..=0x11FF | 0x3130..=0x318F | 0xA960..=0xA97F | 0xAC00..=0xD7FF => {
            Some(Script::Hangul)
        }
        0x3040..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F => Some(Script::Kana),
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x3FFFF => {
            Some(Script::Han)
        }
        _ => None,
    }
}

/// True when `text` is written mostly in an RTL script. Invisible direction marks don't count.
pub fn is_rtl_text(text: &str) -> bool {
    let (mut rtl, mut ltr) = (0usize, 0usize);
    for c in text.chars() {
        if matches!(c, '\u{200E}' | '\u{200F}' | '\u{61C}') {
            continue;
        }
        match bidi_class(c) {
            BidiClass::R | BidiClass::AL => rtl += 1,
            BidiClass::L => ltr += 1,
            _ => {}
        }
    }
    rtl > ltr
}

/// Direction of the first strong character (Unicode bidi P2 and P3): `Some(true)` for RTL,
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

/// Alignment for a line of content text in a panel or table.
pub fn dir_align(text: &str) -> TextAlign {
    match first_strong_rtl(text) {
        Some(true) => TextAlign::Right,
        _ => TextAlign::Left,
    }
}

/// Text for a `text` widget that honours Right or Center on RTL lines. Display only: never put this
/// in a text input or editor value, or in a saved file.
///
/// iced 0.14 draws a one-line RTL paragraph with `align_x` Left/Right/Center at the wrong place, or
/// off the widget entirely. LRM makes the line's base direction LTR, so iced never takes that path,
/// and RLI..PDI keeps the content RTL.
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

/// Turn U+2029, U+0085 and U+001C..U+001E into `\n`. Run on all untrusted text before display.
///
/// `\r\n` and a lone `\r` become `\n` too. They don't panic, but a `\r` left inside a line would
/// fall between the marks of [`isolate_rtl`], which works line by line.
pub fn clean_separators(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                out.push('\n');
            }
            '\u{2029}' | '\u{85}' | '\u{1C}'..='\u{1E}' => out.push('\n'),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use unicode_bidi::BidiInfo;

    /// What cosmic-text uses for the line's base direction (shape.rs: `bidi.paragraphs[0]`).
    fn shaper_rtl(text: &str) -> bool {
        BidiInfo::new(text, None)
            .paragraphs
            .first()
            .is_some_and(|p| p.level.is_rtl())
    }

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
        let out = clean_separators("abc\u{2029}سلام\u{85}x\u{1C}y\u{1D}z\u{1E}w\r\nq\rr");
        for line in out.split('\n') {
            assert_eq!(BidiInfo::new(line, None).paragraphs.len(), 1, "{line:?}");
        }
        assert_eq!(out.matches('\n').count(), 7);
        assert!(!out.contains('\r'));
    }

    #[test]
    fn no_paragraph_separator_survives_cleaning() {
        // every character the bidi tables class as a paragraph break, so a new one can't slip by
        for c in (0..=0x10FFFFu32).filter_map(char::from_u32) {
            if bidi_class(c) == BidiClass::B {
                let cleaned = clean_separators(&c.to_string());
                assert_eq!(cleaned, "\n", "U+{:04X}", c as u32);
            }
        }
    }

    #[test]
    fn clean_leaves_ordinary_text_alone() {
        let s = "سلام، می\u{200C}روم\nabc\t def";
        assert_eq!(clean_separators(s), s);
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
        assert_eq!(dir_align("نسخه 5G را API"), TextAlign::Right);
        // digits are not strong, but the G in 5G is: this line's base direction is LTR
        assert_eq!(dir_align("5G نسخه"), TextAlign::Left);
        assert_eq!(dir_align("12 نسخه"), TextAlign::Right);
        assert_eq!(dir_align("؟"), TextAlign::Right); // U+061F is class AL, so it counts as strong
        assert_eq!(dir_align("API نسخه"), TextAlign::Left);
        assert_eq!(dir_align("\u{200F}API نسخه"), TextAlign::Right);
        assert_eq!(dir_align("12:30"), TextAlign::Left);
    }

    #[test]
    fn mostly_rtl_counts_letters_and_ignores_marks() {
        assert!(is_rtl_text("نسخه 5G را API"));
        assert!(is_rtl_text("שלום עולם"));
        assert!(!is_rtl_text("the API نسخه"));
        assert!(!is_rtl_text("Hello"));
        assert!(!is_rtl_text("123 !?"));
        assert!(!is_rtl_text(""));
        assert!(!is_rtl_text("\u{200F}\u{200F}\u{200F}abcd"));
    }

    #[test]
    fn table_is_consistent() {
        let mut codes: Vec<&str> = LANGUAGES.iter().map(|l| l.code).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), LANGUAGES.len(), "duplicate code");
        assert!(
            LANGUAGES.windows(2).all(|w| w[0].name < w[1].name),
            "not sorted by name"
        );
    }

    #[test]
    fn rtl_flags_agree_with_expected_scripts() {
        for l in LANGUAGES {
            let scripts = expected_scripts(l.code);
            if scripts.contains(&Script::Arabic) || scripts.contains(&Script::Hebrew) {
                assert!(l.rtl, "{} writes RTL", l.code);
            } else if !scripts.is_empty() {
                assert!(!l.rtl, "{} writes LTR", l.code);
            }
        }
        assert!(language("dv").unwrap().rtl); // its script is not in `Script`
    }

    #[test]
    fn lookup_falls_back_to_the_language() {
        assert_eq!(language("fa").unwrap().name, "Persian");
        assert_eq!(language("FA").unwrap().name, "Persian");
        assert_eq!(language("fa-IR").unwrap().name, "Persian");
        assert_eq!(language("pt_BR").unwrap().name, "Portuguese");
        assert_eq!(language("zh-Hant").unwrap().name, "Chinese (Traditional)");
        assert_eq!(language("zh").unwrap().name, "Chinese (Simplified)");
        assert_eq!(language("xx"), None);
        assert_eq!(language(""), None);
    }

    #[test]
    fn expected_scripts_by_language() {
        assert_eq!(expected_scripts("fa"), [Script::Arabic]);
        assert_eq!(expected_scripts("zh-Hant"), [Script::Han]);
        assert_eq!(expected_scripts("ja"), [Script::Han, Script::Kana]);
        assert_eq!(expected_scripts("sr"), [Script::Cyrillic, Script::Latin]);
        assert_eq!(expected_scripts("bn"), []);
        assert_eq!(expected_scripts("xx"), []);
    }

    #[test]
    fn scripts_of_letters() {
        for (c, want) in [
            ('a', Some(Script::Latin)),
            ('é', Some(Script::Latin)),
            ('ß', Some(Script::Latin)),
            ('پ', Some(Script::Arabic)),
            ('ی', Some(Script::Arabic)),
            ('ש', Some(Script::Hebrew)),
            ('я', Some(Script::Cyrillic)),
            ('λ', Some(Script::Greek)),
            ('ह', Some(Script::Devanagari)),
            ('漢', Some(Script::Han)),
            ('か', Some(Script::Kana)),
            ('カ', Some(Script::Kana)),
            ('한', Some(Script::Hangul)),
            ('ก', Some(Script::Thai)),
            ('5', None),
            ('۵', None),
            ('؟', None),
            ('.', None),
            ('×', None),
            ('ব', None),
        ] {
            assert_eq!(script_of(c), want, "{c}");
        }
    }
}
