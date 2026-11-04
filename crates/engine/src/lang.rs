//! Languages and direction-aware display of content text.
//!
//! Stub from the foundation step; agent A fills in the bodies. A working prototype of the direction
//! helpers and their tests is in `spikes/rtl/src/dir.rs`. Findings from that spike:
//! - `dir_align` alone makes short one-line RTL text vanish in iced 0.14, so subtitles use
//!   `isolate_rtl` for display, and panels use a right-aligned container with Default-aligned text.
//! - U+2029, U+0085 and U+001C..U+001E between text of differing direction panic in cosmic-text, so
//!   `clean_separators` must run on untrusted text before any widget gets it.

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

/// All supported target languages. Stub: empty until agent A fills it.
pub const LANGUAGES: &[Language] = &[];

pub fn language(_code: &str) -> Option<&'static Language> {
    todo!("agent A")
}

/// Scripts a correct translation into `code` may be written in. Empty when unknown, which turns the
/// wrong-script guard off.
pub fn expected_scripts(_code: &str) -> &'static [Script] {
    todo!("agent A")
}

/// True when `text` is written mostly in an RTL script.
pub fn is_rtl_text(_text: &str) -> bool {
    todo!("agent A")
}

/// Direction of the first strong character (Unicode bidi P2 and P3): `Some(true)` for RTL,
/// `Some(false)` for LTR, `None` when the first paragraph has no strong character.
pub fn first_strong_rtl(_text: &str) -> Option<bool> {
    todo!("agent A")
}

/// Alignment for a line of content text in a panel or table.
pub fn dir_align(_text: &str) -> TextAlign {
    todo!("agent A")
}

/// Text for a `text` widget that honours Right or Center on RTL lines. Display only: never put this
/// in a text input or editor value, or in a saved file.
pub fn isolate_rtl(_text: &str) -> String {
    todo!("agent A")
}

/// Turn U+2029, U+0085 and U+001C..U+001E into `\n`. Run on all untrusted text before display.
pub fn clean_separators(_text: &str) -> String {
    todo!("agent A")
}
