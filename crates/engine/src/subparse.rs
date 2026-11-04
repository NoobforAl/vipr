//! Reading subtitle files: decoding bytes to text, then tolerant SRT, VTT and ASS parsing.
//!
//! Stub from the foundation step; agent A fills in the bodies.

use crate::project::{ImportWarning, SubFormat, TimedText};

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
    /// Ids count from 1 in file order. Text is as written in the file (still untrusted).
    pub cues: Vec<TimedText>,
    pub warnings: Vec<ImportWarning>,
}

/// BOM, then strict UTF-8, then `chardetng` + `encoding_rs`.
pub fn decode(_bytes: &[u8]) -> Decoded {
    todo!("agent A")
}

/// Decode with an encoding the user picked in the import preview. `None` for an unknown label.
pub fn decode_as(_bytes: &[u8], _label: &str) -> Option<Decoded> {
    todo!("agent A")
}

/// Parse decoded text. The format is detected from the content.
pub fn parse(_text: &str) -> Parsed {
    todo!("agent A")
}
