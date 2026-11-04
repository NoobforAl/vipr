//! Translation prompts, reply parsing and language-agnostic guards. Pure code used by `llm-worker`.
//!
//! Stub from the foundation step; agent B fills in the bodies.

use crate::project::{CueFlag, TimedText};

/// The most cues in one model call.
pub const MAX_CHUNK_CUES: usize = 12;

/// Earlier cues with their translations, then the following source lines, shown as context only.
#[derive(Debug, Clone, Copy, Default)]
pub struct Context<'a> {
    /// Last 3 `(source, translation)` pairs.
    pub before: &'a [(String, String)],
    /// Next 2 source lines.
    pub after: &'a [String],
}

/// Why a translated line was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reject {
    /// Empty, or contains control characters.
    Empty,
    /// The reply echoed a cue id.
    EchoedId,
    /// Too long or too short for the source, by script.
    Length,
    /// Same as the source, where the source should have changed.
    Unchanged,
    /// Not written in a script of the target language.
    WrongScript,
    /// Runaway repetition.
    Repetition,
}

/// Returns the soft flags for a usable line, or why it is rejected.
pub fn check_line(_source: &str, _text: &str, _target: &str) -> Result<Vec<CueFlag>, Reject> {
    todo!("agent B")
}

/// `(system, user)` prompt for one chunk. Runs `sanitize_untrusted` on all text.
pub fn build_prompt(
    _cues: &[TimedText],
    _source_lang: &str,
    _target: &str,
    _context: Context<'_>,
    _profile_prompt: Option<&str>,
    _strict: bool,
) -> (String, String) {
    todo!("agent B")
}

/// Parses `id|translation` lines. Ids must match `expected` exactly and duplicates fail.
pub fn parse_reply(_reply: &str, _expected: &[u64]) -> Result<Vec<(u64, String)>, String> {
    todo!("agent B")
}

/// `max(4*source, 32) + 16*lines + 64`
pub fn token_cap(_source_tokens: usize, _lines: usize) -> usize {
    todo!("agent B")
}
