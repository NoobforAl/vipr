//! Compare prompts, reply parsing and guards. Pure code used by `llm-worker` and the app.
//!
//! Stub from the foundation step; agent B fills in the bodies.

use crate::project::{AiPick, CompareInRow, CompareRow};

/// Longest reason kept from the model, in characters.
pub const MAX_REASON_CHARS: usize = 120;

/// True for rows with text on both sides that differ. All others never reach the LLM.
pub fn needs_llm(_row: &CompareRow) -> bool {
    todo!("agent B")
}

/// True when merged text is really built from A and B: bigram coverage from A∪B ≥ 0.7 and length
/// ≤ 1.3× the longer side.
pub fn check_merged(_merged: &str, _a: &str, _b: &str) -> bool {
    todo!("agent B")
}

/// `(system, user)` prompt for a chunk of rows. Rows are JSON-encoded so they can't forge new rows,
/// A and B are swapped by row-id parity, and all text goes through `sanitize_untrusted`.
pub fn build_prompt(_rows: &[CompareInRow], _profile_prompt: Option<&str>) -> (String, String) {
    todo!("agent B")
}

/// Parses `id|A|reason`, `id|B|reason` and `id|M|merged|reason`. Undoes the parity swap, runs
/// `check_merged`, and caps reasons at [`MAX_REASON_CHARS`]. Rows without a usable answer are absent.
pub fn parse_reply(_reply: &str, _rows: &[CompareInRow]) -> Vec<(u64, AiPick)> {
    todo!("agent B")
}
