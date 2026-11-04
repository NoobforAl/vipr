//! `--mode compare`: asks the model to judge rows.
//!
//! Stub from the foundation step; agent B fills in the body. Prompts, parsing and guards are in
//! `engine::compare`, and the JSON types in `engine::project`.

use anyhow::{Result, bail};
use engine::project::{CompareInput, CompareOutput};

use crate::refine::Chat;

/// Judges every input row. `progress` receives `(done, total)`.
pub fn compare(
    _chat: &mut dyn Chat,
    _input: &CompareInput,
    _progress: &mut dyn FnMut(usize, usize),
) -> Result<CompareOutput> {
    bail!("--mode compare is not implemented yet")
}
