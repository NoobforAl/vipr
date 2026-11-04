//! `--mode translate`: drives the model over all cues.
//!
//! Stub from the foundation step; agent B fills in the body. Prompts, parsing and guards are in
//! `engine::translate`, and the JSON types in `engine::project`.

use std::path::Path;

use anyhow::{Result, bail};
use engine::project::{TranslateInput, TranslateOutput};

use crate::refine::Chat;

/// Translates every cue, resuming from `checkpoint` if it exists and deleting it on success.
/// `progress` receives `(done, total)`.
pub fn translate(
    _chat: &mut dyn Chat,
    _input: &TranslateInput,
    _checkpoint: Option<&Path>,
    _progress: &mut dyn FnMut(usize, usize),
) -> Result<TranslateOutput> {
    bail!("--mode translate is not implemented yet")
}
