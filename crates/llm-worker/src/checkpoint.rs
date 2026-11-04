//! Resume file for translation: `<stem>.<lang>.ckpt.jsonl`, one `TranslationCue` per line.
//!
//! Stub from the foundation step; agent B fills in the bodies.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use engine::project::TranslationCue;

/// An open checkpoint that cues are appended to.
#[derive(Debug)]
pub struct Checkpoint {
    pub path: PathBuf,
}

impl Checkpoint {
    /// Returns the checkpoint and the cues already in it. A missing file means no cues, and a torn
    /// last line from a crash is ignored.
    pub fn open(_path: &Path) -> Result<(Checkpoint, Vec<TranslationCue>)> {
        bail!("checkpoint is not implemented yet")
    }

    /// Appends cues and flushes them to disk.
    pub fn append(&mut self, _cues: &[TranslationCue]) -> Result<()> {
        bail!("checkpoint is not implemented yet")
    }

    /// Deletes the file, after a successful run.
    pub fn finish(self) -> Result<()> {
        bail!("checkpoint is not implemented yet")
    }
}
