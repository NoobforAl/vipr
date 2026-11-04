//! Bookmarks and flagged subtitle lines of one recording (`<stem>.marks.json`).
//!
//! Stub from the foundation step; the types are final, agent D fills in the bodies.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MarkKind {
    /// A moment, with an optional note.
    Bookmark,
    /// A subtitle line that needs attention.
    Flag,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mark {
    /// Unique within the file. Never reused after a delete.
    pub id: u64,
    pub kind: MarkKind,
    /// Seconds from the start of the media. Authoritative: `cue` is only a hint, since cue ids
    /// can change when subtitles are rebuilt.
    pub time: f64,
    #[serde(default)]
    pub note: String,
    /// Id of the flagged cue, when there is one.
    #[serde(default)]
    pub cue: Option<u64>,
}

/// The whole `<stem>.marks.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarksFile {
    #[serde(default = "crate::project::version")]
    pub version: u32,
    #[serde(default)]
    pub marks: Vec<Mark>,
}

/// One recording's marks and the file they save to.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkStore {
    pub path: PathBuf,
    /// Sorted by `time`.
    pub marks: Vec<Mark>,
}

impl MarkStore {
    /// A missing file gives an empty store. A corrupt one is an error, so nothing overwrites it.
    pub fn load(_path: &Path) -> Result<MarkStore> {
        todo!("agent D")
    }

    /// Atomic.
    pub fn save(&self) -> Result<()> {
        todo!("agent D")
    }

    /// Adds a mark in time order and returns its id. Does not save.
    pub fn add(&mut self, _kind: MarkKind, _time: f64, _note: String, _cue: Option<u64>) -> u64 {
        todo!("agent D")
    }

    /// Returns false if there is no such mark. Does not save.
    pub fn set_note(&mut self, _id: u64, _note: String) -> bool {
        todo!("agent D")
    }

    /// Returns false if there is no such mark. Does not save.
    pub fn remove(&mut self, _id: u64) -> bool {
        todo!("agent D")
    }
}
