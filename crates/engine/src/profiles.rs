//! Named profiles: a prompt and glossary keywords, in `<config>/ViPr/profiles.json`.
//!
//! Stub from the foundation step; the types are final, agent D fills in the bodies. A corrupt file
//! is renamed to `.bak` with a visible notice, so user work is never silently reset.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::Result;

/// One glossary keyword. With `target`, it also fixes how the term is translated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlossaryTerm {
    pub source: String,
    #[serde(default)]
    pub target: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Profile {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub glossary: Vec<GlossaryTerm>,
}

/// The whole `profiles.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfilesFile {
    #[serde(default = "crate::project::version")]
    pub version: u32,
    #[serde(default)]
    pub profiles: Vec<Profile>,
}

/// In-memory profiles and the file they save to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileStore {
    pub path: PathBuf,
    pub profiles: Vec<Profile>,
}

/// Result of [`ProfileStore::load`]. It always yields a usable store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedProfiles {
    pub store: ProfileStore,
    /// What the user must be told, when the file was corrupt and moved to `.bak`.
    pub notice: Option<String>,
}

impl ProfileStore {
    /// `<config>/ViPr/profiles.json`.
    pub fn default_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(crate::APP_DIR_NAME)
            .join("profiles.json")
    }

    /// A missing file gives an empty store; a corrupt one is renamed to `.bak`, with a notice.
    pub fn load(_path: &Path) -> LoadedProfiles {
        todo!("agent D")
    }

    /// Atomic.
    pub fn save(&self) -> Result<()> {
        todo!("agent D")
    }

    pub fn get(&self, _id: &str) -> Option<&Profile> {
        todo!("agent D")
    }

    /// Adds a new profile with a fresh id and returns that id. Does not save.
    pub fn create(&mut self, _name: &str) -> String {
        todo!("agent D")
    }

    /// Replaces the profile with the same id. Returns false if there is none. Does not save.
    pub fn update(&mut self, _profile: Profile) -> bool {
        todo!("agent D")
    }

    /// Returns false if there is no such profile. Does not save.
    pub fn remove(&mut self, _id: &str) -> bool {
        todo!("agent D")
    }
}

/// Parses the keywords box: one term per line or comma separated, each `source` or `source=target`.
pub fn parse_glossary(_text: &str) -> Vec<GlossaryTerm> {
    todo!("agent D")
}

/// Formats terms back into the text [`parse_glossary`] reads.
pub fn format_glossary(_terms: &[GlossaryTerm]) -> String {
    todo!("agent D")
}
