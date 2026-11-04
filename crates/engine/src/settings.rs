//! User settings saved between runs (`settings.json` in the per-user config folder).

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::registry::Role;
use crate::{Error, Result, io_err};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Settings {
    /// Chosen model id per role. Missing role = use the recommended model.
    #[serde(default)]
    pub models: BTreeMap<Role, String>,
    /// Where results are written. `None` = `<Documents>/<app folder>`.
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    /// `"system"`, `"light"` or `"dark"`.
    #[serde(default = "default_theme")]
    pub theme: String,
    /// Run the LLM cleanup step when its model and worker are available.
    #[serde(default = "yes")]
    pub cleanup: bool,
    /// An ffmpeg the user picked by hand (for formats the built-in decoders cannot read). `None` = search automatically.
    #[serde(default)]
    pub ffmpeg_path: Option<PathBuf>,
    /// Number of speakers: 0 = detect automatically (default), otherwise force this many.
    #[serde(default)]
    pub num_speakers: usize,
}

fn default_theme() -> String {
    "system".into()
}
fn yes() -> bool {
    true
}

impl Default for Settings {
    fn default() -> Self {
        Settings { models: BTreeMap::new(), output_dir: None, ffmpeg_path: None, theme: default_theme(), cleanup: true, num_speakers: 0 }
    }
}

impl Settings {
    /// Default results folder: `<Documents>/<app folder>`.
    pub fn default_output_dir() -> PathBuf {
        dirs::document_dir().unwrap_or_else(|| PathBuf::from(".")).join(crate::APP_DIR_NAME)
    }

    pub fn path() -> PathBuf {
        if let Some(p) = std::env::var_os("MT_SETTINGS") {
            return PathBuf::from(p);
        }
        dirs::config_dir().unwrap_or_else(|| PathBuf::from(".")).join(crate::APP_DIR_NAME).join("settings.json")
    }

    /// A missing or unreadable file gives the defaults (settings must never stop the app from starting).
    pub fn load() -> Settings {
        std::fs::read_to_string(Self::path()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::path();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(io_err(dir))?;
        }
        let json = serde_json::to_string_pretty(self).map_err(|e| Error::Registry { path: path.display().to_string(), msg: e.to_string() })?;
        std::fs::write(&path, json).map_err(io_err(&path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut s = Settings::default();
        s.models.insert(Role::Asr, "whisper-small-en-q5_1".into());
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<Settings>(&json).unwrap(), s);
        assert!(json.contains("\"asr\""));
    }

    #[test]
    fn garbage_file_gives_defaults() {
        assert_eq!(serde_json::from_str::<Settings>("{}").unwrap(), Settings::default());
    }
}
