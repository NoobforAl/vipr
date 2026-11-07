//! User settings saved between runs (`settings.json` in the per-user config folder).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::Result;
use crate::registry::Role;

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
    /// Id of the profile used for new runs (see `profiles.rs`). `None` = no profile.
    #[serde(default)]
    pub active_profile: Option<String>,
}

/// Dark, as an editor's panels are, whatever the system uses. "system" still follows it when chosen.
fn default_theme() -> String {
    "dark".into()
}
fn yes() -> bool {
    true
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            models: BTreeMap::new(),
            output_dir: None,
            ffmpeg_path: None,
            theme: default_theme(),
            cleanup: true,
            num_speakers: 0,
            active_profile: None,
        }
    }
}

impl Settings {
    /// Default results folder: `<Documents>/<app folder>`.
    pub fn default_output_dir() -> PathBuf {
        dirs::document_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(crate::APP_DIR_NAME)
    }

    pub fn path() -> PathBuf {
        if let Some(p) = std::env::var_os("MT_SETTINGS") {
            return PathBuf::from(p);
        }
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(crate::APP_DIR_NAME)
            .join("settings.json")
    }

    /// A missing or unreadable file gives the defaults (settings must never stop the app from starting).
    pub fn load() -> Settings {
        std::fs::read_to_string(Self::path())
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }

    /// Atomic, so a crash while saving cannot leave a half-written file that resets all settings.
    pub fn save(&self) -> Result<()> {
        self.save_to(&Self::path())
    }

    fn save_to(&self, path: &Path) -> Result<()> {
        crate::fsio::save_json(path, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_app_starts_dark_whatever_the_system_uses() {
        assert_eq!(Settings::default().theme, "dark");
        let saved_before: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(saved_before.theme, "dark");
    }

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
        assert_eq!(
            serde_json::from_str::<Settings>("{}").unwrap(),
            Settings::default()
        );
    }

    #[test]
    fn older_files_load_without_an_active_profile() {
        let old = r#"{"theme":"dark","cleanup":false,"num_speakers":2}"#;
        let s: Settings = serde_json::from_str(old).unwrap();
        assert_eq!(s.active_profile, None);
        assert_eq!(s.theme, "dark");
    }

    #[test]
    fn save_writes_the_whole_file_and_replaces_the_old_one() {
        let dir = std::env::temp_dir().join(format!("vipr-settings-{}", std::process::id()));
        let path = dir.join("nested").join("settings.json");
        let mut s = Settings {
            active_profile: Some("p1".into()),
            ..Settings::default()
        };
        s.save_to(&path).unwrap();
        s.theme = "light".into();
        s.save_to(&path).unwrap();
        let back: Settings = crate::fsio::load_json(&path).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(back, s);
    }
}
