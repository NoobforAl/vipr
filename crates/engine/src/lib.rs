//! ViPr engine library.
//!
//! Phase 1 contains the model registry and the downloader. Audio, ASR, diarization,
//! refine, translate and export stages are added in later phases (see PLAN.md).

pub mod audio;
pub mod chapters;
pub mod compare;
pub mod diar;
pub mod diff;
pub mod download;
pub mod fsio;
pub mod lang;
pub mod marks;
pub mod profiles;
pub mod project;
pub mod prompts;
pub mod registry;
pub mod sanitize;
pub mod settings;
pub mod subalign;
pub mod subparse;
pub mod subtitles;
pub mod tools;
pub mod transcript;
pub mod translate;

/// Folder name under the per-user data / config dirs (`%LOCALAPPDATA%\ViPr`, ...).
pub const APP_DIR_NAME: &str = "ViPr";

pub use gpu_detect;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error on {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("invalid registry {path}: {msg}")]
    Registry { path: String, msg: String },
    #[error("invalid data in {path}: {msg}")]
    Json { path: String, msg: String },
    #[error("audio: {0}")]
    Audio(String),
    #[error("unknown model id '{0}'")]
    UnknownModel(String),
    #[error("model '{0}' is a local file and cannot be downloaded")]
    NotDownloadable(String),
    #[error("download failed: {0}")]
    Http(String),
    #[error("sha256 mismatch for {id}: expected {expected}, got {actual}")]
    HashMismatch {
        id: String,
        expected: String,
        actual: String,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn io_err(path: &std::path::Path) -> impl FnOnce(std::io::Error) -> Error + '_ {
    move |source| Error::Io {
        path: path.display().to_string(),
        source,
    }
}
