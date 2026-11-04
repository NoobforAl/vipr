//! Model registry: the user-selectable list of models (`models.json` + optional `models.user.json`).

use std::collections::HashSet;
use std::path::Path;

use gpu_detect::{Backend, Detection};
use serde::{Deserialize, Serialize};

use crate::{Error, Result, io_err};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Role {
    Asr,
    DiarSegmentation,
    DiarEmbedding,
    Llm,
    Translate,
    /// Voice-activity detection (skips silence, stops Whisper hallucinating in it).
    Vad,
}

impl Role {
    pub const ALL: [Role; 6] =
        [Role::Asr, Role::DiarSegmentation, Role::DiarEmbedding, Role::Llm, Role::Translate, Role::Vad];

    pub fn as_str(self) -> &'static str {
        match self {
            Role::Asr => "asr",
            Role::DiarSegmentation => "diar-segmentation",
            Role::DiarEmbedding => "diar-embedding",
            Role::Llm => "llm",
            Role::Translate => "translate",
            Role::Vad => "vad",
        }
    }

    /// The only model format the engine can run for this role.
    pub fn required_format(self) -> Format {
        match self {
            Role::Asr | Role::Vad => Format::WhisperGgml,
            Role::DiarSegmentation | Role::DiarEmbedding => Format::Onnx,
            Role::Llm | Role::Translate => Format::Gguf,
        }
    }
}

impl std::str::FromStr for Role {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        Role::ALL
            .into_iter()
            .find(|r| r.as_str() == s)
            .ok_or_else(|| format!("unknown role '{s}' (expected one of: asr, diar-segmentation, diar-embedding, llm, translate, vad)"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Format {
    WhisperGgml,
    Onnx,
    Gguf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Preset {
    Fast,
    Balanced,
    Best,
}

impl std::str::FromStr for Preset {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s {
            "fast" => Ok(Preset::Fast),
            "balanced" => Ok(Preset::Balanced),
            "best" => Ok(Preset::Best),
            _ => Err(format!("unknown preset '{s}' (fast, balanced, best)")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Source {
    /// A file in a Hugging Face repo.
    Hf {
        repo: String,
        file: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        revision: Option<String>,
    },
    /// Any direct download link.
    Url { url: String },
    /// A file the user already has on disk (never downloaded or deleted by us).
    Local { path: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub roles: Vec<Role>,
    pub format: Format,
    pub source: Source,
    /// File name on disk; defaults to the last part of the source path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(default)]
    pub size_bytes: u64,
    /// Lower-case hex SHA-256. Entries without a hash are shown as "unverified".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub languages: Vec<String>,
    /// Estimated GPU memory needed (MiB). 0 = runs fine on CPU / tiny.
    #[serde(default)]
    pub min_vram_mb: u64,
    pub backends: Vec<Backend>,
    #[serde(default)]
    pub presets: Vec<Preset>,
    /// The download needs a Hugging Face token (gated model).
    #[serde(default)]
    pub gated: bool,
    #[serde(default)]
    pub notes: String,
}

impl ModelEntry {
    pub fn file_name(&self) -> String {
        if let Some(f) = &self.filename {
            return f.clone();
        }
        let tail = match &self.source {
            Source::Hf { file, .. } => file.as_str(),
            Source::Url { url } => url.split('?').next().unwrap_or(url),
            Source::Local { path } => path.as_str(),
        };
        tail.rsplit(['/', '\\']).next().unwrap_or(tail).to_string()
    }

    pub fn download_url(&self) -> Option<String> {
        match &self.source {
            Source::Hf { repo, file, revision } => {
                Some(format!("https://huggingface.co/{repo}/resolve/{}/{file}", revision.as_deref().unwrap_or("main")))
            }
            Source::Url { url } => Some(url.clone()),
            Source::Local { .. } => None,
        }
    }

    pub fn has_role(&self, role: Role) -> bool {
        self.roles.contains(&role)
    }
}

/// Will this model run on the detected hardware?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Fit {
    /// Enough VRAM with a safety margin (or CPU backend).
    Fits,
    /// Fits on paper but with less than the safety margin left.
    Tight,
    TooBig,
    /// GPU backend but VRAM is not known (non-NVIDIA).
    Unknown,
    /// The model does not list the backend we would use.
    Unsupported,
}

/// MiB kept free for the CUDA context, KV cache and activations.
const VRAM_MARGIN_MB: u64 = 400;

pub fn fit(entry: &ModelEntry, det: &Detection) -> Fit {
    if !entry.backends.contains(&det.recommended) {
        return Fit::Unsupported;
    }
    if det.recommended == Backend::Cpu {
        return Fit::Fits; // system RAM is not checked yet
    }
    let vram = det.recommended_gpu.and_then(|i| det.gpus.get(i)).and_then(|g| g.vram_mb);
    match vram {
        None => Fit::Unknown,
        Some(total) if entry.min_vram_mb + VRAM_MARGIN_MB <= total => Fit::Fits,
        Some(total) if entry.min_vram_mb <= total => Fit::Tight,
        Some(_) => Fit::TooBig,
    }
}

/// Find `models.json`: search upwards from the current folder, then from the executable's folder.
pub fn locate_registry() -> Option<std::path::PathBuf> {
    let mut starts = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        starts.push(cwd);
    }
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.to_path_buf())) {
        starts.push(dir);
    }
    starts
        .iter()
        .flat_map(|s| s.ancestors())
        .map(|d| d.join("models.json"))
        .find(|p| p.is_file())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Registry {
    pub version: u32,
    pub models: Vec<ModelEntry>,
    #[serde(default, rename = "_comment", skip_serializing)]
    _comment: Option<String>,
}

impl Registry {
    pub fn load(path: &Path) -> Result<Registry> {
        let text = std::fs::read_to_string(path).map_err(io_err(path))?;
        let reg: Registry = serde_json::from_str(&text)
            .map_err(|e| Error::Registry { path: path.display().to_string(), msg: e.to_string() })?;
        reg.validate().map_err(|msg| Error::Registry { path: path.display().to_string(), msg })?;
        Ok(reg)
    }

    /// Built-in list plus the user's own list; a user entry with the same id replaces the built-in one.
    pub fn load_with_user(builtin: &Path, user: Option<&Path>) -> Result<Registry> {
        let mut reg = Registry::load(builtin)?;
        if let Some(u) = user.filter(|u| u.exists()) {
            for m in Registry::load(u)?.models {
                match reg.models.iter_mut().find(|x| x.id == m.id) {
                    Some(slot) => *slot = m,
                    None => reg.models.push(m),
                }
            }
        }
        Ok(reg)
    }

    /// Load the registry from `explicit`, or from the located `models.json`, plus `models.user.json` next to it.
    pub fn load_default(explicit: Option<&Path>) -> Result<Registry> {
        let path = match explicit {
            Some(p) => p.to_path_buf(),
            None => locate_registry().ok_or_else(|| Error::Registry {
                path: "models.json".into(),
                msg: "not found; pass --registry <path> or set MT_REGISTRY".into(),
            })?,
        };
        let user = path.with_file_name("models.user.json");
        Registry::load_with_user(&path, Some(&user))
    }

    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.version != 1 {
            return Err(format!("unsupported registry version {} (this build reads version 1)", self.version));
        }
        let mut ids = HashSet::new();
        for m in &self.models {
            let id = &m.id;
            if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) {
                return Err(format!("model id '{id}' must be non-empty and use only letters, digits, - _ ."));
            }
            if !ids.insert(id.as_str()) {
                return Err(format!("duplicate model id '{id}'"));
            }
            if m.roles.is_empty() {
                return Err(format!("{id}: needs at least one role"));
            }
            for r in &m.roles {
                if r.required_format() != m.format {
                    return Err(format!(
                        "{id}: role '{}' needs format {:?} but the entry says {:?}",
                        r.as_str(),
                        r.required_format(),
                        m.format
                    ));
                }
            }
            if m.backends.is_empty() {
                return Err(format!("{id}: needs at least one backend"));
            }
            if let Some(h) = &m.sha256 {
                if h.len() != 64 || !h.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')) {
                    return Err(format!("{id}: sha256 must be 64 lower-case hex characters"));
                }
            }
            let name = m.file_name();
            if name.is_empty() || name.contains(['/', '\\']) || name == ".." {
                return Err(format!("{id}: bad file name '{name}'"));
            }
        }
        Ok(())
    }

    pub fn get(&self, id: &str) -> Result<&ModelEntry> {
        self.models.iter().find(|m| m.id == id).ok_or_else(|| Error::UnknownModel(id.to_string()))
    }

    pub fn for_role(&self, role: Role) -> impl Iterator<Item = &ModelEntry> {
        self.models.iter().filter(move |m| m.has_role(role))
    }

    /// Default model for a role on this machine: the biggest model of the preset that still fits,
    /// else the smallest model of the role that is not too big.
    pub fn recommend(&self, role: Role, preset: Preset, det: &Detection) -> Option<&ModelEntry> {
        let ok = |m: &&ModelEntry| matches!(fit(m, det), Fit::Fits | Fit::Unknown);
        self.for_role(role)
            .filter(|m| m.presets.contains(&preset))
            .filter(ok)
            .max_by_key(|m| m.min_vram_mb)
            .or_else(|| self.for_role(role).filter(|m| fit(m, det) != Fit::Unsupported && fit(m, det) != Fit::TooBig).min_by_key(|m| m.min_vram_mb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_detect::{GpuInfo, Source as GSource, Vendor};

    fn root_models() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models.json")
    }

    fn det(backend: Backend, vram: Option<u64>) -> Detection {
        Detection {
            os: "windows".into(),
            gpus: vec![GpuInfo {
                name: "test".into(),
                vendor: Vendor::Nvidia,
                vram_mb: vram,
                vram_free_mb: None,
                driver: None,
                cuda_driver: None,
                compute_capability: None,
                integrated: false,
                source: GSource::Nvml,
            }],
            recommended: backend,
            recommended_gpu: Some(0),
            reason: String::new(),
            notes: vec![],
        }
    }

    #[test]
    fn builtin_registry_is_valid_and_complete() {
        let reg = Registry::load(&root_models()).expect("models.json must load");
        for role in Role::ALL {
            assert!(reg.for_role(role).next().is_some(), "no model for role {}", role.as_str());
        }
        assert!(reg.models.iter().all(|m| m.sha256.is_some() && m.size_bytes > 0));
    }

    #[test]
    fn rejects_wrong_format_for_role() {
        let json = r#"{"version":1,"models":[{"id":"x","name":"x","roles":["llm"],"format":"onnx",
            "source":{"kind":"url","url":"http://a/b.onnx"},"backends":["cpu"]}]}"#;
        let reg: Registry = serde_json::from_str(json).unwrap();
        assert!(reg.validate().unwrap_err().contains("needs format"));
    }

    #[test]
    fn rejects_duplicate_ids_and_bad_hash() {
        let one = r#"{"id":"x","name":"x","roles":["llm"],"format":"gguf","source":{"kind":"url","url":"http://a/b.gguf"},"backends":["cpu"]"#;
        let dup: Registry = serde_json::from_str(&format!(r#"{{"version":1,"models":[{one}}},{one}}}]}}"#)).unwrap();
        assert!(dup.validate().unwrap_err().contains("duplicate"));
        let bad: Registry = serde_json::from_str(&format!(r#"{{"version":1,"models":[{one},"sha256":"ABC"}}]}}"#)).unwrap();
        assert!(bad.validate().unwrap_err().contains("sha256"));
    }

    #[test]
    fn fit_levels() {
        let reg = Registry::load(&root_models()).unwrap();
        let q5 = reg.get("qwen3-4b-instruct-2507-q5_k_m").unwrap(); // needs 3800
        assert_eq!(fit(q5, &det(Backend::Cuda, Some(4096))), Fit::Tight);
        assert_eq!(fit(q5, &det(Backend::Cuda, Some(2048))), Fit::TooBig);
        assert_eq!(fit(q5, &det(Backend::Cuda, Some(8192))), Fit::Fits);
        assert_eq!(fit(q5, &det(Backend::Cuda, None)), Fit::Unknown);
        assert_eq!(fit(q5, &det(Backend::Cpu, None)), Fit::Fits);
    }

    #[test]
    fn recommends_for_a_4gb_cuda_card() {
        let reg = Registry::load(&root_models()).unwrap();
        let d = det(Backend::Cuda, Some(4096));
        assert_eq!(reg.recommend(Role::Asr, Preset::Balanced, &d).unwrap().id, "whisper-large-v3-turbo-q8_0");
        assert_eq!(reg.recommend(Role::Llm, Preset::Best, &d).unwrap().id, "qwen3-4b-instruct-2507-q4_k_m");
    }

    #[test]
    fn file_names_and_urls() {
        let reg = Registry::load(&root_models()).unwrap();
        let m = reg.get("whisper-large-v3-turbo-q5_0").unwrap();
        assert_eq!(m.file_name(), "ggml-large-v3-turbo-q5_0.bin");
        assert_eq!(
            m.download_url().unwrap(),
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin"
        );
    }
}
