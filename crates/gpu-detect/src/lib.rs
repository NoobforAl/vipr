//! GPU detection: decides which inference backend the engine should use.
//!
//! * NVIDIA GPU with a working driver (found through NVML, loaded at run time) -> [`Backend::Cuda`]
//! * macOS                                                                      -> [`Backend::Metal`]
//! * any other real GPU (AMD, Intel, NVIDIA without NVML) found through wgpu     -> [`Backend::Vulkan`]
//! * nothing usable                                                             -> [`Backend::Cpu`]
//!
//! NVML is the only source of exact VRAM and CUDA details. wgpu only gives name / vendor / type,
//! so VRAM is `None` for non-NVIDIA cards (the model picker then shows "VRAM unknown").

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Cuda,
    Vulkan,
    Metal,
    Cpu,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Vulkan => "vulkan",
            Backend::Metal => "metal",
            Backend::Cpu => "cpu",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Vendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Other,
}

impl Vendor {
    fn from_pci(id: u32) -> Self {
        match id & 0xffff {
            0x10de => Vendor::Nvidia,
            0x1002 | 0x1022 => Vendor::Amd,
            0x8086 => Vendor::Intel,
            0x106b => Vendor::Apple,
            _ => Vendor::Other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Nvml,
    Wgpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vendor: Vendor,
    /// Total VRAM in MiB. Only known for NVIDIA cards (NVML).
    pub vram_mb: Option<u64>,
    /// Free VRAM in MiB at detection time (NVML only).
    pub vram_free_mb: Option<u64>,
    pub driver: Option<String>,
    /// Highest CUDA version the driver supports, e.g. "13.4" (NVML only).
    pub cuda_driver: Option<String>,
    /// CUDA compute capability, e.g. "8.6" (NVML only).
    pub compute_capability: Option<String>,
    pub integrated: bool,
    pub source: Source,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub os: String,
    pub gpus: Vec<GpuInfo>,
    pub recommended: Backend,
    /// Index into `gpus` of the card the recommended backend should use.
    pub recommended_gpu: Option<usize>,
    /// Human readable explanation, shown in the GUI.
    pub reason: String,
    /// Non-fatal problems met while probing (missing driver, no Vulkan, ...).
    pub notes: Vec<String>,
}

/// Probe the machine. Never panics and never needs a GPU; the worst result is `Backend::Cpu`.
pub fn detect() -> Detection {
    let mut notes = Vec::new();
    let mut gpus = probe_nvml(&mut notes);
    let nvml_count = gpus.len();
    for g in probe_wgpu(&mut notes) {
        // NVML already described every NVIDIA card it can see; wgpu would only repeat them
        let already = g.vendor == Vendor::Nvidia && nvml_count > 0;
        if !already {
            gpus.push(g);
        }
    }
    decide(gpus, notes)
}

/// NVIDIA-only detection (NVML). Never touches the graphics stack, so it is safe to call inside a running
/// GUI process: creating a second wgpu instance there can make the GUI's own graphics device fail
/// ("Parent device is lost"). The GUI normally runs the `detect` tool in a separate process instead and only
/// uses this as a fallback.
pub fn detect_without_graphics() -> Detection {
    let mut notes =
        vec!["Graphics adapters were not probed; only NVIDIA cards are listed.".to_string()];
    let gpus = probe_nvml(&mut notes);
    decide(gpus, notes)
}

fn decide(gpus: Vec<GpuInfo>, notes: Vec<String>) -> Detection {
    let os = std::env::consts::OS;

    let best = |pred: &dyn Fn(&GpuInfo) -> bool| -> Option<usize> {
        gpus.iter()
            .enumerate()
            .filter(|(_, g)| pred(g))
            // prefer discrete cards, then the one with the most VRAM
            .max_by_key(|(_, g)| (!g.integrated, g.vram_mb.unwrap_or(0)))
            .map(|(i, _)| i)
    };

    let (recommended, idx, reason) = if let Some(i) = best(&|g| g.source == Source::Nvml) {
        let g = &gpus[i];
        (
            Backend::Cuda,
            Some(i),
            format!(
                "NVIDIA GPU with a working driver: {} ({} MiB, CUDA {})",
                g.name,
                g.vram_mb.unwrap_or(0),
                g.cuda_driver.as_deref().unwrap_or("?")
            ),
        )
    } else if os == "macos" {
        let i = best(&|_| true);
        (Backend::Metal, i, "macOS: using Metal".to_string())
    } else if let Some(i) = best(&|_| true) {
        let g = &gpus[i];
        let why = if g.vendor == Vendor::Nvidia {
            "NVIDIA GPU found but its driver (NVML) is not usable, so CUDA is skipped; using Vulkan"
        } else {
            "No NVIDIA driver; using Vulkan on the available GPU"
        };
        (Backend::Vulkan, Some(i), format!("{why}: {}", g.name))
    } else {
        (
            Backend::Cpu,
            None,
            "No usable GPU found; running on the CPU".to_string(),
        )
    };

    Detection {
        os: os.to_string(),
        gpus,
        recommended,
        recommended_gpu: idx,
        reason,
        notes,
    }
}

fn probe_nvml(notes: &mut Vec<String>) -> Vec<GpuInfo> {
    use nvml_wrapper::Nvml;
    let nvml = match Nvml::init() {
        Ok(n) => n,
        Err(e) => {
            notes.push(format!(
                "NVML not available ({e}); no NVIDIA driver or no NVIDIA GPU"
            ));
            return Vec::new();
        }
    };
    let driver = nvml.sys_driver_version().ok();
    let cuda_driver = nvml.sys_cuda_driver_version().ok().map(|v| {
        format!(
            "{}.{}",
            nvml_wrapper::cuda_driver_version_major(v),
            nvml_wrapper::cuda_driver_version_minor(v)
        )
    });
    let count = nvml.device_count().unwrap_or(0);
    let mut out = Vec::new();
    for i in 0..count {
        let Ok(dev) = nvml.device_by_index(i) else {
            continue;
        };
        let mem = dev.memory_info().ok();
        out.push(GpuInfo {
            name: dev.name().unwrap_or_else(|_| format!("NVIDIA GPU {i}")),
            vendor: Vendor::Nvidia,
            vram_mb: mem.as_ref().map(|m| m.total / (1024 * 1024)),
            vram_free_mb: mem.as_ref().map(|m| m.free / (1024 * 1024)),
            driver: driver.clone(),
            cuda_driver: cuda_driver.clone(),
            compute_capability: dev
                .cuda_compute_capability()
                .ok()
                .map(|c| format!("{}.{}", c.major, c.minor)),
            integrated: false,
            source: Source::Nvml,
        });
    }
    out
}

fn probe_wgpu(notes: &mut Vec<String>) -> Vec<GpuInfo> {
    use wgpu::{Backends, DeviceType, Instance};
    let instance = Instance::default();
    let adapters = pollster::block_on(
        instance.enumerate_adapters(Backends::VULKAN | Backends::DX12 | Backends::METAL),
    );
    if adapters.is_empty() {
        notes.push("wgpu found no graphics adapters (no Vulkan / DX12 / Metal driver?)".into());
    }
    let mut seen: Vec<(u32, u32)> = Vec::new();
    let mut out = Vec::new();
    for a in adapters {
        let info = a.get_info();
        if matches!(info.device_type, DeviceType::Cpu) {
            continue; // software renderers (WARP, llvmpipe) are not a GPU for us
        }
        // the same card shows up once per graphics API (Vulkan + DX12)
        if seen.contains(&(info.vendor, info.device)) {
            continue;
        }
        seen.push((info.vendor, info.device));
        out.push(GpuInfo {
            name: info.name,
            vendor: Vendor::from_pci(info.vendor),
            vram_mb: None,
            vram_free_mb: None,
            driver: if info.driver.is_empty() {
                None
            } else {
                Some(info.driver)
            },
            cuda_driver: None,
            compute_capability: None,
            integrated: matches!(info.device_type, DeviceType::IntegratedGpu),
            source: Source::Wgpu,
        });
    }
    out
}

/// One reading of how busy the GPU is. Only NVIDIA cards report this (through NVML); everything is `None` elsewhere.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct GpuUsage {
    /// Share of time the GPU was busy, 0-100.
    pub gpu_percent: Option<u32>,
    pub vram_used_mb: Option<u64>,
    pub vram_total_mb: Option<u64>,
}

/// Keeps NVML open so that reading the usage once a second is cheap.
pub struct GpuMonitor {
    nvml: Option<nvml_wrapper::Nvml>,
}

impl GpuMonitor {
    pub fn new() -> Self {
        GpuMonitor {
            nvml: nvml_wrapper::Nvml::init().ok(),
        }
    }

    /// Usage of the first NVIDIA GPU, or all `None` when there is none.
    pub fn sample(&self) -> GpuUsage {
        let Some(nvml) = &self.nvml else {
            return GpuUsage::default();
        };
        let Ok(dev) = nvml.device_by_index(0) else {
            return GpuUsage::default();
        };
        let mem = dev.memory_info().ok();
        GpuUsage {
            gpu_percent: dev.utilization_rates().ok().map(|u| u.gpu),
            vram_used_mb: mem.as_ref().map(|m| m.used / (1024 * 1024)),
            vram_total_mb: mem.as_ref().map(|m| m.total / (1024 * 1024)),
        }
    }
}

impl Default for GpuMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(vendor: Vendor, source: Source, vram: Option<u64>, integrated: bool) -> GpuInfo {
        GpuInfo {
            name: format!("{vendor:?}"),
            vendor,
            vram_mb: vram,
            vram_free_mb: None,
            driver: None,
            cuda_driver: Some("13.4".into()),
            compute_capability: None,
            integrated,
            source,
        }
    }

    #[test]
    fn nvidia_with_driver_picks_cuda() {
        let d = decide(
            vec![
                gpu(Vendor::Intel, Source::Wgpu, None, true),
                gpu(Vendor::Nvidia, Source::Nvml, Some(4096), false),
            ],
            vec![],
        );
        assert_eq!(d.recommended, Backend::Cuda);
        assert_eq!(d.recommended_gpu, Some(1));
    }

    #[test]
    fn amd_picks_vulkan() {
        let d = decide(vec![gpu(Vendor::Amd, Source::Wgpu, None, false)], vec![]);
        if std::env::consts::OS != "macos" {
            assert_eq!(d.recommended, Backend::Vulkan);
        }
    }

    #[test]
    fn nvidia_without_nvml_falls_back_to_vulkan() {
        let d = decide(vec![gpu(Vendor::Nvidia, Source::Wgpu, None, false)], vec![]);
        if std::env::consts::OS != "macos" {
            assert_eq!(d.recommended, Backend::Vulkan);
            assert!(d.reason.contains("NVML"));
        }
    }

    #[test]
    fn nothing_picks_cpu() {
        let d = decide(vec![], vec![]);
        assert_eq!(d.recommended, Backend::Cpu);
        assert_eq!(d.recommended_gpu, None);
    }

    #[test]
    fn prefers_discrete_over_integrated() {
        let d = decide(
            vec![
                gpu(Vendor::Intel, Source::Wgpu, None, true),
                gpu(Vendor::Amd, Source::Wgpu, None, false),
            ],
            vec![],
        );
        if std::env::consts::OS != "macos" {
            assert_eq!(d.recommended_gpu, Some(1));
        }
    }
}
