//! Counters, per-phase summaries and per-process CPU accounting.

use serde_json::{Value, json};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};
use windows_sys::Win32::Foundation::FILETIME;
use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, GetProcessTimes, GetSystemTimes, OpenProcess,
    PROCESS_QUERY_LIMITED_INFORMATION,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Video,
    Audio,
}

#[derive(Default)]
pub struct Counters {
    pub frames_read: AtomicU64,
    pub published: AtomicU64,
    pub skipped_late: AtomicU64,
    pub uploaded: AtomicU64,
    pub view_calls: AtomicU64,
    pub redraws: AtomicU64,
    pub underruns: AtomicU64,
    pub audio_errors: AtomicU64,
    pub device_lost: AtomicU64,
    pub uncaptured_gpu_errors: AtomicU64,
    pub lag_restarts: AtomicU64,
    pub frame_allocs: AtomicU64,
    pub ffmpeg_spawned: AtomicU64,
}

impl Counters {
    fn snapshot(&self) -> [u64; 13] {
        let g = |a: &AtomicU64| a.load(Relaxed);
        [
            g(&self.frames_read),
            g(&self.published),
            g(&self.skipped_late),
            g(&self.uploaded),
            g(&self.view_calls),
            g(&self.redraws),
            g(&self.underruns),
            g(&self.audio_errors),
            g(&self.device_lost),
            g(&self.uncaptured_gpu_errors),
            g(&self.lag_restarts),
            g(&self.frame_allocs),
            g(&self.ffmpeg_spawned),
        ]
    }
}

const COUNTER_NAMES: [&str; 13] = [
    "frames_read",
    "published",
    "skipped_late",
    "uploaded",
    "view_calls",
    "redraws",
    "underruns",
    "audio_errors",
    "device_lost",
    "uncaptured_gpu_errors",
    "lag_restarts",
    "frame_allocs",
    "ffmpeg_spawned",
];

#[derive(Default, Clone, Copy)]
pub struct CpuSample {
    pub app: f32,
    pub video: f32,
    pub audio: f32,
    pub system: f32,
    pub app_rss_mb: f32,
}

#[derive(Default)]
struct Series {
    clock_err_ms: Vec<f32>,
    upload_gap_ms: Vec<f32>,
    upload_cost_ms: Vec<f32>,
    cpu: Vec<CpuSample>,
    last_upload: Option<Instant>,
}

struct Phase {
    name: String,
    started: Instant,
    clock_start: f64,
    counters: [u64; 13],
    note: String,
}

pub struct SeekRecord {
    pub generation: u64,
    pub requested: Instant,
    pub target: f64,
    pub first_read_ms: Option<f64>,
}

#[derive(Default)]
pub struct Stats {
    pub c: Counters,
    series: Mutex<Series>,
    phase: Mutex<Option<Phase>>,
    phases: Mutex<Vec<Value>>,
    pending_seek: Mutex<Option<SeekRecord>>,
    seeks: Mutex<Vec<Value>>,
    children: Mutex<Vec<(Kind, u32, isize)>>,
    pub surface_format: Mutex<String>,
    pub device_lost_log: Mutex<Vec<String>>,
}

fn secs(f: FILETIME) -> f64 {
    (((f.dwHighDateTime as u64) << 32) | f.dwLowDateTime as u64) as f64 * 100e-9
}

fn process_cpu_secs(handle: *mut core::ffi::c_void) -> f64 {
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut c, mut e, mut k, mut u) = (zero, zero, zero, zero);
    let ok = unsafe { GetProcessTimes(handle, &mut c, &mut e, &mut k, &mut u) };
    if ok == 0 { 0.0 } else { secs(k) + secs(u) }
}

fn system_busy_secs() -> f64 {
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut i, mut k, mut u) = (zero, zero, zero);
    if unsafe { GetSystemTimes(&mut i, &mut k, &mut u) } == 0 {
        return 0.0;
    }
    // kernel time includes idle
    secs(k) + secs(u) - secs(i)
}

fn rss_mb() -> f32 {
    let mut m: PROCESS_MEMORY_COUNTERS = unsafe { core::mem::zeroed() };
    m.cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut m, m.cb) };
    if ok == 0 { 0.0 } else { m.WorkingSetSize as f32 / 1048576.0 }
}

pub fn percentile(values: &[f32], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v: Vec<f32> = values.to_vec();
    v.sort_by(f32::total_cmp);
    let i = ((v.len() - 1) as f64 * p).round() as usize;
    v[i] as f64
}

fn dist(values: &[f32]) -> Value {
    if values.is_empty() {
        return json!({"n": 0});
    }
    let mean = values.iter().map(|v| *v as f64).sum::<f64>() / values.len() as f64;
    json!({
        "n": values.len(),
        "mean": mean,
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": percentile(values, 1.0),
    })
}

impl Stats {
    /// The handle stays open, so `GetProcessTimes` still works after the process has been killed.
    pub fn track_child(&self, kind: Kind, pid: u32) {
        let h = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if !h.is_null()
            && let Ok(mut g) = self.children.lock()
        {
            g.push((kind, pid, h as isize));
        }
        self.c.ffmpeg_spawned.fetch_add(1, Relaxed);
    }

    /// (app, video children, audio children) CPU seconds so far, including children already killed.
    fn cpu_totals(&self) -> (f64, f64, f64) {
        let app = process_cpu_secs(unsafe { GetCurrentProcess() });
        let (mut video, mut audio) = (0.0, 0.0);
        if let Ok(g) = self.children.lock() {
            for (kind, _, h) in g.iter() {
                let t = process_cpu_secs(*h as *mut core::ffi::c_void);
                match kind {
                    Kind::Video => video += t,
                    Kind::Audio => audio += t,
                }
            }
        }
        (app, video, audio)
    }

    pub fn spawn_cpu_sampler(self: &std::sync::Arc<Self>) {
        let s = self.clone();
        std::thread::spawn(move || {
            let mut last = s.cpu_totals();
            let mut last_sys = system_busy_secs();
            let mut last_at = Instant::now();
            loop {
                std::thread::sleep(Duration::from_secs(1));
                let now = s.cpu_totals();
                let sys = system_busy_secs();
                let dt = last_at.elapsed().as_secs_f64();
                last_at = Instant::now();
                let sample = CpuSample {
                    app: ((now.0 - last.0) / dt) as f32,
                    video: ((now.1 - last.1) / dt) as f32,
                    audio: ((now.2 - last.2) / dt) as f32,
                    system: ((sys - last_sys) / dt) as f32,
                    app_rss_mb: rss_mb(),
                };
                last = now;
                last_sys = sys;
                if let Ok(mut g) = s.series.lock() {
                    g.cpu.push(sample);
                }
            }
        });
    }

    pub fn record_upload(&self, clock_err_ms: f32, cost_ms: f32) {
        self.c.uploaded.fetch_add(1, Relaxed);
        if let Ok(mut s) = self.series.lock() {
            s.clock_err_ms.push(clock_err_ms);
            s.upload_cost_ms.push(cost_ms);
            let now = Instant::now();
            if let Some(prev) = s.last_upload.replace(now) {
                s.upload_gap_ms.push(prev.elapsed().as_secs_f32() * 1000.0);
            }
        }
    }

    pub fn begin_seek(&self, generation: u64, requested: Instant, target: f64) {
        if let Ok(mut g) = self.pending_seek.lock() {
            *g = Some(SeekRecord { generation, requested, target, first_read_ms: None });
        }
    }

    pub fn seek_first_read(&self, generation: u64) {
        if let Ok(mut g) = self.pending_seek.lock()
            && let Some(r) = g.as_mut()
            && r.generation == generation
            && r.first_read_ms.is_none()
        {
            r.first_read_ms = Some(r.requested.elapsed().as_secs_f64() * 1000.0);
        }
    }

    /// Called at texture upload: this is the moment the new picture enters the frame being drawn.
    pub fn seek_first_upload(&self, generation: u64) {
        let Ok(mut g) = self.pending_seek.lock() else { return };
        if g.as_ref().is_some_and(|r| r.generation == generation)
            && let Some(r) = g.take()
            && let Ok(mut out) = self.seeks.lock()
        {
            out.push(json!({
                "target": r.target,
                "to_first_read_ms": r.first_read_ms,
                "to_first_upload_ms": r.requested.elapsed().as_secs_f64() * 1000.0,
                "system_cores_busy": self.series.lock().ok().and_then(|s| s.cpu.last().map(|c| c.system)),
            }));
        }
    }

    pub fn begin_phase(&self, name: &str, clock_now: f64) {
        let phase = Phase {
            name: name.to_string(),
            started: Instant::now(),
            clock_start: clock_now,
            counters: self.c.snapshot(),
            note: String::new(),
        };
        if let Ok(mut s) = self.series.lock() {
            *s = Series::default();
        }
        if let Ok(mut g) = self.phase.lock() {
            *g = Some(phase);
        }
    }

    pub fn end_phase(&self, clock_now: f64, extra: Value) {
        let Some(phase) = self.phase.lock().ok().and_then(|mut g| g.take()) else { return };
        let wall = phase.started.elapsed().as_secs_f64();
        let now = self.c.snapshot();
        let mut counters = serde_json::Map::new();
        for i in 0..now.len() {
            counters.insert(COUNTER_NAMES[i].into(), json!(now[i] - phase.counters[i]));
        }
        let d = |i: usize| (now[i] - phase.counters[i]) as f64;
        let (read, uploaded) = (d(0), d(3));
        let drop_pct = if read > 0.0 { (read - uploaded) / read * 100.0 } else { f64::NAN };
        let s = self.series.lock();
        let summary = s.map(|s| {
            let abs: Vec<f32> = s.clock_err_ms.iter().map(|v| v.abs()).collect();
            let col = |f: fn(&CpuSample) -> f32| s.cpu.iter().map(f).collect::<Vec<f32>>();
            let total: Vec<f32> = s.cpu.iter().map(|c| c.app + c.video + c.audio).collect();
            json!({
                "clock_error_abs_ms": dist(&abs),
                "clock_error_signed_ms": dist(&s.clock_err_ms),
                "upload_gap_ms": dist(&s.upload_gap_ms),
                "upload_cost_ms": dist(&s.upload_cost_ms),
                "cpu_cores": {
                    "total": dist(&total),
                    "app": dist(&col(|c| c.app)),
                    "ffmpeg_video": dist(&col(|c| c.video)),
                    "ffmpeg_audio": dist(&col(|c| c.audio)),
                    "whole_system": dist(&col(|c| c.system)),
                },
                "app_rss_mb": dist(&col(|c| c.app_rss_mb)),
            })
        });
        let view_per_sec = d(4) / wall;
        let upload_per_sec = uploaded / wall;
        let value = json!({
            "name": phase.name,
            "note": phase.note,
            "wall_secs": wall,
            "clock_secs": clock_now - phase.clock_start,
            "counters": counters,
            "drop_pct_read_vs_uploaded": drop_pct,
            "view_per_sec": view_per_sec,
            "uploads_per_sec": upload_per_sec,
            "series": summary.unwrap_or(Value::Null),
            "extra": extra,
        });
        if let Ok(mut g) = self.phases.lock() {
            g.push(value);
        }
    }

    pub fn current_uploaded(&self) -> u64 {
        self.c.uploaded.load(Relaxed)
    }

    pub fn final_json(&self, meta: Value) -> Value {
        let seeks = self.seeks.lock().map(|g| g.clone()).unwrap_or_default();
        let totals: Vec<f32> = seeks
            .iter()
            .filter_map(|s| s["to_first_upload_ms"].as_f64().map(|v| v as f32))
            .collect();
        let reads: Vec<f32> = seeks
            .iter()
            .filter_map(|s| s["to_first_read_ms"].as_f64().map(|v| v as f32))
            .collect();
        let pids: Vec<u32> = self
            .children
            .lock()
            .map(|g| g.iter().map(|c| c.1).collect())
            .unwrap_or_default();
        json!({
            "meta": meta,
            "surface_format": self.surface_format.lock().map(|g| g.clone()).unwrap_or_default(),
            "device_lost_log": self.device_lost_log.lock().map(|g| g.clone()).unwrap_or_default(),
            "totals": {
                "device_lost": self.c.device_lost.load(Relaxed),
                "uncaptured_gpu_errors": self.c.uncaptured_gpu_errors.load(Relaxed),
                "lag_restarts": self.c.lag_restarts.load(Relaxed),
                "frame_allocs": self.c.frame_allocs.load(Relaxed),
                "ffmpeg_spawned": self.c.ffmpeg_spawned.load(Relaxed),
            },
            "phases": self.phases.lock().map(|g| g.clone()).unwrap_or_default(),
            "seek_summary": {
                "to_first_upload_ms": dist(&totals),
                "to_first_read_ms": dist(&reads),
                "note": "measured from the last slider event, so it includes the 150 ms coalescing window",
            },
            "seeks": seeks,
            "ffmpeg_pids": pids,
        })
    }
}
