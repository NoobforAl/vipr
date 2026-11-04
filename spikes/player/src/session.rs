//! Playback session: ffmpeg children, the audio stream, the master clock, seek and pause.

use crate::stats::{Kind, Stats};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, OutputCallbackInfo, SampleFormat, SizedSample};
use rtrb::{Consumer, Producer, RingBuffer};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::*};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const SEEK_COALESCE: Duration = Duration::from_millis(150);
const LAG_RESTART_SECS: f64 = 2.0;
const AUDIO_PREFILL_SECS: f64 = 0.1;

#[derive(Clone, Debug)]
pub struct Config {
    pub ffmpeg: PathBuf,
    pub input: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub audio_track: usize,
    pub has_audio: bool,
    pub use_audio_device: bool,
    pub muted: bool,
    pub out_dir: PathBuf,
    pub duration: f64,
    /// `-threads` for the video decoder; `None` lets ffmpeg decide.
    pub threads: Option<u32>,
    /// Test only: hold back the first frame, as a cold disk or slow share would.
    pub first_frame_delay: Duration,
}

#[derive(Debug)]
pub struct Frame {
    pub seq: u64,
    pub generation: u64,
    pub pts: f64,
    pub width: u32,
    pub height: u32,
    /// Y, then U, then V, tightly packed.
    pub data: Vec<u8>,
}

impl Frame {
    pub fn empty() -> Self {
        Self { seq: 0, generation: 0, pts: 0.0, width: 0, height: 0, data: Vec::new() }
    }
    pub fn planes(&self) -> (&[u8], &[u8], &[u8]) {
        let y = (self.width * self.height) as usize;
        let c = (self.width / 2 * (self.height / 2)) as usize;
        (&self.data[..y], &self.data[y..y + c], &self.data[y + c..y + 2 * c])
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------------------------------------------------------

/// Written only by the audio callback, read by the video thread and the renderer (a small seqlock).
#[derive(Default)]
struct CallbackPoint {
    seq: AtomicU64,
    frames_before: AtomicU64,
    at_ns: AtomicU64,
    latency_ns: AtomicU64,
    consumed: AtomicU64,
}

pub struct Clock {
    epoch: Instant,
    /// Position of frame 0 of the current audio ring, f64 bits.
    base: AtomicU64,
    rate: AtomicU64,
    point: CallbackPoint,
    audio_master: AtomicBool,
    /// While set, `now` returns `frozen_value`: paused, or seeking until the first new picture.
    frozen: AtomicBool,
    frozen_value: AtomicU64,
    /// Wall clock fallback: position = base + (now - wall_started).
    wall_started_ns: AtomicU64,
    /// Filled by the callback, reported in the stats.
    pub last_latency_ns: AtomicU64,
}

impl Clock {
    fn new() -> Self {
        Self {
            epoch: Instant::now(),
            base: AtomicU64::new(0f64.to_bits()),
            rate: AtomicU64::new(48000),
            point: CallbackPoint::default(),
            audio_master: AtomicBool::new(false),
            frozen: AtomicBool::new(true),
            frozen_value: AtomicU64::new(0f64.to_bits()),
            wall_started_ns: AtomicU64::new(0),
            last_latency_ns: AtomicU64::new(0),
        }
    }

    fn ns(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }

    pub fn is_audio_master(&self) -> bool {
        self.audio_master.load(Relaxed)
    }

    pub fn now(&self) -> f64 {
        if self.frozen.load(Acquire) {
            return f64::from_bits(self.frozen_value.load(Acquire));
        }
        let base = f64::from_bits(self.base.load(Acquire));
        if !self.audio_master.load(Relaxed) {
            return base + self.ns().saturating_sub(self.wall_started_ns.load(Acquire)) as f64 * 1e-9;
        }
        let rate = self.rate.load(Relaxed) as f64;
        let p = &self.point;
        loop {
            let s0 = p.seq.load(Acquire);
            if s0 & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let frames = p.frames_before.load(Acquire);
            let at = p.at_ns.load(Acquire);
            let latency = p.latency_ns.load(Acquire);
            let consumed = p.consumed.load(Acquire);
            if p.seq.load(Acquire) != s0 {
                continue;
            }
            // The first frame of the last callback is heard at at + latency. Time moves on inside that callback's
            // data only, so an underrun (consumed = 0) stops the clock instead of letting it run ahead of the sound.
            let since = (self.ns().saturating_sub(at) as f64 * 1e-9).min(consumed as f64 / rate);
            let pos = frames as f64 / rate + since - latency as f64 * 1e-9;
            return base + pos.max(0.0);
        }
    }

    fn freeze_at(&self, pos: f64) {
        self.frozen_value.store(pos.to_bits(), Release);
        self.frozen.store(true, Release);
    }

    /// Wall clock only; with audio the callback unfreezes on its first real samples.
    fn unfreeze_wall(&self) {
        let pos = f64::from_bits(self.frozen_value.load(Acquire));
        self.base.store(pos.to_bits(), Release);
        self.wall_started_ns.store(self.ns(), Release);
        self.frozen.store(false, Release);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------------------------------------------------

pub struct Shared {
    pub latest: Mutex<Arc<Frame>>,
    pub clock: Clock,
    pub stats: Arc<Stats>,
    pub generation: AtomicU64,
    pub paused: AtomicBool,
    pub ended: AtomicBool,
    pub stop: AtomicBool,
    /// Audio stays silent and the clock stays frozen until the first new picture exists.
    hold: AtomicBool,
    /// Set by the first `prepare`. Creating the window takes seconds (fonts, GPU wake-up), and the clock must not run
    /// through that time, or the first seconds of the video play to nobody.
    pub surface_ready: AtomicBool,
    pub frame_interval: f64,
}

// ---------------------------------------------------------------------------------------------------------------------
// ffmpeg
// ---------------------------------------------------------------------------------------------------------------------

fn no_console(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    cmd.creation_flags(0x0800_0000);
}

fn log_file(dir: &Path, name: &str) -> Stdio {
    match OpenOptions::new().create(true).append(true).open(dir.join(name)) {
        Ok(f) => Stdio::from(f),
        Err(_) => Stdio::null(),
    }
}

fn record_pid(cfg: &Config, kind: &str, pid: u32) {
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(cfg.out_dir.join("pids.txt")) {
        let _ = writeln!(f, "{kind} {pid}");
    }
}

fn spawn_video(cfg: &Config, start: f64) -> std::io::Result<Child> {
    let mut cmd = Command::new(&cfg.ffmpeg);
    cmd.args(["-nostdin", "-hide_banner", "-loglevel", "warning"]);
    if let Some(t) = cfg.threads {
        cmd.args(["-threads", &t.to_string()]);
    }
    if start > 0.0 {
        cmd.args(["-ss", &format!("{start:.3}")]);
    }
    let vf = format!(
        "fps={},scale={}:{}:flags=fast_bilinear:out_color_matrix=bt709:out_range=tv,setsar=1,format=yuv420p",
        cfg.fps, cfg.width, cfg.height
    );
    cmd.arg("-i")
        .arg(&cfg.input)
        .args(["-map", "0:v:0", "-an", "-sn", "-dn", "-vf", &vf, "-f", "rawvideo", "pipe:1"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(log_file(&cfg.out_dir, "ffmpeg_video.log"));
    no_console(&mut cmd);
    cmd.spawn()
}

fn spawn_audio(cfg: &Config, start: f64, channels: u16, rate: u32) -> std::io::Result<Child> {
    let mut cmd = Command::new(&cfg.ffmpeg);
    cmd.args(["-nostdin", "-hide_banner", "-loglevel", "warning"]);
    if start > 0.0 {
        cmd.args(["-ss", &format!("{start:.3}")]);
    }
    cmd.arg("-i")
        .arg(&cfg.input)
        .args(["-map", &format!("0:a:{}", cfg.audio_track), "-vn", "-sn", "-dn"])
        .args(["-ac", &channels.to_string(), "-ar", &rate.to_string(), "-f", "f32le", "pipe:1"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(log_file(&cfg.out_dir, "ffmpeg_audio.log"));
    no_console(&mut cmd);
    cmd.spawn()
}

// ---------------------------------------------------------------------------------------------------------------------
// Video reader thread
// ---------------------------------------------------------------------------------------------------------------------

fn frame_size(w: u32, h: u32) -> usize {
    (w * h + 2 * (w / 2 * (h / 2))) as usize
}

/// Latest-wins with recycled buffers: reuse a pooled frame nobody else holds, else allocate.
fn take_buffer(pool: &mut Vec<Arc<Frame>>, size: usize, stats: &Stats) -> Arc<Frame> {
    if let Some(i) = pool.iter().position(|f| Arc::strong_count(f) == 1) {
        return pool.swap_remove(i);
    }
    stats.c.frame_allocs.fetch_add(1, Relaxed);
    Arc::new(Frame {
        seq: 0,
        generation: 0,
        pts: 0.0,
        width: 0,
        height: 0,
        data: vec![0; size],
    })
}

struct VideoJob {
    shared: Arc<Shared>,
    cfg: Config,
    generation: u64,
    start: f64,
    stdout: std::process::ChildStdout,
    seek_tx: mpsc::Sender<SeekCmd>,
}

fn video_thread(mut job: VideoJob) {
    let VideoJob { shared, cfg, generation, start, seek_tx, .. } = &job;
    let (shared, cfg, generation, start, seek_tx) =
        (shared.clone(), cfg.clone(), *generation, *start, seek_tx.clone());
    let size = frame_size(cfg.width, cfg.height);
    let interval = 1.0 / cfg.fps;
    let stats = shared.stats.clone();
    let mut pool: Vec<Arc<Frame>> = Vec::new();
    let mut n: u64 = 0;
    let mut seq_base = 0u64;
    if let Ok(g) = shared.latest.lock() {
        seq_base = g.seq;
    }
    let current = |shared: &Shared| shared.generation.load(Acquire) == generation && !shared.stop.load(Relaxed);

    loop {
        let mut buf = take_buffer(&mut pool, size, &stats);
        let Some(frame) = Arc::get_mut(&mut buf) else {
            continue;
        };
        if job.stdout.read_exact(&mut frame.data).is_err() {
            if current(&shared) {
                shared.ended.store(true, Release);
            }
            return;
        }
        if !current(&shared) {
            return;
        }
        let pts = start + n as f64 * interval;
        frame.seq = seq_base + n + 1;
        frame.generation = generation;
        frame.pts = pts;
        frame.width = cfg.width;
        frame.height = cfg.height;
        stats.c.frames_read.fetch_add(1, Relaxed);
        let first = n == 0;
        n += 1;

        if first {
            stats.seek_first_read(generation);
            std::thread::sleep(cfg.first_frame_delay);
            if !current(&shared) {
                return;
            }
        } else {
            // wait if early
            loop {
                let d = pts - shared.clock.now();
                if d <= 0.0 {
                    break;
                }
                std::thread::sleep(Duration::from_secs_f64(d.min(0.01)));
                if !current(&shared) {
                    return;
                }
            }
            let late = shared.clock.now() - pts;
            if late > LAG_RESTART_SECS {
                stats.c.lag_restarts.fetch_add(1, Relaxed);
                let _ = seek_tx.send(SeekCmd::Internal(shared.clock.now() + 0.5));
                return;
            }
            // skip publishing if more than one interval late
            if late > interval {
                stats.c.skipped_late.fetch_add(1, Relaxed);
                pool.push(buf);
                continue;
            }
        }

        if let Ok(mut g) = shared.latest.lock() {
            *g = buf.clone();
        }
        stats.c.published.fetch_add(1, Relaxed);
        pool.push(buf);
        if first {
            while !shared.surface_ready.load(Acquire) {
                std::thread::sleep(Duration::from_millis(5));
                if !current(&shared) {
                    return;
                }
            }
            shared.hold.store(false, Release);
            if !shared.clock.is_audio_master() && !shared.paused.load(Acquire) {
                shared.clock.unfreeze_wall();
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------------------------------------------------

struct Swap {
    consumer: Consumer<f32>,
}

struct Callback {
    shared: Arc<Shared>,
    channels: usize,
    rate: u32,
    consumer: Option<Consumer<f32>>,
    swap_rx: Consumer<Swap>,
    /// Old rings go back to a normal thread, so the callback never frees memory.
    return_tx: Producer<Consumer<f32>>,
    frames: u64,
    ready: bool,
    gain: f32,
}

impl Callback {
    fn publish(&self, frames_before: u64, consumed: u64, latency_ns: u64) {
        let clock = &self.shared.clock;
        let p = &clock.point;
        p.seq.fetch_add(1, AcqRel);
        p.frames_before.store(frames_before, Release);
        p.at_ns.store(clock.ns(), Release);
        p.latency_ns.store(latency_ns, Release);
        p.consumed.store(consumed, Release);
        p.seq.fetch_add(1, AcqRel);
    }

    fn render<T: SizedSample + FromSample<f32>>(&mut self, out: &mut [T], info: &OutputCallbackInfo) {
        let ts = info.timestamp();
        let latency_ns = ts.playback.duration_since(ts.callback).as_nanos() as u64;
        let clock = &self.shared.clock;
        clock.last_latency_ns.store(latency_ns, Relaxed);

        while let Ok(swap) = self.swap_rx.pop() {
            if let Some(old) = self.consumer.replace(swap.consumer) {
                let _ = self.return_tx.push(old);
            }
            self.frames = 0;
            self.ready = false;
        }

        let prefill = (self.rate as f64 * AUDIO_PREFILL_SECS) as usize * self.channels;
        let hold = self.shared.hold.load(Acquire);
        let Some(consumer) = self.consumer.as_mut() else {
            out.fill(T::EQUILIBRIUM);
            return;
        };
        if !self.ready && !hold && (consumer.slots() >= prefill || consumer.is_abandoned()) {
            self.ready = true;
        }
        if hold || !self.ready {
            out.fill(T::EQUILIBRIUM);
            self.publish(self.frames, 0, latency_ns);
            return;
        }

        let want = out.len() - out.len() % self.channels;
        let available = consumer.slots();
        let n = (want.min(available) / self.channels) * self.channels;
        if n > 0
            && let Ok(chunk) = consumer.read_chunk(n)
        {
            let (a, b) = chunk.as_slices();
            for (o, s) in out.iter_mut().zip(a.iter().chain(b.iter())) {
                *o = T::from_sample(*s * self.gain);
            }
            chunk.commit_all();
        }
        for o in &mut out[n..] {
            *o = T::EQUILIBRIUM;
        }
        if n < want && !consumer.is_abandoned() {
            self.shared.stats.c.underruns.fetch_add(1, Relaxed);
        }
        let consumed = (n / self.channels) as u64;
        self.publish(self.frames, consumed, latency_ns);
        self.frames += consumed;
        if consumed > 0 {
            clock.frozen.store(false, Release);
        }
    }
}

struct AudioOut {
    stream: cpal::Stream,
    channels: u16,
    rate: u32,
    format: SampleFormat,
    swap_tx: Producer<Swap>,
    return_rx: Consumer<Consumer<f32>>,
}

fn build_stream<T: SizedSample + FromSample<f32> + Send + 'static>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    mut cb: Callback,
    stats: Arc<Stats>,
) -> Result<cpal::Stream, cpal::Error> {
    device.build_output_stream::<T, _, _>(
        config,
        move |out: &mut [T], info: &OutputCallbackInfo| cb.render(out, info),
        move |err| {
            stats.c.audio_errors.fetch_add(1, Relaxed);
            log::error!("audio stream error: {err}");
        },
        None,
    )
}

fn open_audio(shared: &Arc<Shared>, muted: bool) -> Result<AudioOut, String> {
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or("no output device")?;
    let supported = device.default_output_config().map_err(|e| e.to_string())?;
    let format = supported.sample_format();
    let config = supported.config();
    let (channels, rate) = (config.channels, config.sample_rate);
    let (swap_tx, swap_rx) = RingBuffer::new(1);
    let (return_tx, return_rx) = RingBuffer::new(8);
    let cb = Callback {
        shared: shared.clone(),
        channels: channels as usize,
        rate,
        consumer: None,
        swap_rx,
        return_tx,
        frames: 0,
        ready: false,
        gain: if muted { 0.0 } else { 1.0 },
    };
    let stats = shared.stats.clone();
    let stream = match format {
        SampleFormat::F32 => build_stream::<f32>(&device, config, cb, stats),
        SampleFormat::I16 => build_stream::<i16>(&device, config, cb, stats),
        SampleFormat::I24 => build_stream::<cpal::I24>(&device, config, cb, stats),
        SampleFormat::I32 => build_stream::<i32>(&device, config, cb, stats),
        other => return Err(format!("unhandled device format {other:?}")),
    }
    .map_err(|e| e.to_string())?;
    Ok(AudioOut { stream, channels, rate, format, swap_tx, return_rx })
}

fn audio_feeder(mut stdout: std::process::ChildStdout, mut producer: Producer<f32>) {
    let mut bytes = vec![0u8; 16384];
    let mut carry = 0usize;
    loop {
        let got = match stdout.read(&mut bytes[carry..]) {
            Ok(0) | Err(_) => return,
            Ok(n) => n,
        };
        let total = carry + got;
        let whole = total - total % 4;
        let mut offset = 0;
        while offset < whole {
            if producer.is_abandoned() {
                return;
            }
            let left = (whole - offset) / 4;
            let n = left.min(producer.slots());
            if n == 0 {
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }
            if let Ok(mut chunk) = producer.write_chunk(n) {
                let (a, b) = chunk.as_mut_slices();
                for s in a.iter_mut().chain(b.iter_mut()) {
                    let c = &bytes[offset..offset + 4];
                    *s = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    offset += 4;
                }
                chunk.commit_all();
            }
        }
        bytes.copy_within(whole..total, 0);
        carry = total - whole;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------------------------------------------------

enum SeekCmd {
    User(f64, Instant),
    Internal(f64),
    Stop,
}

struct Children {
    video: Option<Child>,
    audio: Option<Child>,
}

struct Audio {
    out: AudioOut,
    pending: Option<Swap>,
}

struct Inner {
    shared: Arc<Shared>,
    cfg: Config,
    children: Mutex<Children>,
    audio: Mutex<Option<Audio>>,
    seek_tx: mpsc::Sender<SeekCmd>,
}

pub struct Session {
    inner: Arc<Inner>,
    pub audio_desc: String,
}

fn kill(child: &mut Option<Child>) {
    if let Some(mut c) = child.take() {
        let _ = c.kill();
        let _ = c.wait();
    }
}

impl Inner {
    fn start_at(self: &Arc<Self>, target: f64) {
        let shared = &self.shared;
        shared.clock.freeze_at(target);
        shared.hold.store(true, Release);
        shared.ended.store(false, Release);
        let generation = shared.generation.fetch_add(1, AcqRel) + 1;

        let mut children = self.children.lock().expect("children");
        kill(&mut children.video);
        kill(&mut children.audio);

        if let Some(audio) = self.audio.lock().expect("audio").as_mut() {
            let out = &mut audio.out;
            while out.return_rx.pop().is_ok() {}
            match spawn_audio(&self.cfg, target, out.channels, out.rate) {
                Ok(mut child) => {
                    shared.stats.track_child(Kind::Audio, child.id());
                    record_pid(&self.cfg, "audio", child.id());
                    let capacity = out.rate as usize * out.channels as usize * 2;
                    let (producer, consumer) = RingBuffer::new(capacity);
                    if let Some(stdout) = child.stdout.take() {
                        std::thread::spawn(move || audio_feeder(stdout, producer));
                    }
                    children.audio = Some(child);
                    shared.clock.base.store(target.to_bits(), Release);
                    audio.pending = Some(Swap { consumer });
                    if !shared.paused.load(Acquire) {
                        flush_swap(audio);
                    }
                }
                Err(e) => log::error!("audio ffmpeg: {e}"),
            }
        }

        match spawn_video(&self.cfg, target) {
            Ok(mut child) => {
                shared.stats.track_child(Kind::Video, child.id());
                record_pid(&self.cfg, "video", child.id());
                if let Some(stdout) = child.stdout.take() {
                    let job = VideoJob {
                        shared: shared.clone(),
                        cfg: self.cfg.clone(),
                        generation,
                        start: target,
                        stdout,
                        seek_tx: self.seek_tx.clone(),
                    };
                    std::thread::spawn(move || video_thread(job));
                }
                children.video = Some(child);
            }
            Err(e) => log::error!("video ffmpeg: {e}"),
        }
    }
}

/// The callback does not run while paused, and its swap slot holds one entry.
fn flush_swap(audio: &mut Audio) {
    if let Some(swap) = audio.pending.take()
        && let Err(rtrb::PushError::Full(swap)) = audio.out.swap_tx.push(swap)
    {
        audio.pending = Some(swap);
    }
}

fn seek_worker(inner: Arc<Inner>, rx: mpsc::Receiver<SeekCmd>) {
    // only a seek from the slider is measured; a lag restart keeps the screen going by itself
    let mut waiting: Option<(f64, Option<Instant>)> = None;
    loop {
        let cmd = if waiting.is_some() {
            match rx.recv_timeout(SEEK_COALESCE) {
                Ok(c) => Some(c),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => return,
            }
        } else {
            match rx.recv() {
                Ok(c) => Some(c),
                Err(_) => return,
            }
        };
        match cmd {
            Some(SeekCmd::User(t, at)) => waiting = Some((t, Some(at))),
            Some(SeekCmd::Internal(t)) => {
                if waiting.is_none() {
                    waiting = Some((t, None));
                }
            }
            Some(SeekCmd::Stop) => return,
            None => {
                if let Some((target, requested)) = waiting.take() {
                    if let Some(at) = requested {
                        let next = inner.shared.generation.load(Acquire) + 1;
                        inner.shared.stats.begin_seek(next, at, target);
                    }
                    inner.start_at(target);
                }
            }
        }
    }
}

impl Session {
    pub fn start(cfg: Config, shared: Arc<Shared>, start_pos: f64) -> Self {
        let (seek_tx, seek_rx) = mpsc::channel();
        let mut audio_desc = "wall clock (no audio stream used)".to_string();
        let mut audio = None;
        if cfg.has_audio && cfg.use_audio_device {
            match open_audio(&shared, cfg.muted) {
                Ok(out) => {
                    audio_desc = format!(
                        "cpal default device: {:?} {} ch {} Hz, track {}{}",
                        out.format,
                        out.channels,
                        out.rate,
                        cfg.audio_track,
                        if cfg.muted { " (gain 0: muted)" } else { "" }
                    );
                    shared.clock.rate.store(out.rate as u64, Relaxed);
                    shared.clock.audio_master.store(true, Relaxed);
                    if let Err(e) = out.stream.play() {
                        log::error!("audio play: {e}");
                    }
                    audio = Some(Audio { out, pending: None });
                }
                Err(e) => {
                    audio_desc = format!("wall clock (audio device failed: {e})");
                    log::warn!("{audio_desc}");
                }
            }
        }
        let inner = Arc::new(Inner {
            shared,
            cfg,
            children: Mutex::new(Children { video: None, audio: None }),
            audio: Mutex::new(audio),
            seek_tx,
        });
        inner.start_at(start_pos);
        let worker = inner.clone();
        std::thread::spawn(move || seek_worker(worker, seek_rx));
        Self { inner, audio_desc }
    }

    /// Coalesced: the last request wins after 150 ms of quiet.
    pub fn seek(&self, target: f64) {
        let max = (self.inner.cfg.duration - 0.5).max(0.0);
        let _ = self.inner.seek_tx.send(SeekCmd::User(target.clamp(0.0, max), Instant::now()));
    }

    pub fn seek_sender(&self) -> impl Fn(f64) + Send + 'static {
        let tx = self.inner.seek_tx.clone();
        let max = (self.inner.cfg.duration - 0.5).max(0.0);
        move |t: f64| {
            let _ = tx.send(SeekCmd::User(t.clamp(0.0, max), Instant::now()));
        }
    }

    pub fn set_paused(&self, paused: bool) {
        let shared = &self.inner.shared;
        if shared.paused.swap(paused, AcqRel) == paused {
            return;
        }
        let mut guard = self.inner.audio.lock().expect("audio");
        if paused {
            shared.clock.freeze_at(shared.clock.now());
            if let Some(audio) = guard.as_ref()
                && let Err(e) = audio.out.stream.pause()
            {
                log::error!("audio pause: {e}");
            }
        } else if let Some(audio) = guard.as_mut() {
            flush_swap(audio);
            if let Err(e) = audio.out.stream.play() {
                log::error!("audio play: {e}");
            }
        } else if !shared.hold.load(Acquire) {
            shared.clock.unfreeze_wall();
        }
    }

    pub fn shutdown(&self) {
        let shared = &self.inner.shared;
        shared.stop.store(true, Release);
        let _ = self.inner.seek_tx.send(SeekCmd::Stop);
        if let Ok(mut c) = self.inner.children.lock() {
            kill(&mut c.video);
            kill(&mut c.audio);
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub fn new_shared(stats: Arc<Stats>, fps: f64) -> Arc<Shared> {
    Arc::new(Shared {
        latest: Mutex::new(Arc::new(Frame::empty())),
        clock: Clock::new(),
        stats,
        generation: AtomicU64::new(0),
        paused: AtomicBool::new(false),
        ended: AtomicBool::new(false),
        stop: AtomicBool::new(false),
        hold: AtomicBool::new(true),
        surface_ready: AtomicBool::new(false),
        frame_interval: 1.0 / fps,
    })
}
