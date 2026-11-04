//! Playback session: ffmpeg children, the audio stream, the master clock, seek and pause.
//!
//! ffmpeg decodes in child processes and writes raw `yuv420p` and float samples to pipes. The video
//! thread publishes frames latest-wins and pipe backpressure throttles ffmpeg. A seek kills and
//! restarts both children, keeping the last frame on screen until the first new one exists.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::*};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, OutputCallbackInfo, SampleFormat, SizedSample};
use rtrb::{Consumer, Producer, RingBuffer};

/// A drag on the seek bar reaches ffmpeg after this much quiet.
const SEEK_COALESCE: Duration = Duration::from_millis(150);
/// Restart at the clock position when video falls this far behind.
const LAG_RESTART_SECS: f64 = 2.0;
/// Where a lag restart lands, ahead of the clock, since starting ffmpeg takes a moment.
const LAG_RESTART_AHEAD_SECS: f64 = 0.5;
const AUDIO_PREFILL_SECS: f64 = 0.1;
const AUDIO_RING_SECS: usize = 2;
/// Just under full scale: the I24 conversion does not saturate, so exactly 1.0 lands out of range.
const MAX_SAMPLE: f32 = 1.0 - f32::EPSILON / 2.0;
/// Buffers above this are freed, not kept, when the renderer briefly holds more than usual.
const POOL_MAX: usize = 8;

/// The only track played: `0:a:0`. There is no picker.
const AUDIO_MAP: &str = "0:a:0";
/// Capital V skips attached pictures.
const VIDEO_MAP: &str = "0:V:0";

#[derive(Clone, Debug)]
pub struct Config {
    pub ffmpeg: PathBuf,
    pub input: PathBuf,
    pub has_video: bool,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub has_audio: bool,
    /// False keeps the wall clock and never touches an audio device.
    pub use_audio_device: bool,
    /// Zero when the file doesn't say.
    pub duration: f64,
    pub volume: f32,
}

// ------------------------------------------------------------------------- frames

/// Unique across sessions, so the surface can't mistake a new session's frame for one it uploaded.
static NEXT_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub struct Frame {
    /// Zero only for [`Frame::empty`].
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
        Self {
            seq: 0,
            generation: 0,
            pts: 0.0,
            width: 0,
            height: 0,
            data: Vec::new(),
        }
    }

    pub fn planes(&self) -> (&[u8], &[u8], &[u8]) {
        let y = (self.width * self.height) as usize;
        let c = (self.width / 2 * (self.height / 2)) as usize;
        (
            &self.data[..y],
            &self.data[y..y + c],
            &self.data[y + c..y + 2 * c],
        )
    }
}

fn frame_size(w: u32, h: u32) -> usize {
    (w * h + 2 * (w / 2 * (h / 2))) as usize
}

/// Recycled frame buffers, kept for the whole session so a seek doesn't reallocate them.
#[derive(Default)]
struct FramePool {
    free: Mutex<Vec<Arc<Frame>>>,
    allocs: AtomicU64,
}

impl FramePool {
    /// A buffer nobody else holds, else a new one. The result always has one owner.
    fn take(&self, size: usize) -> Arc<Frame> {
        let mut free = lock(&self.free);
        if let Some(i) = free
            .iter()
            .position(|f| Arc::strong_count(f) == 1 && f.data.len() == size)
        {
            return free.swap_remove(i);
        }
        self.allocs.fetch_add(1, Relaxed);
        Arc::new(Frame {
            data: vec![0; size],
            ..Frame::empty()
        })
    }

    /// The renderer or `latest` may still hold it: it comes back into use once they let go.
    fn give(&self, frame: Arc<Frame>) {
        let mut free = lock(&self.free);
        if free.len() < POOL_MAX {
            free.push(frame);
        }
    }
}

/// A panic elsewhere must not take playback down with it, and these locks guard plain values.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ------------------------------------------------------------------------- frame policy

#[derive(Debug, Clone, Copy, PartialEq)]
enum FrameStep {
    /// Early: sleep this long, then ask again.
    Wait(f64),
    Publish,
    /// More than one interval late: drop it and read the next.
    Skip,
    /// Too far behind to catch up by skipping.
    Restart,
}

/// What to do with a decoded frame, given the master clock.
fn frame_step(pts: f64, clock: f64, interval: f64) -> FrameStep {
    let early = pts - clock;
    if early > 0.0 {
        FrameStep::Wait(early.min(0.01))
    } else if -early > LAG_RESTART_SECS {
        FrameStep::Restart
    } else if -early > interval {
        FrameStep::Skip
    } else {
        FrameStep::Publish
    }
}

// ------------------------------------------------------------------------- clock

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
        }
    }

    fn ns(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }

    fn is_audio_master(&self) -> bool {
        self.audio_master.load(Relaxed)
    }

    /// Seconds from the start of the media.
    pub fn now(&self) -> f64 {
        self.now_at(self.ns())
    }

    /// `now` for a given reading of the counter.
    fn now_at(&self, ns: u64) -> f64 {
        if self.frozen.load(Acquire) {
            return f64::from_bits(self.frozen_value.load(Acquire));
        }
        let base = f64::from_bits(self.base.load(Acquire));
        if !self.audio_master.load(Relaxed) {
            return base + ns.saturating_sub(self.wall_started_ns.load(Acquire)) as f64 * 1e-9;
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
            // The first frame of the last callback is heard at at + latency. Time moves on inside
            // that callback's data only, so an underrun (consumed = 0) stops the clock instead of
            // letting it run ahead of the sound.
            let since = (ns.saturating_sub(at) as f64 * 1e-9).min(consumed as f64 / rate);
            let pos = frames as f64 / rate + since - latency as f64 * 1e-9;
            return base + pos.max(0.0);
        }
    }

    /// Only the audio callback calls this.
    fn publish_point(&self, frames_before: u64, consumed: u64, latency_ns: u64, at_ns: u64) {
        let p = &self.point;
        p.seq.fetch_add(1, AcqRel);
        p.frames_before.store(frames_before, Release);
        p.at_ns.store(at_ns, Release);
        p.latency_ns.store(latency_ns, Release);
        p.consumed.store(consumed, Release);
        p.seq.fetch_add(1, AcqRel);
    }

    fn freeze_at(&self, pos: f64) {
        self.frozen_value.store(pos.to_bits(), Release);
        self.frozen.store(true, Release);
    }

    /// Wall clock only; with audio the callback unfreezes on its first real samples.
    fn unfreeze_wall(&self) {
        self.unfreeze_wall_at(self.ns());
    }

    fn unfreeze_wall_at(&self, ns: u64) {
        let pos = f64::from_bits(self.frozen_value.load(Acquire));
        self.base.store(pos.to_bits(), Release);
        self.wall_started_ns.store(ns, Release);
        self.frozen.store(false, Release);
    }
}

// ------------------------------------------------------------------------- shared state

pub struct Shared {
    latest: Mutex<Arc<Frame>>,
    pub clock: Clock,
    pool: FramePool,
    generation: AtomicU64,
    /// Counts user seeks. The picture has caught up when `seek_shown` reaches `seek_requested`.
    seek_requested: AtomicU64,
    seek_shown: AtomicU64,
    paused: AtomicBool,
    /// The two halves of [`Shared::ended`]. A stream that isn't there counts as done.
    video_done: AtomicBool,
    audio_done: AtomicBool,
    /// The video child gave no frame at all.
    video_failed: AtomicBool,
    stop: AtomicBool,
    /// Audio stays silent and the clock stays frozen until the first new picture exists.
    hold: AtomicBool,
    /// Set by each `prepare`. The clock must not run until the surface can really show a picture,
    /// or the first seconds of the video play to nobody.
    pub surface_ready: AtomicBool,
    /// Stop waiting for the surface: no GPU backend, or the window is minimized.
    no_surface: AtomicBool,
    /// The device went away: the session moves to the wall clock on the next tick.
    audio_failed: AtomicBool,
    volume: AtomicU32,
    pub frame_interval: f64,
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shared")
            .field("generation", &self.generation)
            .field("paused", &self.paused)
            .finish_non_exhaustive()
    }
}

impl Shared {
    fn new(fps: f64, volume: f32) -> Arc<Self> {
        Arc::new(Self {
            latest: Mutex::new(Arc::new(Frame::empty())),
            clock: Clock::new(),
            pool: FramePool::default(),
            generation: AtomicU64::new(0),
            seek_requested: AtomicU64::new(0),
            seek_shown: AtomicU64::new(0),
            paused: AtomicBool::new(false),
            video_done: AtomicBool::new(false),
            audio_done: AtomicBool::new(false),
            video_failed: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            hold: AtomicBool::new(true),
            surface_ready: AtomicBool::new(false),
            no_surface: AtomicBool::new(false),
            audio_failed: AtomicBool::new(false),
            volume: AtomicU32::new(volume.to_bits()),
            frame_interval: 1.0 / fps,
        })
    }

    /// The frame to show. Between a seek and its first new picture this is still the old one.
    pub fn latest(&self) -> Arc<Frame> {
        lock(&self.latest).clone()
    }

    /// A user seek hasn't reached the screen yet.
    pub fn is_seeking(&self) -> bool {
        self.seek_requested.load(Acquire) != self.seek_shown.load(Acquire)
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Acquire)
    }

    /// Everything has played out.
    pub fn ended(&self) -> bool {
        self.video_done.load(Acquire) && self.audio_done.load(Acquire)
    }

    pub fn video_failed(&self) -> bool {
        self.video_failed.load(Acquire)
    }

    /// Bumped under the `latest` lock so that [`Shared::publish`] can't let a stale frame through.
    fn next_generation(&self) -> u64 {
        let _latest = lock(&self.latest);
        self.generation.fetch_add(1, AcqRel) + 1
    }

    fn is_current(&self, generation: u64) -> bool {
        self.generation.load(Acquire) == generation && !self.stop.load(Acquire)
    }

    /// Latest wins, but only within the current generation: a frame from before a seek is dropped
    /// however late it arrives.
    fn publish(&self, frame: &Arc<Frame>) -> bool {
        let mut latest = lock(&self.latest);
        if !self.is_current(frame.generation) {
            return false;
        }
        *latest = frame.clone();
        true
    }

    /// A kill during a seek also ends the old pipe, and must not be taken for the end of the video.
    fn mark_video_done(&self, generation: u64, failed: bool) {
        let _latest = lock(&self.latest);
        if self.is_current(generation) {
            self.video_failed.store(failed, Release);
            self.video_done.store(true, Release);
        }
    }

    fn seek_reached(&self, seq: Option<u64>) {
        if let Some(seq) = seq {
            self.seek_shown.fetch_max(seq, AcqRel);
        }
    }

    /// The new position is on screen, or won't be: let sound and time go.
    fn release_hold(&self) {
        self.hold.store(false, Release);
        if !self.clock.is_audio_master() && !self.is_paused() {
            self.clock.unfreeze_wall();
        }
    }

    fn surface_can_show(&self) -> bool {
        self.surface_ready.load(Acquire) || self.no_surface.load(Acquire)
    }
}

// ------------------------------------------------------------------------- ffmpeg

fn no_console(_cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        _cmd.creation_flags(0x0800_0000);
    }
}

fn base_command(cfg: &Config, start: f64) -> Command {
    let mut cmd = Command::new(&cfg.ffmpeg);
    cmd.args(["-nostdin", "-hide_banner", "-loglevel", "error"]);
    if start > 0.0 {
        cmd.args(["-ss", &format!("{start:.3}")]);
    }
    cmd.arg("-i").arg(&cfg.input);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    no_console(&mut cmd);
    cmd
}

fn spawn_video(cfg: &Config, start: f64) -> std::io::Result<Child> {
    let vf = format!(
        "fps={},scale={}:{}:flags=fast_bilinear:out_color_matrix=bt709:out_range=tv,setsar=1,format=yuv420p",
        cfg.fps, cfg.width, cfg.height
    );
    base_command(cfg, start)
        .args(["-map", VIDEO_MAP, "-an", "-sn", "-dn", "-vf", &vf])
        .args(["-f", "rawvideo", "pipe:1"])
        .spawn()
}

fn spawn_audio(cfg: &Config, start: f64, channels: u16, rate: u32) -> std::io::Result<Child> {
    base_command(cfg, start)
        .args(["-map", AUDIO_MAP, "-vn", "-sn", "-dn"])
        .args(["-ac", &channels.to_string(), "-ar", &rate.to_string()])
        .args(["-f", "f32le", "pipe:1"])
        .spawn()
}

// ------------------------------------------------------------------------- video thread

struct VideoJob {
    shared: Arc<Shared>,
    size: usize,
    width: u32,
    height: u32,
    generation: u64,
    start: f64,
    /// The user seek this start answers, if any.
    seek: Option<u64>,
    stdout: std::process::ChildStdout,
    seek_tx: mpsc::Sender<SeekCmd>,
}

fn video_thread(mut job: VideoJob) {
    let shared = job.shared.clone();
    let generation = job.generation;
    let interval = shared.frame_interval;
    let mut n: u64 = 0;

    loop {
        let mut buf = shared.pool.take(job.size);
        let Some(frame) = Arc::get_mut(&mut buf) else {
            continue;
        };
        if job.stdout.read_exact(&mut frame.data).is_err() {
            shared.pool.give(buf);
            shared.mark_video_done(generation, n == 0);
            if n == 0 && shared.is_current(generation) {
                // a file ffmpeg can't decode: sound and subtitles may still work
                shared.seek_reached(job.seek);
                shared.release_hold();
            }
            return;
        }
        let pts = job.start + n as f64 * interval;
        frame.seq = NEXT_SEQ.fetch_add(1, Relaxed);
        frame.generation = generation;
        frame.pts = pts;
        frame.width = job.width;
        frame.height = job.height;
        let first = n == 0;
        n += 1;

        if !first {
            let step = loop {
                if !shared.is_current(generation) {
                    shared.pool.give(buf);
                    return;
                }
                match frame_step(pts, shared.clock.now(), interval) {
                    FrameStep::Wait(secs) => std::thread::sleep(Duration::from_secs_f64(secs)),
                    step => break step,
                }
            };
            match step {
                FrameStep::Restart => {
                    let target = shared.clock.now() + LAG_RESTART_AHEAD_SECS;
                    let _ = job.seek_tx.send(SeekCmd::Internal(target));
                    shared.pool.give(buf);
                    return;
                }
                FrameStep::Skip => {
                    shared.pool.give(buf);
                    continue;
                }
                _ => {}
            }
        }

        let published = shared.publish(&buf);
        shared.pool.give(buf);
        if !published {
            return;
        }
        if first {
            shared.seek_reached(job.seek);
            while !shared.surface_can_show() {
                std::thread::sleep(Duration::from_millis(5));
                if !shared.is_current(generation) {
                    return;
                }
            }
            shared.release_hold();
        }
    }
}

// ------------------------------------------------------------------------- audio

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
}

impl Callback {
    fn render<T: SizedSample + FromSample<f32>>(&mut self, out: &mut [T], latency_ns: u64) {
        while let Ok(swap) = self.swap_rx.pop() {
            if let Some(old) = self.consumer.replace(swap.consumer) {
                let _ = self.return_tx.push(old);
            }
            self.frames = 0;
            self.ready = false;
        }

        let shared = &self.shared;
        let clock = &shared.clock;
        // Some backends can't really pause, and closing must silence the ring at once.
        if shared.stop.load(Acquire) || shared.is_paused() {
            out.fill(T::EQUILIBRIUM);
            return;
        }
        let prefill = (self.rate as f64 * AUDIO_PREFILL_SECS) as usize * self.channels;
        let hold = shared.hold.load(Acquire);
        let Some(consumer) = self.consumer.as_mut() else {
            out.fill(T::EQUILIBRIUM);
            return;
        };
        if !self.ready && !hold && (consumer.slots() >= prefill || consumer.is_abandoned()) {
            self.ready = true;
        }
        if hold || !self.ready {
            out.fill(T::EQUILIBRIUM);
            clock.publish_point(self.frames, 0, latency_ns, clock.ns());
            return;
        }

        let want = out.len() - out.len() % self.channels;
        let available = consumer.slots();
        let n = (want.min(available) / self.channels) * self.channels;
        if n > 0
            && let Ok(chunk) = consumer.read_chunk(n)
        {
            let gain = f32::from_bits(shared.volume.load(Relaxed));
            let (a, b) = chunk.as_slices();
            for (o, s) in out.iter_mut().zip(a.iter().chain(b.iter())) {
                *o = T::from_sample((*s * gain).clamp(-1.0, MAX_SAMPLE));
            }
            chunk.commit_all();
        }
        for o in &mut out[n..] {
            *o = T::EQUILIBRIUM;
        }
        // Past the end of the track silence counts as played, so video longer than its sound still
        // has a running clock.
        let finished = n == 0 && consumer.is_abandoned();
        if finished {
            shared.audio_done.store(true, Release);
        }
        let played = if finished { want } else { n };
        let consumed = (played / self.channels) as u64;
        clock.publish_point(self.frames, consumed, latency_ns, clock.ns());
        self.frames += consumed;
        if consumed > 0 && !shared.is_paused() {
            clock.frozen.store(false, Release);
        }
    }
}

struct AudioOut {
    stream: cpal::Stream,
    channels: u16,
    rate: u32,
    swap_tx: Producer<Swap>,
    return_rx: Consumer<Consumer<f32>>,
}

fn build_stream<T: SizedSample + FromSample<f32> + Send + 'static>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    mut cb: Callback,
) -> Result<cpal::Stream, cpal::Error> {
    let shared = cb.shared.clone();
    device.build_output_stream::<T, _, _>(
        config,
        move |out: &mut [T], info: &OutputCallbackInfo| {
            let ts = info.timestamp();
            let latency = ts.playback.duration_since(ts.callback).as_nanos() as u64;
            cb.render(out, latency);
        },
        move |err| {
            if is_fatal(err.kind()) {
                shared.audio_failed.store(true, Release);
            }
        },
        None,
    )
}

/// The stream won't produce sound again. Reroutes and passing xruns aren't this.
fn is_fatal(kind: cpal::ErrorKind) -> bool {
    matches!(
        kind,
        cpal::ErrorKind::DeviceNotAvailable
            | cpal::ErrorKind::HostUnavailable
            | cpal::ErrorKind::StreamInvalidated
    )
}

fn open_audio(shared: &Arc<Shared>) -> Result<AudioOut, String> {
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
    };
    let stream = match format {
        SampleFormat::F32 => build_stream::<f32>(&device, config, cb),
        SampleFormat::I16 => build_stream::<i16>(&device, config, cb),
        SampleFormat::I24 => build_stream::<cpal::I24>(&device, config, cb),
        SampleFormat::I32 => build_stream::<i32>(&device, config, cb),
        other => return Err(format!("unhandled device format {other:?}")),
    }
    .map_err(|e| e.to_string())?;
    Ok(AudioOut {
        stream,
        channels,
        rate,
        swap_tx,
        return_rx,
    })
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

// ------------------------------------------------------------------------- session

enum SeekCmd {
    User(f64, u64),
    /// A lag restart: keeps the screen going by itself, so it gives way to any user seek.
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
    /// Set when sound was wanted and no device could play it.
    pub audio_problem: Option<String>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("input", &self.inner.cfg.input)
            .finish()
    }
}

fn kill(child: &mut Option<Child>) {
    if let Some(mut c) = child.take() {
        let _ = c.kill();
        let _ = c.wait();
    }
}

/// Half a second before the end, so a seek still has something to show.
fn clamp_seek(target: f64, duration: f64) -> f64 {
    if duration > 0.0 {
        target.clamp(0.0, (duration - 0.5).max(0.0))
    } else {
        target.max(0.0)
    }
}

impl Inner {
    fn start_at(self: &Arc<Self>, target: f64, seek: Option<u64>) {
        let shared = &self.shared;
        let mut children = lock(&self.children);
        if shared.stop.load(Acquire) {
            return;
        }
        shared.clock.freeze_at(target);
        shared.hold.store(true, Release);
        let generation = shared.next_generation();
        kill(&mut children.video);
        kill(&mut children.audio);

        let mut audio_guard = lock(&self.audio);
        let has_output = audio_guard.is_some();
        // With neither stream `ended` never comes: the page ends it by the duration instead.
        shared
            .video_done
            .store(!self.cfg.has_video && has_output, Release);
        shared.audio_done.store(!has_output, Release);
        shared.video_failed.store(false, Release);

        if let Some(audio) = audio_guard.as_mut() {
            let out = &mut audio.out;
            while out.return_rx.pop().is_ok() {}
            match spawn_audio(&self.cfg, target, out.channels, out.rate) {
                Ok(mut child) => {
                    let capacity = out.rate as usize * out.channels as usize * AUDIO_RING_SECS;
                    let (producer, consumer) = RingBuffer::new(capacity);
                    if let Some(stdout) = child.stdout.take() {
                        std::thread::spawn(move || audio_feeder(stdout, producer));
                    }
                    children.audio = Some(child);
                    shared.clock.base.store(target.to_bits(), Release);
                    audio.pending = Some(Swap { consumer });
                    if !shared.is_paused() {
                        // the callback empties the one slot within a buffer or two
                        for _ in 0..40 {
                            flush_swap(audio);
                            if audio.pending.is_none() {
                                break;
                            }
                            std::thread::sleep(Duration::from_millis(5));
                        }
                    }
                }
                Err(_) => shared.audio_done.store(true, Release),
            }
        }
        drop(audio_guard);

        if !self.cfg.has_video {
            shared.seek_reached(seek);
            shared.release_hold();
            return;
        }
        let spawned = spawn_video(&self.cfg, target).and_then(|mut child| {
            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| std::io::Error::other("no pipe"))?;
            Ok((child, stdout))
        });
        match spawned {
            Ok((child, stdout)) => {
                let job = VideoJob {
                    shared: shared.clone(),
                    size: frame_size(self.cfg.width, self.cfg.height),
                    width: self.cfg.width,
                    height: self.cfg.height,
                    generation,
                    start: target,
                    seek,
                    stdout,
                    seek_tx: self.seek_tx.clone(),
                };
                std::thread::spawn(move || video_thread(job));
                children.video = Some(child);
            }
            Err(_) => {
                shared.mark_video_done(generation, true);
                shared.seek_reached(seek);
                shared.release_hold();
            }
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
    let mut waiting: Option<(f64, Option<u64>)> = None;
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
            Some(SeekCmd::User(t, seq)) => waiting = Some((t, Some(seq))),
            Some(SeekCmd::Internal(t)) => {
                if waiting.is_none() {
                    waiting = Some((clamp_seek(t, inner.cfg.duration), None));
                }
            }
            Some(SeekCmd::Stop) => return,
            None => {
                if let Some((target, seq)) = waiting.take() {
                    inner.start_at(target, seq);
                }
            }
        }
    }
}

impl Session {
    /// Opens the audio device and starts ffmpeg, so call it off the UI thread.
    pub fn start(cfg: Config, start_pos: f64) -> Self {
        let fps = if cfg.has_video { cfg.fps } else { 30.0 };
        let shared = Shared::new(fps, cfg.volume);
        let (seek_tx, seek_rx) = mpsc::channel();
        let mut audio_problem = None;
        let mut audio = None;
        if cfg.has_audio && cfg.use_audio_device {
            match open_audio(&shared) {
                Ok(out) => {
                    shared.clock.rate.store(out.rate as u64, Relaxed);
                    shared.clock.audio_master.store(true, Relaxed);
                    match out.stream.play() {
                        Ok(()) => audio = Some(Audio { out, pending: None }),
                        Err(e) => {
                            shared.clock.audio_master.store(false, Relaxed);
                            audio_problem = Some(e.to_string());
                        }
                    }
                }
                Err(e) => audio_problem = Some(e),
            }
        }
        let inner = Arc::new(Inner {
            shared,
            cfg,
            children: Mutex::new(Children {
                video: None,
                audio: None,
            }),
            audio: Mutex::new(audio),
            seek_tx,
        });
        inner.start_at(start_pos.max(0.0), Some(0));
        let worker = inner.clone();
        std::thread::spawn(move || seek_worker(worker, seek_rx));
        Self {
            inner,
            audio_problem,
        }
    }

    pub fn shared(&self) -> Arc<Shared> {
        self.inner.shared.clone()
    }

    /// Coalesced: the last request wins after 150 ms of quiet.
    pub fn seek(&self, target: f64) {
        let seq = self.inner.shared.seek_requested.fetch_add(1, AcqRel) + 1;
        let target = clamp_seek(target, self.inner.cfg.duration);
        let _ = self.inner.seek_tx.send(SeekCmd::User(target, seq));
    }

    pub fn set_paused(&self, paused: bool) {
        let shared = &self.inner.shared;
        if shared.paused.swap(paused, AcqRel) == paused {
            return;
        }
        let mut guard = lock(&self.inner.audio);
        if paused {
            shared.clock.freeze_at(shared.clock.now());
            if let Some(audio) = guard.as_ref() {
                let _ = audio.out.stream.pause();
            }
        } else if let Some(audio) = guard.as_mut() {
            flush_swap(audio);
            let _ = audio.out.stream.play();
        } else if !shared.hold.load(Acquire) {
            shared.clock.unfreeze_wall();
        }
    }

    pub fn set_volume(&self, volume: f32) {
        self.inner
            .shared
            .volume
            .store(volume.clamp(0.0, 1.0).to_bits(), Relaxed);
    }

    /// The surface isn't drawing: no GPU backend, or a minimized window. Play on without it. If
    /// `prepare` runs later, the picture simply appears.
    pub fn set_no_surface(&self) {
        self.inner.shared.no_surface.store(true, Release);
    }

    /// True once, after the device went away. Playback moves to the wall clock, without sound.
    pub fn take_audio_failure(&self) -> bool {
        let shared = &self.inner.shared;
        if !shared.audio_failed.swap(false, AcqRel) {
            return false;
        }
        let at = shared.clock.now();
        shared.clock.freeze_at(at);
        shared.clock.audio_master.store(false, Release);
        *lock(&self.inner.audio) = None;
        kill(&mut lock(&self.inner.children).audio);
        shared.audio_done.store(true, Release);
        if !shared.is_paused() && !shared.hold.load(Acquire) {
            shared.clock.unfreeze_wall();
        }
        true
    }

    pub fn shutdown(&self) {
        let shared = &self.inner.shared;
        shared.stop.store(true, Release);
        let _ = self.inner.seek_tx.send(SeekCmd::Stop);
        // the ring may hold 2 s of sound, so don't wait for it to drain
        *lock(&self.inner.audio) = None;
        let mut children = lock(&self.inner.children);
        kill(&mut children.video);
        kill(&mut children.audio);
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cpal::Sample;

    const SEC: u64 = 1_000_000_000;

    fn near(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }

    // ---- clock, driven by a fake counter

    fn running_audio_clock(base: f64) -> Clock {
        let c = Clock::new();
        c.rate.store(48000, Relaxed);
        c.audio_master.store(true, Relaxed);
        c.base.store(base.to_bits(), Release);
        c.frozen.store(false, Release);
        c
    }

    #[test]
    fn audio_clock_counts_played_frames_minus_latency() {
        let c = running_audio_clock(10.0);
        // 1 s played before this callback of 480 frames (10 ms), heard 20 ms from now
        c.publish_point(48000, 480, 20_000_000, 5 * SEC);
        near(c.now_at(5 * SEC), 10.0 + 1.0 - 0.020);
        near(c.now_at(5 * SEC + 5_000_000), 10.0 + 1.0 - 0.020 + 0.005);
    }

    #[test]
    fn audio_clock_stops_at_the_end_of_the_callback_data() {
        let c = running_audio_clock(0.0);
        c.publish_point(48000, 480, 0, 5 * SEC);
        let limit = 1.0 + 480.0 / 48000.0;
        near(c.now_at(5 * SEC + 10_000_000), limit);
        // the audio thread is late, or the device underran: time must not run ahead of the sound
        near(c.now_at(9 * SEC), limit);
    }

    #[test]
    fn audio_clock_underrun_holds_still() {
        let c = running_audio_clock(0.0);
        c.publish_point(96000, 0, 0, SEC);
        near(c.now_at(SEC), 2.0);
        near(c.now_at(3 * SEC), 2.0);
    }

    #[test]
    fn audio_clock_never_reads_before_the_base() {
        let c = running_audio_clock(7.0);
        c.publish_point(0, 480, 50_000_000, SEC);
        near(c.now_at(SEC), 7.0);
    }

    #[test]
    fn frozen_clock_returns_its_value() {
        let c = running_audio_clock(0.0);
        c.publish_point(48000, 480, 0, SEC);
        c.freeze_at(42.5);
        near(c.now_at(SEC), 42.5);
        near(c.now_at(100 * SEC), 42.5);
    }

    #[test]
    fn wall_clock_runs_from_where_it_was_frozen() {
        let c = Clock::new();
        near(c.now_at(SEC), 0.0); // starts frozen
        c.freeze_at(12.0);
        c.unfreeze_wall_at(2 * SEC);
        near(c.now_at(2 * SEC), 12.0);
        near(c.now_at(5 * SEC), 15.0);
        c.freeze_at(c.now_at(5 * SEC));
        near(c.now_at(50 * SEC), 15.0);
        c.unfreeze_wall_at(60 * SEC);
        near(c.now_at(61 * SEC), 16.0);
    }

    // ---- timing and drop policy

    #[test]
    fn early_frames_wait_in_short_steps() {
        let i = 1.0 / 30.0;
        assert_eq!(frame_step(1.005, 1.0, i), FrameStep::Wait(1.005 - 1.0));
        assert_eq!(frame_step(5.0, 1.0, i), FrameStep::Wait(0.01));
    }

    #[test]
    fn frames_publish_up_to_one_interval_late() {
        let i = 1.0 / 30.0;
        assert_eq!(frame_step(1.0, 1.0, i), FrameStep::Publish);
        assert_eq!(frame_step(1.0, 1.0 + i * 0.9, i), FrameStep::Publish);
    }

    #[test]
    fn later_frames_are_dropped_and_far_behind_restarts() {
        let i = 1.0 / 30.0;
        assert_eq!(frame_step(1.0, 1.0 + i * 1.1, i), FrameStep::Skip);
        assert_eq!(frame_step(1.0, 3.0, i), FrameStep::Skip);
        assert_eq!(frame_step(1.0, 3.01, i), FrameStep::Restart);
    }

    #[test]
    fn seeks_stay_inside_the_media() {
        near(clamp_seek(-3.0, 100.0), 0.0);
        near(clamp_seek(50.0, 100.0), 50.0);
        near(clamp_seek(100.0, 100.0), 99.5);
        near(clamp_seek(1.0, 0.2), 0.0);
        near(clamp_seek(1000.0, 0.0), 1000.0); // unknown duration
    }

    // ---- frames

    fn frame_of(shared: &Shared, generation: u64, size: usize) -> Arc<Frame> {
        let mut f = shared.pool.take(size);
        Arc::get_mut(&mut f).unwrap().generation = generation;
        f
    }

    #[test]
    fn pool_reuses_buffers_across_generations() {
        let s = Shared::new(30.0, 1.0);
        s.next_generation();
        let f = frame_of(&s, 1, 100);
        s.publish(&f);
        s.pool.give(f);
        // `latest` still holds that one
        let g = frame_of(&s, 1, 100);
        assert_eq!(s.pool.allocs.load(Relaxed), 2);
        s.publish(&g);
        s.pool.give(g);
        // now the first is free, and a later thread gets it after a seek
        s.next_generation();
        let h = frame_of(&s, 2, 100);
        assert_eq!(s.pool.allocs.load(Relaxed), 2);
        assert_eq!(Arc::strong_count(&h), 1);
    }

    #[test]
    fn pool_only_hands_out_buffers_with_one_owner_and_stays_bounded() {
        let p = FramePool::default();
        let held: Vec<_> = (0..12).map(|_| p.take(8)).collect();
        for f in &held {
            p.give(f.clone());
        }
        assert_eq!(lock(&p.free).len(), POOL_MAX);
        let mut extra = p.take(8);
        assert!(Arc::get_mut(&mut extra).is_some());
        assert_eq!(p.allocs.load(Relaxed), 13);
    }

    #[test]
    fn frames_from_before_a_seek_are_dropped() {
        let s = Shared::new(30.0, 1.0);
        s.next_generation();
        let old = frame_of(&s, 1, 10);
        assert!(s.publish(&old));
        assert_eq!(s.latest().generation, 1);

        s.next_generation();
        let stale = frame_of(&s, 1, 10);
        assert!(!s.publish(&stale));
        // the old picture stays up until a new one exists
        assert_eq!(s.latest().generation, 1);
        let new = frame_of(&s, 2, 10);
        assert!(s.publish(&new));
        assert_eq!(s.latest().generation, 2);
    }

    #[test]
    fn a_killed_old_pipe_is_not_the_end_of_the_video() {
        let s = Shared::new(30.0, 1.0);
        s.next_generation();
        s.next_generation();
        s.mark_video_done(1, false);
        assert!(!s.video_done.load(Acquire));
        s.mark_video_done(2, false);
        assert!(s.video_done.load(Acquire));
    }

    #[test]
    fn seeking_ends_only_when_the_newest_seek_is_on_screen() {
        let s = Shared::new(30.0, 1.0);
        assert!(!s.is_seeking());
        s.seek_requested.store(2, Release);
        assert!(s.is_seeking());
        s.seek_reached(Some(1));
        assert!(s.is_seeking());
        s.seek_reached(None); // a lag restart answers nobody
        assert!(s.is_seeking());
        s.seek_reached(Some(2));
        assert!(!s.is_seeking());
    }

    #[test]
    fn planes_split_tightly_packed_yuv420() {
        let f = Frame {
            width: 4,
            height: 2,
            data: vec![0; frame_size(4, 2)],
            ..Frame::empty()
        };
        let (y, u, v) = f.planes();
        assert_eq!((y.len(), u.len(), v.len()), (8, 2, 2));
    }

    // ---- audio callback, without a device. Every sample format the device may ask for.

    struct Rig {
        cb: Callback,
        swap_tx: Producer<Swap>,
        producer: Option<Producer<f32>>,
        _return_rx: Consumer<Consumer<f32>>,
    }

    /// A callback with a ring holding `samples`, two channels at 48 kHz, no longer held.
    fn rig(samples: &[f32]) -> Rig {
        let shared = Shared::new(30.0, 1.0);
        shared.clock.audio_master.store(true, Relaxed);
        shared.clock.rate.store(48000, Relaxed);
        shared.hold.store(false, Release);
        let (mut swap_tx, swap_rx) = RingBuffer::new(1);
        let (return_tx, return_rx) = RingBuffer::new(8);
        let (mut producer, consumer) = RingBuffer::new(samples.len().max(1));
        for s in samples {
            producer.push(*s).unwrap();
        }
        swap_tx.push(Swap { consumer }).ok().unwrap();
        Rig {
            cb: Callback {
                shared,
                channels: 2,
                rate: 48000,
                consumer: None,
                swap_rx,
                return_tx,
                frames: 0,
                ready: false,
            },
            swap_tx,
            producer: Some(producer),
            _return_rx: return_rx,
        }
    }

    /// Enough ring for the prefill (100 ms at 48 kHz, stereo).
    const PREFILL: usize = 9600;

    fn ramp(first: &[f32]) -> Vec<f32> {
        let mut v = first.to_vec();
        v.resize(PREFILL, 0.25);
        v
    }

    #[test]
    fn f32_samples_pass_through_with_gain_and_clipping() {
        let mut r = rig(&ramp(&[0.5, -0.5, 1.0, -1.0]));
        r.cb.shared.volume.store(0.5f32.to_bits(), Relaxed);
        let mut out = [9.0f32; 4];
        r.cb.render(&mut out, 0);
        assert_eq!(out, [0.25, -0.25, 0.5, -0.5]);

        let mut r = rig(&ramp(&[3.0, -3.0]));
        let mut out = [0.0f32; 2];
        r.cb.render(&mut out, 0);
        assert_eq!(out, [MAX_SAMPLE, -1.0]);
    }

    #[test]
    fn i16_samples_convert_without_wrapping() {
        let mut r = rig(&ramp(&[0.5, -0.5, 1.0, -1.0]));
        let mut out = [1i16; 4];
        r.cb.render(&mut out, 0);
        assert_eq!(out[0], 16384);
        assert_eq!(out[1], -16384);
        assert!(out[2] >= 32766, "{}", out[2]);
        assert_eq!(out[3], -32768);
    }

    #[test]
    fn i24_samples_convert_without_wrapping() {
        let mut r = rig(&ramp(&[0.5, -0.5, 1.0, -1.0]));
        let mut out = [cpal::I24::EQUILIBRIUM; 4];
        r.cb.render(&mut out, 0);
        assert_eq!(out[0].inner(), 1 << 22);
        assert_eq!(out[1].inner(), -(1 << 22));
        let top = out[2].inner();
        assert!(((1 << 23) - 2..1 << 23).contains(&top), "{top}");
        assert_eq!(out[3].inner(), -(1 << 23));
    }

    #[test]
    fn i32_samples_convert_without_wrapping() {
        let mut r = rig(&ramp(&[0.5, -0.5, 1.0, -1.0]));
        let mut out = [1i32; 4];
        r.cb.render(&mut out, 0);
        assert_eq!(out[0], 1 << 30);
        assert_eq!(out[1], -(1 << 30));
        assert!(out[2] > 2_000_000_000, "{}", out[2]);
        assert_eq!(out[3], i32::MIN);
    }

    #[test]
    fn hold_is_silent_and_keeps_the_clock_frozen() {
        let mut r = rig(&ramp(&[0.5]));
        r.cb.shared.hold.store(true, Release);
        r.cb.shared.clock.freeze_at(3.0);
        let mut out = [1.0f32; 8];
        r.cb.render(&mut out, 0);
        assert_eq!(out, [0.0; 8]);
        near(r.cb.shared.clock.now(), 3.0);

        // released: the first real samples start the clock
        r.cb.shared.hold.store(false, Release);
        r.cb.render(&mut out, 0);
        assert_eq!(out[0], 0.5);
        assert!(!r.cb.shared.clock.frozen.load(Acquire));
    }

    #[test]
    fn paused_and_stopped_are_silent_and_consume_nothing() {
        let mut r = rig(&ramp(&[0.5]));
        r.cb.shared.paused.store(true, Release);
        r.cb.shared.clock.freeze_at(3.0);
        let mut out = [1.0f32; 8];
        r.cb.render(&mut out, 0);
        assert_eq!(out, [0.0; 8]);
        assert_eq!(r.cb.frames, 0);
        near(r.cb.shared.clock.now(), 3.0);

        r.cb.shared.paused.store(false, Release);
        r.cb.shared.stop.store(true, Release);
        let mut out = [1.0f32; 8];
        r.cb.render(&mut out, 0);
        assert_eq!(out, [0.0; 8]);
    }

    #[test]
    fn prefill_is_waited_for_until_the_pipe_ends() {
        let mut r = rig(&[0.5, 0.5, 0.25, 0.25]);
        let mut out = [1.0f32; 2];
        r.cb.render(&mut out, 0);
        assert_eq!(out, [0.0, 0.0], "not enough buffered yet");

        // a clip shorter than the prefill still plays
        r.producer = None;
        r.cb.render(&mut out, 0);
        assert_eq!(out, [0.5, 0.5]);
    }

    #[test]
    fn underrun_pads_silence_and_counts_only_real_frames() {
        let mut r = rig(&ramp(&[]));
        let mut big = vec![1.0f32; PREFILL + 8];
        r.cb.render(&mut big, 0);
        assert_eq!(big[PREFILL - 1], 0.25);
        assert_eq!(big[PREFILL], 0.0);
        assert_eq!(r.cb.frames, (PREFILL / 2) as u64);
        assert!(!r.cb.shared.audio_done.load(Acquire));
    }

    #[test]
    fn past_the_end_of_the_track_the_clock_runs_on_silence() {
        let mut r = rig(&[0.5, 0.5]);
        r.producer = None;
        let mut out = [1.0f32; 2];
        r.cb.render(&mut out, 0);
        assert_eq!(r.cb.frames, 1);
        assert!(!r.cb.shared.audio_done.load(Acquire));

        let mut out = [1.0f32; 4];
        r.cb.render(&mut out, 0);
        assert_eq!(out, [0.0; 4]);
        assert_eq!(r.cb.frames, 3);
        assert!(r.cb.shared.audio_done.load(Acquire));
    }

    #[test]
    fn a_new_ring_restarts_the_count() {
        let mut r = rig(&ramp(&[0.5]));
        let mut out = [0.0f32; 4];
        r.cb.render(&mut out, 0);
        assert_eq!(r.cb.frames, 2);

        let (mut producer, consumer) = RingBuffer::new(PREFILL);
        for _ in 0..PREFILL {
            producer.push(0.125).unwrap();
        }
        r.swap_tx.push(Swap { consumer }).ok().unwrap();
        r.cb.render(&mut out, 0);
        assert_eq!(out[0], 0.125);
        assert_eq!(r.cb.frames, 2);
    }

    #[test]
    fn a_partial_trailing_frame_is_left_silent() {
        let mut r = rig(&ramp(&[0.5, 0.5, 0.5, 0.5]));
        let mut out = [1.0f32; 3];
        r.cb.render(&mut out, 0);
        assert_eq!(out, [0.5, 0.5, 0.0]);
        assert_eq!(r.cb.frames, 1);
    }

    // ---- real ffmpeg, no window: `cargo test -p vipr player::session -- --ignored --test-threads=1`

    fn make_clip(name: &str, video: bool) -> Option<(PathBuf, PathBuf)> {
        let ffmpeg = engine::audio::find_ffmpeg()?;
        let dir = std::env::temp_dir().join(format!("vipr-player-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok()?;
        let clip = dir.join(name);
        let mut cmd = Command::new(&ffmpeg);
        cmd.args(["-y", "-hide_banner", "-loglevel", "error"]);
        if video {
            cmd.args(["-f", "lavfi", "-i", "testsrc=s=320x180:r=30:d=8"]);
        }
        cmd.args(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]);
        cmd.args(["-t", "8"]).arg(&clip);
        assert!(cmd.status().ok()?.success());
        Some((ffmpeg, clip))
    }

    fn wait_until(what: &str, secs: u64, f: impl Fn() -> bool) {
        let t = Instant::now();
        while !f() {
            assert!(t.elapsed().as_secs() < secs, "timed out: {what}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn config(ffmpeg: PathBuf, input: PathBuf, device: bool, video: bool) -> Config {
        Config {
            ffmpeg,
            input,
            has_video: video,
            width: 320,
            height: 180,
            fps: 30.0,
            has_audio: true,
            use_audio_device: device,
            duration: 8.0,
            volume: 0.0, // tests stay silent
        }
    }

    fn drive_video_session(device: bool) {
        let Some((ffmpeg, clip)) = make_clip(&format!("v{device}.mp4"), true) else {
            eprintln!("no ffmpeg, skipped");
            return;
        };
        let session = Session::start(config(ffmpeg, clip, device, true), 0.0);
        let s = session.shared();

        // held until the surface exists
        wait_until("first frame", 20, || s.latest().seq > 0);
        std::thread::sleep(Duration::from_millis(300));
        near(s.clock.now(), 0.0);
        s.surface_ready.store(true, Release);
        wait_until("clock starts", 20, || s.clock.now() > 0.5);

        // seek forward: last request wins, the old picture stays until the new one exists
        session.seek(1.0);
        session.seek(5.0);
        wait_until("seek", 20, || !s.is_seeking());
        assert!(s.latest().pts >= 5.0, "{}", s.latest().pts);

        // pause holds the clock, and a seek while paused still updates the picture
        session.set_paused(true);
        let at = s.clock.now();
        std::thread::sleep(Duration::from_millis(300));
        near(s.clock.now(), at);
        session.seek(2.0);
        wait_until("seek while paused", 20, || !s.is_seeking());
        assert!((s.latest().pts - 2.0).abs() < 0.2);
        near(s.clock.now(), 2.0);
        session.set_paused(false);
        wait_until("resume", 20, || s.clock.now() > 2.3);

        // no reallocation storm from the seeks
        assert!(s.pool.allocs.load(Relaxed) <= POOL_MAX as u64);

        // plays to the end
        session.seek(7.0);
        wait_until("end", 30, || s.ended());
        assert!(!s.video_failed());
    }

    #[test]
    #[ignore = "needs ffmpeg"]
    fn real_video_on_the_wall_clock() {
        drive_video_session(false);
    }

    #[test]
    #[ignore = "needs ffmpeg and an audio device"]
    fn real_video_with_the_audio_device() {
        drive_video_session(true);
    }

    #[test]
    #[ignore = "needs ffmpeg and an audio device"]
    fn real_audio_only() {
        let Some((ffmpeg, clip)) = make_clip("a.m4a", false) else {
            return;
        };
        let session = Session::start(config(ffmpeg, clip, true, false), 0.0);
        let s = session.shared();
        wait_until("clock starts", 20, || s.clock.now() > 0.3);
        session.seek(7.0);
        wait_until("seek", 20, || !s.is_seeking());
        wait_until("end", 30, || s.ended());
    }

    #[test]
    #[ignore = "needs ffmpeg"]
    fn undecodable_input_reports_failure() {
        let Some(ffmpeg) = engine::audio::find_ffmpeg() else {
            return;
        };
        let dir = std::env::temp_dir().join(format!("vipr-player-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("bad.mp4");
        std::fs::write(&bad, b"not media").unwrap();
        let mut cfg = config(ffmpeg, bad, false, true);
        cfg.has_audio = false;
        let session = Session::start(cfg, 0.0);
        let s = session.shared();
        wait_until("failure", 20, || s.video_failed());
        assert!(!s.is_seeking());
    }

    #[test]
    #[ignore = "needs ffmpeg"]
    fn closing_kills_the_children() {
        let Some((ffmpeg, clip)) = make_clip("k.mp4", true) else {
            return;
        };
        let session = Session::start(config(ffmpeg, clip, false, true), 0.0);
        let s = session.shared();
        wait_until("first frame", 20, || s.latest().seq > 0);
        session.shutdown();
        assert!(lock(&session.inner.children).video.is_none());
        // a seek after closing must not start anything
        session.inner.start_at(1.0, None);
        assert!(lock(&session.inner.children).video.is_none());
    }
}
