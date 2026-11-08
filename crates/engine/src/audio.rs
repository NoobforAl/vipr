//! Decode any video / audio file (MP4, MKV, WebM, MP3, WAV, FLAC, OGG, ...) to mono 16 kHz `f32`,
//! the input format Whisper and the diarizer expect. Pure Rust (symphonia + rubato), no ffmpeg.
//!
//! Audio is resampled packet by packet as it is decoded, so only the 16 kHz result is held in memory
//! (about 230 MB for one hour), never the recording at its original rate.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Indexing, Resampler};
use symphonia::core::audio::sample::Sample;
use symphonia::core::codecs::CodecParameters;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::codecs::registry::CodecRegistry;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader, Track};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::{Error, Result, io_err};

pub const TARGET_RATE: u32 = 16_000;

/// symphonia's codecs plus Opus (WhatsApp / Telegram voice messages, WebM and most `.opus` / `.ogg` recordings),
/// which symphonia cannot decode by itself. libopus is compiled into the program, nothing to install.
fn codecs() -> &'static CodecRegistry {
    static REGISTRY: OnceLock<CodecRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut registry = CodecRegistry::new();
        symphonia::default::register_enabled_codecs(&mut registry);
        registry.register_audio_decoder::<symphonia_adapter_libopus::OpusDecoder>();
        registry
    })
}

fn audio_err(path: &Path, what: impl std::fmt::Display) -> Error {
    Error::Audio(format!("{}: {what}", path.display()))
}

/// The one audio track ViPr uses: the file's first, whatever its flags say. Speech recognition and the
/// player must hear the same track, and ffmpeg's `-map 0:a:0` counts the same way. A first track whose
/// codec the built-in decoders lack fails here and goes on to ffmpeg, so a later track is never used instead.
fn first_audio_track(format: &dyn FormatReader) -> Option<&Track> {
    format
        .tracks()
        .iter()
        .find(|t| matches!(t.codec_params, Some(CodecParameters::Audio(_))))
}

/// Decode the first audio track to mono samples at the file's own sample rate. Returns `(samples, rate)`.
pub fn decode_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut mono: Vec<f32> = Vec::new();
    let rate = decode_mono_each(path, &mut |_rate, hint, chunk| {
        if mono.is_empty() {
            mono.reserve(hint);
        }
        mono.extend_from_slice(chunk);
        Ok(())
    })?;
    Ok((mono, rate))
}

/// Receives decoded mono audio: `(rate, frames_hint, samples)`.
type MonoSink<'a> = dyn FnMut(u32, usize, &[f32]) -> Result<()> + 'a;

/// Decode the first audio track and hand it to `sink` as mono pieces, one per packet, at the file's own rate:
/// `sink(rate, frames_hint, samples)`. `frames_hint` is the length the file says it has (0 when unknown), for
/// reserving space up front. Returns the sample rate. The whole recording is never held here, so a caller that
/// resamples as it goes keeps memory flat however long the file is.
fn decode_mono_each(path: &Path, sink: &mut MonoSink<'_>) -> Result<u32> {
    let file = File::open(path).map_err(io_err(path))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| audio_err(path, format!("unsupported or damaged file ({e})")))?;

    let track =
        first_audio_track(format.as_ref()).ok_or_else(|| audio_err(path, "no audio track"))?;
    let track_id = track.id;
    let known_delay = track.delay;
    let frames_hint = track
        .num_frames
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(0);
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| audio_err(path, "audio track has no codec parameters"))?;
    // AAC (MP4 / MOV / M4A) starts with 1024 priming samples that ffmpeg trims but symphonia does not,
    // which would shift every timestamp by 21 ms at 48 kHz. Only applied when the file gives no delay itself.
    let aac_priming: usize = if params.codec
        == symphonia::core::codecs::audio::well_known::CODEC_ID_AAC
        && known_delay.is_none()
    {
        1024
    } else {
        0
    };
    let mut decoder = codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(|e| audio_err(path, format!("unsupported audio codec ({e})")))?;

    let mut scratch: Vec<f32> = Vec::new();
    let mut mixed: Vec<f32> = Vec::new();
    let mut rate = 0u32;
    let mut skipped = 0usize;
    // frames decoded so far, the priming ones included
    let mut decoded = 0usize;

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(audio_err(path, format!("read error ({e})"))),
        };
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buf) => {
                let spec = buf.spec();
                // the first rate is kept: a stream that changes rate part-way is not something recorders make
                if rate == 0 {
                    rate = spec.rate();
                }
                let channels = spec.channels().count().max(1);
                scratch.resize(buf.samples_interleaved(), f32::MID);
                buf.copy_to_slice_interleaved(&mut scratch);
                let frames: &[f32] = if channels == 1 {
                    &scratch
                } else {
                    let scale = 1.0 / channels as f32;
                    mixed.clear();
                    mixed.extend(
                        scratch
                            .chunks_exact(channels)
                            .map(|f| f.iter().sum::<f32>() * scale),
                    );
                    &mixed
                };
                // the priming frames are dropped as they arrive, not cut off the front of the whole recording
                let skip = aac_priming.saturating_sub(decoded).min(frames.len());
                decoded += frames.len();
                if skip < frames.len() {
                    sink(rate, frames_hint, &frames[skip..])?;
                }
            }
            // a damaged packet is skipped; too many means the file is unusable
            Err(SymError::DecodeError(_)) | Err(SymError::IoError(_)) => {
                skipped += 1;
                if skipped > 200 {
                    return Err(audio_err(path, "too many damaged audio packets"));
                }
            }
            Err(e) => return Err(audio_err(path, format!("decode error ({e})"))),
        }
    }

    if decoded <= aac_priming || rate == 0 {
        return Err(audio_err(path, "no audio could be decoded"));
    }
    Ok(rate)
}

/// Resample mono audio to `to` Hz (no-op when the rate already matches).
pub fn resample_mono(samples: Vec<f32>, from: u32, to: u32) -> Result<Vec<f32>> {
    if from == to {
        return Ok(samples);
    }
    let mut rs = MonoResampler::new(from, to, samples.len())?;
    rs.push(&samples)?;
    drop(samples);
    rs.finish()
}

fn resample_err(e: impl std::fmt::Display) -> Error {
    Error::Audio(format!("resampling failed: {e}"))
}

/// Resamples mono audio piece by piece as it is decoded, so a long recording is never held at its original rate.
/// The result is the same, sample for sample, as rubato's `process_all` over the whole recording at once: the
/// resampler's start-up delay is trimmed and the length is `ceil(input * to / from)`.
pub struct MonoResampler {
    /// `None` when the rates already match: samples pass straight through.
    fft: Option<Fft<f32>>,
    /// Input not yet resampled, shorter than one resampler chunk between calls.
    pending: Vec<f32>,
    /// One chunk of resampler output.
    chunk: Vec<f32>,
    /// Start-up delay frames still to drop.
    to_trim: usize,
    input_len: usize,
    out: Vec<f32>,
}

impl MonoResampler {
    /// `expected_input` is how many input frames are likely to come (0 when unknown); it only reserves space.
    pub fn new(from: u32, to: u32, expected_input: usize) -> Result<Self> {
        let fft = if from == to {
            None
        } else {
            Some(
                Fft::<f32>::new(from as usize, to as usize, 1024, 1, FixedSync::Both)
                    .map_err(|e| Error::Audio(format!("resampler {from} -> {to} Hz: {e}")))?,
            )
        };
        let (to_trim, chunk_len, pending_len) = fft.as_ref().map_or((0, 0, 0), |f| {
            (
                f.output_delay(),
                f.output_frames_max(),
                f.input_frames_max(),
            )
        });
        let expected_out = (expected_input as u128 * to as u128 / from.max(1) as u128) as usize;
        Ok(Self {
            fft,
            pending: Vec::with_capacity(pending_len * 2),
            chunk: vec![0.0; chunk_len],
            to_trim,
            input_len: 0,
            out: Vec::with_capacity(expected_out + chunk_len),
        })
    }

    /// Feed the next piece of input.
    pub fn push(&mut self, samples: &[f32]) -> Result<()> {
        self.input_len += samples.len();
        let Some(fft) = self.fft.as_mut() else {
            self.out.extend_from_slice(samples);
            return Ok(());
        };
        // resample straight from `samples` when nothing is waiting, to skip a copy
        let mut input = samples;
        if !self.pending.is_empty() {
            let need = fft.input_frames_next();
            let take = need.saturating_sub(self.pending.len()).min(input.len());
            self.pending.extend_from_slice(&input[..take]);
            input = &input[take..];
            if self.pending.len() < need {
                return Ok(());
            }
            let pending = std::mem::take(&mut self.pending);
            self.run(&pending, None)?;
            self.pending = pending;
            self.pending.clear();
        }
        loop {
            let need = self.fft.as_ref().map_or(0, |f| f.input_frames_next());
            if input.len() < need {
                break;
            }
            let used = self.run(&input[..need], None)?;
            input = &input[used..];
        }
        self.pending.extend_from_slice(input);
        Ok(())
    }

    /// Resample whatever is left and return the whole result.
    pub fn finish(mut self) -> Result<Vec<f32>> {
        let Some(fft) = self.fft.as_ref() else {
            return Ok(self.out);
        };
        let expected = (fft.resample_ratio() * self.input_len as f64).ceil() as usize;
        let pending = std::mem::take(&mut self.pending);
        if !pending.is_empty() {
            self.run(&pending, Some(pending.len()))?;
        }
        // the resampler still holds the end of the recording: pump silence through until it is all out
        while self.out.len() < expected {
            let before = self.out.len();
            self.run(&[], Some(0))?;
            if self.out.len() == before && self.to_trim == 0 {
                break;
            }
        }
        self.out.truncate(expected);
        Ok(self.out)
    }

    /// Resample one chunk (or the last, partial one), dropping start-up delay. Returns the input frames used.
    fn run(&mut self, input: &[f32], partial_len: Option<usize>) -> Result<usize> {
        let Some(fft) = self.fft.as_mut() else {
            return Ok(0);
        };
        let buf_in = InterleavedSlice::new(input, 1, input.len()).map_err(resample_err)?;
        let frames = self.chunk.len();
        let mut buf_out =
            InterleavedSlice::new_mut(&mut self.chunk, 1, frames).map_err(resample_err)?;
        let indexing = Indexing {
            input_offset: 0,
            output_offset: 0,
            partial_len,
            active_channels_mask: None,
        };
        let (used, made) = fft
            .process_into_buffer(&buf_in, &mut buf_out, Some(&indexing))
            .map_err(resample_err)?;
        let skip = self.to_trim.min(made);
        self.to_trim -= skip;
        self.out.extend_from_slice(&self.chunk[skip..made]);
        Ok(used)
    }
}

static FFMPEG_OVERRIDE: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

/// Use this ffmpeg instead of searching (the user picked one in Settings). `None` goes back to searching.
pub fn set_ffmpeg_override(path: Option<PathBuf>) {
    if let Ok(mut g) = FFMPEG_OVERRIDE.lock() {
        *g = path;
    }
}

/// The ffmpeg the user picked in Settings, if any (passed on to the worker programs).
pub fn ffmpeg_override() -> Option<PathBuf> {
    FFMPEG_OVERRIDE
        .lock()
        .ok()
        .and_then(|g| g.clone())
        .filter(|p| p.is_file())
}

/// Places where ffmpeg is usually installed but which are often **not** on the PATH (winget, scoop, Chocolatey,
/// Homebrew, ...). GUI programs on macOS do not even see the shell's PATH, so these matter there.
fn well_known_ffmpeg(name: &str) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    let env = |k: &str| std::env::var_os(k).map(PathBuf::from);
    if cfg!(windows) {
        if let Some(local) = env("LOCALAPPDATA") {
            candidates.push(local.join("Microsoft/WinGet/Links").join(name));
            // winget unpacks the "Gyan.FFmpeg" / "BtbN.FFmpeg" packages into a versioned folder without a PATH entry
            if let Ok(pkgs) = std::fs::read_dir(local.join("Microsoft/WinGet/Packages")) {
                for pkg in pkgs.flatten().filter(|e| {
                    e.file_name()
                        .to_string_lossy()
                        .to_lowercase()
                        .contains("ffmpeg")
                }) {
                    if let Some(found) = find_file(&pkg.path(), name, 4) {
                        return Some(found);
                    }
                }
            }
        }
        if let Some(user) = env("USERPROFILE") {
            candidates.push(user.join("scoop/shims").join(name));
            candidates.push(user.join("scoop/apps/ffmpeg/current/bin").join(name));
        }
        if let Some(pd) = env("ProgramData") {
            candidates.push(pd.join("chocolatey/bin").join(name));
        }
        for pf in ["ProgramFiles", "ProgramFiles(x86)"] {
            if let Some(dir) = env(pf) {
                candidates.push(dir.join("ffmpeg/bin").join(name));
            }
        }
        candidates.push(PathBuf::from("C:/ffmpeg/bin").join(name));
    } else {
        for dir in [
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/snap/bin",
            "/opt/local/bin",
        ] {
            candidates.push(Path::new(dir).join(name));
        }
    }
    candidates.into_iter().find(|p| p.is_file())
}

/// Look for a file called `name` below `dir`, at most `depth` folders deep.
fn find_file(dir: &Path, name: &str, depth: usize) -> Option<PathBuf> {
    let direct = dir.join(name);
    if direct.is_file() {
        return Some(direct);
    }
    if depth == 0 {
        return None;
    }
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter(|e| e.path().is_dir())
        .find_map(|e| find_file(&e.path(), name, depth - 1))
}

/// Where an `ffmpeg` program can be found, if any: the one picked in Settings, `VIPR_FFMPEG`, next to the running
/// program (also in `ffmpeg/`, `tools/`), the `PATH`, then the usual install places (winget, scoop, Chocolatey,
/// Homebrew, ...). ffmpeg is optional; it lets ViPr read formats the built-in decoders cannot
/// (AVI, WMV / WMA, FLV, MPEG-TS, MPG / VOB, 3GP / AMR, AC3, DTS, ...).
pub fn find_ffmpeg() -> Option<PathBuf> {
    let name = if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };
    if let Some(p) = ffmpeg_override() {
        return Some(p);
    }
    if let Some(p) = std::env::var_os("VIPR_FFMPEG")
        .map(PathBuf::from)
        .filter(|p| p.is_file())
    {
        return Some(p);
    }
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
    {
        for rel in ["", "ffmpeg", "tools", "workers"] {
            let p = if rel.is_empty() {
                dir.join(name)
            } else {
                dir.join(rel).join(name)
            };
            if p.is_file() {
                return Some(p);
            }
        }
    }
    // the copy ViPr downloaded itself (Settings > More formats)
    if let Some(p) = crate::tools::installed_ffmpeg() {
        return Some(p);
    }
    // dev helper: pretend there is no system-wide ffmpeg (to try the download flow on a machine that has one)
    if std::env::var_os("VIPR_HIDE_SYSTEM_FFMPEG").is_some() {
        return None;
    }
    if let Some(p) = std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|d| d.join(name))
            .find(|p| p.is_file())
    }) {
        return Some(p);
    }
    well_known_ffmpeg(name)
}

/// Ask ffmpeg to turn any media file into mono 16 kHz `f32` samples.
fn decode_with_ffmpeg(ffmpeg: &Path, path: &Path) -> Result<Vec<f32>> {
    let mut cmd = Command::new(ffmpeg);
    cmd.args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args(["-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000"])
        .args(["-f", "f32le", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // no console window
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| audio_err(path, format!("could not start ffmpeg ({e})")))?;
    // stderr is read on its own thread so a chatty ffmpeg cannot fill the pipe and stall
    let stderr = child.stderr.take().map(|mut err| {
        std::thread::spawn(move || {
            let mut s = Vec::new();
            let _ = err.read_to_end(&mut s);
            s
        })
    });
    // the samples are converted as they arrive rather than holding every byte and then a second, f32 copy
    let mut samples = Vec::new();
    let read = child
        .stdout
        .take()
        .map_or(Ok(()), |out| read_f32le(out, &mut samples));
    let status = child.wait();
    let stderr = stderr.and_then(|t| t.join().ok()).unwrap_or_default();
    let ok = read.is_ok() && status.is_ok_and(|s| s.success());
    if !ok || samples.is_empty() {
        let why = String::from_utf8_lossy(&stderr)
            .lines()
            .last()
            .unwrap_or("no sound found")
            .trim()
            .to_string();
        return Err(audio_err(
            path,
            format!("ffmpeg could not read this file ({why})"),
        ));
    }
    samples.shrink_to_fit();
    Ok(samples)
}

/// Read little-endian `f32` samples from `r` until it ends, appending them to `out`.
fn read_f32le(mut r: impl Read, out: &mut Vec<f32>) -> std::io::Result<()> {
    let mut buf = vec![0u8; 1 << 16];
    // bytes of a sample split across two reads
    let mut carry = 0usize;
    loop {
        let n = match r.read(&mut buf[carry..]) {
            Ok(0) => return Ok(()),
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        let have = carry + n;
        let (whole, rest) = buf[..have].as_chunks::<4>();
        out.extend(whole.iter().map(|b| f32::from_le_bytes(*b)));
        carry = rest.len();
        let tail = have - carry;
        buf.copy_within(tail..have, 0);
    }
}

/// Decode a media file to mono 16 kHz samples. This is the entry point the engine stages use.
///
/// The built-in decoders are tried first; if they cannot read the file, ffmpeg is used when it is available.
pub fn decode_to_mono_16k(path: &Path) -> Result<Vec<f32>> {
    // made on the first piece, once the file's rate is known
    let mut rs: Option<MonoResampler> = None;
    let native = decode_mono_each(path, &mut |rate, hint, chunk| {
        let rs = match &mut rs {
            Some(rs) => rs,
            None => rs.insert(MonoResampler::new(rate, TARGET_RATE, hint)?),
        };
        rs.push(chunk)
    });
    match native {
        Ok(_) => rs
            .ok_or_else(|| audio_err(path, "no audio could be decoded"))?
            .finish(),
        Err(native) => {
            // a missing file is not a format problem
            if matches!(native, Error::Io { .. }) {
                return Err(native);
            }
            match find_ffmpeg() {
                Some(ffmpeg) => decode_with_ffmpeg(&ffmpeg, path).map_err(|e| {
                    Error::Audio(format!("{e} (the built-in decoders said: {native})"))
                }),
                None => Err(Error::Audio(format!(
                    "{native} - this format needs ffmpeg to be read (ffmpeg was not found)"
                ))),
            }
        }
    }
}

/// Result of checking whether a file can be turned into text before it is added to the queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Probe {
    /// The built-in decoders (or ffmpeg) can read sound from it.
    Ok,
    /// A real media format, but one only ffmpeg can read, and ffmpeg was not found.
    NeedsFfmpeg,
    /// Not usable; the text says why in plain words.
    Invalid(String),
}

/// Extensions that can only be read through ffmpeg (used to tell "needs ffmpeg" from "not a media file").
const FFMPEG_ONLY: [&str; 18] = [
    "avi", "wmv", "wma", "asf", "flv", "ts", "mts", "m2ts", "mpg", "mpeg", "vob", "amr", "ac3",
    "eac3", "dts", "wv", "ape", "spx",
];

/// Can the built-in decoders really decode sound from this file? Opens it, picks the audio track and decodes a few packets,
/// so a text file renamed to `.mp3` is not accepted just because the name looks right.
fn native_can_read(path: &Path) -> bool {
    let Ok(file) = File::open(path) else {
        return false;
    };
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let Ok(mut format) = symphonia::default::get_probe().probe(
        &hint,
        mss,
        FormatOptions::default(),
        MetadataOptions::default(),
    ) else {
        return false;
    };
    let Some(track) = first_audio_track(format.as_ref()) else {
        return false;
    };
    let track_id = track.id;
    let Some(params) = track.codec_params.as_ref().and_then(|p| p.audio()) else {
        return false;
    };
    let Ok(mut decoder) = codecs().make_audio_decoder(params, &AudioDecoderOptions::default())
    else {
        return false;
    };
    let (mut decoded, mut attempts) = (0, 0);
    while attempts < 60 && decoded < 3 {
        attempts += 1;
        match format.next_packet() {
            Ok(Some(packet)) if packet.track_id == track_id => {
                if decoder.decode(&packet).is_ok() {
                    decoded += 1;
                }
            }
            Ok(Some(_)) => {}
            _ => break,
        }
    }
    decoded > 0
}

/// Ask ffmpeg whether it can read sound from the file (decodes the first second only).
fn ffmpeg_can_read(ffmpeg: &Path, path: &Path) -> bool {
    let mut cmd = Command::new(ffmpeg);
    cmd.args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args(["-map", "0:a:0", "-vn", "-t", "1", "-f", "null", "-"])
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    cmd.status().is_ok_and(|s| s.success())
}

/// Check a file before it goes into the queue. Reads only a little of the file, but may touch the disk, so call it
/// off the UI thread.
pub fn probe_media(path: &Path) -> Probe {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return Probe::Invalid("cannot be opened".into()),
    };
    if meta.is_dir() {
        return Probe::Invalid("is a folder, not a file".into());
    }
    if meta.len() == 0 {
        return Probe::Invalid("is empty".into());
    }
    if native_can_read(path) {
        return Probe::Ok;
    }
    match find_ffmpeg() {
        Some(ffmpeg) => {
            if ffmpeg_can_read(&ffmpeg, path) {
                Probe::Ok
            } else {
                Probe::Invalid(
                    "has no sound that can be read (it may be damaged or not a media file)".into(),
                )
            }
        }
        None => {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            if FFMPEG_ONLY.contains(&ext.as_str()) {
                Probe::NeedsFfmpeg
            } else {
                Probe::Invalid("is not a video or audio file ViPr can read".into())
            }
        }
    }
}

/// Write 16-bit mono PCM WAV (for debugging and for comparing with the Python pipeline's WAV files).
pub fn write_wav_16k(path: &Path, samples: &[f32]) -> Result<()> {
    use std::io::Write;
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&TARGET_RATE.to_le_bytes());
    out.extend_from_slice(&(TARGET_RATE * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    let mut f = File::create(path).map_err(io_err(path))?;
    f.write_all(&out).map_err(io_err(path))
}

/// Save mono 16 kHz audio as a 32-bit float WAV, losslessly: reading it back gives exactly `samples`, so one worker
/// can decode a recording and hand the result to the next instead of each decoding the video again. Written to a
/// temporary name first, so a half-written file is never left under `path`.
pub fn write_wav_f32_16k(path: &Path, samples: &[f32]) -> Result<()> {
    use std::io::{BufWriter, Write};
    let data_len = u32::try_from(samples.len() * 4)
        .ok()
        .filter(|n| *n <= u32::MAX - 36)
        .ok_or_else(|| Error::Audio("audio too long for a WAV file".into()))?;
    let tmp = path.with_extension("wav.part");
    let write = || -> std::io::Result<()> {
        let mut w = BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
        w.write_all(b"RIFF")?;
        w.write_all(&(36 + data_len).to_le_bytes())?;
        w.write_all(b"WAVEfmt ")?;
        w.write_all(&16u32.to_le_bytes())?;
        w.write_all(&3u16.to_le_bytes())?; // IEEE float
        w.write_all(&1u16.to_le_bytes())?; // mono
        w.write_all(&TARGET_RATE.to_le_bytes())?;
        w.write_all(&(TARGET_RATE * 4).to_le_bytes())?;
        w.write_all(&4u16.to_le_bytes())?;
        w.write_all(&32u16.to_le_bytes())?;
        w.write_all(b"data")?;
        w.write_all(&data_len.to_le_bytes())?;
        for s in samples {
            w.write_all(&s.to_le_bytes())?;
        }
        w.into_inner().map_err(|e| e.into_error())?.sync_all()
    };
    if let Err(e) = write().and_then(|_| std::fs::rename(&tmp, path)) {
        let _ = std::fs::remove_file(&tmp);
        return Err(io_err(path)(e));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn float_wav_reads_back_exactly() {
        let dir = std::env::temp_dir().join(format!("vipr-f32wav-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.16k.wav");
        // values a 16-bit file would round, and a few that are out of range
        let src: Vec<f32> = (0..20_001)
            .map(|i| ((i as f32 * 0.013).sin() * 1.3) + 1e-7 * i as f32)
            .collect();
        super::write_wav_f32_16k(&p, &src).unwrap();
        assert!(!p.with_extension("wav.part").exists());
        let back = super::decode_to_mono_16k(&p).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(back.len(), src.len());
        assert!(
            back.iter()
                .zip(&src)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    #[test]
    fn ffmpeg_search_honours_the_environment_variable() {
        // a path that does not exist is ignored (falls through to the other places)
        unsafe { std::env::set_var("VIPR_FFMPEG", "definitely-not-here.exe") };
        let found = super::find_ffmpeg();
        unsafe { std::env::remove_var("VIPR_FFMPEG") };
        assert!(found.is_none_or(|p| p.is_file()));
    }

    use super::*;

    #[test]
    fn resample_keeps_length_ratio_and_tone() {
        // 1 s of a 440 Hz tone at 48 kHz -> 16 kHz
        let tone: Vec<f32> = (0..48_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin() * 0.5)
            .collect();
        let out = resample_mono(tone, 48_000, 16_000).unwrap();
        assert!(
            (out.len() as i64 - 16_000).abs() < 64,
            "got {} samples",
            out.len()
        );
        let rms = (out.iter().map(|x| x * x).sum::<f32>() / out.len() as f32).sqrt();
        assert!((rms - 0.3535).abs() < 0.03, "rms {rms}");
    }

    /// What resampling the whole clip at once gives, the result streaming must match.
    fn resample_at_once(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
        use audioadapter_buffers::owned::InterleavedOwned;
        use rubato::{Fft, FixedSync, Resampler};
        let input = InterleavedOwned::new_from(samples.to_vec(), 1, samples.len()).unwrap();
        let mut rs = Fft::<f32>::new(from as usize, to as usize, 1024, 1, FixedSync::Both).unwrap();
        rs.process_all(&input, samples.len(), None)
            .unwrap()
            .take_data()
    }

    #[test]
    fn streaming_resample_matches_resampling_all_at_once() {
        // a sweep plus a little noise, odd lengths, fed in pieces of awkward sizes (packets are not chunk-sized)
        let mut seed = 7u32;
        let mut noise = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        for (from, to, len) in [
            (48_000, 16_000, 100_003),
            (44_100, 16_000, 77_777),
            (22_050, 16_000, 5),
            (48_000, 16_000, 0),
            (8_000, 16_000, 12_345),
        ] {
            let clip: Vec<f32> = (0..len)
                .map(|i| (i as f32 * i as f32 * 1e-7).sin() * 0.4 + noise() * 0.05)
                .collect();
            let want = resample_at_once(&clip, from, to);
            for piece in [1usize, 960, 1024, 4097] {
                let mut rs = MonoResampler::new(from, to, len).unwrap();
                for chunk in clip.chunks(piece) {
                    rs.push(chunk).unwrap();
                }
                let got = rs.finish().unwrap();
                assert_eq!(
                    got.len(),
                    want.len(),
                    "{from}->{to} len {len} piece {piece}"
                );
                let worst = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    worst < 1e-6,
                    "{from}->{to} len {len} piece {piece}: off by {worst}"
                );
            }
        }
    }

    #[test]
    fn same_rate_passes_through() {
        let clip: Vec<f32> = (0..3000).map(|i| i as f32).collect();
        let mut rs = MonoResampler::new(16_000, 16_000, 0).unwrap();
        rs.push(&clip[..1234]).unwrap();
        rs.push(&clip[1234..]).unwrap();
        assert_eq!(rs.finish().unwrap(), clip);
    }

    #[test]
    fn f32_samples_split_across_reads_come_out_whole() {
        /// Hands out its bytes a few at a time, as a pipe does.
        struct Trickle(Vec<u8>, usize);
        impl std::io::Read for Trickle {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let n = 3.min(buf.len()).min(self.0.len() - self.1);
                buf[..n].copy_from_slice(&self.0[self.1..self.1 + n]);
                self.1 += n;
                Ok(n)
            }
        }
        let want: Vec<f32> = (0..1000).map(|i| i as f32 * 0.25 - 7.0).collect();
        let bytes: Vec<u8> = want.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut got = Vec::new();
        read_f32le(Trickle(bytes, 0), &mut got).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn wav_round_trip() {
        let dir = std::env::temp_dir().join("mt_audio_test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.wav");
        let src: Vec<f32> = (0..1600).map(|i| (i as f32 / 1600.0) - 0.5).collect();
        write_wav_16k(&p, &src).unwrap();
        let (back, rate) = decode_mono(&p).unwrap();
        assert_eq!(rate, 16_000);
        assert_eq!(back.len(), src.len());
        assert!(back.iter().zip(&src).all(|(a, b)| (a - b).abs() < 0.001));
    }

    #[test]
    fn probe_accepts_real_audio_and_rejects_everything_else() {
        let dir = std::env::temp_dir().join("mt_probe_test");
        std::fs::create_dir_all(&dir).unwrap();

        // a real WAV file
        let good = dir.join("good.wav");
        let tone: Vec<f32> = (0..16_000).map(|i| (i as f32 / 40.0).sin() * 0.3).collect();
        write_wav_16k(&good, &tone).unwrap();
        assert_eq!(probe_media(&good), Probe::Ok);

        // a text file pretending to be an MP3
        let fake = dir.join("fake.mp3");
        std::fs::write(&fake, "this is just text, not audio\n".repeat(200)).unwrap();
        assert!(
            matches!(probe_media(&fake), Probe::Invalid(_)),
            "{:?}",
            probe_media(&fake)
        );

        // empty file, folder, missing file
        let empty = dir.join("empty.wav");
        std::fs::write(&empty, b"").unwrap();
        assert!(matches!(probe_media(&empty), Probe::Invalid(_)));
        assert!(matches!(probe_media(&dir), Probe::Invalid(_)));
        assert!(matches!(
            probe_media(&dir.join("nope.mp3")),
            Probe::Invalid(_)
        ));

        // a text file called .avi: never accepted (needs ffmpeg when it is missing, invalid when it is there)
        let fake_avi = dir.join("fake.avi");
        std::fs::write(&fake_avi, "not a video\n".repeat(100)).unwrap();
        assert_ne!(probe_media(&fake_avi), Probe::Ok);
    }

    #[test]
    fn missing_file_is_a_clean_error() {
        assert!(decode_to_mono_16k(Path::new("definitely-not-here.mp4")).is_err());
    }
}
