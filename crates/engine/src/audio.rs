//! Decode any video / audio file (MP4, MKV, WebM, MP3, WAV, FLAC, OGG, ...) to mono 16 kHz `f32`,
//! the input format Whisper and the diarizer expect. Pure Rust (symphonia + rubato), no ffmpeg.
//!
//! The whole recording is held in memory at its original sample rate before resampling
//! (about 700 MB for one hour of 48 kHz audio). Streaming resampling can replace this later.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use audioadapter_buffers::owned::InterleavedOwned;
use rubato::{Fft, FixedSync, Resampler};
use symphonia::core::audio::sample::Sample;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::codecs::registry::CodecRegistry;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
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

/// Decode the first audio track to mono samples at the file's own sample rate. Returns `(samples, rate)`.
pub fn decode_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
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

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| audio_err(path, "no audio track"))?;
    let track_id = track.id;
    let known_delay = track.delay;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| audio_err(path, "audio track has no codec parameters"))?;
    // AAC (MP4 / MOV / M4A) starts with 1024 priming samples that ffmpeg trims but symphonia does not,
    // which would shift every timestamp by 21 ms at 48 kHz. Only applied when the file gives no delay itself.
    let aac_priming = if params.codec == symphonia::core::codecs::audio::well_known::CODEC_ID_AAC
        && known_delay.is_none()
    {
        1024
    } else {
        0
    };
    let mut decoder = codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(|e| audio_err(path, format!("unsupported audio codec ({e})")))?;

    let mut mono: Vec<f32> = Vec::new();
    let mut scratch: Vec<f32> = Vec::new();
    let mut rate = 0u32;
    let mut skipped = 0usize;

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
                rate = spec.rate();
                let channels = spec.channels().count().max(1);
                scratch.resize(buf.samples_interleaved(), f32::MID);
                buf.copy_to_slice_interleaved(&mut scratch);
                if channels == 1 {
                    mono.extend_from_slice(&scratch);
                } else {
                    mono.extend(
                        scratch
                            .chunks_exact(channels)
                            .map(|f| f.iter().sum::<f32>() / channels as f32),
                    );
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

    if mono.len() <= aac_priming || rate == 0 {
        return Err(audio_err(path, "no audio could be decoded"));
    }
    mono.drain(..aac_priming);
    Ok((mono, rate))
}

/// Resample mono audio to `to` Hz (no-op when the rate already matches).
pub fn resample_mono(samples: Vec<f32>, from: u32, to: u32) -> Result<Vec<f32>> {
    if from == to {
        return Ok(samples);
    }
    let frames = samples.len();
    let input = InterleavedOwned::new_from(samples, 1, frames)
        .map_err(|e| Error::Audio(format!("resample buffer: {e}")))?;
    let mut resampler = Fft::<f32>::new(from as usize, to as usize, 1024, 1, FixedSync::Both)
        .map_err(|e| Error::Audio(format!("resampler {from} -> {to} Hz: {e}")))?;
    let out = resampler
        .process_all(&input, frames, None)
        .map_err(|e| Error::Audio(format!("resampling failed: {e}")))?;
    Ok(out.take_data())
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
        .args(["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // no console window
    }
    let out = cmd
        .output()
        .map_err(|e| audio_err(path, format!("could not start ffmpeg ({e})")))?;
    if !out.status.success() || out.stdout.len() < 4 {
        let why = String::from_utf8_lossy(&out.stderr)
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
    Ok(out
        .stdout
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Decode a media file to mono 16 kHz samples. This is the entry point the engine stages use.
///
/// The built-in decoders are tried first; if they cannot read the file, ffmpeg is used when it is available.
pub fn decode_to_mono_16k(path: &Path) -> Result<Vec<f32>> {
    match decode_mono(path) {
        Ok((mono, rate)) => resample_mono(mono, rate, TARGET_RATE),
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
    let Some(track) = format.default_track(TrackType::Audio) else {
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
        .args(["-vn", "-t", "1", "-f", "null", "-"])
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

#[cfg(test)]
mod tests {
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
