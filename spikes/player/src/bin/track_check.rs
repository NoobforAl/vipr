//! Which audio track does ASR use, compared with `-map 0:a:K`?
//!
//! Runs the same symphonia call as `crates/engine/src/audio.rs` (`default_track(TrackType::Audio)`), decodes a window,
//! and correlates it with each ffmpeg track and with ffmpeg's own default choice (the ASR fallback has no `-map`).
//! Prints numbers only, no audio leaves this program.
//!
//! Usage: track_check FFMPEG FILE [SECONDS]

use std::fs::File;
use std::path::Path;
use std::process::{Command, Stdio};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::default::{get_codecs, get_probe};

fn symphonia_mono(path: &Path, seconds: f64) -> (Vec<f32>, u32, usize, usize) {
    let file = File::open(path).expect("open");
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("mp4");
    let mut format = get_probe()
        .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
        .expect("probe");
    let audio_ids: Vec<u32> = format
        .tracks()
        .iter()
        .filter(|t| t.codec_params.as_ref().and_then(|p| p.audio()).is_some())
        .map(|t| t.id)
        .collect();
    println!("symphonia audio track ids in order: {audio_ids:?}");
    for t in format.tracks() {
        println!("  track id {} flags {:?}", t.id, t.flags);
    }
    let track = format.default_track(TrackType::Audio).expect("default track");
    let id = track.id;
    let index = audio_ids.iter().position(|i| *i == id).unwrap_or(usize::MAX);
    println!("default_track(Audio) = id {id}, i.e. audio index {index}");
    let params = track.codec_params.as_ref().and_then(|p| p.audio()).expect("params").clone();
    let mut decoder = get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .expect("decoder");
    let (mut mono, mut scratch, mut rate) = (Vec::new(), Vec::<f32>::new(), 0u32);
    loop {
        if rate > 0 && mono.len() as f64 >= seconds * rate as f64 {
            break;
        }
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) | Err(SymError::ResetRequired) => break,
            Err(e) => panic!("{e}"),
        };
        if packet.track_id != id {
            continue;
        }
        if let Ok(buf) = decoder.decode(&packet) {
            let spec = buf.spec();
            rate = spec.rate();
            let ch = spec.channels().count().max(1);
            scratch.resize(buf.samples_interleaved(), 0.0);
            buf.copy_to_slice_interleaved(&mut scratch);
            mono.extend(scratch.chunks_exact(ch).map(|f| f.iter().sum::<f32>() / ch as f32));
        }
    }
    (mono, rate, index, audio_ids.len())
}

fn ffmpeg_mono(ffmpeg: &str, path: &Path, map: Option<usize>, rate: u32, seconds: f64) -> Vec<f32> {
    let mut cmd = Command::new(ffmpeg);
    cmd.args(["-v", "error", "-nostdin", "-t", &seconds.to_string(), "-i"]).arg(path);
    match map {
        Some(k) => cmd.args(["-map", &format!("0:a:{k}")]),
        None => cmd.arg("-vn"),
    };
    cmd.args(["-ac", "1", "-ar", &rate.to_string(), "-f", "f32le", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let out = cmd.output().expect("ffmpeg");
    out.stdout.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

/// Best normalized correlation over small shifts (priming and rounding differ between decoders).
fn best_corr(a: &[f32], b: &[f32]) -> (f64, i32) {
    let mut best = (f64::MIN, 0);
    let n = a.len().min(b.len()).saturating_sub(4096);
    if n == 0 {
        return (0.0, 0);
    }
    let energy = |x: &[f32]| x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
    for shift in (-2048i32..=2048).step_by(64) {
        let (x, y) = if shift >= 0 {
            (&a[shift as usize..shift as usize + n], &b[..n])
        } else {
            (&a[..n], &b[(-shift) as usize..(-shift) as usize + n])
        };
        let dot: f64 = x.iter().zip(y).map(|(p, q)| *p as f64 * *q as f64).sum();
        let c = dot / (energy(x) * energy(y)).sqrt().max(1e-12);
        if c > best.0 {
            best = (c, shift);
        }
    }
    best
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let ffmpeg = &args[0];
    let path = Path::new(&args[1]);
    let seconds: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(120.0);
    let (mono, rate, index, count) = symphonia_mono(path, seconds);
    let rms = |x: &[f32]| (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64).sqrt();
    println!("symphonia decoded {} samples at {rate} Hz, rms {:.5}, index {index}", mono.len(), rms(&mono));
    for k in 0..count {
        let other = ffmpeg_mono(ffmpeg, path, Some(k), rate, seconds);
        let (c, s) = best_corr(&mono, &other);
        println!("ffmpeg -map 0:a:{k}: rms {:.5}, correlation with symphonia {c:.4} (shift {s})", rms(&other));
    }
    let auto = ffmpeg_mono(ffmpeg, path, None, rate, seconds);
    let (c, s) = best_corr(&mono, &auto);
    println!("ffmpeg no -map (ASR fallback): rms {:.5}, correlation with symphonia {c:.4} (shift {s})", rms(&auto));
    for k in 0..count {
        let other = ffmpeg_mono(ffmpeg, path, Some(k), rate, seconds);
        let (c, _) = best_corr(&auto, &other);
        println!("ffmpeg no -map vs -map 0:a:{k}: correlation {c:.4}");
    }
}
