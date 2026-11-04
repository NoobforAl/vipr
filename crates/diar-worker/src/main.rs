//! diar-worker - speaker diarization worker process (pyannote segmentation-3.0 + WeSpeaker embeddings,
//! run with ONNX Runtime on the CPU).
//!
//!   diar-worker --audio meeting.mp4 --segmentation segmentation-3.0.onnx \
//!               --embedding wespeaker_en_voxceleb_CAM++.onnx --out meeting.diar.json [--num-speakers 2]
//!
//! stdout: JSON-lines events (`audio`, `progress`, `done`, `error`), same style as `asr-worker`.
//! Output file: `{"num_speakers":2,"audio_seconds":..,"elapsed_seconds":..,"turns":[{"start","end","speaker"}]}`.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use diar_worker::*;
use engine::diar::DiarTurn;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use serde_json::json;

#[derive(Parser)]
#[command(about = "Speaker diarization worker (pyannote segmentation + WeSpeaker, ONNX Runtime)")]
struct Args {
    /// Video or audio file
    #[arg(long)]
    audio: PathBuf,
    /// segmentation-3.0 ONNX model
    #[arg(long)]
    segmentation: PathBuf,
    /// Speaker embedding ONNX model (WeSpeaker CAM++)
    #[arg(long)]
    embedding: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Number of speakers: 0 = detect automatically (default), N >= 1 = force N
    #[arg(long, default_value_t = 0)]
    num_speakers: usize,
    /// Upper limit for automatic detection
    #[arg(long, default_value_t = 8)]
    max_speakers: usize,
    /// Automatic mode: speakers whose voices are at least this similar (cosine, 0..1) are the same person
    #[arg(long, default_value_t = 0.5, hide = true)]
    stop_sim: f32,
    /// Automatic mode: a "speaker" with less speech than this (seconds) is noise and is merged away
    #[arg(long, default_value_t = 10.0, hide = true)]
    min_speaker_seconds: f32,
    /// Print the similarity of the last cluster merges to stderr (for tuning)
    #[arg(long, hide = true)]
    debug_merges: bool,
    /// CPU threads for ONNX Runtime
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// Scale applied to the waveform before segmentation (1.0 = samples in -1..1)
    #[arg(long, default_value_t = 1.0, hide = true)]
    seg_scale: f32,
    /// Bridge pauses up to this many seconds inside one speaker's speech
    #[arg(long, default_value_t = 0.3, hide = true)]
    max_gap: f64,
}

fn event(v: serde_json::Value) {
    println!("{v}");
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(&args) {
        event(json!({"event": "error", "message": format!("{e:#}")}));
        std::process::exit(1);
    }
}

fn ort_err<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow!("{e}")
}

fn session(env: &ort::environment::Environment, path: &PathBuf, threads: usize) -> Result<Session> {
    Session::builder(env)
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_err)?
        .with_intra_threads(threads)
        .map_err(ort_err)?
        .commit_from_file(path)
        .map_err(ort_err)
        .with_context(|| format!("loading {}", path.display()))
}

/// Run segmentation on one window (zero-padded to 10 s).
fn segment_window(sess: &mut Session, audio: &[f32], start: usize, scale: f32) -> Result<WindowSeg> {
    let end = (start + WINDOW).min(audio.len());
    let mut data = vec![0f32; WINDOW];
    for (d, s) in data.iter_mut().zip(&audio[start..end]) {
        *d = s * scale;
    }
    let tensor = Tensor::from_array((vec![1i64, 1, WINDOW as i64], data)).map_err(ort_err)?;
    let outs = sess.run(ort::inputs![tensor]).map_err(ort_err)?;
    let (shape, values) = outs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
    let frames = shape[1] as usize;
    let classes_n = shape[2] as usize;
    if classes_n != 7 {
        return Err(anyhow!("unexpected segmentation output with {classes_n} classes (expected 7)"));
    }
    let mut classes = Vec::with_capacity(frames);
    let mut single_logp = Vec::with_capacity(frames);
    for f in 0..frames {
        let row = &values[f * 7..f * 7 + 7];
        let best = (0..7).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap_or(0);
        classes.push(best as u8);
        single_logp.push([row[1], row[2], row[3]]);
    }
    Ok(WindowSeg { start, classes, single_logp })
}

/// WeSpeaker embedding of a piece of audio (unit length).
fn embed(sess: &mut Session, samples: &[f32]) -> Result<Vec<f32>> {
    let feats = knf_rs::compute_fbank(samples).map_err(|e| anyhow!("fbank: {e}"))?;
    let (frames, bins) = feats.dim();
    let data: Vec<f32> = feats.as_standard_layout().iter().copied().collect();
    let tensor = Tensor::from_array((vec![1i64, frames as i64, bins as i64], data)).map_err(ort_err)?;
    let outs = sess.run(ort::inputs!["feats" => tensor]).map_err(ort_err)?;
    let (_, values) = outs["embs"].try_extract_tensor::<f32>().map_err(ort_err)?;
    let mut v = values.to_vec();
    normalize(&mut v);
    Ok(v)
}

/// One local speaker of one window that got an embedding.
struct Item {
    window: usize,
    local: usize,
    weight: f32,
}

const MIN_EMBED_SECONDS: f64 = 0.3;
const MAX_EMBED_SECONDS: f64 = 20.0;

fn run(args: &Args) -> Result<()> {
    let started = Instant::now();
    let audio = engine::audio::decode_to_mono_16k(&args.audio).context("decoding audio")?;
    let audio_seconds = audio.len() as f64 / SAMPLE_RATE;
    event(json!({"event": "audio", "seconds": audio_seconds}));

    let env = ort::init().build().map_err(ort_err)?;
    let mut seg_sess = session(&env, &args.segmentation, args.threads)?;
    let mut emb_sess = session(&env, &args.embedding, args.threads)?;

    // 1. segmentation on consecutive 10 s windows
    let n_windows = audio.len().div_ceil(WINDOW).max(1);
    let mut windows: Vec<WindowSeg> = Vec::with_capacity(n_windows);
    for w in 0..n_windows {
        windows.push(segment_window(&mut seg_sess, &audio, w * WINDOW, args.seg_scale)?);
        event(json!({"event": "progress", "percent": (w + 1) * 40 / n_windows}));
    }

    // 2. one embedding per (window, local speaker): clean audio if there is enough, else everything it says
    let mut items: Vec<Item> = Vec::new();
    let mut embs: Vec<Vec<f32>> = Vec::new();
    let candidates: Vec<(usize, usize)> =
        (0..windows.len()).flat_map(|w| (0..3).map(move |l| (w, l))).filter(|&(w, l)| windows[w].active_seconds(l) > 0.0).collect();
    for (done, &(w, l)) in candidates.iter().enumerate() {
        let win = &windows[w];
        let (ranges, seconds) = if win.clean_seconds(l) >= MIN_EMBED_SECONDS {
            (win.clean_ranges(l), win.clean_seconds(l))
        } else if win.active_seconds(l) >= MIN_EMBED_SECONDS {
            // mostly overlapped speech: use all frames of this speaker
            let mut r: Vec<(usize, usize)> = Vec::new();
            for (i, &c) in win.classes.iter().enumerate() {
                if class_members(c as usize).contains(&l) {
                    let (a, b) = frame_range(win.start, i);
                    match r.last_mut() {
                        Some(last) if last.1 == a => last.1 = b,
                        _ => r.push((a, b)),
                    }
                }
            }
            (r, win.active_seconds(l))
        } else {
            continue; // too short to identify; its frames are filled from the neighbours later
        };
        let mut piece: Vec<f32> = Vec::new();
        for (a, b) in ranges {
            let (a, b) = (a.min(audio.len()), b.min(audio.len()));
            piece.extend_from_slice(&audio[a..b]);
            if piece.len() as f64 / SAMPLE_RATE >= MAX_EMBED_SECONDS {
                break;
            }
        }
        if (piece.len() as f64 / SAMPLE_RATE) < MIN_EMBED_SECONDS {
            continue;
        }
        embs.push(embed(&mut emb_sess, &piece)?);
        items.push(Item { window: w, local: l, weight: seconds as f32 });
        if done % 8 == 0 {
            event(json!({"event": "progress", "percent": 40 + (done + 1) * 55 / candidates.len().max(1)}));
        }
    }

    // 3. cluster
    let weights: Vec<f32> = items.iter().map(|i| i.weight).collect();
    let labels: Vec<usize> = if embs.is_empty() {
        vec![]
    } else if args.num_speakers >= 1 {
        let c = cluster_k(&embs, &weights, args.num_speakers);
        c.labels
    } else {
        if args.debug_merges {
            let probe = agglomerative(&embs, &weights, -1.0, 1);
            let tail: Vec<String> = probe.merge_sims.iter().rev().take(8).map(|s| format!("{s:.2}")).collect();
            eprintln!("points: {}  last merges (similarity, last first): {}", embs.len(), tail.join(" "));
            // what the clusters look like before tiny ones are merged away
            let raw = cluster_auto(&embs, &weights, args.stop_sim, args.max_speakers, 0.0);
            let mut secs = vec![0f32; raw.centroids.len()];
            for (l, w) in raw.labels.iter().zip(&weights) {
                secs[*l] += w;
            }
            let mut order: Vec<usize> = (0..secs.len()).collect();
            order.sort_by(|&a, &b| secs[b].total_cmp(&secs[a]));
            let desc: Vec<String> = order.iter().map(|&c| format!("{:.0}s", secs[c])).collect();
            eprintln!("clusters before merging tiny ones: {}", desc.join(" "));
            for (i, &a) in order.iter().enumerate().take(4) {
                for &b in order.iter().skip(i + 1).take(4) {
                    let sim: f32 = raw.centroids[a].iter().zip(&raw.centroids[b]).map(|(x, y)| x * y).sum();
                    eprintln!("   centroid similarity {:.0}s vs {:.0}s: {sim:.2}", secs[a], secs[b]);
                }
            }
        }
        let c = cluster_auto(&embs, &weights, args.stop_sim, args.max_speakers, args.min_speaker_seconds);
        c.labels
    };
    let mut cluster_of = vec![[None::<usize>; 3]; windows.len()];
    for (item, &lab) in items.iter().zip(&labels) {
        cluster_of[item.window][item.local] = Some(lab);
    }

    // 4. frame by frame: speaker = cluster of the local speaker (overlaps go to the stronger one)
    let mut frames: Vec<(f64, f64, usize)> = Vec::new();
    let mut pending: Vec<(f64, f64)> = Vec::new(); // frames of speakers we could not identify
    let mut last_label: Option<usize> = None;
    for (w, win) in windows.iter().enumerate() {
        for (i, &c) in win.classes.iter().enumerate() {
            let members = class_members(c as usize);
            if members.is_empty() {
                continue;
            }
            let (a, b) = frame_range(win.start, i);
            if a >= audio.len() {
                continue;
            }
            let (t0, t1) = (a as f64 / SAMPLE_RATE, b.min(audio.len()) as f64 / SAMPLE_RATE);
            let pick = members
                .iter()
                .copied()
                .filter(|&m| cluster_of[w][m].is_some())
                .max_by(|&x, &y| win.single_logp[i][x].total_cmp(&win.single_logp[i][y]));
            match pick.and_then(|m| cluster_of[w][m]) {
                Some(label) => {
                    // frames waiting for a neighbour take this speaker
                    for (p0, p1) in pending.drain(..) {
                        frames.push((p0, p1, label));
                    }
                    frames.push((t0, t1, label));
                    last_label = Some(label);
                }
                None => match last_label {
                    Some(label) => frames.push((t0, t1, label)),
                    None => pending.push((t0, t1)),
                },
            }
        }
    }
    frames.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut turns: Vec<DiarTurn> = merge_frames(&frames, args.max_gap, 0.1);
    renumber_by_first_appearance(&mut turns);
    let speakers_found = turns.iter().map(|t| t.speaker).max().map(|m| m + 1).unwrap_or(0);

    let elapsed = started.elapsed().as_secs_f64();
    let result = json!({
        "num_speakers": speakers_found,
        "audio_seconds": audio_seconds,
        "elapsed_seconds": elapsed,
        "turns": turns,
    });
    if let Some(dir) = args.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&result)?).with_context(|| format!("writing {}", args.out.display()))?;
    event(json!({
        "event": "done",
        "output": args.out,
        "turns": turns.len(),
        "elapsed_seconds": elapsed,
        "realtime_factor": audio_seconds / elapsed.max(1e-9),
    }));
    Ok(())
}
