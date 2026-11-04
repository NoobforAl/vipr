//! asr-worker - speech-to-text worker process (whisper.cpp through whisper-rs).
//!
//! Runs as its own executable because whisper.cpp and llama.cpp each bundle ggml and cannot be
//! linked into one binary (PLAN.md, Phase 0 finding 2). The GUI / engine starts it and reads
//! JSON-lines events from stdout; all logging from ggml goes to stderr.
//!
//!   asr-worker --model ggml-large-v3-turbo-q8_0.bin --audio meeting.mp4 --out meeting.asr.json
//!
//! Events (one JSON object per line on stdout):
//!   {"event":"audio","seconds":123.4}            audio decoded
//!   {"event":"model_loaded","seconds":1.2}
//!   {"event":"progress","percent":42}
//!   {"event":"done","output":"...","segments":200,"audio_seconds":..,"elapsed_seconds":..,"realtime_factor":..}
//!   {"event":"error","message":"..."}            followed by exit code 1

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use serde_json::json;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperVadContext, WhisperVadContextParams,
    WhisperVadParams,
};

#[derive(Parser)]
#[command(about = "Speech-to-text worker (whisper.cpp)")]
struct Args {
    /// Whisper GGML model file (.bin)
    #[arg(long)]
    model: PathBuf,
    /// Video or audio file (anything the engine can decode)
    #[arg(long)]
    audio: PathBuf,
    /// Where to write the result JSON
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "en")]
    language: String,
    /// Glossary file: names / jargon, one per line or comma separated, `#` comments
    #[arg(long)]
    glossary: Option<PathBuf>,
    /// Beam search width (1 = greedy, fastest)
    #[arg(long, default_value_t = 5)]
    beam: i32,
    /// Silero VAD model (GGML). Skips silence and prevents looping / invented words. Strongly recommended.
    #[arg(long)]
    vad: Option<PathBuf>,
    /// Do not use the GPU
    #[arg(long)]
    cpu: bool,
    #[arg(long, default_value_t = 4)]
    threads: i32,
}

#[derive(Serialize)]
struct Word {
    text: String,
    start: f64,
    end: f64,
    prob: f32,
}

#[derive(Serialize)]
struct Segment {
    start: f64,
    end: f64,
    text: String,
    no_speech_prob: f32,
    words: Vec<Word>,
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

/// Same format as the Python pipeline: one item per line or comma separated, `#` starts a comment.
fn load_glossary(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading glossary {}", path.display()))?;
    let mut terms: Vec<String> = Vec::new();
    for line in text.trim_start_matches('\u{feff}').lines() {
        for t in line.split('#').next().unwrap_or("").split(',') {
            let t = t.trim();
            if !t.is_empty() && !terms.iter().any(|x| x == t) {
                terms.push(t.to_string());
            }
        }
    }
    Ok(terms)
}

/// Turn whisper's segments / tokens into our segment + word list. `offset` (seconds) is the start of the
/// audio chunk that was transcribed, so times land on the original timeline.
fn collect_segments(state: &whisper_rs::WhisperState, offset: f64, out: &mut Vec<Segment>) {
    for seg in state.as_iter() {
        let text = seg.to_str_lossy().map(|s| s.trim().to_string()).unwrap_or_default();
        if text.is_empty() {
            continue;
        }
        let seg_start = offset + seg.start_timestamp() as f64 / 100.0;
        let seg_end = offset + seg.end_timestamp() as f64 / 100.0;

        // group tokens into words: a token that starts with a space begins a new word
        let mut words: Vec<Word> = Vec::new();
        for i in 0..seg.n_tokens() {
            let Some(tok) = seg.get_token(i) else { continue };
            let piece = tok.to_str_lossy().map(|s| s.into_owned()).unwrap_or_default();
            if piece.is_empty() || piece.starts_with("[_") || piece.starts_with("<|") {
                continue; // special token
            }
            let d = tok.token_data();
            let t0 = if d.t0 >= 0 { offset + d.t0 as f64 / 100.0 } else { seg_start };
            let t1 = if d.t1 >= 0 { offset + d.t1 as f64 / 100.0 } else { seg_end };
            match words.last_mut() {
                Some(w) if !piece.starts_with(' ') => {
                    w.text.push_str(&piece);
                    w.end = t1;
                    w.prob = w.prob.min(d.p);
                }
                _ => words.push(Word { text: piece.trim_start().to_string(), start: t0, end: t1, prob: d.p }),
            }
        }
        words.retain(|w| !w.text.trim().is_empty());
        out.push(Segment { start: seg_start, end: seg_end, text, no_speech_prob: seg.no_speech_probability(), words });
    }
}

fn run(args: &Args) -> Result<()> {
    let started = Instant::now();

    let samples = engine::audio::decode_to_mono_16k(&args.audio).context("decoding audio")?;
    let audio_seconds = samples.len() as f64 / 16_000.0;
    event(json!({"event": "audio", "seconds": audio_seconds}));

    let t = Instant::now();
    let mut cparams = WhisperContextParameters::default();
    cparams.use_gpu(!args.cpu);
    cparams.flash_attn(!args.cpu);
    let model_path = args.model.to_str().context("model path is not valid UTF-8")?;
    let ctx = WhisperContext::new_with_params(model_path, cparams)
        .with_context(|| format!("loading model {}", args.model.display()))?;
    let mut state = ctx.create_state().context("creating whisper state")?;
    event(json!({"event": "model_loaded", "seconds": t.elapsed().as_secs_f64()}));

    let prompt = match &args.glossary {
        Some(p) => {
            let terms = load_glossary(p)?;
            if terms.is_empty() { None } else { Some(format!("Glossary: {}.", terms.join(", "))) }
        }
        None => None,
    };

    // Fresh parameters for every call (whisper consumes them).
    let build_params = |with_progress: bool| {
        let strategy = if args.beam > 1 {
            SamplingStrategy::BeamSearch { beam_size: args.beam, patience: -1.0 }
        } else {
            SamplingStrategy::Greedy { best_of: 1 }
        };
        let mut params = FullParams::new(strategy);
        params.set_language(Some(args.language.as_str()));
        params.set_n_threads(args.threads);
        params.set_token_timestamps(true);
        params.set_no_context(true); // like condition_on_previous_text=False in the Python pipeline
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params.set_no_speech_thold(0.6);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        if let Some(p) = &prompt {
            params.set_initial_prompt(p);
        }
        if with_progress {
            let mut last_pct = -1;
            params.set_progress_callback_safe(move |pct: i32| {
                if pct != last_pct {
                    last_pct = pct;
                    event(json!({"event": "progress", "percent": pct}));
                }
            });
        }
        params
    };

    let mut segments: Vec<Segment> = Vec::new();

    if let Some(vad_model) = &args.vad {
        // whisper.cpp only applies its VAD inside whisper_full(), not with a state, so we run the
        // detector ourselves and transcribe the speech regions (merged into chunks of <= 28 s).
        let vad_path = vad_model.to_str().context("VAD model path is not valid UTF-8")?;
        let mut vctx_params = WhisperVadContextParams::new();
        vctx_params.set_use_gpu(false); // tiny model, CPU is fine
        let mut vctx = WhisperVadContext::new(vad_path, vctx_params).context("loading VAD model")?;
        let mut vparams = WhisperVadParams::new();
        vparams.set_min_silence_duration(600);
        vparams.set_speech_pad(200);
        vparams.set_max_speech_duration(25.0);
        let regions: Vec<(f64, f64)> = vctx
            .segments_from_samples(vparams, &samples)
            .context("running VAD")?
            .map(|seg| (seg.start as f64 / 100.0, seg.end as f64 / 100.0))
            .collect();
        let speech_seconds: f64 = regions.iter().map(|(s, e)| e - s).sum();
        event(json!({"event": "vad", "regions": regions.len(), "speech_seconds": speech_seconds}));

        let mut chunks: Vec<(f64, f64)> = Vec::new();
        for (s, e) in regions {
            match chunks.last_mut() {
                Some(c) if e - c.0 <= 28.0 => c.1 = e,
                _ => chunks.push((s, e)),
            }
        }
        let total = chunks.len().max(1);
        for (i, (cs, ce)) in chunks.iter().enumerate() {
            let from = ((cs * 16_000.0) as usize).min(samples.len());
            let to = ((ce * 16_000.0) as usize).min(samples.len());
            if to <= from + 1_600 {
                continue; // under 0.1 s
            }
            state.full(build_params(false), &samples[from..to]).context("transcription failed")?;
            collect_segments(&state, *cs, &mut segments);
            event(json!({"event": "progress", "percent": (i + 1) * 100 / total}));
        }
    } else {
        state.full(build_params(true), &samples).context("transcription failed")?;
        collect_segments(&state, 0.0, &mut segments);
    }

    let elapsed = started.elapsed().as_secs_f64();
    let result = json!({
        "model": args.model.file_name().map(|n| n.to_string_lossy().into_owned()),
        "language": args.language,
        "audio_seconds": audio_seconds,
        "elapsed_seconds": elapsed,
        "segments": segments,
    });
    if let Some(dir) = args.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&result)?).with_context(|| format!("writing {}", args.out.display()))?;

    event(json!({
        "event": "done",
        "output": args.out,
        "segments": segments.len(),
        "audio_seconds": audio_seconds,
        "elapsed_seconds": elapsed,
        "realtime_factor": audio_seconds / elapsed.max(1e-9),
    }));
    Ok(())
}
