//! llm-worker - text-cleanup worker process (llama.cpp through llama-cpp-2).
//!
//! Own executable because whisper.cpp and llama.cpp each bundle ggml and cannot share a binary
//! (PLAN.md, Phase 0 finding 2). The app starts it and reads JSON-lines events from stdout.
//!
//!   llm-worker --model Qwen3-4B-Instruct-2507-Q4_K_M.gguf --input utterances.json --out refined.json \
//!              [--glossary glossary.txt] [--cpu] [--ctx 4096]
//!
//! Input JSON:
//!   {"utterances":[{"id":1,"start":12.3,"speaker":"Speaker 1","text":"..."}],
//!    "about":{"description":"...","speaker_names":{"Speaker 1":"Eric"},"category":"software-meeting"},
//!    "glossary":["Cloudflare"]}                                  (about / glossary optional)
//! Output JSON:
//!   {"utterances":[same ids, cleaned text / speaker],
//!    "changes":[{"id":N,"kind":"TEXT|SPEAKER|REJECTED","before":"..","after":"..","reason":"rewrite|speaker-flip"}],
//!    "stats":{"text_edits":N,"speaker_edits":N,"rejected":N,"skipped":N,"elapsed_seconds":..}}
//!
//! Events on stdout (one JSON object per line):
//!   {"event":"model_loaded","seconds":..}  {"event":"progress","percent":..}
//!   {"event":"done","output":"..","text_edits":..,"speaker_edits":..,"rejected":..,"skipped":..,"elapsed_seconds":..}
//!   {"event":"error","message":".."}  followed by exit code 1

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use engine::prompts;
use llm_worker::llama::{LlamaChat, init_backend, load_model};
use llm_worker::refine::{DEFAULT_CHUNK_LINES, Input, Refiner};
use serde_json::json;

#[derive(Parser)]
#[command(about = "Transcript text-cleanup worker (llama.cpp)")]
struct Args {
    /// GGUF model file (Qwen3-4B-Instruct-2507 recommended)
    #[arg(long)]
    model: PathBuf,
    /// Input JSON (see the top of main.rs)
    #[arg(long)]
    input: PathBuf,
    /// Where to write the result JSON
    #[arg(long)]
    out: PathBuf,
    /// Glossary file (names / jargon, one per line or comma separated, `#` comments); merged with the input's glossary
    #[arg(long)]
    glossary: Option<PathBuf>,
    /// `prompts/` folder with the category files (default: searched upwards from the current folder / exe)
    #[arg(long)]
    prompts_dir: Option<PathBuf>,
    /// Do not use the GPU
    #[arg(long)]
    cpu: bool,
    /// Context size in tokens (a 20-line chunk with context needs about 1500)
    #[arg(long, default_value_t = 4096)]
    ctx: u32,
    /// Lines per model call
    #[arg(long, default_value_t = DEFAULT_CHUNK_LINES)]
    chunk_lines: usize,
    /// Keep the attention cache in 16 bit instead of 8 bit (uses about 0.3 GB more at ctx 4096)
    #[arg(long)]
    kv_f16: bool,
    #[arg(long, default_value_t = 4)]
    threads: i32,
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

/// One item per line or comma separated, `#` starts a comment (same format as the Python pipeline / asr-worker).
fn load_glossary_file(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading glossary {}", path.display()))?;
    let mut terms = Vec::new();
    for line in text.trim_start_matches('\u{feff}').lines() {
        for t in line.split('#').next().unwrap_or("").split(',') {
            let t = t.trim();
            if !t.is_empty() {
                terms.push(t.to_string());
            }
        }
    }
    Ok(terms)
}

fn run(args: &Args) -> Result<()> {
    let started = Instant::now();
    let text = std::fs::read_to_string(&args.input)
        .with_context(|| format!("reading {}", args.input.display()))?;
    let input: Input =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", args.input.display()))?;
    if input.utterances.is_empty() {
        bail!("the input has no utterances");
    }

    let file_glossary = match &args.glossary {
        Some(p) => load_glossary_file(p)?,
        None => Vec::new(),
    };
    let glossary = prompts::merge_glossary([input.glossary.as_slice(), file_glossary.as_slice()]);

    let prompts_dir = args.prompts_dir.clone().or_else(prompts::locate_dir);
    let (category, found) =
        prompts::load_category_or_default(prompts_dir.as_deref(), input.about.category.as_deref());
    if !found {
        eprintln!(
            "warning: prompt category {:?} not found, using the default",
            input.about.category
        );
    }

    let mut labels: Vec<String> = Vec::new();
    for u in &input.utterances {
        if !labels.contains(&u.speaker) {
            labels.push(u.speaker.clone());
        }
    }

    let t = Instant::now();
    let backend = init_backend()?;
    let model = load_model(&backend, &args.model, args.cpu)?;
    let chat = LlamaChat::new(&backend, &model, args.ctx, args.threads, !args.kv_f16)?;
    event(json!({"event": "model_loaded", "seconds": t.elapsed().as_secs_f64()}));

    let mut refiner = Refiner::new(chat, &category, &labels, glossary, input.about.clone());
    let mut last_pct = -1i64;
    let output = refiner.refine(&input.utterances, args.chunk_lines, &mut |done, total| {
        let pct = (done * 100 / total.max(1)) as i64;
        if pct != last_pct {
            last_pct = pct;
            event(json!({"event": "progress", "percent": pct}));
        }
    })?;

    if let Some(dir) = args.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&output)?)
        .with_context(|| format!("writing {}", args.out.display()))?;

    event(json!({
        "event": "done",
        "output": args.out,
        "text_edits": output.stats.text_edits,
        "speaker_edits": output.stats.speaker_edits,
        "rejected": output.stats.rejected,
        "skipped": output.stats.skipped,
        "elapsed_seconds": started.elapsed().as_secs_f64(),
    }));
    Ok(())
}
