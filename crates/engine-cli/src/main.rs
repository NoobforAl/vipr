//! engine-cli - command line front end of the engine (the GUI will spawn this).
//!
//!   engine-cli detect [--json]
//!   engine-cli models list [--role asr]
//!   engine-cli models recommend [--preset balanced]
//!   engine-cli models fetch <id>...        download + verify
//!   engine-cli models verify <id>...       re-hash files on disk

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use engine::download::{ModelStore, Status};
use engine::registry::{Fit, Preset, Registry, Role, fit};

#[derive(Parser)]
#[command(name = "engine-cli", about = "ViPr engine")]
struct Cli {
    /// Path to models.json (default: search upwards from the current folder, then next to the exe)
    #[arg(long, global = true, env = "MT_REGISTRY")]
    registry: Option<PathBuf>,
    /// Folder for downloaded models (default: per-user data folder, or MT_MODELS_DIR)
    #[arg(long, global = true)]
    models_dir: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Show the GPU and the backend that would be used
    Detect {
        #[arg(long)]
        json: bool,
    },
    #[command(subcommand)]
    Models(ModelsCmd),
    #[command(subcommand)]
    Audio(AudioCmd),
    /// Download and unpack ffmpeg into ViPr's own folder (VIPR_TOOLS_DIR overrides the folder)
    InstallFfmpeg,
    /// Work out topic chapters from a plain text result ([HH:MM:SS] Speaker: text), to test them without the app
    Chapters {
        /// The .txt result
        #[arg(long)]
        txt: PathBuf,
        /// A general AI model (.gguf)
        #[arg(long)]
        model: PathBuf,
        /// Where the chapters go. The worker's input is written next to it.
        #[arg(long)]
        out: PathBuf,
        /// llm-worker to use (default: next to this program)
        #[arg(long)]
        worker: Option<PathBuf>,
    },
    /// Build the canonical transcript from an ASR result (and optional speaker turns) and write .json / .txt / .srt / .vtt
    Export {
        /// Result file written by asr-worker
        asr: PathBuf,
        /// Optional speaker turns file written by diar-worker
        #[arg(long)]
        turns: Option<PathBuf>,
        /// Output folder
        #[arg(long)]
        out_dir: PathBuf,
        /// File name without extension
        #[arg(long)]
        name: String,
        /// Speaker names in order, e.g. --speaker Eric --speaker Farshad
        #[arg(long = "speaker")]
        speakers: Vec<String>,
    },
}

#[derive(Subcommand)]
enum AudioCmd {
    /// Decode a video / audio file to mono 16 kHz and report its length (optionally save a WAV)
    Decode {
        input: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Compare our decoder with a reference WAV (e.g. the Python pipeline's audio/*.wav)
    Compare { input: PathBuf, reference: PathBuf },
}

#[derive(Subcommand)]
enum ModelsCmd {
    List {
        #[arg(long)]
        role: Option<Role>,
    },
    /// Default model per role for this machine
    Recommend {
        #[arg(long, default_value = "balanced")]
        preset: Preset,
    },
    Fetch {
        #[arg(required = true)]
        ids: Vec<String>,
    },
    Verify {
        #[arg(required = true)]
        ids: Vec<String>,
    },
}

fn load_registry(explicit: Option<PathBuf>) -> Result<Registry> {
    Registry::load_default(explicit.as_deref()).context("loading the model registry")
}

fn mb(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1} GB", bytes as f64 / (1u64 << 30) as f64)
    } else {
        format!("{} MB", bytes >> 20)
    }
}

fn fit_label(f: Fit) -> &'static str {
    match f {
        Fit::Fits => "fits",
        Fit::Tight => "TIGHT",
        Fit::TooBig => "too big",
        Fit::Unknown => "vram?",
        Fit::Unsupported => "no backend",
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let store = ModelStore::new(
        cli.models_dir
            .clone()
            .unwrap_or_else(ModelStore::default_dir),
    );

    match cli.cmd {
        Cmd::Detect { json } => {
            let d = gpu_detect::detect();
            if json {
                println!("{}", serde_json::to_string_pretty(&d)?);
            } else {
                for (i, g) in d.gpus.iter().enumerate() {
                    let vram = g
                        .vram_mb
                        .map(|m| format!("{m} MiB"))
                        .unwrap_or_else(|| "VRAM unknown".into());
                    println!(
                        "{} [{i}] {} ({vram})",
                        if d.recommended_gpu == Some(i) {
                            "*"
                        } else {
                            " "
                        },
                        g.name
                    );
                }
                println!("backend: {} - {}", d.recommended.as_str(), d.reason);
            }
        }
        Cmd::InstallFfmpeg => {
            let tool = engine::tools::ffmpeg_download()
                .context("no ffmpeg download is defined for this system")?;
            println!(
                "{} {} ({}), into {}",
                tool.name,
                tool.version,
                tool.licence,
                engine::tools::tools_dir().display()
            );
            let mut last = 0u64;
            let exe = engine::tools::install_ffmpeg(&mut |step| match step {
                engine::tools::InstallStep::Downloading { done, total } => {
                    if done.saturating_sub(last) >= (4 << 20) || done == total {
                        last = done;
                        print!(
                            "\rdownloading {} / {} ({}%)   ",
                            mb(done),
                            mb(total),
                            (done * 100).checked_div(total).unwrap_or(0)
                        );
                        let _ = std::io::stdout().flush();
                    }
                }
                engine::tools::InstallStep::Unpacking => println!("\nunpacking..."),
            })?;
            println!("ready: {}", exe.display());
        }
        Cmd::Chapters {
            txt,
            model,
            out,
            worker,
        } => {
            let lines = engine::chapters::lines_from_txt(
                &std::fs::read_to_string(&txt).context("reading the text result")?,
            );
            let duration = lines.last().map_or(0.0, |l| l.start + 30.0);
            let input = engine::project::ChaptersInput {
                version: engine::project::VERSION,
                duration,
                lines,
            };
            let input_path = out.with_extension("input.json");
            engine::fsio::save_json(&input_path, &input)?;
            let worker = match worker {
                Some(path) => path,
                None => std::env::current_exe()?.with_file_name("llm-worker"),
            };
            let status = std::process::Command::new(&worker)
                .args(["--mode", "chapters", "--model"])
                .arg(&model)
                .arg("--input")
                .arg(&input_path)
                .arg("--out")
                .arg(&out)
                .status()
                .with_context(|| format!("starting {}", worker.display()))?;
            anyhow::ensure!(status.success(), "the worker failed");
            let output: engine::project::ChaptersOutput = engine::fsio::load_json(&out)?;
            for c in &output.chapters {
                println!("{:>8}  {}", engine::chapters::clock(c.start), c.title);
            }
        }
        Cmd::Export {
            asr,
            turns,
            out_dir,
            name,
            speakers,
        } => {
            let asr_result = engine::transcript::AsrResult::load(&asr)?;
            let turns: Option<Vec<engine::transcript::SpeakerTurn>> = match turns {
                Some(p) => {
                    #[derive(serde::Deserialize)]
                    struct File {
                        turns: Vec<engine::transcript::SpeakerTurn>,
                    }
                    let f: File = serde_json::from_str(&std::fs::read_to_string(&p)?)
                        .context("reading speaker turns")?;
                    Some(f.turns)
                }
                None => None,
            };
            let t = engine::transcript::Transcript::build(
                &name,
                &asr_result,
                turns.as_deref(),
                &speakers,
            );
            std::fs::create_dir_all(&out_dir)?;
            t.save(&out_dir.join(format!("{name}.json")))?;
            std::fs::write(out_dir.join(format!("{name}.txt")), t.to_txt())?;
            let cues =
                engine::subtitles::build_cues(&t, &engine::subtitles::SubtitleOptions::default());
            std::fs::write(
                out_dir.join(format!("{name}.srt")),
                engine::subtitles::to_srt(&cues),
            )?;
            std::fs::write(
                out_dir.join(format!("{name}.vtt")),
                engine::subtitles::to_vtt(&cues),
            )?;
            println!(
                "{} utterances, {} subtitle cues -> {}",
                t.utterances.len(),
                cues.len(),
                out_dir.display()
            );
        }
        Cmd::Audio(AudioCmd::Decode { input, out }) => {
            let t = std::time::Instant::now();
            let samples = engine::audio::decode_to_mono_16k(&input)?;
            let secs = samples.len() as f64 / 16_000.0;
            println!(
                "{}: {:.2} s of audio, decoded in {:.1} s ({:.0}x real time)",
                input.display(),
                secs,
                t.elapsed().as_secs_f64(),
                secs / t.elapsed().as_secs_f64().max(1e-9)
            );
            if let Some(out) = out {
                engine::audio::write_wav_16k(&out, &samples)?;
                println!("saved {}", out.display());
            }
        }
        Cmd::Audio(AudioCmd::Compare { input, reference }) => {
            let ours = engine::audio::decode_to_mono_16k(&input)?;
            let (theirs, rate) = engine::audio::decode_mono(&reference)?;
            let theirs = engine::audio::resample_mono(theirs, rate, 16_000)?;
            // compare a 20 s window from the middle, searching +-100 ms for the best alignment
            let win = 20 * 16_000usize;
            let start = ours.len().min(theirs.len()).saturating_sub(win + 4_000) / 2;
            let (mut best_lag, mut best) = (0i64, f64::MIN);
            for lag in -1600i64..=1600 {
                let (mut sab, mut saa, mut sbb) = (0f64, 0f64, 0f64);
                for i in 0..win {
                    let (Some(a), Some(b)) = (
                        ours.get(start + i),
                        theirs
                            .get((start + i) as i64 as usize)
                            .filter(|_| true)
                            .and_then(|_| theirs.get(((start + i) as i64 + lag) as usize)),
                    ) else {
                        continue;
                    };
                    let (a, b) = (*a as f64, *b as f64);
                    sab += a * b;
                    saa += a * a;
                    sbb += b * b;
                }
                let c = sab / (saa.sqrt() * sbb.sqrt()).max(1e-12);
                if c > best {
                    best = c;
                    best_lag = lag;
                }
            }
            let n = ours.len().min(theirs.len());
            let (mut sab, mut saa, mut sbb) = (0f64, 0f64, 0f64);
            for i in 0..n {
                let (a, b) = (ours[i] as f64, theirs[i] as f64);
                sab += a * b;
                saa += a * a;
                sbb += b * b;
            }
            println!(
                "whole file at 0 ms shift: correlation {:.4}",
                sab / (saa.sqrt() * sbb.sqrt()).max(1e-12)
            );
            println!(
                "{}: ours {:.2} s | reference {:.2} s | diff {:+.3} s | best match {:.4} at shift {:+.1} ms",
                input
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                ours.len() as f64 / 16e3,
                theirs.len() as f64 / 16e3,
                (ours.len() as f64 - theirs.len() as f64) / 16e3,
                best,
                best_lag as f64 / 16.0
            );
        }
        Cmd::Models(ModelsCmd::List { role }) => {
            let reg = load_registry(cli.registry)?;
            let det = gpu_detect::detect();
            println!(
                "backend: {} | models dir: {}\n",
                det.recommended.as_str(),
                store.dir().display()
            );
            println!(
                "{:<34} {:<20} {:>8}  {:<10} {:<12} hash",
                "id", "roles", "size", "fit", "installed"
            );
            for m in reg
                .models
                .iter()
                .filter(|m| role.is_none_or(|r| m.has_role(r)))
            {
                let roles = m
                    .roles
                    .iter()
                    .map(|r| r.as_str())
                    .collect::<Vec<_>>()
                    .join(",");
                let state = match store.status(m) {
                    Status::Present => "yes".to_string(),
                    Status::Partial(n) => format!("partial {}", mb(n)),
                    Status::Missing => "no".into(),
                    Status::LocalMissing => "file missing".into(),
                };
                let hash = if m.sha256.is_some() {
                    "sha256"
                } else {
                    "UNVERIFIED"
                };
                println!(
                    "{:<34} {:<20} {:>8}  {:<10} {:<12} {}",
                    m.id,
                    roles,
                    mb(m.size_bytes),
                    fit_label(fit(m, &det)),
                    state,
                    hash
                );
            }
        }
        Cmd::Models(ModelsCmd::Recommend { preset }) => {
            let reg = load_registry(cli.registry)?;
            let det = gpu_detect::detect();
            println!("backend: {} | preset: {preset:?}", det.recommended.as_str());
            for role in Role::ALL {
                match reg.recommend(role, preset, &det) {
                    Some(m) => println!("{:<18} {}", role.as_str(), m.id),
                    None => println!("{:<18} (nothing fits)", role.as_str()),
                }
            }
        }
        Cmd::Models(ModelsCmd::Fetch { ids }) => {
            let reg = load_registry(cli.registry)?;
            let token = std::env::var("HF_TOKEN").ok();
            for id in ids {
                let m = reg.get(&id)?;
                let started = std::time::Instant::now();
                let mut last = 0u64;
                let path = store.fetch(m, token.as_deref(), &mut |done, total| {
                    if done - last >= 8 << 20 || (total > 0 && done == total) {
                        last = done;
                        let pct = (done * 100).checked_div(total).unwrap_or(0);
                        print!("\r{id}: {} / {} ({pct}%)   ", mb(done), mb(total));
                        let _ = std::io::stdout().flush();
                    }
                })?;
                println!(
                    "\r{id}: ok in {:.0}s -> {}{}",
                    started.elapsed().as_secs_f32(),
                    path.display(),
                    if m.sha256.is_some() {
                        " (sha256 verified)"
                    } else {
                        " (no hash in registry, NOT verified)"
                    }
                );
            }
        }
        Cmd::Models(ModelsCmd::Verify { ids }) => {
            let reg = load_registry(cli.registry)?;
            let mut bad = false;
            for id in ids {
                let m = reg.get(&id)?;
                match store.status(m) {
                    Status::Present => match store.verify(m)? {
                        Some(true) => println!("{id}: OK"),
                        Some(false) => {
                            println!("{id}: HASH MISMATCH");
                            bad = true;
                        }
                        None => println!("{id}: present, registry has no hash"),
                    },
                    other => {
                        println!("{id}: not installed ({other:?})");
                        bad = true;
                    }
                }
            }
            if bad {
                std::process::exit(1);
            }
        }
    }
    Ok(())
}
