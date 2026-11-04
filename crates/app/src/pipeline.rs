//! Runs one recording through the worker processes and writes the results.
//!
//!   asr-worker  ->  (diar-worker)  ->  merge  ->  (llm-worker)  ->  .json / .txt / .srt / .vtt
//!
//! Speaker detection and the AI cleanup are optional: when their model or worker is missing, or they
//! fail, the job continues without them and says so. Speech recognition is required.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use engine::project::ProjectPaths;
use engine::subtitles::{SubtitleOptions, build_cues, to_srt, to_vtt};
use engine::transcript::{AsrResult, SpeakerTurn, Transcript};
use serde_json::{Value, json};

#[derive(Clone, Debug, Default)]
pub struct About {
    pub description: String,
    pub category: String,
    /// Display name per speaker index; empty = "Speaker N".
    pub speaker_names: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct JobInput {
    pub path: PathBuf,
    pub out_dir: PathBuf,
    pub about: About,
    pub num_speakers: usize,
    pub asr_model: PathBuf,
    pub vad_model: PathBuf,
    pub diar_models: Option<(PathBuf, PathBuf)>,
    pub llm_model: Option<PathBuf>,
    pub glossary: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub enum Event {
    /// All steps of this run, in order (known before the first one starts).
    Plan(Vec<String>),
    /// Step `index` (0-based) has started.
    Step(usize),
    /// What the current step is doing right now, in plain words.
    Detail(String),
    /// Progress inside the current step and over the whole run, both 0..1.
    Progress {
        step: f32,
        overall: f32,
    },
    /// Something the user should know (a step was skipped, ...).
    Note(String),
    Done {
        transcript: Box<Transcript>,
        outputs: Vec<PathBuf>,
        /// Where the results went, with the real stem (see [`result_stem`]).
        paths: ProjectPaths,
    },
    Failed(String),
}

/// Look for a worker executable next to the app, in `workers/`, or in a build folder (development).
pub fn find_worker(name: &str) -> Option<PathBuf> {
    let file = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    let exe = std::env::current_exe().ok()?;
    for dir in exe.parent()?.ancestors() {
        for rel in [
            "",
            "workers",
            "release",
            "target/release",
            "target-diar/release",
            "target-llm/release",
        ] {
            let p = if rel.is_empty() {
                dir.join(&file)
            } else {
                dir.join(rel).join(&file)
            };
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// `glossary.txt` next to `models.json` (the project folder), else in the config folder.
pub fn find_glossary() -> Option<PathBuf> {
    let beside = engine::registry::locate_registry()
        .and_then(|p| p.parent().map(|d| d.join("glossary.txt")));
    let config = dirs::config_dir().map(|d| d.join(engine::APP_DIR_NAME).join("glossary.txt"));
    [beside, config].into_iter().flatten().find(|p| p.is_file())
}

pub fn open_in_file_manager(path: &Path) {
    #[cfg(windows)]
    let cmd = ("explorer", path.as_os_str().to_owned());
    #[cfg(target_os = "macos")]
    let cmd = ("open", path.as_os_str().to_owned());
    #[cfg(all(unix, not(target_os = "macos")))]
    let cmd = ("xdg-open", path.as_os_str().to_owned());
    let _ = Command::new(cmd.0).arg(cmd.1).spawn();
}

static RUNNING_WORKER: Mutex<Option<u32>> = Mutex::new(None);
static CANCELLED: AtomicBool = AtomicBool::new(false);

/// Forget an earlier cancel; call before a job starts.
pub fn reset_cancel() {
    CANCELLED.store(false, Ordering::SeqCst);
}

/// Stop the worker that is running now (if any) and make the rest of the job give up.
pub fn request_cancel() {
    CANCELLED.store(true, Ordering::SeqCst);
    kill_running_worker();
}

/// Kill the running worker process (used by Cancel and when the app closes, so nothing keeps the GPU busy).
pub fn kill_running_worker() {
    let pid = RUNNING_WORKER.lock().ok().and_then(|g| *g);
    let Some(pid) = pid else { return };
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .creation_flags(0x0800_0000)
            .status();
    }
    #[cfg(unix)]
    {
        let _ = Command::new("kill").args(["-9", &pid.to_string()]).status();
    }
}

/// Start a worker, feed every JSON event line to `on_event`, and return its `done` event.
fn run_worker(
    exe: &Path,
    args: &[String],
    log: &Path,
    on_event: &mut dyn FnMut(&Value),
) -> Result<Value, String> {
    let name = exe
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut cmd = Command::new(exe);
    cmd.args(args).stdout(Stdio::piped());
    // workers are separate programs: hand them the ffmpeg this app found (picked, downloaded or system-wide)
    if let Some(ff) = engine::audio::find_ffmpeg() {
        cmd.env("VIPR_FFMPEG", ff);
    }
    match File::create(log) {
        Ok(f) => cmd.stderr(Stdio::from(f)),
        Err(_) => cmd.stderr(Stdio::null()),
    };
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // no console window
    }
    if CANCELLED.load(Ordering::SeqCst) {
        return Err("Cancelled".into());
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("could not start {name}: {e}"))?;
    if let Ok(mut g) = RUNNING_WORKER.lock() {
        *g = Some(child.id());
    }
    let stdout = child.stdout.take().ok_or("no worker output")?;

    let mut done: Option<Value> = None;
    let mut error: Option<String> = None;
    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        let Ok(v) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        match v.get("event").and_then(Value::as_str) {
            Some("error") => {
                error = Some(
                    v.get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown error")
                        .to_string(),
                )
            }
            Some("done") => done = Some(v.clone()),
            _ => {}
        }
        on_event(&v);
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    if let Ok(mut g) = RUNNING_WORKER.lock() {
        *g = None;
    }
    if CANCELLED.load(Ordering::SeqCst) {
        return Err("Cancelled".into());
    }
    if let Some(e) = error {
        return Err(e);
    }
    if !status.success() {
        return Err(format!(
            "{name} stopped unexpectedly (exit code {:?}); details: {}",
            status.code(),
            log.display()
        ));
    }
    done.ok_or_else(|| format!("{name} finished without a result"))
}

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Turn the engine's technical audio errors into something a person can act on (no paths, no jargon).
fn friendly_error(file: &str, raw: &str) -> String {
    let lower = raw.to_lowercase();
    if lower.contains("needs ffmpeg") {
        format!(
            "\"{file}\" is in a format ViPr cannot read by itself (for example AVI, WMV, FLV, MPEG-TS, AMR or AC3). \
             Install ffmpeg or put ffmpeg next to vipr (see Settings > More formats), then add the file again."
        )
    } else if lower.contains("ffmpeg could not read") {
        format!(
            "ffmpeg could not read the sound in \"{file}\". The file may be damaged or have no audio."
        )
    } else if lower.contains("unsupported audio codec") {
        format!(
            "ViPr cannot read the audio in \"{file}\": this audio format is not supported yet. \
             Convert it to MP3, WAV or M4A and add it again."
        )
    } else if lower.contains("no audio track") {
        format!("\"{file}\" has no audio track, so there is nothing to transcribe.")
    } else if lower.contains("unsupported or damaged file") || lower.contains("too many damaged") {
        format!(
            "ViPr cannot open \"{file}\". The file may be damaged, or it is in a format that is not supported."
        )
    } else if lower.contains("no audio could be decoded") {
        format!("No sound could be read from \"{file}\".")
    } else {
        raw.to_string()
    }
}

fn pct_of(v: &Value) -> f32 {
    v.get("percent").and_then(Value::as_f64).unwrap_or(0.0) as f32 / 100.0
}

/// Progress over the whole run, 0..1: the steps before `step` are finished, and `step` itself is
/// `fraction` done. `steps` holds each step's name and weight.
fn overall_progress(steps: &[(&str, f32)], step: usize, fraction: f32) -> f32 {
    let total: f32 = steps.iter().map(|s| s.1).sum();
    let finished: f32 = steps.iter().take(step).map(|s| s.1).sum();
    let current = steps.get(step).map_or(0.0, |s| s.1);
    (finished + current * fraction.clamp(0.0, 1.0)) / total
}

/// Name of a recording's results. `earlier_source` is the file that already has results under the
/// plain name, if any: meeting.mp4 and meeting.mp3 must not overwrite each other's results, so the
/// second one gets "meeting (mp3)".
fn result_stem(path: &Path, earlier_source: Option<&str>) -> String {
    let stem = path
        .file_stem()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "recording".into());
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if earlier_source.is_some_and(|s| s != file_name) {
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().into_owned())
            .unwrap_or_default();
        return format!("{stem} ({ext})");
    }
    stem
}

/// Run the whole job. Blocking; call from a worker thread.
pub fn run(input: &JobInput, emit: &mut dyn FnMut(Event)) -> Result<(), String> {
    let file_name = input
        .path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let plain_stem = input
        .path
        .file_stem()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "recording".into());
    let earlier_source = std::fs::read_to_string(
        input
            .out_dir
            .join(&plain_stem)
            .join(format!("{plain_stem}.json")),
    )
    .ok()
    .and_then(|t| serde_json::from_str::<Value>(&t).ok())
    .and_then(|v| v.get("source").and_then(Value::as_str).map(str::to_string));
    let stem = result_stem(&input.path, earlier_source.as_deref());
    let dir = input.out_dir.join(&stem);
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;

    let asr_exe = find_worker("asr-worker")
        .ok_or("The speech engine (asr-worker) was not found next to the app.")?;
    let diar_exe = input
        .diar_models
        .as_ref()
        .and_then(|_| find_worker("diar-worker"));
    let llm_exe = input
        .llm_model
        .as_ref()
        .and_then(|_| find_worker("llm-worker"));

    // the steps of this run, and each step's share of the overall progress
    let mut steps: Vec<(&str, f32)> = vec![("Transcribing speech", 0.55)];
    if diar_exe.is_some() {
        steps.push(("Detecting speakers", 0.15));
    }
    if llm_exe.is_some() {
        steps.push(("Cleaning up the text", 0.25));
    }
    steps.push(("Saving files", 0.05));
    emit(Event::Plan(steps.iter().map(|s| s.0.to_string()).collect()));
    let mut step = 0usize;

    // 1. speech to text
    emit(Event::Step(step));
    emit(Event::Detail("Reading the audio from the file".into()));
    let asr_json = dir.join(format!("{stem}.asr.json"));
    let mut args = vec![
        "--model".into(),
        s(&input.asr_model),
        "--vad".into(),
        s(&input.vad_model),
        "--audio".into(),
        s(&input.path),
        "--out".into(),
        s(&asr_json),
        "--beam".into(),
        "1".into(),
    ];
    if let Some(g) = &input.glossary {
        args.extend(["--glossary".into(), s(g)]);
    }
    run_worker(
        &asr_exe,
        &args,
        &dir.join("asr-worker.log"),
        &mut |v| match v.get("event").and_then(Value::as_str) {
            Some("audio") => {
                let secs = v.get("seconds").and_then(Value::as_f64).unwrap_or(0.0);
                emit(Event::Detail(format!(
                    "Audio ready ({}). Loading the speech model",
                    engine::transcript::clock(secs)
                )));
            }
            Some("model_loaded") => emit(Event::Detail("Looking for the parts with speech".into())),
            Some("vad") => {
                let n = v.get("regions").and_then(Value::as_u64).unwrap_or(0);
                let sp = v
                    .get("speech_seconds")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                emit(Event::Detail(format!(
                    "Found {n} stretches of speech ({sp:.0} s). Writing down the words"
                )));
            }
            Some("progress") => emit(Event::Progress {
                step: pct_of(v),
                overall: overall_progress(&steps, step, pct_of(v)),
            }),
            _ => {}
        },
    )
    .map_err(|e| friendly_error(&file_name, &e))?;
    step += 1;
    let asr = AsrResult::load(&asr_json).map_err(|e| e.to_string())?;

    // 2. who speaks when (optional)
    let mut turns: Option<Vec<SpeakerTurn>> = None;
    if let (Some(exe), Some((seg, emb))) = (&diar_exe, &input.diar_models) {
        emit(Event::Step(step));
        emit(Event::Detail("Loading the speaker models".into()));
        let out = dir.join(format!("{stem}.turns.json"));
        let mut args = vec![
            "--audio".into(),
            s(&input.path),
            "--segmentation".into(),
            s(seg),
            "--embedding".into(),
            s(emb),
            "--out".into(),
            s(&out),
        ];
        // 0 = Auto: the worker works out how many people are talking
        if input.num_speakers > 0 {
            args.extend(["--num-speakers".into(), input.num_speakers.to_string()]);
        }
        let res = run_worker(exe, &args, &dir.join("diar-worker.log"), &mut |v| match v
            .get("event")
            .and_then(Value::as_str)
        {
            Some("audio") => emit(Event::Detail("Finding who is speaking when".into())),
            Some("progress") => emit(Event::Progress {
                step: pct_of(v),
                overall: overall_progress(&steps, step, pct_of(v)),
            }),
            _ => {}
        });
        match res.and_then(|_| std::fs::read_to_string(&out).map_err(|e| e.to_string())) {
            Ok(text) => match serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| serde_json::from_value(v["turns"].clone()).ok())
            {
                Some(t) => turns = Some(t),
                None => emit(Event::Note(
                    "Speaker detection gave an unreadable result; continuing without speakers."
                        .into(),
                )),
            },
            Err(e) => emit(Event::Note(format!(
                "Speaker detection failed ({e}); continuing without speakers."
            ))),
        }
        step += 1;
    } else {
        emit(Event::Note(
            "Speaker detection is not available; all speech is labelled as one speaker.".into(),
        ));
    }

    let mut transcript = Transcript::build(
        &input
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        &asr,
        turns.as_deref(),
        &input.about.speaker_names,
    );

    // 3. AI cleanup (optional)
    if let (Some(exe), Some(model)) = (&llm_exe, &input.llm_model) {
        emit(Event::Step(step));
        emit(Event::Detail("Loading the cleanup model".into()));
        let llm_in = dir.join(format!("{stem}.llm-in.json"));
        let llm_out = dir.join(format!("{stem}.llm-out.json"));
        let names: serde_json::Map<String, Value> = input
            .about
            .speaker_names
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.trim().is_empty())
            .map(|(i, n)| (format!("Speaker {}", i + 1), json!(n.trim())))
            .collect();
        let payload = json!({
            "utterances": transcript.utterances.iter().map(|u| json!({
                "id": u.id, "start": u.start, "speaker": format!("Speaker {}", u.speaker + 1), "text": u.raw_text,
            })).collect::<Vec<_>>(),
            "about": { "description": input.about.description, "speaker_names": names, "category": input.about.category },
        });
        std::fs::write(&llm_in, serde_json::to_string(&payload).unwrap_or_default())
            .map_err(|e| e.to_string())?;
        let mut args = vec![
            "--model".into(),
            s(model),
            "--input".into(),
            s(&llm_in),
            "--out".into(),
            s(&llm_out),
        ];
        if let Some(g) = &input.glossary {
            args.extend(["--glossary".into(), s(g)]);
        }
        let res = run_worker(exe, &args, &dir.join("llm-worker.log"), &mut |v| {
            if v.get("event").and_then(Value::as_str) == Some("progress") {
                emit(Event::Detail(
                    "Fixing misheard words and punctuation, a few lines at a time".into(),
                ));
                emit(Event::Progress {
                    step: pct_of(v),
                    overall: overall_progress(&steps, step, pct_of(v)),
                });
            }
        });
        match res.and_then(|_| std::fs::read_to_string(&llm_out).map_err(|e| e.to_string())) {
            Ok(text) => {
                let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
                let refined: Vec<(usize, String, Option<usize>)> = v["utterances"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|u| {
                                let id = u["id"].as_u64()? as usize;
                                let text = u["text"].as_str()?.to_string();
                                let speaker = u["speaker"]
                                    .as_str()
                                    .and_then(|l| l.strip_prefix("Speaker "))
                                    .and_then(|n| n.trim().parse::<usize>().ok())
                                    .map(|n| n.saturating_sub(1));
                                Some((id, text, speaker))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if refined.is_empty() {
                    emit(Event::Note(
                        "The text cleanup returned nothing; keeping the original text.".into(),
                    ));
                } else {
                    transcript.apply_refined(&refined);
                }
            }
            Err(e) => emit(Event::Note(format!(
                "Text cleanup failed ({e}); keeping the original text."
            ))),
        }
        step += 1;
    }

    // 4. write the files
    emit(Event::Step(step));
    emit(Event::Detail("Writing .txt, .srt, .vtt and .json".into()));
    let outputs = write_outputs(&transcript, &dir, &stem)?;
    emit(Event::Progress {
        step: 1.0,
        overall: 1.0,
    });
    emit(Event::Done {
        transcript: Box::new(transcript),
        outputs,
        paths: ProjectPaths::new(dir, stem),
    });
    Ok(())
}

/// Write `.json`, `.txt`, `.srt` and `.vtt` for a transcript into `dir`.
pub fn write_outputs(t: &Transcript, dir: &Path, stem: &str) -> Result<Vec<PathBuf>, String> {
    let cues = build_cues(t, &SubtitleOptions::default());
    let files = [
        (format!("{stem}.json"), None),
        (format!("{stem}.txt"), Some(t.to_txt())),
        (format!("{stem}.srt"), Some(to_srt(&cues))),
        (format!("{stem}.vtt"), Some(to_vtt(&cues))),
    ];
    let mut out = Vec::new();
    for (name, text) in files {
        let path = dir.join(name);
        match text {
            None => t.save(&path).map_err(|e| e.to_string())?,
            Some(body) => std::fs::write(&path, body)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?,
        }
        out.push(path);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_STEPS: [(&str, f32); 4] = [
        ("Transcribing speech", 0.55),
        ("Detecting speakers", 0.15),
        ("Cleaning up the text", 0.25),
        ("Saving files", 0.05),
    ];

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn a_step_after_cleanup_counts_the_cleanup_as_finished() {
        // the save step starts at 95%; it used to count from 70%, as if cleanup had not run
        assert!(close(overall_progress(&ALL_STEPS, 3, 0.0), 0.95));
        assert!(close(overall_progress(&ALL_STEPS, 3, 0.5), 0.975));
        assert!(close(overall_progress(&ALL_STEPS, 3, 1.0), 1.0));
    }

    #[test]
    fn progress_rises_through_the_steps() {
        assert!(close(overall_progress(&ALL_STEPS, 0, 0.0), 0.0));
        assert!(close(overall_progress(&ALL_STEPS, 0, 1.0), 0.55));
        assert!(close(overall_progress(&ALL_STEPS, 1, 0.0), 0.55));
        assert!(close(overall_progress(&ALL_STEPS, 2, 0.5), 0.825));
        let mut last = -1.0;
        for step in 0..ALL_STEPS.len() {
            for tenth in 0..=10 {
                let now = overall_progress(&ALL_STEPS, step, tenth as f32 / 10.0);
                assert!(now >= last - 1e-5, "went back at step {step}");
                last = now;
            }
        }
    }

    #[test]
    fn progress_scales_when_optional_steps_are_missing() {
        let steps = [("Transcribing speech", 0.55), ("Saving files", 0.05)];
        assert!(close(overall_progress(&steps, 1, 0.0), 0.55 / 0.60));
        assert!(close(overall_progress(&steps, 1, 1.0), 1.0));
    }

    #[test]
    fn progress_stays_in_range() {
        assert!(close(overall_progress(&ALL_STEPS, 0, -3.0), 0.0));
        assert!(close(overall_progress(&ALL_STEPS, 3, 7.0), 1.0));
        assert!(close(overall_progress(&ALL_STEPS, 99, 0.5), 1.0));
    }

    #[test]
    fn second_format_of_a_recording_gets_its_own_stem() {
        let mp3 = Path::new("C:/media/meeting.mp3");
        assert_eq!(result_stem(mp3, None), "meeting");
        assert_eq!(result_stem(mp3, Some("meeting.mp3")), "meeting");
        assert_eq!(result_stem(mp3, Some("meeting.mp4")), "meeting (mp3)");
    }

    #[test]
    fn saving_again_with_the_stored_stem_rewrites_the_same_files() {
        let dir = std::env::temp_dir().join(format!("vipr-pipeline-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let media = Path::new("C:/media/meeting.mp3");
        let stem = result_stem(media, Some("meeting.mp4"));
        let t = Transcript::build("meeting.mp3", &empty_asr(), None, &[]);
        let first = write_outputs(&t, &dir, &stem).unwrap();
        let paths = ProjectPaths::new(&dir, stem);
        let second = write_outputs(&t, &paths.dir, &paths.stem).unwrap();
        let wrote_transcript = paths.transcript().is_file();
        let plain_name_exists = dir.join("meeting.json").exists();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(first, second);
        assert!(wrote_transcript);
        assert!(!plain_name_exists, "would overwrite meeting.mp4's results");
    }

    fn empty_asr() -> AsrResult {
        AsrResult {
            model: None,
            language: String::new(),
            audio_seconds: 0.0,
            elapsed_seconds: 0.0,
            segments: Vec::new(),
        }
    }
}
