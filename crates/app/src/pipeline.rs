//! Runs one recording through the worker processes and writes the results.
//!
//!   asr-worker + (diar-worker, alongside)  ->  merge  ->  (llm-worker)  ->  .json / .txt / .srt / .vtt
//!
//! Speaker detection runs beside speech recognition (that one is mostly the GPU, this the CPU), starting as soon as
//! asr-worker has decoded the recording and saved it as 16 kHz audio, which it reads instead of decoding the video
//! again.
//!
//! Speaker detection and the AI cleanup are optional: when their model or worker is missing, or they
//! fail, the job continues without them and says so. Speech recognition is required.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, mpsc};

use crate::export::{Encoder, ExportOptions, SubtitleFormat};
use engine::cuts::Cut;
use engine::fsio;
use engine::profiles::{GlossaryTerm, Profile};
use engine::project::{CueStatus, ProjectMeta, ProjectPaths, TimedText, TranslationFile};
use engine::style::SubtitleStyle;
use engine::subtitles::{
    Cue, SubtitleOptions, build_cues, mark_rtl, mark_rtl_text, to_srt, to_vtt, wrap,
};
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
        for rel in ["", "workers", "release", "target/release"] {
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

/// Worker processes running now: usually one, two while speaker detection runs beside transcription.
static RUNNING_WORKERS: Mutex<Vec<u32>> = Mutex::new(Vec::new());
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

/// Keeps a worker in `RUNNING_WORKERS` until dropped, so an early return cannot leave a stale process id behind
/// (one a later Cancel would kill after the system has reused it).
struct Tracked(u32);

impl Tracked {
    fn new(pid: u32) -> Self {
        if let Ok(mut g) = RUNNING_WORKERS.lock() {
            g.push(pid);
        }
        Self(pid)
    }
}

impl Drop for Tracked {
    fn drop(&mut self) {
        if let Ok(mut g) = RUNNING_WORKERS.lock() {
            g.retain(|&p| p != self.0);
        }
    }
}

/// Kill the running worker processes (used by Cancel and when the app closes, so nothing keeps the GPU busy).
pub fn kill_running_worker() {
    let pids = RUNNING_WORKERS
        .lock()
        .map(|g| g.clone())
        .unwrap_or_default();
    for pid in pids {
        kill_pid(pid);
    }
}

fn kill_pid(pid: u32) {
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
pub(crate) fn run_worker(
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
    let tracked = Tracked::new(child.id());
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
    drop(tracked);
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

pub(crate) fn s(p: &Path) -> String {
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

pub(crate) fn pct_of(v: &Value) -> f32 {
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

/// Where a recording's results go: `<out_dir>/<stem>/`, with the stem from [`result_stem`].
fn result_paths(path: &Path, out_dir: &Path) -> ProjectPaths {
    let plain_stem = path
        .file_stem()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "recording".into());
    let earlier_source =
        std::fs::read_to_string(out_dir.join(&plain_stem).join(format!("{plain_stem}.json")))
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .and_then(|v| v.get("source").and_then(Value::as_str).map(str::to_string));
    let stem = result_stem(path, earlier_source.as_deref());
    ProjectPaths::new(out_dir.join(&stem), stem)
}

/// Terms of a glossary file: one per line or comma separated, `#` starts a comment.
fn parse_glossary_file(text: &str) -> Vec<String> {
    text.trim_start_matches('\u{feff}')
        .lines()
        .flat_map(|line| line.split('#').next().unwrap_or("").split(','))
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

/// A profile term as one glossary file entry: anything the file format gives a meaning to is removed,
/// so a term can't turn into several terms or a comment. Empty when nothing usable is left.
fn glossary_entry(term: &str) -> String {
    let cleaned: String = term
        .chars()
        .map(|c| {
            if c.is_control() || matches!(c, ',' | '#' | '=' | '\u{2028}' | '\u{2029}') {
                ' '
            } else {
                c
            }
        })
        .collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The merged glossary: the terms of the `base` file, then the profile's. A term's `target` is not in
/// here: the translate worker gets those through its input.
pub(crate) fn merged_glossary(base: Option<&str>, terms: &[GlossaryTerm]) -> String {
    let from_base = base.map(parse_glossary_file).unwrap_or_default();
    let from_profile: Vec<String> = terms.iter().map(|t| glossary_entry(&t.source)).collect();
    let mut out =
        engine::prompts::merge_glossary([from_base.as_slice(), from_profile.as_slice()]).join("\n");
    if !out.is_empty() {
        out.push('\n');
    }
    out
}

/// Writes `<stem>.glossary.txt` from the `base` file and the profile's terms, and returns its path.
/// `None` when there are no terms at all.
pub(crate) fn write_merged_glossary(
    paths: &ProjectPaths,
    base: Option<&Path>,
    profile: Option<&Profile>,
) -> Result<Option<PathBuf>, String> {
    let base = match base {
        Some(p) => Some(
            std::fs::read_to_string(p).map_err(|e| format!("cannot read {}: {e}", p.display()))?,
        ),
        None => None,
    };
    let terms = profile.map_or(&[][..], |p| p.glossary.as_slice());
    let merged = merged_glossary(base.as_deref(), terms);
    if merged.is_empty() {
        return Ok(None);
    }
    let path = paths.glossary();
    fsio::write_atomic(&path, merged.as_bytes()).map_err(|e| e.to_string())?;
    Ok(Some(path))
}

/// Adds the profile's background to what the cleanup step is told about the recording. It stays marked
/// as background there, so the fixed rules still win.
fn with_background(description: &str, prompt: &str) -> String {
    match (description.trim(), prompt.trim()) {
        (d, "") => d.to_string(),
        ("", p) => p.to_string(),
        (d, p) => format!("{d} {p}"),
    }
}

/// Makes a job use the active profile: its glossary terms go into `<stem>.glossary.txt` and through
/// `--glossary`, and its prompt becomes background for the cleanup. Call just before [`run`]. On an
/// error the job is left as it was and can still run; tell the user.
pub fn apply_profile(input: &mut JobInput, profile: &Profile) -> Result<(), String> {
    if !profile.glossary.is_empty() {
        let paths = result_paths(&input.path, &input.out_dir);
        let base = input.glossary.as_deref();
        if let Some(path) = write_merged_glossary(&paths, base, Some(profile))? {
            input.glossary = Some(path);
        }
    }
    input.about.description = with_background(&input.about.description, &profile.prompt);
    Ok(())
}

/// Run the whole job. Blocking; call from a worker thread.
pub fn run(input: &JobInput, emit: &mut dyn FnMut(Event)) -> Result<(), String> {
    let file_name = input
        .path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let ProjectPaths { dir, stem } = result_paths(&input.path, &input.out_dir);
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
    // the speaker step reads the audio ASR already decoded instead of decoding the video a second time;
    // removed when this run ends, however it ends
    let shared_audio = diar_exe
        .as_ref()
        .map(|_| TempFile(dir.join(format!("{stem}.16k.wav"))));
    if let Some(a) = &shared_audio {
        args.extend(["--save-audio".into(), s(&a.0)]);
    }
    // the speaker worker, less its --audio; it runs beside transcription (that is mostly the GPU, this the CPU)
    let diar_out = dir.join(format!("{stem}.turns.json"));
    let diar = diar_exe
        .as_ref()
        .zip(input.diar_models.as_ref())
        .map(|(exe, (seg, emb))| {
            let mut args = vec![
                "--segmentation".into(),
                s(seg),
                "--embedding".into(),
                s(emb),
                "--out".into(),
                s(&diar_out),
            ];
            // 0 = Auto: the worker works out how many people are talking
            if input.num_speakers > 0 {
                args.extend(["--num-speakers".into(), input.num_speakers.to_string()]);
            }
            (exe.clone(), args)
        });
    let diar_log = dir.join("diar-worker.log");

    let (asr, turns) = std::thread::scope(|scope| -> Result<_, String> {
        let (diar_tx, diar_rx) = mpsc::channel::<Value>();
        let mut diar_tx = Some(diar_tx);
        let mut diar_run: Option<std::thread::ScopedJoinHandle<Result<Value, String>>> = None;
        // starts the speaker worker, on the shared audio once ASR has saved it, else on the recording itself
        let mut start_diar = |diar_run: &mut Option<_>| {
            let (Some((exe, args)), Some(tx)) = (&diar, diar_tx.take()) else {
                return;
            };
            let audio = shared_audio
                .as_ref()
                .map(|a| a.0.clone())
                .filter(|p| p.is_file())
                .unwrap_or_else(|| input.path.clone());
            let mut args = args.clone();
            args.extend(["--audio".into(), s(&audio)]);
            let log = &diar_log;
            *diar_run = Some(scope.spawn(move || {
                run_worker(exe, &args, log, &mut |v| {
                    let _ = tx.send(v.clone());
                })
            }));
        };

        let asr_run = run_worker(
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
                    start_diar(&mut diar_run);
                }
                Some("model_loaded") => {
                    emit(Event::Detail("Looking for the parts with speech".into()))
                }
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
        );
        if let Err(e) = asr_run {
            // the run is over: stop the speaker worker too rather than wait for it
            if diar_run.is_some() {
                kill_running_worker();
            }
            return Err(friendly_error(&file_name, &e));
        }
        step += 1;
        let asr = AsrResult::load(&asr_json).map_err(|e| e.to_string())?;

        // 2. who speaks when (optional); usually well under way by now
        let mut turns: Option<Vec<SpeakerTurn>> = None;
        if diar.is_none() {
            emit(Event::Note(
                "Speaker detection is not available; all speech is labelled as one speaker.".into(),
            ));
            return Ok((asr, turns));
        }
        emit(Event::Step(step));
        emit(Event::Detail("Finding who is speaking when".into()));
        if diar_run.is_none() {
            start_diar(&mut diar_run);
        }
        drop(start_diar);
        // the sender lives in the speaker thread, so this ends when the worker does
        for v in diar_rx {
            if v.get("event").and_then(Value::as_str) == Some("progress") {
                emit(Event::Progress {
                    step: pct_of(&v),
                    overall: overall_progress(&steps, step, pct_of(&v)),
                });
            }
        }
        let res = diar_run
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err("speaker detection crashed".into()))
            })
            .unwrap_or_else(|| Err("speaker detection did not start".into()));
        match res.and_then(|_| std::fs::read_to_string(&diar_out).map_err(|e| e.to_string())) {
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
            // a Cancel stops the whole run, not just this step
            Err(e) if CANCELLED.load(Ordering::SeqCst) => return Err(e),
            Err(e) => emit(Event::Note(format!(
                "Speaker detection failed ({e}); continuing without speakers."
            ))),
        }
        step += 1;
        Ok((asr, turns))
    })?;
    drop(shared_audio);

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
            match v.get("event").and_then(Value::as_str) {
                // said as soon as the model is ready: the first progress comes only after a whole batch of lines
                Some("model_loaded") => emit(Event::Detail(
                    "Fixing misheard words and punctuation, a few lines at a time".into(),
                )),
                Some("progress") => emit(Event::Progress {
                    step: pct_of(v),
                    overall: overall_progress(&steps, step, pct_of(v)),
                }),
                _ => {}
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
    let paths = ProjectPaths::new(dir, stem);
    // only Open result needs this, and it can ask for the media instead
    if let Err(e) = ProjectMeta::new(&input.path, &paths.stem).save(&paths.project()) {
        emit(Event::Note(format!(
            "Could not save the link to the media file ({e}); Open result will ask where it is."
        )));
    }
    emit(Event::Progress {
        step: 1.0,
        overall: 1.0,
    });
    emit(Event::Done {
        transcript: Box::new(transcript),
        outputs,
        paths,
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
            Some(body) => fsio::write_atomic(&path, body.as_bytes())
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?,
        }
        out.push(path);
    }
    Ok(out)
}

/// The subtitles an export uses, on the original timeline: ViPr's own, or a translation's. With the language they are
/// in, for the marks a right-to-left language needs.
pub fn export_cues(t: &Transcript, translation: Option<&TranslationFile>) -> (Vec<Cue>, String) {
    match translation {
        Some(file) => (translated_cues(file), file.target.clone()),
        None => (
            build_cues(t, &SubtitleOptions::default()),
            t.language.clone(),
        ),
    }
}

/// The cues as they fall once `cuts` are left out. A cue inside a cut, or left too short by one, goes.
pub fn retime_cues(cues: &[Cue], cuts: &[Cut]) -> Vec<Cue> {
    cues.iter()
        .map(|c| Cue {
            start: engine::cuts::shift(cuts, c.start),
            end: engine::cuts::shift(cuts, c.end),
            ..c.clone()
        })
        .filter(|c| c.end - c.start >= 0.2)
        .collect()
}

/// A subtitle file's text: `srt`, `vtt` or `txt`.
pub fn subtitle_text(
    cues: &[Cue],
    lang: &str,
    format: SubtitleFormat,
    style: &SubtitleStyle,
) -> String {
    match format {
        SubtitleFormat::Ass => engine::style::to_ass(&mark_rtl(cues, lang), style),
        SubtitleFormat::Srt => to_srt(&mark_rtl(cues, lang)),
        SubtitleFormat::Vtt => to_vtt(&mark_rtl(cues, lang)),
        SubtitleFormat::Txt => {
            let txt: String = cues.iter().map(|c| txt_line(c.start, &c.text)).collect();
            mark_rtl_text(&txt, lang)
        }
    }
}

/// The subtitle file ffmpeg reads while exporting, without its ending. A fixed plain name in the folder ffmpeg runs in:
/// a full Windows path would need escaping in a filter, and a wrong escape is a broken export.
const EXPORT_SUBTITLES: &str = "vipr-export";

/// Runs an export with ffmpeg: `srt` is the subtitle file it reads, if any. `dirs` is the folder to work in and the
/// file to write. Each of `options`' encoder attempts is tried in turn until one runs. Writes to a `.part` file and
/// renames it at the end, so a failed or cancelled export never leaves a broken file behind. `on_progress` gets 0..1.
pub fn run_export(
    ffmpeg: &Path,
    (options, media, subtitles): (&ExportOptions, &Path, Option<(&str, &str)>),
    dirs: (&Path, &Path),
    (duration, cuts): (f64, &[Cut]),
    log: &Path,
    on_progress: &mut dyn FnMut(f32),
) -> Result<(), String> {
    let (work_dir, out) = dirs;
    let name = subtitles.map(|(_, ext)| format!("{EXPORT_SUBTITLES}.{ext}"));
    let subs = work_dir.join(name.as_deref().unwrap_or(EXPORT_SUBTITLES));
    if let Some((text, _)) = subtitles {
        fsio::write_atomic(&subs, text.as_bytes())
            .map_err(|e| format!("cannot write the subtitles: {e}"))?;
    }
    let part = out.with_extension(format!("{}.part", options.extension()));
    let out_seconds = duration - engine::cuts::removed(cuts);
    let mut result = Err("no encoder could run".to_string());
    for video in options.video_attempts() {
        let args = options.ffmpeg_args(media, name.as_deref(), (duration, cuts), &video, &part);
        result = run_ffmpeg(ffmpeg, work_dir, &args, out_seconds, log, on_progress);
        if result.is_ok() || CANCELLED.load(Ordering::SeqCst) {
            break;
        }
    }
    let _ = std::fs::remove_file(&subs);
    match result {
        Ok(()) => std::fs::rename(&part, out)
            .map_err(|e| format!("could not save {} ({e})", out.display())),
        Err(e) if options.encoder == Encoder::Gpu && e != "Cancelled" => {
            let _ = std::fs::remove_file(&part);
            Err(format!(
                "the graphics card could not make {} video here. Choose Auto or Processor. ({e})",
                options.codec.label()
            ))
        }
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            Err(e)
        }
    }
}

/// One frame of `media` at `at` seconds with the ASS subtitles `ass` drawn on, as PNG bytes about 960 pixels wide.
pub fn preview_frame(ffmpeg: &Path, media: &Path, ass: &str, at: f64) -> Result<Vec<u8>, String> {
    let dir = std::env::temp_dir().join(format!("vipr-preview-{}", std::process::id()));
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("cannot make a folder for the preview: {e}"))?;
    fsio::write_atomic(&dir.join("preview.ass"), ass.as_bytes())
        .map_err(|e| format!("cannot write the preview subtitles: {e}"))?;
    let mut cmd = Command::new(ffmpeg);
    // seeking before the input is fast, and -copyts keeps the original times the subtitles are matched against
    cmd.current_dir(&dir)
        .args(["-hide_banner", "-nostdin", "-v", "error", "-ss"])
        .arg(format!("{at:.3}"))
        .args(["-copyts", "-i"])
        .arg(media)
        .args(["-map", "0:v:0", "-vf", "subtitles=preview.ass,scale=960:-2"])
        .args(["-frames:v", "1", "-f", "image2", "-c:v", "png", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // no console window
    }
    let out = cmd
        .output()
        .map_err(|e| format!("could not start ffmpeg: {e}"))?;
    let _ = std::fs::remove_dir_all(&dir);
    if out.status.success() && !out.stdout.is_empty() {
        return Ok(out.stdout);
    }
    let why = String::from_utf8_lossy(&out.stderr);
    if why.contains("matches no streams") {
        return Err("this recording has no picture to preview on.".into());
    }
    Err(format!(
        "ffmpeg could not make the preview ({})",
        engine::lang::clean_separators(why.lines().last().unwrap_or("no reason given").trim())
    ))
}

fn run_ffmpeg(
    ffmpeg: &Path,
    work_dir: &Path,
    args: &[String],
    duration: f64,
    log: &Path,
    on_progress: &mut dyn FnMut(f32),
) -> Result<(), String> {
    let mut cmd = Command::new(ffmpeg);
    cmd.current_dir(work_dir).args(args).stdout(Stdio::piped());
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
        .map_err(|e| format!("could not start ffmpeg: {e}"))?;
    let tracked = Tracked::new(child.id());
    if let Some(stdout) = child.stdout.take() {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Some(done) = line
                .strip_prefix("out_time_us=")
                .and_then(|us| us.trim().parse::<f64>().ok())
                .filter(|_| duration > 0.0)
            {
                on_progress((done / 1e6 / duration).clamp(0.0, 1.0) as f32);
            }
        }
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    drop(tracked);
    if CANCELLED.load(Ordering::SeqCst) {
        return Err("Cancelled".into());
    }
    if status.success() {
        return Ok(());
    }
    let details = std::fs::read_to_string(log).unwrap_or_default();
    if details.contains("matches no streams") || details.contains("does not contain any stream") {
        return Err("this recording has no picture or no sound for this kind of export.".into());
    }
    Err(format!(
        "ffmpeg could not make the file (exit code {:?}); details: {}",
        status.code(),
        log.display()
    ))
}

/// ViPr's own subtitles as a worker or compare takes them. Ids are the position in `build_cues`, plus one.
pub(crate) fn transcript_cues(t: &Transcript) -> Vec<TimedText> {
    build_cues(t, &SubtitleOptions::default())
        .into_iter()
        .enumerate()
        .map(|(i, c)| TimedText {
            id: i as u64 + 1,
            start: c.start,
            end: c.end,
            text: c.text,
        })
        .collect()
}

/// Text as a subtitle block can hold it, however it was produced: one blank line would end the cue in a
/// player, and `-->` could be read as the start of another one.
fn cue_text(text: &str) -> String {
    let cleaned = engine::lang::clean_separators(text);
    let lines: Vec<&str> = cleaned
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    lines.join("\n").replace("-->", "->")
}

/// One line for a `.txt` file: `[HH:MM:SS] text`.
fn txt_line(start: f64, text: &str) -> String {
    let flat = text.split('\n').collect::<Vec<_>>().join(" ");
    format!("[{}] {flat}\n", engine::transcript::clock(start))
}

/// A translation's cues as saved files show them. `[untranslated]` is added here and never stored. A
/// cue with no text is left out, as a blank subtitle block would break the file for some players.
fn translated_cues(file: &TranslationFile) -> Vec<Cue> {
    let opts = SubtitleOptions::default();
    file.cues
        .iter()
        .filter(|c| !cue_text(&c.text).is_empty())
        .map(|c| {
            let flat = cue_text(&c.text).replace('\n', " ");
            let text = match c.status {
                CueStatus::Untranslated => format!("[untranslated] {flat}"),
                CueStatus::Translated | CueStatus::Edited | CueStatus::Unknown => flat,
            };
            Cue {
                start: c.start,
                end: c.end,
                text: wrap(&text, opts.max_chars_per_line, opts.max_lines),
                speaker: 0,
                utterance: 0,
            }
        })
        .collect()
}

/// Writes `srt`, `vtt` and `txt` to `path_for(ext)`, with U+200F per line when `lang` is an RTL language.
fn write_subtitle_files(
    cues: &[Cue],
    lang: &str,
    path_for: impl Fn(&str) -> PathBuf,
) -> Result<Vec<PathBuf>, String> {
    let shown = mark_rtl(cues, lang);
    let txt: String = cues.iter().map(|c| txt_line(c.start, &c.text)).collect();
    let txt = mark_rtl_text(&txt, lang);
    let mut out = Vec::new();
    for (ext, body) in [
        ("srt", to_srt(&shown)),
        ("vtt", to_vtt(&shown)),
        ("txt", txt),
    ] {
        let path = path_for(ext);
        fsio::write_atomic(&path, body.as_bytes())
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        out.push(path);
    }
    Ok(out)
}

/// Writes `<stem>.<target>.{srt,vtt,txt}` from a translation. Also for use after a cue was edited by hand.
pub fn write_translated(
    file: &TranslationFile,
    paths: &ProjectPaths,
) -> Result<Vec<PathBuf>, String> {
    write_subtitle_files(&translated_cues(file), &file.target, |ext| {
        paths.translated(&file.target, ext)
    })
}

/// A file deleted when this goes out of scope.
struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use engine::project::TranslationCue;

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

    fn term(source: &str, target: Option<&str>) -> GlossaryTerm {
        GlossaryTerm {
            source: source.into(),
            target: target.map(str::to_string),
        }
    }

    /// A fresh folder under the system temp dir, removed when dropped.
    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> TestDir {
            let dir = std::env::temp_dir().join(format!(
                "vipr-pipeline-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            TestDir(dir)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn merged_glossary_joins_both_lists_once() {
        let base = "\u{feff}# names\nCloudflare, WhatsApp\nwebhook # jargon\n";
        let terms = [
            term("cloudflare", None),
            term("Kubernetes", Some("کوبرنتیز")),
        ];
        assert_eq!(
            merged_glossary(Some(base), &terms),
            "Cloudflare\nWhatsApp\nwebhook\nKubernetes\n"
        );
        assert_eq!(merged_glossary(None, &[]), "");
        assert_eq!(merged_glossary(Some("# only a comment\n"), &[]), "");
    }

    #[test]
    fn a_profile_term_stays_one_term() {
        let terms = [
            term("Acme, Inc", None),
            term("C# code", None),
            term("two\nlines\u{2029}here", None),
            term("a=b", None),
            term(" ,#\n", None),
        ];
        assert_eq!(
            merged_glossary(None, &terms),
            "Acme Inc\nC code\ntwo lines here\na b\n"
        );
    }

    #[test]
    fn profile_prompt_becomes_background() {
        assert_eq!(
            with_background("A call.", " Use formal names. "),
            "A call. Use formal names."
        );
        assert_eq!(
            with_background("", "Use formal names."),
            "Use formal names."
        );
        assert_eq!(with_background(" A call. ", "  "), "A call.");
    }

    fn job_input(path: &Path, out_dir: &Path, glossary: Option<PathBuf>) -> JobInput {
        JobInput {
            path: path.to_path_buf(),
            out_dir: out_dir.to_path_buf(),
            about: About::default(),
            num_speakers: 0,
            asr_model: PathBuf::new(),
            vad_model: PathBuf::new(),
            diar_models: None,
            llm_model: None,
            glossary,
        }
    }

    fn profile(prompt: &str, glossary: Vec<GlossaryTerm>) -> Profile {
        Profile {
            id: "p".into(),
            name: "P".into(),
            prompt: prompt.into(),
            glossary,
        }
    }

    #[test]
    fn apply_profile_writes_the_glossary_under_the_real_stem() {
        let dir = TestDir::new("apply");
        // meeting.mp4 already has results, so meeting.mp3 is "meeting (mp3)"
        let earlier = ProjectPaths::new(dir.0.join("meeting"), "meeting");
        let t = Transcript::build("meeting.mp4", &empty_asr(), None, &[]);
        t.save(&earlier.transcript()).unwrap();
        let base = dir.0.join("glossary.txt");
        std::fs::write(&base, "Cloudflare\n").unwrap();

        let mut input = job_input(Path::new("C:/m/meeting.mp3"), &dir.0, Some(base));
        let p = profile("Formal.", vec![term("Acme", None)]);
        apply_profile(&mut input, &p).unwrap();

        let want = ProjectPaths::new(dir.0.join("meeting (mp3)"), "meeting (mp3)").glossary();
        assert_eq!(input.glossary.as_deref(), Some(want.as_path()));
        assert_eq!(
            std::fs::read_to_string(&want).unwrap(),
            "Cloudflare\nAcme\n"
        );
        assert_eq!(input.about.description, "Formal.");
    }

    #[test]
    fn a_profile_without_terms_leaves_the_glossary_alone() {
        let dir = TestDir::new("noterms");
        let mut input = job_input(Path::new("C:/m/a.mp3"), &dir.0, None);
        apply_profile(&mut input, &profile("Formal.", vec![])).unwrap();
        assert_eq!(input.glossary, None);
        assert_eq!(input.about.description, "Formal.");
        assert!(!dir.0.join("a").exists());
    }

    fn cue(id: u64, text: &str, status: CueStatus) -> TranslationCue {
        TranslationCue {
            id,
            start: id as f64,
            end: id as f64 + 0.5,
            source: "source".into(),
            text: text.into(),
            status,
            flags: vec![],
        }
    }

    fn translation(target: &str, cues: Vec<TranslationCue>) -> TranslationFile {
        TranslationFile {
            version: 1,
            target: target.into(),
            source_hash: String::new(),
            model: String::new(),
            profile: None,
            cues,
        }
    }

    #[test]
    fn untranslated_lines_are_marked_only_in_saved_files() {
        let dir = TestDir::new("translated");
        let paths = ProjectPaths::new(&dir.0, "a (mp3)");
        let file = translation(
            "de",
            vec![
                cue(1, "hallo", CueStatus::Translated),
                cue(2, "source", CueStatus::Untranslated),
                cue(3, "von Hand", CueStatus::Edited),
            ],
        );
        let written = write_translated(&file, &paths).unwrap();
        assert_eq!(
            written,
            ["srt", "vtt", "txt"].map(|e| paths.translated("de", e))
        );
        let srt = std::fs::read_to_string(paths.translated("de", "srt")).unwrap();
        assert!(
            srt.contains("1\n00:00:01,000 --> 00:00:01,500\nhallo\n\n"),
            "{srt}"
        );
        assert!(srt.contains("[untranslated] source\n"), "{srt}");
        assert!(srt.contains("von Hand\n"), "{srt}");
        assert!(!srt.contains('\u{200f}'), "no marks on an LTR target");
        let vtt = std::fs::read_to_string(paths.translated("de", "vtt")).unwrap();
        assert!(vtt.starts_with("WEBVTT\n") && vtt.contains("[untranslated] source\n"));
        let txt = std::fs::read_to_string(paths.translated("de", "txt")).unwrap();
        assert_eq!(
            txt,
            "[00:00:01] hallo\n[00:00:02] [untranslated] source\n[00:00:03] von Hand\n"
        );
        assert_eq!(file.cues[1].text, "source", "the marker is never stored");
    }

    #[test]
    fn a_cue_with_no_text_never_reaches_the_saved_files() {
        let dir = TestDir::new("empty-cue");
        let paths = ProjectPaths::new(&dir.0, "a");
        let file = translation(
            "fa",
            vec![
                cue(1, "", CueStatus::Translated),
                cue(2, "جالبه", CueStatus::Translated),
                cue(3, " \n ", CueStatus::Edited),
                cue(4, "بعدی", CueStatus::Translated),
            ],
        );
        write_translated(&file, &paths).unwrap();
        let srt = std::fs::read_to_string(paths.translated("fa", "srt")).unwrap();
        assert_eq!(
            srt,
            "1\n00:00:02,000 --> 00:00:02,500\n\u{200f}جالبه\n\n\
             2\n00:00:04,000 --> 00:00:04,500\n\u{200f}بعدی\n\n"
        );
        let parsed = engine::subparse::parse(&srt);
        assert_eq!(parsed.cues.len(), 2, "{:?}", parsed.warnings);
        assert!(parsed.warnings.is_empty(), "{:?}", parsed.warnings);
        let vtt = std::fs::read_to_string(paths.translated("fa", "vtt")).unwrap();
        assert_eq!(vtt.matches("-->").count(), 2);
        assert!(!vtt.contains("\n\n\n"), "{vtt:?}");
        let txt = std::fs::read_to_string(paths.translated("fa", "txt")).unwrap();
        assert_eq!(txt.lines().count(), 2, "{txt}");
    }

    #[test]
    fn rtl_targets_get_a_mark_per_line_in_every_file() {
        let dir = TestDir::new("rtl");
        let paths = ProjectPaths::new(&dir.0, "a");
        // long enough to wrap
        let long = "این یک جملهٔ بلند است که باید در دو خط نوشته شود تا روی صفحه خوانا بماند";
        let file = translation("fa", vec![cue(1, long, CueStatus::Translated)]);
        write_translated(&file, &paths).unwrap();
        let srt = std::fs::read_to_string(paths.translated("fa", "srt")).unwrap();
        let text: Vec<&str> = srt.lines().skip(2).take_while(|l| !l.is_empty()).collect();
        assert_eq!(text.len(), 2, "{srt}");
        assert!(text.iter().all(|l| l.starts_with('\u{200f}')), "{srt}");
        let txt = std::fs::read_to_string(paths.translated("fa", "txt")).unwrap();
        assert!(txt.starts_with("\u{200f}[00:00:01] "), "{txt}");
        assert_eq!(file.cues[0].text, long, "the marks are never stored");
    }

    #[test]
    fn saved_text_cannot_break_out_of_its_cue() {
        let file = translation(
            "de",
            vec![cue(
                1,
                "eins\n\n2\n00:00:09,000 --> 00:00:10,000\nforged",
                CueStatus::Translated,
            )],
        );
        let cues = translated_cues(&file);
        assert_eq!(cues.len(), 1);
        assert!(!cues[0].text.contains("\n\n") && !cues[0].text.contains("-->"));
    }

    #[test]
    fn own_cues_are_numbered_from_one() {
        let asr = AsrResult {
            model: None,
            language: "en".into(),
            audio_seconds: 9.0,
            elapsed_seconds: 0.0,
            segments: vec![engine::transcript::AsrSegment {
                start: 0.0,
                end: 8.0,
                text: "one. two.".into(),
                no_speech_prob: 0.0,
                words: vec![
                    engine::transcript::AsrWord {
                        text: "one.".into(),
                        start: 0.0,
                        end: 1.0,
                        prob: 1.0,
                    },
                    engine::transcript::AsrWord {
                        text: "two.".into(),
                        start: 6.0,
                        end: 7.0,
                        prob: 1.0,
                    },
                ],
            }],
        };
        let t = Transcript::build("a.mp3", &asr, None, &[]);
        let cues = transcript_cues(&t);
        assert_eq!(cues.iter().map(|c| c.id).collect::<Vec<_>>(), [1, 2]);
        assert_eq!(cues[1].text, "two.");
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
