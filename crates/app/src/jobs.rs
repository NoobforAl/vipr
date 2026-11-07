//! Background work on finished results: the queue of translate and compare runs, and Open result.
//!
//! Only one worker runs at a time (`pipeline::RUNNING_WORKER`), so `main.rs` shares the slot:
//! `App::start_next` runs waiting transcriptions first, then asks this queue, and starts nothing
//! while [`busy`] is true. Runs use `pipeline`'s worker runner, so Cancel and closing the window stop
//! them. A translation stopped this way resumes from its checkpoint when it is started again.
//!
//! Everything a worker returns is checked here before it is saved, and all subtitle text is untrusted.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use engine::chapters::transcript_lines;
use engine::compare::{MAX_REASON_CHARS, check_merged, needs_llm};
use engine::fsio;
use engine::lang::clean_separators;
use engine::profiles::Profile;
use engine::project::{
    AiChoice, AiPick, AiStatus, ChaptersFile, ChaptersInput, ChaptersOutput, CompareFile,
    CompareInRow, CompareInput, CompareOutRow, CompareOutput, CompareRow, CompareSource, CueStatus,
    ImportFile, ProjectMeta, ProjectPaths, TimedText, TranslateInput, TranslateOutput,
    TranslationCue, TranslationFile, VERSION, source_hash,
};
use engine::registry::{Registry, Role};
use engine::subalign::{align, best_offset};
use engine::transcript::Transcript;
use iced::futures::channel::mpsc;
use iced::futures::{SinkExt, Stream, StreamExt};
use iced::widget::{button, column, container, row, text};
use iced::{Alignment, Element, Length, Task};
use serde_json::Value;

use crate::icons;
use crate::out::{JobKind, JobRef, Out, Page, TranslateSource};
use crate::pipeline;
use crate::progress;
use crate::theme::{self, Tokens};

/// Text of the error a run ends with when it was cancelled (`pipeline::run_worker`).
const CANCELLED: &str = "Cancelled";

#[derive(Debug, Default)]
pub struct State {
    queue: VecDeque<JobKind>,
    running: Option<Running>,
}

#[derive(Debug)]
struct Running {
    kind: JobKind,
    /// The recording's name.
    stem: String,
    detail: String,
    progress: f32,
    /// `progress` as drawn: it glides there.
    shown: f32,
    cancelling: bool,
}

#[derive(Debug, Clone)]
pub enum Msg {
    Event(RunEvent),
    /// A run could not start. Only here so `main.rs` looks at the queue again.
    Skipped,
    /// Something to tell the user after a run opened a page.
    Tell(String),
    Cancel,
    /// Takes a waiting run out of the queue, by its place in it.
    Remove(usize),
}

/// What a running translate or compare reports.
#[derive(Debug, Clone)]
pub enum RunEvent {
    Detail(String),
    Progress(f32),
    Finished(Finished),
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finished {
    Translated {
        job: u64,
        target: String,
        translated: usize,
        untranslated: usize,
        flagged: usize,
    },
    Compared {
        job: u64,
        /// Rows that needed the AI and did not get a pick, because no model is available.
        not_judged: usize,
    },
    Exported {
        path: PathBuf,
    },
    Chapters {
        count: usize,
        /// Parts of the recording the AI gave nothing usable for.
        failed_windows: usize,
    },
}

/// What `main.rs` knows and a queued run needs.
#[derive(Debug, Clone)]
pub struct JobContext {
    /// `None` when the job was removed while this run was waiting.
    pub job: Option<JobRef>,
    /// The chosen LLM, when it is downloaded.
    pub model: Option<PathBuf>,
    pub profile: Option<Profile>,
}

/// A finished job rebuilt from disk by Open result. Jobs otherwise exist only in memory.
#[derive(Debug, Clone)]
pub struct LoadedResult {
    /// `None` when the media has moved and the user must locate it.
    pub media: Option<PathBuf>,
    pub paths: ProjectPaths,
    pub transcript: Transcript,
}

/// Queues a run, unless the same run is already waiting or running.
pub fn enqueue(state: &mut State, kind: JobKind) {
    let running = state.running.as_ref().is_some_and(|r| r.kind == kind);
    if !running && !state.queue.contains(&kind) {
        state.queue.push_back(kind);
    }
}

/// How far along this run is, 0..1, when it is waiting or running. `None` otherwise.
pub fn pending(state: &State, kind: &JobKind) -> Option<f32> {
    match &state.running {
        Some(r) if r.kind == *kind => Some(r.shown),
        _ => state.queue.contains(kind).then_some(0.0),
    }
}

/// One step of the animation, while [`busy`].
pub fn tick(state: &mut State) {
    if let Some(r) = &mut state.running {
        r.shown = progress::approach(r.shown, r.progress);
    }
}

/// Takes the next queued run.
pub fn next(state: &mut State) -> Option<JobKind> {
    state.queue.pop_front()
}

/// True while a run started by [`start`] has not finished.
pub fn busy(state: &State) -> bool {
    state.running.is_some()
}

/// Starts a run taken from [`next`]. `main.rs` has checked that the worker slot is free.
pub fn start(state: &mut State, kind: JobKind, context: JobContext) -> (Task<Msg>, Out) {
    let Some(job) = context.job else {
        let notice = format!(
            "{} was skipped: its recording is no longer in the list.",
            label(&kind)
        );
        return (Task::done(Msg::Skipped), Out::Notice(notice));
    };
    state.running = Some(Running {
        kind: kind.clone(),
        stem: job.paths.stem.clone(),
        detail: "Starting".into(),
        progress: 0.0,
        shown: 0.0,
        cancelling: false,
    });
    pipeline::reset_cancel();
    let run = Run {
        kind,
        job,
        model: context.model,
        profile: context.profile,
    };
    (Task::run(run_stream(run), Msg::Event), Out::None)
}

/// `main.rs` calls `App::start_next` after this whenever [`busy`] is false.
pub fn update(state: &mut State, msg: Msg) -> (Task<Msg>, Out) {
    let out = match msg {
        Msg::Event(RunEvent::Detail(detail)) => {
            if let Some(r) = &mut state.running {
                r.detail = detail;
            }
            Out::None
        }
        Msg::Event(RunEvent::Progress(progress)) => {
            if let Some(r) = &mut state.running {
                r.progress = progress;
            }
            Out::None
        }
        Msg::Event(RunEvent::Finished(finished)) => {
            state.running = None;
            return finished_out(finished);
        }
        Msg::Event(RunEvent::Failed(error)) => {
            let what = state.running.take().map(|r| label(&r.kind));
            let what = what.as_deref().unwrap_or("The run");
            if error == CANCELLED {
                Out::Notice(format!("{what} was cancelled."))
            } else {
                Out::Alert(format!("{what} failed: {error}"))
            }
        }
        Msg::Skipped => Out::None,
        Msg::Tell(notice) => Out::Notice(notice),
        Msg::Cancel => {
            if let Some(r) = &mut state.running {
                r.cancelling = true;
                r.detail = "Stopping".into();
                pipeline::request_cancel();
            }
            Out::None
        }
        Msg::Remove(index) => {
            state.queue.remove(index);
            Out::None
        }
    };
    (Task::none(), out)
}

fn finished_out(finished: Finished) -> (Task<Msg>, Out) {
    match finished {
        Finished::Translated {
            target,
            translated,
            untranslated,
            flagged,
            ..
        } => {
            let mut notice = format!(
                "Translated {translated} lines to {}.",
                language_name(&target)
            );
            if untranslated > 0 {
                notice.push_str(&format!(
                    " {untranslated} could not be translated and stay in the original language."
                ));
            }
            if flagged > 0 {
                notice.push_str(&format!(" {flagged} are worth checking."));
            }
            (Task::none(), Out::Notice(notice))
        }
        Finished::Compared { job, not_judged } => {
            let open = Out::Open(Page::Review(job));
            if not_judged == 0 {
                return (Task::none(), open);
            }
            let notice = format!(
                "No AI model is available, so {not_judged} rows have no AI pick. Choose them yourself."
            );
            (Task::done(Msg::Tell(notice)), open)
        }
        Finished::Exported { path } => {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            (
                Task::none(),
                Out::Notice(format!("Saved the video with subtitles: {name}")),
            )
        }
        Finished::Chapters {
            count,
            failed_windows,
        } => (
            Task::none(),
            Out::Notice(chapters_notice(count, failed_windows)),
        ),
    }
}

fn chapters_notice(count: usize, failed_windows: usize) -> String {
    let mut notice = if count == 1 {
        "Found one topic in this recording.".to_string()
    } else {
        format!("Found {count} chapters.")
    };
    if failed_windows > 0 {
        notice.push_str(" Part of the recording could not be read, so a topic may be missing.");
    }
    notice
}

/// `43%`. Empty until there is a first figure, since "0%" reads as stuck while a model loads.
pub fn percent_text(fraction: f32) -> String {
    let percent = (fraction * 100.0).round();
    if percent < 1.0 {
        String::new()
    } else {
        format!("{percent:.0}%")
    }
}

/// The running and waiting runs. Empty when there are none.
pub fn view(state: &State, t: Tokens, clock: f32) -> Element<'_, Msg> {
    let mut list = column![].spacing(8);
    if let Some(r) = &state.running {
        let mut cancel = button(text("Cancel").size(12)).style(theme::outline_btn(t));
        if !r.cancelling {
            cancel = cancel.on_press(Msg::Cancel);
        }
        let title = format!("{}: {}", label(&r.kind), clean_separators(&r.stem));
        list = list
            .push(
                row![
                    icons::spinner(icons::spin_angle(clock), 18.0, t.accent),
                    column![
                        text(title).size(13).color(t.text),
                        text(r.detail.as_str()).size(12).color(t.muted),
                    ]
                    .spacing(2)
                    .width(Length::Fill),
                    text(percent_text(r.shown)).size(14).color(t.accent),
                    cancel,
                ]
                .spacing(12)
                .align_y(Alignment::Center),
            )
            .push(progress::bar(r.shown, clock, 8.0, t));
    }
    for (index, kind) in state.queue.iter().enumerate() {
        list = list.push(
            row![
                text(clean_separators(&format!("Waiting: {}", label(kind))))
                    .size(13)
                    .color(t.muted)
                    .width(Length::Fill),
                button(text("Remove").size(12))
                    .style(theme::ghost(t))
                    .on_press(Msg::Remove(index)),
            ]
            .spacing(12)
            .align_y(Alignment::Center),
        );
    }
    if state.running.is_none() && state.queue.is_empty() {
        return list.into();
    }
    container(list)
        .padding(14)
        .width(Length::Fill)
        .style(theme::card(t))
        .into()
}

pub fn language_name(code: &str) -> String {
    engine::lang::language(code).map_or_else(|| code.to_string(), |l| l.name.to_string())
}

fn label(kind: &JobKind) -> String {
    match kind {
        JobKind::Translate { target, .. } => format!("Translation to {}", language_name(target)),
        JobKind::Compare { .. } => "Subtitle compare".to_string(),
        JobKind::Chapters { .. } => "Finding chapters".to_string(),
        JobKind::Export { .. } => "Exporting video".to_string(),
    }
}

// ------------------------------------------------------------------------- running

/// Everything one run needs, moved to its thread.
struct Run {
    kind: JobKind,
    job: JobRef,
    model: Option<PathBuf>,
    profile: Option<Profile>,
}

/// Runs one translate or compare on its own thread and streams its events to the UI. Always ends with
/// `Finished` or `Failed`.
fn run_stream(run: Run) -> impl Stream<Item = RunEvent> {
    iced::stream::channel(64, async move |mut out: mpsc::Sender<RunEvent>| {
        let (tx, mut rx) = mpsc::unbounded::<RunEvent>();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut emit = |e: RunEvent| {
                    let _ = tx.unbounded_send(e);
                };
                execute(&run, &mut emit)
            }));
            let last = match result {
                Ok(Ok(finished)) => RunEvent::Finished(finished),
                Ok(Err(error)) => RunEvent::Failed(error),
                Err(_) => RunEvent::Failed("an internal error stopped it.".into()),
            };
            let _ = tx.unbounded_send(last);
        });
        while let Some(event) = rx.next().await {
            let _ = out.send(event).await;
        }
    })
}

fn execute(run: &Run, emit: &mut dyn FnMut(RunEvent)) -> Result<Finished, String> {
    match &run.kind {
        JobKind::Translate { target, source, .. } => translate(run, target, source, emit),
        JobKind::Compare { a, b, slug, .. } => compare(run, a, b, slug, emit),
        JobKind::Chapters { .. } => chapters(run, emit),
        JobKind::Export { lang, out, .. } => export(run, lang.as_deref(), out, emit),
    }
}

/// A language code or slug ends up in a file name, so it must not be able to leave the result folder.
fn check_name(name: &str) -> Result<(), String> {
    let usable = !name.is_empty()
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if usable {
        Ok(())
    } else {
        Err(format!("\"{name}\" cannot be used in a file name."))
    }
}

/// Models made only for translation each need a prompt format of their own, which isn't wired up.
const TRANSLATE_ONLY_FILES: [&str; 3] = ["translategemma", "hunyuan-mt", "seed-x"];

/// The registry id of a model, or the reason it can't be used for this work.
fn model_id(model: &Path, registry: Option<&Registry>) -> Result<String, String> {
    let file = model
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let entry = registry.and_then(|r| r.models.iter().find(|m| m.file_name() == file));
    let lower = file.to_lowercase();
    if entry.is_some_and(|m| !m.has_role(Role::Llm))
        || TRANSLATE_ONLY_FILES.iter().any(|n| lower.contains(n))
    {
        return Err(format!(
            "\"{file}\" is made only for translation and needs its own prompt format, which ViPr does \
             not have yet. Choose a general AI model in Settings."
        ));
    }
    Ok(entry.map_or(file, |m| m.id.clone()))
}

/// Starts `llm-worker --mode <mode>` and turns its progress into events.
fn call_worker(
    mode: &str,
    exe: &Path,
    model: &Path,
    files: (&Path, &Path),
    extra: Vec<String>,
    log: &Path,
    emit: &mut dyn FnMut(RunEvent),
) -> Result<(), String> {
    let (input, out) = files;
    let mut args = vec![
        "--mode".to_string(),
        mode.to_string(),
        "--model".to_string(),
        pipeline::s(model),
        "--input".to_string(),
        pipeline::s(input),
        "--out".to_string(),
        pipeline::s(out),
    ];
    args.extend(extra);
    emit(RunEvent::Detail("Loading the AI model".into()));
    pipeline::run_worker(exe, &args, log, &mut |v: &Value| match v
        .get("event")
        .and_then(Value::as_str)
    {
        Some("model_loaded") => emit(RunEvent::Detail("Working through the lines".into())),
        Some("progress") => emit(RunEvent::Progress(pipeline::pct_of(v))),
        _ => {}
    })
    .map(|_| ())
}

fn glossary_args(glossary: Option<&Path>) -> Vec<String> {
    glossary
        .map(|g| vec!["--glossary".to_string(), pipeline::s(g)])
        .unwrap_or_default()
}

fn load_error(what: &str, e: engine::Error) -> String {
    format!("could not read {what} ({e}).")
}

/// `cues` as they fall on the media's timeline. Import files keep their own times.
fn shifted(cues: Vec<TimedText>, offset: f64) -> Vec<TimedText> {
    cues.into_iter()
        .map(|c| TimedText {
            start: (c.start + offset).max(0.0),
            end: (c.end + offset).max(0.0),
            ..c
        })
        .collect()
}

fn own_cues(paths: &ProjectPaths) -> Result<(Vec<TimedText>, String), String> {
    let t = Transcript::load(&paths.transcript()).map_err(|e| load_error("the transcript", e))?;
    Ok((pipeline::transcript_cues(&t), t.language))
}

fn import_file(paths: &ProjectPaths, slug: &str) -> Result<ImportFile, String> {
    check_name(slug)?;
    fsio::load_json(&paths.import(slug)).map_err(|e| load_error("the imported subtitles", e))
}

// ------------------------------------------------------------------------- export

fn export(
    run: &Run,
    lang: Option<&str>,
    out: &Path,
    emit: &mut dyn FnMut(RunEvent),
) -> Result<Finished, String> {
    let ffmpeg = engine::audio::find_ffmpeg()
        .ok_or("putting subtitles on a video needs ffmpeg. Get it in Settings first.")?;
    if !run.job.media.is_file() {
        return Err("the video is not where it was. Open the player once to find it again.".into());
    }
    let paths = &run.job.paths;
    let t = Transcript::load(&paths.transcript()).map_err(|e| load_error("the transcript", e))?;
    let srt = match lang {
        None => pipeline::transcript_srt(&t),
        Some(code) => {
            check_name(code)?;
            let file: TranslationFile = fsio::load_json(&paths.translation(code))
                .map_err(|e| load_error("the translation", e))?;
            pipeline::translation_srt(&file)
        }
    };
    emit(RunEvent::Detail(
        "Putting the subtitles on the video".into(),
    ));
    pipeline::burn_subtitles(
        &ffmpeg,
        &run.job.media,
        &srt,
        (&paths.dir, out),
        t.audio_seconds,
        &paths.dir.join("export.log"),
        &mut |done| emit(RunEvent::Progress(done)),
    )?;
    Ok(Finished::Exported {
        path: out.to_path_buf(),
    })
}

// ------------------------------------------------------------------------- chapters

fn chapters(run: &Run, emit: &mut dyn FnMut(RunEvent)) -> Result<Finished, String> {
    let model = run
        .model
        .as_deref()
        .ok_or("no AI model is downloaded. Get one in Settings first.")?;
    let exe = pipeline::find_worker("llm-worker")
        .ok_or("the AI engine (llm-worker) was not found next to the app.")?;
    let model_name = model_id(model, Registry::load_default(None).ok().as_ref())?;

    let paths = &run.job.paths;
    let t = Transcript::load(&paths.transcript()).map_err(|e| load_error("the transcript", e))?;
    let lines = transcript_lines(&t);
    if lines.is_empty() {
        return Err("there are no lines to work from.".into());
    }
    let input = ChaptersInput {
        version: VERSION,
        duration: t.audio_seconds,
        lines,
    };
    let input_path = paths.dir.join(format!("{}.chapters-in.json", paths.stem));
    let out_path = paths.dir.join(format!("{}.chapters-out.json", paths.stem));
    fsio::save_json(&input_path, &input).map_err(|e| e.to_string())?;
    let finished = call_worker(
        "chapters",
        &exe,
        model,
        (&input_path, &out_path),
        Vec::new(),
        &paths.dir.join("llm-chapters.log"),
        emit,
    );
    // also after a failure or Cancel, since nothing can continue from these
    let _ = std::fs::remove_file(&input_path);
    finished?;
    let output: ChaptersOutput =
        fsio::load_json(&out_path).map_err(|e| load_error("the chapters", e))?;
    let _ = std::fs::remove_file(&out_path);
    let file = ChaptersFile {
        version: VERSION,
        source_hash: source_hash(&input.lines),
        model: model_name,
        chapters: output.chapters,
    };
    fsio::save_json(&paths.chapters(), &file).map_err(|e| e.to_string())?;
    Ok(Finished::Chapters {
        count: file.chapters.len(),
        failed_windows: output.stats.failed_windows,
    })
}

// ------------------------------------------------------------------------- translate

fn translate(
    run: &Run,
    target: &str,
    source: &TranslateSource,
    emit: &mut dyn FnMut(RunEvent),
) -> Result<Finished, String> {
    check_name(target)?;
    let model = run
        .model
        .as_deref()
        .ok_or("no AI model is downloaded. Get one in Settings first.")?;
    let exe = pipeline::find_worker("llm-worker")
        .ok_or("the AI engine (llm-worker) was not found next to the app.")?;
    let model_name = model_id(model, Registry::load_default(None).ok().as_ref())?;

    let paths = &run.job.paths;
    let (cues, source_lang) = match source {
        TranslateSource::Transcript => own_cues(paths)?,
        TranslateSource::Import { slug } => {
            let import = import_file(paths, slug)?;
            (shifted(import.cues, import.offset), String::new())
        }
    };
    if cues.is_empty() {
        return Err("there are no lines to translate.".into());
    }

    let input_path = paths
        .dir
        .join(format!("{}.{target}.translate-in.json", paths.stem));
    let out_path = paths
        .dir
        .join(format!("{}.{target}.translate-out.json", paths.stem));
    let profile = run.profile.as_ref();
    let glossary =
        pipeline::write_merged_glossary(paths, pipeline::find_glossary().as_deref(), profile)?;
    let input = TranslateInput {
        version: VERSION,
        source_lang,
        target: target.to_string(),
        cues: cues.clone(),
        profile_prompt: profile_prompt(profile),
        glossary: profile.map(|p| p.glossary.clone()).unwrap_or_default(),
        hints: Vec::new(),
    };
    fsio::save_json(&input_path, &input).map_err(|e| e.to_string())?;
    let mut extra = glossary_args(glossary.as_deref());
    extra.extend([
        "--checkpoint".to_string(),
        pipeline::s(&paths.checkpoint(target)),
    ]);
    call_worker(
        "translate",
        &exe,
        model,
        (&input_path, &out_path),
        extra,
        &paths.dir.join("llm-translate.log"),
        emit,
    )?;
    let output: TranslateOutput =
        fsio::load_json(&out_path).map_err(|e| load_error("the translation", e))?;

    emit(RunEvent::Detail("Writing the translated files".into()));
    let file_path = paths.translation(target);
    let previous: Option<TranslationFile> = fsio::load_json(&file_path).ok();
    let file = translation_file(
        target,
        &cues,
        output,
        model_name,
        run.profile.clone(),
        previous.as_ref(),
    );
    fsio::save_json(&file_path, &file).map_err(|e| e.to_string())?;
    pipeline::write_translated(&file, paths)?;
    let _ = std::fs::remove_file(&input_path);
    let _ = std::fs::remove_file(&out_path);

    let count = |status: CueStatus| file.cues.iter().filter(|c| c.status == status).count();
    Ok(Finished::Translated {
        job: run.job.id,
        target: target.to_string(),
        translated: count(CueStatus::Translated) + count(CueStatus::Edited),
        untranslated: count(CueStatus::Untranslated),
        flagged: file.cues.iter().filter(|c| !c.flags.is_empty()).count(),
    })
}

fn profile_prompt(profile: Option<&Profile>) -> Option<String> {
    profile
        .map(|p| p.prompt.trim().to_string())
        .filter(|p| !p.is_empty())
}

/// Wraps the worker's output into the file saved on disk. Ids, times and source always come from `cues`.
/// A cue with no usable translation holds its source and is `Untranslated`. Exactly empty is usable: the
/// worker leaves a cue that way when its sentence had too few words to reach it. A cue the user edited
/// in `previous` is kept when it was made from the same source text, while the rest of its sentence
/// takes the new translation.
fn translation_file(
    target: &str,
    cues: &[TimedText],
    output: TranslateOutput,
    model: String,
    profile: Option<Profile>,
    previous: Option<&TranslationFile>,
) -> TranslationFile {
    let done: Vec<&TranslationCue> = output.cues.iter().collect();
    let cues_out = cues
        .iter()
        .map(|c| {
            let edited = previous
                .and_then(|p| p.cues.iter().find(|o| o.id == c.id && o.source == c.text))
                .filter(|o| o.status == CueStatus::Edited);
            if let Some(o) = edited {
                return TranslationCue {
                    start: c.start,
                    end: c.end,
                    ..o.clone()
                };
            }
            let usable = done.iter().find(|o| o.id == c.id).filter(|o| {
                o.status == CueStatus::Translated
                    && (o.text.is_empty() || !o.text.trim().is_empty())
            });
            TranslationCue {
                id: c.id,
                start: c.start,
                end: c.end,
                source: c.text.clone(),
                text: usable.map_or_else(|| c.text.clone(), |o| o.text.clone()),
                status: usable.map_or(CueStatus::Untranslated, |_| CueStatus::Translated),
                flags: usable.map_or_else(Vec::new, |o| o.flags.clone()),
            }
        })
        .collect();
    TranslationFile {
        version: VERSION,
        target: target.to_string(),
        source_hash: source_hash(cues),
        model,
        profile,
        cues: cues_out,
    }
}

// ------------------------------------------------------------------------- compare

/// The cues of one compare column on the media's timeline, and the import offset already in them.
fn side_cues(paths: &ProjectPaths, side: &CompareSource) -> Result<(Vec<TimedText>, f64), String> {
    match side {
        CompareSource::Transcript => own_cues(paths).map(|(cues, _)| (cues, 0.0)),
        CompareSource::Translation { lang } => {
            check_name(lang)?;
            let file: TranslationFile = fsio::load_json(&paths.translation(lang))
                .map_err(|e| load_error("the translation", e))?;
            // an untranslated cue holds the source language, so this side has no text for it, and an
            // empty one has its text on a neighbour
            let cues = file
                .cues
                .into_iter()
                .filter(|c| c.status != CueStatus::Untranslated && !c.text.trim().is_empty())
                .map(|c| TimedText {
                    id: c.id,
                    start: c.start,
                    end: c.end,
                    text: c.text,
                })
                .collect();
            Ok((cues, 0.0))
        }
        CompareSource::Import { slug } => {
            let import = import_file(paths, slug)?;
            Ok((shifted(import.cues, import.offset), import.offset))
        }
    }
}

fn compare(
    run: &Run,
    a: &CompareSource,
    b: &CompareSource,
    slug: &str,
    emit: &mut dyn FnMut(RunEvent),
) -> Result<Finished, String> {
    check_name(slug)?;
    let paths = &run.job.paths;
    let (a_cues, _) = side_cues(paths, a)?;
    let (b_cues, b_offset) = side_cues(paths, b)?;
    if a_cues.is_empty() && b_cues.is_empty() {
        return Err("both sides are empty.".into());
    }
    emit(RunEvent::Detail("Lining up the subtitles".into()));
    let shift = best_offset(&a_cues, &b_cues);
    let mut rows = align(&a_cues, &b_cues, shift);

    let asks = rows_to_judge(&rows);
    let mut model_name = String::new();
    let mut not_judged = 0;
    if !asks.is_empty() {
        let llm = run
            .model
            .as_deref()
            .zip(pipeline::find_worker("llm-worker"));
        match llm {
            Some((model, exe)) => {
                model_name = model_id(model, Registry::load_default(None).ok().as_ref())?;
                let output = judge(run, slug, asks, model, &exe, emit)?;
                apply_picks(&mut rows, &output.rows);
            }
            None => not_judged = asks.len(),
        }
    }

    let file_path = paths.compare(slug);
    if let Ok(previous) = fsio::load_json::<CompareFile>(&file_path) {
        keep_user_picks(&mut rows, &previous.rows);
    }
    let file = CompareFile {
        version: VERSION,
        slug: slug.to_string(),
        a: a.clone(),
        b: b.clone(),
        model: model_name,
        offset: b_offset + shift,
        rows,
    };
    fsio::save_json(&file_path, &file).map_err(|e| e.to_string())?;
    Ok(Finished::Compared {
        job: run.job.id,
        not_judged,
    })
}

/// Identical and one-sided rows never reach the LLM.
fn rows_to_judge(rows: &[CompareRow]) -> Vec<CompareInRow> {
    rows.iter()
        .filter(|r| needs_llm(r))
        .filter_map(|r| {
            Some(CompareInRow {
                id: r.id,
                a: r.a.clone()?,
                b: r.b.clone()?,
            })
        })
        .collect()
}

fn judge(
    run: &Run,
    slug: &str,
    rows: Vec<CompareInRow>,
    model: &Path,
    exe: &Path,
    emit: &mut dyn FnMut(RunEvent),
) -> Result<CompareOutput, String> {
    let paths = &run.job.paths;
    let input_path = paths
        .dir
        .join(format!("{}.compare-{slug}.in.json", paths.stem));
    let out_path = paths
        .dir
        .join(format!("{}.compare-{slug}.out.json", paths.stem));
    let profile = run.profile.as_ref();
    let glossary =
        pipeline::write_merged_glossary(paths, pipeline::find_glossary().as_deref(), profile)?;
    let input = CompareInput {
        version: VERSION,
        rows,
        profile_prompt: profile_prompt(profile),
        glossary: profile.map(|p| p.glossary.clone()).unwrap_or_default(),
        hints: Vec::new(),
    };
    fsio::save_json(&input_path, &input).map_err(|e| e.to_string())?;
    call_worker(
        "compare",
        exe,
        model,
        (&input_path, &out_path),
        glossary_args(glossary.as_deref()),
        &paths.dir.join("llm-compare.log"),
        emit,
    )?;
    let output = fsio::load_json(&out_path).map_err(|e| load_error("the AI's picks", e))?;
    let _ = std::fs::remove_file(&input_path);
    let _ = std::fs::remove_file(&out_path);
    Ok(output)
}

/// Puts the worker's picks into the rows. Only rows that were asked about can change, and each pick is
/// checked again here: a merge has to pass `check_merged`, and A or B carry no text from the model.
/// A row that was asked about and has no usable pick ends up `Failed`.
fn apply_picks(rows: &mut [CompareRow], picks: &[CompareOutRow]) {
    for row in rows.iter_mut().filter(|r| needs_llm(r)) {
        let pick = picks.iter().find(|p| p.id == row.id).map(|p| &p.ai);
        row.ai = checked_pick(row, pick);
    }
}

fn checked_pick(row: &CompareRow, pick: Option<&AiPick>) -> AiPick {
    let failed = |reason: String| AiPick {
        choice: None,
        merged: None,
        reason,
        status: AiStatus::Failed,
    };
    let Some(pick) = pick else {
        return failed(String::new());
    };
    let reason: String = pick.reason.chars().take(MAX_REASON_CHARS).collect();
    if pick.status != AiStatus::Picked {
        return failed(reason);
    }
    let (a, b) = (
        row.a.as_deref().unwrap_or(""),
        row.b.as_deref().unwrap_or(""),
    );
    let merged = match pick.choice {
        None => return failed(reason),
        Some(AiChoice::A | AiChoice::B) => None,
        Some(AiChoice::Merged) => match pick.merged.as_deref() {
            Some(m) if check_merged(m, a, b) => Some(m.to_string()),
            _ => return failed(reason),
        },
    };
    AiPick {
        choice: pick.choice,
        merged,
        reason,
        status: AiStatus::Picked,
    }
}

/// A run with an existing slug keeps what the user decided, on rows whose two texts are unchanged.
fn keep_user_picks(rows: &mut [CompareRow], previous: &[CompareRow]) {
    for row in rows {
        row.user = previous
            .iter()
            .find(|o| o.id == row.id && o.a == row.a && o.b == row.b)
            .and_then(|o| o.user.clone());
    }
}

// ------------------------------------------------------------------------- Open result

/// Rebuilds a finished job from a result folder or its `<stem>.json`.
pub fn load_result(path: &Path) -> Result<LoadedResult, String> {
    let paths = find_result(path)?;
    let transcript = Transcript::load(&paths.transcript()).map_err(|_| {
        format!(
            "\"{}\" is not a ViPr result, or it is damaged.",
            paths.transcript().display()
        )
    })?;
    let media = find_media(&paths, &transcript);
    Ok(LoadedResult {
        media,
        paths,
        transcript,
    })
}

/// Records where the media is after the user located it, so Open result finds it next time.
pub fn locate_media(paths: &ProjectPaths, media: &Path) -> Result<(), String> {
    ProjectMeta::new(media, &paths.stem)
        .save(&paths.project())
        .map_err(|e| e.to_string())
}

fn find_result(path: &Path) -> Result<ProjectPaths, String> {
    if path.is_dir() {
        return result_in_folder(path);
    }
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let stem = name
        .strip_suffix(".project.json")
        .or_else(|| name.strip_suffix(".json"))
        .ok_or_else(|| {
            format!("\"{name}\" is not a ViPr result. Choose its folder or .json file.")
        })?;
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    Ok(ProjectPaths::new(dir, stem))
}

/// The folder is normally named after the stem. If it was renamed, the one transcript inside it counts.
fn result_in_folder(dir: &Path) -> Result<ProjectPaths, String> {
    if let Some(name) = dir.file_name().map(|n| n.to_string_lossy().into_owned()) {
        let paths = ProjectPaths::new(dir, name);
        if paths.transcript().is_file() {
            return Ok(paths);
        }
    }
    let listing = std::fs::read_dir(dir).map_err(|e| format!("cannot read that folder ({e})."))?;
    let mut found: Vec<ProjectPaths> = listing
        .filter_map(|entry| {
            let name = entry.ok()?.file_name().to_string_lossy().into_owned();
            let stem = name.strip_suffix(".json")?.to_string();
            let paths = ProjectPaths::new(dir, stem);
            Transcript::load(&paths.transcript()).ok().map(|_| paths)
        })
        .collect();
    match found.len() {
        0 => Err(format!("\"{}\" holds no ViPr result.", dir.display())),
        1 => Ok(found.remove(0)),
        _ => Err(format!(
            "\"{}\" holds several results. Choose one of their .json files.",
            dir.display()
        )),
    }
}

/// The linked media if it is still there, else a file of the same name beside the results.
fn find_media(paths: &ProjectPaths, transcript: &Transcript) -> Option<PathBuf> {
    let linked = ProjectMeta::load(&paths.project())
        .ok()
        .map(|m| m.media_path)
        .filter(|p| p.is_file());
    linked.or_else(|| {
        // only the name: `source` comes from a file that may have been edited
        let name = Path::new(&transcript.source).file_name()?;
        [Some(paths.dir.as_path()), paths.dir.parent()]
            .into_iter()
            .flatten()
            .map(|d| d.join(name))
            .find(|p| p.is_file())
    })
}

#[cfg(test)]
mod tests {
    use engine::project::{CueFlag, TranslateStats, UserPick};

    use super::*;

    /// A fresh folder under the system temp dir, removed when dropped.
    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> TestDir {
            let dir = std::env::temp_dir().join(format!(
                "vipr-jobs-{name}-{}-{:?}",
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

    fn kind(target: &str) -> JobKind {
        JobKind::Translate {
            job: 1,
            target: target.into(),
            source: TranslateSource::Transcript,
        }
    }

    fn timed(id: u64, text: &str) -> TimedText {
        TimedText {
            id,
            start: id as f64,
            end: id as f64 + 0.5,
            text: text.into(),
        }
    }

    fn running(k: JobKind) -> Running {
        Running {
            kind: k,
            stem: "a".into(),
            detail: String::new(),
            progress: 0.0,
            shown: 0.0,
            cancelling: false,
        }
    }

    #[test]
    fn a_chapters_run_counts_as_pending_only_for_its_own_recording() {
        let mine = JobKind::Chapters { job: 1 };
        let mut state = State::default();
        assert_eq!(pending(&state, &mine), None);
        enqueue(&mut state, mine.clone());
        assert_eq!(pending(&state, &mine), Some(0.0));
        assert_eq!(pending(&state, &JobKind::Chapters { job: 2 }), None);
        let _ = next(&mut state);
        assert_eq!(
            pending(&state, &mine),
            None,
            "off the queue, not yet running"
        );
    }

    #[test]
    fn a_percentage_only_shows_once_there_is_one() {
        assert_eq!(percent_text(0.0), "");
        assert_eq!(percent_text(0.004), "");
        assert_eq!(percent_text(0.43), "43%");
        assert_eq!(percent_text(1.0), "100%");
    }

    #[test]
    fn queue_runs_in_order_and_never_holds_a_run_twice() {
        let mut state = State::default();
        enqueue(&mut state, kind("fa"));
        enqueue(&mut state, kind("de"));
        enqueue(&mut state, kind("fa"));
        assert_eq!(state.queue.len(), 2);
        assert_eq!(next(&mut state), Some(kind("fa")));
        state.running = Some(running(kind("fa")));
        enqueue(&mut state, kind("fa"));
        assert_eq!(state.queue.len(), 1, "that one is running now");
        assert_eq!(next(&mut state), Some(kind("de")));
        assert_eq!(next(&mut state), None);
    }

    #[test]
    fn a_run_whose_job_is_gone_does_not_hold_the_slot() {
        let mut state = State::default();
        let context = JobContext {
            job: None,
            model: None,
            profile: None,
        };
        let (_, out) = start(&mut state, kind("fa"), context);
        assert!(!busy(&state));
        assert!(matches!(out, Out::Notice(_)));
    }

    #[test]
    fn the_slot_is_free_after_any_ending() {
        let endings = [
            RunEvent::Failed("no disk".into()),
            RunEvent::Failed(CANCELLED.into()),
            RunEvent::Finished(Finished::Compared {
                job: 7,
                not_judged: 0,
            }),
        ];
        for ending in endings {
            let mut state = State {
                running: Some(running(kind("fa"))),
                ..State::default()
            };
            assert!(busy(&state));
            let _ = update(&mut state, Msg::Event(ending));
            assert!(!busy(&state));
        }
    }

    #[test]
    fn events_after_the_end_are_harmless_and_a_cancelled_run_says_so() {
        let mut state = State {
            running: Some(running(kind("fa"))),
            ..State::default()
        };
        let (_, out) = update(&mut state, Msg::Event(RunEvent::Failed(CANCELLED.into())));
        let Out::Notice(notice) = out else {
            panic!("expected a notice");
        };
        assert!(
            notice.contains("cancelled") && !notice.contains("failed"),
            "{notice}"
        );
        let _ = update(&mut state, Msg::Event(RunEvent::Progress(0.5)));
        let _ = update(&mut state, Msg::Cancel);
        assert!(!busy(&state));
    }

    #[test]
    fn compare_opens_review_when_done() {
        let mut state = State::default();
        let finished = Finished::Compared {
            job: 7,
            not_judged: 0,
        };
        let (_, out) = update(&mut state, Msg::Event(RunEvent::Finished(finished)));
        assert!(matches!(out, Out::Open(Page::Review(7))));
    }

    #[test]
    fn remove_takes_a_waiting_run_out_and_ignores_a_stale_index() {
        let mut state = State::default();
        enqueue(&mut state, kind("fa"));
        enqueue(&mut state, kind("de"));
        let _ = update(&mut state, Msg::Remove(0));
        let _ = update(&mut state, Msg::Remove(9));
        assert_eq!(next(&mut state), Some(kind("de")));
    }

    #[test]
    fn names_cannot_leave_the_result_folder() {
        for good in ["fa", "zh-Hans", "my_import.2"] {
            assert!(check_name(good).is_ok(), "{good}");
        }
        for bad in ["", "..", "../x", "a/b", "a\\b", "c:x", ".hidden", "a b"] {
            assert!(check_name(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn translate_only_models_are_refused_and_general_ones_named_by_id() {
        let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models.json");
        let registry = Registry::load_default(Some(&file)).unwrap();
        let mut refused = 0;
        for m in &registry.models {
            let result = model_id(Path::new(&m.file_name()), Some(&registry));
            if m.has_role(Role::Llm) {
                assert_eq!(result.as_deref(), Ok(m.id.as_str()));
            } else if m.has_role(Role::Translate) {
                refused += 1;
                assert!(result.is_err(), "{}", m.id);
            }
        }
        assert!(refused >= 3, "the registry should list these models");
        // the same files outside the registry, or with no registry at all
        for name in [
            "TranslateGemma-4b.gguf",
            "Hunyuan-MT-7B-Q4.gguf",
            "Seed-X-Instruct.gguf",
        ] {
            assert!(model_id(Path::new(name), None).is_err(), "{name}");
        }
        assert_eq!(
            model_id(Path::new("C:/m/mine.gguf"), None).unwrap(),
            "mine.gguf"
        );
    }

    fn out_cue(id: u64, text: &str, status: CueStatus) -> TranslationCue {
        TranslationCue {
            id,
            start: 99.0,
            end: 100.0,
            source: "changed by the worker".into(),
            text: text.into(),
            status,
            flags: vec![],
        }
    }

    fn output(cues: Vec<TranslationCue>) -> TranslateOutput {
        TranslateOutput {
            version: 1,
            target: "de".into(),
            cues,
            stats: TranslateStats::default(),
        }
    }

    #[test]
    fn translation_file_trusts_its_own_cues_over_the_output() {
        let cues = [
            timed(1, "one"),
            timed(2, "two"),
            timed(3, "three"),
            timed(4, "four"),
        ];
        let flagged = TranslationCue {
            flags: vec![CueFlag::NumberMismatch],
            ..out_cue(1, "eins", CueStatus::Translated)
        };
        let file = translation_file(
            "de",
            &cues,
            output(vec![
                flagged,
                out_cue(2, "   ", CueStatus::Translated),
                out_cue(3, "whatever", CueStatus::Untranslated),
                out_cue(9, "stray", CueStatus::Translated),
            ]),
            "m".into(),
            None,
            None,
        );
        assert_eq!(file.source_hash, source_hash(&cues));
        assert_eq!(file.cues.len(), 4);
        assert_eq!(file.cues[0].text, "eins");
        assert_eq!(file.cues[0].source, "one");
        assert_eq!(file.cues[0].start, 1.0);
        assert_eq!(file.cues[0].flags, [CueFlag::NumberMismatch]);
        // only blanks are a failure here: "" is how the worker leaves a cue with no words of its own
        for (cue, source) in file.cues[1..].iter().zip(["two", "three", "four"]) {
            assert_eq!(cue.status, CueStatus::Untranslated);
            assert_eq!(cue.text, source, "holds its source");
            assert!(cue.flags.is_empty());
        }
    }

    #[test]
    fn a_cue_left_empty_by_its_sentence_stays_empty() {
        let cues = [timed(1, "That's"), timed(2, "interesting.")];
        let file = translation_file(
            "de",
            &cues,
            output(vec![
                out_cue(1, "", CueStatus::Translated),
                out_cue(2, "Interessant.", CueStatus::Translated),
            ]),
            "m".into(),
            None,
            None,
        );
        assert_eq!(file.cues[0].text, "");
        assert_eq!(file.cues[0].status, CueStatus::Translated);
        assert_eq!(file.cues[0].source, "That's");
        assert_eq!(file.cues[1].text, "Interessant.");
    }

    #[test]
    fn an_edit_stays_when_the_rest_of_its_sentence_is_translated_again() {
        let previous = TranslationFile {
            version: 1,
            target: "de".into(),
            source_hash: String::new(),
            model: String::new(),
            profile: None,
            cues: vec![TranslationCue {
                source: "first half".into(),
                ..out_cue(1, "von Hand", CueStatus::Edited)
            }],
        };
        let cues = [timed(1, "first half"), timed(2, "second half.")];
        let file = translation_file(
            "de",
            &cues,
            output(vec![
                out_cue(1, "erste", CueStatus::Translated),
                out_cue(2, "zweite Hälfte.", CueStatus::Translated),
            ]),
            "m".into(),
            None,
            Some(&previous),
        );
        assert_eq!(file.cues[0].text, "von Hand");
        assert_eq!(file.cues[0].status, CueStatus::Edited);
        assert_eq!(file.cues[1].text, "zweite Hälfte.");
        assert_eq!(file.cues[1].status, CueStatus::Translated);
    }

    #[test]
    fn a_rerun_keeps_edits_of_the_same_source_only() {
        let previous = TranslationFile {
            version: 1,
            target: "de".into(),
            source_hash: String::new(),
            model: String::new(),
            profile: None,
            cues: vec![
                TranslationCue {
                    source: "one".into(),
                    ..out_cue(1, "von Hand", CueStatus::Edited)
                },
                TranslationCue {
                    source: "old two".into(),
                    ..out_cue(2, "von Hand", CueStatus::Edited)
                },
                TranslationCue {
                    source: "three".into(),
                    ..out_cue(3, "alt", CueStatus::Translated)
                },
            ],
        };
        let cues = [timed(1, "one"), timed(2, "two"), timed(3, "three")];
        let file = translation_file(
            "de",
            &cues,
            output(vec![
                out_cue(1, "eins", CueStatus::Translated),
                out_cue(2, "zwei", CueStatus::Translated),
                out_cue(3, "drei", CueStatus::Translated),
            ]),
            "m".into(),
            None,
            Some(&previous),
        );
        assert_eq!(file.cues[0].text, "von Hand");
        assert_eq!(file.cues[0].status, CueStatus::Edited);
        assert_eq!(file.cues[0].start, 1.0, "times follow the new cues");
        assert_eq!(file.cues[1].text, "zwei");
        assert_eq!(file.cues[2].text, "drei");
    }

    #[test]
    fn import_times_move_by_the_offset_and_stay_on_the_timeline() {
        let cues = shifted(vec![timed(1, "x")], 2.5);
        assert_eq!((cues[0].start, cues[0].end), (3.5, 4.0));
        let cues = shifted(vec![timed(1, "x")], -1.2);
        assert_eq!(cues[0].start, 0.0);
        assert!((cues[0].end - 0.3).abs() < 1e-9);
    }

    #[test]
    fn untranslated_lines_are_not_a_compare_side() {
        let dir = TestDir::new("side");
        let paths = ProjectPaths::new(&dir.0, "a");
        let file = TranslationFile {
            version: 1,
            target: "de".into(),
            source_hash: String::new(),
            model: String::new(),
            profile: None,
            cues: vec![
                out_cue(1, "eins", CueStatus::Translated),
                out_cue(2, "two", CueStatus::Untranslated),
                out_cue(3, "drei", CueStatus::Edited),
                out_cue(4, "", CueStatus::Translated),
            ],
        };
        fsio::save_json(&paths.translation("de"), &file).unwrap();
        let side = CompareSource::Translation { lang: "de".into() };
        let (cues, offset) = side_cues(&paths, &side).unwrap();
        assert_eq!(offset, 0.0);
        assert_eq!(cues.iter().map(|c| c.id).collect::<Vec<_>>(), [1, 3]);
        let escape = CompareSource::Translation {
            lang: "../de".into(),
        };
        assert!(side_cues(&paths, &escape).is_err());
    }

    fn row(id: u64, a: Option<&str>, b: Option<&str>, ai: AiPick) -> CompareRow {
        CompareRow {
            id,
            start: 0.0,
            end: 1.0,
            a: a.map(str::to_string),
            b: b.map(str::to_string),
            ai,
            user: None,
        }
    }

    fn picked(choice: AiChoice, merged: Option<&str>, reason: &str) -> AiPick {
        AiPick {
            choice: Some(choice),
            merged: merged.map(str::to_string),
            reason: reason.into(),
            status: AiStatus::Picked,
        }
    }

    fn out_row(id: u64, ai: AiPick) -> CompareOutRow {
        CompareOutRow { id, ai }
    }

    #[test]
    fn only_rows_that_differ_are_sent_to_the_model() {
        let rows = [
            row(1, Some("same"), Some("same"), AiPick::default()),
            row(2, Some("only a"), None, AiPick::default()),
            row(3, Some("left"), Some("right"), AiPick::default()),
        ];
        let asks = rows_to_judge(&rows);
        assert_eq!(asks.len(), 1);
        assert_eq!((asks[0].id, asks[0].a.as_str()), (3, "left"));
    }

    #[test]
    fn picks_cannot_touch_rows_that_were_not_asked_about() {
        let settled = AiPick {
            choice: Some(AiChoice::A),
            status: AiStatus::Identical,
            ..AiPick::default()
        };
        let mut rows = [row(1, Some("same"), Some("same"), settled.clone())];
        let forged = out_row(1, picked(AiChoice::Merged, Some("forged"), "x"));
        apply_picks(&mut rows, &[forged]);
        assert_eq!(rows[0].ai, settled);
    }

    #[test]
    fn a_or_b_carry_no_model_text_and_reasons_are_capped() {
        let mut rows = [row(1, Some("left"), Some("right"), AiPick::default())];
        let long = "r".repeat(500);
        let pick = picked(AiChoice::B, Some("ignore me"), &long);
        apply_picks(&mut rows, &[out_row(1, pick)]);
        assert_eq!(rows[0].ai.status, AiStatus::Picked);
        assert_eq!(rows[0].ai.merged, None);
        assert_eq!(rows[0].ai.reason.chars().count(), MAX_REASON_CHARS);
        assert_eq!(rows[0].resolved_text(), Some("right"));
    }

    #[test]
    fn unusable_answers_fail_the_row() {
        let a = "the quick brown fox jumps";
        let b = "the quick brown fox leaps";
        let rejected_merge = picked(AiChoice::Merged, Some("visit evil dot example now"), "x");
        let no_merge = picked(AiChoice::Merged, None, "x");
        let not_picked = AiPick {
            status: AiStatus::Failed,
            ..picked(AiChoice::A, None, "x")
        };
        let no_choice = AiPick {
            choice: None,
            ..picked(AiChoice::A, None, "x")
        };
        let answers = [
            Some(rejected_merge),
            Some(no_merge),
            Some(not_picked),
            Some(no_choice),
            None,
        ];
        for answer in answers {
            let mut rows = [row(1, Some(a), Some(b), AiPick::default())];
            let picks: Vec<CompareOutRow> =
                answer.iter().cloned().map(|ai| out_row(1, ai)).collect();
            apply_picks(&mut rows, &picks);
            assert_eq!(rows[0].ai.status, AiStatus::Failed, "{answer:?}");
            assert_eq!(rows[0].ai.choice, None);
            assert_eq!(rows[0].ai.merged, None);
            assert_eq!(rows[0].resolved_text(), None);
        }
    }

    #[test]
    fn a_merge_built_from_both_sides_is_kept() {
        let (a, b) = ("the quick brown fox", "the quick brown fox jumps");
        let mut rows = [row(1, Some(a), Some(b), AiPick::default())];
        let pick = picked(AiChoice::Merged, Some("the quick brown fox jumps"), "x");
        apply_picks(&mut rows, &[out_row(1, pick)]);
        assert_eq!(rows[0].ai.status, AiStatus::Picked);
        assert_eq!(rows[0].resolved_text(), Some("the quick brown fox jumps"));
    }

    #[test]
    fn a_rerun_keeps_user_picks_where_the_text_is_unchanged() {
        let mut old_same = row(1, Some("a"), Some("b"), AiPick::default());
        old_same.user = Some(UserPick::B);
        let mut old_changed = row(2, Some("a"), Some("b"), AiPick::default());
        old_changed.user = Some(UserPick::B);
        let mut rows = [
            row(1, Some("a"), Some("b"), AiPick::default()),
            row(2, Some("a"), Some("new"), AiPick::default()),
        ];
        keep_user_picks(&mut rows, &[old_same, old_changed]);
        assert_eq!(rows[0].user, Some(UserPick::B));
        assert_eq!(rows[1].user, None);
    }

    /// A result folder with a transcript of `source`.
    fn result(dir: &Path, stem: &str, source: &str) -> ProjectPaths {
        let paths = ProjectPaths::new(dir, stem);
        let asr = engine::transcript::AsrResult {
            model: None,
            language: "en".into(),
            audio_seconds: 0.0,
            elapsed_seconds: 0.0,
            segments: vec![],
        };
        Transcript::build(source, &asr, None, &[])
            .save(&paths.transcript())
            .unwrap();
        paths
    }

    #[test]
    fn open_result_takes_a_folder_or_its_json_files() {
        let root = TestDir::new("open");
        let dir = root.0.join("a (mp3)");
        let paths = result(&dir, "a (mp3)", "a.mp3");
        let media = root.0.join("elsewhere.mp3");
        std::fs::write(&media, b"x").unwrap();
        locate_media(&paths, &media).unwrap();

        for pick in [dir.clone(), paths.transcript(), paths.project()] {
            let loaded = load_result(&pick).unwrap();
            assert_eq!(loaded.paths, paths, "{}", pick.display());
            assert_eq!(loaded.media.as_deref(), Some(media.as_path()));
            assert_eq!(loaded.transcript.source, "a.mp3");
        }
    }

    #[test]
    fn open_result_finds_the_one_transcript_in_a_renamed_folder() {
        let root = TestDir::new("renamed");
        let dir = root.0.join("renamed");
        let paths = result(&dir, "talk", "talk.mp4");
        // other sidecars in the folder are not transcripts
        fsio::save_json(&paths.marks(), &vec![1u32]).unwrap();
        let asr = "{\"audio_seconds\":1,\"segments\":[]}";
        std::fs::write(dir.join("talk.asr.json"), asr).unwrap();
        assert_eq!(load_result(&dir).unwrap().paths, paths);

        result(&dir, "second", "second.mp4");
        let error = load_result(&dir).unwrap_err();
        assert!(error.contains("several"), "{error}");
    }

    #[test]
    fn open_result_says_when_the_media_has_moved() {
        let root = TestDir::new("moved");
        let dir = root.0.join("a");
        let paths = result(&dir, "a", "a.mp3");
        locate_media(&paths, &root.0.join("gone.mp3")).unwrap();
        assert_eq!(load_result(&dir).unwrap().media, None);

        // moved together with its results
        let beside = root.0.join("a.mp3");
        std::fs::write(&beside, b"x").unwrap();
        assert_eq!(load_result(&dir).unwrap().media, Some(beside.clone()));
        std::fs::remove_file(&beside).unwrap();
        let inside = dir.join("a.mp3");
        std::fs::write(&inside, b"x").unwrap();
        assert_eq!(load_result(&dir).unwrap().media, Some(inside));
    }

    #[test]
    fn open_result_works_without_a_project_file_and_ignores_paths_in_source() {
        let root = TestDir::new("oldresult");
        let dir = root.0.join("deep").join("a");
        let paths = result(&dir, "a", "../../secret.mp3");
        std::fs::write(root.0.join("secret.mp3"), b"x").unwrap();
        let loaded = load_result(&paths.transcript()).unwrap();
        assert_eq!(loaded.media, None);
        assert_eq!(loaded.transcript.source, "../../secret.mp3");
    }

    #[test]
    fn open_result_refuses_what_is_not_a_result() {
        let root = TestDir::new("junk");
        assert!(load_result(&root.0).is_err(), "an empty folder");
        let missing = root.0.join("missing.json");
        assert!(load_result(&missing).is_err());
        let text = root.0.join("notes.txt");
        std::fs::write(&text, "x").unwrap();
        assert!(load_result(&text).is_err());
        let broken = root.0.join("broken.json");
        std::fs::write(&broken, "{not json").unwrap();
        let error = load_result(&broken).unwrap_err();
        assert!(error.contains("not a ViPr result"), "{error}");
        let marks = ProjectPaths::new(&root.0, "a").marks();
        fsio::save_json(&marks, &vec![1u32]).unwrap();
        assert!(load_result(&marks).is_err(), "a sidecar is not a result");
    }
}
