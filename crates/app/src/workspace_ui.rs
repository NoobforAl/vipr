//! The open project (`engine::workspace`): its recordings and their order, saving and opening it,
//! restoring an unsaved one, the change log and project undo.
//!
//! There is always a project once a file is added: dropping a file with none open starts an unsaved
//! one in `workspace::temporary_root()`, saved after every change, so nothing is lost and nothing
//! has to be clicked first. Save As gives it a name and a place. A named project marks itself
//! changed instead, keeps autosaved copies in `<file>.backups/`, and asks before closing.
//!
//! Unlike the other feature modules this one changes the job list itself (a project *is* the list of
//! recordings on Home), so its handlers are an `impl App` block here instead of an `update` that
//! returns an `Out`. Every job of the project carries its clip id in `Job::clip`.

use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use engine::history::{Entry, HistoryLog, Kind, Undo};
use engine::workspace::{self, ResultRef, Workspace};
use iced::widget::{button, column, container, pick_list, row, scrollable, text, text_input};
use iced::{Alignment, Element, Length, Task};

use crate::theme::{self, Tokens};
use crate::{App, Job, JobState, Message, export_ui, icons, import_ui, jobs, saver, semibold};

/// Recent projects remembered for the Open list.
const RECENT_MAX: usize = 8;
/// Entries shown in the History list.
const HISTORY_SHOWN: usize = 50;
/// Unsaved projects kept for restoring; older ones are deleted at start-up.
const KEEP_UNSAVED: usize = 5;
/// A named project's autosaved copy is written at most this often.
const BACKUP_EVERY: Duration = Duration::from_secs(30);

pub struct Open {
    pub path: PathBuf,
    pub data: Workspace,
    /// Changed since the user last saved it (named projects only; unsaved ones save themselves).
    pub dirty: bool,
    /// ViPr made it in its own folder for unsaved projects (`workspace::is_temporary`). Only such a
    /// project's folder is ever deleted.
    temporary: bool,
    log: HistoryLog,
    last_backup: Option<Instant>,
}

impl Open {
    pub fn temporary(&self) -> bool {
        self.temporary
    }

    /// For the title bar: the project's name, `•` while it has unsaved changes.
    pub fn title(&self) -> String {
        let name = if self.temporary() {
            "Unsaved project".to_string()
        } else {
            file_stem(&self.path)
        };
        if self.dirty {
            format!("{name} •")
        } else {
            name
        }
    }
}

/// What to do once the "save changes?" question is answered.
#[derive(Debug, Clone, PartialEq)]
pub enum After {
    Close,
    New,
    Open(PathBuf),
}

#[derive(Default)]
pub struct State {
    pub open: Option<Open>,
    undo: Undo<Workspace>,
    recent: Vec<PathBuf>,
    /// An unsaved project from an earlier run, and how many recordings it has.
    restorable: Option<(PathBuf, usize)>,
    /// The "save changes?" question is showing, for this.
    asking: Option<After>,
    /// The user answered it, so closing goes ahead.
    pub close_confirmed: bool,
    history_open: bool,
    history: Vec<Entry>,
    /// Counts opens, so clips still loading for an earlier project are dropped.
    generation: u64,
    /// The combination whose name is being typed: its first letter made the one Undo step for all of
    /// them. Any other change ends it.
    renaming: Option<u64>,
    /// `None`: `workspace::temporary_root()`. Tests point it elsewhere.
    temp_root: Option<PathBuf>,
}

impl State {
    fn root(&self) -> PathBuf {
        self.temp_root
            .clone()
            .unwrap_or_else(workspace::temporary_root)
    }

    pub fn is_asking(&self) -> bool {
        self.asking.is_some()
    }
}

#[derive(Debug, Clone)]
pub enum Msg {
    New,
    Open,
    Picked(Option<PathBuf>),
    OpenRecent(Recent),
    Loaded(Result<(PathBuf, Workspace), String>),
    ClipsLoaded(u64, Vec<ClipLoad>),
    Save,
    SaveAs,
    SaveAsPicked(Option<PathBuf>),
    Restore,
    Discard,
    /// Move a recording (by job id) up or down.
    Move(u64, isize),
    Undo,
    Redo,
    ToggleHistory,
    /// A new combination of all the recordings, in their order.
    SeqNew,
    SeqRename(u64, String),
    SeqAdd(u64, ClipChoice),
    /// Move the item at an index of a combination.
    SeqMove(u64, usize, isize),
    SeqRemoveItem(u64, usize),
    SeqDelete(u64),
    /// Open the export dialog for a combination.
    SeqExport(u64),
    /// Answers to "save changes?".
    AnswerSave,
    AnswerDiscard,
    AnswerCancel,
}

/// A recent project in the Open list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recent(pub PathBuf);

impl fmt::Display for Recent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", crate::clean_separators(&file_stem(&self.0)))
    }
}

/// A recording offered in a combination's "Add" list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipChoice {
    id: u64,
    title: String,
}

impl fmt::Display for ClipChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", crate::clean_separators(&self.title))
    }
}

/// A clip read back from disk, ready to become a job.
#[derive(Debug, Clone)]
pub struct ClipLoad {
    clip: u64,
    media: Option<PathBuf>,
    result: Option<Result<jobs::LoadedResult, String>>,
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn recent_file() -> Option<PathBuf> {
    if cfg!(test) {
        return None; // tests must not change the user's recent list
    }
    Some(
        dirs::config_dir()?
            .join(engine::APP_DIR_NAME)
            .join("recent-projects.json"),
    )
}

fn load_recent() -> Vec<PathBuf> {
    recent_file()
        .and_then(|f| engine::fsio::load_json::<Vec<PathBuf>>(&f).ok())
        .unwrap_or_default()
        .into_iter()
        .filter(|p| p.is_file())
        .take(RECENT_MAX)
        .collect()
}

/// Reads each clip's results and finds its media, off the UI thread.
fn load_clips(project: Option<PathBuf>, data: Workspace, clips: Vec<u64>) -> Vec<ClipLoad> {
    clips
        .into_iter()
        .filter_map(|id| {
            let clip = data.clip(id)?;
            let media = data.media_path(id, project.as_deref());
            let result = clip
                .result
                .as_ref()
                .filter(|r| r.paths().transcript().is_file())
                .map(|r| jobs::load_result(&r.paths().transcript()));
            Some(ClipLoad {
                clip: id,
                media,
                result,
            })
        })
        .collect()
}

/// "5 min ago", for the History list. There is no time zone library, and relative times read well.
fn ago(t: u64) -> String {
    let s = workspace::now().saturating_sub(t);
    match s {
        0..60 => "just now".to_string(),
        60..3600 => format!("{} min ago", s / 60),
        3600..86_400 => format!("{} h ago", s / 3600),
        _ => format!("{} days ago", s / 86_400),
    }
}

/// An entry's measurements in words.
fn metrics_text(e: &Entry) -> String {
    let m = |k: &str| e.metrics.get(k).copied();
    match e.kind {
        Kind::Transcribed => match (m("audio_seconds"), m("elapsed_seconds")) {
            (Some(audio), Some(took)) if took > 0.0 => format!(
                "{} of sound in {}, {:.0}× real time",
                crate::mmss(audio as u64),
                crate::mmss(took as u64),
                audio / took
            ),
            _ => String::new(),
        },
        Kind::Exported | Kind::Translated => {
            let mut parts = Vec::new();
            if let Some(b) = m("bytes") {
                parts.push(crate::human(b as u64));
            }
            if let Some(c) = m("cues") {
                parts.push(format!("{c:.0} lines"));
            }
            if let Some(took) = m("elapsed_seconds") {
                parts.push(format!("in {}", crate::mmss(took as u64)));
            }
            parts.join(", ")
        }
        Kind::MarksChanged => match m("marks") {
            Some(n) => format!("{n:.0} mark{}", if n == 1.0 { "" } else { "s" }),
            None => String::new(),
        },
        Kind::CutsChanged => match (m("cuts"), m("cut_seconds")) {
            (Some(0.0), _) => "no cuts".to_string(),
            (Some(n), Some(left_out)) => format!(
                "{n:.0} cut{}, {} left out",
                if n == 1.0 { "" } else { "s" },
                crate::mmss(left_out as u64)
            ),
            _ => String::new(),
        },
        _ => String::new(),
    }
}

impl App {
    /// At start-up: the recent list, and an unsaved project from last time to offer back.
    pub(crate) fn ws_init(&mut self) {
        if cfg!(test) {
            return; // tests must never touch the user's own unsaved projects
        }
        self.workspace.recent = load_recent();
        let root = self.workspace.root();
        let found = workspace::temporary_projects(&root);
        for (old, _) in found.iter().skip(KEEP_UNSAVED) {
            let _ = workspace::remove_temporary(old, &root);
        }
        // folders of projects that never got a clip
        if let Ok(read) = std::fs::read_dir(&root) {
            let kept: HashSet<&Path> = found.iter().map(|(p, _)| p.as_path()).collect();
            for e in read.flatten() {
                let file = e.path().join(format!("untitled.{}", workspace::EXTENSION));
                if !kept.contains(file.as_path()) {
                    let _ = workspace::remove_temporary(&file, &root);
                }
            }
        }
        self.workspace.restorable = found.first().map(|(p, w)| (p.clone(), w.clips.len()));
    }

    /// The open project, starting an unsaved one if there is none.
    fn ws_open(&mut self) -> &mut Open {
        if self.workspace.open.is_none() {
            let path = workspace::new_temporary(&self.workspace.root());
            let log = HistoryLog::new(workspace::history_path(&path));
            self.workspace.open = Some(Open {
                path,
                data: Workspace::new(""),
                dirty: false,
                temporary: true,
                log,
                last_backup: None,
            });
            self.ws_log(Entry::new(Kind::Created, ""));
        }
        self.workspace.open.as_mut().expect("just opened")
    }

    /// Writes to the change log on the save thread, and to the History list.
    pub(crate) fn ws_log(&mut self, entry: Entry) {
        let Some(open) = &self.workspace.open else {
            return;
        };
        let log = open.log.clone();
        let line = entry.clone();
        saver::run(move || {
            let _ = log.append(&line); // the log is a record; failing to write it must not stop work
        });
        self.workspace.history.push(entry);
        let extra = self.workspace.history.len().saturating_sub(HISTORY_SHOWN);
        self.workspace.history.drain(..extra);
    }

    /// After a change: an unsaved project saves itself, a named one is marked changed and backed up.
    fn ws_persist(&mut self) {
        let Some(open) = self.workspace.open.as_mut() else {
            return;
        };
        if open.temporary() {
            let mut data = open.data.clone();
            let path = open.path.clone();
            saver::run(move || {
                let _ = data.save(&path);
            });
            return;
        }
        open.dirty = true;
        if open.last_backup.is_none_or(|t| t.elapsed() >= BACKUP_EVERY) {
            open.last_backup = Some(Instant::now());
            let data = open.data.clone();
            let path = open.path.clone();
            saver::run(move || {
                let _ = data.backup(&path, workspace::KEEP_BACKUPS);
            });
        }
    }

    /// Records the project as it is, so the change about to be made can be undone.
    fn ws_snapshot(&mut self, what: &str) {
        let before = self.ws_open().data.clone();
        self.workspace.undo.record(what, before);
        self.workspace.renaming = None;
    }

    /// Files just added to the list become the project's clips.
    pub(crate) fn ws_jobs_added(&mut self, added: &[u64]) {
        if added.is_empty() {
            return;
        }
        let what = if added.len() == 1 {
            "Add a recording".to_string()
        } else {
            format!("Add {} recordings", added.len())
        };
        self.ws_snapshot(&what);
        for id in added {
            let Some((path, size)) = self.job(*id).map(|j| (j.path.clone(), j.size)) else {
                continue;
            };
            let clip = self.ws_open().data.add_clip(&path, size);
            if let Some(j) = self.job_mut(*id) {
                j.clip = Some(clip);
            }
            self.ws_log(Entry::new(Kind::ClipAdded, crate::file_name(&path)).clip(clip));
        }
        self.ws_persist();
    }

    /// A transcription finished: the clip remembers its results, and the log how long it took.
    pub(crate) fn ws_job_done(&mut self, id: u64) {
        let Some(j) = self.job(id) else {
            return;
        };
        let (Some(clip), Some(paths)) = (j.clip, j.paths.clone()) else {
            return;
        };
        let mut entry = Entry::new(Kind::Transcribed, j.name()).clip(clip);
        if let Some(t) = &j.transcript {
            entry = entry.metric("audio_seconds", t.audio_seconds);
        }
        if let Some(started) = j.started {
            entry = entry.metric("elapsed_seconds", started.elapsed().as_secs_f64());
        }
        if self
            .ws_open()
            .data
            .set_result(clip, ResultRef::from(&paths))
        {
            self.ws_persist();
        }
        self.ws_log(entry);
    }

    /// A finished job's own clip, so its export and translation can be logged.
    pub(crate) fn ws_clip_of(&self, job: u64) -> Option<u64> {
        self.job(job).and_then(|j| j.clip)
    }

    pub(crate) fn ws_job_removed(&mut self, job: &Job) {
        let Some(clip) = job.clip else {
            return;
        };
        self.ws_snapshot("Remove a recording");
        if self.ws_open().data.remove_clip(clip) {
            self.ws_log(Entry::new(Kind::ClipRemoved, job.name()).clip(clip));
            self.ws_persist();
        }
    }

    pub(crate) fn ws_relinked(&mut self, job: u64, media: &Path) {
        let Some(clip) = self.ws_clip_of(job) else {
            return;
        };
        self.ws_snapshot("Find a recording");
        if self.ws_open().data.relink(clip, media) {
            self.ws_log(Entry::new(Kind::ClipRelinked, crate::file_name(media)).clip(clip));
            self.ws_persist();
        }
    }

    /// Before the window closes: `Some` when it must wait for an answer about unsaved changes.
    pub(crate) fn ws_before_close(&mut self) -> Option<Task<Message>> {
        if self.workspace.close_confirmed {
            return None;
        }
        match &self.workspace.open {
            Some(open) if open.dirty && !open.temporary() => {
                self.workspace.asking = Some(After::Close);
                Some(Task::none())
            }
            Some(open) if open.temporary() && open.data.clips.is_empty() => {
                let _ = workspace::remove_temporary(&open.path, &self.workspace.root());
                None
            }
            _ => None,
        }
    }

    /// Puts the job list in the project's order, drops jobs whose clip is gone, and loads the clips
    /// that have no job yet.
    fn ws_sync_jobs(&mut self) -> Task<Message> {
        let Some(open) = &self.workspace.open else {
            return Task::none();
        };
        let order: Vec<u64> = open.data.clips.iter().map(|c| c.id).collect();
        let running = self.running;
        self.jobs
            .retain(|j| Some(j.id) == running || j.clip.is_some_and(|c| order.contains(&c)));
        let position = |j: &Job| {
            j.clip
                .and_then(|c| order.iter().position(|o| *o == c))
                .unwrap_or(usize::MAX)
        };
        self.jobs.sort_by_key(position);
        // results the jobs know but an older snapshot doesn't
        let known: Vec<(u64, ResultRef)> = self
            .jobs
            .iter()
            .filter_map(|j| Some((j.clip?, ResultRef::from(j.paths.as_ref()?))))
            .collect();
        let open = self.workspace.open.as_mut().expect("checked above");
        for (clip, result) in known {
            open.data.set_result(clip, result);
        }
        let have: HashSet<u64> = self.jobs.iter().filter_map(|j| j.clip).collect();
        let missing: Vec<u64> = order.into_iter().filter(|c| !have.contains(c)).collect();
        if missing.is_empty() {
            return Task::none();
        }
        let project = (!open.temporary()).then(|| open.path.clone());
        let data = open.data.clone();
        let generation = self.workspace.generation;
        Task::perform(
            import_ui::on_thread(move || Ok(load_clips(project, data, missing))),
            move |loads: Result<Vec<ClipLoad>, String>| {
                Message::Workspace(Msg::ClipsLoaded(generation, loads.unwrap_or_default()))
            },
        )
    }

    fn ws_add_loaded(&mut self, loads: Vec<ClipLoad>) {
        for load in loads {
            let Some(clip) = self
                .workspace
                .open
                .as_ref()
                .and_then(|o| o.data.clip(load.clip).cloned())
            else {
                continue;
            };
            let id = match load.result {
                Some(Ok(mut loaded)) => {
                    // the project's idea of where the media is wins over the result folder's
                    if load.media.is_some() {
                        loaded.media = load.media.clone();
                    }
                    self.add_loaded_result(loaded)
                }
                other => {
                    let path = load
                        .media
                        .clone()
                        .unwrap_or_else(|| clip.media.path.clone());
                    let id = self.next_id;
                    self.next_id += 1;
                    let mut job = Job::new(id, path, clip.media.size);
                    if load.media.is_none() {
                        job.media_missing = true;
                        job.state = JobState::Failed;
                        job.error = Some(
                            "The recording is not where it was. Find it to transcribe it.".into(),
                        );
                    } else if let Some(Err(e)) = other {
                        job.notes.push(format!(
                            "Its earlier results could not be read ({e}), so it runs again."
                        ));
                    }
                    self.jobs.push(job);
                    id
                }
            };
            if let Some(j) = self.job_mut(id) {
                j.clip = Some(load.clip);
            }
        }
        let _ = self.ws_sync_jobs();
    }

    fn ws_add_recent(&mut self, path: &Path) {
        let recent = &mut self.workspace.recent;
        recent.retain(|p| p != path);
        recent.insert(0, path.to_path_buf());
        recent.truncate(RECENT_MAX);
        if let Some(file) = recent_file() {
            let list = recent.clone();
            saver::run(move || {
                let _ = engine::fsio::save_json(&file, &list);
            });
        }
    }

    /// Leaves the current project: an unsaved one without clips is deleted, one with clips stays to
    /// be restored later.
    fn ws_leave(&mut self) {
        let root = self.workspace.root();
        if let Some(open) = self.workspace.open.take()
            && open.temporary()
            && open.data.clips.is_empty()
        {
            let _ = workspace::remove_temporary(&open.path, &root);
        }
        self.workspace.undo.clear();
        self.workspace.history.clear();
        self.workspace.generation += 1;
        let running = self.running;
        self.jobs.retain(|j| Some(j.id) == running);
        if let Some(id) = running
            && let Some(j) = self.job_mut(id)
        {
            j.clip = None; // it belongs to the project that was left
        }
    }

    /// Whether `after` needs the user to say what happens to unsaved changes first.
    fn ws_ask_first(&mut self, after: After) -> bool {
        let dirty = self
            .workspace
            .open
            .as_ref()
            .is_some_and(|o| o.dirty && !o.temporary());
        if dirty {
            self.workspace.asking = Some(after);
        }
        dirty
    }

    fn ws_continue(&mut self, after: After) -> Task<Message> {
        match after {
            After::Close => {
                self.workspace.close_confirmed = true;
                self.update(Message::CloseWindow)
            }
            After::New => {
                self.ws_leave();
                self.page = crate::Page::Home;
                Task::none()
            }
            After::Open(path) => Task::perform(
                import_ui::on_thread(move || {
                    Workspace::load(&path)
                        .map(|w| (path.clone(), w))
                        .map_err(|e| e.to_string())
                }),
                |r| Message::Workspace(Msg::Loaded(r)),
            ),
        }
    }

    fn ws_save_to(&mut self, path: PathBuf) -> Result<(), String> {
        let root = self.workspace.root();
        if workspace::is_inside(&path, &root) {
            return Err(
                "that folder is where ViPr keeps unsaved projects. Choose another place.".into(),
            );
        }
        let Some(open) = self.workspace.open.as_mut() else {
            return Ok(());
        };
        let was_temporary = open.temporary();
        let old = open.path.clone();
        let mut data = open.data.clone();
        if path != old || data.name.is_empty() {
            data.name = file_stem(&path);
        }
        data.save(&path).map_err(|e| e.to_string())?;
        open.data = data;
        if path != old {
            let _ = open.log.move_to(&workspace::history_path(&path));
            open.path = path.clone();
            open.temporary = false;
            if was_temporary {
                let _ = workspace::remove_temporary(&old, &root);
            }
        }
        open.dirty = false;
        self.ws_add_recent(&path);
        self.ws_log(Entry::new(Kind::Saved, file_stem(&path)));
        Ok(())
    }

    pub(crate) fn ws_update(&mut self, msg: Msg) -> Task<Message> {
        match msg {
            Msg::New => {
                if self.running.is_some() {
                    self.tell("Finish or cancel the transcription that is running first.");
                } else if !self.ws_ask_first(After::New) {
                    return self.ws_continue(After::New);
                }
                Task::none()
            }
            Msg::Open => Task::perform(pick_project(), |p| Message::Workspace(Msg::Picked(p))),
            Msg::OpenRecent(Recent(path)) => self.ws_update(Msg::Picked(Some(path))),
            Msg::Picked(None) => Task::none(),
            Msg::Picked(Some(path)) => {
                if self.running.is_some() {
                    self.tell("Finish or cancel the transcription that is running first.");
                    return Task::none();
                }
                if self.workspace.open.as_ref().is_some_and(|o| o.path == path) {
                    return Task::none();
                }
                if self.ws_ask_first(After::Open(path.clone())) {
                    return Task::none();
                }
                self.ws_continue(After::Open(path))
            }
            Msg::Loaded(Err(e)) => {
                self.alert(&format!("That project could not be opened: {e}"));
                Task::none()
            }
            Msg::Loaded(Ok((path, data))) => {
                self.ws_leave();
                let temporary = workspace::is_temporary(&path, &self.workspace.root());
                let log = HistoryLog::new(workspace::history_path(&path));
                self.workspace.history = log.tail(HISTORY_SHOWN);
                self.workspace.open = Some(Open {
                    path: path.clone(),
                    data,
                    dirty: false,
                    temporary,
                    log,
                    last_backup: None,
                });
                if self
                    .workspace
                    .restorable
                    .as_ref()
                    .is_some_and(|(p, _)| *p == path)
                {
                    self.workspace.restorable = None;
                }
                if !temporary {
                    self.ws_add_recent(&path);
                }
                self.ws_log(Entry::new(Kind::Opened, file_stem(&path)));
                self.page = crate::Page::Home;
                self.ws_sync_jobs()
            }
            Msg::ClipsLoaded(generation, loads) => {
                if generation == self.workspace.generation {
                    self.ws_add_loaded(loads);
                }
                Task::none()
            }
            Msg::Save => match &self.workspace.open {
                None => Task::none(),
                Some(open) if open.temporary() => self.ws_update(Msg::SaveAs),
                Some(open) => {
                    let path = open.path.clone();
                    if let Err(e) = self.ws_save_to(path) {
                        self.alert(&format!("The project could not be saved: {e}"));
                    }
                    Task::none()
                }
            },
            Msg::SaveAs => {
                let open = self.ws_open();
                let name = if open.temporary() {
                    "Project".to_string()
                } else {
                    file_stem(&open.path)
                };
                Task::perform(save_project_as(name), |p| {
                    Message::Workspace(Msg::SaveAsPicked(p))
                })
            }
            Msg::SaveAsPicked(None) => {
                self.workspace.asking = None;
                Task::none()
            }
            Msg::SaveAsPicked(Some(path)) => {
                let path = if path.extension().is_some_and(|e| e == workspace::EXTENSION) {
                    path
                } else {
                    let mut p = path.into_os_string();
                    p.push(format!(".{}", workspace::EXTENSION));
                    PathBuf::from(p)
                };
                if let Err(e) = self.ws_save_to(path) {
                    self.alert(&format!("The project could not be saved: {e}"));
                    return Task::none();
                }
                match self.workspace.asking.take() {
                    Some(after) => self.ws_continue(after),
                    None => Task::none(),
                }
            }
            Msg::Restore => match self.workspace.restorable.clone() {
                Some((path, _)) => self.ws_update(Msg::Picked(Some(path))),
                None => Task::none(),
            },
            Msg::Discard => {
                if let Some((path, _)) = self.workspace.restorable.take() {
                    let _ = workspace::remove_temporary(&path, &self.workspace.root());
                }
                Task::none()
            }
            Msg::Move(job, by) => {
                let Some(clip) = self.ws_clip_of(job) else {
                    return Task::none();
                };
                self.ws_snapshot("Change the order");
                if self.ws_open().data.move_clip(clip, by) {
                    self.ws_log(Entry::new(Kind::Reordered, "").clip(clip));
                    self.ws_persist();
                    return self.ws_sync_jobs();
                }
                Task::none()
            }
            Msg::Undo | Msg::Redo => {
                let redo = matches!(msg, Msg::Redo);
                if self.workspace.open.is_none() {
                    return Task::none();
                }
                let current = self.ws_open().data.clone();
                let step = if redo {
                    self.workspace.undo.redo(current)
                } else {
                    self.workspace.undo.undo(current)
                };
                let Some((what, state)) = step else {
                    return Task::none();
                };
                self.ws_open().data = state;
                self.workspace.renaming = None;
                let kind = if redo { Kind::Redone } else { Kind::Undone };
                self.ws_log(Entry::new(kind, what));
                self.ws_persist();
                self.ws_sync_jobs()
            }
            Msg::SeqNew => {
                let open = self.ws_open();
                let n = open.data.sequences.len() + 1;
                let clips: Vec<u64> = open.data.clips.iter().map(|c| c.id).collect();
                self.ws_snapshot("New combination");
                let data = &mut self.ws_open().data;
                let id = data.add_sequence(format!("Combination {n}"));
                for clip in clips {
                    data.push_to_sequence(id, clip);
                }
                self.ws_log(Entry::new(Kind::SequenceAdded, format!("Combination {n}")));
                self.ws_persist();
                Task::none()
            }
            Msg::SeqRename(id, name) => {
                // typing one name is one Undo step, not one per letter
                let name = crate::clean_separators(&name);
                let unchanged = self
                    .ws_open()
                    .data
                    .sequence(id)
                    .is_none_or(|s| s.name == name);
                if unchanged {
                    return Task::none();
                }
                if self.workspace.renaming != Some(id) {
                    let old = self
                        .ws_open()
                        .data
                        .sequence(id)
                        .map(|s| s.name.clone())
                        .unwrap_or_default();
                    self.ws_snapshot("Rename a combination");
                    self.workspace.renaming = Some(id);
                    self.ws_log(Entry::new(Kind::SequenceChanged, format!("renamed {old}")));
                }
                if let Some(seq) = self.ws_open().data.sequence_mut(id) {
                    seq.name = name;
                }
                self.ws_persist();
                Task::none()
            }
            Msg::SeqAdd(id, choice) => self.ws_seq_change(id, "Add to a combination", |d| {
                d.push_to_sequence(id, choice.id)
            }),
            Msg::SeqMove(id, at, by) => {
                self.ws_seq_change(id, "Change a combination's order", |d| {
                    d.move_in_sequence(id, at, by)
                })
            }
            Msg::SeqRemoveItem(id, at) => {
                self.ws_seq_change(id, "Take out of a combination", |d| {
                    d.remove_from_sequence(id, at)
                })
            }
            Msg::SeqDelete(id) => {
                let name = self
                    .ws_open()
                    .data
                    .sequence(id)
                    .map(|s| s.name.clone())
                    .unwrap_or_default();
                self.ws_snapshot("Delete a combination");
                if self.ws_open().data.remove_sequence(id) {
                    self.ws_log(Entry::new(Kind::SequenceRemoved, name));
                    self.ws_persist();
                }
                Task::none()
            }
            Msg::SeqExport(id) => {
                self.ws_export_sequence(id);
                Task::none()
            }
            Msg::ToggleHistory => {
                self.workspace.history_open = !self.workspace.history_open;
                Task::none()
            }
            Msg::AnswerSave => {
                let Some(after) = self.workspace.asking.clone() else {
                    return Task::none();
                };
                match &self.workspace.open {
                    Some(open) if !open.temporary() => {
                        let path = open.path.clone();
                        self.workspace.asking = None;
                        if let Err(e) = self.ws_save_to(path) {
                            self.alert(&format!("The project could not be saved: {e}"));
                            return Task::none();
                        }
                        self.ws_continue(after)
                    }
                    _ => self.ws_update(Msg::SaveAs), // continues after the file is chosen
                }
            }
            Msg::AnswerDiscard => match self.workspace.asking.take() {
                Some(after) => {
                    if let Some(open) = self.workspace.open.as_mut() {
                        open.dirty = false;
                    }
                    self.ws_continue(after)
                }
                None => Task::none(),
            },
            Msg::AnswerCancel => {
                self.workspace.asking = None;
                Task::none()
            }
        }
    }

    /// Keyboard shortcuts for the project. `None` when the key isn't one.
    pub(crate) fn ws_key(
        &self,
        key: &iced::keyboard::Key,
        mods: iced::keyboard::Modifiers,
        on_home: bool,
    ) -> Option<Message> {
        use iced::keyboard::Key;
        if !mods.command() {
            return None;
        }
        let Key::Character(c) = key else {
            return None;
        };
        let msg = match (c.to_lowercase().as_str(), mods.shift()) {
            ("s", false) => Msg::Save,
            ("s", true) => Msg::SaveAs,
            ("o", _) => Msg::Open,
            ("n", _) => Msg::New,
            ("z", false) if on_home => Msg::Undo,
            ("z", true) | ("y", _) if on_home => Msg::Redo,
            _ => return None,
        };
        Some(Message::Workspace(msg))
    }

    /// The project's name for the title bar.
    pub(crate) fn ws_title(&self) -> Option<String> {
        self.workspace.open.as_ref().map(Open::title)
    }

    /// The bar above the recordings on Home: the project, and what can be done with it.
    pub(crate) fn ws_bar(&self, t: Tokens) -> Element<'_, Message> {
        let ws = &self.workspace;
        let small = |label: &'static str, msg: Option<Msg>| {
            button(text(label).size(12))
                .padding([5, 10])
                .style(theme::ghost(t))
                .on_press_maybe(msg.map(Message::Workspace))
        };
        let name = ws
            .open
            .as_ref()
            .map_or_else(|| "No project yet".to_string(), Open::title);
        let has_clips = ws.open.as_ref().is_some_and(|o| !o.data.clips.is_empty());
        let can_save = ws
            .open
            .as_ref()
            .is_some_and(|o| o.dirty || (o.temporary() && has_clips));
        let mut bar = row![
            icons::icon(icons::FOLDER, 15.0, t.muted),
            text(crate::clean_separators(&name))
                .size(13)
                .font(semibold())
                .color(t.text),
            iced::widget::space::horizontal(),
            small("New", Some(Msg::New)),
            small("Open...", Some(Msg::Open)),
        ]
        .spacing(6)
        .align_y(Alignment::Center);
        if !ws.recent.is_empty() {
            let recent: Vec<Recent> = ws.recent.iter().cloned().map(Recent).collect();
            bar = bar.push(
                pick_list(recent, None::<Recent>, |r| {
                    Message::Workspace(Msg::OpenRecent(r))
                })
                .placeholder("Recent")
                .text_size(12)
                .padding([4, 8])
                .style(theme::pick(t)),
            );
        }
        bar = bar
            .push(small("Save", can_save.then_some(Msg::Save)))
            .push(small("Save as...", has_clips.then_some(Msg::SaveAs)))
            .push(
                button(text("Undo").size(12))
                    .padding([5, 10])
                    .style(theme::ghost(t))
                    .on_press_maybe(ws.undo.next_undo().map(|_| Message::Workspace(Msg::Undo))),
            )
            .push(
                button(text("Redo").size(12))
                    .padding([5, 10])
                    .style(theme::ghost(t))
                    .on_press_maybe(ws.undo.next_redo().map(|_| Message::Workspace(Msg::Redo))),
            )
            .push(small(
                if ws.history_open {
                    "Hide history"
                } else {
                    "History"
                },
                ws.open.is_some().then_some(Msg::ToggleHistory),
            ));

        let mut col = column![bar].spacing(10);
        if let Some((_, clips)) = &ws.restorable {
            col = col.push(
                container(
                    row![
                        icons::icon(icons::INFO, 18.0, t.accent),
                        text(format!(
                            "A project you didn't save last time has {clips} recording{}.",
                            if *clips == 1 { "" } else { "s" }
                        ))
                        .size(13)
                        .color(t.text)
                        .width(Length::Fill),
                        button(text("Restore").size(12))
                            .padding([6, 12])
                            .style(theme::primary(t))
                            .on_press(Message::Workspace(Msg::Restore)),
                        button(text("Discard").size(12))
                            .padding([6, 12])
                            .style(theme::ghost(t))
                            .on_press(Message::Workspace(Msg::Discard)),
                    ]
                    .spacing(10)
                    .align_y(Alignment::Center),
                )
                .padding(12)
                .width(Length::Fill)
                .style(theme::card(t)),
            );
        }
        if ws.history_open {
            col = col.push(self.ws_history(t));
        }
        col.into()
    }

    fn ws_history(&self, t: Tokens) -> Element<'_, Message> {
        let mut list = column![].spacing(6);
        if self.workspace.history.is_empty() {
            list = list.push(
                text("Nothing has happened in this project yet.")
                    .size(12)
                    .color(t.muted),
            );
        }
        for e in self.workspace.history.iter().rev() {
            let what = match e.detail.trim() {
                "" => e.kind.label().to_string(),
                d => format!("{}: {}", e.kind.label(), crate::clean_separators(d)),
            };
            let mut line = row![text(what).size(12).color(t.text).width(Length::Fill),].spacing(10);
            let numbers = metrics_text(e);
            if !numbers.is_empty() {
                line = line.push(text(numbers).size(12).color(t.muted));
            }
            line = line.push(text(ago(e.t)).size(11).color(t.faint));
            list = list.push(line);
        }
        container(scrollable(list).height(Length::Shrink))
            .max_height(260)
            .padding(12)
            .width(Length::Fill)
            .style(theme::card(t))
            .into()
    }

    fn ws_seq_change(
        &mut self,
        id: u64,
        what: &str,
        change: impl FnOnce(&mut Workspace) -> bool,
    ) -> Task<Message> {
        let before = self.ws_open().data.clone();
        if change(&mut self.ws_open().data) {
            self.workspace.undo.record(what, before);
            self.workspace.renaming = None;
            let name = self
                .ws_open()
                .data
                .sequence(id)
                .map(|s| s.name.clone())
                .unwrap_or_default();
            self.ws_log(Entry::new(Kind::SequenceChanged, name));
            self.ws_persist();
        }
        Task::none()
    }

    /// The finished job of each item of a combination, or the title of the first that isn't ready.
    fn ws_sequence_jobs(&self, id: u64) -> Result<Vec<u64>, String> {
        let open = self.workspace.open.as_ref().ok_or_else(String::new)?;
        let seq = open.data.sequence(id).ok_or_else(String::new)?;
        seq.items
            .iter()
            .map(|clip| {
                let title = open.data.clip(*clip).map(|c| c.title()).unwrap_or_default();
                self.jobs
                    .iter()
                    .find(|j| {
                        j.clip == Some(*clip) && j.state == JobState::Done && j.paths.is_some()
                    })
                    .map(|j| j.id)
                    .ok_or(title)
            })
            .collect()
    }

    fn ws_export_sequence(&mut self, id: u64) {
        let Some(name) = self
            .workspace
            .open
            .as_ref()
            .and_then(|o| o.data.sequence(id))
            .map(|s| s.name.clone())
        else {
            return;
        };
        let jobs = match self.ws_sequence_jobs(id) {
            Ok(jobs) if !jobs.is_empty() => jobs,
            Ok(_) => {
                self.tell("Add recordings to the combination first.");
                return;
            }
            Err(title) => {
                self.tell(&format!(
                    "Every recording in a combination has to be transcribed first, and {title} isn't yet."
                ));
                return;
            }
        };
        let items: Vec<crate::JobRef> = jobs.iter().filter_map(|j| self.job_ref(*j)).collect();
        if let Some(gone) = items.iter().find(|i| !i.media.is_file()) {
            self.alert(&format!(
                "The recording {} is not where it was. Find it first.",
                crate::file_name(&gone.media)
            ));
            return;
        }
        let durations: Vec<f64> = jobs
            .iter()
            .map(|j| {
                self.job(*j)
                    .and_then(|j| j.transcript.as_ref())
                    .map_or(0.0, |t| t.audio_seconds)
            })
            .collect();
        let gpu = self
            .detection
            .as_ref()
            .is_none_or(|d| d.recommended == gpu_detect::Backend::Cuda);
        let style = self.settings.subtitle_style.clone();
        export_ui::open_sequence(
            &mut self.export,
            name,
            items,
            &durations,
            (None, style),
            gpu,
        );
    }

    /// The combinations on Home, below the recordings. `None` while the project has no recordings.
    pub(crate) fn ws_combinations(&self, t: Tokens) -> Option<Element<'_, Message>> {
        let open = self.workspace.open.as_ref()?;
        if open.data.clips.is_empty() {
            return None;
        }
        let data = &open.data;
        let small = |label: &'static str, msg: Option<Msg>| {
            button(text(label).size(12))
                .padding([5, 10])
                .style(theme::ghost(t))
                .on_press_maybe(msg.map(Message::Workspace))
        };
        let icon_btn = |icon: &'static str, msg: Option<Msg>| {
            button(icons::icon(
                icon,
                13.0,
                if msg.is_some() { t.muted } else { t.faint },
            ))
            .padding(5)
            .style(theme::ghost(t))
            .on_press_maybe(msg.map(Message::Workspace))
        };
        let mut col = column![
            row![
                column![
                    text("Combinations").size(15).font(semibold()).color(t.text),
                    text("Join recordings into one video, one after another. Each keeps its own cuts.")
                        .size(12)
                        .color(t.muted),
                ]
                .spacing(2)
                .width(Length::Fill),
                button(text("New combination").size(13))
                    .padding([7, 12])
                    .style(theme::outline_btn(t))
                    .on_press(Message::Workspace(Msg::SeqNew)),
            ]
            .spacing(12)
            .align_y(Alignment::Center)
        ]
        .spacing(12);
        let choices: Vec<ClipChoice> = data
            .clips
            .iter()
            .map(|c| ClipChoice {
                id: c.id,
                title: c.title(),
            })
            .collect();
        for seq in &data.sequences {
            let id = seq.id;
            let ready = self.ws_sequence_jobs(id).is_ok_and(|j| !j.is_empty());
            let mut card = column![
                row![
                    text_input("Name", &seq.name)
                        .on_input(move |v| Message::Workspace(Msg::SeqRename(id, v)))
                        .style(theme::input(t))
                        .padding(7)
                        .size(13)
                        .width(Length::Fill),
                    button(text("Export...").size(13))
                        .padding([7, 12])
                        .style(theme::primary(t))
                        .on_press_maybe(ready.then_some(Message::Workspace(Msg::SeqExport(id)))),
                    small("Delete", Some(Msg::SeqDelete(id))),
                ]
                .spacing(8)
                .align_y(Alignment::Center)
            ]
            .spacing(6);
            let mut length = 0.0;
            for (at, clip) in seq.items.iter().enumerate() {
                let title = data.clip(*clip).map(|c| c.title()).unwrap_or_default();
                let job = self.jobs.iter().find(|j| j.clip == Some(*clip));
                let state = match job.map(|j| (&j.state, j.transcript.as_ref())) {
                    Some((JobState::Done, Some(t))) => {
                        length += t.audio_seconds;
                        crate::clock(t.audio_seconds)
                    }
                    _ => "not transcribed yet".to_string(),
                };
                card = card.push(
                    row![
                        text(format!("{}.", at + 1)).size(12).color(t.faint),
                        text(crate::clean_separators(&title))
                            .size(13)
                            .color(t.text)
                            .width(Length::Fill),
                        text(state).size(12).color(t.muted),
                        icon_btn(
                            icons::CHEVRON_UP,
                            (at > 0).then_some(Msg::SeqMove(id, at, -1))
                        ),
                        icon_btn(
                            icons::CHEVRON_DOWN,
                            (at + 1 < seq.items.len()).then_some(Msg::SeqMove(id, at, 1))
                        ),
                        icon_btn(icons::X, Some(Msg::SeqRemoveItem(id, at))),
                    ]
                    .spacing(8)
                    .align_y(Alignment::Center),
                );
            }
            if seq.items.is_empty() {
                card = card.push(text("No recordings yet.").size(12).color(t.muted));
            }
            card = card.push(
                row![
                    pick_list(choices.clone(), None::<ClipChoice>, move |c| {
                        Message::Workspace(Msg::SeqAdd(id, c))
                    })
                    .placeholder("Add a recording")
                    .text_size(12)
                    .padding([5, 9])
                    .style(theme::pick(t)),
                    iced::widget::space::horizontal(),
                    text(if length > 0.0 {
                        format!("{} before cuts", crate::clock(length))
                    } else {
                        String::new()
                    })
                    .size(12)
                    .color(t.muted),
                ]
                .align_y(Alignment::Center),
            );
            col = col.push(
                container(card)
                    .padding(12)
                    .width(Length::Fill)
                    .style(theme::card(t)),
            );
        }
        Some(col.into())
    }

    /// Up / down buttons for a recording's card.
    pub(crate) fn ws_move_buttons(&self, job: u64, t: Tokens) -> Option<Element<'_, Message>> {
        let open = self.workspace.open.as_ref()?;
        let clip = self.ws_clip_of(job)?;
        if open.data.clips.len() < 2 {
            return None;
        }
        let at = open.data.clips.iter().position(|c| c.id == clip)?;
        let arrow = |icon: &'static str, by: isize, enabled: bool| {
            button(icons::icon(
                icon,
                14.0,
                if enabled { t.muted } else { t.faint },
            ))
            .padding(6)
            .style(theme::ghost(t))
            .on_press_maybe(enabled.then_some(Message::Workspace(Msg::Move(job, by))))
        };
        Some(
            column![
                arrow(icons::CHEVRON_UP, -1, at > 0),
                arrow(icons::CHEVRON_DOWN, 1, at + 1 < open.data.clips.len()),
            ]
            .into(),
        )
    }

    /// "Save changes?" over everything, while it is being asked.
    pub(crate) fn ws_question(&self, t: Tokens) -> Option<Element<'_, Message>> {
        let after = self.workspace.asking.as_ref()?;
        let name = self
            .workspace
            .open
            .as_ref()
            .map_or_else(String::new, |o| file_stem(&o.path));
        let then = match after {
            After::Close => "before closing",
            After::New => "before starting a new project",
            After::Open(_) => "before opening another project",
        };
        Some(
            container(
                column![
                    text(format!(
                        "Save the changes to {}?",
                        crate::clean_separators(&name)
                    ))
                    .size(16)
                    .font(semibold())
                    .color(t.text),
                    text(format!(
                        "The project has changes that are not saved yet, {then}."
                    ))
                    .size(13)
                    .color(t.muted),
                    row![
                        iced::widget::space::horizontal(),
                        button(text("Cancel").size(13))
                            .padding([8, 14])
                            .style(theme::ghost(t))
                            .on_press(Message::Workspace(Msg::AnswerCancel)),
                        button(text("Don't save").size(13))
                            .padding([8, 14])
                            .style(theme::outline_btn(t))
                            .on_press(Message::Workspace(Msg::AnswerDiscard)),
                        button(text("Save").size(13))
                            .padding([8, 14])
                            .style(theme::primary(t))
                            .on_press(Message::Workspace(Msg::AnswerSave)),
                    ]
                    .spacing(8),
                ]
                .spacing(12),
            )
            .padding(20)
            .max_width(460)
            .style(theme::card(t))
            .into(),
        )
    }
}

async fn pick_project() -> Option<PathBuf> {
    rfd::AsyncFileDialog::new()
        .set_title("Open a project")
        .add_filter("ViPr project", &[workspace::EXTENSION])
        .pick_file()
        .await
        .map(|f| f.path().to_path_buf())
}

async fn save_project_as(name: String) -> Option<PathBuf> {
    rfd::AsyncFileDialog::new()
        .set_title("Save the project")
        .add_filter("ViPr project", &[workspace::EXTENSION])
        .set_file_name(format!("{name}.{}", workspace::EXTENSION))
        .save_file()
        .await
        .map(|f| f.path().to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh folder under the system temp dir, removed when dropped.
    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> TestDir {
            let dir = std::env::temp_dir().join(format!(
                "vipr-wsui-{name}-{}-{:?}",
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

    fn app_in(dir: &TestDir) -> App {
        let (mut app, _) = App::new();
        app.workspace.temp_root = Some(dir.0.join("unsaved"));
        app
    }

    /// Job paths in list order, as file names.
    fn order(app: &App) -> Vec<String> {
        app.jobs.iter().map(Job::name).collect()
    }

    /// Loads the clips that have no job, as the background task would.
    fn finish_loading(app: &mut App) {
        let open = app.workspace.open.as_ref().unwrap();
        let have: HashSet<u64> = app.jobs.iter().filter_map(|j| j.clip).collect();
        let missing: Vec<u64> = open
            .data
            .clips
            .iter()
            .map(|c| c.id)
            .filter(|c| !have.contains(c))
            .collect();
        let project = (!open.temporary()).then(|| open.path.clone());
        let loads = load_clips(project, open.data.clone(), missing);
        app.ws_add_loaded(loads);
    }

    #[test]
    fn a_dropped_file_starts_an_unsaved_project_that_saves_itself() {
        let dir = TestDir::new("temp");
        let mut app = app_in(&dir);
        let media = dir.0.join("a.mp4");
        std::fs::write(&media, b"x").unwrap();
        app.jobs.push(Job::new(1, media.clone(), 1));
        app.ws_jobs_added(&[1]);
        saver::wait();
        let open = app.workspace.open.as_ref().unwrap();
        assert!(open.temporary() && !open.dirty);
        assert!(open.path.starts_with(dir.0.join("unsaved")));
        let saved = Workspace::load(&open.path).unwrap();
        assert_eq!(saved.clips.len(), 1);
        let log = HistoryLog::new(workspace::history_path(&open.path)).tail(10);
        assert_eq!(
            log.iter().map(|e| e.kind).collect::<Vec<_>>(),
            [Kind::Created, Kind::ClipAdded]
        );
        assert_eq!(app.jobs[0].clip, Some(saved.clips[0].id));
        // an unsaved project never asks before closing
        assert!(app.ws_before_close().is_none());
    }

    #[test]
    fn a_project_opens_reorders_undoes_and_saves() {
        let dir = TestDir::new("named");
        let file = dir.0.join("Meeting.vipr");
        let mut w = Workspace::new("Meeting");
        for name in ["a.mp4", "b.mp4"] {
            let media = dir.0.join("media").join(name);
            std::fs::create_dir_all(media.parent().unwrap()).unwrap();
            std::fs::write(&media, b"x").unwrap();
            w.add_clip(&media, 1);
        }
        w.save(&file).unwrap();

        let mut app = app_in(&dir);
        let _ = app.ws_update(Msg::Loaded(Ok((
            file.clone(),
            Workspace::load(&file).unwrap(),
        ))));
        finish_loading(&mut app);
        assert_eq!(order(&app), ["a.mp4", "b.mp4"]);
        assert_eq!(app.ws_title().as_deref(), Some("Meeting"));

        let b = app.jobs[1].id;
        let _ = app.ws_update(Msg::Move(b, -1));
        assert_eq!(order(&app), ["b.mp4", "a.mp4"]);
        assert_eq!(app.ws_title().as_deref(), Some("Meeting •"));
        let _ = app.ws_update(Msg::Undo);
        assert_eq!(order(&app), ["a.mp4", "b.mp4"]);
        let _ = app.ws_update(Msg::Redo);
        assert_eq!(order(&app), ["b.mp4", "a.mp4"]);

        // removing a recording, and undoing that, brings its job back
        let a = app.jobs[1].id;
        let _ = app.update(Message::Remove(a));
        assert_eq!(order(&app), ["b.mp4"]);
        let _ = app.ws_update(Msg::Undo);
        finish_loading(&mut app);
        assert_eq!(order(&app), ["b.mp4", "a.mp4"]);

        // closing with unsaved changes asks first; Save writes the file and the question goes
        assert!(app.ws_before_close().is_some());
        assert!(app.workspace.is_asking());
        app.workspace.asking = None;
        let _ = app.ws_update(Msg::Save);
        saver::wait();
        assert_eq!(app.ws_title().as_deref(), Some("Meeting"));
        let saved = Workspace::load(&file).unwrap();
        let names: Vec<String> = saved.clips.iter().map(|c| c.title()).collect();
        assert_eq!(names, ["b.mp4", "a.mp4"]);
        assert!(app.ws_before_close().is_none());

        let kinds: Vec<Kind> = HistoryLog::new(workspace::history_path(&file))
            .tail(20)
            .iter()
            .map(|e| e.kind)
            .collect();
        assert_eq!(
            kinds,
            [
                Kind::Opened,
                Kind::Reordered,
                Kind::Undone,
                Kind::Redone,
                Kind::ClipRemoved,
                Kind::Undone,
                Kind::Saved
            ]
        );
    }

    #[test]
    fn a_users_project_in_a_folder_named_untitled_is_never_deleted() {
        let dir = TestDir::new("lookalike");
        let theirs = dir.0.join("untitled-trip").join("untitled.vipr");
        Workspace::new("").save(&theirs).unwrap(); // no clips, like an empty unsaved project
        let mut app = app_in(&dir);
        let _ = app.ws_update(Msg::Loaded(Ok((
            theirs.clone(),
            Workspace::load(&theirs).unwrap(),
        ))));
        assert!(!app.workspace.open.as_ref().unwrap().temporary());
        let _ = app.ws_update(Msg::New); // leaves the project
        assert!(app.ws_before_close().is_none());
        assert!(theirs.exists(), "the user's folder is still there");

        // and Save As can't put a project where ViPr keeps its unsaved ones
        app.jobs.push(Job::new(1, dir.0.join("a.mp4"), 1));
        app.ws_jobs_added(&[1]);
        saver::wait();
        let temp_dir = app
            .workspace
            .open
            .as_ref()
            .unwrap()
            .path
            .parent()
            .unwrap()
            .to_path_buf();
        let err = app.ws_save_to(temp_dir.join("Mine.vipr")).unwrap_err();
        assert!(err.contains("unsaved projects"), "{err}");
        assert!(app.workspace.open.as_ref().unwrap().temporary());
    }

    #[test]
    fn combinations_are_made_changed_undone_and_wait_for_transcripts() {
        let dir = TestDir::new("combine");
        let mut app = app_in(&dir);
        for (id, name) in [(1, "a.mp4"), (2, "b.mp4")] {
            let media = dir.0.join(name);
            std::fs::write(&media, b"x").unwrap();
            app.jobs.push(Job::new(id, media, 1));
        }
        app.ws_jobs_added(&[1, 2]);
        let _ = app.ws_update(Msg::SeqNew);
        let data = |app: &App| app.workspace.open.as_ref().unwrap().data.clone();
        let seq = data(&app).sequences[0].clone();
        assert_eq!(seq.name, "Combination 1");
        assert_eq!(seq.items.len(), 2, "starts with every recording, in order");

        let _ = app.ws_update(Msg::SeqMove(seq.id, 1, -1));
        let _ = app.ws_update(Msg::SeqRemoveItem(seq.id, 1));
        assert_eq!(data(&app).sequences[0].items, [seq.items[1]]);
        let _ = app.ws_update(Msg::Undo);
        assert_eq!(data(&app).sequences[0].items, [seq.items[1], seq.items[0]]);
        // typing a name letter by letter is one step for Undo
        for typed in ["W", "We", "Week", "Week 12"] {
            let _ = app.ws_update(Msg::SeqRename(seq.id, typed.into()));
        }
        assert_eq!(data(&app).sequences[0].name, "Week 12");
        let _ = app.ws_update(Msg::Undo);
        assert_eq!(
            data(&app).sequences[0].name,
            "Combination 1",
            "the whole name at once"
        );
        let _ = app.ws_update(Msg::Redo);
        assert_eq!(data(&app).sequences[0].name, "Week 12");
        // a new name after something else is a step of its own
        let _ = app.ws_update(Msg::SeqMove(seq.id, 1, -1));
        let _ = app.ws_update(Msg::SeqRename(seq.id, "Week 13".into()));
        let _ = app.ws_update(Msg::Undo);
        assert_eq!(data(&app).sequences[0].name, "Week 12");
        let _ = app.ws_update(Msg::Undo);
        assert_eq!(
            data(&app).sequences[0].items,
            [seq.items[1], seq.items[0]],
            "and then the move"
        );

        // nothing is transcribed yet, so the export dialog doesn't open
        let _ = app.ws_update(Msg::SeqExport(seq.id));
        assert!(!export_ui::is_open(&app.export));
        assert!(app.ws_sequence_jobs(seq.id).is_err());

        let _ = app.ws_update(Msg::SeqDelete(seq.id));
        assert!(data(&app).sequences.is_empty());
        let _ = app.ws_update(Msg::Undo);
        assert_eq!(data(&app).sequences.len(), 1, "deleting can be undone");
    }

    #[test]
    fn in_the_player_one_shortcut_undoes_cuts_and_marks_newest_first() {
        use crate::{Page, marks_ui, player};
        use iced::keyboard::{Key, Modifiers};
        let dir = TestDir::new("shortcut");
        let mut app = app_in(&dir);
        let job = crate::JobRef {
            id: 1,
            media: dir.0.join("a.mp4"),
            paths: engine::project::ProjectPaths::new(dir.0.clone(), "a".to_string()),
        };
        // the recording is in a project, so its changes are logged
        app.jobs.push(Job::new(1, job.media.clone(), 1));
        app.ws_jobs_added(&[1]);
        app.player = player::for_test(job.clone(), 60.0, 10.0);
        let _ = marks_ui::open(&mut app.marks, &job);
        app.page = Page::Player(1);
        let key = |app: &mut App, c: &str, mods| {
            let _ = app.route_key(Key::Character(c.into()), mods);
            saver::wait();
        };
        let marks = |app: &App| marks_ui::marks(&app.marks).len();

        // a cut, then a bookmark, then another cut
        let _ = app.update(Message::Player(player::Msg::CutStart));
        player::seek_for_test(&mut app.player, 12.0);
        let _ = app.update(Message::Player(player::Msg::CutEnd));
        let _ = app.update(Message::Marks(marks_ui::Msg::AddAt(20.0)));
        player::seek_for_test(&mut app.player, 30.0);
        let _ = app.update(Message::Player(player::Msg::CutStart));
        player::seek_for_test(&mut app.player, 35.0);
        let _ = app.update(Message::Player(player::Msg::CutEnd));
        assert_eq!((player::cut_count(&app.player), marks(&app)), (2, 1));

        key(&mut app, "z", Modifiers::CTRL);
        assert_eq!(
            (player::cut_count(&app.player), marks(&app)),
            (1, 1),
            "the second cut first"
        );
        key(&mut app, "z", Modifiers::CTRL);
        assert_eq!(
            (player::cut_count(&app.player), marks(&app)),
            (1, 0),
            "then the bookmark"
        );
        key(&mut app, "z", Modifiers::CTRL);
        assert_eq!(
            (player::cut_count(&app.player), marks(&app)),
            (0, 0),
            "then the first cut"
        );
        key(&mut app, "y", Modifiers::CTRL);
        assert_eq!(
            (player::cut_count(&app.player), marks(&app)),
            (1, 0),
            "redo goes the other way"
        );
        key(&mut app, "z", Modifiers::CTRL | Modifiers::SHIFT);
        assert_eq!((player::cut_count(&app.player), marks(&app)), (1, 1));
        key(&mut app, "y", Modifiers::CTRL);
        assert_eq!((player::cut_count(&app.player), marks(&app)), (2, 1));

        // each change reached the project's change log
        let kinds: Vec<Kind> = app.workspace.history.iter().map(|e| e.kind).collect();
        assert!(
            kinds.contains(&Kind::CutsChanged) && kinds.contains(&Kind::MarksChanged),
            "{kinds:?}"
        );
    }

    #[test]
    fn a_missing_recording_is_shown_as_missing_and_can_be_found_again() {
        let dir = TestDir::new("missing");
        let file = dir.0.join("p.vipr");
        let mut w = Workspace::new("p");
        let clip = w.add_clip(&dir.0.join("gone.mp4"), 9);
        w.save(&file).unwrap();
        let mut app = app_in(&dir);
        let _ = app.ws_update(Msg::Loaded(Ok((
            file.clone(),
            Workspace::load(&file).unwrap(),
        ))));
        finish_loading(&mut app);
        assert!(app.jobs[0].media_missing && app.jobs[0].state == JobState::Failed);

        let found = dir.0.join("found.mp4");
        std::fs::write(&found, b"abc").unwrap();
        let id = app.jobs[0].id;
        let _ = app.update(Message::MediaLocated(id, Some(found.clone())));
        assert_eq!(
            app.workspace
                .open
                .as_ref()
                .unwrap()
                .data
                .media_path(clip, Some(&file)),
            Some(found)
        );
        let job = &app.jobs[0];
        assert!(
            !job.media_missing && job.state == JobState::Waiting && job.error.is_none(),
            "can run now"
        );
    }

    #[test]
    fn transcription_metrics_read_as_words() {
        let e = Entry::new(Kind::Transcribed, "a.mp4")
            .metric("audio_seconds", 1082.0)
            .metric("elapsed_seconds", 57.0);
        assert_eq!(metrics_text(&e), "18:02 of sound in 0:57, 19× real time");
        assert_eq!(metrics_text(&Entry::new(Kind::Saved, "")), "");
        let cuts = Entry::new(Kind::CutsChanged, "Cut")
            .metric("cuts", 2.0)
            .metric("cut_seconds", 65.0);
        assert_eq!(metrics_text(&cuts), "2 cuts, 1:05 left out");
    }

    #[test]
    fn times_read_as_how_long_ago() {
        let now = workspace::now();
        assert_eq!(ago(now), "just now");
        assert_eq!(ago(now - 600), "10 min ago");
        assert_eq!(ago(now - 7200), "2 h ago");
        assert_eq!(ago(now - 3 * 86_400), "3 days ago");
    }

    #[test]
    fn a_named_project_shows_unsaved_changes_in_its_title() {
        let open = |path: &str, dirty, temporary| Open {
            path: PathBuf::from(path),
            data: Workspace::new(""),
            dirty,
            temporary,
            log: HistoryLog::new("x"),
            last_backup: None,
        };
        assert_eq!(open("C:/p/Meeting.vipr", true, false).title(), "Meeting •");
        assert_eq!(open("C:/p/Meeting.vipr", false, false).title(), "Meeting");
        assert_eq!(
            open("C:/u/untitled-1/untitled.vipr", false, true).title(),
            "Unsaved project"
        );
    }
}
