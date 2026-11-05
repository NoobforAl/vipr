//! The compare review page: a paged table of A, B and the AI's pick, where the user overrides or
//! edits rows and saves the result. Nothing reaches `<stem>.reviewed.*` without this page.
//!
//! [`Table`] is the pure reducer and [`reviewed_output`] the pure save step. The user's picks are also
//! written to `<stem>.compare-<slug>.json` after every change, so closing the app loses nothing.
//!
//! A, B, merged text and AI notes are all untrusted. They are shown as plain text after
//! `lang::clean_separators`; the app copies A and B from its own row and never from the model.

use std::path::{Path, PathBuf};

use engine::compare::MAX_REASON_CHARS;
use engine::fsio;
use engine::lang;
use engine::project::{
    AiChoice, AiPick, AiStatus, CompareFile, CompareRow, CompareSource, ProjectPaths, UserPick,
};
use engine::subtitles::{self, Cue};
use iced::keyboard::key::Named;
use iced::keyboard::{Key, Modifiers};
use iced::widget::{
    Column, Id, button, column, container, operation, row, scrollable, space, text, text_editor,
};
use iced::{Alignment, Color, Element, Length, Task};

use crate::icons;
use crate::import_ui::{clean_action, clock, content_text, on_thread, truncate_chars};
use crate::out::{JobRef, Out, Page};
use crate::theme::{self, Tokens};

pub const PAGE_SIZE: usize = 25;
const ROWS_ID: Id = Id::new("review-rows");

// ------------------------------------------------------------------------- reducer

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Filter {
    #[default]
    All,
    /// Rows with text on both sides that differs.
    Differ,
    /// Rows that still need a decision before saving.
    Undecided,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    A,
    B,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Override the AI. Refused when the row has no such text.
    Pick {
        id: u64,
        pick: UserPick,
    },
    /// Back to the AI's pick.
    Clear {
        id: u64,
    },
    /// Every undecided row takes this side, or its only side.
    FillUndecided(Side),
    SetFilter(Filter),
    NextPage,
    PrevPage,
}

/// True when the row has something to save but nobody has picked it.
fn is_undecided(row: &CompareRow) -> bool {
    row.resolved_text().is_none() && (row.a.is_some() || row.b.is_some())
}

fn differs(row: &CompareRow) -> bool {
    matches!((&row.a, &row.b), (Some(a), Some(b)) if a != b)
}

fn merged_text(row: &CompareRow) -> Option<&str> {
    (row.ai.choice == Some(AiChoice::Merged))
        .then_some(row.ai.merged.as_deref())
        .flatten()
}

fn can_pick(row: &CompareRow, pick: &UserPick) -> bool {
    match pick {
        UserPick::A => row.a.is_some(),
        UserPick::B => row.b.is_some(),
        UserPick::Merged => merged_text(row).is_some(),
        UserPick::Edited { .. } => true,
    }
}

/// Which text of a row counts now, after the user's pick and then the AI's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shown {
    A,
    B,
    Merged,
    Edited,
}

fn shown(row: &CompareRow) -> Option<Shown> {
    match &row.user {
        Some(UserPick::A) => Some(Shown::A),
        Some(UserPick::B) => Some(Shown::B),
        Some(UserPick::Merged) => Some(Shown::Merged),
        Some(UserPick::Edited { .. }) => Some(Shown::Edited),
        None => match row.ai.choice? {
            AiChoice::A => Some(Shown::A),
            AiChoice::B => Some(Shown::B),
            AiChoice::Merged => Some(Shown::Merged),
        },
    }
}

/// One line for each row of text, trimmed, with no blank lines: a blank line would end an SRT cue
/// early. Also splits on the separators that are unsafe to display, so saved files have none.
fn normalize_lines(text: &str) -> String {
    text.split([
        '\n', '\r', '\u{b}', '\u{c}', '\u{85}', '\u{2028}', '\u{2029}',
    ])
    .flat_map(|l| l.split(['\u{1c}', '\u{1d}', '\u{1e}']))
    .map(str::trim)
    .filter(|l| !l.is_empty())
    .collect::<Vec<_>>()
    .join("\n")
}

/// The compare table and where the user is in it.
#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    file: CompareFile,
    page: usize,
    filter: Filter,
}

impl Table {
    fn new(file: CompareFile) -> Table {
        Table {
            file,
            page: 0,
            filter: Filter::default(),
        }
    }

    /// Applies an action. True when a pick changed, so the file must be written again.
    fn reduce(&mut self, action: Action) -> bool {
        let changed = match action {
            Action::Pick { id, pick } => self.pick(id, pick),
            Action::Clear { id } => self
                .row_mut(id)
                .is_some_and(|row| row.user.take().is_some()),
            Action::FillUndecided(side) => self.fill(side),
            Action::SetFilter(filter) => {
                self.filter = filter;
                self.page = 0;
                false
            }
            Action::NextPage => {
                self.page += 1;
                false
            }
            Action::PrevPage => {
                self.page = self.page.saturating_sub(1);
                false
            }
        };
        self.page = self.page.min(self.page_count() - 1);
        changed
    }

    fn pick(&mut self, id: u64, pick: UserPick) -> bool {
        let Some(row) = self.row_mut(id) else {
            return false;
        };
        if !can_pick(row, &pick) {
            return false;
        }
        let pick = match pick {
            UserPick::Edited { text } => {
                let text = normalize_lines(&text);
                // confirming the editor without typing is not an override
                if row.resolved_text().map(normalize_lines).as_ref() == Some(&text) {
                    return false;
                }
                UserPick::Edited { text }
            }
            other => other,
        };
        if row.user.as_ref() == Some(&pick) {
            return false;
        }
        row.user = Some(pick);
        true
    }

    fn fill(&mut self, side: Side) -> bool {
        let mut changed = false;
        for row in self.file.rows.iter_mut().filter(|r| is_undecided(r)) {
            let pick = match (side, row.a.is_some(), row.b.is_some()) {
                (Side::A, true, _) | (Side::B, true, false) => UserPick::A,
                _ => UserPick::B,
            };
            row.user = Some(pick);
            changed = true;
        }
        changed
    }

    fn row(&self, id: u64) -> Option<&CompareRow> {
        self.file.rows.iter().find(|r| r.id == id)
    }

    fn row_mut(&mut self, id: u64) -> Option<&mut CompareRow> {
        self.file.rows.iter_mut().find(|r| r.id == id)
    }

    fn matches_filter(&self, row: &CompareRow) -> bool {
        match self.filter {
            Filter::All => true,
            Filter::Differ => differs(row),
            Filter::Undecided => is_undecided(row),
        }
    }

    fn visible_count(&self) -> usize {
        self.file
            .rows
            .iter()
            .filter(|r| self.matches_filter(r))
            .count()
    }

    /// Never zero: an empty table still has one, empty, page.
    fn page_count(&self) -> usize {
        self.visible_count().div_ceil(PAGE_SIZE).max(1)
    }

    fn page_rows(&self) -> impl Iterator<Item = &CompareRow> {
        self.file
            .rows
            .iter()
            .filter(|r| self.matches_filter(r))
            .skip(self.page * PAGE_SIZE)
            .take(PAGE_SIZE)
    }

    fn undecided(&self) -> usize {
        self.file.rows.iter().filter(|r| is_undecided(r)).count()
    }
}

// ------------------------------------------------------------------------- save output

/// What a save writes.
#[derive(Debug, Clone, PartialEq)]
struct Reviewed {
    /// In time order.
    cues: Vec<Cue>,
    /// Rows whose text is empty, which is how a line is removed on purpose.
    dropped: usize,
}

/// The saved text of every row, or the ids of rows that have no pick yet. Nothing is guessed: a row
/// without a decision blocks the save.
fn reviewed_output(rows: &[CompareRow]) -> Result<Reviewed, Vec<u64>> {
    let undecided: Vec<u64> = rows
        .iter()
        .filter(|r| is_undecided(r))
        .map(|r| r.id)
        .collect();
    if !undecided.is_empty() {
        return Err(undecided);
    }
    let mut cues = Vec::with_capacity(rows.len());
    let mut dropped = 0;
    for row in rows {
        let Some(raw) = row.resolved_text() else {
            continue;
        };
        let text = normalize_lines(raw);
        if text.is_empty() {
            dropped += 1;
            continue;
        }
        cues.push(Cue {
            start: row.start,
            end: row.end.max(row.start),
            text,
            speaker: 0,
        });
    }
    cues.sort_by(|a, b| a.start.total_cmp(&b.start));
    Ok(Reviewed { cues, dropped })
}

/// One cue per line.
fn to_txt(cues: &[Cue]) -> String {
    let mut out = String::new();
    for c in cues {
        out.push_str(&c.text.replace('\n', " "));
        out.push('\n');
    }
    out
}

/// The language of the saved text, when the compare file says: only translations carry one.
fn output_lang(file: &CompareFile) -> Option<&str> {
    [&file.a, &file.b].into_iter().find_map(|s| match s {
        CompareSource::Translation { lang } => Some(lang.as_str()),
        CompareSource::Transcript | CompareSource::Import { .. } => None,
    })
}

/// RLM at the start of each line that reads right to left. For saved text whose language isn't known,
/// such as a transcript compared with an import.
fn mark_rtl_lines(cues: &[Cue]) -> Vec<Cue> {
    let mark = |line: &str| {
        if lang::is_rtl_text(line) && !line.starts_with('\u{200F}') {
            format!("\u{200F}{line}")
        } else {
            line.to_string()
        }
    };
    cues.iter()
        .map(|c| Cue {
            text: c.text.split('\n').map(mark).collect::<Vec<_>>().join("\n"),
            ..c.clone()
        })
        .collect()
}

/// RTL line marks go into the written files only, and never into the cues themselves.
fn output_files(
    paths: &ProjectPaths,
    reviewed: &Reviewed,
    lang: Option<&str>,
) -> Vec<(PathBuf, String)> {
    let cues = match lang.filter(|code| lang::language(code).is_some_and(|l| l.rtl)) {
        Some(code) => subtitles::mark_rtl(&reviewed.cues, code),
        None => mark_rtl_lines(&reviewed.cues),
    };
    let cues = cues.as_slice();
    vec![
        (paths.reviewed("srt"), subtitles::to_srt(cues)),
        (paths.reviewed("vtt"), subtitles::to_vtt(cues)),
        (paths.reviewed("txt"), to_txt(cues)),
    ]
}

/// One write at a time, so an older snapshot can't land on top of a newer one. Requests made
/// meanwhile collapse into the next write.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Writer {
    in_flight: bool,
    want_json: bool,
    want_outputs: bool,
}

impl Writer {
    /// `Some(with_outputs)` when a write should start now.
    fn begin(&mut self) -> Option<bool> {
        if self.in_flight || !(self.want_json || self.want_outputs) {
            return None;
        }
        let with_outputs = self.want_outputs;
        *self = Writer {
            in_flight: true,
            want_json: false,
            want_outputs: false,
        };
        Some(with_outputs)
    }

    fn finish(&mut self) {
        self.in_flight = false;
    }

    /// True when memory holds something the disk doesn't have yet.
    fn pending(&self) -> bool {
        self.in_flight || self.want_json || self.want_outputs
    }
}

/// The picks first: if the outputs then fail, nothing the user decided is lost.
fn write_all(path: &Path, file: &CompareFile, outputs: &[(PathBuf, String)]) -> Result<(), String> {
    fsio::save_json(path, file).map_err(|e| e.to_string())?;
    for (path, body) in outputs {
        fsio::write_atomic(path, body.as_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(())
}

// ------------------------------------------------------------------------- loading

/// The slug of a `<stem>.compare-<slug>.json` file name.
fn compare_slug<'a>(file_name: &'a str, stem: &str) -> Option<&'a str> {
    let slug = file_name
        .strip_prefix(stem)?
        .strip_prefix(".compare-")?
        .strip_suffix(".json")?;
    (!slug.is_empty()).then_some(slug)
}

/// The most recently written compare file of a result folder.
fn newest_compare(paths: &ProjectPaths) -> Option<PathBuf> {
    std::fs::read_dir(&paths.dir)
        .ok()?
        .flatten()
        .filter(|e| compare_slug(&e.file_name().to_string_lossy(), &paths.stem).is_some())
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .max_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)))
        .map(|(_, path)| path)
}

/// This result has a comparison to review.
pub fn has_comparison(paths: &ProjectPaths) -> bool {
    newest_compare(paths).is_some()
}

fn load_newest(paths: &ProjectPaths) -> Result<Option<(PathBuf, CompareFile)>, String> {
    let Some(path) = newest_compare(paths) else {
        return Ok(None);
    };
    let file = fsio::load_json(&path).map_err(|e| e.to_string())?;
    Ok(Some((path, file)))
}

// ------------------------------------------------------------------------- state

#[derive(Debug, Clone, Default, PartialEq, Eq)]
enum Load {
    #[default]
    Loading,
    /// This job has no compare file.
    Missing,
    /// The file stays as it is; the user can fix or delete it.
    Failed(String),
    Ready,
}

#[derive(Debug)]
struct Editing {
    id: u64,
    content: text_editor::Content,
}

#[derive(Debug, Default)]
pub struct State {
    job: Option<JobRef>,
    /// Bumped on every load, so late answers about another file are not taken for this one.
    generation: u64,
    load: Load,
    path: Option<PathBuf>,
    table: Option<Table>,
    /// Where the user was, to restore when the same file comes back.
    restore: Option<(PathBuf, usize, Filter)>,
    editing: Option<Editing>,
    writer: Writer,
    /// Counts changes to the picks.
    revision: u64,
    /// The revision in the saved files, with cue and dropped counts.
    saved: Option<(u64, usize, usize)>,
    /// The last write failed, so memory holds more than the disk.
    write_error: Option<String>,
    /// Why Save did nothing. Goes away with the next change of a pick.
    error: Option<String>,
}

#[derive(Debug, Clone)]
pub enum Msg {
    Loaded {
        generation: u64,
        result: Result<Option<(PathBuf, CompareFile)>, String>,
    },
    Choose(u64, UserPick),
    Clear(u64),
    FillUndecided(Side),
    SetFilter(Filter),
    NextPage,
    PrevPage,
    EditStart(u64),
    EditAction(text_editor::Action),
    EditCommit,
    EditCancel,
    Play(u64),
    Save,
    Written {
        generation: u64,
        /// Revision and counts when this write included the output files.
        outputs: Option<(u64, usize, usize)>,
        result: Result<(), String>,
    },
    /// An older file was flushed on the way to another job.
    Flushed(Result<(), String>),
    Back,
}

/// The page is opening on `job`: load its newest `<stem>.compare-<slug>.json`.
///
/// Coming back from the player must not lose anything, so unwritten picks or an open editor on the
/// same job keep what is in memory. Otherwise the disk holds everything, and is read again in case a
/// new compare has finished.
pub fn open(state: &mut State, job: &JobRef) -> (Task<Msg>, Out) {
    let same_job = state.job.as_ref().is_some_and(|j| j.id == job.id);
    let unwritten =
        state.writer.pending() || state.editing.is_some() || state.write_error.is_some();
    if same_job && state.table.is_some() && unwritten {
        return (Task::none(), Out::None);
    }
    let flush = if same_job { Task::none() } else { flush(state) };
    state.restore = match (&state.path, &state.table) {
        (Some(path), Some(table)) if same_job => Some((path.clone(), table.page, table.filter)),
        _ => None,
    };
    state.job = Some(job.clone());
    state.generation += 1;
    state.load = Load::Loading;
    state.path = None;
    state.table = None;
    state.editing = None;
    state.writer = Writer::default();
    state.saved = None;
    state.write_error = None;
    state.error = None;
    let generation = state.generation;
    let paths = job.paths.clone();
    let load = Task::perform(on_thread(move || load_newest(&paths)), move |result| {
        Msg::Loaded { generation, result }
    });
    (Task::batch([flush, load]), Out::None)
}

/// Leaving for another job with a pick still unwritten: write it now.
fn flush(state: &State) -> Task<Msg> {
    let (Some(path), Some(table)) = (&state.path, &state.table) else {
        return Task::none();
    };
    if !state.writer.want_json && state.write_error.is_none() {
        return Task::none();
    }
    let path = path.clone();
    let file = table.file.clone();
    Task::perform(
        on_thread(move || write_all(&path, &file, &[])),
        Msg::Flushed,
    )
}

pub fn update(state: &mut State, msg: Msg) -> (Task<Msg>, Out) {
    match msg {
        Msg::Loaded { generation, result } => {
            if generation == state.generation {
                loaded(state, result);
            }
        }
        Msg::Choose(id, pick) => {
            commit_edit(state);
            apply(state, Action::Pick { id, pick });
        }
        Msg::Clear(id) => {
            commit_edit(state);
            apply(state, Action::Clear { id });
        }
        Msg::FillUndecided(side) => {
            commit_edit(state);
            apply(state, Action::FillUndecided(side));
        }
        Msg::SetFilter(filter) => {
            commit_edit(state);
            apply(state, Action::SetFilter(filter));
            return (Task::batch([write_next(state), to_top()]), Out::None);
        }
        Msg::NextPage => {
            commit_edit(state);
            apply(state, Action::NextPage);
            return (Task::batch([write_next(state), to_top()]), Out::None);
        }
        Msg::PrevPage => {
            commit_edit(state);
            apply(state, Action::PrevPage);
            return (Task::batch([write_next(state), to_top()]), Out::None);
        }
        Msg::EditStart(id) => {
            commit_edit(state);
            start_edit(state, id);
        }
        Msg::EditAction(action) => {
            if let Some(editing) = &mut state.editing {
                editing.content.perform(clean_action(action));
            }
        }
        Msg::EditCommit => commit_edit(state),
        Msg::EditCancel => state.editing = None,
        Msg::Play(id) => {
            let row = state.table.as_ref().and_then(|t| t.row(id));
            if let (Some(job), Some(row)) = (&state.job, row) {
                return (
                    Task::none(),
                    Out::PlayAt {
                        job: job.id,
                        time: row.start.max(0.0),
                    },
                );
            }
        }
        Msg::Save => {
            commit_edit(state);
            request_save(state);
        }
        Msg::Written {
            generation,
            outputs,
            result,
        } => {
            if generation != state.generation {
                // that page is gone, so the only place left to say so is a notice
                return (Task::none(), notice_on_error(result));
            }
            state.writer.finish();
            match result {
                Ok(()) => {
                    if outputs.is_some() {
                        state.saved = outputs;
                    }
                    state.write_error = None;
                }
                Err(e) => state.write_error = Some(format!("Could not save: {e}")),
            }
        }
        Msg::Flushed(result) => return (Task::none(), notice_on_error(result)),
        Msg::Back => {
            commit_edit(state);
            let task = write_next(state);
            let out = state
                .job
                .as_ref()
                .map_or(Out::None, |j| Out::Open(Page::Transcript(j.id)));
            return (task, out);
        }
    }
    (write_next(state), Out::None)
}

fn notice_on_error(result: Result<(), String>) -> Out {
    match result {
        Ok(()) => Out::None,
        Err(e) => Out::Notice(format!("Could not save the review: {e}")),
    }
}

fn to_top() -> Task<Msg> {
    operation::snap_to(ROWS_ID, scrollable::RelativeOffset::START)
}

fn loaded(state: &mut State, result: Result<Option<(PathBuf, CompareFile)>, String>) {
    let restore = state.restore.take();
    match result {
        Ok(None) => state.load = Load::Missing,
        Err(e) => state.load = Load::Failed(e),
        Ok(Some((path, file))) => {
            let mut table = Table::new(file);
            if let Some((_, page, filter)) = restore.filter(|(p, _, _)| *p == path) {
                table.filter = filter;
                table.page = page.min(table.page_count() - 1);
            }
            state.path = Some(path);
            state.table = Some(table);
            state.load = Load::Ready;
        }
    }
}

fn apply(state: &mut State, action: Action) {
    let Some(table) = &mut state.table else {
        return;
    };
    if table.reduce(action) {
        state.revision += 1;
        state.writer.want_json = true;
        state.error = None;
    }
}

fn start_edit(state: &mut State, id: u64) {
    let Some(row) = state.table.as_ref().and_then(|t| t.row(id)) else {
        return;
    };
    let start = row
        .resolved_text()
        .or(row.a.as_deref())
        .or(row.b.as_deref())
        .unwrap_or_default();
    let content = text_editor::Content::with_text(&lang::clean_separators(start));
    state.editing = Some(Editing { id, content });
}

fn commit_edit(state: &mut State) {
    if let Some(Editing { id, content }) = state.editing.take() {
        let pick = UserPick::Edited {
            text: content.text(),
        };
        apply(state, Action::Pick { id, pick });
    }
}

fn request_save(state: &mut State) {
    let Some(table) = &mut state.table else {
        return;
    };
    let undecided = table.undecided();
    if undecided > 0 {
        table.reduce(Action::SetFilter(Filter::Undecided));
        state.error = Some(format!(
            "Rows with no pick yet: {undecided}. Choose one for each, or use a side for all of them."
        ));
        return;
    }
    state.writer.want_outputs = true;
}

/// Starts the next write, if one is wanted and none is running.
fn write_next(state: &mut State) -> Task<Msg> {
    let (Some(job), Some(path), Some(table)) = (&state.job, &state.path, &state.table) else {
        return Task::none();
    };
    let Some(with_outputs) = state.writer.begin() else {
        return Task::none();
    };
    let mut files = Vec::new();
    let mut outputs = None;
    if with_outputs {
        // picks can have changed since the request, so check them again
        match reviewed_output(&table.file.rows) {
            Ok(reviewed) => {
                files = output_files(&job.paths, &reviewed, output_lang(&table.file));
                outputs = Some((state.revision, reviewed.cues.len(), reviewed.dropped));
            }
            Err(ids) => {
                state.error = Some(format!("Rows with no pick yet: {}.", ids.len()));
            }
        }
    }
    let generation = state.generation;
    let path = path.clone();
    let file = table.file.clone();
    Task::perform(
        on_thread(move || write_all(&path, &file, &files)),
        move |result| Msg::Written {
            generation,
            outputs,
            result,
        },
    )
}

/// A key nothing else used, while this page is open.
pub fn key(state: &State, key: &Key, mods: Modifiers) -> Option<Msg> {
    state.table.as_ref()?;
    match key.as_ref() {
        Key::Character("s") if mods.command() => Some(Msg::Save),
        Key::Named(Named::PageDown) if state.editing.is_none() => Some(Msg::NextPage),
        Key::Named(Named::PageUp) if state.editing.is_none() => Some(Msg::PrevPage),
        _ => None,
    }
}

// ------------------------------------------------------------------------- view

fn source_label(source: &CompareSource) -> String {
    let label = match source {
        CompareSource::Transcript => "Transcript".to_string(),
        CompareSource::Translation { lang } => format!("Translation ({lang})"),
        CompareSource::Import { slug } => format!("Import ({slug})"),
    };
    lang::clean_separators(&label)
}

/// What the AI did with a row, for its chip. Empty when there is nothing to say.
fn ai_label(ai: &AiPick) -> &'static str {
    match (ai.status, ai.choice) {
        (AiStatus::Identical, _) => "Same text",
        (AiStatus::OneSided, _) => "One side only",
        (AiStatus::Failed, _) => "AI could not decide",
        (AiStatus::Pending, _) => "Not judged",
        (_, Some(AiChoice::A)) => "AI pick: A",
        (_, Some(AiChoice::B)) => "AI pick: B",
        (_, Some(AiChoice::Merged)) => "AI pick: merged",
        (_, None) => "",
    }
}

pub fn view(state: &State, t: Tokens) -> Element<'_, Msg> {
    let message = |line: String, color: Color| -> Element<'_, Msg> {
        column![
            back_button(t),
            container(content_text(&line, 14.0, color)).padding(16),
        ]
        .spacing(8)
        .padding(16)
        .into()
    };

    match (&state.load, &state.table) {
        (Load::Ready, Some(table)) => column![
            row![back_button(t)].padding([10, 16]),
            table_view(state, table, t)
        ]
        .width(Length::Fill)
        .height(Length::Fill)
        .into(),
        (Load::Missing, _) => message("This result has no comparison yet.".into(), t.muted),
        (Load::Failed(e), _) => message(format!("Could not open the comparison: {e}"), t.bad),
        _ => message("Loading…".into(), t.muted),
    }
}

fn back_button<'a>(t: Tokens) -> Element<'a, Msg> {
    button(
        row![
            icons::icon(icons::ARROW_LEFT, 15.0, t.muted),
            text("Back").size(13)
        ]
        .spacing(6)
        .align_y(Alignment::Center),
    )
    .padding([6, 10])
    .style(theme::ghost(t))
    .on_press(Msg::Back)
    .into()
}

fn table_view<'a>(state: &'a State, table: &'a Table, t: Tokens) -> Element<'a, Msg> {
    let a_label = source_label(&table.file.a);
    let b_label = source_label(&table.file.b);
    let undecided = table.undecided();

    let seg = |label: &'static str, filter: Filter| {
        button(text(label).size(13))
            .padding([6, 12])
            .style(theme::segment(t, table.filter == filter))
            .on_press(Msg::SetFilter(filter))
    };
    let mut head = row![
        text("Review").size(18).color(t.text),
        text(format!("{} rows", table.file.rows.len()))
            .size(13)
            .color(t.muted),
        space::horizontal(),
        seg("All", Filter::All),
        seg("Different", Filter::Differ),
        seg("Undecided", Filter::Undecided),
    ]
    .spacing(10)
    .align_y(Alignment::Center);
    if undecided > 0 {
        head = head.push(
            button(text("Use A for undecided").size(13))
                .padding([6, 12])
                .style(theme::outline_btn(t))
                .on_press(Msg::FillUndecided(Side::A)),
        );
        head = head.push(
            button(text("Use B for undecided").size(13))
                .padding([6, 12])
                .style(theme::outline_btn(t))
                .on_press(Msg::FillUndecided(Side::B)),
        );
    }

    let mut rows = Column::new()
        .spacing(10)
        .padding(iced::Padding::ZERO.right(12));
    for r in table.page_rows() {
        rows = rows.push(row_view(r, &a_label, &b_label, state.editing.as_ref(), t));
    }
    if table.visible_count() == 0 {
        rows = rows.push(text("No rows here.").size(13).color(t.muted));
    }

    let pager = row![
        button(text("Previous").size(13))
            .padding([6, 12])
            .style(theme::outline_btn(t))
            .on_press_maybe((table.page > 0).then_some(Msg::PrevPage)),
        text(format!("Page {} of {}", table.page + 1, table.page_count()))
            .size(13)
            .color(t.muted),
        button(text("Next").size(13))
            .padding([6, 12])
            .style(theme::outline_btn(t))
            .on_press_maybe((table.page + 1 < table.page_count()).then_some(Msg::NextPage)),
        space::horizontal(),
        status_view(state, undecided, t),
        button(
            row![
                icons::icon(icons::DOWNLOAD, 15.0, t.on_accent),
                text("Save reviewed files").size(13)
            ]
            .spacing(6)
            .align_y(Alignment::Center),
        )
        .padding([8, 12])
        .style(theme::primary(t))
        .on_press(Msg::Save),
    ]
    .spacing(10)
    .align_y(Alignment::Center);

    column![
        head,
        scrollable(rows).id(ROWS_ID).height(Length::Fill),
        pager
    ]
    .spacing(12)
    .padding(iced::Padding::new(16.0).top(0.0))
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}

fn status_view<'a>(state: &'a State, undecided: usize, t: Tokens) -> Element<'a, Msg> {
    if let Some(error) = state.write_error.as_ref().or(state.error.as_ref()) {
        return container(text(lang::clean_separators(error)).size(13).color(t.bad))
            .max_width(420)
            .into();
    }
    let (line, color) = match state.saved {
        Some((revision, cues, dropped)) if revision == state.revision => {
            let dropped = match dropped {
                0 => String::new(),
                n => format!(", {n} left out"),
            };
            (format!("Saved {cues} lines{dropped}"), t.ok)
        }
        _ if undecided > 0 => (format!("{undecided} undecided"), t.warn),
        _ => ("Not saved yet".to_string(), t.muted),
    };
    text(line).size(13).color(color).into()
}

fn row_view<'a>(
    r: &'a CompareRow,
    a_label: &str,
    b_label: &str,
    editing: Option<&'a Editing>,
    t: Tokens,
) -> Element<'a, Msg> {
    let id = r.id;
    let current = shown(r);
    let time = format!("#{id}   {} to {}", clock(r.start), clock(r.end));

    let mut head = row![text(time).size(12).color(t.faint)]
        .spacing(8)
        .align_y(Alignment::Center);
    let ai = ai_label(&r.ai);
    if !ai.is_empty() {
        head = head.push(
            container(text(ai).size(11).color(t.accent))
                .padding([2, 8])
                .style(theme::chip(t.accent)),
        );
    }
    if r.user.is_some() {
        head = head.push(
            container(text("Your pick").size(11).color(t.ok))
                .padding([2, 8])
                .style(theme::chip(t.ok)),
        );
    }
    head = head.push(space::horizontal());
    if r.user.is_some() {
        let label = if r.ai.choice.is_some() {
            "Use AI pick"
        } else {
            "Clear"
        };
        head = head.push(small_button(label, Msg::Clear(id), t));
    }
    head = head.push(small_button("Edit", Msg::EditStart(id), t));
    head = head.push(
        button(
            row![
                icons::icon(icons::PLAY, 13.0, t.muted),
                text("Play this moment").size(12)
            ]
            .spacing(5)
            .align_y(Alignment::Center),
        )
        .padding([4, 8])
        .style(theme::ghost(t))
        .on_press(Msg::Play(id)),
    );

    let sides = row![
        side_button(
            a_label,
            r.a.as_deref(),
            current == Some(Shown::A),
            Msg::Choose(id, UserPick::A),
            t
        ),
        side_button(
            b_label,
            r.b.as_deref(),
            current == Some(Shown::B),
            Msg::Choose(id, UserPick::B),
            t
        ),
    ]
    .spacing(8);

    let mut col = column![head, sides].spacing(8);

    if let Some(merged) = merged_text(r) {
        col = col.push(side_button(
            "AI merged",
            Some(merged),
            current == Some(Shown::Merged),
            Msg::Choose(id, UserPick::Merged),
            t,
        ));
    }
    if let Some(UserPick::Edited { text: edited }) = &r.user {
        let body = if edited.is_empty() {
            text("This line is left out of the saved files.")
                .size(13)
                .color(t.muted)
                .into()
        } else {
            content_text(edited, 14.0, t.text)
        };
        col = col.push(
            container(column![text("Your edit").size(11).color(t.ok), body].spacing(4))
                .padding(10)
                .width(Length::Fill)
                .style(theme::chip(t.ok)),
        );
    }
    if !r.ai.reason.is_empty() {
        // the model's words, never trusted: plain text, capped, and labelled as what it is
        let note = truncate_chars(&r.ai.reason, MAX_REASON_CHARS);
        col = col.push(
            row![
                text("AI hint, may be wrong:").size(12).color(t.faint),
                content_text(&note, 12.0, t.muted)
            ]
            .spacing(8),
        );
    }
    if let Some(editing) = editing.filter(|e| e.id == id) {
        col = col.push(edit_view(editing, t));
    }

    container(col)
        .padding(12)
        .width(Length::Fill)
        .style(theme::bubble(t))
        .into()
}

fn edit_view(editing: &Editing, t: Tokens) -> Element<'_, Msg> {
    column![
        text_editor(&editing.content)
            .on_action(Msg::EditAction)
            .size(14)
            .padding(8)
            .height(Length::Fixed(84.0)),
        text("Leave empty to remove this line. On right-to-left lines the cursor and selection can appear on the wrong side.")
            .size(11)
            .color(t.faint),
        row![
            button(text("Keep edit").size(12))
                .padding([5, 10])
                .style(theme::primary(t))
                .on_press(Msg::EditCommit),
            button(text("Cancel").size(12))
                .padding([5, 10])
                .style(theme::outline_btn(t))
                .on_press(Msg::EditCancel),
        ]
        .spacing(8),
    ]
    .spacing(6)
    .into()
}

fn small_button<'a>(label: &'static str, msg: Msg, t: Tokens) -> Element<'a, Msg> {
    button(text(label).size(12))
        .padding([4, 8])
        .style(theme::ghost(t))
        .on_press(msg)
        .into()
}

/// One selectable text. A side with no text at this moment can't be chosen.
fn side_button<'a>(
    label: &str,
    body: Option<&str>,
    selected: bool,
    msg: Msg,
    t: Tokens,
) -> Element<'a, Msg> {
    let content: Element<'a, Msg> = match body {
        Some(body) => content_text(body, 14.0, t.text),
        None => text("No text here").size(13).color(t.faint).into(),
    };
    button(column![text(label.to_string()).size(11).color(t.muted), content].spacing(4))
        .padding(10)
        .width(Length::FillPortion(1))
        .style(theme::segment(t, selected))
        .on_press_maybe(body.map(|_| msg))
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::project::VERSION;

    fn row(id: u64, a: Option<&str>, b: Option<&str>, ai: AiPick) -> CompareRow {
        CompareRow {
            id,
            start: id as f64,
            end: id as f64 + 0.5,
            a: a.map(str::to_string),
            b: b.map(str::to_string),
            ai,
            user: None,
        }
    }

    fn ai(choice: AiChoice) -> AiPick {
        AiPick {
            choice: Some(choice),
            merged: (choice == AiChoice::Merged).then(|| "merged text".to_string()),
            reason: "model says so".into(),
            status: AiStatus::Picked,
        }
    }

    fn file(rows: Vec<CompareRow>) -> CompareFile {
        CompareFile {
            version: VERSION,
            slug: "s".into(),
            a: CompareSource::Transcript,
            b: CompareSource::Import { slug: "s".into() },
            model: "m".into(),
            offset: 0.0,
            rows,
        }
    }

    fn table(rows: Vec<CompareRow>) -> Table {
        Table::new(file(rows))
    }

    /// Rows where the AI picked B between two different texts.
    fn many(n: u64) -> Table {
        table(
            (1..=n)
                .map(|i| row(i, Some("a"), Some("b"), ai(AiChoice::B)))
                .collect(),
        )
    }

    fn resolved(t: &Table, id: u64) -> Option<String> {
        t.row(id).unwrap().resolved_text().map(str::to_string)
    }

    #[test]
    fn override_beats_the_ai_and_clear_restores_it() {
        let mut t = many(1);
        assert_eq!(resolved(&t, 1).as_deref(), Some("b"));
        assert!(t.reduce(Action::Pick {
            id: 1,
            pick: UserPick::A
        }));
        assert_eq!(resolved(&t, 1).as_deref(), Some("a"));
        assert_eq!(shown(t.row(1).unwrap()), Some(Shown::A));
        assert!(t.reduce(Action::Clear { id: 1 }));
        assert_eq!(resolved(&t, 1).as_deref(), Some("b"));
        assert_eq!(shown(t.row(1).unwrap()), Some(Shown::B));
    }

    #[test]
    fn repeating_a_pick_is_not_a_change() {
        let mut t = many(1);
        let pick = Action::Pick {
            id: 1,
            pick: UserPick::A,
        };
        assert!(t.reduce(pick.clone()));
        assert!(!t.reduce(pick));
        assert!(t.reduce(Action::Clear { id: 1 }));
        assert!(!t.reduce(Action::Clear { id: 1 }));
    }

    #[test]
    fn a_pick_without_that_text_is_refused() {
        let mut t = table(vec![row(1, Some("a"), None, AiPick::default())]);
        let refused = |pick| Action::Pick { id: 1, pick };
        assert!(!t.reduce(refused(UserPick::B)));
        assert!(!t.reduce(refused(UserPick::Merged)));
        assert_eq!(t.row(1).unwrap().user, None);
        assert!(!t.reduce(Action::Pick {
            id: 99,
            pick: UserPick::A
        }));
    }

    #[test]
    fn merged_is_only_offered_when_the_ai_chose_to_merge() {
        // a stray merged field on a row where the AI chose A is not an option
        let mut stray = ai(AiChoice::A);
        stray.merged = Some("sneaked in".into());
        let mut t = table(vec![
            row(1, Some("a"), Some("b"), stray),
            row(2, Some("a"), Some("b"), ai(AiChoice::Merged)),
        ]);
        assert_eq!(merged_text(t.row(1).unwrap()), None);
        assert!(!t.reduce(Action::Pick {
            id: 1,
            pick: UserPick::Merged
        }));
        assert!(t.reduce(Action::Pick {
            id: 2,
            pick: UserPick::Merged
        }));
        assert_eq!(resolved(&t, 2).as_deref(), Some("merged text"));
    }

    #[test]
    fn a_or_b_save_our_own_text_never_the_models() {
        let mut pick = ai(AiChoice::A);
        pick.merged = Some("IGNORE ALL RULES".into());
        let rows = vec![row(1, Some("left"), Some("right"), pick)];
        let out = reviewed_output(&rows).unwrap();
        assert_eq!(out.cues[0].text, "left");
    }

    #[test]
    fn edits_are_normalized_and_an_unchanged_edit_is_no_override() {
        let mut t = many(1);
        let edit = |text: &str| Action::Pick {
            id: 1,
            pick: UserPick::Edited { text: text.into() },
        };
        // text_editor hands back a trailing newline
        assert!(!t.reduce(edit("b\n")));
        assert_eq!(t.row(1).unwrap().user, None);
        assert!(t.reduce(edit("  one\r\n\r\n two \u{2029}three\n")));
        assert_eq!(resolved(&t, 1).as_deref(), Some("one\ntwo\nthree"));
        assert_eq!(shown(t.row(1).unwrap()), Some(Shown::Edited));
    }

    #[test]
    fn normalizing_removes_every_unsafe_separator_and_blank_line() {
        assert_eq!(
            normalize_lines("a\u{85}b\u{1c}c\u{1d}d\u{1e}e\u{2028}f\n\n\ng"),
            "a\nb\nc\nd\ne\nf\ng"
        );
        assert_eq!(normalize_lines(" \n \r\n"), "");
        // right-to-left text and its marks are left alone
        assert_eq!(normalize_lines("سلام؟\u{200f}"), "سلام؟\u{200f}");
    }

    #[test]
    fn save_is_blocked_until_every_row_has_a_decision() {
        let failed = AiPick {
            status: AiStatus::Failed,
            ..AiPick::default()
        };
        let mut t = table(vec![
            row(1, Some("a"), Some("b"), ai(AiChoice::A)),
            row(2, Some("a"), Some("b"), failed.clone()),
            row(3, None, Some("only b"), failed),
        ]);
        assert_eq!(reviewed_output(&t.file.rows).unwrap_err(), [2, 3]);
        assert_eq!(t.undecided(), 2);

        assert!(t.reduce(Action::FillUndecided(Side::A)));
        assert_eq!(t.undecided(), 0);
        assert_eq!(resolved(&t, 2).as_deref(), Some("a"));
        // only side wins over the requested one
        assert_eq!(resolved(&t, 3).as_deref(), Some("only b"));
        // rows that were decided stay as they were
        assert_eq!(t.row(1).unwrap().user, None);
        assert!(!t.reduce(Action::FillUndecided(Side::B)));

        let out = reviewed_output(&t.file.rows).unwrap();
        assert_eq!(out.cues.len(), 3);
    }

    #[test]
    fn a_row_with_no_text_at_all_blocks_nothing() {
        let rows = vec![row(1, None, None, AiPick::default())];
        assert!(!is_undecided(&rows[0]));
        let out = reviewed_output(&rows).unwrap();
        assert_eq!((out.cues.len(), out.dropped), (0, 0));
    }

    #[test]
    fn an_emptied_row_is_left_out_on_purpose() {
        let mut t = many(2);
        assert!(t.reduce(Action::Pick {
            id: 1,
            pick: UserPick::Edited { text: " \n".into() },
        }));
        let out = reviewed_output(&t.file.rows).unwrap();
        assert_eq!(out.dropped, 1);
        assert_eq!(out.cues.len(), 1);
        assert_eq!(out.cues[0].start, 2.0);
    }

    #[test]
    fn saved_cues_are_in_time_order_with_a_valid_length() {
        let mut late = row(1, Some("late"), None, ai(AiChoice::A));
        late.start = 9.0;
        late.end = 8.0;
        let early = row(2, Some("early\n\nline"), None, ai(AiChoice::A));
        let out = reviewed_output(&[late, early]).unwrap();
        assert_eq!(out.cues[0].text, "early\nline");
        assert_eq!(out.cues[1].text, "late");
        assert_eq!(out.cues[1].end, 9.0);
    }

    #[test]
    fn output_files_hold_the_chosen_text() {
        let dir = TestDir::new("outputs");
        let paths = ProjectPaths::new(&dir.0, "rec (mp4)");
        let mut t = table(vec![
            row(1, Some("first\nline"), Some("no"), ai(AiChoice::A)),
            row(2, Some("no"), Some("second"), ai(AiChoice::A)),
        ]);
        t.reduce(Action::Pick {
            id: 2,
            pick: UserPick::B,
        });
        let compare = paths.compare("s");
        let reviewed = reviewed_output(&t.file.rows).unwrap();
        let files = output_files(&paths, &reviewed, output_lang(&t.file));
        write_all(&compare, &t.file, &files).unwrap();

        let read = |ext: &str| std::fs::read_to_string(paths.reviewed(ext)).unwrap();
        assert_eq!(
            read("srt"),
            "1\n00:00:01,000 --> 00:00:01,500\nfirst\nline\n\n2\n00:00:02,000 --> 00:00:02,500\nsecond\n\n"
        );
        assert!(read("vtt").starts_with("WEBVTT\n\n00:00:01.000 --> 00:00:01.500\nfirst\nline\n"));
        assert_eq!(read("txt"), "first line\nsecond\n");

        // and the picks come back when the page opens again
        let (path, back) = load_newest(&paths).unwrap().unwrap();
        assert_eq!(path, compare);
        assert_eq!(back.rows[1].user, Some(UserPick::B));
    }

    #[test]
    fn rtl_marks_reach_only_the_files_with_rtl_text() {
        let paths = ProjectPaths::new("out", "a");
        let reviewed = reviewed_output(&[row(1, Some("سلام"), None, ai(AiChoice::A))]).unwrap();
        let body = |files: &[(PathBuf, String)], i: usize| files[i].1.clone();

        let mut f = file(vec![]);
        f.b = CompareSource::Translation { lang: "fa".into() };
        assert_eq!(output_lang(&f), Some("fa"));
        let files = output_files(&paths, &reviewed, output_lang(&f));
        assert!(body(&files, 0).contains("\u{200f}سلام"));
        assert!(body(&files, 1).contains("\u{200f}سلام"));
        assert_eq!(body(&files, 2), "\u{200f}سلام\n");
        assert_eq!(reviewed.cues[0].text, "سلام", "the cue itself stays clean");

        // a transcript compared with an import has no language, so each line decides for itself
        assert_eq!(output_lang(&file(vec![])), None);
        let files = output_files(&paths, &reviewed, None);
        assert!(body(&files, 0).contains("\u{200f}سلام"));
        let english = reviewed_output(&[row(1, Some("hello"), None, ai(AiChoice::A))]).unwrap();
        let files = output_files(&paths, &english, None);
        assert!(!body(&files, 0).contains('\u{200f}'));
    }

    #[test]
    fn rows_from_align_with_several_cues_on_a_side_save_every_line() {
        let cue = |id, start, end, text: &str| engine::project::TimedText {
            id,
            start,
            end,
            text: text.into(),
        };
        let a = [cue(1, 0.0, 2.0, "one"), cue(2, 2.0, 4.0, "two")];
        let b = [cue(1, 0.0, 4.0, "one two")];
        let rows = engine::subalign::align(&a, &b, 0.0);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].a.as_deref(), Some("one\ntwo"));

        let mut t = table(rows);
        assert!(t.reduce(Action::Pick {
            id: t.file.rows[0].id,
            pick: UserPick::A,
        }));
        let out = reviewed_output(&t.file.rows).unwrap();
        assert_eq!(out.cues[0].text, "one\ntwo");
        assert_eq!(to_txt(&out.cues), "one two\n");
    }

    #[test]
    fn paging_stays_in_range() {
        let mut t = many(PAGE_SIZE as u64 * 2 + 1);
        assert_eq!(t.page_count(), 3);
        assert_eq!(t.page_rows().count(), PAGE_SIZE);
        assert_eq!(t.page_rows().next().unwrap().id, 1);
        t.reduce(Action::PrevPage);
        assert_eq!(t.page, 0);
        t.reduce(Action::NextPage);
        assert_eq!(t.page_rows().next().unwrap().id, PAGE_SIZE as u64 + 1);
        t.reduce(Action::NextPage);
        t.reduce(Action::NextPage);
        assert_eq!(t.page, 2);
        assert_eq!(t.page_rows().count(), 1);

        assert_eq!(table(vec![]).page_count(), 1);
        let mut empty = table(vec![]);
        assert!(!empty.reduce(Action::NextPage));
        assert_eq!(empty.page, 0);
    }

    #[test]
    fn filters_narrow_the_table_and_a_decision_can_empty_a_page() {
        let failed = AiPick {
            status: AiStatus::Failed,
            ..AiPick::default()
        };
        let mut t = table(vec![
            row(1, Some("same"), Some("same"), ai(AiChoice::A)),
            row(2, Some("a"), None, ai(AiChoice::A)),
            row(3, Some("a"), Some("b"), failed),
            row(4, Some("a"), Some("b"), ai(AiChoice::B)),
        ]);
        let ids = |t: &Table| t.page_rows().map(|r| r.id).collect::<Vec<_>>();
        assert_eq!(ids(&t), [1, 2, 3, 4]);
        t.reduce(Action::SetFilter(Filter::Differ));
        assert_eq!(ids(&t), [3, 4]);
        t.reduce(Action::SetFilter(Filter::Undecided));
        assert_eq!(ids(&t), [3]);
        assert!(t.reduce(Action::Pick {
            id: 3,
            pick: UserPick::A
        }));
        assert_eq!(ids(&t), Vec::<u64>::new());
        assert_eq!(t.page, 0);
    }

    #[test]
    fn deciding_the_last_row_of_a_page_moves_back_a_page() {
        let failed = AiPick {
            status: AiStatus::Failed,
            ..AiPick::default()
        };
        let rows = (1..=PAGE_SIZE as u64 + 1)
            .map(|i| row(i, Some("a"), Some("b"), failed.clone()))
            .collect();
        let mut t = table(rows);
        t.reduce(Action::SetFilter(Filter::Undecided));
        t.reduce(Action::NextPage);
        assert_eq!(t.page, 1);
        t.reduce(Action::Pick {
            id: PAGE_SIZE as u64 + 1,
            pick: UserPick::A,
        });
        assert_eq!(t.page, 0);
    }

    #[test]
    fn writes_never_overlap_and_requests_collapse() {
        let mut w = Writer::default();
        assert!(!w.pending());
        assert_eq!(w.begin(), None);

        w.want_json = true;
        assert_eq!(w.begin(), Some(false));
        assert!(w.pending());

        // more changes and a save arrive while that write runs
        w.want_json = true;
        w.want_outputs = true;
        assert_eq!(w.begin(), None);

        w.finish();
        assert_eq!(w.begin(), Some(true), "one write covers both");
        w.finish();
        assert!(!w.pending());
        assert_eq!(w.begin(), None);
    }

    #[test]
    fn save_with_undecided_rows_shows_them_and_writes_nothing() {
        let mut state = State {
            table: Some(table(vec![row(1, Some("a"), Some("b"), AiPick::default())])),
            load: Load::Ready,
            ..State::default()
        };
        request_save(&mut state);
        assert!(!state.writer.want_outputs);
        assert_eq!(state.table.as_ref().unwrap().filter, Filter::Undecided);
        assert!(state.error.as_deref().unwrap().contains(": 1."));
    }

    #[test]
    fn a_change_is_counted_and_queued_for_writing() {
        let mut state = State {
            table: Some(many(1)),
            ..State::default()
        };
        apply(&mut state, Action::NextPage);
        assert_eq!(state.revision, 0);
        assert!(!state.writer.want_json);
        apply(
            &mut state,
            Action::Pick {
                id: 1,
                pick: UserPick::A,
            },
        );
        assert_eq!(state.revision, 1);
        assert!(state.writer.want_json);
    }

    #[test]
    fn coming_back_with_unwritten_picks_keeps_them() {
        let job = JobRef {
            id: 1,
            media: PathBuf::from("a.mp4"),
            paths: ProjectPaths::new("no/such/folder", "a"),
        };
        let mut state = State {
            job: Some(job.clone()),
            load: Load::Ready,
            table: Some(many(1)),
            writer: Writer {
                in_flight: true,
                ..Writer::default()
            },
            ..State::default()
        };
        let generation = state.generation;
        let _ = open(&mut state, &job);
        assert!(state.table.is_some());
        assert_eq!(state.generation, generation);
    }

    #[test]
    fn a_stale_load_is_ignored_and_the_page_comes_back_for_the_same_file() {
        let mut state = State {
            generation: 5,
            restore: Some((PathBuf::from("a.json"), 1, Filter::Differ)),
            ..State::default()
        };
        let answer = |generation| Msg::Loaded {
            generation,
            result: Ok(Some((PathBuf::from("a.json"), many(30).file))),
        };
        let _ = update(&mut state, answer(4));
        assert!(state.table.is_none());
        let _ = update(&mut state, answer(5));
        let table = state.table.as_ref().unwrap();
        assert_eq!((table.page, table.filter), (1, Filter::Differ));
        assert_eq!(state.load, Load::Ready);
    }

    #[test]
    fn playing_a_moment_asks_for_its_time() {
        let job = JobRef {
            id: 8,
            media: PathBuf::from("a.mp4"),
            paths: ProjectPaths::new("out", "a"),
        };
        let mut state = State {
            job: Some(job),
            table: Some(many(3)),
            ..State::default()
        };
        let (_, out) = update(&mut state, Msg::Play(2));
        assert!(matches!(out, Out::PlayAt { job: 8, time } if time == 2.0));
    }

    #[test]
    fn a_failed_write_is_reported_and_clears_the_way_for_the_next() {
        let mut state = State {
            table: Some(many(1)),
            writer: Writer {
                in_flight: true,
                ..Writer::default()
            },
            ..State::default()
        };
        let _ = update(
            &mut state,
            Msg::Written {
                generation: 0,
                outputs: Some((0, 1, 0)),
                result: Err("disk full".into()),
            },
        );
        assert!(state.write_error.as_deref().unwrap().contains("disk full"));
        assert!(!state.writer.in_flight);
        assert_eq!(state.saved, None);

        // the picks are only in memory now, so coming back must not read the older file
        let job = JobRef {
            id: 1,
            media: PathBuf::from("a.mp4"),
            paths: ProjectPaths::new("no/such/folder", "a"),
        };
        state.job = Some(job.clone());
        let _ = open(&mut state, &job);
        assert!(state.table.is_some());
    }

    #[test]
    fn newest_compare_ignores_other_files() {
        let dir = TestDir::new("newest");
        let paths = ProjectPaths::new(&dir.0, "a (mp4)");
        assert_eq!(newest_compare(&paths), None);
        assert_eq!(load_newest(&paths).unwrap(), None);
        std::fs::write(dir.0.join("a (mp4).json"), "{}").unwrap();
        std::fs::write(dir.0.join("a (mp4).import-x.json"), "{}").unwrap();
        std::fs::write(paths.compare("x"), "{not json").unwrap();
        assert_eq!(newest_compare(&paths), Some(paths.compare("x")));
        assert!(load_newest(&paths).is_err());
        assert_eq!(compare_slug("a.compare-.json", "a"), None);
        assert_eq!(compare_slug("a.compare-b-2.json", "a"), Some("b-2"));
        assert_eq!(compare_slug("b.compare-x.json", "a"), None);
    }

    #[test]
    fn labels_say_what_happened() {
        let status = |status| AiPick {
            status,
            ..AiPick::default()
        };
        assert_eq!(ai_label(&status(AiStatus::Identical)), "Same text");
        assert_eq!(ai_label(&status(AiStatus::Failed)), "AI could not decide");
        assert_eq!(ai_label(&status(AiStatus::Unknown)), "");
        assert_eq!(ai_label(&ai(AiChoice::Merged)), "AI pick: merged");
    }

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> TestDir {
            let dir = std::env::temp_dir().join(format!(
                "vipr-review-{name}-{}-{:?}",
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
}
