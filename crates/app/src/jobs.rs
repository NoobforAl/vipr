//! Background work on finished results: the queue of translate and compare runs, and Open result.
//! Stub from the foundation step; agent E fills it in.
//!
//! Only one worker runs at a time (`pipeline::RUNNING_WORKER`), so `main.rs` shares the slot:
//! `App::start_next` runs waiting transcriptions first, then asks this queue, and starts nothing
//! while [`busy`] is true. Use `pipeline`'s worker runner, so Cancel and closing the window stop it.

// temporary: `view` and `load_result` have no caller until integration
#![allow(dead_code)]

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use engine::profiles::Profile;
use engine::project::ProjectPaths;
use engine::transcript::Transcript;
use iced::widget::text;
use iced::{Element, Task};

use crate::out::{JobKind, JobRef, Out};
use crate::theme::Tokens;

#[derive(Debug, Default)]
pub struct State {
    queue: VecDeque<JobKind>,
}

#[derive(Debug, Clone)]
pub enum Msg {}

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

pub fn enqueue(state: &mut State, kind: JobKind) {
    state.queue.push_back(kind);
}

/// Takes the next queued run.
pub fn next(state: &mut State) -> Option<JobKind> {
    state.queue.pop_front()
}

/// True while a run started by [`start`] has not finished.
pub fn busy(_state: &State) -> bool {
    false
}

/// Starts a run taken from [`next`]. `main.rs` has checked that the worker slot is free.
pub fn start(_state: &mut State, _kind: JobKind, _context: JobContext) -> (Task<Msg>, Out) {
    let notice = "Translate and compare are not available yet.".to_string();
    (Task::none(), Out::Notice(notice))
}

/// `main.rs` calls `App::start_next` after this whenever [`busy`] is false.
pub fn update(_state: &mut State, msg: Msg) -> (Task<Msg>, Out) {
    match msg {}
}

pub fn view(_state: &State, _tokens: Tokens) -> Element<'_, Msg> {
    text("Jobs").into()
}

/// Rebuilds a finished job from a result folder or its `<stem>.json`.
pub fn load_result(_path: &Path) -> Result<LoadedResult, String> {
    Err("Open result is not available yet.".to_string())
}
