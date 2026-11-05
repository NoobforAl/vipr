//! The contract between `main.rs` and the feature modules (`player`, `marks_ui`, `profiles_ui`,
//! `import_ui`, `review_ui`, `jobs`).
//!
//! Every module has `State`, `Msg`, `update -> (Task<Msg>, Out)` and `view`. A module changes only
//! its own state and returns an [`Out`] for anything beyond it; `main.rs` acts on that in
//! `App::handle_out`, so owners can fill in their bodies without touching `main.rs`.

use std::path::PathBuf;

use engine::lang::{TextAlign, dir_align};
use engine::project::{CompareSource, ProjectPaths};

/// Which page fills the window. The `u64` is a job id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Home,
    Transcript(u64),
    Player(u64),
    /// Opens the newest compare file of that job (`review_ui::open`).
    Review(u64),
}

/// What a module asks `main.rs` to do after an update.
#[derive(Debug, Clone, Default)]
pub enum Out {
    #[default]
    None,
    /// Go to a page. A job without finished results stays on the current page.
    Open(Page),
    /// Open the player on a job if needed, and seek to `time` seconds.
    PlayAt { job: u64, time: f64 },
    /// Queue background work. There is one worker slot, so this runs after what is running now.
    Enqueue(JobKind),
    /// Tell the user something. Shown on every page for a while, then it goes away by itself.
    Notice(String),
    /// Like `Notice`, for something the user must not miss: it stays until dismissed.
    Alert(String),
}

/// Background work on a finished job's results.
#[derive(Debug, Clone, PartialEq)]
pub enum JobKind {
    /// Writes `<stem>.<target>.translation.json` and the translated outputs.
    Translate {
        job: u64,
        target: String,
        source: TranslateSource,
    },
    /// Writes `<stem>.compare-<slug>.json`, then opens `Page::Review`.
    Compare {
        job: u64,
        a: CompareSource,
        b: CompareSource,
        slug: String,
    },
}

impl JobKind {
    pub fn job(&self) -> u64 {
        match self {
            JobKind::Translate { job, .. } | JobKind::Compare { job, .. } => *job,
        }
    }
}

/// What a translation starts from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranslateSource {
    /// ViPr's own subtitles.
    Transcript,
    /// An imported subtitle file, by slug.
    Import { slug: String },
}

/// A finished job, as modules see it. Everything else is read from the sidecars under `paths`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobRef {
    pub id: u64,
    pub media: PathBuf,
    pub paths: ProjectPaths,
}

/// Where playback is, for marks (`player::position`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Position {
    /// Seconds from the start of the media.
    pub time: f64,
    /// Id of the subtitle cue on screen.
    pub cue: Option<u64>,
}

/// For `container(..).align_x(..)` around content text with default alignment. Setting the alignment
/// on the text itself makes short RTL lines vanish (spikes/rtl).
pub fn content_align(text: &str) -> iced::alignment::Horizontal {
    match dir_align(text) {
        TextAlign::Left => iced::alignment::Horizontal::Left,
        TextAlign::Right => iced::alignment::Horizontal::Right,
    }
}
