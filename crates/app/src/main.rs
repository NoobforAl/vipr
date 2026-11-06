//! Desktop app (Iced). Version 0.1.0 is English only.
//!
//! Layout: a sidebar (wide windows) or top bar (narrow), the file queue with one card per recording,
//! a transcript viewer, and a Settings popup. Work is done by separate worker processes
//! (see `pipeline.rs`); this file only draws and routes messages.

mod fade;
mod icons;
mod import_ui;
mod jobs;
mod marks_ui;
mod out;
mod pipeline;
mod player;
mod profiles_ui;
mod progress;
mod review_ui;
mod saver;
mod theme;

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use engine::download::{ModelStore, Status};
use engine::lang::{LANGUAGES, clean_separators};
use engine::project::ProjectPaths;
use engine::registry::{Fit, ModelEntry, Preset, Registry, Role, fit};
use engine::settings::Settings;
use engine::transcript::{Transcript, clock};
use gpu_detect::Detection;
use iced::futures::channel::mpsc;
use iced::futures::{SinkExt, Stream, StreamExt};
use iced::widget::{
    button, center, column, combo_box, container, mouse_area, opaque, pick_list, progress_bar,
    responsive, row, scrollable, space, stack, svg, text, text_editor, text_input, toggler,
};
use iced::{
    Alignment, Color, Element, Font, Length, Size, Subscription, Task, event, font, keyboard,
    window,
};
use out::{JobRef, Out, Page};
use theme::{DARK, LIGHT, Tokens};

/// Product name (PLAN.md D9). Also used for the data folders, see `engine::APP_DIR_NAME`.
const APP_NAME: &str = "ViPr";
const VERSION: &str = "0.1.0";

/// Models every run needs, listed in Settings.
/// `Role::Translate` is left out: it is optional and follows the cleanup model until one is chosen.
const ROLES: [Role; 5] = [
    Role::Asr,
    Role::Vad,
    Role::DiarSegmentation,
    Role::DiarEmbedding,
    Role::Llm,
];

/// A change of page fades in over this.
const PAGE_FADE: Duration = Duration::from_millis(280);

const CATEGORIES: [(&str, &str); 7] = [
    ("default", "General"),
    ("software-meeting", "Software / dev meeting"),
    ("interview", "Interview"),
    ("lecture-training", "Lecture / training"),
    ("client-sales-call", "Client or sales call"),
    ("podcast-conversation", "Podcast / conversation"),
    ("casual-chat", "Casual chat"),
];

/// File types offered by the file picker. The first group is read by ViPr itself; the second needs ffmpeg.
const MEDIA_EXTENSIONS: &[&str] = &[
    // built-in decoders
    "mp4", "m4v", "mov", "mkv", "webm", "mka", "ogg", "oga", "opus", "mp3", "mp2", "wav", "flac",
    "m4a", "aac", "aif", "aiff", "caf", "3gp", // through ffmpeg
    "avi", "wmv", "wma", "asf", "flv", "ts", "mts", "m2ts", "mpg", "mpeg", "vob", "amr", "ac3",
    "eac3", "dts", "wv", "ape", "spx", "rm", "mxf",
];

/// Extensions worth looking into (the file picker list plus a few rarer names). Files with any other name are ignored.
const KNOWN_EXTENSIONS: &[&str] = &[
    "m4b", "m4p", "wave", "3g2", "ogv", "weba", "f4v", "mpe", "m2v", "m2a", "mp4a", "qt", "tod",
    "dv", "divx", "xvid", "ogx",
];

fn has_media_extension(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .is_some_and(|e| {
            MEDIA_EXTENSIONS.contains(&e.as_str()) || KNOWN_EXTENSIONS.contains(&e.as_str())
        })
}

const PAGE: usize = 150; // transcript rows drawn at once

/// A build without a console would lose a panic's message, so it also goes to `panic.log` in the
/// config folder.
fn log_panics() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Some(dir) = dirs::config_dir().map(|d| d.join(engine::APP_DIR_NAME)) {
            let _ = std::fs::create_dir_all(&dir);
            let log = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join("panic.log"));
            if let Ok(mut log) = log {
                use std::io::Write;
                let trace = std::backtrace::Backtrace::force_capture();
                let _ = writeln!(log, "{info}\n{trace}\n");
            }
        }
        default_hook(info);
    }));
}

fn main() -> iced::Result {
    log_panics();
    iced::application(App::new, App::step, App::view)
        .title(|_: &App| format!("{APP_NAME} {VERSION}"))
        .theme(|app: &App| app.tokens.theme())
        .subscription(App::subscription)
        .window(window_settings())
        .run()
}

/// True on macOS, where the window keeps its native traffic-light buttons. Windows and Linux share one look:
/// we draw the title strip and the minimize / maximize / close buttons ourselves.
const MAC: bool = cfg!(target_os = "macos");

fn window_settings() -> window::Settings {
    let mut w = window::Settings {
        icon: window_icon(),
        size: Size::new(1040.0, 720.0),
        min_size: Some(Size::new(560.0, 460.0)),
        position: window::Position::Centered,
        // closing goes through us, so running workers can be stopped first
        exit_on_close_request: false,
        ..Default::default()
    };
    #[cfg(target_os = "macos")]
    {
        // keep the native traffic lights, but let our content run under a transparent title bar
        w.platform_specific.title_hidden = true;
        w.platform_specific.titlebar_transparent = true;
        w.platform_specific.fullsize_content_view = true;
    }
    #[cfg(not(target_os = "macos"))]
    {
        w.decorations = false;
    }
    #[cfg(target_os = "windows")]
    {
        // like VS Code on Windows 11: rounded corners and the system's drop shadow
        w.platform_specific.corner_preference = window::settings::platform::CornerPreference::Round;
        w.platform_specific.undecorated_shadow = true;
    }
    w
}

/// Radius of the window corners, in px. Windows 11 (build 22000+) rounds them for us; our frame line follows
/// the same curve. Windows 10 and Linux windows stay square.
fn corner_radius() -> f32 {
    static RADIUS: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *RADIUS.get_or_init(|| {
        if cfg!(target_os = "windows") {
            let build = sysinfo::System::kernel_version()
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(0);
            if build >= 22000 { 8.0 } else { 0.0 }
        } else {
            0.0
        }
    })
}

/// The app icon for the title strip (vector, so it stays sharp on any screen).
const APP_ICON_SVG: &[u8] = include_bytes!("../../../assets/icon/vipr.svg");

/// The ViPr icon (assets/icon/vipr-256.png) as the window / taskbar icon.
fn window_icon() -> Option<window::Icon> {
    let mut reader = png::Decoder::new(std::io::Cursor::new(include_bytes!(
        "../../../assets/icon/vipr-256.png"
    )))
    .read_info()
    .ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    if info.color_type != png::ColorType::Rgba || info.bit_depth != png::BitDepth::Eight {
        return None;
    }
    buf.truncate(info.buffer_size());
    window::icon::from_rgba(buf, info.width, info.height).ok()
}

// ------------------------------------------------------------------ small types

#[derive(Debug, Clone, PartialEq, Eq)]
struct Choice {
    id: String,
    label: String,
}

impl fmt::Display for Choice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label)
    }
}

#[derive(Debug, Clone)]
enum DownloadEvent {
    Progress { done: u64, total: u64 },
    Finished(Result<(), String>),
}

#[derive(Default, Clone, Copy)]
struct Progress {
    done: u64,
    total: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum JobState {
    Waiting,
    Running,
    Done,
    Failed,
}

struct Job {
    id: u64,
    path: PathBuf,
    size: u64,
    state: JobState,
    steps: Vec<String>,
    step: usize,
    detail: String,
    step_progress: f32,
    progress: f32,
    /// `progress` as drawn: it glides there.
    shown_progress: f32,
    started: Option<Instant>,
    cancelled: bool,
    notes: Vec<String>,
    expanded: bool,
    about: pipeline::About,
    /// Speaker names as typed (comma separated, in order of first speaking).
    names_text: String,
    transcript: Option<Transcript>,
    outputs: Vec<PathBuf>,
    /// Where the results are, with the real stem. Set when the job finishes.
    paths: Option<ProjectPaths>,
    error: Option<String>,
    /// Opened from saved results and its media is not where it was.
    media_missing: bool,
}

impl Job {
    fn new(id: u64, path: PathBuf, size: u64) -> Job {
        Job {
            id,
            path,
            size,
            state: JobState::Waiting,
            steps: Vec::new(),
            step: 0,
            detail: String::new(),
            step_progress: 0.0,
            progress: 0.0,
            shown_progress: 0.0,
            started: None,
            cancelled: false,
            notes: Vec::new(),
            expanded: false,
            about: pipeline::About {
                category: "default".into(),
                ..Default::default()
            },
            names_text: String::new(),
            transcript: None,
            outputs: Vec::new(),
            paths: None,
            error: None,
            media_missing: false,
        }
    }

    /// The file name, safe to show. It comes from the disk, so it can hold anything.
    fn name(&self) -> String {
        file_name(&self.path)
    }
}

/// A path's last part, cleaned for the text layout.
fn file_name(path: &std::path::Path) -> String {
    path.file_name()
        .map(|n| clean_separators(&n.to_string_lossy()))
        .unwrap_or_default()
}

/// A message from a feature module, shown on every page.
#[derive(Debug, Clone)]
struct Notice {
    text: String,
    expires: Instant,
}

/// How long an `Out::Notice` stays up.
const NOTICE_TIME: Duration = Duration::from_secs(5);
const MAX_NOTICES: usize = 8;
/// The message in the corner of the window.
const TOAST_WIDTH: f32 = 380.0;
const TOAST_MARGIN: f32 = 16.0;

/// Progress of the one-click ffmpeg download.
#[derive(Debug, Clone)]
enum FfmpegEvent {
    Downloading { done: u64, total: u64 },
    Unpacking,
    Finished(Result<PathBuf, String>),
}

#[derive(Debug, Clone, Copy, Default)]
struct FfmpegProgress {
    done: u64,
    total: u64,
    unpacking: bool,
}

/// An entry of the translate language picker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LanguageChoice(&'static engine::lang::Language);

impl fmt::Display for LanguageChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.0.name, self.0.code)
    }
}

/// A row's selectable text. Text comes in cleaned for a widget or is cleaned here, so it can't break the layout.
fn row_content(line: &str) -> text_editor::Content {
    text_editor::Content::with_text(&clean_separators(line))
}

/// The languages a transcript can be translated to. Not its own: that would fail in the worker.
fn translate_options(source: Option<&str>) -> Vec<LanguageChoice> {
    LANGUAGES
        .iter()
        .filter(|l| Some(l.code) != source)
        .map(LanguageChoice)
        .collect()
}

/// Which copy button was pressed last (it shows "Copied" for a moment).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CopyTarget {
    All,
    Row(usize),
}

#[derive(Debug, Clone)]
enum Message {
    SystemTheme(iced::theme::Mode),
    Usage(Usage),
    /// Hardware detection finished (runs in the background so the window opens at once).
    Detected(Option<Detection>),
    /// Animation clock for spinners; only runs while something is loading.
    Tick,
    // window
    WinId(Option<window::Id>),
    Minimize,
    ToggleMaximize,
    Maximized(bool),
    ToggleFullscreen,
    DragWindow,
    ResizeEdge(window::Direction),
    CloseWindow,
    Cancel(u64),
    OpenSettings,
    CloseSettings,
    /// Closes whichever dialog is open.
    Escape,
    // queue
    AddFiles,
    /// A file dropped on the window: for the import dialog while it is open, else for the queue.
    FileDropped(PathBuf),
    /// Files from the picker, drag and drop (or MT_PRELOAD): they are checked before they are added.
    FilesPicked(Vec<PathBuf>),
    FilesChecked(Vec<(PathBuf, engine::audio::Probe)>),
    DismissNotice,
    Remove(u64),
    ToggleDetails(u64),
    Description(u64, String),
    Category(u64, Choice),
    SpeakerNames(u64, String),
    StartAll,
    JobEvent(u64, pipeline::Event),
    // transcript page
    OpenJob(u64),
    Back,
    /// Go to a page, as a button asks for it.
    Go(Page),
    OpenImport(u64),
    TranslateTarget(LanguageChoice),
    /// The translate box lost focus: whatever was typed there is dropped.
    TranslateBoxClosed,
    /// A job's saved translations were read, and which of them to show, by name.
    TranslationsLoaded {
        job: u64,
        translations: Vec<player::TranslatedLines>,
        show: Option<String>,
    },
    /// Show this translation on the transcript page. `None` is the original text.
    ViewLanguage(Option<usize>),
    Translate(u64),
    // saved results
    OpenResult {
        folder: bool,
    },
    ResultPicked(Option<PathBuf>),
    ResultLoaded(Result<jobs::LoadedResult, String>),
    LocateMedia(u64),
    MediaLocated(u64, Option<PathBuf>),
    MoreRows,
    OpenFolder(u64),
    CopyText(u64),
    /// Mouse / keyboard action inside one message of the transcript (selecting text).
    RowAction(usize, text_editor::Action),
    CopyRow(usize),
    RenameSpeaker(usize, String),
    SaveNames,
    // settings
    SetTheme(String),
    ChooseOutput,
    ChooseFfmpeg,
    GetFfmpeg,
    FfmpegUpdate(FfmpegEvent),
    RemoveFfmpeg,
    FfmpegChosen(Option<PathBuf>),
    ResetFfmpeg,
    OutputChosen(Option<PathBuf>),
    ResetOutput,
    ToggleCleanup(bool),
    SetSpeakers(usize),
    Pick(Role, Choice),
    Download(String),
    DownloadAllMissing,
    DownloadUpdate(String, DownloadEvent),
    // feature modules (see out.rs)
    /// A key press that no widget used, so typing in a field can't trigger a shortcut.
    Key(iced::keyboard::Key, iced::keyboard::Modifiers),
    Player(player::Msg),
    Marks(marks_ui::Msg),
    /// The screen is ready for the next step of whatever is moving.
    Frame(Instant),
    Profiles(profiles_ui::Msg),
    Import(import_ui::Msg),
    Review(review_ui::Msg),
    Jobs(jobs::Msg),
}

// ------------------------------------------------------------------ app state

struct App {
    /// `None` until the background detection has finished.
    detection: Option<Detection>,
    epoch: Instant,
    spin: f32,
    registry: Result<Registry, String>,
    store: ModelStore,
    selection: HashMap<Role, String>,
    downloading: HashMap<String, Progress>,
    errors: HashMap<String, String>,
    settings: Settings,
    system_dark: bool,
    tokens: Tokens,
    settings_open: bool,
    page: Page,
    jobs: Vec<Job>,
    next_id: u64,
    running: Option<u64>,
    shown: usize,
    name_edits: Vec<String>,
    usage: Option<Usage>,
    win: Option<window::Id>,
    maximized: bool,
    fullscreen: bool,
    /// ffmpeg (optional, for more file formats), looked up at start and whenever Settings opens
    ffmpeg: Option<PathBuf>,
    /// One read-only, selectable text view per transcript message that is on screen.
    rows: Vec<text_editor::Content>,
    copied: Option<(CopyTarget, Instant)>,
    /// Files that were not added, with the reason; shown for a few seconds.
    notice: Option<(Vec<String>, Instant)>,
    /// Messages from the feature modules.
    notices: Vec<Notice>,
    /// A compare that finished while the user was busy: it waits behind a button instead of taking over.
    review_ready: Option<u64>,
    /// Jobs known to have a compare file, for the Review button.
    reviewable: HashSet<u64>,
    /// The language picked for "Translate" on the result page.
    translate_target: Option<LanguageChoice>,
    /// The saved translations of the transcript page's job.
    translations: Vec<player::TranslatedLines>,
    /// The one of `translations` the page shows. `None` is the original text.
    viewing: Option<usize>,
    /// What the translate box searches, and the transcript language it was made for. See [`App::sync_translate_box`].
    translate_box: combo_box::State<LanguageChoice>,
    translate_box_source: Option<Option<&'static str>>,
    /// True from pressing Start until the user goes anywhere else: the player may open by itself.
    auto_open: bool,
    /// ffmpeg download in progress, or the last error
    ffmpeg_dl: Option<FfmpegProgress>,
    ffmpeg_err: Option<String>,
    /// Files that were skipped only because they need ffmpeg; they are added once ffmpeg is ready.
    waiting_for_ffmpeg: Vec<PathBuf>,
    player: player::State,
    marks: marks_ui::State,
    /// A page change is fading in: since when, and the old page's background.
    page_fade: Option<(Instant, iced::Color)>,
    profiles: profiles_ui::State,
    import: import_ui::State,
    review: review_ui::State,
    queue: jobs::State,
}

impl App {
    fn new() -> (Self, Task<Message>) {
        let registry = Registry::load_default(None).map_err(|e| e.to_string());
        let settings = Settings::load();
        // saved choices now; the recommended models for the other roles are filled in once the hardware is known
        let mut selection = HashMap::new();
        if let Ok(reg) = &registry {
            for role in ROLES {
                if let Some(id) = settings
                    .models
                    .get(&role)
                    .filter(|id| reg.get(id).is_ok_and(|m| m.has_role(role)))
                {
                    selection.insert(role, id.clone());
                }
            }
            if let Some(id) = settings
                .models
                .get(&Role::Translate)
                .filter(|id| reg.get(id).is_ok_and(can_translate))
            {
                selection.insert(Role::Translate, id.clone());
            }
        }
        let mut app = App {
            detection: None,
            epoch: Instant::now(),
            spin: 0.0,
            registry,
            store: ModelStore::new(ModelStore::default_dir()),
            selection,
            downloading: HashMap::new(),
            errors: HashMap::new(),
            settings,
            system_dark: true,
            tokens: DARK,
            settings_open: false,
            page: Page::Home,
            jobs: Vec::new(),
            next_id: 1,
            running: None,
            shown: PAGE,
            name_edits: Vec::new(),
            usage: None,
            win: None,
            maximized: false,
            fullscreen: false,
            ffmpeg: engine::audio::find_ffmpeg(),
            rows: Vec::new(),
            copied: None,
            notice: None,
            notices: Vec::new(),
            review_ready: None,
            reviewable: HashSet::new(),
            translate_target: None,
            translations: Vec::new(),
            viewing: None,
            translate_box: combo_box::State::default(),
            translate_box_source: None,
            auto_open: false,
            ffmpeg_dl: None,
            ffmpeg_err: None,
            waiting_for_ffmpeg: Vec::new(),
            player: player::State::default(),
            marks: marks_ui::State::default(),
            page_fade: None,
            profiles: profiles_ui::State::default(),
            import: import_ui::State::default(),
            review: review_ui::State::default(),
            queue: jobs::State::default(),
        };
        app.recompute_tokens();
        engine::audio::set_ffmpeg_override(app.settings.ffmpeg_path.clone());
        app.ffmpeg = engine::audio::find_ffmpeg();
        // dev helper: MT_PRELOAD="a.mp4;b.mp4" queues files at start-up
        let preload: Vec<PathBuf> = std::env::var("MT_PRELOAD")
            .ok()
            .map(|v| {
                v.split(';')
                    .filter(|p| !p.is_empty())
                    .map(PathBuf::from)
                    .collect()
            })
            .unwrap_or_default();
        let mut tasks = vec![
            iced::system::theme().map(Message::SystemTheme),
            window::oldest().map(Message::WinId),
            Task::perform(detect_async(), Message::Detected),
        ];
        if !preload.is_empty() {
            tasks.push(Task::done(Message::FilesPicked(preload)));
        }
        let (task, out) = profiles_ui::init(&mut app.profiles);
        tasks.push(task.map(Message::Profiles));
        tasks.push(app.handle_out(out));
        (app, Task::batch(tasks))
    }

    fn recompute_tokens(&mut self) {
        self.tokens = match self.settings.theme.as_str() {
            "light" => LIGHT,
            "dark" => DARK,
            _ => {
                if self.system_dark {
                    DARK
                } else {
                    LIGHT
                }
            }
        };
    }

    fn save_settings(&mut self) {
        self.settings.models = self
            .selection
            .iter()
            .map(|(r, id)| (*r, id.clone()))
            .collect();
        // on the save thread, so that a busy disk can't freeze the window
        let settings = self.settings.clone();
        saver::run(move || {
            let _ = settings.save(); // a failed save must not disturb the user
        });
    }

    fn output_dir(&self) -> PathBuf {
        self.settings
            .output_dir
            .clone()
            .unwrap_or_else(Settings::default_output_dir)
    }

    fn entry(&self, role: Role) -> Option<&ModelEntry> {
        let reg = self.registry.as_ref().ok()?;
        reg.get(self.selection.get(&role)?).ok()
    }

    /// Path of the chosen model for `role` if it is downloaded.
    fn model_path(&self, role: Role) -> Option<PathBuf> {
        let m = self.entry(role)?;
        matches!(self.store.status(m), Status::Present).then(|| self.store.path_for(m))
    }

    /// The model translations use: the one chosen for them, else the cleanup model. A chosen model
    /// that isn't downloaded yet gives `None`, not a silent switch to a different model.
    fn translate_model_path(&self) -> Option<PathBuf> {
        self.model_path(translate_role(
            self.selection.contains_key(&Role::Translate),
        ))
    }

    fn selected_entries(&self) -> Vec<&ModelEntry> {
        ROLES
            .iter()
            .chain(std::iter::once(&Role::Translate))
            .filter_map(|r| self.entry(*r))
            .collect()
    }

    /// Speech recognition and silence detection are required to transcribe.
    fn can_transcribe(&self) -> bool {
        self.model_path(Role::Asr).is_some() && self.model_path(Role::Vad).is_some()
    }

    /// True while something on screen is waiting for data (drives the spinner animation).
    fn is_loading(&self) -> bool {
        self.detection.is_none()
            || (self.settings_open && self.usage.is_none())
            || self.running.is_some()
            || self.copied.is_some()
            || self.notice.is_some()
            || !self.notices.is_empty()
            || self.ffmpeg_dl.is_some()
            || jobs::busy(&self.queue)
    }

    /// Rotating arc, for "working on it" states.
    fn spinner<'a>(&self, size: f32, color: Color) -> Element<'a, Message> {
        icons::spinner(self.spin, size, color)
    }

    /// Grey placeholder bar shown where data is about to appear.
    fn skeleton<'a>(&self, width: Length, height: f32) -> Element<'a, Message> {
        container(space::horizontal())
            .width(width)
            .height(Length::Fixed(height))
            .style(theme::rounded_fill(self.tokens.hover, 6.0))
            .into()
    }

    fn meter_placeholder<'a>(&self, label: &'a str) -> Element<'a, Message> {
        let t = self.tokens;
        row![
            text(label)
                .size(13)
                .color(t.faint)
                .width(Length::Fixed(120.0)),
            self.skeleton(Length::Fill, 8.0),
            self.skeleton(Length::Fixed(70.0), 8.0),
        ]
        .spacing(12)
        .align_y(Alignment::Center)
        .into()
    }

    fn job(&self, id: u64) -> Option<&Job> {
        self.jobs.iter().find(|j| j.id == id)
    }

    fn job_mut(&mut self, id: u64) -> Option<&mut Job> {
        self.jobs.iter_mut().find(|j| j.id == id)
    }

    /// A job with finished results, as the feature modules see it.
    fn job_ref(&self, id: u64) -> Option<JobRef> {
        let job = self.job(id)?;
        Some(JobRef {
            id,
            media: job.path.clone(),
            paths: job.paths.clone()?,
        })
    }

    // -------------------------------------------------------------- feature modules

    /// Do what a feature module asked for (see out.rs).
    fn handle_out(&mut self, out: Out) -> Task<Message> {
        match out {
            Out::None => Task::none(),
            Out::Open(page) => self.open_page(page),
            Out::PlayAt { job, time } => {
                let opened = if self.page == Page::Player(job) {
                    Task::none()
                } else {
                    self.open_page(Page::Player(job))
                };
                if self.page != Page::Player(job) {
                    return opened;
                }
                let seek = self.update(Message::Player(player::Msg::SeekTo(time)));
                Task::batch([opened, seek])
            }
            Out::Enqueue(kind) => {
                jobs::enqueue(&mut self.queue, kind);
                self.start_next()
            }
            Out::Notice(line) => {
                self.tell(&line);
                Task::none()
            }
            Out::Alert(line) => {
                self.tell(&line);
                Task::none()
            }
        }
    }

    /// All notice text passes here: errors can quote paths and file content.
    fn tell(&mut self, line: &str) {
        if self.notices.len() == MAX_NOTICES {
            self.notices.remove(0);
        }
        self.notices.push(Notice {
            text: clean_separators(line),
            expires: Instant::now() + NOTICE_TIME,
        });
    }

    /// True when a result that just finished may take over the window: the user is not playing,
    /// reviewing, in a dialog, or looking at another recording.
    fn can_take_over(&self, job: u64) -> bool {
        !self.settings_open
            && !import_ui::is_open(&self.import)
            && (self.page == Page::Home || self.page == Page::Transcript(job))
    }

    /// `jobs::update`'s answer. A finished run opens its page only when that interrupts nothing.
    fn handle_job_out(&mut self, out: Out) -> Task<Message> {
        match out {
            Out::Open(Page::Review(job)) if !self.can_take_over(job) => {
                self.review_ready = Some(job);
                Task::none()
            }
            out => self.handle_out(out),
        }
    }

    /// Go to a page, opening or closing the feature modules it needs. A job without finished
    /// results stays on the current page.
    fn open_page(&mut self, page: Page) -> Task<Message> {
        let job = match page {
            Page::Player(id) | Page::Review(id) => match self.job_ref(id) {
                Some(job) => Some(job),
                None => return Task::none(),
            },
            Page::Home | Page::Transcript(_) => None,
        };
        if self.page != page {
            self.auto_open = false;
        }
        if matches!(page, Page::Review(id) if self.review_ready == Some(id)) {
            self.review_ready = None;
        }
        if matches!(self.page, Page::Player(_)) && self.page != page {
            player::close(&mut self.player);
            marks_ui::flush(&mut self.marks);
        }
        match (page, job) {
            (Page::Transcript(id), _) => self.update(Message::OpenJob(id)),
            (Page::Player(_), Some(job)) => {
                self.page = page;
                let (task, out) = player::open(&mut self.player, &job);
                let (marks_task, marks_out) = marks_ui::open(&mut self.marks, &job);
                Task::batch([
                    task.map(Message::Player),
                    marks_task.map(Message::Marks),
                    self.handle_out(out),
                    self.handle_out(marks_out),
                ])
            }
            (Page::Review(_), Some(job)) => {
                self.page = page;
                let (task, out) = review_ui::open(&mut self.review, &job);
                Task::batch([task.map(Message::Review), self.handle_out(out)])
            }
            _ => {
                self.page = page;
                Task::none()
            }
        }
    }

    /// A key press that no widget used. Pages get them only while no dialog is open.
    fn route_key(
        &mut self,
        key: iced::keyboard::Key,
        mods: iced::keyboard::Modifiers,
    ) -> Task<Message> {
        if self.settings_open || import_ui::is_open(&self.import) {
            return Task::none();
        }
        let message = match self.page {
            Page::Player(_) if player::is_playing(&self.player) => {
                player::key(&self.player, &key, mods)
                    .map(Message::Player)
                    .or_else(|| {
                        let at = player::position(&self.player);
                        marks_ui::key(&self.marks, &key, mods, at).map(Message::Marks)
                    })
            }
            Page::Player(_) => None,
            Page::Review(_) => review_ui::key(&self.review, &key, mods).map(Message::Review),
            Page::Home | Page::Transcript(_) => None,
        };
        match message {
            Some(message) => self.update(message),
            None => Task::none(),
        }
    }

    /// The video and its controls, next to the marks.
    fn player_page(&self, id: u64) -> Element<'_, Message> {
        let t = theme::PLAYER;
        let marks = marks_ui::marks(&self.marks);
        let finding_chapters = jobs::pending(&self.queue, &out::JobKind::Chapters { job: id });
        let name = self.job(id).map(Job::name).unwrap_or_default();
        let mut top = row![
            button(icons::icon(icons::BACK, 16.0, t.text))
                .padding(5)
                .style(theme::flat(t, false))
                .on_press(Message::Go(Page::Transcript(id))),
            text(name).size(13).color(t.muted).width(Length::Fill),
        ]
        .spacing(8)
        .padding([2, 8])
        .align_y(Alignment::Center);
        if self.reviewable.contains(&id) {
            top = top.push(
                button(text("Review").size(12))
                    .padding([3, 10])
                    .style(theme::flat(t, false))
                    .on_press(Message::Go(Page::Review(id))),
            );
        }
        let mut video = column![
            container(top).width(Length::Fill).style(theme::strip(t)),
            player::view(&self.player, t, marks, finding_chapters).map(Message::Player)
        ]
        .width(Length::Fill)
        .height(Length::Fill);
        if player::needs_ffmpeg(&self.player) {
            video = video.push(container(self.ffmpeg_offer()).padding(12));
        }
        let at = player::is_playing(&self.player).then(|| player::position(&self.player));
        let panel = marks_ui::view(
            &self.marks,
            t,
            at,
            player::cues(&self.player),
            player::current_cue(&self.player),
        );
        row![video, panel.map(Message::Marks)].into()
    }

    // -------------------------------------------------------------- pipeline control

    /// Run a queued translate or compare. Called when no transcription is waiting.
    fn start_queued(&mut self) -> Task<Message> {
        let Some(kind) = jobs::next(&mut self.queue) else {
            return Task::none();
        };
        let model = match kind {
            out::JobKind::Translate { .. } => self.translate_model_path(),
            out::JobKind::Compare { .. } | out::JobKind::Chapters { .. } => {
                self.model_path(Role::Llm)
            }
        };
        let context = jobs::JobContext {
            job: self.job_ref(kind.job()),
            model,
            profile: profiles_ui::active(&self.profiles, &self.settings),
        };
        let (task, out) = jobs::start(&mut self.queue, kind, context);
        Task::batch([task.map(Message::Jobs), self.handle_out(out)])
    }

    fn start_next(&mut self) -> Task<Message> {
        if self.running.is_some() || jobs::busy(&self.queue) {
            return Task::none();
        }
        let Some(idx) = self.jobs.iter().position(|j| j.state == JobState::Waiting) else {
            return self.start_queued();
        };
        let (Some(asr), Some(vad)) = (self.model_path(Role::Asr), self.model_path(Role::Vad))
        else {
            self.jobs[idx].state = JobState::Failed;
            self.jobs[idx].error = Some("Download the speech models in Settings first.".into());
            return self.start_next();
        };
        let diar_models = self
            .model_path(Role::DiarSegmentation)
            .zip(self.model_path(Role::DiarEmbedding));
        let llm_model = if self.settings.cleanup {
            self.model_path(Role::Llm)
        } else {
            None
        };
        let out_dir = self.output_dir();
        let num_speakers = self.settings.num_speakers;

        let job = &mut self.jobs[idx];
        job.about.speaker_names = job
            .names_text
            .split(',')
            .map(|n| n.trim().to_string())
            .collect();
        job.state = JobState::Running;
        job.steps.clear();
        job.step = 0;
        job.detail = "Starting".into();
        job.step_progress = 0.0;
        job.progress = 0.0;
        job.shown_progress = 0.0;
        job.started = Some(Instant::now());
        job.cancelled = false;
        job.notes.clear();
        job.error = None;
        job.expanded = false;
        let id = job.id;
        let mut input = pipeline::JobInput {
            path: job.path.clone(),
            out_dir,
            about: job.about.clone(),
            num_speakers,
            asr_model: asr,
            vad_model: vad,
            diar_models,
            llm_model,
            glossary: pipeline::find_glossary(),
        };
        self.running = Some(id);
        pipeline::reset_cancel();
        if let Some(profile) = profiles_ui::active(&self.profiles, &self.settings)
            && let Err(e) = pipeline::apply_profile(&mut input, &profile)
        {
            self.tell(&format!(
                "The active profile could not be used for this run: {e}"
            ));
        }
        Task::run(job_stream(input), move |ev| Message::JobEvent(id, ev))
    }

    fn start_download(&mut self, id: &str) -> Task<Message> {
        if self.downloading.contains_key(id) {
            return Task::none();
        }
        let Some(entry) = self
            .registry
            .as_ref()
            .ok()
            .and_then(|r| r.get(id).ok())
            .cloned()
        else {
            return Task::none();
        };
        self.errors.remove(id);
        self.downloading.insert(
            id.to_string(),
            Progress {
                done: 0,
                total: entry.size_bytes,
            },
        );
        let id = id.to_string();
        let stream = download_stream(
            entry,
            self.store.dir().to_path_buf(),
            std::env::var("HF_TOKEN").ok(),
        );
        Task::run(stream, move |ev| Message::DownloadUpdate(id.clone(), ev))
    }

    // -------------------------------------------------------------- update

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::SystemTheme(mode) => {
                self.system_dark = !matches!(mode, iced::theme::Mode::Light);
                self.recompute_tokens();
                Task::none()
            }
            Message::Detected(d) => {
                self.detection = d;
                if let (Some(det), Ok(reg)) = (&self.detection, &self.registry) {
                    for role in ROLES {
                        if !self.selection.contains_key(&role)
                            && let Some(m) = reg.recommend(role, Preset::Balanced, det)
                        {
                            self.selection.insert(role, m.id.clone());
                        }
                    }
                }
                Task::none()
            }
            Message::Tick => {
                self.spin = icons::spin_angle(self.epoch.elapsed().as_secs_f32());
                jobs::tick(&mut self.queue);
                if let Some(id) = self.running
                    && let Some(j) = self.job_mut(id)
                {
                    j.shown_progress = progress::approach(j.shown_progress, j.progress);
                }
                if self
                    .copied
                    .is_some_and(|(_, at)| at.elapsed() > Duration::from_millis(1600))
                {
                    self.copied = None;
                }
                // the notice stays while it offers the ffmpeg download
                if self.waiting_for_ffmpeg.is_empty()
                    && self.ffmpeg_dl.is_none()
                    && self
                        .notice
                        .as_ref()
                        .is_some_and(|(_, at)| at.elapsed() > NOTICE_TIME)
                {
                    self.notice = None;
                }
                let now = Instant::now();
                self.notices.retain(|n| n.expires > now);
                Task::none()
            }
            Message::WinId(id) => {
                self.win = id;
                Task::none()
            }
            Message::Minimize => match self.win {
                Some(id) => window::minimize(id, true),
                None => Task::none(),
            },
            Message::ToggleMaximize => match self.win {
                Some(id) => window::toggle_maximize(id)
                    .chain(window::is_maximized(id).map(Message::Maximized)),
                None => Task::none(),
            },
            Message::Maximized(m) => {
                self.maximized = m;
                Task::none()
            }
            Message::ToggleFullscreen => match self.win {
                Some(id) => {
                    self.fullscreen = !self.fullscreen;
                    window::set_mode(
                        id,
                        if self.fullscreen {
                            window::Mode::Fullscreen
                        } else {
                            window::Mode::Windowed
                        },
                    )
                }
                None => Task::none(),
            },
            Message::DragWindow => match self.win {
                Some(id) => window::drag(id),
                None => Task::none(),
            },
            Message::ResizeEdge(dir) => match self.win {
                Some(id) => window::drag_resize(id, dir),
                None => Task::none(),
            },
            Message::CloseWindow => {
                // do not leave a worker running (and the GPU busy) after the window is gone
                pipeline::request_cancel();
                player::close(&mut self.player);
                marks_ui::flush(&mut self.marks);
                profiles_ui::flush(&mut self.profiles);
                // the last click's save has to land before the process goes
                saver::wait();
                // an open row editor ends with a write, which has to finish before the window goes
                let (write, _) = review_ui::update(&mut self.review, review_ui::Msg::Back);
                let write = write.map(Message::Review);
                match self.win {
                    Some(id) => write.chain(window::close(id)),
                    None => write.chain(iced::exit()),
                }
            }
            Message::Cancel(id) => {
                if self.running == Some(id) {
                    if let Some(j) = self.job_mut(id) {
                        j.cancelled = true;
                    }
                    pipeline::request_cancel();
                }
                Task::none()
            }
            Message::Usage(u) => {
                self.usage = Some(u);
                Task::none()
            }
            Message::OpenSettings => {
                self.settings_open = true;
                self.auto_open = false;
                self.ffmpeg = engine::audio::find_ffmpeg();
                Task::none()
            }
            Message::CloseSettings => {
                self.settings_open = false;
                profiles_ui::flush(&mut self.profiles);
                Task::none()
            }
            Message::Escape => {
                if import_ui::is_open(&self.import) {
                    self.update(Message::Import(import_ui::Msg::Close))
                } else if self.settings_open {
                    self.update(Message::CloseSettings)
                } else {
                    Task::none()
                }
            }

            Message::AddFiles => Task::perform(pick_files(), Message::FilesPicked),
            Message::FileDropped(path) => {
                if import_ui::is_open(&self.import) {
                    self.update(Message::Import(import_ui::Msg::FileDropped(path)))
                } else {
                    self.update(Message::FilesPicked(vec![path]))
                }
            }
            Message::FilesPicked(files) => {
                // anything that is not a readable video / audio file is ignored (see FilesChecked)
                let files: Vec<PathBuf> = files
                    .into_iter()
                    .filter(|f| !self.jobs.iter().any(|j| &j.path == f))
                    .collect();
                if files.is_empty() {
                    return Task::none();
                }
                Task::perform(check_files(files), Message::FilesChecked)
            }
            Message::FilesChecked(results) => {
                let mut skipped: Vec<String> = Vec::new();
                for (f, verdict) in results {
                    let name = match file_name(&f) {
                        name if name.is_empty() => clean_separators(&f.display().to_string()),
                        name => name,
                    };
                    match verdict {
                        engine::audio::Probe::Ok => {
                            if self.jobs.iter().any(|j| j.path == f) {
                                continue;
                            }
                            let size = std::fs::metadata(&f).map(|m| m.len()).unwrap_or(0);
                            let id = self.next_id;
                            self.next_id += 1;
                            self.jobs.push(Job::new(id, f, size));
                        }
                        engine::audio::Probe::NeedsFfmpeg => {
                            skipped.push(format!("{name}: this format needs ffmpeg"));
                            if !self.waiting_for_ffmpeg.contains(&f) {
                                self.waiting_for_ffmpeg.push(f);
                            }
                        }
                        engine::audio::Probe::Invalid(why) => {
                            skipped.push(format!("{name}: {}", clean_separators(&why)))
                        }
                    }
                }
                if !skipped.is_empty() {
                    let mut lines = self.notice.take().map(|(l, _)| l).unwrap_or_default();
                    lines.extend(skipped);
                    lines.truncate(8);
                    self.notice = Some((lines, Instant::now()));
                }
                // adding a file must not interrupt playback or a review
                if !matches!(self.page, Page::Player(_) | Page::Review(_)) {
                    self.page = Page::Home;
                }
                Task::none()
            }
            Message::DismissNotice => {
                self.notice = None;
                self.notices.clear();
                self.review_ready = None;
                if self.ffmpeg_dl.is_none() {
                    self.waiting_for_ffmpeg.clear();
                    self.ffmpeg_err = None;
                }
                Task::none()
            }
            Message::GetFfmpeg => {
                if self.ffmpeg_dl.is_some() {
                    return Task::none();
                }
                self.ffmpeg_err = None;
                self.ffmpeg_dl = Some(FfmpegProgress::default());
                Task::run(ffmpeg_stream(), Message::FfmpegUpdate)
            }
            Message::FfmpegUpdate(ev) => match ev {
                FfmpegEvent::Downloading { done, total } => {
                    self.ffmpeg_dl = Some(FfmpegProgress {
                        done,
                        total,
                        unpacking: false,
                    });
                    Task::none()
                }
                FfmpegEvent::Unpacking => {
                    let (done, total) = self.ffmpeg_dl.map_or((0, 0), |p| (p.done, p.total));
                    self.ffmpeg_dl = Some(FfmpegProgress {
                        done,
                        total,
                        unpacking: true,
                    });
                    Task::none()
                }
                FfmpegEvent::Finished(Ok(_)) => {
                    self.ffmpeg_dl = None;
                    self.ffmpeg = engine::audio::find_ffmpeg();
                    // add the files that were waiting for it
                    let files = std::mem::take(&mut self.waiting_for_ffmpeg);
                    self.notice = None;
                    let waiting = if files.is_empty() {
                        Task::none()
                    } else {
                        Task::done(Message::FilesPicked(files))
                    };
                    if player::needs_ffmpeg(&self.player) {
                        let retry = self.update(Message::Player(player::Msg::Retry));
                        Task::batch([waiting, retry])
                    } else {
                        waiting
                    }
                }
                FfmpegEvent::Finished(Err(e)) => {
                    self.ffmpeg_dl = None;
                    self.ffmpeg_err = Some(e);
                    Task::none()
                }
            },
            Message::RemoveFfmpeg => {
                let _ = engine::tools::remove_installed_ffmpeg();
                self.ffmpeg = engine::audio::find_ffmpeg();
                Task::none()
            }
            Message::Remove(id) => {
                if self.running != Some(id) {
                    self.jobs.retain(|j| j.id != id);
                    if matches!(
                        self.page,
                        Page::Transcript(p) | Page::Player(p) | Page::Review(p) if p == id
                    ) {
                        return self.open_page(Page::Home);
                    }
                }
                Task::none()
            }
            Message::ToggleDetails(id) => {
                if let Some(j) = self.job_mut(id) {
                    j.expanded = !j.expanded;
                }
                Task::none()
            }
            Message::Description(id, v) => {
                if let Some(j) = self.job_mut(id) {
                    j.about.description = clean_separators(&v);
                }
                Task::none()
            }
            Message::Category(id, c) => {
                if let Some(j) = self.job_mut(id) {
                    j.about.category = c.id;
                }
                Task::none()
            }
            Message::SpeakerNames(id, v) => {
                if let Some(j) = self.job_mut(id) {
                    j.names_text = clean_separators(&v);
                }
                Task::none()
            }
            Message::StartAll => {
                // failed jobs can be run again
                for j in &mut self.jobs {
                    if matches!(j.state, JobState::Failed) {
                        j.state = JobState::Waiting;
                    }
                }
                self.auto_open = self.page == Page::Home && !self.settings_open;
                self.start_next()
            }
            Message::JobEvent(id, ev) => {
                match ev {
                    pipeline::Event::Plan(steps) => {
                        if let Some(j) = self.job_mut(id) {
                            j.steps = steps;
                            j.step = 0;
                        }
                    }
                    pipeline::Event::Step(i) => {
                        if let Some(j) = self.job_mut(id) {
                            j.step = i;
                            j.step_progress = 0.0;
                            j.detail.clear();
                        }
                    }
                    pipeline::Event::Detail(d) => {
                        if let Some(j) = self.job_mut(id) {
                            j.detail = d;
                        }
                    }
                    pipeline::Event::Progress { step, overall } => {
                        if let Some(j) = self.job_mut(id) {
                            j.step_progress = step.clamp(0.0, 1.0);
                            j.progress = overall.clamp(0.0, 1.0);
                        }
                    }
                    pipeline::Event::Note(n) => {
                        if let Some(j) = self.job_mut(id) {
                            j.notes.push(n);
                        }
                    }
                    pipeline::Event::Done {
                        transcript,
                        outputs,
                        paths,
                    } => {
                        if let Some(j) = self.job_mut(id) {
                            j.state = JobState::Done;
                            j.progress = 1.0;
                            j.transcript = Some(*transcript);
                            j.outputs = outputs;
                            j.paths = Some(paths);
                        }
                        self.running = None;
                        let next = self.start_next();
                        // only when nothing else is left to do and the user stayed where they were
                        let idle = self.running.is_none() && !jobs::busy(&self.queue);
                        if idle && self.auto_open && self.can_take_over(id) {
                            let open = self.open_page(Page::Player(id));
                            return Task::batch([next, open]);
                        }
                        return next;
                    }
                    pipeline::Event::Failed(e) => {
                        let mut was_cancelled = false;
                        if let Some(j) = self.job_mut(id) {
                            if j.cancelled {
                                was_cancelled = true;
                                j.state = JobState::Waiting;
                                j.cancelled = false;
                                j.notes
                                    .push("Cancelled. Press Start to run it again.".into());
                            } else {
                                j.state = JobState::Failed;
                                j.error = Some(e);
                            }
                        }
                        self.running = None;
                        if was_cancelled {
                            return Task::none(); // cancel stops the whole queue
                        }
                        return self.start_next();
                    }
                }
                Task::none()
            }

            Message::OpenJob(id) => {
                if let Some(t) = self.job(id).and_then(|j| j.transcript.as_ref()) {
                    let names: Vec<String> = t
                        .speakers
                        .iter()
                        .map(|s| clean_separators(&s.name))
                        .collect();
                    self.name_edits = names;
                    self.copied = None;
                    self.shown = PAGE;
                    self.page = Page::Transcript(id);
                    self.auto_open = false;
                    self.translations.clear();
                    self.viewing = None;
                    self.rebuild_rows(id);
                    self.note_review_files(id);
                    return self.load_translations(id, None);
                }
                Task::none()
            }
            Message::TranslationsLoaded {
                job,
                translations,
                show,
            } => {
                // from before the user moved on
                if self.page == Page::Transcript(job) {
                    let shown_now = self.viewing_label().map(str::to_string);
                    let wanted = show.or(shown_now);
                    self.viewing =
                        wanted.and_then(|w| translations.iter().position(|l| l.label == w));
                    self.translations = translations;
                    self.rebuild_rows(job);
                }
                Task::none()
            }
            Message::ViewLanguage(index) => {
                self.viewing = index.filter(|i| *i < self.translations.len());
                if let Page::Transcript(id) = self.page {
                    self.rebuild_rows(id);
                }
                Task::none()
            }
            Message::Back => {
                // leaving a review keeps a row still being edited
                let write = if matches!(self.page, Page::Review(_)) {
                    let (write, _) = review_ui::update(&mut self.review, review_ui::Msg::Back);
                    write.map(Message::Review)
                } else {
                    Task::none()
                };
                Task::batch([write, self.open_page(Page::Home)])
            }
            Message::Go(page) => {
                if let Page::Player(id) | Page::Review(id) = page {
                    self.note_review_files(id);
                }
                self.open_page(page)
            }
            Message::OpenImport(id) => match self.job_ref(id) {
                Some(job) => {
                    self.auto_open = false;
                    let (task, out) = import_ui::open(&mut self.import, &job);
                    Task::batch([task.map(Message::Import), self.handle_out(out)])
                }
                None => Task::none(),
            },
            Message::TranslateTarget(choice) => {
                self.translate_target = Some(choice);
                Task::none()
            }
            Message::TranslateBoxClosed => {
                let source = self.translate_box_source.flatten();
                self.translate_box = combo_box::State::new(translate_options(source));
                Task::none()
            }
            Message::Translate(job) => {
                let Some(target) = self.translate_target else {
                    return Task::none();
                };
                self.handle_out(Out::Enqueue(out::JobKind::Translate {
                    job,
                    target: target.0.code.to_string(),
                    source: out::TranslateSource::Transcript,
                }))
            }
            Message::OpenResult { folder } => {
                Task::perform(pick_result(folder), Message::ResultPicked)
            }
            Message::ResultPicked(None) => Task::none(),
            Message::ResultPicked(Some(path)) => Task::perform(
                import_ui::on_thread(move || jobs::load_result(&path)),
                Message::ResultLoaded,
            ),
            Message::ResultLoaded(Err(e)) => {
                self.tell(&format!("That result could not be opened: {e}"));
                Task::none()
            }
            Message::ResultLoaded(Ok(loaded)) => {
                let id = self.add_loaded_result(loaded);
                self.open_page(Page::Transcript(id))
            }
            Message::LocateMedia(id) => {
                let name = self.job(id).map(Job::name).unwrap_or_default();
                Task::perform(pick_media(name), move |path| {
                    Message::MediaLocated(id, path)
                })
            }
            Message::MediaLocated(_, None) => Task::none(),
            Message::MediaLocated(id, Some(path)) => {
                let Some(paths) = self.job(id).and_then(|j| j.paths.clone()) else {
                    return Task::none();
                };
                if let Err(e) = jobs::locate_media(&paths, &path) {
                    self.tell(&format!(
                        "The new place of the media could not be saved: {e}"
                    ));
                }
                if let Some(j) = self.job_mut(id) {
                    j.size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    j.path = path;
                    j.media_missing = false;
                }
                Task::none()
            }
            Message::MoreRows => {
                self.shown += PAGE;
                if let Page::Transcript(id) = self.page {
                    let have = self.rows.len();
                    let more: Vec<text_editor::Content> = self
                        .job(id)
                        .and_then(|j| j.transcript.as_ref())
                        .map(|t| {
                            (have..t.utterances.len().min(self.shown))
                                .map(|i| row_content(&self.row_text(t, i)))
                                .collect()
                        })
                        .unwrap_or_default();
                    self.rows.extend(more);
                }
                Task::none()
            }
            Message::RowAction(i, action) => {
                // read-only: selecting and moving the caret are fine, typing and deleting are ignored
                if !action.is_edit()
                    && let Some(c) = self.rows.get_mut(i)
                {
                    c.perform(action);
                }
                Task::none()
            }
            Message::CopyRow(i) => {
                let text = match self.page {
                    Page::Transcript(id) => self
                        .job(id)
                        .and_then(|j| j.transcript.as_ref())
                        .filter(|t| i < t.utterances.len())
                        .map(|t| self.row_text(t, i)),
                    Page::Home | Page::Player(_) | Page::Review(_) => None,
                };
                match text {
                    Some(s) => {
                        self.copied = Some((CopyTarget::Row(i), Instant::now()));
                        iced::clipboard::write(s)
                    }
                    None => Task::none(),
                }
            }
            Message::OpenFolder(id) => {
                if let Some(paths) = self.job(id).and_then(|j| j.paths.as_ref()) {
                    pipeline::open_in_file_manager(&paths.dir);
                }
                Task::none()
            }
            Message::CopyText(id) => {
                match self.job(id).and_then(|j| j.transcript.as_ref()).map(|t| {
                    match self.viewed_texts() {
                        Some(texts) => t.to_txt_with(texts),
                        None => t.to_txt(),
                    }
                }) {
                    Some(text) => {
                        self.copied = Some((CopyTarget::All, Instant::now()));
                        iced::clipboard::write(text)
                    }
                    None => Task::none(),
                }
            }
            Message::RenameSpeaker(i, v) => {
                if i < self.name_edits.len() {
                    self.name_edits[i] = clean_separators(&v);
                }
                Task::none()
            }
            Message::SaveNames => {
                if let Page::Transcript(id) = self.page {
                    let edits = self.name_edits.clone();
                    if let Some(j) = self.job_mut(id)
                        && let Some(t) = j.transcript.as_mut()
                    {
                        for (i, n) in edits.iter().enumerate() {
                            if !n.trim().is_empty() {
                                t.rename_speaker(i, n);
                            }
                        }
                        if let Some(paths) = &j.paths {
                            let _ = pipeline::write_outputs(t, &paths.dir, &paths.stem);
                        }
                    }
                }
                Task::none()
            }

            Message::SetTheme(mode) => {
                self.settings.theme = mode;
                self.recompute_tokens();
                self.save_settings();
                Task::none()
            }
            Message::ChooseOutput => Task::perform(pick_folder(), Message::OutputChosen),
            Message::ChooseFfmpeg => Task::perform(pick_ffmpeg(), Message::FfmpegChosen),
            Message::FfmpegChosen(path) => {
                if let Some(p) = path {
                    self.settings.ffmpeg_path = Some(p.clone());
                    engine::audio::set_ffmpeg_override(Some(p));
                    self.ffmpeg = engine::audio::find_ffmpeg();
                    self.save_settings();
                }
                Task::none()
            }
            Message::ResetFfmpeg => {
                self.settings.ffmpeg_path = None;
                engine::audio::set_ffmpeg_override(None);
                self.ffmpeg = engine::audio::find_ffmpeg();
                self.save_settings();
                Task::none()
            }
            Message::OutputChosen(dir) => {
                if let Some(d) = dir {
                    self.settings.output_dir = Some(d);
                    self.save_settings();
                }
                Task::none()
            }
            Message::ResetOutput => {
                self.settings.output_dir = None;
                self.save_settings();
                Task::none()
            }
            Message::ToggleCleanup(on) => {
                self.settings.cleanup = on;
                self.save_settings();
                Task::none()
            }
            Message::SetSpeakers(n) => {
                self.settings.num_speakers = n;
                self.save_settings();
                Task::none()
            }
            Message::Pick(role, choice) => {
                if choice.id.is_empty() {
                    // "Same as text cleanup"
                    self.selection.remove(&role);
                } else {
                    self.selection.insert(role, choice.id);
                }
                self.save_settings();
                Task::none()
            }
            Message::Download(id) => self.start_download(&id),
            Message::DownloadAllMissing => {
                let ids: Vec<String> = self
                    .selected_entries()
                    .into_iter()
                    .filter(|m| !matches!(self.store.status(m), Status::Present))
                    .filter(|m| {
                        self.detection
                            .as_ref()
                            .is_none_or(|d| !matches!(fit(m, d), Fit::TooBig | Fit::Unsupported))
                    })
                    .map(|m| m.id.clone())
                    .collect();
                Task::batch(ids.iter().map(|id| self.start_download(id)))
            }
            Message::DownloadUpdate(id, ev) => {
                match ev {
                    DownloadEvent::Progress { done, total } => {
                        self.downloading.insert(id, Progress { done, total });
                    }
                    DownloadEvent::Finished(res) => {
                        self.downloading.remove(&id);
                        if let Err(e) = res {
                            self.errors.insert(id, e);
                        }
                    }
                }
                Task::none()
            }

            Message::Key(key, mods) => self.route_key(key, mods),
            Message::Frame(at) => {
                if self
                    .page_fade
                    .is_some_and(|(since, _)| !fade::running(since, PAGE_FADE))
                {
                    self.page_fade = None;
                }
                marks_ui::frame(&mut self.marks, at).map(Message::Marks)
            }
            Message::Player(msg) => {
                let (task, out) = player::update(&mut self.player, msg);
                marks_ui::follow(
                    &mut self.marks,
                    player::cues(&self.player),
                    player::current_cue(&self.player),
                    player::position(&self.player).time,
                );
                Task::batch([task.map(Message::Player), self.handle_out(out)])
            }
            Message::Marks(msg) => {
                let (task, out) = marks_ui::update(&mut self.marks, msg);
                Task::batch([task.map(Message::Marks), self.handle_out(out)])
            }
            Message::Profiles(msg) => {
                let active = self.settings.active_profile.clone();
                let (task, out) = profiles_ui::update(&mut self.profiles, msg, &mut self.settings);
                if self.settings.active_profile != active {
                    self.save_settings();
                }
                Task::batch([task.map(Message::Profiles), self.handle_out(out)])
            }
            Message::Import(msg) => {
                let (task, out) = import_ui::update(&mut self.import, msg);
                Task::batch([task.map(Message::Import), self.handle_out(out)])
            }
            Message::Review(msg) => {
                let (task, out) = review_ui::update(&mut self.review, msg);
                Task::batch([task.map(Message::Review), self.handle_out(out)])
            }
            Message::Jobs(msg) => {
                let finished = match &msg {
                    jobs::Msg::Event(jobs::RunEvent::Finished(finished)) => Some(finished.clone()),
                    _ => None,
                };
                let (task, out) = jobs::update(&mut self.queue, msg);
                if let Some(jobs::Finished::Compared { job, .. }) = finished {
                    self.reviewable.insert(job);
                }
                let reload = match finished {
                    Some(jobs::Finished::Translated { job, target, .. })
                        if self.page == Page::Transcript(job) =>
                    {
                        self.load_translations(job, Some(jobs::language_name(&target)))
                    }
                    Some(jobs::Finished::Translated { .. } | jobs::Finished::Chapters { .. })
                        if matches!(self.page, Page::Player(_)) =>
                    {
                        self.update(Message::Player(player::Msg::ReloadTracks))
                    }
                    _ => Task::none(),
                };
                let handled = self.handle_job_out(out);
                Task::batch([task.map(Message::Jobs), handled, reload, self.start_next()])
            }
        }
    }

    /// `update`, and a soft change when it moved to another page.
    fn step(&mut self, message: Message) -> Task<Message> {
        let (page, background) = (self.page, self.chrome_tokens().bg);
        let task = self.update(message);
        if self.page != page {
            self.page_fade = Some((Instant::now(), background));
        }
        self.sync_translate_box();
        task
    }

    /// The translation on show, one text per utterance. `None` when the page shows the original.
    fn viewed_texts(&self) -> Option<&[String]> {
        self.viewing
            .and_then(|i| self.translations.get(i))
            .map(|l| l.texts.as_slice())
    }

    fn viewing_label(&self) -> Option<&str> {
        self.viewing
            .and_then(|i| self.translations.get(i))
            .map(|l| l.label.as_str())
    }

    /// What utterance `i` says in the language on show.
    fn row_text(&self, t: &Transcript, i: usize) -> String {
        match self.viewed_texts() {
            Some(texts) => texts.get(i).cloned().unwrap_or_default(),
            None => t
                .utterances
                .get(i)
                .map(|u| u.text.clone())
                .unwrap_or_default(),
        }
    }

    /// The selectable text of the rows built so far, in the language on show.
    fn rebuild_rows(&mut self, id: u64) {
        let rows = self
            .job(id)
            .and_then(|j| j.transcript.as_ref())
            .map(|t| {
                (0..t.utterances.len().min(self.shown))
                    .map(|i| row_content(&self.row_text(t, i)))
                    .collect()
            })
            .unwrap_or_default();
        self.rows = rows;
    }

    /// Reads a job's saved translations off the UI thread, then shows the one called `show`, if any.
    fn load_translations(&self, job: u64, show: Option<String>) -> Task<Message> {
        let found = self.job(job).and_then(|j| {
            let paths = j.paths.clone()?;
            let starts = j.transcript.as_ref()?.utterances.iter().map(|u| u.start);
            Some((paths, starts.collect::<Vec<f64>>()))
        });
        let Some((paths, starts)) = found else {
            return Task::none();
        };
        Task::perform(
            import_ui::on_thread(move || Ok(player::translated_lines(&paths, &starts))),
            move |translations| Message::TranslationsLoaded {
                job,
                translations: translations.unwrap_or_default(),
                show: show.clone(),
            },
        )
    }

    /// The box keeps its own list, so it is rebuilt when the open transcript has another language.
    fn sync_translate_box(&mut self) {
        let Page::Transcript(id) = self.page else {
            return;
        };
        let source = self
            .job(id)
            .and_then(|j| j.transcript.as_ref())
            .and_then(|tr| engine::lang::language(&tr.language))
            .map(|l| l.code);
        if self.translate_box_source != Some(source) {
            self.translate_box_source = Some(source);
            self.translate_box = combo_box::State::new(translate_options(source));
        }
    }

    /// Something on screen is mid-motion and needs a redraw every frame.
    fn animating(&self) -> bool {
        self.page_fade.is_some()
            || (matches!(self.page, Page::Player(_))
                && (player::is_fading(&self.player) || marks_ui::is_animating(&self.marks)))
    }

    fn subscription(&self) -> Subscription<Message> {
        // system usage is only sampled while it is shown (Settings) or useful (a job is running)
        let usage = if self.settings_open || self.running.is_some() {
            Subscription::run(usage_stream).map(Message::Usage)
        } else {
            Subscription::none()
        };
        let tick = if self.is_loading() {
            Subscription::run(tick_stream).map(|_| Message::Tick)
        } else {
            Subscription::none()
        };
        let frames = if self.animating() {
            window::frames().map(Message::Frame)
        } else {
            Subscription::none()
        };
        Subscription::batch([
            usage,
            tick,
            frames,
            iced::system::theme_changes().map(Message::SystemTheme),
            window::close_requests().map(|_| Message::CloseWindow),
            player::subscription(&self.player).map(Message::Player),
            event::listen_with(|ev, status, _id| match ev {
                iced::Event::Window(window::Event::FileDropped(path)) => {
                    Some(Message::FileDropped(path))
                }
                iced::Event::Keyboard(keyboard::Event::KeyPressed {
                    key,
                    physical_key,
                    modifiers,
                    ..
                }) => {
                    use keyboard::{Key, key::Named};
                    // shortcuts belong to the key, whichever layout is active (Persian, for example)
                    let key = latin_key(key, physical_key);
                    match key.as_ref() {
                        Key::Named(Named::Escape) => Some(Message::Escape),
                        Key::Named(Named::F11) => Some(Message::ToggleFullscreen),
                        // Cmd on macOS, Ctrl elsewhere
                        Key::Character(",") if modifiers.command() => Some(Message::OpenSettings),
                        Key::Character("w") if MAC && modifiers.command() => {
                            Some(Message::CloseWindow)
                        }
                        Key::Character("m") if MAC && modifiers.command() => {
                            Some(Message::Minimize)
                        }
                        _ if status == event::Status::Ignored => {
                            Some(Message::Key(key.clone(), modifiers))
                        }
                        _ => None,
                    }
                }
                _ => None,
            }),
        ])
    }

    // -------------------------------------------------------------- views: shell

    fn view(&self) -> Element<'_, Message> {
        let t = self.tokens;
        let shell: Element<'_, Message> = responsive(move |size: Size| {
            let narrow = size.width < 760.0;
            let page = match self.page {
                Page::Home => self.home_page(narrow),
                Page::Transcript(id) => self.transcript_page(id),
                Page::Player(id) => self.player_page(id),
                Page::Review(_) => review_ui::view(&self.review, t).map(Message::Review),
            };
            let content = page;
            let divider = container(space::horizontal())
                .width(Length::Fixed(1.0))
                .height(Length::Fill)
                .style(theme::divider(t));
            // the player keeps the whole window, as a media player does; its back button leaves
            let body: Element<'_, Message> = if narrow || matches!(self.page, Page::Player(_)) {
                content
            } else {
                row![self.sidebar(), divider, content].into()
            };
            column![self.titlebar(narrow), body].into()
        })
        .into();

        // our own frame only where we also draw the title strip (Windows / Linux, not maximized)
        let framed = !MAC && !self.maximized && !self.fullscreen;
        let radius = if framed { corner_radius() } else { 0.0 };
        let mut layers: Vec<Element<'_, Message>> = vec![
            container(shell)
                .padding(1)
                .width(Length::Fill)
                .height(Length::Fill)
                .style(theme::window_frame(self.chrome_tokens(), framed, radius))
                .into(),
        ];

        // the new page comes up through a veil that starts as the old page's background, so the
        // window never changes brightness at a stroke
        if let Some((since, from)) = self.page_fade {
            let done = fade::eased(since, PAGE_FADE);
            let veil =
                theme::with_alpha(theme::mix(from, self.chrome_tokens().bg, done), 1.0 - done);
            layers.push(
                container(space())
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .style(theme::rounded_fill(veil, radius))
                    .into(),
            );
        }

        if self.settings_open {
            layers.push(opaque(
                mouse_area(
                    container(center(opaque(self.settings_dialog())).padding(24))
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .style(theme::scrim()),
                )
                .on_press(Message::CloseSettings),
            ));
        }
        if import_ui::is_open(&self.import) {
            layers.push(opaque(
                container(center(
                    import_ui::view(&self.import, t).map(Message::Import),
                ))
                .padding(24)
                .width(Length::Fill)
                .height(Length::Fill)
                .style(theme::scrim()),
            ));
        }
        // invisible handles on the window edges so the borderless window can still be resized
        if !MAC && !self.maximized && !self.fullscreen {
            layers.push(resize_edges());
        }
        // on top of everything, and only as big as the message: what's under it keeps its place and stays usable
        if let Some(message) = self.notice_banner() {
            layers.push(
                container(opaque(container(message).max_width(TOAST_WIDTH)))
                    .padding(TOAST_MARGIN)
                    .align_right(Length::Fill)
                    .align_bottom(Length::Fill)
                    .into(),
            );
        }

        stack(layers).into()
    }

    /// Top strip: drag area, (settings button on narrow windows) and the window buttons.
    /// macOS: native traffic lights live in the reserved space on the left, so no buttons are drawn.
    /// The window's own frame and title strip: dark around the player whatever the app's theme is.
    fn chrome_tokens(&self) -> Tokens {
        if matches!(self.page, Page::Player(_)) {
            theme::PLAYER
        } else {
            self.tokens
        }
    }

    fn titlebar(&self, narrow: bool) -> Element<'_, Message> {
        let t = self.chrome_tokens();
        let height = if MAC { 40.0 } else { 36.0 };
        let drag = mouse_area(
            container(space::horizontal())
                .width(Length::Fill)
                .height(Length::Fixed(height)),
        )
        .on_press(Message::DragWindow)
        .on_double_click(Message::ToggleMaximize);

        let mut r = row![].align_y(Alignment::Center);
        if MAC {
            r = r.push(space::horizontal().width(Length::Fixed(84.0)));
        } else {
            r = r.push(
                container(
                    row![
                        svg(svg::Handle::from_memory(APP_ICON_SVG))
                            .width(Length::Fixed(18.0))
                            .height(Length::Fixed(18.0)),
                        text(APP_NAME).size(13).font(semibold()).color(t.muted),
                    ]
                    .spacing(9)
                    .align_y(Alignment::Center),
                )
                .padding([0, 12])
                .center_y(Length::Fixed(height)),
            );
        }
        r = r.push(drag);
        if narrow {
            r = r.push(
                button(icons::icon(icons::SLIDERS, 17.0, t.text))
                    .padding([7, 9])
                    .style(theme::ghost(t))
                    .on_press(Message::OpenSettings),
            );
            r = r.push(space::horizontal().width(Length::Fixed(6.0)));
        }
        if !MAC {
            let win_btn = |icon: &'static str, msg: Message, close: bool| -> Element<'_, Message> {
                button(
                    container(icons::icon(icon, 10.0, t.text))
                        .center_x(Length::Fixed(46.0))
                        .center_y(Length::Fixed(height)),
                )
                .padding(0)
                .style(theme::win_button(t, close))
                .on_press(msg)
                .into()
            };
            r = r
                .push(win_btn(icons::WIN_MINIMIZE, Message::Minimize, false))
                .push(win_btn(
                    if self.maximized {
                        icons::WIN_RESTORE
                    } else {
                        icons::WIN_MAXIMIZE
                    },
                    Message::ToggleMaximize,
                    false,
                ))
                .push(win_btn(icons::WIN_CLOSE, Message::CloseWindow, true));
        }
        container(r)
            .width(Length::Fill)
            .height(Length::Fixed(height))
            .style(theme::strip(t))
            .into()
    }

    fn sidebar(&self) -> Element<'_, Message> {
        let t = self.tokens;
        let hw_inner: Element<'_, Message> = match &self.detection {
            None => row![
                self.spinner(16.0, t.muted),
                column![
                    text("Detecting hardware").size(12).color(t.muted),
                    self.skeleton(Length::Fixed(84.0), 8.0)
                ]
                .spacing(5),
            ]
            .spacing(10)
            .align_y(Alignment::Center)
            .into(),
            Some(d) => {
                let gpu = d.recommended_gpu.and_then(|i| d.gpus.get(i)).map(|g| {
                    g.name
                        .replace("NVIDIA GeForce ", "")
                        .replace(" Laptop GPU", "")
                });
                row![
                    icons::icon(icons::CPU, 18.0, t.muted),
                    column![
                        text(d.recommended.as_str().to_uppercase())
                            .size(12)
                            .font(semibold())
                            .color(t.text),
                        text(gpu.unwrap_or_else(|| "No GPU found".into()))
                            .size(12)
                            .color(t.muted),
                    ]
                    .spacing(1),
                ]
                .spacing(10)
                .align_y(Alignment::Center)
                .into()
            }
        };
        let hw = container(hw_inner)
            .padding(12)
            .width(Length::Fill)
            .style(theme::card(t));

        container(
            column![
                nav_button(
                    t,
                    icons::FILE_TEXT,
                    "Transcripts",
                    !self.settings_open,
                    Message::Back
                ),
                nav_button(
                    t,
                    icons::SLIDERS,
                    "Settings",
                    self.settings_open,
                    Message::OpenSettings
                ),
                space::vertical(),
                hw,
                text(format!("Version {VERSION}  -  English"))
                    .size(11)
                    .color(t.faint),
            ]
            .spacing(6)
            .padding([8, 16]),
        )
        .width(Length::Fixed(243.0))
        .height(Length::Fill)
        .style(theme::sidebar(
            t,
            corner_radius() > 0.0 && !self.maximized && !self.fullscreen,
        ))
        .into()
    }

    // -------------------------------------------------------------- views: home

    fn home_page(&self, narrow: bool) -> Element<'_, Message> {
        let t = self.tokens;
        let waiting = self
            .jobs
            .iter()
            .filter(|j| matches!(j.state, JobState::Waiting | JobState::Failed))
            .count();
        let can_start = waiting > 0 && self.running.is_none() && self.can_transcribe();

        let start_label = if self.running.is_some() {
            "Working...".to_string()
        } else if waiting > 1 {
            format!("Start {waiting} files")
        } else {
            "Start".to_string()
        };
        let actions = row![
            button(
                row![
                    icons::icon(icons::PLUS, 16.0, t.text),
                    text("Add files").size(14)
                ]
                .spacing(8)
                .align_y(Alignment::Center)
            )
            .padding([9, 14])
            .style(theme::outline_btn(t))
            .on_press(Message::AddFiles),
            button(
                row![
                    icons::icon(
                        icons::PLAY,
                        15.0,
                        if can_start { t.on_accent } else { t.faint }
                    ),
                    text(start_label).size(14)
                ]
                .spacing(8)
                .align_y(Alignment::Center)
            )
            .padding([9, 16])
            .style(theme::primary(t))
            .on_press_maybe(can_start.then_some(Message::StartAll)),
        ]
        .spacing(10);

        let title = column![
            text("Transcripts").size(26).font(semibold()).color(t.text),
            text("Turn recordings into clean, speaker-labelled text and subtitles. Everything runs on this computer.")
                .size(13)
                .color(t.muted),
        ]
        .spacing(4)
        .width(Length::Fill);

        let header: Element<'_, Message> = if narrow {
            column![title, actions].spacing(12).into()
        } else {
            row![title, actions]
                .spacing(16)
                .align_y(Alignment::Center)
                .into()
        };

        let open_saved = row![
            text("Open a saved result:").size(12).color(t.muted),
            button(text("Folder...").size(12))
                .padding([4, 10])
                .style(theme::ghost(t))
                .on_press(Message::OpenResult { folder: true }),
            button(text("File...").size(12))
                .padding([4, 10])
                .style(theme::ghost(t))
                .on_press(Message::OpenResult { folder: false }),
        ]
        .spacing(6)
        .align_y(Alignment::Center);

        let mut page = column![
            header,
            open_saved,
            jobs::view(&self.queue, t, self.epoch.elapsed().as_secs_f32()).map(Message::Jobs)
        ]
        .spacing(18);

        if self.detection.is_some() && !self.can_transcribe() {
            page = page.push(
                container(
                    row![
                        icons::icon(icons::ALERT, 20.0, t.warn),
                        column![
                            text("Speech models are not downloaded yet").size(14).font(semibold()).color(t.text),
                            text("Open Settings and download the models once; after that everything works offline.")
                                .size(13)
                                .color(t.muted),
                        ]
                        .spacing(2)
                        .width(Length::Fill),
                        button(text("Open Settings").size(13)).padding([8, 12]).style(theme::primary(t)).on_press(Message::OpenSettings),
                    ]
                    .spacing(12)
                    .align_y(Alignment::Center),
                )
                .padding(14)
                .width(Length::Fill)
                .style(theme::card(t)),
            );
        }

        if self.jobs.is_empty() {
            page = page.push(self.empty_state());
        } else {
            for j in &self.jobs {
                page = page.push(self.job_card(j));
            }
        }

        scrollable(
            container(page)
                .padding(if narrow { 16 } else { 28 })
                .max_width(980),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
    }

    /// Everything the app has to tell the user, on any page. `None` when there is nothing.
    fn notice_banner(&self) -> Option<Element<'_, Message>> {
        if self.notice.is_none() && self.notices.is_empty() && self.review_ready.is_none() {
            return None;
        }
        let t = self.tokens;
        let heading = |line: String| text(line).size(14).font(semibold()).color(t.text);
        let mut list = column![].spacing(3).width(Length::Fill);
        if let Some(job) = self.review_ready {
            list = list.push(
                row![
                    heading("The comparison is ready".to_string()).width(Length::Fill),
                    button(text("Review").size(12))
                        .padding([4, 10])
                        .style(theme::primary(t))
                        .on_press(Message::Go(Page::Review(job))),
                ]
                .spacing(8)
                .align_y(Alignment::Center),
            );
        }
        if !self.notices.is_empty() {
            list = list.push(heading(if self.notices.len() == 1 {
                "Notice".to_string()
            } else {
                format!("{} notices", self.notices.len())
            }));
            for n in &self.notices {
                list = list.push(text(n.text.clone()).size(12).color(t.muted));
            }
        }
        if let Some((lines, _)) = &self.notice {
            list = list.push(heading(if lines.len() == 1 {
                "1 file was not added".to_string()
            } else {
                format!("{} files were not added", lines.len())
            }));
            for l in lines {
                list = list.push(text(l.clone()).size(12).color(t.muted));
            }
            if !self.waiting_for_ffmpeg.is_empty() {
                list = list
                    .push(space::vertical().height(Length::Fixed(6.0)))
                    .push(self.ffmpeg_offer());
            }
        }
        Some(
            container(
                row![
                    icons::icon(icons::ALERT, 20.0, t.warn),
                    list,
                    button(icons::icon(icons::X, 14.0, t.muted))
                        .padding(6)
                        .style(theme::ghost(t))
                        .on_press(Message::DismissNotice),
                ]
                .spacing(12)
                .align_y(Alignment::Start),
            )
            .padding(14)
            .width(Length::Fill)
            .style(theme::toast(t))
            .into(),
        )
    }

    fn empty_state(&self) -> Element<'_, Message> {
        let t = self.tokens;
        container(
            column![
                container(icons::icon(icons::VIDEO, 30.0, t.accent))
                    .center_x(Length::Fixed(68.0))
                    .center_y(Length::Fixed(68.0))
                    .style(theme::disc(theme::with_alpha(t.accent, 0.14))),
                text("Drop videos or audio here").size(20).font(semibold()).color(t.text),
                text("Most video and audio formats: mp4, mkv, mov, webm, avi, mp3, wav, m4a, flac, ogg, opus and more").size(13).color(t.muted),
                button(row![icons::icon(icons::PLUS, 16.0, t.on_accent), text("Add files").size(14)].spacing(8).align_y(Alignment::Center))
                    .padding([10, 18])
                    .style(theme::primary(t))
                    .on_press(Message::AddFiles),
            ]
            .spacing(14)
            .align_x(Alignment::Center),
        )
        .padding(48)
        .width(Length::Fill)
        .center_x(Length::Fill)
        .style(theme::card(t))
        .into()
    }

    fn job_card<'a>(&'a self, j: &'a Job) -> Element<'a, Message> {
        let t = self.tokens;
        let id = j.id;
        let name = j.name();
        let folder = j
            .path
            .parent()
            .map(|p| clean_separators(&p.display().to_string()))
            .unwrap_or_default();
        let meta = format!("{}  -  {}", human(j.size), folder);

        let (label, color) = match j.state {
            JobState::Waiting => ("Waiting".to_string(), t.muted),
            JobState::Running => (
                j.steps
                    .get(j.step)
                    .cloned()
                    .unwrap_or_else(|| "Working".to_string()),
                t.accent,
            ),
            JobState::Done => ("Done".to_string(), t.ok),
            JobState::Failed => ("Failed".to_string(), t.bad),
        };

        let mut head = row![
            container(icons::icon(
                if is_audio_file(&j.path) {
                    icons::AUDIO
                } else {
                    icons::VIDEO
                },
                20.0,
                t.accent
            ))
            .center_x(Length::Fixed(42.0))
            .center_y(Length::Fixed(42.0))
            .style(theme::rounded_fill(
                theme::with_alpha(t.accent, 0.12),
                theme::RADIUS_S
            )),
            column![
                text(name).size(15).font(semibold()).color(t.text),
                text(meta).size(12).color(t.muted)
            ]
            .spacing(2)
            .width(Length::Fill),
            container(text(label).size(12).color(color))
                .padding([4, 10])
                .style(theme::chip(color)),
        ]
        .spacing(12)
        .align_y(Alignment::Center);

        if matches!(j.state, JobState::Waiting | JobState::Failed) {
            let chevron = if j.expanded {
                icons::CHEVRON_UP
            } else {
                icons::CHEVRON_DOWN
            };
            head = head.push(
                button(
                    row![icons::icon(chevron, 14.0, t.muted), text("About").size(13)]
                        .spacing(6)
                        .align_y(Alignment::Center),
                )
                .padding([6, 10])
                .style(theme::ghost(t))
                .on_press(Message::ToggleDetails(id)),
            );
        }
        if self.running != Some(id) {
            head = head.push(
                button(icons::icon(icons::X, 15.0, t.muted))
                    .padding(7)
                    .style(theme::ghost(t))
                    .on_press(Message::Remove(id)),
            );
        }

        let mut body = column![head].spacing(12);

        match j.state {
            JobState::Running => {
                body = body.push(self.running_panel(j));
            }
            JobState::Failed => {
                if let Some(e) = &j.error {
                    body = body.push(text(clean_separators(e)).size(13).color(t.bad));
                }
            }
            JobState::Done => {
                let speakers = j.transcript.as_ref().map_or(0, |tr| {
                    tr.utterances
                        .iter()
                        .map(|u| u.speaker)
                        .collect::<std::collections::BTreeSet<_>>()
                        .len()
                });
                let count = j.transcript.as_ref().map_or(0, |tr| tr.utterances.len());
                body = body.push(
                    row![
                        text(format!("{count} turns  -  {speakers} speaker(s)"))
                            .size(13)
                            .color(t.muted)
                            .width(Length::Fill),
                        button(
                            row![
                                icons::icon(icons::FOLDER, 15.0, t.text),
                                text("Open folder").size(13)
                            ]
                            .spacing(6)
                            .align_y(Alignment::Center)
                        )
                        .padding([7, 12])
                        .style(theme::outline_btn(t))
                        .on_press(Message::OpenFolder(id)),
                        button(
                            row![
                                icons::icon(icons::FILE_TEXT, 15.0, t.text),
                                text("View transcript").size(13)
                            ]
                            .spacing(6)
                            .align_y(Alignment::Center)
                        )
                        .padding([7, 12])
                        .style(theme::outline_btn(t))
                        .on_press(Message::OpenJob(id)),
                        self.play_button(j),
                    ]
                    .spacing(8)
                    .align_y(Alignment::Center),
                );
            }
            JobState::Waiting => {}
        }

        for n in &j.notes {
            body = body.push(
                row![
                    icons::icon(icons::ALERT, 13.0, t.warn),
                    text(clean_separators(n)).size(12).color(t.muted)
                ]
                .spacing(6),
            );
        }

        if j.expanded && matches!(j.state, JobState::Waiting | JobState::Failed) {
            let cats: Vec<Choice> = CATEGORIES
                .iter()
                .map(|(s, n)| Choice {
                    id: s.to_string(),
                    label: n.to_string(),
                })
                .collect();
            let selected = cats.iter().find(|c| c.id == j.about.category).cloned();
            body = body.push(
                container(
                    column![
                        text("Tell the app about this recording (optional). It helps fix names, jargon and wording.").size(12).color(t.muted),
                        row![
                            column![
                                text("Type").size(12).color(t.muted),
                                pick_list(cats, selected, move |c| Message::Category(id, c)).style(theme::pick(t)).padding(9).width(Length::Fill)
                            ]
                            .spacing(4)
                            .width(Length::FillPortion(2)),
                            column![
                                text("What is it about?").size(12).color(t.muted),
                                text_input("e.g. weekly call about the WhatsApp integration", &j.about.description)
                                    .on_input(move |s| Message::Description(id, s))
                                    .style(theme::input(t))
                                    .padding(9)
                            ]
                            .spacing(4)
                            .width(Length::FillPortion(3)),
                        ]
                        .spacing(12),
                        column![
                            text("Speaker names (optional, in the order they first speak, separated by commas)").size(12).color(t.muted),
                            text_input("e.g. Eric, Farshad", &j.names_text).on_input(move |v| Message::SpeakerNames(id, v)).style(theme::input(t)).padding(9)
                        ]
                        .spacing(4),
                    ]
                    .spacing(12),
                )
                .padding(14)
                .width(Length::Fill)
                .style(theme::bubble(t)),
            );
        }

        container(body)
            .padding(16)
            .width(Length::Fill)
            .style(theme::card(t))
            .into()
    }

    /// "Open player", or "Locate media" when the file is gone from where it was.
    fn play_button<'a>(&self, j: &Job) -> Element<'a, Message> {
        let t = self.tokens;
        let (icon, label, message) = if j.media_missing {
            (icons::VIDEO, "Locate media", Message::LocateMedia(j.id))
        } else {
            (icons::PLAY, "Open player", Message::Go(Page::Player(j.id)))
        };
        button(
            row![icons::icon(icon, 15.0, t.on_accent), text(label).size(13)]
                .spacing(6)
                .align_y(Alignment::Center),
        )
        .padding([7, 12])
        .style(theme::primary(t))
        .on_press(message)
        .into()
    }

    /// Looks on disk for a compare file of this job, so the Review button can show. Not done while
    /// drawing, because the view runs often.
    fn note_review_files(&mut self, id: u64) {
        let found = self
            .job(id)
            .and_then(|j| j.paths.as_ref())
            .is_some_and(review_ui::has_comparison);
        if found {
            self.reviewable.insert(id);
        }
    }

    /// Adds a finished job rebuilt from disk, or finds the one already in the list. Returns its id.
    fn add_loaded_result(&mut self, loaded: jobs::LoadedResult) -> u64 {
        let jobs::LoadedResult {
            media,
            paths,
            transcript,
        } = loaded;
        if let Some(j) = self.jobs.iter().find(|j| j.paths.as_ref() == Some(&paths)) {
            return j.id;
        }
        let media_missing = media.is_none();
        let path = media.unwrap_or_else(|| {
            let name = std::path::Path::new(&transcript.source)
                .file_name()
                .map_or_else(|| paths.stem.clone().into(), ToOwned::to_owned);
            paths.dir.join(name)
        });
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let id = self.next_id;
        self.next_id += 1;
        let mut job = Job::new(id, path, size);
        job.state = JobState::Done;
        job.progress = 1.0;
        job.outputs = vec![paths.transcript()];
        job.transcript = Some(transcript);
        job.paths = Some(paths);
        job.media_missing = media_missing;
        self.jobs.push(job);
        id
    }

    /// Numbered steps, one bar for the whole run, what it is doing, and time / system usage.
    fn running_panel<'a>(&'a self, j: &'a Job) -> Element<'a, Message> {
        let t = self.tokens;
        let total = j.steps.len().max(1);
        let current = j
            .steps
            .get(j.step)
            .cloned()
            .unwrap_or_else(|| "Working".into());

        let mut items: Vec<Element<'a, Message>> = Vec::new();
        for (i, name) in j.steps.iter().enumerate() {
            let badge: Element<'a, Message> = if i < j.step {
                container(icons::icon(icons::CHECK, 12.0, t.ok))
                    .center_x(Length::Fixed(22.0))
                    .center_y(Length::Fixed(22.0))
                    .style(theme::disc(theme::with_alpha(t.ok, 0.18)))
                    .into()
            } else if i == j.step {
                container(
                    text((i + 1).to_string())
                        .size(12)
                        .font(semibold())
                        .color(t.on_accent),
                )
                .center_x(Length::Fixed(22.0))
                .center_y(Length::Fixed(22.0))
                .style(theme::disc(t.accent_fill))
                .into()
            } else {
                container(text((i + 1).to_string()).size(12).color(t.faint))
                    .center_x(Length::Fixed(22.0))
                    .center_y(Length::Fixed(22.0))
                    .style(theme::ring(t.faint))
                    .into()
            };
            let label_color = if i == j.step {
                t.text
            } else if i < j.step {
                t.muted
            } else {
                t.faint
            };
            items.push(
                row![badge, text(name.clone()).size(13).color(label_color)]
                    .spacing(8)
                    .align_y(Alignment::Center)
                    .into(),
            );
            if i + 1 < total {
                items.push(
                    container(space::horizontal())
                        .width(Length::Fixed(26.0))
                        .height(Length::Fixed(1.0))
                        .style(theme::rounded_fill(
                            if i < j.step { t.ok } else { t.rule },
                            1.0,
                        ))
                        .into(),
                );
            }
        }
        let stepper = row(items)
            .spacing(8)
            .align_y(Alignment::Center)
            .wrap()
            .vertical_spacing(8);

        let elapsed = j.started.map(|s| s.elapsed().as_secs()).unwrap_or(0);
        let eta = (j.progress > 0.04 && j.progress < 1.0)
            .then(|| (elapsed as f32 * (1.0 - j.progress) / j.progress) as u64);
        let mut meta = format!("{} elapsed", mmss(elapsed));
        if let Some(e) = eta {
            meta.push_str(&format!("  -  {}", time_left(e)));
        }
        let clock = self.epoch.elapsed().as_secs_f32();

        let mut col = column![
            stepper,
            row![
                text(format!(
                    "Step {} of {}  -  {}",
                    (j.step + 1).min(total),
                    total,
                    current
                ))
                .size(14)
                .font(semibold())
                .color(t.text)
                .width(Length::Fill),
                text(jobs::percent_text(j.shown_progress))
                    .size(16)
                    .font(semibold())
                    .color(t.accent),
                button(
                    row![
                        icons::icon(icons::X, 12.0, t.muted),
                        text("Cancel").size(12)
                    ]
                    .spacing(5)
                    .align_y(Alignment::Center)
                )
                .padding([4, 9])
                .style(theme::ghost(t))
                .on_press(Message::Cancel(j.id)),
            ]
            .spacing(10)
            .align_y(Alignment::Center),
            progress::bar(j.shown_progress, clock, 10.0, t),
        ]
        .spacing(10);
        let detail = if j.detail.is_empty() {
            "Working...".to_string()
        } else {
            clean_separators(&j.detail)
        };
        col = col.push(
            row![
                self.spinner(18.0, t.accent),
                text(detail).size(13).color(t.muted).width(Length::Fill),
                text(step_share(j.step_progress)).size(12).color(t.faint),
            ]
            .spacing(8)
            .align_y(Alignment::Center),
        );
        col = col.push(text(meta).size(12).color(t.faint));
        if let Some(u) = &self.usage {
            col = col.push(text(u.summary()).size(12).color(t.faint));
        }
        container(col)
            .padding(14)
            .width(Length::Fill)
            .style(theme::bubble(t))
            .into()
    }

    /// "This computer": which GPU / backend is used, plus live CPU / memory / GPU readings.
    /// Shows loading placeholders until the hardware is known and the first reading has arrived.
    fn system_section(&self) -> Element<'_, Message> {
        let t = self.tokens;
        let hardware: Element<'_, Message> = match &self.detection {
            None => row![
                self.spinner(18.0, t.accent),
                text("Detecting your hardware...").size(14).color(t.muted)
            ]
            .spacing(10)
            .align_y(Alignment::Center)
            .into(),
            Some(d) => {
                let gpu = d.recommended_gpu.and_then(|i| d.gpus.get(i));
                row![
                    container(
                        text(d.recommended.as_str().to_uppercase())
                            .size(13)
                            .font(semibold())
                            .color(t.accent)
                    )
                    .padding([5, 12])
                    .style(theme::chip(t.accent)),
                    column![
                        text(
                            gpu.map(|g| g.name.clone())
                                .unwrap_or_else(|| "No GPU found - using the CPU".into())
                        )
                        .size(14)
                        .color(t.text),
                        text(
                            gpu.and_then(|g| g.vram_mb)
                                .map(|m| format!("{:.1} GB video memory", m as f64 / 1024.0))
                                .unwrap_or_else(|| d.reason.clone())
                        )
                        .size(12)
                        .color(t.muted),
                    ]
                    .spacing(2),
                ]
                .spacing(14)
                .align_y(Alignment::Center)
                .into()
            }
        };

        let Some(u) = &self.usage else {
            // the first reading arrives within a second: show where the meters will appear
            let placeholders = column![
                self.meter_placeholder("Processor"),
                self.meter_placeholder("Memory (RAM)"),
                self.meter_placeholder("Graphics card"),
                self.meter_placeholder("Video memory"),
            ]
            .spacing(12);
            return section(
                t,
                "This computer",
                column![
                    hardware,
                    row![
                        self.spinner(14.0, t.accent),
                        text("Measuring usage...").size(12).color(t.muted)
                    ]
                    .spacing(8)
                    .align_y(Alignment::Center),
                    placeholders,
                ]
                .spacing(14)
                .into(),
            );
        };
        let ram_frac = if u.ram_total > 0 {
            u.ram_used as f32 / u.ram_total as f32
        } else {
            0.0
        };
        let mut col = column![
            hardware,
            meter(t, "Processor", u.cpu / 100.0, format!("{:.0}%", u.cpu)),
            meter(
                t,
                "Memory (RAM)",
                ram_frac,
                format!("{} / {}", human(u.ram_used), human(u.ram_total))
            ),
        ]
        .spacing(12);
        match (u.gpu.gpu_percent, u.gpu.vram_used_mb, u.gpu.vram_total_mb) {
            (Some(g), Some(used), Some(total)) => {
                col = col
                    .push(meter(t, "Graphics card", g as f32 / 100.0, format!("{g}%")))
                    .push(meter(
                        t,
                        "Video memory",
                        if total > 0 {
                            used as f32 / total as f32
                        } else {
                            0.0
                        },
                        format!("{} / {}", human(used << 20), human(total << 20)),
                    ));
            }
            _ => {
                col = col.push(
                    text("Graphics card usage is not available for this GPU.")
                        .size(12)
                        .color(t.muted),
                );
            }
        }
        section(t, "This computer", col.into())
    }

    // -------------------------------------------------------------- views: transcript

    fn transcript_page(&self, id: u64) -> Element<'_, Message> {
        let t = self.tokens;
        let Some(j) = self.job(id) else {
            return text("Transcript not found").into();
        };
        let Some(tr) = j.transcript.as_ref() else {
            return text("Transcript not found").into();
        };
        let name = j.name();
        let copied_all = matches!(self.copied, Some((CopyTarget::All, _)));

        let header = row![
            button(icons::icon(icons::ARROW_LEFT, 18.0, t.text))
                .padding(9)
                .style(theme::outline_btn(t))
                .on_press(Message::Back),
            column![
                text(name).size(20).font(semibold()).color(t.text),
                text(format!(
                    "{}  -  {} turns  -  text {}",
                    clock(tr.audio_seconds),
                    tr.utterances.len(),
                    if tr.refined {
                        "cleaned up by AI"
                    } else {
                        "as heard"
                    }
                ))
                .size(12)
                .color(t.muted),
            ]
            .spacing(2)
            .width(Length::Fill),
            button(
                row![
                    icons::icon(
                        if copied_all {
                            icons::CHECK
                        } else {
                            icons::COPY
                        },
                        15.0,
                        if copied_all { t.ok } else { t.text }
                    ),
                    text(if copied_all { "Copied" } else { "Copy all" }).size(13)
                ]
                .spacing(6)
                .align_y(Alignment::Center)
            )
            .padding([8, 12])
            .style(theme::outline_btn(t))
            .on_press(Message::CopyText(id)),
            button(
                row![
                    icons::icon(icons::FOLDER, 15.0, t.text),
                    text("Open folder").size(13)
                ]
                .spacing(6)
                .align_y(Alignment::Center)
            )
            .padding([8, 12])
            .style(theme::outline_btn(t))
            .on_press(Message::OpenFolder(id)),
            self.play_button(j),
        ]
        .spacing(10)
        .align_y(Alignment::Center);

        // speaker names: edit, then save (the files are rewritten)
        let mut name_items: Vec<Element<'_, Message>> = Vec::new();
        for (i, s) in tr.speakers.iter().enumerate() {
            let c = t.speaker(i);
            name_items.push(
                row![
                    container(text("").size(1))
                        .width(Length::Fixed(10.0))
                        .height(Length::Fixed(10.0))
                        .style(theme::disc(c)),
                    text_input(
                        &clean_separators(&s.name),
                        self.name_edits.get(i).map(String::as_str).unwrap_or("")
                    )
                    .on_input(move |v| Message::RenameSpeaker(i, v))
                    .style(theme::input(t))
                    .padding(7)
                    .width(Length::Fixed(150.0)),
                ]
                .spacing(6)
                .align_y(Alignment::Center)
                .into(),
            );
        }
        let names = row(name_items)
            .spacing(10)
            .align_y(Alignment::Center)
            .wrap();
        let legend = row![
            text("Speakers").size(12).color(t.muted),
            names,
            button(text("Save names").size(12))
                .padding([6, 10])
                .style(theme::outline_btn(t))
                .on_press(Message::SaveNames),
            space::horizontal(),
            text("Select text with the mouse, then Ctrl+C. Also saved as .txt  .srt  .vtt  .json")
                .size(12)
                .color(t.faint),
        ]
        .spacing(12)
        .align_y(Alignment::Center);

        let mut list = column![].spacing(14);
        for (i, u) in tr.utterances.iter().enumerate().take(self.shown) {
            let c = t.speaker(u.speaker);
            let who = clean_separators(&tr.speaker_name(u.speaker));
            let initial = who
                .chars()
                .next()
                .map(|ch| ch.to_uppercase().to_string())
                .unwrap_or_else(|| "?".into());
            let copied_here = matches!(self.copied, Some((CopyTarget::Row(r), _)) if r == i);
            // the message text: selectable (mouse drag, double-click a word, Ctrl+C)
            let body: Element<'_, Message> = match self.rows.get(i) {
                Some(content) => text_editor(content)
                    .on_action(move |a| Message::RowAction(i, a))
                    .size(15)
                    .padding(0)
                    .style(theme::selectable(t))
                    .into(),
                None => text(clean_separators(&u.text))
                    .size(15)
                    .color(t.text)
                    .into(),
            };
            list = list.push(
                row![
                    container(text(initial).size(14).font(semibold()).color(c))
                        .center_x(Length::Fixed(34.0))
                        .center_y(Length::Fixed(34.0))
                        .style(theme::disc(theme::with_alpha(c, 0.16))),
                    column![
                        row![
                            text(who).size(13).font(semibold()).color(c),
                            text(clock(u.start)).size(12).color(t.faint),
                            space::horizontal(),
                            button(
                                row![
                                    icons::icon(
                                        if copied_here {
                                            icons::CHECK
                                        } else {
                                            icons::COPY
                                        },
                                        13.0,
                                        if copied_here { t.ok } else { t.muted }
                                    ),
                                    text(if copied_here { "Copied" } else { "Copy" })
                                        .size(12)
                                        .color(if copied_here { t.ok } else { t.muted }),
                                ]
                                .spacing(5)
                                .align_y(Alignment::Center)
                            )
                            .padding([3, 8])
                            .style(theme::ghost(t))
                            .on_press(Message::CopyRow(i)),
                        ]
                        .spacing(8)
                        .align_y(Alignment::Center),
                        container(body)
                            .padding([10, 14])
                            .width(Length::Fill)
                            .style(theme::bubble(t)),
                    ]
                    .spacing(4)
                    .width(Length::Fill),
                ]
                .spacing(12),
            );
        }
        if tr.utterances.len() > self.shown {
            list = list.push(
                button(
                    text(format!(
                        "Show more ({} left)",
                        tr.utterances.len() - self.shown
                    ))
                    .size(13),
                )
                .padding([9, 16])
                .style(theme::outline_btn(t))
                .on_press(Message::MoreRows),
            );
        }

        column![
            container(
                column![
                    header,
                    self.result_actions(id, tr),
                    self.language_switch(),
                    legend,
                    jobs::view(&self.queue, t, self.epoch.elapsed().as_secs_f32())
                        .map(Message::Jobs)
                ]
                .spacing(14)
            )
            .padding([18, 28])
            .width(Length::Fill)
            .style(theme::panel(t)),
            scrollable(container(list).padding([20, 28]).max_width(920))
                .width(Length::Fill)
                .height(Length::Fill),
        ]
        .into()
    }

    /// Original and each saved translation, to choose what the page below shows. Nothing without a translation.
    fn language_switch(&self) -> Element<'_, Message> {
        if self.translations.is_empty() {
            return space().into();
        }
        let t = self.tokens;
        let choice = |label: &str, index: Option<usize>| -> Element<'_, Message> {
            button(text(label.to_string()).size(13))
                .padding([5, 12])
                .style(theme::segment(t, self.viewing == index))
                .on_press(Message::ViewLanguage(index))
                .into()
        };
        let mut items = vec![
            text("Showing").size(12).color(t.muted).into(),
            choice("Original", None),
        ];
        for (i, language) in self.translations.iter().enumerate() {
            items.push(choice(language.label.as_str(), Some(i)));
        }
        row(items)
            .spacing(8)
            .align_y(Alignment::Center)
            .wrap()
            .vertical_spacing(8)
            .into()
    }

    /// What can be done with a finished result: translate it, import subtitles, review a comparison.
    fn result_actions<'a>(&'a self, id: u64, tr: &Transcript) -> Element<'a, Message> {
        let t = self.tokens;
        let source = engine::lang::language(&tr.language).map(|l| l.code);
        let has_model = self.translate_model_path().is_some();
        let target = self.translate_target.filter(|c| Some(c.0.code) != source);
        let again = target
            .zip(self.job(id).and_then(|j| j.paths.as_ref()))
            .is_some_and(|(c, paths)| paths.translation(c.0.code).is_file());
        let mut r = row![
            combo_box(
                &self.translate_box,
                "Translate to... (type to search)",
                target.as_ref(),
                Message::TranslateTarget,
            )
            .size(13)
            .padding([6, 10])
            .width(Length::Fixed(240.0))
            .on_close(Message::TranslateBoxClosed)
            .input_style(theme::input(t))
            .menu_style(theme::search_menu(t)),
            button(
                text(if again {
                    "Translate again"
                } else {
                    "Translate"
                })
                .size(13)
            )
            .padding([7, 12])
            .style(theme::outline_btn(t))
            .on_press_maybe((has_model && target.is_some()).then_some(Message::Translate(id))),
            button(text("Import subtitles").size(13))
                .padding([7, 12])
                .style(theme::outline_btn(t))
                .on_press(Message::OpenImport(id)),
        ]
        .spacing(10)
        .align_y(Alignment::Center);
        if self.reviewable.contains(&id) {
            r = r.push(
                button(text("Review").size(13))
                    .padding([7, 12])
                    .style(theme::outline_btn(t))
                    .on_press(Message::Go(Page::Review(id))),
            );
        }
        if !has_model {
            r = r.push(
                text("Translating needs its AI model: download it in Settings.")
                    .size(12)
                    .color(t.muted),
            );
        }
        r.into()
    }

    // -------------------------------------------------------------- views: settings popup

    fn settings_dialog(&self) -> Element<'_, Message> {
        let t = self.tokens;

        let seg = |label: &'static str, value: &'static str| {
            button(text(label).size(13))
                .padding([7, 14])
                .style(theme::segment(t, self.settings.theme == value))
                .on_press(Message::SetTheme(value.into()))
        };
        let appearance = section(
            t,
            "Appearance",
            row![
                seg("System", "system"),
                seg("Light", "light"),
                seg("Dark", "dark")
            ]
            .spacing(8)
            .into(),
        );

        let out = self.output_dir();
        let folder = section(
            t,
            "Results folder",
            column![
                text(clean_separators(&out.display().to_string()))
                    .size(13)
                    .color(t.muted),
                row![
                    button(text("Change...").size(13))
                        .padding([7, 12])
                        .style(theme::outline_btn(t))
                        .on_press(Message::ChooseOutput),
                    button(text("Use default").size(13))
                        .padding([7, 12])
                        .style(theme::ghost(t))
                        .on_press_maybe(
                            self.settings
                                .output_dir
                                .is_some()
                                .then_some(Message::ResetOutput)
                        ),
                ]
                .spacing(8),
            ]
            .spacing(10)
            .into(),
        );

        let speakers_row = column![
            row![
                text("Number of speakers").size(14).color(t.text).width(Length::Fill),
                row((0..=6usize).map(|n| {
                    button(text(if n == 0 { "Auto".to_string() } else { n.to_string() }).size(13))
                        .padding([6, 12])
                        .style(theme::segment(t, self.settings.num_speakers == n))
                        .on_press(Message::SetSpeakers(n))
                        .into()
                }))
                .spacing(6),
            ]
            .align_y(Alignment::Center),
            text("Auto works out how many people are talking (1, 2, 3 or more). Choose a number only if Auto gets it wrong.")
                .size(12)
                .color(t.muted),
        ]
        .spacing(4);
        let cleanup_row = row![
            icons::icon(icons::SPARKLES, 18.0, t.accent),
            column![
                text("Clean up text with AI").size(14).color(t.text),
                text("Fixes misheard words and punctuation. Slower; needs the cleanup model.")
                    .size(12)
                    .color(t.muted)
            ]
            .spacing(2)
            .width(Length::Fill),
            toggler(self.settings.cleanup).on_toggle(Message::ToggleCleanup),
        ]
        .align_y(Alignment::Center)
        .spacing(12);
        let processing = section(
            t,
            "Processing",
            column![speakers_row, cleanup_row].spacing(16).into(),
        );

        let body = column![
            self.system_section(),
            appearance,
            folder,
            processing,
            section(
                t,
                "Profiles",
                profiles_ui::view(&self.profiles, &self.settings, t).map(Message::Profiles)
            ),
            self.formats_section(),
            self.models_section()
        ]
        .spacing(14)
        .padding([0, 22]);

        container(
            column![
                row![
                    text("Settings").size(22).font(semibold()).color(t.text),
                    space::horizontal(),
                    button(icons::icon(icons::X, 16.0, t.text))
                        .padding(8)
                        .style(theme::ghost(t))
                        .on_press(Message::CloseSettings),
                ]
                .align_y(Alignment::Center)
                .padding([0, 22]),
                scrollable(body).height(Length::Fill),
            ]
            .spacing(14),
        )
        .padding([18, 0])
        .width(Length::Fill)
        .max_width(780)
        .height(Length::Fill)
        .max_height(700)
        .style(theme::dialog(t))
        .into()
    }

    /// Download offer / progress / error for ffmpeg (used in the notice and in Settings).
    fn ffmpeg_offer(&self) -> Element<'_, Message> {
        let t = self.tokens;
        if let Some(p) = &self.ffmpeg_dl {
            let (label, frac) = if p.unpacking {
                ("Unpacking ffmpeg...".to_string(), 1.0)
            } else if p.total > 0 && p.done > 0 {
                (
                    format!("Downloading ffmpeg  {} / {}", human(p.done), human(p.total)),
                    p.done as f32 / p.total as f32,
                )
            } else {
                ("Connecting to the download server...".to_string(), 0.0)
            };
            return column![
                row![
                    self.spinner(14.0, t.accent),
                    text(label).size(12).color(t.muted)
                ]
                .spacing(8)
                .align_y(Alignment::Center),
                progress_bar(0.0..=1.0, frac)
                    .length(Length::Fill)
                    .girth(6)
                    .style(theme::bar(t)),
            ]
            .spacing(6)
            .into();
        }
        let mut col = column![].spacing(6);
        if let Some(e) = &self.ffmpeg_err {
            col = col.push(
                text(format!(
                    "The download did not work: {}",
                    clean_separators(e)
                ))
                .size(12)
                .color(t.bad),
            );
        }
        match engine::tools::ffmpeg_download() {
            Some(tool) => {
                col = col.push(
                    row![
                        button(
                            row![
                                icons::icon(icons::DOWNLOAD, 14.0, t.on_accent),
                                text(format!("Download ffmpeg ({} MB)", tool.size_bytes / 1_000_000)).size(13)
                            ]
                            .spacing(6)
                            .align_y(Alignment::Center)
                        )
                        .padding([7, 12])
                        .style(theme::primary(t))
                        .on_press(Message::GetFfmpeg),
                        text("One time, into ViPr's own folder. Nothing is installed on your computer. Waiting files are added afterwards.")
                            .size(12)
                            .color(t.muted)
                            .width(Length::Fill),
                    ]
                    .spacing(12)
                    .align_y(Alignment::Center),
                );
            }
            None => {
                col = col.push(
                    text("Install ffmpeg with your package manager (for example `brew install ffmpeg` or `apt install ffmpeg`), or choose it in Settings > More formats. Then add the files again.")
                        .size(12)
                        .color(t.muted),
                );
            }
        }
        col.into()
    }

    /// Which file formats work: always the built-in ones, more when ffmpeg is available.
    fn formats_section(&self) -> Element<'_, Message> {
        let t = self.tokens;
        let ours = self
            .ffmpeg
            .as_ref()
            .is_some_and(|p| engine::tools::installed_ffmpeg().as_ref() == Some(p));
        let status: Element<'_, Message> = match &self.ffmpeg {
            Some(p) => row![
                icons::icon(icons::CHECK, 16.0, t.ok),
                column![
                    text(if ours { "ffmpeg ready (downloaded by ViPr) - extra formats are on" } else { "ffmpeg found - extra formats are on" })
                        .size(13)
                        .color(t.ok),
                    text(clean_separators(&p.display().to_string()))
                        .size(12)
                        .color(t.muted),
                ]
                .spacing(2)
                .width(Length::Fill),
                button(text("Remove").size(12))
                    .padding([6, 10])
                    .style(theme::ghost(t))
                    .on_press_maybe(ours.then_some(Message::RemoveFfmpeg)),
            ]
            .spacing(8)
            .align_y(Alignment::Center)
            .into(),
            None => column![
                row![
                    icons::icon(icons::ALERT, 16.0, t.warn),
                    column![
                        text("ffmpeg not found - only the built-in formats work").size(13).color(t.warn),
                        text("ffmpeg adds AVI, WMV / WMA, FLV, MPEG-TS, MPG / VOB, 3GP, AMR, AC3 and more.").size(12).color(t.muted),
                    ]
                    .spacing(2),
                ]
                .spacing(8)
                .align_y(Alignment::Start),
                self.ffmpeg_offer(),
            ]
            .spacing(10)
            .into(),
        };
        section(
            t,
            "More formats",
            column![
                text("Built in: MP4, M4V, MOV, MKV, WebM, 3GP, MP3, WAV, FLAC, OGG, Opus, M4A, AAC, AIFF, CAF").size(12).color(t.muted),
                status,
                row![
                    button(text("Choose ffmpeg...").size(13)).padding([7, 12]).style(theme::outline_btn(t)).on_press(Message::ChooseFfmpeg),
                    button(text("Search automatically").size(13))
                        .padding([7, 12])
                        .style(theme::ghost(t))
                        .on_press_maybe(self.settings.ffmpeg_path.is_some().then_some(Message::ResetFfmpeg)),
                ]
                .spacing(8),
            ]
            .spacing(10)
            .into(),
        )
    }

    fn models_section(&self) -> Element<'_, Message> {
        let t = self.tokens;
        let reg = match &self.registry {
            Err(e) => {
                return section(
                    t,
                    "Models",
                    text(format!(
                        "The model list could not be loaded: {}",
                        clean_separators(e)
                    ))
                    .color(t.bad)
                    .into(),
                );
            }
            Ok(r) => r,
        };
        let Some(det) = self.detection.as_ref() else {
            return section(
                t,
                "Models",
                row![
                    self.spinner(18.0, t.accent),
                    text("Checking your hardware to pick the best models...")
                        .size(13)
                        .color(t.muted)
                ]
                .spacing(10)
                .align_y(Alignment::Center)
                .into(),
            );
        };
        let missing: Vec<&ModelEntry> = self
            .selected_entries()
            .into_iter()
            .filter(|m| !matches!(self.store.status(m), Status::Present))
            .collect();
        let missing_bytes: u64 = missing.iter().map(|m| m.size_bytes).sum();
        let busy = !self.downloading.is_empty();

        let summary: Element<'_, Message> = if missing.is_empty() {
            row![
                icons::icon(icons::CHECK, 16.0, t.ok),
                text("All selected models are downloaded.")
                    .size(13)
                    .color(t.ok)
            ]
            .spacing(8)
            .align_y(Alignment::Center)
            .into()
        } else {
            row![
                text(format!(
                    "{} of {} models still need downloading ({})",
                    missing.len(),
                    self.selected_entries().len(),
                    human(missing_bytes)
                ))
                .size(13)
                .color(t.warn)
                .width(Length::Fill),
                button(
                    row![
                        icons::icon(icons::DOWNLOAD, 15.0, t.on_accent),
                        text("Download all").size(13)
                    ]
                    .spacing(6)
                    .align_y(Alignment::Center)
                )
                .padding([7, 12])
                .style(theme::primary(t))
                .on_press_maybe((!busy).then_some(Message::DownloadAllMissing)),
            ]
            .align_y(Alignment::Center)
            .into()
        };

        let mut col = column![summary].spacing(16);
        for role in ROLES {
            col = col.push(self.model_row(reg, role, det));
        }
        // right after text cleanup, the last of the models above
        col = col.push(self.model_row(reg, Role::Translate, det));
        section(t, "Models", col.into())
    }

    fn model_row<'a>(
        &'a self,
        reg: &'a Registry,
        role: Role,
        det: &'a Detection,
    ) -> Element<'a, Message> {
        let t = self.tokens;
        let mut options: Vec<Choice> = reg
            .for_role(role)
            .filter(|m| role != Role::Translate || can_translate(m))
            .map(|m| Choice {
                id: m.id.clone(),
                label: m.name.clone(),
            })
            .collect();
        if role == Role::Translate {
            options.insert(
                0,
                Choice {
                    id: String::new(),
                    label: "Same as text cleanup".to_string(),
                },
            );
        }
        let selected = match self.selection.get(&role) {
            Some(id) => options.iter().find(|c| &c.id == id).cloned(),
            None if role == Role::Translate => options.first().cloned(),
            None => None,
        };
        let entry = self.selection.get(&role).and_then(|id| reg.get(id).ok());

        let mut right = column![
            pick_list(options, selected, move |c| Message::Pick(role, c))
                .style(theme::pick(t))
                .padding(9)
                .width(Length::Fill)
        ]
        .spacing(6);

        if let Some(m) = entry {
            let f = fit(m, det);
            let (fit_text, fit_color) = match f {
                Fit::Fits => ("Fits your GPU", t.ok),
                Fit::Tight => ("Tight on your GPU", t.warn),
                Fit::TooBig => ("Too big for your GPU", t.bad),
                Fit::Unknown => ("GPU memory unknown", t.muted),
                Fit::Unsupported => ("Not supported on this backend", t.bad),
            };
            let facts = if m.min_vram_mb > 0 {
                format!(
                    "{}  -  needs about {} of GPU memory",
                    human(m.size_bytes),
                    human(m.min_vram_mb << 20)
                )
            } else {
                format!("{}  -  small, runs anywhere", human(m.size_bytes))
            };
            let status: Element<'a, Message> = if let Some(p) = self.downloading.get(&m.id) {
                let pct = if p.total > 0 {
                    p.done as f32 / p.total as f32
                } else {
                    0.0
                };
                if p.done == 0 {
                    row![
                        self.spinner(14.0, t.accent),
                        text("Connecting to the download server...")
                            .size(12)
                            .color(t.muted)
                    ]
                    .spacing(8)
                    .align_y(Alignment::Center)
                    .into()
                } else {
                    row![
                        progress_bar(0.0..=1.0, pct)
                            .length(Length::Fill)
                            .girth(6)
                            .style(theme::bar(t)),
                        text(format!("{} / {}", human(p.done), human(p.total)))
                            .size(12)
                            .color(t.muted),
                    ]
                    .spacing(10)
                    .align_y(Alignment::Center)
                    .into()
                }
            } else if let Some(err) = self.errors.get(&m.id) {
                row![
                    text(format!("Download failed: {}", clean_separators(err)))
                        .size(12)
                        .color(t.bad)
                        .width(Length::Fill),
                    button(text("Retry").size(12))
                        .padding([6, 10])
                        .style(theme::outline_btn(t))
                        .on_press(Message::Download(m.id.clone())),
                ]
                .spacing(10)
                .align_y(Alignment::Center)
                .into()
            } else {
                match self.store.status(m) {
                    Status::Present => row![
                        icons::icon(icons::CHECK, 14.0, t.ok),
                        text("Downloaded").size(12).color(t.ok)
                    ]
                    .spacing(6)
                    .align_y(Alignment::Center)
                    .into(),
                    Status::LocalMissing => text("File not found").size(12).color(t.bad).into(),
                    st => {
                        let label = match st {
                            Status::Partial(n) => format!("Resume ({} done)", human(n)),
                            _ => format!("Download ({})", human(m.size_bytes)),
                        };
                        let blocked = matches!(f, Fit::TooBig | Fit::Unsupported);
                        row![
                            text("Not downloaded").size(12).color(t.warn),
                            space::horizontal(),
                            button(
                                row![
                                    icons::icon(icons::DOWNLOAD, 14.0, t.on_accent),
                                    text(label).size(12)
                                ]
                                .spacing(6)
                                .align_y(Alignment::Center)
                            )
                            .padding([6, 11])
                            .style(theme::primary(t))
                            .on_press_maybe((!blocked).then(|| Message::Download(m.id.clone()))),
                        ]
                        .align_y(Alignment::Center)
                        .into()
                    }
                }
            };
            right = right
                .push(
                    row![
                        text(fit_text).size(12).color(fit_color),
                        text("-").size(12).color(t.faint),
                        text(facts).size(12).color(t.muted)
                    ]
                    .spacing(8),
                )
                .push(status);
        }

        row![
            text(role_label(role))
                .size(14)
                .color(t.text)
                .width(Length::Fixed(150.0)),
            right.width(Length::Fill)
        ]
        .spacing(14)
        .into()
    }
}

// ------------------------------------------------------------------ view helpers

fn semibold() -> Font {
    Font {
        weight: font::Weight::Semibold,
        ..Font::DEFAULT
    }
}

fn section<'a>(t: Tokens, title: &'a str, body: Element<'a, Message>) -> Element<'a, Message> {
    container(column![text(title).size(15).font(semibold()).color(t.text), body].spacing(14))
        .padding(16)
        .width(Length::Fill)
        .style(theme::card(t))
        .into()
}

fn nav_button<'a>(
    t: Tokens,
    icon: &'static str,
    label: &'a str,
    active: bool,
    msg: Message,
) -> Element<'a, Message> {
    button(
        row![
            icons::icon(icon, 18.0, if active { t.accent } else { t.muted }),
            text(label).size(14)
        ]
        .spacing(12)
        .align_y(Alignment::Center),
    )
    .padding([10, 12])
    .width(Length::Fill)
    .style(theme::nav_item(t, active))
    .on_press(msg)
    .into()
}

/// File types that are sound only (shown with a waveform icon instead of a camera).
fn is_audio_file(path: &std::path::Path) -> bool {
    const AUDIO: [&str; 18] = [
        "mp3", "wav", "m4a", "aac", "flac", "ogg", "opus", "oga", "mka", "wma", "amr", "mp2",
        "aif", "aiff", "caf", "ac3", "wv", "ape",
    ];
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| AUDIO.contains(&e.to_lowercase().as_str()))
}

/// Models tagged for translation only each need a prompt format of their own, which isn't wired up
/// (`jobs::model_id` refuses them), so they aren't offered.
fn can_translate(model: &ModelEntry) -> bool {
    model.has_role(Role::Translate) && model.has_role(Role::Llm)
}

/// Whose model a translation runs on.
fn translate_role(own_model_chosen: bool) -> Role {
    if own_model_chosen {
        Role::Translate
    } else {
        Role::Llm
    }
}

fn role_label(role: Role) -> &'static str {
    match role {
        Role::Asr => "Speech to text",
        Role::Vad => "Silence detection",
        Role::DiarSegmentation => "Speaker segmentation",
        Role::DiarEmbedding => "Speaker identity",
        Role::Llm => "Text cleanup (AI)",
        Role::Translate => "Translation",
    }
}

fn human(bytes: u64) -> String {
    const MB: f64 = 1024.0 * 1024.0;
    let mb = bytes as f64 / MB;
    if mb >= 1024.0 {
        format!("{:.1} GB", mb / 1024.0)
    } else if mb >= 1.0 {
        format!("{mb:.0} MB")
    } else {
        format!("{} KB", bytes / 1024)
    }
}

/// One reading of system load.
#[derive(Debug, Clone, Copy)]
struct Usage {
    cpu: f32,
    ram_used: u64,
    ram_total: u64,
    gpu: gpu_detect::GpuUsage,
}

impl Usage {
    /// One line: "CPU 34%  -  RAM 6.1 / 16 GB  -  GPU 78%  -  VRAM 1.4 / 4.0 GB".
    fn summary(&self) -> String {
        let mut s = format!(
            "CPU {:.0}%  -  RAM {} / {}",
            self.cpu,
            human(self.ram_used),
            human(self.ram_total)
        );
        if let (Some(g), Some(used), Some(total)) = (
            self.gpu.gpu_percent,
            self.gpu.vram_used_mb,
            self.gpu.vram_total_mb,
        ) {
            s.push_str(&format!(
                "  -  GPU {g}%  -  VRAM {} / {}",
                human(used << 20),
                human(total << 20)
            ));
        }
        s
    }
}

/// Sample the system once a second on a plain thread; stops when the subscription is dropped.
fn usage_stream() -> impl Stream<Item = Usage> {
    iced::stream::channel(4, async |mut out: mpsc::Sender<Usage>| {
        let (tx, mut rx) = mpsc::unbounded::<Usage>();
        std::thread::spawn(move || {
            test_delay();
            let Ok(mut sys) = sysinfo::System::new() else {
                return;
            };
            let gpu = gpu_detect::GpuMonitor::new();
            loop {
                sys.refresh_cpu_usage();
                sys.refresh_memory();
                let reading = Usage {
                    cpu: sys.global_cpu_usage(),
                    ram_used: sys.used_memory(),
                    ram_total: sys.total_memory(),
                    gpu: gpu.sample(),
                };
                if tx.unbounded_send(reading).is_err() {
                    break; // the app stopped listening
                }
                std::thread::sleep(Duration::from_millis(1000));
            }
        });
        while let Some(u) = rx.next().await {
            if out.send(u).await.is_err() {
                break;
            }
        }
    })
}

fn mmss(secs: u64) -> String {
    format!("{}:{:02}", secs / 60, secs % 60)
}

/// Label, filled bar (turns amber / red when nearly full) and value.
fn meter<'a>(t: Tokens, label: &'a str, fraction: f32, value: String) -> Element<'a, Message> {
    let f = fraction.clamp(0.0, 1.0);
    let color = if f > 0.92 {
        t.bad
    } else if f > 0.75 {
        t.warn
    } else {
        t.accent
    };
    row![
        text(label)
            .size(13)
            .color(t.text)
            .width(Length::Fixed(120.0)),
        progress_bar(0.0..=1.0, f)
            .length(Length::Fill)
            .girth(8)
            .style(theme::bar_tint(t, color)),
        text(value)
            .size(12)
            .color(t.muted)
            .width(Length::Fixed(130.0)),
    ]
    .spacing(12)
    .align_y(Alignment::Center)
    .into()
}

/// Thin invisible areas along the window border that start a resize drag.
fn resize_edges<'a>() -> Element<'a, Message> {
    use iced::mouse::Interaction;
    use window::Direction as D;
    let edge = |dir: D, w: Length, h: Length| -> Element<'a, Message> {
        let cursor = match dir {
            D::North | D::South => Interaction::ResizingVertically,
            D::East | D::West => Interaction::ResizingHorizontally,
            D::NorthWest | D::SouthEast => Interaction::ResizingDiagonallyDown,
            D::NorthEast | D::SouthWest => Interaction::ResizingDiagonallyUp,
        };
        mouse_area(space::horizontal().width(w).height(h))
            .on_press(Message::ResizeEdge(dir))
            .interaction(cursor)
            .into()
    };
    let (corner, thin) = (Length::Fixed(10.0), Length::Fixed(4.0));
    let top = row![
        edge(D::NorthWest, corner, corner),
        edge(D::North, Length::Fill, thin),
        edge(D::NorthEast, corner, corner)
    ];
    let middle = row![
        edge(D::West, thin, Length::Fill),
        space::horizontal(),
        edge(D::East, thin, Length::Fill)
    ]
    .height(Length::Fill);
    let bottom = row![
        edge(D::SouthWest, corner, corner),
        edge(D::South, Length::Fill, thin),
        edge(D::SouthEast, corner, corner)
    ]
    .align_y(Alignment::End);
    container(column![top, middle, bottom])
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

// ------------------------------------------------------------------ async helpers

/// Check every file (name first, then what is inside) on a plain thread, so the window never waits for the disk.
async fn check_files(files: Vec<PathBuf>) -> Vec<(PathBuf, engine::audio::Probe)> {
    let (tx, rx) = iced::futures::channel::oneshot::channel();
    std::thread::spawn(move || {
        let results = files
            .into_iter()
            .map(|f| {
                let verdict = if f.is_dir() {
                    engine::audio::Probe::Invalid("is a folder, not a file".into())
                } else if !has_media_extension(&f) {
                    engine::audio::Probe::Invalid("is not a video or audio file".into())
                } else {
                    engine::audio::probe_media(&f)
                };
                (f, verdict)
            })
            .collect();
        let _ = tx.send(results);
    });
    rx.await.unwrap_or_default()
}

/// A pressed key as its Latin letter, so shortcuts work on any keyboard layout. Keys without one are
/// left as they are.
fn latin_key(key: keyboard::Key, physical: keyboard::key::Physical) -> keyboard::Key {
    use keyboard::key::{Code, Physical};
    let latin = key.to_latin(physical).or(
        // `to_latin` knows only letters and digits, and Settings is Ctrl+,
        match (&key, physical) {
            (keyboard::Key::Character(_), Physical::Code(Code::Comma)) => Some(','),
            _ => None,
        },
    );
    match latin {
        Some(c) => keyboard::Key::Character(c.to_string().into()),
        None => key,
    }
}

/// Dev helper: `MT_SLOW=5000` delays hardware detection and the first usage reading by 5 s, to look at the loading states.
fn test_delay() {
    if let Some(ms) = std::env::var("MT_SLOW")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        std::thread::sleep(Duration::from_millis(ms));
    }
}

/// Hardware detection, off the UI thread and **out of this process**.
///
/// Detection lists the graphics adapters through wgpu. Doing that inside the running GUI creates a second
/// graphics instance next to the one Iced uses and made the GUI crash with "Parent device is lost", so the
/// `detect` tool (shipped next to the app) does it in its own process. If that tool is missing, a detection
/// that only asks NVML (no graphics stack) is used instead.
async fn detect_async() -> Option<Detection> {
    let (tx, rx) = iced::futures::channel::oneshot::channel();
    std::thread::spawn(move || {
        test_delay();
        let found = detect_in_child_process().unwrap_or_else(gpu_detect::detect_without_graphics);
        let _ = tx.send(found);
    });
    rx.await.ok()
}

fn detect_in_child_process() -> Option<Detection> {
    let exe = pipeline::find_worker("detect")?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--json");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // no console window
    }
    let out = cmd.output().ok().filter(|o| o.status.success())?;
    serde_json::from_slice(&out.stdout).ok()
}

/// Download + unpack ffmpeg on its own thread and report progress.
fn ffmpeg_stream() -> impl Stream<Item = FfmpegEvent> {
    iced::stream::channel(32, async move |mut out: mpsc::Sender<FfmpegEvent>| {
        let (tx, mut rx) = mpsc::unbounded::<FfmpegEvent>();
        std::thread::spawn(move || {
            let mut last = 0u64;
            let result = engine::tools::install_ffmpeg(&mut |step| match step {
                engine::tools::InstallStep::Downloading { done, total } => {
                    if done.saturating_sub(last) >= (512 << 10) || done == total {
                        last = done;
                        let _ = tx.unbounded_send(FfmpegEvent::Downloading { done, total });
                    }
                }
                engine::tools::InstallStep::Unpacking => {
                    let _ = tx.unbounded_send(FfmpegEvent::Unpacking);
                }
            });
            let _ = tx.unbounded_send(FfmpegEvent::Finished(result.map_err(|e| e.to_string())));
        });
        while let Some(ev) = rx.next().await {
            let finished = matches!(ev, FfmpegEvent::Finished(_));
            let _ = out.send(ev).await;
            if finished {
                break;
            }
        }
    })
}

/// A tick every 70 ms for spinner animation; the thread ends when the subscription is dropped.
/// A rough figure on purpose: the estimate moves around, and a clock counting in seconds would jitter with it.
fn time_left(seconds: u64) -> String {
    match seconds {
        0..=59 => "less than a minute left".to_string(),
        60..=119 => "about a minute left".to_string(),
        _ => format!("about {} minutes left", (seconds + 30) / 60),
    }
}

/// How far the current step is, next to what it is doing.
fn step_share(fraction: f32) -> String {
    let percent = jobs::percent_text(fraction);
    if percent.is_empty() {
        percent
    } else {
        format!("{percent} of this step")
    }
}

fn tick_stream() -> impl Stream<Item = ()> {
    iced::stream::channel(2, async |mut out: mpsc::Sender<()>| {
        let (tx, mut rx) = mpsc::unbounded::<()>();
        std::thread::spawn(move || {
            loop {
                // often enough for the spinner and progress bars to move smoothly
                std::thread::sleep(Duration::from_millis(33));
                if tx.unbounded_send(()).is_err() {
                    break;
                }
            }
        });
        while rx.next().await.is_some() {
            if out.send(()).await.is_err() {
                break;
            }
        }
    })
}

async fn pick_files() -> Vec<PathBuf> {
    rfd::AsyncFileDialog::new()
        .set_title("Add videos or audio")
        .add_filter("Video / audio", MEDIA_EXTENSIONS)
        .pick_files()
        .await
        .map(|files| files.into_iter().map(|f| f.path().to_path_buf()).collect())
        .unwrap_or_default()
}

async fn pick_ffmpeg() -> Option<PathBuf> {
    let mut dialog = rfd::AsyncFileDialog::new().set_title("Choose ffmpeg");
    if cfg!(windows) {
        dialog = dialog.add_filter("ffmpeg", &["exe"]);
    }
    dialog.pick_file().await.map(|f| f.path().to_path_buf())
}

/// A result folder, or the `<stem>.json` inside one.
async fn pick_result(folder: bool) -> Option<PathBuf> {
    let dialog = rfd::AsyncFileDialog::new().set_title("Open a saved result");
    let picked = if folder {
        dialog.pick_folder().await
    } else {
        dialog
            .add_filter("ViPr result", &["json"])
            .pick_file()
            .await
    };
    picked.map(|f| f.path().to_path_buf())
}

async fn pick_media(name: String) -> Option<PathBuf> {
    rfd::AsyncFileDialog::new()
        .set_title(format!("Locate {name}"))
        .add_filter("Video / audio", MEDIA_EXTENSIONS)
        .add_filter("All files", &["*"])
        .pick_file()
        .await
        .map(|f| f.path().to_path_buf())
}

async fn pick_folder() -> Option<PathBuf> {
    rfd::AsyncFileDialog::new()
        .set_title("Choose the results folder")
        .pick_folder()
        .await
        .map(|f| f.path().to_path_buf())
}

/// Runs the whole pipeline for one file on its own thread and streams events to the UI.
fn job_stream(input: pipeline::JobInput) -> impl Stream<Item = pipeline::Event> {
    iced::stream::channel(64, async move |mut out: mpsc::Sender<pipeline::Event>| {
        let (tx, mut rx) = mpsc::unbounded::<pipeline::Event>();
        std::thread::spawn(move || {
            let tx2 = tx.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut emit = |e: pipeline::Event| {
                    let _ = tx.unbounded_send(e);
                };
                pipeline::run(&input, &mut emit)
            }));
            match result {
                Ok(Ok(())) => {}
                Ok(Err(msg)) => {
                    let _ = tx2.unbounded_send(pipeline::Event::Failed(msg));
                }
                Err(_) => {
                    let _ = tx2.unbounded_send(pipeline::Event::Failed(
                        "Internal error while processing this file.".into(),
                    ));
                }
            }
        });
        while let Some(ev) = rx.next().await {
            let _ = out.send(ev).await;
        }
    })
}

/// Runs the blocking download on its own thread and streams progress to the UI.
fn download_stream(
    entry: ModelEntry,
    dir: PathBuf,
    token: Option<String>,
) -> impl Stream<Item = DownloadEvent> {
    iced::stream::channel(32, async move |mut out: mpsc::Sender<DownloadEvent>| {
        let (tx, mut rx) = mpsc::unbounded::<DownloadEvent>();
        std::thread::spawn(move || {
            let store = ModelStore::new(dir);
            let mut last = 0u64;
            let result = store.fetch(&entry, token.as_deref(), &mut |done, total| {
                if done.saturating_sub(last) >= (1 << 20) || (total > 0 && done == total) {
                    last = done;
                    let _ = tx.unbounded_send(DownloadEvent::Progress { done, total });
                }
            });
            let _ = tx.unbounded_send(DownloadEvent::Finished(
                result.map(|_| ()).map_err(|e| e.to_string()),
            ));
        });
        while let Some(ev) = rx.next().await {
            let finished = matches!(ev, DownloadEvent::Finished(_));
            let _ = out.send(ev).await;
            if finished {
                break;
            }
        }
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn typed_letters_find_languages_anywhere_in_their_name() {
        let found = |query: &str| -> Vec<String> {
            super::translate_options(None)
                .iter()
                .map(ToString::to_string)
                .filter(|name| name.to_lowercase().contains(query))
                .collect()
        };
        assert!(found("pers").contains(&"Persian (fa)".to_string()), "start");
        assert!(
            found("rsia").contains(&"Persian (fa)".to_string()),
            "middle"
        );
        assert!(found("(fa)").contains(&"Persian (fa)".to_string()), "code");
    }

    #[test]
    fn a_transcript_is_not_offered_its_own_language() {
        let names = |source| -> Vec<&str> {
            super::translate_options(source)
                .iter()
                .map(|l| l.0.code)
                .collect()
        };
        assert!(names(None).contains(&"en"));
        assert!(!names(Some("en")).contains(&"en"));
        assert_eq!(names(Some("en")).len() + 1, names(None).len());
    }

    #[test]
    fn the_time_left_stays_a_rough_figure() {
        assert_eq!(super::time_left(20), "less than a minute left");
        assert_eq!(super::time_left(90), "about a minute left");
        assert_eq!(super::time_left(200), "about 3 minutes left");
        assert_eq!(super::time_left(3600), "about 60 minutes left");
    }

    #[test]
    fn translation_uses_its_own_model_only_once_one_is_chosen() {
        assert_eq!(super::translate_role(false), super::Role::Llm);
        assert_eq!(super::translate_role(true), super::Role::Translate);
    }

    use super::*;
    use keyboard::Key;
    use keyboard::key::{Code, Named, Physical};

    #[test]
    fn shortcut_keys_work_on_any_layout() {
        let persian = |c: &str, code| latin_key(Key::Character(c.into()), Physical::Code(code));
        assert_eq!(persian("\u{0645}", Code::KeyM), Key::Character("m".into()));
        assert_eq!(persian("\u{0633}", Code::KeyS), Key::Character("s".into()));
        assert_eq!(persian("\u{0648}", Code::Comma), Key::Character(",".into()));
        // a Latin layout keeps its own letter, wherever that key is
        assert_eq!(persian("z", Code::KeyY), Key::Character("z".into()));
        let named = Key::Named(Named::Escape);
        assert_eq!(
            latin_key(named.clone(), Physical::Code(Code::Escape)),
            named
        );
    }

    #[test]
    fn a_file_name_is_cleaned_before_it_is_shown() {
        let name = "clip\u{2029}\u{0633}\u{0644}\u{0627}\u{0645} x.mp4";
        let job = Job::new(1, std::path::Path::new("media").join(name), 0);
        assert_eq!(job.path.file_name().unwrap().to_string_lossy(), name);
        assert!(!job.name().contains('\u{2029}'));
        assert!(job.name().contains("x.mp4"));
    }
}
