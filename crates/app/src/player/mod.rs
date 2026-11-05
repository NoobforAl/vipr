//! The player page: video with a subtitle overlay, or subtitles alone for audio files.
//!
//! `main.rs` depends only on the items in this file. `session` runs ffmpeg, the audio stream and the
//! clock, `surface` draws frames on Iced's device, `probe` reads what the file holds and `subs` loads
//! the subtitle tracks.

mod probe;
mod session;
mod subs;
mod surface;

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::Ordering::Acquire;
use std::time::{Duration, Instant};

use engine::marks::{Mark, MarkKind};
use iced::futures::channel::{mpsc, oneshot};
use iced::futures::{SinkExt, Stream, StreamExt};
use iced::keyboard::key::Named;
use iced::keyboard::{Key, Modifiers};
use iced::widget::text::{self, Shaping, Wrapping};
use iced::widget::{
    button, column, container, row, scrollable, shader, slider, space, stack, text as label,
};
use iced::{Alignment, Color, Element, Length, Subscription, Task};

use crate::out::{JobRef, Out, Position, content_align};
use crate::theme::{self, Tokens};

use probe::Probe;
use session::{Config, Session, Shared};
use subs::{Track, TrackSet, load_tracks};

const NUDGE_SECS: f64 = 5.0;
const VOLUME_STEP: f32 = 0.1;
/// With no `prepare` by now there's no GPU backend, or the window is minimized.
const NO_SURFACE_AFTER: Duration = Duration::from_secs(4);
/// Sound can run a little past the container's duration before the pipe closes.
const END_GRACE_SECS: f64 = 0.25;
/// Cues shown before and after the current one in the audio-only view.
const TRANSCRIPT_CONTEXT: usize = 3;
/// Half the slider's handle: its rail is inset by this, and the mark ticks have to line up.
const SLIDER_INSET: f32 = 7.0;
const SEEK_BAR_HEIGHT: f32 = 16.0;

#[derive(Debug, Clone, Default, PartialEq)]
enum Phase {
    #[default]
    Idle,
    /// Reading the file, then starting the session. The video area stays black.
    Opening,
    NoFfmpeg,
    /// Can't play this, for the reason given.
    Failed(String),
    Playing,
}

#[derive(Debug)]
pub struct State {
    job: Option<JobRef>,
    /// Results from before the last `open` or `close` are dropped by this.
    token: u64,
    phase: Phase,
    session: Option<Arc<Session>>,
    shared: Option<Arc<Shared>>,
    has_video: bool,
    /// Zero when the file doesn't say.
    duration: f64,
    tracks: Vec<Track>,
    active: usize,
    position: f64,
    paused: bool,
    /// Played to the end: paused there, and the next play starts over.
    finished: bool,
    /// Shown instead of the clock until the seek reaches the screen.
    seek_target: Option<f64>,
    /// A seek that came before the session existed, as `Out::PlayAt` does.
    pending_seek: Option<f64>,
    /// Kept between files.
    volume: f32,
    opened_at: Option<Instant>,
    surface_seen: bool,
    no_surface: bool,
    video_failure_told: bool,
    notices: Vec<String>,
}

impl Default for State {
    fn default() -> Self {
        State {
            job: None,
            token: 0,
            phase: Phase::Idle,
            session: None,
            shared: None,
            has_video: false,
            duration: 0.0,
            tracks: Vec::new(),
            active: 0,
            position: 0.0,
            paused: false,
            finished: false,
            seek_target: None,
            pending_seek: None,
            volume: 1.0,
            opened_at: None,
            surface_seen: false,
            no_surface: false,
            video_failure_told: false,
            notices: Vec::new(),
        }
    }
}

/// What the background step found.
#[derive(Debug)]
pub struct Loaded {
    ffmpeg: Option<PathBuf>,
    media_found: bool,
    probe: Probe,
    set: TrackSet,
}

#[derive(Debug, Clone)]
pub enum Msg {
    /// Seek to this many seconds. `main.rs` sends it for `Out::PlayAt`.
    SeekTo(f64),
    Loaded(u64, Option<Arc<Loaded>>),
    Started(u64, Option<Arc<Session>>),
    Tick,
    TogglePause,
    Nudge(f64),
    Volume(f32),
    SelectTrack(usize),
    NextTrack,
    /// Read the tracks again, after a translation finishes.
    ReloadTracks,
    TracksLoaded(u64, Option<Arc<TrackSet>>),
    /// Try opening the same job again, after ffmpeg arrives or the media comes back.
    Retry,
}

fn background<T, W, D>(work: W, done: D) -> Task<Msg>
where
    T: Send + 'static,
    W: FnOnce() -> T + Send + 'static,
    D: FnOnce(Option<T>) -> Msg + Send + 'static,
{
    let (tx, rx) = oneshot::channel();
    std::thread::spawn(move || {
        let _ = tx.send(work());
    });
    Task::perform(async move { rx.await.ok() }, done)
}

fn run_probe(ffmpeg: &std::path::Path, input: &std::path::Path) -> Probe {
    let mut cmd = Command::new(ffmpeg);
    cmd.args(["-hide_banner", "-nostdin", "-i"]).arg(input);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // no console window
    }
    match cmd.output() {
        Ok(out) => probe::parse_probe(&String::from_utf8_lossy(&out.stderr)),
        Err(_) => Probe::default(),
    }
}

fn load(job: &JobRef) -> Loaded {
    let ffmpeg = engine::audio::find_ffmpeg();
    let media_found = job.media.is_file();
    let probe = match &ffmpeg {
        Some(path) if media_found => run_probe(path, &job.media),
        _ => Probe::default(),
    };
    Loaded {
        ffmpeg,
        media_found,
        probe,
        set: load_tracks(&job.paths),
    }
}

fn start_loading(state: &mut State) -> Task<Msg> {
    let Some(job) = state.job.clone() else {
        return Task::none();
    };
    state.phase = Phase::Opening;
    let token = state.token;
    background(
        move || Arc::new(load(&job)),
        move |loaded| Msg::Loaded(token, loaded),
    )
}

/// The file name only, as paths in notices tell nothing more and can run long.
fn media_name(job: &JobRef) -> String {
    let name = job.media.file_name().unwrap_or_default().to_string_lossy();
    engine::lang::clean_separators(&name)
}

/// What to do with a loaded file, or why it can't play.
fn phase_after_load(loaded: &Loaded, media: &str) -> Result<PathBuf, Phase> {
    if !loaded.media_found {
        return Err(Phase::Failed(format!("The media file is missing: {media}")));
    }
    let Some(ffmpeg) = &loaded.ffmpeg else {
        return Err(Phase::NoFfmpeg);
    };
    if !loaded.probe.has_video && !loaded.probe.has_audio() {
        return Err(Phase::Failed(format!(
            "ffmpeg found no video or sound in {media}"
        )));
    }
    Ok(ffmpeg.clone())
}

/// More tracks than the one being played. Titles come from the file, so they're untrusted.
fn track_notice(probe: &Probe) -> Option<String> {
    let total = probe.audio_tracks.len();
    if total < 2 {
        return None;
    }
    let title = probe.audio_tracks[0]
        .title
        .as_deref()
        .map(|t| format!(" ({})", engine::lang::clean_separators(t)))
        .unwrap_or_default();
    Some(format!("Playing audio track 1 of {total}{title}."))
}

/// The page is opening on `job`.
pub fn open(state: &mut State, job: &JobRef) -> (Task<Msg>, Out) {
    close(state);
    state.job = Some(job.clone());
    (start_loading(state), Out::None)
}

/// The page is closing, or the window is. Ends the playback session.
pub fn close(state: &mut State) {
    if let Some(session) = state.session.take() {
        session.shutdown();
    }
    let volume = state.volume;
    let token = state.token + 1;
    *state = State {
        volume,
        token,
        ..State::default()
    };
}

fn seek(state: &mut State, target: f64) {
    let target = if state.duration > 0.0 {
        target.clamp(0.0, state.duration)
    } else {
        target.max(0.0)
    };
    let Some(session) = &state.session else {
        state.pending_seek = Some(target);
        return;
    };
    if state.finished {
        state.finished = false;
        state.paused = false;
        session.set_paused(false);
    }
    session.seek(target);
    state.seek_target = Some(target);
    state.position = target;
}

fn tick(state: &mut State) {
    let (Some(session), Some(shared)) = (state.session.clone(), state.shared.clone()) else {
        return;
    };
    if state.seek_target.is_some() && !shared.is_seeking() {
        state.seek_target = None;
    }
    let clock = shared.clock.now();
    state.position = state.seek_target.unwrap_or(clock);

    if session.take_audio_failure() {
        state
            .notices
            .push("The sound device stopped. Playing on without sound.".to_string());
    }
    if shared.video_failed() && !state.video_failure_told {
        state.video_failure_told = true;
        state
            .notices
            .push("ffmpeg couldn't decode the picture of this file.".to_string());
    }
    if state.has_video && !state.surface_seen {
        if shared.surface_ready.load(Acquire) {
            state.surface_seen = true;
            state.no_surface = false;
        } else if !state.no_surface
            && state
                .opened_at
                .is_some_and(|t| t.elapsed() >= NO_SURFACE_AFTER)
        {
            session.set_no_surface();
            state.no_surface = true;
        }
    }

    let past_end = state.duration > 0.0 && clock >= state.duration + END_GRACE_SECS;
    if !state.finished && !shared.is_seeking() && (shared.ended() || past_end) {
        state.finished = true;
        state.paused = true;
        session.set_paused(true);
        if state.duration > 0.0 {
            state.position = state.duration;
        }
    }
}

pub fn update(state: &mut State, msg: Msg) -> (Task<Msg>, Out) {
    let mut task = Task::none();
    match msg {
        Msg::SeekTo(time) => seek(state, time),
        Msg::Loaded(token, loaded) if token == state.token => {
            task = on_loaded(state, loaded);
        }
        Msg::Started(token, session) if token == state.token => match session {
            Some(session) => on_started(state, session),
            None => {
                state.phase = Phase::Failed("Playback couldn't start.".to_string());
            }
        },
        Msg::Tick => tick(state),
        Msg::TogglePause => {
            if state.finished {
                seek(state, 0.0);
            } else if let Some(session) = &state.session {
                state.paused = !state.paused;
                session.set_paused(state.paused);
            }
        }
        Msg::Nudge(delta) => {
            let from = state.seek_target.unwrap_or(state.position);
            seek(state, from + delta);
        }
        Msg::Volume(volume) => {
            state.volume = volume.clamp(0.0, 1.0);
            if let Some(session) = &state.session {
                session.set_volume(state.volume);
            }
        }
        Msg::SelectTrack(i) => {
            if i < state.tracks.len() {
                state.active = i;
            }
        }
        Msg::NextTrack => {
            if !state.tracks.is_empty() {
                state.active = (state.active + 1) % state.tracks.len();
            }
        }
        Msg::ReloadTracks => {
            if let Some(job) = state.job.clone() {
                let token = state.token;
                task = background(
                    move || Arc::new(load_tracks(&job.paths)),
                    move |set| Msg::TracksLoaded(token, set),
                );
            }
        }
        Msg::TracksLoaded(token, Some(set)) if token == state.token => {
            set_tracks(state, set.tracks.clone());
        }
        Msg::Retry => {
            state.token += 1;
            task = start_loading(state);
        }
        // from before the last open or close: a late session is dropped here, which ends it
        Msg::Loaded(..) | Msg::Started(..) | Msg::TracksLoaded(..) => {}
    }
    (task, Out::None)
}

/// Replace the tracks, staying on the same one if it's still there.
fn set_tracks(state: &mut State, tracks: Vec<Track>) {
    let current = state.tracks.get(state.active).map(|t| t.label.clone());
    state.active = current
        .and_then(|label| tracks.iter().position(|t| t.label == label))
        .unwrap_or(0);
    state.tracks = tracks;
}

fn on_loaded(state: &mut State, loaded: Option<Arc<Loaded>>) -> Task<Msg> {
    let (Some(loaded), Some(job)) = (loaded, state.job.clone()) else {
        state.phase = Phase::Failed("This file couldn't be opened.".to_string());
        return Task::none();
    };
    state.notices = loaded.set.problems.clone();
    set_tracks(state, loaded.set.tracks.clone());
    let ffmpeg = match phase_after_load(&loaded, &media_name(&job)) {
        Ok(ffmpeg) => ffmpeg,
        Err(phase) => {
            state.phase = phase;
            return Task::none();
        }
    };
    let probe = &loaded.probe;
    state.has_video = probe.has_video;
    state.duration = probe.duration_secs.unwrap_or(0.0);
    state.notices.extend(track_notice(probe));

    let (width, height, fps) = probe::output_format(probe);
    let config = Config {
        ffmpeg,
        input: job.media,
        has_video: probe.has_video,
        width,
        height,
        fps,
        has_audio: probe.has_audio(),
        use_audio_device: true,
        duration: state.duration,
        volume: state.volume,
    };
    let start = state.pending_seek.take().unwrap_or(0.0);
    state.position = start;
    let token = state.token;
    background(
        move || Arc::new(Session::start(config, start)),
        move |session| Msg::Started(token, session),
    )
}

fn on_started(state: &mut State, session: Arc<Session>) {
    if let Some(problem) = &session.audio_problem {
        state.notices.push(format!(
            "No sound: {}.",
            engine::lang::clean_separators(problem)
        ));
    }
    state.shared = Some(session.shared());
    if let Some(target) = state.pending_seek.take() {
        session.seek(target);
        state.seek_target = Some(target);
        state.position = target;
    }
    session.set_volume(state.volume);
    state.session = Some(session);
    state.opened_at = Some(Instant::now());
    state.phase = Phase::Playing;
}

// ------------------------------------------------------------------------- view

/// `H:MM:SS`, or `M:SS` under an hour.
fn clock_text(secs: f64) -> String {
    let t = secs.max(0.0) as u64;
    if t >= 3600 {
        format!("{}:{:02}:{:02}", t / 3600, t / 60 % 60, t % 60)
    } else {
        format!("{}:{:02}", t / 60, t % 60)
    }
}

/// Parts of the seek bar's width. Fine enough that a pixel is never lost on a screen.
const TICK_SCALE: u32 = 10_000;

/// Where to draw mark ticks: for each, the gap before it, then the gap after the last, as portions
/// of the bar. Marks on the same spot are drawn once. Empty without a duration.
fn tick_layout(marks: &[Mark], duration: f64) -> (Vec<(u16, MarkKind)>, u16) {
    if duration <= 0.0 {
        return (Vec::new(), TICK_SCALE as u16);
    }
    let mut at: Vec<(u32, MarkKind)> = marks
        .iter()
        .map(|m| {
            let f = (m.time / duration).clamp(0.0, 1.0);
            ((f * TICK_SCALE as f64).round() as u32, m.kind)
        })
        .collect();
    at.sort_by_key(|(p, _)| *p);
    at.dedup_by_key(|(p, _)| *p);
    let mut last = 0;
    let gaps = at
        .into_iter()
        .map(|(p, kind)| {
            let gap = p - last;
            last = p;
            (gap as u16, kind)
        })
        .collect();
    (gaps, (TICK_SCALE - last) as u16)
}

fn gap<'a>(portion: u16) -> Option<Element<'a, Msg>> {
    (portion > 0).then(|| space().width(Length::FillPortion(portion)).into())
}

fn seek_bar<'a>(state: &'a State, t: Tokens, marks: &'a [Mark]) -> Element<'a, Msg> {
    if state.duration <= 0.0 {
        return space().width(Length::Fill).into();
    }
    let bar = slider(
        0.0..=state.duration,
        state.position.clamp(0.0, state.duration),
        Msg::SeekTo,
    )
    .step(0.1)
    .height(SEEK_BAR_HEIGHT);

    // no widget here takes the mouse, so the slider under it still works everywhere
    let (ticks, tail) = tick_layout(marks, state.duration);
    let mut line = row![]
        .height(SEEK_BAR_HEIGHT)
        .align_y(Alignment::Center)
        .padding([0.0, SLIDER_INSET]);
    for (portion, kind) in ticks {
        line = line.push(gap(portion));
        let color = match kind {
            MarkKind::Bookmark => t.accent,
            MarkKind::Flag => t.warn,
        };
        line = line.push(
            container(space())
                .width(2)
                .height(SEEK_BAR_HEIGHT - 4.0)
                .style(theme::rounded_fill(color, 1.0)),
        );
    }
    line = line.push(gap(tail));
    stack![bar, line].width(Length::Fill).into()
}

fn controls<'a>(state: &'a State, t: Tokens, marks: &'a [Mark]) -> Element<'a, Msg> {
    let playing = !state.paused && !state.finished;
    let time = if state.duration > 0.0 {
        format!(
            "{} / {}",
            clock_text(state.position),
            clock_text(state.duration)
        )
    } else {
        clock_text(state.position)
    };
    let transport = row![
        button(label(if playing { "Pause" } else { "Play" }).size(13))
            .on_press(Msg::TogglePause)
            .style(theme::outline_btn(t)),
        label(time).size(12).color(t.muted),
        seek_bar(state, t, marks),
        label("Volume").size(12).color(t.muted),
        slider(0.0..=1.0, state.volume, Msg::Volume)
            .step(0.05f32)
            .height(SEEK_BAR_HEIGHT)
            .width(90),
    ]
    .spacing(10)
    .align_y(Alignment::Center);

    let mut tracks = row![label("Subtitles").size(12).color(t.muted)]
        .spacing(6)
        .align_y(Alignment::Center);
    for (i, track) in state.tracks.iter().enumerate() {
        tracks = tracks.push(
            button(label(track.label.as_str()).size(12))
                .on_press(Msg::SelectTrack(i))
                .style(theme::segment(t, i == state.active)),
        );
    }
    if state.tracks.is_empty() {
        tracks = tracks.push(label("None").size(12).color(t.faint));
    }
    tracks = tracks.push(
        button(label("Refresh").size(12))
            .on_press(Msg::ReloadTracks)
            .style(theme::ghost(t)),
    );

    let mut col = column![].spacing(8);
    if state.no_surface {
        col = col.push(
            label(
                "The picture can't be drawn right now: no GPU backend, or the window is \
                 minimized. Sound and subtitles continue.",
            )
            .size(12)
            .color(t.warn),
        );
    }
    for notice in &state.notices {
        col = col.push(
            label(notice.as_str())
                .size(12)
                .color(t.warn)
                .shaping(Shaping::Advanced)
                .wrapping(Wrapping::WordOrGlyph),
        );
    }
    container(col.push(transport).push(tracks))
        .width(Length::Fill)
        .padding([10, 14])
        .style(theme::strip(t))
        .into()
}

fn black<'a>(content: Element<'a, Msg>) -> Element<'a, Msg> {
    container(content)
        .width(Length::Fill)
        .height(Length::Fill)
        .style(|_| container::Style {
            background: Some(Color::BLACK.into()),
            ..Default::default()
        })
        .into()
}

/// Black behind the shader, so the theme background can't show there whatever the shader does.
fn video_area(state: &State) -> Element<'_, Msg> {
    let picture: Element<'_, Msg> = match &state.shared {
        Some(shared) => black(
            shader(surface::VideoProgram {
                shared: shared.clone(),
            })
            .width(Length::Fill)
            .height(Length::Fill)
            .into(),
        ),
        None => black(space().into()),
    };
    let Some(cue) = state
        .tracks
        .get(state.active)
        .and_then(|t| t.cue_at(state.position))
    else {
        return picture;
    };
    let line = container(
        label(cue.overlay.as_str())
            .size(24)
            .color(Color::WHITE)
            .align_x(text::Alignment::Center)
            .shaping(Shaping::Advanced)
            .wrapping(Wrapping::WordOrGlyph),
    )
    .padding([4, 12])
    .style(theme::rounded_fill(
        Color::from_rgba(0.0, 0.0, 0.0, 0.65),
        6.0,
    ));
    stack![
        picture,
        column![
            space().height(Length::Fill),
            container(line)
                .width(Length::Fill)
                .align_x(Alignment::Center)
                .padding([12, 24]),
        ]
        .width(Length::Fill)
        .height(Length::Fill),
    ]
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}

/// Audio files: the current line among its neighbours, each one a place to jump to.
fn transcript_area<'a>(state: &'a State, t: Tokens) -> Element<'a, Msg> {
    let Some(track) = state.tracks.get(state.active) else {
        return message_area(t, "This file has sound only, and no subtitles yet.", None);
    };
    let current = track.cue_index_at(state.position);
    let around = current.unwrap_or_else(|| {
        track
            .cues
            .partition_point(|c| c.start <= state.position)
            .min(track.cues.len().saturating_sub(1))
    });
    let from = around.saturating_sub(TRANSCRIPT_CONTEXT);
    let to = (around + TRANSCRIPT_CONTEXT + 1).min(track.cues.len());
    let mut list = column![].spacing(4).width(Length::Fill).padding([0, 12]);
    for (i, cue) in track.cues[from..to].iter().enumerate() {
        let now = current == Some(from + i);
        let line = label(cue.text.as_str())
            .size(if now { 22 } else { 16 })
            .color(if now { t.text } else { t.muted })
            .shaping(Shaping::Advanced)
            .wrapping(Wrapping::WordOrGlyph);
        // default-aligned text in an aligned container: aligning RTL text itself can make it vanish
        list = list.push(
            button(
                container(line)
                    .width(Length::Fill)
                    .align_x(content_align(&cue.text)),
            )
            .width(Length::Fill)
            .on_press(Msg::SeekTo(cue.start))
            .style(theme::ghost(t)),
        );
    }
    container(scrollable(list))
        .width(Length::Fill)
        .height(Length::Fill)
        .align_y(Alignment::Center)
        .into()
}

fn message_area<'a>(t: Tokens, message: &'a str, retry: Option<&'a str>) -> Element<'a, Msg> {
    let mut col = column![
        label(message)
            .size(14)
            .color(t.muted)
            .shaping(Shaping::Advanced)
            .wrapping(Wrapping::WordOrGlyph)
    ]
    .spacing(12)
    .align_x(Alignment::Center);
    if let Some(retry) = retry {
        col = col.push(
            button(label(retry).size(13))
                .on_press(Msg::Retry)
                .style(theme::outline_btn(t)),
        );
    }
    container(col)
        .width(Length::Fill)
        .height(Length::Fill)
        .padding(24)
        .center(Length::Fill)
        .into()
}

/// A session is running, so [`position`] means something.
pub fn is_playing(state: &State) -> bool {
    state.phase == Phase::Playing
}

/// `marks` are drawn as ticks on the seek bar. They are owned by `marks_ui`.
pub fn view<'a>(state: &'a State, tokens: Tokens, marks: &'a [Mark]) -> Element<'a, Msg> {
    match &state.phase {
        Phase::Idle | Phase::Opening => black(space().into()),
        Phase::NoFfmpeg => message_area(
            tokens,
            "Playback needs ffmpeg, which isn't installed yet.",
            Some("Check again"),
        ),
        Phase::Failed(reason) => message_area(tokens, reason, Some("Try again")),
        Phase::Playing => {
            let main = if state.has_video {
                video_area(state)
            } else {
                transcript_area(state, tokens)
            };
            column![main, controls(state, tokens, marks)]
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        }
    }
}

/// True while the page should show `main.rs`'s `ffmpeg_offer`, which can install it.
pub fn needs_ffmpeg(state: &State) -> bool {
    state.phase == Phase::NoFfmpeg
}

// ------------------------------------------------------------------------- subscription and keys

/// Whether the 10 Hz tick for slider, subtitle and end detection has anything to do. It stops while
/// paused: only a seek on its way, or a surface that hasn't shown up yet, can change anything.
fn wants_tick(state: &State) -> bool {
    state.phase == Phase::Playing
        && !state.finished
        && (!state.paused
            || state.seek_target.is_some()
            || (state.has_video && !state.surface_seen))
}

pub fn subscription(state: &State) -> Subscription<Msg> {
    if wants_tick(state) {
        Subscription::run(tick_stream).map(|()| Msg::Tick)
    } else {
        Subscription::none()
    }
}

fn tick_stream() -> impl Stream<Item = ()> {
    iced::stream::channel(2, async |mut out: mpsc::Sender<()>| {
        let (tx, mut rx) = mpsc::unbounded::<()>();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_millis(100));
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

/// A key nothing else used, while this page is open.
pub fn key(state: &State, key: &Key, mods: Modifiers) -> Option<Msg> {
    if state.phase != Phase::Playing || mods.control() || mods.alt() || mods.logo() {
        return None;
    }
    match key.as_ref() {
        Key::Named(Named::Space) => Some(Msg::TogglePause),
        Key::Named(Named::ArrowLeft) => Some(Msg::Nudge(-NUDGE_SECS)),
        Key::Named(Named::ArrowRight) => Some(Msg::Nudge(NUDGE_SECS)),
        Key::Named(Named::ArrowUp) => Some(Msg::Volume(state.volume + VOLUME_STEP)),
        Key::Named(Named::ArrowDown) => Some(Msg::Volume(state.volume - VOLUME_STEP)),
        // `main.rs` turns keys into their Latin letter, so this works on any layout
        Key::Character(c) if c.eq_ignore_ascii_case("s") && state.tracks.len() > 1 => {
            Some(Msg::NextTrack)
        }
        _ => None,
    }
}

pub fn position(state: &State) -> Position {
    let time = match (&state.seek_target, &state.shared) {
        (Some(target), _) => *target,
        (None, Some(shared)) if !state.finished => shared.clock.now(),
        _ => state.position,
    };
    Position {
        time,
        cue: state
            .tracks
            .get(state.active)
            .and_then(|t| t.cue_at(time))
            .map(|c| c.id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::out::JobRef;
    use engine::project::ProjectPaths;

    fn send(state: &mut State, msg: Msg) {
        let _ = update(state, msg);
    }

    fn mark(time: f64, kind: MarkKind) -> Mark {
        Mark {
            id: 1,
            kind,
            time,
            note: String::new(),
            cue: None,
        }
    }

    fn job() -> JobRef {
        JobRef {
            id: 1,
            media: PathBuf::from("Z:/nowhere/talk.mp4"),
            paths: ProjectPaths::new("Z:/nowhere", "talk"),
        }
    }

    fn playing_state() -> State {
        State {
            phase: Phase::Playing,
            has_video: true,
            surface_seen: true,
            tracks: vec![
                Track {
                    label: "Original".to_string(),
                    cues: vec![],
                },
                Track {
                    label: "Persian".to_string(),
                    cues: vec![],
                },
            ],
            ..State::default()
        }
    }

    #[test]
    fn clock_text_formats() {
        assert_eq!(clock_text(0.0), "0:00");
        assert_eq!(clock_text(-3.0), "0:00");
        assert_eq!(clock_text(65.9), "1:05");
        assert_eq!(clock_text(4722.73), "1:18:42");
    }

    #[test]
    fn ticks_land_in_proportion_and_add_up() {
        let marks = [
            mark(75.0, MarkKind::Flag),
            mark(25.0, MarkKind::Bookmark),
            mark(25.0, MarkKind::Flag),
        ];
        let (gaps, tail) = tick_layout(&marks, 100.0);
        assert_eq!(gaps.len(), 2, "same spot drawn once");
        assert_eq!(gaps[0].0, 2500);
        assert_eq!(gaps[1], (5000, MarkKind::Flag));
        assert_eq!(tail, 2500);
    }

    #[test]
    fn ticks_outside_the_media_are_clamped_and_no_duration_means_none() {
        let (gaps, tail) = tick_layout(&[mark(500.0, MarkKind::Flag)], 100.0);
        assert_eq!(gaps[0].0 as u32, TICK_SCALE);
        assert_eq!(tail, 0);
        assert!(gap(tail).is_none(), "a zero portion would not stay zero");
        let (gaps, _) = tick_layout(&[mark(1.0, MarkKind::Flag)], 0.0);
        assert!(gaps.is_empty());
    }

    #[test]
    fn keys_map_to_messages_only_while_playing_and_unmodified() {
        let s = playing_state();
        let none = Modifiers::empty();
        let k = |n| Key::Named(n);
        assert!(matches!(
            key(&s, &k(Named::Space), none),
            Some(Msg::TogglePause)
        ));
        assert!(
            matches!(key(&s, &k(Named::ArrowLeft), none), Some(Msg::Nudge(d)) if d == -NUDGE_SECS)
        );
        assert!(
            matches!(key(&s, &k(Named::ArrowRight), none), Some(Msg::Nudge(d)) if d == NUDGE_SECS)
        );
        assert!(matches!(
            key(&s, &k(Named::ArrowUp), none),
            Some(Msg::Volume(_))
        ));
        for c in ["s", "S"] {
            assert!(
                matches!(
                    key(&s, &Key::Character(c.into()), none),
                    Some(Msg::NextTrack)
                ),
                "{c}"
            );
        }
        // M and F belong to marks_ui
        assert!(key(&s, &Key::Character("m".into()), none).is_none());
        assert!(key(&s, &Key::Character("f".into()), none).is_none());
        assert!(key(&s, &k(Named::Space), Modifiers::CTRL).is_none());
        assert!(key(&State::default(), &k(Named::Space), none).is_none());
    }

    #[test]
    fn s_is_left_alone_with_one_track() {
        let mut s = playing_state();
        s.tracks.truncate(1);
        assert!(key(&s, &Key::Character("s".into()), Modifiers::empty()).is_none());
    }

    #[test]
    fn next_track_wraps_and_select_is_bounded() {
        let mut s = playing_state();
        send(&mut s, Msg::NextTrack);
        assert_eq!(s.active, 1);
        send(&mut s, Msg::NextTrack);
        assert_eq!(s.active, 0);
        send(&mut s, Msg::SelectTrack(1));
        assert_eq!(s.active, 1);
        send(&mut s, Msg::SelectTrack(9));
        assert_eq!(s.active, 1);
    }

    #[test]
    fn reloading_tracks_stays_on_the_same_language() {
        let mut s = playing_state();
        s.active = 1;
        let new = |label: &str| Track {
            label: label.to_string(),
            cues: vec![],
        };
        set_tracks(&mut s, vec![new("Original"), new("German"), new("Persian")]);
        assert_eq!(s.active, 2);
        set_tracks(&mut s, vec![new("Original")]);
        assert_eq!(s.active, 0);
    }

    #[test]
    fn volume_is_clamped_and_kept_across_close() {
        let mut s = playing_state();
        send(&mut s, Msg::Volume(1.7));
        assert_eq!(s.volume, 1.0);
        send(&mut s, Msg::Volume(-1.0));
        assert_eq!(s.volume, 0.0);
        send(&mut s, Msg::Volume(0.4));
        close(&mut s);
        assert_eq!(s.volume, 0.4);
        assert_eq!(s.phase, Phase::Idle);
    }

    #[test]
    fn a_seek_before_the_session_exists_waits_for_it() {
        let mut s = State::default();
        send(&mut s, Msg::SeekTo(42.0));
        assert_eq!(s.pending_seek, Some(42.0));
        assert!(s.seek_target.is_none());
    }

    #[test]
    fn late_results_from_an_old_open_are_dropped() {
        let mut s = State::default();
        let (_task, _out) = open(&mut s, &job());
        let old = s.token;
        close(&mut s);
        assert_ne!(old, s.token);
        send(&mut s, Msg::Loaded(old, None));
        send(&mut s, Msg::Started(old, None));
        assert_eq!(s.phase, Phase::Idle);
    }

    #[test]
    fn the_tick_stops_while_paused_unless_something_is_coming() {
        let mut s = playing_state();
        assert!(wants_tick(&s));
        s.paused = true;
        assert!(!wants_tick(&s));
        s.seek_target = Some(3.0);
        assert!(wants_tick(&s));
        s.seek_target = None;
        s.surface_seen = false;
        assert!(wants_tick(&s));
        s.surface_seen = true;
        s.paused = false;
        s.finished = true;
        assert!(!wants_tick(&s));
        assert!(!wants_tick(&State::default()));
    }

    fn loaded(ffmpeg: bool, found: bool, probe: Probe) -> Loaded {
        Loaded {
            ffmpeg: ffmpeg.then(|| PathBuf::from("ffmpeg")),
            media_found: found,
            probe,
            set: TrackSet::default(),
        }
    }

    #[test]
    fn opening_tells_each_failure_apart() {
        let video = Probe {
            has_video: true,
            ..Probe::default()
        };
        assert!(phase_after_load(&loaded(true, true, video.clone()), "x").is_ok());
        assert_eq!(
            phase_after_load(&loaded(false, true, video.clone()), "x"),
            Err(Phase::NoFfmpeg)
        );
        assert!(matches!(
            phase_after_load(&loaded(true, false, video), "x"),
            Err(Phase::Failed(m)) if m.contains("missing")
        ));
        assert!(matches!(
            phase_after_load(&loaded(true, true, Probe::default()), "x"),
            Err(Phase::Failed(m)) if m.contains("no video or sound")
        ));
    }

    #[test]
    fn many_audio_tracks_are_mentioned_and_their_titles_cleaned() {
        let one = Probe {
            audio_tracks: vec![probe::AudioTrack::default()],
            ..Probe::default()
        };
        assert_eq!(track_notice(&one), None);
        let three = Probe {
            audio_tracks: vec![
                probe::AudioTrack {
                    title: Some("Mix\u{2029}Main".to_string()),
                },
                probe::AudioTrack::default(),
                probe::AudioTrack::default(),
            ],
            ..Probe::default()
        };
        assert_eq!(
            track_notice(&three).unwrap(),
            "Playing audio track 1 of 3 (Mix\nMain)."
        );
    }

    #[test]
    fn the_page_builds_in_every_phase() {
        // view must not panic before a session exists; drawing needs a window
        let t = crate::theme::DARK;
        let marks = [mark(1.0, MarkKind::Bookmark)];
        for phase in [
            Phase::Idle,
            Phase::Opening,
            Phase::NoFfmpeg,
            Phase::Failed("reason".to_string()),
        ] {
            let s = State {
                phase,
                ..State::default()
            };
            drop(view(&s, t, &marks));
        }
        let mut audio = playing_state();
        audio.has_video = false;
        audio.duration = 10.0;
        drop(view(&audio, t, &marks));
        let video = playing_state();
        drop(view(&video, t, &marks));
    }
}
