//! Right panel of the player page: the subtitle timeline, with bookmarks and flags in time order.
//!
//! A click on a line jumps there, the flag on a line marks it, and the strip between two lines adds
//! a bookmark in the gap. Every change saves at once, on the background thread (`saver`), so a slow
//! disk can't freeze the window and a crash loses at most a note being typed.
//!
//! A long recording has thousands of lines and this page redraws ten times a second, so only the rows
//! near the visible part are built. Every row's height is fixed up front for that.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use engine::diff::DiffKind;
use engine::lang::clean_separators;
use engine::marks::{Mark, MarkKind, MarkStore};
use iced::futures::channel::oneshot;
use iced::keyboard::{Key, Modifiers};
use iced::widget::scrollable::{AbsoluteOffset, Direction, Scrollbar};
use iced::widget::text::{Shaping, Wrapping};
use iced::widget::{
    Id, button, center, column, container, rich_text, row, scrollable, space, span, text,
    text_input,
};
use iced::{Alignment, Element, Length, Task};

use crate::fade;
use crate::icons;
use crate::out::{JobRef, Out, Position, content_align};
use crate::player::{AiDiff, SubCue};
use crate::saver;
use crate::theme::{self, Tokens};

const PANEL_WIDTH: f32 = 340.0;
const NOTE_INPUT: &str = "marks-note";
const TIMELINE: &str = "marks-timeline";

const LINE_HEIGHT: f32 = 18.0;
const CUE_PADDING: f32 = 14.0;
/// Tall enough to aim at, short enough that the lines still read as one list.
const GAP_HEIGHT: f32 = 12.0;
const MARK_HEIGHT: f32 = 38.0;
const NOTE_LINE_CHARS: usize = 36;
const EDITING_HEIGHT: f32 = 104.0;
const REMOVING_HEIGHT: f32 = 34.0;
/// Rows built beyond the visible part, so a scroll doesn't show empty space first.
const OVERSCAN: f32 = 500.0;
/// Before the panel has reported its size.
const ASSUMED_VIEW_HEIGHT: f32 = 600.0;
/// Where the current line sits in the panel when it follows playback, from the top.
const FOLLOW_AT: f32 = 0.3;
/// A scroll by less than this is rounding, not the user.
const SCROLL_SLACK: f32 = 2.0;
/// The panel glides to the next line over this.
const GLIDE: Duration = Duration::from_millis(420);
/// How far a glide travels at most, in panel heights. A far jump starts this close to its target, so
/// the list doesn't fly past.
const GLIDE_MAX_VIEWS: f32 = 1.0;
const GLOW: Duration = Duration::from_millis(220);
/// Reports of somewhere else tolerated after our own scroll, before it counts as the user's.
const STALE_REPORTS: u8 = 2;

/// The panel on its way to a line.
#[derive(Debug, Clone, Copy)]
struct Glide {
    from: f32,
    to: f32,
    since: Instant,
}

#[derive(Debug, Default)]
pub struct State {
    job: Option<u64>,
    /// `None` until a job is open, and when its marks file could not be read. Marks can't be added
    /// then, so the file is never overwritten.
    store: Option<MarkStore>,
    /// Why the marks could not be read. Shown at the top of the panel.
    problem: Option<String>,
    /// Why the last save failed, from the save thread.
    save_problem: Arc<Mutex<Option<String>>>,
    /// Ends when the save just started has finished, to show its result.
    saving: Option<oneshot::Receiver<()>>,
    /// The mark whose note is open, with the text typed so far.
    editing: Option<(u64, String)>,
    /// The mark waiting for "Remove?" to be answered.
    removing: Option<u64>,
    /// The panel's scroll position.
    offset: f32,
    /// Zero until the panel has reported its size.
    view_height: f32,
    /// The user scrolled away, so the panel stays there until they ask for it to follow again.
    not_following: bool,
    /// The line the panel last scrolled to.
    followed: Option<usize>,
    /// A scroll of ours that hasn't been reported back yet.
    expected: Option<f32>,
    /// Reports of somewhere else since `expected` was set.
    stale_reports: u8,
    glide: Option<Glide>,
    /// The highlighted line, and when it lit up.
    lit: Option<(usize, Instant)>,
    /// The line whose highlight is fading out.
    unlit: Option<(usize, Instant)>,
    /// The lines also show what the AI cleanup changed.
    show_changes: bool,
}

#[derive(Debug, Clone)]
pub enum Msg {
    Add(MarkKind, Position),
    /// A bookmark at this moment, from the strip between two lines.
    AddAt(f64),
    /// Flag this line, or take the flag off.
    ToggleFlag {
        cue: u64,
        time: f64,
    },
    Jump(f64),
    EditNote(u64),
    NoteChanged(String),
    SaveNote,
    CancelNote,
    AskRemove(u64),
    CancelRemove,
    Remove(u64),
    /// A save finished: time to show how it went.
    Saved,
    /// Offset and height of the panel.
    Scrolled(f32, f32),
    FollowAgain,
    /// Show or hide what the AI cleanup changed.
    ToggleChanges,
}

/// The player page is opening on `job`: load its marks. A note still open is saved first.
pub fn open(state: &mut State, job: &JobRef) -> (Task<Msg>, Out) {
    flush(state);
    // a save still on its way has to land before the file is read back
    saver::wait();
    state.job = Some(job.id);
    state.removing = None;
    state.problem = None;
    state.offset = 0.0;
    state.not_following = false;
    state.followed = None;
    state.expected = None;
    *lock_problem(state) = None;
    let path = job.paths.marks();
    state.store = match MarkStore::load(&path) {
        Ok(store) => Some(store),
        Err(e) => {
            // the error can quote the file's content and its path, so it counts as untrusted
            state.problem = Some(clean_separators(&format!(
                "These marks could not be read, so the file was left as it is and new marks can't be added. {e}"
            )));
            None
        }
    };
    (Task::none(), Out::None)
}

pub fn update(state: &mut State, msg: Msg) -> (Task<Msg>, Out) {
    let (task, out) = handle(state, msg);
    let Some(saving) = state.saving.take() else {
        return (task, out);
    };
    let saved = Task::perform(
        async move {
            let _ = saving.await;
        },
        |()| Msg::Saved,
    );
    (Task::batch([task, saved]), out)
}

fn handle(state: &mut State, msg: Msg) -> (Task<Msg>, Out) {
    match msg {
        Msg::Add(kind, at) => add(state, kind, at),
        Msg::AddAt(time) => add(state, MarkKind::Bookmark, Position { time, cue: None }),
        Msg::ToggleFlag { cue, time } => {
            let flag = find_flag(marks(state), cue);
            match flag {
                Some(id) => ask_remove(state, id),
                None => add(
                    state,
                    MarkKind::Flag,
                    Position {
                        time,
                        cue: Some(cue),
                    },
                ),
            }
        }
        Msg::Jump(time) => {
            state.not_following = false;
            state.followed = None;
            if let Some(job) = state.job {
                return (Task::none(), Out::PlayAt { job, time });
            }
        }
        Msg::EditNote(id) => {
            flush(state);
            let note = state
                .store
                .as_ref()
                .and_then(|store| store.marks.iter().find(|m| m.id == id))
                .map(|mark| one_line(&mark.note));
            if let Some(note) = note {
                state.editing = Some((id, note));
                return (
                    iced::widget::operation::focus(Id::new(NOTE_INPUT)),
                    Out::None,
                );
            }
        }
        Msg::NoteChanged(typed) => {
            if let Some((_, draft)) = &mut state.editing {
                *draft = one_line(&typed);
            }
        }
        Msg::SaveNote => flush(state),
        Msg::CancelNote => state.editing = None,
        Msg::AskRemove(id) => ask_remove(state, id),
        Msg::CancelRemove => state.removing = None,
        Msg::Remove(id) => remove(state, id),
        // the redraw after this message shows the result
        Msg::Saved => {}
        Msg::Scrolled(offset, height) => scrolled(state, offset, height),
        Msg::FollowAgain => {
            state.not_following = false;
            state.followed = None;
        }
        Msg::ToggleChanges => {
            state.show_changes = !state.show_changes;
            // row heights changed, so the line followed has moved
            state.followed = None;
            state.glide = None;
        }
    }
    (Task::none(), Out::None)
}

fn scrolled(state: &mut State, offset: f32, height: f32) {
    state.view_height = height;
    // every report during a glide is one of its own steps, a frame behind
    if state.glide.is_some() {
        return;
    }
    if let Some(expected) = state.expected {
        if (offset - expected).abs() <= SCROLL_SLACK {
            state.expected = None;
            state.stale_reports = 0;
            state.offset = offset;
            return;
        }
        // our scroll_to may report earlier places before the new one
        if state.stale_reports < STALE_REPORTS {
            state.stale_reports += 1;
            return;
        }
        state.expected = None;
    }
    state.stale_reports = 0;
    if (offset - state.offset).abs() > SCROLL_SLACK {
        state.not_following = true;
    }
    state.offset = offset;
}

fn find_flag(marks: &[Mark], cue: u64) -> Option<u64> {
    marks
        .iter()
        .find(|m| m.kind == MarkKind::Flag && m.cue == Some(cue))
        .map(|m| m.id)
}

fn add(state: &mut State, kind: MarkKind, at: Position) {
    let Some(store) = &mut state.store else {
        return;
    };
    // F twice on one line is one flag, however long the line stays on screen.
    let flagged =
        kind == MarkKind::Flag && at.cue.is_some_and(|c| find_flag(&store.marks, c).is_some());
    if !flagged {
        let cue = if kind == MarkKind::Flag { at.cue } else { None };
        store.add(kind, at.time, String::new(), cue);
        save(state);
    }
}

fn ask_remove(state: &mut State, id: u64) {
    let has_note = marks(state)
        .iter()
        .find(|m| m.id == id)
        .is_some_and(|mark| !mark.note.trim().is_empty());
    if has_note {
        state.removing = Some(id);
    } else {
        remove(state, id);
    }
}

fn remove(state: &mut State, id: u64) {
    state.removing = None;
    if state.editing.as_ref().is_some_and(|(open, _)| *open == id) {
        state.editing = None;
    }
    if state.store.as_mut().is_some_and(|store| store.remove(id)) {
        save(state);
    }
}

fn lock_problem(state: &State) -> std::sync::MutexGuard<'_, Option<String>> {
    state.save_problem.lock().unwrap_or_else(|e| e.into_inner())
}

/// Starts the save and returns at once. The file is replaced whole, so the last save wins.
fn save(state: &mut State) {
    let Some(store) = state.store.clone() else {
        return;
    };
    let problem = state.save_problem.clone();
    let (done, saving) = oneshot::channel();
    state.saving = Some(saving);
    saver::run(move || {
        let result = store
            .save()
            .err()
            .map(|e| clean_separators(&format!("Marks could not be saved: {e}")));
        *problem.lock().unwrap_or_else(|e| e.into_inner()) = result;
        let _ = done.send(());
    });
}

/// Keep a note still being typed. Integration calls this when the window closes, then
/// `saver::wait`.
pub fn flush(state: &mut State) {
    let Some((id, draft)) = state.editing.take() else {
        return;
    };
    let note = draft.trim().to_string();
    if state
        .store
        .as_mut()
        .is_some_and(|store| store.set_note(id, note))
    {
        save(state);
    }
}

/// Notes come from a file, so they may hold separators that crash the text layout. The note box is
/// one line, so line breaks become spaces there.
fn one_line(text: &str) -> String {
    clean_separators(text).replace(['\r', '\n'], " ")
}

/// `m:ss`, or `h:mm:ss` from an hour on.
fn clock(time: f64) -> String {
    let total = time.max(0.0) as u64;
    let (hours, minutes, seconds) = (total / 3600, total / 60 % 60, total % 60);
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

// ------------------------------------------------------------------------- timeline layout

/// One row of the timeline.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Item {
    /// The strip after this line, or before the first one.
    Gap(Option<usize>),
    Cue(usize),
    Mark(usize),
}

/// The moment the strip after line `prev` stands for: the middle of the silence between the lines,
/// or where the next one starts if they touch.
fn gap_time(cues: &[SubCue], prev: Option<usize>) -> f64 {
    let Some(prev) = prev else {
        return cues.first().map_or(0.0, |c| c.start / 2.0);
    };
    let end = cues[prev].end;
    match cues.get(prev + 1) {
        Some(next) if next.start > end => (end + next.start) / 2.0,
        Some(next) => next.start,
        None => end,
    }
}

fn cue_height(state: &State, cue: &SubCue) -> f32 {
    let diff_lines = match &cue.ai_diff {
        Some(diff) if state.show_changes => diff.panel_lines,
        _ => 0,
    };
    f32::from(cue.panel_lines + diff_lines) * LINE_HEIGHT + CUE_PADDING
}

fn mark_height(state: &State, mark: &Mark) -> f32 {
    let mut height = MARK_HEIGHT;
    if state.removing == Some(mark.id) {
        height += REMOVING_HEIGHT;
    }
    if state.editing.as_ref().is_some_and(|(id, _)| *id == mark.id) {
        height += EDITING_HEIGHT;
    } else if !mark.note.is_empty() {
        let lines = mark.note.chars().count().div_ceil(NOTE_LINE_CHARS);
        height += lines as f32 * LINE_HEIGHT + 6.0;
    }
    height
}

/// Every row in time order, with its height. A mark comes after the line it falls under, so a
/// bookmark made in the gap after a line sits above that gap's strip.
fn layout(state: &State, cues: &[SubCue], marks: &[Mark]) -> Vec<(Item, f32)> {
    let mut items = Vec::with_capacity(cues.len() * 2 + marks.len() + 1);
    let mut next_mark = 0;
    let push_marks = |items: &mut Vec<(Item, f32)>, next_mark: &mut usize, before: f64| {
        while let Some(mark) = marks.get(*next_mark).filter(|m| m.time < before) {
            items.push((Item::Mark(*next_mark), mark_height(state, mark)));
            *next_mark += 1;
        }
    };
    if let Some(first) = cues.first() {
        push_marks(&mut items, &mut next_mark, first.start);
        items.push((Item::Gap(None), GAP_HEIGHT));
    }
    for (i, cue) in cues.iter().enumerate() {
        items.push((Item::Cue(i), cue_height(state, cue)));
        let before = cues.get(i + 1).map_or(f64::INFINITY, |next| next.start);
        push_marks(&mut items, &mut next_mark, before);
        items.push((Item::Gap(Some(i)), GAP_HEIGHT));
    }
    push_marks(&mut items, &mut next_mark, f64::INFINITY);
    items
}

/// Which rows to build for this view: their range, and the height of those left out above and below.
fn window(items: &[(Item, f32)], top: f32, height: f32) -> (std::ops::Range<usize>, f32, f32) {
    let (from, to) = (top - OVERSCAN, top + height + OVERSCAN);
    let (mut y, mut first, mut last, mut above) = (0.0, items.len(), items.len(), 0.0);
    for (i, (_, h)) in items.iter().enumerate() {
        if first == items.len() && y + h > from {
            first = i;
            above = y;
        }
        if y >= to {
            last = i;
            break;
        }
        y += h;
    }
    let below: f32 = items[last..].iter().map(|(_, h)| h).sum();
    (first.min(last)..last, above, below)
}

fn view_height(state: &State) -> f32 {
    if state.view_height > 0.0 {
        state.view_height
    } else {
        ASSUMED_VIEW_HEIGHT
    }
}

/// Keep the line on screen lit and in view, unless the user scrolled away. Call it when playback
/// moves. The scrolling itself happens in [`frame`].
pub fn follow(state: &mut State, cues: &[SubCue], current: Option<usize>, position: f64) {
    if state.lit.map(|(i, _)| i) != current {
        let now = Instant::now();
        state.unlit = state.lit.map(|(i, _)| (i, now));
        state.lit = current.map(|i| (i, now));
    }
    if state.not_following || cues.is_empty() {
        return;
    }
    // between lines, and before the first one, the nearest line that has started
    let target = current.unwrap_or_else(|| {
        cues.partition_point(|c| c.start <= position)
            .saturating_sub(1)
    });
    if state.followed == Some(target) {
        return;
    }
    state.followed = Some(target);
    let items = layout(state, cues, marks(state));
    let total: f32 = items.iter().map(|(_, h)| h).sum();
    let top: f32 = items
        .iter()
        .take_while(|(item, _)| *item != Item::Cue(target))
        .map(|(_, h)| h)
        .sum();
    let height = view_height(state);
    let to = (top - height * FOLLOW_AT).clamp(0.0, (total - height).max(0.0));
    if (to - state.offset).abs() <= SCROLL_SLACK {
        return;
    }
    let reach = height * GLIDE_MAX_VIEWS;
    state.glide = Some(Glide {
        from: state.offset.clamp(to - reach, to + reach),
        to,
        since: Instant::now(),
    });
}

/// Something in the panel is mid-motion, so it needs [`frame`] every frame.
pub fn is_animating(state: &State) -> bool {
    let fading =
        |light: Option<(usize, Instant)>| light.is_some_and(|(_, at)| fade::running(at, GLOW));
    state.glide.is_some() || fading(state.lit) || fading(state.unlit)
}

/// The next step of the glide, for the moment `at`.
pub fn frame(state: &mut State, at: Instant) -> Task<Msg> {
    let Some(glide) = state.glide else {
        return Task::none();
    };
    let t = fade::eased_at(glide.since, at, GLIDE);
    let y = glide.from + (glide.to - glide.from) * t;
    state.offset = y;
    if t >= 1.0 {
        state.glide = None;
        state.expected = Some(y);
        state.stale_reports = 0;
    }
    iced::widget::operation::scroll_to(
        Id::new(TIMELINE),
        AbsoluteOffset {
            x: None,
            y: Some(y),
        },
    )
}

/// How lit line `index` is, 0 to 1.
fn glow(state: &State, index: usize) -> f32 {
    match (state.lit, state.unlit) {
        (Some((i, at)), _) if i == index => fade::eased(at, GLOW),
        (_, Some((i, at))) if i == index => 1.0 - fade::eased(at, GLOW),
        _ => 0.0,
    }
}

// ------------------------------------------------------------------------- view

/// `at` is where playback is now, `None` when nothing is playing, for the add buttons. `cues` are
/// the lines of the track on screen and `current` the one showing.
pub fn view<'a>(
    state: &'a State,
    tokens: Tokens,
    at: Option<Position>,
    cues: &'a [SubCue],
    current: Option<usize>,
) -> Element<'a, Msg> {
    let t = tokens;
    let add = |icon: &'static str, label: &'static str, kind: MarkKind| {
        button(
            row![icons::icon(icon, 14.0, t.muted), text(label).size(12)]
                .spacing(4)
                .align_y(Alignment::Center),
        )
        .padding([4, 8])
        .style(theme::flat(t, false))
        .on_press_maybe(
            at.filter(|_| state.store.is_some())
                .map(|at| Msg::Add(kind, at)),
        )
    };
    let mut title = row![
        text("Timeline").size(14).color(t.text).width(Length::Fill),
        add(icons::BOOKMARK, "Bookmark", MarkKind::Bookmark),
        add(icons::FLAG, "Flag line", MarkKind::Flag),
    ]
    .spacing(4)
    .align_y(Alignment::Center);
    if state.not_following && current.is_some() {
        title = title.push(
            button(
                row![
                    icons::icon(icons::FOLLOW, 14.0, t.text),
                    text("Follow").size(12)
                ]
                .spacing(4)
                .align_y(Alignment::Center),
            )
            .padding([4, 8])
            .style(theme::flat(t, true))
            .on_press(Msg::FollowAgain),
        );
    }
    let mut head = column![title].spacing(4);
    for problem in [state.problem.clone(), lock_problem(state).clone()]
        .into_iter()
        .flatten()
    {
        head = head.push(text(problem).size(12).color(t.bad));
    }
    head = head.push(
        text("Click a line to jump. The flag marks a line, + marks the gap between two lines.")
            .size(12)
            .color(t.muted),
    );
    if cues.iter().any(|c| c.ai_edited) {
        let label = if state.show_changes {
            "Hide AI changes"
        } else {
            "Show AI changes"
        };
        let mut changes = row![
            button(text(label).size(12))
                .padding([4, 8])
                .style(theme::flat(t, state.show_changes))
                .on_press(Msg::ToggleChanges)
        ]
        .spacing(8)
        .align_y(Alignment::Center);
        if state.show_changes {
            changes = changes.push(
                rich_text(vec![
                    span("removed").color(t.bad).strikethrough(true),
                    span("  "),
                    span("added").color(t.ok),
                ])
                .on_link_click(iced::never)
                .size(12),
            );
        }
        head = head.push(changes);
    }

    let marks = marks(state);
    let items = layout(state, cues, marks);
    let body: Element<'a, Msg> = if items.is_empty() {
        text("No subtitles or marks yet.")
            .size(13)
            .color(t.faint)
            .into()
    } else {
        let (range, above, below) = window(&items, state.offset, view_height(state));
        let mut list = column![].width(Length::Fill).padding(iced::Padding {
            right: 10.0,
            ..iced::Padding::ZERO
        });
        list = list.push(space().height(above));
        for &(item, height) in &items[range] {
            list = list.push(match item {
                Item::Gap(prev) => gap_row(state, cues, prev, t),
                Item::Cue(i) => cue_row(state, &cues[i], i, current == Some(i), height, t),
                Item::Mark(i) => mark_row(state, &marks[i], height, t),
            });
        }
        list = list.push(space().height(below));
        scrollable(list)
            .id(Id::new(TIMELINE))
            .direction(Direction::Vertical(
                Scrollbar::new().width(6).scroller_width(6),
            ))
            .style(theme::dark_scroll(t))
            .on_scroll(|v| Msg::Scrolled(v.absolute_offset().y, v.bounds().height))
            .height(Length::Fill)
            .into()
    };
    container(column![head, body].spacing(8))
        .padding(10)
        .width(Length::Fixed(PANEL_WIDTH))
        .height(Length::Fill)
        .style(theme::panel(t))
        .into()
}

fn gap_row<'a>(state: &State, cues: &[SubCue], prev: Option<usize>, t: Tokens) -> Element<'a, Msg> {
    button(center(icons::icon(icons::MARK_PLUS, 10.0, t.faint)).height(GAP_HEIGHT))
        .padding(0)
        .width(Length::Fill)
        .height(GAP_HEIGHT)
        .style(theme::flat(t, false))
        .on_press_maybe(
            state
                .store
                .is_some()
                .then(|| Msg::AddAt(gap_time(cues, prev))),
        )
        .into()
}

fn cue_row<'a>(
    state: &State,
    cue: &'a SubCue,
    index: usize,
    current: bool,
    height: f32,
    t: Tokens,
) -> Element<'a, Msg> {
    let line = cue.text.replace('\n', " ");
    let align = content_align(&line);
    let words = container(
        text(line)
            .size(13)
            .color(if current { t.text } else { t.muted })
            .shaping(Shaping::Advanced)
            .wrapping(Wrapping::WordOrGlyph),
    )
    .width(Length::Fill)
    .align_x(align);
    let mut lines = column![words];
    if state.show_changes
        && let Some(diff) = &cue.ai_diff
    {
        lines = lines.push(change_line(diff, t));
    }
    let edited = state.show_changes && cue.ai_edited;
    let jump = button(
        row![
            text(clock(cue.start))
                .size(11)
                .color(if edited { t.accent } else { t.faint })
                .width(Length::Fixed(44.0)),
            lines,
        ]
        .spacing(4),
    )
    .padding([6, 6])
    .width(Length::Fill)
    .height(Length::Fill)
    .style(theme::flat(t, false))
    .on_press(Msg::Jump(cue.start));

    let flagged = find_flag(marks(state), cue.id).is_some();
    let flag = button(icons::icon(
        if flagged {
            icons::FLAG_SOLID
        } else {
            icons::FLAG
        },
        15.0,
        if flagged { t.warn } else { t.faint },
    ))
    .padding(5)
    .style(theme::flat(t, false))
    .on_press_maybe(state.store.is_some().then_some(Msg::ToggleFlag {
        cue: cue.id,
        time: cue.start,
    }));

    let strength = glow(state, index);
    container(row![jump, flag].align_y(Alignment::Center))
        .width(Length::Fill)
        .height(height)
        .clip(true)
        .style(theme::current_line(t, strength))
        .into()
}

/// The words of an utterance with what the AI cleanup removed and added.
fn change_line<'a>(diff: &'a AiDiff, t: Tokens) -> Element<'a, Msg> {
    let plain: Vec<&str> = diff.spans.iter().map(|s| s.text.as_str()).collect();
    let mut spans = Vec::with_capacity(diff.spans.len() * 2);
    for (i, piece) in diff.spans.iter().enumerate() {
        if i > 0 {
            spans.push(span(" "));
        }
        spans.push(match piece.kind {
            DiffKind::Same => span(piece.text.as_str()).color(t.faint),
            DiffKind::Removed => span(piece.text.as_str()).color(t.bad).strikethrough(true),
            DiffKind::Added => span(piece.text.as_str()).color(t.ok),
        });
    }
    // default-aligned text in an aligned container: aligning RTL text itself can make it vanish
    container(
        rich_text(spans)
            .on_link_click(iced::never)
            .size(12)
            .wrapping(Wrapping::WordOrGlyph),
    )
    .width(Length::Fill)
    .align_x(content_align(&plain.join(" ")))
    .into()
}

fn mark_row<'a>(state: &'a State, mark: &'a Mark, height: f32, t: Tokens) -> Element<'a, Msg> {
    let (label, color) = match mark.kind {
        MarkKind::Bookmark => ("Bookmark", t.accent),
        MarkKind::Flag => ("Flag", t.warn),
    };
    let top = row![
        button(text(clock(mark.time)).size(12).color(t.text))
            .padding([2, 6])
            .style(theme::flat(t, false))
            .on_press(Msg::Jump(mark.time)),
        container(text(label).size(11).color(color))
            .padding([1, 7])
            .style(theme::chip(color)),
        space::horizontal(),
        button(text("Note").size(12))
            .padding([2, 6])
            .style(theme::flat(t, false))
            .on_press(Msg::EditNote(mark.id)),
        button(icons::icon(icons::CLOSE, 13.0, t.muted))
            .padding(5)
            .style(theme::flat(t, false))
            .on_press(Msg::AskRemove(mark.id)),
    ]
    .spacing(4)
    .align_y(Alignment::Center);

    let mut body = column![top].spacing(4);
    if state.removing == Some(mark.id) {
        body = body.push(
            row![
                text("Remove this mark and its note?")
                    .size(12)
                    .color(t.text)
                    .width(Length::Fill),
                button(text("Remove").size(12))
                    .padding([2, 8])
                    .style(theme::flat(t, true))
                    .on_press(Msg::Remove(mark.id)),
                button(text("Keep").size(12))
                    .padding([2, 8])
                    .style(theme::flat(t, false))
                    .on_press(Msg::CancelRemove),
            ]
            .spacing(4)
            .align_y(Alignment::Center),
        );
    }
    match &state.editing {
        Some((id, draft)) if *id == mark.id => {
            body = body.push(
                column![
                    text_input("Add a note", draft)
                        .id(Id::new(NOTE_INPUT))
                        .size(13)
                        .padding(6)
                        .style(theme::input(t))
                        .on_input(Msg::NoteChanged)
                        .on_submit(Msg::SaveNote),
                    text("Right-to-left notes may show the caret on the wrong side.")
                        .size(11)
                        .color(t.faint),
                    row![
                        button(text("Save").size(12))
                            .padding([2, 8])
                            .style(theme::flat(t, true))
                            .on_press(Msg::SaveNote),
                        button(text("Cancel").size(12))
                            .padding([2, 8])
                            .style(theme::flat(t, false))
                            .on_press(Msg::CancelNote),
                    ]
                    .spacing(4),
                ]
                .spacing(4),
            );
        }
        _ if !mark.note.is_empty() => {
            let note = clean_separators(&mark.note);
            let align = content_align(&note);
            body = body.push(
                container(
                    text(note)
                        .size(13)
                        .color(t.text)
                        .shaping(Shaping::Advanced)
                        .wrapping(Wrapping::WordOrGlyph),
                )
                .width(Length::Fill)
                .align_x(align),
            );
        }
        _ => {}
    }
    container(
        container(body)
            .padding(4)
            .width(Length::Fill)
            .height(Length::Fill)
            .clip(true)
            .style(theme::flat_card(t)),
    )
    .padding([1, 0])
    .height(height)
    .into()
}

/// M and F on the player page. `at` is where playback is now. `player::key` is asked first.
pub fn key(state: &State, key: &Key, mods: Modifiers, at: Position) -> Option<Msg> {
    if state.store.is_none() || mods.control() || mods.alt() || mods.logo() {
        return None;
    }
    let Key::Character(c) = key else {
        return None;
    };
    if c.eq_ignore_ascii_case("m") {
        Some(Msg::Add(MarkKind::Bookmark, at))
    } else if c.eq_ignore_ascii_case("f") {
        Some(Msg::Add(MarkKind::Flag, at))
    } else {
        None
    }
}

/// For the seek bar's ticks.
pub fn marks(state: &State) -> &[Mark] {
    state.store.as_ref().map_or(&[], |store| &store.marks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::project::ProjectPaths;
    use std::path::PathBuf;

    /// A fresh folder under the system temp dir, removed when dropped.
    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> TestDir {
            let dir = std::env::temp_dir().join(format!(
                "vipr-marks-ui-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            TestDir(dir)
        }

        fn job(&self) -> JobRef {
            JobRef {
                id: 7,
                media: self.0.join("a.mp4"),
                paths: ProjectPaths {
                    dir: self.0.clone(),
                    stem: "a".into(),
                },
            }
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn opened(dir: &TestDir) -> State {
        let mut state = State::default();
        let _ = open(&mut state, &dir.job());
        state
    }

    /// Waits for the save thread, as the tests read the file or the save's result next.
    fn send(state: &mut State, msg: Msg) -> Out {
        let out = update(state, msg).1;
        saver::wait();
        out
    }

    fn cue(id: u64, start: f64, end: f64) -> SubCue {
        SubCue {
            id,
            start,
            end,
            text: "line".to_string(),
            overlay: "line".to_string(),
            panel_lines: 1,
            ai_edited: false,
            ai_diff: None,
        }
    }

    #[test]
    fn a_shown_change_makes_its_row_taller_so_it_is_not_cut_off() {
        let mut changed = cue(1, 0.0, 1.0);
        changed.ai_edited = true;
        changed.ai_diff = Some(Arc::new(AiDiff {
            spans: vec![],
            panel_lines: 2,
        }));
        let mut state = State::default();
        let closed = cue_height(&state, &changed);
        state.show_changes = true;
        assert_eq!(cue_height(&state, &changed), closed + 2.0 * LINE_HEIGHT);
        assert_eq!(cue_height(&state, &cue(2, 1.0, 2.0)), closed);
    }

    fn at(time: f64, cue: Option<u64>) -> Position {
        Position { time, cue }
    }

    #[test]
    fn adding_saves_at_once_and_a_reopen_finds_it() {
        let dir = TestDir::new("add");
        let mut state = opened(&dir);
        send(&mut state, Msg::Add(MarkKind::Bookmark, at(9.0, Some(3))));
        send(&mut state, Msg::Add(MarkKind::Flag, at(2.0, Some(1))));

        let back = opened(&dir);
        let marks = marks(&back);
        assert_eq!(marks.len(), 2);
        assert_eq!((marks[0].kind, marks[0].cue), (MarkKind::Flag, Some(1)));
        assert_eq!((marks[1].kind, marks[1].cue), (MarkKind::Bookmark, None));
    }

    #[test]
    fn flagging_one_line_twice_is_one_flag_but_lines_without_an_id_are_separate() {
        let dir = TestDir::new("flag");
        let mut state = opened(&dir);
        send(&mut state, Msg::Add(MarkKind::Flag, at(1.0, Some(4))));
        send(&mut state, Msg::Add(MarkKind::Flag, at(1.5, Some(4))));
        assert_eq!(marks(&state).len(), 1);
        send(&mut state, Msg::Add(MarkKind::Flag, at(2.0, None)));
        send(&mut state, Msg::Add(MarkKind::Flag, at(3.0, None)));
        assert_eq!(marks(&state).len(), 3);
    }

    #[test]
    fn jump_asks_for_this_job_and_time() {
        let dir = TestDir::new("jump");
        let mut state = opened(&dir);
        let out = send(&mut state, Msg::Jump(12.5));
        assert!(matches!(out, Out::PlayAt { job: 7, time } if time == 12.5));
    }

    #[test]
    fn a_note_is_kept_on_save_and_dropped_on_cancel() {
        let dir = TestDir::new("note");
        let mut state = opened(&dir);
        send(&mut state, Msg::Add(MarkKind::Bookmark, at(1.0, None)));
        let id = marks(&state)[0].id;

        send(&mut state, Msg::EditNote(id));
        send(&mut state, Msg::NoteChanged("  سلام  ".into()));
        send(&mut state, Msg::SaveNote);
        assert_eq!(marks(&opened(&dir))[0].note, "سلام");

        send(&mut state, Msg::EditNote(id));
        send(&mut state, Msg::NoteChanged("other".into()));
        send(&mut state, Msg::CancelNote);
        assert_eq!(marks(&state)[0].note, "سلام");
    }

    #[test]
    fn a_note_being_typed_survives_reopening_the_page() {
        let dir = TestDir::new("flush");
        let mut state = opened(&dir);
        send(&mut state, Msg::Add(MarkKind::Bookmark, at(1.0, None)));
        let id = marks(&state)[0].id;
        send(&mut state, Msg::EditNote(id));
        send(&mut state, Msg::NoteChanged("half typed".into()));
        let _ = open(&mut state, &dir.job());
        assert_eq!(marks(&state)[0].note, "half typed");
    }

    #[test]
    fn separators_from_a_file_never_reach_the_note_box() {
        let dir = TestDir::new("separators");
        std::fs::write(
            dir.job().paths.marks(),
            r#"{"marks":[{"id":1,"kind":"bookmark","time":1.0,"note":"abc سلام\u0085x\ny"}]}"#,
        )
        .unwrap();
        let mut state = opened(&dir);
        send(&mut state, Msg::EditNote(1));
        assert_eq!(state.editing.unwrap().1, "abc سلام x y");
    }

    #[test]
    fn only_a_mark_with_a_note_asks_before_removal() {
        let dir = TestDir::new("remove");
        let mut state = opened(&dir);
        send(&mut state, Msg::Add(MarkKind::Bookmark, at(1.0, None)));
        send(&mut state, Msg::Add(MarkKind::Bookmark, at(2.0, None)));
        let (plain, noted) = (marks(&state)[0].id, marks(&state)[1].id);
        send(&mut state, Msg::EditNote(noted));
        send(&mut state, Msg::NoteChanged("keep me".into()));
        send(&mut state, Msg::SaveNote);

        send(&mut state, Msg::AskRemove(plain));
        assert_eq!(marks(&state).len(), 1);

        send(&mut state, Msg::AskRemove(noted));
        assert_eq!(marks(&state).len(), 1);
        send(&mut state, Msg::CancelRemove);
        assert_eq!(marks(&state).len(), 1);
        send(&mut state, Msg::AskRemove(noted));
        send(&mut state, Msg::Remove(noted));
        assert!(marks(&state).is_empty());
        assert!(marks(&opened(&dir)).is_empty());
    }

    fn has_separator(text: &str) -> bool {
        text.contains(['\u{2029}', '\u{85}', '\u{1C}', '\u{1D}', '\u{1E}'])
    }

    #[test]
    fn a_marks_file_quoting_a_separator_in_its_error_is_cleaned_for_display() {
        let dir = TestDir::new("error-quote");
        let path = dir.job().paths.marks();
        std::fs::write(&path, r#"{"marks":[{"id":1,"kind":"abc سلام","time":1.0}]}"#).unwrap();
        let raw = MarkStore::load(&path).unwrap_err().to_string();
        assert!(has_separator(&raw), "the engine error quotes it: {raw:?}");

        let state = opened(&dir);
        let problem = state.problem.unwrap();
        assert!(!has_separator(&problem), "{problem:?}");
        assert!(problem.contains("سلام"), "{problem:?}");
    }

    #[test]
    fn a_failed_save_naming_a_path_with_a_separator_is_cleaned_for_display() {
        let dir = TestDir::new("save-path");
        let mut job = dir.job();
        // the stem comes from the media's file name
        job.paths.stem = "abc\u{2029}سلام".into();
        let mut state = State::default();
        let _ = open(&mut state, &job);
        assert_eq!(state.problem, None);
        // a folder at the file's path, so the save can't finish
        std::fs::create_dir(job.paths.marks()).unwrap();
        send(&mut state, Msg::Add(MarkKind::Bookmark, at(1.0, None)));
        let problem = lock_problem(&state).clone().unwrap();
        assert!(problem.contains("could not be saved"), "{problem:?}");
        assert!(!has_separator(&problem), "{problem:?}");
    }

    #[test]
    fn a_corrupt_file_is_reported_and_never_overwritten() {
        let dir = TestDir::new("corrupt");
        let path = dir.job().paths.marks();
        std::fs::write(&path, "{not json").unwrap();
        let mut state = opened(&dir);
        assert!(state.problem.is_some());
        assert!(
            key(
                &state,
                &Key::Character("m".into()),
                Modifiers::empty(),
                at(1.0, None)
            )
            .is_none()
        );
        send(&mut state, Msg::Add(MarkKind::Bookmark, at(1.0, None)));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{not json");
    }

    #[test]
    fn keys_are_m_and_f_in_either_case_without_command_modifiers() {
        let dir = TestDir::new("keys");
        let state = opened(&dir);
        let pos = at(4.0, Some(2));
        let press = |c: &str, mods| key(&state, &Key::Character(c.into()), mods, pos);
        assert!(matches!(
            press("m", Modifiers::empty()),
            Some(Msg::Add(MarkKind::Bookmark, p)) if p == pos
        ));
        assert!(matches!(
            press("F", Modifiers::SHIFT),
            Some(Msg::Add(MarkKind::Flag, _))
        ));
        assert!(press("m", Modifiers::CTRL).is_none());
        assert!(press("f", Modifiers::ALT).is_none());
        assert!(press("x", Modifiers::empty()).is_none());
    }

    #[test]
    fn nothing_open_means_no_marks_no_jump_and_no_keys() {
        let mut state = State::default();
        assert!(marks(&state).is_empty());
        assert!(matches!(send(&mut state, Msg::Jump(1.0)), Out::None));
        assert!(
            key(
                &state,
                &Key::Character("m".into()),
                Modifiers::empty(),
                at(0.0, None)
            )
            .is_none()
        );
    }

    #[test]
    fn a_flag_on_a_line_comes_off_again_unless_it_has_a_note() {
        let dir = TestDir::new("toggle");
        let mut state = opened(&dir);
        let toggle = Msg::ToggleFlag { cue: 4, time: 10.0 };
        send(&mut state, toggle.clone());
        assert_eq!(marks(&state)[0].cue, Some(4));
        assert_eq!(marks(&state)[0].kind, MarkKind::Flag);
        send(&mut state, toggle.clone());
        assert!(marks(&state).is_empty());

        send(&mut state, toggle.clone());
        let id = marks(&state)[0].id;
        send(&mut state, Msg::EditNote(id));
        send(&mut state, Msg::NoteChanged("keep".into()));
        send(&mut state, Msg::SaveNote);
        send(&mut state, toggle);
        assert_eq!(state.removing, Some(id), "asks first");
        assert_eq!(marks(&state).len(), 1);
    }

    #[test]
    fn a_mark_between_two_lines_is_a_bookmark_in_the_gap_and_is_saved() {
        let dir = TestDir::new("gap");
        let mut state = opened(&dir);
        let cues = [cue(1, 10.0, 12.0), cue(2, 20.0, 22.0), cue(3, 22.0, 24.0)];
        assert_eq!(gap_time(&cues, None), 5.0);
        assert_eq!(gap_time(&cues, Some(0)), 16.0, "the middle of the silence");
        assert_eq!(gap_time(&cues, Some(1)), 22.0, "lines that touch");
        assert_eq!(gap_time(&cues, Some(2)), 24.0, "after the last");
        assert_eq!(gap_time(&[], None), 0.0);

        send(&mut state, Msg::AddAt(16.0));
        let back = opened(&dir);
        let mark = &marks(&back)[0];
        assert_eq!(
            (mark.kind, mark.time, mark.cue),
            (MarkKind::Bookmark, 16.0, None)
        );
    }

    #[test]
    fn the_timeline_keeps_time_order_with_marks_after_their_line() {
        let dir = TestDir::new("order");
        let mut state = opened(&dir);
        for time in [5.0, 11.0, 16.0, 99.0] {
            send(&mut state, Msg::AddAt(time));
        }
        let cues = [cue(1, 10.0, 12.0), cue(2, 20.0, 22.0)];
        let items: Vec<Item> = layout(&state, &cues, marks(&state))
            .into_iter()
            .map(|(item, _)| item)
            .collect();
        assert_eq!(
            items,
            [
                Item::Mark(0),
                Item::Gap(None),
                Item::Cue(0),
                Item::Mark(1),
                Item::Mark(2),
                Item::Gap(Some(0)),
                Item::Cue(1),
                Item::Mark(3),
                Item::Gap(Some(1)),
            ]
        );
        assert!(layout(&State::default(), &[], &[]).is_empty());
    }

    #[test]
    fn only_rows_near_the_view_are_built_and_the_scroll_height_stays_right() {
        let items: Vec<(Item, f32)> = (0..10_000).map(|i| (Item::Cue(i), 20.0)).collect();
        let total = 200_000.0;
        let (range, above, below) = window(&items, 100_000.0, 600.0);
        assert!(range.len() < 200, "{range:?}");
        assert!(range.contains(&5000) && range.contains(&5029));
        let built: f32 = items[range].iter().map(|(_, h)| h).sum();
        assert_eq!(above + built + below, total);

        let (top, above, _) = window(&items, 0.0, 600.0);
        assert_eq!((top.start, above), (0, 0.0));
        let (end, _, below) = window(&items, total - 600.0, 600.0);
        assert_eq!((end.end, below), (10_000, 0.0));
        let (none, above, below) = window(&[], 0.0, 600.0);
        assert_eq!((none, above, below), (0..0, 0.0, 0.0));
    }

    #[test]
    fn the_panel_follows_playback_until_the_user_scrolls_away() {
        let cues: Vec<SubCue> = (0..200)
            .map(|i| cue(i + 1, i as f64 * 10.0, i as f64 * 10.0 + 5.0))
            .collect();
        let mut state = State {
            view_height: 300.0,
            ..State::default()
        };
        let later = || Instant::now() + GLIDE * 2;
        follow(&mut state, &cues, Some(50), 0.0);
        assert_eq!(state.followed, Some(50));
        assert!(is_animating(&state), "glides there");
        let _ = frame(&mut state, later());
        assert!(state.glide.is_none());
        let followed_to = state.offset;
        assert!(followed_to > 0.0);

        // our own scroll comes back, even from the old place first: still following
        scrolled(&mut state, 0.0, 300.0);
        assert!(!state.not_following);
        scrolled(&mut state, followed_to, 300.0);
        assert!(!state.not_following);

        // the user's own
        scrolled(&mut state, followed_to + 200.0, 300.0);
        assert!(state.not_following);
        follow(&mut state, &cues, Some(60), 0.0);
        assert_eq!(state.followed, Some(50), "stays where the user went");
        assert!(state.glide.is_none());

        send(&mut state, Msg::FollowAgain);
        follow(&mut state, &cues, Some(60), 0.0);
        assert_eq!(state.followed, Some(60));
        let _ = frame(&mut state, later());
        assert!(state.offset > followed_to);
    }

    #[test]
    fn a_glide_starts_gently_and_a_far_jump_does_not_fly_past_the_whole_list() {
        let cues: Vec<SubCue> = (0..2000)
            .map(|i| cue(i + 1, i as f64 * 10.0, i as f64 * 10.0 + 5.0))
            .collect();
        let mut state = State {
            view_height: 300.0,
            ..State::default()
        };
        follow(&mut state, &cues, Some(1500), 0.0);
        let glide = state.glide.unwrap();
        assert_eq!(glide.to - glide.from, 300.0);
        let _ = frame(&mut state, glide.since + GLIDE / 10);
        assert!(state.offset - glide.from < 300.0 * 0.1, "{}", state.offset);
        assert!(state.glide.is_some());
    }

    #[test]
    fn the_highlight_moves_softly_even_when_the_panel_is_not_following() {
        let cues = [cue(1, 0.0, 1.0), cue(2, 2.0, 3.0)];
        let mut state = State {
            not_following: true,
            ..State::default()
        };
        follow(&mut state, &cues, Some(0), 0.0);
        follow(&mut state, &cues, Some(1), 0.0);
        assert_eq!(state.lit.unwrap().0, 1);
        assert_eq!(state.unlit.unwrap().0, 0);
        assert!(is_animating(&state));
        assert!(glow(&state, 1) < 0.5);
        assert!(glow(&state, 0) > 0.5);
        assert_eq!(glow(&state, 5), 0.0);
    }

    #[test]
    fn with_no_line_showing_the_panel_goes_to_the_nearest_one() {
        let cues: Vec<SubCue> = (0..100)
            .map(|i| cue(i + 1, 1.0 + i as f64 * 10.0, 5.0 + i as f64 * 10.0))
            .collect();
        let mut state = State {
            view_height: 300.0,
            ..State::default()
        };
        follow(&mut state, &cues, Some(50), 501.0);
        let _ = frame(&mut state, Instant::now() + GLIDE * 2);
        assert!(state.offset > 0.0);

        // Stop: position 0, before the first line, and nothing on screen
        follow(&mut state, &cues, None, 0.0);
        assert_eq!(state.followed, Some(0));
        let _ = frame(&mut state, Instant::now() + GLIDE * 2);
        assert_eq!(state.offset, 0.0);

        // in the silence after line 20
        follow(&mut state, &cues, None, 207.0);
        assert_eq!(state.followed, Some(20));
        assert!(state.lit.is_none(), "nothing is lit between lines");

        follow(&mut State::default(), &[], None, 5.0);
    }

    #[test]
    fn following_never_scrolls_past_the_end() {
        let cues: Vec<SubCue> = (0..10)
            .map(|i| cue(i + 1, i as f64, i as f64 + 0.5))
            .collect();
        let mut state = State {
            view_height: 300.0,
            ..State::default()
        };
        follow(&mut state, &cues, Some(9), 0.0);
        let _ = frame(&mut state, Instant::now() + GLIDE * 2);
        let total: f32 = layout(&state, &cues, &[]).iter().map(|(_, h)| h).sum();
        assert!(state.offset <= (total - 300.0).max(0.0));
    }

    #[test]
    fn a_save_does_not_hold_up_the_click() {
        let dir = TestDir::new("async");
        let mut state = opened(&dir);
        // jam the save thread: the click must still return at once
        let (release, jammed) = std::sync::mpsc::channel::<()>();
        saver::run(move || {
            let _ = jammed.recv();
        });
        let (task, _) = update(&mut state, Msg::Add(MarkKind::Bookmark, at(1.0, None)));
        drop(task);
        assert_eq!(marks(&state).len(), 1, "memory has it already");
        assert!(!dir.job().paths.marks().exists(), "the disk not yet");
        release.send(()).unwrap();
        saver::wait();
        assert!(dir.job().paths.marks().exists());
    }

    #[test]
    fn clock_formats() {
        assert_eq!(clock(0.0), "0:00");
        assert_eq!(clock(65.9), "1:05");
        assert_eq!(clock(3725.0), "1:02:05");
        assert_eq!(clock(-3.0), "0:00");
    }
}
