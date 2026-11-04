//! Side panel of bookmarks and flagged lines on the player page. Click-to-jump returns
//! `Out::PlayAt`. Every change saves at once, so a crash loses at most a note being typed.

use engine::lang::clean_separators;
use engine::marks::{Mark, MarkKind, MarkStore};
use iced::keyboard::{Key, Modifiers};
use iced::widget::{Id, button, column, container, row, scrollable, space, text, text_input};
use iced::{Alignment, Element, Length, Task};

use crate::icons;
use crate::out::{JobRef, Out, Position, content_align};
use crate::theme::{self, Tokens};

const PANEL_WIDTH: f32 = 290.0;
const NOTE_INPUT: &str = "marks-note";

#[derive(Debug, Default)]
pub struct State {
    job: Option<u64>,
    /// `None` until a job is open, and when its marks file could not be read. Marks can't be added
    /// then, so the file is never overwritten.
    store: Option<MarkStore>,
    /// Shown at the top of the panel.
    problem: Option<String>,
    /// The mark whose note is open, with the text typed so far.
    editing: Option<(u64, String)>,
    /// The mark waiting for "Remove?" to be answered.
    removing: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum Msg {
    Add(MarkKind, Position),
    Jump(f64),
    EditNote(u64),
    NoteChanged(String),
    SaveNote,
    CancelNote,
    AskRemove(u64),
    CancelRemove,
    Remove(u64),
}

/// The player page is opening on `job`: load its marks. A note still open is saved first.
pub fn open(state: &mut State, job: &JobRef) -> (Task<Msg>, Out) {
    flush(state);
    state.job = Some(job.id);
    state.removing = None;
    state.problem = None;
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
    match msg {
        Msg::Add(kind, at) => add(state, kind, at),
        Msg::Jump(time) => {
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
        Msg::AskRemove(id) => {
            let has_note = state
                .store
                .as_ref()
                .and_then(|store| store.marks.iter().find(|m| m.id == id))
                .is_some_and(|mark| !mark.note.trim().is_empty());
            if has_note {
                state.removing = Some(id);
            } else {
                remove(state, id);
            }
        }
        Msg::CancelRemove => state.removing = None,
        Msg::Remove(id) => remove(state, id),
    }
    (Task::none(), Out::None)
}

fn add(state: &mut State, kind: MarkKind, at: Position) {
    let Some(store) = &mut state.store else {
        return;
    };
    // F twice on one line is one flag, however long the line stays on screen.
    let flagged = kind == MarkKind::Flag
        && at.cue.is_some()
        && store
            .marks
            .iter()
            .any(|m| m.kind == MarkKind::Flag && m.cue == at.cue);
    if !flagged {
        let cue = if kind == MarkKind::Flag { at.cue } else { None };
        store.add(kind, at.time, String::new(), cue);
        save(state);
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

fn save(state: &mut State) {
    let Some(store) = &state.store else {
        return;
    };
    state.problem = store
        .save()
        .err()
        .map(|e| clean_separators(&format!("Marks could not be saved: {e}")));
}

/// Keep a note still being typed. Integration calls this when the window closes.
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

pub fn view(state: &State, tokens: Tokens) -> Element<'_, Msg> {
    let t = tokens;
    let mut head = column![text("Marks").size(15).color(t.text)].spacing(4);
    if let Some(problem) = &state.problem {
        head = head.push(text(problem.as_str()).size(12).color(t.bad));
    }
    head = head.push(
        text("M adds a bookmark, F flags the line on screen. Right-to-left notes may show the caret on the wrong side while typing.")
            .size(12)
            .color(t.muted),
    );
    let marks = state.store.as_ref().map_or(&[][..], |s| &s.marks[..]);
    let list: Element<'_, Msg> = if marks.is_empty() {
        text("No marks yet.").size(13).color(t.faint).into()
    } else {
        scrollable(column(marks.iter().map(|m| mark_row(state, m, t))).spacing(8))
            .height(Length::Fill)
            .into()
    };
    container(column![head, list].spacing(12))
        .padding(12)
        .width(Length::Fixed(PANEL_WIDTH))
        .height(Length::Fill)
        .style(theme::panel(t))
        .into()
}

fn mark_row<'a>(state: &'a State, mark: &'a Mark, t: Tokens) -> Element<'a, Msg> {
    let (label, color) = match mark.kind {
        MarkKind::Bookmark => ("Bookmark", t.accent),
        MarkKind::Flag => ("Flag", t.warn),
    };
    let top = row![
        button(text(clock(mark.time)).size(13))
            .padding([3, 6])
            .style(theme::ghost(t))
            .on_press(Msg::Jump(mark.time)),
        container(text(label).size(11).color(color))
            .padding([1, 7])
            .style(theme::chip(color)),
        space::horizontal(),
        button(text("Note").size(12))
            .padding([3, 6])
            .style(theme::ghost(t))
            .on_press(Msg::EditNote(mark.id)),
        button(icons::icon(icons::X, 13.0, t.muted))
            .padding(5)
            .style(theme::ghost(t))
            .on_press(Msg::AskRemove(mark.id)),
    ]
    .spacing(6)
    .align_y(Alignment::Center);

    let mut body = column![top].spacing(6);
    if state.removing == Some(mark.id) {
        body = body.push(
            row![
                text("Remove this mark and its note?")
                    .size(12)
                    .color(t.text)
                    .width(Length::Fill),
                button(text("Remove").size(12))
                    .padding([3, 8])
                    .style(theme::outline_btn(t))
                    .on_press(Msg::Remove(mark.id)),
                button(text("Keep").size(12))
                    .padding([3, 8])
                    .style(theme::ghost(t))
                    .on_press(Msg::CancelRemove),
            ]
            .spacing(6)
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
                        .padding(7)
                        .style(theme::input(t))
                        .on_input(Msg::NoteChanged)
                        .on_submit(Msg::SaveNote),
                    row![
                        button(text("Save").size(12))
                            .padding([3, 8])
                            .style(theme::outline_btn(t))
                            .on_press(Msg::SaveNote),
                        button(text("Cancel").size(12))
                            .padding([3, 8])
                            .style(theme::ghost(t))
                            .on_press(Msg::CancelNote),
                    ]
                    .spacing(6),
                ]
                .spacing(6),
            );
        }
        _ if !mark.note.is_empty() => {
            let note = clean_separators(&mark.note);
            let align = content_align(&note);
            body = body.push(
                container(text(note).size(13).color(t.text))
                    .width(Length::Fill)
                    .align_x(align),
            );
        }
        _ => {}
    }
    container(body)
        .padding(8)
        .width(Length::Fill)
        .style(theme::card(t))
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

    fn send(state: &mut State, msg: Msg) -> Out {
        update(state, msg).1
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
        let problem = state.problem.unwrap();
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
    fn clock_formats() {
        assert_eq!(clock(0.0), "0:00");
        assert_eq!(clock(65.9), "1:05");
        assert_eq!(clock(3725.0), "1:02:05");
        assert_eq!(clock(-3.0), "0:00");
    }
}
