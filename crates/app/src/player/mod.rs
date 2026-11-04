//! The player page. Stub from the foundation step, deliberately minimal because the S1 spike can
//! still change the design; agent C owns everything here and may add files to this folder.
//! `main.rs` depends only on the items below.

// temporary: nothing reads the `SeekTo` time until agent C does
#![allow(dead_code)]

use engine::marks::Mark;
use iced::keyboard::{Key, Modifiers};
use iced::widget::text;
use iced::{Element, Subscription, Task};

use crate::out::{JobRef, Out, Position};
use crate::theme::Tokens;

#[derive(Debug, Default)]
pub struct State {}

#[derive(Debug, Clone)]
pub enum Msg {
    /// Seek to this many seconds. `main.rs` sends it for `Out::PlayAt`.
    SeekTo(f64),
}

/// The page is opening on `job`.
pub fn open(_state: &mut State, _job: &JobRef) -> (Task<Msg>, Out) {
    (Task::none(), Out::None)
}

/// The page is closing, or the window is. Ends the playback session.
pub fn close(_state: &mut State) {}

pub fn update(_state: &mut State, msg: Msg) -> (Task<Msg>, Out) {
    match msg {
        Msg::SeekTo(_) => (Task::none(), Out::None),
    }
}

/// `marks` are drawn as ticks on the seek bar. They are owned by `marks_ui`.
pub fn view<'a>(_state: &'a State, _tokens: Tokens, _marks: &'a [Mark]) -> Element<'a, Msg> {
    text("Player").into()
}

pub fn subscription(_state: &State) -> Subscription<Msg> {
    Subscription::none()
}

/// A key nothing else used, while this page is open.
pub fn key(_state: &State, _key: &Key, _mods: Modifiers) -> Option<Msg> {
    None
}

pub fn position(_state: &State) -> Position {
    Position::default()
}
