//! Side panel of bookmarks and flagged lines on the player page. Stub from the foundation step;
//! agent D fills it in. Click-to-jump returns `Out::PlayAt`.

use engine::marks::Mark;
use iced::keyboard::{Key, Modifiers};
use iced::widget::text;
use iced::{Element, Task};

use crate::out::{JobRef, Out, Position};
use crate::theme::Tokens;

#[derive(Debug, Default)]
pub struct State {}

#[derive(Debug, Clone)]
pub enum Msg {}

/// The player page is opening on `job`: load its marks.
pub fn open(_state: &mut State, _job: &JobRef) -> (Task<Msg>, Out) {
    (Task::none(), Out::None)
}

pub fn update(_state: &mut State, msg: Msg) -> (Task<Msg>, Out) {
    match msg {}
}

pub fn view(_state: &State, _tokens: Tokens) -> Element<'_, Msg> {
    text("Marks").into()
}

/// M and F on the player page. `at` is where playback is now. `player::key` is asked first.
pub fn key(_state: &State, _key: &Key, _mods: Modifiers, _at: Position) -> Option<Msg> {
    None
}

/// For the seek bar's ticks.
pub fn marks(_state: &State) -> &[Mark] {
    &[]
}
