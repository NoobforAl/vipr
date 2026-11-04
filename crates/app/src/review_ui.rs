//! The compare review page. Stub from the foundation step; agent G fills it in. "Play this moment"
//! returns `Out::PlayAt`.

use iced::keyboard::{Key, Modifiers};
use iced::widget::text;
use iced::{Element, Task};

use crate::out::{JobRef, Out};
use crate::theme::Tokens;

#[derive(Debug, Default)]
pub struct State {}

#[derive(Debug, Clone)]
pub enum Msg {}

/// The page is opening on `job`: load its newest `<stem>.compare-<slug>.json`.
pub fn open(_state: &mut State, _job: &JobRef) -> (Task<Msg>, Out) {
    (Task::none(), Out::None)
}

pub fn update(_state: &mut State, msg: Msg) -> (Task<Msg>, Out) {
    match msg {}
}

pub fn view(_state: &State, _tokens: Tokens) -> Element<'_, Msg> {
    text("Review").into()
}

/// A key nothing else used, while this page is open.
pub fn key(_state: &State, _key: &Key, _mods: Modifiers) -> Option<Msg> {
    None
}
