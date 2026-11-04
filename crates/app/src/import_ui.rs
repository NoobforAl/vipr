//! Import dialog: file, drop or paste, then a preview, then translate or compare. Stub from the
//! foundation step; agent G fills it in. `main.rs` draws `view` over the window while `is_open`.

// temporary: `open` has no caller until integration adds the button
#![allow(dead_code)]

use iced::widget::text;
use iced::{Element, Task};

use crate::out::{JobRef, Out};
use crate::theme::Tokens;

#[derive(Debug, Default)]
pub struct State {}

#[derive(Debug, Clone)]
pub enum Msg {}

/// Open the dialog for imports into `job`.
pub fn open(_state: &mut State, _job: &JobRef) -> (Task<Msg>, Out) {
    (Task::none(), Out::None)
}

pub fn is_open(_state: &State) -> bool {
    false
}

pub fn update(_state: &mut State, msg: Msg) -> (Task<Msg>, Out) {
    match msg {}
}

pub fn view(_state: &State, _tokens: Tokens) -> Element<'_, Msg> {
    text("Import").into()
}
