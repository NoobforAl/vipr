//! Profiles section of Settings. Stub from the foundation step; agent D fills it in. Integration
//! puts `view` into the Settings dialog; until then nothing draws it.

// temporary: `view` has no caller until integration
#![allow(dead_code)]

use engine::profiles::Profile;
use engine::settings::Settings;
use iced::widget::text;
use iced::{Element, Task};

use crate::out::Out;
use crate::theme::Tokens;

#[derive(Debug, Default)]
pub struct State {}

#[derive(Debug, Clone)]
pub enum Msg {}

/// Load the store at start-up. A notice about a corrupt file comes back as `Out::Notice`.
pub fn init(_state: &mut State) -> (Task<Msg>, Out) {
    (Task::none(), Out::None)
}

/// `settings.active_profile` lives in `settings`. `main.rs` saves `settings` when that changes.
pub fn update(_state: &mut State, msg: Msg, _settings: &mut Settings) -> (Task<Msg>, Out) {
    match msg {}
}

pub fn view<'a>(_state: &'a State, _settings: &'a Settings, _tokens: Tokens) -> Element<'a, Msg> {
    text("Profiles").into()
}

/// The profile new work should use.
pub fn active(_state: &State, _settings: &Settings) -> Option<Profile> {
    None
}
