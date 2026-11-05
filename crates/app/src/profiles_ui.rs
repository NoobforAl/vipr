//! Profiles section of Settings. Integration puts `view` into the Settings dialog; until then
//! nothing draws it.
//!
//! Edits go into the store as they are typed, so `active` always sees them. Saving happens at most
//! once per [`SAVE_DELAY`], since each save flushes a file to disk, and at once for anything that
//! isn't typing.

// temporary: `view` has no caller until integration
#![allow(dead_code)]

use std::fmt;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use engine::lang::clean_separators;
use engine::profiles::{Profile, ProfileStore, format_glossary, parse_glossary};
use engine::settings::Settings;
use iced::widget::text_editor::{Action, Content, Edit};
use iced::widget::{
    button, column, container, pick_list, row, space, text, text_editor, text_input,
};
use iced::{Alignment, Background, Border, Element, Length, Task, Theme, border};

use crate::icons;
use crate::out::Out;
use crate::theme::{self, Tokens};

const SAVE_DELAY: Duration = Duration::from_millis(800);
const EDITOR_HEIGHT: f32 = 110.0;

#[derive(Debug, Default)]
pub struct State {
    /// `None` only before `init`.
    store: Option<ProfileStore>,
    /// The profile open for editing, which need not be the active one.
    selected: Option<String>,
    name: String,
    /// Why `name` isn't in the store yet.
    name_problem: Option<&'static str>,
    prompt: Content,
    glossary: Content,
    confirm_delete: bool,
    /// From loading. Stays up for the session, and also goes out as `Out::Alert`.
    notice: Option<String>,
    save_error: Option<String>,
    /// The store holds edits the file doesn't.
    dirty: bool,
    /// A `SaveDue` is on its way.
    timer_running: bool,
}

/// An entry of the active profile selector. `id: None` is "no profile".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Choice {
    id: Option<String>,
    label: String,
}

impl fmt::Display for Choice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label)
    }
}

#[derive(Debug, Clone)]
pub enum Msg {
    SetActive(Choice),
    Select(String),
    New,
    NameChanged(String),
    PromptEdited(Action),
    GlossaryEdited(Action),
    AskDelete,
    CancelDelete,
    Delete,
    SaveDue,
}

/// Load the store at start-up. A notice about a corrupt file comes back as `Out::Alert`.
pub fn init(state: &mut State) -> (Task<Msg>, Out) {
    (
        Task::none(),
        init_from(state, &ProfileStore::default_path()),
    )
}

fn init_from(state: &mut State, path: &Path) -> Out {
    let loaded = ProfileStore::load(path);
    let first = loaded.store.profiles.first().map(|p| p.id.clone());
    state.store = Some(loaded.store);
    // the notice can quote the file's content and its path, so it counts as untrusted
    let notice = loaded.notice.as_deref().map(clean_separators);
    state.notice = notice.clone();
    select(state, first);
    notice.map_or(Out::None, Out::Alert)
}

/// `settings.active_profile` lives in `settings`. `main.rs` saves `settings` when that changes.
pub fn update(state: &mut State, msg: Msg, settings: &mut Settings) -> (Task<Msg>, Out) {
    let task = match msg {
        Msg::SetActive(choice) => {
            settings.active_profile = choice.id;
            Task::none()
        }
        Msg::Select(id) => {
            save_now(state);
            select(state, Some(id));
            Task::none()
        }
        Msg::New => {
            let id = state.store.as_mut().map(|store| store.create(""));
            state.dirty = true;
            save_now(state);
            select(state, id);
            Task::none()
        }
        Msg::NameChanged(typed) => name_changed(state, &typed),
        Msg::PromptEdited(action) => {
            let edited = action.is_edit();
            state.prompt.perform(clean_paste(action));
            if edited {
                let prompt = state.prompt.text();
                change_selected(state, |profile| profile.prompt = prompt)
            } else {
                Task::none()
            }
        }
        Msg::GlossaryEdited(action) => {
            let edited = action.is_edit();
            state.glossary.perform(clean_paste(action));
            if edited {
                let terms = parse_glossary(&state.glossary.text());
                change_selected(state, |profile| profile.glossary = terms)
            } else {
                Task::none()
            }
        }
        Msg::AskDelete => {
            state.confirm_delete = true;
            Task::none()
        }
        Msg::CancelDelete => {
            state.confirm_delete = false;
            Task::none()
        }
        Msg::Delete => {
            delete_selected(state, settings);
            Task::none()
        }
        Msg::SaveDue => {
            state.timer_running = false;
            save_now(state);
            Task::none()
        }
    };
    (task, Out::None)
}

/// Pasted text can hold separators that crash the text layout (spikes/rtl).
fn clean_paste(action: Action) -> Action {
    match action {
        Action::Edit(Edit::Paste(pasted)) => {
            Action::Edit(Edit::Paste(Arc::new(clean_separators(&pasted))))
        }
        other => other,
    }
}

/// For the one-line boxes and the selector. Names come from a file, or a paste.
fn one_line(text: &str) -> String {
    clean_separators(text).replace(['\r', '\n'], " ")
}

/// Open a profile for editing. Profile text comes from a file, so it is cleaned on the way in.
fn select(state: &mut State, id: Option<String>) {
    let profile = id
        .as_deref()
        .and_then(|id| state.store.as_ref()?.get(id))
        .cloned();
    state.confirm_delete = false;
    state.name_problem = None;
    let Some(profile) = profile else {
        state.selected = None;
        state.name.clear();
        state.prompt = Content::new();
        state.glossary = Content::new();
        return;
    };
    state.selected = Some(profile.id);
    state.name = one_line(&profile.name);
    state.prompt = Content::with_text(&clean_separators(&profile.prompt));
    state.glossary = Content::with_text(&clean_separators(&format_glossary(&profile.glossary)));
}

fn name_changed(state: &mut State, typed: &str) -> Task<Msg> {
    state.name = one_line(typed);
    let name = state.name.trim().to_string();
    let selected = state.selected.as_deref();
    let taken = state
        .store
        .as_ref()
        .is_some_and(|store| store.name_taken(&name, selected));
    state.name_problem = if name.is_empty() {
        Some("A profile needs a name. The old one stays until this is filled in.")
    } else if taken {
        Some("Another profile has this name. The old one stays until this is changed.")
    } else {
        None
    };
    if state.name_problem.is_some() {
        return Task::none();
    }
    change_selected(state, |profile| profile.name = name)
}

fn change_selected(state: &mut State, change: impl FnOnce(&mut Profile)) -> Task<Msg> {
    let found = state.selected.as_deref().and_then(|id| {
        let store = state.store.as_mut()?;
        let mut profile = store.get(id)?.clone();
        change(&mut profile);
        Some(store.update(profile))
    });
    if found != Some(true) {
        return Task::none();
    }
    state.dirty = true;
    if state.timer_running {
        return Task::none();
    }
    state.timer_running = true;
    Task::perform(wait(SAVE_DELAY), |()| Msg::SaveDue)
}

/// Sleep on a thread of its own: iced runs without an async timer here.
async fn wait(delay: Duration) {
    let (done, finished) = iced::futures::channel::oneshot::channel();
    std::thread::spawn(move || {
        std::thread::sleep(delay);
        let _ = done.send(());
    });
    let _ = finished.await;
}

fn delete_selected(state: &mut State, settings: &mut Settings) {
    let Some(id) = state.selected.clone() else {
        return;
    };
    if state.store.as_mut().is_some_and(|store| store.remove(&id)) {
        if settings.active_profile.as_deref() == Some(id.as_str()) {
            settings.active_profile = None;
        }
        state.dirty = true;
        save_now(state);
    }
    let next = state
        .store
        .as_ref()
        .and_then(|store| store.profiles.first())
        .map(|p| p.id.clone());
    select(state, next);
}

fn save_now(state: &mut State) {
    if !state.dirty {
        return;
    }
    let Some(store) = &state.store else {
        return;
    };
    match store.save() {
        Ok(()) => {
            state.dirty = false;
            state.save_error = None;
        }
        Err(e) => {
            state.save_error = Some(clean_separators(&format!(
                "Profiles could not be saved: {e}"
            )));
        }
    }
}

/// Save edits the delay is still holding back. Integration calls this when Settings or the window
/// closes.
pub fn flush(state: &mut State) {
    save_now(state);
}

/// The body of the section; the caller draws the card and the title.
pub fn view<'a>(state: &'a State, settings: &'a Settings, tokens: Tokens) -> Element<'a, Msg> {
    let t = tokens;
    let mut body = column![].spacing(14);
    for problem in [&state.notice, &state.save_error].into_iter().flatten() {
        body = body.push(
            row![
                icons::icon(icons::ALERT, 16.0, t.warn),
                text(problem.as_str())
                    .size(12)
                    .color(t.warn)
                    .width(Length::Fill),
            ]
            .spacing(8),
        );
    }

    let profiles = state.store.as_ref().map_or(&[][..], |s| &s.profiles[..]);
    let none = Choice {
        id: None,
        label: "No profile".into(),
    };
    let mut choices = vec![none.clone()];
    choices.extend(profiles.iter().map(|p| Choice {
        id: Some(p.id.clone()),
        label: one_line(&p.name),
    }));
    let current = choices
        .iter()
        .find(|c| c.id.is_some() && c.id == settings.active_profile)
        .cloned()
        .unwrap_or(none);
    body = body.push(
        row![
            column![
                text("Active profile").size(14).color(t.text),
                text("New translations and comparisons use its prompt and keywords.")
                    .size(12)
                    .color(t.muted),
            ]
            .spacing(2)
            .width(Length::Fill),
            pick_list(choices, Some(current), Msg::SetActive)
                .style(theme::pick(t))
                .padding(9)
                .width(Length::Fixed(240.0)),
        ]
        .spacing(12)
        .align_y(Alignment::Center),
    );

    let tabs = profiles.iter().map(|p| {
        let selected = state.selected.as_deref() == Some(p.id.as_str());
        button(text(one_line(&p.name)).size(13))
            .padding([7, 12])
            .style(theme::segment(t, selected))
            .on_press(Msg::Select(p.id.clone()))
            .into()
    });
    let new = button(
        row![
            icons::icon(icons::PLUS, 14.0, t.text),
            text("New profile").size(13)
        ]
        .spacing(6)
        .align_y(Alignment::Center),
    )
    .padding([7, 12])
    .style(theme::outline_btn(t))
    .on_press(Msg::New);
    body = body.push(
        row(tabs)
            .push(new)
            .spacing(8)
            .align_y(Alignment::Center)
            .wrap()
            .vertical_spacing(8),
    );

    if state.selected.is_some() {
        body = body.push(editor(state, t));
    }
    body.into()
}

fn editor(state: &State, t: Tokens) -> Element<'_, Msg> {
    let mut name = column![
        label("Name", t),
        text_input("Profile name", &state.name)
            .size(13)
            .padding(9)
            .style(theme::input(t))
            .on_input(Msg::NameChanged),
    ]
    .spacing(4);
    if let Some(problem) = state.name_problem {
        name = name.push(text(problem).size(12).color(t.warn));
    }
    let prompt = column![
        label("Prompt", t),
        text("Background for the AI: the subject, the tone, who it's for.")
            .size(12)
            .color(t.muted),
        text_editor(&state.prompt)
            .placeholder("This recording is a lecture on cell biology, for students.")
            .size(13)
            .padding(9)
            .height(Length::Fixed(EDITOR_HEIGHT))
            .style(editor_style(t))
            .on_action(Msg::PromptEdited),
    ]
    .spacing(4);
    let keywords = column![
        label("Glossary keywords", t),
        text("One per line or separated by commas. Write source=target to fix how a term is translated.")
            .size(12)
            .color(t.muted),
        text_editor(&state.glossary)
            .placeholder("Kubernetes\nGPU=graphics processor")
            .size(13)
            .padding(9)
            .height(Length::Fixed(EDITOR_HEIGHT))
            .style(editor_style(t))
            .on_action(Msg::GlossaryEdited),
    ]
    .spacing(4);
    let limitation = text(
        "In right-to-left text the caret and selection can appear on the wrong side while editing. The text itself is saved correctly.",
    )
    .size(12)
    .color(t.faint);

    let delete: Element<'_, Msg> = if state.confirm_delete {
        row![
            text("Delete this profile? This can't be undone.")
                .size(13)
                .color(t.bad)
                .width(Length::Fill),
            button(text("Delete").size(13))
                .padding([7, 12])
                .style(theme::outline_btn(t))
                .on_press(Msg::Delete),
            button(text("Keep").size(13))
                .padding([7, 12])
                .style(theme::ghost(t))
                .on_press(Msg::CancelDelete),
        ]
        .spacing(8)
        .align_y(Alignment::Center)
        .into()
    } else {
        row![
            space::horizontal(),
            button(text("Delete profile").size(13))
                .padding([7, 12])
                .style(theme::ghost(t))
                .on_press(Msg::AskDelete),
        ]
        .into()
    };
    container(column![name, prompt, keywords, limitation, delete].spacing(12))
        .width(Length::Fill)
        .into()
}

fn label(title: &str, t: Tokens) -> Element<'_, Msg> {
    text(title).size(13).color(t.text).into()
}

/// `theme::selectable` is for read-only text, so these get the look of `theme::input`.
fn editor_style(t: Tokens) -> impl Fn(&Theme, text_editor::Status) -> text_editor::Style {
    move |_, status| {
        let color = match status {
            text_editor::Status::Focused { .. } => t.accent,
            text_editor::Status::Hovered => t.faint,
            _ => t.rule,
        };
        text_editor::Style {
            background: Background::Color(t.bg),
            border: Border {
                color,
                width: 1.0,
                radius: border::radius(theme::RADIUS_S),
            },
            placeholder: t.faint,
            value: t.text,
            selection: theme::with_alpha(t.accent, 0.35),
        }
    }
}

/// The profile new work should use.
pub fn active(state: &State, settings: &Settings) -> Option<Profile> {
    let id = settings.active_profile.as_deref()?;
    state.store.as_ref()?.get(id).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::profiles::GlossaryTerm;
    use std::path::PathBuf;

    /// A fresh folder under the system temp dir, removed when dropped.
    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> TestDir {
            let dir = std::env::temp_dir().join(format!(
                "vipr-profiles-ui-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            TestDir(dir)
        }

        fn file(&self) -> PathBuf {
            self.0.join("profiles.json")
        }

        fn state(&self) -> State {
            let mut state = State::default();
            init_from(&mut state, &self.file());
            state
        }

        fn on_disk(&self) -> Vec<Profile> {
            ProfileStore::load(&self.file()).store.profiles
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn send(state: &mut State, settings: &mut Settings, msg: Msg) {
        let _ = update(state, msg, settings);
    }

    fn paste(text: &str) -> Action {
        Action::Edit(Edit::Paste(Arc::new(text.to_string())))
    }

    #[test]
    fn a_corrupt_file_gives_a_notice_out_and_keeps_it_in_the_section() {
        let dir = TestDir::new("corrupt");
        std::fs::write(dir.file(), "{not json").unwrap();
        let mut state = State::default();
        let out = init_from(&mut state, &dir.file());
        assert!(matches!(out, Out::Alert(line) if line.contains(".bak")));
        assert!(state.notice.is_some());
        assert!(state.selected.is_none());
    }

    fn has_separator(text: &str) -> bool {
        text.contains(['\u{2029}', '\u{85}', '\u{1C}', '\u{1D}', '\u{1E}'])
    }

    #[test]
    fn notices_and_save_errors_with_a_separator_are_cleaned_for_display() {
        let dir = TestDir::new("separators");
        // serde escapes this content when quoting it, but the notice names the folder unescaped
        let folder = dir.0.join("abc\u{2029}سلام");
        std::fs::create_dir(&folder).unwrap();
        let path = folder.join("profiles.json");
        std::fs::write(&path, r#"{"version":"abc سلام"}"#).unwrap();

        let raw = ProfileStore::load(&path).notice.unwrap();
        assert!(has_separator(&raw), "the engine notice quotes it: {raw:?}");

        std::fs::write(&path, r#"{"version":"abc سلام"}"#).unwrap();
        let (mut state, mut settings) = (State::default(), Settings::default());
        let out = init_from(&mut state, &path);
        let Out::Alert(line) = out else {
            panic!("expected a notice");
        };
        assert!(!has_separator(&line), "{line:?}");
        assert!(!has_separator(state.notice.as_deref().unwrap()));

        // a folder at the file's path, so the save fails and its error names the path
        std::fs::create_dir(&path).unwrap();
        send(&mut state, &mut settings, Msg::New);
        let error = state.save_error.unwrap();
        assert!(!has_separator(&error), "{error:?}");
    }

    #[test]
    fn profile_text_from_a_file_is_cleaned_before_any_widget_gets_it() {
        let dir = TestDir::new("file-text");
        std::fs::write(
            dir.file(),
            r#"{"profiles":[{"id":"a","name":"abc سلام","prompt":"abc\u0085سلام",
                "glossary":[{"source":"abc\u001Cسلام","target":"x سلام"}]}]}"#,
        )
        .unwrap();
        let state = dir.state();
        assert!(!has_separator(&state.name), "{:?}", state.name);
        assert!(!has_separator(&state.prompt.text()));
        assert!(!has_separator(&state.glossary.text()));
    }

    #[test]
    fn new_saves_at_once_and_opens_the_profile() {
        let dir = TestDir::new("new");
        let (mut state, mut settings) = (dir.state(), Settings::default());
        send(&mut state, &mut settings, Msg::New);
        let disk = dir.on_disk();
        assert_eq!(disk.len(), 1);
        assert_eq!(state.selected.as_deref(), Some(disk[0].id.as_str()));
        assert_eq!(state.name, "New profile");
        assert_eq!(settings.active_profile, None);
    }

    #[test]
    fn edits_reach_active_at_once_and_the_file_after_the_delay() {
        let dir = TestDir::new("edits");
        let (mut state, mut settings) = (dir.state(), Settings::default());
        send(&mut state, &mut settings, Msg::New);
        let id = state.selected.clone().unwrap();
        settings.active_profile = Some(id);

        send(&mut state, &mut settings, Msg::PromptEdited(paste("سلام")));
        send(
            &mut state,
            &mut settings,
            Msg::GlossaryEdited(paste("API=رابط, GPU")),
        );
        let profile = active(&state, &settings).unwrap();
        assert_eq!(profile.prompt, "سلام");
        assert_eq!(
            profile.glossary,
            [
                GlossaryTerm {
                    source: "API".into(),
                    target: Some("رابط".into())
                },
                GlossaryTerm {
                    source: "GPU".into(),
                    target: None
                },
            ]
        );
        assert_eq!(dir.on_disk()[0].prompt, "", "not saved yet");
        assert!(state.dirty && state.timer_running);

        send(&mut state, &mut settings, Msg::SaveDue);
        assert_eq!(dir.on_disk()[0].prompt, "سلام");
        assert!(!state.dirty && !state.timer_running);
    }

    #[test]
    fn saved_text_reopens_exactly_with_no_growing_line_ends() {
        let dir = TestDir::new("exact");
        let (mut state, mut settings) = (dir.state(), Settings::default());
        send(&mut state, &mut settings, Msg::New);
        send(
            &mut state,
            &mut settings,
            Msg::PromptEdited(paste("one\ntwo")),
        );
        flush(&mut state);

        for _ in 0..3 {
            let mut state = dir.state();
            assert_eq!(state.prompt.text().trim_end(), "one\ntwo");
            state.prompt.perform(Action::Edit(Edit::Insert('!')));
            state.prompt.perform(Action::Edit(Edit::Backspace));
            assert_eq!(state.prompt.text().trim_end(), "one\ntwo");
        }
        assert_eq!(dir.on_disk()[0].prompt, "one\ntwo");
    }

    #[test]
    fn pasted_separators_are_cleaned_before_the_editor_sees_them() {
        let dir = TestDir::new("paste");
        let (mut state, mut settings) = (dir.state(), Settings::default());
        send(&mut state, &mut settings, Msg::New);
        send(
            &mut state,
            &mut settings,
            Msg::PromptEdited(paste("abc\u{2029}سلام\u{85}x")),
        );
        assert_eq!(state.prompt.text().trim_end(), "abc\nسلام\nx");
    }

    #[test]
    fn a_bad_name_stays_in_the_box_but_not_in_the_store() {
        let dir = TestDir::new("name");
        let (mut state, mut settings) = (dir.state(), Settings::default());
        send(&mut state, &mut settings, Msg::New);
        send(&mut state, &mut settings, Msg::New);
        send(&mut state, &mut settings, Msg::NameChanged("Work".into()));
        let id = state.selected.clone().unwrap();
        let name = |state: &State| state.store.as_ref().unwrap().get(&id).unwrap().name.clone();
        assert_eq!(name(&state), "Work");

        send(&mut state, &mut settings, Msg::NameChanged("   ".into()));
        assert!(state.name_problem.is_some());
        assert_eq!(state.name, "   ");
        assert_eq!(name(&state), "Work");

        send(
            &mut state,
            &mut settings,
            Msg::NameChanged("new PROFILE".into()),
        );
        assert!(state.name_problem.is_some());
        assert_eq!(name(&state), "Work");

        send(&mut state, &mut settings, Msg::NameChanged(" Home ".into()));
        assert_eq!(state.name_problem, None);
        assert_eq!(name(&state), "Home");
    }

    #[test]
    fn switching_profiles_loads_the_other_one_and_saves_pending_edits() {
        let dir = TestDir::new("switch");
        let (mut state, mut settings) = (dir.state(), Settings::default());
        send(&mut state, &mut settings, Msg::New);
        let first = state.selected.clone().unwrap();
        send(
            &mut state,
            &mut settings,
            Msg::PromptEdited(paste("first prompt")),
        );
        send(&mut state, &mut settings, Msg::New);
        let second = state.selected.clone().unwrap();
        assert_eq!(state.prompt.text().trim_end(), "");
        assert_ne!(first, second);

        send(&mut state, &mut settings, Msg::Select(first));
        assert_eq!(state.prompt.text().trim_end(), "first prompt");
        assert_eq!(dir.on_disk()[0].prompt, "first prompt");
    }

    #[test]
    fn deleting_the_active_profile_clears_it_and_opens_another() {
        let dir = TestDir::new("delete");
        let (mut state, mut settings) = (dir.state(), Settings::default());
        send(&mut state, &mut settings, Msg::New);
        let keep = state.selected.clone().unwrap();
        send(&mut state, &mut settings, Msg::New);
        let gone = state.selected.clone().unwrap();
        settings.active_profile = Some(gone.clone());

        send(&mut state, &mut settings, Msg::AskDelete);
        assert!(state.confirm_delete);
        send(&mut state, &mut settings, Msg::CancelDelete);
        assert_eq!(dir.on_disk().len(), 2);

        send(&mut state, &mut settings, Msg::Delete);
        assert_eq!(settings.active_profile, None);
        assert_eq!(state.selected, Some(keep));
        assert!(dir.on_disk().iter().all(|p| p.id != gone));
    }

    #[test]
    fn deleting_another_profile_leaves_the_active_one() {
        let dir = TestDir::new("delete-other");
        let (mut state, mut settings) = (dir.state(), Settings::default());
        send(&mut state, &mut settings, Msg::New);
        let active_id = state.selected.clone().unwrap();
        send(&mut state, &mut settings, Msg::New);
        settings.active_profile = Some(active_id.clone());
        send(&mut state, &mut settings, Msg::Delete);
        assert_eq!(settings.active_profile, Some(active_id));
    }

    #[test]
    fn set_active_writes_to_settings_and_an_unknown_id_is_no_profile() {
        let dir = TestDir::new("active");
        let (mut state, mut settings) = (dir.state(), Settings::default());
        send(&mut state, &mut settings, Msg::New);
        let id = state.selected.clone().unwrap();
        let choice = Choice {
            id: Some(id.clone()),
            label: "x".into(),
        };
        send(&mut state, &mut settings, Msg::SetActive(choice));
        assert_eq!(settings.active_profile, Some(id));
        assert!(active(&state, &settings).is_some());

        settings.active_profile = Some("gone".into());
        assert_eq!(active(&state, &settings), None);
    }

    #[test]
    fn a_failed_save_is_kept_visible_and_the_edit_stays_unsaved() {
        let dir = TestDir::new("blocked");
        // a folder at the file's path, so the store loads with saving blocked
        std::fs::create_dir(dir.file()).unwrap();
        let (mut state, mut settings) = (dir.state(), Settings::default());
        assert!(state.notice.is_some());
        send(&mut state, &mut settings, Msg::New);
        assert!(state.save_error.is_some());
        assert!(state.dirty, "the edit is still waiting to be saved");
    }
}
