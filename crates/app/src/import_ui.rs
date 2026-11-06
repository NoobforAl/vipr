//! Import dialog: a subtitle file (picked or dropped) or pasted text, then a preview, then translate
//! or compare. `main.rs` draws `view` over the window while `is_open`.
//!
//! Everything in an imported file is untrusted. The parsed text is stored as written, and goes through
//! `lang::clean_separators` right before any widget shows it, because some separators panic in
//! cosmic-text (spikes/rtl).

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use engine::fsio;
use engine::lang::{self, LANGUAGES, Language};
use engine::project::{
    CompareSource, ImportFile, ImportWarning, ProjectPaths, SubFormat, TimedText, VERSION,
};
use engine::subparse::{self, Decoded, Parsed};
use iced::widget::text::Wrapping;
use iced::widget::{
    Column, button, column, combo_box, container, pick_list, row, scrollable, space, text,
    text_editor,
};
use iced::{Alignment, Color, Element, Length, Task};

use crate::icons;
use crate::out::{JobKind, JobRef, Out, TranslateSource, content_align};
use crate::theme::{self, Tokens};

/// Subtitle files are small; anything larger is not one, and would stall the preview.
const MAX_FILE_BYTES: u64 = 32 * 1024 * 1024;
/// Formats are told apart by content, so `.txt` is fine too.
const SUB_EXTENSIONS: &[&str] = &["srt", "vtt", "ass", "ssa", "txt"];
/// `encoding_rs` labels offered when detection guesses wrong.
const ENCODINGS: &[&str] = &[
    "UTF-8",
    "UTF-16LE",
    "UTF-16BE",
    "windows-1250",
    "windows-1251",
    "windows-1252",
    "windows-1253",
    "windows-1254",
    "windows-1255",
    "windows-1256",
    "windows-1257",
    "windows-1258",
    "windows-874",
    "KOI8-R",
    "Shift_JIS",
    "EUC-JP",
    "EUC-KR",
    "GBK",
    "gb18030",
    "Big5",
];
const PREVIEW_CUES: usize = 5;
const PREVIEW_WARNINGS: usize = 5;
const PREVIEW_LINE_CHARS: usize = 160;

/// What the imported subtitles are for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Action {
    #[default]
    Translate,
    CompareTranscript,
    CompareTranslation,
}

/// An entry of the encoding picker. `None` means use what detection found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodingChoice(Option<&'static str>);

impl fmt::Display for EncodingChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0.unwrap_or("Detect automatically"))
    }
}

/// An entry of the language picker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target(&'static Language);

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.0.name, self.0.code)
    }
}

/// A parsed subtitle source, as the dialog shows it.
#[derive(Debug, Clone, PartialEq)]
struct Preview {
    /// File name. `None` when pasted.
    source_name: Option<String>,
    format: SubFormat,
    encoding: String,
    cues: Vec<TimedText>,
    warnings: Vec<ImportWarning>,
}

impl Preview {
    fn new(source_name: Option<String>, decoded: Decoded, parsed: Parsed) -> Preview {
        let mut warnings = parsed.warnings;
        if decoded.lossy {
            warnings.insert(
                0,
                ImportWarning {
                    line: None,
                    message: "Some bytes did not fit this encoding and were replaced.".into(),
                },
            );
        }
        Preview {
            source_name,
            format: parsed.format,
            encoding: decoded.encoding,
            cues: parsed.cues,
            warnings,
        }
    }

    /// The sidecar for this preview. Times stay the file's own: the compare run finds the offset.
    fn to_import(&self, slug: &str) -> ImportFile {
        ImportFile {
            version: VERSION,
            slug: slug.to_string(),
            source_name: self.source_name.clone(),
            format: self.format,
            encoding: self.encoding.clone(),
            offset: 0.0,
            warnings: self.warnings.clone(),
            cues: self.cues.clone(),
        }
    }
}

#[derive(Debug, Default)]
pub struct State {
    job: Option<JobRef>,
    /// Bumped whenever the dialog opens or closes, so late results of an earlier one are dropped.
    generation: u64,
    /// Bytes of the current file, kept for the encoding override. `None` for pasted text.
    bytes: Option<Arc<Vec<u8>>>,
    paste: text_editor::Content,
    encoding: Option<&'static str>,
    preview: Option<Preview>,
    action: Action,
    target: Option<Target>,
    /// What the target language box searches.
    targets: combo_box::State<Target>,
    /// The target language already has a translation, which this import would replace.
    target_exists: bool,
    overwrite_confirmed: bool,
    translations: Vec<String>,
    translation: Option<String>,
    reading: bool,
    saving: bool,
    error: Option<String>,
}

#[derive(Debug, Clone)]
pub enum Msg {
    Close,
    PickFile,
    Picked(Option<PathBuf>),
    /// A file dropped on the window. Integration routes it here while the dialog is open.
    FileDropped(PathBuf),
    FileRead {
        generation: u64,
        name: String,
        result: Result<Arc<Vec<u8>>, String>,
    },
    PasteEdit(text_editor::Action),
    PreviewPaste,
    Encoding(EncodingChoice),
    Action(Action),
    Target(Target),
    /// The target language box lost focus: whatever was typed there is dropped.
    TargetBoxClosed,
    ConfirmOverwrite(bool),
    Translation(String),
    Start,
    Saved {
        generation: u64,
        result: Result<String, String>,
    },
}

/// Open the dialog for imports into `job`.
fn target_box() -> combo_box::State<Target> {
    combo_box::State::new(LANGUAGES.iter().map(Target).collect())
}

pub fn open(state: &mut State, job: &JobRef) -> (Task<Msg>, Out) {
    let translations = existing_translations(&job.paths);
    *state = State {
        generation: state.generation + 1,
        job: Some(job.clone()),
        translation: translations.first().cloned(),
        translations,
        targets: target_box(),
        ..State::default()
    };
    (Task::none(), Out::None)
}

pub fn is_open(state: &State) -> bool {
    state.job.is_some()
}

pub fn update(state: &mut State, msg: Msg) -> (Task<Msg>, Out) {
    match msg {
        Msg::Close => {
            // while saving, the answer still has to queue the work it was saved for
            if !state.saving {
                close(state);
            }
        }
        Msg::PickFile => return (Task::perform(pick_file(), Msg::Picked), Out::None),
        Msg::Picked(None) => {}
        Msg::Picked(Some(path)) | Msg::FileDropped(path) => {
            if is_open(state) && !state.saving {
                return (read_file(state, path), Out::None);
            }
        }
        Msg::FileRead {
            generation,
            name,
            result,
        } => {
            if generation == state.generation {
                state.reading = false;
                match result {
                    Ok(bytes) => show_file(state, name, bytes),
                    Err(e) => state.error = Some(e),
                }
            }
        }
        Msg::PasteEdit(action) => {
            let action = clean_action(action);
            let pasted = matches!(
                action,
                text_editor::Action::Edit(text_editor::Edit::Paste(_))
            );
            state.paste.perform(action);
            if pasted {
                show_paste(state);
            }
        }
        Msg::PreviewPaste => show_paste(state),
        Msg::Encoding(choice) => apply_encoding(state, choice),
        Msg::Action(action) => state.action = action,
        Msg::Target(target) => {
            state.target_exists = state
                .job
                .as_ref()
                .is_some_and(|job| job.paths.translation(target.0.code).is_file());
            state.overwrite_confirmed = false;
            state.target = Some(target);
        }
        Msg::TargetBoxClosed => state.targets = target_box(),
        Msg::ConfirmOverwrite(confirmed) => state.overwrite_confirmed = confirmed,
        Msg::Translation(lang) => state.translation = Some(lang),
        Msg::Start => return (start(state), Out::None),
        Msg::Saved { generation, result } => {
            if generation == state.generation {
                return (Task::none(), saved(state, result));
            }
        }
    }
    (Task::none(), Out::None)
}

fn close(state: &mut State) {
    *state = State {
        generation: state.generation + 1,
        ..State::default()
    };
}

fn read_file(state: &mut State, path: PathBuf) -> Task<Msg> {
    if !has_subtitle_extension(&path) {
        state.error = Some(format!(
            "Choose a subtitle file: {}.",
            SUB_EXTENSIONS.join(", ")
        ));
        return Task::none();
    }
    state.generation += 1;
    state.reading = true;
    state.error = None;
    let generation = state.generation;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    Task::perform(
        on_thread(move || read_limited(&path, MAX_FILE_BYTES).map(Arc::new)),
        move |result| Msg::FileRead {
            generation,
            name: name.clone(),
            result,
        },
    )
}

fn show_file(state: &mut State, name: String, bytes: Arc<Vec<u8>>) {
    let decoded = subparse::decode(&bytes);
    state.bytes = Some(bytes);
    state.encoding = None;
    show_decoded(state, Some(name), decoded);
}

fn show_paste(state: &mut State) {
    let pasted = state.paste.text();
    if pasted.trim().is_empty() {
        state.error = Some("Paste subtitle text first.".into());
        return;
    }
    state.bytes = None;
    state.encoding = None;
    let decoded = Decoded {
        text: pasted,
        encoding: "UTF-8".into(),
        lossy: false,
    };
    show_decoded(state, None, decoded);
}

fn apply_encoding(state: &mut State, choice: EncodingChoice) {
    let Some(bytes) = state.bytes.clone() else {
        return;
    };
    let decoded = match choice.0 {
        None => Some(subparse::decode(&bytes)),
        Some(label) => subparse::decode_as(&bytes, label),
    };
    match decoded {
        Some(decoded) => {
            state.encoding = choice.0;
            let name = state.preview.as_ref().and_then(|p| p.source_name.clone());
            show_decoded(state, name, decoded);
        }
        None => state.error = Some("That encoding is not available.".into()),
    }
}

fn show_decoded(state: &mut State, name: Option<String>, decoded: Decoded) {
    let parsed = subparse::parse(&decoded.text);
    state.preview = Some(Preview::new(name, decoded, parsed));
    state.error = None;
}

fn start(state: &mut State) -> Task<Msg> {
    if state.saving || !can_start(state) {
        return Task::none();
    }
    let (Some(job), Some(preview)) = (&state.job, &state.preview) else {
        return Task::none();
    };
    let paths = job.paths.clone();
    let file = preview.to_import("");
    state.saving = true;
    state.error = None;
    let generation = state.generation;
    Task::perform(
        on_thread(move || save_import(&paths, file)),
        move |result| Msg::Saved { generation, result },
    )
}

fn saved(state: &mut State, result: Result<String, String>) -> Out {
    state.saving = false;
    let slug = match result {
        Ok(slug) => slug,
        Err(e) => {
            state.error = Some(e);
            return Out::None;
        }
    };
    let kind = state.job.as_ref().and_then(|job| {
        job_for(
            job.id,
            state.action,
            &slug,
            state.target.map(|t| t.0.code),
            state.translation.as_deref(),
        )
    });
    close(state);
    kind.map_or(Out::None, Out::Enqueue)
}

fn can_start(state: &State) -> bool {
    let has_cues = state.preview.as_ref().is_some_and(|p| !p.cues.is_empty());
    let replaces_unconfirmed =
        state.action == Action::Translate && state.target_exists && !state.overwrite_confirmed;
    has_cues
        && !replaces_unconfirmed
        && job_for(
            0,
            state.action,
            "",
            state.target.map(|t| t.0.code),
            state.translation.as_deref(),
        )
        .is_some()
}

// ------------------------------------------------------------------------- pure helpers

/// The background work for a saved import. `None` until the action has what it needs.
fn job_for(
    job: u64,
    action: Action,
    slug: &str,
    target: Option<&str>,
    translation: Option<&str>,
) -> Option<JobKind> {
    let import = CompareSource::Import {
        slug: slug.to_string(),
    };
    let compare = |a: CompareSource| JobKind::Compare {
        job,
        a,
        b: import.clone(),
        slug: slug.to_string(),
    };
    match action {
        Action::Translate => Some(JobKind::Translate {
            job,
            target: target?.to_string(),
            source: TranslateSource::Import {
                slug: slug.to_string(),
            },
        }),
        Action::CompareTranscript => Some(compare(CompareSource::Transcript)),
        Action::CompareTranslation => Some(compare(CompareSource::Translation {
            lang: translation?.to_string(),
        })),
    }
}

/// A file-name part from a source name: lowercase ASCII letters, digits and single dashes.
fn base_slug(source_name: Option<&str>) -> String {
    let Some(name) = source_name else {
        return "pasted".to_string();
    };
    let stem = Path::new(name)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut slug = String::new();
    for c in stem.chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c.to_ascii_lowercase());
        } else if !slug.is_empty() && !slug.ends_with('-') {
            slug.push('-');
        }
        if slug.len() >= 40 {
            break;
        }
    }
    let slug = slug.trim_end_matches('-');
    if slug.is_empty() {
        "import".to_string()
    } else {
        slug.to_string()
    }
}

/// `base`, or `base-2`, `base-3`, ... when that is taken.
fn unique_slug(base: &str, taken: impl Fn(&str) -> bool) -> String {
    if !taken(base) {
        return base.to_string();
    }
    (2u32..)
        .map(|n| format!("{base}-{n}"))
        .find(|slug| !taken(slug))
        .unwrap_or_else(|| base.to_string())
}

/// Writes the import under a new slug, so earlier imports and compares stay as they are.
fn save_import(paths: &ProjectPaths, mut file: ImportFile) -> Result<String, String> {
    let base = base_slug(file.source_name.as_deref());
    let slug = unique_slug(&base, |s| {
        paths.import(s).exists() || paths.compare(s).exists()
    });
    file.slug.clone_from(&slug);
    fsio::save_json(&paths.import(&slug), &file).map_err(|e| e.to_string())?;
    Ok(slug)
}

fn read_limited(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let name = path.display();
    let meta = std::fs::metadata(path).map_err(|e| format!("Could not read {name}: {e}"))?;
    if !meta.is_file() {
        return Err(format!("{name} is not a file."));
    }
    if meta.len() > limit {
        return Err(format!("{name} is too large to be a subtitle file."));
    }
    std::fs::read(path).map_err(|e| format!("Could not read {name}: {e}"))
}

fn has_subtitle_extension(path: &Path) -> bool {
    path.extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .is_some_and(|e| SUB_EXTENSIONS.contains(&e.as_str()))
}

/// The language of a `<stem>.<lang>.translation.json` file name.
fn translation_lang<'a>(file_name: &'a str, stem: &str) -> Option<&'a str> {
    let lang = file_name
        .strip_prefix(stem)?
        .strip_prefix('.')?
        .strip_suffix(".translation.json")?;
    (!lang.is_empty() && !lang.contains('.')).then_some(lang)
}

/// Languages this job already has a translation for, sorted.
fn existing_translations(paths: &ProjectPaths) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(&paths.dir) else {
        return Vec::new();
    };
    let mut langs: Vec<String> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name();
            translation_lang(&name.to_string_lossy(), &paths.stem).map(str::to_string)
        })
        // shown in a picker, and no language code holds these
        .filter(|lang| lang::clean_separators(lang) == *lang)
        .collect();
    langs.sort();
    langs
}

/// One cue as a single short line. Not yet safe to show: see [`content_text`].
fn preview_line(text: &str) -> String {
    let flat = text
        .split('\n')
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" / ");
    truncate_chars(&flat, PREVIEW_LINE_CHARS)
}

/// At most `max` characters, with an ellipsis when something was cut.
pub(crate) fn truncate_chars(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((end, _)) => format!("{}…", &text[..end]),
        None => text.to_string(),
    }
}

/// `m:ss`, or `h:mm:ss` past the first hour.
pub(crate) fn clock(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    let (h, m, s) = (total / 3600, (total / 60) % 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

fn format_name(format: SubFormat) -> &'static str {
    match format {
        SubFormat::Srt => "SRT",
        SubFormat::Vtt => "VTT",
        SubFormat::Ass => "ASS",
        SubFormat::Unknown => "Not recognised",
    }
}

// ------------------------------------------------------------------------- shared with review_ui

/// Pasted text is as untrusted as an imported file, so it is cleaned on its way into an editor.
pub(crate) fn clean_action(action: text_editor::Action) -> text_editor::Action {
    match action {
        text_editor::Action::Edit(text_editor::Edit::Paste(pasted)) => {
            let cleaned = lang::clean_separators(&pasted);
            text_editor::Action::Edit(text_editor::Edit::Paste(Arc::new(cleaned)))
        }
        other => other,
    }
}

/// Untrusted content text: cleaned, with default alignment in a container aligned by direction.
pub(crate) fn content_text<'a, M: 'a>(raw: &str, size: f32, color: Color) -> Element<'a, M> {
    let clean = lang::clean_separators(raw);
    let align = content_align(&clean);
    container(
        text(clean)
            .size(size)
            .color(color)
            .wrapping(Wrapping::WordOrGlyph),
    )
    .width(Length::Fill)
    .align_x(align)
    .into()
}

/// Runs blocking file work on its own thread, like `check_files` in `main.rs`.
pub(crate) async fn on_thread<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    let (tx, rx) = iced::futures::channel::oneshot::channel();
    std::thread::spawn(move || {
        let _ = tx.send(work());
    });
    rx.await
        .unwrap_or_else(|_| Err("The background task stopped unexpectedly.".to_string()))
}

async fn pick_file() -> Option<PathBuf> {
    rfd::AsyncFileDialog::new()
        .set_title("Import subtitles")
        .add_filter("Subtitles", SUB_EXTENSIONS)
        .pick_file()
        .await
        .map(|f| f.path().to_path_buf())
}

// ------------------------------------------------------------------------- view

pub fn view(state: &State, t: Tokens) -> Element<'_, Msg> {
    let header = row![
        text("Import subtitles").size(18).color(t.text),
        space::horizontal(),
        button(icons::icon(icons::X, 14.0, t.muted))
            .padding(6)
            .style(theme::ghost(t))
            .on_press_maybe((!state.saving).then_some(Msg::Close)),
    ]
    .align_y(Alignment::Center);

    let mut body = Column::new().spacing(14).push(source_view(state, t));
    if let Some(preview) = &state.preview {
        body = body.push(preview_view(state, preview, t));
        if !preview.cues.is_empty() {
            body = body.push(action_view(state, t));
        }
    }

    let mut dialog = Column::new()
        .spacing(14)
        .push(header)
        .push(scrollable(body.padding(iced::Padding::ZERO.right(12))).height(Length::Shrink));
    if let Some(error) = &state.error {
        dialog = dialog.push(content_text(error, 13.0, t.bad));
    }
    dialog = dialog.push(footer(state, t));

    container(dialog)
        .padding(20)
        .max_width(720)
        .style(theme::dialog(t))
        .into()
}

fn source_view(state: &State, t: Tokens) -> Element<'_, Msg> {
    let pick = button(
        row![
            icons::icon(icons::FOLDER, 15.0, t.text),
            text("Choose file").size(13)
        ]
        .spacing(6)
        .align_y(Alignment::Center),
    )
    .padding([8, 12])
    .style(theme::outline_btn(t))
    .on_press_maybe((!state.saving).then_some(Msg::PickFile));

    let hint = if state.reading {
        "Reading the file…"
    } else {
        "SRT, VTT or ASS. You can also drop a file here, or paste below."
    };

    column![
        row![pick, text(hint).size(12).color(t.muted)]
            .spacing(10)
            .align_y(Alignment::Center),
        text_editor(&state.paste)
            .placeholder("Paste subtitle text")
            .on_action(Msg::PasteEdit)
            .height(Length::Fixed(110.0))
            .size(13)
            .padding(9),
        button(text("Preview pasted text").size(13))
            .padding([6, 12])
            .style(theme::ghost(t))
            .on_press_maybe((!state.saving).then_some(Msg::PreviewPaste)),
    ]
    .spacing(8)
    .into()
}

fn preview_view<'a>(state: &'a State, preview: &'a Preview, t: Tokens) -> Element<'a, Msg> {
    let chip = |label: String, color: Color| {
        container(text(label).size(12).color(color))
            .padding([2, 9])
            .style(theme::chip(color))
    };
    let count = match preview.cues.len() {
        1 => "1 cue".to_string(),
        n => format!("{n} cues"),
    };
    let count_color = if preview.cues.is_empty() { t.bad } else { t.ok };
    let format_color = if preview.format == SubFormat::Unknown {
        t.bad
    } else {
        t.accent
    };

    let mut facts = row![
        chip(format_name(preview.format).to_string(), format_color),
        chip(count, count_color),
    ]
    .spacing(8)
    .align_y(Alignment::Center);

    if state.bytes.is_some() {
        let mut options = vec![EncodingChoice(None)];
        options.extend(ENCODINGS.iter().map(|e| EncodingChoice(Some(e))));
        facts = facts.push(space::horizontal());
        facts = facts.push(
            text(format!("Read as {}", preview.encoding))
                .size(12)
                .color(t.muted),
        );
        facts = facts.push(
            pick_list(options, Some(EncodingChoice(state.encoding)), Msg::Encoding)
                .text_size(12)
                .padding([4, 8])
                .style(theme::pick(t)),
        );
    } else {
        facts = facts.push(chip("Pasted".to_string(), t.muted));
    }

    let mut col = Column::new().spacing(8).push(facts);
    if let Some(name) = &preview.source_name {
        col = col.push(content_text(name, 12.0, t.muted));
    }

    if preview.cues.is_empty() {
        col = col.push(
            text("No subtitle cues found in this text.")
                .size(13)
                .color(t.bad),
        );
    }
    for w in preview.warnings.iter().take(PREVIEW_WARNINGS) {
        let line = match w.line {
            Some(n) => format!("Line {n}: {}", w.message),
            None => w.message.clone(),
        };
        col = col.push(content_text(&line, 12.0, t.warn));
    }
    if preview.warnings.len() > PREVIEW_WARNINGS {
        let more = preview.warnings.len() - PREVIEW_WARNINGS;
        col = col.push(
            text(format!("and {more} more warnings"))
                .size(12)
                .color(t.warn),
        );
    }
    for cue in preview.cues.iter().take(PREVIEW_CUES) {
        col = col.push(
            row![
                text(clock(cue.start)).size(12).color(t.faint),
                content_text(&preview_line(&cue.text), 13.0, t.text),
            ]
            .spacing(10),
        );
    }

    container(col)
        .padding(12)
        .width(Length::Fill)
        .style(theme::card(t))
        .into()
}

fn action_view(state: &State, t: Tokens) -> Element<'_, Msg> {
    let seg = |label: &'static str, action: Action, enabled: bool| {
        button(text(label).size(13))
            .padding([7, 12])
            .style(theme::segment(t, state.action == action))
            .on_press_maybe(enabled.then_some(Msg::Action(action)))
    };
    let choices = row![
        seg("Translate", Action::Translate, true),
        seg("Compare with transcript", Action::CompareTranscript, true),
        seg(
            "Compare with translation",
            Action::CompareTranslation,
            !state.translations.is_empty()
        ),
    ]
    .spacing(8);

    let detail: Element<'_, Msg> = match state.action {
        Action::Translate => {
            let picker = combo_box(
                &state.targets,
                "Target language (type to search)",
                state.target.as_ref(),
                Msg::Target,
            )
            .size(13)
            .padding(8)
            .on_close(Msg::TargetBoxClosed)
            .input_style(theme::input(t))
            .menu_style(theme::search_menu(t));
            if state.target_exists {
                column![
                    picker,
                    text(
                        "This result already has a translation to that language. Translating this \
                         file replaces it and its saved files. Your edits stay only on lines with \
                         the same source text."
                    )
                    .size(12)
                    .color(t.warn),
                    button(text("Replace the existing translation").size(13))
                        .padding([7, 12])
                        .style(theme::segment(t, state.overwrite_confirmed))
                        .on_press(Msg::ConfirmOverwrite(!state.overwrite_confirmed)),
                ]
                .spacing(8)
                .into()
            } else {
                picker.into()
            }
        }
        Action::CompareTranscript => {
            text("The AI picks the better line for each moment. You review its picks before anything is saved.")
                .size(12)
                .color(t.muted)
                .into()
        }
        Action::CompareTranslation => pick_list(
            state.translations.as_slice(),
            state.translation.as_ref(),
            Msg::Translation,
        )
        .text_size(13)
        .padding(8)
        .style(theme::pick(t))
        .into(),
    };

    column![choices, detail].spacing(10).into()
}

fn footer(state: &State, t: Tokens) -> Element<'_, Msg> {
    let label = match state.action {
        Action::Translate => "Add to queue: translate",
        Action::CompareTranscript | Action::CompareTranslation => "Add to queue: compare",
    };
    let cancel = button(text("Cancel").size(13))
        .padding([8, 12])
        .style(theme::outline_btn(t))
        .on_press_maybe((!state.saving).then_some(Msg::Close));
    let go = button(text(label).size(13))
        .padding([8, 12])
        .style(theme::primary(t))
        .on_press_maybe((can_start(state) && !state.saving).then_some(Msg::Start));
    row![space::horizontal(), cancel, go]
        .spacing(10)
        .align_y(Alignment::Center)
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(id: u64, text: &str) -> TimedText {
        TimedText {
            id,
            start: id as f64,
            end: id as f64 + 1.0,
            text: text.into(),
        }
    }

    fn preview(lossy: bool) -> Preview {
        Preview::new(
            Some("Movie.fa.srt".into()),
            Decoded {
                text: String::new(),
                encoding: "windows-1256".into(),
                lossy,
            },
            Parsed {
                format: SubFormat::Srt,
                cues: vec![cue(1, "hello")],
                warnings: vec![ImportWarning {
                    line: Some(4),
                    message: "bad time".into(),
                }],
            },
        )
    }

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> TestDir {
            let dir = std::env::temp_dir().join(format!(
                "vipr-import-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            TestDir(dir)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn slug_is_safe_for_a_file_name() {
        assert_eq!(base_slug(None), "pasted");
        assert_eq!(
            base_slug(Some("My Movie (2021).EN.srt")),
            "my-movie-2021-en"
        );
        assert_eq!(base_slug(Some("..\\..\\evil.srt")), "evil");
        assert_eq!(base_slug(Some("زیرنویس.srt")), "import");
        assert!(base_slug(Some(&format!("{}.srt", "a".repeat(200)))).len() <= 40);
    }

    #[test]
    fn slug_never_reuses_a_taken_name() {
        assert_eq!(unique_slug("a", |_| false), "a");
        assert_eq!(unique_slug("a", |s| s == "a"), "a-2");
        assert_eq!(unique_slug("a", |s| s == "a" || s == "a-2"), "a-3");
    }

    #[test]
    fn saving_keeps_earlier_imports() {
        let dir = TestDir::new("save");
        let paths = ProjectPaths::new(&dir.0, "rec (mp4)");
        let p = preview(false);
        let first = save_import(&paths, p.to_import("")).unwrap();
        let second = save_import(&paths, p.to_import("")).unwrap();
        assert_eq!(first, "movie-fa");
        assert_eq!(second, "movie-fa-2");
        let back: ImportFile = fsio::load_json(&paths.import(&second)).unwrap();
        assert_eq!(back.slug, second);
        assert_eq!(back.encoding, "windows-1256");
        assert_eq!(back.source_name.as_deref(), Some("Movie.fa.srt"));
        assert_eq!(back.cues, p.cues);
    }

    #[test]
    fn a_compare_with_that_slug_also_takes_the_name() {
        let dir = TestDir::new("taken");
        let paths = ProjectPaths::new(&dir.0, "rec");
        std::fs::write(paths.compare("movie-fa"), "{}").unwrap();
        let slug = save_import(&paths, preview(false).to_import("")).unwrap();
        assert_eq!(slug, "movie-fa-2");
    }

    #[test]
    fn lossy_decoding_becomes_a_warning_that_is_saved() {
        assert_eq!(preview(false).warnings.len(), 1);
        let lossy = preview(true);
        assert_eq!(lossy.warnings.len(), 2);
        assert!(lossy.warnings[0].message.contains("replaced"));
        assert_eq!(lossy.to_import("s").warnings, lossy.warnings);
    }

    #[test]
    fn each_action_asks_for_the_right_job() {
        assert_eq!(job_for(7, Action::Translate, "s", None, Some("fa")), None);
        assert_eq!(
            job_for(7, Action::Translate, "s", Some("de"), None),
            Some(JobKind::Translate {
                job: 7,
                target: "de".into(),
                source: TranslateSource::Import { slug: "s".into() },
            })
        );
        assert_eq!(
            job_for(7, Action::CompareTranscript, "s", None, None),
            Some(JobKind::Compare {
                job: 7,
                a: CompareSource::Transcript,
                b: CompareSource::Import { slug: "s".into() },
                slug: "s".into(),
            })
        );
        assert_eq!(
            job_for(7, Action::CompareTranslation, "s", Some("de"), None),
            None
        );
        assert_eq!(
            job_for(7, Action::CompareTranslation, "s", None, Some("fa")),
            Some(JobKind::Compare {
                job: 7,
                a: CompareSource::Translation { lang: "fa".into() },
                b: CompareSource::Import { slug: "s".into() },
                slug: "s".into(),
            })
        );
    }

    #[test]
    fn start_needs_cues_and_a_complete_choice() {
        let mut state = State::default();
        assert!(!can_start(&state));
        let mut empty = preview(false);
        empty.cues.clear();
        state.preview = Some(empty);
        state.action = Action::CompareTranscript;
        assert!(!can_start(&state), "no cues");
        state.preview = Some(preview(false));
        assert!(can_start(&state));
        state.action = Action::Translate;
        assert!(!can_start(&state), "no target yet");
        state.action = Action::CompareTranslation;
        assert!(!can_start(&state), "no translation on disk");
    }

    #[test]
    fn replacing_a_translation_needs_a_confirmation() {
        let dir = TestDir::new("replace");
        let paths = ProjectPaths::new(&dir.0, "a");
        std::fs::write(paths.translation("fa"), "{}").unwrap();
        let mut state = State {
            job: Some(JobRef {
                id: 1,
                media: PathBuf::from("a.mp4"),
                paths,
            }),
            preview: Some(preview(false)),
            ..State::default()
        };
        let target = |code: &str| Target(lang::language(code).unwrap());

        let _ = update(&mut state, Msg::Target(target("de")));
        assert!(!state.target_exists);
        assert!(can_start(&state), "nothing to replace");

        let _ = update(&mut state, Msg::Target(target("fa")));
        assert!(state.target_exists);
        assert!(!can_start(&state), "would replace it unasked");
        let _ = update(&mut state, Msg::ConfirmOverwrite(true));
        assert!(can_start(&state));

        // the confirmation was for that language only
        let _ = update(&mut state, Msg::Target(target("de")));
        let _ = update(&mut state, Msg::Target(target("fa")));
        assert!(!can_start(&state));
        // a compare doesn't touch the translation
        state.action = Action::CompareTranscript;
        assert!(can_start(&state));
    }

    #[test]
    fn saved_import_queues_the_work_and_closes() {
        let mut state = State {
            job: Some(JobRef {
                id: 3,
                media: PathBuf::from("a.mp4"),
                paths: ProjectPaths::new("out", "a"),
            }),
            action: Action::CompareTranscript,
            preview: Some(preview(false)),
            saving: true,
            ..State::default()
        };
        let out = saved(&mut state, Ok("s".into()));
        assert!(matches!(out, Out::Enqueue(JobKind::Compare { job: 3, .. })));
        assert!(!is_open(&state));
    }

    #[test]
    fn a_failed_save_keeps_the_dialog_and_queues_nothing() {
        let mut state = State {
            job: Some(JobRef {
                id: 3,
                media: PathBuf::from("a.mp4"),
                paths: ProjectPaths::new("out", "a"),
            }),
            action: Action::CompareTranscript,
            saving: true,
            ..State::default()
        };
        let out = saved(&mut state, Err("disk full".into()));
        assert!(matches!(out, Out::None));
        assert!(is_open(&state));
        assert!(!state.saving);
        assert_eq!(state.error.as_deref(), Some("disk full"));
    }

    #[test]
    fn translations_are_found_by_their_exact_file_name() {
        assert_eq!(translation_lang("a.fa.translation.json", "a"), Some("fa"));
        assert_eq!(
            translation_lang("a (mp3).pt-BR.translation.json", "a (mp3)"),
            Some("pt-BR")
        );
        assert_eq!(translation_lang("a.translation.json", "a"), None);
        assert_eq!(translation_lang("a..translation.json", "a"), None);
        assert_eq!(translation_lang("b.fa.translation.json", "a"), None);
        assert_eq!(translation_lang("a.fa.srt", "a"), None);

        let dir = TestDir::new("langs");
        for name in ["a.fa.translation.json", "a.de.translation.json", "a.json"] {
            std::fs::write(dir.0.join(name), "{}").unwrap();
        }
        let paths = ProjectPaths::new(&dir.0, "a");
        assert_eq!(existing_translations(&paths), ["de", "fa"]);
        assert!(existing_translations(&ProjectPaths::new("no/such/folder", "a")).is_empty());
    }

    #[test]
    fn only_small_subtitle_files_are_read() {
        assert!(has_subtitle_extension(Path::new("a.SRT")));
        assert!(has_subtitle_extension(Path::new("a.ass")));
        assert!(!has_subtitle_extension(Path::new("a.mp4")));
        assert!(!has_subtitle_extension(Path::new("srt")));

        let dir = TestDir::new("read");
        let path = dir.0.join("a.srt");
        std::fs::write(&path, b"12345").unwrap();
        assert_eq!(read_limited(&path, 5).unwrap(), b"12345");
        assert!(read_limited(&path, 4).unwrap_err().contains("too large"));
        assert!(read_limited(&dir.0, 5).unwrap_err().contains("not a file"));
        assert!(read_limited(&dir.0.join("gone.srt"), 5).is_err());
    }

    #[test]
    fn preview_lines_are_short_and_flat() {
        assert_eq!(preview_line("one\n\n two \n"), "one / two");
        let long = preview_line(&"س".repeat(500));
        assert_eq!(long.chars().count(), PREVIEW_LINE_CHARS + 1);
        assert!(long.ends_with('…'));
        assert_eq!(truncate_chars("abc", 3), "abc");
    }

    #[test]
    fn every_offered_encoding_is_one_the_parser_knows() {
        for label in ENCODINGS {
            assert!(subparse::decode_as(b"abc", label).is_some(), "{label}");
        }
    }

    #[test]
    fn a_file_shows_a_preview_and_the_encoding_can_be_changed() {
        // "سلام" in windows-1256
        let mut bytes = b"1\r\n00:00:01,000 --> 00:00:02,000\r\n".to_vec();
        bytes.extend([0xD3, 0xE1, 0xC7, 0xE3]);
        bytes.extend(b"\r\n\r\n");
        let mut state = State::default();
        show_file(&mut state, "a.srt".into(), Arc::new(bytes));
        assert_eq!(state.preview.as_ref().unwrap().format, SubFormat::Srt);
        assert_eq!(state.preview.as_ref().unwrap().cues.len(), 1);

        apply_encoding(&mut state, EncodingChoice(Some("windows-1256")));
        let preview = state.preview.as_ref().unwrap();
        assert_eq!(preview.encoding, "windows-1256");
        assert_eq!(preview.cues[0].text, "سلام");
        assert_eq!(preview.source_name.as_deref(), Some("a.srt"));
        assert_eq!(state.encoding, Some("windows-1256"));
    }

    #[test]
    fn pasted_text_previews_without_separators_that_crash_the_renderer() {
        let mut state = State::default();
        let srt = "1\n00:00:01,000 --> 00:00:02,000\nhello\u{2029}سلام\n\n";
        let _ = update(
            &mut state,
            Msg::PasteEdit(text_editor::Action::Edit(text_editor::Edit::Paste(
                Arc::new(srt.to_string()),
            ))),
        );
        assert!(!state.paste.text().contains('\u{2029}'));
        let preview = state.preview.as_ref().unwrap();
        assert_eq!(preview.source_name, None);
        assert_eq!(preview.cues.len(), 1);
        assert!(!preview.cues[0].text.contains('\u{2029}'));
    }

    #[test]
    fn empty_paste_asks_for_text_and_keeps_the_preview() {
        let mut state = State {
            preview: Some(preview(false)),
            ..State::default()
        };
        let _ = update(&mut state, Msg::PreviewPaste);
        assert!(state.error.is_some());
        assert!(state.preview.is_some());
    }

    #[test]
    fn text_that_is_not_subtitles_is_not_recognised_and_cannot_start() {
        let mut state = State {
            action: Action::CompareTranscript,
            ..State::default()
        };
        let _ = update(
            &mut state,
            Msg::PasteEdit(text_editor::Action::Edit(text_editor::Edit::Paste(
                Arc::new("just a note, no timing lines".to_string()),
            ))),
        );
        let preview = state.preview.as_ref().unwrap();
        assert_eq!(preview.format, SubFormat::Unknown);
        assert_eq!(format_name(preview.format), "Not recognised");
        assert!(!preview.warnings.is_empty());
        assert!(!can_start(&state));
    }

    #[test]
    fn bytes_that_do_not_fit_warn_in_the_preview() {
        let mut bytes = b"1\n00:00:01,000 --> 00:00:02,000\nhi ".to_vec();
        bytes.extend([0xFF, 0xFE, 0x80]);
        bytes.extend(b"\n\n");
        let mut state = State::default();
        show_file(&mut state, "a.srt".into(), Arc::new(bytes));
        apply_encoding(&mut state, EncodingChoice(Some("UTF-8")));
        let preview = state.preview.as_ref().unwrap();
        assert!(preview.warnings[0].message.contains("replaced"));
        assert_eq!(preview.cues.len(), 1);
    }

    #[test]
    fn clock_reads_like_a_player() {
        assert_eq!(clock(0.0), "0:00");
        assert_eq!(clock(-3.0), "0:00");
        assert_eq!(clock(65.4), "1:05");
        assert_eq!(clock(3661.0), "1:01:01");
    }
}
