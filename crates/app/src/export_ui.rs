//! The export dialog: what to make, in which format, how, and with which subtitles. A summary line says what will
//! come out before anything runs.

use std::path::PathBuf;

use engine::cuts::{self, Cut, CutsFile};
use engine::style::{Align, Place, SubtitleStyle};
use iced::widget::{
    Column, button, column, container, image, row, scrollable, slider, space, text,
};
use iced::{Alignment, Element, Length, Task};

use crate::export::{
    AudioFormat, Codec, Container, Encoder, ExportOptions, Kind, Quality, Resolution,
    SubtitleFormat, SubtitleMode,
};
use crate::icons;
use crate::out::{JobKind, JobRef, Out};
use crate::theme::{self, Tokens};

#[derive(Debug, Default)]
pub struct State {
    /// `None` while closed.
    job: Option<JobRef>,
    options: ExportOptions,
    /// The result's translations, as `(code, name)`.
    languages: Vec<(String, String)>,
    cuts: Vec<Cut>,
    /// The recording's length in seconds.
    duration: f64,
    /// An NVIDIA graphics card was found.
    gpu: bool,
    error: Option<String>,
    /// The last preview frame, the moment it shows, and the choices it was made with.
    preview: Option<(image::Handle, f64, ExportOptions)>,
    previewing: bool,
    preview_error: Option<String>,
    /// Set when exporting a combination: its name, and its recordings in order. `job` is then the first.
    sequence: Option<(String, Vec<JobRef>)>,
}

#[derive(Debug, Clone)]
pub enum Msg {
    Close,
    Kind(Kind),
    Container(Container),
    Codec(Codec),
    Encoder(Encoder),
    Quality(Quality),
    Resolution(Resolution),
    Subtitles(SubtitleMode),
    Language(Option<String>),
    Font(String),
    Size(f32),
    Bold(bool),
    Italic(bool),
    Color(u32),
    OutlineColor(u32),
    Outline(f32),
    Shadow(f32),
    Box(bool),
    BoxColor(u32),
    BoxOpacity(f32),
    Place(Place),
    Align(Align),
    Margin(f32),
    ResetStyle,
    /// Draw a frame with these choices.
    Preview,
    PreviewReady(Result<(Vec<u8>, f64), String>),
    ApplyCuts(bool),
    SubtitleFormat(SubtitleFormat),
    AudioFormat(AudioFormat),
    /// Ask where to save.
    Export,
    Chosen(Option<PathBuf>),
}

/// Opens the dialog for `job`. Choices made earlier stay, except the language, which follows what is on show.
/// `style` is the one used last.
pub fn open(
    state: &mut State,
    job: JobRef,
    (lang, style): (Option<String>, SubtitleStyle),
    duration: f64,
    gpu: bool,
) {
    let cuts = std::fs::read_to_string(job.paths.cuts())
        .ok()
        .and_then(|text| serde_json::from_str::<CutsFile>(&text).ok())
        .map(|file| cuts::normalize(&file.cuts, duration))
        .unwrap_or_default();
    state.languages = crate::player::translation_languages(&job.paths);
    state.options.lang = lang.filter(|code| state.languages.iter().any(|(c, _)| c == code));
    state.cuts = cuts;
    state.duration = duration;
    state.gpu = gpu;
    state.error = None;
    state.options.style = style;
    state.preview = None;
    state.previewing = false;
    state.preview_error = None;
    if !gpu && state.options.encoder == Encoder::Gpu {
        state.options.encoder = Encoder::Auto;
    }
    state.sequence = None;
    state.job = Some(job);
}

/// Opens the dialog for a combination: `items` one after another. The preview shows the first; the
/// languages are the first one's (a recording without the chosen translation shows its own words), and the
/// cuts are all of theirs, laid end to end, for the summary. `durations` are the recordings' lengths.
pub fn open_sequence(
    state: &mut State,
    name: String,
    items: Vec<JobRef>,
    durations: &[f64],
    (lang, style): (Option<String>, SubtitleStyle),
    gpu: bool,
) {
    let Some(first) = items.first().cloned() else {
        return;
    };
    let total: f64 = durations.iter().sum();
    open(state, first, (lang, style), total, gpu);
    let mut all = Vec::new();
    let mut offset = 0.0;
    for (item, duration) in items.iter().zip(durations) {
        let own = std::fs::read_to_string(item.paths.cuts())
            .ok()
            .and_then(|text| serde_json::from_str::<CutsFile>(&text).ok())
            .map(|file| cuts::normalize(&file.cuts, *duration))
            .unwrap_or_default();
        all.extend(own.into_iter().map(|c| Cut {
            start: c.start + offset,
            end: c.end + offset,
        }));
        offset += duration;
    }
    state.cuts = all;
    state.sequence = Some((name, items));
}

pub fn is_open(state: &State) -> bool {
    state.job.is_some()
}

pub fn update(state: &mut State, msg: Msg) -> (Task<Msg>, Out) {
    let o = &mut state.options;
    match msg {
        Msg::Close => {
            state.job = None;
            state.sequence = None;
        }
        Msg::Kind(k) => o.kind = k,
        Msg::Container(c) => o.container = c,
        Msg::Codec(c) => o.codec = c,
        Msg::Encoder(e) => o.encoder = e,
        Msg::Quality(q) => o.quality = q,
        Msg::Resolution(r) => o.resolution = r,
        Msg::Subtitles(m) => o.subtitles = m,
        Msg::Language(l) => o.lang = l,
        Msg::Font(f) => o.style.font = f,
        Msg::Size(v) => o.style.size = v.round() as u32,
        Msg::Bold(b) => o.style.bold = b,
        Msg::Italic(i) => o.style.italic = i,
        Msg::Color(rgb) => o.style.color = rgb,
        Msg::OutlineColor(rgb) => o.style.outline_color = rgb,
        Msg::Outline(v) => o.style.outline = (v * 2.0).round() / 2.0,
        Msg::Shadow(v) => o.style.shadow = (v * 2.0).round() / 2.0,
        Msg::Box(on) => o.style.box_on = on,
        Msg::BoxColor(rgb) => o.style.box_color = rgb,
        Msg::BoxOpacity(v) => o.style.box_opacity = v.round().clamp(0.0, 100.0) as u8,
        Msg::Place(pl) => o.style.place = pl,
        Msg::Align(a) => o.style.align = a,
        Msg::Margin(v) => o.style.margin = v.round() as u32,
        Msg::ResetStyle => o.style = SubtitleStyle::default(),
        Msg::Preview => {
            let Some(job) = state.job.clone() else {
                return (Task::none(), Out::None);
            };
            let options = state.options.clone();
            state.previewing = true;
            state.preview_error = None;
            return (
                Task::perform(
                    crate::import_ui::on_thread(move || {
                        crate::jobs::export_preview(&job, &options)
                    }),
                    Msg::PreviewReady,
                ),
                Out::None,
            );
        }
        Msg::PreviewReady(result) => {
            state.previewing = false;
            match result {
                Ok((png, at)) => {
                    state.preview =
                        Some((image::Handle::from_bytes(png), at, state.options.clone()));
                }
                Err(e) => state.preview_error = Some(e),
            }
        }
        Msg::ApplyCuts(yes) => o.apply_cuts = yes,
        Msg::SubtitleFormat(f) => o.subtitle_format = f,
        Msg::AudioFormat(f) => o.audio_format = f,
        Msg::Export => {
            let Some(job) = &state.job else {
                return (Task::none(), Out::None);
            };
            let dir = job.media.parent().map(std::path::Path::to_path_buf);
            let ext = state.options.extension();
            let stem = match &state.sequence {
                Some((name, _)) => name.clone(),
                None => job.paths.stem.clone(),
            };
            let name = format!(
                "{}.{ext}",
                file_name(&stem, &state.options, &state.languages)
            );
            return (Task::perform(pick(dir, name, ext), Msg::Chosen), Out::None);
        }
        Msg::Chosen(None) => {}
        Msg::Chosen(Some(path)) => {
            let Some(job) = &state.job else {
                return (Task::none(), Out::None);
            };
            let path = path.with_extension(state.options.extension());
            let replaces_a_recording = match &state.sequence {
                Some((_, items)) => items.iter().any(|i| i.media == path),
                None => path == job.media,
            };
            if replaces_a_recording {
                state.error = Some(
                    "The export can't replace the recording itself. Choose another name.".into(),
                );
                return (Task::none(), Out::None);
            }
            let kind = match state.sequence.take() {
                Some((name, items)) => JobKind::ExportSequence {
                    job: job.id,
                    name,
                    items,
                    options: state.options.clone(),
                    out: path,
                },
                None => JobKind::Export {
                    job: job.id,
                    options: state.options.clone(),
                    out: path,
                },
            };
            state.job = None;
            return (Task::none(), Out::Enqueue(kind));
        }
    }
    state.options.fix();
    (Task::none(), Out::None)
}

/// A name to start the save dialog with, without its ending.
fn file_name(stem: &str, o: &ExportOptions, languages: &[(String, String)]) -> String {
    let language = language_name(o, languages);
    // a combination's name is typed by the user, so it must stay a plain file name
    let stem: String = stem
        .chars()
        .map(|c| {
            if c.is_control() || r#"<>:"/\|?*"#.contains(c) {
                '_'
            } else {
                c
            }
        })
        .collect();
    let stem = stem.trim();
    match o.kind {
        Kind::Video if o.subtitles == SubtitleMode::Off => format!("{stem} (export)"),
        Kind::Video => format!("{stem} - {language} subtitles"),
        Kind::Subtitles => format!("{stem} - {language}"),
        Kind::Audio => format!("{stem} (sound)"),
    }
}

fn language_name(o: &ExportOptions, languages: &[(String, String)]) -> String {
    o.lang
        .as_ref()
        .and_then(|code| languages.iter().find(|(c, _)| c == code))
        .map_or_else(|| "original".to_string(), |(_, name)| name.clone())
}

async fn pick(dir: Option<PathBuf>, name: String, ext: &'static str) -> Option<PathBuf> {
    let mut dialog = rfd::AsyncFileDialog::new()
        .set_title("Export")
        .set_file_name(name)
        .add_filter(ext.to_uppercase(), &[ext]);
    if let Some(dir) = dir {
        dialog = dialog.set_directory(dir);
    }
    dialog.save_file().await.map(|f| f.path().to_path_buf())
}

/// What will come out, in one line.
fn summary(state: &State) -> String {
    let o = &state.options;
    let language = language_name(o, &state.languages);
    let mut parts: Vec<String> = match o.kind {
        Kind::Video => {
            let made_by = match (o.encoder, o.codec.gpu_encoder().is_some() && state.gpu) {
                (Encoder::Cpu, _) | (Encoder::Auto, false) => "on the processor",
                (Encoder::Gpu, _) => "on the graphics card",
                (Encoder::Auto, true) => "on the graphics card, or the processor if it can't",
            };
            let mut v = vec![format!(
                "{} video in {}, {made_by}",
                o.container.label(),
                o.codec.label()
            )];
            v.push(match o.subtitles {
                SubtitleMode::Burn => format!("{language} subtitles on the picture"),
                SubtitleMode::Track => format!("{language} subtitles as a track"),
                SubtitleMode::Off => "no subtitles".to_string(),
            });
            v
        }
        Kind::Subtitles => vec![format!(
            "{language} subtitles as a .{} file",
            o.subtitle_format.extension()
        )],
        Kind::Audio => vec![format!("the sound as a .{} file", o.extension())],
    };
    let removed = cuts::removed(&state.cuts);
    if !state.cuts.is_empty() && o.apply_cuts {
        parts.push(format!(
            "{} cut {} left out, {} long",
            state.cuts.len(),
            if state.cuts.len() == 1 {
                "part"
            } else {
                "parts"
            },
            engine::transcript::clock((state.duration - removed).max(0.0))
        ));
    }
    let line = parts.join(" - ");
    let mut chars = line.chars();
    chars.next().map_or_else(String::new, |first| {
        first.to_uppercase().chain(chars).collect()
    })
}

/// Something worth knowing about the current choices, if any.
fn note(state: &State) -> Option<&'static str> {
    let o = &state.options;
    if o.kind != Kind::Video {
        return None;
    }
    match (o.codec, o.encoder) {
        (Codec::Av1, Encoder::Auto | Encoder::Gpu) => Some(
            "AV1 on the graphics card needs an NVIDIA RTX 40 or newer. On other cards Auto makes it on the processor, \
             which is slow.",
        ),
        (Codec::Av1, Encoder::Cpu) => {
            Some("AV1 files are small, but making them on the processor is slow.")
        }
        (Codec::Vp9, _) => Some("VP9 is made on the processor only, which is slower than H.264."),
        (Codec::H265, _) => {
            Some("H.265 files are smaller than H.264, but some older players can't open them.")
        }
        (Codec::H264, _) if o.subtitles == SubtitleMode::Track => Some(
            "Players can turn a subtitle track on and off. Some web sites ignore it: drawn subtitles always show.",
        ),
        (Codec::H264, _) => None,
    }
}

// ------------------------------------------------------------------------- text style

/// Fonts offered, with the file that has to be there for each. Each covers Latin letters; Tahoma, Segoe UI, Arial and
/// Times New Roman also cover Persian and Arabic.
const FONTS: [(&str, &str); 10] = [
    ("Arial", "arial.ttf"),
    ("Segoe UI", "segoeui.ttf"),
    ("Tahoma", "tahoma.ttf"),
    ("Verdana", "verdana.ttf"),
    ("Calibri", "calibri.ttf"),
    ("Trebuchet MS", "trebuc.ttf"),
    ("Georgia", "georgia.ttf"),
    ("Times New Roman", "times.ttf"),
    ("Consolas", "consola.ttf"),
    ("Impact", "impact.ttf"),
];

/// The fonts this computer has. Elsewhere than Windows all are offered, and the system finds a near one.
fn fonts() -> Vec<&'static str> {
    let Some(dir) = std::env::var_os("WINDIR").map(|w| PathBuf::from(w).join("Fonts")) else {
        return FONTS.iter().map(|(name, _)| *name).collect();
    };
    FONTS
        .iter()
        .filter(|(_, file)| dir.join(file).is_file())
        .map(|(name, _)| *name)
        .collect()
}

const TEXT_COLORS: [(&str, u32); 7] = [
    ("White", 0xFFFFFF),
    ("Yellow", 0xFFE14D),
    ("Cyan", 0x66E0FF),
    ("Green", 0x7CFC8A),
    ("Orange", 0xFFA040),
    ("Pink", 0xFF7EC8),
    ("Black", 0x000000),
];
const EDGE_COLORS: [(&str, u32); 3] = [
    ("Black", 0x000000),
    ("Dark grey", 0x404040),
    ("White", 0xFFFFFF),
];
const BOX_COLORS: [(&str, u32); 4] = [
    ("Black", 0x000000),
    ("Dark grey", 0x303030),
    ("Dark blue", 0x0B3D91),
    ("White", 0xFFFFFF),
];

fn rgb(c: u32) -> iced::Color {
    iced::Color::from_rgb8((c >> 16) as u8, (c >> 8) as u8, c as u8)
}

/// A colour to pick: a small swatch, outlined when chosen.
fn swatch<'a>(name: &'a str, color: u32, chosen: bool, msg: Msg, t: Tokens) -> Element<'a, Msg> {
    let fill = container(space())
        .width(18)
        .height(18)
        .style(theme::rounded_fill(rgb(color), 3.0));
    iced::widget::tooltip(
        button(fill)
            .padding(4)
            .style(theme::segment(t, chosen))
            .on_press(msg),
        container(text(name).size(12).color(t.text))
            .padding([4, 8])
            .style(theme::toast(t)),
        iced::widget::tooltip::Position::Top,
    )
    .into()
}

/// A slider with its value next to it.
fn amount<'a>(
    range: std::ops::RangeInclusive<f32>,
    value: f32,
    step: f32,
    shown: String,
    msg: fn(f32) -> Msg,
    t: Tokens,
) -> Vec<Element<'a, Msg>> {
    vec![
        slider(range, value, msg)
            .step(step)
            .width(Length::Fixed(220.0))
            .into(),
        text(shown).size(12).color(t.muted).into(),
    ]
}

fn style_section(state: &State, t: Tokens) -> Element<'_, Msg> {
    let s = &state.options.style;
    let fonts = fonts()
        .into_iter()
        .map(|name| choice(name, s.font == name, true, Msg::Font(name.to_string()), t))
        .collect();
    let weight = vec![
        choice("Bold", s.bold, true, Msg::Bold(!s.bold), t),
        choice("Italic", s.italic, true, Msg::Italic(!s.italic), t),
    ];
    let colors = TEXT_COLORS
        .iter()
        .map(|(name, c)| swatch(name, *c, s.color == *c, Msg::Color(*c), t))
        .collect();
    let mut lines = vec![
        line("Font", fonts, t),
        line(
            "Size",
            amount(
                20.0..=120.0,
                s.size as f32,
                1.0,
                format!("{} px", s.size),
                Msg::Size,
                t,
            ),
            t,
        ),
        line("Style", weight, t),
        line("Colour", colors, t),
        line(
            "Background",
            vec![
                choice("Outline", !s.box_on, true, Msg::Box(false), t),
                choice("Box", s.box_on, true, Msg::Box(true), t),
            ],
            t,
        ),
    ];
    if s.box_on {
        let box_colors = BOX_COLORS
            .iter()
            .map(|(name, c)| swatch(name, *c, s.box_color == *c, Msg::BoxColor(*c), t))
            .collect();
        lines.push(line("Box colour", box_colors, t));
        lines.push(line(
            "Box cover",
            amount(
                0.0..=100.0,
                f32::from(s.box_opacity),
                5.0,
                format!("{}%", s.box_opacity),
                Msg::BoxOpacity,
                t,
            ),
            t,
        ));
    } else {
        let mut edge: Vec<Element<'_, Msg>> = EDGE_COLORS
            .iter()
            .map(|(name, c)| swatch(name, *c, s.outline_color == *c, Msg::OutlineColor(*c), t))
            .collect();
        edge.extend(amount(
            0.0..=8.0,
            s.outline,
            0.5,
            format!("{:.1} px", s.outline),
            Msg::Outline,
            t,
        ));
        lines.push(line("Outline", edge, t));
        lines.push(line(
            "Shadow",
            amount(
                0.0..=6.0,
                s.shadow,
                0.5,
                format!("{:.1} px", s.shadow),
                Msg::Shadow,
                t,
            ),
            t,
        ));
    }
    let places = [
        ("Bottom", Place::Bottom),
        ("Middle", Place::Middle),
        ("Top", Place::Top),
    ]
    .iter()
    .map(|(l, p)| choice(l, s.place == *p, true, Msg::Place(*p), t))
    .collect();
    let aligns = [
        ("Left", Align::Left),
        ("Centre", Align::Center),
        ("Right", Align::Right),
    ]
    .iter()
    .map(|(l, a)| choice(l, s.align == *a, true, Msg::Align(*a), t))
    .collect();
    lines.push(line("Position", places, t));
    lines.push(line("Line up", aligns, t));
    if s.place != Place::Middle {
        lines.push(line(
            "From the edge",
            amount(
                0.0..=300.0,
                s.margin as f32,
                5.0,
                format!("{} px", s.margin),
                Msg::Margin,
                t,
            ),
            t,
        ));
    }

    // the preview, when the video is what is made
    if state.options.kind == Kind::Video {
        let stale = state
            .preview
            .as_ref()
            .is_some_and(|(_, _, made_with)| made_with != &state.options);
        let label = match (state.previewing, &state.preview, stale) {
            (true, ..) => "Making the preview...",
            (false, None, _) => "Show a preview",
            (false, Some(_), true) => "Update the preview",
            (false, Some(_), false) => "Preview is up to date",
        };
        // the button stands out while the frame shown is not what the export would make
        let hot = stale || state.preview.is_none();
        let mut actions = row![
            button(text(label).size(12))
                .padding([5, 10])
                .style(move |theme_, status| {
                    if hot {
                        theme::primary(t)(theme_, status)
                    } else {
                        theme::outline_btn(t)(theme_, status)
                    }
                })
                .on_press_maybe((!state.previewing).then_some(Msg::Preview)),
            button(text("Reset the style").size(12))
                .padding([5, 10])
                .style(theme::outline_btn(t))
                .on_press(Msg::ResetStyle),
        ]
        .spacing(8)
        .align_y(Alignment::Center);
        if let Some((_, at, _)) = &state.preview {
            actions = actions.push(
                text(format!("A frame at {}", engine::transcript::clock(*at)))
                    .size(12)
                    .color(t.faint),
            );
        }
        lines.push(actions.into());
        if let Some(e) = &state.preview_error {
            lines.push(
                text(format!("No preview: {e}"))
                    .size(12)
                    .color(t.bad)
                    .into(),
            );
        }
        if let Some((handle, ..)) = &state.preview {
            lines.push(
                container(image(handle.clone()).width(Length::Fill))
                    .width(Length::Fill)
                    .style(theme::rounded_fill(iced::Color::BLACK, 4.0))
                    .into(),
            );
        }
    } else {
        lines.push(
            button(text("Reset the style").size(12))
                .padding([5, 10])
                .style(theme::outline_btn(t))
                .on_press(Msg::ResetStyle)
                .into(),
        );
    }
    section("Text style", lines, t)
}

// ------------------------------------------------------------------------- view

/// One choice in a row: the selected one is filled, one that doesn't fit the other choices is off.
fn choice<'a>(label: &str, selected: bool, enabled: bool, msg: Msg, t: Tokens) -> Element<'a, Msg> {
    button(text(label.to_string()).size(12))
        .padding([5, 10])
        .style(theme::segment(t, selected))
        .on_press_maybe(enabled.then_some(msg))
        .into()
}

/// A labelled row of choices.
fn line<'a>(label: &'a str, items: Vec<Element<'a, Msg>>, t: Tokens) -> Element<'a, Msg> {
    row![
        text(label)
            .size(12)
            .color(t.muted)
            .width(Length::Fixed(110.0)),
        row(items).spacing(6).wrap().vertical_spacing(6),
    ]
    .spacing(10)
    .align_y(Alignment::Center)
    .into()
}

fn section<'a>(title: &'a str, lines: Vec<Element<'a, Msg>>, t: Tokens) -> Element<'a, Msg> {
    let mut col = Column::new()
        .spacing(10)
        .push(text(title).size(13).color(t.text));
    for l in lines {
        col = col.push(l);
    }
    container(col)
        .padding(12)
        .width(Length::Fill)
        .style(theme::card(t))
        .into()
}

pub fn view(state: &State, t: Tokens) -> Element<'_, Msg> {
    let o = &state.options;
    let header = row![
        text("Export").size(18).color(t.text),
        space::horizontal(),
        button(icons::icon(icons::X, 14.0, t.muted))
            .padding(6)
            .style(theme::ghost(t))
            .on_press(Msg::Close),
    ]
    .align_y(Alignment::Center);

    let kinds = vec![
        choice(
            "Video",
            o.kind == Kind::Video,
            true,
            Msg::Kind(Kind::Video),
            t,
        ),
        choice(
            "Subtitles only",
            o.kind == Kind::Subtitles,
            true,
            Msg::Kind(Kind::Subtitles),
            t,
        ),
        choice(
            "Sound only",
            o.kind == Kind::Audio,
            true,
            Msg::Kind(Kind::Audio),
            t,
        ),
    ];
    let mut body = Column::new().spacing(12).push(line("Export", kinds, t));

    let language_line = |t: Tokens| -> Element<'_, Msg> {
        let mut items = vec![choice(
            "Original",
            o.lang.is_none(),
            true,
            Msg::Language(None),
            t,
        )];
        for (code, name) in &state.languages {
            items.push(choice(
                name,
                o.lang.as_deref() == Some(code.as_str()),
                true,
                Msg::Language(Some(code.clone())),
                t,
            ));
        }
        line("Language", items, t)
    };

    match o.kind {
        Kind::Video => {
            let containers = Container::ALL
                .iter()
                .map(|c| choice(c.label(), o.container == *c, true, Msg::Container(*c), t))
                .collect();
            let codecs = [Codec::H264, Codec::H265, Codec::Av1, Codec::Vp9]
                .iter()
                .map(|c| {
                    choice(
                        c.label(),
                        o.codec == *c,
                        o.container.codecs().contains(c),
                        Msg::Codec(*c),
                        t,
                    )
                })
                .collect();
            let gpu_ok = state.gpu && o.codec.gpu_encoder().is_some();
            let encoders = vec![
                choice(
                    "Auto",
                    o.encoder == Encoder::Auto,
                    true,
                    Msg::Encoder(Encoder::Auto),
                    t,
                ),
                choice(
                    "Graphics card",
                    o.encoder == Encoder::Gpu,
                    gpu_ok,
                    Msg::Encoder(Encoder::Gpu),
                    t,
                ),
                choice(
                    "Processor",
                    o.encoder == Encoder::Cpu,
                    true,
                    Msg::Encoder(Encoder::Cpu),
                    t,
                ),
            ];
            let qualities = [
                ("High", Quality::High),
                ("Balanced", Quality::Balanced),
                ("Small file", Quality::Small),
            ]
            .iter()
            .map(|(l, q)| choice(l, o.quality == *q, true, Msg::Quality(*q), t))
            .collect();
            let sizes = [
                ("Original", Resolution::Original),
                ("1080p", Resolution::P1080),
                ("720p", Resolution::P720),
                ("480p", Resolution::P480),
            ]
            .iter()
            .map(|(l, r)| choice(l, o.resolution == *r, true, Msg::Resolution(*r), t))
            .collect();
            body = body.push(section(
                "Video",
                vec![
                    line("Format", containers, t),
                    line("Codec", codecs, t),
                    line("Made on", encoders, t),
                    line("Quality", qualities, t),
                    line("Picture size", sizes, t),
                ],
                t,
            ));

            let modes = [
                ("On the picture", SubtitleMode::Burn),
                ("As a track", SubtitleMode::Track),
                ("None", SubtitleMode::Off),
            ]
            .iter()
            .map(|(l, m)| choice(l, o.subtitles == *m, true, Msg::Subtitles(*m), t))
            .collect();
            let mut lines = vec![line("Subtitles", modes, t)];
            if o.subtitles != SubtitleMode::Off {
                lines.push(language_line(t));
            }
            body = body.push(section("Subtitles", lines, t));
        }
        Kind::Subtitles => {
            let formats = [
                ("SRT", SubtitleFormat::Srt),
                ("WebVTT", SubtitleFormat::Vtt),
                ("Text", SubtitleFormat::Txt),
                ("ASS (styled)", SubtitleFormat::Ass),
            ]
            .iter()
            .map(|(l, f)| choice(l, o.subtitle_format == *f, true, Msg::SubtitleFormat(*f), t))
            .collect();
            body = body.push(section(
                "Subtitles",
                vec![line("File type", formats, t), language_line(t)],
                t,
            ));
        }
        Kind::Audio => {
            let formats = [
                ("MP3", AudioFormat::Mp3),
                ("M4A (AAC)", AudioFormat::M4a),
                ("WAV", AudioFormat::Wav),
                ("FLAC", AudioFormat::Flac),
            ]
            .iter()
            .map(|(l, f)| choice(l, o.audio_format == *f, true, Msg::AudioFormat(*f), t))
            .collect();
            body = body.push(section("Sound", vec![line("File type", formats, t)], t));
        }
    }

    if o.uses_style() {
        body = body.push(style_section(state, t));
    }

    if !state.cuts.is_empty() {
        let removed = engine::transcript::clock(cuts::removed(&state.cuts));
        let cut_line = vec![
            choice(
                "Leave them out",
                o.apply_cuts,
                true,
                Msg::ApplyCuts(true),
                t,
            ),
            choice(
                "Keep everything",
                !o.apply_cuts,
                true,
                Msg::ApplyCuts(false),
                t,
            ),
        ];
        body = body.push(section(
            "Cuts",
            vec![
                text(format!(
                    "{} cut {} in the player, {removed} in all.",
                    state.cuts.len(),
                    if state.cuts.len() == 1 {
                        "part"
                    } else {
                        "parts"
                    }
                ))
                .size(12)
                .color(t.muted)
                .into(),
                line("Cut parts", cut_line, t),
            ],
            t,
        ));
    }

    let mut footer_text = column![text(summary(state)).size(12).color(t.text)].spacing(4);
    if let Some(note) = note(state) {
        footer_text = footer_text.push(text(note).size(12).color(t.faint));
    }
    if let Some(error) = &state.error {
        footer_text = footer_text.push(text(error.clone()).size(12).color(t.bad));
    }
    let footer = row![
        footer_text.width(Length::Fill),
        button(text("Cancel").size(13))
            .padding([8, 14])
            .style(theme::outline_btn(t))
            .on_press(Msg::Close),
        button(text("Export...").size(13))
            .padding([8, 16])
            .style(theme::primary(t))
            .on_press(Msg::Export),
    ]
    .spacing(10)
    .align_y(Alignment::Center);

    // a video's many choices outgrow the window: they scroll, and the summary and Export stay in view. The shorter
    // kinds keep a dialog only as tall as they are.
    let tall = o.kind == Kind::Video || o.uses_style();
    let height = if tall { Length::Fill } else { Length::Shrink };
    container(
        column![
            header,
            scrollable(body.padding(iced::Padding::ZERO.right(12))).height(height),
            footer,
        ]
        .spacing(14),
    )
    .padding(20)
    .max_width(680)
    .height(height)
    .style(theme::dialog(t))
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::project::ProjectPaths;

    fn job() -> JobRef {
        JobRef {
            id: 1,
            media: PathBuf::from("talk.mp4"),
            paths: ProjectPaths::new("results", "talk"),
        }
    }

    fn opened() -> State {
        let mut s = State::default();
        open(&mut s, job(), (None, SubtitleStyle::default()), 600.0, true);
        s
    }

    #[test]
    fn choosing_a_container_keeps_the_codec_valid() {
        let mut s = opened();
        let _ = update(&mut s, Msg::Codec(Codec::H265));
        let _ = update(&mut s, Msg::Container(Container::Webm));
        assert_eq!(s.options.codec, Codec::Vp9);
        assert!(summary(&s).starts_with("WebM video in VP9, on the processor"));
        let _ = update(&mut s, Msg::Kind(Kind::Audio));
        assert!(summary(&s).starts_with("The sound as a .mp3 file"));
    }

    #[test]
    fn the_saved_file_gets_the_right_ending_and_queues_the_options() {
        let mut s = opened();
        let _ = update(&mut s, Msg::Kind(Kind::Audio));
        let _ = update(&mut s, Msg::AudioFormat(AudioFormat::Flac));
        let (_, out) = update(&mut s, Msg::Chosen(Some(PathBuf::from("x/sound.mp3"))));
        let Out::Enqueue(JobKind::Export { options, out, .. }) = out else {
            panic!("nothing queued");
        };
        assert_eq!(out, PathBuf::from("x/sound.flac"));
        assert_eq!(options.kind, Kind::Audio);
        assert!(!is_open(&s), "the dialog closes once it is queued");
    }

    #[test]
    fn style_choices_land_in_the_options_and_reset_brings_the_defaults_back() {
        let mut s = opened();
        let _ = update(&mut s, Msg::Size(71.6));
        let _ = update(&mut s, Msg::Box(true));
        let _ = update(&mut s, Msg::BoxOpacity(140.0));
        let _ = update(&mut s, Msg::Outline(2.3));
        let _ = update(&mut s, Msg::Place(Place::Top));
        let st = &s.options.style;
        assert_eq!(st.size, 72);
        assert!(st.box_on);
        assert_eq!(st.box_opacity, 100, "never more than all");
        assert_eq!(st.outline, 2.5, "half pixel steps");
        assert_eq!(st.place, Place::Top);
        let _ = update(&mut s, Msg::ResetStyle);
        assert_eq!(s.options.style, SubtitleStyle::default());
    }

    #[test]
    fn the_style_used_last_comes_back_when_the_dialog_opens() {
        let mut s = State::default();
        let mine = SubtitleStyle {
            size: 80,
            ..SubtitleStyle::default()
        };
        open(&mut s, job(), (None, mine.clone()), 600.0, true);
        assert_eq!(s.options.style, mine);
    }

    #[test]
    fn the_recording_itself_is_never_the_target() {
        let mut s = opened();
        let (_, out) = update(&mut s, Msg::Chosen(Some(PathBuf::from("talk.mp4"))));
        assert!(matches!(out, Out::None));
        assert!(s.error.is_some());
        assert!(is_open(&s));
    }

    #[test]
    fn without_an_nvidia_card_the_graphics_card_is_not_kept() {
        let mut s = State {
            options: ExportOptions {
                encoder: Encoder::Gpu,
                ..ExportOptions::default()
            },
            ..State::default()
        };
        open(
            &mut s,
            job(),
            (None, SubtitleStyle::default()),
            600.0,
            false,
        );
        assert_eq!(s.options.encoder, Encoder::Auto);
        assert!(summary(&s).contains("on the processor"));
    }

    #[test]
    fn a_language_the_result_lacks_falls_back_to_the_original() {
        let mut s = State::default();
        open(
            &mut s,
            job(),
            (Some("fa".into()), SubtitleStyle::default()),
            600.0,
            true,
        );
        assert_eq!(s.options.lang, None);
    }
}
