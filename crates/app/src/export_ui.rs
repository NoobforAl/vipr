//! The export dialog: what to make, in which format, how, and with which subtitles. A summary line says what will
//! come out before anything runs.

use std::path::PathBuf;

use engine::cuts::{self, Cut, CutsFile};
use iced::widget::{Column, button, column, container, row, scrollable, space, text};
use iced::{Alignment, Element, Length, Task};

use crate::export::{
    AudioFormat, Codec, Container, Encoder, ExportOptions, Kind, Quality, Resolution,
    SubtitleFormat, SubtitleMode, TextSize,
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
    TextSize(TextSize),
    ApplyCuts(bool),
    SubtitleFormat(SubtitleFormat),
    AudioFormat(AudioFormat),
    /// Ask where to save.
    Export,
    Chosen(Option<PathBuf>),
}

/// Opens the dialog for `job`. Choices made earlier stay, except the language, which follows what is on show.
pub fn open(state: &mut State, job: JobRef, lang: Option<String>, duration: f64, gpu: bool) {
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
    if !gpu && state.options.encoder == Encoder::Gpu {
        state.options.encoder = Encoder::Auto;
    }
    state.job = Some(job);
}

pub fn is_open(state: &State) -> bool {
    state.job.is_some()
}

pub fn update(state: &mut State, msg: Msg) -> (Task<Msg>, Out) {
    let o = &mut state.options;
    match msg {
        Msg::Close => state.job = None,
        Msg::Kind(k) => o.kind = k,
        Msg::Container(c) => o.container = c,
        Msg::Codec(c) => o.codec = c,
        Msg::Encoder(e) => o.encoder = e,
        Msg::Quality(q) => o.quality = q,
        Msg::Resolution(r) => o.resolution = r,
        Msg::Subtitles(m) => o.subtitles = m,
        Msg::Language(l) => o.lang = l,
        Msg::TextSize(s) => o.text_size = s,
        Msg::ApplyCuts(yes) => o.apply_cuts = yes,
        Msg::SubtitleFormat(f) => o.subtitle_format = f,
        Msg::AudioFormat(f) => o.audio_format = f,
        Msg::Export => {
            let Some(job) = &state.job else {
                return (Task::none(), Out::None);
            };
            let dir = job.media.parent().map(std::path::Path::to_path_buf);
            let ext = state.options.extension();
            let name = format!("{}.{ext}", file_name(job, &state.options, &state.languages));
            return (Task::perform(pick(dir, name, ext), Msg::Chosen), Out::None);
        }
        Msg::Chosen(None) => {}
        Msg::Chosen(Some(path)) => {
            let Some(job) = &state.job else {
                return (Task::none(), Out::None);
            };
            let path = path.with_extension(state.options.extension());
            if path == job.media {
                state.error = Some(
                    "The export can't replace the recording itself. Choose another name.".into(),
                );
                return (Task::none(), Out::None);
            }
            let kind = JobKind::Export {
                job: job.id,
                options: state.options.clone(),
                out: path,
            };
            state.job = None;
            return (Task::none(), Out::Enqueue(kind));
        }
    }
    state.options.fix();
    (Task::none(), Out::None)
}

/// A name to start the save dialog with, without its ending.
fn file_name(job: &JobRef, o: &ExportOptions, languages: &[(String, String)]) -> String {
    let language = language_name(o, languages);
    match o.kind {
        Kind::Video if o.subtitles == SubtitleMode::Off => format!("{} (export)", job.paths.stem),
        Kind::Video => format!("{} - {language} subtitles", job.paths.stem),
        Kind::Subtitles => format!("{} - {language}", job.paths.stem),
        Kind::Audio => format!("{} (sound)", job.paths.stem),
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
            if o.subtitles == SubtitleMode::Burn {
                let text_sizes = [
                    ("Small", TextSize::Small),
                    ("Medium", TextSize::Medium),
                    ("Large", TextSize::Large),
                ]
                .iter()
                .map(|(l, s)| choice(l, o.text_size == *s, true, Msg::TextSize(*s), t))
                .collect();
                lines.push(line("Text size", text_sizes, t));
            }
            body = body.push(section("Subtitles", lines, t));
        }
        Kind::Subtitles => {
            let formats = [
                ("SRT", SubtitleFormat::Srt),
                ("WebVTT", SubtitleFormat::Vtt),
                ("Text", SubtitleFormat::Txt),
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

    container(
        column![
            header,
            scrollable(body.padding(iced::Padding::ZERO.right(12))).height(Length::Shrink),
            footer,
        ]
        .spacing(14),
    )
    .padding(20)
    .max_width(680)
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
        open(&mut s, job(), None, 600.0, true);
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
        open(&mut s, job(), None, 600.0, false);
        assert_eq!(s.options.encoder, Encoder::Auto);
        assert!(summary(&s).contains("on the processor"));
    }

    #[test]
    fn a_language_the_result_lacks_falls_back_to_the_original() {
        let mut s = State::default();
        open(&mut s, job(), Some("fa".into()), 600.0, true);
        assert_eq!(s.options.lang, None);
    }
}
