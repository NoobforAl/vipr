//! Line icons (24x24, stroke only). Shapes follow the Lucide set (ISC licence). The SVG widget tints
//! them with the colour we pass, so the stroke colour inside the file does not matter.

use iced::widget::svg;
use iced::{Color, Element, Length};

macro_rules! icon_src {
    ($body:expr) => {
        concat!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="white" "#,
            r#"stroke-width="2" stroke-linecap="round" stroke-linejoin="round">"#,
            $body,
            "</svg>"
        )
    };
}

pub const PLUS: &str = icon_src!(r#"<path d="M12 5v14M5 12h14"/>"#);
pub const X: &str = icon_src!(r#"<path d="M18 6 6 18M6 6l12 12"/>"#);
pub const CHECK: &str = icon_src!(r#"<path d="M20 6 9 17l-5-5"/>"#);
pub const CHEVRON_DOWN: &str = icon_src!(r#"<path d="m6 9 6 6 6-6"/>"#);
pub const CHEVRON_UP: &str = icon_src!(r#"<path d="m18 15-6-6-6 6"/>"#);
pub const ARROW_LEFT: &str = icon_src!(r#"<path d="M19 12H5M12 19l-7-7 7-7"/>"#);
pub const DOWNLOAD: &str =
    icon_src!(r#"<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/>"#);
pub const FOLDER: &str = icon_src!(
    r#"<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>"#
);
pub const VIDEO: &str = icon_src!(
    r#"<path d="m16 13 5.22 3.48a.5.5 0 0 0 .78-.42V7.87a.5.5 0 0 0-.75-.43L16 10.5"/><rect x="2" y="6" width="14" height="12" rx="2"/>"#
);
pub const SLIDERS: &str = icon_src!(
    r#"<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M2 14h4M10 8h4M18 16h4"/>"#
);
pub const AUDIO: &str = icon_src!(r#"<path d="M2 10v3M6 6v11M10 3v18M14 8v7M18 5v13M22 10v3"/>"#);
pub const PLAY: &str = icon_src!(r#"<polygon points="6 3 20 12 6 21 6 3"/>"#);
pub const FILE_TEXT: &str = icon_src!(
    r#"<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4M10 9H8M16 13H8M16 17H8"/>"#
);
pub const SPARKLES: &str = icon_src!(
    r#"<path d="M9.94 15.5A2 2 0 0 0 8.5 14.06l-6.14-1.58a.5.5 0 0 1 0-.96L8.5 9.94A2 2 0 0 0 9.94 8.5l1.58-6.14a.5.5 0 0 1 .96 0L14.06 8.5A2 2 0 0 0 15.5 9.94l6.14 1.58a.5.5 0 0 1 0 .96L15.5 14.06a2 2 0 0 0-1.44 1.44l-1.58 6.14a.5.5 0 0 1-.96 0z"/>"#
);
pub const COPY: &str = icon_src!(
    r#"<rect x="8" y="8" width="14" height="14" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>"#
);
pub const CPU: &str = icon_src!(
    r#"<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>"#
);
pub const ALERT: &str =
    icon_src!(r#"<circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>"#);

// The player page: Phosphor Icons (MIT licence, assets/icons/phosphor/LICENSE).
pub const PLAY_SOLID: &str = include_str!("../../../assets/icons/phosphor/play-fill.svg");
pub const PAUSE: &str = include_str!("../../../assets/icons/phosphor/pause-fill.svg");
pub const STOP: &str = include_str!("../../../assets/icons/phosphor/stop-fill.svg");
pub const SKIP_BACK: &str = include_str!("../../../assets/icons/phosphor/skip-back-fill.svg");
pub const SKIP_FORWARD: &str = include_str!("../../../assets/icons/phosphor/skip-forward-fill.svg");
pub const REWIND: &str = include_str!("../../../assets/icons/phosphor/rewind-fill.svg");
pub const FAST_FORWARD: &str = include_str!("../../../assets/icons/phosphor/fast-forward-fill.svg");
pub const VOLUME: &str = include_str!("../../../assets/icons/phosphor/speaker-high-fill.svg");
pub const VOLUME_OFF: &str = include_str!("../../../assets/icons/phosphor/speaker-x-fill.svg");
pub const FLAG: &str = include_str!("../../../assets/icons/phosphor/flag.svg");
pub const FLAG_SOLID: &str = include_str!("../../../assets/icons/phosphor/flag-fill.svg");
pub const MARK_PLUS: &str = include_str!("../../../assets/icons/phosphor/plus.svg");
pub const CLOSE: &str = include_str!("../../../assets/icons/phosphor/x.svg");
pub const BACK: &str = include_str!("../../../assets/icons/phosphor/arrow-left.svg");
pub const BOOKMARK: &str = include_str!("../../../assets/icons/phosphor/bookmark-simple.svg");
pub const REFRESH: &str = include_str!("../../../assets/icons/phosphor/arrow-clockwise.svg");
pub const FOLLOW: &str = include_str!("../../../assets/icons/phosphor/crosshair-simple.svg");

/// Spinner: a faint ring with a comet on it, bright at its head and fading along its tail. Rotate it to animate.
pub const SPINNER: &str = icon_src!(
    r#"<circle cx="12" cy="12" r="9" stroke-opacity="0.15" stroke-width="2.6"/><g stroke-width="2.6"><path d="M3 12a9 9 0 0 1 2.64-6.36" stroke-opacity="0.2" stroke-linecap="butt"/><path d="M5.64 5.64A9 9 0 0 1 12 3" stroke-opacity="0.45" stroke-linecap="butt"/><path d="M12 3a9 9 0 0 1 6.36 2.64" stroke-opacity="0.72" stroke-linecap="butt"/><path d="M18.36 5.64A9 9 0 0 1 21 12"/></g>"#
);

// Window buttons (minimize / maximize / restore / close). A separate 10 x 10 grid with 1 px strokes:
// the same proportions as the Windows caption glyphs, and much crisper than the 24 px line icons at this size.
macro_rules! win_src {
    ($body:expr) => {
        concat!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10" fill="none" stroke="white" "#,
            r#"stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round">"#,
            $body,
            "</svg>"
        )
    };
}

pub const WIN_MINIMIZE: &str = win_src!(r#"<path d="M0.5 5.5h9"/>"#);
pub const WIN_MAXIMIZE: &str =
    win_src!(r#"<rect x="0.55" y="0.55" width="8.9" height="8.9" rx="1.4"/>"#);
pub const WIN_RESTORE: &str = win_src!(
    r#"<path d="M2.6 2.5V2A1.45 1.45 0 0 1 4.05.55H8A1.45 1.45 0 0 1 9.45 2V6A1.45 1.45 0 0 1 8 7.45H7.5"/><rect x="0.55" y="2.55" width="6.9" height="6.9" rx="1.4"/>"#
);
pub const WIN_CLOSE: &str = win_src!(r#"<path d="M0.9 0.9 9.1 9.1M9.1 0.9 0.9 9.1"/>"#);

/// Draw an icon `size` px wide, tinted `color`.
pub fn icon<'a, M: 'a>(src: &'static str, size: f32, color: Color) -> Element<'a, M> {
    svg(svg::Handle::from_memory(src.as_bytes()))
        .width(Length::Fixed(size))
        .height(Length::Fixed(size))
        .style(move |_, _| svg::Style { color: Some(color) })
        .into()
}

/// How far a spinner has turned after `clock` seconds, in radians.
pub fn spin_angle(clock: f32) -> f32 {
    (clock * 7.0) % std::f32::consts::TAU
}

/// A spinner turned by `angle` radians (call again with a new angle to animate it).
pub fn spinner<'a, M: 'a>(angle: f32, size: f32, color: Color) -> Element<'a, M> {
    svg(svg::Handle::from_memory(SPINNER.as_bytes()))
        .width(Length::Fixed(size))
        .height(Length::Fixed(size))
        .style(move |_, _| svg::Style { color: Some(color) })
        .rotation(iced::Rotation::Floating(iced::Radians(angle)))
        .into()
}
