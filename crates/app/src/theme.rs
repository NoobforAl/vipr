//! Design tokens and widget styles. The palette and radii follow the ddIRC app: calm layered
//! surfaces, one soft accent, hairline borders, generous radii, no shadows or ripples.

use iced::theme::Palette;
use iced::widget::{
    button, container, pick_list, progress_bar, scrollable, slider, text_editor, text_input,
};
use iced::{Background, Border, Color, Shadow, Theme, border};

#[derive(Clone, Copy, Debug)]
pub struct Tokens {
    pub dark: bool,
    pub bg: Color,
    pub surface: Color,
    pub hover: Color,
    pub rule: Color,
    pub text: Color,
    pub muted: Color,
    pub faint: Color,
    pub accent: Color,
    pub on_accent: Color,
    pub ok: Color,
    pub warn: Color,
    pub bad: Color,
    pub bubble: Color,
}

const fn hex(v: u32) -> Color {
    Color::from_rgb8((v >> 16) as u8, (v >> 8) as u8, v as u8)
}

pub const DARK: Tokens = Tokens {
    dark: true,
    bg: hex(0x15171C),
    surface: hex(0x1B1E24),
    hover: hex(0x242832),
    rule: hex(0x2E323C),
    text: hex(0xE8E9EE),
    muted: hex(0x9DA1AD),
    faint: hex(0x626773),
    accent: hex(0x8AB4F8),
    on_accent: hex(0x15171C),
    ok: hex(0x7BCF95),
    warn: hex(0xE2BA55),
    bad: hex(0xE57A7A),
    bubble: hex(0x20242C),
};

pub const LIGHT: Tokens = Tokens {
    dark: false,
    bg: hex(0xF6F5F2),
    surface: hex(0xEFEEEA),
    hover: hex(0xE6E4DF),
    rule: hex(0xDAD8D2),
    text: hex(0x1E1F24),
    muted: hex(0x5F616B),
    faint: hex(0x85878F),
    accent: hex(0x3563C0),
    on_accent: hex(0xFFFFFF),
    ok: hex(0x19703B),
    warn: hex(0x845D00),
    bad: hex(0xB1362C),
    bubble: hex(0xFFFFFF),
};

/// The player page in any app theme: the dark, flat look of a classic desktop media player.
pub const PLAYER: Tokens = Tokens {
    dark: true,
    bg: hex(0x1F1F1F),
    surface: hex(0x2B2B2B),
    hover: hex(0x3A3A3A),
    rule: hex(0x3F3F3F),
    text: hex(0xE6E6E6),
    muted: hex(0xA8A8A8),
    faint: hex(0x707070),
    accent: hex(0x2F9BE0),
    on_accent: hex(0xFFFFFF),
    ok: hex(0x7BCF95),
    warn: hex(0xE2BA55),
    bad: hex(0xE57A7A),
    bubble: hex(0x262626),
};

pub const RADIUS_S: f32 = 10.0;
pub const RADIUS_M: f32 = 14.0;
pub const RADIUS_L: f32 = 18.0;
pub const PILL: f32 = 999.0;

impl Tokens {
    pub fn theme(self) -> Theme {
        Theme::custom(
            if self.dark { "Dark" } else { "Light" },
            Palette {
                background: self.bg,
                text: self.text,
                primary: self.accent,
                success: self.ok,
                warning: self.warn,
                danger: self.bad,
            },
        )
    }

    /// Colour for speaker `i` (avatar, name, legend).
    pub fn speaker(self, i: usize) -> Color {
        const DARKS: [u32; 6] = [0x8AB4F8, 0x7BCF95, 0xE2BA55, 0xE59ACB, 0xB59BF0, 0x6FD0D0];
        const LIGHTS: [u32; 6] = [0x3563C0, 0x19703B, 0x845D00, 0xA33D80, 0x6B45C2, 0x1B7F86];
        hex(if self.dark {
            DARKS[i % 6]
        } else {
            LIGHTS[i % 6]
        })
    }
}

pub fn with_alpha(c: Color, a: f32) -> Color {
    Color { a, ..c }
}

pub fn mix(a: Color, b: Color, t: f32) -> Color {
    Color {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
        a: a.a + (b.a - a.a) * t,
    }
}

fn outline(c: Color, radius: f32) -> Border {
    Border {
        color: c,
        width: 1.0,
        radius: border::radius(radius),
    }
}

// ------------------------------------------------------------------ containers

/// Raised panel: surface colour, hairline border.
pub fn card(t: Tokens) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(t.surface)),
        border: outline(t.rule, RADIUS_M),
        ..Default::default()
    }
}

/// A card floating over the page, so it is told apart from what's under it.
pub fn toast(t: Tokens) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(t.surface)),
        border: outline(t.rule, RADIUS_M),
        shadow: Shadow {
            color: Color::from_rgba(0.0, 0.0, 0.0, 0.45),
            offset: iced::Vector::new(0.0, 6.0),
            blur_radius: 20.0,
        },
        ..Default::default()
    }
}

pub fn panel(t: Tokens) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(t.surface)),
        border: Border {
            color: t.rule,
            width: 1.0,
            radius: border::radius(0),
        },
        ..Default::default()
    }
}

pub fn bubble(t: Tokens) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(t.bubble)),
        border: outline(t.rule, RADIUS_M),
        ..Default::default()
    }
}

/// Small rounded label tinted with `color`.
pub fn chip(color: Color) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(with_alpha(color, 0.14))),
        border: Border {
            color: with_alpha(color, 0.35),
            width: 1.0,
            radius: border::radius(PILL),
        },
        ..Default::default()
    }
}

pub fn disc(color: Color) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(color)),
        border: Border {
            radius: border::radius(PILL),
            ..Default::default()
        },
        ..Default::default()
    }
}

pub fn rounded_fill(color: Color, radius: f32) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(color)),
        border: Border {
            radius: border::radius(radius),
            ..Default::default()
        },
        ..Default::default()
    }
}

pub fn scrim() -> impl Fn(&Theme) -> container::Style {
    |_| container::Style {
        background: Some(Background::Color(Color {
            a: 0.62,
            ..Color::BLACK
        })),
        ..Default::default()
    }
}

pub fn dialog(t: Tokens) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(t.bg)),
        border: outline(t.rule, RADIUS_L),
        shadow: Shadow {
            color: Color {
                a: 0.45,
                ..Color::BLACK
            },
            offset: iced::Vector::new(0.0, 12.0),
            blur_radius: 40.0,
        },
        ..Default::default()
    }
}

// ------------------------------------------------------------------ buttons

fn press_dim(c: Color, status: button::Status, t: Tokens) -> Color {
    match status {
        button::Status::Hovered => mix(c, if t.dark { Color::WHITE } else { Color::BLACK }, 0.10),
        button::Status::Pressed => mix(c, Color::BLACK, 0.12),
        _ => c,
    }
}

/// Filled accent button (main action).
pub fn primary(t: Tokens) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        if matches!(status, button::Status::Disabled) {
            return button::Style {
                background: Some(Background::Color(t.hover)),
                text_color: t.faint,
                border: Border {
                    radius: border::radius(RADIUS_S),
                    ..Default::default()
                },
                ..Default::default()
            };
        }
        button::Style {
            background: Some(Background::Color(press_dim(t.accent, status, t))),
            text_color: t.on_accent,
            border: Border {
                radius: border::radius(RADIUS_S),
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// Bordered button (secondary action).
pub fn outline_btn(t: Tokens) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let (bg, fg) = match status {
            button::Status::Hovered => (Some(Background::Color(t.hover)), t.text),
            button::Status::Pressed => (Some(Background::Color(mix(t.hover, t.rule, 0.5))), t.text),
            button::Status::Disabled => (None, t.faint),
            _ => (None, t.text),
        };
        button::Style {
            background: bg,
            text_color: fg,
            border: outline(t.rule, RADIUS_S),
            ..Default::default()
        }
    }
}

/// Borderless button that only shows a background on hover.
pub fn ghost(t: Tokens) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let (bg, fg) = match status {
            button::Status::Hovered | button::Status::Pressed => {
                (Some(Background::Color(t.hover)), t.text)
            }
            button::Status::Disabled => (None, t.faint),
            _ => (None, t.muted),
        };
        button::Style {
            background: bg,
            text_color: fg,
            border: Border {
                radius: border::radius(RADIUS_S),
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// Sidebar item; `active` shows the accent tint.
pub fn nav_item(t: Tokens, active: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let bg = if active {
            Some(Background::Color(with_alpha(t.accent, 0.16)))
        } else if matches!(status, button::Status::Hovered | button::Status::Pressed) {
            Some(Background::Color(t.hover))
        } else {
            None
        };
        button::Style {
            background: bg,
            text_color: if active { t.accent } else { t.text },
            border: Border {
                radius: border::radius(RADIUS_S),
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// Segmented-control button.
pub fn segment(t: Tokens, selected: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
        button::Style {
            background: Some(Background::Color(if selected {
                with_alpha(t.accent, 0.18)
            } else if hovered {
                t.hover
            } else {
                Color::TRANSPARENT
            })),
            text_color: if selected { t.accent } else { t.muted },
            border: outline(
                if selected {
                    with_alpha(t.accent, 0.5)
                } else {
                    t.rule
                },
                RADIUS_S,
            ),
            ..Default::default()
        }
    }
}

// ------------------------------------------------------------------ player

/// Square toolbar button: nothing until hovered, tinted when `selected`.
pub fn flat(t: Tokens, selected: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
        let background = if selected {
            Some(Background::Color(with_alpha(t.accent, 0.28)))
        } else if hovered {
            Some(Background::Color(t.hover))
        } else {
            None
        };
        button::Style {
            background,
            text_color: match status {
                button::Status::Disabled => t.faint,
                _ if selected || hovered => t.text,
                _ => t.muted,
            },
            border: Border {
                radius: border::radius(2),
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// A flat fill with a one pixel edge.
pub fn flat_card(t: Tokens) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(t.bubble)),
        border: Border {
            color: t.rule,
            width: 1.0,
            radius: border::radius(2),
        },
        ..Default::default()
    }
}

/// The subtitle line on screen in the timeline.
pub fn current_line(t: Tokens, strength: f32) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(with_alpha(t.accent, 0.22 * strength))),
        border: Border {
            radius: border::radius(2),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// A bar as tall as the widget with a thin marker. `filled` is the part already played.
fn bar_slider(
    filled: Color,
    rest: Color,
    edge: Color,
    marker: Color,
    marker_width: u16,
    height: f32,
) -> impl Fn(&Theme, slider::Status) -> slider::Style {
    move |_, _| slider::Style {
        rail: slider::Rail {
            backgrounds: (Background::Color(filled), Background::Color(rest)),
            width: height,
            border: Border {
                color: edge,
                width: 1.0,
                radius: border::radius(0),
            },
        },
        handle: slider::Handle {
            shape: slider::HandleShape::Rectangle {
                width: marker_width,
                border_radius: border::radius(0),
            },
            background: Background::Color(marker),
            border_width: 0.0,
            border_color: Color::TRANSPARENT,
        },
    }
}

/// `height` has to match the widget's: the rail is drawn at this height whatever the layout says.
pub fn seek_slider(t: Tokens, height: f32) -> impl Fn(&Theme, slider::Status) -> slider::Style {
    bar_slider(
        t.rule,
        t.surface,
        t.rule,
        t.accent,
        SEEK_MARKER_WIDTH,
        height,
    )
}

pub fn volume_slider(t: Tokens, height: f32) -> impl Fn(&Theme, slider::Status) -> slider::Style {
    bar_slider(
        with_alpha(t.accent, 0.45),
        t.bg,
        t.rule,
        t.accent,
        2,
        height,
    )
}

/// Width of the seek bar's marker. Ticks on the bar have to be inset by half of it to line up.
pub const SEEK_MARKER_WIDTH: u16 = 4;

/// Scrollbars that belong on a dark panel, in any app theme.
pub fn dark_scroll(t: Tokens) -> impl Fn(&Theme, scrollable::Status) -> scrollable::Style {
    move |theme, status| {
        let mut style = scrollable::default(theme, status);
        let scroller = scrollable::Scroller {
            background: Background::Color(t.rule),
            border: Border {
                radius: border::radius(2),
                ..Default::default()
            },
        };
        style.vertical_rail = scrollable::Rail {
            background: None,
            border: Border::default(),
            scroller,
        };
        style
    }
}

// ------------------------------------------------------------------ inputs

pub fn input(t: Tokens) -> impl Fn(&Theme, text_input::Status) -> text_input::Style {
    move |_, status| {
        let border_color = match status {
            text_input::Status::Focused { .. } => t.accent,
            text_input::Status::Hovered => t.faint,
            _ => t.rule,
        };
        text_input::Style {
            background: Background::Color(t.bg),
            border: outline(border_color, RADIUS_S),
            icon: t.muted,
            placeholder: t.faint,
            value: t.text,
            selection: with_alpha(t.accent, 0.35),
        }
    }
}

pub fn pick(t: Tokens) -> impl Fn(&Theme, pick_list::Status) -> pick_list::Style {
    move |_, status| {
        let border_color = match status {
            pick_list::Status::Opened { .. } => t.accent,
            pick_list::Status::Hovered => t.faint,
            _ => t.rule,
        };
        pick_list::Style {
            text_color: t.text,
            placeholder_color: t.faint,
            handle_color: t.muted,
            background: Background::Color(t.bg),
            border: outline(border_color, RADIUS_S),
        }
    }
}

/// The list that opens under a search box.
pub fn search_menu(t: Tokens) -> impl Fn(&Theme) -> iced::overlay::menu::Style {
    move |_| iced::overlay::menu::Style {
        background: Background::Color(t.surface),
        border: outline(t.rule, RADIUS_S),
        text_color: t.text,
        selected_text_color: t.on_accent,
        selected_background: Background::Color(t.accent),
        shadow: Default::default(),
    }
}

/// Thin outline circle (a step that has not started).
pub fn ring(color: Color) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        border: Border {
            color,
            width: 1.5,
            radius: border::radius(PILL),
        },
        ..Default::default()
    }
}

/// Progress bar in a chosen colour.
pub fn bar_tint(t: Tokens, color: Color) -> impl Fn(&Theme) -> progress_bar::Style {
    move |_| progress_bar::Style {
        background: Background::Color(t.rule),
        bar: Background::Color(color),
        border: Border {
            radius: border::radius(PILL),
            ..Default::default()
        },
    }
}

pub fn bar(t: Tokens) -> impl Fn(&Theme) -> progress_bar::Style {
    move |_| progress_bar::Style {
        background: Background::Color(t.rule),
        bar: Background::Color(t.accent),
        border: Border {
            radius: border::radius(PILL),
            ..Default::default()
        },
    }
}

// ------------------------------------------------------------------ window chrome

/// Title strip at the top of the window. Same colour as the sidebar, so strip + sidebar form one
/// L-shaped frame around the content (like the VS Code title bar and side bar).
pub fn strip(t: Tokens) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(t.surface)),
        ..Default::default()
    }
}

/// Sidebar: the strip colour, no border (a separate 1 px divider sits on its right edge).
/// `round_bottom_left` follows a rounded window corner.
pub fn sidebar(t: Tokens, round_bottom_left: bool) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(t.surface)),
        border: Border {
            radius: border::Radius::new(0).bottom_left(if round_bottom_left { 7.0 } else { 0.0 }),
            ..Default::default()
        },
        ..Default::default()
    }
}

pub fn divider(t: Tokens) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(t.rule)),
        ..Default::default()
    }
}

/// One pixel frame around the window (only drawn when we draw the window chrome ourselves).
/// `radius` matches the system's rounded window corners (Windows 11), 0 otherwise.
pub fn window_frame(
    t: Tokens,
    draw_border: bool,
    radius: f32,
) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(t.bg)),
        border: Border {
            color: if draw_border {
                t.rule
            } else {
                Color::TRANSPARENT
            },
            width: 1.0,
            radius: border::radius(radius),
        },
        ..Default::default()
    }
}

/// Minimize / maximize / close. Flat, square, full height; close turns red on hover.
pub fn win_button(t: Tokens, close: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
        let bg = if !hovered {
            None
        } else if close {
            Some(Background::Color(mix(t.bad, t.bg, 0.55)))
        } else {
            Some(Background::Color(t.hover))
        };
        button::Style {
            background: bg,
            text_color: t.text,
            border: Border::default(),
            ..Default::default()
        }
    }
}

/// Read-only text that can be selected: no background or border, a soft accent highlight for the selection.
pub fn selectable(t: Tokens) -> impl Fn(&Theme, text_editor::Status) -> text_editor::Style {
    move |_, _| text_editor::Style {
        background: Background::Color(Color::TRANSPARENT),
        border: Border {
            width: 0.0,
            radius: border::radius(0),
            color: Color::TRANSPARENT,
        },
        placeholder: t.faint,
        value: t.text,
        selection: with_alpha(t.accent, 0.38),
    }
}
