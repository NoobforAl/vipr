//! How subtitles look when they are drawn: font, size, colours, outline, shadow, a background box and where on the
//! picture they sit. Written out as an ASS file, the format ffmpeg's subtitle renderer (libass) reads natively.

use serde::{Deserialize, Serialize};

use crate::subtitles::Cue;

/// The size the style is measured in. libass scales it to the real picture, so a style looks the same on any video.
const PLAY_WIDTH: u32 = 1920;
const PLAY_HEIGHT: u32 = 1080;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Place {
    Bottom,
    Middle,
    Top,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Align {
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SubtitleStyle {
    pub font: String,
    /// Text height in pixels of a 1080-line picture.
    pub size: u32,
    pub bold: bool,
    pub italic: bool,
    /// 0xRRGGBB.
    pub color: u32,
    pub outline_color: u32,
    /// Outline width, in pixels of a 1080-line picture.
    pub outline: f32,
    pub shadow: f32,
    /// A box behind the text instead of an outline.
    pub box_on: bool,
    pub box_color: u32,
    /// How much the box covers, 0 to 100.
    pub box_opacity: u8,
    pub place: Place,
    pub align: Align,
    /// Distance from the edge the text sits at, in pixels of a 1080-line picture.
    pub margin: u32,
}

impl Default for SubtitleStyle {
    fn default() -> Self {
        SubtitleStyle {
            font: "Arial".to_string(),
            size: 54,
            bold: false,
            italic: false,
            color: 0xFFFFFF,
            outline_color: 0x000000,
            outline: 3.0,
            shadow: 1.0,
            box_on: false,
            box_color: 0x000000,
            box_opacity: 60,
            place: Place::Bottom,
            align: Align::Center,
            margin: 60,
        }
    }
}

/// An ASS colour: `&HAABBGGRR`, where alpha 00 is solid and FF is invisible.
fn ass_color(rgb: u32, opacity: u8) -> String {
    let alpha = 255 - (u32::from(opacity.min(100)) * 255 / 100);
    let (r, g, b) = ((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF);
    format!("&H{alpha:02X}{b:02X}{g:02X}{r:02X}")
}

impl SubtitleStyle {
    /// ASS alignment, laid out like a number pad: 1 2 3 at the bottom, 7 8 9 at the top.
    fn alignment(&self) -> u8 {
        let row = match self.place {
            Place::Bottom => 0,
            Place::Middle => 3,
            Place::Top => 6,
        };
        row + match self.align {
            Align::Left => 1,
            Align::Center => 2,
            Align::Right => 3,
        }
    }

    /// The `Style:` line.
    fn style_line(&self) -> String {
        // a box is drawn in the outline colour with the outline as its padding; the back colour matches it so every
        // renderer shows the same
        let (border, outline_color, back_color, outline, shadow) = if self.box_on {
            let c = ass_color(self.box_color, self.box_opacity);
            (3, c.clone(), c, self.outline.max(6.0), 0.0)
        } else {
            (
                1,
                ass_color(self.outline_color, 100),
                ass_color(0x000000, 60),
                self.outline,
                self.shadow,
            )
        };
        let font: String = self.font.chars().filter(|c| *c != ',').collect();
        format!(
            "Style: Default,{font},{},{},{},{outline_color},{back_color},{},{},0,0,100,100,0,0,{border},{outline:.1},{shadow:.1},{},40,40,{},1",
            self.size.clamp(8, 200),
            ass_color(self.color, 100),
            ass_color(self.color, 100),
            if self.bold { -1 } else { 0 },
            if self.italic { -1 } else { 0 },
            self.alignment(),
            self.margin.min(PLAY_HEIGHT / 2),
        )
    }
}

/// `0:01:02.35`: ASS times are in hundredths.
fn ass_time(seconds: f64) -> String {
    let cs = (seconds.max(0.0) * 100.0).round() as u64;
    format!(
        "{}:{:02}:{:02}.{:02}",
        cs / 360_000,
        cs / 6000 % 60,
        cs / 100 % 60,
        cs % 100
    )
}

/// Text as an ASS line holds it: lines joined with `\N`, and braces, which would start an override block, made
/// plain.
fn ass_text(text: &str) -> String {
    text.replace('\\', "/")
        .replace('{', "(")
        .replace('}', ")")
        .lines()
        .collect::<Vec<_>>()
        .join("\\N")
}

/// A whole ASS file for `cues` in `style`.
pub fn to_ass(cues: &[Cue], style: &SubtitleStyle) -> String {
    let mut out = format!(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: {PLAY_WIDTH}\nPlayResY: {PLAY_HEIGHT}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n\
         [V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n{}\n\n\
         [Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
        style.style_line()
    );
    for c in cues {
        out.push_str(&format!(
            "Dialogue: 0,{},{},Default,,0,0,0,,{}\n",
            ass_time(c.start),
            ass_time(c.end),
            ass_text(&c.text)
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(start: f64, end: f64, text: &str) -> Cue {
        Cue {
            start,
            end,
            text: text.to_string(),
            speaker: 0,
            utterance: 0,
        }
    }

    #[test]
    fn colours_are_written_alpha_blue_green_red() {
        assert_eq!(ass_color(0xFFFFFF, 100), "&H00FFFFFF");
        assert_eq!(
            ass_color(0xFF8000, 100),
            "&H000080FF",
            "orange: red FF, green 80, blue 00"
        );
        assert_eq!(ass_color(0x000000, 0), "&HFF000000", "nothing to see");
        assert_eq!(ass_color(0x000000, 60), "&H66000000");
    }

    #[test]
    fn place_and_alignment_make_the_number_pad_position() {
        let at = |place, align| {
            SubtitleStyle {
                place,
                align,
                ..SubtitleStyle::default()
            }
            .alignment()
        };
        assert_eq!(at(Place::Bottom, Align::Center), 2);
        assert_eq!(at(Place::Bottom, Align::Left), 1);
        assert_eq!(at(Place::Middle, Align::Right), 6);
        assert_eq!(at(Place::Top, Align::Center), 8);
    }

    #[test]
    fn a_box_replaces_the_outline_and_uses_its_colour_twice() {
        let boxed = SubtitleStyle {
            box_on: true,
            box_color: 0x202020,
            box_opacity: 75,
            ..SubtitleStyle::default()
        };
        let line = boxed.style_line();
        assert!(line.contains(",&H40202020,&H40202020,"), "{line}");
        assert!(
            line.contains(",3,6.0,0.0,2,"),
            "border style 3, padding 6, no shadow: {line}"
        );
        let plain = SubtitleStyle::default().style_line();
        assert!(plain.contains(",1,3.0,1.0,2,40,40,60,1"), "{plain}");
    }

    #[test]
    fn bold_italic_size_and_font_reach_the_style() {
        let s = SubtitleStyle {
            font: "Segoe UI, Bad".to_string(),
            size: 72,
            bold: true,
            italic: true,
            ..SubtitleStyle::default()
        };
        let line = s.style_line();
        assert!(
            line.starts_with("Style: Default,Segoe UI Bad,72,"),
            "a comma would break the fields: {line}"
        );
        assert!(line.contains(",-1,-1,0,0,"), "{line}");
    }

    #[test]
    fn events_have_hundredths_line_breaks_and_no_override_blocks() {
        let ass = to_ass(
            &[cue(62.354, 3725.0, "first\nsecond {\\b1}")],
            &SubtitleStyle::default(),
        );
        assert!(ass.contains("PlayResY: 1080"));
        assert!(
            ass.contains(
                "Dialogue: 0,0:01:02.35,1:02:05.00,Default,,0,0,0,,first\\Nsecond (/b1)\n"
            ),
            "{ass}"
        );
    }

    #[test]
    fn a_saved_style_missing_fields_takes_the_defaults() {
        let s: SubtitleStyle = serde_json::from_str(r#"{"size": 40, "place": "top"}"#).unwrap();
        assert_eq!(s.size, 40);
        assert_eq!(s.place, Place::Top);
        assert_eq!(s.font, "Arial");
    }
}
