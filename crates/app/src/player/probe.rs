//! Pure parser for the stderr of `ffmpeg -hide_banner -i FILE`, so no ffprobe download is needed.
//! Also the choice of output size and rate from what it found.

#[derive(Debug, Clone, PartialEq, Default)]
pub struct AudioTrack {
    /// `title`, or OBS's `name`, or the handler name. Untrusted file metadata.
    pub title: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Probe {
    pub duration_secs: Option<f64>,
    pub width: u32,
    pub height: u32,
    pub fps: Option<f64>,
    /// Clockwise degrees needed to display the coded picture upright: 0, 90, 180 or 270.
    pub rotation: u32,
    /// A real video stream. Cover art doesn't count, matching `-map 0:V:0`.
    pub has_video: bool,
    pub audio_tracks: Vec<AudioTrack>,
}

impl Probe {
    pub fn has_audio(&self) -> bool {
        !self.audio_tracks.is_empty()
    }

    /// Size after ffmpeg's autorotation, which is what the filter chain receives.
    pub fn display_size(&self) -> (u32, u32) {
        if self.rotation % 180 == 90 {
            (self.height, self.width)
        } else {
            (self.width, self.height)
        }
    }
}

/// Playback is capped at 1080p (either way round) and 30 fps, even for 60 fps recordings.
const MAX_LONG: f64 = 1920.0;
const MAX_SHORT: f64 = 1080.0;
const MAX_FPS: f64 = 30.0;
/// Used when the probe can't read a size. Wrong proportions beat no picture.
const FALLBACK_SIZE: (u32, u32) = (1280, 720);

/// Width, height and rate of the raw frames, kept in proportion. Sizes are even and at least 2, so
/// the `yuv420p` chroma planes are never empty.
pub fn output_format(probe: &Probe) -> (u32, u32, f64) {
    let (w, h) = match probe.display_size() {
        (0, _) | (_, 0) => FALLBACK_SIZE,
        size => size,
    };
    let (long, short) = (w.max(h) as f64, w.min(h) as f64);
    let scale = (MAX_LONG / long).min(MAX_SHORT / short).min(1.0);
    let even = |v: u32| ((v as f64 * scale).round() as u32 & !1).max(2);
    let fps = probe
        .fps
        .filter(|f| f.is_finite() && *f > 0.0)
        .unwrap_or(MAX_FPS)
        .min(MAX_FPS);
    (even(w), even(h), fps)
}

/// Split on commas outside parentheses: `yuv420p(tv, bt709), 1920x1080 [SAR 1:1], 60 fps`.
fn top_level_fields(s: &str) -> Vec<&str> {
    let (mut out, mut depth, mut start) = (Vec::new(), 0i32, 0usize);
    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(s[start..].trim());
    out
}

fn parse_duration(line: &str) -> Option<f64> {
    let rest = line.trim().strip_prefix("Duration:")?.trim();
    let stamp = rest.split(',').next()?.trim();
    let mut parts = stamp.split(':');
    let h: f64 = parts.next()?.parse().ok()?;
    let m: f64 = parts.next()?.parse().ok()?;
    let s: f64 = parts.next()?.parse().ok()?;
    Some(h * 3600.0 + m * 60.0 + s)
}

fn parse_size(field: &str) -> Option<(u32, u32)> {
    let token = field.split_whitespace().next()?;
    let (w, h) = token.split_once('x')?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

/// `Stream #0:1[0x2](und): Audio: aac ...` gives `("Audio", "aac ...")`.
fn stream_kind(line: &str) -> Option<(&str, &str)> {
    let rest = line.trim().strip_prefix("Stream #")?;
    let after_id = rest.split_once(": ")?.1;
    after_id.split_once(": ")
}

fn normalize_rotation(degrees: f64) -> u32 {
    let d = degrees.round() as i64;
    (((d % 360) + 360) % 360) as u32
}

#[derive(PartialEq)]
enum In {
    Other,
    Video,
    Audio,
}

/// Only the first input and the first real video stream count: `-map 0:V:0` selects that one and
/// skips attached pictures.
pub fn parse_probe(stderr: &str) -> Probe {
    let mut p = Probe::default();
    let mut section = In::Other;
    let mut title_rank = 0;

    for line in stderr.lines() {
        let t = line.trim();
        if t.starts_with("Output #") {
            break;
        }
        if t.starts_with("Duration:") {
            p.duration_secs = parse_duration(t);
            continue;
        }
        if let Some((kind, body)) = stream_kind(t) {
            let fields = top_level_fields(body);
            title_rank = 0;
            match kind {
                "Video" if !p.has_video && !t.contains("(attached pic)") => {
                    p.has_video = true;
                    section = In::Video;
                    if let Some((w, h)) = fields.iter().find_map(|f| parse_size(f)) {
                        p.width = w;
                        p.height = h;
                    }
                    let number_before = |unit: &str| {
                        fields.iter().find_map(|f| {
                            f.strip_suffix(unit)
                                .and_then(|n| n.trim().parse::<f64>().ok())
                        })
                    };
                    p.fps = number_before(" fps").or_else(|| number_before(" tbr"));
                }
                "Audio" => {
                    section = In::Audio;
                    p.audio_tracks.push(AudioTrack::default());
                }
                _ => section = In::Other,
            }
            continue;
        }
        if section == In::Video {
            // "displaymatrix: rotation of" in older builds, "Display Matrix: rotation of" in ffmpeg 9
            let lower = t.to_ascii_lowercase();
            let matrix = lower
                .strip_prefix("displaymatrix: rotation of ")
                .or_else(|| lower.strip_prefix("display matrix: rotation of "));
            if let Some(rest) = matrix {
                // ffmpeg prints the counter-clockwise angle, so -90 means turn 90 clockwise
                if let Some(deg) = rest
                    .split_whitespace()
                    .next()
                    .and_then(|n| n.parse::<f64>().ok())
                {
                    p.rotation = normalize_rotation(-deg);
                }
            } else if let Some((key, value)) = t.split_once(':')
                && key.trim() == "rotate"
                && let Ok(deg) = value.trim().parse::<f64>()
                && p.rotation == 0
            {
                p.rotation = normalize_rotation(deg);
            }
        }
        if section == In::Audio
            && let Some((key, value)) = t.split_once(':')
        {
            let rank = match key.trim() {
                "title" => 3,
                "name" => 2,
                "handler_name" => 1,
                _ => 0,
            };
            if rank > title_rank
                && let Some(track) = p.audio_tracks.last_mut()
            {
                track.title = Some(value.trim().to_string());
                title_rank = rank;
            }
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    const OBS: &str = r#"[in#0 @ 000001bc9bfba840] UDTA parsing failed retrying raw
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '2026-11-02 19-07-02.mp4':
  Metadata:
    encoder         : OBS Studio (33.0.1)
  Duration: 01:18:42.73, start: 0.000000, bitrate: 2388 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1920x1080 [SAR 1:1 DAR 16:9], 1945 kb/s, 60 fps, 60 tbr, 60 tbn (default)
    Metadata:
      handler_name    : OBS Video Handler
      encoder         : obs_nvenc_h264_tex
      name            : advanced_video_recording
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 195 kb/s (default)
    Metadata:
      handler_name    : OBS Audio Handler
      name            : Mix (Mic + System)
  Stream #0:2[0x3](und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 36 kb/s (default)
    Metadata:
      handler_name    : OBS Audio Handler
      name            : Mic
  Stream #0:3[0x4](und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 193 kb/s (default)
    Metadata:
      handler_name    : OBS Audio Handler
      name            : System Sound
At least one output file must be specified
"#;

    #[test]
    fn obs_three_tracks() {
        let p = parse_probe(OBS);
        assert_eq!((p.width, p.height), (1920, 1080));
        assert_eq!(p.fps, Some(60.0));
        assert_eq!(p.rotation, 0);
        assert!((p.duration_secs.unwrap() - 4722.73).abs() < 0.01);
        assert_eq!(p.audio_tracks.len(), 3);
        assert_eq!(
            p.audio_tracks[0].title.as_deref(),
            Some("Mix (Mic + System)")
        );
        assert_eq!(p.audio_tracks[1].title.as_deref(), Some("Mic"));
    }

    #[test]
    fn rotated_side_data_swaps_display_size() {
        let s = r#"  Duration: 00:00:10.00, start: 0.000000, bitrate: 100 kb/s
  Stream #0:0(und): Video: hevc (Main), yuv420p(tv, bt709), 1920x1080, 100 kb/s, 29.97 fps, 29.97 tbr, 600 tbn
    Side data:
      displaymatrix: rotation of -90.00 degrees
"#;
        let p = parse_probe(s);
        assert_eq!(p.rotation, 90);
        assert_eq!(p.display_size(), (1080, 1920));
        assert_eq!(p.fps, Some(29.97));
        assert!(!p.has_audio());
    }

    #[test]
    fn counter_clockwise_and_old_rotate_tag() {
        let s = "  Stream #0:0: Video: h264, yuv420p, 640x360, 25 tbr\n    Side data:\n      displaymatrix: rotation of 90.00 degrees\n";
        assert_eq!(parse_probe(s).rotation, 270);
        assert_eq!(parse_probe(s).fps, Some(25.0));
        let old = "  Stream #0:0: Video: h264, yuv420p, 640x360, 25 fps\n    Metadata:\n      rotate          : 180\n";
        assert_eq!(parse_probe(old).rotation, 180);
    }

    /// Real output of ffmpeg 9.0.2 for a clip made with `-display_rotation 90`.
    #[test]
    fn ffmpeg_9_display_matrix_spelling() {
        let s = "  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709/unknown/unknown, progressive), 640x360 [SAR 1:1 DAR 16:9], 8 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Side data:
      Display Matrix: rotation of 90.00 degrees
";
        let p = parse_probe(s);
        assert_eq!(p.rotation, 270);
        assert_eq!(p.display_size(), (360, 640));
    }

    #[test]
    fn audio_only_and_no_duration() {
        let s = "  Duration: N/A, bitrate: N/A\n  Stream #0:0: Audio: opus, 48000 Hz, mono, fltp\n      title           : Voice\n";
        let p = parse_probe(s);
        assert!(!p.has_video);
        assert_eq!(p.duration_secs, None);
        assert_eq!(p.audio_tracks[0].title.as_deref(), Some("Voice"));
    }

    #[test]
    fn second_video_stream_does_not_replace_the_first_and_output_stops_parsing() {
        let s = "  Stream #0:0: Video: h264, yuv420p, 1280x720, 30 fps\n  Stream #0:1: Video: mjpeg, yuvj420p, 300x300, 90k tbr (attached pic)\nOutput #0, null:\n  Stream #0:0: Video: wrapped_avframe, 1x1, 1 fps\n";
        let p = parse_probe(s);
        assert_eq!((p.width, p.height), (1280, 720));
        assert_eq!(p.fps, Some(30.0));
    }

    #[test]
    fn cover_art_alone_is_audio_only() {
        let s = "  Duration: 00:03:00.00, bitrate: 192 kb/s\n  Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 192 kb/s\n  Stream #0:1: Video: mjpeg (Baseline), yuvj420p(pc, bt470bg/unknown/unknown), 500x500 [SAR 96:96 DAR 1:1], 90k tbr, 90k tbn (attached pic)\n";
        let p = parse_probe(s);
        assert!(!p.has_video);
        assert!(p.has_audio());
    }

    #[test]
    fn cover_art_before_the_video_is_skipped() {
        let s = "  Stream #0:0: Video: mjpeg, yuvj420p, 500x500, 90k tbr (attached pic)\n  Stream #0:1: Video: h264, yuv420p, 1280x720, 25 fps\n";
        let p = parse_probe(s);
        assert!(p.has_video);
        assert_eq!((p.width, p.height), (1280, 720));
    }

    fn video(width: u32, height: u32, fps: Option<f64>, rotation: u32) -> Probe {
        Probe {
            width,
            height,
            fps,
            rotation,
            has_video: true,
            ..Probe::default()
        }
    }

    #[test]
    fn output_is_capped_at_1080p30_and_keeps_proportions() {
        assert_eq!(
            output_format(&video(1920, 1080, Some(60.0), 0)),
            (1920, 1080, 30.0)
        );
        assert_eq!(
            output_format(&video(3840, 2160, Some(24.0), 0)),
            (1920, 1080, 24.0)
        );
        // portrait after rotation: the cap applies either way round
        assert_eq!(
            output_format(&video(3840, 2160, Some(30.0), 90)),
            (1080, 1920, 30.0)
        );
        // ultrawide: the long side decides
        assert_eq!(
            output_format(&video(3440, 1440, None, 0)),
            (1920, 804, 30.0)
        );
    }

    #[test]
    fn output_sizes_are_even_and_never_empty() {
        assert_eq!(
            output_format(&video(641, 361, Some(25.0), 0)),
            (640, 360, 25.0)
        );
        let (w, h, _) = output_format(&video(1, 1, Some(25.0), 0));
        assert_eq!((w, h), (2, 2));
        let (w, h, _) = output_format(&video(1920, 2, Some(25.0), 0));
        assert_eq!((w, h), (1920, 2));
        assert_eq!(output_format(&video(0, 0, None, 0)), (1280, 720, 30.0));
    }

    #[test]
    fn odd_rates_fall_back() {
        assert_eq!(output_format(&video(640, 360, Some(0.0), 0)).2, 30.0);
        assert_eq!(output_format(&video(640, 360, Some(f64::NAN), 0)).2, 30.0);
    }
}
