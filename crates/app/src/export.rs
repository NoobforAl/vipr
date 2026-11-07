//! What an export makes, and the ffmpeg command for it. Kept apart from the dialog so it can be tested on its own.

use std::path::Path;

use engine::cuts::{self, Cut};

/// What comes out of an export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Video,
    Subtitles,
    Audio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    Mp4,
    Mkv,
    Mov,
    Webm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    H264,
    H265,
    Av1,
    Vp9,
}

/// What does the encoding work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoder {
    /// The graphics card when it can, the processor otherwise.
    Auto,
    Gpu,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    High,
    Balanced,
    Small,
}

/// The picture's height at most. The width follows, and a smaller picture is never made larger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
    Original,
    P1080,
    P720,
    P480,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleMode {
    /// Drawn onto the picture: every player shows them, nobody can turn them off.
    Burn,
    /// A subtitle track in the file: players can turn it on and off.
    Track,
    Off,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextSize {
    Small,
    Medium,
    Large,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleFormat {
    Srt,
    Vtt,
    Txt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Mp3,
    M4a,
    Wav,
    Flac,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExportOptions {
    pub kind: Kind,
    pub container: Container,
    pub codec: Codec,
    pub encoder: Encoder,
    pub quality: Quality,
    pub resolution: Resolution,
    pub subtitles: SubtitleMode,
    /// A translation's language code. `None` is the original text.
    pub lang: Option<String>,
    pub text_size: TextSize,
    pub apply_cuts: bool,
    pub subtitle_format: SubtitleFormat,
    pub audio_format: AudioFormat,
}

impl Default for ExportOptions {
    fn default() -> Self {
        ExportOptions {
            kind: Kind::Video,
            container: Container::Mp4,
            codec: Codec::H264,
            encoder: Encoder::Auto,
            quality: Quality::Balanced,
            resolution: Resolution::Original,
            subtitles: SubtitleMode::Burn,
            lang: None,
            text_size: TextSize::Medium,
            apply_cuts: true,
            subtitle_format: SubtitleFormat::Srt,
            audio_format: AudioFormat::Mp3,
        }
    }
}

impl Container {
    pub const ALL: [Container; 4] = [
        Container::Mp4,
        Container::Mkv,
        Container::Mov,
        Container::Webm,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Container::Mp4 => "MP4",
            Container::Mkv => "MKV",
            Container::Mov => "MOV",
            Container::Webm => "WebM",
        }
    }

    fn extension(self) -> &'static str {
        match self {
            Container::Mp4 => "mp4",
            Container::Mkv => "mkv",
            Container::Mov => "mov",
            Container::Webm => "webm",
        }
    }

    fn ffmpeg_format(self) -> &'static str {
        match self {
            Container::Mp4 => "mp4",
            Container::Mkv => "matroska",
            Container::Mov => "mov",
            Container::Webm => "webm",
        }
    }

    /// The video codecs this kind of file can hold, as players expect them.
    pub fn codecs(self) -> &'static [Codec] {
        match self {
            Container::Mp4 | Container::Mov => &[Codec::H264, Codec::H265, Codec::Av1],
            Container::Mkv => &[Codec::H264, Codec::H265, Codec::Av1, Codec::Vp9],
            Container::Webm => &[Codec::Vp9, Codec::Av1],
        }
    }

    fn subtitle_codec(self) -> &'static str {
        match self {
            Container::Mp4 | Container::Mov => "mov_text",
            Container::Mkv => "srt",
            Container::Webm => "webvtt",
        }
    }
}

impl Codec {
    pub fn label(self) -> &'static str {
        match self {
            Codec::H264 => "H.264",
            Codec::H265 => "H.265 (HEVC)",
            Codec::Av1 => "AV1",
            Codec::Vp9 => "VP9",
        }
    }

    /// The NVIDIA encoder, when there is one. Not every card has every one: AV1 needs an RTX 40 or newer.
    pub fn gpu_encoder(self) -> Option<&'static str> {
        match self {
            Codec::H264 => Some("h264_nvenc"),
            Codec::H265 => Some("hevc_nvenc"),
            Codec::Av1 => Some("av1_nvenc"),
            Codec::Vp9 => None,
        }
    }

    /// The quality setting for High, Balanced and Small: lower is better and bigger.
    fn quality(self, q: Quality) -> u32 {
        let [high, balanced, small] = match self {
            Codec::H264 => [18, 21, 26],
            Codec::H265 => [20, 24, 28],
            Codec::Av1 => [24, 32, 40],
            Codec::Vp9 => [26, 33, 40],
        };
        match q {
            Quality::High => high,
            Quality::Balanced => balanced,
            Quality::Small => small,
        }
    }
}

impl Resolution {
    fn height(self) -> Option<u32> {
        match self {
            Resolution::Original => None,
            Resolution::P1080 => Some(1080),
            Resolution::P720 => Some(720),
            Resolution::P480 => Some(480),
        }
    }
}

impl TextSize {
    fn font_size(self) -> u32 {
        match self {
            TextSize::Small => 13,
            TextSize::Medium => 16,
            TextSize::Large => 20,
        }
    }
}

impl SubtitleFormat {
    pub fn extension(self) -> &'static str {
        match self {
            SubtitleFormat::Srt => "srt",
            SubtitleFormat::Vtt => "vtt",
            SubtitleFormat::Txt => "txt",
        }
    }
}

impl AudioFormat {
    fn extension(self) -> &'static str {
        match self {
            AudioFormat::Mp3 => "mp3",
            AudioFormat::M4a => "m4a",
            AudioFormat::Wav => "wav",
            AudioFormat::Flac => "flac",
        }
    }
}

impl ExportOptions {
    /// The file name ending for what this makes.
    pub fn extension(&self) -> &'static str {
        match self.kind {
            Kind::Video => self.container.extension(),
            Kind::Subtitles => self.subtitle_format.extension(),
            Kind::Audio => self.audio_format.extension(),
        }
    }

    /// Settles choices that don't go together after one of them changed: a codec the container can't hold, and the
    /// graphics card for a codec it has no encoder for.
    pub fn fix(&mut self) {
        if !self.container.codecs().contains(&self.codec) {
            self.codec = self.container.codecs()[0];
        }
        if self.encoder == Encoder::Gpu && self.codec.gpu_encoder().is_none() {
            self.encoder = Encoder::Auto;
        }
    }

    /// The video encoder settings to try, in order: an attempt that fails moves on to the next.
    pub fn video_attempts(&self) -> Vec<Vec<String>> {
        if self.kind != Kind::Video {
            return vec![Vec::new()];
        }
        let q = self.codec.quality(self.quality).to_string();
        let tag =
            matches!(self.container, Container::Mp4 | Container::Mov) && self.codec == Codec::H265;
        let finish = |mut args: Vec<&str>| -> Vec<String> {
            if tag {
                // what Apple players need to show H.265
                args.extend(["-tag:v", "hvc1"]);
            }
            args.into_iter().map(String::from).collect()
        };
        let gpu = self
            .codec
            .gpu_encoder()
            .map(|name| finish(vec!["-c:v", name, "-preset", "p5", "-cq", &q]));
        let cpu = finish(match self.codec {
            Codec::H264 => vec!["-c:v", "libx264", "-preset", "veryfast", "-crf", &q],
            Codec::H265 => vec!["-c:v", "libx265", "-preset", "fast", "-crf", &q],
            Codec::Av1 => vec!["-c:v", "libsvtav1", "-preset", "8", "-crf", &q],
            Codec::Vp9 => vec![
                "-c:v",
                "libvpx-vp9",
                "-crf",
                &q,
                "-b:v",
                "0",
                "-row-mt",
                "1",
                "-deadline",
                "good",
                "-cpu-used",
                "4",
            ],
        });
        match self.encoder {
            Encoder::Auto => gpu.into_iter().chain([cpu]).collect(),
            Encoder::Gpu => gpu.into_iter().collect(),
            Encoder::Cpu => vec![cpu],
        }
    }

    /// The whole ffmpeg command after the program name, for one attempt. `subtitles` is the subtitle file's name in the
    /// folder ffmpeg runs in. `cuts` are [`cuts::normalize`]d and already empty when the cuts aren't applied.
    pub fn ffmpeg_args(
        &self,
        media: &Path,
        subtitles: Option<&str>,
        (duration, cuts): (f64, &[Cut]),
        video: &[String],
        out: &Path,
    ) -> Vec<String> {
        let mut a: Vec<String> = ["-hide_banner", "-nostdin", "-y", "-i"]
            .map(String::from)
            .to_vec();
        a.push(media.to_string_lossy().into_owned());
        let parts = cuts::kept(cuts, duration);
        match self.kind {
            Kind::Video => {
                let track = subtitles.filter(|_| self.subtitles == SubtitleMode::Track);
                if let Some(file) = track {
                    a.extend(["-i".to_string(), file.to_string()]);
                }
                let mut chain = Vec::new();
                if let Some(file) = subtitles.filter(|_| self.subtitles == SubtitleMode::Burn) {
                    chain.push(format!(
                        "subtitles={file}:force_style='FontName=Arial,FontSize={},Outline=2,Shadow=1,MarginV=20'",
                        self.text_size.font_size()
                    ));
                }
                if let Some(h) = self.resolution.height() {
                    chain.push(format!("scale=-2:'min(ih,{h})'"));
                }
                if cuts.is_empty() {
                    a.extend(["-map", "0:v:0", "-map", "0:a?"].map(String::from));
                    if !chain.is_empty() {
                        a.extend(["-vf".to_string(), chain.join(",")]);
                    }
                } else {
                    let draw = if chain.is_empty() {
                        "null".to_string()
                    } else {
                        chain.join(",")
                    };
                    a.extend([
                        "-filter_complex".to_string(),
                        cuts::filter_complex(Some(&draw), &parts, true),
                        "-map".to_string(),
                        "[v]".to_string(),
                        "-map".to_string(),
                        "[a]".to_string(),
                    ]);
                }
                if track.is_some() {
                    a.extend(
                        ["-map", "1:0", "-c:s", self.container.subtitle_codec()].map(String::from),
                    );
                }
                a.extend(video.iter().cloned());
                a.extend(["-pix_fmt", "yuv420p"].map(String::from));
                a.extend(
                    match self.container {
                        Container::Webm => ["-c:a", "libopus", "-b:a", "128k"],
                        _ => ["-c:a", "aac", "-b:a", "192k"],
                    }
                    .map(String::from),
                );
                if matches!(self.container, Container::Mp4 | Container::Mov) {
                    a.extend(["-movflags", "+faststart"].map(String::from));
                }
                a.extend(["-f", self.container.ffmpeg_format()].map(String::from));
            }
            Kind::Audio | Kind::Subtitles => {
                if cuts.is_empty() {
                    a.extend(["-map", "0:a:0"].map(String::from));
                } else {
                    a.extend([
                        "-filter_complex".to_string(),
                        cuts::filter_complex(None, &parts, true),
                        "-map".to_string(),
                        "[a]".to_string(),
                    ]);
                }
                a.push("-vn".to_string());
                let (codec, format): (&[&str], &str) = match self.audio_format {
                    AudioFormat::Mp3 => (&["-c:a", "libmp3lame", "-q:a", "2"], "mp3"),
                    AudioFormat::M4a => (&["-c:a", "aac", "-b:a", "192k"], "ipod"),
                    AudioFormat::Wav => (&["-c:a", "pcm_s16le"], "wav"),
                    AudioFormat::Flac => (&["-c:a", "flac"], "flac"),
                };
                a.extend(codec.iter().map(|s| s.to_string()));
                a.extend(["-f", format].map(String::from));
            }
        }
        a.extend(["-progress", "pipe:1", "-nostats"].map(String::from));
        a.push(out.to_string_lossy().into_owned());
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn joined(args: &[String]) -> String {
        args.join(" ")
    }

    #[test]
    fn a_codec_the_container_cant_hold_is_swapped_and_the_gpu_only_kept_when_it_can_work() {
        let mut o = ExportOptions {
            codec: Codec::H264,
            container: Container::Webm,
            encoder: Encoder::Gpu,
            ..ExportOptions::default()
        };
        o.fix();
        assert_eq!(o.codec, Codec::Vp9);
        assert_eq!(o.encoder, Encoder::Auto, "VP9 has no graphics card encoder");
        let mut keep = ExportOptions {
            container: Container::Mkv,
            codec: Codec::H265,
            ..ExportOptions::default()
        };
        keep.fix();
        assert_eq!(keep.codec, Codec::H265);
    }

    #[test]
    fn auto_tries_the_graphics_card_first_and_the_processor_after() {
        let auto = ExportOptions::default().video_attempts();
        assert_eq!(auto.len(), 2);
        assert!(joined(&auto[0]).contains("h264_nvenc"));
        assert!(joined(&auto[1]).contains("libx264"));
        let cpu = ExportOptions {
            encoder: Encoder::Cpu,
            ..ExportOptions::default()
        };
        assert_eq!(cpu.video_attempts().len(), 1);
        let gpu_h265_mp4 = ExportOptions {
            encoder: Encoder::Gpu,
            codec: Codec::H265,
            ..ExportOptions::default()
        }
        .video_attempts();
        assert_eq!(gpu_h265_mp4.len(), 1);
        assert!(joined(&gpu_h265_mp4[0]).contains("hevc_nvenc"));
        assert!(joined(&gpu_h265_mp4[0]).contains("hvc1"));
    }

    #[test]
    fn quality_changes_the_number_in_the_right_direction() {
        let q = |quality| {
            let o = ExportOptions {
                quality,
                encoder: Encoder::Cpu,
                ..ExportOptions::default()
            };
            let args = &o.video_attempts()[0];
            args[args.iter().position(|a| a == "-crf").unwrap() + 1]
                .parse::<u32>()
                .unwrap()
        };
        assert!(q(Quality::High) < q(Quality::Balanced));
        assert!(q(Quality::Balanced) < q(Quality::Small));
    }

    #[test]
    fn drawn_subtitles_scaling_and_cuts_go_into_one_graph() {
        let o = ExportOptions {
            resolution: Resolution::P720,
            text_size: TextSize::Large,
            ..ExportOptions::default()
        };
        let cuts = [Cut {
            start: 10.0,
            end: 20.0,
        }];
        let args = o.ffmpeg_args(
            Path::new("in.mp4"),
            Some("s.srt"),
            (100.0, &cuts),
            &[],
            Path::new("out.part"),
        );
        let all = joined(&args);
        assert!(all.contains(
            "-filter_complex [0:v:0]subtitles=s.srt:force_style='FontName=Arial,FontSize=20"
        ));
        assert!(all.contains(",scale=-2:'min(ih,720)',split=2"));
        assert!(all.contains("-map [v] -map [a]"));
        assert!(all.ends_with("-f mp4 -progress pipe:1 -nostats out.part"));
        assert!(!all.contains("-vf"));
    }

    #[test]
    fn a_subtitle_track_is_a_second_input_in_the_container_s_own_format() {
        for (container, codec) in [
            (Container::Mp4, "mov_text"),
            (Container::Mkv, "srt"),
            (Container::Webm, "webvtt"),
        ] {
            let o = ExportOptions {
                container,
                subtitles: SubtitleMode::Track,
                ..ExportOptions::default()
            };
            let all = joined(&o.ffmpeg_args(
                Path::new("in.mp4"),
                Some("s.srt"),
                (100.0, &[]),
                &[],
                Path::new("o"),
            ));
            assert!(all.contains("-i in.mp4 -i s.srt"), "{all}");
            assert!(all.contains(&format!("-map 1:0 -c:s {codec}")), "{all}");
            assert!(!all.contains("subtitles="), "a track is not drawn");
        }
    }

    #[test]
    fn audio_only_has_no_picture_and_its_own_codec() {
        let o = ExportOptions {
            kind: Kind::Audio,
            audio_format: AudioFormat::Flac,
            ..ExportOptions::default()
        };
        assert_eq!(o.extension(), "flac");
        let all =
            joined(&o.ffmpeg_args(Path::new("in.mp4"), None, (100.0, &[]), &[], Path::new("o")));
        assert!(all.contains("-map 0:a:0 -vn -c:a flac -f flac"), "{all}");
        let cuts = [Cut {
            start: 1.0,
            end: 2.0,
        }];
        let cut = joined(&o.ffmpeg_args(
            Path::new("in.mp4"),
            None,
            (100.0, &cuts),
            &[],
            Path::new("o"),
        ));
        assert!(cut.contains("-map [a] -vn"), "{cut}");
        assert!(!cut.contains("[0:v:0]"), "{cut}");
    }

    #[test]
    fn file_endings_follow_what_is_made() {
        let mut o = ExportOptions::default();
        assert_eq!(o.extension(), "mp4");
        o.container = Container::Webm;
        assert_eq!(o.extension(), "webm");
        o.kind = Kind::Subtitles;
        o.subtitle_format = SubtitleFormat::Vtt;
        assert_eq!(o.extension(), "vtt");
    }
}
