//! What an export makes, and the ffmpeg command for it. Kept apart from the dialog so it can be tested on its own.

use std::path::{Path, PathBuf};

use engine::cuts::{self, Cut};
use engine::style::SubtitleStyle;

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
pub enum SubtitleFormat {
    Srt,
    Vtt,
    Txt,
    /// Keeps the text style: font, colours, box and position.
    Ass,
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
    /// How drawn subtitles, a styled track and an ASS file look.
    pub style: SubtitleStyle,
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
            style: SubtitleStyle::default(),
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
            // an ASS track keeps the text style in players that show it
            Container::Mkv => "ass",
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

impl SubtitleFormat {
    pub fn extension(self) -> &'static str {
        match self {
            SubtitleFormat::Srt => "srt",
            SubtitleFormat::Vtt => "vtt",
            SubtitleFormat::Txt => "txt",
            SubtitleFormat::Ass => "ass",
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

    /// The subtitle file ffmpeg reads for this export, by its ending: styled ASS where the style is used, plain SRT
    /// for a track in a container that only takes plain text. `None` when no subtitles go in.
    pub fn subtitle_file(&self) -> Option<&'static str> {
        match (self.kind, self.subtitles) {
            (Kind::Video, SubtitleMode::Burn) => Some("ass"),
            (Kind::Video, SubtitleMode::Track) if self.container == Container::Mkv => Some("ass"),
            (Kind::Video, SubtitleMode::Track) => Some("srt"),
            _ => None,
        }
    }

    /// The text style is used by what this export makes.
    pub fn uses_style(&self) -> bool {
        self.subtitle_file() == Some("ass")
            || (self.kind == Kind::Subtitles && self.subtitle_format == SubtitleFormat::Ass)
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
                    // the ASS file carries the whole style
                    chain.push(format!("subtitles={file}"));
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
                self.video_output(&mut a, video);
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
                self.audio_output(&mut a);
            }
        }
        a.extend(["-progress", "pipe:1", "-nostats"].map(String::from));
        a.push(out.to_string_lossy().into_owned());
        a
    }

    /// Encoder, pixel format, sound codec and container of a video export.
    fn video_output(&self, a: &mut Vec<String>, video: &[String]) {
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

    /// Codec and container of a sound-only export.
    fn audio_output(&self, a: &mut Vec<String>) {
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

    /// The ffmpeg command for a combination: the recordings one after another, each with its own cuts, as one
    /// file. Every picture is scaled into `frame` (bars where its shape differs) at one frame rate, and every sound
    /// made 48 kHz stereo, since joining needs them all alike; a recording without sound gets silence. Subtitles
    /// (`subtitles`, timed on the joined result) are drawn after joining.
    pub fn sequence_ffmpeg_args(
        &self,
        inputs: &[SeqInput],
        frame: Frame,
        subtitles: Option<&str>,
        video: &[String],
        out: &Path,
    ) -> Vec<String> {
        let with_video = self.kind == Kind::Video;
        let mut a: Vec<String> = ["-hide_banner", "-nostdin", "-y"]
            .map(String::from)
            .to_vec();
        for input in inputs {
            a.extend(["-i".to_string(), input.media.to_string_lossy().into_owned()]);
        }
        let track = subtitles.filter(|_| with_video && self.subtitles == SubtitleMode::Track);
        if let Some(file) = track {
            a.extend(["-i".to_string(), file.to_string()]);
        }
        let range = |start: f64, end: Option<f64>| match end {
            Some(end) => format!("start={start:.3}:end={end:.3}"),
            None => format!("start={start:.3}"),
        };
        let Frame { width, height, fps } = frame;
        let mut graph = String::new();
        let mut joined = String::new();
        let mut count = 0;
        for (i, input) in inputs.iter().enumerate() {
            let parts = cuts::kept(&input.cuts, input.duration);
            let n = parts.len();
            if with_video {
                graph.push_str(&format!(
                    "[{i}:v:0]scale={width}:{height}:force_original_aspect_ratio=decrease,\
                     pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p,split={n}"
                ));
                for k in 0..n {
                    graph.push_str(&format!("[vs{i}_{k}]"));
                }
                graph.push(';');
            }
            if input.has_audio {
                graph.push_str(&format!(
                    "[{i}:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,asplit={n}"
                ));
                for k in 0..n {
                    graph.push_str(&format!("[as{i}_{k}]"));
                }
                graph.push(';');
            }
            for (k, (start, end)) in parts.iter().enumerate() {
                if with_video {
                    graph.push_str(&format!(
                        "[vs{i}_{k}]trim={},setpts=PTS-STARTPTS[v{i}_{k}];",
                        range(*start, *end)
                    ));
                    joined.push_str(&format!("[v{i}_{k}]"));
                }
                if input.has_audio {
                    graph.push_str(&format!(
                        "[as{i}_{k}]atrim={},asetpts=PTS-STARTPTS[a{i}_{k}];",
                        range(*start, *end)
                    ));
                } else {
                    let length = end.unwrap_or(input.duration) - start;
                    graph.push_str(&format!(
                        "anullsrc=r=48000:cl=stereo,atrim=duration={length:.3},\
                         aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}_{k}];"
                    ));
                }
                joined.push_str(&format!("[a{i}_{k}]"));
                count += 1;
            }
        }
        graph.push_str(&joined);
        graph.push_str(&format!("concat=n={count}:v={}:a=1", u8::from(with_video)));
        if with_video {
            graph.push_str("[cv][a];");
            match subtitles.filter(|_| self.subtitles == SubtitleMode::Burn) {
                Some(file) => graph.push_str(&format!("[cv]subtitles={file}[v]")),
                None => graph.push_str("[cv]null[v]"),
            }
        } else {
            graph.push_str("[a]");
        }
        a.extend(["-filter_complex".to_string(), graph]);
        if with_video {
            a.extend(["-map", "[v]", "-map", "[a]"].map(String::from));
            if track.is_some() {
                a.extend([
                    "-map".to_string(),
                    format!("{}:0", inputs.len()),
                    "-c:s".to_string(),
                    self.container.subtitle_codec().to_string(),
                ]);
            }
            self.video_output(&mut a, video);
        } else {
            a.extend(["-map", "[a]"].map(String::from));
            self.audio_output(&mut a);
        }
        a.extend(["-progress", "pipe:1", "-nostats"].map(String::from));
        a.push(out.to_string_lossy().into_owned());
        a
    }
}

/// One recording of a combination, as its export sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct SeqInput {
    pub media: PathBuf,
    /// Its length in seconds.
    pub duration: f64,
    /// [`cuts::normalize`]d, and empty when the cuts aren't applied.
    pub cuts: Vec<Cut>,
    pub has_audio: bool,
}

/// The picture of a combination: every recording is fitted into it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
}

impl Frame {
    /// The first recording's size and rate, or its proportions at the chosen height. As for one recording, a
    /// smaller picture is never made larger. `first` is `(width, height, fps)`; zeros mean unknown.
    pub fn for_export(first: (u32, u32, Option<f64>), resolution: Resolution) -> Frame {
        let (w, h) = match (first.0, first.1) {
            (0, _) | (_, 0) => (1280, 720),
            size => size,
        };
        let height = resolution.height().map_or(h, |most| most.min(h));
        let even = |v: f64| ((v.round() as u32) & !1).max(2);
        Frame {
            width: even(w as f64 * height as f64 / h as f64),
            height: even(height as f64),
            fps: first
                .2
                .filter(|f| f.is_finite() && *f > 0.0)
                .unwrap_or(30.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn joined(args: &[String]) -> String {
        args.join(" ")
    }

    fn seq_inputs() -> Vec<SeqInput> {
        vec![
            SeqInput {
                media: PathBuf::from("a.mp4"),
                duration: 10.0,
                cuts: vec![Cut {
                    start: 2.0,
                    end: 4.0,
                }],
                has_audio: true,
            },
            SeqInput {
                media: PathBuf::from("b.mov"),
                duration: 5.0,
                cuts: Vec::new(),
                has_audio: false,
            },
        ]
    }

    const FRAME: Frame = Frame {
        width: 1280,
        height: 720,
        fps: 30.0,
    };

    #[test]
    fn a_combination_joins_every_kept_part_of_every_recording() {
        let o = ExportOptions::default();
        let args =
            o.sequence_ffmpeg_args(&seq_inputs(), FRAME, Some("s.ass"), &[], Path::new("o.mp4"));
        let all = joined(&args);
        assert!(
            all.starts_with("-hide_banner -nostdin -y -i a.mp4 -i b.mov -filter_complex "),
            "{all}"
        );
        let graph = &args[args.iter().position(|a| a == "-filter_complex").unwrap() + 1];
        // both fitted into one frame and rate, the cut recording split in two parts
        assert!(graph.contains("[0:v:0]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p,split=2[vs0_0][vs0_1];"), "{graph}");
        assert!(
            graph.contains("[vs0_0]trim=start=0.000:end=2.000"),
            "{graph}"
        );
        assert!(graph.contains("[vs0_1]trim=start=4.000,"), "{graph}");
        assert!(graph.contains("[1:v:0]scale=1280:720"), "{graph}");
        // the silent recording gets silence of its own length
        assert!(
            graph.contains("anullsrc=r=48000:cl=stereo,atrim=duration=5.000"),
            "{graph}"
        );
        assert!(!graph.contains("[1:a:0]"), "{graph}");
        assert!(graph.contains("[v0_0][a0_0][v0_1][a0_1][v1_0][a1_0]concat=n=3:v=1:a=1[cv][a];[cv]subtitles=s.ass[v]"), "{graph}");
        assert!(all.contains("-map [v] -map [a]"), "{all}");
        assert!(all.ends_with("-progress pipe:1 -nostats o.mp4"), "{all}");
    }

    #[test]
    fn a_combination_can_be_just_its_sound_or_carry_a_subtitle_track() {
        let sound = ExportOptions {
            kind: Kind::Audio,
            ..ExportOptions::default()
        };
        let args = sound.sequence_ffmpeg_args(&seq_inputs(), FRAME, None, &[], Path::new("o.mp3"));
        let all = joined(&args);
        assert!(!all.contains(":v:0"), "no picture at all: {all}");
        assert!(
            all.contains("concat=n=3:v=0:a=1[a] -map [a] -vn -c:a libmp3lame"),
            "{all}"
        );

        let track = ExportOptions {
            subtitles: SubtitleMode::Track,
            ..ExportOptions::default()
        };
        let all = joined(&track.sequence_ffmpeg_args(
            &seq_inputs(),
            FRAME,
            Some("s.srt"),
            &[],
            Path::new("o.mp4"),
        ));
        assert!(all.contains("-i b.mov -i s.srt"), "{all}");
        assert!(all.contains("[cv]null[v]"), "a track is not drawn: {all}");
        assert!(all.contains("-map 2:0 -c:s mov_text"), "{all}");
    }

    #[test]
    fn a_combination_takes_the_first_recordings_shape_and_never_enlarges_it() {
        let frame = |first, r| Frame::for_export(first, r);
        assert_eq!(
            frame((1920, 1080, Some(60.0)), Resolution::Original),
            Frame {
                width: 1920,
                height: 1080,
                fps: 60.0
            }
        );
        assert_eq!(
            frame((1920, 1080, Some(60.0)), Resolution::P720),
            Frame {
                width: 1280,
                height: 720,
                fps: 60.0
            }
        );
        assert_eq!(
            frame((640, 480, None), Resolution::P1080),
            Frame {
                width: 640,
                height: 480,
                fps: 30.0
            }
        );
        assert_eq!(
            frame((1080, 1920, Some(30.0)), Resolution::P720).width,
            404,
            "upright, kept even"
        );
        assert_eq!(
            frame((0, 0, None), Resolution::Original),
            Frame {
                width: 1280,
                height: 720,
                fps: 30.0
            }
        );
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
            ..ExportOptions::default()
        };
        let cuts = [Cut {
            start: 10.0,
            end: 20.0,
        }];
        let args = o.ffmpeg_args(
            Path::new("in.mp4"),
            Some("s.ass"),
            (100.0, &cuts),
            &[],
            Path::new("out.part"),
        );
        let all = joined(&args);
        assert!(
            all.contains("-filter_complex [0:v:0]subtitles=s.ass,scale=-2:'min(ih,720)',split=2"),
            "{all}"
        );
        assert!(all.contains("-map [v] -map [a]"));
        assert!(all.ends_with("-f mp4 -progress pipe:1 -nostats out.part"));
        assert!(!all.contains("-vf"));
    }

    #[test]
    fn a_subtitle_track_is_a_second_input_in_the_container_s_own_format() {
        for (container, codec) in [
            (Container::Mp4, "mov_text"),
            (Container::Mkv, "ass"),
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
    fn the_style_is_used_where_it_can_show_and_plain_text_elsewhere() {
        let at = |subtitles, container| ExportOptions {
            subtitles,
            container,
            ..ExportOptions::default()
        };
        assert_eq!(
            at(SubtitleMode::Burn, Container::Mp4).subtitle_file(),
            Some("ass")
        );
        assert_eq!(
            at(SubtitleMode::Track, Container::Mkv).subtitle_file(),
            Some("ass")
        );
        assert_eq!(
            at(SubtitleMode::Track, Container::Mp4).subtitle_file(),
            Some("srt")
        );
        assert_eq!(at(SubtitleMode::Off, Container::Mp4).subtitle_file(), None);
        assert!(!at(SubtitleMode::Track, Container::Mp4).uses_style());
        let ass_file = ExportOptions {
            kind: Kind::Subtitles,
            subtitle_format: SubtitleFormat::Ass,
            ..ExportOptions::default()
        };
        assert!(ass_file.uses_style());
        assert_eq!(ass_file.extension(), "ass");
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
