//! Player spike. See README.md for the commands.
#![windows_subsystem = "windows"]

mod probe;
mod session;
mod stats;
mod surface;

use iced::futures::channel::mpsc;
use iced::futures::{SinkExt, Stream, StreamExt};
use iced::widget::{button, column, container, row, shader, slider, space, stack, text};
use iced::window;
use iced::{Color, Element, Length, Size, Subscription, Task};
use serde_json::json;
use session::{Config, Session, Shared};
use stats::Stats;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::Ordering::*;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
struct Args {
    input: PathBuf,
    width: Option<u32>,
    height: Option<u32>,
    fps: Option<f64>,
    track: usize,
    muted: bool,
    no_audio: bool,
    start: f64,
    play_secs: Option<f64>,
    seeks: usize,
    window_test: bool,
    out: PathBuf,
    threads: Option<u32>,
    pause_test: bool,
    wait_after: f64,
    pause_after: Option<f64>,
    theme: Option<iced::Theme>,
    first_frame_delay: f64,
    no_window_bg: bool,
    no_container_bg: bool,
    seed: u64,
}

fn parse_args() -> Args {
    let mut a = Args {
        input: PathBuf::new(),
        width: None,
        height: None,
        fps: None,
        track: 0,
        muted: false,
        no_audio: false,
        start: 0.0,
        play_secs: None,
        seeks: 0,
        window_test: false,
        out: PathBuf::from("spikes/player/out/run"),
        threads: None,
        pause_test: false,
        wait_after: 0.0,
        pause_after: None,
        theme: None,
        first_frame_delay: 0.0,
        no_window_bg: false,
        no_container_bg: false,
        seed: 0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_default();
        match arg.as_str() {
            "--w" => a.width = val().parse().ok(),
            "--h" => a.height = val().parse().ok(),
            "--fps" => a.fps = val().parse().ok(),
            "--track" => a.track = val().parse().unwrap_or(0),
            "--mute" => a.muted = true,
            "--no-audio" => a.no_audio = true,
            "--start" => a.start = val().parse().unwrap_or(0.0),
            "--play-secs" => a.play_secs = val().parse().ok(),
            "--seeks" => a.seeks = val().parse().unwrap_or(0),
            "--window-test" => a.window_test = true,
            "--pause-test" => a.pause_test = true,
            "--pause-after" => a.pause_after = val().parse().ok(),
            "--first-frame-delay" => a.first_frame_delay = val().parse().unwrap_or(0.0),
            "--no-window-bg" => a.no_window_bg = true,
            "--no-container-bg" => a.no_container_bg = true,
            "--seed" => a.seed = val().parse().unwrap_or(0),
            "--theme" => {
                a.theme = match val().as_str() {
                    "light" => Some(iced::Theme::Light),
                    "dark" => Some(iced::Theme::Dark),
                    _ => None,
                }
            }
            "--out" => a.out = PathBuf::from(val()),
            "--threads" => a.threads = val().parse().ok(),
            "--wait-after" => a.wait_after = val().parse().unwrap_or(0.0),
            other => a.input = PathBuf::from(other),
        }
    }
    a
}

/// Same search order idea as `engine::audio::find_ffmpeg`, without depending on the engine crate.
fn find_ffmpeg() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("VIPR_FFMPEG").map(PathBuf::from).filter(|p| p.is_file()) {
        return Some(p);
    }
    let local = std::env::var_os("LOCALAPPDATA").map(PathBuf::from)?;
    let tools = local.join("ViPr/tools/ffmpeg");
    for p in [tools.join("ffmpeg.exe"), tools.join("bin/ffmpeg.exe")] {
        if p.is_file() {
            return Some(p);
        }
    }
    if let Some(p) = std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).map(|d| d.join("ffmpeg.exe")).find(|p| p.is_file())
    }) {
        return Some(p);
    }
    fn find(dir: &Path, depth: usize) -> Option<PathBuf> {
        let direct = dir.join("ffmpeg.exe");
        if direct.is_file() {
            return Some(direct);
        }
        if depth == 0 {
            return None;
        }
        std::fs::read_dir(dir).ok()?.flatten().filter(|e| e.path().is_dir()).find_map(|e| find(&e.path(), depth - 1))
    }
    for pkg in std::fs::read_dir(local.join("Microsoft/WinGet/Packages")).ok()?.flatten() {
        if pkg.file_name().to_string_lossy().to_lowercase().contains("ffmpeg")
            && let Some(p) = find(&pkg.path(), 4)
        {
            return Some(p);
        }
    }
    None
}

fn run_probe(ffmpeg: &Path, input: &Path) -> probe::Probe {
    use std::os::windows::process::CommandExt;
    let out = Command::new(ffmpeg)
        .args(["-hide_banner", "-nostdin", "-i"])
        .arg(input)
        .creation_flags(0x0800_0000)
        .output()
        .expect("run ffmpeg -i");
    probe::parse_probe(&String::from_utf8_lossy(&out.stderr))
}

struct FileLogger(std::sync::Mutex<std::fs::File>);

impl log::Log for FileLogger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Warn
            || m.target().starts_with("iced_wgpu")
            || m.target().starts_with("player_spike")
    }
    fn log(&self, r: &log::Record) {
        use std::io::Write;
        if self.enabled(r.metadata())
            && let Ok(mut f) = self.0.lock()
        {
            let _ = writeln!(f, "{} {}: {}", r.level(), r.target(), r.args());
        }
    }
    fn flush(&self) {}
}

#[derive(Debug, Clone)]
enum Message {
    Tick,
    Seek(f64),
    TogglePause,
    ToggleFullscreen,
    WindowId(Option<window::Id>),
    WindowClosed,
}

enum Stage {
    /// Window, fonts and pipeline start: not part of any measured phase.
    Settle { began: Instant, first_upload: Option<Instant> },
    Play { until: Instant },
    Seeks { done: usize, next_at: Instant },
    PauseTest { step: usize, next_at: Instant },
    Window { step: usize, next_at: Instant, uploaded_at_step: u64, log: Vec<serde_json::Value> },
    Done,
}

struct App {
    shared: Arc<Shared>,
    session: Session,
    args: Args,
    cfg: Config,
    probe: probe::Probe,
    stage: Stage,
    position: f64,
    duration: f64,
    subtitle: String,
    paused: bool,
    fullscreen: bool,
    window: Option<window::Id>,
    lcg: u64,
    finish_at: Option<Instant>,
    started: Instant,
    startup_ms: f64,
}

const WINDOW_STEPS: [&str; 16] = [
    "resize 1280x720",
    "resize 800x450",
    "resize 1600x900",
    "resize 640x900",
    "resize 1280x720",
    "fullscreen",
    "windowed",
    "minimize",
    "restore",
    "fullscreen",
    "minimize",
    "restore",
    "windowed",
    "maximize",
    "unmaximize",
    "resize 1280x720",
];

impl App {
    fn new() -> (Self, Task<Message>) {
        let args = parse_args();
        std::fs::create_dir_all(&args.out).expect("out dir");
        // this is a windows-subsystem program: a panic has no console to go to
        let panic_path = args.out.join("panic.txt");
        std::panic::set_hook(Box::new(move |info| {
            let _ = std::fs::write(&panic_path, info.to_string());
        }));
        if let Ok(f) = std::fs::File::create(args.out.join("app.log")) {
            let logger = Box::leak(Box::new(FileLogger(std::sync::Mutex::new(f))));
            let _ = log::set_logger(logger);
            log::set_max_level(log::LevelFilter::Info);
        }
        let _ = std::fs::write(args.out.join("app.pid"), std::process::id().to_string());
        let _ = std::fs::remove_file(args.out.join("pids.txt"));

        let ffmpeg = find_ffmpeg().expect("ffmpeg not found");
        let probe = run_probe(&ffmpeg, &args.input);
        log::info!(target: "player_spike", "probe: {probe:?}");
        assert!(probe.has_video, "this spike needs a video stream");
        let (dw, dh) = probe.display_size();
        let fps = args.fps.or(probe.fps).unwrap_or(30.0).min(probe.fps.unwrap_or(f64::MAX));
        let even = |v: u32| v & !1;
        let (width, height) = (
            even(args.width.unwrap_or(dw).min(dw.max(2))),
            even(args.height.unwrap_or(dh).min(dh.max(2))),
        );
        let duration = probe.duration_secs.unwrap_or(0.0);
        let cfg = Config {
            ffmpeg,
            input: args.input.clone(),
            width,
            height,
            fps,
            audio_track: args.track,
            has_audio: probe.has_audio(),
            use_audio_device: !args.no_audio,
            muted: args.muted,
            out_dir: args.out.clone(),
            duration,
            threads: args.threads,
            first_frame_delay: Duration::from_secs_f64(args.first_frame_delay),
        };

        let stats = Arc::new(Stats::default());
        stats.spawn_cpu_sampler();
        let shared = session::new_shared(stats, fps);
        let session = Session::start(cfg.clone(), shared.clone(), args.start);
        log::info!(target: "player_spike", "audio: {}", session.audio_desc);

        let stage = Stage::Settle { began: Instant::now(), first_upload: None };
        let app = Self {
            shared,
            session,
            position: args.start,
            duration,
            subtitle: String::new(),
            paused: false,
            fullscreen: false,
            window: None,
            lcg: 0x2545_F491_4F6C_DD1D ^ args.seed,
            finish_at: None,
            started: Instant::now(),
            startup_ms: 0.0,
            stage,
            cfg,
            probe,
            args,
        };
        (app, window::latest().map(Message::WindowId))
    }

    fn next_random(&mut self, lo: f64, hi: f64) -> f64 {
        self.lcg = self.lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = (self.lcg >> 11) as f64 / (1u64 << 53) as f64;
        lo + r * (hi - lo)
    }

    fn write_stats(&self) {
        let (s, p) = (&self.shared.stats, &self.probe);
        let meta = json!({
            "input": self.args.input.file_name().map(|n| n.to_string_lossy().to_string()),
            "ffmpeg": self.cfg.ffmpeg.display().to_string(),
            "video": format!("{}x{} @ {}", self.cfg.width, self.cfg.height, self.cfg.fps),
            "source": format!("{}x{} @ {:?}, rotation {}, {} audio tracks", p.width, p.height, p.fps, p.rotation, p.audio_tracks.len()),
            "audio": self.session.audio_desc,
            "audio_latency_ms_last": self.shared.clock.last_latency_ns.load(Relaxed) as f64 / 1e6,
            "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
            "threads_arg": self.cfg.threads,
            "start_to_first_upload_ms": self.startup_ms,
        });
        let v = s.final_json(meta);
        let _ = std::fs::write(
            self.args.out.join("stats.json"),
            serde_json::to_string_pretty(&v).unwrap_or_default(),
        );
    }

    fn after_play(&mut self) -> Stage {
        let now = Instant::now();
        let clock = self.shared.clock.now();
        if self.args.seeks > 0 {
            self.shared.stats.begin_phase("seek", clock);
            Stage::Seeks { done: 0, next_at: now }
        } else {
            self.after_seeks()
        }
    }

    fn after_seeks(&mut self) -> Stage {
        let now = Instant::now();
        let clock = self.shared.clock.now();
        if self.args.pause_test {
            self.shared.stats.begin_phase("pause", clock);
            Stage::PauseTest { step: 0, next_at: now }
        } else if self.args.window_test {
            self.shared.stats.begin_phase("window", clock);
            Stage::Window { step: 0, next_at: now, uploaded_at_step: 0, log: Vec::new() }
        } else {
            Stage::Done
        }
    }

    fn drive(&mut self) -> Task<Message> {
        let now = Instant::now();
        let clock = self.shared.clock.now();
        if let Some(after) = self.args.pause_after
            && !self.paused
            && self.started.elapsed().as_secs_f64() >= after
        {
            self.paused = true;
            self.args.pause_after = None;
            self.session.set_paused(true);
            log::info!(target: "player_spike", "paused and staying paused");
        }
        match &mut self.stage {
            Stage::Settle { began, first_upload } => {
                if first_upload.is_none() && self.shared.stats.current_uploaded() > 0 {
                    *first_upload = Some(now);
                    self.startup_ms = began.elapsed().as_secs_f64() * 1000.0;
                }
                if first_upload.is_some_and(|t| now >= t + Duration::from_secs(2)) {
                    if let Some(secs) = self.args.play_secs {
                        self.shared.stats.begin_phase("play", clock);
                        self.stage = Stage::Play { until: now + Duration::from_secs_f64(secs) };
                    } else {
                        self.stage = self.after_play();
                    }
                }
                Task::none()
            }
            Stage::Done => {
                if self.args.play_secs.is_none()
                    && self.args.seeks == 0
                    && !self.args.window_test
                    && !self.args.pause_test
                {
                    return Task::none();
                }
                if self.finish_at.is_none() {
                    self.write_stats();
                    self.finish_at = Some(now + Duration::from_secs_f64(self.args.wait_after));
                }
                if self.finish_at.is_some_and(|t| now >= t) {
                    log::info!(target: "player_spike", "script finished");
                    self.session.shutdown();
                    return iced::exit();
                }
                Task::none()
            }
            Stage::Play { until } => {
                if now >= *until {
                    self.shared.stats.end_phase(clock, json!({}));
                    self.write_stats();
                    self.stage = self.after_play();
                }
                Task::none()
            }
            Stage::Seeks { done, next_at } => {
                if now < *next_at {
                    return Task::none();
                }
                let d = *done;
                if d >= self.args.seeks {
                    self.shared.stats.end_phase(clock, json!({}));
                    self.write_stats();
                    self.stage = self.after_seeks();
                    return Task::none();
                }
                *done += 1;
                *next_at = now + Duration::from_secs(5);
                log::info!(target: "player_spike", "scripted seek {d}");
                let hi = (self.duration - 60.0).max(1.0);
                let target = self.next_random(10f64.min(hi), hi);
                // every fourth seek is a slider drag: only the last value may reach ffmpeg
                if d % 4 == 3 {
                    let send = self.session.seek_sender();
                    std::thread::spawn(move || {
                        for i in (0..8).rev() {
                            send(target - i as f64 * 7.0);
                            std::thread::sleep(Duration::from_millis(30));
                        }
                    });
                } else {
                    self.session.seek(target);
                }
                Task::none()
            }
            Stage::PauseTest { step, next_at } => {
                if now < *next_at {
                    return Task::none();
                }
                let s = *step;
                *step += 1;
                *next_at = now + Duration::from_secs(3);
                match s {
                    0 | 2 => {
                        self.paused = true;
                        self.session.set_paused(true);
                    }
                    1 | 3 => {
                        self.paused = false;
                        self.session.set_paused(false);
                    }
                    _ => {
                        self.shared.stats.end_phase(clock, json!({}));
                        self.stage = if self.args.window_test {
                            self.shared.stats.begin_phase("window", clock);
                            Stage::Window { step: 0, next_at: now, uploaded_at_step: 0, log: Vec::new() }
                        } else {
                            Stage::Done
                        };
                    }
                }
                Task::none()
            }
            Stage::Window { step, next_at, uploaded_at_step, log } => {
                if now < *next_at {
                    return Task::none();
                }
                let uploaded = self.shared.stats.current_uploaded();
                if *step > 0 {
                    log.push(json!({
                        "step": WINDOW_STEPS[*step - 1],
                        "uploads_during_step": uploaded - *uploaded_at_step,
                        "device_lost_so_far": self.shared.stats.c.device_lost.load(Relaxed),
                    }));
                }
                *uploaded_at_step = uploaded;
                if *step >= WINDOW_STEPS.len() {
                    let steps = std::mem::take(log);
                    self.shared.stats.end_phase(clock, json!({ "steps": steps }));
                    self.stage = Stage::Done;
                    return Task::none();
                }
                let name = WINDOW_STEPS[*step];
                *step += 1;
                *next_at = now + Duration::from_secs(4);
                let Some(id) = self.window else {
                    return Task::none();
                };
                match name {
                    "fullscreen" => window::set_mode(id, window::Mode::Fullscreen),
                    "windowed" => window::set_mode(id, window::Mode::Windowed),
                    "minimize" => window::minimize(id, true),
                    "restore" => window::minimize(id, false),
                    "maximize" => window::maximize(id, true),
                    "unmaximize" => window::maximize(id, false),
                    other => {
                        let mut parts = other[7..].split('x');
                        let w: f32 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(800.0);
                        let h: f32 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(450.0);
                        window::resize(id, Size::new(w, h))
                    }
                }
            }
        }
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Tick => {
                self.position = self.shared.clock.now();
                let t = self.position as u64;
                self.subtitle = format!(
                    "Sample subtitle overlay, line {} ({:02}:{:02}:{:02})",
                    t / 4,
                    t / 3600,
                    t / 60 % 60,
                    t % 60
                );
                return self.drive();
            }
            Message::Seek(v) => {
                self.position = v;
                self.session.seek(v);
            }
            Message::TogglePause => {
                self.paused = !self.paused;
                self.session.set_paused(self.paused);
            }
            Message::ToggleFullscreen => {
                self.fullscreen = !self.fullscreen;
                if let Some(id) = self.window {
                    let mode = if self.fullscreen { window::Mode::Fullscreen } else { window::Mode::Windowed };
                    return window::set_mode(id, mode);
                }
            }
            Message::WindowId(id) => self.window = id,
            Message::WindowClosed => {
                log::warn!(target: "player_spike", "close requested at {:.1} s after start, ending", self.started.elapsed().as_secs_f64());
                self.shared.stats.end_phase(self.shared.clock.now(), json!({ "closed_early": true }));
                self.write_stats();
                self.session.shutdown();
                return iced::exit();
            }
        }
        Task::none()
    }

    fn view(&self) -> Element<'_, Message> {
        self.shared.stats.c.view_calls.fetch_add(1, Relaxed);
        let video = shader(surface::VideoProgram { shared: self.shared.clone() })
            .width(Length::Fill)
            .height(Length::Fill);
        // Black behind the shader, so the theme background cannot show there whatever the shader does.
        let black_behind = !self.args.no_container_bg;
        let video = container(video).width(Length::Fill).height(Length::Fill).style(move |_| container::Style {
            background: black_behind.then(|| Color::BLACK.into()),
            ..Default::default()
        });
        let has_picture = self.shared.latest.lock().map(|f| f.seq > 0).unwrap_or(false);
        let subtitle = container(text(self.subtitle.clone()).size(28).color(Color::WHITE))
            .padding(8)
            .style(move |_| container::Style {
                background: has_picture.then(|| Color::from_rgba(0.0, 0.0, 0.0, 0.65).into()),
                ..Default::default()
            });
        let overlay = column![
            space().height(Length::Fill),
            container(subtitle).width(Length::Fill).align_x(iced::Center),
            row![
                button(if self.paused { "Play" } else { "Pause" }).on_press(Message::TogglePause),
                slider(0.0..=self.duration.max(1.0), self.position, Message::Seek).step(0.1),
                button("Fullscreen").on_press(Message::ToggleFullscreen),
            ]
            .spacing(10)
            .padding(10)
            .align_y(iced::Center),
        ]
        .width(Length::Fill)
        .height(Length::Fill);
        stack![video, overlay].width(Length::Fill).height(Length::Fill).into()
    }

    fn subscription(&self) -> Subscription<Message> {
        Subscription::batch([
            Subscription::run(tick_stream).map(|_| Message::Tick),
            window::close_requests().map(|_| Message::WindowClosed),
        ])
    }
}

/// 10 Hz: slider, subtitle text and the auto script.
fn tick_stream() -> impl Stream<Item = ()> {
    iced::stream::channel(2, async |mut out: mpsc::Sender<()>| {
        let (tx, mut rx) = mpsc::unbounded::<()>();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_millis(100));
                if tx.unbounded_send(()).is_err() {
                    break;
                }
            }
        });
        while rx.next().await.is_some() {
            if out.send(()).await.is_err() {
                break;
            }
        }
    })
}

fn main() -> iced::Result {
    iced::application(App::new, App::update, App::view)
        .title(|app: &App| {
            if app.args.play_secs.is_some() || app.args.seeks > 0 || app.args.window_test || app.args.pause_test {
                "ViPr player spike - automated test, closes by itself".to_string()
            } else {
                "ViPr player spike".to_string()
            }
        })
        .style(|app: &App, theme: &iced::Theme| {
            let base = iced::theme::Base::base(theme);
            // the first thing the renderer clears to, and what any gap around the surface shows
            iced::theme::Style {
                background_color: if app.args.no_window_bg { base.background_color } else { Color::BLACK },
                text_color: base.text_color,
            }
        })
        .theme(|app: &App| app.args.theme.clone())
        .subscription(App::subscription)
        .window(window::Settings {
            size: Size::new(1280.0, 720.0),
            exit_on_close_request: false,
            ..Default::default()
        })
        .run()
}
