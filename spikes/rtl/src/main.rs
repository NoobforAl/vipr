//! RTL rendering spike. One scrollable page; every row has an id and says what to check.
//!
//!   cargo run                   the test page
//!   cargo run -- --section N    only section N of the page
//!   cargo run -- --report       measured facts from the shaper (no window)
//!   cargo run -- --capture DIR  screenshots of every section except 8, then exit
//!   cargo run -- --carets DIR   scripted caret and selection states in section 8, for an outside
//!                               grabber: writes DIR/request.txt, waits until it is deleted
//!
//! Section 8 needs the outside grabber: in iced 0.14 `window::screenshot` leaves text_editor text
//! out of the image, while the real window shows it.

mod dir;
mod report;

use dir::dir_align;
use iced::alignment::Horizontal;
use iced::widget::text::{Alignment as TextAlign, Wrapping};
use iced::widget::{
    Column, column, container, operation, row, scrollable, text, text_editor, text_input,
};
use iced::{Color, Element, Font, Length::Fill, Size, Task, Theme, window};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Duration;

const LONG_FA: &str = "این یک جملهٔ بلند فارسی است که باید در چند خط شکسته شود تا رفتار تراز متن را در پنجره‌های پهن و باریک ببینیم.";
const LONG_AR: &str = "هذه جملة عربية طويلة يجب أن تنقسم إلى عدة أسطر حتى نتمكن من رؤية سلوك محاذاة النص في النوافذ العريضة والضيقة.";
const LONG_HE: &str = "זהו משפט ארוך בעברית שצריך להישבר למספר שורות כדי שנוכל לראות כיצד הטקסט מיושר בחלונות רחבים וצרים.";

const SCROLL_ID: &str = "page";
const EDITOR_TEXT: &str = "سلام دنیا\nنسخه 5G را API\nHello دنیا\nאבג שלום\nاین یک خط بلند فارسی است که باید در ویرایشگر شکسته شود تا رفتار مکان‌نما در خطوط شکسته‌شده دیده شود.";

fn main() -> iced::Result {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--report") {
        report::run();
        return Ok(());
    }
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
    };
    let shots = value("--capture").map(PathBuf::from);
    let carets = value("--carets").map(PathBuf::from);
    let only = value("--section").and_then(|n| n.parse::<u8>().ok());
    iced::application(
        move || App::new(shots.clone(), carets.clone(), only),
        App::update,
        App::view,
    )
    .title("ViPr RTL spike")
    .theme(|_: &App| Theme::Dark)
    .window(window::Settings {
        size: Size::new(1040.0, 720.0),
        ..Default::default()
    })
    .run()
}

/// Scripted capture steps. One runs, the window settles, the next one runs.
#[derive(Clone)]
enum Step {
    Resize(f32, f32),
    Section(Option<u8>),
    Paged(String),
    Shot(String),
    Focus(&'static str),
    Cursor(&'static str, usize),
    Select(&'static str, usize, usize),
    Editor(fn() -> text_editor::Action),
    Quit,
}

struct App {
    only: Option<u8>,
    input: [String; 4],
    editors: [text_editor::Content; 3],
    fonts: Vec<(&'static str, &'static str, String, usize)>,
    shots: Option<PathBuf>,
    outside: bool,
    steps: VecDeque<Step>,
    window: Option<window::Id>,
    paged: Option<(String, u32, Vec<u8>)>,
    viewport_h: f32,
}

#[derive(Debug, Clone)]
enum Message {
    Input(usize, String),
    Editor(usize, text_editor::Action),
    Window(Option<window::Id>),
    Next,
    PagedShoot,
    PageShot(window::Screenshot),
    Saved(String, window::Screenshot),
}

/// Lets the window draw before the next step. Blocks one pool thread, not the UI.
fn settle(ms: u64, msg: Message) -> Task<Message> {
    Task::perform(
        async move { std::thread::sleep(Duration::from_millis(ms)) },
        move |()| msg.clone(),
    )
}

fn save(dir: &std::path::Path, name: &str, s: &window::Screenshot) {
    let file = std::fs::File::create(dir.join(format!("{name}.png"))).expect("create png");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), s.size.width, s.size.height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .expect("png header")
        .write_image_data(&s.rgba)
        .expect("png data");
}

fn capture_script() -> VecDeque<Step> {
    let mut s = VecDeque::new();
    for (w, h, tag) in [(1040.0, 720.0, "wide"), (560.0, 720.0, "narrow")] {
        s.push_back(Step::Resize(w, h));
        for n in (1..=10u8).filter(|n| *n != 8) {
            s.push_back(Step::Section(Some(n)));
            s.push_back(Step::Paged(format!("{tag}_s{n}")));
        }
    }
    s.push_back(Step::Quit);
    s
}

fn caret_script() -> VecDeque<Step> {
    use text_editor::{Action, Edit, Motion};
    let mut s = VecDeque::new();
    s.push_back(Step::Resize(1040.0, 720.0));
    s.push_back(Step::Section(Some(8)));
    s.push_back(Step::Shot("caret_00_nothing_focused".into()));
    let input_cases: [(&str, &str, usize); 7] = [
        ("input0", "fa_default_pos0", 0),
        ("input0", "fa_default_pos4", 4),
        ("input0", "fa_default_pos9", 9),
        ("input1", "fa_right_pos0", 0),
        ("input1", "fa_right_pos9", 9),
        ("input2", "mixed_pos5", 5),
        ("input2", "mixed_pos14", 14),
    ];
    for (i, (id, name, pos)) in input_cases.into_iter().enumerate() {
        s.push_back(Step::Focus(id));
        s.push_back(Step::Cursor(id, pos));
        s.push_back(Step::Shot(format!("caret_1{i}_input_{name}")));
    }
    s.push_back(Step::Focus("input0"));
    s.push_back(Step::Select("input0", 0, 4));
    s.push_back(Step::Shot("caret_20_input_select_0_4".into()));
    s.push_back(Step::Focus("input1"));
    s.push_back(Step::Select("input1", 0, 4));
    s.push_back(Step::Shot("caret_21_input_right_select_0_4".into()));

    s.push_back(Step::Focus("editor0"));
    let moves: [(&str, fn() -> Action); 9] = [
        ("start", || Action::Move(Motion::DocumentStart)),
        ("right1", || Action::Move(Motion::Right)),
        ("right2", || Action::Move(Motion::Right)),
        ("left1", || Action::Move(Motion::Left)),
        ("end_of_line", || Action::Move(Motion::End)),
        ("home", || Action::Move(Motion::Home)),
        ("select_right", || Action::Select(Motion::Right)),
        ("select_right2", || Action::Select(Motion::Right)),
        ("select_line", || Action::SelectLine),
    ];
    for (i, (name, f)) in moves.into_iter().enumerate() {
        s.push_back(Step::Editor(f));
        s.push_back(Step::Shot(format!("caret_3{i}_editor_{name}")));
    }
    s.push_back(Step::Editor(|| Action::Move(Motion::DocumentStart)));
    s.push_back(Step::Editor(|| Action::Move(Motion::End)));
    s.push_back(Step::Editor(|| Action::Edit(Edit::Insert('پ'))));
    s.push_back(Step::Editor(|| Action::Edit(Edit::Insert('ی'))));
    s.push_back(Step::Shot("caret_40_editor_typed_at_line_end".into()));
    s.push_back(Step::Resize(560.0, 720.0));
    s.push_back(Step::Shot("caret_50_narrow".into()));
    s.push_back(Step::Quit);
    s
}

/// Hands one shot to the outside grabber and waits for it. Blocks one pool thread, not the UI.
fn request_shot(dir: &std::path::Path, name: String) -> Task<Message> {
    let request = dir.join("request.txt");
    Task::perform(
        async move {
            std::fs::write(&request, name).expect("write request");
            while request.exists() {
                std::thread::sleep(Duration::from_millis(50));
            }
        },
        |()| Message::Next,
    )
}

impl App {
    fn new(
        capture: Option<PathBuf>,
        carets: Option<PathBuf>,
        only: Option<u8>,
    ) -> (Self, Task<Message>) {
        let outside = carets.is_some();
        let steps = if outside {
            caret_script()
        } else if capture.is_some() {
            capture_script()
        } else {
            VecDeque::new()
        };
        let shots = carets.or(capture);
        if let Some(dir) = &shots {
            std::fs::create_dir_all(dir).expect("shots dir");
        }
        let task = if shots.is_some() {
            Task::perform(
                async { std::thread::sleep(Duration::from_millis(1200)) },
                |()| (),
            )
            .discard()
            .chain(window::latest().map(Message::Window))
        } else {
            Task::none()
        };
        let app = Self {
            only,
            input: [
                "سلام دنیا".into(),
                "سلام دنیا".into(),
                "نسخه 5G را API".into(),
                String::new(),
            ],
            editors: [
                text_editor::Content::with_text(EDITOR_TEXT),
                text_editor::Content::new(),
                text_editor::Content::with_text("Latin control line\nsecond line"),
            ],
            fonts: report::font_rows(),
            shots,
            outside,
            steps,
            window: None,
            paged: None,
            viewport_h: 720.0,
        };
        (app, task)
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Input(i, v) => self.input[i] = v,
            Message::Editor(i, a) => self.editors[i].perform(a),
            Message::Window(id) => {
                self.window = id;
                return Task::done(Message::Next);
            }
            Message::Next => return self.step(),
            Message::PagedShoot => {
                let id = self.window.expect("window");
                return window::screenshot(id).map(Message::PageShot);
            }
            Message::PageShot(s) => {
                let Some((prefix, page, last)) = self.paged.as_mut() else {
                    return Task::none();
                };
                if *last == s.rgba.as_ref() {
                    self.paged = None;
                    return Task::done(Message::Next);
                }
                last.clear();
                last.extend_from_slice(&s.rgba);
                let name = format!("{prefix}_p{page:02}");
                save(self.shots.as_ref().expect("dir"), &name, &s);
                *page += 1;
                let y = *page as f32 * self.viewport_h * 0.85;
                return operation::scroll_to(SCROLL_ID, scrollable::AbsoluteOffset { x: 0.0, y })
                    .chain(settle(350, Message::PagedShoot));
            }
            Message::Saved(name, s) => {
                save(self.shots.as_ref().expect("dir"), &name, &s);
                return Task::done(Message::Next);
            }
        }
        Task::none()
    }

    fn step(&mut self) -> Task<Message> {
        let Some(step) = self.steps.pop_front() else {
            return Task::none();
        };
        let id = self.window.expect("window");
        let next = settle(250, Message::Next);
        match step {
            Step::Resize(w, h) => {
                self.viewport_h = h;
                window::resize(id, Size::new(w, h)).chain(settle(600, Message::Next))
            }
            Step::Section(n) => {
                self.only = n;
                operation::scroll_to(SCROLL_ID, scrollable::AbsoluteOffset { x: 0.0, y: 0.0 })
                    .chain(next)
            }
            Step::Paged(prefix) => {
                self.paged = Some((prefix, 0, vec![]));
                settle(350, Message::PagedShoot)
            }
            Step::Shot(name) if self.outside => {
                request_shot(self.shots.as_ref().expect("dir"), name)
            }
            Step::Shot(name) => {
                window::screenshot(id).map(move |s| Message::Saved(name.clone(), s))
            }
            Step::Focus(w) => operation::focus(w).chain(next),
            Step::Cursor(w, pos) => {
                operation::move_cursor_to(w, pos).chain(settle(60, Message::Next))
            }
            Step::Select(w, a, b) => operation::select_range(w, a, b).chain(next),
            Step::Editor(f) => {
                self.editors[0].perform(f());
                settle(60, Message::Next)
            }
            Step::Quit => iced::exit(),
        }
    }

    fn view(&self) -> Element<'_, Message> {
        let show = |n: u8| self.only.is_none_or(|o| o == n);
        let mut page = Column::new().spacing(26).padding(16).width(Fill);
        if show(1) {
            page = page.push(section1());
        }
        if show(2) {
            page = page.push(section2());
        }
        if show(3) {
            page = page.push(section3());
        }
        if show(4) {
            page = page.push(section4());
        }
        if show(5) {
            page = page.push(section5());
        }
        if show(6) {
            page = page.push(section6());
        }
        if show(7) {
            page = page.push(section7());
        }
        if show(8) {
            page = page.push(self.section8());
        }
        if show(9) {
            page = page.push(self.section9());
        }
        if show(10) {
            page = page.push(section10());
        }
        scrollable(page)
            .id(SCROLL_ID)
            .width(Fill)
            .height(Fill)
            .into()
    }

    fn section8(&self) -> Element<'_, Message> {
        let input = |i: usize, id: &'static str| {
            text_input("type here", &self.input[i])
                .id(id)
                .size(24)
                .on_input(move |v| Message::Input(i, v))
        };
        let editor = |i: usize, id: &'static str| {
            text_editor(&self.editors[i])
                .id(id)
                .size(22)
                .height(if i == 0 { 200 } else { 90 })
                .on_action(move |a| Message::Editor(i, a))
        };
        section(
            "8. text_input and text_editor with RTL (live: click, arrows, Home/End, shift-select, type; needs a Persian keyboard)",
            vec![
                sample(
                    "8.1",
                    "text_input, default alignment. Does text start at the left edge? Caret where you click? Right arrow moves the way you expect?",
                    input(0, "input0").into(),
                ),
                sample(
                    "8.2",
                    "text_input with .align_x(Right): text at the right edge? Caret still on the text, not off to one side?",
                    input(1, "input1").align_x(TextAlign::Right).into(),
                ),
                sample(
                    "8.3",
                    "text_input, mixed direction. Caret across the Persian/Latin boundaries; select across them.",
                    input(2, "input2").align_x(TextAlign::Right).into(),
                ),
                sample(
                    "8.4",
                    "text_input, empty, right aligned. Type Persian: letters join live? Caret on the correct side of the last letter?",
                    input(3, "input3").align_x(TextAlign::Right).into(),
                ),
                sample(
                    "8.5",
                    "text_editor (no align_x exists). Lines start at the right edge? Caret and selection on the correct letters, also on the wrapped last line?",
                    editor(0, "editor0").into(),
                ),
                sample(
                    "8.6",
                    "text_editor, empty. Type Persian, press Enter, type Latin. Letters join? Each line follows its own direction?",
                    editor(1, "editor1").into(),
                ),
                sample(
                    "8.7",
                    "text_editor, Latin control: is the text visible at all?",
                    editor(2, "editor2").into(),
                ),
            ],
        )
    }

    fn section9(&self) -> Element<'_, Message> {
        let mut rows = vec![];
        for (i, (label, sample_text, faces, notdef)) in self.fonts.iter().enumerate() {
            let body = column![
                rtl(sample_text),
                text(format!("shaper picked: {faces}. Missing glyphs: {notdef}"))
                    .size(13)
                    .color(GREY),
            ]
            .spacing(4);
            rows.push(sample_owned(
                format!("9.{}", i + 1),
                format!("{label}: any box or tofu?"),
                body.into(),
            ));
        }
        section("9. Fonts: no font is bundled in this build", rows)
    }
}

const GREY: Color = Color::from_rgb(0.68, 0.70, 0.74);

fn section<'a>(title: &'a str, rows: Vec<Element<'a, Message>>) -> Element<'a, Message> {
    let mut col = Column::new().spacing(12).width(Fill);
    col = col.push(text(title).size(19).color(Color::from_rgb(0.55, 0.8, 1.0)));
    for r in rows {
        col = col.push(r);
    }
    col.into()
}

fn sample<'a>(id: &'a str, check: &'a str, body: Element<'a, Message>) -> Element<'a, Message> {
    sample_owned(id.to_string(), check.to_string(), body)
}

fn sample_owned<'a>(id: String, check: String, body: Element<'a, Message>) -> Element<'a, Message> {
    column![
        // LRM: a label that quotes an Arabic letter early would otherwise flip its own line
        text(format!("\u{200E}{id}  {check}")).size(13).color(GREY),
        container(body).width(Fill).padding(8).style(frame),
    ]
    .spacing(3)
    .into()
}

fn frame(_: &Theme) -> container::Style {
    container::Style {
        background: Some(Color::from_rgb(0.15, 0.16, 0.19).into()),
        border: iced::Border {
            color: Color::from_rgb(0.32, 0.34, 0.38),
            width: 1.0,
            radius: 0.0.into(),
        },
        ..Default::default()
    }
}

/// Content text aligned by its own direction, through a container. `.align_x(dir_align(s))` on the
/// text itself makes a one-line RTL string vanish in iced 0.14; see section 5 and section 10.
fn rtl(s: &str) -> Element<'_, Message> {
    let t = text(s).size(24).wrapping(Wrapping::WordOrGlyph);
    let h = if dir::first_strong_rtl(s) == Some(true) {
        Horizontal::Right
    } else {
        Horizontal::Left
    };
    container(t).width(Fill).align_x(h).into()
}

fn s<'a>(id: &'a str, check: &'a str, sample_text: &'a str) -> Element<'a, Message> {
    sample(id, check, rtl(sample_text))
}

fn band<'a>(body: Element<'a, Message>) -> Element<'a, Message> {
    container(body)
        .width(Fill)
        .style(|_: &Theme| container::Style {
            background: Some(Color::from_rgb(0.22, 0.24, 0.3).into()),
            ..Default::default()
        })
        .into()
}

/// Workaround A: text keeps Alignment::Default and shrinks; the container places it.
fn via_container(s: &str, h: Horizontal) -> Element<'_, Message> {
    let t = text(s).size(20).wrapping(Wrapping::WordOrGlyph);
    container(t).width(Fill).align_x(h).into()
}

/// Workaround B: LRM + RLI..PDI. The line's base is LTR, so iced's RTL path is never taken, and the
/// content still runs RTL inside the isolate.
fn via_isolate(s: &str, a: TextAlign) -> Element<'static, Message> {
    text(format!("\u{200E}\u{2067}{s}\u{2069}"))
        .size(20)
        .width(Fill)
        .align_x(a)
        .wrapping(Wrapping::WordOrGlyph)
        .into()
}

/// Workaround C: a trailing line break, so iced takes its multi-line path.
fn via_newline(s: &str, a: TextAlign) -> Element<'static, Message> {
    text(format!("{s}\n"))
        .size(20)
        .width(Fill)
        .align_x(a)
        .wrapping(Wrapping::WordOrGlyph)
        .into()
}

fn workaround_rows(s: &'static str) -> Element<'static, Message> {
    let rows: Vec<(&str, Element<'static, Message>)> = vec![
        ("A Right", via_container(s, Horizontal::Right)),
        ("A Center", via_container(s, Horizontal::Center)),
        ("B Right", via_isolate(s, TextAlign::Right)),
        ("B Center", via_isolate(s, TextAlign::Center)),
        ("C Right", via_newline(s, TextAlign::Right)),
        ("C Center", via_newline(s, TextAlign::Center)),
    ];
    let mut col = Column::new().spacing(3);
    for (name, body) in rows {
        col = col.push(row![text(name).size(12).color(GREY).width(64), band(body)].spacing(6));
    }
    col.into()
}

const LONG2_FA: &str = "این یک جملهٔ بلند فارسی است که باید در چند خط شکسته شود تا رفتار تراز متن را در پنجره‌های پهن و باریک ببینیم. جملهٔ دوم نسخه 5G را با API می‌سنجد و آخرین خط باید کوتاه باشد.";

fn section10<'a>() -> Element<'a, Message> {
    section(
        "10. Workarounds for section 5. A: container places the text. B: LRM + RLI..PDI. C: trailing newline",
        vec![
            sample(
                "10.1",
                "Short Persian. Each row at the side it names, nothing missing or extra-tall",
                workaround_rows("سلام دنیا"),
            ),
            sample(
                "10.2",
                "Mixed line, same word order as 2.1 in every row",
                workaround_rows("نسخه 5G را API"),
            ),
            sample(
                "10.3",
                "Wrapped. Sentence starts on the top line at the right, lines read downward, no words swapped between lines. A Center: lines are right-aligned inside a centred block",
                workaround_rows(LONG2_FA),
            ),
            sample(
                "10.4",
                "Ends with ؟: mark on the far left of the text in every row",
                workaround_rows("آیا آماده‌ای؟"),
            ),
        ],
    )
}

fn section1<'a>() -> Element<'a, Message> {
    section(
        "1. Letter shapes",
        vec![
            s(
                "1.1",
                "Persian: words read correctly, letters joined, پ چ ژ گ correct",
                "پدر چای ژاله گل پنجره چشم مژده گرگ",
            ),
            s(
                "1.2",
                "Persian: the same letters in a running sentence",
                "به پیچ و خم ژرف گلستان چشم بدوز و پرچم را بگیر",
            ),
            s(
                "1.3",
                "Letter by letter: isolated, then joined three times. ژ never joins to the letter after it",
                "پ پپپ چ چچچ ژ ژژژ گ گگگ",
            ),
            s(
                "1.4",
                "Zero-width non-joiner (U+200C): no box, می and خواهم stay apart without a gap, کتاب‌ها too",
                "می‌خواهم کتاب‌ها را بخوانم",
            ),
            s(
                "1.5",
                "Arabic: joined correctly, lam-alef ligature in لا",
                "لا تقل لي إن السلام بعيد عن هذه الأرض",
            ),
            s(
                "1.6",
                "Arabic with diacritics: each mark above or below its own letter, none floating or colliding",
                "مَرْحَبًا بِكُمْ فِي الْعَالَمِ رَبَّنَا عَلَّمَ الْقَلَمُ",
            ),
            s(
                "1.7",
                "Hebrew: correct letters, final forms ם ן ץ ף ך",
                "שלום עולם, זהו מבחן קצר של עברית",
            ),
            s(
                "1.8",
                "Hebrew with niqqud: points in the right place",
                "שָׁלוֹם עוֹלָם בְּרֵאשִׁית",
            ),
        ],
    )
}

fn section2<'a>() -> Element<'a, Message> {
    section(
        "2. Mixed direction",
        vec![
            s(
                "2.1",
                "Order from the right: first Persian word, then 5G, then the second Persian word, then API on the far left",
                "نسخه 5G را API",
            ),
            s(
                "2.2",
                "Number keeps its digit order and separators: 1,250.50 and 2026",
                "قیمت 1,250.50 دلار در سال 2026",
            ),
            s(
                "2.3",
                "Persian digits read ۱۴۰۵, percent sign after ۲۵ on the left",
                "سال ۱۴۰۵ و ۲۵٪",
            ),
            s(
                "2.4",
                "Latin at the line start, no RLM: base is LTR, so API is on the left and the text is left aligned",
                "API نسخه جدید",
            ),
            s(
                "2.5",
                "Latin at the line end: API on the far left, text right aligned",
                "نسخه جدید API",
            ),
            s(
                "2.6",
                "Parentheses surround GPT-4 and open the right way",
                "این مدل (GPT-4) بهتر است",
            ),
            s(
                "2.7",
                "Parentheses at the line start: surround نسخه اول, on the right",
                "(نسخه اول) شروع شد",
            ),
            s(
                "2.8",
                "Times stay 00:12:34 and 00:12:36, first time is further right",
                "زمان 00:12:34 تا 00:12:36",
            ),
            s(
                "2.9",
                "URL reads normally, then the Persian continues on its left",
                "https://example.com/a?b=1 را باز کنید",
            ),
            s("2.10", "Arabic, same order as 2.1", "الإصدار 5G من API"),
            s("2.11", "Hebrew, same order as 2.1", "גרסה 5G של API"),
        ],
    )
}

fn section3<'a>() -> Element<'a, Message> {
    section(
        "3. Punctuation at the line end: the mark sits on the far left, after the last word",
        vec![
            s("3.1", "؟ on the far left", "آیا آماده‌ای؟"),
            s("3.2", "، after سلام and on the far left", "سلام، دنیا،"),
            s("3.3", "؛ on the far left", "این پایان است؛"),
            s(
                "3.4",
                "Period on the far left, also with ؟! together",
                "پایان. چرا؟!",
            ),
            s(
                "3.5",
                "Line ending in a Latin word plus ؟: ؟ stays on the far left",
                "این نسخه API؟",
            ),
            s("3.6", "Line ending in a number plus ،", "سال 2026،"),
            s(
                "3.7",
                "Arabic: ؟ and ، on the far left",
                "هل أنت مستعد، يا صديقي؟",
            ),
            s("3.8", "Hebrew: comma and ? on the left", "שלום, מה שלומך?"),
        ],
    )
}

fn pair<'a>(id: &'a str, check: &'a str, without: &'a str, with: &'a str) -> Element<'a, Message> {
    let side = |title: &'a str, t: &'a str| -> Column<'a, Message> {
        column![text(title).size(12).color(GREY), rtl(t)]
            .spacing(2)
            .width(Fill)
    };
    sample(
        id,
        check,
        row![side("without RLM", without), side("with RLM", with)]
            .spacing(16)
            .into(),
    )
}

fn section4<'a>() -> Element<'a, Message> {
    section(
        "4. U+200F prefix (RLM): left column without, right column with",
        vec![
            pair(
                "4.1",
                "Plain Persian: both must look identical, no box on the right edge of the second",
                "سلام دنیا",
                "\u{200F}سلام دنیا",
            ),
            pair(
                "4.2",
                "Starts with Latin. Without: API on the left. With: API on the far right, then the Persian going left",
                "API نسخه جدید است",
                "\u{200F}API نسخه جدید است",
            ),
            pair(
                "4.3",
                "Starts with 5G: the G is a Latin letter, so without RLM the base is LTR and 5G is on the far left. With RLM, far right",
                "5G سریع است",
                "\u{200F}5G سریع است",
            ),
            pair(
                "4.7",
                "Starts with a plain number: digits are not strong, so both are RTL and 12 is on the far right in both",
                "12 نفر آمدند",
                "\u{200F}12 نفر آمدند",
            ),
            pair(
                "4.4",
                "Ends with ؟ and starts with Latin. With RLM, ؟ on the far left",
                "Hello دنیا؟",
                "\u{200F}Hello دنیا؟",
            ),
            pair(
                "4.5",
                "Hebrew: no box",
                "API גרסה חדשה",
                "\u{200F}API גרסה חדשה",
            ),
            pair(
                "4.6",
                "Arabic: no box",
                "API إصدار جديد",
                "\u{200F}API إصدار جديد",
            ),
        ],
    )
}

fn aligned_rows<'a>(sample_text: &'a str) -> Element<'a, Message> {
    let mut col = Column::new().spacing(3);
    for (name, a) in [
        ("Default", TextAlign::Default),
        ("Left", TextAlign::Left),
        ("Right", TextAlign::Right),
        ("Center", TextAlign::Center),
        ("dir_align", dir_align(sample_text)),
    ] {
        let t = text(sample_text)
            .size(20)
            .width(Fill)
            .align_x(a)
            .wrapping(Wrapping::WordOrGlyph);
        col = col.push(row![text(name).size(12).color(GREY).width(64), band(t.into())].spacing(6));
    }
    let fixed = text(dir::isolate_rtl(sample_text))
        .size(20)
        .width(Fill)
        .align_x(dir_align(sample_text))
        .wrapping(Wrapping::WordOrGlyph);
    col = col.push(
        row![
            text("dir_align + isolate_rtl")
                .size(12)
                .color(GREY)
                .width(64),
            band(fixed.into())
        ]
        .spacing(6),
    );
    col.into()
}

fn section5<'a>() -> Element<'a, Message> {
    let c = "Each lighter band is a Fill-width box. Default: does RTL hug the left? Right, Center, dir_align as named?";
    section(
        "5. Alignment",
        vec![
            sample("5.1", c, aligned_rows("سلام دنیا")),
            sample(
                "5.2",
                "Wrapped Persian. Last line stays with the block's own alignment, nothing clipped, no word lost",
                aligned_rows(LONG_FA),
            ),
            sample("5.3", "Hebrew", aligned_rows("שלום עולם")),
            sample("5.4", "Wrapped Arabic", aligned_rows(LONG_AR)),
            sample("5.5", "English control", aligned_rows("Hello world")),
            sample(
                "5.6",
                "Latin first: dir_align gives Left",
                aligned_rows("API نسخه جدید"),
            ),
            sample(
                "5.7",
                "Plain number first: dir_align gives Right (but 5G first gives Left, the G is Latin)",
                aligned_rows("12 نسخه جدید"),
            ),
            sample(
                "5.8",
                "RLM + Latin first: dir_align gives Right",
                aligned_rows("\u{200F}API نسخه جدید"),
            ),
            sample(
                "5.9",
                "No strong character: dir_align gives Left",
                aligned_rows("12:30 - 13:45"),
            ),
        ],
    )
}

fn overlay(t: impl text::IntoFragment<'static>) -> Element<'static, Message> {
    container(
        container(
            text(t)
                .size(28)
                .color(Color::WHITE)
                .width(Fill)
                .align_x(TextAlign::Center)
                .wrapping(Wrapping::WordOrGlyph),
        )
        .width(Fill)
        .padding(10)
        .style(|_: &Theme| container::Style {
            background: Some(Color::from_rgba(0.0, 0.0, 0.0, 0.82).into()),
            ..Default::default()
        }),
    )
    .width(Fill)
    .padding(iced::Padding {
        top: 50.0,
        bottom: 16.0,
        left: 40.0,
        right: 40.0,
    })
    .style(|_: &Theme| container::Style {
        background: Some(Color::from_rgb(0.33, 0.42, 0.45).into()),
        ..Default::default()
    })
    .into()
}

fn section6<'a>() -> Element<'a, Message> {
    let c = "Every line centred on the same axis, wrapped between words, nothing clipped";
    section(
        "6. Subtitles over a dark band",
        vec![
            sample("6.1", c, overlay(LONG_FA)),
            sample("6.2", c, overlay(LONG_AR)),
            sample("6.3", c, overlay(LONG_HE)),
            sample(
                "6.4",
                "Two explicit lines with the RLM prefix the writer adds: both centred, no box",
                overlay("\u{200F}سلام دوستان، خوش آمدید\n\u{200F}API نسخه 5G آماده است؟"),
            ),
            sample("6.5", "One short line: centred", overlay("چرا؟")),
            sample(
                "6.6",
                "English control",
                overlay(
                    "This is a long English sentence that has to wrap on several lines so we can compare it with the right-to-left samples.",
                ),
            ),
            sample(
                "6.7",
                "isolate_rtl fix, one short line: visible and centred (compare 6.5)",
                overlay(dir::isolate_rtl("چرا؟")),
            ),
            sample(
                "6.8",
                "isolate_rtl fix, wrapped Persian: same as 6.1",
                overlay(dir::isolate_rtl(LONG_FA)),
            ),
            sample(
                "6.9",
                "isolate_rtl fix, explicit lines with RLM: same as 6.4",
                overlay(dir::isolate_rtl(
                    "\u{200F}سلام دوستان، خوش آمدید\n\u{200F}API نسخه 5G آماده است؟",
                )),
            ),
            sample(
                "6.10",
                "isolate_rtl fix, one short Hebrew line and one Arabic: visible and centred",
                overlay(dir::isolate_rtl("שלום לכולם\nأهلا وسهلا")),
            ),
        ],
    )
}

fn cjk<'a>(id: &'a str, check: &'a str, t: &'a str) -> Element<'a, Message> {
    let narrow = |title: &'a str, w: Wrapping| {
        column![
            text(title).size(12).color(GREY),
            container(text(t).size(20).wrapping(w).width(Fill))
                .width(200)
                .style(frame),
        ]
        .spacing(2)
    };
    sample(
        id,
        check,
        row![
            narrow("WordOrGlyph, 200 wide", Wrapping::WordOrGlyph),
            narrow("Word (default), 200 wide", Wrapping::Word)
        ]
        .spacing(16)
        .into(),
    )
}

fn section7<'a>() -> Element<'a, Message> {
    let c = "Left box wraps inside its border, nothing past the right border, glyphs look right for the language";
    section(
        "7. CJK wrapping",
        vec![
            cjk(
                "7.1",
                c,
                "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。",
            ),
            cjk(
                "7.2",
                c,
                "床前明月光疑是地上霜举头望明月低头思故乡春眠不觉晓处处闻啼鸟夜来风雨声花落知多少",
            ),
            cjk(
                "7.3",
                "Japanese kanji drawn in a Japanese style (e.g. 直 骨 角 look Japanese, not Chinese). Only if you read Japanese",
                "直骨角今令海写真直接",
            ),
            cjk(
                "7.4",
                "Long Latin without spaces: same box",
                "https://example.com/a/very/long/path/without/any/spaces/that/cannot/wrap",
            ),
            cjk(
                "7.5",
                "Long Persian without spaces: same box",
                "پپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپپ",
            ),
            cjk(
                "7.6",
                "Long Latin with no break opportunity at all: left box breaks it, right box may overflow",
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ),
            sample(
                "7.7",
                "Japanese with Font::with_name(\"Yu Gothic\"). Compare 7.1: kanji and kana now from one font?",
                text("吾輩は猫である。名前はまだ無い。直骨角今令海")
                    .size(20)
                    .font(Font::with_name("Yu Gothic"))
                    .into(),
            ),
        ],
    )
}
