//! `--report`: numbers from the same text stack iced draws with (cosmic-text + harfrust), so the
//! parts of the checklist that can be measured do not depend on anyone reading a screenshot.

use iced::{Font, Pixels, Size};
use iced_graphics::core::alignment::Vertical;
use iced_graphics::core::text::{Alignment, LineHeight, Paragraph as _, Shaping, Text, Wrapping};
use iced_graphics::text::cosmic_text::{self, Attrs, Buffer, FontSystem, Metrics};
use iced_graphics::text::{Paragraph, font_system};

fn attrs() -> Attrs<'static> {
    iced_graphics::text::to_attributes(Font::default())
}

fn shape(fs: &mut FontSystem, s: &str, width: Option<f32>) -> Buffer {
    let mut b = Buffer::new(fs, Metrics::new(24.0, 30.0));
    b.set_size(fs, width, None);
    b.set_text(fs, s, &attrs(), cosmic_text::Shaping::Advanced, None);
    b
}

fn face(fs: &FontSystem, id: cosmic_text::fontdb::ID) -> String {
    fs.db()
        .face(id)
        .map_or("?".into(), |f| f.post_script_name.clone())
}

/// Font faces with their glyph counts, and how many glyphs came out as .notdef (the "box").
pub fn fonts_used(fs: &mut FontSystem, s: &str) -> (String, usize) {
    let b = shape(fs, s, None);
    let mut counts: Vec<(String, usize)> = vec![];
    let mut notdef = 0;
    for run in b.layout_runs() {
        for g in run.glyphs {
            notdef += usize::from(g.glyph_id == 0);
            let name = face(fs, g.font_id);
            match counts.iter_mut().find(|(n, _)| *n == name) {
                Some((_, c)) => *c += 1,
                None => counts.push((name, 1)),
            }
        }
    }
    let list: Vec<String> = counts.iter().map(|(n, c)| format!("{n} ×{c}")).collect();
    (list.join(", "), notdef)
}

pub const FONT_SAMPLES: &[(&str, &str)] = &[
    ("Persian", "پدر چای ژاله گل می‌خواهم ۱۲۳"),
    ("Arabic", "لا تقل لي إن السلام بعيد ٤٥٦"),
    ("Arabic marks", "مَرْحَبًا بِكُمْ فِي الْعَالَمِ"),
    ("Hebrew", "שלום עולם זהו מבחן"),
    ("Hebrew marks", "שָׁלוֹם עוֹלָם"),
    ("Punctuation", "؟ ، ؛ ٪"),
    ("Format chars", "\u{200F}a\u{200E}b\u{200C}c\u{200D}d"),
    ("Japanese", "吾輩は猫である。名前はまだ無い。カタカナ"),
    ("Chinese", "床前明月光，疑是地上霜。"),
    ("Korean", "안녕하세요"),
    ("Latin", "Hello 123"),
];

/// Rows for section 9 of the page: (label, sample, faces, notdef).
pub fn font_rows() -> Vec<(&'static str, &'static str, String, usize)> {
    let mut guard = font_system().write().expect("font system");
    let fs = guard.raw();
    FONT_SAMPLES
        .iter()
        .map(|(label, s)| {
            let (faces, notdef) = fonts_used(fs, s);
            (*label, *s, faces, notdef)
        })
        .collect()
}

/// Glyph ids per source character, in logical order. Each entry is the cluster's first glyph.
fn logical_ids(fs: &mut FontSystem, s: &str) -> Vec<u16> {
    let b = shape(fs, s, None);
    let mut glyphs: Vec<(usize, u16)> = b
        .layout_runs()
        .flat_map(|r| r.glyphs.iter().map(|g| (g.start, g.glyph_id)))
        .collect();
    glyphs.sort_by_key(|g| g.0);
    glyphs.dedup_by_key(|g| g.0);
    glyphs.into_iter().map(|g| g.1).collect()
}

fn joining(fs: &mut FontSystem) {
    println!("== 1. joining: glyph ids (a different id means a different contextual form)");
    println!("   dual-joining, expected: initial, medial, final all differ from isolated");
    for c in "پچگبتسشعغفقکلمنهی".chars() {
        let iso = logical_ids(fs, &c.to_string());
        let forms = logical_ids(fs, &format!("{c}{c}{c}"));
        let ok = forms.len() == 3
            && !forms.contains(&iso[0])
            && forms[0] != forms[1]
            && forms[1] != forms[2];
        println!(
            "   {c} U+{:04X} iso={:?} init/med/fin={forms:?} {}",
            c as u32,
            iso[0],
            if ok { "ok" } else { "CHECK" }
        );
    }
    println!(
        "   right-joining, expected: final form differs from isolated, and the next letter restarts"
    );
    for c in "ژاددذرزو".chars() {
        let iso = logical_ids(fs, &c.to_string());
        let after = logical_ids(fs, &format!("ب{c}"));
        let ok = after.len() == 2 && after[1] != iso[0];
        println!(
            "   {c} U+{:04X} iso={:?} after-beh={after:?} {}",
            c as u32,
            iso[0],
            if ok { "ok" } else { "CHECK" }
        );
    }
    let plain = logical_ids(fs, "می");
    let zwnj = logical_ids(fs, "م\u{200C}ی");
    println!(
        "   ZWNJ: می={plain:?}  م+ZWNJ+ی={zwnj:?} (the letters should differ: joining broken)"
    );
    let la = logical_ids(fs, "لا");
    println!(
        "   lam-alef: {} glyph cluster(s), ids {la:?} (1 = ligature)",
        la.len()
    );
}

fn marks(fs: &mut FontSystem) {
    println!("== 1. marks: how many glyphs are zero-width marks with their own position");
    for s in ["مَرْحَبًا", "الْعَالَمِ", "رَبَّنَا", "שָׁלוֹם"] {
        let b = shape(fs, s, None);
        let (mut total, mut zero, mut moved, mut boxes) = (0, 0, 0, 0);
        for r in b.layout_runs() {
            for g in r.glyphs {
                total += 1;
                zero += usize::from(g.w == 0.0);
                moved += usize::from(g.x_offset != 0.0 || g.y_offset != 0.0);
                boxes += usize::from(g.glyph_id == 0);
            }
        }
        println!("   {s}: glyphs={total} zero-width={zero} offset={moved} notdef={boxes}");
    }
}

/// Visual order, left to right, grouped into runs of the same embedding level.
fn visual(fs: &mut FontSystem, s: &str) -> String {
    let b = shape(fs, s, None);
    let mut out = String::new();
    for r in b.layout_runs() {
        out += if r.rtl { "base=RTL " } else { "base=LTR " };
        let mut seen: Vec<(u8, usize, usize)> = vec![];
        let mut glyphs: Vec<_> = r.glyphs.iter().collect();
        glyphs.sort_by(|a, b| a.x.total_cmp(&b.x));
        for g in glyphs {
            let lvl = g.level.number();
            match seen.last_mut() {
                Some(l) if l.0 == lvl => {
                    l.1 = l.1.min(g.start);
                    l.2 = l.2.max(g.end);
                }
                _ => seen.push((lvl, g.start, g.end)),
            }
        }
        for (lvl, a, e) in seen {
            out += &format!("[{}|L{lvl}] ", r.text[a..e].trim());
        }
    }
    out
}

fn order(fs: &mut FontSystem) {
    println!(
        "== 2,3,4. visual order, left to right on screen (each [..] prints its own text logically)"
    );
    for s in [
        "نسخه 5G را API",
        "קיבלנו גרסה 5G של API",
        "قیمت 1,250.50 دلار در سال 2026",
        "سال ۱۴۰۵ و ۲۵٪",
        "API نسخه جدید",
        "نسخه جدید API",
        "این مدل (GPT-4) بهتر است",
        "(نسخه اول) شروع شد",
        "زمان 00:12:34 تا 00:12:36",
        "آیا آماده‌ای؟",
        "سلام، دنیا،",
        "این پایان است؛",
        "\u{200F}API نسخه جدید است",
        "\u{200F}5G سریع است",
        "5G سریع است",
    ] {
        println!("   {s:?}\n      {}", visual(fs, s));
    }
    println!(
        "== 3. final punctuation: x of each mark against the line (RTL: it must be the leftmost)"
    );
    for s in [
        "آیا آماده‌ای؟",
        "سلام، دنیا،",
        "این پایان است؛",
        "שלום, עולם?",
    ] {
        let b = shape(fs, s, None);
        for r in b.layout_runs() {
            let last = r.glyphs.iter().max_by_key(|g| g.start).expect("glyph");
            let min_x = r.glyphs.iter().map(|g| g.x).fold(f32::MAX, f32::min);
            println!(
                "   {s}: last char x={:.1}, line min x={min_x:.1}, {}",
                last.x,
                if last.x <= min_x + 0.01 {
                    "leftmost"
                } else {
                    "CHECK"
                }
            );
        }
    }
    println!("== 2. mirroring: '(' glyph id in an LTR line and in an RTL line");
    let ltr = logical_ids(fs, "(");
    let rtl = logical_ids(fs, "سلام (");
    println!("   ltr={:?} rtl={:?} (should differ)", ltr[0], rtl.last());
}

fn format_chars(fs: &mut FontSystem) {
    println!("== 4. format characters: glyph, advance, face");
    for (name, s) in [
        ("RLM U+200F", "\u{200F}"),
        ("RLM+Persian", "\u{200F}سلام"),
        ("Persian only", "سلام"),
        ("RLM+Hebrew", "\u{200F}שלום"),
        ("Hebrew only", "שלום"),
        ("RLM+Latin", "\u{200F}Hello"),
        ("Latin only", "Hello"),
        ("ZWNJ", "\u{200C}"),
        ("LRM", "\u{200E}"),
        ("RLI..PDI", "\u{2067}a\u{2069}"),
    ] {
        let b = shape(fs, s, None);
        let mut list = vec![];
        let mut w = 0.0;
        for r in b.layout_runs() {
            w = r.line_w;
            for g in r.glyphs {
                list.push(format!(
                    "#{} w={:.2} {}",
                    g.glyph_id,
                    g.w,
                    face(fs, g.font_id)
                ));
            }
        }
        let shown = if list.len() > 4 {
            format!("{} glyphs", list.len())
        } else {
            list.join("; ")
        };
        println!("   {name}: line_w={w:.2} {shown}");
    }
}

fn paragraph(s: &str, align: Alignment, width: f32, wrapping: Wrapping) -> Paragraph {
    Paragraph::with_text(Text {
        content: s,
        bounds: Size::new(width, f32::INFINITY),
        size: Pixels(24.0),
        line_height: LineHeight::default(),
        font: Font::default(),
        align_x: align,
        align_y: Vertical::Top,
        shaping: Shaping::Auto,
        wrapping,
    })
}

fn alignment() {
    println!(
        "== 5. what iced does to the buffer: buffer width after layout, against the requested bounds"
    );
    let long = "این یک جملهٔ بلند فارسی است که باید در چند خط شکسته شود تا رفتار تراز متن را ببینیم";
    for (name, s) in [
        ("short fa", "سلام دنیا"),
        ("long fa", long),
        ("short en", "Hello world"),
    ] {
        let width = 400.0;
        let plain = {
            let mut guard = font_system().write().expect("font system");
            shape(guard.raw(), s, Some(width)).layout_runs().count()
        };
        for (an, a) in [
            ("Default", Alignment::Default),
            ("Left", Alignment::Left),
            ("Right", Alignment::Right),
            ("Center", Alignment::Center),
        ] {
            let p = paragraph(s, a, width, Wrapping::WordOrGlyph);
            let lines = p.buffer().layout_runs().count();
            let (bw, _) = p.buffer().size();
            let min = p.min_bounds();
            println!(
                "   {name} {an}: requested={width} buffer_w={bw:?} min_w={:.1} lines={lines} (lines at requested width={plain})",
                min.width
            );
        }
    }
}

fn carets() {
    println!("== 8. text_input geometry: grapheme_position(x) per logical index, Advanced shaping");
    println!("   RTL word: x should go down as the index goes up");
    for s in ["سلام دنیا", "نسخه 5G را API", "Hello دنیا", "שלום"] {
        let p = paragraph(s, Alignment::Default, f32::INFINITY, Wrapping::None);
        let n = s.chars().count();
        let xs: Vec<String> = (0..=n)
            .map(|i| {
                p.grapheme_position(0, i)
                    .map_or("-".into(), |pt| format!("{:.0}", pt.x))
            })
            .collect();
        println!(
            "   {s}: min_w={:.0} x={}",
            p.min_bounds().width,
            xs.join(",")
        );
    }
    println!("   hit_test across the line (x -> index):");
    for s in ["سلام دنیا", "نسخه 5G را API"] {
        let p = paragraph(s, Alignment::Default, f32::INFINITY, Wrapping::None);
        let w = p.min_bounds().width;
        let hits: Vec<String> = (0..=8)
            .map(|i| {
                let x = w * i as f32 / 8.0;
                let hit = p.hit_test(iced::Point::new(x, 10.0));
                format!(
                    "{x:.0}->{}",
                    hit.map_or("-".into(), |h| h.cursor().to_string())
                )
            })
            .collect();
        println!("   {s}: {}", hits.join(" "));
    }
}

fn separators() {
    println!(
        "== extra. cosmic-text asserts all bidi paragraphs in one line share a direction (shape.rs:960)"
    );
    let mut fs = FontSystem::new();
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    for (name, s) in [
        ("U+2029 latin then RTL", "abc\u{2029}سلام"),
        ("U+2029 RTL then latin", "سلام\u{2029}abc"),
        ("U+0085 latin then RTL", "abc\u{85}سلام"),
        ("U+001C latin then RTL", "abc\u{1C}سلام"),
        ("U+2028 latin then RTL", "abc\u{2028}سلام"),
        ("CR latin then RTL", "abc\rسلام"),
        ("LF latin then RTL", "abc\nسلام"),
    ] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            shape(&mut fs, s, Some(400.0)).layout_runs().count()
        }));
        match r {
            Ok(lines) => println!("   {name}: ok, {lines} line(s)"),
            Err(_) => println!("   {name}: PANIC"),
        }
    }
    let cleaned = crate::dir::clean_separators("abc\u{2029}سلام\u{85}def\u{1C}سلام");
    let lines = shape(&mut fs, &cleaned, Some(400.0)).layout_runs().count();
    println!("   after clean_separators, all of them mixed: ok, {lines} line(s)");
    // same direction on both sides must not trip the assert
    let same = shape(&mut fs, "abc\u{2029}def", Some(400.0))
        .layout_runs()
        .count();
    println!("   U+2029 latin then latin: ok, {same} line(s)");
    // the real iced path. Last, because a panic here poisons the global font system lock.
    let r = std::panic::catch_unwind(|| {
        paragraph("abc\u{2029}سلام", Alignment::Default, 400.0, Wrapping::Word).min_bounds()
    });
    println!(
        "   iced Paragraph::with_text, U+2029 latin then RTL: {}",
        if r.is_err() { "PANIC" } else { "ok" }
    );
    std::panic::set_hook(hook);
}

pub fn run() {
    {
        let mut guard = font_system().write().expect("font system");
        let fs = guard.raw();
        println!(
            "== 9. locale={} faces in db={}",
            fs.locale(),
            fs.db().faces().count()
        );
        for family in [
            "Arial",
            "Segoe UI",
            "Tahoma",
            "Microsoft YaHei UI",
            "Yu Gothic",
            "Noto Sans Arabic",
            "Noto Naskh Arabic",
            "Vazirmatn",
        ] {
            let has = fs
                .db()
                .faces()
                .any(|f| f.families.iter().any(|(n, _)| n == family));
            println!(
                "   {family}: {}",
                if has { "installed" } else { "not installed" }
            );
        }
        println!(
            "   default sans-serif family: {}",
            fs.db().family_name(&cosmic_text::fontdb::Family::SansSerif)
        );
        for (label, s) in FONT_SAMPLES {
            let (faces, notdef) = fonts_used(fs, s);
            println!("   {label}: {faces}; notdef={notdef}");
        }
        joining(fs);
        marks(fs);
        order(fs);
        format_chars(fs);
    }
    // Paragraph::with_text takes the font system lock itself, so the guard above must be gone
    alignment();
    carets();
    separators();
}
