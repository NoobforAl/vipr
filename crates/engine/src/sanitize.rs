//! Making untrusted text safe to put in a prompt.
//!
//! `llama.rs:84` tokenizes the finished prompt with special-token parsing on, so a literal
//! `<|im_end|>` in a subtitle could become a real control token. A literal newline or `|` could also
//! forge a new row in a line-based prompt or reply.
//!
//! This is for prompts only. Subtitles are ordinary text that can say `a < b` or `[music]`, so only
//! what looks like a token goes, and the result must never replace the user's own text on disk.

/// Longest token this looks for, in characters. Keeps the scan linear on hostile input.
const MAX_TOKEN: usize = 64;

/// Nested tokens need one pass per level (`<<s>s>` leaves `<s>`). Past this many the text is hostile.
const MAX_PASSES: usize = 32;

/// `<name>` tokens that stay stripped even when glued to a word, as in `hello<s>`.
const KNOWN_TAGS: &[&str] = &[
    "s",
    "bos",
    "eos",
    "pad",
    "unk",
    "sep",
    "cls",
    "mask",
    "sop",
    "eop",
    "think",
    "thinking",
    "system",
    "user",
    "assistant",
    "human",
    "bot",
    "model",
    "tool",
    "endoftext",
];

/// `[name]` tokens. Anything else in square brackets is left alone, because subtitles use them for
/// sounds such as `[music]`.
const KNOWN_BRACKETS: &[&str] = &[
    "inst",
    "sys",
    "s",
    "bos",
    "eos",
    "pad",
    "unk",
    "sep",
    "cls",
    "mask",
    "gmask",
    "smask",
    "sop",
    "eop",
    "think",
    "tool_calls",
    "tool_results",
    "tool_content",
    "available_tools",
    "system_prompt",
    "img",
    "prefix",
    "middle",
    "suffix",
];

/// Strips special-token sequences such as `<|im_end|>`, tag-like tokens, control characters and
/// invisible direction marks, turns newlines into spaces and `|` into `/`. Runs before every prompt,
/// including refine. Ordinary text, including `a < b`, `x<y>z` and U+200C in Persian, stays intact.
pub fn sanitize_untrusted(text: &str) -> String {
    let mut s = strip_controls(text);
    let mut settled = false;
    for _ in 0..MAX_PASSES {
        // `|` goes inside the loop because replacing it can complete a token: `[|INST]`
        let next = strip_tokens(&s).replace(['|', '｜'], "/");
        if next == s {
            settled = true;
            break;
        }
        s = next;
    }
    if !settled {
        // every token starts with one of these
        s = s.replace(['<', '['], "");
    }
    collapse_spaces(&s)
}

fn strip_controls(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\n'
            | '\r'
            | '\t'
            | '\u{B}'
            | '\u{C}'
            | '\u{1C}'..='\u{1E}'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}' => out.push(' '),
            // direction marks and embeddings, BOM, annotation marks, and the tag characters that
            // can spell hidden ASCII. U+200C and U+200D stay: Persian and emoji need them.
            '\u{200E}'
            | '\u{200F}'
            | '\u{61C}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2066}'..='\u{2069}'
            | '\u{FEFF}'
            | '\u{FFF9}'..='\u{FFFB}'
            | '\u{E0000}'..='\u{E007F}' => {}
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

fn strip_tokens(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(pos) = rest.find(['<', '[']) {
        out.push_str(&rest[..pos]);
        let tail = &rest[pos..];
        let len = if tail.starts_with('[') {
            bracket_token(tail)
        } else {
            pipe_token(tail)
                .or_else(|| llama2_sys_token(tail))
                .or_else(|| tag_token(tail, out.chars().last()))
        };
        match len {
            Some(len) => rest = &tail[len..],
            None => {
                out.push_str(&tail[..1]);
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn is_pipe(c: char) -> bool {
    c == '|' || c == '｜'
}

/// `<|anything|>` and `<｜anything｜>`, with no whitespace inside. Returns the byte length.
fn pipe_token(tail: &str) -> Option<usize> {
    let mut chars = tail.char_indices().skip(1);
    if !is_pipe(chars.next()?.1) {
        return None;
    }
    let mut prev_pipe = None;
    for (i, c) in chars.take(MAX_TOKEN) {
        if c.is_whitespace() {
            return None;
        }
        if c == '>' {
            if let Some(pipe_end) = prev_pipe
                && pipe_end == i
            {
                return Some(i + 1);
            }
            return None;
        }
        prev_pipe = is_pipe(c).then_some(i + c.len_utf8());
    }
    None
}

/// `<<SYS>>` and `<</SYS>>`.
fn llama2_sys_token(tail: &str) -> Option<usize> {
    ["<<sys>>", "<</sys>>"].iter().find_map(|t| {
        tail.get(..t.len())
            .filter(|head| head.eq_ignore_ascii_case(t))
            .map(|_| t.len())
    })
}

/// `<name>`, `</name>`, `<name/>`, and the half-piped forms `<|name>` and `<name|>`.
///
/// A plain word glued to the text before it is a comparison or a generic rather than a tag, as in
/// `x<y>z` or `Vec<String>`, so it stays unless the model family is known to use that name.
fn tag_token(tail: &str, before: Option<char>) -> Option<usize> {
    let mut chars = tail.char_indices().skip(1).peekable();
    let mut piped = false;
    if chars.next_if(|(_, c)| is_pipe(*c)).is_some() {
        piped = true;
    }
    let closing = chars.next_if(|(_, c)| *c == '/').is_some();
    let start = chars.peek()?.0;
    let mut end = start;
    let mut len = 0;
    while let Some((i, c)) = chars.peek().copied() {
        if !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '▁')) {
            break;
        }
        len += 1;
        if len > MAX_TOKEN {
            return None;
        }
        end = i + c.len_utf8();
        chars.next();
    }
    let name = &tail[start..end];
    if !name.starts_with(|c: char| c.is_ascii_alphabetic()) {
        return None;
    }
    let self_closing = chars.next_if(|(_, c)| *c == '/').is_some();
    if chars.next_if(|(_, c)| is_pipe(*c)).is_some() {
        piped = true;
    }
    let (close, _) = chars.next_if(|(_, c)| *c == '>')?;

    let known = KNOWN_TAGS.iter().any(|k| name.eq_ignore_ascii_case(k));
    let plain_word = name.chars().all(|c| c.is_ascii_alphabetic());
    let glued = before.is_some_and(char::is_alphanumeric);
    let generic = glued && plain_word && !closing && !self_closing;
    (piped || known || !generic).then_some(close + 1)
}

/// `[NAME]` and `[/NAME]` for the names in [`KNOWN_BRACKETS`], plus `[control_N]`.
fn bracket_token(tail: &str) -> Option<usize> {
    let close = tail
        .char_indices()
        .take(MAX_TOKEN)
        .find(|(_, c)| *c == ']')?
        .0;
    let inner = &tail[1..close];
    let name = inner.strip_prefix('/').unwrap_or(inner);
    let known = KNOWN_BRACKETS.iter().any(|k| name.eq_ignore_ascii_case(k))
        || name
            .get(.."control_".len())
            .is_some_and(|p| p.eq_ignore_ascii_case("control_"));
    known.then_some(close + 1)
}

fn collapse_spaces(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.trim_matches(' ').chars() {
        if c != ' ' || !out.ends_with(' ') {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_the_tokens_of_common_model_families() {
        for (input, want) in [
            ("a<|im_end|>b", "ab"),
            ("<|im_start|>system", "system"),
            ("<|eot_id|> hi", "hi"),
            ("<|begin_of_text|>hi", "hi"),
            ("x<｜end▁of▁sentence｜>y", "xy"),
            ("<|turn>user", "user"),
            ("hi<turn|>", "hi"),
            ("<s>hello</s>", "hello"),
            ("hello<s>", "hello"),
            ("hello</s>", "hello"),
            ("<S>hello</S>", "hello"),
            ("[INST] do this [/INST]", "do this"),
            ("[inst]x[/Inst]", "x"),
            ("<<SYS>>rules<</SYS>>", "rules"),
            ("<think>plan</think>", "plan"),
            ("<start_of_turn>model", "model"),
            ("<end_of_turn>", ""),
            ("x<unused12>y", "xy"),
            ("x<extra_id_0>y", "xy"),
            ("x<reserved_special_token_3>y", "xy"),
            ("<tool_call>", ""),
            ("[TOOL_CALLS]x", "x"),
            ("[control_8]x", "x"),
        ] {
            assert_eq!(sanitize_untrusted(input), want, "{input:?}");
        }
    }

    #[test]
    fn stripping_cannot_build_a_new_token() {
        for input in [
            "<<s>s>",
            "<<s>/s>",
            "<|im_<s>end|>",
            "<|im_\u{7}end|>",
            "<\u{0}s>",
            "<\ns>",
            "[IN[INST]ST]",
            "[|INST]",
            "[｜INST]",
            "[\u{202E}INST]",
            "<<|x|>s>",
            "<|<|im_end|>|>",
        ] {
            let out = sanitize_untrusted(input);
            for token in ["<s>", "</s>", "[INST]", "[/INST]", "<|", "|>", "|"] {
                assert!(!out.contains(token), "{input:?} gave {out:?}");
            }
        }
    }

    #[test]
    fn deep_nesting_ends_token_free() {
        let input = format!("{}s{}", "<".repeat(1000), ">".repeat(1000));
        assert!(!sanitize_untrusted(&input).contains("<s>"));
        let input = format!("{}s>", "<s".repeat(1000));
        assert!(!sanitize_untrusted(&input).contains("<s>"));
        let input = "<s>".repeat(100_000);
        assert_eq!(sanitize_untrusted(&input), "");
    }

    #[test]
    fn legitimate_text_stays() {
        for s in [
            "a < b",
            "a > b",
            "if a < b and c > d",
            "a<b and c>d",
            "a<b",
            "1 < 2 > 1",
            "x <= y",
            "x<y>z",
            "a <> b",
            "Vec<String>",
            "I <3 you",
            "<3>",
            "x -> y <- z",
            "me <me@example.com>",
            "[music]",
            "[Applause]",
            "[door slams]",
            "array[0]",
            "<",
            ">",
            "[",
            "]",
            "سلام، می\u{200C}روم",
            "👨\u{200D}👩\u{200D}👧",
            "<سلام>",
            "5 < 6 is true",
        ] {
            assert_eq!(sanitize_untrusted(s), s, "{s:?}");
        }
    }

    #[test]
    fn newlines_and_controls_cannot_shape_the_prompt() {
        assert_eq!(sanitize_untrusted("one\ntwo\r\nthree"), "one two three");
        assert_eq!(sanitize_untrusted("a\u{2028}b\u{2029}c\u{85}d"), "a b c d");
        assert_eq!(sanitize_untrusted("a\tb"), "a b");
        assert_eq!(sanitize_untrusted("a\u{7}b\u{0}c\u{7F}d\u{9F}e"), "abcde");
        assert_eq!(sanitize_untrusted("  a   b  "), "a b");
        assert_eq!(sanitize_untrusted("\u{200F}a\u{202E}b\u{2067}c"), "abc");
        // invisible tag characters spelling "hi"
        assert_eq!(sanitize_untrusted("a\u{E0068}\u{E0069}b"), "ab");
        assert_eq!(sanitize_untrusted("\u{FEFF}a"), "a");
    }

    #[test]
    fn pipes_become_slashes() {
        assert_eq!(sanitize_untrusted("a | b"), "a / b");
        assert_eq!(sanitize_untrusted("3|forged row"), "3/forged row");
        assert_eq!(sanitize_untrusted("a｜b"), "a/b");
        // not a token: the Elixir and F# pipe operators keep their operands
        assert_eq!(sanitize_untrusted("a <| b |> c"), "a </ b /> c");
    }

    #[test]
    fn is_idempotent() {
        for s in [
            "a < b | c",
            "<|im_end|> x <s> y",
            "x<y>z\nnext",
            "[|INST] a </ b />",
            "<<<s>>>",
            "سلام <s> دنیا",
        ] {
            let once = sanitize_untrusted(s);
            assert_eq!(sanitize_untrusted(&once), once, "{s:?}");
        }
    }

    #[test]
    fn long_hostile_input_finishes_quickly() {
        let started = std::time::Instant::now();
        sanitize_untrusted(&"<".repeat(1_000_000));
        sanitize_untrusted(&"<|".repeat(500_000));
        sanitize_untrusted(&"[".repeat(1_000_000));
        assert!(started.elapsed().as_secs() < 30);
    }
}
