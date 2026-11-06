//! `--mode chapters`: finds where the topic changes. Prompts, parsing and rules are in `engine::chapters`, and the JSON
//! types in `engine::project`.
//!
//! A long recording doesn't fit one prompt, so overlapping windows are read one at a time, and a last call over the
//! short list of titles joins what they split.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Result, bail};
use engine::chapters as rules;
use engine::project::{Chapter, ChaptersInput, ChaptersOutput, ChaptersStats, VERSION};

use crate::batch::{BadReply, MAX_HARD_ERRORS};
use crate::refine::Chat;

/// Transcript tokens per window. With a context of 4096 this leaves room for the instructions and the reply.
const WINDOW_TOKENS: usize = 2800;
/// Repeated from the window before, so a change on the edge is seen with the words around it.
const OVERLAP_TOKENS: usize = 200;
const WINDOW_REPLY_TOKENS: usize = 256;

/// Works out the chapters. `progress` receives `(done, total)` model calls. An `Err` means nothing usable came back, or
/// [`MAX_HARD_ERRORS`] hard errors in a row.
pub fn chapters(
    chat: &mut dyn Chat,
    input: &ChaptersInput,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<ChaptersOutput> {
    let started = Instant::now();
    let rendered: Vec<String> = input.lines.iter().map(rules::prompt_line).collect();
    let tokens: Vec<usize> = rendered.iter().map(|l| chat.count_tokens(l)).collect();
    let windows = rules::windows(&tokens, WINDOW_TOKENS, OVERLAP_TOKENS);
    let start_of: HashMap<u64, f64> = input.lines.iter().map(|l| (l.id, l.start)).collect();
    let total = windows.len() + 1;

    let mut candidates: Vec<Chapter> = Vec::new();
    let mut failed_windows = 0;
    let mut hard_errors = 0;
    for (i, window) in windows.iter().enumerate() {
        let ids: Vec<u64> = input.lines[window.clone()].iter().map(|l| l.id).collect();
        match ask_window(chat, &rendered[window.clone()], &ids) {
            Ok(found) => {
                hard_errors = 0;
                if found.is_empty() {
                    failed_windows += 1;
                }
                candidates.extend(found.into_iter().map(|(id, title)| Chapter {
                    start: start_of[&id],
                    title,
                }));
            }
            Err(e) => {
                hard_errors += 1;
                failed_windows += 1;
                if hard_errors >= MAX_HARD_ERRORS {
                    return Err(e);
                }
            }
        }
        progress(i + 1, total);
    }
    if candidates.is_empty() {
        bail!("the model gave no usable chapters");
    }
    let candidate_count = candidates.len();
    let settled = rules::settle(candidates, input.duration);

    // a failed merge only leaves more chapters than needed
    let limit = rules::most_chapters(input.duration);
    let merged = if settled.len() > 1 {
        ask_merge(chat, &settled, limit, input.duration).unwrap_or(settled)
    } else {
        settled
    };
    // the model doesn't always keep to the limit, so the code makes sure of it
    let merged = rules::limit_chapters(merged, limit, input.duration);
    progress(total, total);

    let stats = ChaptersStats {
        windows: windows.len(),
        failed_windows,
        candidates: candidate_count,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    };
    Ok(ChaptersOutput {
        version: VERSION,
        chapters: merged,
        stats,
    })
}

/// One retry with stricter wording when a reply has nothing usable.
fn ask_window(chat: &mut dyn Chat, lines: &[String], ids: &[u64]) -> Result<Vec<(u64, String)>> {
    for strict in [false, true] {
        let (system, user) = rules::window_prompt(lines, strict);
        match chat.complete(&system, &user, WINDOW_REPLY_TOKENS) {
            Ok(reply) => {
                let found = rules::parse_titles(&reply, ids);
                if !found.is_empty() {
                    return Ok(found);
                }
            }
            Err(e) if e.is::<BadReply>() => {}
            Err(e) => return Err(e),
        }
    }
    Ok(Vec::new())
}

fn ask_merge(
    chat: &mut dyn Chat,
    chapters: &[Chapter],
    limit: usize,
    duration: f64,
) -> Option<Vec<Chapter>> {
    let cap = rules::merge_token_cap(chapters.len());
    for strict in [false, true] {
        let (system, user) = rules::merge_prompt(chapters, limit, strict);
        let reply = chat.complete(&system, &user, cap).ok()?;
        if let Some(merged) = rules::parse_merge(&reply, chapters, duration) {
            return Some(merged);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use engine::project::TimedText;

    /// Answers from a list, one per call, counting words as tokens.
    struct Script {
        replies: Vec<Result<String, String>>,
        prompts: Vec<String>,
    }

    impl Script {
        fn new(replies: &[&str]) -> Script {
            Script {
                replies: replies.iter().rev().map(|r| Ok(r.to_string())).collect(),
                prompts: Vec::new(),
            }
        }
    }

    impl Chat for Script {
        fn complete(&mut self, _: &str, user: &str, _: usize) -> Result<String> {
            self.prompts.push(user.to_string());
            match self.replies.pop() {
                Some(Ok(reply)) => Ok(reply),
                Some(Err(e)) => Err(anyhow!(e)),
                None => Ok(String::new()),
            }
        }
        fn count_tokens(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
    }

    fn input(lines: usize, words: usize, seconds_apart: f64) -> ChaptersInput {
        let text = "word ".repeat(words);
        ChaptersInput {
            version: VERSION,
            duration: lines as f64 * seconds_apart,
            lines: (0..lines)
                .map(|i| TimedText {
                    id: i as u64 + 1,
                    start: i as f64 * seconds_apart,
                    end: (i + 1) as f64 * seconds_apart,
                    text: text.clone(),
                })
                .collect(),
        }
    }

    fn run(chat: &mut Script, input: &ChaptersInput) -> Result<ChaptersOutput> {
        chapters(chat, input, &mut |_, _| {})
    }

    #[test]
    fn times_come_from_the_lines_and_the_merge_can_rename() {
        let mut chat = Script::new(&["1|Intro\n7|Budget\n", "1|Welcome\n2|Budget review\n"]);
        let out = run(&mut chat, &input(20, 5, 30.0)).unwrap();
        assert_eq!(out.chapters.len(), 2);
        assert_eq!(out.chapters[0].start, 0.0);
        assert_eq!(out.chapters[1].start, 180.0, "line 7 starts at 180 s");
        assert_eq!(out.chapters[1].title, "Budget review");
    }

    #[test]
    fn a_long_recording_is_read_in_windows_and_a_bad_window_is_survivable() {
        // 60 lines of about 100 tokens: three windows
        let data = input(60, 100, 30.0);
        let mut chat = Script::new(&["1|First\n", "garbage", "garbage", "55|Late subject\n"]);
        let out = run(&mut chat, &data).unwrap();
        assert!(out.stats.windows >= 3, "{:?}", out.stats);
        assert_eq!(out.stats.failed_windows, 1);
        assert_eq!(out.chapters[0].title, "First");
        assert_eq!(out.chapters.last().unwrap().title, "Late subject");
    }

    #[test]
    fn a_failed_merge_keeps_the_chapters_from_the_windows() {
        let mut chat = Script::new(&["1|Intro\n7|Budget\n", "nonsense", "nonsense"]);
        let out = run(&mut chat, &input(20, 5, 30.0)).unwrap();
        let titles: Vec<&str> = out.chapters.iter().map(|c| c.title.as_str()).collect();
        assert_eq!(titles, ["Intro", "Budget"]);
    }

    #[test]
    fn one_chapter_needs_no_merge_call() {
        let mut chat = Script::new(&["1|Only subject\n"]);
        let out = run(&mut chat, &input(3, 5, 30.0)).unwrap();
        assert_eq!(out.chapters.len(), 1);
        assert_eq!(chat.prompts.len(), 1);
    }

    #[test]
    fn nothing_usable_is_an_error() {
        let mut chat = Script::new(&["no", "no"]);
        assert!(run(&mut chat, &input(3, 5, 30.0)).is_err());
    }

    #[test]
    fn hard_errors_in_a_row_stop_the_run() {
        let mut chat = Script {
            replies: vec![Err("decode failed".to_string()); 10],
            prompts: Vec::new(),
        };
        let err = run(&mut chat, &input(100, 100, 30.0)).unwrap_err();
        assert!(err.to_string().contains("decode failed"));
        assert_eq!(chat.prompts.len(), MAX_HARD_ERRORS);
    }
}
