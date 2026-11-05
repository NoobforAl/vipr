//! Split-and-retry around model calls, shared by translate and compare. `refine.rs` keeps its own logic.

use std::fmt;

use anyhow::Result;

use crate::refine::Chat;

/// Hard errors in a row that stop the run. The checkpoint allows resuming.
pub const MAX_HARD_ERRORS: usize = 3;
/// Room kept for what the chat template adds around the two messages.
const TEMPLATE_TOKENS: usize = 64;

/// The model answered, but the answer can't be used. `call` returns this for a failure worth a
/// split or a retry; any other error is a hard error, such as a failed decode.
#[derive(Debug)]
pub struct BadReply(pub String);

impl fmt::Display for BadReply {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unusable reply: {}", self.0)
    }
}

impl std::error::Error for BadReply {}

/// Makes a prompt that can't fit the model's context a [`BadReply`], not a hard error, so imported
/// text that is too long fails its own chunk (which then splits) instead of stopping the run.
pub struct FitContext<C> {
    pub chat: C,
    n_ctx: usize,
}

impl<C: Chat> FitContext<C> {
    pub fn new(chat: C, n_ctx: usize) -> FitContext<C> {
        FitContext { chat, n_ctx }
    }
}

impl<C: Chat> Chat for FitContext<C> {
    fn complete(&mut self, system: &str, user: &str, max_new_tokens: usize) -> Result<String> {
        let prompt =
            self.chat.count_tokens(system) + self.chat.count_tokens(user) + TEMPLATE_TOKENS;
        if prompt + max_new_tokens > self.n_ctx {
            let message = format!(
                "a prompt of about {prompt} tokens and a reply of up to {max_new_tokens} don't fit \
                 the context of {}",
                self.n_ctx
            );
            return Err(BadReply(message).into());
        }
        // TEMPLATE_TOKENS is an estimate, so the backend's own check can still refuse the prompt.
        self.chat
            .complete(system, user, max_new_tokens)
            .map_err(|e| {
                if e.to_string().contains("does not fit") {
                    BadReply(e.to_string()).into()
                } else {
                    e
                }
            })
    }

    fn count_tokens(&self, text: &str) -> usize {
        self.chat.count_tokens(text)
    }
}

/// Runs `call` on chunks of `items`. On failure a chunk is split in half, then a single item gets one
/// stricter retry (`call`'s `bool` is `strict`), then `give_up` decides its result. Output stays in
/// input order. `on_done` receives each finished piece of results and how many items are done.
///
/// `call` must return one result per item; a wrong count is treated as [`BadReply`]. Hard errors take
/// the same path, but [`MAX_HARD_ERRORS`] in a row end the run with the last error. Any answer from
/// the model, even an unusable one, resets that count.
pub fn run_chunks<T, R>(
    items: &[T],
    chunk_size: usize,
    call: impl FnMut(&[T], bool) -> Result<Vec<R>>,
    give_up: impl FnMut(&T) -> R,
    on_done: impl FnMut(&[R], usize),
) -> Result<Vec<R>> {
    run_weighted_chunks(items, chunk_size, |_| 1, call, give_up, on_done)
}

/// [`run_chunks`] for items of different sizes: a chunk's weights add up to at most `max_weight`, though
/// a heavier item still gets a chunk of its own.
pub fn run_weighted_chunks<T, R>(
    items: &[T],
    max_weight: usize,
    weight: impl Fn(&T) -> usize,
    mut call: impl FnMut(&[T], bool) -> Result<Vec<R>>,
    mut give_up: impl FnMut(&T) -> R,
    mut on_done: impl FnMut(&[R], usize),
) -> Result<Vec<R>> {
    let mut run = Run {
        call: &mut call,
        give_up: &mut give_up,
        on_done: &mut on_done,
        hard_errors: 0,
        out: Vec::with_capacity(items.len()),
    };
    let mut start = 0;
    while start < items.len() {
        let mut end = start + 1;
        let mut used = weight(&items[start]);
        while end < items.len() && used + weight(&items[end]) <= max_weight {
            used += weight(&items[end]);
            end += 1;
        }
        run.solve(&items[start..end])?;
        start = end;
    }
    Ok(run.out)
}

type Call<'a, T, R> = &'a mut dyn FnMut(&[T], bool) -> Result<Vec<R>>;

struct Run<'a, T, R> {
    call: Call<'a, T, R>,
    give_up: &'a mut dyn FnMut(&T) -> R,
    on_done: &'a mut dyn FnMut(&[R], usize),
    hard_errors: usize,
    out: Vec<R>,
}

impl<T, R> Run<'_, T, R> {
    /// `Ok(None)` is a failed attempt; `Err` stops the run.
    fn attempt(&mut self, chunk: &[T], strict: bool) -> Result<Option<Vec<R>>> {
        match (self.call)(chunk, strict) {
            Ok(results) => {
                self.hard_errors = 0;
                Ok((results.len() == chunk.len()).then_some(results))
            }
            Err(e) if e.is::<BadReply>() => {
                self.hard_errors = 0;
                Ok(None)
            }
            Err(e) => {
                self.hard_errors += 1;
                if self.hard_errors >= MAX_HARD_ERRORS {
                    return Err(e);
                }
                Ok(None)
            }
        }
    }

    fn finish(&mut self, results: Vec<R>) {
        let start = self.out.len();
        self.out.extend(results);
        (self.on_done)(&self.out[start..], self.out.len());
    }

    fn solve(&mut self, chunk: &[T]) -> Result<()> {
        if let Some(results) = self.attempt(chunk, false)? {
            self.finish(results);
            return Ok(());
        }
        if let [only] = chunk {
            let results = match self.attempt(chunk, true)? {
                Some(results) => results,
                None => vec![(self.give_up)(only)],
            };
            self.finish(results);
            return Ok(());
        }
        let (left, right) = chunk.split_at(chunk.len() / 2);
        self.solve(left)?;
        self.solve(right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    /// The result, each call as `(items, strict)`, and each `on_done` count.
    type Outcome = (Result<Vec<u32>>, Vec<(Vec<u32>, bool)>, Vec<usize>);

    /// Doubles every item.
    fn run(
        items: &[u32],
        size: usize,
        mut fails: impl FnMut(&[u32], bool) -> Option<anyhow::Error>,
    ) -> Outcome {
        let mut calls = Vec::new();
        let mut done = Vec::new();
        let result = run_chunks(
            items,
            size,
            |chunk: &[u32], strict| {
                calls.push((chunk.to_vec(), strict));
                match fails(chunk, strict) {
                    Some(e) => Err(e),
                    None => Ok(chunk.iter().map(|x| x * 2).collect()),
                }
            },
            |_| 0,
            |_, n| done.push(n),
        );
        (result, calls, done)
    }

    fn bad() -> anyhow::Error {
        BadReply("x".into()).into()
    }

    /// Counts words as tokens and answers with `reply`, remembering how many calls it got.
    struct Fixed {
        reply: fn() -> Result<String>,
        calls: usize,
    }

    impl Chat for Fixed {
        fn complete(&mut self, _system: &str, _user: &str, _max: usize) -> Result<String> {
            self.calls += 1;
            (self.reply)()
        }
        fn count_tokens(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
    }

    #[test]
    fn a_prompt_that_cannot_fit_is_a_bad_reply_without_a_model_call() {
        let inner = Fixed {
            reply: || Ok("fine".into()),
            calls: 0,
        };
        let mut chat = FitContext::new(inner, TEMPLATE_TOKENS + 10);
        assert_eq!(chat.complete("two words", "two words", 6).unwrap(), "fine");
        let err = chat.complete("two words", "two words", 7).unwrap_err();
        assert!(err.is::<BadReply>(), "{err}");
        assert_eq!(chat.chat.calls, 1);
    }

    #[test]
    fn only_the_backends_overflow_error_becomes_a_bad_reply() {
        let overflow = Fixed {
            reply: || {
                Err(anyhow!(
                    "prompt of 5000 tokens does not fit the context of 4096"
                ))
            },
            calls: 0,
        };
        let err = FitContext::new(overflow, 4096)
            .complete("s", "u", 1)
            .unwrap_err();
        assert!(err.is::<BadReply>(), "{err}");

        let lost = Fixed {
            reply: || Err(anyhow!("decoding: gpu lost")),
            calls: 0,
        };
        let err = FitContext::new(lost, 4096)
            .complete("s", "u", 1)
            .unwrap_err();
        assert!(!err.is::<BadReply>(), "{err}");
    }

    #[test]
    fn good_chunks_run_once_in_order() {
        let (result, calls, done) = run(&[1, 2, 3, 4, 5], 2, |_, _| None);
        assert_eq!(result.unwrap(), [2, 4, 6, 8, 10]);
        assert_eq!(calls.len(), 3);
        assert!(calls.iter().all(|(_, strict)| !strict));
        assert_eq!(done, [2, 4, 5]);
    }

    #[test]
    fn weights_decide_what_shares_a_chunk() {
        let mut calls: Vec<Vec<u32>> = Vec::new();
        let result = run_weighted_chunks(
            &[1, 2, 3, 9, 1, 1],
            4,
            |x: &u32| *x as usize,
            |chunk: &[u32], _| {
                calls.push(chunk.to_vec());
                Ok(chunk.to_vec())
            },
            |_| 0,
            |_, _| {},
        );
        assert_eq!(result.unwrap(), [1, 2, 3, 9, 1, 1]);
        assert_eq!(calls, [vec![1, 2], vec![3], vec![9], vec![1, 1]]);
    }

    #[test]
    fn no_items_means_no_calls() {
        let (result, calls, _) = run(&[], 12, |_, _| None);
        assert!(result.unwrap().is_empty());
        assert!(calls.is_empty());
        // a zero chunk size can't loop forever
        assert_eq!(run(&[1], 0, |_, _| None).0.unwrap(), [2]);
    }

    #[test]
    fn a_failed_chunk_splits_then_retries_strictly_then_gives_up() {
        // 3 never works, and everything longer than one item fails too
        let (result, calls, done) = run(&[1, 2, 3], 3, |chunk, _| {
            (chunk.len() > 1 || chunk == [3]).then(bad)
        });
        assert_eq!(result.unwrap(), [2, 4, 0], "input order, with 3 given up");
        let expected = [
            (vec![1, 2, 3], false),
            (vec![1], false),
            (vec![2, 3], false),
            (vec![2], false),
            (vec![3], false),
            (vec![3], true),
        ];
        assert_eq!(calls, expected);
        assert_eq!(done, [1, 2, 3]);
    }

    #[test]
    fn the_strict_retry_can_save_an_item() {
        let (result, _, _) = run(&[1], 1, |_, strict| (!strict).then(bad));
        assert_eq!(result.unwrap(), [2]);
    }

    #[test]
    fn a_wrong_result_count_is_a_failure() {
        let result = run_chunks(
            &[1, 2],
            2,
            |chunk: &[i32], _| {
                Ok(if chunk.len() == 2 {
                    vec![9]
                } else {
                    vec![chunk[0]]
                })
            },
            |_| 0,
            |_, _| {},
        );
        assert_eq!(result.unwrap(), [1, 2]);
    }

    #[test]
    fn three_hard_errors_in_a_row_stop_the_run() {
        let (result, calls, done) = run(&[1, 2, 3, 4], 4, |_, _| Some(anyhow!("gpu lost")));
        assert_eq!(result.unwrap_err().to_string(), "gpu lost");
        assert_eq!(calls.len(), MAX_HARD_ERRORS);
        assert!(done.is_empty());
    }

    #[test]
    fn earlier_results_are_reported_before_the_stop() {
        let (result, _, done) = run(&[1, 2, 3], 1, |chunk, _| {
            (chunk[0] > 1).then(|| anyhow!("gpu lost"))
        });
        assert!(result.is_err());
        assert_eq!(done[0], 1);
    }

    #[test]
    fn an_answer_resets_the_hard_error_count() {
        let mut n = 0;
        // hard, hard, then an unusable answer, forever: never three in a row
        let (result, _, _) = run(&[1, 2, 3, 4], 4, |_, _| {
            n += 1;
            Some(if n % 3 == 0 { bad() } else { anyhow!("flaky") })
        });
        assert_eq!(result.unwrap(), [0, 0, 0, 0]);
    }
}
