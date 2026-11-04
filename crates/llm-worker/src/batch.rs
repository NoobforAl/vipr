//! Split-and-retry around model calls, shared by translate and compare. `refine.rs` keeps its own logic.

use std::fmt;

use anyhow::Result;

/// Hard errors in a row that stop the run. The checkpoint allows resuming.
pub const MAX_HARD_ERRORS: usize = 3;

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
    for chunk in items.chunks(chunk_size.max(1)) {
        run.solve(chunk)?;
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

    #[test]
    fn good_chunks_run_once_in_order() {
        let (result, calls, done) = run(&[1, 2, 3, 4, 5], 2, |_, _| None);
        assert_eq!(result.unwrap(), [2, 4, 6, 8, 10]);
        assert_eq!(calls.len(), 3);
        assert!(calls.iter().all(|(_, strict)| !strict));
        assert_eq!(done, [2, 4, 5]);
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
