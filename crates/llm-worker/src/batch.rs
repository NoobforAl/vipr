//! Split-and-retry around model calls, shared by translate and compare.
//!
//! Stub from the foundation step; agent B fills in the body. `refine.rs` keeps its own logic.

use anyhow::{Result, bail};

/// Hard errors in a row that stop the run. The checkpoint allows resuming.
pub const MAX_HARD_ERRORS: usize = 3;

/// Runs `call` on chunks of `items`. On failure a chunk is split in half, then a single item gets one
/// stricter retry (`call`'s `bool` is `strict`), then `give_up` decides its result. Output stays in
/// input order. `on_done` receives each finished piece of results and how many items are done.
pub fn run_chunks<T, R>(
    _items: &[T],
    _chunk_size: usize,
    _call: impl FnMut(&[T], bool) -> Result<Vec<R>>,
    _give_up: impl FnMut(&T) -> R,
    _on_done: impl FnMut(&[R], usize),
) -> Result<Vec<R>> {
    bail!("batch::run_chunks is not implemented yet")
}
