//! Lining up two subtitle tracks for the compare table.
//!
//! Stub from the foundation step; agent A fills in the bodies.

use crate::project::{CompareRow, TimedText};

/// The largest constant shift [`best_offset`] looks for, in seconds.
pub const MAX_OFFSET: f64 = 30.0;

/// Seconds to add to `b`'s times so it best overlaps `a`, within ±[`MAX_OFFSET`].
pub fn best_offset(_a: &[TimedText], _b: &[TimedText]) -> f64 {
    todo!("agent A")
}

/// Rows from cues with overlap ≥ 0.3 of the shorter duration, after adding `offset` to `b`'s times.
/// Rows are in time order with ids from 1. Cues with no partner become one-sided rows. AI status is
/// `Identical` or `OneSided` where that is already known and `Pending` otherwise.
pub fn align(_a: &[TimedText], _b: &[TimedText], _offset: f64) -> Vec<CompareRow> {
    todo!("agent A")
}
