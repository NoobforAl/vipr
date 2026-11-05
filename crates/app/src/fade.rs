//! Soft motion: how far along a fade is, eased so that it starts and ends gently.

use std::time::{Duration, Instant};

/// 0 at `since`, 1 from `since + length` on, for the moment `at`.
pub fn eased_at(since: Instant, at: Instant, length: Duration) -> f32 {
    let t = at.saturating_duration_since(since).as_secs_f32() / length.as_secs_f32();
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// [`eased_at`] for now.
pub fn eased(since: Instant, length: Duration) -> f32 {
    eased_at(since, Instant::now(), length)
}

pub fn running(since: Instant, length: Duration) -> bool {
    since.elapsed() < length
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fade_starts_and_ends_gently_and_stays_in_range() {
        let start = Instant::now();
        let length = Duration::from_millis(100);
        let at = |ms| eased_at(start, start + Duration::from_millis(ms), length);
        assert_eq!(at(0), 0.0);
        assert_eq!(at(100), 1.0);
        assert_eq!(at(500), 1.0, "holds at the end");
        assert!((at(50) - 0.5).abs() < 1e-4);
        assert!(at(10) < 0.1, "slow at first: {}", at(10));
        assert!(at(90) > 0.9, "slow at the end: {}", at(90));
        // a moment before the start is the start
        assert_eq!(eased_at(start + length, start, length), 0.0);
    }
}
