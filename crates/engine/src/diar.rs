//! Speaker turns from the diarization worker and the word -> speaker merge.
//!
//! The worker (`diar-worker`) writes `{"turns":[{"start":..,"end":..,"speaker":0}, ...]}`.
//! `assign_speaker` gives each word (or any time span) the speaker whose turns overlap it most,
//! and falls back to the nearest turn when the span lies in a gap (Whisper word times and
//! diarization boundaries never line up exactly).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DiarTurn {
    pub start: f64,
    pub end: f64,
    /// 0-based speaker index (Speaker 1 = 0).
    pub speaker: usize,
}

/// The diarization worker's output file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarResult {
    pub num_speakers: usize,
    pub audio_seconds: f64,
    #[serde(default)]
    pub elapsed_seconds: f64,
    pub turns: Vec<DiarTurn>,
}

impl DiarResult {
    pub fn load(path: &std::path::Path) -> crate::Result<DiarResult> {
        let text = std::fs::read_to_string(path).map_err(crate::io_err(path))?;
        serde_json::from_str(&text).map_err(|e| crate::Error::Registry {
            path: path.display().to_string(),
            msg: e.to_string(),
        })
    }
}

/// Speaker of the span `[t0, t1]`: the one with the largest total overlap; if nothing overlaps,
/// the speaker of the nearest turn. `None` only when there are no turns at all.
/// `turns` must be sorted by `start` (the worker guarantees it).
pub fn assign_speaker(turns: &[DiarTurn], t0: f64, t1: f64) -> Option<usize> {
    if turns.is_empty() {
        return None;
    }
    let t1 = t1.max(t0);
    // first turn that could overlap: turns are sorted and non-overlapping, so binary search on end
    let first = turns.partition_point(|t| t.end <= t0);
    let mut best: Option<(usize, f64)> = None;
    let mut per_speaker: Vec<(usize, f64)> = Vec::new();
    for t in &turns[first..] {
        if t.start >= t1 {
            break;
        }
        let overlap = t1.min(t.end) - t0.max(t.start);
        if overlap > 0.0 {
            match per_speaker.iter_mut().find(|(s, _)| *s == t.speaker) {
                Some((_, o)) => *o += overlap,
                None => per_speaker.push((t.speaker, overlap)),
            }
        }
    }
    for (s, o) in per_speaker {
        if best.is_none_or(|(_, bo)| o > bo) {
            best = Some((s, o));
        }
    }
    if let Some((s, _)) = best {
        return Some(s);
    }
    // in a gap (or a zero-length word on a boundary): nearest turn by distance to its edge
    let mid = (t0 + t1) / 2.0;
    turns
        .iter()
        .min_by(|a, b| dist(a, mid).total_cmp(&dist(b, mid)))
        .map(|t| t.speaker)
}

fn dist(t: &DiarTurn, x: f64) -> f64 {
    if x < t.start {
        t.start - x
    } else if x > t.end {
        x - t.end
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turns() -> Vec<DiarTurn> {
        vec![
            DiarTurn {
                start: 0.0,
                end: 4.0,
                speaker: 0,
            },
            DiarTurn {
                start: 4.0,
                end: 6.0,
                speaker: 1,
            },
            DiarTurn {
                start: 8.0,
                end: 10.0,
                speaker: 0,
            },
        ]
    }

    #[test]
    fn word_inside_a_turn() {
        assert_eq!(assign_speaker(&turns(), 1.0, 1.4), Some(0));
        assert_eq!(assign_speaker(&turns(), 4.5, 5.0), Some(1));
    }

    #[test]
    fn word_across_a_boundary_takes_the_larger_overlap() {
        // 3.5-4.2: 0.5 s in speaker 0, 0.2 s in speaker 1
        assert_eq!(assign_speaker(&turns(), 3.5, 4.2), Some(0));
        // 3.9-5.0: 0.1 s vs 1.0 s
        assert_eq!(assign_speaker(&turns(), 3.9, 5.0), Some(1));
    }

    #[test]
    fn word_in_a_gap_uses_the_nearest_turn() {
        assert_eq!(assign_speaker(&turns(), 6.2, 6.5), Some(1)); // nearer to the turn ending at 6
        assert_eq!(assign_speaker(&turns(), 7.5, 7.8), Some(0)); // nearer to the turn starting at 8
        assert_eq!(assign_speaker(&turns(), 20.0, 20.5), Some(0)); // after the end
    }

    #[test]
    fn zero_length_word_and_empty_turns() {
        assert_eq!(assign_speaker(&turns(), 5.0, 5.0), Some(1));
        assert_eq!(assign_speaker(&[], 1.0, 2.0), None);
    }
}
