//! Lining up two subtitle tracks for the compare table.

use crate::project::{AiChoice, AiPick, AiStatus, CompareRow, TimedText};

/// The largest constant shift [`best_offset`] looks for, in seconds.
pub const MAX_OFFSET: f64 = 30.0;

/// A cue overlaps its partner by at least this share of the shorter of the two.
const MIN_OVERLAP: f64 = 0.3;

/// A shift must beat no shift at all by this share, or it is noise. Without this, tracks that have
/// little to do with each other would get a confident-looking offset.
const MIN_GAIN: f64 = 0.01;

fn is_timed(c: &TimedText) -> bool {
    c.start.is_finite() && c.end.is_finite() && c.end > c.start
}

/// Time covered by a track, in milliseconds: sorted, with overlapping cues merged, so that a track
/// stacking several cues at one moment isn't counted twice.
fn coverage(cues: &[TimedText]) -> Vec<(i64, i64)> {
    let mut spans: Vec<(i64, i64)> = cues
        .iter()
        .filter(|c| is_timed(c))
        .map(|c| {
            (
                (c.start * 1000.0).round() as i64,
                (c.end * 1000.0).round() as i64,
            )
        })
        .collect();
    spans.sort_unstable();
    let mut merged: Vec<(i64, i64)> = Vec::with_capacity(spans.len());
    for (start, end) in spans {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// Milliseconds that `a` and `b` shifted by `shift` have in common. Both must be from [`coverage`].
fn shared(a: &[(i64, i64)], b: &[(i64, i64)], shift: i64) -> i64 {
    let (mut i, mut j, mut total) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        let (b_start, b_end) = (b[j].0 + shift, b[j].1 + shift);
        total += (a[i].1.min(b_end) - a[i].0.max(b_start)).max(0);
        if a[i].1 < b_end {
            i += 1;
        } else {
            j += 1;
        }
    }
    total
}

/// The best shift among `center`, `center ± step`, `center ± 2 step` and so on up to `reach`, nearest
/// to `center` on a tie. Returns the shift and its score.
fn search(a: &[(i64, i64)], b: &[(i64, i64)], center: i64, step: i64, reach: i64) -> (i64, i64) {
    let limit = (MAX_OFFSET * 1000.0) as i64;
    let mut best = (center, shared(a, b, center));
    for n in 1..=reach / step {
        for shift in [center + n * step, center - n * step] {
            if shift.abs() <= limit {
                let score = shared(a, b, shift);
                if score > best.1 {
                    best = (shift, score);
                }
            }
        }
    }
    best
}

/// Seconds to add to `b`'s times so it best overlaps `a`, within ±[`MAX_OFFSET`]. `0.0` when nothing
/// beats leaving the track where it is.
pub fn best_offset(a: &[TimedText], b: &[TimedText]) -> f64 {
    let (a, b) = (coverage(a), coverage(b));
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let (coarse, _) = search(&a, &b, 0, 100, (MAX_OFFSET * 1000.0) as i64);
    let (fine, score) = search(&a, &b, coarse, 10, 100);
    let unshifted = shared(&a, &b, 0);
    if score as f64 <= unshifted as f64 * (1.0 + MIN_GAIN) {
        return 0.0;
    }
    fine as f64 / 1000.0
}

/// Whitespace and invisible direction marks don't make two lines different.
fn same_words(a: &str, b: &str) -> bool {
    let words = |s: &'_ str| -> Vec<String> {
        s.replace(['\u{200E}', '\u{200F}'], "")
            .split_whitespace()
            .map(str::to_string)
            .collect()
    };
    words(a) == words(b)
}

/// One cue with `b`'s offset already applied.
#[derive(Clone, Copy)]
struct Span {
    start: f64,
    end: f64,
}

fn spans(cues: &[TimedText], offset: f64) -> Vec<Span> {
    cues.iter()
        .map(|c| Span {
            start: c.start + offset,
            end: c.end + offset,
        })
        .collect()
}

/// Every pair of cues that overlaps by enough, biggest overlap first: `(a index, b index)`.
fn qualifying_pairs(a: &[TimedText], b: &[TimedText], offset: f64) -> Vec<(usize, usize)> {
    let b_spans = spans(b, offset);
    let mut b_order: Vec<usize> = (0..b.len()).filter(|&j| is_timed(&b[j])).collect();
    b_order.sort_by(|&x, &y| b_spans[x].start.total_cmp(&b_spans[y].start));
    // the latest end so far, which lets the scan skip what has ended even if cues overlap
    let mut latest = f64::NEG_INFINITY;
    let latest_end: Vec<f64> = b_order
        .iter()
        .map(|&j| {
            latest = latest.max(b_spans[j].end);
            latest
        })
        .collect();

    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (i, cue) in a.iter().enumerate().filter(|(_, c)| is_timed(c)) {
        let first = latest_end.partition_point(|&end| end <= cue.start);
        for &j in &b_order[first..] {
            let other = b_spans[j];
            if other.start >= cue.end {
                break;
            }
            let overlap = cue.end.min(other.end) - cue.start.max(other.start);
            let shorter = (cue.end - cue.start).min(other.end - other.start);
            if overlap > 0.0 && overlap >= MIN_OVERLAP * shorter - 1e-9 {
                pairs.push((overlap, i, j));
            }
        }
    }
    pairs.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
    pairs.into_iter().map(|(_, i, j)| (i, j)).collect()
}

/// Cue indexes on each side of one row.
#[derive(Default)]
struct Group {
    a: Vec<usize>,
    b: Vec<usize>,
}

/// Each cue pairs with its best free partner first, then a cue left over joins the row of a paired
/// cue it overlaps, so one long line against two short ones is a single row. Only paired cues take
/// in others: two tracks that overlap each other all the way along must not chain into one huge row.
fn group_cues(a_len: usize, b_len: usize, pairs: &[(usize, usize)]) -> Vec<Group> {
    let mut a_group: Vec<Option<usize>> = vec![None; a_len];
    let mut b_group: Vec<Option<usize>> = vec![None; b_len];
    let mut groups: Vec<Group> = Vec::new();

    for &(i, j) in pairs {
        if a_group[i].is_none() && b_group[j].is_none() {
            a_group[i] = Some(groups.len());
            b_group[j] = Some(groups.len());
            groups.push(Group {
                a: vec![i],
                b: vec![j],
            });
        }
    }
    let (a_hosts, b_hosts) = (a_group.clone(), b_group.clone());
    for &(i, j) in pairs {
        if a_group[i].is_none()
            && let Some(g) = b_hosts[j]
        {
            a_group[i] = Some(g);
            groups[g].a.push(i);
        } else if b_group[j].is_none()
            && let Some(g) = a_hosts[i]
        {
            b_group[j] = Some(g);
            groups[g].b.push(j);
        }
    }
    for (i, _) in a_group.iter().enumerate().filter(|(_, g)| g.is_none()) {
        groups.push(Group {
            a: vec![i],
            b: Vec::new(),
        });
    }
    for (j, _) in b_group.iter().enumerate().filter(|(_, g)| g.is_none()) {
        groups.push(Group {
            a: Vec::new(),
            b: vec![j],
        });
    }
    groups
}

/// The cues' text in time order, or `None` when there is none.
fn side_text(cues: &[TimedText], members: &mut [usize], times: &[Span]) -> Option<String> {
    members.sort_by(|&x, &y| times[x].start.total_cmp(&times[y].start));
    let parts: Vec<&str> = members
        .iter()
        .map(|&m| cues[m].text.trim())
        .filter(|t| !t.is_empty())
        .collect();
    (!parts.is_empty()).then(|| parts.join("\n"))
}

/// What can be said about a row without asking the LLM.
fn known_pick(a: Option<&str>, b: Option<&str>) -> AiPick {
    let settled = |choice, status| AiPick {
        choice: Some(choice),
        status,
        ..AiPick::default()
    };
    match (a, b) {
        (Some(a), Some(b)) if same_words(a, b) => settled(AiChoice::A, AiStatus::Identical),
        (Some(_), None) => settled(AiChoice::A, AiStatus::OneSided),
        (None, Some(_)) => settled(AiChoice::B, AiStatus::OneSided),
        _ => AiPick::default(),
    }
}

/// Rows from cues with overlap ≥ 0.3 of the shorter duration, after adding `offset` to `b`'s times.
/// Rows are in time order with ids from 1. Cues with no partner become one-sided rows. AI status is
/// `Identical` or `OneSided` where that is already known and `Pending` otherwise.
///
/// A row can hold several cues of one side, their text joined with `\n`, when one line on one side
/// covers two on the other. Cues with no text leave no row.
pub fn align(a: &[TimedText], b: &[TimedText], offset: f64) -> Vec<CompareRow> {
    let pairs = qualifying_pairs(a, b, offset);
    let a_spans = spans(a, 0.0);
    let b_spans = spans(b, offset);

    let mut rows: Vec<CompareRow> = group_cues(a.len(), b.len(), &pairs)
        .into_iter()
        .filter_map(|mut group| {
            let a_text = side_text(a, &mut group.a, &a_spans);
            let b_text = side_text(b, &mut group.b, &b_spans);
            if a_text.is_none() && b_text.is_none() {
                return None;
            }
            let members = || {
                let a_members = group.a.iter().map(|&i| a_spans[i]);
                a_members.chain(group.b.iter().map(|&j| b_spans[j]))
            };
            let ai = known_pick(a_text.as_deref(), b_text.as_deref());
            Some(CompareRow {
                id: 0,
                start: members().map(|s| s.start).fold(f64::INFINITY, f64::min),
                end: members().map(|s| s.end).fold(f64::NEG_INFINITY, f64::max),
                a: a_text,
                b: b_text,
                ai,
                user: None,
            })
        })
        .collect();

    rows.sort_by(|x, y| x.start.total_cmp(&y.start).then(x.end.total_cmp(&y.end)));
    for (n, row) in rows.iter_mut().enumerate() {
        row.id = n as u64 + 1;
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(id: u64, start: f64, end: f64, text: &str) -> TimedText {
        TimedText {
            id,
            start,
            end,
            text: text.into(),
        }
    }

    /// Irregular cues over about ten minutes, from a small fixed generator.
    fn track(n: u64) -> Vec<TimedText> {
        let mut seed = 12345u64;
        let mut next = move |range: u64| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) % range) as f64 / 1000.0
        };
        let mut t = 2.0;
        (1..=n)
            .map(|id| {
                let start = t + next(2500);
                let end = start + 0.8 + next(3500);
                t = end;
                cue(id, start, end, &format!("line {id}"))
            })
            .collect()
    }

    fn shifted(cues: &[TimedText], by: f64) -> Vec<TimedText> {
        cues.iter()
            .map(|c| TimedText {
                start: c.start + by,
                end: c.end + by,
                ..c.clone()
            })
            .collect()
    }

    #[test]
    fn offset_recovers_a_known_shift() {
        let a = track(150);
        for shift in [7.3, -12.84, 0.5, 29.5, -29.5] {
            let b = shifted(&a, -shift);
            let found = best_offset(&a, &b);
            assert!((found - shift).abs() < 0.011, "{shift}: found {found}");
        }
    }

    #[test]
    fn offset_survives_missing_cues_and_jitter() {
        let a = track(200);
        let b: Vec<TimedText> = shifted(&a, -11.2)
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i % 5 != 0)
            .map(|(i, c)| {
                let jitter = [-0.06, 0.04, 0.02, -0.03][i % 4];
                TimedText {
                    start: c.start + jitter,
                    end: c.end - jitter,
                    ..c
                }
            })
            .collect();
        let found = best_offset(&a, &b);
        assert!((found - 11.2).abs() < 0.1, "found {found}");
    }

    #[test]
    fn no_shift_when_already_in_line_or_no_evidence() {
        let a = track(100);
        assert_eq!(best_offset(&a, &a), 0.0);
        assert_eq!(best_offset(&a, &[]), 0.0);
        assert_eq!(best_offset(&[], &a), 0.0);
        let b = [cue(1, 5000.0, 5001.0, "far away")];
        assert_eq!(best_offset(&a, &b), 0.0);
        let bad = [cue(1, 3.0, 3.0, "no length"), cue(2, 9.0, 4.0, "backwards")];
        assert_eq!(best_offset(&a, &bad), 0.0);
    }

    #[test]
    fn regular_spacing_does_not_invent_a_shift() {
        // every 3 s looks like every other 3 s, so the tracks fit at many shifts
        let a: Vec<TimedText> = (0..200)
            .map(|i| cue(i + 1, i as f64 * 3.0, i as f64 * 3.0 + 2.0, "x"))
            .collect();
        assert_eq!(best_offset(&a, &a), 0.0);
    }

    #[test]
    fn stacked_cues_are_not_counted_twice() {
        let a = [cue(1, 10.0, 12.0, "x")];
        let b = [
            cue(1, 0.0, 2.0, "y"),
            cue(2, 0.0, 2.0, "y again"),
            cue(3, 20.0, 21.0, "z"),
        ];
        assert_eq!(coverage(&b), [(0, 2000), (20000, 21000)]);
        assert!((best_offset(&a, &b) - 10.0).abs() < 0.011);
    }

    #[test]
    fn rows_pair_by_overlap_and_settle_what_they_can() {
        let a = [
            cue(1, 0.0, 2.0, "same"),
            cue(2, 3.0, 5.0, "left"),
            cue(3, 6.0, 7.0, "only a"),
        ];
        let b = [
            cue(1, 0.1, 2.1, "same"),
            cue(2, 3.2, 5.2, "right"),
            cue(3, 9.0, 10.0, "only b"),
        ];
        let rows = align(&a, &b, 0.0);
        assert_eq!(rows.len(), 4);
        assert_eq!(rows.iter().map(|r| r.id).collect::<Vec<_>>(), [1, 2, 3, 4]);

        assert_eq!(rows[0].ai.status, AiStatus::Identical);
        assert_eq!(rows[0].ai.choice, Some(AiChoice::A));
        assert_eq!((rows[0].start, rows[0].end), (0.0, 2.1));

        assert_eq!(rows[1].a.as_deref(), Some("left"));
        assert_eq!(rows[1].b.as_deref(), Some("right"));
        assert_eq!(rows[1].ai.status, AiStatus::Pending);
        assert_eq!(rows[1].ai.choice, None);

        assert_eq!(rows[2].a.as_deref(), Some("only a"));
        assert_eq!(rows[2].b, None);
        assert_eq!(rows[2].ai.status, AiStatus::OneSided);
        assert_eq!(rows[2].ai.choice, Some(AiChoice::A));

        assert_eq!(rows[3].a, None);
        assert_eq!(rows[3].b.as_deref(), Some("only b"));
        assert_eq!(rows[3].ai.choice, Some(AiChoice::B));
    }

    #[test]
    fn overlap_threshold_is_of_the_shorter_cue() {
        let a = [cue(1, 0.0, 2.0, "a")];
        // 0.3 s of a 1 s cue: exactly enough
        let rows = align(&a, &[cue(1, 1.7, 2.7, "b")], 0.0);
        assert_eq!(rows.len(), 1);
        // 0.29 s of a 1 s cue: not enough
        let rows = align(&a, &[cue(1, 1.71, 2.71, "b")], 0.0);
        assert_eq!(rows.len(), 2);
        // a tiny cue fully inside a long one
        let rows = align(&[cue(1, 0.0, 60.0, "a")], &[cue(1, 30.0, 30.5, "b")], 0.0);
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn offset_applies_to_b() {
        let a = [cue(1, 10.0, 12.0, "a")];
        let b = [cue(1, 0.0, 2.0, "b")];
        assert_eq!(align(&a, &b, 0.0).len(), 2);
        let rows = align(&a, &b, 10.0);
        assert_eq!(rows.len(), 1);
        assert_eq!((rows[0].start, rows[0].end), (10.0, 12.0));
    }

    #[test]
    fn one_long_line_against_two_short_ones_is_one_row() {
        let a = [cue(1, 0.0, 6.0, "the whole sentence")];
        let b = [
            cue(1, 3.1, 6.0, "second half"),
            cue(2, 0.0, 3.0, "first half"),
        ];
        let rows = align(&a, &b, 0.0);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].b.as_deref(), Some("first half\nsecond half"));
        assert_eq!((rows[0].start, rows[0].end), (0.0, 6.0));
        assert_eq!(rows[0].ai.status, AiStatus::Pending);
    }

    #[test]
    fn offset_by_half_a_cue_does_not_chain_into_one_row() {
        let a: Vec<TimedText> = (0..100)
            .map(|i| cue(i + 1, i as f64 * 2.0, i as f64 * 2.0 + 2.0, "a"))
            .collect();
        let b: Vec<TimedText> = (0..100)
            .map(|i| cue(i + 1, i as f64 * 2.0 + 1.0, i as f64 * 2.0 + 3.0, "b"))
            .collect();
        let rows = align(&a, &b, 0.0);
        assert!(rows.len() >= 90, "chained into {} rows", rows.len());
        assert!(rows.iter().all(|r| r.end - r.start <= 6.0));
    }

    #[test]
    fn every_cue_lands_in_exactly_one_row() {
        let a = track(120);
        let mut b = shifted(&a, 0.7);
        b.truncate(100);
        for (i, c) in b.iter_mut().enumerate() {
            c.text = format!("other {i}");
        }
        let rows = align(&a, &b, 0.0);
        let count = |side: fn(&CompareRow) -> Option<&String>| -> usize {
            rows.iter()
                .filter_map(side)
                .map(|t| t.split('\n').count())
                .sum()
        };
        assert_eq!(count(|r| r.a.as_ref()), a.len());
        assert_eq!(count(|r| r.b.as_ref()), b.len());
        assert!(rows.windows(2).all(|w| w[0].start <= w[1].start));
    }

    #[test]
    fn identical_ignores_line_breaks_and_direction_marks() {
        let a = [cue(1, 0.0, 2.0, "one two\nthree")];
        let b = [cue(1, 0.0, 2.0, "\u{200F}one\ntwo three ")];
        assert_eq!(align(&a, &b, 0.0)[0].ai.status, AiStatus::Identical);
        let c = [cue(1, 0.0, 2.0, "One two three")];
        assert_eq!(align(&a, &c, 0.0)[0].ai.status, AiStatus::Pending);
    }

    #[test]
    fn empty_and_broken_cues_are_harmless() {
        assert_eq!(align(&[], &[], 0.0), []);
        let rows = align(&[cue(1, 0.0, 1.0, "  ")], &[cue(1, 0.0, 1.0, "b")], 0.0);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].a, None);
        assert_eq!(rows[0].ai.status, AiStatus::OneSided);
        assert_eq!(align(&[cue(1, 0.0, 1.0, " ")], &[], 0.0), []);
        // no length: kept as a one-sided row, never paired
        let rows = align(&[cue(1, 5.0, 5.0, "a")], &[cue(1, 0.0, 10.0, "b")], 0.0);
        assert_eq!(rows.len(), 2);
    }
}
