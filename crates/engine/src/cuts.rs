//! Parts of a recording to leave out of an export, as an editor's In and Out marks with Extract do.
//!
//! The cuts are ranges of the original timeline. Everything that refers to the recording (subtitles, marks, chapters)
//! keeps its own times: only an export closes the gaps.

use serde::{Deserialize, Serialize};

/// A cut shorter than this is a slip of the hand, not an edit.
pub const MIN_CUT_SECONDS: f64 = 0.2;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Cut {
    pub start: f64,
    pub end: f64,
}

/// `<stem>.cuts.json`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CutsFile {
    #[serde(default = "version")]
    pub version: u32,
    #[serde(default)]
    pub cuts: Vec<Cut>,
}

fn version() -> u32 {
    1
}

impl CutsFile {
    pub fn new(cuts: Vec<Cut>) -> CutsFile {
        CutsFile {
            version: version(),
            cuts,
        }
    }
}

/// The cuts as an export can use them: inside `[0, duration]`, in order, overlapping and touching ones joined, too
/// short ones dropped. A `duration` of 0 or less means the length isn't known, and nothing is clamped at the end.
pub fn normalize(cuts: &[Cut], duration: f64) -> Vec<Cut> {
    let end_of = |t: f64| if duration > 0.0 { t.min(duration) } else { t };
    let mut sorted: Vec<Cut> = cuts
        .iter()
        .filter(|c| c.start.is_finite() && c.end.is_finite())
        .map(|c| Cut {
            start: end_of(c.start.min(c.end).max(0.0)),
            end: end_of(c.start.max(c.end).max(0.0)),
        })
        .collect();
    sorted.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut out: Vec<Cut> = Vec::new();
    for c in sorted {
        match out.last_mut() {
            Some(last) if c.start <= last.end => last.end = last.end.max(c.end),
            _ => out.push(c),
        }
    }
    out.retain(|c| c.end - c.start >= MIN_CUT_SECONDS);
    out
}

/// The parts that stay, as `(start, end)` of the original timeline, `end` `None` for "to the end of the recording".
/// Expects [`normalize`]d cuts.
pub fn kept(cuts: &[Cut], duration: f64) -> Vec<(f64, Option<f64>)> {
    let mut parts = Vec::new();
    let mut from = 0.0;
    for c in cuts {
        if c.start > from {
            parts.push((from, Some(c.start)));
        }
        from = c.end;
    }
    if duration <= 0.0 || from < duration {
        parts.push((from, None));
    }
    parts
}

/// Seconds the cuts take out. Expects [`normalize`]d cuts.
pub fn removed(cuts: &[Cut]) -> f64 {
    cuts.iter().map(|c| c.end - c.start).sum()
}

/// The cut that covers `time`, if any.
pub fn cut_at(cuts: &[Cut], time: f64) -> Option<&Cut> {
    cuts.iter().find(|c| c.start <= time && time < c.end)
}

/// Where `time` of the original lands once the cuts are left out. A moment inside a cut lands where that cut was.
/// Expects [`normalize`]d cuts.
pub fn shift(cuts: &[Cut], time: f64) -> f64 {
    let mut out = time;
    for c in cuts {
        if time >= c.end {
            out -= c.end - c.start;
        } else if time > c.start {
            out -= time - c.start;
        }
    }
    out.max(0.0)
}

/// An ffmpeg `filter_complex` that keeps only `parts` and joins them. `video` runs on the whole picture first, so
/// subtitles drawn there keep their own times; `None` leaves the picture out. Trim and concat in one graph keep the
/// sound with the picture, which `select` does not over many cuts. Gives `[v]` and/or `[a]`; the sound is the first
/// sound track.
pub fn filter_complex(
    video: Option<&str>,
    parts: &[(f64, Option<f64>)],
    with_audio: bool,
) -> String {
    let n = parts.len().max(1);
    let range = |start: f64, end: Option<f64>| match end {
        Some(end) => format!("start={start:.3}:end={end:.3}"),
        None => format!("start={start:.3}"),
    };
    let mut graph = String::new();
    if let Some(filter) = video {
        graph.push_str(&format!("[0:v:0]{filter},split={n}"));
        for i in 0..n {
            graph.push_str(&format!("[vs{i}]"));
        }
        graph.push(';');
    }
    if with_audio {
        graph.push_str(&format!("[0:a:0]asplit={n}"));
        for i in 0..n {
            graph.push_str(&format!("[as{i}]"));
        }
        graph.push(';');
    }
    for (i, (start, end)) in parts.iter().enumerate() {
        if video.is_some() {
            graph.push_str(&format!(
                "[vs{i}]trim={},setpts=PTS-STARTPTS[v{i}];",
                range(*start, *end)
            ));
        }
        if with_audio {
            graph.push_str(&format!(
                "[as{i}]atrim={},asetpts=PTS-STARTPTS[a{i}];",
                range(*start, *end)
            ));
        }
    }
    for i in 0..n {
        if video.is_some() {
            graph.push_str(&format!("[v{i}]"));
        }
        if with_audio {
            graph.push_str(&format!("[a{i}]"));
        }
    }
    graph.push_str(&format!(
        "concat=n={n}:v={}:a={}",
        usize::from(video.is_some()),
        usize::from(with_audio)
    ));
    if video.is_some() {
        graph.push_str("[v]");
    }
    if with_audio {
        graph.push_str("[a]");
    }
    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cut(start: f64, end: f64) -> Cut {
        Cut { start, end }
    }

    #[test]
    fn cuts_are_sorted_joined_clamped_and_tiny_ones_dropped() {
        let got = normalize(
            &[
                cut(50.0, 40.0),
                cut(10.0, 20.0),
                cut(15.0, 25.0),
                cut(25.0, 26.0),
                cut(30.0, 30.1),
                cut(95.0, 200.0),
                cut(-5.0, 1.0),
            ],
            100.0,
        );
        assert_eq!(
            got,
            [
                cut(0.0, 1.0),
                cut(10.0, 26.0),
                cut(40.0, 50.0),
                cut(95.0, 100.0)
            ]
        );
        assert_eq!(removed(&got), 32.0);
    }

    #[test]
    fn what_stays_is_everything_between_the_cuts() {
        let cuts = normalize(&[cut(10.0, 20.0), cut(40.0, 50.0)], 100.0);
        assert_eq!(
            kept(&cuts, 100.0),
            [(0.0, Some(10.0)), (20.0, Some(40.0)), (50.0, None)]
        );
        // a cut at the very start or end leaves no empty part
        let edges = normalize(&[cut(0.0, 5.0), cut(90.0, 100.0)], 100.0);
        assert_eq!(kept(&edges, 100.0), [(5.0, Some(90.0))]);
        assert_eq!(kept(&[], 100.0), [(0.0, None)]);
    }

    #[test]
    fn the_cut_under_a_moment_is_found() {
        let cuts = [cut(10.0, 20.0)];
        assert!(cut_at(&cuts, 9.99).is_none());
        assert!(cut_at(&cuts, 10.0).is_some());
        assert!(
            cut_at(&cuts, 20.0).is_none(),
            "the end is where playback goes on"
        );
    }

    #[test]
    fn the_filter_draws_subtitles_first_then_joins_the_parts() {
        let graph = filter_complex(
            Some("subtitles=s.srt"),
            &[(0.0, Some(10.0)), (20.0, None)],
            true,
        );
        assert!(graph.starts_with("[0:v:0]subtitles=s.srt,split=2[vs0][vs1];"));
        assert!(graph.contains("[vs0]trim=start=0.000:end=10.000,setpts=PTS-STARTPTS[v0];"));
        assert!(graph.contains("[as1]atrim=start=20.000,asetpts=PTS-STARTPTS[a1];"));
        assert!(graph.ends_with("[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"));
        let silent = filter_complex(Some("subtitles=s.srt"), &[(5.0, None)], false);
        assert!(!silent.contains("[0:a:0]"));
        assert!(silent.ends_with("[v0]concat=n=1:v=1:a=0[v]"));
    }

    #[test]
    fn sound_alone_is_trimmed_and_joined_without_a_picture() {
        let graph = filter_complex(None, &[(0.0, Some(10.0)), (20.0, None)], true);
        assert!(!graph.contains("[0:v:0]"));
        assert!(graph.ends_with("[a0][a1]concat=n=2:v=0:a=1[a]"));
    }

    #[test]
    fn times_after_a_cut_move_back_by_its_length_and_inside_one_land_on_its_place() {
        let cuts = normalize(&[cut(10.0, 20.0), cut(40.0, 50.0)], 100.0);
        assert_eq!(shift(&cuts, 5.0), 5.0);
        assert_eq!(shift(&cuts, 15.0), 10.0, "inside the first cut");
        assert_eq!(shift(&cuts, 25.0), 15.0);
        assert_eq!(shift(&cuts, 60.0), 40.0);
        assert_eq!(shift(&[], 60.0), 60.0);
    }

    #[test]
    fn an_old_or_empty_file_reads_as_no_cuts() {
        let file: CutsFile = serde_json::from_str("{}").unwrap();
        assert!(file.cuts.is_empty());
    }
}
