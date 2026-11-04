//! Speaker diarization pieces (pure logic; the ONNX Runtime calls live in `main.rs`).
//!
//! Pipeline (a simplified pyannote 3.x):
//! 1. `segmentation-3.0` runs on 10 s windows and gives, per frame (270 samples = 16.9 ms), one of
//!    7 "powerset" classes: silence, speaker A/B/C (local to the window), or a pair of them.
//! 2. For every local speaker of every window (use only frames where it speaks alone) one
//!    WeSpeaker embedding is computed.
//! 3. All embeddings are clustered into the global speakers (cosine k-means, weighted by speech time).
//! 4. Every frame gets the global speaker of its local speaker; frames become non-overlapping turns.

use engine::diar::DiarTurn;

/// Samples per segmentation window (10 s at 16 kHz).
pub const WINDOW: usize = 160_000;
/// Samples between two segmentation frames.
pub const FRAME_STEP: usize = 270;
/// The model's receptive field is 721 samples; a frame is centred half of that after its start.
pub const FRAME_CENTER: usize = 360;
pub const SAMPLE_RATE: f64 = 16_000.0;

/// Local speakers (0..3) active in a powerset class: 0 = none, 1..=3 = one speaker, 4..=6 = a pair.
pub fn class_members(class: usize) -> &'static [usize] {
    match class {
        1 => &[0],
        2 => &[1],
        3 => &[2],
        4 => &[0, 1],
        5 => &[0, 2],
        6 => &[1, 2],
        _ => &[],
    }
}

/// Sample range `[from, to)` covered by frame `i` of the window starting at `window_start`.
pub fn frame_range(window_start: usize, i: usize) -> (usize, usize) {
    let center = window_start + FRAME_CENTER + i * FRAME_STEP;
    (center - FRAME_STEP / 2, center + FRAME_STEP / 2)
}

/// Segmentation result for one window.
#[derive(Debug, Clone)]
pub struct WindowSeg {
    /// First sample of the window.
    pub start: usize,
    /// Argmax powerset class per frame.
    pub classes: Vec<u8>,
    /// Log-probability of the three single-speaker classes per frame (used to split overlaps).
    pub single_logp: Vec<[f32; 3]>,
}

impl WindowSeg {
    /// Sample ranges where local speaker `local` speaks alone (clean audio for its embedding).
    pub fn clean_ranges(&self, local: usize) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = Vec::new();
        for (i, &c) in self.classes.iter().enumerate() {
            let m = class_members(c as usize);
            if m.len() == 1 && m[0] == local {
                let (a, b) = frame_range(self.start, i);
                match out.last_mut() {
                    Some(last) if last.1 == a => last.1 = b,
                    _ => out.push((a, b)),
                }
            }
        }
        out
    }

    /// Seconds in which local speaker `local` speaks alone.
    pub fn clean_seconds(&self, local: usize) -> f64 {
        self.classes
            .iter()
            .filter(|&&c| class_members(c as usize) == [local])
            .count() as f64
            * FRAME_STEP as f64
            / SAMPLE_RATE
    }

    /// Seconds in which local speaker `local` speaks at all (alone or in an overlap).
    pub fn active_seconds(&self, local: usize) -> f64 {
        self.classes
            .iter()
            .filter(|&&c| class_members(c as usize).contains(&local))
            .count() as f64
            * FRAME_STEP as f64
            / SAMPLE_RATE
    }
}

// ---------------------------------------------------------------------------------- clustering

pub fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[derive(Debug, Clone)]
pub struct Clustering {
    pub labels: Vec<usize>,
    pub centroids: Vec<Vec<f32>>,
    /// Weighted mean cosine similarity to the own centroid (higher = tighter).
    pub score: f32,
}

/// Cosine k-means on unit vectors with per-point weights and farthest-point initialisation.
/// `embs` must be normalised. `k` is capped at the number of points.
pub fn kmeans_cosine(
    embs: &[Vec<f32>],
    weights: &[f32],
    k: usize,
    first_seed: usize,
) -> Clustering {
    let n = embs.len();
    let k = k.min(n).max(1);
    let dim = embs[0].len();

    // farthest-point init starting from `first_seed`
    let mut centroids: Vec<Vec<f32>> = vec![embs[first_seed].clone()];
    while centroids.len() < k {
        let far = (0..n)
            .map(|i| {
                let best = centroids
                    .iter()
                    .map(|c| dot(&embs[i], c))
                    .fold(f32::MIN, f32::max);
                (i, 1.0 - best)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        centroids.push(embs[far].clone());
    }

    let mut labels = vec![0usize; n];
    for iter in 0..100 {
        let mut changed = false;
        for i in 0..n {
            let best = (0..k)
                .max_by(|&a, &b| {
                    dot(&embs[i], &centroids[a]).total_cmp(&dot(&embs[i], &centroids[b]))
                })
                .unwrap_or(0);
            if labels[i] != best || iter == 0 {
                changed |= labels[i] != best;
                labels[i] = best;
            }
        }
        let mut sums = vec![vec![0f32; dim]; k];
        for i in 0..n {
            for (s, x) in sums[labels[i]].iter_mut().zip(&embs[i]) {
                *s += weights[i] * x;
            }
        }
        for (c, s) in centroids.iter_mut().zip(sums) {
            if s.iter().any(|x| *x != 0.0) {
                *c = s;
                normalize(c);
            }
        }
        if !changed && iter > 0 {
            break;
        }
    }
    let total_w: f32 = weights.iter().sum::<f32>().max(1e-9);
    let score = (0..n)
        .map(|i| weights[i] * dot(&embs[i], &centroids[labels[i]]))
        .sum::<f32>()
        / total_w;
    Clustering {
        labels,
        centroids,
        score,
    }
}

/// Best of a few initialisations (heaviest point, point farthest from the mean, a middle point).
pub fn cluster_k(embs: &[Vec<f32>], weights: &[f32], k: usize) -> Clustering {
    let n = embs.len();
    let heaviest = (0..n)
        .max_by(|&a, &b| weights[a].total_cmp(&weights[b]))
        .unwrap_or(0);
    let mut mean = vec![0f32; embs[0].len()];
    for e in embs {
        mean.iter_mut().zip(e).for_each(|(m, x)| *m += x);
    }
    normalize(&mut mean);
    let farthest_from_mean = (0..n)
        .min_by(|&a, &b| dot(&embs[a], &mean).total_cmp(&dot(&embs[b], &mean)))
        .unwrap_or(0);
    [heaviest, farthest_from_mean, n / 2]
        .into_iter()
        .map(|seed| kmeans_cosine(embs, weights, k, seed))
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .expect("at least one initialisation")
}

/// Result of average-linkage agglomerative clustering.
#[derive(Debug, Clone)]
pub struct Agglo {
    /// Cluster id (0..n_clusters) per input point.
    pub labels: Vec<usize>,
    pub n_clusters: usize,
    /// Cosine similarity at which each merge happened, in merge order (non-increasing for average linkage,
    /// apart from small float noise). The last entries show how separable the final speakers are.
    pub merge_sims: Vec<f32>,
}

/// Weighted average-linkage agglomerative clustering on cosine similarity.
///
/// Clusters are merged while the most similar pair has similarity >= `stop_sim`, and also while there are
/// more than `max_clusters` clusters (a hard cap on the number of speakers). `embs` must be normalised.
pub fn agglomerative(
    embs: &[Vec<f32>],
    weights: &[f32],
    stop_sim: f32,
    max_clusters: usize,
) -> Agglo {
    let n = embs.len();
    let mut sim = vec![0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dot(&embs[i], &embs[j]);
            sim[i * n + j] = d;
            sim[j * n + i] = d;
        }
    }
    let mut w: Vec<f32> = weights.iter().map(|x| x.max(1e-6)).collect();
    let mut active = vec![true; n];
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    // nearest active neighbour of a cluster
    let nearest = |i: usize, sim: &[f32], active: &[bool]| -> (usize, f32) {
        let mut best = (usize::MAX, f32::MIN);
        for j in 0..n {
            if j != i && active[j] && sim[i * n + j] > best.1 {
                best = (j, sim[i * n + j]);
            }
        }
        best
    };
    let mut nn: Vec<(usize, f32)> = (0..n).map(|i| nearest(i, &sim, &active)).collect();

    let mut count = n;
    let mut merge_sims = Vec::new();
    while count > 1 {
        let (i, (j, s)) = (0..n)
            .filter(|&i| active[i] && nn[i].0 != usize::MAX)
            .map(|i| (i, nn[i]))
            .max_by(|a, b| a.1.1.total_cmp(&b.1.1))
            .expect("two active clusters have a nearest neighbour");
        if s < stop_sim && count <= max_clusters {
            break;
        }
        // merge j into i (Lance-Williams update for weighted average linkage)
        for k in 0..n {
            if active[k] && k != i && k != j {
                let v = (w[i] * sim[i * n + k] + w[j] * sim[j * n + k]) / (w[i] + w[j]);
                sim[i * n + k] = v;
                sim[k * n + i] = v;
            }
        }
        w[i] += w[j];
        let moved = std::mem::take(&mut members[j]);
        members[i].extend(moved);
        active[j] = false;
        count -= 1;
        merge_sims.push(s);
        nn[i] = nearest(i, &sim, &active);
        for k in 0..n {
            if active[k] && k != i {
                if nn[k].0 == i || nn[k].0 == j {
                    nn[k] = nearest(k, &sim, &active);
                } else if sim[k * n + i] > nn[k].1 {
                    nn[k] = (i, sim[k * n + i]);
                }
            }
        }
    }

    let mut labels = vec![0usize; n];
    let mut next = 0;
    for i in 0..n {
        if active[i] {
            for &m in &members[i] {
                labels[m] = next;
            }
            next += 1;
        }
    }
    Agglo {
        labels,
        n_clusters: next,
        merge_sims,
    }
}

/// Weighted, normalised centroid of every cluster.
pub fn centroids_of(
    embs: &[Vec<f32>],
    weights: &[f32],
    labels: &[usize],
    k: usize,
) -> Vec<Vec<f32>> {
    let dim = embs.first().map_or(0, |e| e.len());
    let mut sums = vec![vec![0f32; dim]; k];
    for (i, e) in embs.iter().enumerate() {
        for (s, x) in sums[labels[i]].iter_mut().zip(e) {
            *s += weights[i] * x;
        }
    }
    for c in sums.iter_mut() {
        normalize(c);
    }
    sums
}

fn nearest_centroid(e: &[f32], centroids: &[Vec<f32>]) -> usize {
    (0..centroids.len())
        .max_by(|&a, &b| dot(e, &centroids[a]).total_cmp(&dot(e, &centroids[b])))
        .unwrap_or(0)
}

/// A small cluster (under this share of all speech) counts as a variant of another speaker when ...
pub const VARIANT_SHARE: f32 = 0.10;
/// ... its centroid is at least this similar (cosine) to that speaker's centroid.
pub const VARIANT_SIM: f32 = 0.4;

/// Automatic speaker clustering: agglomerative clustering with a similarity threshold, then a few
/// rounds of "move every point to its nearest centroid", then tiny clusters (less than `min_speech`
/// seconds of speech in total - noise, laughter, a cough) are merged into the closest real speaker so they
/// cannot become phantom speakers. Never returns more than `max_speakers` clusters. Labels are 0..k.
pub fn cluster_auto(
    embs: &[Vec<f32>],
    weights: &[f32],
    stop_sim: f32,
    max_speakers: usize,
    min_speech: f32,
) -> Clustering {
    let agg = agglomerative(embs, weights, stop_sim, max_speakers.max(1));
    let mut labels = agg.labels;
    let mut k = agg.n_clusters;

    let mut centroids = centroids_of(embs, weights, &labels, k);
    for _ in 0..5 {
        let new: Vec<usize> = embs
            .iter()
            .map(|e| nearest_centroid(e, &centroids))
            .collect();
        if new == labels {
            break;
        }
        labels = new;
        centroids = centroids_of(embs, weights, &labels, k);
    }

    // Clean-up of clusters that are not real speakers. Weights are seconds of clean speech.
    let mut total = vec![0f32; k];
    for (i, &l) in labels.iter().enumerate() {
        total[l] += weights[i];
    }
    let all: f32 = total.iter().sum();
    // (1) "established" speakers have at least `min_speech` seconds; everything else (noise, laughter, a
    //     cough, a stray word) is merged into the most similar established speaker. Merging into established
    //     clusters only - not into each other - stops many tiny clusters adding up to a phantom speaker.
    let mut keep: Vec<bool> = total.iter().map(|&t| t >= min_speech).collect();
    if !keep.iter().any(|&b| b) {
        let heaviest = (0..k)
            .max_by(|&a, &b| total[a].total_cmp(&total[b]))
            .unwrap_or(0);
        keep[heaviest] = true;
    }
    // (2) a small established cluster (under VARIANT_SHARE of all speech) that sounds like another established
    //     speaker (similarity >= VARIANT_SIM) is that speaker in a different mood / on a bad line: merge it.
    loop {
        let mut weakest: Option<(usize, usize)> = None; // (cluster, the speaker it is a variant of)
        let mut weakest_total = f32::MAX;
        for c in 0..k {
            if !(keep[c] && total[c] < VARIANT_SHARE * all && total[c] < weakest_total) {
                continue;
            }
            let host = (0..k)
                .filter(|&o| o != c && keep[o] && dot(&centroids[c], &centroids[o]) >= VARIANT_SIM)
                .max_by(|&a, &b| {
                    dot(&centroids[c], &centroids[a]).total_cmp(&dot(&centroids[c], &centroids[b]))
                });
            if let Some(h) = host {
                weakest = Some((c, h));
                weakest_total = total[c];
            }
        }
        let Some((c, h)) = weakest else { break };
        keep[c] = false;
        total[h] += total[c];
        total[c] = 0.0;
        for l in labels.iter_mut() {
            if *l == c {
                *l = h;
            }
        }
    }
    // route what is left over, then number the clusters 0..k again
    let target: Vec<usize> = (0..k)
        .map(|c| {
            if keep[c] {
                c
            } else {
                (0..k)
                    .filter(|&o| keep[o])
                    .max_by(|&a, &b| {
                        dot(&centroids[c], &centroids[a])
                            .total_cmp(&dot(&centroids[c], &centroids[b]))
                    })
                    .expect("at least one established speaker")
            }
        })
        .collect();
    let mut new_id = vec![usize::MAX; k];
    let mut next = 0;
    for c in 0..k {
        if keep[c] {
            new_id[c] = next;
            next += 1;
        }
    }
    for l in labels.iter_mut() {
        *l = new_id[target[*l]];
    }
    k = next;
    centroids = centroids_of(embs, weights, &labels, k);

    let total_w: f32 = weights.iter().sum::<f32>().max(1e-9);
    let score = (0..embs.len())
        .map(|i| weights[i] * dot(&embs[i], &centroids[labels[i]]))
        .sum::<f32>()
        / total_w;
    Clustering {
        labels,
        centroids,
        score,
    }
}

// ---------------------------------------------------------------------------------- turns

/// Merge consecutive frames of the same speaker into turns. Gaps up to `max_gap` seconds inside one
/// speaker's speech are bridged; turns shorter than `min_len` seconds are dropped.
/// `frames` = `(start_s, end_s, speaker)` sorted by start.
pub fn merge_frames(frames: &[(f64, f64, usize)], max_gap: f64, min_len: f64) -> Vec<DiarTurn> {
    let mut turns: Vec<DiarTurn> = Vec::new();
    for &(s, e, spk) in frames {
        match turns.last_mut() {
            Some(t) if t.speaker == spk && s - t.end <= max_gap => t.end = t.end.max(e),
            _ => turns.push(DiarTurn {
                start: s,
                end: e,
                speaker: spk,
            }),
        }
    }
    turns.retain(|t| t.end - t.start >= min_len);
    // dropping short turns can make neighbours of the same speaker adjacent again
    let mut merged: Vec<DiarTurn> = Vec::new();
    for t in turns {
        match merged.last_mut() {
            Some(m) if m.speaker == t.speaker && t.start - m.end <= max_gap => {
                m.end = m.end.max(t.end)
            }
            _ => merged.push(t),
        }
    }
    // frames never overlap, but gap bridging must not push a turn into the next one
    for i in 1..merged.len() {
        if merged[i - 1].end > merged[i].start {
            merged[i - 1].end = merged[i].start;
        }
    }
    merged
}

/// Renumber speakers by first appearance so "Speaker 1" is whoever talks first.
pub fn renumber_by_first_appearance(turns: &mut [DiarTurn]) {
    let mut order: Vec<usize> = Vec::new();
    for t in turns.iter() {
        if !order.contains(&t.speaker) {
            order.push(t.speaker);
        }
    }
    for t in turns.iter_mut() {
        t.speaker = order.iter().position(|&s| s == t.speaker).unwrap_or(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(x: f32, y: f32) -> Vec<f32> {
        let mut v = vec![x, y, 0.0];
        normalize(&mut v);
        v
    }

    #[test]
    fn powerset_members() {
        assert_eq!(class_members(0), &[] as &[usize]);
        assert_eq!(class_members(2), &[1]);
        assert_eq!(class_members(5), &[0, 2]);
    }

    #[test]
    fn clean_ranges_join_adjacent_frames_and_skip_overlap() {
        let w = WindowSeg {
            start: 0,
            classes: vec![1, 1, 4, 1, 0, 2],
            single_logp: vec![[0.0; 3]; 6],
        };
        let r = w.clean_ranges(0);
        assert_eq!(r.len(), 2); // frames 0-1, then frame 3 (frame 2 is an overlap)
        assert_eq!(r[0].1 - r[0].0, 2 * FRAME_STEP);
        assert!((w.clean_seconds(0) - 3.0 * 270.0 / 16000.0).abs() < 1e-9);
        assert!((w.active_seconds(0) - 4.0 * 270.0 / 16000.0).abs() < 1e-9);
    }

    #[test]
    fn kmeans_separates_two_directions() {
        let embs: Vec<Vec<f32>> = vec![
            unit(1.0, 0.05),
            unit(1.0, -0.1),
            unit(0.9, 0.0),
            unit(0.05, 1.0),
            unit(-0.1, 1.0),
            unit(0.0, 0.9),
        ];
        let w = vec![1.0; 6];
        let c = cluster_k(&embs, &w, 2);
        assert_eq!(c.labels[0], c.labels[1]);
        assert_eq!(c.labels[0], c.labels[2]);
        assert_eq!(c.labels[3], c.labels[4]);
        assert_ne!(c.labels[0], c.labels[3]);
    }

    fn blob(centre: (f32, f32), count: usize) -> Vec<Vec<f32>> {
        (0..count)
            .map(|i| {
                unit(
                    centre.0 + (i as f32 % 5.0) * 0.01,
                    centre.1 + (i as f32 % 3.0) * 0.01,
                )
            })
            .collect()
    }

    fn third_axis(count: usize) -> Vec<Vec<f32>> {
        (0..count)
            .map(|i| {
                let mut v = vec![0.02 * i as f32, 0.0, 1.0];
                normalize(&mut v);
                v
            })
            .collect()
    }

    #[test]
    fn auto_finds_one_two_three_speakers() {
        let one = blob((1.0, 0.0), 12);
        assert_eq!(
            cluster_auto(&one, &[3.0; 12], 0.5, 8, 10.0).centroids.len(),
            1
        );

        let mut two = blob((1.0, 0.0), 8);
        two.extend(blob((0.0, 1.0), 8));
        assert_eq!(
            cluster_auto(&two, &[3.0; 16], 0.5, 8, 10.0).centroids.len(),
            2
        );

        let mut three = two.clone();
        three.extend(third_axis(8));
        assert_eq!(
            cluster_auto(&three, &[3.0; 24], 0.5, 8, 10.0)
                .centroids
                .len(),
            3
        );
    }

    #[test]
    fn auto_respects_the_cap_and_ignores_tiny_clusters() {
        let mut e = blob((1.0, 0.0), 8);
        e.extend(blob((0.0, 1.0), 8));
        let mut w = vec![3.0; 16];
        // a stray 1.5 s "speaker" in a third direction must not become a speaker
        e.extend(third_axis(1));
        w.push(1.5);
        let c = cluster_auto(&e, &w, 0.5, 8, 10.0);
        assert_eq!(c.centroids.len(), 2);
        assert_eq!(c.labels.len(), 17);

        // three real speakers but at most two allowed
        let mut three = blob((1.0, 0.0), 8);
        three.extend(blob((0.0, 1.0), 8));
        three.extend(third_axis(8));
        assert_eq!(
            cluster_auto(&three, &[3.0; 24], 0.5, 2, 10.0)
                .centroids
                .len(),
            2
        );
    }

    #[test]
    fn many_tiny_clusters_do_not_add_up_to_a_phantom_speaker() {
        // two real speakers (8 x 3 s each) plus five scattered 1.5 s noise points in different directions
        let mut e = blob((1.0, 0.0), 8);
        e.extend(blob((0.0, 1.0), 8));
        let mut w = vec![3.0; 16];
        for i in 0..5 {
            let mut v = vec![0.0, 0.3 + i as f32 * 0.05, 1.0];
            v[0] = -0.2 * i as f32;
            normalize(&mut v);
            e.push(v);
            w.push(1.5);
        }
        // 7.5 s of noise in total: below 10 s even if the noise points cluster together
        assert_eq!(cluster_auto(&e, &w, 0.9, 8, 10.0).centroids.len(), 2);
    }

    #[test]
    fn small_cluster_that_sounds_like_a_big_one_is_merged() {
        // speaker A (60 s), speaker B (60 s) and a 12 s cluster that sits between A and B's directions
        // (similar to both, > 0.4) but with enough speech to pass the 10 s rule: merged as a variant
        let mut e = blob((1.0, 0.0), 10);
        e.extend(blob((0.0, 1.0), 10));
        let mut w = vec![6.0; 20]; // 60 s each, so 12 s is 9% of all speech
        for i in 0..4 {
            e.push(unit(0.8 + 0.01 * i as f32, 0.6));
            w.push(3.0);
        }
        let c = cluster_auto(&e, &w, 0.9, 8, 10.0);
        assert_eq!(c.centroids.len(), 2, "variant cluster should disappear");
        // a genuinely different 12 s speaker (orthogonal) is kept
        let mut e2 = blob((1.0, 0.0), 10);
        e2.extend(blob((0.0, 1.0), 10));
        e2.extend((0..4).map(|i| {
            let mut v = vec![0.0, 0.0, 1.0];
            v[0] = 0.01 * i as f32;
            normalize(&mut v);
            v
        }));
        let c2 = cluster_auto(&e2, &w, 0.9, 8, 10.0);
        assert_eq!(c2.centroids.len(), 3);
    }

    #[test]
    fn agglomerative_reports_merge_similarities() {
        let mut e = blob((1.0, 0.0), 4);
        e.extend(blob((0.0, 1.0), 4));
        let a = agglomerative(&e, &[1.0; 8], 0.5, 8);
        assert_eq!(a.n_clusters, 2);
        assert_eq!(a.merge_sims.len(), 6);
        assert!(a.merge_sims.iter().all(|&s| s > 0.9));
    }

    #[test]
    fn merge_bridges_small_gaps_and_drops_blips() {
        let f = [
            (0.0, 1.0, 0),
            (1.2, 2.0, 0),
            (2.0, 2.05, 1),
            (2.05, 4.0, 0),
            (5.0, 6.0, 1),
        ];
        let t = merge_frames(&f, 0.4, 0.1);
        assert_eq!(t.len(), 2);
        assert_eq!((t[0].start, t[0].end, t[0].speaker), (0.0, 4.0, 0));
        assert_eq!(t[1].speaker, 1);
    }

    #[test]
    fn turns_never_overlap() {
        let f = [(0.0, 1.0, 0), (1.3, 2.0, 1), (2.1, 3.0, 0)];
        let t = merge_frames(&f, 0.5, 0.0);
        assert!(t.windows(2).all(|w| w[0].end <= w[1].start));
    }

    #[test]
    fn renumbering_follows_first_speaker() {
        let mut t = vec![
            DiarTurn {
                start: 0.0,
                end: 1.0,
                speaker: 5,
            },
            DiarTurn {
                start: 1.0,
                end: 2.0,
                speaker: 2,
            },
            DiarTurn {
                start: 2.0,
                end: 3.0,
                speaker: 5,
            },
        ];
        renumber_by_first_appearance(&mut t);
        assert_eq!(
            t.iter().map(|x| x.speaker).collect::<Vec<_>>(),
            vec![0, 1, 0]
        );
    }
}
