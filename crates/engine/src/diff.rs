//! Comparing two lists of words: the matching itself (a port of Python's `difflib.SequenceMatcher`, including its
//! `autojunk` rule for `len(b) >= 200`), and a word-level difference to show what an edit changed.

use std::collections::HashMap;

/// Runs of equal elements, in order, as `(position in a, position in b, length)`.
pub fn matching_blocks<S: AsRef<str>>(a: &[S], b: &[S]) -> Vec<(usize, usize, usize)> {
    // b2j: element -> ascending positions in b
    let mut b2j: HashMap<&str, Vec<usize>> = HashMap::new();
    for (j, w) in b.iter().enumerate() {
        b2j.entry(w.as_ref()).or_default().push(j);
    }
    if b.len() >= 200 {
        let ntest = b.len() / 100 + 1;
        b2j.retain(|_, v| v.len() <= ntest);
    }

    let mut queue = vec![(0usize, a.len(), 0usize, b.len())];
    let mut blocks = Vec::new();
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = longest_match(a, b, &b2j, alo, ahi, blo, bhi);
        if k > 0 {
            blocks.push((i, j, k));
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    blocks.sort_unstable();
    blocks
}

fn longest_match<S: AsRef<str>>(
    a: &[S],
    b: &[S],
    b2j: &HashMap<&str, Vec<usize>>,
    alo: usize,
    ahi: usize,
    blo: usize,
    bhi: usize,
) -> (usize, usize, usize) {
    let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0usize);
    let mut j2len: HashMap<usize, usize> = HashMap::new();
    for (i, word) in a.iter().enumerate().take(ahi).skip(alo) {
        let mut newj2len: HashMap<usize, usize> = HashMap::new();
        if let Some(js) = b2j.get(word.as_ref()) {
            for &j in js {
                if j < blo {
                    continue;
                }
                if j >= bhi {
                    break;
                }
                let k = if j > 0 {
                    j2len.get(&(j - 1)).copied().unwrap_or(0)
                } else {
                    0
                } + 1;
                newj2len.insert(j, k);
                if k > bestsize {
                    besti = i + 1 - k;
                    bestj = j + 1 - k;
                    bestsize = k;
                }
            }
        }
        j2len = newj2len;
    }
    // extend the best match over equal neighbours (only matters when popular elements were dropped)
    while besti > alo && bestj > blo && a[besti - 1].as_ref() == b[bestj - 1].as_ref() {
        besti -= 1;
        bestj -= 1;
        bestsize += 1;
    }
    while besti + bestsize < ahi
        && bestj + bestsize < bhi
        && a[besti + bestsize].as_ref() == b[bestj + bestsize].as_ref()
    {
        bestsize += 1;
    }
    (besti, bestj, bestsize)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffKind {
    Same,
    Removed,
    Added,
}

/// Words next to each other that share a [`DiffKind`], joined with single spaces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffSpan {
    pub kind: DiffKind,
    pub text: String,
}

/// What changed from `old` to `new`, word by word. Words split on whitespace and punctuation stays on its word, so a
/// new comma shows as a changed word. Where words were replaced, the removed ones come first.
pub fn diff_words(old: &str, new: &str) -> Vec<DiffSpan> {
    let a: Vec<&str> = old.split_whitespace().collect();
    let b: Vec<&str> = new.split_whitespace().collect();
    let mut spans = Vec::new();
    let mut push = |kind, words: &[&str]| {
        if !words.is_empty() {
            spans.push(DiffSpan {
                kind,
                text: words.join(" "),
            });
        }
    };
    let (mut i, mut j) = (0, 0);
    for (bi, bj, k) in matching_blocks(&a, &b) {
        push(DiffKind::Removed, &a[i..bi]);
        push(DiffKind::Added, &b[j..bj]);
        push(DiffKind::Same, &a[bi..bi + k]);
        (i, j) = (bi + k, bj + k);
    }
    push(DiffKind::Removed, &a[i..]);
    push(DiffKind::Added, &b[j..]);
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(spans: &[DiffSpan]) -> Vec<(DiffKind, &str)> {
        spans.iter().map(|s| (s.kind, s.text.as_str())).collect()
    }

    #[test]
    fn a_replaced_word_shows_removed_then_added() {
        let d = diff_words("so the web hook failed", "so the webhook failed");
        assert_eq!(
            kinds(&d),
            [
                (DiffKind::Same, "so the"),
                (DiffKind::Removed, "web hook"),
                (DiffKind::Added, "webhook"),
                (DiffKind::Same, "failed"),
            ]
        );
    }

    #[test]
    fn additions_and_removals_at_the_edges_are_kept() {
        let d = diff_words("um hello there", "hello there friend");
        assert_eq!(
            kinds(&d),
            [
                (DiffKind::Removed, "um"),
                (DiffKind::Same, "hello there"),
                (DiffKind::Added, "friend"),
            ]
        );
    }

    #[test]
    fn punctuation_counts_as_a_change() {
        let d = diff_words("hello there", "hello, there");
        assert_eq!(
            kinds(&d),
            [
                (DiffKind::Removed, "hello"),
                (DiffKind::Added, "hello,"),
                (DiffKind::Same, "there"),
            ]
        );
    }

    #[test]
    fn only_spacing_differs_means_no_change() {
        let d = diff_words("hello  there\n", " hello there");
        assert_eq!(kinds(&d), [(DiffKind::Same, "hello there")]);
    }

    #[test]
    fn empty_sides_are_fine() {
        assert_eq!(diff_words("", ""), []);
        assert_eq!(kinds(&diff_words("a b", "")), [(DiffKind::Removed, "a b")]);
        assert_eq!(kinds(&diff_words("", "a b")), [(DiffKind::Added, "a b")]);
    }

    #[test]
    fn right_to_left_words_compare_like_any_other() {
        let d = diff_words("سلام دنیا", "سلام جهان");
        assert_eq!(
            kinds(&d),
            [
                (DiffKind::Same, "سلام"),
                (DiffKind::Removed, "دنیا"),
                (DiffKind::Added, "جهان"),
            ]
        );
    }

    #[test]
    fn rebuilding_either_side_from_the_spans_gives_it_back() {
        let (old, new) = ("a b c d e f g", "a x c d y f z g h");
        let side = |skip| {
            diff_words(old, new)
                .iter()
                .filter(|s| s.kind != skip)
                .map(|s| s.text.clone())
                .collect::<Vec<_>>()
                .join(" ")
        };
        assert_eq!(side(DiffKind::Added), old);
        assert_eq!(side(DiffKind::Removed), new);
    }
}
