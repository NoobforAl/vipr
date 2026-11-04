//! Safety nets for the LLM text cleanup: ported one-to-one from `scripts/refine_transcripts.py`.
//! Small models love to over-edit, so every proposed line is checked against the original here, in code.
//! Nothing in a prompt (category, video description) can change these rules.

use std::collections::{HashMap, HashSet};

/// Filler words ignored when comparing a line with its edit.
pub const FILLERS: [&str; 6] = ["uh", "um", "uhm", "erm", "umm", "uhh"];
/// Only short lines may change speaker.
pub const MAX_FLIP_WORDS: usize = 12;
/// If more than this share of a chunk flips speaker, the model is guessing.
pub const MAX_FLIP_FRACTION: f64 = 0.3;

/// `re.findall(r"[a-z0-9']+", s.lower())` without the fillers.
pub fn norm_words(s: &str) -> Vec<String> {
    let lower = s.to_lowercase();
    let mut words = Vec::new();
    let mut cur = String::new();
    for c in lower.chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '\'' {
            cur.push(c);
        } else if !cur.is_empty() {
            words.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    words.retain(|w| !FILLERS.contains(&w.as_str()));
    words
}

/// Number of `,.?!;:` characters.
pub fn punct_count(s: &str) -> usize {
    s.chars().filter(|c| ",.?!;:".contains(*c)).count()
}

/// Port of `difflib.SequenceMatcher(None, a, b).ratio()` (including its `autojunk` rule for `len(b) >= 200`).
pub fn sequence_ratio(a: &[String], b: &[String]) -> f64 {
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    2.0 * matched(a, b) as f64 / total as f64
}

fn matched(a: &[String], b: &[String]) -> usize {
    // b2j: element -> ascending positions in b
    let mut b2j: HashMap<&str, Vec<usize>> = HashMap::new();
    for (j, w) in b.iter().enumerate() {
        b2j.entry(w.as_str()).or_default().push(j);
    }
    if b.len() >= 200 {
        let ntest = b.len() / 100 + 1;
        b2j.retain(|_, v| v.len() <= ntest);
    }

    let mut queue = vec![(0usize, a.len(), 0usize, b.len())];
    let mut sum = 0;
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = longest_match(a, b, &b2j, alo, ahi, blo, bhi);
        if k > 0 {
            sum += k;
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    sum
}

fn longest_match(
    a: &[String],
    b: &[String],
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
        if let Some(js) = b2j.get(word.as_str()) {
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
    while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
        besti -= 1;
        bestj -= 1;
        bestsize += 1;
    }
    while besti + bestsize < ahi
        && bestj + bestsize < bhi
        && a[besti + bestsize] == b[bestj + bestsize]
    {
        bestsize += 1;
    }
    (besti, bestj, bestsize)
}

/// Reject rewrites: the corrected line must stay close to what was said.
pub fn acceptable_edit(old: &str, new: &str) -> bool {
    let wo = norm_words(old);
    let wn = norm_words(new);
    if wo.is_empty() {
        return false; // nothing but fillers: leave as is
    }
    if wn.is_empty() {
        return false; // model deleted the content
    }
    let ratio = sequence_ratio(&wo, &wn);
    let len_ratio = wn.len() as f64 / wo.len() as f64;
    if !(ratio >= 0.6 && (0.6..=1.5).contains(&len_ratio)) {
        return false;
    }
    // the model must not strip punctuation that was already there
    let (po, pn) = (punct_count(old), punct_count(new));
    !(po >= 3 && (pn as f64) < 0.6 * po as f64)
}

/// If this line continues an unfinished sentence, undo a capital the model added.
pub fn fix_continuation_case(
    prev: &str,
    old: &str,
    new: &str,
    keep_caps: &HashSet<String>,
) -> String {
    if prev.is_empty() || old.is_empty() || new.is_empty() {
        return new.to_string();
    }
    // Python: `prev.rstrip()[-1:] in ".?!"` is also true for an empty string
    let prev_ends_sentence = prev
        .trim_end()
        .chars()
        .last()
        .is_none_or(|c| ".?!".contains(c));
    let old_first = old.chars().next().unwrap();
    let new_first = new.chars().next().unwrap();
    if prev_ends_sentence || !old_first.is_lowercase() || !new_first.is_uppercase() {
        return new.to_string();
    }
    let Some(token) = new.split_whitespace().next() else {
        return new.to_string();
    };
    let first: String = token
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '\'')
        .collect();
    if first == "I" || first.starts_with("I'") || keep_caps.contains(&first.to_lowercase()) {
        return new.to_string();
    }
    let mut chars = new.chars();
    let head: String = chars.next().unwrap().to_lowercase().collect();
    format!("{head}{}", chars.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn norm_words_drops_fillers_and_punctuation() {
        assert_eq!(
            norm_words("Uh, I'm gonna - um - merge it."),
            vec!["i'm", "gonna", "merge", "it"]
        );
        assert!(norm_words("uh um").is_empty());
    }

    #[test]
    fn sequence_ratio_matches_python_difflib() {
        // values computed with difflib.SequenceMatcher(None, a, b).ratio()
        assert!((sequence_ratio(&w("a b c d"), &w("a b c d")) - 1.0).abs() < 1e-12);
        assert!((sequence_ratio(&w("a b c d"), &w("a x c d")) - 0.75).abs() < 1e-12);
        assert!((sequence_ratio(&w("a b c d"), &w("d c b a")) - 0.25).abs() < 1e-12);
        assert!(
            (sequence_ratio(&w("the cat sat on the mat"), &w("the cat sat on a mat"))
                - 10.0 / 12.0)
                .abs()
                < 1e-12
        );
        assert!((sequence_ratio(&w("a b c"), &w("x y z")) - 0.0).abs() < 1e-12);
        // classic difflib example: "abcd" vs "bcde" -> 0.75
        assert!((sequence_ratio(&w("a b c d"), &w("b c d e")) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn accepts_small_fixes_rejects_rewrites() {
        assert!(acceptable_edit(
            "so we merge the pr uh today",
            "So we merge the PR today."
        ));
        assert!(acceptable_edit(
            "i think its a web hook issue",
            "I think it's a webhook issue."
        ));
        // rewrite / summary
        assert!(!acceptable_edit(
            "yeah i think we should probably deploy that on friday",
            "Deploy on Friday."
        ));
        // deleted everything / nothing but fillers
        assert!(!acceptable_edit("okay", ""));
        assert!(!acceptable_edit("uh um", "Um."));
        // too many words invented
        assert!(!acceptable_edit(
            "okay",
            "okay that sounds good to me let us do that"
        ));
    }

    #[test]
    fn rejects_stripped_punctuation() {
        let old = "Well, I think, honestly, that it works.";
        assert!(!acceptable_edit(old, "Well I think honestly that it works"));
        assert!(acceptable_edit(
            old,
            "Well, I think, honestly, that it works!"
        ));
    }

    #[test]
    fn continuation_case_is_undone_only_mid_sentence() {
        let keep: HashSet<String> = ["cloudflare".to_string()].into();
        // previous line unfinished, original lowercase, model capitalised -> lowercase it again
        assert_eq!(
            fix_continuation_case("we should go to", "the store", "The store.", &keep),
            "the store."
        );
        // previous line finished a sentence -> keep
        assert_eq!(
            fix_continuation_case("that is all.", "the store", "The store.", &keep),
            "The store."
        );
        // "I", glossary terms and original capitals are kept
        assert_eq!(
            fix_continuation_case("and then", "i left", "I left.", &keep),
            "I left."
        );
        assert_eq!(
            fix_continuation_case("we use", "cloudflare for it", "Cloudflare for it.", &keep),
            "Cloudflare for it."
        );
        assert_eq!(
            fix_continuation_case("we use", "Docker", "Docker", &keep),
            "Docker"
        );
        // no previous line
        assert_eq!(
            fix_continuation_case("", "the store", "The store.", &keep),
            "The store."
        );
    }
}
