//! The cleanup driver: chunking, prompt building, reply parsing, split-and-retry and the guards.
//! Ported from `scripts/refine_transcripts.py` (`Refiner._try`, `refine_chunk`, `refine_utterances`).

use std::collections::{HashMap, HashSet};

use anyhow::Result;
use engine::prompts::{self, About, Category};
use serde::{Deserialize, Serialize};

use crate::guards::{self, MAX_FLIP_FRACTION, MAX_FLIP_WORDS};

/// Lines per model call, and how many earlier lines are shown as context.
pub const DEFAULT_CHUNK_LINES: usize = 20;
const CONTEXT_LINES: usize = 4;

/// A language model that answers one chat turn deterministically.
pub trait Chat {
    fn complete(&mut self, system: &str, user: &str, max_new_tokens: usize) -> Result<String>;
    fn count_tokens(&self, text: &str) -> usize;
}

// ---------------------------------------------------------------------------- data

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct InUtterance {
    pub id: u64,
    #[serde(default)]
    pub start: f64,
    pub speaker: String,
    pub text: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Input {
    pub utterances: Vec<InUtterance>,
    #[serde(default)]
    pub about: About,
    #[serde(default)]
    pub glossary: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Change {
    pub id: u64,
    /// `TEXT`, `SPEAKER` or `REJECTED`
    pub kind: &'static str,
    pub before: String,
    pub after: String,
    /// Only for `REJECTED`: `rewrite` (text guard) or `speaker-flip`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct Stats {
    pub text_edits: usize,
    pub speaker_edits: usize,
    /// Rejected text rewrites + rejected speaker flips (same meaning as the Python `rejected`).
    pub rejected: usize,
    /// Lines the model never answered correctly (kept as they were).
    pub skipped: usize,
    pub elapsed_seconds: f64,
}

struct Utt {
    id: u64,
    start: f64,
    speaker: String,
    text: String,
    new_speaker: String,
    new_text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Output {
    pub utterances: Vec<InUtterance>,
    pub changes: Vec<Change>,
    pub stats: Stats,
}

// ---------------------------------------------------------------------------- parsing

/// `^\s*(\d+)\s*\|\s*(<label>)\s*\|\s*(.*?)\s*$`
pub fn parse_reply_line(line: &str) -> Option<(usize, String, String)> {
    let mut parts = line.splitn(3, '|');
    let id = parts.next()?.trim().parse::<usize>().ok()?;
    let speaker = parts.next()?.trim();
    let text = parts.next()?.trim();
    (!speaker.is_empty()).then(|| (id, speaker.to_string(), text.to_string()))
}

// ---------------------------------------------------------------------------- the refiner

pub struct Refiner<C: Chat> {
    chat: C,
    system: String,
    glossary: Vec<String>,
    about: About,
}

type Parsed = HashMap<usize, (String, String)>;

impl<C: Chat> Refiner<C> {
    pub fn new(
        chat: C,
        category: &Category,
        speaker_labels: &[String],
        glossary: Vec<String>,
        about: About,
    ) -> Self {
        Refiner {
            chat,
            system: prompts::build_system_prompt(category, speaker_labels),
            glossary,
            about,
        }
    }

    pub fn system_prompt(&self) -> &str {
        &self.system
    }

    fn try_chunk(
        &mut self,
        idx: &[usize],
        utts: &[Utt],
        context: &[&Utt],
    ) -> Result<Option<Parsed>> {
        let lines: Vec<String> = idx
            .iter()
            .map(|&i| format!("{i}|{}|{}", utts[i].new_speaker, utts[i].new_text))
            .collect();
        let ctx: Vec<(String, String)> = context
            .iter()
            .map(|u| (u.new_speaker.clone(), u.new_text.clone()))
            .collect();
        let user = prompts::build_user_message(&self.glossary, &self.about, &ctx, &lines);
        let expected_tokens = self.chat.count_tokens(&lines.join("\n"));
        let reply = self.chat.complete(
            &self.system,
            &user,
            (expected_tokens as f64 * 1.4) as usize + 64,
        )?;

        let mut parsed: Parsed = HashMap::new();
        for line in reply.lines() {
            if let Some((id, spk, text)) = parse_reply_line(line) {
                parsed.insert(id, (spk, text));
            }
        }
        let want: HashSet<usize> = idx.iter().copied().collect();
        let got: HashSet<usize> = parsed.keys().copied().collect();
        Ok((want == got).then_some(parsed))
    }

    /// Ask the model about `idx`; if it breaks the format, split the chunk and retry, then give up on a single line.
    fn refine_chunk(&mut self, idx: &[usize], utts: &[Utt], context: &[&Utt]) -> Result<Parsed> {
        if let Some(r) = self.try_chunk(idx, utts, context)? {
            return Ok(r);
        }
        if idx.len() == 1 {
            return Ok(HashMap::new()); // give up on this line, keep the original
        }
        let mid = idx.len() / 2;
        let left = self.refine_chunk(&idx[..mid], utts, context)?;
        // Python: `context + [utts[i] for i in idx[:mid]][-4:]` (the last 4 lines of the left half)
        let left_part: Vec<&Utt> = idx[..mid].iter().map(|&i| &utts[i]).collect();
        let mut right_context: Vec<&Utt> = context.to_vec();
        right_context.extend(
            left_part[left_part.len().saturating_sub(CONTEXT_LINES)..]
                .iter()
                .copied(),
        );
        let right = self.refine_chunk(&idx[mid..], utts, &right_context)?;
        let mut all = left;
        all.extend(right);
        Ok(all)
    }

    /// Run the whole cleanup. `progress(done_chunks, total_chunks)` is called after every chunk.
    pub fn refine(
        &mut self,
        input: &[InUtterance],
        chunk_lines: usize,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<Output> {
        let started = std::time::Instant::now();
        let mut utts: Vec<Utt> = input
            .iter()
            .map(|u| Utt {
                id: u.id,
                start: u.start,
                speaker: u.speaker.clone(),
                text: u.text.clone(),
                new_speaker: u.speaker.clone(),
                new_text: u.text.clone(),
            })
            .collect();
        let labels: HashSet<String> = utts.iter().map(|u| u.speaker.clone()).collect();
        let keep_caps: HashSet<String> = self
            .glossary
            .iter()
            .flat_map(|t| t.split_whitespace().map(|w| w.to_lowercase()))
            .collect();
        let mut stats = Stats::default();
        let mut changes: Vec<Change> = Vec::new();

        let chunk_lines = chunk_lines.max(1);
        let total = utts.len().div_ceil(chunk_lines);
        for (n, start) in (0..utts.len()).step_by(chunk_lines).enumerate() {
            let idx: Vec<usize> = (start..(start + chunk_lines).min(utts.len())).collect();
            let result = {
                let context: Vec<&Utt> = utts[start.saturating_sub(CONTEXT_LINES)..start]
                    .iter()
                    .collect();
                self.refine_chunk(&idx, &utts, &context)?
            };
            stats.skipped += idx.len() - result.len();

            let flips = result
                .iter()
                .filter(|(i, (spk, _))| labels.contains(spk) && *spk != utts[**i].speaker)
                .count();
            let allow_flips = flips <= 1.max((MAX_FLIP_FRACTION * idx.len() as f64) as usize);

            let mut ids: Vec<usize> = result.keys().copied().collect();
            ids.sort_unstable();
            for i in ids {
                let (spk, mut text) = result[&i].clone();
                if i > 0 {
                    text = guards::fix_continuation_case(
                        &utts[i - 1].new_text,
                        &utts[i].text,
                        &text,
                        &keep_caps,
                    );
                }
                let old_text = utts[i].text.clone();
                if text != old_text {
                    if guards::acceptable_edit(&old_text, &text) {
                        stats.text_edits += 1;
                        changes.push(Change {
                            id: utts[i].id,
                            kind: "TEXT",
                            before: old_text,
                            after: text.clone(),
                            reason: None,
                        });
                        utts[i].new_text = text;
                    } else {
                        stats.rejected += 1;
                        changes.push(Change {
                            id: utts[i].id,
                            kind: "REJECTED",
                            before: old_text,
                            after: text,
                            reason: Some("rewrite"),
                        });
                    }
                }
                if spk != utts[i].speaker && labels.contains(&spk) {
                    if allow_flips && utts[i].text.split_whitespace().count() <= MAX_FLIP_WORDS {
                        stats.speaker_edits += 1;
                        changes.push(Change {
                            id: utts[i].id,
                            kind: "SPEAKER",
                            before: utts[i].speaker.clone(),
                            after: spk.clone(),
                            reason: None,
                        });
                        utts[i].new_speaker = spk;
                    } else {
                        stats.rejected += 1;
                        changes.push(Change {
                            id: utts[i].id,
                            kind: "REJECTED",
                            before: utts[i].speaker.clone(),
                            after: spk,
                            reason: Some("speaker-flip"),
                        });
                    }
                }
            }
            progress(n + 1, total);
        }

        stats.elapsed_seconds = started.elapsed().as_secs_f64();
        let utterances = utts
            .into_iter()
            .map(|u| InUtterance {
                id: u.id,
                start: u.start,
                speaker: u.new_speaker,
                text: u.new_text,
            })
            .collect();
        Ok(Output {
            utterances,
            changes,
            stats,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// Replies with scripted answers, remembering what it was asked.
    struct Mock {
        replies: VecDeque<String>,
        asked: Vec<String>,
    }

    impl Chat for Mock {
        fn complete(&mut self, _system: &str, user: &str, _max: usize) -> Result<String> {
            self.asked.push(user.to_string());
            Ok(self.replies.pop_front().unwrap_or_default())
        }
        fn count_tokens(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
    }

    fn utt(id: u64, spk: &str, text: &str) -> InUtterance {
        InUtterance {
            id,
            start: id as f64,
            speaker: spk.into(),
            text: text.into(),
        }
    }

    fn refiner(replies: &[&str]) -> Refiner<Mock> {
        let mock = Mock {
            replies: replies.iter().map(|s| s.to_string()).collect(),
            asked: vec![],
        };
        Refiner::new(
            mock,
            &Category::default(),
            &["Speaker 1".into(), "Speaker 2".into()],
            vec!["Cloudflare".into()],
            About::default(),
        )
    }

    #[test]
    fn parses_reply_lines() {
        assert_eq!(
            parse_reply_line(" 3 | Speaker 2 | Hello, there. "),
            Some((3, "Speaker 2".into(), "Hello, there.".into()))
        );
        assert_eq!(
            parse_reply_line("3|Speaker 1|a | b"),
            Some((3, "Speaker 1".into(), "a | b".into()))
        );
        assert_eq!(parse_reply_line("Sure! Here are the lines:"), None);
        assert_eq!(parse_reply_line("x|Speaker 1|hi"), None);
    }

    #[test]
    fn applies_good_edit_and_logs_it() {
        let input = vec![
            utt(10, "Speaker 1", "so we use cloud flare uh today"),
            utt(11, "Speaker 2", "okay"),
        ];
        let mut r = refiner(&["0|Speaker 1|So we use Cloudflare today.\n1|Speaker 2|Okay."]);
        let out = r.refine(&input, 20, &mut |_, _| {}).unwrap();
        assert_eq!(out.utterances.len(), 2);
        assert_eq!(out.utterances[0].id, 10);
        assert_eq!(out.utterances[0].text, "So we use Cloudflare today.");
        assert_eq!(out.stats.text_edits, 2);
        assert_eq!(out.changes[0].kind, "TEXT");
        assert_eq!(out.changes[0].id, 10);
    }

    #[test]
    fn rewrites_are_rejected_and_original_kept() {
        let input = vec![utt(
            1,
            "Speaker 1",
            "yeah i think we should probably deploy that on friday",
        )];
        let mut r = refiner(&["0|Speaker 1|Deploy on Friday."]);
        let out = r.refine(&input, 20, &mut |_, _| {}).unwrap();
        assert_eq!(out.utterances[0].text, input[0].text);
        assert_eq!(out.stats.rejected, 1);
        assert_eq!(out.changes[0].kind, "REJECTED");
        assert_eq!(out.changes[0].reason, Some("rewrite"));
    }

    #[test]
    fn broken_format_splits_the_chunk_then_gives_up_on_a_line() {
        let input = vec![
            utt(1, "Speaker 1", "hello there"),
            utt(2, "Speaker 2", "good morning"),
            utt(3, "Speaker 1", "how are you"),
        ];
        // full chunk: garbage; left half [0]: ok; right half [1,2]: garbage; then [1]: ok; [2]: garbage -> kept
        let mut r = refiner(&[
            "I fixed everything for you",
            "0|Speaker 1|Hello there.",
            "nope",
            "1|Speaker 2|Good morning.",
            "still nope",
        ]);
        let out = r.refine(&input, 20, &mut |_, _| {}).unwrap();
        assert_eq!(out.utterances[0].text, "Hello there.");
        assert_eq!(out.utterances[1].text, "Good morning.");
        assert_eq!(out.utterances[2].text, "how are you");
        assert_eq!(out.stats.skipped, 1);
        assert_eq!(r.chat.asked.len(), 5);
    }

    #[test]
    fn missing_or_extra_ids_count_as_broken() {
        let input = vec![utt(1, "Speaker 1", "hello there")];
        // asks for id 0 but the model answers id 5 -> format broken, single line -> skipped
        let mut r = refiner(&["5|Speaker 1|Hello there."]);
        let out = r.refine(&input, 20, &mut |_, _| {}).unwrap();
        assert_eq!(out.stats.skipped, 1);
        assert_eq!(out.utterances[0].text, "hello there");
    }

    #[test]
    fn speaker_flip_allowed_for_short_line_but_capped() {
        let input = vec![
            utt(1, "Speaker 1", "what do you think"),
            utt(2, "Speaker 1", "okay"),
            utt(3, "Speaker 2", "yes"),
        ];
        let mut r =
            refiner(&["0|Speaker 1|What do you think?\n1|Speaker 2|Okay.\n2|Speaker 2|Yes."]);
        let out = r.refine(&input, 20, &mut |_, _| {}).unwrap();
        assert_eq!(out.utterances[1].speaker, "Speaker 2");
        assert_eq!(out.stats.speaker_edits, 1);

        // too many flips in one chunk (> max(1, 30%)): all rejected
        // (both labels must exist in the input, otherwise an unknown label is simply ignored)
        let mut input: Vec<InUtterance> = (0..4).map(|i| utt(i, "Speaker 1", "okay")).collect();
        input.push(utt(4, "Speaker 2", "right"));
        let mut r = refiner(&[
            "0|Speaker 2|Okay.\n1|Speaker 2|Okay.\n2|Speaker 2|Okay.\n3|Speaker 2|Okay.\n4|Speaker 2|Right.",
        ]);
        let out = r.refine(&input, 20, &mut |_, _| {}).unwrap();
        assert!(out.utterances[..4].iter().all(|u| u.speaker == "Speaker 1"));
        assert_eq!(out.stats.speaker_edits, 0);
        assert_eq!(out.stats.rejected, 4);
        assert_eq!(
            out.changes
                .iter()
                .filter(|c| c.reason == Some("speaker-flip"))
                .count(),
            4
        );
    }

    #[test]
    fn unknown_speaker_label_is_ignored_and_long_lines_never_flip() {
        let input = vec![
            utt(1, "Speaker 1", "okay"),
            utt(
                2,
                "Speaker 2",
                "one two three four five six seven eight nine ten eleven twelve thirteen",
            ),
        ];
        let mut r = refiner(&[
            "0|Speaker 9|Okay.\n1|Speaker 1|One two three four five six seven eight nine ten eleven twelve thirteen.",
        ]);
        let out = r.refine(&input, 20, &mut |_, _| {}).unwrap();
        assert_eq!(out.utterances[0].speaker, "Speaker 1");
        assert_eq!(out.utterances[1].speaker, "Speaker 2");
    }

    #[test]
    fn chunks_carry_context_and_report_progress() {
        let input: Vec<InUtterance> = (0..5)
            .map(|i| utt(i, "Speaker 1", &format!("line number {i}")))
            .collect();
        let mut r = refiner(&[
            "0|Speaker 1|line number 0\n1|Speaker 1|line number 1",
            "2|Speaker 1|line number 2\n3|Speaker 1|line number 3",
            "4|Speaker 1|line number 4",
        ]);
        let mut seen = vec![];
        r.refine(&input, 2, &mut |d, t| seen.push((d, t))).unwrap();
        assert_eq!(seen, vec![(1, 3), (2, 3), (3, 3)]);
        assert!(!r.chat.asked[0].contains("Previous lines"));
        assert!(r.chat.asked[1].contains("Previous lines (context only, do NOT output them):\nSpeaker 1: line number 0\nSpeaker 1: line number 1"));
        assert!(r.chat.asked[0].contains("Glossary (preferred spellings): Cloudflare"));
    }

    #[test]
    fn same_line_count_and_ids_always() {
        let input: Vec<InUtterance> = (0..45)
            .map(|i| utt(100 + i, "Speaker 1", "hello"))
            .collect();
        let mut r = refiner(&[]); // model never answers
        let out = r.refine(&input, 20, &mut |_, _| {}).unwrap();
        assert_eq!(out.utterances.len(), 45);
        assert!(out.utterances.iter().zip(&input).all(|(a, b)| a == b));
        assert_eq!(out.stats.skipped, 45);
    }
}
