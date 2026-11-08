//! A project's change log and metrics (`<project>.history.jsonl`), and the in-memory undo stack.
//!
//! The log is append-only, one JSON object per line, so writing an entry never rewrites the file and a
//! crash can at worst cut off the last line, which reading then skips. Editors keep undo in memory
//! only, and so does ViPr ([`Undo`]); the log is the lasting record of what happened and how long the
//! heavy steps took.

use std::collections::{BTreeMap, VecDeque};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Error, Result, io_err};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Created,
    Opened,
    Saved,
    ClipAdded,
    ClipRemoved,
    ClipRelinked,
    Reordered,
    SequenceAdded,
    SequenceChanged,
    SequenceRemoved,
    CutsChanged,
    Undone,
    Redone,
    Transcribed,
    Translated,
    Exported,
    Failed,
    /// Written by a newer ViPr.
    #[serde(other)]
    Unknown,
}

impl Kind {
    /// A few words for the History list.
    pub fn label(self) -> &'static str {
        match self {
            Kind::Created => "Project created",
            Kind::Opened => "Opened",
            Kind::Saved => "Saved",
            Kind::ClipAdded => "Recording added",
            Kind::ClipRemoved => "Recording removed",
            Kind::ClipRelinked => "Recording found again",
            Kind::Reordered => "Order changed",
            Kind::SequenceAdded => "Combination added",
            Kind::SequenceChanged => "Combination changed",
            Kind::SequenceRemoved => "Combination removed",
            Kind::CutsChanged => "Cuts changed",
            Kind::Undone => "Undone",
            Kind::Redone => "Redone",
            Kind::Transcribed => "Transcribed",
            Kind::Translated => "Translated",
            Kind::Exported => "Exported",
            Kind::Failed => "Failed",
            Kind::Unknown => "Change",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    /// Seconds since 1970.
    pub t: u64,
    pub kind: Kind,
    /// The clip it is about, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip: Option<u64>,
    /// What it was about in words: a file name, a language, an export target.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub detail: String,
    /// Measurements, such as `audio_seconds`, `elapsed_seconds`, `bytes`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metrics: BTreeMap<String, f64>,
}

impl Entry {
    pub fn new(kind: Kind, detail: impl Into<String>) -> Entry {
        Entry {
            t: crate::workspace::now(),
            kind,
            clip: None,
            detail: detail.into(),
            metrics: BTreeMap::new(),
        }
    }

    pub fn clip(mut self, id: u64) -> Entry {
        self.clip = Some(id);
        self
    }

    /// Adds a measurement. NaN and infinity can't be written as JSON, so they are left out.
    pub fn metric(mut self, name: &str, value: f64) -> Entry {
        if value.is_finite() {
            self.metrics.insert(name.to_string(), value);
        }
        self
    }
}

/// The change log of one project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryLog {
    pub path: PathBuf,
}

impl HistoryLog {
    pub fn new(path: impl Into<PathBuf>) -> HistoryLog {
        HistoryLog { path: path.into() }
    }

    /// Appends one entry as one line.
    pub fn append(&self, entry: &Entry) -> Result<()> {
        let mut line = serde_json::to_string(entry).map_err(|e| Error::Json {
            path: self.path.display().to_string(),
            msg: e.to_string(),
        })?;
        line.push('\n');
        if let Some(dir) = self.path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(io_err(dir))?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(io_err(&self.path))?;
        file.write_all(line.as_bytes()).map_err(io_err(&self.path))
    }

    /// The last `n` entries, oldest first. A missing file is an empty log; damaged lines are skipped.
    pub fn tail(&self, n: usize) -> Vec<Entry> {
        read_entries(&self.path, n)
    }

    /// Moves the log along with its project (Save As of a temporary project). A missing log is fine.
    pub fn move_to(&mut self, path: &Path) -> Result<()> {
        if self.path.exists() && self.path != path {
            if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
                std::fs::create_dir_all(dir).map_err(io_err(dir))?;
            }
            if std::fs::rename(&self.path, path).is_err() {
                // another drive: copy, then keep appending to the copy
                std::fs::copy(&self.path, path).map_err(io_err(path))?;
                let _ = std::fs::remove_file(&self.path);
            }
        }
        self.path = path.to_path_buf();
        Ok(())
    }
}

fn read_entries(path: &Path, n: usize) -> Vec<Entry> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let entries: Vec<Entry> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let skip = entries.len().saturating_sub(n);
    entries.into_iter().skip(skip).collect()
}

/// Undo and redo by snapshots: before each change the caller [`record`](Undo::record)s the state it
/// had. Snapshots of a project are a few kilobytes, so this is simpler and safer than inverse steps.
#[derive(Debug, Clone)]
pub struct Undo<T> {
    past: VecDeque<(String, T)>,
    future: Vec<(String, T)>,
    limit: usize,
}

impl<T> Default for Undo<T> {
    fn default() -> Undo<T> {
        Undo::new(100)
    }
}

impl<T> Undo<T> {
    pub fn new(limit: usize) -> Undo<T> {
        Undo {
            past: VecDeque::new(),
            future: Vec::new(),
            limit: limit.max(1),
        }
    }

    /// Remembers `before`, the state just before a change called `what`. Clears redo.
    pub fn record(&mut self, what: impl Into<String>, before: T) {
        self.future.clear();
        self.past.push_back((what.into(), before));
        while self.past.len() > self.limit {
            self.past.pop_front();
        }
    }

    /// The state to go back to, given the `current` one, and what the change was.
    pub fn undo(&mut self, current: T) -> Option<(String, T)> {
        let (what, before) = self.past.pop_back()?;
        self.future.push((what.clone(), current));
        Some((what, before))
    }

    pub fn redo(&mut self, current: T) -> Option<(String, T)> {
        let (what, after) = self.future.pop()?;
        self.past.push_back((what.clone(), current));
        Some((what, after))
    }

    /// What Undo would take back, for its tooltip.
    pub fn next_undo(&self) -> Option<&str> {
        self.past.back().map(|(w, _)| w.as_str())
    }

    pub fn next_redo(&self) -> Option<&str> {
        self.future.last().map(|(w, _)| w.as_str())
    }

    pub fn clear(&mut self) {
        self.past.clear();
        self.future.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "vipr-history-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir.join("p.vipr.history.jsonl")
    }

    #[test]
    fn entries_append_and_read_back_in_order_with_their_metrics() {
        let path = temp_file("append");
        let log = HistoryLog::new(&path);
        assert!(log.tail(10).is_empty(), "no file yet");
        log.append(&Entry::new(Kind::Created, "Meeting")).unwrap();
        let done = Entry::new(Kind::Transcribed, "a.mp4")
            .clip(3)
            .metric("audio_seconds", 1082.0)
            .metric("elapsed_seconds", 57.5)
            .metric("broken", f64::NAN);
        log.append(&done).unwrap();
        let back = log.tail(10);
        assert_eq!(back.len(), 2);
        assert_eq!(
            back[1],
            Entry {
                metrics: done.metrics.clone(),
                ..done.clone()
            }
        );
        assert!(!back[1].metrics.contains_key("broken"));
        assert_eq!(log.tail(1), vec![done]);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn a_damaged_or_newer_line_does_not_lose_the_rest() {
        let path = temp_file("damaged");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            "{\"t\":1,\"kind\":\"saved\"}\n{\"t\":2,\"ki\n\n{\"t\":3,\"kind\":\"teleported\",\"new\":true}\n",
        )
        .unwrap();
        let back = HistoryLog::new(&path).tail(10);
        assert_eq!(back.iter().map(|e| e.t).collect::<Vec<_>>(), [1, 3]);
        assert_eq!(back[1].kind, Kind::Unknown);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn the_log_moves_with_its_project() {
        let from = temp_file("move-from");
        let to = temp_file("move-to");
        let mut log = HistoryLog::new(&from);
        log.append(&Entry::new(Kind::Created, "")).unwrap();
        log.move_to(&to).unwrap();
        log.append(&Entry::new(Kind::Saved, "")).unwrap();
        assert!(!from.exists());
        assert_eq!(log.tail(5).len(), 2);
        for p in [from, to] {
            let _ = std::fs::remove_dir_all(p.parent().unwrap());
        }
    }

    #[test]
    fn undo_and_redo_walk_the_snapshots() {
        let mut u: Undo<i32> = Undo::new(3);
        let mut state = 0;
        for next in 1..=5 {
            u.record(format!("set {next}"), state);
            state = next;
        }
        assert_eq!(u.next_undo(), Some("set 5"));
        // only the last 3 changes can be undone
        for expected in [4, 3, 2] {
            let (_, back) = u.undo(state).unwrap();
            state = back;
            assert_eq!(state, expected);
        }
        assert!(u.undo(state).is_none());
        let (what, again) = u.redo(state).unwrap();
        assert_eq!((what.as_str(), again), ("set 3", 3));
        state = again;
        u.record("set 9", state);
        assert!(u.redo(9).is_none(), "a new change clears redo");
        assert_eq!(u.undo(9).unwrap().1, 3);
    }
}
