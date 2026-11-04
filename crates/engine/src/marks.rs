//! Bookmarks and flagged subtitle lines of one recording (`<stem>.marks.json`).

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Error, Result, fsio, io_err};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MarkKind {
    /// A moment, with an optional note.
    Bookmark,
    /// A subtitle line that needs attention.
    Flag,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mark {
    /// Unique within the file. Never reused after a delete.
    pub id: u64,
    pub kind: MarkKind,
    /// Seconds from the start of the media. Authoritative: `cue` is only a hint, since cue ids
    /// can change when subtitles are rebuilt.
    pub time: f64,
    #[serde(default)]
    pub note: String,
    /// Id of the flagged cue, when there is one.
    #[serde(default)]
    pub cue: Option<u64>,
}

/// The whole `<stem>.marks.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarksFile {
    #[serde(default = "crate::project::version")]
    pub version: u32,
    /// Saved so that deleting the newest mark can't free its id for the next one.
    #[serde(default)]
    pub next_id: u64,
    #[serde(default)]
    pub marks: Vec<Mark>,
}

/// One recording's marks and the file they save to.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkStore {
    pub path: PathBuf,
    /// Sorted by `time`.
    pub marks: Vec<Mark>,
    /// The id the next new mark gets.
    pub next_id: u64,
}

impl MarkStore {
    /// A missing file gives an empty store. A corrupt or unreadable one is an error, so nothing
    /// overwrites it.
    pub fn load(path: &Path) -> Result<MarkStore> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == ErrorKind::NotFound => {
                return Ok(MarkStore {
                    path: path.to_path_buf(),
                    marks: Vec::new(),
                    next_id: 1,
                });
            }
            Err(e) => return Err(io_err(path)(e)),
        };
        let file: MarksFile = serde_json::from_str(&text).map_err(|e| Error::Json {
            path: path.display().to_string(),
            msg: e.to_string(),
        })?;
        let mut marks = file.marks;
        marks.sort_by(|a, b| a.time.total_cmp(&b.time));
        // A hand-edited file may hold an id at or above its counter.
        let above_all = marks.iter().map(|m| m.id.saturating_add(1)).max();
        Ok(MarkStore {
            path: path.to_path_buf(),
            marks,
            next_id: file.next_id.max(above_all.unwrap_or(1)).max(1),
        })
    }

    /// Atomic.
    pub fn save(&self) -> Result<()> {
        fsio::save_json(
            &self.path,
            &MarksFile {
                version: crate::project::VERSION,
                next_id: self.next_id,
                marks: self.marks.clone(),
            },
        )
    }

    /// Adds a mark in time order and returns its id. Does not save. A negative or non-finite time
    /// becomes 0: JSON can't hold NaN, so one would make the saved file unreadable.
    pub fn add(&mut self, kind: MarkKind, time: f64, note: String, cue: Option<u64>) -> u64 {
        let time = if time.is_finite() { time.max(0.0) } else { 0.0 };
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let at = self.marks.partition_point(|m| m.time <= time);
        self.marks.insert(
            at,
            Mark {
                id,
                kind,
                time,
                note,
                cue,
            },
        );
        id
    }

    /// Returns false if there is no such mark. Does not save.
    pub fn set_note(&mut self, id: u64, note: String) -> bool {
        match self.marks.iter_mut().find(|m| m.id == id) {
            Some(mark) => {
                mark.note = note;
                true
            }
            None => false,
        }
    }

    /// Returns false if there is no such mark. Does not save.
    pub fn remove(&mut self, id: u64) -> bool {
        let before = self.marks.len();
        self.marks.retain(|m| m.id != id);
        self.marks.len() != before
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh folder under the system temp dir, removed when dropped.
    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> TestDir {
            let dir = std::env::temp_dir().join(format!(
                "vipr-marks-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            TestDir(dir)
        }

        fn file(&self) -> PathBuf {
            self.0.join("a.marks.json")
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn times(store: &MarkStore) -> Vec<f64> {
        store.marks.iter().map(|m| m.time).collect()
    }

    #[test]
    fn a_missing_file_gives_an_empty_store() {
        let dir = TestDir::new("missing");
        let store = MarkStore::load(&dir.file()).unwrap();
        assert!(store.marks.is_empty());
        assert_eq!(store.path, dir.file());
    }

    #[test]
    fn round_trip_keeps_everything() {
        let dir = TestDir::new("round");
        let mut store = MarkStore::load(&dir.file()).unwrap();
        store.add(MarkKind::Flag, 12.5, "سلام <|im_end|>".into(), Some(7));
        store.add(MarkKind::Bookmark, 3.0, String::new(), None);
        store.save().unwrap();
        assert_eq!(MarkStore::load(&dir.file()).unwrap(), store);
    }

    #[test]
    fn adding_keeps_time_order_and_equal_times_stay_in_order_of_adding() {
        let dir = TestDir::new("order");
        let mut store = MarkStore::load(&dir.file()).unwrap();
        let first = store.add(MarkKind::Bookmark, 5.0, String::new(), None);
        store.add(MarkKind::Bookmark, 1.0, String::new(), None);
        let second = store.add(MarkKind::Flag, 5.0, String::new(), None);
        store.add(MarkKind::Bookmark, 9.0, String::new(), None);
        assert_eq!(times(&store), [1.0, 5.0, 5.0, 9.0]);
        assert_eq!(store.marks[1].id, first);
        assert_eq!(store.marks[2].id, second);
    }

    #[test]
    fn a_bad_time_becomes_zero_and_the_file_stays_readable() {
        let dir = TestDir::new("nan");
        let mut store = MarkStore::load(&dir.file()).unwrap();
        store.add(MarkKind::Bookmark, f64::NAN, String::new(), None);
        store.add(MarkKind::Bookmark, f64::INFINITY, String::new(), None);
        store.add(MarkKind::Bookmark, -4.0, String::new(), None);
        store.save().unwrap();
        assert_eq!(
            times(&MarkStore::load(&dir.file()).unwrap()),
            [0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn ids_are_not_reused_after_a_delete_even_across_a_reload() {
        let dir = TestDir::new("ids");
        let mut store = MarkStore::load(&dir.file()).unwrap();
        store.add(MarkKind::Bookmark, 1.0, String::new(), None);
        let newest = store.add(MarkKind::Bookmark, 2.0, String::new(), None);
        assert!(store.remove(newest));
        assert!(!store.remove(newest));
        store.save().unwrap();

        let mut back = MarkStore::load(&dir.file()).unwrap();
        assert!(back.add(MarkKind::Bookmark, 3.0, String::new(), None) > newest);
    }

    #[test]
    fn set_note_finds_by_id() {
        let dir = TestDir::new("note");
        let mut store = MarkStore::load(&dir.file()).unwrap();
        let id = store.add(MarkKind::Bookmark, 1.0, String::new(), None);
        assert!(store.set_note(id, "look here".into()));
        assert_eq!(store.marks[0].note, "look here");
        assert!(!store.set_note(id + 1, "nothing".into()));
    }

    #[test]
    fn a_corrupt_file_is_an_error_and_stays_as_it_was() {
        let dir = TestDir::new("corrupt");
        std::fs::write(dir.file(), "{not json").unwrap();
        assert!(matches!(
            MarkStore::load(&dir.file()),
            Err(Error::Json { .. })
        ));
        assert_eq!(std::fs::read_to_string(dir.file()).unwrap(), "{not json");
    }

    #[test]
    fn an_unreadable_path_is_an_error_not_an_empty_store() {
        let dir = TestDir::new("unreadable");
        // a folder at the file's path: reading fails with something other than "not found"
        std::fs::create_dir(dir.file()).unwrap();
        assert!(matches!(
            MarkStore::load(&dir.file()),
            Err(Error::Io { .. })
        ));
    }

    #[test]
    fn a_file_with_only_the_foundation_fields_loads_unsorted_and_ids_stay_clear() {
        let dir = TestDir::new("older");
        std::fs::write(
            dir.file(),
            r#"{"marks":[
                {"id":9,"kind":"flag","time":8.0,"cue":3},
                {"id":2,"kind":"bookmark","time":1.0,"note":"n"}
            ]}"#,
        )
        .unwrap();
        let mut store = MarkStore::load(&dir.file()).unwrap();
        assert_eq!(times(&store), [1.0, 8.0]);
        assert_eq!(store.marks[1].cue, Some(3));
        assert_eq!(store.add(MarkKind::Bookmark, 2.0, String::new(), None), 10);
    }

    #[test]
    fn the_saved_file_names_kinds_in_snake_case_and_has_a_version() {
        let dir = TestDir::new("json");
        let mut store = MarkStore::load(&dir.file()).unwrap();
        store.add(MarkKind::Bookmark, 1.0, String::new(), None);
        store.save().unwrap();
        let file: MarksFile = fsio::load_json(&dir.file()).unwrap();
        assert_eq!(file.version, crate::project::VERSION);
        let text = std::fs::read_to_string(dir.file()).unwrap();
        assert!(text.contains("\"bookmark\""));
    }
}
