//! A ViPr project file (`<name>.vipr`): the recordings worked on together, in their order, and the
//! combinations made of them.
//!
//! Modelled on what video editors do (Kdenlive, Shotcut, Premiere, OpenTimelineIO):
//! - one readable JSON file with a schema `version`; fields added later get `serde(default)` and
//!   unknown fields are ignored, so older and newer files both load;
//! - media is referenced, never copied: the absolute path plus the path relative to the project file,
//!   so a folder moved together with its project still resolves, and a missing file can be relinked;
//! - everything that can be rebuilt (transcripts, translations, cuts) stays in each recording's result
//!   folder (`project::ProjectPaths`), under the user's output folder;
//! - autosaved copies go to `<file>.backups/` (the last [`KEEP_BACKUPS`]), apart from the user's own
//!   Save, and the change log to `<file>.history.jsonl` (`history`).
//!
//! Not to be confused with `project.rs`, which is the sidecar format of one recording's result folder.

use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::fsio;
use crate::project::ProjectPaths;
use crate::{Result, io_err};

pub const VERSION: u32 = 1;
/// File extension of a project, without the dot.
pub const EXTENSION: &str = "vipr";
/// Autosaved copies kept next to a project.
pub const KEEP_BACKUPS: usize = 10;

fn version() -> u32 {
    VERSION
}

fn first_id() -> u64 {
    1
}

/// Seconds since 1970, the time stamp of project files and the change log.
pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Workspace {
    #[serde(default = "version")]
    pub version: u32,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub created: u64,
    #[serde(default)]
    pub modified: u64,
    /// In the order the user sees them.
    #[serde(default)]
    pub clips: Vec<Clip>,
    #[serde(default)]
    pub sequences: Vec<Sequence>,
    /// Ids are shared by clips and sequences and never reused.
    #[serde(default = "first_id")]
    pub next_id: u64,
}

/// One recording in the project.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Clip {
    pub id: u64,
    pub media: MediaRef,
    /// Its result folder, once it has been transcribed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<ResultRef>,
    /// A name the user gave it; the file name otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MediaRef {
    /// Absolute, as it was when added or last relinked.
    pub path: PathBuf,
    /// Relative to the project file's folder, written on every save.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relative: Option<PathBuf>,
    #[serde(default)]
    pub size: u64,
}

/// Where a clip's results are: `<dir>/<stem>.json` and its sidecars.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResultRef {
    pub dir: PathBuf,
    pub stem: String,
}

impl ResultRef {
    pub fn paths(&self) -> ProjectPaths {
        ProjectPaths::new(self.dir.clone(), self.stem.clone())
    }
}

impl From<&ProjectPaths> for ResultRef {
    fn from(p: &ProjectPaths) -> ResultRef {
        ResultRef {
            dir: p.dir.clone(),
            stem: p.stem.clone(),
        }
    }
}

/// Clips played one after another and exported as one file, each with its own cuts applied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sequence {
    pub id: u64,
    pub name: String,
    /// Clip ids in playing order. A clip may appear more than once.
    #[serde(default)]
    pub items: Vec<u64>,
}

impl Clip {
    /// What the user calls it: the label, else the media's file name.
    pub fn title(&self) -> String {
        match &self.label {
            Some(l) if !l.trim().is_empty() => l.clone(),
            _ => self.media.path.file_name().map_or_else(
                || self.media.path.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            ),
        }
    }
}

impl Workspace {
    pub fn new(name: impl Into<String>) -> Workspace {
        let t = now();
        Workspace {
            version: VERSION,
            name: name.into(),
            created: t,
            modified: t,
            clips: Vec::new(),
            sequences: Vec::new(),
            next_id: 1,
        }
    }

    /// Reads a project file. A missing file is `Error::Io`, a damaged one `Error::Json`. Ids and
    /// sequences are made consistent, so a hand-edited file can't break the app.
    pub fn load(path: &Path) -> Result<Workspace> {
        let mut w: Workspace = fsio::load_json(path)?;
        w.repair();
        Ok(w)
    }

    /// Writes the project to `path` atomically, with media paths relative to its folder.
    pub fn save(&mut self, path: &Path) -> Result<()> {
        self.modified = now();
        self.version = self.version.max(VERSION);
        let base = path.parent().unwrap_or(Path::new(""));
        for clip in &mut self.clips {
            clip.media.relative = relative_path(&clip.media.path, base);
        }
        fsio::save_json(path, self)
    }

    fn repair(&mut self) {
        let mut seen = std::collections::HashSet::new();
        self.clips.retain(|c| seen.insert(c.id));
        self.sequences.retain(|s| seen.insert(s.id));
        let clip_ids: std::collections::HashSet<u64> = self.clips.iter().map(|c| c.id).collect();
        for seq in &mut self.sequences {
            seq.items.retain(|id| clip_ids.contains(id));
        }
        let highest = seen.into_iter().max().unwrap_or(0);
        self.next_id = self.next_id.max(highest + 1);
    }

    fn take_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn clip(&self, id: u64) -> Option<&Clip> {
        self.clips.iter().find(|c| c.id == id)
    }

    pub fn clip_mut(&mut self, id: u64) -> Option<&mut Clip> {
        self.clips.iter_mut().find(|c| c.id == id)
    }

    /// The clip of this media file, if the project has it.
    pub fn clip_for_media(&self, media: &Path) -> Option<&Clip> {
        self.clips.iter().find(|c| same_path(&c.media.path, media))
    }

    /// Adds a recording at the end, or returns the clip it already has.
    pub fn add_clip(&mut self, media: &Path, size: u64) -> u64 {
        if let Some(c) = self.clip_for_media(media) {
            return c.id;
        }
        let id = self.take_id();
        self.clips.push(Clip {
            id,
            media: MediaRef {
                path: media.to_path_buf(),
                relative: None,
                size,
            },
            result: None,
            label: None,
        });
        id
    }

    /// Removes a clip and every place a combination used it. `false` if there was no such clip.
    pub fn remove_clip(&mut self, id: u64) -> bool {
        let before = self.clips.len();
        self.clips.retain(|c| c.id != id);
        for seq in &mut self.sequences {
            seq.items.retain(|i| *i != id);
        }
        self.clips.len() != before
    }

    /// Moves a clip `by` places (negative = up), stopping at either end. `false` if nothing moved.
    pub fn move_clip(&mut self, id: u64, by: isize) -> bool {
        move_item(&mut self.clips, |c| c.id == id, by)
    }

    pub fn set_result(&mut self, id: u64, result: ResultRef) -> bool {
        match self.clip_mut(id) {
            Some(c) if c.result.as_ref() != Some(&result) => {
                c.result = Some(result);
                true
            }
            _ => false,
        }
    }

    /// Points a clip at a recording that moved. `false` if there was no such clip.
    pub fn relink(&mut self, id: u64, media: &Path) -> bool {
        let size = std::fs::metadata(media).map_or(0, |m| m.len());
        match self.clip_mut(id) {
            Some(c) => {
                c.media.path = media.to_path_buf();
                c.media.relative = None;
                c.media.size = size;
                true
            }
            None => false,
        }
    }

    /// Where the clip's recording is now: its absolute path, else its path relative to the project
    /// file (`project_file`), else `None`, which the app shows as "media missing".
    pub fn media_path(&self, id: u64, project_file: Option<&Path>) -> Option<PathBuf> {
        let clip = self.clip(id)?;
        if clip.media.path.is_file() {
            return Some(clip.media.path.clone());
        }
        let base = project_file?.parent()?;
        let moved = base.join(clip.media.relative.as_ref()?);
        moved.is_file().then_some(moved)
    }

    pub fn sequence(&self, id: u64) -> Option<&Sequence> {
        self.sequences.iter().find(|s| s.id == id)
    }

    pub fn sequence_mut(&mut self, id: u64) -> Option<&mut Sequence> {
        self.sequences.iter_mut().find(|s| s.id == id)
    }

    /// A new, empty combination at the end.
    pub fn add_sequence(&mut self, name: impl Into<String>) -> u64 {
        let id = self.take_id();
        self.sequences.push(Sequence {
            id,
            name: name.into(),
            items: Vec::new(),
        });
        id
    }

    pub fn remove_sequence(&mut self, id: u64) -> bool {
        let before = self.sequences.len();
        self.sequences.retain(|s| s.id != id);
        self.sequences.len() != before
    }

    /// Adds a clip to the end of a combination. `false` for an unknown clip or combination.
    pub fn push_to_sequence(&mut self, seq: u64, clip: u64) -> bool {
        if self.clip(clip).is_none() {
            return false;
        }
        match self.sequence_mut(seq) {
            Some(s) => {
                s.items.push(clip);
                true
            }
            None => false,
        }
    }

    /// Moves the item at `index` of a combination `by` places.
    pub fn move_in_sequence(&mut self, seq: u64, index: usize, by: isize) -> bool {
        let Some(s) = self.sequence_mut(seq) else {
            return false;
        };
        let Some(to) = index.checked_add_signed(by).filter(|t| *t < s.items.len()) else {
            return false;
        };
        if index >= s.items.len() || to == index {
            return false;
        }
        let item = s.items.remove(index);
        s.items.insert(to, item);
        true
    }

    pub fn remove_from_sequence(&mut self, seq: u64, index: usize) -> bool {
        match self.sequence_mut(seq) {
            Some(s) if index < s.items.len() => {
                s.items.remove(index);
                true
            }
            _ => false,
        }
    }

    /// Writes a copy to `<project>.backups/` and deletes all but the newest `keep`. Returns the copy.
    pub fn backup(&self, project_file: &Path, keep: usize) -> Result<PathBuf> {
        let dir = backups_dir(project_file);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        let copy = dir.join(format!("project-{stamp}.{EXTENSION}"));
        fsio::save_json(&copy, self)?;
        let mut copies = backups(project_file);
        while copies.len() > keep {
            let oldest = copies.remove(0);
            let _ = std::fs::remove_file(oldest);
        }
        Ok(copy)
    }
}

/// `<project>.history.jsonl`, the change log next to a project file.
pub fn history_path(project_file: &Path) -> PathBuf {
    sibling(project_file, "history.jsonl")
}

/// `<project>.backups`, the autosaved copies of a project file.
pub fn backups_dir(project_file: &Path) -> PathBuf {
    sibling(project_file, "backups")
}

/// The autosaved copies of a project, oldest first.
pub fn backups(project_file: &Path) -> Vec<PathBuf> {
    let Ok(read) = std::fs::read_dir(backups_dir(project_file)) else {
        return Vec::new();
    };
    let mut copies: Vec<(u128, PathBuf)> = read
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let stamp = name
                .strip_prefix("project-")?
                .strip_suffix(&format!(".{EXTENSION}"))?
                .parse()
                .ok()?;
            Some((stamp, e.path()))
        })
        .collect();
    copies.sort();
    copies.into_iter().map(|(_, p)| p).collect()
}

fn sibling(project_file: &Path, suffix: &str) -> PathBuf {
    let name = project_file
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    project_file.with_file_name(format!("{name}.{suffix}"))
}

/// Where temporary projects live: `<root>/untitled-<time>/untitled.vipr`, one folder each.
pub fn new_temporary(root: &Path) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    root.join(format!("untitled-{stamp}"))
        .join(format!("untitled.{EXTENSION}"))
}

/// The default `root` for [`new_temporary`]: `<system temp>/vipr`.
pub fn temporary_root() -> PathBuf {
    std::env::temp_dir().join("vipr")
}

/// Temporary projects under `root` that hold at least one clip, newest first: what can be restored
/// after a crash or a close.
pub fn temporary_projects(root: &Path) -> Vec<(PathBuf, Workspace)> {
    let Ok(read) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut found: Vec<(PathBuf, Workspace)> = read
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("untitled-"))
        .map(|e| e.path().join(format!("untitled.{EXTENSION}")))
        .filter_map(|p| Workspace::load(&p).ok().map(|w| (p, w)))
        .filter(|(_, w)| !w.clips.is_empty())
        .collect();
    found.sort_by_key(|a| std::cmp::Reverse(a.1.modified));
    found
}

/// Deletes a temporary project's folder. Refuses anything that isn't one.
pub fn remove_temporary(project_file: &Path) -> Result<()> {
    let Some(dir) = project_file.parent() else {
        return Ok(());
    };
    let is_temp = dir
        .file_name()
        .is_some_and(|n| n.to_string_lossy().starts_with("untitled-"));
    if !is_temp || !dir.exists() {
        return Ok(());
    }
    std::fs::remove_dir_all(dir).map_err(io_err(dir))
}

/// `true` if the project file lives in a temporary project folder.
pub fn is_temporary(project_file: &Path) -> bool {
    project_file
        .parent()
        .and_then(Path::file_name)
        .is_some_and(|n| n.to_string_lossy().starts_with("untitled-"))
}

fn move_item<T>(items: &mut Vec<T>, is: impl Fn(&T) -> bool, by: isize) -> bool {
    let Some(at) = items.iter().position(is) else {
        return false;
    };
    let to = at
        .saturating_add_signed(by)
        .min(items.len().saturating_sub(1));
    if to == at {
        return false;
    }
    let item = items.remove(at);
    items.insert(to, item);
    true
}

/// Windows paths compare without regard to case.
fn same_path(a: &Path, b: &Path) -> bool {
    if cfg!(windows) {
        a.to_string_lossy().to_lowercase() == b.to_string_lossy().to_lowercase()
    } else {
        a == b
    }
}

/// `path` relative to `base`, if both are absolute and on the same drive.
fn relative_path(path: &Path, base: &Path) -> Option<PathBuf> {
    if !path.is_absolute() || !base.is_absolute() {
        return None;
    }
    let key = |c: Component| -> String {
        let s = c.as_os_str().to_string_lossy().into_owned();
        if cfg!(windows) { s.to_lowercase() } else { s }
    };
    let p: Vec<Component> = path.components().collect();
    let b: Vec<Component> = base.components().collect();
    // a different drive or share has no relative path
    if p.first().map(|c| key(*c)) != b.first().map(|c| key(*c)) {
        return None;
    }
    let common = p
        .iter()
        .zip(&b)
        .take_while(|(x, y)| key(**x) == key(**y))
        .count();
    let mut rel = PathBuf::new();
    for _ in common..b.len() {
        rel.push("..");
    }
    for c in &p[common..] {
        rel.push(c.as_os_str());
    }
    Some(rel)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh folder under the system temp dir, removed when dropped.
    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> TestDir {
            let dir = std::env::temp_dir().join(format!(
                "vipr-workspace-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            TestDir(dir)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"media").unwrap();
    }

    #[test]
    fn a_project_saves_and_loads_back_the_same() {
        let dir = TestDir::new("roundtrip");
        let file = dir.0.join("Meeting.vipr");
        let mut w = Workspace::new("Meeting");
        let a = w.add_clip(&dir.0.join("media/a.mp4"), 10);
        let b = w.add_clip(&dir.0.join("media/b.mp4"), 20);
        assert_eq!(
            w.add_clip(&dir.0.join("media/a.mp4"), 10),
            a,
            "same file, same clip"
        );
        w.set_result(
            a,
            ResultRef {
                dir: dir.0.join("out/a"),
                stem: "a".into(),
            },
        );
        let seq = w.add_sequence("Both");
        assert!(w.push_to_sequence(seq, a) && w.push_to_sequence(seq, b));
        w.save(&file).unwrap();

        let back = Workspace::load(&file).unwrap();
        assert_eq!(back, w);
        assert_eq!(
            back.clip(a).unwrap().media.relative,
            Some(PathBuf::from("media").join("a.mp4"))
        );
        assert_eq!(
            back.clip(a)
                .unwrap()
                .result
                .as_ref()
                .unwrap()
                .paths()
                .transcript(),
            dir.0.join("out/a/a.json")
        );
        assert_eq!(back.next_id, 4);
    }

    #[test]
    fn old_minimal_and_newer_files_still_load() {
        let dir = TestDir::new("compat");
        let file = dir.0.join("p.vipr");
        std::fs::write(
            &file,
            r#"{"clips":[{"id":7,"media":{"path":"C:/m/a.mp4"}}]}"#,
        )
        .unwrap();
        let w = Workspace::load(&file).unwrap();
        assert_eq!(w.version, VERSION);
        assert_eq!(w.next_id, 8, "ids are never reused");

        let newer = r#"{"version":9,"name":"n","future":{"x":1},"clips":[],
            "sequences":[{"id":3,"name":"s","items":[3,99],"transition":"fade"}],"next_id":1}"#;
        std::fs::write(&file, newer).unwrap();
        let w = Workspace::load(&file).unwrap();
        assert!(w.sequences[0].items.is_empty(), "unknown clips are dropped");
        assert_eq!(w.next_id, 4);

        std::fs::write(&file, "{ not json").unwrap();
        assert!(matches!(
            Workspace::load(&file),
            Err(crate::Error::Json { .. })
        ));
    }

    #[test]
    fn a_moved_folder_is_found_through_the_relative_path_and_can_be_relinked() {
        let dir = TestDir::new("relink");
        let old = dir.0.join("old");
        touch(&old.join("media/a.mp4"));
        let file = old.join("p.vipr");
        let mut w = Workspace::new("p");
        let a = w.add_clip(&old.join("media/a.mp4"), 5);
        w.save(&file).unwrap();

        let new = dir.0.join("new");
        std::fs::rename(&old, &new).unwrap();
        let moved = new.join("p.vipr");
        let w = Workspace::load(&moved).unwrap();
        assert_eq!(
            w.media_path(a, Some(&moved)),
            Some(new.join("media").join("a.mp4"))
        );
        assert_eq!(
            w.media_path(a, None),
            None,
            "no project file, no relative path"
        );

        let mut w = w;
        std::fs::remove_file(new.join("media/a.mp4")).unwrap();
        assert_eq!(w.media_path(a, Some(&moved)), None, "missing");
        let elsewhere = dir.0.join("elsewhere/a2.mp4");
        touch(&elsewhere);
        assert!(w.relink(a, &elsewhere));
        assert_eq!(w.media_path(a, Some(&moved)), Some(elsewhere));
        assert_eq!(w.clip(a).unwrap().media.size, 5);
    }

    #[test]
    fn clips_and_combination_items_reorder_and_remove() {
        let mut w = Workspace::new("p");
        let a = w.add_clip(Path::new("/a.mp4"), 0);
        let b = w.add_clip(Path::new("/b.mp4"), 0);
        let c = w.add_clip(Path::new("/c.mp4"), 0);
        assert!(w.move_clip(c, -5));
        assert_eq!(w.clips.iter().map(|x| x.id).collect::<Vec<_>>(), [c, a, b]);
        assert!(!w.move_clip(c, -1), "already first");
        assert!(w.move_clip(c, 1));
        assert_eq!(w.clips.iter().map(|x| x.id).collect::<Vec<_>>(), [a, c, b]);

        let s = w.add_sequence("s");
        for id in [a, b, a] {
            assert!(w.push_to_sequence(s, id));
        }
        assert!(!w.push_to_sequence(s, 999));
        assert!(w.move_in_sequence(s, 2, -2));
        assert_eq!(w.sequence(s).unwrap().items, [a, a, b]);
        assert!(!w.move_in_sequence(s, 0, -1));
        assert!(w.remove_clip(a));
        assert_eq!(w.sequence(s).unwrap().items, [b], "removed everywhere");
        assert!(w.remove_from_sequence(s, 0));
        assert!(!w.remove_from_sequence(s, 0));
        assert!(w.remove_sequence(s) && !w.remove_sequence(s));
    }

    #[test]
    fn backups_keep_only_the_newest() {
        let dir = TestDir::new("backups");
        let file = dir.0.join("p.vipr");
        let mut w = Workspace::new("p");
        let mut last = PathBuf::new();
        for i in 0..13 {
            w.name = format!("v{i}");
            last = w.backup(&file, 10).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let copies = backups(&file);
        assert_eq!(copies.len(), 10);
        assert_eq!(copies.last(), Some(&last));
        assert_eq!(Workspace::load(&last).unwrap().name, "v12");
        assert_eq!(Workspace::load(&copies[0]).unwrap().name, "v3");
        assert_eq!(history_path(&file), dir.0.join("p.vipr.history.jsonl"));
    }

    #[test]
    fn temporary_projects_with_clips_can_be_found_and_removed() {
        let dir = TestDir::new("temp");
        let empty = new_temporary(&dir.0);
        Workspace::new("").save(&empty).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let used = new_temporary(&dir.0);
        assert_ne!(empty, used);
        let mut w = Workspace::new("");
        w.add_clip(Path::new("/a.mp4"), 1);
        w.save(&used).unwrap();
        assert!(is_temporary(&used) && !is_temporary(&dir.0.join("p.vipr")));

        let found = temporary_projects(&dir.0);
        assert_eq!(found.len(), 1, "an empty one is not worth restoring");
        assert_eq!(found[0].0, used);

        remove_temporary(&used).unwrap();
        assert!(!used.parent().unwrap().exists());
        // never deletes a folder that isn't a temporary project
        let keep = dir.0.join("mine/p.vipr");
        touch(&keep);
        remove_temporary(&keep).unwrap();
        assert!(keep.exists());
    }

    #[test]
    fn relative_paths_only_within_one_drive() {
        if cfg!(windows) {
            assert_eq!(
                relative_path(Path::new(r"C:\a\b\c.mp4"), Path::new(r"c:\a\x")),
                Some(PathBuf::from(r"..\b\c.mp4"))
            );
            assert_eq!(
                relative_path(Path::new(r"D:\c.mp4"), Path::new(r"C:\a")),
                None
            );
        } else {
            assert_eq!(
                relative_path(Path::new("/a/b/c.mp4"), Path::new("/a/x")),
                Some(PathBuf::from("../b/c.mp4"))
            );
        }
        assert_eq!(relative_path(Path::new("c.mp4"), Path::new("/a")), None);
    }
}
