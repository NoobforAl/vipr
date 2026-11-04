//! Resume file for translation: `<stem>.<lang>.ckpt.jsonl`, one `TranslationCue` per line.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use engine::project::TranslationCue;

/// An open checkpoint that cues are appended to.
#[derive(Debug)]
pub struct Checkpoint {
    pub path: PathBuf,
    file: File,
}

/// The cues in the file and how many leading bytes end in a whole line.
fn read_cues(bytes: &[u8]) -> (Vec<TranslationCue>, usize) {
    let mut cues = Vec::new();
    let mut whole = 0;
    let mut rest = bytes;
    while let Some(end) = rest.iter().position(|b| *b == b'\n') {
        // A bad line in the middle is skipped: its cue simply gets translated again.
        if let Ok(cue) = serde_json::from_slice::<TranslationCue>(&rest[..end]) {
            cues.push(cue);
        }
        whole += end + 1;
        rest = &rest[end + 1..];
    }
    (cues, whole)
}

impl Checkpoint {
    /// Returns the checkpoint and the cues already in it. A missing file means no cues, and a torn
    /// last line from a crash is ignored, and cut off so the next append starts on a fresh line.
    pub fn open(path: &Path) -> Result<(Checkpoint, Vec<TranslationCue>)> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        let (cues, whole) = read_cues(&bytes);
        if whole < bytes.len() {
            // Its own handle: Windows refuses to resize through an append-only one.
            OpenOptions::new()
                .write(true)
                .open(path)
                .and_then(|f| f.set_len(whole as u64))
                .with_context(|| format!("repairing {}", path.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        let checkpoint = Checkpoint {
            path: path.to_path_buf(),
            file,
        };
        Ok((checkpoint, cues))
    }

    /// Appends cues and flushes them to disk.
    pub fn append(&mut self, cues: &[TranslationCue]) -> Result<()> {
        let mut text = String::new();
        for cue in cues {
            text.push_str(&serde_json::to_string(cue)?);
            text.push('\n');
        }
        self.file
            .write_all(text.as_bytes())
            .and_then(|()| self.file.sync_data())
            .with_context(|| format!("writing {}", self.path.display()))
    }

    /// Deletes the file, after a successful run.
    pub fn finish(self) -> Result<()> {
        let Checkpoint { path, file } = self;
        drop(file); // Windows can't delete an open file
        std::fs::remove_file(&path).with_context(|| format!("deleting {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::project::CueStatus;

    /// A path in a fresh folder, removed when dropped.
    struct TestPath(PathBuf);

    impl TestPath {
        fn new(name: &str) -> TestPath {
            let dir = std::env::temp_dir().join(format!(
                "vipr-ckpt-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            TestPath(dir.join("a.fa.ckpt.jsonl"))
        }
    }

    impl Drop for TestPath {
        fn drop(&mut self) {
            if let Some(dir) = self.0.parent() {
                let _ = std::fs::remove_dir_all(dir);
            }
        }
    }

    fn cue(id: u64, text: &str) -> TranslationCue {
        TranslationCue {
            id,
            start: 0.0,
            end: 1.0,
            source: "source".into(),
            text: text.into(),
            status: CueStatus::Translated,
            flags: vec![],
        }
    }

    #[test]
    fn missing_file_means_no_cues_and_parents_are_created() {
        let path = TestPath::new("missing");
        let (_cp, cues) = Checkpoint::open(&path.0).unwrap();
        assert!(cues.is_empty());
    }

    #[test]
    fn cues_survive_reopening_and_finish_deletes_the_file() {
        let path = TestPath::new("reopen");
        let (mut cp, _) = Checkpoint::open(&path.0).unwrap();
        cp.append(&[cue(1, "یک"), cue(2, "دو")]).unwrap();
        drop(cp);

        let (mut cp, cues) = Checkpoint::open(&path.0).unwrap();
        assert_eq!(cues, [cue(1, "یک"), cue(2, "دو")]);
        cp.append(&[cue(3, "سه")]).unwrap();
        drop(cp);

        let (cp, cues) = Checkpoint::open(&path.0).unwrap();
        assert_eq!(cues.len(), 3);
        cp.finish().unwrap();
        assert!(!path.0.exists());
    }

    #[test]
    fn a_torn_last_line_is_dropped_and_does_not_eat_the_next_append() {
        let path = TestPath::new("torn");
        let (mut cp, _) = Checkpoint::open(&path.0).unwrap();
        cp.append(&[cue(1, "یک"), cue(2, "دو")]).unwrap();
        drop(cp);
        // cut inside the multi-byte text of the last cue, as a crash can
        let bytes = std::fs::read(&path.0).unwrap();
        let cut = bytes.len() - "دو".len() - 10;
        std::fs::write(&path.0, &bytes[..cut]).unwrap();

        let (mut cp, cues) = Checkpoint::open(&path.0).unwrap();
        assert_eq!(cues, [cue(1, "یک")]);
        cp.append(&[cue(2, "دو")]).unwrap();
        drop(cp);

        let (_cp, cues) = Checkpoint::open(&path.0).unwrap();
        assert_eq!(cues, [cue(1, "یک"), cue(2, "دو")]);
    }

    #[test]
    fn a_bad_line_in_the_middle_is_skipped() {
        let path = TestPath::new("middle");
        let (cp, _) = Checkpoint::open(&path.0).unwrap();
        drop(cp);
        let good = serde_json::to_string(&cue(5, "پنج")).unwrap();
        std::fs::write(&path.0, format!("{{not json\n{good}\n")).unwrap();
        let (_cp, cues) = Checkpoint::open(&path.0).unwrap();
        assert_eq!(cues, [cue(5, "پنج")]);
    }
}
