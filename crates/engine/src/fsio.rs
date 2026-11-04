//! Small file helpers shared by everything that saves user data.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{Error, Result, io_err};

/// Attempts at the final rename. On Windows it fails for a moment while another program (a virus
/// scanner, the search indexer, a text editor) has the old file open.
const RENAME_ATTEMPTS: u32 = 6;
const RENAME_WAIT: Duration = Duration::from_millis(25);

/// Replace `path` with `bytes` so a reader or a crash sees the whole old file or the whole new one,
/// never part of either. Writes a temp file in the same folder, flushes it, then renames it over
/// `path`. Creates the parent folder if needed. On failure `path` is untouched and no temp file stays.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(io_err(dir))?;
    }
    let tmp = temp_path(path);
    let written = write_and_flush(&tmp, bytes).and_then(|()| rename_with_retry(&tmp, path));
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written.map_err(io_err(path))
}

/// Pretty-printed JSON through [`write_atomic`].
pub fn save_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let json = serde_json::to_vec_pretty(value).map_err(|e| Error::Json {
        path: path.display().to_string(),
        msg: e.to_string(),
    })?;
    write_atomic(path, &json)
}

/// Read a JSON file. A missing file is `Error::Io`, a corrupt one `Error::Json`.
pub fn load_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path).map_err(io_err(path))?;
    serde_json::from_str(&text).map_err(|e| Error::Json {
        path: path.display().to_string(),
        msg: e.to_string(),
    })
}

/// Unique within the machine: two threads or two app instances saving at once never share a temp file.
fn temp_path(path: &Path) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let unique = NEXT.fetch_add(1, Ordering::Relaxed);
    path.with_file_name(format!(".{name}.{}.{unique}.tmp", std::process::id()))
}

fn write_and_flush(tmp: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = File::create(tmp)?;
    file.write_all(bytes)?;
    file.sync_all()
}

fn rename_with_retry(from: &Path, to: &Path) -> std::io::Result<()> {
    let mut attempt = 1;
    loop {
        match std::fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(e) if attempt >= RENAME_ATTEMPTS || to.is_dir() => return Err(e),
            Err(_) => {
                attempt += 1;
                std::thread::sleep(RENAME_WAIT);
            }
        }
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
                "vipr-fsio-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            TestDir(dir)
        }

        fn entries(&self) -> Vec<String> {
            let mut names: Vec<String> = std::fs::read_dir(&self.0)
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();
            names
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn creates_then_replaces_and_leaves_no_temp_file() {
        let dir = TestDir::new("replace");
        let path = dir.0.join("a.json");
        write_atomic(&path, b"first").unwrap();
        write_atomic(&path, b"second, and longer than the first").unwrap();
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"second, and longer than the first"
        );
        write_atomic(&path, b"3").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"3");
        assert_eq!(dir.entries(), ["a.json"]);
    }

    #[test]
    fn creates_missing_parent_folders() {
        let dir = TestDir::new("parent");
        let path = dir.0.join("x").join("y").join("a.txt");
        write_atomic(&path, b"hello").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"hello");
    }

    #[test]
    fn failure_cleans_up() {
        let dir = TestDir::new("fail");
        // the target is a folder, so the rename cannot succeed
        let target = dir.0.join("taken");
        std::fs::create_dir(&target).unwrap();
        assert!(write_atomic(&target, b"data").is_err());
        assert_eq!(dir.entries(), ["taken"]);
    }

    #[test]
    fn json_round_trip_and_errors_are_told_apart() {
        let dir = TestDir::new("json");
        let path = dir.0.join("v.json");
        assert!(matches!(
            load_json::<Vec<u32>>(&path),
            Err(Error::Io { .. })
        ));
        save_json(&path, &vec![1u32, 2]).unwrap();
        assert_eq!(load_json::<Vec<u32>>(&path).unwrap(), [1, 2]);
        std::fs::write(&path, "{not json").unwrap();
        assert!(matches!(
            load_json::<Vec<u32>>(&path),
            Err(Error::Json { .. })
        ));
    }

    #[test]
    fn concurrent_writers_each_leave_a_complete_file() {
        let dir = TestDir::new("concurrent");
        let path = dir.0.join("a.txt");
        let bodies: Vec<Vec<u8>> = (0..4u8).map(|i| vec![b'a' + i; 4096]).collect();
        std::thread::scope(|s| {
            for body in &bodies {
                let path = &path;
                s.spawn(move || {
                    for _ in 0..5 {
                        // a rename can still lose to another writer on Windows; the file must stay whole
                        let _ = write_atomic(path, body);
                    }
                });
            }
        });
        let got = std::fs::read(&path).unwrap();
        assert!(bodies.contains(&got), "a mix of two writes");
        assert_eq!(dir.entries(), ["a.txt"]);
    }
}
