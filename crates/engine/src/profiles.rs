//! Named profiles: a prompt and glossary keywords, in `<config>/ViPr/profiles.json`.
//!
//! A corrupt file is moved to `.bak` with a visible notice, so user work is never silently reset.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::{Result, fsio, io_err};

/// Name of a profile created without a usable one.
const DEFAULT_NAME: &str = "New profile";

/// One glossary keyword. With `target`, it also fixes how the term is translated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlossaryTerm {
    pub source: String,
    #[serde(default)]
    pub target: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Profile {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub glossary: Vec<GlossaryTerm>,
}

/// The whole `profiles.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfilesFile {
    #[serde(default = "crate::project::version")]
    pub version: u32,
    #[serde(default)]
    pub profiles: Vec<Profile>,
}

/// In-memory profiles and the file they save to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileStore {
    pub path: PathBuf,
    pub profiles: Vec<Profile>,
    /// The file exists but could be neither read nor moved aside (say another program has it
    /// locked). Saving would destroy it, so [`ProfileStore::save`] refuses.
    pub save_blocked: bool,
}

/// Result of [`ProfileStore::load`]. It always yields a usable store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedProfiles {
    pub store: ProfileStore,
    /// What the user must be told, when the file was corrupt and moved to `.bak`, or when it could
    /// not be read at all.
    pub notice: Option<String>,
}

impl ProfileStore {
    /// `<config>/ViPr/profiles.json`.
    pub fn default_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(crate::APP_DIR_NAME)
            .join("profiles.json")
    }

    /// A missing file gives an empty store. A corrupt one is moved to `.bak` with a notice. If the
    /// file can't be read or moved, the store is empty, saving is blocked and the notice says so.
    pub fn load(path: &Path) -> LoadedProfiles {
        let store = |profiles, save_blocked| ProfileStore {
            path: path.to_path_buf(),
            profiles,
            save_blocked,
        };
        let name = file_name(path);
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == ErrorKind::NotFound => {
                return LoadedProfiles {
                    store: store(Vec::new(), false),
                    notice: None,
                };
            }
            Err(e) => {
                return LoadedProfiles {
                    store: store(Vec::new(), true),
                    notice: Some(format!(
                        "{name} could not be read ({e}), so your profiles are not available and \
                         changes to them won't be saved. Close whatever is using the file and \
                         restart ViPr."
                    )),
                };
            }
        };
        // Notepad and friends may add a byte order mark, which serde_json rejects.
        let json = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&bytes);
        let problem = match serde_json::from_slice::<ProfilesFile>(json) {
            Ok(file) => {
                return LoadedProfiles {
                    store: store(file.profiles, false),
                    notice: None,
                };
            }
            Err(e) => e,
        };
        let backup = backup_path(path);
        match std::fs::rename(path, &backup) {
            Ok(()) => LoadedProfiles {
                store: store(Vec::new(), false),
                notice: Some(format!(
                    "{name} was damaged ({problem}). It was kept as {}, and ViPr started with no \
                     profiles.",
                    backup.display()
                )),
            },
            Err(e) => LoadedProfiles {
                store: store(Vec::new(), true),
                notice: Some(format!(
                    "{name} is damaged ({problem}) and could not be moved aside ({e}). Your \
                     profiles are not available and changes to them won't be saved."
                )),
            },
        }
    }

    /// Atomic. Fails while saving is blocked (see [`ProfileStore::save_blocked`]).
    pub fn save(&self) -> Result<()> {
        if self.save_blocked {
            return Err(io_err(&self.path)(std::io::Error::other(
                "not saved, because the existing file could not be read",
            )));
        }
        fsio::save_json(
            &self.path,
            &ProfilesFile {
                version: crate::project::VERSION,
                profiles: self.profiles.clone(),
            },
        )
    }

    pub fn get(&self, id: &str) -> Option<&Profile> {
        self.profiles.iter().find(|p| p.id == id)
    }

    /// True when a profile other than `except` already has this name, ignoring case.
    pub fn name_taken(&self, name: &str, except: Option<&str>) -> bool {
        let name = name.trim().to_lowercase();
        self.profiles
            .iter()
            .any(|p| Some(p.id.as_str()) != except && p.name.trim().to_lowercase() == name)
    }

    /// Adds a new profile with a fresh id and returns that id. Does not save. An empty or taken
    /// name gets a number, since the selector shows profiles by name alone.
    pub fn create(&mut self, name: &str) -> String {
        let name = name.trim();
        let base = if name.is_empty() { DEFAULT_NAME } else { name };
        let mut unique = base.to_string();
        let mut n = 2;
        while self.name_taken(&unique, None) {
            unique = format!("{base} {n}");
            n += 1;
        }
        let id = self.fresh_id();
        self.profiles.push(Profile {
            id: id.clone(),
            name: unique,
            prompt: String::new(),
            glossary: Vec::new(),
        });
        id
    }

    /// Time-based rather than counted, so a deleted profile's id doesn't come back and match a
    /// `Settings::active_profile` or a saved snapshot that still names it.
    fn fresh_id(&self) -> String {
        let mut n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64);
        loop {
            let id = format!("p{n:x}");
            if self.get(&id).is_none() {
                return id;
            }
            n += 1;
        }
    }

    /// Replaces the profile with the same id. Returns false if there is none. Does not save.
    pub fn update(&mut self, profile: Profile) -> bool {
        match self.profiles.iter_mut().find(|p| p.id == profile.id) {
            Some(slot) => {
                *slot = profile;
                true
            }
            None => false,
        }
    }

    /// Returns false if there is no such profile. Does not save.
    pub fn remove(&mut self, id: &str) -> bool {
        let before = self.profiles.len();
        self.profiles.retain(|p| p.id != id);
        self.profiles.len() != before
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map_or_else(|| "The file".into(), |n| n.to_string_lossy().into_owned())
}

/// `<name>.bak`, or `<name>.2.bak` and so on when that is taken: an older backup may be the only
/// copy of someone's profiles, and a rename would replace it.
fn backup_path(path: &Path) -> PathBuf {
    let name = file_name(path);
    let mut candidate = path.with_file_name(format!("{name}.bak"));
    let mut n = 2;
    while candidate.exists() {
        candidate = path.with_file_name(format!("{name}.{n}.bak"));
        n += 1;
    }
    candidate
}

/// Between terms: line breaks, including the rarer ones a paste can bring in, and the comma as
/// Latin, Arabic and CJK text write it.
fn splits_terms(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r' | ',' | '\u{60C}' | '\u{FF0C}' | '\u{3001}' | '\u{85}' | '\u{1C}'
            ..='\u{1E}' | '\u{2028}' | '\u{2029}'
    )
}

/// Whitespace and invisible direction marks at the ends of a term. ZWNJ (U+200C) is not here,
/// since Persian words use it, and control characters go wherever they are.
fn tidy_term(text: &str) -> String {
    let inner: String = text.chars().filter(|c| !c.is_control()).collect();
    inner
        .trim_matches(|c: char| {
            c.is_whitespace()
                || matches!(c, '\u{200E}' | '\u{200F}' | '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}' | '\u{FEFF}')
        })
        .to_string()
}

/// Parses the keywords box: one term per line or comma separated, each `source` or `source=target`.
/// Only the first `=` splits. A term with no source is dropped. A term repeated under the same
/// source, ignoring case, is dropped too, except that a later target fills in a missing one.
pub fn parse_glossary(text: &str) -> Vec<GlossaryTerm> {
    let mut terms: Vec<GlossaryTerm> = Vec::new();
    for part in text.split(splits_terms) {
        let (source, target) = match part.split_once('=') {
            Some((source, target)) => (tidy_term(source), tidy_term(target)),
            None => (tidy_term(part), String::new()),
        };
        if source.is_empty() {
            continue;
        }
        let target = (!target.is_empty()).then_some(target);
        let key = source.to_lowercase();
        match terms.iter_mut().find(|t| t.source.to_lowercase() == key) {
            Some(known) => {
                if known.target.is_none() {
                    known.target = target;
                }
            }
            None => terms.push(GlossaryTerm { source, target }),
        }
    }
    terms
}

/// Formats terms back into the text [`parse_glossary`] reads, one per line.
pub fn format_glossary(terms: &[GlossaryTerm]) -> String {
    let lines: Vec<String> = terms
        .iter()
        .map(|term| match &term.target {
            Some(target) => format!("{}={target}", term.source),
            None => term.source.clone(),
        })
        .collect();
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh folder under the system temp dir, removed when dropped.
    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> TestDir {
            let dir = std::env::temp_dir().join(format!(
                "vipr-profiles-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            TestDir(dir)
        }

        fn file(&self) -> PathBuf {
            self.0.join("profiles.json")
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

    fn term(source: &str, target: Option<&str>) -> GlossaryTerm {
        GlossaryTerm {
            source: source.into(),
            target: target.map(Into::into),
        }
    }

    #[test]
    fn a_missing_file_gives_an_empty_store_and_no_notice() {
        let dir = TestDir::new("missing");
        let loaded = ProfileStore::load(&dir.file());
        assert!(loaded.store.profiles.is_empty());
        assert!(!loaded.store.save_blocked);
        assert_eq!(loaded.notice, None);
        assert!(dir.entries().is_empty());
    }

    #[test]
    fn round_trip_keeps_everything() {
        let dir = TestDir::new("round");
        let mut store = ProfileStore::load(&dir.file()).store;
        let id = store.create("Medical");
        let mut profile = store.get(&id).unwrap().clone();
        profile.prompt = "ترجمهٔ رسمی\nsecond line".into();
        profile.glossary = vec![term("API", Some("رابط")), term("Kubernetes", None)];
        assert!(store.update(profile));
        store.save().unwrap();

        let back = ProfileStore::load(&dir.file());
        assert_eq!(back.notice, None);
        assert_eq!(back.store, store);
    }

    #[test]
    fn a_corrupt_file_moves_to_bak_with_a_notice_and_the_store_is_usable() {
        let dir = TestDir::new("corrupt");
        std::fs::write(dir.file(), "{not json").unwrap();

        let loaded = ProfileStore::load(&dir.file());
        assert!(loaded.store.profiles.is_empty());
        assert!(!loaded.store.save_blocked);
        let notice = loaded.notice.as_deref().unwrap();
        assert!(notice.contains("profiles.json.bak"), "{notice}");
        assert_eq!(dir.entries(), ["profiles.json.bak"]);
        assert_eq!(
            std::fs::read_to_string(dir.0.join("profiles.json.bak")).unwrap(),
            "{not json"
        );

        let mut store = loaded.store;
        store.create("Fresh");
        store.save().unwrap();
        assert_eq!(ProfileStore::load(&dir.file()).store.profiles.len(), 1);
    }

    #[test]
    fn valid_json_of_the_wrong_shape_and_an_empty_file_count_as_corrupt() {
        for body in ["[1, 2]", r#"{"profiles":[{"id":"x"}]}"#, ""] {
            let dir = TestDir::new("shape");
            std::fs::write(dir.file(), body).unwrap();
            let loaded = ProfileStore::load(&dir.file());
            assert!(loaded.notice.is_some(), "{body:?}");
            assert_eq!(dir.entries(), ["profiles.json.bak"], "{body:?}");
        }
    }

    #[test]
    fn invalid_utf8_counts_as_corrupt() {
        let dir = TestDir::new("utf8");
        std::fs::write(dir.file(), [b'{', 0xFF, 0xFE, b'}']).unwrap();
        assert!(ProfileStore::load(&dir.file()).notice.is_some());
        assert_eq!(dir.entries(), ["profiles.json.bak"]);
    }

    #[test]
    fn an_older_backup_is_never_replaced() {
        let dir = TestDir::new("older-bak");
        std::fs::write(dir.0.join("profiles.json.bak"), "first").unwrap();
        std::fs::write(dir.file(), "second").unwrap();
        let notice = ProfileStore::load(&dir.file()).notice.unwrap();
        assert!(notice.contains("profiles.json.2.bak"), "{notice}");
        assert_eq!(
            std::fs::read_to_string(dir.0.join("profiles.json.bak")).unwrap(),
            "first"
        );
        assert_eq!(
            std::fs::read_to_string(dir.0.join("profiles.json.2.bak")).unwrap(),
            "second"
        );
    }

    #[test]
    fn an_unreadable_file_blocks_saving_and_says_so() {
        let dir = TestDir::new("unreadable");
        // a folder at the file's path: reading fails with something other than "not found"
        std::fs::create_dir(dir.file()).unwrap();
        let loaded = ProfileStore::load(&dir.file());
        assert!(loaded.notice.is_some());
        assert!(loaded.store.save_blocked);
        assert!(loaded.store.save().is_err());
        assert!(dir.file().is_dir(), "left in place");
    }

    #[test]
    fn a_byte_order_mark_is_not_corruption() {
        let dir = TestDir::new("bom");
        std::fs::write(
            dir.file(),
            "\u{FEFF}{\"profiles\":[{\"id\":\"a\",\"name\":\"A\"}]}",
        )
        .unwrap();
        let loaded = ProfileStore::load(&dir.file());
        assert_eq!(loaded.notice, None);
        assert_eq!(loaded.store.profiles[0].name, "A");
        assert_eq!(loaded.store.profiles[0].prompt, "");
    }

    #[test]
    fn created_ids_differ_even_within_one_millisecond_and_after_a_delete() {
        let dir = TestDir::new("ids");
        let mut store = ProfileStore::load(&dir.file()).store;
        let a = store.create("A");
        let b = store.create("B");
        assert_ne!(a, b);
        assert!(store.remove(&b));
        assert!(!store.remove(&b));
        assert_eq!(store.get(&b), None);
        assert_ne!(store.create("C"), a);
    }

    #[test]
    fn create_gives_every_profile_a_distinct_name() {
        let dir = TestDir::new("names");
        let mut store = ProfileStore::load(&dir.file()).store;
        let names: Vec<String> = ["  ", "", "Work", "work", "  Work "]
            .iter()
            .map(|name| {
                let id = store.create(name);
                store.get(&id).unwrap().name.clone()
            })
            .collect();
        assert_eq!(
            names,
            ["New profile", "New profile 2", "Work", "work 2", "Work 3"]
        );
    }

    #[test]
    fn name_taken_can_ignore_the_profile_being_renamed() {
        let dir = TestDir::new("taken");
        let mut store = ProfileStore::load(&dir.file()).store;
        let a = store.create("Work");
        assert!(store.name_taken(" WORK ", None));
        assert!(!store.name_taken("work", Some(&a)));
        assert!(!store.name_taken("Home", None));
    }

    #[test]
    fn update_of_an_unknown_id_changes_nothing() {
        let dir = TestDir::new("update");
        let mut store = ProfileStore::load(&dir.file()).store;
        store.create("A");
        let before = store.clone();
        let ghost = Profile {
            id: "ghost".into(),
            name: "Ghost".into(),
            prompt: String::new(),
            glossary: Vec::new(),
        };
        assert!(!store.update(ghost));
        assert_eq!(store, before);
    }

    #[test]
    fn glossary_splits_on_lines_and_commas_with_optional_targets() {
        assert_eq!(
            parse_glossary("  API = رابط \r\nKubernetes, GPU=پردازنده\n\n ,  "),
            [
                term("API", Some("رابط")),
                term("Kubernetes", None),
                term("GPU", Some("پردازنده")),
            ]
        );
    }

    #[test]
    fn glossary_splits_on_other_scripts_commas_and_pasted_separators() {
        assert_eq!(
            parse_glossary("a،b，c、d\u{2029}e\u{85}f\u{1C}g"),
            ["a", "b", "c", "d", "e", "f", "g"].map(|s| term(s, None))
        );
    }

    #[test]
    fn glossary_only_the_first_equals_splits_and_empty_sides_are_handled() {
        assert_eq!(
            parse_glossary("a=b=c\nd=\n=e\n="),
            [term("a", Some("b=c")), term("d", None)]
        );
    }

    #[test]
    fn glossary_repeats_collapse_and_a_later_target_fills_a_missing_one() {
        assert_eq!(
            parse_glossary("API\napi=رابط\nAPI=other\nGPU=x\ngpu"),
            [term("API", Some("رابط")), term("GPU", Some("x"))]
        );
    }

    #[test]
    fn glossary_terms_lose_invisible_edges_but_keep_zwnj_inside() {
        assert_eq!(
            parse_glossary("\u{200F}می\u{200C}خواهم\u{200E}=\u{2067}x\u{2069}\t"),
            [term("می\u{200C}خواهم", Some("x"))]
        );
    }

    #[test]
    fn glossary_text_round_trips() {
        let terms = vec![
            term("API", Some("رابط")),
            term("Kubernetes", None),
            term("a", Some("b=c")),
        ];
        let text = format_glossary(&terms);
        assert_eq!(text, "API=رابط\nKubernetes\na=b=c");
        assert_eq!(parse_glossary(&text), terms);
        assert_eq!(format_glossary(&[]), "");
        assert!(parse_glossary("").is_empty());
    }
}
