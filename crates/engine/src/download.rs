//! Model store + resumable, hash-checked downloader.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::registry::{ModelEntry, Source};
use crate::{Error, Result, io_err};

/// Where downloaded models live: `<dir>/<model id>/<file>`.
#[derive(Debug, Clone)]
pub struct ModelStore {
    dir: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Missing,
    /// A `.part` file of this many bytes exists (interrupted download).
    Partial(u64),
    /// File present with the expected size (hash not re-checked).
    Present,
    /// A local-source file that does not exist.
    LocalMissing,
}

impl ModelStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        ModelStore { dir: dir.into() }
    }

    /// `MT_MODELS_DIR` if set, else the per-user data folder (`%LOCALAPPDATA%\ViPr\models` on Windows).
    pub fn default_dir() -> PathBuf {
        if let Some(d) = std::env::var_os("MT_MODELS_DIR") {
            return PathBuf::from(d);
        }
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(crate::APP_DIR_NAME)
            .join("models")
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn path_for(&self, e: &ModelEntry) -> PathBuf {
        match &e.source {
            Source::Local { path } => PathBuf::from(path),
            _ => self.dir.join(&e.id).join(e.file_name()),
        }
    }

    fn part_path(&self, e: &ModelEntry) -> PathBuf {
        let mut p = self.path_for(e).into_os_string();
        p.push(".part");
        PathBuf::from(p)
    }

    pub fn status(&self, e: &ModelEntry) -> Status {
        let path = self.path_for(e);
        if matches!(e.source, Source::Local { .. }) {
            return if path.is_file() {
                Status::Present
            } else {
                Status::LocalMissing
            };
        }
        match fs::metadata(&path) {
            Ok(m) if m.is_file() && (e.size_bytes == 0 || m.len() == e.size_bytes) => {
                Status::Present
            }
            _ => match fs::metadata(self.part_path(e)) {
                Ok(m) => Status::Partial(m.len()),
                Err(_) => Status::Missing,
            },
        }
    }

    /// Hash the file on disk. `Ok(None)` if the entry has no hash to compare with.
    pub fn verify(&self, e: &ModelEntry) -> Result<Option<bool>> {
        let Some(expected) = &e.sha256 else {
            return Ok(None);
        };
        let actual = sha256_file(&self.path_for(e))?;
        Ok(Some(&actual == expected))
    }

    /// Download (or resume) a model and check its hash. Returns the final path.
    /// `progress(done, total)` is called about every 256 KiB; `total` is 0 when unknown.
    /// `hf_token` is only sent to huggingface.co.
    pub fn fetch(
        &self,
        e: &ModelEntry,
        hf_token: Option<&str>,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<PathBuf> {
        let url = e
            .download_url()
            .ok_or_else(|| Error::NotDownloadable(e.id.clone()))?;
        self.fetch_file(
            &e.id,
            &url,
            &self.path_for(e),
            e.size_bytes,
            e.sha256.as_deref(),
            hf_token,
            progress,
        )
    }

    /// Download any file: resumable (`<path>.part` + Range), size and SHA-256 checked, moved into place when complete.
    /// `size_bytes` / `sha256` may be 0 / `None` when unknown (then nothing is verified).
    #[allow(clippy::too_many_arguments)]
    pub fn fetch_file(
        &self,
        id: &str,
        url: &str,
        final_path: &Path,
        size_bytes: u64,
        sha256: Option<&str>,
        hf_token: Option<&str>,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<PathBuf> {
        let final_path = final_path.to_path_buf();
        let part = {
            let mut p = final_path.clone().into_os_string();
            p.push(".part");
            PathBuf::from(p)
        };
        fs::create_dir_all(final_path.parent().unwrap_or(Path::new(".")))
            .map_err(io_err(&final_path))?;

        // already complete and correct?
        if let Ok(meta) = fs::metadata(&final_path)
            && meta.is_file()
            && (size_bytes == 0 || meta.len() == size_bytes)
        {
            match sha256 {
                Some(expected) if sha256_file(&final_path)? != expected => {}
                _ => return Ok(final_path),
            }
        }
        let _ = fs::remove_file(&final_path); // wrong size / hash: start again

        let client = reqwest::blocking::Client::builder()
            .user_agent(concat!("vipr/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(30))
            .timeout(None::<Duration>)
            .build()
            .map_err(|x| Error::Http(x.to_string()))?;

        let mut have = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        let mut attempt = 0;
        loop {
            attempt += 1;
            let mut req = client.get(url);
            if let Some(t) = hf_token.filter(|_| url.starts_with("https://huggingface.co/")) {
                req = req.bearer_auth(t);
            }
            if have > 0 {
                req = req.header("Range", format!("bytes={have}-"));
            }
            let mut resp = req.send().map_err(|x| Error::Http(x.to_string()))?;
            let status = resp.status();
            if status.as_u16() == 416 && attempt == 1 {
                // our .part is not a valid prefix (or already complete): restart cleanly
                let _ = fs::remove_file(&part);
                have = 0;
                continue;
            }
            if status.as_u16() == 401 || status.as_u16() == 403 {
                return Err(Error::Http(format!(
                    "{status} from {url}: this download needs a Hugging Face token (set HF_TOKEN) and accepted terms on its page"
                )));
            }
            if !status.is_success() {
                return Err(Error::Http(format!("{status} from {url}")));
            }
            let resumed = status.as_u16() == 206;
            if !resumed {
                have = 0; // server ignored the Range header: it sends the whole file
            }
            let total = if size_bytes > 0 {
                size_bytes
            } else {
                have + resp.content_length().unwrap_or(0)
            };
            let mut out = if resumed {
                OpenOptions::new().append(true).open(&part)
            } else {
                File::create(&part)
            }
            .map_err(io_err(&part))?;

            let mut buf = vec![0u8; 256 * 1024];
            let mut done = have;
            progress(done, total);
            loop {
                let n = resp.read(&mut buf).map_err(|x| {
                    Error::Http(format!("connection lost: {x} (run again to resume)"))
                })?;
                if n == 0 {
                    break;
                }
                out.write_all(&buf[..n]).map_err(io_err(&part))?;
                done += n as u64;
                progress(done, total);
            }
            out.flush().map_err(io_err(&part))?;
            break;
        }

        let size = fs::metadata(&part).map_err(io_err(&part))?.len();
        if size_bytes > 0 && size != size_bytes {
            return Err(Error::Http(format!(
                "incomplete download: got {size} of {size_bytes} bytes (run again to resume)"
            )));
        }
        if let Some(expected) = sha256 {
            let actual = sha256_file(&part)?;
            if actual != expected {
                let _ = fs::remove_file(&part);
                return Err(Error::HashMismatch {
                    id: id.to_string(),
                    expected: expected.to_string(),
                    actual,
                });
            }
        }
        fs::rename(&part, &final_path).map_err(io_err(&final_path))?;
        Ok(final_path)
    }
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = File::open(path).map_err(io_err(path))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf).map_err(io_err(path))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Format, Role};
    use gpu_detect::Backend;

    fn entry(path: &Path) -> ModelEntry {
        ModelEntry {
            id: "t".into(),
            name: "t".into(),
            roles: vec![Role::Llm],
            format: Format::Gguf,
            source: Source::Local {
                path: path.display().to_string(),
            },
            filename: None,
            size_bytes: 0,
            sha256: None,
            license: String::new(),
            languages: vec![],
            min_vram_mb: 0,
            backends: vec![Backend::Cpu],
            presets: vec![],
            gated: false,
            notes: String::new(),
        }
    }

    #[test]
    fn sha256_of_known_content() {
        let p = std::env::temp_dir().join("mt_sha_test.txt");
        fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_file(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn local_models_are_never_downloaded() {
        let store = ModelStore::new(std::env::temp_dir().join("mt_store_test"));
        let e = entry(Path::new("does-not-exist.gguf"));
        assert_eq!(store.status(&e), Status::LocalMissing);
        assert!(matches!(
            store.fetch(&e, None, &mut |_, _| {}),
            Err(Error::NotDownloadable(_))
        ));
    }
}
