//! Optional helper programs that ViPr can download for the user (so nothing has to be installed by hand).
//!
//! Today that is **ffmpeg**, used only for rare formats (AVI, WMV, FLV, MPEG-TS, AMR, AC3, ...). It is downloaded
//! on request into ViPr's own folder (`%LOCALAPPDATA%\ViPr\tools\ffmpeg` on Windows), checked against a pinned
//! SHA-256, and unpacked there. ViPr runs it as a separate program, it is never linked into ViPr.

use std::fs;
use std::path::{Path, PathBuf};

use crate::download::ModelStore;
use crate::{Error, Result, io_err};

/// A pinned download of a helper program.
#[derive(Debug, Clone, Copy)]
pub struct ToolDownload {
    pub name: &'static str,
    pub version: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size_bytes: u64,
    /// Where to find the program inside the archive (path suffix).
    pub exe_in_archive: &'static str,
    pub licence: &'static str,
}

/// ffmpeg "essentials" build from the official Gyan.dev releases (the same builds winget installs for
/// `Gyan.FFmpeg`). Only Windows x86-64 has a pinned download; elsewhere use the system package manager.
pub fn ffmpeg_download() -> Option<ToolDownload> {
    if cfg!(all(windows, target_arch = "x86_64")) {
        Some(ToolDownload {
            name: "ffmpeg",
            version: "9.0.2",
            url: "https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.7z",
            sha256: "4705843ccaaf54257c16ad90f3e952ece33c17df964ecf7bfdbb0f49c7171077",
            size_bytes: 35_400_000,
            exe_in_archive: "/bin/ffmpeg.exe",
            licence: "GPL v3 (ffmpeg build by gyan.dev); it runs as a separate program",
        })
    } else {
        None
    }
}

/// Folder for downloaded helpers: `VIPR_TOOLS_DIR`, else `<per-user data folder>/ViPr/tools`.
pub fn tools_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("VIPR_TOOLS_DIR") {
        return PathBuf::from(d);
    }
    dirs::data_local_dir().unwrap_or_else(|| PathBuf::from(".")).join(crate::APP_DIR_NAME).join("tools")
}

fn ffmpeg_dir() -> PathBuf {
    tools_dir().join("ffmpeg")
}

fn ffmpeg_exe_name() -> &'static str {
    if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" }
}

/// The ffmpeg that ViPr downloaded itself, if it is there.
pub fn installed_ffmpeg() -> Option<PathBuf> {
    let p = ffmpeg_dir().join(ffmpeg_exe_name());
    p.is_file().then_some(p)
}

/// Delete the ffmpeg that ViPr downloaded.
pub fn remove_installed_ffmpeg() -> Result<()> {
    let dir = ffmpeg_dir();
    if dir.exists() {
        fs::remove_dir_all(&dir).map_err(io_err(&dir))?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub enum InstallStep {
    Downloading { done: u64, total: u64 },
    Unpacking,
}

/// Download, verify and unpack ffmpeg. Returns the path of `ffmpeg.exe`.
pub fn install_ffmpeg(progress: &mut dyn FnMut(InstallStep)) -> Result<PathBuf> {
    let tool = ffmpeg_download().ok_or_else(|| {
        Error::Http("ViPr cannot download ffmpeg for this system; install it with your package manager (for example `brew install ffmpeg` or `apt install ffmpeg`)".into())
    })?;
    let dir = ffmpeg_dir();
    fs::create_dir_all(&dir).map_err(io_err(&dir))?;
    let archive = dir.join(format!("ffmpeg-{}.7z", tool.version));

    let store = ModelStore::new(tools_dir());
    store.fetch_file("ffmpeg", tool.url, &archive, 0, Some(tool.sha256), None, &mut |done, total| {
        progress(InstallStep::Downloading { done, total: if total > 0 { total } else { tool.size_bytes } });
    })?;

    progress(InstallStep::Unpacking);
    let exe = unpack(&archive, &dir, tool.exe_in_archive)?;
    let _ = fs::remove_file(&archive); // the unpacked program is all we need
    Ok(exe)
}

/// Pull `ffmpeg.exe` (and the licence text) out of the 7z archive.
fn unpack(archive: &Path, dir: &Path, exe_suffix: &str) -> Result<PathBuf> {
    let bad = |msg: String| Error::Audio(format!("could not unpack ffmpeg: {msg}"));
    let mut reader = sevenz_rust2::ArchiveReader::open(archive, sevenz_rust2::Password::empty()).map_err(|e| bad(e.to_string()))?;
    let exe_path = dir.join(ffmpeg_exe_name());
    let part = dir.join(format!("{}.part", ffmpeg_exe_name()));
    let mut found = false;
    reader
        .for_each_entries(|entry, data| {
            let name = entry.name().replace('\\', "/");
            if entry.is_directory() {
                return Ok(true);
            }
            if name.ends_with(exe_suffix) {
                let mut out = fs::File::create(&part)?;
                std::io::copy(data, &mut out)?;
                found = true;
            } else if name.ends_with("/LICENSE") || name.ends_with("/README.txt") {
                let mut out = fs::File::create(dir.join(Path::new(&name).file_name().unwrap_or_default()))?;
                std::io::copy(data, &mut out)?;
            }
            Ok(true)
        })
        .map_err(|e| bad(e.to_string()))?;
    if !found {
        let _ = fs::remove_file(&part);
        return Err(bad("the archive does not contain ffmpeg.exe".into()));
    }
    let _ = fs::remove_file(&exe_path);
    fs::rename(&part, &exe_path).map_err(io_err(&exe_path))?;
    Ok(exe_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_is_pinned_with_a_hash() {
        if let Some(t) = ffmpeg_download() {
            assert_eq!(t.sha256.len(), 64);
            assert!(t.url.starts_with("https://"));
            assert!(t.exe_in_archive.ends_with("ffmpeg.exe"));
        }
    }

    #[test]
    fn tools_dir_can_be_overridden() {
        unsafe { std::env::set_var("VIPR_TOOLS_DIR", "somewhere-else") };
        assert_eq!(tools_dir(), PathBuf::from("somewhere-else"));
        unsafe { std::env::remove_var("VIPR_TOOLS_DIR") };
    }
}
