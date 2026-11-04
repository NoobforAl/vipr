# Build everything for one GPU backend and assemble a ready-to-run folder: dist/<backend>/
#
#   scripts\build-release.ps1 -Backend cuda      # NVIDIA (CUDA 13 toolkit + MSVC needed)
#   scripts\build-release.ps1 -Backend vulkan    # AMD / Intel / any Vulkan GPU
#   scripts\build-release.ps1 -Backend cpu       # no GPU
#   scripts\build-release.ps1 -Backend cuda -SkipBuild   # only (re)assemble from existing builds
#
# If Windows refuses to run scripts:  powershell -ExecutionPolicy Bypass -File scripts\build-release.ps1 -Backend cuda
#
# The app finds its workers next to itself, so the folder can be moved or zipped as it is.
# Models are NOT included (they are downloaded on first start); models.json, prompts/ and
# glossary.txt are.
param(
    [ValidateSet('cuda', 'vulkan', 'cpu')][string]$Backend = 'cuda',
    [string]$Out = 'dist',
    [switch]$SkipBuild,
    [switch]$Zip,
    # Optional: an ffmpeg.exe to ship next to vipr.exe so AVI, WMV, FLV, MPEG-TS, AMR, AC3 ... also work.
    # Use an LGPL build (no GPL parts) if you distribute it, and ship its licence text with it.
    [string]$FfmpegPath = ''
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $SkipBuild) {
    if (-not (Test-Path 'tools\libclang18\clang\native\libclang.dll')) { & "$PSScriptRoot\setup-build-tools.ps1" }
    # Several GPU generations in one build (the dev default in .cargo/config.toml is only sm_86).
    if ($Backend -eq 'cuda') { $env:CMAKE_CUDA_ARCHITECTURES = '75;80;86;89' }
    # Plain engines: no native-code GPU libraries.
    cargo build --release -p vipr -p engine-cli -p gpu-detect -p diar-worker
    if ($LASTEXITCODE -ne 0) { throw 'build failed (app / cli / diar-worker)' }
    # whisper.cpp and llama.cpp each bring their own ggml, so they are separate executables.
    $features = if ($Backend -eq 'cpu') { @() } else { @('--features', $Backend) }
    cargo build --release -p asr-worker -p llm-worker @features
    if ($LASTEXITCODE -ne 0) { throw 'build failed (asr-worker / llm-worker)' }
}

# Find each executable in the usual build folders (agents / developers may use separate target dirs).
function Find-Exe([string]$name) {
    foreach ($dir in 'target', 'target-diar', 'target-llm') {
        $p = Join-Path $root "$dir\release\$name.exe"
        if (Test-Path $p) { return $p }
    }
    return $null
}

$dest = Join-Path $root "$Out\$Backend"
if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
New-Item -ItemType Directory -Force $dest | Out-Null

$missing = @()
foreach ($name in 'vipr', 'engine-cli', 'detect', 'asr-worker', 'diar-worker', 'llm-worker') {
    $src = Find-Exe $name
    if ($src) { Copy-Item $src $dest; "  + $name.exe" } else { $missing += $name }
}
if ($FfmpegPath) {
    if (-not (Test-Path $FfmpegPath)) { throw "ffmpeg not found: $FfmpegPath" }
    Copy-Item $FfmpegPath (Join-Path $dest (Split-Path $FfmpegPath -Leaf)); '  + ffmpeg (extra formats)'
}
Copy-Item (Join-Path $root 'models.json') $dest
Copy-Item (Join-Path $root 'prompts') (Join-Path $dest 'prompts') -Recurse
if (Test-Path (Join-Path $root 'glossary.txt')) { Copy-Item (Join-Path $root 'glossary.txt') $dest }

if ($missing.Count -gt 0) { Write-Warning ("not found (build them first): " + ($missing -join ', ')) }
"Assembled $dest"

if ($Zip) {
    $zipPath = Join-Path $root "$Out\vipr-$Backend.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath }
    Compress-Archive -Path "$dest\*" -DestinationPath $zipPath
    "Zip: $zipPath"
}

