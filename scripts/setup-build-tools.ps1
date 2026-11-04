# One-time setup: libclang 18 for bindgen (libclang 22 from LLVM / Visual Studio breaks whisper-rs).
# Needs uv. Result: tools/libclang18/clang/native/libclang.dll  (about 80 MB, not committed)
$root = Split-Path -Parent $PSScriptRoot
uv pip install --target "$root\tools\libclang18" "libclang==18.1.1"
if (Test-Path "$root\tools\libclang18\clang\native\libclang.dll") { "ok" } else { throw "libclang.dll not found" }
