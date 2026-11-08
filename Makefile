# ViPr developer commands. Run `make` to list them.
#
# Recipes are single plain commands, so they work the same from PowerShell, cmd and Git Bash.
# Needs: cargo, uv (for ruff and taplo). `make workers` / `make package` also need MSVC, CMake and,
# for BACKEND=cuda, the CUDA toolkit (run `make setup` once for libclang).

# GPU backend for the worker builds: cuda | vulkan | cpu
BACKEND ?= cuda

PY_DIRS := scripts validation
TOML_FILES := Cargo.toml .cargo/config.toml scripts/pyproject.toml $(wildcard crates/*/Cargo.toml)
RUFF := uvx ruff
TAPLO := uvx taplo
# cpu is the workers' default build, so it has no feature flag.
WORKER_FEATURES := $(if $(filter cpu,$(BACKEND)),,--features $(BACKEND))

.DEFAULT_GOAL := help
.PHONY: help dev run build workers package setup fix lint test bench check models gpu clean

define HELP_TEXT
ViPr commands

Run
  make dev       Run the app in debug mode, with backtraces on a crash
  make run       Run the app in release mode (slower build, faster app)

Code quality
  make fix       Apply lint fixes and format all Rust, Python and TOML files
  make lint      Report format and lint problems, changing nothing
  make test      Run all Rust tests
  make bench     Run the engine benchmarks (resampling, subtitle parsing)
  make check     lint + test, for use before a commit

Build
  make build     Build the app and the CLI tools (debug)
  make workers   Build the release workers that transcribing needs (BACKEND=$(BACKEND))
  make package   Assemble dist/<backend>/, replacing it (BACKEND=$(BACKEND))
  make setup     One-time setup: download libclang 18 for the worker builds
  make clean     Delete target/ (the next worker build takes minutes)

Tools
  make models    List the models in the registry
  make gpu       Show the detected GPU

Choose a backend with BACKEND=cuda|vulkan|cpu, for example: make workers BACKEND=vulkan
endef

# `cd .` does nothing; it stops make adding "Nothing to be done for 'help'".
help:
	$(info $(HELP_TEXT))
	@cd .

# --- Run -------------------------------------------------------------------------------------

# The app finds workers in target/release, so run `make workers` once to be able to transcribe.
dev: export RUST_BACKTRACE := 1
dev:
	cargo run -p vipr

run:
	cargo run --release -p vipr

# --- Code quality ----------------------------------------------------------------------------

# Clippy goes first because its edits may need reformatting. Python is formatted twice: the first
# pass settles lints ruff cannot fix by itself (`a; b` on one line), the second tidies ruff's edits.
# The last check fails on anything that still has to be fixed by hand.
fix:
	cargo clippy --fix --allow-dirty --allow-staged --workspace --all-targets
	cargo fmt --all
	$(TAPLO) fmt $(TOML_FILES)
	$(RUFF) format $(PY_DIRS)
	$(RUFF) check --fix-only $(PY_DIRS)
	$(RUFF) format $(PY_DIRS)
	$(RUFF) check $(PY_DIRS)

lint:
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets -- -D warnings
	$(RUFF) format --check $(PY_DIRS)
	$(RUFF) check $(PY_DIRS)
	$(TAPLO) fmt --check $(TOML_FILES)

test:
	cargo test --workspace

bench:
	cargo bench -p engine

check: lint test

# --- Build -----------------------------------------------------------------------------------

build:
	cargo build

workers:
	cargo build --release -p diar-worker
	cargo build --release -p asr-worker -p llm-worker $(WORKER_FEATURES)

package:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-release.ps1 -Backend $(BACKEND)

setup:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup-build-tools.ps1

clean:
	cargo clean

# --- Tools -----------------------------------------------------------------------------------

models:
	cargo run -q -p engine-cli -- models list

gpu:
	cargo run -q -p gpu-detect --bin detect
