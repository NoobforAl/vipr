#!/bin/bash
# CUDA build test for both engines (separate binaries). Needs libclang 18 (see PLAN.md notes).
# /Zc:preprocessor is required by the CUDA 13 headers (otherwise: fatal error C1189 in cccl/preprocessor.h)
export GGML_NATIVE=OFF CMAKE_CUDA_ARCHITECTURES=86 CMAKE_CUDA_FLAGS="-Xcompiler=/Zc:preprocessor"
export LIBCLANG_PATH="$1"
cd "$(dirname "$0")"
cargo build --features cuda --bin asr 2>&1 | tail -25; echo "ASR_EXIT=${PIPESTATUS[0]}"
cargo build --features cuda --bin llm 2>&1 | tail -25; echo "LLM_EXIT=${PIPESTATUS[0]}"
