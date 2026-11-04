//! llm-worker library: guards + cleanup driver (pure Rust, unit-tested with a mock model) and the
//! llama.cpp chat backend. The binary (`main.rs`) is the process the app starts.

pub mod guards;
pub mod llama;
pub mod refine;
