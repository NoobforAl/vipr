//! llm-worker library: guards + cleanup driver (pure Rust, unit-tested with a mock model) and the
//! llama.cpp chat backend. The binary (`main.rs`) is the process the app starts.

pub mod batch;
pub mod chapters;
pub mod checkpoint;
pub mod compare;
pub mod guards;
pub mod llama;
pub mod refine;
pub mod translate;
