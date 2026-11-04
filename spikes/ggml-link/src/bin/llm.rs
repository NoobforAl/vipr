//! llama.cpp alone: initialises the backend and reports GPU offload support.
use llama_cpp_2::llama_backend::LlamaBackend;
fn main() {
    let b = LlamaBackend::init().expect("llama backend init");
    println!("llama.cpp gpu offload supported: {}", b.supports_gpu_offload());
}
