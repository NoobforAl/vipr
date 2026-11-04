//! whisper.cpp alone: prints the ggml features it was built with.
fn main() {
    println!("whisper.cpp: {}", whisper_rs::print_system_info());
}
