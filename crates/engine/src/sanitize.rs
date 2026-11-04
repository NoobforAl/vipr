//! Making untrusted text safe to put in a prompt.
//!
//! Stub from the foundation step; agent A fills in the body. `llama.rs:84` tokenizes the finished
//! prompt with special-token parsing on, so a literal `<|im_end|>` in a subtitle could become a real
//! control token.

/// Strips special-token sequences such as `<|im_end|>`. Runs before every prompt, including refine.
pub fn sanitize_untrusted(_text: &str) -> String {
    todo!("agent A")
}
