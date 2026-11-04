//! `detect` - print what GPU the engine would use.   `detect --json` for machine output.

fn main() {
    let d = gpu_detect::detect();
    if std::env::args().any(|a| a == "--json") {
        println!("{}", serde_json::to_string_pretty(&d).expect("serialize"));
        return;
    }
    println!("OS: {}", d.os);
    if d.gpus.is_empty() {
        println!("GPUs: none");
    }
    for (i, g) in d.gpus.iter().enumerate() {
        let mark = if d.recommended_gpu == Some(i) {
            "*"
        } else {
            " "
        };
        let vram = g
            .vram_mb
            .map(|m| format!("{m} MiB"))
            .unwrap_or_else(|| "VRAM unknown".into());
        println!(
            "{mark} [{i}] {} | {:?} | {vram}{} | via {:?}",
            g.name,
            g.vendor,
            if g.integrated { " | integrated" } else { "" },
            g.source
        );
        if let Some(c) = &g.compute_capability {
            println!(
                "      compute capability {c}, driver {}, CUDA {}",
                g.driver.as_deref().unwrap_or("?"),
                g.cuda_driver.as_deref().unwrap_or("?")
            );
        }
    }
    println!("Backend: {}  ({})", d.recommended.as_str(), d.reason);
    for n in &d.notes {
        println!("note: {n}");
    }
}
