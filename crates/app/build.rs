//! Embeds the ViPr icon and version info into the Windows executable.

fn main() {
    println!("cargo:rerun-if-changed=../../assets/icon/vipr.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../../assets/icon/vipr.ico");
        res.set("ProductName", "ViPr");
        res.set("FileDescription", "ViPr - video and audio to text");
        res.set("CompanyName", "ViPr");
        if let Err(e) = res.compile() {
            // the app works without the embedded icon; do not fail the build over it
            println!("cargo:warning=could not embed the icon: {e}");
        }
    }
}
