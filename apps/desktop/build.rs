#[path = "../../crates/whispernow-domain/src/ipc/command_names.rs"]
mod command_names;

use command_names::COMMAND_NAMES;

fn main() {
    println!("cargo:rerun-if-changed=../../crates/whispernow-domain/src/ipc/command_names.rs");
    let attributes = tauri_build::Attributes::new()
        .app_manifest(tauri_build::AppManifest::new().commands(&COMMAND_NAMES))
        .windows_attributes(tauri_build::WindowsAttributes::new_without_app_manifest());
    embed_windows_manifest();
    tauri_build::try_build(attributes).expect("Tauri build configuration");
}

/// Links Tauri's default manifest (Common Controls v6) into every binary, tests included.
/// tauri-build 2.7.4 embeds it only in `[[bin]]` targets, and a test binary without it can't
/// start on Windows (`STATUS_ENTRYPOINT_NOT_FOUND`).
fn embed_windows_manifest() {
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        return;
    }
    let manifest =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("windows-app-manifest.xml");
    println!("cargo:rerun-if-changed={}", manifest.display());
    println!("cargo:rustc-link-arg=/MANIFEST:EMBED");
    println!("cargo:rustc-link-arg=/MANIFESTINPUT:{}", manifest.display());
}
