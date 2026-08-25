#[path = "../../crates/whispernow-api/src/command_names.rs"]
mod command_names;

use command_names::COMMAND_NAMES;

fn main() {
    println!("cargo:rerun-if-changed=../../crates/whispernow-api/src/command_names.rs");
    // Exposes the exact list handed to Tauri so the command catalog test can
    // check what the build generated permissions for.
    println!(
        "cargo:rustc-env=WHISPERNOW_TAURI_MANIFEST_COMMANDS={}",
        COMMAND_NAMES.join(",")
    );
    let attributes = tauri_build::Attributes::new()
        .app_manifest(tauri_build::AppManifest::new().commands(&COMMAND_NAMES));
    tauri_build::try_build(attributes).expect("Tauri build configuration");
}
