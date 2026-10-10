#[path = "../../crates/whispernow-domain/src/ipc/command_names.rs"]
mod command_names;

use command_names::COMMAND_NAMES;

fn main() {
    println!("cargo:rerun-if-changed=../../crates/whispernow-domain/src/ipc/command_names.rs");
    let attributes = tauri_build::Attributes::new()
        .app_manifest(tauri_build::AppManifest::new().commands(&COMMAND_NAMES));
    tauri_build::try_build(attributes).expect("Tauri build configuration");
}
