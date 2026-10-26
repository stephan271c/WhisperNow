//! Platform adapters and executable-wide observability setup.

mod asr;
mod atomic_file;
mod audio;
mod clear_data;
mod credential_store;
mod destination;
mod llama;
mod llm;
mod local_cleanup;
mod models;
mod opener;
mod paste_delivery;
mod paste_keys;
mod paths;
mod settings_store;
mod sqlite_history;
mod voice_activity;
#[cfg(target_os = "linux")]
pub mod wayland;

pub use asr::SherpaAsrEngine;
pub use audio::CpalAudioInput;
pub use clear_data::ClearPlan;
#[cfg(any(test, feature = "test-support"))]
pub use credential_store::{CredentialStore, CredentialStoreUnavailable, MemoryCredentialStore};
pub use llm::{complete, list_models};
pub use local_cleanup::{HeldUnloaded, LocalCleanupService};
#[cfg(any(test, feature = "test-support"))]
pub use models::DownloadTimeouts;
pub use models::{DownloadProgress, ModelListEntry, ModelStore, ModelStoreError};
pub use opener::{open_directory, open_privacy_settings, open_url};
#[cfg(target_os = "linux")]
pub use paste_delivery::x11_paste_injection_available;
pub use paste_delivery::{
    ClipboardContents, NativeClipboard, NativePasteError, NativePastePlatform,
};
#[cfg(target_os = "macos")]
pub use paste_delivery::{accessibility_trusted, request_accessibility_trust};
pub use paste_keys::PasteKeys;
pub use paths::AppPaths;
pub use settings_store::{SettingsLoad, SettingsNotice, SettingsStore};
pub use sqlite_history::SqliteHistoryRepository;
use tracing_subscriber::EnvFilter;
pub use voice_activity::{silero_vad_model_path, silero_voice_activity};

pub fn init_logging() {
    let mut filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    // Rig provider targets can include request/response bodies and, for Gemini,
    // credential-bearing request URLs.
    for directive in ["rig=off", "rig_core=off"] {
        filter = filter.add_directive(directive.parse().expect("static logging directive"));
    }
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}
