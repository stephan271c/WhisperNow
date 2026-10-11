//! Platform adapters and executable-wide observability setup.

mod asr;
mod atomic_file;
mod audio;
mod clear_data;
mod credential_store;
mod destination;
mod file_manager;
mod llama;
mod llm;
mod local_cleanup;
mod models;
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
pub use clear_data::ClearDataError;
#[cfg(any(test, feature = "test-support"))]
pub use credential_store::MemoryCredentialStore;
pub use credential_store::{CredentialStore, CredentialStoreUnavailable};
pub use file_manager::{open_in_file_manager, open_privacy_settings};
pub use llama::{CUDA_LIBRARIES, Llama};
pub use llm::complete;
pub use local_cleanup::{CleanupModelLocator, CudaPackLocator, LocalCleanupService};
pub use models::{
    CudaPackArchive, CudaPackFormat, CudaPackRemovalOutcome, CudaPackState, DownloadProgress,
    DownloadTimeouts, ModelListEntry, ModelStore, ModelStoreError,
};
#[cfg(target_os = "macos")]
pub use paste_delivery::accessibility_trusted;
#[cfg(target_os = "linux")]
pub use paste_delivery::x11_paste_injection_available;
pub use paste_delivery::{
    ClipboardContents, NativeClipboard, NativePasteError, NativePastePlatform,
};
pub use paste_keys::PasteKeys;
pub use paths::AppPaths;
pub use settings_store::{SettingsNotice, SettingsStore};
pub use sqlite_history::SqliteHistoryRepository;
use tracing_subscriber::EnvFilter;
pub use voice_activity::{silero_vad_model_path, silero_voice_activity};

pub fn init_logging() {
    let mut filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    // Rig provider targets can include request/response bodies and, for Gemini,
    // credential-bearing request URLs. Application code uses its own metadata-only
    // events, so provider-library logging is disabled structurally at the root.
    for directive in ["rig=off", "rig_core=off"] {
        filter = filter.add_directive(directive.parse().expect("static logging directive"));
    }
    let _subscriber_was_installed = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}
