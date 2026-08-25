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
pub use clear_data::{ClearDataCategory, ClearDataError, DataCleaner};
pub use credential_store::{CredentialStore, CredentialStoreUnavailable, MemoryCredentialStore};
pub use file_manager::{PrivacySettingsPage, open_in_file_manager, open_privacy_settings};
pub use llama::{CUDA_LIBRARIES, Llama};
pub use llm::{RemoteProvider, RigLlmService};
pub use local_cleanup::{CleanupModelLocator, CudaPackLocator, LocalCleanupService};
pub use models::{
    CudaPackArchive, CudaPackFormat, CudaPackState, DownloadObserver, DownloadProgress,
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
pub use voice_activity::{SileroVoiceActivitySource, silero_vad_model_path};

/// Initializes content-free structured logging once at a composition root.
pub fn init_logging() {
    let mut filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    // Rig provider targets can include request/response bodies and, for Gemini,
    // credential-bearing request URLs. Application code uses its own metadata-only
    // events, so provider-library logging is disabled structurally at the root.
    for directive in ["rig=off", "rig_core=off"] {
        if let Ok(disable_rig) = directive.parse() {
            filter = filter.add_directive(disable_rig);
        }
    }
    let _subscriber_was_installed = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}
