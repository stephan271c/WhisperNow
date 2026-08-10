//! Platform adapters and executable-wide observability setup.

mod asr;
mod audio;
mod llama;
mod llm;
mod local_cleanup;
mod models;
mod paths;
mod voice_activity;

pub use asr::SherpaAsrEngine;
pub use audio::CpalAudioInput;
pub use llama::{CUDA_LIBRARIES, Llama};
pub use llm::{RemoteProvider, RigLlmService};
pub use local_cleanup::{CleanupModelLocator, CudaPackLocator, LocalCleanupService};
pub use models::{
    CudaPackArchive, CudaPackFormat, CudaPackState, DownloadObserver, DownloadProgress,
    DownloadTimeouts, ModelListEntry, ModelStore, ModelStoreError,
};
pub use paths::AppPaths;
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
