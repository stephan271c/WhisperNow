use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use whispernow_domain::{AppError, ErrorCategory};

use whispernow_core::{
    audio::AudioInput,
    llm::{LlmFuture, LlmRequest, LlmService},
};
use whispernow_platform::{
    AppPaths, CpalAudioInput, LocalCleanupService, ModelStore, SettingsLoad, SettingsStore,
    SherpaAsrEngine, silero_vad_model_path, silero_voice_activity,
};

use super::{
    Dictation, Events, NativeServices, StateCell,
    dictation::{AiCleanup, DictationParts},
    follow_cleanup_provider,
    history::HistoryStore,
    models::ModelCoordinator,
    output::OutputCoordinator,
    permissions::MicrophoneAccess,
    settings::SettingsCoordinator,
    state::State,
};

impl NativeServices {
    pub(crate) fn compose(
        events: Events,
        native_wayland: bool,
        resource_dir: &Path,
    ) -> Result<(Self, bool), AppError> {
        let paths = AppPaths::discover()
            .inspect_err(|error| tracing::warn!(%error, "application storage could not be found"))
            .map_err(|_| {
                AppError::new(
                    ErrorCategory::General,
                    "app_paths",
                    "Application storage is unavailable",
                )
            })?;
        let wayland_token_path = native_wayland.then(|| paths.wayland_remote_desktop_token_file());
        let output = Arc::new(OutputCoordinator::native(wayland_token_path));
        let audio: Arc<dyn AudioInput> = Arc::new(CpalAudioInput::new());
        let persistence = Arc::new(SettingsStore::new(paths.clone()));
        let SettingsLoad {
            settings: persisted,
            notice: settings_notice,
            enable_autostart,
        } = persistence.load()?;
        let (history, startup_history_warning) = HistoryStore::open(paths.clone());
        let history = Arc::new(history);
        let llama_dir = llama_dir(resource_dir);
        let vad_model = silero_vad_model_path(resource_dir);
        let initial = State::new(persisted.selected_model_id().clone());
        let state = Arc::new(StateCell::new(initial, events));
        if let Some(warning) = startup_history_warning {
            state.publish_error(warning);
        }
        let settings = Arc::new(SettingsCoordinator::new(
            Arc::clone(&persistence),
            Arc::clone(&state),
            persisted.clone(),
        ));
        settings.report_loaded(&persisted, settings_notice);
        let store = Arc::new(ModelStore::new(paths.clone()));
        let cleanup_state = Arc::clone(&state);
        let local_cleanup = Arc::new(LocalCleanupService::new(
            llama_dir,
            Arc::clone(&store),
            move |device| cleanup_state.apply(|state| state.cleanup_device = device),
        ));
        follow_cleanup_provider(&local_cleanup, &persisted);
        let models = ModelCoordinator::new(
            Arc::clone(&store),
            Arc::clone(&state),
            Arc::clone(&local_cleanup),
        );
        let cleanup = Arc::new(AiCleanup {
            service: Arc::new(ProviderRouter {
                local: Arc::clone(&local_cleanup),
            }),
            api_keys: persistence,
        });
        let dictation = Dictation::spawn(DictationParts {
            state: Arc::clone(&state),
            settings: Arc::clone(&settings),
            history: Arc::clone(&history),
            audio: Arc::clone(&audio),
            models: store,
            voice_activity: Arc::new(move || silero_voice_activity(&vad_model)),
            cleanup,
            output: output.clone(),
            engine: Box::new(|| Box::new(SherpaAsrEngine::default())),
        })?;
        Ok((
            Self {
                settings,
                history,
                audio,
                models,
                state,
                dictation,
                paths,
                output,
                microphone_access: MicrophoneAccess::default(),
                local_cleanup,
            },
            enable_autostart,
        ))
    }
}

/// On macOS the libraries sit in the bundle's `Contents/Frameworks/`, where they are signed. A
/// development build can point `WHISPERNOW_LLAMA_DIR` at a llama.cpp b11259 release instead.
fn llama_dir(resource_dir: &Path) -> PathBuf {
    if let Some(directory) = std::env::var_os("WHISPERNOW_LLAMA_DIR") {
        PathBuf::from(directory)
    } else if cfg!(target_os = "macos") {
        resource_dir.parent().map_or_else(
            || resource_dir.join("llama"),
            |contents| contents.join("Frameworks"),
        )
    } else {
        resource_dir.join("llama")
    }
}

struct ProviderRouter {
    local: Arc<LocalCleanupService>,
}

impl LlmService for ProviderRouter {
    fn complete(&self, request: LlmRequest) -> LlmFuture<'_> {
        match request {
            LlmRequest::Local {
                model,
                transcript,
                max_output_tokens,
                draft,
            } => Box::pin(
                self.local
                    .complete(model, transcript, max_output_tokens, draft),
            ),
            LlmRequest::Remote {
                target,
                api_key,
                prompt,
            } => Box::pin(whispernow_platform::complete(target, api_key, prompt)),
        }
    }
}
