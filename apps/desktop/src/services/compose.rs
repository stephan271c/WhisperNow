use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use whispernow_domain::AppError;

use whispernow_core::{
    audio::AudioInput,
    llm::{LlmFuture, LlmProvider, LlmRequest, LlmService},
};
use whispernow_domain::ipc::ErrorCategory;
use whispernow_domain::{ModelId, ModelLayout};
use whispernow_platform::{
    AppPaths, CleanupModelLocator, CpalAudioInput, CudaPackLocator, LocalCleanupService,
    ModelStore, SettingsStore, SherpaAsrEngine, silero_vad_model_path, silero_voice_activity,
};

use super::{
    Dictation, Events, NativeServices, StateCell,
    dictation::{AiCleanup, DictationParts},
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
        resource_dir: PathBuf,
    ) -> Result<Self, AppError> {
        let paths = AppPaths::discover().map_err(|_| {
            AppError::new(
                ErrorCategory::General,
                "app_paths",
                "Application storage is unavailable",
            )
        })?;
        let wayland_token_path = native_wayland.then(|| paths.wayland_remote_desktop_token_file());
        let paste_helper_setup_dir = paths.paste_helper_setup_dir();
        let output = Arc::new(OutputCoordinator::native(wayland_token_path));
        let audio: Arc<dyn AudioInput> = Arc::new(CpalAudioInput::new());
        let persistence = Arc::new(SettingsStore::new(paths.clone()));
        let persisted = persistence.load()?;
        let history = Arc::new(HistoryStore::open(paths.clone()));
        let startup_history_warning = history.take_warning();
        let llm_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("whispernow-llm")
            .enable_all()
            .build()
            .map_err(|_| {
                AppError::new(
                    ErrorCategory::General,
                    "llm_runtime",
                    "AI cleanup is unavailable",
                )
            })?;
        let llama_dir = llama_dir(&resource_dir);
        let vad_model = silero_vad_model_path(resource_dir);
        let initial = State::new(persisted.model_id.clone(), persisted.first_run_complete);
        let state = Arc::new(StateCell::new(initial, events));
        if let Some(warning) = startup_history_warning {
            state.publish_error(warning);
        }
        let settings = Arc::new(SettingsCoordinator::new(
            Arc::clone(&persistence),
            Arc::clone(&state),
        ));
        settings.report_loaded(&persisted);
        let store = Arc::new(ModelStore::new(paths.clone()));
        store.finish_cuda_pack_removal();
        let local_cleanup = Arc::new(LocalCleanupService::new(
            llama_dir,
            installed_cleanup_model(Arc::clone(&store)),
            installed_cuda_pack(Arc::clone(&store)),
        ));
        if persisted.llm_provider == LlmProvider::LocalCleanup && persisted.cleanup_enabled {
            local_cleanup.preload(persisted.cleanup_model_id());
        }
        let models = Arc::new(ModelCoordinator::new(
            Arc::clone(&store),
            Arc::clone(&state),
            Arc::clone(&local_cleanup),
        ));
        let llm = Arc::new(AiCleanup {
            service: Arc::new(ProviderRouter {
                local: Arc::clone(&local_cleanup),
            }),
            runtime: Arc::new(llm_runtime),
            api_keys: settings.api_keys(),
        });
        let dictation = Dictation::spawn(DictationParts {
            state: Arc::clone(&state),
            settings: Arc::clone(&settings),
            history: Arc::clone(&history),
            audio: Arc::clone(&audio),
            models: store,
            voice_activity: Arc::new(move || silero_voice_activity(&vad_model)),
            llm,
            output: output.clone(),
            engine: Box::new(|| Box::new(SherpaAsrEngine::new())),
        })?;
        Ok(Self {
            settings,
            history,
            audio,
            models,
            state,
            dictation,
            paths,
            output,
            paste_helper_setup_dir,
            microphone_access: MicrophoneAccess::default(),
            local_cleanup,
        })
    }
}

/// Where llama.cpp's libraries are: the resources' `llama/`, or on macOS the bundle's
/// `Contents/Frameworks/`, beside `Contents/Resources/`, where they are signed. A development
/// build can point `WHISPERNOW_LLAMA_DIR` at a llama.cpp b11259 release instead.
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

/// An installed cleanup model's GGUF, looked up by id when it is needed.
fn installed_cleanup_model(store: Arc<ModelStore>) -> CleanupModelLocator {
    Arc::new(move |model_id| {
        let id = ModelId::new(model_id).ok()?;
        let model = store.resolve(&id).ok().flatten()?;
        match &model.definition.layout {
            ModelLayout::Gguf { model: artifact } => Some(model.root.join(artifact)),
            _ => None,
        }
    })
}

/// The installed CUDA pack's directory, looked up when a cleanup model loads.
fn installed_cuda_pack(store: Arc<ModelStore>) -> CudaPackLocator {
    Arc::new(move || store.cuda_pack_dir())
}

/// AI cleanup's providers: the local cleanup model in process, every other one through Rig.
struct ProviderRouter {
    local: Arc<LocalCleanupService>,
}

impl LlmService for ProviderRouter {
    fn complete(&self, request: LlmRequest) -> LlmFuture<'_> {
        if request.config.provider == LlmProvider::LocalCleanup {
            self.local.complete(request)
        } else {
            Box::pin(whispernow_platform::complete(request))
        }
    }
}
