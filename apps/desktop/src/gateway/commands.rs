use whispernow_domain::AppError;
use whispernow_domain::ipc::{
    AppState, AudioDevice, ClearDataResult, DataLocation, ErrorCategory, EventPayload,
    MainWindowSection, MicrophoneTestResponse, ModelSummary, PermissionKind, PermissionsResponse,
    SaveSettingsResponse, SetApiKeyRequest, SettingsView, SetupStateView,
};
use whispernow_domain::{
    ClearDataCategory, EntryId, HistoryPage, HistoryPageRequest, LlmProvider, ModelId,
    RemoteProvider, Settings, SetupCheckpoint,
};

use super::{Failure, Gateway, run_blocking};
use crate::services::{NativeServices, RecordingTrigger};

#[tauri::command]
pub(crate) async fn get_settings(
    state: tauri::State<'_, Gateway>,
) -> Result<SettingsView, AppError> {
    state.settings_view().await
}

#[tauri::command]
pub(crate) async fn update_settings(
    request: Settings,
    state: tauri::State<'_, Gateway>,
) -> Result<SaveSettingsResponse, AppError> {
    state.update_settings(request).await
}

#[tauri::command]
pub(crate) async fn set_start_on_login(
    request: bool,
    state: tauri::State<'_, Gateway>,
) -> Result<SaveSettingsResponse, AppError> {
    state.set_start_on_login(request).await
}

#[tauri::command]
pub(crate) async fn suspend_shortcuts(state: tauri::State<'_, Gateway>) -> Result<(), AppError> {
    state.set_shortcut_capture(true).await
}

#[tauri::command]
pub(crate) async fn resume_shortcuts(state: tauri::State<'_, Gateway>) -> Result<(), AppError> {
    state.set_shortcut_capture(false).await
}

#[tauri::command]
pub(crate) async fn set_api_key(
    request: SetApiKeyRequest,
    state: tauri::State<'_, Gateway>,
) -> Result<SettingsView, AppError> {
    let mut view = state
        .run_command("set_api_key", Failure::Local, move |services| {
            services.settings.set_api_key(&request)
        })
        .await?;
    state.read_start_on_login(&mut view);
    Ok(view)
}

#[tauri::command]
pub(crate) async fn clear_api_key(
    request: LlmProvider,
    state: tauri::State<'_, Gateway>,
) -> Result<SettingsView, AppError> {
    let mut view = state
        .run_command("clear_api_key", Failure::Local, move |services| {
            services.settings.clear_api_key(request)
        })
        .await?;
    state.read_start_on_login(&mut view);
    Ok(view)
}

/// The model names `request` offers, asked with its saved endpoint and API key.
#[tauri::command]
pub(crate) async fn list_provider_models(
    request: LlmProvider,
    state: tauri::State<'_, Gateway>,
) -> Result<Vec<String>, AppError> {
    let (provider, api_key) = state
        .run_command("list_provider_models", Failure::Local, move |services| {
            let settings = services.settings.load();
            let endpoint = settings.provider_settings(request).api_base.clone();
            let provider =
                RemoteProvider::new(request, endpoint).ok_or_else(provider_models_unavailable)?;
            let api_key = if settings.api_keys.contains(&request) {
                services.settings.api_keys().api_key(request)?
            } else {
                None
            };
            Ok((provider, api_key))
        })
        .await?;
    whispernow_platform::list_models(&provider, api_key.as_ref())
        .await
        .map_err(|error| {
            tracing::warn!(?error, "the provider's models could not be listed");
            provider_models_unavailable()
        })
}

fn provider_models_unavailable() -> AppError {
    AppError::new(
        ErrorCategory::Cleanup,
        "provider_models_unavailable",
        "The provider's models could not be listed",
    )
}

#[tauri::command]
pub(crate) async fn list_audio_devices(
    state: tauri::State<'_, Gateway>,
) -> Result<Vec<AudioDevice>, AppError> {
    state
        .run_command("list_audio_devices", Failure::Published, |services| {
            services.list_audio_devices()
        })
        .await
}

#[tauri::command]
pub(crate) async fn test_microphone(
    state: tauri::State<'_, Gateway>,
) -> Result<MicrophoneTestResponse, AppError> {
    state
        .run_command("test_microphone", Failure::Local, |services| {
            services.test_microphone()
        })
        .await
}

#[tauri::command]
pub(crate) async fn list_models(
    state: tauri::State<'_, Gateway>,
) -> Result<Vec<ModelSummary>, AppError> {
    state
        .run_command("list_models", Failure::Published, |services| {
            services.models.list()
        })
        .await
}

#[tauri::command]
pub(crate) async fn get_setup_state(
    state: tauri::State<'_, Gateway>,
) -> Result<SetupStateView, AppError> {
    state
        .run_command("get_setup_state", Failure::Published, |services| {
            Ok(services.settings.get_setup_state())
        })
        .await
}

#[tauri::command]
pub(crate) async fn data_locations(
    state: tauri::State<'_, Gateway>,
) -> Result<Vec<DataLocation>, AppError> {
    state
        .run_command("data_locations", Failure::Published, |services| {
            services.data_locations()
        })
        .await
}

/// Dismisses the failure the recording bar shows.
#[tauri::command]
pub(crate) async fn dismiss_recording_overlay(
    state: tauri::State<'_, Gateway>,
) -> Result<(), AppError> {
    state.services.state.dismiss_error();
    Ok(())
}

/// Copies the last transcript for a manual paste and dismisses the "not pasted" failure it
/// recovers from.
#[tauri::command]
pub(crate) async fn copy_last_transcript(state: tauri::State<'_, Gateway>) -> Result<(), AppError> {
    state
        .run_command("copy_last_transcript", Failure::Published, |services| {
            services.copy_last_transcript()
        })
        .await?;
    state.services.state.dismiss_error();
    Ok(())
}

/// Shows the main window at the section an overlay action points to and dismisses the failure
/// it came from.
#[tauri::command]
pub(crate) async fn open_main_window(
    request: MainWindowSection,
    app: tauri::AppHandle,
    state: tauri::State<'_, Gateway>,
) -> Result<(), AppError> {
    state
        .services
        .state
        .events()
        .publish(EventPayload::NavigationRequested { section: request });
    crate::show_settings_window(&app);
    state.services.state.dismiss_error();
    Ok(())
}

#[tauri::command]
pub(crate) async fn get_state(state: tauri::State<'_, Gateway>) -> Result<AppState, AppError> {
    state
        .run_command("get_state", Failure::Published, |services| {
            Ok(services.state.snapshot())
        })
        .await
}

#[tauri::command]
pub(crate) async fn get_history(
    request: HistoryPageRequest,
    state: tauri::State<'_, Gateway>,
) -> Result<HistoryPage, AppError> {
    state
        .run_command("get_history", Failure::Local, move |services| {
            services.history(&request)
        })
        .await
}

/// Returns an entry's kept audio as raw WAV bytes, never as a path.
#[tauri::command]
pub(crate) async fn get_history_audio(
    request: EntryId,
    state: tauri::State<'_, Gateway>,
) -> Result<tauri::ipc::Response, AppError> {
    state
        .run_command("get_history_audio", Failure::Local, move |services| {
            services.history.audio(request)
        })
        .await
        .map(tauri::ipc::Response::new)
}

#[tauri::command]
pub(crate) async fn open_recordings_folder(
    state: tauri::State<'_, Gateway>,
) -> Result<(), AppError> {
    state
        .run_command("open_recordings_folder", Failure::Local, |services| {
            services.open_recordings_folder()
        })
        .await
}

#[tauri::command]
pub(crate) async fn delete_history_entry(
    request: EntryId,
    state: tauri::State<'_, Gateway>,
) -> Result<(), AppError> {
    state
        .run_command(
            "delete_history_entry",
            Failure::Published,
            move |services| services.delete_history_entry(request),
        )
        .await
}

#[tauri::command]
pub(crate) async fn clear_history(state: tauri::State<'_, Gateway>) -> Result<(), AppError> {
    state
        .run_command("clear_history", Failure::Published, |services| {
            services.clear_history()
        })
        .await
}

#[tauri::command]
pub(crate) async fn clear_data(
    request: Vec<ClearDataCategory>,
    state: tauri::State<'_, Gateway>,
) -> Result<ClearDataResult, AppError> {
    state.clear_data(request).await
}

#[tauri::command]
pub(crate) async fn download_model(
    request: ModelId,
    state: tauri::State<'_, Gateway>,
) -> Result<(), AppError> {
    state
        .run_command("download_model", Failure::Published, move |services| {
            services.models.download(&request)
        })
        .await
}

#[tauri::command]
pub(crate) async fn cancel_download(
    request: ModelId,
    state: tauri::State<'_, Gateway>,
) -> Result<(), AppError> {
    state
        .run_command("cancel_download", Failure::Published, move |services| {
            services.models.cancel_download(&request)
        })
        .await
}

#[tauri::command]
pub(crate) async fn select_model(
    request: ModelId,
    state: tauri::State<'_, Gateway>,
) -> Result<(), AppError> {
    state
        .run_command("select_model", Failure::Published, move |services| {
            services
                .models
                .select(request, &services.settings, &services.dictation)
        })
        .await
}

#[tauri::command]
pub(crate) async fn delete_model(
    request: ModelId,
    state: tauri::State<'_, Gateway>,
) -> Result<(), AppError> {
    state
        .run_command("delete_model", Failure::Published, move |services| {
            services.models.delete(&request)
        })
        .await
}

#[tauri::command]
pub(crate) async fn save_setup_checkpoint(
    request: SetupCheckpoint,
    state: tauri::State<'_, Gateway>,
) -> Result<SetupStateView, AppError> {
    state
        .run_command(
            "save_setup_checkpoint",
            Failure::Published,
            move |services| {
                services
                    .settings
                    .save_setup_checkpoint(request, services.models.model_store())
            },
        )
        .await
}

#[tauri::command]
pub(crate) async fn complete_setup(
    request: SetupCheckpoint,
    state: tauri::State<'_, Gateway>,
) -> Result<SaveSettingsResponse, AppError> {
    state.complete_setup(request).await
}

#[tauri::command]
pub(crate) async fn restart_setup(
    state: tauri::State<'_, Gateway>,
) -> Result<SetupStateView, AppError> {
    state
        .run_command("restart_setup", Failure::Local, |services| {
            services.settings.restart_setup()
        })
        .await
}

#[tauri::command]
pub(crate) async fn get_permissions(
    state: tauri::State<'_, Gateway>,
) -> Result<PermissionsResponse, AppError> {
    state.permissions().await
}

#[tauri::command]
pub(crate) async fn request_permission(
    request: PermissionKind,
    state: tauri::State<'_, Gateway>,
) -> Result<PermissionsResponse, AppError> {
    state.request_permission(request).await
}

#[tauri::command]
pub(crate) async fn open_permission_settings(request: PermissionKind) -> Result<(), AppError> {
    run_blocking(move || NativeServices::open_permission_settings(request)).await?
}

/// `stop_recording` ends it, and its transcript is neither pasted nor kept.
#[tauri::command]
pub(crate) async fn start_test_dictation(state: tauri::State<'_, Gateway>) -> Result<(), AppError> {
    state
        .run_command("start_test_dictation", Failure::Local, |services| {
            services
                .dictation
                .start(RecordingTrigger::SetupTest)
                .map(drop)
        })
        .await
}

#[tauri::command]
pub(crate) async fn stop_recording(state: tauri::State<'_, Gateway>) -> Result<(), AppError> {
    state
        .run_command("stop_recording", Failure::Published, |services| {
            services.dictation.stop()
        })
        .await
}

#[tauri::command]
pub(crate) async fn cancel_recording(state: tauri::State<'_, Gateway>) -> Result<(), AppError> {
    state
        .run_command("cancel_recording", Failure::Published, |services| {
            services.dictation.cancel()
        })
        .await
}
