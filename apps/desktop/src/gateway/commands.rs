use whispernow_domain::AppError;
use whispernow_domain::ipc::{
    AppState, AudioDevice, ClearDataResult, DataLocation, EventPayload, MainWindowSection,
    MicrophoneTestResponse, ModelSummary, PermissionKind, PermissionsResponse,
    SaveSettingsResponse, SetApiKeyRequest, SettingsView,
};
use whispernow_domain::{
    EntryId, HistoryPage, HistoryPageRequest, LlmProvider, ModelId, SettingChange, SetupCheckpoint,
};

use super::{Failure, Gateway, run_blocking};
use crate::services::{RecordingTrigger, provider_models_unavailable};

#[tauri::command]
pub(crate) async fn get_settings(
    state: tauri::State<'_, Gateway>,
) -> Result<SettingsView, AppError> {
    state.settings_view().await
}

#[tauri::command]
pub(crate) async fn update_settings(
    request: SettingChange,
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
    state.set_api_key(request).await
}

#[tauri::command]
pub(crate) async fn clear_api_key(
    request: LlmProvider,
    state: tauri::State<'_, Gateway>,
) -> Result<SettingsView, AppError> {
    state.clear_api_key(request).await
}

#[tauri::command]
pub(crate) async fn list_provider_models(
    request: LlmProvider,
    state: tauri::State<'_, Gateway>,
) -> Result<Vec<String>, AppError> {
    let (provider, api_key) = state
        .run_command("list_provider_models", Failure::Local, move |backend| {
            backend.settings.provider_and_key(request)
        })
        .await?;
    whispernow_platform::list_models(&provider, api_key.as_ref())
        .await
        .map_err(|error| {
            tracing::warn!(?error, "the provider's models could not be listed");
            provider_models_unavailable()
        })
}

#[tauri::command]
pub(crate) async fn list_audio_devices(
    state: tauri::State<'_, Gateway>,
) -> Result<Vec<AudioDevice>, AppError> {
    state
        .run_command("list_audio_devices", Failure::Published, |backend| {
            backend.list_audio_devices()
        })
        .await
}

#[tauri::command]
pub(crate) async fn test_microphone(
    state: tauri::State<'_, Gateway>,
) -> Result<MicrophoneTestResponse, AppError> {
    state
        .run_command("test_microphone", Failure::Local, |backend| {
            backend.test_microphone()
        })
        .await
}

#[tauri::command]
pub(crate) async fn list_models(
    state: tauri::State<'_, Gateway>,
) -> Result<Vec<ModelSummary>, AppError> {
    state
        .run_command("list_models", Failure::Published, |backend| {
            backend.models.list()
        })
        .await
}

#[tauri::command]
pub(crate) async fn data_locations(
    state: tauri::State<'_, Gateway>,
) -> Result<Vec<DataLocation>, AppError> {
    state
        .run_command("data_locations", Failure::Published, |backend| {
            backend.data_locations()
        })
        .await
}

#[tauri::command]
pub(crate) async fn dismiss_recording_overlay(
    state: tauri::State<'_, Gateway>,
) -> Result<(), AppError> {
    state.backend.state.dismiss_error();
    Ok(())
}

#[tauri::command]
pub(crate) async fn copy_text(
    request: String,
    state: tauri::State<'_, Gateway>,
) -> Result<(), AppError> {
    state
        .run_command("copy_text", Failure::Local, move |backend| {
            backend.output.copy(&request)
        })
        .await
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
        .backend
        .state
        .events()
        .publish(EventPayload::NavigationRequested { section: request });
    crate::show_settings_window(&app);
    state.backend.state.dismiss_error();
    Ok(())
}

#[tauri::command]
pub(crate) async fn get_state(state: tauri::State<'_, Gateway>) -> Result<AppState, AppError> {
    state
        .run_command("get_state", Failure::Published, |backend| {
            Ok(backend.state.snapshot())
        })
        .await
}

#[tauri::command]
pub(crate) async fn get_history(
    request: HistoryPageRequest,
    state: tauri::State<'_, Gateway>,
) -> Result<HistoryPage, AppError> {
    state
        .run_command("get_history", Failure::Local, move |backend| {
            backend.history_page(&request)
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
        .run_command("get_history_audio", Failure::Local, move |backend| {
            backend.history.audio(request)
        })
        .await
        .map(tauri::ipc::Response::new)
}

#[tauri::command]
pub(crate) async fn open_recordings_folder(
    state: tauri::State<'_, Gateway>,
) -> Result<(), AppError> {
    state
        .run_command("open_recordings_folder", Failure::Local, |backend| {
            backend.open_recordings_folder()
        })
        .await
}

#[tauri::command]
pub(crate) async fn delete_history_entry(
    request: EntryId,
    state: tauri::State<'_, Gateway>,
) -> Result<(), AppError> {
    state
        .run_command("delete_history_entry", Failure::Published, move |backend| {
            backend.delete_history_entry(request)
        })
        .await
}

#[tauri::command]
pub(crate) async fn clear_history(state: tauri::State<'_, Gateway>) -> Result<(), AppError> {
    state
        .run_command("clear_history", Failure::Published, |backend| {
            backend.clear_history()
        })
        .await
}

#[tauri::command]
pub(crate) async fn clear_data(
    state: tauri::State<'_, Gateway>,
) -> Result<ClearDataResult, AppError> {
    state.clear_data().await
}

#[tauri::command]
pub(crate) async fn download_model(
    request: ModelId,
    state: tauri::State<'_, Gateway>,
) -> Result<(), AppError> {
    state
        .run_command("download_model", Failure::Local, move |backend| {
            backend.models.download(&request)
        })
        .await
}

#[tauri::command]
pub(crate) async fn cancel_download(
    request: ModelId,
    state: tauri::State<'_, Gateway>,
) -> Result<(), AppError> {
    state
        .run_command("cancel_download", Failure::Local, move |backend| {
            backend.models.cancel_download(&request)
        })
        .await
}

#[tauri::command]
pub(crate) async fn select_model(
    request: ModelId,
    state: tauri::State<'_, Gateway>,
) -> Result<(), AppError> {
    state
        .run_command("select_model", Failure::Local, move |backend| {
            backend.select_model(request)
        })
        .await
}

#[tauri::command]
pub(crate) async fn delete_model(
    request: ModelId,
    state: tauri::State<'_, Gateway>,
) -> Result<(), AppError> {
    state
        .run_command("delete_model", Failure::Local, move |backend| {
            backend.models.delete(&request)
        })
        .await
}

#[tauri::command]
pub(crate) async fn save_setup_checkpoint(
    request: SetupCheckpoint,
    state: tauri::State<'_, Gateway>,
) -> Result<SettingsView, AppError> {
    state.save_setup_checkpoint(request).await
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
) -> Result<SettingsView, AppError> {
    state.restart_setup().await
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
    run_blocking(move || crate::services::open_permission_settings(request)).await?
}

/// Pasting chooses between the helper and the portal at startup, so the app restarts once the
/// helper is installed.
#[tauri::command]
pub(crate) async fn set_up_paste_helper(
    app: tauri::AppHandle,
    state: tauri::State<'_, Gateway>,
) -> Result<(), AppError> {
    if state.set_up_paste_helper().await? {
        app.request_restart();
    }
    Ok(())
}

/// `stop_recording` ends it, and its transcript is neither pasted nor kept.
#[tauri::command]
pub(crate) async fn start_test_dictation(state: tauri::State<'_, Gateway>) -> Result<(), AppError> {
    state
        .run_command("start_test_dictation", Failure::Local, |backend| {
            backend.dictation.start(RecordingTrigger::SetupTest)
        })
        .await
}

#[tauri::command]
pub(crate) async fn stop_recording(state: tauri::State<'_, Gateway>) -> Result<(), AppError> {
    state
        .run_command("stop_recording", Failure::Published, |backend| {
            backend.dictation.stop()
        })
        .await
}

#[tauri::command]
pub(crate) async fn cancel_recording(state: tauri::State<'_, Gateway>) -> Result<(), AppError> {
    state
        .run_command("cancel_recording", Failure::Published, |backend| {
            backend.dictation.cancel()
        })
        .await
}
