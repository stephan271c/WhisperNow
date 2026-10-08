use whispernow_api::{
    ApiErrorDto, AppStateDto, ClearApiKeyRequest, ClearDataRequest, ClearDataResult,
    DataLocationDto, DeleteHistoryEntryRequest, GetHistoryAudioRequest, HistoryPage,
    HistoryPageRequest, ListAudioDevicesResponse, ListModelsResponse, MicrophoneTestResponse,
    ModelId, OpenMainWindowRequest, PermissionRequest, PermissionsResponse, SaveSettingsResponse,
    SetApiKeyRequest, SettingsView, SetupCheckpoint, SetupStateView, TestPromptRequest,
    TestPromptResponse, UpdateSettingsRequest,
};

use super::{Failure, Gateway, run_blocking};
use crate::services::{NativeServices, RecordingTrigger};

#[tauri::command]
pub(crate) async fn get_settings(
    state: tauri::State<'_, Gateway>,
) -> Result<SettingsView, ApiErrorDto> {
    state.settings_view().await
}

#[tauri::command]
pub(crate) async fn update_settings(
    request: UpdateSettingsRequest,
    state: tauri::State<'_, Gateway>,
) -> Result<SaveSettingsResponse, ApiErrorDto> {
    state.update_settings(request).await
}

#[tauri::command]
pub(crate) async fn test_prompt(
    request: TestPromptRequest,
    state: tauri::State<'_, Gateway>,
) -> Result<TestPromptResponse, ApiErrorDto> {
    state
        .run_command("test_prompt", Failure::Local, move |services, shutdown| {
            services.test_prompt(&request, &shutdown)
        })
        .await
}

#[tauri::command]
pub(crate) async fn suspend_shortcuts(state: tauri::State<'_, Gateway>) -> Result<(), ApiErrorDto> {
    state.set_shortcut_capture(true).await
}

#[tauri::command]
pub(crate) async fn resume_shortcuts(state: tauri::State<'_, Gateway>) -> Result<(), ApiErrorDto> {
    state.set_shortcut_capture(false).await
}

#[tauri::command]
pub(crate) async fn set_api_key(
    request: SetApiKeyRequest,
    state: tauri::State<'_, Gateway>,
) -> Result<SettingsView, ApiErrorDto> {
    state
        .run_command("set_api_key", Failure::Local, move |services, _| {
            services.settings.set_api_key(&request)
        })
        .await
}

#[tauri::command]
pub(crate) async fn clear_api_key(
    request: ClearApiKeyRequest,
    state: tauri::State<'_, Gateway>,
) -> Result<SettingsView, ApiErrorDto> {
    state
        .run_command("clear_api_key", Failure::Local, move |services, _| {
            services.settings.clear_api_key(request)
        })
        .await
}

#[tauri::command]
pub(crate) async fn list_audio_devices(
    state: tauri::State<'_, Gateway>,
) -> Result<ListAudioDevicesResponse, ApiErrorDto> {
    state
        .run_command("list_audio_devices", Failure::Published, |services, _| {
            Ok(services.list_audio_devices())
        })
        .await
}

#[tauri::command]
pub(crate) async fn test_microphone(
    state: tauri::State<'_, Gateway>,
) -> Result<MicrophoneTestResponse, ApiErrorDto> {
    state
        .run_command("test_microphone", Failure::Local, |services, _| {
            services.test_microphone()
        })
        .await
}

#[tauri::command]
pub(crate) async fn list_models(
    state: tauri::State<'_, Gateway>,
) -> Result<ListModelsResponse, ApiErrorDto> {
    state
        .run_command("list_models", Failure::Published, |services, _| {
            services.models.list()
        })
        .await
}

#[tauri::command]
pub(crate) async fn get_setup_state(
    state: tauri::State<'_, Gateway>,
) -> Result<SetupStateView, ApiErrorDto> {
    state
        .run_command("get_setup_state", Failure::Published, |services, _| {
            services.settings.get_setup_state()
        })
        .await
}

#[tauri::command]
pub(crate) async fn data_locations(
    state: tauri::State<'_, Gateway>,
) -> Result<Vec<DataLocationDto>, ApiErrorDto> {
    state
        .run_command("data_locations", Failure::Published, |services, _| {
            services.data_locations()
        })
        .await
}

/// Dismisses the failure the recording bar shows.
#[tauri::command]
pub(crate) async fn dismiss_recording_overlay(
    state: tauri::State<'_, Gateway>,
) -> Result<(), ApiErrorDto> {
    state.services.state.dismiss_error();
    Ok(())
}

/// Copies the last transcript for a manual paste and dismisses the "not pasted" failure it
/// recovers from.
#[tauri::command]
pub(crate) async fn copy_last_transcript(
    state: tauri::State<'_, Gateway>,
) -> Result<(), ApiErrorDto> {
    state
        .run_command("copy_last_transcript", Failure::Published, |services, _| {
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
    request: OpenMainWindowRequest,
    app: tauri::AppHandle,
    state: tauri::State<'_, Gateway>,
) -> Result<(), ApiErrorDto> {
    state.request_navigation(request.section);
    crate::show_settings_window(&app);
    state.services.state.dismiss_error();
    Ok(())
}

#[tauri::command]
pub(crate) async fn get_state(
    state: tauri::State<'_, Gateway>,
) -> Result<AppStateDto, ApiErrorDto> {
    state
        .run_command("get_state", Failure::Published, |services, _| {
            Ok(services.state.snapshot())
        })
        .await
}

#[tauri::command]
pub(crate) async fn get_history(
    request: HistoryPageRequest,
    state: tauri::State<'_, Gateway>,
) -> Result<HistoryPage, ApiErrorDto> {
    state
        .run_command("get_history", Failure::Local, move |services, _| {
            services.history(&request)
        })
        .await
}

/// Returns an entry's kept audio as raw WAV bytes, never as a path.
#[tauri::command]
pub(crate) async fn get_history_audio(
    request: GetHistoryAudioRequest,
    state: tauri::State<'_, Gateway>,
) -> Result<tauri::ipc::Response, ApiErrorDto> {
    state
        .run_command("get_history_audio", Failure::Local, move |services, _| {
            services.history.audio(request.id)
        })
        .await
        .map(tauri::ipc::Response::new)
}

#[tauri::command]
pub(crate) async fn open_recordings_folder(
    state: tauri::State<'_, Gateway>,
) -> Result<(), ApiErrorDto> {
    state
        .run_command("open_recordings_folder", Failure::Local, |services, _| {
            services.open_recordings_folder()
        })
        .await
}

#[tauri::command]
pub(crate) async fn delete_history_entry(
    request: DeleteHistoryEntryRequest,
    state: tauri::State<'_, Gateway>,
) -> Result<(), ApiErrorDto> {
    state
        .run_command(
            "delete_history_entry",
            Failure::Published,
            move |services, _| services.delete_history_entry(request),
        )
        .await
}

#[tauri::command]
pub(crate) async fn clear_history(state: tauri::State<'_, Gateway>) -> Result<(), ApiErrorDto> {
    state
        .run_command("clear_history", Failure::Published, |services, _| {
            services.clear_history()
        })
        .await
}

#[tauri::command]
pub(crate) async fn clear_data(
    request: ClearDataRequest,
    state: tauri::State<'_, Gateway>,
) -> Result<ClearDataResult, ApiErrorDto> {
    state.clear_data(request).await
}

#[tauri::command]
pub(crate) async fn download_model(
    request: ModelId,
    state: tauri::State<'_, Gateway>,
) -> Result<(), ApiErrorDto> {
    state
        .run_command(
            "download_model",
            Failure::Published,
            move |services, shutdown| services.models.download(&request, &shutdown),
        )
        .await
}

#[tauri::command]
pub(crate) async fn cancel_download(
    request: ModelId,
    state: tauri::State<'_, Gateway>,
) -> Result<(), ApiErrorDto> {
    state
        .run_command("cancel_download", Failure::Published, move |services, _| {
            services.models.cancel_download(&request)
        })
        .await
}

#[tauri::command]
pub(crate) async fn select_model(
    request: ModelId,
    state: tauri::State<'_, Gateway>,
) -> Result<(), ApiErrorDto> {
    state
        .run_command("select_model", Failure::Published, move |services, _| {
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
) -> Result<(), ApiErrorDto> {
    state
        .run_command("delete_model", Failure::Published, move |services, _| {
            services.models.delete(&request, &services.settings)
        })
        .await
}

#[tauri::command]
pub(crate) async fn save_setup_checkpoint(
    request: SetupCheckpoint,
    state: tauri::State<'_, Gateway>,
) -> Result<SetupStateView, ApiErrorDto> {
    state
        .run_command(
            "save_setup_checkpoint",
            Failure::Published,
            move |services, _| {
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
) -> Result<SaveSettingsResponse, ApiErrorDto> {
    state.complete_setup(request).await
}

#[tauri::command]
pub(crate) async fn restart_setup(
    state: tauri::State<'_, Gateway>,
) -> Result<SetupStateView, ApiErrorDto> {
    state
        .run_command("restart_setup", Failure::Local, |services, _| {
            services.settings.restart_setup()
        })
        .await
}

#[tauri::command]
pub(crate) async fn get_permissions(
    state: tauri::State<'_, Gateway>,
) -> Result<PermissionsResponse, ApiErrorDto> {
    state.permissions().await
}

#[tauri::command]
pub(crate) async fn request_permission(
    request: PermissionRequest,
    state: tauri::State<'_, Gateway>,
) -> Result<PermissionsResponse, ApiErrorDto> {
    state.request_permission(request).await
}

#[tauri::command]
pub(crate) async fn open_permission_settings(
    request: PermissionRequest,
) -> Result<(), ApiErrorDto> {
    run_blocking(move || NativeServices::open_permission_settings(request.kind)).await?
}

/// `stop_recording` ends it, and its transcript is neither pasted nor kept.
#[tauri::command]
pub(crate) async fn start_test_dictation(
    state: tauri::State<'_, Gateway>,
) -> Result<(), ApiErrorDto> {
    state
        .run_command("start_test_dictation", Failure::Local, |services, _| {
            services
                .dictation
                .start(RecordingTrigger::SetupTest)
                .map(drop)
        })
        .await
}

#[tauri::command]
pub(crate) async fn stop_recording(state: tauri::State<'_, Gateway>) -> Result<(), ApiErrorDto> {
    state
        .run_command("stop_recording", Failure::Published, |services, _| {
            services.dictation.stop()
        })
        .await
}

#[tauri::command]
pub(crate) async fn cancel_recording(state: tauri::State<'_, Gateway>) -> Result<(), ApiErrorDto> {
    state
        .run_command("cancel_recording", Failure::Published, |services, _| {
            services.dictation.cancel()
        })
        .await
}
