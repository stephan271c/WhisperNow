//! One async function per desktop command, over a single transport: Tauri's `invoke` in the
//! browser, and a scripted fake in native tests.
#![cfg_attr(
    not(target_arch = "wasm32"),
    allow(dead_code, reason = "natively only tests call the desktop")
)]

use serde::{Serialize, de::DeserializeOwned};
use whispernow_api::{
    ApiErrorDto, AppStateDto, ClearApiKeyRequest, ClearDataRequest, ClearDataResult,
    DataLocationDto, DeleteHistoryEntryRequest, HistoryFilter, HistoryPage, HistoryPageRequest,
    ListAudioDevicesResponse, ListModelsResponse, MicrophoneTestResponse, ModelId,
    OpenMainWindowRequest, PermissionRequest, PermissionsResponse, SaveSettingsResponse,
    SetApiKeyRequest, SettingsView, SetupCheckpoint, SetupStateView, TestPromptRequest,
    TestPromptResponse, UpdateSettingsRequest, UpdateStateDto,
};

#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod fake;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClientError {
    Api(ApiErrorDto),
    /// The call or its answer could not cross the bridge.
    Transport,
}

impl ClientError {
    #[must_use]
    pub const fn api_error(&self) -> Option<&ApiErrorDto> {
        match self {
            Self::Api(error) => Some(error),
            Self::Transport => None,
        }
    }

    #[must_use]
    pub fn summary(&self) -> String {
        match self {
            Self::Api(error) => error.summary.clone(),
            Self::Transport => "The desktop connection failed".to_owned(),
        }
    }
}

type Answer<R> = Result<R, ClientError>;

/// Invokes `command` with `request`, or with no input when it takes none.
async fn invoke<Q, R>(command: &'static str, request: Option<&Q>) -> Answer<R>
where
    Q: Serialize,
    R: DeserializeOwned,
{
    #[cfg(target_arch = "wasm32")]
    {
        wasm::invoke(command, request).await
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let request = request.map(|request| serde_json::to_value(request).unwrap_or_default());
        let answer = fake::invoke(command, request).await?;
        serde_json::from_value(answer).map_err(|_| ClientError::Transport)
    }
}

async fn call<R: DeserializeOwned>(command: &'static str) -> Answer<R> {
    invoke::<(), R>(command, None).await
}

pub(crate) async fn get_state() -> Answer<AppStateDto> {
    call("get_state").await
}

pub(crate) async fn get_settings() -> Answer<SettingsView> {
    call("get_settings").await
}

pub(crate) async fn update_settings(
    request: &UpdateSettingsRequest,
) -> Answer<SaveSettingsResponse> {
    invoke("update_settings", Some(request)).await
}

pub(crate) async fn set_api_key(request: &SetApiKeyRequest) -> Answer<SettingsView> {
    invoke("set_api_key", Some(request)).await
}

pub(crate) async fn clear_api_key(request: &ClearApiKeyRequest) -> Answer<SettingsView> {
    invoke("clear_api_key", Some(request)).await
}

pub(crate) async fn test_prompt(request: &TestPromptRequest) -> Answer<TestPromptResponse> {
    invoke("test_prompt", Some(request)).await
}

pub(crate) async fn list_audio_devices() -> Answer<ListAudioDevicesResponse> {
    call("list_audio_devices").await
}

pub(crate) async fn test_microphone() -> Answer<MicrophoneTestResponse> {
    call("test_microphone").await
}

pub(crate) async fn list_models() -> Answer<ListModelsResponse> {
    call("list_models").await
}

pub(crate) async fn download_model(model_id: &ModelId) -> Answer<()> {
    invoke("download_model", Some(model_id)).await
}

pub(crate) async fn cancel_download(model_id: &ModelId) -> Answer<()> {
    invoke("cancel_download", Some(model_id)).await
}

pub(crate) async fn select_model(model_id: &ModelId) -> Answer<()> {
    invoke("select_model", Some(model_id)).await
}

pub(crate) async fn delete_model(model_id: &ModelId) -> Answer<()> {
    invoke("delete_model", Some(model_id)).await
}

pub(crate) async fn get_setup_state() -> Answer<SetupStateView> {
    call("get_setup_state").await
}

pub(crate) async fn save_setup_checkpoint(checkpoint: &SetupCheckpoint) -> Answer<SetupStateView> {
    invoke("save_setup_checkpoint", Some(checkpoint)).await
}

pub(crate) async fn complete_setup(checkpoint: &SetupCheckpoint) -> Answer<SaveSettingsResponse> {
    invoke("complete_setup", Some(checkpoint)).await
}

pub(crate) async fn restart_setup() -> Answer<SetupStateView> {
    call("restart_setup").await
}

pub(crate) async fn get_permissions() -> Answer<PermissionsResponse> {
    call("get_permissions").await
}

pub(crate) async fn request_permission(request: &PermissionRequest) -> Answer<PermissionsResponse> {
    invoke("request_permission", Some(request)).await
}

pub(crate) async fn open_permission_settings(request: &PermissionRequest) -> Answer<()> {
    invoke("open_permission_settings", Some(request)).await
}

pub(crate) async fn start_test_dictation() -> Answer<()> {
    call("start_test_dictation").await
}

pub(crate) async fn stop_recording() -> Answer<()> {
    call("stop_recording").await
}

pub(crate) async fn cancel_recording() -> Answer<()> {
    call("cancel_recording").await
}

pub(crate) async fn dismiss_recording_overlay() -> Answer<()> {
    call("dismiss_recording_overlay").await
}

pub(crate) async fn copy_last_transcript() -> Answer<()> {
    call("copy_last_transcript").await
}

pub(crate) async fn open_main_window(request: &OpenMainWindowRequest) -> Answer<()> {
    invoke("open_main_window", Some(request)).await
}

pub(crate) async fn get_history(request: &HistoryPageRequest) -> Answer<HistoryPage> {
    invoke("get_history", Some(request)).await
}

/// The newest `pages` pages matching `filter`, read as one page.
pub(crate) async fn history_window(pages: u32, filter: HistoryFilter) -> Answer<HistoryPage> {
    get_history(&HistoryPageRequest {
        limit: pages.saturating_mul(HistoryPageRequest::DEFAULT_PAGE_SIZE),
        ..HistoryPageRequest::first_page(filter)
    })
    .await
}

pub(crate) async fn open_recordings_folder() -> Answer<()> {
    call("open_recordings_folder").await
}

pub(crate) async fn delete_history_entry(request: &DeleteHistoryEntryRequest) -> Answer<()> {
    invoke("delete_history_entry", Some(request)).await
}

pub(crate) async fn clear_history() -> Answer<()> {
    call("clear_history").await
}

pub(crate) async fn data_locations() -> Answer<Vec<DataLocationDto>> {
    call("data_locations").await
}

pub(crate) async fn clear_data(request: &ClearDataRequest) -> Answer<ClearDataResult> {
    invoke("clear_data", Some(request)).await
}

pub(crate) async fn suspend_shortcuts() -> Answer<()> {
    call("suspend_shortcuts").await
}

pub(crate) async fn resume_shortcuts() -> Answer<()> {
    call("resume_shortcuts").await
}

pub(crate) async fn get_update_state() -> Answer<UpdateStateDto> {
    call("get_update_state").await
}

pub(crate) async fn check_for_updates() -> Answer<UpdateStateDto> {
    call("check_for_updates").await
}

pub(crate) async fn dismiss_update() -> Answer<UpdateStateDto> {
    call("dismiss_update").await
}

pub(crate) async fn restart_to_update() -> Answer<UpdateStateDto> {
    call("restart_to_update").await
}

pub(crate) async fn open_update_release_page() -> Answer<UpdateStateDto> {
    call("open_update_release_page").await
}

/// Waits `ms` milliseconds; native tests run every waiting timer with [`fake::run_timers`].
#[cfg(not(target_arch = "wasm32"))]
pub(crate) async fn delay(_ms: u32) {
    fake::delay().await;
}

/// Natively no window is ever hidden.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const fn document_hidden() -> bool {
    false
}

pub(crate) async fn write_clipboard(text: &str) -> Answer<()> {
    #[cfg(target_arch = "wasm32")]
    {
        wasm::write_clipboard_text(text).await
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        invoke("write_clipboard", Some(&text)).await
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::cell::RefCell;

    use js_sys::Function;
    use serde::{Serialize, de::DeserializeOwned};
    use wasm_bindgen::{JsCast as _, closure::Closure, prelude::*};
    use whispernow_api::{EventPayload, GetHistoryAudioRequest};

    use super::{Answer, ClientError};

    #[wasm_bindgen(module = "/public/tauri_bridge.js")]
    extern "C" {
        #[wasm_bindgen(catch, js_name = invokeTyped)]
        async fn invoke_typed(command: &str, request: JsValue) -> Result<JsValue, JsValue>;

        #[wasm_bindgen(catch, js_name = audioBlob)]
        async fn audio_blob(command: &str, request: JsValue) -> Result<JsValue, JsValue>;

        #[wasm_bindgen(js_name = objectUrl)]
        fn object_url(blob: &JsValue) -> String;

        #[wasm_bindgen(js_name = revokeObjectUrl)]
        pub fn revoke_object_url(url: &str);

        #[wasm_bindgen(catch, js_name = writeClipboardText)]
        async fn write_clipboard_text_js(text: &str) -> Result<(), JsValue>;

        #[wasm_bindgen(catch, js_name = listenTyped)]
        async fn listen_typed(name: &str, callback: &Function) -> Result<u32, JsValue>;

        #[wasm_bindgen(js_name = unlistenTyped)]
        fn unlisten_typed(listener_id: u32);

        #[wasm_bindgen(js_name = currentWindowLabel)]
        fn current_window_label_js() -> String;

        #[wasm_bindgen(js_name = resetSettingsScroll)]
        pub fn reset_settings_scroll();

        #[wasm_bindgen(js_name = delay)]
        async fn delay_js(ms: u32) -> JsValue;

        #[wasm_bindgen(js_name = applyTheme)]
        pub fn apply_theme(theme: Option<String>);

        #[wasm_bindgen(catch, js_name = hasCustomTitlebar)]
        async fn has_custom_titlebar_js() -> Result<JsValue, JsValue>;

        #[wasm_bindgen(catch, js_name = minimizeWindow)]
        async fn minimize_window_js() -> Result<(), JsValue>;

        #[wasm_bindgen(catch, js_name = toggleMaximizeWindow)]
        async fn toggle_maximize_window_js() -> Result<(), JsValue>;

        #[wasm_bindgen(catch, js_name = closeWindow)]
        async fn close_window_js() -> Result<(), JsValue>;

        #[wasm_bindgen(catch, js_name = listenMaximized)]
        async fn listen_maximized_js(callback: &Function) -> Result<u32, JsValue>;
    }

    fn from_js(value: JsValue) -> ClientError {
        serde_wasm_bindgen::from_value(value).map_or(ClientError::Transport, ClientError::Api)
    }

    pub(super) async fn invoke<Q, R>(command: &str, request: Option<&Q>) -> Answer<R>
    where
        Q: Serialize,
        R: DeserializeOwned,
    {
        let request = match request {
            Some(request) => {
                serde_wasm_bindgen::to_value(request).map_err(|_| ClientError::Transport)?
            }
            None => JsValue::UNDEFINED,
        };
        let response = invoke_typed(command, request).await.map_err(from_js)?;
        serde_wasm_bindgen::from_value(response).map_err(|_| ClientError::Transport)
    }

    pub(super) async fn write_clipboard_text(text: &str) -> Answer<()> {
        write_clipboard_text_js(text).await.map_err(from_js)
    }

    /// A recording's retained audio held in this window. Dropping it only leaves bytes for the
    /// garbage collector; an object URL exists once [`Self::object_url`] makes one.
    #[derive(Debug)]
    pub struct RecordingAudio(JsValue);

    impl RecordingAudio {
        /// An object URL an `<audio>` element can play; revoke it with [`revoke_object_url`]
        /// once the player is gone.
        #[must_use]
        pub fn object_url(&self) -> String {
            object_url(&self.0)
        }
    }

    pub async fn history_audio(request: &GetHistoryAudioRequest) -> Answer<RecordingAudio> {
        let request = serde_wasm_bindgen::to_value(request).map_err(|_| ClientError::Transport)?;
        audio_blob("get_history_audio", request)
            .await
            .map(RecordingAudio)
            .map_err(from_js)
    }

    /// Only the recording bar may pass the partial transcript.
    pub async fn listen_all(
        event_names: &[&str],
        callback: impl FnMut(EventPayload) + 'static,
    ) -> Result<ListenerOwner, ClientError> {
        let mut callback = callback;
        let closure = Closure::wrap(Box::new(move |value: JsValue| {
            if let Ok(event) = serde_wasm_bindgen::from_value::<EventPayload>(value) {
                callback(event);
            }
        }) as Box<dyn FnMut(JsValue)>);
        let mut owner = ListenerOwner::new(closure);
        for event_name in event_names {
            let listener_id = listen_typed(
                event_name,
                owner.callback.as_ref().unchecked_ref::<Function>(),
            )
            .await
            .map_err(from_js)?;
            owner.listener_ids.push(listener_id);
        }
        Ok(owner)
    }

    /// Whether this window is undecorated and so draws its own titlebar. A window whose
    /// decoration cannot be read keeps its native frame, so the answer is `false`.
    pub(crate) async fn has_custom_titlebar() -> bool {
        has_custom_titlebar_js()
            .await
            .ok()
            .and_then(|custom| custom.as_bool())
            .unwrap_or(false)
    }

    pub(crate) async fn minimize_window() -> Answer<()> {
        minimize_window_js().await.map_err(from_js)
    }

    pub(crate) async fn toggle_maximize_window() -> Answer<()> {
        toggle_maximize_window_js().await.map_err(from_js)
    }

    /// Asks to close this window; the desktop hides it to the tray.
    pub(crate) async fn close_window() -> Answer<()> {
        close_window_js().await.map_err(from_js)
    }

    /// Reports whether this window is maximized, now and after each resize.
    pub(crate) async fn listen_maximized(
        callback: impl FnMut(bool) + 'static,
    ) -> Result<ListenerOwner, ClientError> {
        let mut callback = callback;
        let closure = Closure::wrap(Box::new(move |value: JsValue| {
            if let Some(maximized) = value.as_bool() {
                callback(maximized);
            }
        }) as Box<dyn FnMut(JsValue)>);
        let mut owner = ListenerOwner::new(closure);
        let listener_id = listen_maximized_js(owner.callback.as_ref().unchecked_ref::<Function>())
            .await
            .map_err(from_js)?;
        owner.listener_ids.push(listener_id);
        Ok(owner)
    }

    /// Desktop listeners and their callback; dropping it unregisters each listener.
    pub struct ListenerOwner {
        listener_ids: Vec<u32>,
        callback: Closure<dyn FnMut(JsValue)>,
    }

    impl ListenerOwner {
        const fn new(callback: Closure<dyn FnMut(JsValue)>) -> Self {
            Self {
                listener_ids: Vec::new(),
                callback,
            }
        }
    }

    impl Drop for ListenerOwner {
        fn drop(&mut self) {
            for listener_id in &self.listener_ids {
                unlisten_typed(*listener_id);
            }
        }
    }

    pub(crate) async fn delay(ms: u32) {
        let _ = delay_js(ms).await;
    }

    /// Whether the page is hidden, as when its window is hidden in the tray.
    pub(crate) fn document_hidden() -> bool {
        let global = js_sys::global();
        js_sys::Reflect::get(&global, &JsValue::from_str("document"))
            .and_then(|document| js_sys::Reflect::get(&document, &JsValue::from_str("hidden")))
            .ok()
            .and_then(|hidden| hidden.as_bool())
            .unwrap_or(false)
    }

    #[must_use]
    pub fn current_window_label() -> String {
        current_window_label_js()
    }

    /// The global-shortcut suspension the UI wants and the one last sent. One
    /// sender applies changes in request order, outside any component's
    /// lifetime, so a control unmounted mid-capture still restores shortcuts.
    #[derive(Default)]
    struct ShortcutCaptureSync {
        desired: bool,
        applied: bool,
        sending: bool,
    }

    thread_local! {
        static SHORTCUT_CAPTURE: RefCell<ShortcutCaptureSync> =
            RefCell::new(ShortcutCaptureSync::default());
    }

    /// Asks the desktop to suspend (`true`) or restore (`false`) its global
    /// shortcuts while a shortcut capture control is listening.
    pub(crate) fn request_shortcut_capture(capturing: bool) {
        let start_sender = SHORTCUT_CAPTURE.with_borrow_mut(|sync| {
            sync.desired = capturing;
            !std::mem::replace(&mut sync.sending, true)
        });
        if start_sender {
            wasm_bindgen_futures::spawn_local(send_shortcut_capture_changes());
        }
    }

    async fn send_shortcut_capture_changes() {
        loop {
            let target = SHORTCUT_CAPTURE.with_borrow_mut(|sync| {
                let target = (sync.desired != sync.applied).then_some(sync.desired);
                sync.sending = target.is_some();
                target
            });
            let Some(target) = target else {
                return;
            };
            // Both commands are idempotent, so a failed suspension is not
            // retried; a later restore request is still sent and harmless.
            let _ = if target {
                super::suspend_shortcuts().await
            } else {
                super::resume_shortcuts().await
            };
            SHORTCUT_CAPTURE.with_borrow_mut(|sync| sync.applied = target);
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm::*;
