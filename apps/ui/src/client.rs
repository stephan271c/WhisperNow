#![cfg_attr(
    not(target_arch = "wasm32"),
    allow(dead_code, reason = "natively only tests call the desktop")
)]

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use whispernow_domain::AppError;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod fake;

pub(crate) fn connection_error() -> AppError {
    AppError::new(
        whispernow_domain::ErrorCategory::General,
        "desktop_unreachable",
        "The desktop connection failed",
    )
}

fn contract_failed(command: &str, error: &impl std::fmt::Display) -> AppError {
    let message = format!("Desktop command {command} contract failed: {error}");
    #[cfg(target_arch = "wasm32")]
    wasm::console_error(&message);
    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("{message}");
    connection_error()
}

type Answer<R> = Result<R, AppError>;

pub(crate) async fn invoke<Q, R>(command: &'static str, request: Q) -> Answer<R>
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
        let request =
            serde_json::to_value(request).map_err(|error| contract_failed(command, &error))?;
        let answer = fake::invoke(command, request).await?;
        serde_json::from_value(answer).map_err(|error| contract_failed(command, &error))
    }
}

pub(crate) async fn call<R: DeserializeOwned>(command: &'static str) -> Answer<R> {
    invoke(command, ()).await
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

/// Natively there is no window to measure.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const fn viewport_height() -> Option<f64> {
    None
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum WindowFrame {
    /// The OS draws the whole frame, or there is no window at all.
    #[default]
    Native,
    /// The window is undecorated; the page draws the titlebar and its buttons.
    Drawn,
    /// macOS draws its buttons over the page; the page draws a drag strip under them.
    Overlay,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Platform {
    Windows,
    Macos,
    Linux,
    #[default]
    Other,
}

impl Platform {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Windows => "windows",
            Self::Macos => "macos",
            Self::Linux => "linux",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum WindowAction {
    Minimize,
    ToggleMaximize,
    Close,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn run_window_action(_action: WindowAction) {}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct ListenerOwner;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn listen_events(
    _callback: impl FnMut(whispernow_domain::ipc::EventPayload) + 'static,
) -> std::future::Ready<Answer<ListenerOwner>> {
    std::future::ready(Ok(ListenerOwner))
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn current_window_label() -> String {
    "settings".to_owned()
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn apply_theme(_theme: Option<&str>) {}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) const fn reset_settings_scroll() {}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) const fn track_scroll_activity() {}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn window_frame() -> std::future::Ready<WindowFrame> {
    std::future::ready(fake::frame().0)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn host_platform() -> Platform {
    fake::frame().1
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn listen_maximized(
    _callback: impl FnMut(bool) + 'static,
) -> std::future::Ready<Answer<ListenerOwner>> {
    std::future::ready(Ok(ListenerOwner))
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) const fn request_shortcut_capture(_capturing: bool) {}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn history_audio_url(
    _id: whispernow_domain::EntryId,
) -> std::future::Ready<Answer<String>> {
    std::future::ready(Err(AppError::new(
        whispernow_domain::ErrorCategory::General,
        "audio_unavailable",
        "Audio plays in the desktop app",
    )))
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) const fn revoke_object_url(_url: &str) {}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::cell::RefCell;

    use js_sys::Function;
    use serde::{Serialize, de::DeserializeOwned};
    use wasm_bindgen::{JsCast as _, closure::Closure, prelude::*};
    use whispernow_domain::EntryId;
    use whispernow_domain::ipc::EventPayload;

    use super::{
        Answer, AppError, Platform, WindowAction, WindowFrame, connection_error, contract_failed,
    };

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = console, js_name = error)]
        pub(super) fn console_error(message: &str);
    }

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

        #[wasm_bindgen(js_name = trackScrollActivity)]
        pub fn track_scroll_activity();

        #[wasm_bindgen(js_name = applyTheme)]
        pub fn apply_theme(theme: Option<&str>);

        #[wasm_bindgen(catch, js_name = windowFrame)]
        async fn window_frame_js() -> Result<JsValue, JsValue>;

        #[wasm_bindgen(js_name = hostPlatform)]
        fn host_platform_js() -> JsValue;

        #[wasm_bindgen(catch, js_name = minimizeWindow)]
        async fn minimize_window_js() -> Result<(), JsValue>;

        #[wasm_bindgen(catch, js_name = toggleMaximizeWindow)]
        async fn toggle_maximize_window_js() -> Result<(), JsValue>;

        #[wasm_bindgen(catch, js_name = closeWindow)]
        async fn close_window_js() -> Result<(), JsValue>;

        #[wasm_bindgen(catch, js_name = listenMaximized)]
        async fn listen_maximized_js(callback: &Function) -> Result<u32, JsValue>;
    }

    fn from_js(value: &JsValue) -> AppError {
        serde_wasm_bindgen::from_value(value.clone()).unwrap_or_else(|_| {
            console_error(&format!("Desktop call rejected: {}", rejection_text(value)));
            connection_error()
        })
    }

    fn rejection_text(value: &JsValue) -> String {
        value
            .as_string()
            .or_else(|| {
                value
                    .dyn_ref::<js_sys::Error>()
                    .map(|error| String::from(error.message()))
            })
            .or_else(|| js_sys::JSON::stringify(value).ok()?.as_string())
            .unwrap_or_else(|| format!("{value:?}"))
    }

    // Tauri sends arguments through `JSON.stringify`, which drops `undefined`, so `None` and `()`
    // must reach it as `null`.
    fn request_to_js(request: &impl Serialize) -> Result<JsValue, serde_wasm_bindgen::Error> {
        request.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
    }

    pub(super) async fn invoke<Q, R>(command: &str, request: Q) -> Answer<R>
    where
        Q: Serialize,
        R: DeserializeOwned,
    {
        let request = request_to_js(&request).map_err(|error| contract_failed(command, &error))?;
        let response = invoke_typed(command, request)
            .await
            .map_err(|error| from_js(&error))?;
        serde_wasm_bindgen::from_value(response).map_err(|error| contract_failed(command, &error))
    }

    // Make the URL after the last await, in the same step that hands it to the player.
    pub(crate) async fn history_audio_url(id: EntryId) -> Answer<String> {
        let request =
            request_to_js(&id).map_err(|error| contract_failed("get_history_audio", &error))?;
        let audio = audio_blob("get_history_audio", request)
            .await
            .map_err(|error| from_js(&error))?;
        Ok(object_url(&audio))
    }

    pub async fn listen_events(
        callback: impl FnMut(EventPayload) + 'static,
    ) -> Result<ListenerOwner, AppError> {
        let mut callback = callback;
        let closure = Closure::wrap(Box::new(move |value: JsValue| {
            match serde_wasm_bindgen::from_value::<EventPayload>(value) {
                Ok(event) => callback(event),
                Err(error) => console_error(&format!("Desktop event contract failed: {error}")),
            }
        }) as Box<dyn FnMut(JsValue)>);
        let listener_id = listen_typed(
            whispernow_domain::ipc::EVENT,
            closure.as_ref().unchecked_ref::<Function>(),
        )
        .await
        .map_err(|error| from_js(&error))?;
        Ok(ListenerOwner {
            listener_id,
            _callback: closure,
        })
    }

    /// A window whose decoration cannot be read keeps its native frame.
    pub(crate) async fn window_frame() -> WindowFrame {
        let frame = window_frame_js().await.unwrap_or(JsValue::NULL);
        serde_wasm_bindgen::from_value(frame).unwrap_or_default()
    }

    pub(crate) fn host_platform() -> Platform {
        serde_wasm_bindgen::from_value(host_platform_js()).unwrap_or_default()
    }

    pub(crate) fn run_window_action(action: WindowAction) {
        dioxus::prelude::spawn(async move {
            // A refused window action leaves the window as it was, which the user can see.
            let _ = match action {
                WindowAction::Minimize => minimize_window_js().await,
                WindowAction::ToggleMaximize => toggle_maximize_window_js().await,
                WindowAction::Close => close_window_js().await,
            };
        });
    }

    /// Reports whether this window is maximized, now and after each resize.
    pub(crate) async fn listen_maximized(
        callback: impl FnMut(bool) + 'static,
    ) -> Result<ListenerOwner, AppError> {
        let mut callback = callback;
        let closure = Closure::wrap(Box::new(move |value: JsValue| {
            if let Some(maximized) = value.as_bool() {
                callback(maximized);
            }
        }) as Box<dyn FnMut(JsValue)>);
        let listener_id = listen_maximized_js(closure.as_ref().unchecked_ref::<Function>())
            .await
            .map_err(|error| from_js(&error))?;
        Ok(ListenerOwner {
            listener_id,
            _callback: closure,
        })
    }

    /// A desktop listener and its callback; dropping it unregisters the listener.
    pub struct ListenerOwner {
        listener_id: u32,
        _callback: Closure<dyn FnMut(JsValue)>,
    }

    impl Drop for ListenerOwner {
        fn drop(&mut self) {
            unlisten_typed(self.listener_id);
        }
    }

    pub(crate) async fn delay(ms: u32) {
        let _ = delay_js(ms).await;
    }

    pub(crate) fn document_hidden() -> bool {
        let global = js_sys::global();
        js_sys::Reflect::get(&global, &JsValue::from_str("document"))
            .and_then(|document| js_sys::Reflect::get(&document, &JsValue::from_str("hidden")))
            .ok()
            .and_then(|hidden| hidden.as_bool())
            .unwrap_or(false)
    }

    pub(crate) fn viewport_height() -> Option<f64> {
        js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("innerHeight"))
            .ok()
            .and_then(|height| height.as_f64())
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
                super::call::<()>("suspend_shortcuts").await
            } else {
                super::call::<()>("resume_shortcuts").await
            };
            SHORTCUT_CAPTURE.with_borrow_mut(|sync| sync.applied = target);
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm::*;
