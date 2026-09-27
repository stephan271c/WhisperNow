use dioxus::prelude::*;
use wasm_bindgen::{JsCast as _, JsValue};
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::*;
use whispernow_api::{AppStateDto, ModelAvailability, ModelDto, ModelId, ModelStateDto, PhaseDto};

use crate::{AppShell, ManagementState, Navigator, Route, UiContext, UiState};

wasm_bindgen_test_configure!(run_in_browser);

#[component]
fn BrowserSettingsSurface() -> Element {
    let state = use_signal(|| {
        let mut state = UiState::default();
        state.hydrate(idle_snapshot());
        state
    });
    let management = use_signal(management_with_history);
    let navigator = Navigator::new(use_signal(|| Route::Home), use_signal(|| 0));
    use_context_provider(|| UiContext {
        state,
        management,
        navigator,
    });
    rsx! { AppShell {} }
}

#[wasm_bindgen_test(async)]
async fn real_history_clicks_reach_tauri_and_clipboard_adapters() {
    install_tauri_probe();
    install_mount_point();
    dioxus::launch(BrowserSettingsSurface);
    wait_for("the settings shell to render", || rendered("nav-history")).await;

    click("nav-history");
    wait_for("the history page to render", || rendered("history-copy-7")).await;
    click("history-copy-7");
    wait_for("the copy to reach writeClipboardText", || {
        adapter_calls().contains("plugin:clipboard-manager|write_text")
    })
    .await;
    let calls = adapter_calls();
    assert!(
        calls.contains("cleaned 7"),
        "the pasted text is copied: {calls}"
    );

    click("history-delete-7");
    wait_for("the delete to ask for confirmation", || {
        rendered("history-delete-7-confirm")
    })
    .await;
    click("history-delete-7-confirm");
    wait_for("the delete to reach invoke", || {
        adapter_calls().contains("delete_history_entry")
    })
    .await;
}

fn adapter_calls() -> String {
    js_sys::eval("JSON.stringify(window.__whispernowAdapterCalls)")
        .expect("serialize adapter calls")
        .as_string()
        .expect("adapter call JSON")
}

fn install_mount_point() {
    let document = web_sys::window()
        .expect("window")
        .document()
        .expect("document");
    let mount = document.create_element("div").expect("create mount point");
    mount.set_id("main");
    document
        .document_element()
        .expect("document element")
        .append_child(&mount)
        .expect("append mount point");
}

fn install_tauri_probe() {
    js_sys::eval(
        r#"
        window.__whispernowAdapterCalls = [];
        window.__TAURI__ = {
          core: {
            invoke: async (command, payload) => {
              window.__whispernowAdapterCalls.push({ command, payload });
              if (command === "delete_history_entry") {
                return null;
              }
              if (command === "get_history") {
                return { entries: [], next_cursor: null };
              }
              if (command === "get_state") {
                return {
                  lifecycle: "ready",
                  workflowAdmission: "ready",
                  phase: "processing",
                  selectedModelId: "current-model",
                  modelState: "ready",
                  lastTranscript: null,
                  textOutputStatus: null,
                  error: null,
                  activeDownloads: []
                };
              }
              if (command === "plugin:clipboard-manager|write_text") {
                return null;
              }
              throw new Error(`unexpected command: ${command}`);
            }
          },
          event: { listen: async () => () => {} },
          window: { getCurrentWindow: () => ({ label: "settings" }) }
        };
        "#,
    )
    .expect("install Tauri probe");
}

fn click(id: &str) {
    let document = web_sys::window()
        .expect("window")
        .document()
        .expect("document");
    let element = document
        .get_element_by_id(id)
        .unwrap_or_else(|| panic!("rendered element {id}"));
    let click = js_sys::Reflect::get(&element, &JsValue::from_str("click"))
        .expect("read native click")
        .dyn_into::<js_sys::Function>()
        .expect("click function");
    click.call0(&element).expect("dispatch native click");
}

fn rendered(id: &str) -> bool {
    web_sys::window()
        .expect("window")
        .document()
        .expect("document")
        .get_element_by_id(id)
        .is_some()
}

/// Yields render turns until `condition` holds, failing after about two seconds.
async fn wait_for(what: &str, condition: impl Fn() -> bool) {
    for _ in 0..200 {
        if condition() {
            return;
        }
        let promise = js_sys::Promise::new(&mut |resolve, _reject| {
            web_sys::window()
                .expect("window")
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 10)
                .expect("schedule render turn");
        });
        JsFuture::from(promise).await.expect("render turn");
    }
    panic!("timed out waiting for {what}");
}

fn model_id() -> ModelId {
    ModelId::new("current-model").expect("fixture model id")
}

fn idle_snapshot() -> AppStateDto {
    AppStateDto {
        phase: PhaseDto::Idle,
        selected_model_id: model_id(),
        model_state: ModelStateDto::Ready,
        last_transcript: None,
        error: None,
        active_downloads: Vec::new(),
        overlay_style: whispernow_api::OverlayStyle::default(),
        overlay_position: whispernow_api::OverlayPosition::default(),
        setup_test: None,
    }
}

fn management_with_history() -> ManagementState {
    let mut management = ManagementState::default();
    management.history_rows_filter = Some(whispernow_api::HistoryFilter::default());
    let mut entry = crate::test_support::history_entry(7, "raw 7");
    entry.cleaned_text = Some("cleaned 7".to_owned());
    management.history = vec![entry];
    management.models = vec![ModelDto {
        id: model_id(),
        display_name: "Current model".to_owned(),
        family: whispernow_api::ModelFamily::Whisper,
        availability: ModelAvailability::Downloaded,
        languages: vec!["en".to_owned()],
        automatic_language_detection: true,
        size_bytes: 116_204_861,
        speed: whispernow_api::RelativeRating::High,
        accuracy: whispernow_api::RelativeRating::Low,
        recommended: false,
    }];
    management.models_loaded = true;
    management
}
