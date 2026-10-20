use dioxus::prelude::*;
use wasm_bindgen::{JsCast as _, JsValue};
use wasm_bindgen_test::*;
use whispernow_domain::ipc::{AppState, ModelState, ModelSummary, Phase, SettingsView};
use whispernow_domain::{
    ModelAvailability, ModelId, Settings, Shortcut, ShortcutKey, ShortcutModifier,
};

use crate::{AppShell, ManagementState, Navigator, Route, UiContext, UiState};

wasm_bindgen_test_configure!(run_in_browser);

mod browser_support {
    include!("../tests/browser_support/mod.rs");
}
use browser_support::{install_mount_point, wait_for};

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
async fn shortcut_capture_suspends_global_shortcuts_and_saves_recording_and_cancel_keys() {
    install_tauri_probe();
    install_mount_point();
    dioxus::launch(BrowserSettingsSurface);
    wait_for("the settings shell", || rendered("nav-settings")).await;
    click("nav-settings");
    wait_for("the general settings page", || {
        rendered("recording-shortcut")
    })
    .await;

    for (id, key, code, control, expected) in [
        (
            "recording-shortcut",
            "r",
            "KeyR",
            true,
            Shortcut {
                modifiers: [ShortcutModifier::Control].into(),
                key: ShortcutKey::R,
            },
        ),
        (
            "cancel-shortcut",
            "F8",
            "F8",
            false,
            Shortcut {
                modifiers: std::collections::BTreeSet::default(),
                key: ShortcutKey::F8,
            },
        ),
    ] {
        let before = adapter_count("update_settings");
        let suspended = adapter_count("suspend_shortcuts");
        let resumed = adapter_count("resume_shortcuts");
        click(id);
        wait_for("global shortcuts to suspend", || {
            adapter_count("suspend_shortcuts") > suspended
        })
        .await;
        js_sys::eval(&format!(
            "document.getElementById({id:?}).dispatchEvent(new KeyboardEvent('keydown', {{key: {key:?}, code: {code:?}, ctrlKey: {control}, bubbles: true}}))"
        )).expect("capture shortcut");
        wait_for("the captured shortcut to save", || {
            adapter_count("update_settings") > before
        })
        .await;
        wait_for("global shortcuts to resume", || {
            adapter_count("resume_shortcuts") > resumed
        })
        .await;
        let saved = js_sys::eval("JSON.stringify(window.__whispernowAdapterCalls.filter(({command}) => command === 'update_settings').at(-1).payload.request)")
            .expect("saved settings").as_string().expect("settings JSON");
        let saved: Settings = serde_json::from_str(&saved).expect("settings request");
        if id == "recording-shortcut" {
            assert_eq!(saved.recording_shortcut, expected);
        } else {
            assert_eq!(saved.cancel_shortcut, Some(expected));
        }
    }
}

fn adapter_count(command: &str) -> u32 {
    let count = js_sys::eval(&format!(
        "window.__whispernowAdapterCalls.filter(call => call.command === {command:?}).length"
    ))
    .expect("count adapter calls");
    serde_wasm_bindgen::from_value(count).expect("integer call count")
}

#[wasm_bindgen_test(async)]
async fn a_history_copy_reaches_the_desktop_copy_command() {
    install_tauri_probe();
    install_mount_point();
    dioxus::launch(BrowserSettingsSurface);
    wait_for("the settings shell to render", || rendered("nav-history")).await;

    click("nav-history");
    wait_for("the history page to render", || rendered("history-copy-7")).await;
    click("history-copy-7");
    wait_for("the copy to reach the desktop", || {
        adapter_count("copy_text") == 1
    })
    .await;
    let calls = adapter_calls();
    assert!(
        calls.contains(r#"{"command":"copy_text","payload":{"request":"cleaned 7"}}"#),
        "the pasted text is copied: {calls}"
    );
}

fn adapter_calls() -> String {
    js_sys::eval("JSON.stringify(window.__whispernowAdapterCalls)")
        .expect("serialize adapter calls")
        .as_string()
        .expect("adapter call JSON")
}

fn install_tauri_probe() {
    js_sys::eval(
        r#"
        window.__whispernowAdapterCalls = [];
        window.__TAURI__ = {
          core: {
            invoke: async (command, payload) => {
              window.__whispernowAdapterCalls.push({ command, payload });
              if (command === "update_settings") {
                return { settings: { settings: payload.request, warnings: [] }, warnings: [] };
              }
              if (command === "suspend_shortcuts" || command === "resume_shortcuts") {
                return null;
              }
              if (command === "copy_text") {
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

fn model_id() -> ModelId {
    ModelId::new("current-model").expect("fixture model id")
}

fn idle_snapshot() -> AppState {
    AppState {
        phase: Phase::Idle,
        selected_model_id: model_id(),
        model_state: ModelState::Ready,
        last_transcript: None,
        error: None,
        active_downloads: Vec::new(),
        setup_test: None,
    }
}

fn management_with_history() -> ManagementState {
    let mut management = ManagementState::default();
    management.install_settings(SettingsView {
        start_on_login: false,
        settings: Settings {
            model_id: model_id(),
            first_run_complete: true,
            ..Settings::default()
        },
        warnings: Vec::new(),
    });
    management.history_rows_filter = Some(whispernow_domain::HistoryFilter::default());
    let mut entry = crate::test_fixtures::history_entry(7, "raw 7");
    entry.cleaned_text = Some("cleaned 7".to_owned());
    management.history = vec![entry];
    management.models = vec![ModelSummary {
        id: model_id(),
        display_name: "Current model".to_owned(),
        family: whispernow_domain::ModelFamily::Whisper,
        availability: ModelAvailability::Downloaded,
        languages: vec!["en".to_owned()],
        size_bytes: 116_204_861,
        benchmark: None,
        recommended: false,
    }];
    management.models_loaded = true;
    management
}
