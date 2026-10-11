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
                modifiers: Default::default(),
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

#[wasm_bindgen_test(async)]
async fn a_delayed_shortcut_refusal_disables_other_saves_and_the_next_theme_save_uses_confirmed_keys()
 {
    install_tauri_probe();
    install_mount_point();
    dioxus::launch(BrowserSettingsSurface);
    wait_for("the settings shell", || rendered("nav-settings")).await;
    click("nav-settings");
    wait_for("the shortcut control", || rendered("recording-shortcut")).await;
    js_sys::eval("window.__whispernowDelaySettings = true").expect("delay settings saves");
    click("recording-shortcut");
    wait_for("global shortcuts to suspend", || {
        adapter_count("suspend_shortcuts") == 1.0
    })
    .await;
    js_sys::eval("document.getElementById('recording-shortcut').dispatchEvent(new KeyboardEvent('keydown', {key: 'r', code: 'KeyR', ctrlKey: true, bubbles: true}))").expect("capture shortcut");
    wait_for("the shortcut save and disabled theme control", || {
        adapter_count("update_settings") == 1.0 && disabled("theme")
    })
    .await;
    for id in [
        "recording-shortcut",
        "cancel-shortcut",
        "recording-mode-toggle",
        "cancel-dictation",
        "start-minimized",
        "start-on-login",
    ] {
        assert!(disabled(id), "{id} remains editable during a document save");
    }
    click("start-minimized");
    assert_eq!(adapter_count("update_settings"), 1.0);
    js_sys::eval("window.__whispernowPendingSettings.shift().reject({category: 'settings', code: 'shortcut_refused', summary: 'The recording shortcut is taken'})").expect("refuse shortcut");
    wait_for("the field error and restored controls", || {
        rendered("recording-mode-error") && !disabled("theme")
    })
    .await;
    assert!(
        web_sys::window()
            .expect("window")
            .document()
            .expect("document")
            .get_element_by_id("recording-shortcut-keys")
            .expect("shortcut keys")
            .text_content()
            .expect("shortcut label")
            .contains("Space")
    );
    js_sys::eval("window.__whispernowDelaySettings = false; const theme = document.getElementById('theme'); theme.value = 'dark'; theme.dispatchEvent(new Event('change', {bubbles: true}))").expect("choose dark theme");
    wait_for("the theme to save", || {
        adapter_count("update_settings") == 2.0 && !disabled("theme")
    })
    .await;
    let saved = js_sys::eval("JSON.stringify(window.__whispernowAdapterCalls.filter(({command}) => command === 'update_settings').at(-1).payload.request)").expect("saved settings").as_string().expect("settings JSON");
    let saved: Settings = serde_json::from_str(&saved).expect("settings request");
    assert_eq!(saved.recording_shortcut.key, ShortcutKey::Space);
    assert_eq!(saved.theme, whispernow_domain::ThemePreference::Dark);
    assert!(rendered("recording-mode-error"));
}

fn disabled(id: &str) -> bool {
    js_sys::eval(&format!("document.getElementById({id:?}).disabled"))
        .expect("disabled property")
        .as_bool()
        .expect("boolean disabled property")
}

fn adapter_count(command: &str) -> f64 {
    js_sys::eval(&format!(
        "window.__whispernowAdapterCalls.filter(call => call.command === {command:?}).length"
    ))
    .expect("count adapter calls")
    .as_f64()
    .expect("call count")
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

fn install_tauri_probe() {
    js_sys::eval(
        r#"
        window.__whispernowAdapterCalls = [];
        window.__whispernowDelaySettings = false;
        window.__whispernowPendingSettings = [];
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
              if (command === "update_settings") {
                if (window.__whispernowDelaySettings) {
                  return await new Promise((resolve, reject) => window.__whispernowPendingSettings.push({resolve, reject}));
                }
                return { settings: { settings: payload.request, warnings: [] }, warnings: [] };
              }
              if (command === "suspend_shortcuts" || command === "resume_shortcuts") {
                return null;
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
        speed: whispernow_domain::Rating::new(5),
        accuracy: whispernow_domain::Rating::new(1),
        benchmark: None,
        recommended: false,
    }];
    management.models_loaded = true;
    management
}
