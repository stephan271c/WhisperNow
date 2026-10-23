use dioxus::prelude::*;
use wasm_bindgen::{JsCast as _, JsValue};
use wasm_bindgen_test::*;
use whispernow_domain::ipc::{ModelSummary, Phase};
use whispernow_domain::{
    ModelAvailability, ModelId, SettingChange, Settings, Shortcut, ShortcutKey, ShortcutModifier,
};

use crate::test_fixtures::{management_with, snapshot};
use crate::{AppShell, ManagementState, Navigator, Route, UiContext, UiState};

wasm_bindgen_test_configure!(run_in_browser);

mod browser_support {
    include!("../tests/browser_support/mod.rs");
}
use browser_support::{adapter_count, eval_bool, install_mount_point, install_tauri, wait_for};

#[component]
fn BrowserSettingsSurface() -> Element {
    let state = use_signal(|| {
        let mut state = UiState::default();
        state.hydrate(snapshot(Phase::Idle));
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
        let sent = js_sys::eval("JSON.stringify(window.__whispernowAdapterCalls.filter(({command}) => command === 'update_settings').at(-1).payload.request)")
            .expect("sent change").as_string().expect("change JSON");
        let sent: SettingChange = serde_json::from_str(&sent).expect("setting change");
        let settings = browser_settings();
        let expected = if id == "recording-shortcut" {
            SettingChange::RecordingShortcut {
                shortcut: expected,
                mode: settings.recording_mode,
            }
        } else {
            SettingChange::CancelShortcut(Some(expected))
        };
        assert_eq!(sent, expected);
    }
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

/// Answers every settings save with the surface's unchanged settings.
fn install_tauri_probe() {
    let settings = serde_json::to_string(&browser_settings()).expect("settings JSON");
    install_tauri(&format!(
        r#"
        if (command === "update_settings") {{
          return {{ settings: {{ settings: {settings}, warnings: [] }}, warnings: [] }};
        }}
        if (command === "suspend_shortcuts" || command === "resume_shortcuts") {{
          return null;
        }}
        if (command === "copy_text") {{
          return null;
        }}
        throw new Error(`unexpected command: ${{command}}`);
        "#,
    ));
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
    eval_bool(&format!("!!document.getElementById({id:?})"))
}

fn model_id() -> ModelId {
    snapshot(Phase::Idle).selected_model_id
}

fn browser_settings() -> Settings {
    Settings {
        model_id: model_id(),
        setup: whispernow_domain::SetupProgress::Complete,
        ..Settings::default()
    }
}

fn management_with_history() -> ManagementState {
    let mut management = management_with(browser_settings());
    let mut entry = crate::test_fixtures::history_entry(7, "raw 7");
    entry.cleaned_text = Some("cleaned 7".to_owned());
    management.finish_history_refresh(whispernow_domain::HistoryPage {
        entries: vec![entry],
        has_more: false,
    });
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
