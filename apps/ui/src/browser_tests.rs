use dioxus::prelude::*;
use serde::{Serialize, de::DeserializeOwned};
use std::fmt::Debug;
use wasm_bindgen::{JsCast as _, JsValue};
use wasm_bindgen_test::*;
use whispernow_domain::ipc::{ModelSummary, Phase};
use whispernow_domain::{
    AudioDeviceId, LlmProvider, ModelAvailability, ModelId, ProviderEndpoint, SettingChange,
    Settings, SetupCheckpoint, Shortcut, ShortcutKey, ShortcutModifier, VocabularyReplacement,
};

use crate::components::ui::{Field, Select};
use crate::test_fixtures::{management_with, snapshot};
use crate::{AppShell, ManagementState, Navigator, Route, UiContext, UiState, client};

wasm_bindgen_test_configure!(run_in_browser);

mod browser_support {
    include!("../tests/browser_support/mod.rs");
}
use browser_support::{
    adapter_count, eval_bool, install_mount_point, install_tauri, render_turn, wait_for,
};

fn browser_surface(management: fn() -> ManagementState) -> Element {
    let state = use_signal(|| {
        let mut state = UiState::default();
        state.hydrate(snapshot(Phase::Idle));
        state
    });
    let management = use_signal(management);
    let navigator = Navigator::new(use_signal(|| Route::Home), use_signal(|| 0));
    use_context_provider(|| UiContext {
        state,
        management,
        navigator,
    });
    rsx! { AppShell {} }
}

#[component]
fn BrowserSettingsSurface() -> Element {
    browser_surface(management_with_history)
}

#[component]
fn BrowserVocabularySurface() -> Element {
    browser_surface(|| management_with(vocabulary_settings(&VOCABULARY)))
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
async fn a_deleted_rules_focus_request_does_not_pull_focus_from_the_search_when_its_row_returns() {
    install_tauri_saving(&vocabulary_settings(&VOCABULARY[1..]));
    install_mount_point();
    dioxus::launch(BrowserVocabularySurface);
    wait_for("the settings shell", || rendered("nav-settings")).await;
    click("nav-settings");
    wait_for("the settings pages", || rendered("nav-settings-vocabulary")).await;
    click("nav-settings-vocabulary");
    wait_for("the rules", || rendered("vocabulary-rule-3")).await;

    click("vocabulary-rule-1-delete");
    wait_for("the delete to save", || {
        adapter_count("update_settings") == 1
    })
    .await;
    wait_for("the first rule to take focus", || {
        focused_id() == "vocabulary-rule-1"
    })
    .await;

    type_in_search("gamma");
    wait_for("the filter to hide the first rule", || {
        !rendered("vocabulary-rule-1")
    })
    .await;
    type_in_search("");
    wait_for("the first rule to return", || rendered("vocabulary-rule-1")).await;
    for _ in 0..10 {
        render_turn().await;
    }
    assert_eq!(focused_id(), "vocabulary-search");
}

#[component]
fn EdgeSelect() -> Element {
    rsx! {
        Field { id: "edge", label: "Edge",
            Select { value: 0_usize,
                options: (0..8).map(|index| (index, format!("Option {index}"))).collect::<Vec<_>>(),
                onchange: move |_: usize| {},
            }
        }
    }
}

#[wasm_bindgen_test(async)]
async fn an_open_list_fits_the_room_left_without_writing_a_style_attribute() {
    js_sys::eval(
        r#"
        window.__styleAttributes = [];
        window.__setAttribute = Element.prototype.setAttribute;
        Element.prototype.setAttribute = function (name, value) {
          if (name === "style") window.__styleAttributes.push(String(value));
          return window.__setAttribute.call(this, name, value);
        };
        window.__innerHeight = Object.getOwnPropertyDescriptor(window, "innerHeight");
        Object.defineProperty(window, "innerHeight", { value: 120, configurable: true });
        "#,
    )
    .expect("short window");
    install_mount_point();
    dioxus::launch(EdgeSelect);
    wait_for("the select", || rendered("edge")).await;
    click("edge");
    wait_for("the list to be placed", || {
        eval_bool("!!document.querySelector('#edge-listbox[data-placement]')")
    })
    .await;
    let limit =
        eval_string("document.getElementById('edge-listbox').style.getPropertyValue('max-height')");
    let height = eval_string(
        "String(document.getElementById('edge-listbox').getBoundingClientRect().height)",
    );
    let written = eval_string("JSON.stringify(window.__styleAttributes)");
    js_sys::eval(
        r#"
        Element.prototype.setAttribute = window.__setAttribute;
        if (window.__innerHeight) Object.defineProperty(window, "innerHeight", window.__innerHeight);
        else delete window.innerHeight;
        "#,
    )
    .expect("restore the window");

    assert_eq!(written, "[]", "the CSP forbids inline styles");
    let room = limit
        .strip_prefix("min(18rem, ")
        .and_then(|rest| rest.strip_suffix("px)"))
        .and_then(|room| room.parse::<f64>().ok())
        .unwrap_or_else(|| panic!("a limit of the room left: {limit:?}"));
    let height = height.parse::<f64>().expect("list height");
    assert!(height <= room + 0.5, "{height}px list in {room}px");
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

#[wasm_bindgen_test(async)]
async fn requests_holding_none_reach_the_desktop_intact() {
    install_tauri("return null;");
    let device = AudioDeviceId::new("usb-mic").expect("device id");
    let endpoint = ProviderEndpoint::parse("https://example.test/v1").expect("endpoint");

    for change in [
        SettingChange::InputDevice(None),
        SettingChange::InputDevice(Some(device.clone())),
        SettingChange::CancelShortcut(None),
        SettingChange::CancelShortcut(Some(Shortcut::default_cancel())),
        SettingChange::ProviderEndpoint {
            provider: LlmProvider::Custom,
            api_base: None,
        },
        SettingChange::ProviderEndpoint {
            provider: LlmProvider::Custom,
            api_base: Some(endpoint),
        },
    ] {
        assert_reaches_desktop("update_settings", change).await;
    }
    for input_device in [None, Some(device)] {
        let checkpoint = SetupCheckpoint {
            input_device,
            ..SetupCheckpoint::default()
        };
        assert_reaches_desktop("save_setup_checkpoint", checkpoint).await;
    }
}

#[wasm_bindgen_test(async)]
async fn a_command_without_input_sends_no_request_and_null_answers_read_as_unit_and_none() {
    install_tauri("return null;");

    let none: Option<String> = client::call("get_state")
        .await
        .expect("a null answer is None");
    client::invoke::<_, ()>("copy_text", "text")
        .await
        .expect("a null answer is the unit");

    assert_eq!(none, None);
    let calls = adapter_calls();
    assert!(
        calls.contains(r#"{"command":"get_state","payload":{}}"#),
        "{calls}"
    );
}

async fn assert_reaches_desktop<T>(command: &'static str, request: T)
where
    T: Serialize + DeserializeOwned + PartialEq + Debug,
{
    client::invoke::<_, ()>(command, &request)
        .await
        .expect("the desktop answers");
    let sent = js_sys::eval(&format!(
        "JSON.stringify(window.__whispernowAdapterCalls.filter(call => call.command === {command:?}).at(-1).payload.request)"
    ))
    .expect("read the sent request")
        .as_string()
        .unwrap_or_else(|| panic!("{request:?} must reach {command} as a request"));
    let received: T = serde_json::from_str(&sent)
        .unwrap_or_else(|error| panic!("{command} cannot read {sent}: {error}"));
    assert_eq!(received, request);
}

fn adapter_calls() -> String {
    js_sys::eval("JSON.stringify(window.__whispernowAdapterCalls)")
        .expect("serialize adapter calls")
        .as_string()
        .expect("adapter call JSON")
}

/// Answers every settings save with the surface's unchanged settings.
fn install_tauri_probe() {
    install_tauri_saving(&browser_settings());
}

/// Answers every settings save with `settings`.
fn install_tauri_saving(settings: &Settings) {
    let settings = serde_json::to_string(settings).expect("settings JSON");
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

fn eval_string(script: &str) -> String {
    js_sys::eval(script)
        .expect("evaluate probe script")
        .as_string()
        .expect("string probe result")
}

fn focused_id() -> String {
    eval_string("document.activeElement?.id ?? ''")
}

/// Types into the search field as a user does: focus it, then change its text.
fn type_in_search(text: &str) {
    js_sys::eval(&format!(
        "const search = document.getElementById('vocabulary-search'); search.focus(); search.value = {text:?}; search.dispatchEvent(new Event('input', {{ bubbles: true }}));"
    ))
    .expect("type in the search");
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

const VOCABULARY: [(&str, &str); 3] = [("alpha", "1"), ("beta", "2"), ("gamma", "3")];

fn vocabulary_settings(rules: &[(&str, &str)]) -> Settings {
    Settings {
        vocabulary_replacements: rules
            .iter()
            .map(|(source, replacement)| VocabularyReplacement::new(*source, *replacement))
            .collect(),
        ..browser_settings()
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
