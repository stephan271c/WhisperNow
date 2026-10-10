//! Tauri delivers events from JavaScript with no Dioxus scope on the stack. An event that
//! calls the desktop must still reach the backend instead of panicking the UI.
#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

mod browser_support;
use browser_support::{install_mount_point, render_turn, wait_for};

#[wasm_bindgen_test(async)]
async fn history_changed_event_from_tauri_refreshes_history() {
    install_tauri_probe();
    install_mount_point();
    dioxus::launch(whispernow_ui::app);
    wait_for("the history-changed listener", || {
        eval_bool(&format!(
            "window.__whispernowListeners.has({:?})",
            whispernow_domain::ipc::HISTORY_CHANGED_EVENT
        ))
    })
    .await;
    // Let startup commands settle so only the event's refresh is counted below.
    settle().await;
    let before = history_requests();
    let event = serde_json::to_string(&whispernow_domain::ipc::EventPayload::HistoryChanged)
        .expect("history event");

    let threw = eval_bool(&format!(
        r#"(() => {{
          try {{
            window.__whispernowListeners.get({:?})({{ payload: {event} }});
            return false;
          }} catch (error) {{
            console.error(error);
            return true;
          }}
        }})()"#,
        whispernow_domain::ipc::HISTORY_CHANGED_EVENT
    ));

    assert!(!threw, "the Tauri event callback must not panic");
    wait_for("the history refresh to reach invokeTyped", || {
        history_requests() > before
    })
    .await;
}

fn install_tauri_probe() {
    js_sys::eval(
        r#"
        window.__whispernowAdapterCalls = [];
        window.__whispernowListeners = new Map();
        window.__TAURI__ = {
          core: {
            invoke: async (command, payload) => {
              window.__whispernowAdapterCalls.push({ command, payload });
              throw new Error(`unavailable in this test: ${command}`);
            }
          },
          event: {
            listen: async (name, handler) => {
              window.__whispernowListeners.set(name, handler);
              return () => window.__whispernowListeners.delete(name);
            }
          },
          window: { getCurrentWindow: () => ({ label: "settings" }) }
        };
        "#,
    )
    .expect("install Tauri probe");
}

fn history_requests() -> f64 {
    js_sys::eval(
        r#"window.__whispernowAdapterCalls.filter(({ command }) => command === "get_history").length"#,
    )
    .expect("count history requests")
    .as_f64()
    .expect("history request count")
}

fn eval_bool(script: &str) -> bool {
    js_sys::eval(script)
        .expect("evaluate probe script")
        .as_bool()
        .expect("boolean probe result")
}

async fn settle() {
    for _ in 0..20 {
        render_turn().await;
    }
}
