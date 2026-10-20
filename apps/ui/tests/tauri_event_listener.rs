//! Tauri delivers events from JavaScript with no Dioxus scope on the stack. An event that
//! calls the desktop must still reach the backend instead of panicking the UI.
#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

mod browser_support;
use browser_support::{
    adapter_count, eval_bool, install_mount_point, install_tauri, render_turn, wait_for,
};

#[wasm_bindgen_test(async)]
async fn history_changed_event_from_tauri_refreshes_history() {
    install_tauri("throw new Error(`unavailable in this test: ${command}`);");
    install_mount_point();
    dioxus::launch(whispernow_ui::app);
    wait_for("the event listener", || {
        eval_bool(&format!(
            "window.__whispernowListeners.has({:?})",
            whispernow_domain::ipc::EVENT
        ))
    })
    .await;
    // Let startup commands settle so only the event's refresh is counted below.
    settle().await;
    let before = adapter_count("get_history");
    let event = serde_json::to_string(&whispernow_domain::ipc::EventPayload::HistoryChanged)
        .expect("history event");

    let threw = eval_bool(&format!(
        r"(() => {{
          try {{
            window.__whispernowListeners.get({:?})({{ payload: {event} }});
            return false;
          }} catch (error) {{
            console.error(error);
            return true;
          }}
        }})()",
        whispernow_domain::ipc::EVENT
    ));

    assert!(!threw, "the Tauri event callback must not panic");
    wait_for("the history refresh to reach invokeTyped", || {
        adapter_count("get_history") > before
    })
    .await;
}

async fn settle() {
    for _ in 0..20 {
        render_turn().await;
    }
}
