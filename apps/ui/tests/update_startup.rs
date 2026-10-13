#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;
use whispernow_domain::ipc::{
    AppState, EventPayload, ModelState, Phase, UpdatePhase, UpdateProgress, UpdateState,
};
use whispernow_domain::{AppError, ErrorCategory, ModelId};

wasm_bindgen_test_configure!(run_in_browser);

mod browser_support;
use browser_support::{install_mount_point, render_turn, wait_for};

#[wasm_bindgen_test(async)]
async fn newer_ready_event_survives_the_delayed_initial_update_snapshot() {
    newer_event_survives(Ok(downloading_update())).await;
}

#[wasm_bindgen_test(async)]
async fn newer_ready_event_survives_a_failed_initial_update_read() {
    newer_event_survives(Err(initial_read_error())).await;
}

async fn newer_event_survives(answer: Result<UpdateState, AppError>) {
    launch_with_pending_update_read().await;
    emit(&EventPayload::StateChanged {
        state: idle_state(),
    });
    emit(&EventPayload::UpdateChanged {
        state: update(UpdatePhase::ReadyToRestart {
            version: "0.3.0".to_owned(),
            release_notes: None,
            error: None,
        }),
    });
    wait_for(
        "the event's ready state and restart action",
        ready_to_restart,
    )
    .await;

    finish_initial_read(answer).await;

    assert!(
        ready_to_restart(),
        "the initial read must preserve the ready banner and enabled restart action"
    );
    assert!(!eval_bool("!!document.querySelector('.update-error')"));
}

#[wasm_bindgen_test(async)]
async fn initial_update_snapshot_is_installed_without_an_event() {
    launch_with_pending_update_read().await;

    finish_initial_read(Ok(downloading_update())).await;

    assert!(eval_bool(
        "document.querySelector('.update-banner h3')?.textContent === 'Downloading update to v0.3.0'"
    ));
}

#[wasm_bindgen_test(async)]
async fn initial_update_read_failure_is_visible_without_an_event() {
    launch_with_pending_update_read().await;

    finish_initial_read(Err(initial_read_error())).await;

    assert!(eval_bool(
        "document.querySelector('.update-error h3')?.textContent === 'Update check failed' && document.querySelector('.update-error p')?.textContent === 'The initial update read failed'"
    ));
}

async fn launch_with_pending_update_read() {
    let idle = serde_json::to_string(&idle_state()).expect("idle snapshot");
    js_sys::eval(&format!(
        r#"
        window.__whispernowAdapterCalls = [];
        window.__whispernowListeners = new Map();
        window.__whispernowInitialUpdate = undefined;
        window.__TAURI__ = {{
          core: {{
            invoke: async (command, payload) => {{
              window.__whispernowAdapterCalls.push({{ command, payload }});
              if (command === "get_update_state") {{
                return new Promise((resolve, reject) => {{
                  window.__whispernowInitialUpdate = {{ resolve, reject }};
                }});
              }}
              if (command === "get_state") return {idle};
              throw new Error(`unavailable in this test: ${{command}}`);
            }}
          }},
          event: {{
            listen: async (name, handler) => {{
              window.__whispernowListeners.set(name, handler);
              return () => window.__whispernowListeners.delete(name);
            }}
          }},
          window: {{ getCurrentWindow: () => ({{ label: "settings", isDecorated: async () => true }}) }}
        }};
        "#,
    ))
    .expect("install delayed Tauri update read");
    install_mount_point();
    dioxus::launch(whispernow_ui::app);
    wait_for("the pending initial update read", || {
        eval_bool("!!window.__whispernowInitialUpdate")
    })
    .await;
}

async fn finish_initial_read(answer: Result<UpdateState, AppError>) {
    let (method, value) = match answer {
        Ok(state) => ("resolve", serde_json::to_string(&state)),
        Err(error) => ("reject", serde_json::to_string(&error)),
    };
    let value = value.expect("initial update answer");
    js_sys::eval(&format!(
        "window.__whispernowInitialUpdate.{method}({value})"
    ))
    .expect("complete initial update read");
    wait_for("startup hydration after the update read", || {
        eval_bool("window.__whispernowAdapterCalls.some(({command}) => command === 'get_state')")
    })
    .await;
    render_turn().await;
}

fn emit(event: &EventPayload) {
    let payload = serde_json::to_string(event).expect("event payload");
    js_sys::eval(&format!(
        "window.__whispernowListeners.get({:?})({{payload: {payload}}})",
        event.event_name()
    ))
    .expect("deliver Tauri event");
}

fn ready_to_restart() -> bool {
    eval_bool(
        "document.querySelector('.update-ready h3')?.textContent === 'Restart to update to v0.3.0' && Array.from(document.querySelectorAll('.update-ready button')).some(button => button.textContent.trim() === 'Restart to update' && !button.disabled)",
    )
}

fn eval_bool(script: &str) -> bool {
    js_sys::eval(script)
        .expect("evaluate probe script")
        .as_bool()
        .expect("boolean probe result")
}

fn idle_state() -> AppState {
    AppState {
        phase: Phase::Idle,
        selected_model_id: ModelId::new("current-model").expect("fixture model id"),
        model_state: ModelState::Ready,
        last_transcript: None,
        error: None,
        active_downloads: Vec::new(),
        setup_test: None,
    }
}

fn update(phase: UpdatePhase) -> UpdateState {
    UpdateState {
        phase,
        notification_dismissed: false,
        restart_deferred: false,
    }
}

fn downloading_update() -> UpdateState {
    update(UpdatePhase::Downloading {
        version: "0.3.0".to_owned(),
        release_notes: None,
        progress: UpdateProgress {
            downloaded_bytes: 40,
            total_bytes: Some(100),
        },
    })
}

fn initial_read_error() -> AppError {
    AppError::new(
        ErrorCategory::General,
        "initial_update_read_failed",
        "The initial update read failed",
    )
}
