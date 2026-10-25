use wasm_bindgen_futures::JsFuture;

pub fn install_mount_point() {
    let document = web_sys::window()
        .expect("window")
        .document()
        .expect("document");
    if let Some(previous) = document.get_element_by_id("main") {
        previous.remove();
    }
    let mount = document.create_element("div").expect("create mount point");
    mount.set_id("main");
    document
        .document_element()
        .expect("document element")
        .append_child(&mount)
        .expect("append mount point");
}

pub async fn render_turn() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        web_sys::window()
            .expect("window")
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 10)
            .expect("schedule render turn");
    });
    JsFuture::from(promise).await.expect("render turn");
}

pub async fn wait_for(what: &str, condition: impl Fn() -> bool) {
    for _ in 0..200 {
        if condition() {
            return;
        }
        render_turn().await;
    }
    panic!("timed out waiting for {what}");
}

pub fn eval_bool(script: &str) -> bool {
    js_sys::eval(script)
        .expect("evaluate probe script")
        .as_bool()
        .expect("boolean probe result")
}

/// Records each call as Tauri's IPC sends it: through `JSON.stringify`, which drops `undefined`.
pub fn install_tauri(invoke_js: &str) {
    js_sys::eval(&format!(
        r#"
        window.__whispernowAdapterCalls = [];
        window.__whispernowListeners = new Map();
        window.__TAURI__ = {{
          core: {{
            invoke: async (command, payload) => {{
              const sent = JSON.parse(JSON.stringify(payload, (_key, value) =>
                value instanceof Map ? Object.fromEntries(value) : value));
              window.__whispernowAdapterCalls.push({{ command, payload: sent }});
              {invoke_js}
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
        "#
    ))
    .expect("install Tauri stub");
}

pub fn adapter_count(command: &str) -> u32 {
    let count = js_sys::eval(&format!(
        "window.__whispernowAdapterCalls.filter(call => call.command === {command:?}).length"
    ))
    .expect("count adapter calls");
    serde_wasm_bindgen::from_value(count).expect("integer call count")
}
