#[cfg(target_arch = "wasm32")]
fn main() {
    dioxus::launch(whispernow_ui::app);
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {}
