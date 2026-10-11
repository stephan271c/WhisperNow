#![cfg(target_arch = "wasm32")]

use serde::Deserialize;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

const STYLESHEETS: [&str; 4] = [
    include_str!("../public/tokens.css"),
    include_str!("../public/ui.css"),
    include_str!("../public/settings.css"),
    include_str!("../public/overlay.css"),
];

const STYLE_PROBE: &str = r#"
const frame = document.createElement("iframe");
frame.style.cssText = "position:absolute;left:-10000px;width:1200px;height:900px;border:0";
document.body.append(frame);
try {
  const view = frame.contentWindow;
  const doc = frame.contentDocument;
  doc.documentElement.dataset.theme = theme;
  for (const css of JSON.parse(stylesheets)) {
    const style = doc.createElement("style");
    style.textContent = css;
    doc.head.append(style);
  }

  const selectors = [];
  let mediaRules = 0;
  const collectRules = rules => {
    for (const rule of rules) {
      if (rule.selectorText) {
        selectors.push({
          text: rule.selectorText,
          classes: [...rule.selectorText.matchAll(/\.([-_a-zA-Z][-_a-zA-Z0-9]*)/g)].map(match => match[1])
        });
      }
      if (rule instanceof view.CSSMediaRule && rule.conditionText === "(prefers-reduced-motion: reduce)") {
        // WebDriver has no cross-browser media emulation API; change only the parsed condition.
        rule.media.mediaText = reduced ? "all" : "not all";
        mediaRules++;
      }
      if (rule.cssRules) collectRules(rule.cssRules);
    }
  };
  for (const sheet of doc.styleSheets) {
    collectRules(sheet.cssRules);
  }
  if (mediaRules === 0) throw new Error("production reduced-motion media rules are missing");
  const override = doc.createElement("style");
  override.textContent = extra;
  doc.head.append(override);

  doc.body.innerHTML = `
    <main class="settings-content">
      <div class="page-header"><h1 id="page-title" data-text>Settings</h1></div>
      <h2 id="section-title" class="section-title" data-text>General</h2>
      <p id="section-description" class="section-description" data-text>Dictation preferences</p>
      <label id="field-label" class="field-label" data-text>Microphone</label>
      <p id="field-description" class="field-description" data-text>Select an input</p>
      <p id="field-error" class="field-error-message" data-text>Input unavailable</p>
      <p id="setting-description" class="setting-row-description" data-text>Recording shortcut</p>
      <button id="button" class="btn btn-secondary" data-text data-motion="transition">Save</button>
      <button id="small-button" class="btn btn-sm btn-secondary" data-text data-motion="transition">Retry</button>
      <input id="input" class="text-input" value="Microphone" data-text data-motion="transition">
      <select id="select" class="select" data-text data-motion="transition"><option>Default</option></select>
      <button id="segmented-option" class="segmented-option" data-text data-motion="transition">System</button>
      <button id="toggle" class="toggle" data-motion="transition"><span id="toggle-thumb" class="toggle-thumb" data-motion="transition"></span></button>
      <button id="window-control" class="window-control" data-motion="transition">×</button>
      <button id="nav-item" class="nav-item" data-text data-motion="transition">History</button>
      <button id="shortcut-capture" class="shortcut-capture" data-text data-motion="transition">Ctrl+R</button>
      <kbd id="shortcut-key" class="kbd" data-text>Ctrl</kbd>
      <span class="history-row-meta"><time id="history-time" data-text>Today at 14:00</time></span>
      <p id="history-preview" class="history-row-preview" data-text>Recognized text</p>
      <div class="model-facts"><span id="model-size" data-text>100 MB download</span></div>
      <p id="shortcut-help" class="shortcut-capture-help" data-text>Press a new shortcut</p>
      <svg class="progress-track"><rect id="progress-fill" class="progress-fill" width="20" height="6" data-motion="transition"></rect></svg>
    </main>
    <main class="overlay">
      <section class="overlay-pill overlay-listening">
        <span class="overlay-indicator"><span id="listening-dot" class="overlay-dot" data-motion="animation"></span></span>
        <p id="overlay-status" class="overlay-status" data-text>Listening</p>
        <span id="overlay-elapsed" class="overlay-elapsed" data-text>00:12</span>
        <svg class="overlay-bars" viewBox="0 0 60 20"></svg>
      </section>
      <section class="overlay-pill overlay-starting"><svg class="overlay-bars overlay-bars-arming" viewBox="0 0 60 20"></svg></section>
      <section class="overlay-pill overlay-transcribing"><span class="overlay-indicator"><span id="spinner" class="overlay-spinner" data-motion="animation"></span></span></section>
    </main>`;

  for (const element of doc.querySelectorAll("[data-text]")) {
    const owner = element.closest("[class]");
    if (!owner || owner.classList.length === 0) {
      throw new Error(`text fixture ${element.id} has no styled class`);
    }
    for (const name of owner.classList) {
      if (!selectors.some(rule => rule.classes.includes(name) &&
          (owner.matches(rule.text) || element.matches(rule.text)))) {
        throw new Error(`text fixture ${element.id} uses unstyled class .${name}`);
      }
    }
  }

  for (const bars of doc.querySelectorAll(".overlay-bars")) {
    const arming = bars.classList.contains("overlay-bars-arming");
    for (let index = 0; index < 9; index++) {
      const bar = doc.createElementNS("http://www.w3.org/2000/svg", "rect");
      bar.id = `${arming ? "arming" : "live"}-bar-${index + 1}`;
      bar.setAttribute("class", "overlay-bar");
      bar.setAttribute("x", index * 6);
      bar.setAttribute("y", 8);
      bar.setAttribute("width", 3);
      bar.setAttribute("height", 4);
      bar.dataset.motion = arming ? "animation" : "transition";
      bars.append(bar);
    }
  }

  const rootStyle = view.getComputedStyle(doc.documentElement);
  const fontTokens = [...rootStyle].filter(name => name.startsWith("--font-size-"));
  if (fontTokens.length === 0) throw new Error("production font-size tokens are missing");
  for (const token of fontTokens) {
    const text = doc.createElement("span");
    text.id = token;
    text.dataset.text = "";
    text.style.fontSize = `var(${token})`;
    text.textContent = token;
    doc.body.append(text);
  }

  const hasDuration = (names, durations) => {
    const times = durations.split(",");
    return names.split(",").some((name, index) =>
      name.trim() !== "none" && parseFloat(times[index % times.length]) > 0);
  };
  return JSON.stringify([...doc.querySelectorAll("[data-text], [data-motion]")].map(element => {
    const style = view.getComputedStyle(element);
    return {
      element: element.id,
      text: element.hasAttribute("data-text"),
      motion: element.dataset.motion ?? null,
      font_px: parseFloat(style.fontSize),
      animation: hasDuration(style.animationName, style.animationDuration),
      transition: hasDuration(style.transitionProperty, style.transitionDuration)
    };
  }));
} finally {
  frame.remove();
}
"#;

#[derive(Debug, Deserialize)]
struct ComputedStyle {
    element: String,
    text: bool,
    motion: Option<String>,
    font_px: f64,
    animation: bool,
    transition: bool,
}

fn computed_styles(theme: &str, reduced: bool, extra: &str) -> Vec<ComputedStyle> {
    let stylesheets =
        serde_json::to_string(&STYLESHEETS).expect("serialize production stylesheets");
    let result = js_sys::Function::new_with_args("stylesheets, theme, reduced, extra", STYLE_PROBE)
        .call4(
            &JsValue::NULL,
            &JsValue::from_str(&stylesheets),
            &JsValue::from_str(theme),
            &JsValue::from_bool(reduced),
            &JsValue::from_str(extra),
        )
        .expect("sample computed production styles")
        .as_string()
        .expect("computed style JSON");
    serde_json::from_str(&result).expect("computed styles")
}

fn small_text(styles: &[ComputedStyle]) -> Vec<&ComputedStyle> {
    styles
        .iter()
        .filter(|style| style.text && style.font_px < 12.0)
        .collect()
}

#[wasm_bindgen_test]
fn reduced_motion_stops_overlay_animations_and_control_transitions() {
    let normal = computed_styles("light", false, "");
    for style in &normal {
        match style.motion.as_deref() {
            Some("animation") => assert!(style.animation, "the fixture animates: {style:?}"),
            Some("transition") => assert!(style.transition, "the fixture transitions: {style:?}"),
            None => {}
            Some(other) => panic!("unknown fixture motion: {other}"),
        }
    }

    let reduced = computed_styles("light", true, "");
    let moving: Vec<_> = reduced
        .iter()
        .filter(|style| style.animation || style.transition)
        .collect();
    assert!(moving.is_empty(), "reduced motion still moves: {moving:#?}");
}

#[wasm_bindgen_test]
fn motion_check_detects_later_longhands_that_restore_motion() {
    for (element, extra) in [
        (
            "spinner",
            ".overlay-spinner { animation-name: overlay-spin; animation-duration: 0.8s; }",
        ),
        (
            "live-bar-1",
            ".overlay-bar { transition-property: height; transition-duration: 50ms; }",
        ),
    ] {
        let styles = computed_styles("light", true, extra);
        let restored = styles
            .iter()
            .find(|style| style.element == element)
            .expect("mutated moving element");
        assert!(
            restored.animation || restored.transition,
            "the browser must expose the restored motion: {restored:?}"
        );
    }
}

#[wasm_bindgen_test]
fn font_tokens_and_representative_text_render_at_least_twelve_pixels() {
    for theme in ["light", "dark"] {
        let styles = computed_styles(theme, false, "");
        let small = small_text(&styles);
        assert!(small.is_empty(), "{theme} text below 12px: {small:#?}");
    }
}

#[wasm_bindgen_test]
fn style_probe_rejects_a_renamed_text_fixture_class() {
    let probe = STYLE_PROBE.replace(
        "class=\"field-description\"",
        "class=\"obsolete-field-description\"",
    );
    let stylesheets =
        serde_json::to_string(&STYLESHEETS).expect("serialize production stylesheets");
    let error = js_sys::Function::new_with_args("stylesheets, theme, reduced, extra", &probe)
        .call4(
            &JsValue::NULL,
            &JsValue::from_str(&stylesheets),
            &JsValue::from_str("light"),
            &JsValue::FALSE,
            &JsValue::from_str(""),
        )
        .expect_err("a renamed text fixture class must be rejected");
    let message = js_sys::Reflect::get(&error, &JsValue::from_str("message"))
        .expect("probe error message")
        .as_string()
        .expect("error text");
    assert!(message.contains("unstyled class .obsolete-field-description"));
}

#[wasm_bindgen_test]
fn font_check_detects_undersized_relative_variable_and_calculated_values() {
    for (element, extra) in [
        ("overlay-elapsed", ".overlay-elapsed { font-size: 0.5rem; }"),
        (
            "--font-size-xs",
            ":root { --undersized-font: 9px; --font-size-xs: var(--undersized-font); }",
        ),
        (
            "field-description",
            ".field-description { font-size: calc(var(--font-size-xs) - 1px); }",
        ),
    ] {
        let styles = computed_styles("light", false, extra);
        let small = small_text(&styles);
        assert!(
            small.iter().any(|style| style.element == element),
            "the browser must expose the undersized {element}: {styles:#?}"
        );
    }
}

#[wasm_bindgen_test]
fn readable_relative_variable_and_calculated_values_remain_acceptable() {
    for extra in [
        ".overlay-elapsed { font-size: 1rem; }",
        ":root { --readable-font: 12px; --font-size-xs: var(--readable-font); }",
        ".field-description { font-size: calc(var(--font-size-xs) + 1px); }",
    ] {
        let styles = computed_styles("light", false, extra);
        let small = small_text(&styles);
        assert!(small.is_empty(), "readable text was rejected: {small:#?}");
    }
}
