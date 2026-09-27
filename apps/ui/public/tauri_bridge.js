const listenerOwners = new Map();
let nextListenerId = 1;

function tauri() {
  const api = window.__TAURI__;
  if (!api?.core || !api?.event || !api?.window) {
    throw new Error("Tauri frontend API is unavailable");
  }
  return api;
}

// Commands without input take no arguments, so `request` is sent only when there is one.
export async function invokeTyped(command, request) {
  const args = request === undefined ? {} : { request };
  return await tauri().core.invoke(command, args);
}

// Wraps a command's audio bytes in a Blob that stays in this window. No URL is made here: a
// player closed while the bytes load would never revoke it, so the caller asks objectUrl for
// one only when it is still there to play it.
export async function audioBlob(command, request) {
  const bytes = await tauri().core.invoke(command, { request });
  return new Blob([bytes], { type: "audio/wav" });
}

// An object URL an <audio> element can play; the caller revokes it when the player goes away.
export function objectUrl(blob) {
  return URL.createObjectURL(blob);
}

export function revokeObjectUrl(url) {
  URL.revokeObjectURL(url);
}

export async function writeClipboardText(text) {
  await tauri().core.invoke("plugin:clipboard-manager|write_text", { text });
}

export async function listenTyped(name, callback) {
  const unlisten = await tauri().event.listen(name, (event) => callback(event.payload));
  const id = nextListenerId++;
  listenerOwners.set(id, unlisten);
  return id;
}

export function unlistenTyped(listenerId) {
  const unlisten = listenerOwners.get(listenerId);
  if (unlisten) {
    listenerOwners.delete(listenerId);
    unlisten();
  }
}

export function currentWindowLabel() {
  return window.__TAURI__?.window?.getCurrentWindow?.().label
    ?? new URLSearchParams(window.location.search).get("window")
    ?? "settings";
}

// Called after a settings page renders; no dynamic evaluation is needed under the desktop CSP.
export function resetSettingsScroll() {
  const content = document.querySelector(".settings-content");
  if (content) {
    content.scrollTop = 0;
  }
  window.scrollTo(0, 0);
}

// Resolves after the given number of milliseconds; lets success notices retire themselves.
export function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Applies the chosen colour theme; without one the stylesheets follow the system scheme.
export function applyTheme(theme) {
  const root = document.documentElement;
  if (theme === "light" || theme === "dark") {
    root.dataset.theme = theme;
  } else {
    delete root.dataset.theme;
  }
}

// Window chrome. The desktop leaves the main window undecorated where the page draws its own
// titlebar, so the page asks the window rather than guessing from the platform.
export async function hasCustomTitlebar() {
  return !(await tauri().window.getCurrentWindow().isDecorated());
}

export async function minimizeWindow() {
  await tauri().window.getCurrentWindow().minimize();
}

export async function toggleMaximizeWindow() {
  await tauri().window.getCurrentWindow().toggleMaximize();
}

// A close request, not a destroy: the desktop hides the window to the tray and keeps its geometry.
export async function closeWindow() {
  await tauri().window.getCurrentWindow().close();
}

// Calls back with whether the window is maximized now and after every resize, which includes a
// maximize from the titlebar double-click or a keyboard shortcut.
export async function listenMaximized(callback) {
  const current = tauri().window.getCurrentWindow();
  const report = async () => callback(await current.isMaximized());
  const unlisten = await current.onResized(report);
  const id = nextListenerId++;
  listenerOwners.set(id, unlisten);
  await report();
  return id;
}
