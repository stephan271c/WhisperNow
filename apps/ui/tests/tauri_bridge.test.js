import assert from "node:assert/strict";
import test from "node:test";

import {
  applyTheme,
  audioBlob,
  closeWindow,
  hostPlatform,
  listenMaximized,
  invokeTyped,
  objectUrl,
  revokeObjectUrl,
  trackScrollActivity,
  unlistenTyped,
  windowFrame,
} from "../public/tauri_bridge.js";

function installTauri(invocations) {
  globalThis.window = {
    __TAURI__: {
      core: {
        invoke: async (command, payload) => {
          invocations.push({ command, payload });
          return null;
        },
      },
      event: {},
      window: {},
    },
  };
}

test("a history deletion request reaches the production Tauri invoke adapter", async () => {
  const invocations = [];
  installTauri(invocations);
  const request = 42;

  await invokeTyped("delete_history_entry", request);

  assert.deepEqual(invocations, [
    {
      command: "delete_history_entry",
      payload: { request },
    },
  ]);
});

test("a command without input is invoked without a request argument", async () => {
  for (const request of [undefined, null]) {
    const invocations = [];
    installTauri(invocations);

    await invokeTyped("get_state", request);

    assert.deepEqual(invocations, [{ command: "get_state", payload: {} }]);
  }
});

test("a request that is falsy but present is still sent", async () => {
  const invocations = [];
  installTauri(invocations);

  await invokeTyped("set_start_on_login", false);
  await invokeTyped("copy_text", "");

  assert.deepEqual(invocations, [
    { command: "set_start_on_login", payload: { request: false } },
    { command: "copy_text", payload: { request: "" } },
  ]);
});

function installAudio(invocations, bytes) {
  globalThis.window = {
    __TAURI__: {
      core: {
        invoke: async (command, payload) => {
          invocations.push({ command, payload });
          return bytes;
        },
      },
      event: {},
      window: {},
    },
  };
}

test("loading history audio makes no object URL a closed player could leave behind", async () => {
  installAudio([], new Uint8Array([82, 73, 70, 70]).buffer);
  const createObjectURL = URL.createObjectURL;
  let made = 0;
  URL.createObjectURL = (blob) => {
    made += 1;
    return createObjectURL(blob);
  };
  try {
    const audio = await audioBlob("get_history_audio", 7);
    assert.equal(made, 0, "only a player that is still open asks for a URL");
    assert.equal(audio.type, "audio/wav");
  } finally {
    URL.createObjectURL = createObjectURL;
  }
});

test("history audio bytes become a playable object URL that can be revoked", async () => {
  const invocations = [];
  const bytes = new Uint8Array([82, 73, 70, 70]).buffer;
  installAudio(invocations, bytes);
  const request = 7;

  const url = objectUrl(await audioBlob("get_history_audio", request));

  assert.deepEqual(invocations, [{ command: "get_history_audio", payload: { request } }]);
  assert.match(url, /^blob:/);
  const played = await fetch(url);
  assert.equal(played.headers.get("content-type"), "audio/wav");
  assert.deepEqual(new Uint8Array(await played.arrayBuffer()), new Uint8Array(bytes));
  revokeObjectUrl(url);
  await assert.rejects(fetch(url));
});

test("a chosen theme sets data-theme and the system theme removes it", () => {
  const root = { dataset: {} };
  globalThis.document = { documentElement: root };

  applyTheme("dark");
  assert.equal(root.dataset.theme, "dark");
  applyTheme("light");
  assert.equal(root.dataset.theme, "light");
  applyTheme(undefined);
  assert.equal("theme" in root.dataset, false);
});

function installWindow(calls, state, userAgent = "") {
  const resized = new Set();
  globalThis.window = {
    navigator: { userAgent },
    __TAURI__: {
      core: {},
      event: {},
      window: {
        getCurrentWindow: () => ({
          isDecorated: async () => state.decorated,
          isMaximized: async () => {
            if (state.maximizeError) {
              throw state.maximizeError;
            }
            return state.maximized;
          },
          close: async () => calls.push("close"),
          destroy: async () => calls.push("destroy"),
          onResized: async (handler) => {
            resized.add(handler);
            return () => {
              calls.push("unlisten");
              resized.delete(handler);
            };
          },
        }),
      },
    },
  };
  return {
    get activeSubscriptions() {
      return resized.size;
    },
    resize: async () => {
      for (const handler of resized) {
        await handler();
      }
    },
  };
}

function installPlatform(agent) {
  globalThis.window = { navigator: { userAgent: agent } };
}

test("the platform is read from the webview's user agent", () => {
  installPlatform("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Edg/140.0");
  assert.equal(hostPlatform(), "windows");
  installPlatform("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15");
  assert.equal(hostPlatform(), "macos");
  installPlatform("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15");
  assert.equal(hostPlatform(), "linux");
  installPlatform("Mozilla/5.0 (X11; FreeBSD amd64)");
  assert.equal(hostPlatform(), "other");
});

test("an undecorated window draws its own titlebar on every platform", async () => {
  for (const agent of ["Windows NT 10.0", "Macintosh", "X11; Linux x86_64"]) {
    installWindow([], { decorated: false }, agent);
    assert.equal(await windowFrame(), "drawn", agent);
  }
});

test("a decorated macOS window keeps its own buttons over the page", async () => {
  installWindow([], { decorated: true }, "Macintosh; Intel Mac OS X 10_15_7");
  assert.equal(await windowFrame(), "overlay");
});

test("a decorated window elsewhere keeps its native frame", async () => {
  installWindow([], { decorated: true }, "Windows NT 10.0");
  assert.equal(await windowFrame(), "native");
  installWindow([], { decorated: true }, "X11; Linux x86_64");
  assert.equal(await windowFrame(), "native");
});

test("the close button requests a close so the desktop can hide to the tray", async () => {
  const calls = [];
  installWindow(calls, {});

  await closeWindow();

  assert.deepEqual(calls, ["close"]);
});

test("maximize state is reported at once and after each resize until unlistened", async () => {
  const calls = [];
  const state = { maximized: false };
  const current = installWindow(calls, state);
  const reports = [];

  const id = await listenMaximized((maximized) => reports.push(maximized));
  assert.equal(current.activeSubscriptions, 1);
  state.maximized = true;
  await current.resize();
  unlistenTyped(id);
  assert.equal(current.activeSubscriptions, 0);
  state.maximized = false;
  await current.resize();
  unlistenTyped(id);

  assert.deepEqual(reports, [false, true]);
  assert.deepEqual(calls, ["unlisten"]);
});

test("a failed initial maximize report detaches the abandoned resize subscription", async () => {
  const failure = new Error("maximize state unavailable");
  const calls = [];
  const state = { maximizeError: failure, maximized: false };
  const current = installWindow(calls, state);
  const reports = [];

  await assert.rejects(
    listenMaximized((maximized) => reports.push(maximized)),
    (error) => error === failure,
  );

  assert.equal(current.activeSubscriptions, 0);
  assert.deepEqual(calls, ["unlisten"]);
  delete state.maximizeError;
  state.maximized = true;
  await current.resize();
  assert.deepEqual(reports, []);
});

test("the area that scrolls is marked until it has been still for a moment", (context) => {
  context.mock.timers.enable({ apis: ["setTimeout"] });
  globalThis.Element = class {
    constructor() {
      const names = new Set();
      this.classList = {
        add: (name) => names.add(name),
        remove: (name) => names.delete(name),
        contains: (name) => names.has(name),
      };
    }
  };
  const root = new Element();
  const list = new Element();
  let onScroll;
  globalThis.document = {
    documentElement: root,
    addEventListener: (type, listener, options) => {
      assert.equal(type, "scroll");
      assert.equal(options.capture, true);
      onScroll = listener;
    },
  };

  trackScrollActivity();
  trackScrollActivity();
  onScroll({ target: list });
  onScroll({ target: document });
  assert.equal(list.classList.contains("is-scrolling"), true);
  assert.equal(root.classList.contains("is-scrolling"), true);

  context.mock.timers.tick(800);
  onScroll({ target: list });
  context.mock.timers.tick(800);
  assert.equal(list.classList.contains("is-scrolling"), true);
  assert.equal(root.classList.contains("is-scrolling"), false);

  context.mock.timers.tick(200);
  assert.equal(list.classList.contains("is-scrolling"), false);
});
