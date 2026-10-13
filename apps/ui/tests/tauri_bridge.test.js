import assert from "node:assert/strict";
import test from "node:test";

import {
  applyTheme,
  audioBlob,
  closeWindow,
  hasCustomTitlebar,
  listenMaximized,
  invokeTyped,
  objectUrl,
  revokeObjectUrl,
  unlistenTyped,
  writeClipboardText,
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
  const invocations = [];
  installTauri(invocations);

  await invokeTyped("get_state", undefined);

  assert.deepEqual(invocations, [{ command: "get_state", payload: {} }]);
});

test("copy text reaches the production clipboard adapter", async () => {
  const invocations = [];
  installTauri(invocations);

  await writeClipboardText("cleaned transcript");

  assert.deepEqual(invocations, [
    {
      command: "plugin:clipboard-manager|write_text",
      payload: { text: "cleaned transcript" },
    },
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

function installWindow(calls, state) {
  const resized = new Set();
  globalThis.window = {
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

test("only an undecorated window draws its own titlebar", async () => {
  installWindow([], { decorated: false });
  assert.equal(await hasCustomTitlebar(), true);
  installWindow([], { decorated: true });
  assert.equal(await hasCustomTitlebar(), false);
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
