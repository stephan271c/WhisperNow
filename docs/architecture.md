# Architecture

WhisperNow is a single-user, local dictation app. A shortcut starts a recording; when it ends,
the audio is transcribed on the device, optionally cleaned up by a language model, and pasted into
whichever app has focus. Every decode is an offline batch request over a finished span of audio;
there is no streaming recognition.

Tauri owns the desktop process: the windows, tray, global shortcuts and the Rust backend. The app
has two windows, settings and the recording bar. Both run the same Dioxus app, compiled to
WebAssembly, in Tauri's webview.

## Code map

Dependencies point one way: `domain` ← `core` ← `platform` ← `desktop`. The UI depends only on
`domain`.

- **`crates/whispernow-domain`**: the data types the desktop and the UI share: `Settings` and
  their rules, shortcuts, the model catalog, history entries, errors, and every command and event
  sent between the two (`ipc`).
- **`crates/whispernow-core`**: the dictation logic: audio processing, voice-activity
  segmentation and joining (`workflow`), vocabulary replacements, the cleanup prompt and response
  checks (`llm`), and the paste-and-restore rules (`text_output`). It defines the traits the
  platform implements: `AudioInput`, `AsrEngine`, `LlmService`, `PastePlatform`.
- **`crates/whispernow-platform`**: the adapters: CPAL capture, sherpa-onnx recognizers and Silero
  VAD, the in-process llama.cpp cleanup engine (`local_cleanup`, `llama`), cloud providers
  (`llm`), model downloads (`models`), SQLite history, `settings.json`, the OS credential store,
  per-OS paste (`paste_delivery`) and the Wayland portals (`wayland`). `AppPaths` says where
  everything is stored.
- **`apps/desktop`**: the Tauri app and composition root. `services` builds the `Backend`
  (settings, history, models, the dictation worker, output); `gateway` is the command and event
  layer in front of it. `shortcuts`, `tray`, `overlay` (the recording bar's window) and `updater`
  handle the parts that talk to the OS.
- **`apps/ui`**: the Dioxus frontend. `client` is the only code that talks to the desktop,
  `state` holds the latest snapshot, `management` holds each page's logic, and `components`
  renders them.
- **`apps/input-helper`**: a small helper program for GNOME on Wayland, installed setgid `input`
  so it can read the keyboard. It tells the app when a held shortcut is released, which GNOME's
  portal sometimes fails to report, and it can type the paste keystroke without a
  remote-desktop session. Its desktop side is `input_helper` in `apps/desktop`.
- **`tools/update-signature`**: verifies an update's minisign signature; the release workflow
  runs it.

Modules named `legacy` import the Python release's settings and history once. The Python under
`scripts/` is release and verification tooling, and `training/` is the separate project that
trains the cleanup models.

## Runtime model

**One thread owns the microphone and the speech model.** All dictation work runs on a single
worker thread (`services::dictation`). It is the only code that opens the microphone or runs the
speech model. The rest of the app sends it requests, such as starting or stopping a recording,
loading a model or testing the microphone, and it handles them one at a time in the order they
arrive. Because only this thread uses them, two tasks can never use the microphone or the model
at the same time. The local cleanup model works the same way, on a thread of its own.

**One shared state, sent in full on every change.** The app's state lives in one `AppState`
value. Its `Phase` says what the app is doing: `Idle`, `Recording`, `Processing`, `Paused`,
`ClearingData`, `Updating` or `Error`. Whenever the state changes, the desktop sends a complete
copy of it in a `state_changed` event, and the events arrive in the order the changes happened.
The tray, the recording bar, the shortcuts and the UI decide what to show from the latest copy.
None of them tracks the phase separately, so they cannot disagree about it.

**The main thread never blocks.** Tauri's main thread runs the event loop for every window, so
blocking it freezes the whole app. Commands from the UI run on Tauri's blocking pool
(`Gateway::run_command`). Window changes are queued to the main thread without waiting for them.
Shortcut registration, which can wait minutes for the user to answer a portal dialog, runs on a
worker thread. On shutdown, the app waits for running commands to finish before it releases the
backend.

## Dictation flow

```text
press   -> the worker opens capture with the selected speech model
        -> while recording, VAD splits the audio at long pauses; each segment is decoded in order
release -> flush, wait for the last decodes, join the segment texts
        -> vocabulary replacements
        -> AI cleanup when on (local or cloud); any failure keeps the uncleaned text
        -> history entry
        -> paste into the focused app, then restore the user's clipboard
        -> Idle
```

| Stage | Desktop | Core | Platform |
| --- | --- | --- | --- |
| Capture | `dictation::capture` | `audio` | `audio` |
| Segmentation and decode | `dictation` | `workflow::segmentation` | `voice_activity`, `asr` |
| Cleanup | `dictation::cleanup` | `llm` | `local_cleanup`, `llm` |
| History | `history` | | `sqlite_history` |
| Paste | `output` | `text_output` | `paste_delivery`, `wayland` |

## Where data lives

- **Settings**: `settings.json`, held in memory by the desktop and saved on every change. Hand
  edits take effect after a restart.
- **API keys**: only in the OS credential store. `Settings.api_keys` lists the providers that
  have one, so the store is asked only when a key is needed.
- **History**: one SQLite `entries` row per dictation, plus `<id>.wav` while Keep audio is on.
- **Models**: downloaded speech and cleanup models, managed by `ModelLibrary` in the desktop and
  `models` in the platform.
- **Start on login**: the OS registration itself, not a setting.

## Boundaries and rules

- `domain` and `core` do not depend on Tauri or Dioxus, so the dictation logic is tested without
  a desktop.
- What the UI can see and do is limited:
  - It never receives credentials or clipboard contents.
  - It never sends the desktop a file path to read or open. Commands such as
    `open_recordings_folder` always open a fixed folder.
  - It gets a history recording as audio bytes, not as a file path.
  - It can save or remove an API key, but it can never read one back.
- Audio, transcripts, clipboard values, window titles, personal paths, credentials and provider
  request and response bodies never reach logs. Errors carry a code and a summary written for the
  user, not native error text.
- A missing OS permission disables only the feature that needs it.

## Platform differences

On X11, Windows and macOS, the app registers global shortcuts through Tauri, writes to the system
clipboard and simulates the paste keystroke.

Native Wayland is stricter: an app cannot see keys pressed in other apps or simulate key presses.
Instead it asks the desktop through portals, system services that grant an app a capability,
often after asking the user. So on Wayland:

- Shortcuts go through the GlobalShortcuts portal (`wayland_shortcuts`).
- Pasting goes through the Clipboard and RemoteDesktop portals, or through the input helper if it
  is installed.
- The app's own windows run under XWayland when it is available. GNOME does not let a native
  Wayland window keep itself above other windows, and the recording bar needs that.

macOS, Windows and AppImage installs update themselves. Debian and RPM installs are updated by
the system package manager, so for them the app only links to the release page.
