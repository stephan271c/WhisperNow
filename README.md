# WhisperNow

WhisperNow is a private, local-first shortcut-driven dictation application for Windows, macOS, and
Linux. The 0.2 candidate is built with Rust, Tauri v2, Dioxus, and
the official `k2-fsa/sherpa-onnx` Rust bindings. Its design is in
[`docs/architecture.md`](docs/architecture.md).

## Target 0.2 packages (not yet released)

| Platform | Architectures | Official packages |
| --- | --- | --- |
| Windows 11 | x86_64 | current-user NSIS installer |
| macOS 14+ | Apple Silicon, Intel | self-signed application and DMG (approve the first launch in Privacy & Security) |
| Linux | x86_64 | AppImage, `.deb`, `.rpm` |

The capability matrix is in
[`docs/release-checklist.md`](docs/release-checklist.md#platforms). Unofficial Flatpak, Snap, Nix,
and distribution repackages are not update targets.

These are intended release artifacts; no 0.2 package has shipped yet.

## Install

The [GitHub Releases](https://github.com/stephan271c/WhisperNow/releases) page is authoritative for
versions and packages that are actually available. Do not treat the target table above as a 0.2
release announcement. WhisperNow downloads the
user-selected speech model on demand; a 0.2 installer includes only the small offline
speech-presence model.

Version 0.1.x used the retired Python desktop application. Moving from 0.1.x to 0.2.0 is a
one-time manual installer upgrade because those binaries did not contain the Tauri updater.
Settings, history, models, vocabulary, shortcuts, and other data continue to use the established
`WhisperNow` platform data directory. From 0.2.0, macOS, Windows and AppImage installations update
automatically with signed updates; `.deb` and `.rpm` installations show when a signed update is
available and link to the release page so you can install it with your package manager.
Before a manual upgrade, quit 0.1.x and uninstall it without any data-removal option; do not delete
the data directory.

## Use

1. Launch WhisperNow and complete first-run setup.
2. Select and download an ASR model.
3. Use either recording mode: hold the hold-to-record shortcut while speaking and release it, or
   press the toggle-to-record shortcut once to start and again to stop. You can enable either mode
   or both in setup and Configuration.
4. WhisperNow transcribes locally, optionally applies vocabulary or LLM cleanup, pastes into the
   currently focused application, and restores common clipboard formats on a best-effort basis.

Microphone access is required. macOS paste delivery also needs Accessibility permission. Native Wayland uses consented XDG
portals and does not fall back to raw input devices or a privileged typing daemon.
Both shortcut modes can be configured entirely with the keyboard: each mode has a named enable
control and shortcut-capture control, and validation errors are announced as alerts.

## Develop

Setup, native prerequisites and desktop behaviour are in
[`docs/development.md`](docs/development.md). The quality gates:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
python3 -m pip install -r scripts/tests/requirements.txt
python3 -m unittest discover -s scripts/tests -v
```

## Version and release contract

`Cargo.toml` is the sole product-version source. Workspace crates, the desktop bundle,
installers, updater metadata, runtime logs, and stable `vMAJOR.MINOR.PATCH` tags must agree. Check
the contract without modifying the worktree:

```bash
python3 scripts/release_version.py --check
python3 scripts/release_version.py --check --release-tag v0.2.0
```

Stable clients never receive drafts or prereleases, and downgrades are rejected. The release
checklist (build, sign, manual smoke test per platform, publish) is in
[`docs/release-checklist.md`](docs/release-checklist.md).

## Data and privacy

Audio and local ASR stay on the device. Optional AI cleanup sends the transcript to a provider
only when explicitly configured. Ordinary uninstall and updater replacement remove application
binaries but preserve the shared data directory; only the in-app **Clear All Data** action removes
user data.

WhisperNow is licensed under the [MIT License](LICENSE).
