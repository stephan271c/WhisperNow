![WhisperNow: Write as fast as you think.](brand/banner/whispernow-banner.png)

# WhisperNow

WhisperNow turns your speech into text and pastes it wherever you're typing. It runs on Windows,
macOS and Linux, and your audio never leaves your computer.

## Demo

<!-- Demo video goes here. -->

## Features

- **Works in any app.** Hold <kbd>Ctrl</kbd> + <kbd>Space</kbd> (<kbd>Ctrl</kbd> +
  <kbd>Option</kbd> + <kbd>Space</kbd> on macOS) while you speak, or switch to toggle mode:
  press once to start and again to stop. The text is pasted into the selected text field, and
  your clipboard is preserved.
- **Speech recognition on your computer.** Pick from nine speech models, from a 108 MB English
  model to Whisper Large v3 Turbo with 100 languages.
- **Optional AI cleanup.** Removes filler words, false starts and self-corrections, and formats
  lists. Run one of WhisperNow's own cleanup models on your computer, or use OpenAI, Anthropic,
  Google Gemini, OpenRouter, Ollama or any other compatible service.
- **Vocabulary.** Replacement rules fix names and terms the speech model gets wrong.
- **History.** Search past transcripts and copy them again.

## Privacy

- Recording and speech recognition run on your computer. Audio is never uploaded.
- API keys are stored in your system keychain.

## Download

Download WhisperNow from [GitHub Releases](https://github.com/stephan271c/WhisperNow/releases/latest).

| Platform | Package | Updates |
| --- | --- | --- |
| Windows 11 (x86_64) | `.exe` installer | Automatic |
| macOS 13.3 or later (Apple Silicon or Intel) | `.dmg` | Automatic |
| Linux (x86_64) | `.AppImage` | Automatic |
| Linux (x86_64) | `.deb`, `.rpm` | WhisperNow tells you when one is out; install it with your package manager |

Automatic updates download in the background and install when you restart WhisperNow.
Speech models aren't part of the download: setup downloads the one you choose.

**macOS:** WhisperNow isn't notarized by Apple, so macOS blocks its first launch. Open
**System Settings › Privacy & Security** and choose **Open Anyway**.

**Upgrading from 0.1:** uninstall 0.1, then install 0.2. Your settings and history carry over.

## Getting started

1. Open WhisperNow. Setup asks for microphone access, downloads a speech model and lets you
   choose your shortcut. On macOS it also asks for Accessibility access, which WhisperNow needs
   to paste.
2. Click into any text field, hold your shortcut and speak.
3. Let go. The text appears where your cursor is.

### Linux

On X11, shortcuts and pasting work without extra setup.

On Wayland, your desktop asks you once to allow WhisperNow's shortcut. Pasting uses a small
helper, which the `.deb` and `.rpm` packages install. With the AppImage, WhisperNow offers to set
it up.

## Models

### Speech models

| Model | Languages | Download |
| --- | --- | --- |
| Parakeet TDT v2 0.6B (Int8), the default | English | 482 MB |
| Parakeet TDT v2 0.6B (FP16) | English | 1.1 GB |
| Parakeet Unified 0.6B | English | 501 MB |
| Parakeet TDT v3 0.6B | 25 European languages | 487 MB |
| Parakeet TDT 110M | English | 108 MB |
| SenseVoice Small | Chinese, Cantonese, English, Japanese, Korean | 163 MB |
| Qwen3-ASR 0.6B | 30 languages | 879 MB |
| Whisper Distil Large v3.5 | English | 529 MB |
| Whisper Large v3 Turbo | 100 languages | 564 MB |

### Cleanup models

WhisperNow's cleanup models are trained for dictation cleanup and run on your computer. They use
the GPU through Vulkan on Windows and Linux and Metal on Apple Silicon, and the CPU otherwise.

| Model | Download | Time for 100 words, GPU | Time for 100 words, CPU |
| --- | --- | --- | --- |
| LFM2.5 1.2B, the default | 1.2 GB | 0.2 s | 2.3 s |
| LFM2.5 2.6B | 2.9 GB | 0.3 s | 5.4 s |
| Qwen3.5 4B | 4.5 GB | 0.6 s | 8.9 s |

Times were measured on an RTX 5070 and a Ryzen 9 8945HS.

## Build from source

WhisperNow is written in Rust with [Tauri](https://tauri.app) and
[Dioxus](https://dioxuslabs.com). Speech recognition runs on
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) and the cleanup models on
[llama.cpp](https://github.com/ggml-org/llama.cpp). [docs/architecture.md](docs/architecture.md)
explains how it fits together.

## License

WhisperNow is released under the [MIT License](LICENSE).
