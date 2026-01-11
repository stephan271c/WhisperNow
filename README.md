# WhisperNow

**Push-to-talk speech transcription for desktop** – fast, private, and cross-platform.

WhisperNow is a desktop application that transcribes your speech in real-time with a simple push-to-talk hotkey. It runs locally using NVIDIA's Parakeet ASR model, keeping your voice data private while delivering high-quality transcriptions.

---

## Features

- **Push-to-Talk Recording** – Hold a customizable hotkey to record, release to transcribe
- **High-Quality ASR** – Uses NVIDIA NeMo Parakeet models (with optional HuggingFace backends)
- **GPU Acceleration** – CUDA support for fast transcription on compatible hardware
- **System Tray Integration** – Runs quietly in the background
- **LLM Enhancement** – Optional post-processing via OpenAI, Anthropic, Google, or local Ollama
- **Vocabulary Replacement** – Define custom word/phrase substitutions
- **Cross-Platform** – Works on Windows, macOS, and Linux
- **Auto-Typing** – Automatically types transcriptions into the active window
- **Transcription History** – Browse and copy past transcriptions

---

## Installation

### Prerequisites

- **Python 3.12+**
- **uv** package manager ([install uv](https://github.com/astral-sh/uv))
- **CUDA** (optional, for GPU acceleration)

### From Source

```bash
# Clone the repository
git clone https://github.com/yourname/whispernow.git
cd whispernow

# Create virtual environment and install dependencies
uv sync

# Run the application
source .venv/bin/activate
uv run python -m src.transcribe.app
```

### Pre-built Releases

Download installers for your platform from the [Releases](https://github.com/yourname/whispernow/releases) page:

| Platform | Format |
|----------|--------|
| Windows  | `.exe` installer |
| macOS    | `.dmg` disk image |
| Linux    | `.AppImage` |

---

## Quick Start

1. **Launch WhisperNow** – The app starts in the system tray
2. **Wait for model loading** – First run downloads the ASR model (~600MB)
3. **Press and hold** your hotkey (default: `Ctrl+Shift`) to record
4. **Release** to transcribe – text is automatically typed into the active window

### First-Run Setup Wizard

On first launch, a setup wizard guides you through:
- Microphone permission requests
- Hotkey configuration
- GPU/CPU selection

---

## Configuration

Access settings by clicking the tray icon → **Settings**.

### General Tab
| Setting | Description |
|---------|-------------|
| Hotkey | Keyboard shortcut for push-to-talk |
| Start minimized | Launch to tray without showing window |
| Auto-start on login | Launch WhisperNow at system startup |
| Typing speed | Characters per second (0 = instant paste) |

### Model Tab
| Setting | Description |
|---------|-------------|
| ASR Model | Select from available Parakeet models |
| Use GPU | Enable CUDA acceleration |
| Sample Rate | Audio sample rate (default: 16000 Hz) |
| Input Device | Select microphone |

### Enhancements Tab
Configure optional LLM post-processing:
| Provider | Description |
|----------|-------------|
| OpenAI | GPT models (requires API key) |
| Anthropic | Claude models (requires API key) |
| Google | Gemini models (requires API key) |
| Ollama | Local models (no API key needed) |

Built-in enhancements include:
- **Fix Grammar** – Correct grammar and punctuation
- **Formal Tone** – Convert to professional language
- **Casual Tone** – Make text more conversational
- **Summarize** – Condense transcription

### Vocabulary Tab
Define custom word/phrase replacements. Useful for:
- Technical jargon corrections
- Name spelling fixes
- Abbreviation expansions

---

## Project Structure

```
whispernow/
├── src/transcribe/
│   ├── app.py              # Application entry point
│   ├── core/
│   │   ├── asr/            # Speech recognition backends
│   │   ├── audio/          # Audio recording
│   │   ├── input/          # Hotkey handling
│   │   ├── output/         # Text output controller
│   │   ├── settings/       # Configuration management
│   │   └── transcript_processor/  # LLM & vocabulary processing
│   ├── ui/
│   │   ├── tray.py         # System tray icon
│   │   ├── main_window.py  # Settings window
│   │   ├── setup_wizard.py # First-run wizard
│   │   ├── recording_toast.py  # Recording indicator
│   │   └── tabs/           # Settings tabs
│   └── utils/
│       ├── logger.py       # Logging configuration
│       └── platform.py     # Platform-specific utilities
├── tests/                  # Test suite
├── scripts/
│   └── build.py            # PyInstaller build script
├── pyproject.toml          # Project metadata & dependencies
├── transcribe.spec         # PyInstaller spec file
└── installer.iss           # Windows Inno Setup script
```

---

## Development

### Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
uv run pytest

# Run tests excluding slow integration tests
uv run pytest -m "not slow"

# Run with coverage
uv run pytest --cov=src/transcribe
```

### Building Distributables

```bash
# Build with PyInstaller
uv run python scripts/build.py
```

**Output locations:**
- Windows: `dist/whispernow/` (folder) + `installer-output/*.exe`
- macOS: `dist/WhisperNow.app` + `*.dmg`
- Linux: `dist/whispernow/` + `*.AppImage`

### CI/CD

The project uses GitHub Actions for automated builds:

```bash
# Trigger on:
# - Push to main branch
# - Version tags (v*)
# - Pull requests to main
```

Tagged releases (`v*`) automatically create draft GitHub releases with all platform installers.

---

## Dependencies

### Core
| Package | Purpose |
|---------|---------|
| PySide6 | Qt-based GUI framework |
| nemo-toolkit | NVIDIA NeMo ASR models |
| torch / torchaudio | Neural network inference |
| pynput | Global hotkey capture |
| sounddevice | Audio recording |
| litellm | Unified LLM API client |

### Optional
| Package | Purpose |
|---------|---------|
| transformers + accelerate | HuggingFace model backend |

### Development
| Package | Purpose |
|---------|---------|
| pytest + pytest-qt | Testing framework |
| pyinstaller | Application bundling |

---

## Platform Notes

### Linux
- Requires `xclip` for clipboard operations
- AppImage requires FUSE: `sudo apt install fuse libfuse2`
- Hotkey capture may need X11 or elevated permissions

### macOS
- Requires Accessibility permissions for global hotkeys
- Microphone permission required on first use

### Windows
- No special requirements
- Installer registers start menu entry

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`uv run pytest`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.


