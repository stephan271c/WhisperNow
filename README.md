# WhisperNow

**Push-to-talk speech transcription for desktop** – fast, private, and cross-platform.

WhisperNow is a desktop application that transcribes your speech in real-time with a simple push-to-talk hotkey. It runs locally using NVIDIA NeMo and HuggingFace ASR models, keeping your voice data private while delivering high-quality transcriptions.

---

## Features

- **Push-to-Talk Recording** – Hold a customizable hotkey to record, release to transcribe
- **High-Quality ASR** – Uses NVIDIA NeMo and HuggingFace ASR models
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
git clone https://github.com/stephan271c/WhisperNow.git
cd WhisperNow

# Create virtual environment and install dependencies
uv sync

# Run the application
source .venv/bin/activate
uv run python -m src.whispernow.bootstrap
```

### Pre-built Releases

Download installers for your platform from the [Releases](https://github.com/stephan271c/WhisperNow/releases) page:

| Platform | Format |
|----------|--------|
| Windows  | `.exe` installer |
| macOS    | `.dmg` disk image |
| Linux    | `.AppImage` |

---

## How it works

```mermaid
graph TD
    A[User Produces Speech] --> B(ASR Model)
    B -->|Transcription| C{Vocabulary<br>Substitution?}
    C -->|Yes| D[Replace Keywords/Abbreviations]
    C -->|No| E{LLM<br>Enhancement?}
    D --> E
    E -->|Yes| F[Apply LLM Enhancements]
    E -->|No| G[Output to Textbox]
    F --> G
```

---

## Quick Start

1. **Launch WhisperNow** – The app starts in the system tray
2. **Wait for model loading** – First run downloads the ASR model
3. **Press and hold** your hotkey (default: `Ctrl+Shift`) to record
4. **Release** to transcribe – text is automatically typed into the active window

### First-Run Setup Wizard

On first launch, a setup wizard guides you through:
- Microphone permission requests
- Hotkey configuration
- GPU/CPU selection

---

## Models

### ASR Models

WhisperNow supports Nvidia NeMo and HuggingFace pipeline compatible ASR models, including OpenAI Whisper models. We recommend using parakeet-tdt-0.6b-v3 for the best performance.

| Model | Description |
|-------|-------------|
| `nvidia/parakeet-tdt-0.6b-v3` | NVIDIA NeMo ASR model |
| `nvidia/canary-1b-v2` | NVIDIA NeMo ASR model |
| `openai/whisper-large-v3-turbo` | OpenAI Whisper model |
| `openai/whisper-small` | OpenAI Whisper model |

### LLM Models

WhisperNow uses litellm to interface with LLMs. You can use any LLM that litellm supports. Since the llm is used for simple post-processing, we recommend using a small model.

| Model | Description |
|-------|-------------|
| `ollama/gemma3:1b` | Ollama Google Gemma model |
nvidia/nemotron-3-nano-30b-a3b:free` | OpenRouter model |
| `openai/gpt-5-nano` | OpenAI GPT-5 nano model |
| `anthropic/claude-sonnet-4-5` | Anthropic Claude model |
| `gemini/gemini-flash-latest` | Google Gemini model |

## Configuration

Access settings by clicking the tray icon → **Settings**.

### Configuration Tab
| Setting | Description |
|---------|-------------|
| Typing speed | Characters per second (0 = instant paste) |
| Start minimized | Launch to tray without showing window |
| Auto-start on login | Launch WhisperNow at system startup |
| Hotkey | Keyboard shortcut for push-to-talk |
| Sample Rate | Audio sample rate (default: 16000 Hz) |
| Input Device | Select microphone |
| ASR Model | Select from available ASR models |
| Use GPU | Enable CUDA acceleration |

### Mode Tab
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
├── src/whispernow/
│   ├── app.py                # Application entry point
│   ├── bootstrap.py          # Dependency check & setup wizard
│   ├── core/
│   │   ├── asr/              # Speech recognition backends
│   │   ├── deps/             # Dependency management & installation
│   │   ├── audio/            # Audio recording
│   │   ├── input/            # Hotkey handling
│   │   ├── output/           # Text output controller
│   │   ├── settings/         # Configuration management
│   │   └── transcript_processor/  # LLM & vocabulary processing
│   ├── ui/
│   │   ├── main_window.py    # Settings window
│   │   ├── tray.py           # System tray icon
│   │   ├── setup_wizard.py   # First-run wizard
│   │   ├── dependency_setup_page.py # Installation UI
│   │   ├── recording_toast.py    # Recording indicator
│   │   └── tabs/             # Settings tabs
│   └── utils/
│       ├── logger.py         # Logging configuration
│       └── platform.py       # Platform-specific utilities
├── tests/                    # Test suite
├── scripts/
│   └── build.py              # Briefcase build script
├── pyproject.toml            # Project metadata & dependencies
└── installer.iss             # Windows Inno Setup script
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
uv run pytest --cov=src/whispernow
```

### Building Distributables

```bash
# Build with Briefcase
uv run python scripts/build.py
```

**Output locations:**
- Windows: `build/whispernow/windows/app/` + `installer-output/*.exe`
- macOS: `build/whispernow/macos/app/WhisperNow.app` + `*.dmg`
- Linux: `build/whispernow/linux/appimage/*.AppImage`

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

### Core (Bundled)
These packages are included in the lightweight installer:
| Package | Purpose |
|---------|---------|
| PySide6 | Qt-based GUI framework |
| pynput | Global hotkey capture |
| sounddevice | Audio recording |

### Machine Learning (Downloaded on First Run)
These heavy dependencies are downloaded automatically by the setup wizard:
| Package | Purpose |
|---------|---------|
| torch / torchaudio | Neural network inference |
| nemo-toolkit | NVIDIA NeMo ASR models |
| litellm | Unified LLM API client |

### Optional
| Package | Purpose |
|---------|---------|
| transformers + accelerate | HuggingFace model backend |

### Development
| Package | Purpose |
|---------|---------|
| pytest + pytest-qt | Testing framework |
| briefcase | Application bundling |

---

## Platform Notes

### Linux
- Requires `xclip` for clipboard operations
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

This project is licensed under the MIT License


