# Bundled native dependencies

WhisperNow uses the official `sherpa-onnx` 1.13.8 Rust crate and its default
static CPU libraries. The native code is linked into the application executable;
installers must not contain a separate sherpa-onnx or ONNX Runtime shared
library and installed binaries must not require a development library path.

The official `sherpa-onnx-sys` build downloads an architecture-specific
`*-static-lib.tar.bz2` archive. Release builds first download and verify the
exact archive recorded in `native-allowlist.json`, extract it into a controlled
release directory, and set `SHERPA_ONNX_LIB_DIR` to that verified `lib`
directory. Linux x86_64, Windows x86_64, macOS arm64, and macOS x86_64 have
separate archive names, byte lengths, SHA-256 hashes, and upstream release URLs.

The static link set contains sherpa-onnx, ONNX Runtime, kaldi-decoder,
kaldi-native-fbank, OpenFst/kaldifst, KissFFT, piper-phonemize/eSpeak NG/UCD,
and simple-sentencepiece code. `native-allowlist.json` lists the upstream
components and licenses represented by the archive. In particular, the official
all-features static archive includes eSpeak NG even though WhisperNow does not
call TTS; a release owner must review the resulting GPL-3.0-or-later
distribution obligations before signing a release.

`native-licenses/` contains the exact hash-pinned license snapshots named by
the allowlist, including the ancillary BSD and Unicode terms used by eSpeak NG
and UCD. It also carries the exact 18,011,501-byte eSpeak NG corresponding
source archive pinned by upstream sherpa-onnx. Staging fails if any license or
source-offer byte differs, and emits a deterministic `native-compliance.zip`
that is published with each platform's release packages.

## llama.cpp (the local cleanup model)

AI cleanup's **Local cleanup model** provider runs in process on llama.cpp b11259's
public C API (`llama.h`). Nothing is compiled: the app opens the official release's
shared libraries at run time, from the resources' `llama/` directory (on macOS, the
bundle's `Contents/Frameworks/`, where they are signed with the app).
`native-allowlist.json` pins, under `shared_libraries.llama`, each platform's release
archive (name, bytes, SHA-256, URL) and every file taken from it, renamed to the name
the loader opens (the library's soname). `scripts/stage_llama.py` downloads and
verifies the archive and extracts exactly those files; CI checks the installed
payload against the same pins with `--verify`.

| Platform | Archive | Backends | Files | Unpacked | xz -6 |
| --- | --- | --- | --- | --- | --- |
| Linux x86_64 | `ubuntu-vulkan-x64.tar.gz` | Vulkan, CPU (13 CPU variants, best one chosen at run time) | 18 | 69.9 MB | 8.2 MB |
| Windows x86_64 | `win-vulkan-x64.zip` | Vulkan, CPU (13 CPU variants), with LLVM `libomp.dll` | 19 | 70.4 MB | 8.3 MB |
| macOS arm64 | `macos-arm64.tar.gz` | Metal, CPU (Accelerate BLAS) | 7 | 10.5 MB | 1.5 MB |
| macOS x86_64 | `macos-x64.tar.gz` | CPU (Accelerate BLAS) | 6 | 6.0 MB | 1.6 MB |

Packages grow by roughly the compressed size: about 8 MB with xz or LZMA (deb, rpm,
NSIS), about 23 MB with gzip-class compression (AppImage), and 1.5–3 MB on macOS.
Nothing is compiled, so builds take no longer; staging downloads one 11–33 MB archive.
The Vulkan backend accounts for about 44 MB on Linux and Windows. It loads only where
a Vulkan loader (`libvulkan.so.1`, `vulkan-1.dll`) exists; without one, or without a
usable GPU, the model runs on the CPU. It goes to the first GPU that takes it, discrete
before integrated and the one with the most memory first; AI cleanup's Use GPU switch
keeps it on the CPU. The AppImage leaves the Vulkan loader and `libwayland-client` to the
host (`LINUXDEPLOY_EXCLUDED_LIBRARIES` in the release workflow): copies from the build
machine can keep a newer distribution's Vulkan driver or Mesa from loading.
On macOS, `libggml` links its RPC backend, so that library ships
too, unused. Linux needs the system's GCC OpenMP runtime (`libgomp1`, a package
dependency). If the libraries cannot load, AI cleanup reports the provider unavailable
and keeps the deterministic text.

The Windows libraries import the Microsoft Visual C++ 2015–2022 runtime (`msvcp140.dll`,
`vcruntime140.dll`, `vcruntime140_1.dll`), which a fresh Windows does not have. The
installer carries those three in `llama/`, where the libraries' dependencies resolve
(the engine opens them with `LOAD_WITH_ALTERED_SEARCH_PATH`), copied from the release
build's Visual Studio redistributable directory, as the Visual Studio license allows
for its Distributable Code. The allowlist names them under `msvc_runtime`; CI checks that
they carry Microsoft's Authenticode signature, and `scripts/check_windows_imports.py`
checks that every DLL the installed payload imports is beside its importer or part of
Windows. `whispernow-desktop.exe` itself links the runtime statically (`cargo tauri
build` sets `STATIC_VCRUNTIME`).

llama.cpp and ggml are MIT (`llama.cpp-MIT.txt`, the release's `LICENSE`); the LLVM
OpenMP runtime shipped on Windows is Apache-2.0 WITH LLVM-exception
(`llvm-openmp-Apache-2.0-WITH-LLVM-exception.txt`, the release's
`LICENSE-LLVM-OpenMP`). Both are in `native-compliance.zip`.

The cleanup models' weights, like the ASR models', are not installer resources. The
LFM2.5-1.2B-Base and LFM2.5-2.6B-Base fine-tunes are under the LFM Open License v1.0
(`lfm-open-license-v1.0.txt`), the Qwen3.5-4B-Base fine-tune under Apache-2.0
(`Apache-2.0.txt`).

## Model weights

The ASR model weights are not installer resources. The application downloads a
model from the pinned registry only after an explicit user action and verifies
its declared size and SHA-256 digest before installation.

The installer does include the 643,854-byte Silero VAD v4 model used only for
in-process voice-activity segmentation. `resources/vad/manifest.json` pins its official
sherpa-onnx release URL, SHA-256, MIT license snapshot, and provenance. CI checks
the model's size and SHA-256 in the source tree and in every extracted package.

CI inspects each installed or extracted candidate and rejects packaged
`sherpa-onnx`/`onnxruntime` shared libraries, unresolved native dependencies,
absolute developer paths, missing VAD resources, or a candidate that cannot
start on a clean runner.
