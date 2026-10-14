# Native license snapshots

License texts for the native code statically linked into the WhisperNow executable.

## Why the eSpeak NG source archive is tracked

`espeak-ng-ed530aa113046142eb5115cf2fc9157854d0ffe1.zip` (about 18 MB) is the corresponding
source for the eSpeak NG code in the official sherpa-onnx static libraries. eSpeak NG is
GPL-3.0-or-later, so distributing a binary that links it means offering its source.

WhisperNow never calls text-to-speech, but it can't drop that code yet:

- the sherpa-onnx 1.13.8 `*-static-lib.tar.bz2` archives selected by the Rust build include TTS;
- the `sherpa-onnx-sys` 1.13.8 build script always links `static=espeak-ng` and
  `static=piper_phonemize`.

Removing it would need the upstream no-TTS archives on every release platform and a
`sherpa-onnx-sys` build that omits the TTS libraries. Once that exists, delete the zip,
`espeak-ng-*.txt`, `piper-phonemize-MIT.txt` and `Unicode-DFS-2016.txt`.
