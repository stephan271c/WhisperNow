# Native license snapshots

License texts for the native code statically linked into the WhisperNow executable.

## Why the eSpeak NG source archive is tracked

`espeak-ng-ed530aa113046142eb5115cf2fc9157854d0ffe1.zip` (about 18 MB) is the corresponding
source for the eSpeak NG code in the official sherpa-onnx static libraries. eSpeak NG is
GPL-3.0-or-later, so distributing a binary that links it means offering its source.

WhisperNow never calls text-to-speech, but it can't drop that code yet:

- upstream sherpa-onnx 1.13.6 publishes only all-features `*-static-lib.tar.bz2` archives;
- the `sherpa-onnx-sys` 1.13.6 build script always links `static=espeak-ng` and
  `static=piper_phonemize`.

Removing it would need both a from-source sherpa-onnx build with `SHERPA_ONNX_ENABLE_TTS=OFF`
on every release platform and a patched `sherpa-onnx-sys`. Once that exists, delete the zip,
`espeak-ng-*.txt`, `piper-phonemize-MIT.txt` and `Unicode-DFS-2016.txt`.
