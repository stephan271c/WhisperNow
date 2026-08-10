# Local safe-wrapper patch

This directory is the published `sherpa-onnx` 1.13.6 crate (crate checksum
`93440301c4240bd5d83496ab9adeee889be48d9428f9c01c37e5ae4ef33ba9e2`,
upstream tag commit `1cb484af5e69d3c7803c1eb0b3b5ab8041e0e911`).

The only production-source change is in `src/offline_asr.rs`: the safe
`OfflineRecognizerResult` deserializes `lang`, `segment_timestamps`,
`segment_durations`, and `segment_texts` already emitted by the pinned native
library's `SherpaOnnxGetOfflineStreamResultAsJson`. This is intentionally an
upstreamable result-shape fix. It introduces no new unsafe code, native handle,
model owner, or inference pass.

Remove the Cargo patch and this directory once a pinned published upstream
release exposes these fields.
