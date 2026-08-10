# Native Silero VAD speech fixture

`flite-testing-one-two-three.s16le` is deterministic synthesized speech used to exercise the
real official `sherpa-onnx` Silero VAD adapter. It contains the phrase “testing one two three” spoken by
Flite's `slt` voice, with no captured or user-provided audio.

It was generated on Ubuntu with FFmpeg 6.1.1 and its bundled libflite filter:

```sh
ffmpeg -hide_banner -loglevel error \
  -f lavfi -i "flite=text='testing one two three':voice=slt" \
  -ar 16000 -ac 1 -f s16le \
  flite-testing-one-two-three.s16le
```

The committed file is mono signed 16-bit little-endian PCM at 16 kHz, 59,040 bytes (29,520
samples, 1.845 seconds). Its SHA-256 is
`710ace5fc266bba391f71746a1ff7999a59d92d9af6e46a6d65c11c84df03349`.

The fixture contains 250 ms of leading silence. Speech continues until approximately 1.64
seconds, followed by roughly 205 ms of silence. Tests slice and pad this one pinned source to
cover short speech, speech ending at a partial EOF frame, speech surrounded by silence, and
repeated recordings.
