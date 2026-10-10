use std::path::{Path, PathBuf};

use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};
use thiserror::Error;
use whispernow_core::{
    audio::TARGET_SAMPLE_RATE,
    workflow::{AlwaysSpeech, FRAME_SAMPLES, VoiceActivity},
};

/// Filename of the pinned Silero VAD v4 runtime asset. Release tooling checks its digest.
const SILERO_VAD_MODEL_FILE: &str = "silero_vad.onnx";

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the segmenter's frame is 512 samples"
)]
const WINDOW_SAMPLES: i32 = FRAME_SAMPLES as i32;
const SPEECH_THRESHOLD: f32 = 0.5;
// Short native hysteresis keeps the decision close to per-frame speech probability: the core
// segmenter owns pause length, padding, and minimum-speech policy. The onset lag is bounded by
// the minimum speech duration, which stays well inside the segmenter's leading padding.
const MIN_SPEECH_DURATION_SECONDS: f32 = 0.064;
const MIN_SILENCE_DURATION_SECONDS: f32 = 0.1;
// Segments are forced below 25 s by the core segmenter; this only bounds the native buffer.
const MAX_SPEECH_DURATION_SECONDS: f32 = 30.0;
const VAD_BUFFER_SIZE_SECONDS: f32 = 30.0;

#[must_use]
pub fn silero_vad_model_path(resource_root: impl AsRef<Path>) -> PathBuf {
    resource_root
        .as_ref()
        .join("vad")
        .join(SILERO_VAD_MODEL_FILE)
}

/// Builds fresh detector state for a recording. An unavailable model fails open to
/// [`AlwaysSpeech`], so no recording is rejected as silent.
#[must_use]
pub fn silero_voice_activity(model_path: &Path) -> Box<dyn VoiceActivity> {
    match SileroVoiceActivity::new(model_path) {
        Ok(activity) => Box::new(activity),
        Err(error) => {
            tracing::warn!(
                ?error,
                "Silero VAD is unavailable; the recording is segmented without pause detection"
            );
            Box::new(AlwaysSpeech)
        }
    }
}

struct SileroVoiceActivity {
    detector: VoiceActivityDetector,
}

impl SileroVoiceActivity {
    fn new(model_path: &Path) -> Result<Self, SileroVadError> {
        let model = model_path
            .to_str()
            .filter(|model| !model.as_bytes().contains(&0))
            .ok_or(SileroVadError::InvalidModelPath)?;
        let config = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: Some(model.to_owned()),
                threshold: SPEECH_THRESHOLD,
                min_silence_duration: MIN_SILENCE_DURATION_SECONDS,
                min_speech_duration: MIN_SPEECH_DURATION_SECONDS,
                window_size: WINDOW_SAMPLES,
                max_speech_duration: MAX_SPEECH_DURATION_SECONDS,
            },
            sample_rate: TARGET_SAMPLE_RATE.cast_signed(),
            num_threads: 1,
            provider: Some("cpu".to_owned()),
            debug: false,
            ..VadModelConfig::default()
        };
        let detector = VoiceActivityDetector::create(&config, VAD_BUFFER_SIZE_SECONDS)
            .ok_or(SileroVadError::NativeConstruction)?;
        Ok(Self { detector })
    }
}

impl VoiceActivity for SileroVoiceActivity {
    fn is_speech(&mut self, frame: &[f32]) -> bool {
        self.detector.accept_waveform(frame);
        let speech = self.detector.detected();
        // Native segments are unused: the core segmenter owns segment boundaries, and clearing
        // them keeps the native queue from retaining audio for the whole recording.
        self.detector.clear();
        speech
    }
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
enum SileroVadError {
    #[error("Silero VAD model path is invalid")]
    InvalidModelPath,
    #[error("Silero VAD construction failed")]
    NativeConstruction,
}

#[cfg(test)]
mod tests {
    use sha2::{Digest as _, Sha256};
    use whispernow_core::workflow::{AudioSegment, segment_recording};

    use super::*;

    const SECOND: usize = TARGET_SAMPLE_RATE as usize;

    fn bundled_model_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../resources/vad")
            .join(SILERO_VAD_MODEL_FILE)
    }

    fn native_speech_fixture() -> Vec<f32> {
        const PCM: &[u8] = include_bytes!("../tests/fixtures/flite-testing-one-two-three.s16le");
        assert_eq!(PCM.len(), 59_040, "fixture byte length changed");
        assert_eq!(
            crate::models::hex_lower(&Sha256::digest(PCM)),
            "710ace5fc266bba391f71746a1ff7999a59d92d9af6e46a6d65c11c84df03349",
            "fixture content changed without updating its provenance"
        );
        PCM.chunks_exact(2)
            .map(|bytes| f32::from(i16::from_le_bytes([bytes[0], bytes[1]])) / 32_768.0)
            .collect()
    }

    fn fixture_speech_without_edge_silence() -> Vec<f32> {
        // The pinned fixture has 250 ms of leading silence and begins its
        // trailing silence at approximately 1.64 s.
        native_speech_fixture()[4_000..26_240].to_vec()
    }

    fn segments(samples: &[f32]) -> Vec<AudioSegment> {
        segment_recording(
            samples,
            Box::new(SileroVoiceActivity::new(&bundled_model_path()).expect("native detector")),
        )
    }

    #[test]
    fn an_unavailable_model_fails_open_to_treating_every_frame_as_speech() {
        let mut activity = silero_voice_activity(Path::new("missing.onnx"));
        assert!(activity.is_speech(&[0.0; 512]));
    }

    #[test]
    fn silence_an_isolated_impulse_and_low_background_noise_have_no_segments() {
        let silence = vec![0.0; SECOND];
        assert!(segments(&silence).is_empty());

        let mut impulse = silence;
        impulse[SECOND / 2] = 1.0;
        assert!(segments(&impulse).is_empty());
        assert!(segments(&impulse).is_empty());

        // Two seconds of fixed-seed xorshift32 white noise bounded to 0.01
        // normalized amplitude (approximately -40 dBFS peak). This is a
        // deterministic stand-in for steady low-level room or microphone noise.
        let mut state = 0x4d59_5df4_u32;
        let background_noise = (0..SECOND * 2)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let upper = u16::try_from(state >> 16).expect("upper half fits u16");
                let unit = f32::from(upper) / f32::from(u16::MAX);
                (unit * 2.0 - 1.0) * 0.01
            })
            .collect::<Vec<_>>();
        assert!(segments(&background_noise).is_empty());
    }

    #[test]
    fn speech_surrounded_by_silence_is_one_padded_segment() {
        let speech = fixture_speech_without_edge_silence();
        let mut surrounded = vec![0.0; SECOND];
        surrounded.extend_from_slice(&speech);
        surrounded.resize(surrounded.len() + SECOND, 0.0);

        let found = segments(&surrounded);

        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].start < SECOND, "{found:?}");
        assert!(found[0].end > SECOND + speech.len(), "{found:?}");
    }

    #[test]
    fn speech_ending_at_a_partial_final_frame_is_kept() {
        let speech = fixture_speech_without_edge_silence();
        assert_ne!(
            speech.len() % FRAME_SAMPLES,
            0,
            "fixture must exercise EOF padding"
        );

        assert_eq!(segments(&speech).len(), 1);
    }

    #[test]
    fn each_recording_gets_fresh_detector_state() {
        let speech = fixture_speech_without_edge_silence();
        let silence = vec![0.0; SECOND];

        let first = segments(&speech);
        assert_eq!(first.len(), 1);
        assert!(segments(&silence).is_empty());
        assert_eq!(segments(&speech), first);
    }

    #[test]
    fn pauses_between_utterances_split_segments() {
        let speech = fixture_speech_without_edge_silence();
        let mut recording = vec![0.0; SECOND / 2];
        recording.extend_from_slice(&speech);
        recording.resize(recording.len() + SECOND, 0.0);
        recording.extend_from_slice(&speech);
        recording.resize(recording.len() + SECOND / 2, 0.0);

        assert_eq!(segments(&recording).len(), 2);
    }
}
