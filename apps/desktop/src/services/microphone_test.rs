//! The probe keeps only level summaries. Audio blocks are dropped as they
//! arrive, so a test never stores or transcribes speech.

use std::{
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use whispernow_domain::AppError;

use whispernow_core::audio::{
    AudioBlock, AudioDeviceChoice, AudioEvent, AudioInput, AudioLevelSummary, AudioSink,
};
use whispernow_domain::ipc::{ErrorCategory, MicrophoneTestResponse};

use super::capture_start_error;

pub(super) const TEST_DURATION: Duration = Duration::from_secs(3);
/// How long setup listens to make the system ask for the microphone.
pub(super) const PERMISSION_CHECK_DURATION: Duration = Duration::from_millis(300);
const POLL_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Default)]
struct ProbeState {
    levels: Vec<f32>,
    peak: f32,
    capture_failed: bool,
}

const MAX_LEVELS: usize = 64;

#[derive(Default)]
struct LevelProbe(Mutex<ProbeState>);

impl LevelProbe {
    fn with<R>(&self, work: impl FnOnce(&mut ProbeState) -> R) -> R {
        work(&mut self.0.lock().expect("microphone test level lock poisoned"))
    }
}

impl AudioSink for LevelProbe {
    fn samples(&self, _block: AudioBlock) {}

    fn observe(&self, summary: AudioLevelSummary) {
        self.with(|state| {
            if state.levels.len() < MAX_LEVELS {
                state.levels.push(summary.rms);
            }
            state.peak = state.peak.max(summary.peak);
        });
    }

    fn event(&self, event: AudioEvent) {
        if matches!(event, AudioEvent::DeviceLost | AudioEvent::StreamError) {
            self.with(|state| state.capture_failed = true);
        }
    }
}

/// Captures from `device` for at most `duration`, stopping early when `cancelled` turns true
/// or the device goes away, and returns the observed levels.
pub(super) fn run(
    audio: &dyn AudioInput,
    device: AudioDeviceChoice,
    duration: Duration,
    cancelled: impl Fn() -> bool,
) -> Result<MicrophoneTestResponse, AppError> {
    let probe = Arc::new(LevelProbe::default());
    let capture = audio
        .start(device, probe.clone())
        .map_err(|error| capture_start_error(&error))?;
    let started = Instant::now();
    while started.elapsed() < duration && !cancelled() && !probe.with(|state| state.capture_failed)
    {
        thread::sleep(POLL_INTERVAL.min(duration.saturating_sub(started.elapsed())));
    }
    let _stopped = capture.stop().is_ok();
    let state = probe.with(std::mem::take);
    if state.capture_failed {
        return Err(AppError::new(
            ErrorCategory::Audio,
            "microphone_lost",
            "The microphone stopped during the test. Check that it is connected and try again.",
        ));
    }
    Ok(MicrophoneTestResponse {
        levels: state.levels,
        peak: state.peak,
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use whispernow_core::audio::{AudioDeviceChoice, AudioInputError};

    use super::run;
    use crate::services::test_support::FakeAudioInput;

    #[test]
    fn a_test_is_bounded_and_always_stops_capture() {
        let input = FakeAudioInput::levels();
        let response = run(
            &input,
            AudioDeviceChoice::SystemDefault,
            Duration::from_millis(20),
            || false,
        )
        .expect("test runs");

        assert_eq!(response.levels.len(), 64);
        assert!((response.peak - 1.0).abs() < f32::EPSILON);
        assert_eq!(*input.calls.lock().expect("calls"), ["start", "stop"]);
    }

    #[test]
    fn a_cancelled_test_stops_at_once_and_a_refused_start_reports_permission() {
        let input = FakeAudioInput::levels();
        run(
            &input,
            AudioDeviceChoice::SystemDefault,
            Duration::from_mins(1),
            || true,
        )
        .expect("cancelled test still returns its levels");
        assert_eq!(*input.calls.lock().expect("calls"), ["start", "stop"]);

        let refused = FakeAudioInput::refusing(AudioInputError::PermissionDenied);
        let error = run(
            &refused,
            AudioDeviceChoice::SystemDefault,
            Duration::from_mins(1),
            || false,
        )
        .expect_err("permission denied");
        assert_eq!(error.category, whispernow_domain::ErrorCategory::Permission);
    }
}
