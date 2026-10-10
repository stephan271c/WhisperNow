//! The probe keeps only level summaries. Audio blocks are dropped as they
//! arrive, so a test never stores or transcribes speech.

use std::{
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use whispernow_domain::AppError;

use whispernow_core::audio::{
    AudioBlock, AudioDeviceChoice, AudioEvent, AudioInput, AudioObservation, AudioSink,
    AudioSinkError,
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
    lost: bool,
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
    fn samples(&self, _block: AudioBlock) -> Result<(), AudioSinkError> {
        Ok(())
    }

    fn observe(&self, observation: AudioObservation) {
        if let AudioObservation::Level { summary } = observation {
            self.with(|state| {
                if state.levels.len() < MAX_LEVELS {
                    state.levels.push(unit(summary.rms));
                }
                state.peak = state.peak.max(unit(summary.peak));
            });
        }
    }

    fn event(&self, event: AudioEvent) {
        if matches!(event, AudioEvent::DeviceLost | AudioEvent::StreamError) {
            self.with(|state| state.lost = true);
        }
    }
}

fn unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
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
    while started.elapsed() < duration && !cancelled() && !probe.with(|state| state.lost) {
        thread::sleep(POLL_INTERVAL.min(duration.saturating_sub(started.elapsed())));
    }
    let _stopped = capture.stop().is_ok();
    let state = probe.with(std::mem::take);
    if state.lost {
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
    use std::{
        sync::{Arc, Mutex},
        time::Duration,
    };

    use whispernow_core::audio::{
        AudioCapture, AudioDeviceChoice, AudioDeviceDescriptor, AudioInput, AudioInputError,
        AudioLevelSummary, AudioObservation, AudioSink,
    };

    use super::run;

    #[derive(Default)]
    struct FakeInput {
        calls: Arc<Mutex<Vec<&'static str>>>,
        refuse: bool,
    }

    struct FakeCapture(Arc<Mutex<Vec<&'static str>>>);

    impl AudioCapture for FakeCapture {
        fn stop(self: Box<Self>) -> Result<(), AudioInputError> {
            self.0.lock().expect("calls").push("stop");
            Ok(())
        }
    }

    impl AudioInput for FakeInput {
        fn list_input_devices(&self) -> Result<Vec<AudioDeviceDescriptor>, AudioInputError> {
            Ok(Vec::new())
        }

        fn start(
            &self,
            _device: AudioDeviceChoice,
            sink: Arc<dyn AudioSink>,
        ) -> Result<Box<dyn AudioCapture>, AudioInputError> {
            self.calls.lock().expect("calls").push("start");
            if self.refuse {
                return Err(AudioInputError::PermissionDenied);
            }
            for rms in (0_u8..100).map(|step| f32::from(step) / 50.0) {
                sink.observe(AudioObservation::Level {
                    summary: AudioLevelSummary {
                        peak: rms,
                        rms,
                        sample_count: 800,
                    },
                });
            }
            Ok(Box::new(FakeCapture(Arc::clone(&self.calls))))
        }
    }

    #[test]
    fn a_test_is_bounded_clamped_and_always_stops_capture() {
        let input = FakeInput::default();
        let response = run(
            &input,
            AudioDeviceChoice::SystemDefault,
            Duration::from_millis(20),
            || false,
        )
        .expect("test runs");

        assert_eq!(response.levels.len(), 64);
        assert!(
            response
                .levels
                .iter()
                .all(|level| (0.0..=1.0).contains(level))
        );
        assert!((response.peak - 1.0).abs() < f32::EPSILON);
        assert_eq!(*input.calls.lock().expect("calls"), ["start", "stop"]);
    }

    #[test]
    fn a_cancelled_test_stops_at_once_and_a_refused_start_reports_permission() {
        let input = FakeInput::default();
        run(
            &input,
            AudioDeviceChoice::SystemDefault,
            Duration::from_mins(1),
            || true,
        )
        .expect("cancelled test still returns its levels");
        assert_eq!(*input.calls.lock().expect("calls"), ["start", "stop"]);

        let refused = FakeInput {
            refuse: true,
            ..FakeInput::default()
        };
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
