use std::sync::Arc;

use thiserror::Error;
use whispernow_domain::AudioDeviceId;
use whispernow_domain::ipc::AudioDevice;

use super::{AudioBlock, AudioLevelSummary};

/// No event contains captured audio.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AudioEvent {
    DeviceLost,
    StreamError,
    Backpressure { dropped_frames: usize },
}

/// Consumer called from an adapter's worker, never its real-time audio callback, so a
/// sink may block briefly.
pub trait AudioSink: Send + Sync {
    /// Delivers normalized mono 16 kHz audio.
    fn samples(&self, block: AudioBlock);

    fn observe(&self, summary: AudioLevelSummary);

    fn event(&self, event: AudioEvent);
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum AudioInputError {
    #[error("audio input device was not found")]
    InvalidDevice,
    #[error("audio input device is unavailable")]
    DeviceUnavailable,
    #[error("audio input device was lost")]
    DeviceLost,
    #[error("microphone permission was denied")]
    PermissionDenied,
    #[error("no supported input configuration is available")]
    UnsupportedConfiguration,
    #[error("audio backend failed during {operation}")]
    Backend { operation: &'static str },
}

/// Native callbacks only enqueue bounded raw data; conversion and sink calls happen on a worker.
pub trait AudioInput: Send + Sync {
    fn list_input_devices(&self) -> Result<Vec<AudioDevice>, AudioInputError>;

    /// Records from `device`, or from the system default (`None`) resolved now: a later default
    /// change is an observable transition, never a silent stream replacement. Dropping the
    /// returned capture stops the recording.
    fn start(
        &self,
        device: Option<AudioDeviceId>,
        sink: Arc<dyn AudioSink>,
    ) -> Result<Box<dyn AudioCapture>, AudioInputError>;
}

pub trait AudioCapture: Send {
    /// Stops the stream and delivers the audio still queued.
    fn stop(self: Box<Self>) -> Result<(), AudioInputError>;
}
