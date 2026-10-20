use std::sync::Arc;

use thiserror::Error;
use whispernow_domain::ipc::AudioDevice;

use super::{AudioBlock, AudioLevelSummary};

/// Opaque native input-device identity. Callers must not parse this value.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct AudioDeviceId(String);

impl AudioDeviceId {
    /// `None` for a blank identifier.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Option<Self> {
        let value = value.into();
        (!value.trim().is_empty()).then_some(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum AudioDeviceChoice {
    /// Resolve the system default at start time. A later default change is an
    /// observable transition, never a silent stream replacement.
    SystemDefault,
    Device(AudioDeviceId),
}

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
    /// Lists connected input devices, marking the current system default.
    fn list_input_devices(&self) -> Result<Vec<AudioDevice>, AudioInputError>;

    /// Starts one recording. Dropping the returned capture stops it.
    fn start(
        &self,
        device: AudioDeviceChoice,
        sink: Arc<dyn AudioSink>,
    ) -> Result<Box<dyn AudioCapture>, AudioInputError>;
}

pub trait AudioCapture: Send {
    /// Stops the stream and delivers the audio still queued.
    fn stop(self: Box<Self>) -> Result<(), AudioInputError>;
}
