use std::{sync::Arc, time::Duration};

use thiserror::Error;

use super::{AudioBlock, AudioLevelSummary};

/// Opaque native input-device identity. Callers must not parse this value.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct AudioDeviceId(String);

impl AudioDeviceId {
    pub fn new(value: impl Into<String>) -> Result<Self, AudioInputError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(AudioInputError::InvalidDeviceId);
        }
        Ok(Self(value))
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioDeviceDescriptor {
    pub choice: AudioDeviceChoice,
    pub display_name: String,
    pub is_current_default: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AudioObservation {
    StreamReady { request_to_stream_ready: Duration },
    FirstSamples { request_to_first_sample: Duration },
    Level { summary: AudioLevelSummary },
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
    fn samples(&self, block: AudioBlock) -> Result<(), AudioSinkError>;

    fn observe(&self, observation: AudioObservation);

    fn event(&self, event: AudioEvent);
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum AudioSinkError {
    #[error("audio consumer is closed")]
    Closed,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum AudioInputError {
    #[error("audio device identifier is empty")]
    InvalidDeviceId,
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
    /// Lists an explicit `SystemDefault` entry plus stable physical choices.
    fn list_input_devices(&self) -> Result<Vec<AudioDeviceDescriptor>, AudioInputError>;

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
