mod cpal_audio_input;
mod cpal_backend;

pub use cpal_audio_input::CpalAudioInput;

use cpal_backend::backend;
pub(crate) use cpal_backend::{
    CaptureBackend, CaptureStream, CpalBackend, FaultFlag, OpenedCapture, StreamFault,
};
