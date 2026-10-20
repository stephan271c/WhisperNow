mod cpal_backend;
mod service;

pub use service::CpalAudioInput;

use cpal_backend::backend;
pub(crate) use cpal_backend::{
    CaptureBackend, CaptureStream, CpalBackend, OpenedCapture, StreamFault,
};
