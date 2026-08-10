mod cpal_backend;
mod service;

pub use service::CpalAudioInput;

pub(crate) use cpal_backend::{
    CaptureBackend, CaptureStream, CpalBackend, OpenedCapture, StreamFault,
};
