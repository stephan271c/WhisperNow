//! Framework-independent audio capture and processing contracts.
//!
//! Native callback ownership belongs to a platform adapter. The types in this
//! module are deliberately independent of CPAL and UI frameworks.

mod level;
mod pipeline;
mod ports;
mod processing;

use level::AudioLevelAccumulator;
pub use level::AudioLevelSummary;
pub use pipeline::AudioProcessor;
pub use ports::{
    AudioCapture, AudioDeviceChoice, AudioDeviceDescriptor, AudioDeviceId, AudioEvent, AudioInput,
    AudioInputError, AudioObservation, AudioSink, AudioSinkError, StartCaptureRequest,
};
pub use processing::{AudioBlock, TARGET_SAMPLE_RATE};
use processing::{AudioProcessingError, StreamingResampler, convert_interleaved_to_mono};
