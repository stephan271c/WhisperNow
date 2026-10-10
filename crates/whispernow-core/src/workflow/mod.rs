//! Pure batch transcription preparation. Phase, recording ownership, cancellation, history and
//! output belong to the desktop.

mod join;
mod pipeline;
mod presence;
mod segmentation;
mod types;

pub use join::join_segment_texts;

pub use pipeline::{apply_vocabulary_replacements, prepare_segmented, recognize_segment};
pub use presence::has_microphone_audio;
pub use segmentation::{
    AlwaysSpeech, AudioSegment, FRAME_SAMPLES, SpeechSegmenter, VoiceActivity, segment_recording,
};
pub use types::{PipelineFailureKind, PreparedTranscription};

#[cfg(test)]
mod tests;
