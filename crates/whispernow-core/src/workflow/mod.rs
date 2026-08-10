//! Pure batch transcription preparation.
//!
//! Application phase, recording ownership, cancellation, history, and output are desktop
//! responsibilities. This module splits one immutable 16-kHz recording into speech segments,
//! decodes each segment as an ordinary offline batch request, and joins the results into one
//! final transcript without workflow sessions or destination identity.

mod join;
mod pipeline;
mod presence;
mod segmentation;
mod types;

pub use join::join_segment_texts;

pub use pipeline::{apply_vocabulary_replacements, prepare_segmented, recognize_segment};
pub use presence::has_microphone_audio;
pub use segmentation::{
    AlwaysSpeech, AudioSegment, FRAME_SAMPLES, SpeechSegmenter, VoiceActivity, VoiceActivitySource,
    segment_recording,
};
pub use types::{BATCH_SAMPLE_RATE_HZ, PipelineFailureKind, PreparedTranscription};

/// Exact, case-sensitive marker emitted by the current ASR worker.
pub(crate) const UNKNOWN_TOKEN_PATTERN: &str = "<unk>";

#[cfg(test)]
mod tests;
