//! Pure batch transcription preparation. Phase, recording ownership, cancellation, history and
//! output belong to the desktop.

mod join;
mod pipeline;
mod segmentation;

pub use join::join_segment_texts;

pub use pipeline::{
    PreparedTranscript, TranscriptFailure, apply_vocabulary_replacements, has_microphone_audio,
    prepare_segmented,
};
pub use segmentation::{AlwaysSpeech, AudioSegment, FRAME_SAMPLES, SpeechSegmenter, VoiceActivity};

#[cfg(test)]
mod tests;
