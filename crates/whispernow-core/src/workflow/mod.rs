//! Pure batch transcription preparation. Phase, recording ownership, cancellation, history and
//! output belong to the desktop.

mod join;
mod pipeline;
mod segmentation;

pub use pipeline::{
    PreparedTranscript, TranscriptFailure, has_microphone_audio, prepare_segmented,
};
pub use segmentation::{AlwaysSpeech, AudioSegment, FRAME_SAMPLES, SpeechSegmenter, VoiceActivity};

#[cfg(test)]
mod tests;
