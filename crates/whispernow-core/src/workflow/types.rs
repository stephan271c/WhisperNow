use whispernow_domain::UtcMillis;

use crate::llm::CleanupFallback;

/// Sample rate accepted by the batch transcription pipeline.
pub const BATCH_SAMPLE_RATE_HZ: u32 = 16_000;

/// Fully prepared text. It has not yet caused history or output side effects.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedTranscription {
    pub timestamp: UtcMillis,
    pub raw_text: String,
    /// The raw text after vocabulary replacements: what AI cleanup edits.
    pub deterministic_text: String,
    pub final_text: String,
    /// Why AI cleanup kept the deterministic text.
    pub fallback: Option<CleanupFallback>,
}

impl PreparedTranscription {
    /// Uses AI cleanup's text, or keeps the deterministic text and records why.
    pub fn apply_cleanup(&mut self, cleanup: Result<String, CleanupFallback>) {
        match cleanup {
            Ok(text) => self.final_text = text,
            Err(fallback) => self.fallback = Some(fallback),
        }
    }
}

/// Failure classes produced before history or output side effects are permitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineFailureKind {
    NoAudio,
    NoSpeech,
    Cancelled,
    RecognitionFailed,
    EmptyTranscription,
    UnknownTokens,
}
