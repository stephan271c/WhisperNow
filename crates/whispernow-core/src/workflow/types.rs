use crate::llm::CleanupFallback;

/// Has not yet caused history or output side effects.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedTranscription {
    pub raw_text: String,
    /// The raw text after vocabulary replacements: what AI cleanup edits.
    pub deterministic_text: String,
    pub final_text: String,
    /// Why AI cleanup kept the deterministic text.
    pub fallback: Option<CleanupFallback>,
}

impl PreparedTranscription {
    pub fn apply_cleanup(&mut self, cleanup: Result<String, CleanupFallback>) {
        match cleanup {
            Ok(text) => self.final_text = text,
            Err(fallback) => self.fallback = Some(fallback),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineFailureKind {
    NoAudio,
    NoSpeech,
    RecognitionFailed,
    EmptyTranscription,
    UnknownTokens,
}
