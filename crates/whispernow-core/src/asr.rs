use thiserror::Error;
use whispernow_domain::ModelId;

use crate::ResolvedModel;

/// ASR failures. They carry no model paths or native error strings.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum AsrError {
    #[error("ASR recognizer construction failed")]
    LoadFailed,
    #[error("ASR transcription failed")]
    TranscriptionFailed,
}

/// Speech recognition with one resident model.
pub trait AsrEngine: Send {
    fn loaded_model_id(&self) -> Option<&ModelId>;

    /// Builds a recognizer for `model`, then replaces the resident one. A failed load leaves
    /// the previous model resident.
    fn load(&mut self, model: &ResolvedModel) -> Result<(), AsrError>;

    /// Decodes one mono 16-kHz recording. A failure fails this call only.
    fn transcribe(&mut self, samples: &[f32]) -> Result<String, AsrError>;

    fn unload(&mut self);
}
