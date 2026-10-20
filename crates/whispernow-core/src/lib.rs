pub mod asr;
pub mod audio;
pub mod llm;
pub mod text_output;
pub mod workflow;

use std::path::PathBuf;

use whispernow_domain::ModelDefinition;

pub use tokio_util::sync::CancellationToken;

/// Validated model root passed only to native/platform adapters.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedModel {
    pub definition: ModelDefinition,
    pub root: PathBuf,
}
