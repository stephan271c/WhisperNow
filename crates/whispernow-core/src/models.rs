use std::path::PathBuf;

use whispernow_domain::{ModelAvailability, ModelDefinition};

/// Validated model root passed only to native/platform adapters.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedModel {
    pub definition: ModelDefinition,
    pub root: PathBuf,
    /// Never `NotInstalled`.
    pub availability: ModelAvailability,
}
