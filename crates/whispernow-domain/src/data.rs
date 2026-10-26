use serde::{Deserialize, Serialize};

/// What a data location holds; each maps to paths the platform owns.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClearDataCategory {
    Settings,
    History,
    Models,
}
