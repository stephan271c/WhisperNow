//! The kinds of application data the user can clear.

use serde::{Deserialize, Serialize};

/// A fixed class of application-owned data that may be cleared. Each maps to paths the
/// platform owns; a request cannot name arbitrary paths.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClearDataCategory {
    Settings,
    History,
    Models,
    Logs,
}

impl ClearDataCategory {
    pub const ALL: [Self; 4] = [Self::Settings, Self::History, Self::Models, Self::Logs];
}
