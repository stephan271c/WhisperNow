use whispernow_domain::ipc::{MainWindowSection, ModelState, PermissionKind, Phase};
use whispernow_domain::{AppError, TEXT_OUTPUT_PERMISSION_DENIED};

use crate::UiState;
use crate::client::{Platform, host_platform};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HomeStatusKind {
    Ready,
    Paused,
    Recording,
    Working,
    Waiting,
    Problem,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Fix {
    pub label: &'static str,
    pub target: FixTarget,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixTarget {
    Page(MainWindowSection),
    SystemSettings(PermissionKind),
}

#[derive(Clone, Debug, PartialEq)]
pub struct HomeStatus {
    pub kind: HomeStatusKind,
    pub title: String,
    pub detail: Option<String>,
    pub fix: Option<Fix>,
}

impl HomeStatus {
    #[must_use]
    pub fn from_state(state: &UiState) -> Self {
        let status = |kind, title: &str, detail: Option<&str>, fix| Self {
            kind,
            title: title.to_owned(),
            detail: detail.map(str::to_owned),
            fix,
        };
        let problem =
            |title, detail, fix| status(HomeStatusKind::Problem, title, Some(detail), fix);
        if state.event_stream_failed {
            return problem(
                "Live updates are unavailable",
                "Close and reopen this window to reconnect.",
                None,
            );
        }
        if state.connection_failed {
            return problem(
                "WhisperNow could not connect",
                "Close and reopen this window to retry.",
                None,
            );
        }
        if let Some(error) = state
            .snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.error.as_ref())
        {
            return problem(
                if concerns_dictation(error.category) {
                    "Dictation needs attention"
                } else {
                    "WhisperNow needs attention"
                },
                &error.summary,
                error_fix(error),
            );
        }
        let Some(snapshot) = &state.snapshot else {
            return status(
                HomeStatusKind::Waiting,
                "Connecting to WhisperNow",
                None,
                None,
            );
        };
        let model_fix = category_fix(whispernow_domain::ErrorCategory::Model);
        if snapshot.model_state == ModelState::Failed {
            let detail = if snapshot.failed_download.as_ref() == Some(&snapshot.selected_model_id) {
                "The speech model could not be downloaded."
            } else {
                "The speech model could not be loaded."
            };
            return problem("Dictation is unavailable", detail, model_fix);
        }
        match snapshot.phase {
            Phase::Paused => status(
                HomeStatusKind::Paused,
                "Paused",
                Some("Resume WhisperNow from the system tray to record again."),
                None,
            ),
            Phase::Updating => status(
                HomeStatusKind::Waiting,
                "Installing an update",
                Some("WhisperNow will be ready after the update finishes."),
                None,
            ),
            Phase::ClearingData => status(
                HomeStatusKind::Waiting,
                "Clearing application data",
                None,
                None,
            ),
            Phase::Recording { .. } => status(HomeStatusKind::Recording, "Recording", None, None),
            Phase::Processing { cleaning_up } => status(
                HomeStatusKind::Working,
                if cleaning_up {
                    "Cleaning up transcript"
                } else {
                    "Transcribing recording"
                },
                None,
                None,
            ),
            _ if matches!(
                snapshot.model_state,
                ModelState::Downloading | ModelState::Loading
            ) =>
            {
                status(HomeStatusKind::Waiting, "Loading speech model", None, None)
            }
            Phase::Idle | Phase::Error if snapshot.model_state == ModelState::Unloaded => problem(
                "Speech model not ready",
                "Download or choose a speech model to start dictating.",
                model_fix,
            ),
            Phase::Idle | Phase::Error => status(HomeStatusKind::Ready, "Ready", None, None),
        }
    }
}

const fn concerns_dictation(category: whispernow_domain::ErrorCategory) -> bool {
    use whispernow_domain::ErrorCategory;
    matches!(
        category,
        ErrorCategory::Audio
            | ErrorCategory::Permission
            | ErrorCategory::Model
            | ErrorCategory::Recognition
            | ErrorCategory::Cleanup
            | ErrorCategory::Output
    )
}

pub(crate) const fn category_fix(category: whispernow_domain::ErrorCategory) -> Option<Fix> {
    use whispernow_domain::ErrorCategory;
    let (label, section) = match category {
        ErrorCategory::Audio | ErrorCategory::Permission => {
            ("Check audio settings", MainWindowSection::Audio)
        }
        ErrorCategory::Model | ErrorCategory::Recognition => {
            ("Open speech models", MainWindowSection::SpeechModels)
        }
        ErrorCategory::Cleanup => ("Open AI cleanup", MainWindowSection::AiCleanup),
        ErrorCategory::Settings => ("Open settings", MainWindowSection::Settings),
        ErrorCategory::Output => ("Open history", MainWindowSection::History),
        ErrorCategory::General | ErrorCategory::NotFound | ErrorCategory::HistoryUnavailable => {
            return None;
        }
    };
    Some(Fix {
        label,
        target: FixTarget::Page(section),
    })
}

/// macOS grants typing in its own settings, so the failure points there instead of at ours.
pub(crate) fn error_fix(error: &AppError) -> Option<Fix> {
    if error.code == TEXT_OUTPUT_PERMISSION_DENIED && host_platform() == Platform::Macos {
        return Some(Fix {
            label: "Open system settings",
            target: FixTarget::SystemSettings(PermissionKind::Accessibility),
        });
    }
    category_fix(error.category)
}
