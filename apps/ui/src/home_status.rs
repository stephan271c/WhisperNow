use whispernow_domain::ipc::{MainWindowSection, ModelState, Phase};

use crate::UiState;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HomeStatusKind {
    Ready,
    Paused,
    Recording,
    Working,
    Waiting,
    Problem,
}

/// Home's live status line, with a link to the page that fixes a problem.
#[derive(Clone, Debug, PartialEq)]
pub struct HomeStatus {
    pub kind: HomeStatusKind,
    pub title: String,
    pub detail: Option<String>,
    pub fix: Option<(&'static str, MainWindowSection)>,
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
        if let Some(error) = &state.error {
            return problem(
                "Dictation needs attention",
                &error.summary,
                category_fix(error.category),
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
        if snapshot.model_state == ModelState::Failed {
            return problem(
                "Dictation is unavailable",
                "The speech model could not be loaded.",
                category_fix(whispernow_domain::ErrorCategory::Model),
            );
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
            _ if matches!(
                snapshot.model_state,
                ModelState::Downloading | ModelState::Loading
            ) =>
            {
                status(HomeStatusKind::Waiting, "Loading speech model", None, None)
            }
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
            Phase::Idle | Phase::Error { .. } => status(HomeStatusKind::Ready, "Ready", None, None),
        }
    }
}

pub(crate) const fn category_fix(
    category: whispernow_domain::ErrorCategory,
) -> Option<(&'static str, MainWindowSection)> {
    use whispernow_domain::ErrorCategory;
    match category {
        ErrorCategory::Audio | ErrorCategory::Permission => {
            Some(("Check audio settings", MainWindowSection::Audio))
        }
        ErrorCategory::Model | ErrorCategory::Recognition => {
            Some(("Open speech models", MainWindowSection::SpeechModels))
        }
        ErrorCategory::Cleanup => Some(("Open AI cleanup", MainWindowSection::AiCleanup)),
        ErrorCategory::Settings => Some(("Open settings", MainWindowSection::Settings)),
        ErrorCategory::Output => Some(("Open history", MainWindowSection::History)),
        ErrorCategory::General | ErrorCategory::NotFound | ErrorCategory::HistoryUnavailable => {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{HomeStatus, HomeStatusKind};
    use crate::UiState;
    use whispernow_domain::{AppError, ErrorCategory};

    #[test]
    fn connection_failures_have_no_fix_from_an_earlier_dictation_error() {
        for (connection_failed, event_stream_failed, title) in [
            (true, false, "WhisperNow could not connect"),
            (false, true, "Live updates are unavailable"),
        ] {
            let state = UiState {
                connection_failed,
                event_stream_failed,
                error: Some(AppError::new(
                    ErrorCategory::Audio,
                    "microphone_unavailable",
                    "Choose an available microphone",
                )),
                ..UiState::default()
            };
            let status = HomeStatus::from_state(&state);
            assert_eq!(status.kind, HomeStatusKind::Problem);
            assert_eq!(status.title, title);
            assert_eq!(status.fix, None);
        }
    }
}
