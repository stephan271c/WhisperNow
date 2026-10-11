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
        let status = |kind, title: &str, detail: Option<&str>| Self {
            kind,
            title: title.to_owned(),
            detail: detail.map(str::to_owned),
            fix: (kind == HomeStatusKind::Problem)
                .then(|| problem_fix(state))
                .flatten(),
        };
        let problem = |title, detail| status(HomeStatusKind::Problem, title, Some(detail));
        if state.event_stream_failed {
            return problem(
                "Live updates are unavailable",
                "Close and reopen this window to reconnect.",
            );
        }
        if state.connection_failed {
            return problem(
                "WhisperNow could not connect",
                "Close and reopen this window to retry.",
            );
        }
        if let Some(error) = &state.error {
            return problem("Dictation needs attention", &error.summary);
        }
        let Some(snapshot) = &state.snapshot else {
            return status(HomeStatusKind::Waiting, "Connecting to WhisperNow", None);
        };
        if snapshot.model_state == ModelState::Failed {
            return problem(
                "Dictation is unavailable",
                "The speech model could not be loaded.",
            );
        }
        match snapshot.phase {
            Phase::Paused => status(
                HomeStatusKind::Paused,
                "Paused",
                Some("Resume WhisperNow from the system tray to record again."),
            ),
            Phase::Updating => status(
                HomeStatusKind::Waiting,
                "Installing an update",
                Some("WhisperNow will be ready after the update finishes."),
            ),
            Phase::ClearingData => {
                status(HomeStatusKind::Waiting, "Clearing application data", None)
            }
            _ if state.is_loading_model() => {
                status(HomeStatusKind::Waiting, "Loading speech model", None)
            }
            Phase::Recording { .. } => status(HomeStatusKind::Recording, "Recording", None),
            Phase::Processing { cleaning_up } => status(
                HomeStatusKind::Working,
                if cleaning_up {
                    "Cleaning up transcript"
                } else {
                    "Transcribing recording"
                },
                None,
            ),
            Phase::Idle | Phase::Error { .. } => status(HomeStatusKind::Ready, "Ready", None),
        }
    }
}

fn problem_fix(state: &UiState) -> Option<(&'static str, MainWindowSection)> {
    if let Some(error) = &state.error {
        return category_fix(error.category);
    }
    state
        .snapshot
        .as_ref()
        .filter(|snapshot| snapshot.model_state == ModelState::Failed)
        .and_then(|_| category_fix(whispernow_domain::ErrorCategory::Model))
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
    use super::category_fix;
    use whispernow_domain::ErrorCategory;
    use whispernow_domain::ipc::MainWindowSection;

    #[test]
    fn each_error_category_has_one_fix() {
        let audio = Some(("Check audio settings", MainWindowSection::Audio));
        let model = Some(("Open speech models", MainWindowSection::SpeechModels));
        let cleanup = Some(("Open AI cleanup", MainWindowSection::AiCleanup));
        let settings = Some(("Open settings", MainWindowSection::Settings));
        for (category, fix) in [
            (ErrorCategory::General, None),
            (ErrorCategory::NotFound, None),
            (ErrorCategory::Audio, audio),
            (ErrorCategory::Permission, audio),
            (ErrorCategory::Model, model),
            (ErrorCategory::Recognition, model),
            (ErrorCategory::Cleanup, cleanup),
            (
                ErrorCategory::Output,
                Some(("Open history", MainWindowSection::History)),
            ),
            (ErrorCategory::HistoryUnavailable, None),
            (ErrorCategory::Settings, settings),
        ] {
            assert_eq!(category_fix(category), fix, "{category:?}");
        }
    }
}
