use whispernow_domain::ipc::{ModelState, Phase};

use crate::{EventStreamHealth, Route, SettingsRoute, UiState};

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
    pub fix: Option<(&'static str, Route)>,
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
        if state.event_stream_health == EventStreamHealth::Failed {
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

fn problem_fix(state: &UiState) -> Option<(&'static str, Route)> {
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
) -> Option<(&'static str, Route)> {
    use whispernow_domain::ErrorCategory;
    match category {
        ErrorCategory::Audio | ErrorCategory::Permission => Some((
            "Check audio settings",
            Route::Settings(SettingsRoute::Audio),
        )),
        ErrorCategory::Model | ErrorCategory::Recognition => Some((
            "Open speech models",
            Route::Settings(SettingsRoute::SpeechModels),
        )),
        ErrorCategory::Cleanup => {
            Some(("Open AI cleanup", Route::Settings(SettingsRoute::AiCleanup)))
        }
        ErrorCategory::Settings => Some(("Open settings", Route::Settings(SettingsRoute::General))),
        ErrorCategory::Output => Some(("Open history", Route::History)),
        ErrorCategory::General | ErrorCategory::NotFound | ErrorCategory::HistoryUnavailable => {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::category_fix;
    use crate::{Route, SettingsRoute};
    use whispernow_domain::ErrorCategory;

    #[test]
    fn each_error_category_has_one_fix() {
        let audio = Some((
            "Check audio settings",
            Route::Settings(SettingsRoute::Audio),
        ));
        let model = Some((
            "Open speech models",
            Route::Settings(SettingsRoute::SpeechModels),
        ));
        let cleanup = Some(("Open AI cleanup", Route::Settings(SettingsRoute::AiCleanup)));
        let settings = Some(("Open settings", Route::Settings(SettingsRoute::General)));
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
                Some(("Open history", Route::History)),
            ),
            (ErrorCategory::HistoryUnavailable, None),
            (ErrorCategory::Settings, settings),
        ] {
            assert_eq!(category_fix(category), fix, "{category:?}");
        }
    }
}
