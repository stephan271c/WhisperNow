//! What Home says about dictation, from the application state.

use whispernow_api::{ModelStateDto, PhaseDto};

use crate::{EventStreamHealth, Route, SettingsRoute, UiState};

/// What Home says about dictation right now.
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
        if snapshot.model_state == ModelStateDto::Failed {
            return problem(
                "Dictation is unavailable",
                "The speech model could not be loaded.",
            );
        }
        match snapshot.phase {
            PhaseDto::Paused => status(
                HomeStatusKind::Paused,
                "Paused",
                Some("Resume WhisperNow from the system tray to record again."),
            ),
            PhaseDto::Updating => status(
                HomeStatusKind::Waiting,
                "Installing an update",
                Some("WhisperNow will be ready after the update finishes."),
            ),
            _ if state.is_loading_model() => {
                status(HomeStatusKind::Waiting, "Loading speech model", None)
            }
            PhaseDto::Recording { .. } => status(HomeStatusKind::Recording, "Recording", None),
            PhaseDto::Processing { cleaning_up } => status(
                HomeStatusKind::Working,
                if cleaning_up {
                    "Cleaning up transcript"
                } else {
                    "Transcribing recording"
                },
                None,
            ),
            PhaseDto::Idle | PhaseDto::Error { .. } => status(HomeStatusKind::Ready, "Ready", None),
        }
    }
}

/// The page that resolves the current problem, if one does.
fn problem_fix(state: &UiState) -> Option<(&'static str, Route)> {
    use whispernow_api::ErrorCategoryDto as Category;
    let model_failed = state
        .snapshot
        .as_ref()
        .is_some_and(|snapshot| snapshot.model_state == ModelStateDto::Failed);
    if model_failed {
        return Some((
            "Open speech models",
            Route::Settings(SettingsRoute::SpeechModels),
        ));
    }
    match state.error.as_ref()?.category {
        Category::Audio | Category::Permission => Some((
            "Check audio settings",
            Route::Settings(SettingsRoute::Audio),
        )),
        Category::Model | Category::Recognition => Some((
            "Open speech models",
            Route::Settings(SettingsRoute::SpeechModels),
        )),
        Category::Output => Some(("Open history", Route::History)),
        _ => None,
    }
}
