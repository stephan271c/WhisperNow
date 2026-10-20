use whispernow_core::{
    audio::TARGET_SAMPLE_RATE,
    llm::{CleanupFallback, LlmError},
    workflow::{PipelineFailureKind, PreparedTranscription, has_microphone_audio},
};
use whispernow_domain::AppError;
use whispernow_domain::ipc::{ErrorCategory, Phase, SetupTestResult};
use whispernow_domain::{CleanupTarget, EntryId, EntryStatus, NewHistoryEntry};

use super::{Recording, RecordingTrigger, Worker, capture::CaptureInterruption};
use crate::services::{output::OutputNotice, state::CancelReason};

pub(super) enum Outcome {
    Delivered {
        transcript: String,
        /// The transcript was not pasted.
        failure: Option<AppError>,
        /// The transcript was delivered with a problem worth mentioning.
        warning: Option<AppError>,
    },
    /// Setup's test dictation heard this text.
    Heard(String),
    /// Setup's test dictation heard no speech, or no audio at all.
    HeardNothing,
    Failed(AppError),
    Cancelled,
}

pub(super) enum DecodeFailure {
    Cancelled,
    Failed,
}

impl Worker {
    /// Publishes where the dictation ended and, when an update waited for it, starts it.
    pub(super) fn finish(&mut self, trigger: RecordingTrigger, outcome: Outcome) {
        let setup_test = trigger.is_setup_test();
        self.parts.state.apply(|state| {
            let next = match outcome {
                Outcome::Cancelled => {
                    if setup_test {
                        state.setup_test = Some(SetupTestResult::Cancelled);
                    }
                    Phase::Idle
                }
                Outcome::HeardNothing => {
                    state.setup_test = Some(SetupTestResult::NoSpeech);
                    Phase::Idle
                }
                Outcome::Failed(error) if setup_test => {
                    state.setup_test = Some(SetupTestResult::Failed {
                        error: Box::new(error),
                    });
                    Phase::Idle
                }
                Outcome::Failed(error) => {
                    state.last_error = Some(error);
                    Phase::Error
                }
                Outcome::Heard(text) => {
                    state.setup_test = Some(SetupTestResult::Heard { text });
                    Phase::Idle
                }
                Outcome::Delivered {
                    transcript,
                    failure,
                    warning,
                } => {
                    state.last_transcript = (!transcript.trim().is_empty()).then_some(transcript);
                    let failed = failure.is_some();
                    state.last_error = failure.or(warning);
                    if failed { Phase::Error } else { Phase::Idle }
                }
            };
            state.end_dictation(next);
        });
        if let Some(reply) = self.deferred_update.take() {
            self.begin_update(reply);
        }
    }

    pub(super) fn cancelled(&self, recording: &Recording, samples: &[f32]) -> Outcome {
        let reason = self.parts.state.lock().cancel;
        let (status, error) = match reason {
            Some(CancelReason::User) => (EntryStatus::Cancelled, None),
            Some(CancelReason::Pause) => (
                EntryStatus::Failed,
                Some("WhisperNow was paused before this dictation was transcribed"),
            ),
            Some(CancelReason::Shutdown) => (
                EntryStatus::Failed,
                Some("WhisperNow closed before this dictation was transcribed"),
            ),
            None => return Outcome::Cancelled,
        };
        self.keep_untranscribed(recording, samples, status, error.map(str::to_owned));
        Outcome::Cancelled
    }

    pub(super) fn failed_with_audio(
        &self,
        recording: &Recording,
        samples: &[f32],
        error: AppError,
    ) -> Outcome {
        self.keep_untranscribed(
            recording,
            samples,
            EntryStatus::Failed,
            Some(error.summary.clone()),
        );
        Outcome::Failed(error)
    }

    pub(super) fn pipeline_failure(
        &self,
        recording: &Recording,
        samples: &[f32],
        kind: PipelineFailureKind,
    ) -> Outcome {
        let error = pipeline_error(kind);
        match kind {
            PipelineFailureKind::NoAudio | PipelineFailureKind::NoSpeech => {
                if recording.trigger.is_setup_test() {
                    Outcome::HeardNothing
                } else {
                    Outcome::Failed(error)
                }
            }
            PipelineFailureKind::RecognitionFailed
            | PipelineFailureKind::EmptyTranscription
            | PipelineFailureKind::UnknownTokens => {
                self.failed_with_audio(recording, samples, error)
            }
        }
    }

    /// Records a dictation that has no transcript, when it heard anything.
    pub(super) fn keep_untranscribed(
        &self,
        recording: &Recording,
        samples: &[f32],
        status: EntryStatus,
        error: Option<String>,
    ) {
        if recording.trigger.is_setup_test() || !has_microphone_audio(samples) {
            return;
        }
        if let Err(error) = self.save_entry(recording, samples, status, error, None) {
            tracing::warn!(
                code = error.code,
                "untranscribed recording could not be saved"
            );
        }
    }

    /// Publishes `HistoryChanged` whether or not the save succeeded.
    pub(super) fn save_entry(
        &self,
        recording: &Recording,
        samples: &[f32],
        status: EntryStatus,
        error: Option<String>,
        transcript: Option<&PreparedTranscription>,
    ) -> Result<EntryId, AppError> {
        let saved = self.parts.history.insert(
            NewHistoryEntry {
                created_at: recording.started_at,
                duration_millis: Some(capture_duration_millis(samples)),
                raw_text: transcript.map(|prepared| prepared.raw_text.clone()),
                cleaned_text: transcript
                    .filter(|prepared| prepared.final_text != prepared.raw_text)
                    .map(|prepared| prepared.final_text.clone()),
                status,
                error,
                asr_model_id: Some(recording.settings.model_id.clone()),
            },
            samples,
            recording.settings.keep_audio,
            recording.settings.history_limit,
            recording.settings.history_retention,
        );
        self.parts.state.events().history_changed();
        saved
    }
}

fn capture_duration_millis(samples: &[f32]) -> u64 {
    u64::try_from(samples.len())
        .unwrap_or(u64::MAX)
        .saturating_mul(1_000)
        / u64::from(TARGET_SAMPLE_RATE)
}

impl OutputNotice {
    pub(super) fn failure(self) -> (Option<&'static str>, Option<AppError>) {
        let (reason, error) = match self {
            Self::PermissionDenied => (
                "The text was not pasted because WhisperNow lacks permission to type",
                Some(AppError::new(
                    ErrorCategory::Settings,
                    "text_output_permission_denied",
                    "Automatic text output needs desktop accessibility or portal permission. Enable it in system settings, then retry.",
                )),
            ),
            Self::CapabilityUnavailable => (
                "The text was not pasted because automatic typing is unavailable",
                Some(AppError::new(
                    ErrorCategory::Output,
                    "wayland_portal_unavailable",
                    "Automatic text output is unavailable because the required desktop portal capability could not be started.",
                )),
            ),
            Self::PasteFailed => (
                "The text could not be pasted",
                Some(AppError::new(
                    ErrorCategory::Output,
                    "paste_failed",
                    "The text was not pasted; copy it instead",
                )),
            ),
            Self::Cancelled => ("Pasting was cancelled", None),
            Self::Pasted => return (None, None),
        };
        (Some(reason), error)
    }
}

pub(super) fn cleanup_fallback_warning(
    prepared: &PreparedTranscription,
    target: Option<&CleanupTarget>,
) -> Option<AppError> {
    let (code, summary) = match (prepared.fallback?, target) {
        (CleanupFallback::Cancelled, _) => return None,
        (
            CleanupFallback::ProviderFailed(LlmError::ModelNotFound),
            Some(CleanupTarget::Remote(target)),
        ) => (
            "cleanup_model_not_found",
            format!(
                "AI cleanup couldn't find the model \"{}\"; check its name in Settings → AI cleanup",
                target.model
            ),
        ),
        _ => (
            "cleanup_failed",
            "AI cleanup was unavailable, so the transcript was kept as recognized".to_owned(),
        ),
    };
    Some(AppError::new(ErrorCategory::Cleanup, code, summary))
}

pub(super) fn interruption_warning(interruption: CaptureInterruption) -> Option<AppError> {
    match interruption {
        CaptureInterruption::MaximumDuration => None,
        CaptureInterruption::DeviceLost | CaptureInterruption::StreamError => Some(AppError::new(
            ErrorCategory::Audio,
            "audio_capture_interrupted",
            "The microphone stopped during the recording; the audio captured before it stopped was transcribed",
        )),
    }
}

pub(super) fn pipeline_error(kind: PipelineFailureKind) -> AppError {
    match kind {
        PipelineFailureKind::NoAudio => AppError::new(
            ErrorCategory::Audio,
            "no_microphone_audio",
            "No audio was detected; check the microphone and try again.",
        ),
        PipelineFailureKind::NoSpeech => AppError::new(
            ErrorCategory::Recognition,
            "no_speech",
            "No speech was detected; check the microphone and try again.",
        ),
        PipelineFailureKind::RecognitionFailed
        | PipelineFailureKind::EmptyTranscription
        | PipelineFailureKind::UnknownTokens => AppError::new(
            ErrorCategory::Recognition,
            "transcription_failed",
            "Speech recognition failed",
        ),
    }
}
