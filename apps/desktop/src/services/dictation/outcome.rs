//! How a dictation ends: what it publishes, and what history keeps of one that has no
//! transcript.

use whispernow_api::{
    ApiErrorDto, ErrorCategoryDto, RecoveryActionDto, SetupTestResultDto, TextOutputStatusDto,
    TranscriptResponse,
};
use whispernow_core::{
    llm::{CleanupFallback, LlmError},
    workflow::{
        BATCH_SAMPLE_RATE_HZ, PipelineFailureKind, PreparedTranscription, has_microphone_audio,
    },
};
use whispernow_domain::{EntryStatus, NewHistoryEntry};

use super::{Recording, RecordingTrigger, Worker, capture::CaptureInterruption};
use crate::services::{
    output::OutputNotice,
    state::{CancelReason, Phase},
};

/// How one dictation ended.
pub(super) enum Outcome {
    Delivered {
        transcript: TranscriptResponse,
        /// The transcript was not pasted.
        failure: Option<ApiErrorDto>,
        /// The transcript was delivered with a problem worth mentioning.
        warning: Option<ApiErrorDto>,
    },
    /// Setup's test dictation heard this text.
    Heard(String),
    /// Setup's test dictation heard no speech, or no audio at all.
    HeardNothing,
    Failed(ApiErrorDto),
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
            let reason = state.cancel.take();
            let next = match outcome {
                Outcome::Cancelled => {
                    if setup_test {
                        state.setup_test = Some(SetupTestResultDto::Cancelled);
                    }
                    Phase::Idle
                }
                Outcome::HeardNothing => {
                    state.setup_test = Some(SetupTestResultDto::NoSpeech);
                    Phase::Idle
                }
                Outcome::Failed(error) if setup_test => {
                    state.setup_test = Some(SetupTestResultDto::Failed {
                        error: Box::new(error),
                    });
                    Phase::Idle
                }
                Outcome::Failed(error) => {
                    state.last_error = Some(error.clone());
                    Phase::Error(error)
                }
                Outcome::Heard(text) => {
                    state.setup_test = Some(SetupTestResultDto::Heard { text });
                    Phase::Idle
                }
                Outcome::Delivered {
                    transcript,
                    failure,
                    warning,
                } => {
                    state.last_transcript = Some(transcript);
                    state.last_error = failure.clone().or(warning);
                    failure.map_or(Phase::Idle, Phase::Error)
                }
            };
            state.phase = if reason == Some(CancelReason::Pause) {
                Phase::Paused
            } else {
                next
            };
        });
        if let Some(reply) = self.deferred_update.take() {
            self.begin_update(reply);
        }
    }

    /// Why the dictation was cancelled decides what history keeps of it.
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
        error: ApiErrorDto,
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
        let Some(error) = pipeline_error(kind) else {
            return self.cancelled(recording, samples);
        };
        if kept_failure(kind) {
            return self.failed_with_audio(recording, samples, error);
        }
        let heard_nothing = matches!(
            kind,
            PipelineFailureKind::NoAudio | PipelineFailureKind::NoSpeech
        );
        if heard_nothing && recording.trigger.is_setup_test() {
            Outcome::HeardNothing
        } else {
            Outcome::Failed(error)
        }
    }

    /// Keeps a dictation that has no transcript, with its audio, when it heard anything.
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
        let saved = self.parts.history.insert(
            NewHistoryEntry {
                created_at: recording.started_at,
                duration_millis: Some(capture_duration_millis(samples)),
                raw_text: None,
                cleaned_text: None,
                status,
                error,
                asr_model_id: Some(recording.settings.model_id.clone()),
                llm_model: None,
            },
            samples,
            recording.settings.keep_audio,
            recording.settings.history_limit,
        );
        if let Err(error) = saved {
            tracing::warn!(
                code = error.code,
                "untranscribed recording could not be saved"
            );
        }
        self.history_changed();
        if let Some(warning) = self.parts.history.take_warning() {
            self.parts.state.publish_error(warning);
        }
    }
}

pub(super) fn capture_duration_millis(samples: &[f32]) -> u64 {
    u64::try_from(samples.len())
        .unwrap_or(u64::MAX)
        .saturating_mul(1_000)
        / u64::from(BATCH_SAMPLE_RATE_HZ)
}

pub(super) const fn delivery_failure(notice: OutputNotice) -> Option<&'static str> {
    match notice {
        OutputNotice::PermissionDenied => {
            Some("The text was not pasted because WhisperNow lacks permission to type")
        }
        OutputNotice::CapabilityUnavailable => {
            Some("The text was not pasted because automatic typing is unavailable")
        }
        OutputNotice::PasteFailed => Some("The text could not be pasted"),
        OutputNotice::Cancelled => Some("Pasting was cancelled"),
        OutputNotice::PastedAndRestored
        | OutputNotice::PastedClipboardNotRestored
        | OutputNotice::RestoreSkippedClipboardChanged => None,
    }
}

pub(super) const fn text_output_status(notice: OutputNotice) -> TextOutputStatusDto {
    match notice {
        OutputNotice::PastedAndRestored => TextOutputStatusDto::PastedAndRestored,
        OutputNotice::PastedClipboardNotRestored => TextOutputStatusDto::PastedClipboardNotRestored,
        OutputNotice::RestoreSkippedClipboardChanged => {
            TextOutputStatusDto::RestoreSkippedClipboardChanged
        }
        OutputNotice::PermissionDenied
        | OutputNotice::CapabilityUnavailable
        | OutputNotice::PasteFailed => TextOutputStatusDto::PasteFailed,
        OutputNotice::Cancelled => TextOutputStatusDto::Cancelled,
    }
}

/// Why the transcript was not pasted, when it was not and the user did not cancel it.
pub(super) fn delivery_error(notice: OutputNotice) -> Option<ApiErrorDto> {
    match notice {
        OutputNotice::PermissionDenied => Some(ApiErrorDto::new(
            ErrorCategoryDto::Permission,
            "text_output_permission_denied",
            "Automatic text output needs desktop accessibility or portal permission. Enable it in system settings, then retry.",
        )),
        OutputNotice::CapabilityUnavailable => Some(ApiErrorDto::new(
            ErrorCategoryDto::Output,
            "wayland_portal_unavailable",
            "Automatic text output is unavailable because the required desktop portal capability could not be started.",
        )),
        OutputNotice::PasteFailed => Some(ApiErrorDto::new(
            ErrorCategoryDto::Output,
            "paste_failed",
            "The text was not pasted; copy it instead",
        )),
        OutputNotice::PastedAndRestored
        | OutputNotice::PastedClipboardNotRestored
        | OutputNotice::RestoreSkippedClipboardChanged
        | OutputNotice::Cancelled => None,
    }
}

pub(super) fn cleanup_fallback_warning(prepared: &PreparedTranscription) -> Option<ApiErrorDto> {
    let (code, summary) = match prepared.fallback? {
        CleanupFallback::Cancelled => return None,
        CleanupFallback::ProviderFailed(LlmError::Timeout) => (
            "cleanup_timed_out",
            "AI cleanup took too long, so the transcript was kept as recognized",
        ),
        CleanupFallback::ProviderFailed(_)
        | CleanupFallback::EmptyOutput
        | CleanupFallback::Truncated
        | CleanupFallback::ContentFiltered => (
            "cleanup_failed",
            "AI cleanup was unavailable, so the transcript was kept as recognized",
        ),
    };
    Some(ApiErrorDto {
        recovery: Some(RecoveryActionDto::OpenSettings),
        ..ApiErrorDto::new(ErrorCategoryDto::Recognition, code, summary)
    })
}

pub(super) fn interruption_warning(interruption: CaptureInterruption) -> Option<ApiErrorDto> {
    match interruption {
        CaptureInterruption::MaximumDuration => None,
        CaptureInterruption::DeviceLost | CaptureInterruption::StreamError => {
            Some(ApiErrorDto::new(
                ErrorCategoryDto::Audio,
                "audio_capture_interrupted",
                "The microphone stopped during the recording; the audio captured before it stopped was transcribed",
            ))
        }
    }
}

/// The public error for a pipeline failure. A cancelled pipeline ends without one.
pub(super) fn pipeline_error(kind: PipelineFailureKind) -> Option<ApiErrorDto> {
    Some(match kind {
        PipelineFailureKind::Cancelled => return None,
        PipelineFailureKind::NoAudio => ApiErrorDto {
            recovery: Some(RecoveryActionDto::Retry),
            ..ApiErrorDto::new(
                ErrorCategoryDto::Audio,
                "no_microphone_audio",
                "No microphone audio was detected; check the selected input and try again.",
            )
        },
        PipelineFailureKind::NoSpeech => ApiErrorDto::new(
            ErrorCategoryDto::Recognition,
            "no_speech",
            "No speech was detected; check the microphone and try again.",
        ),
        PipelineFailureKind::RecognitionFailed
        | PipelineFailureKind::EmptyTranscription
        | PipelineFailureKind::UnknownTokens => ApiErrorDto::new(
            ErrorCategoryDto::Recognition,
            "transcription_failed",
            "Speech recognition failed",
        ),
    })
}

/// Whether a pipeline failure is kept in history as a failed dictation.
const fn kept_failure(kind: PipelineFailureKind) -> bool {
    match kind {
        PipelineFailureKind::RecognitionFailed
        | PipelineFailureKind::EmptyTranscription
        | PipelineFailureKind::UnknownTokens => true,
        PipelineFailureKind::NoAudio
        | PipelineFailureKind::NoSpeech
        | PipelineFailureKind::Cancelled => false,
    }
}
