//! Captured text and credentials are redacted from `Debug` output.

use std::{fmt, future::Future, pin::Pin};

use thiserror::Error;
use whispernow_domain::{ApiKey, CleanupTarget, ModelId, RemoteCleanupTarget};

use crate::CancellationToken;

mod align;
mod draft;
pub mod local;
mod prompt;

use prompt::EMPTY_TRANSCRIPT_SENTINEL;
pub use prompt::{LlmPrompt, build_cleanup_prompt, build_local_cleanup_prompt};

const TRANSIENT_RETRIES: u32 = 2;

/// `Debug` never exposes the key, model, or endpoint.
#[derive(Clone, Eq, PartialEq)]
pub struct LlmRemoteConfig {
    pub target: RemoteCleanupTarget,
    pub api_key: Option<ApiKey>,
}

impl fmt::Debug for LlmRemoteConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LlmRemoteConfig")
            .field("target", &self.target)
            .field("credential_present", &self.api_key.is_some())
            .finish_non_exhaustive()
    }
}

/// The pre-cleaned transcript the local cleanup model drafts from (see `draft`).
#[derive(Clone, Eq, PartialEq)]
pub struct DraftText(String);

impl DraftText {
    #[must_use]
    pub fn from_transcript(transcript: &str) -> Self {
        Self(draft::preclean(transcript))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for DraftText {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DraftText")
            .field("character_count", &self.0.chars().count())
            .finish()
    }
}

#[derive(Clone)]
pub enum LlmRequest {
    Local {
        model: ModelId,
        prompt: LlmPrompt,
        max_output_tokens: usize,
        draft: DraftText,
    },
    Remote {
        config: LlmRemoteConfig,
        prompt: LlmPrompt,
    },
}

impl fmt::Debug for LlmRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Local {
                prompt,
                max_output_tokens,
                draft,
                ..
            } => formatter
                .debug_struct("LlmRequest::Local")
                .field("prompt", prompt)
                .field("max_output_tokens", max_output_tokens)
                .field("draft", draft)
                .finish_non_exhaustive(),
            Self::Remote { config, prompt } => formatter
                .debug_struct("LlmRequest::Remote")
                .field("config", config)
                .field("prompt", prompt)
                .finish(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LlmFinishReason {
    #[default]
    Unknown,
    Completed,
    OutputLimit,
    ContentFilter,
}

#[derive(Clone, PartialEq)]
pub struct LlmResponse {
    pub text: String,
    pub finish_reason: LlmFinishReason,
}

impl fmt::Debug for LlmResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LlmResponse")
            .field("text_character_count", &self.text.chars().count())
            .field("finish_reason", &self.finish_reason)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum LlmError {
    #[error("LLM request was cancelled")]
    Cancelled,
    #[error("LLM provider transport failed")]
    Transport,
    #[error("LLM provider rate limit was reached")]
    RateLimited,
    #[error("LLM provider is temporarily unavailable")]
    TemporarilyUnavailable,
    #[error("LLM provider authentication failed")]
    Authentication,
    #[error("LLM provider has no model with that name")]
    ModelNotFound,
    #[error("LLM provider rejected the request")]
    InvalidRequest,
    #[error("LLM provider returned an invalid response")]
    InvalidResponse,
}

impl LlmError {
    #[must_use]
    pub const fn is_retryable(self) -> bool {
        matches!(
            self,
            Self::Transport | Self::RateLimited | Self::TemporarilyUnavailable
        )
    }
}

pub type LlmFuture<'service> =
    Pin<Box<dyn Future<Output = Result<LlmResponse, LlmError>> + Send + 'service>>;

/// One provider call. [`clean_up`] cancels it by dropping the future.
pub trait LlmService: Send + Sync {
    fn complete(&self, request: LlmRequest) -> LlmFuture<'_>;
}

fn strip_think_blocks(text: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let mut rest = text;
    // Some providers stream the reasoning without its opening tag.
    if let Some(close) = rest.find(CLOSE)
        && rest.find(OPEN).is_none_or(|open| open > close)
    {
        rest = &rest[close + CLOSE.len()..];
    }
    let mut kept = String::with_capacity(rest.len());
    while let Some(open) = rest.find(OPEN) {
        kept.push_str(&rest[..open]);
        match rest[open..].find(CLOSE) {
            Some(close) => rest = &rest[open + close + CLOSE.len()..],
            // An unterminated block is all reasoning.
            None => return kept,
        }
    }
    kept.push_str(rest);
    kept
}

/// Accepts a provider response unless it failed mechanically: cut off at the output limit,
/// stopped by a content filter, or empty once formatting is removed.
fn accept_response(response: &LlmResponse) -> Result<String, CleanupFallback> {
    match response.finish_reason {
        LlmFinishReason::OutputLimit => return Err(CleanupFallback::Truncated),
        LlmFinishReason::ContentFilter => return Err(CleanupFallback::ContentFiltered),
        LlmFinishReason::Unknown | LlmFinishReason::Completed => {}
    }
    let text = strip_think_blocks(&response.text).trim().to_owned();
    // The sentinel means the model found no dictated words; the deterministic
    // transcript is kept rather than pasting the sentinel or nothing.
    if text.is_empty() || text == EMPTY_TRANSCRIPT_SENTINEL {
        Err(CleanupFallback::EmptyOutput)
    } else {
        Ok(text)
    }
}

/// Why cleanup kept the deterministic text. Every reason is a mechanical failure;
/// the content of a completed response is never judged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CleanupFallback {
    Cancelled,
    ProviderFailed(LlmError),
    /// The response was empty, or the model reported no dictated words.
    EmptyOutput,
    Truncated,
    ContentFiltered,
}

/// Asks `service` to edit the transcript and returns the reply. It waits as long as the
/// provider takes; cancellation drops the call in flight.
pub async fn clean_up(
    service: &dyn LlmService,
    transcript: &str,
    target: &CleanupTarget,
    api_key: Option<ApiKey>,
    cancellation: &CancellationToken,
) -> Result<String, CleanupFallback> {
    let request = match target {
        CleanupTarget::Local { model } => {
            let word_count = transcript.split_whitespace().count();
            LlmRequest::Local {
                model: model.clone(),
                prompt: build_local_cleanup_prompt(transcript),
                // Evaluation uses 2.5 tokens per input word, with a minimum of 64.
                max_output_tokens: word_count.saturating_mul(5).div_ceil(2).max(64),
                draft: DraftText::from_transcript(transcript),
            }
        }
        CleanupTarget::Remote(target) => LlmRequest::Remote {
            config: LlmRemoteConfig {
                target: target.clone(),
                api_key,
            },
            prompt: build_cleanup_prompt(transcript),
        },
    };
    let result = tokio::select! {
        biased;
        () = cancellation.cancelled() => Err(LlmError::Cancelled),
        result = call_with_retries(service, &request) => result,
    };
    match result {
        Ok(response) => accept_response(&response),
        Err(LlmError::Cancelled) => Err(CleanupFallback::Cancelled),
        Err(error) => Err(CleanupFallback::ProviderFailed(error)),
    }
}

/// Retries a transient failure. Text streamed before the failure is discarded, so a
/// retry is safe.
async fn call_with_retries(
    service: &dyn LlmService,
    request: &LlmRequest,
) -> Result<LlmResponse, LlmError> {
    let mut retries = 0;
    loop {
        match service.complete(request.clone()).await {
            Err(error) if error.is_retryable() && retries < TRANSIENT_RETRIES => retries += 1,
            result => return result,
        }
    }
}

#[cfg(test)]
mod tests;
