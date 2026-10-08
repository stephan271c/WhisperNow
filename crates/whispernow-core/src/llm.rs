//! Captured text and credentials are redacted from `Debug` output.

use std::{fmt, future::Future, pin::Pin};

use thiserror::Error;
use whispernow_domain::ApiKey;
pub use whispernow_domain::{LlmProvider, ProviderEndpoint};

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
pub struct LlmProviderConfig {
    pub provider: LlmProvider,
    pub model: String,
    pub api_key: Option<ApiKey>,
    pub endpoint: Option<ProviderEndpoint>,
}

impl fmt::Debug for LlmProviderConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LlmProviderConfig")
            .field("provider", &self.provider)
            .field("credential_present", &self.api_key.is_some())
            .field("custom_endpoint", &self.endpoint.is_some())
            .finish_non_exhaustive()
    }
}

/// The output budget and draft sent with one request. API providers get neither and decode
/// with their own defaults.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LlmDecoding {
    /// Output tokens after which generation stops and the reply counts as cut off; `None`
    /// leaves the provider's own limit.
    pub max_output_tokens: Option<u64>,
    pub draft: Option<DraftText>,
}

/// The pre-cleaned transcript the local cleanup model drafts from (see `draft`).
#[derive(Clone, Eq, PartialEq)]
pub struct DraftText(String);

impl DraftText {
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

impl LlmDecoding {
    /// The local cleanup model's budget, as its evaluation ran it: 2.5 tokens per input
    /// word (at least 64), so a loop is cut off early and falls back to the deterministic
    /// text. The transcript, pre-cleaned by code, goes along as the draft for speculative
    /// decoding. Its repeat penalty is fixed (`local::REPEAT_PENALTY`).
    #[must_use]
    pub fn local_cleanup(deterministic_text: &str) -> Self {
        let words =
            u64::try_from(deterministic_text.split_whitespace().count()).unwrap_or(u64::MAX);
        Self {
            max_output_tokens: Some(words.saturating_mul(5).div_ceil(2).max(64)),
            draft: Some(DraftText(draft::preclean(deterministic_text))),
        }
    }
}

#[derive(Clone)]
pub struct LlmRequest {
    pub config: LlmProviderConfig,
    pub prompt: LlmPrompt,
    pub decoding: LlmDecoding,
}

impl fmt::Debug for LlmRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LlmRequest")
            .field("config", &self.config)
            .field("prompt", &self.prompt)
            .field("decoding", &self.decoding)
            .finish()
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

/// Removes `<think>` blocks, the reasoning some providers inline with the answer, and the
/// surrounding whitespace. The answer itself is used as returned.
#[must_use]
pub(crate) fn sanitize_output(text: &str) -> String {
    strip_think_blocks(text).trim().to_owned()
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
    let text = sanitize_output(&response.text);
    // The sentinel means the model found no dictated words; the deterministic
    // transcript is kept rather than pasting the sentinel or nothing.
    if text.is_empty() || text == EMPTY_TRANSCRIPT_SENTINEL {
        Err(CleanupFallback::EmptyOutput)
    } else {
        Ok(text)
    }
}

#[derive(Clone, Copy)]
pub struct CleanupRequest<'input> {
    pub deterministic_text: &'input str,
    pub config: &'input LlmProviderConfig,
}

impl fmt::Debug for CleanupRequest<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CleanupRequest")
            .field(
                "deterministic_character_count",
                &self.deterministic_text.chars().count(),
            )
            .field("config", &self.config)
            .finish()
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
    input: CleanupRequest<'_>,
    cancellation: &CancellationToken,
) -> Result<String, CleanupFallback> {
    let (prompt, decoding) = if input.config.provider == LlmProvider::LocalCleanup {
        (
            build_local_cleanup_prompt(input.deterministic_text),
            LlmDecoding::local_cleanup(input.deterministic_text),
        )
    } else {
        (
            build_cleanup_prompt(input.deterministic_text),
            LlmDecoding::default(),
        )
    };
    let request = LlmRequest {
        config: input.config.clone(),
        prompt,
        decoding,
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
