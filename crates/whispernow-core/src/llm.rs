//! Provider-independent AI cleanup: the provider interface, and the call that
//! puts one deadline, cancellation and retries around a provider. The prompts are in
//! `prompt`.
//!
//! Captured text and credentials are redacted from `Debug` output.

use std::{fmt, future::Future, pin::Pin, time::Duration};

use thiserror::Error;
use whispernow_domain::{ApiKey, CleanupPrompt};
pub use whispernow_domain::{LlmProvider, ProviderEndpoint};

use crate::CancellationToken;

mod align;
mod draft;
pub mod local;
mod prompt;

use prompt::EMPTY_TRANSCRIPT_SENTINEL;
pub use prompt::{LlmPrompt, PromptKind, build_cleanup_prompt, build_local_cleanup_prompt};

/// Retries after a transport, rate-limit, or availability failure, while the deadline has
/// time left.
const TRANSIENT_RETRIES: u32 = 2;

/// The provider and model to call, with its credentials. `Debug` never exposes the key,
/// model, or endpoint.
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
    /// What the local cleanup model's speculative decoding drafts from.
    pub draft: Option<DraftText>,
}

/// The pre-cleaned transcript the local cleanup model drafts from (see `draft`). It holds
/// captured text, so `Debug` shows only its length.
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

/// Request delivered to one concrete provider adapter.
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

/// Why the provider stopped generating, when the provider reports it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LlmFinishReason {
    /// The provider did not say why it stopped.
    #[default]
    Unknown,
    /// The model finished its answer.
    Completed,
    /// Generation hit the output token limit, so the text is cut off.
    OutputLimit,
    /// The provider's content filter stopped generation.
    ContentFilter,
}

/// Provider response. Captured output is redacted from `Debug`.
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

/// Content-free provider failures.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum LlmError {
    #[error("LLM request was cancelled")]
    Cancelled,
    #[error("LLM request timed out")]
    Timeout,
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
    /// Failures worth another attempt within the same deadline. A timeout is
    /// not one: the deadline covers every attempt, so nothing is left to retry.
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

/// One provider call. [`clean_up`] enforces the deadline and cancellation by dropping the
/// future.
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

/// Complete input for one cleanup.
#[derive(Clone, Copy)]
pub struct CleanupRequest<'input> {
    pub deterministic_text: &'input str,
    pub prompt: &'input CleanupPrompt,
    pub mode: PromptKind,
    pub config: &'input LlmProviderConfig,
}

impl fmt::Debug for CleanupRequest<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CleanupRequest")
            .field("mode", &self.mode)
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
    /// Generation stopped at the output limit, so the text is cut off.
    Truncated,
    /// The provider's content filter stopped generation.
    ContentFiltered,
}

/// Asks `service` to edit the transcript and returns the reply. One `timeout` covers every
/// attempt, and cancellation drops the call in flight.
///
/// # Errors
///
/// Returns why the deterministic text stays: cancellation, a provider failure, or a
/// mechanically failed reply.
pub async fn clean_up(
    service: &dyn LlmService,
    input: CleanupRequest<'_>,
    timeout: Duration,
    cancellation: &CancellationToken,
) -> Result<String, CleanupFallback> {
    // Settings validation keeps every prompt but Clean Up & Format off the local
    // cleanup model, which knows only that one edit.
    let (prompt, decoding) = if input.config.provider == LlmProvider::LocalCleanup {
        (
            build_local_cleanup_prompt(input.deterministic_text),
            LlmDecoding::local_cleanup(input.deterministic_text),
        )
    } else {
        (
            build_cleanup_prompt(input.deterministic_text, input.prompt, input.mode),
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
        result = tokio::time::timeout(timeout, call_with_retries(service, &request)) => {
            result.unwrap_or(Err(LlmError::Timeout))
        }
    };
    match result {
        Ok(response) => accept_response(&response),
        Err(LlmError::Cancelled) => Err(CleanupFallback::Cancelled),
        Err(error) => Err(CleanupFallback::ProviderFailed(error)),
    }
}

/// Retries a transient failure. Text streamed before the failure is discarded, so a
/// retry is safe; the caller's deadline bounds them all.
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
