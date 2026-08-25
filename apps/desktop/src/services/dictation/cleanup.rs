//! AI cleanup of a recognized transcript, and the saved-prompt test that runs the same call.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use whispernow_api::{ApiErrorDto, ErrorCategoryDto};
use whispernow_core::{
    CancellationToken,
    llm::{
        CleanupFallback, CleanupRequest, LlmError, LlmProviderConfig, LlmService, PromptKind,
        clean_up,
    },
    workflow::{PreparedTranscription, apply_vocabulary_replacements},
};
use whispernow_domain::{CleanupPrompt, VocabularyReplacement};

use whispernow_platform::SettingsStore;

#[derive(Clone)]
pub(crate) struct CleanupPlan {
    pub(crate) prompt: CleanupPrompt,
    pub(crate) config: Option<LlmProviderConfig>,
    /// Whether the credential store holds the provider's API key, so it is asked only then.
    pub(crate) key_saved: bool,
    pub(crate) mode: PromptKind,
    pub(crate) vocabulary: Vec<VocabularyReplacement>,
}

/// The LLM provider service, the runtime its calls run on, and the saved API keys.
pub(crate) struct AiCleanup {
    pub(crate) service: Arc<dyn LlmService>,
    pub(crate) runtime: Arc<tokio::runtime::Runtime>,
    pub(crate) api_keys: Arc<SettingsStore>,
}

impl AiCleanup {
    /// Cleans up `prepared` with `plan`'s provider, and returns the model whose text was
    /// used. Settings validation gives enabled cleanup a provider.
    pub(super) fn apply(
        &self,
        prepared: &mut PreparedTranscription,
        plan: Option<&CleanupPlan>,
        timeout: Duration,
        cancellation: &CancellationToken,
    ) -> Option<String> {
        let plan = plan?;
        let config = plan.config.as_ref()?;
        let started = Instant::now();
        let result = self.run(
            plan,
            config,
            &prepared.deterministic_text,
            timeout,
            cancellation,
        );
        tracing::info!(
            stage = "cleanup",
            duration_ms = started.elapsed().as_secs_f64() * 1_000.0,
            outcome = if result.is_ok() {
                "completed"
            } else {
                "fallback"
            },
            "batch pipeline stage completed"
        );
        let llm_model = result.is_ok().then(|| config.model.clone());
        prepared.apply_cleanup(result);
        llm_model
    }

    /// Runs `plan` once on `sample` with the saved provider, the way a dictation's cleanup
    /// would, and returns the provider's text. A run that keeps the deterministic text is
    /// an error, so the caller never presents unchanged text as cleaned. Application
    /// shutdown cancels the provider call.
    pub(crate) fn run_prompt(
        &self,
        plan: &CleanupPlan,
        sample: &str,
        timeout: Duration,
        shutdown: &CancellationToken,
    ) -> Result<String, ApiErrorDto> {
        let config = plan.config.as_ref().ok_or_else(|| {
            ApiErrorDto::new(
                ErrorCategoryDto::General,
                "cleanup_provider_unconfigured",
                "Choose an AI cleanup provider and model first",
            )
        })?;
        let deterministic = apply_vocabulary_replacements(sample, &plan.vocabulary);
        self.run(plan, config, &deterministic, timeout, shutdown)
            .map_err(|fallback| match fallback {
                CleanupFallback::Cancelled => ApiErrorDto::new(
                    ErrorCategoryDto::Recognition,
                    "operation_cancelled",
                    "The AI cleanup run was cancelled",
                ),
                CleanupFallback::ProviderFailed(LlmError::Timeout) => ApiErrorDto::new(
                    ErrorCategoryDto::Recognition,
                    "cleanup_timed_out",
                    "The provider did not answer before the timeout",
                ),
                _ => ApiErrorDto::new(
                    ErrorCategoryDto::Recognition,
                    "cleanup_failed",
                    "The provider request failed; check the model, endpoint and API key",
                ),
            })
    }

    fn run(
        &self,
        plan: &CleanupPlan,
        config: &LlmProviderConfig,
        deterministic_text: &str,
        timeout: Duration,
        cancellation: &CancellationToken,
    ) -> Result<String, CleanupFallback> {
        let mut config = config.clone();
        if plan.key_saved {
            // An unreadable key leaves the call without one; the provider then rejects it
            // and the transcript keeps its deterministic text.
            config.api_key = self
                .api_keys
                .api_key(config.provider)
                .inspect_err(|error| {
                    tracing::warn!(
                        code = error.code,
                        "the LLM provider API key could not be read"
                    );
                })
                .ok()
                .flatten();
        }
        self.runtime.block_on(clean_up(
            self.service.as_ref(),
            CleanupRequest {
                deterministic_text,
                prompt: &plan.prompt,
                mode: plan.mode,
                config: &config,
            },
            timeout,
            cancellation,
        ))
    }
}
