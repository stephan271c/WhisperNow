use std::{sync::Arc, time::Instant};

use whispernow_core::{
    CancellationToken,
    llm::{CleanupFallback, LlmProviderConfig, LlmService, clean_up},
    workflow::PreparedTranscription,
};
use whispernow_platform::SettingsStore;

#[derive(Clone)]
pub(crate) struct CleanupPlan {
    pub(crate) config: Option<LlmProviderConfig>,
    /// Whether the credential store holds the provider's API key, so it is asked only then.
    pub(crate) key_saved: bool,
}

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
        cancellation: &CancellationToken,
    ) -> Option<String> {
        let plan = plan?;
        let config = plan.config.as_ref()?;
        let started = Instant::now();
        let result = self.run(plan, config, &prepared.deterministic_text, cancellation);
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

    fn run(
        &self,
        plan: &CleanupPlan,
        config: &LlmProviderConfig,
        deterministic_text: &str,
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
            deterministic_text,
            &config,
            cancellation,
        ))
    }
}
