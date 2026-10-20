use std::{sync::Arc, time::Instant};

use whispernow_core::{
    CancellationToken,
    llm::{CleanupFallback, LlmService, clean_up},
    workflow::PreparedTranscription,
};
use whispernow_domain::CleanupTarget;
use whispernow_platform::SettingsStore;

#[derive(Clone)]
pub(crate) struct CleanupPlan {
    pub(crate) target: CleanupTarget,
    /// Whether the credential store holds the provider's API key, so it is asked only then.
    pub(crate) key_saved: bool,
}

pub(crate) struct AiCleanup {
    pub(crate) service: Arc<dyn LlmService>,
    pub(crate) runtime: Arc<tokio::runtime::Runtime>,
    /// Read when a call is prepared, so settings snapshots and recording start never wait on
    /// the OS credential store.
    pub(crate) api_keys: Arc<SettingsStore>,
}

impl AiCleanup {
    pub(super) fn apply(
        &self,
        prepared: &mut PreparedTranscription,
        plan: &CleanupPlan,
        cancellation: &CancellationToken,
    ) {
        let started = Instant::now();
        let result = self.run(plan, &prepared.final_text, cancellation);
        if let Err(fallback) = &result
            && *fallback != CleanupFallback::Cancelled
        {
            tracing::warn!(?fallback, "AI cleanup kept the recognized text");
        }
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
        prepared.apply_cleanup(result);
    }

    fn run(
        &self,
        plan: &CleanupPlan,
        deterministic_text: &str,
        cancellation: &CancellationToken,
    ) -> Result<String, CleanupFallback> {
        let api_key = if let CleanupTarget::Remote(target) = &plan.target
            && plan.key_saved
        {
            // An unreadable key leaves the call without one; the provider then rejects it
            // and the transcript keeps its deterministic text.
            self.api_keys
                .api_key(target.provider.provider())
                .inspect_err(|error| {
                    tracing::warn!(
                        code = error.code,
                        "the LLM provider API key could not be read"
                    );
                })
                .ok()
                .flatten()
        } else {
            None
        };
        self.runtime.block_on(clean_up(
            self.service.as_ref(),
            deterministic_text,
            &plan.target,
            api_key,
            cancellation,
        ))
    }
}
