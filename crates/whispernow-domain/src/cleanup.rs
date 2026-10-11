use std::fmt;

use thiserror::Error;

use crate::{LlmProvider, ModelId, ProviderEndpoint, Settings};

#[derive(Clone, Eq, PartialEq)]
pub enum CleanupTarget {
    Local { model: ModelId },
    Remote(RemoteCleanupTarget),
}

/// `Debug` never exposes the model or endpoint.
#[derive(Clone, Eq, PartialEq)]
pub struct RemoteCleanupTarget {
    pub provider: RemoteProvider,
    pub model: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RemoteProvider {
    OpenAi(Option<ProviderEndpoint>),
    Anthropic(Option<ProviderEndpoint>),
    OpenRouter(Option<ProviderEndpoint>),
    Ollama(Option<ProviderEndpoint>),
    Gemini(Option<ProviderEndpoint>),
    Custom(ProviderEndpoint),
}

impl RemoteProvider {
    #[must_use]
    pub const fn provider(&self) -> LlmProvider {
        match self {
            Self::OpenAi(_) => LlmProvider::OpenAi,
            Self::Anthropic(_) => LlmProvider::Anthropic,
            Self::OpenRouter(_) => LlmProvider::OpenRouter,
            Self::Ollama(_) => LlmProvider::Ollama,
            Self::Gemini(_) => LlmProvider::Gemini,
            Self::Custom(_) => LlmProvider::Custom,
        }
    }
}

impl fmt::Debug for RemoteCleanupTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RemoteCleanupTarget")
            .field("provider", &self.provider)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for CleanupTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Local { .. } => formatter
                .debug_struct("CleanupTarget::Local")
                .finish_non_exhaustive(),
            Self::Remote(target) => target.fmt(formatter),
        }
    }
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum MissingCleanupField {
    #[error("a model")]
    Model,
    #[error("an API endpoint")]
    Endpoint,
}

impl Settings {
    #[must_use]
    pub fn uses_local_cleanup(&self) -> bool {
        self.cleanup_enabled && self.llm_provider == LlmProvider::LocalCleanup
    }

    pub fn cleanup_target(&self) -> Result<CleanupTarget, MissingCleanupField> {
        if self.llm_provider == LlmProvider::LocalCleanup {
            return Ok(CleanupTarget::Local {
                model: self.cleanup_model_id(),
            });
        }
        let settings = self.provider_settings(self.llm_provider);
        let model = self.llm_provider.request_model(&settings.model);
        if model.is_empty() {
            return Err(MissingCleanupField::Model);
        }
        let endpoint = settings.api_base.clone();
        let provider = match self.llm_provider {
            LlmProvider::OpenAi => RemoteProvider::OpenAi(endpoint),
            LlmProvider::Anthropic => RemoteProvider::Anthropic(endpoint),
            LlmProvider::OpenRouter => RemoteProvider::OpenRouter(endpoint),
            LlmProvider::Ollama => RemoteProvider::Ollama(endpoint),
            LlmProvider::Gemini => RemoteProvider::Gemini(endpoint),
            LlmProvider::Custom => {
                RemoteProvider::Custom(endpoint.ok_or(MissingCleanupField::Endpoint)?)
            }
            LlmProvider::LocalCleanup => unreachable!(),
        };
        Ok(CleanupTarget::Remote(RemoteCleanupTarget {
            provider,
            model: model.to_owned(),
        }))
    }

    #[must_use]
    pub fn cleanup_warning(&self) -> Option<String> {
        if !self.cleanup_enabled {
            return None;
        }
        self.cleanup_target().err().map(|missing| {
            format!(
                "AI cleanup needs {missing} for {}; the plain transcript will be pasted until you enter one.",
                self.llm_provider.label()
            )
        })
    }
}
