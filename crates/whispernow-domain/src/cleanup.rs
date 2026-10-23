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
    /// `provider` at `endpoint`, or at its default address when there is none. `None` for the
    /// app's own cleanup model and for a custom provider without an endpoint.
    #[must_use]
    pub fn new(provider: LlmProvider, endpoint: Option<ProviderEndpoint>) -> Option<Self> {
        match provider {
            LlmProvider::OpenAi => Some(Self::OpenAi(endpoint)),
            LlmProvider::Anthropic => Some(Self::Anthropic(endpoint)),
            LlmProvider::OpenRouter => Some(Self::OpenRouter(endpoint)),
            LlmProvider::Ollama => Some(Self::Ollama(endpoint)),
            LlmProvider::Gemini => Some(Self::Gemini(endpoint)),
            LlmProvider::Custom => endpoint.map(Self::Custom),
            LlmProvider::LocalCleanup => None,
        }
    }

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
    pub fn active_cleanup_target(&self) -> Option<CleanupTarget> {
        if self.cleanup_enabled {
            self.cleanup_target().ok()
        } else {
            None
        }
    }

    pub fn cleanup_target(&self) -> Result<CleanupTarget, MissingCleanupField> {
        if self.llm_provider == LlmProvider::LocalCleanup {
            return Ok(CleanupTarget::Local {
                model: self.cleanup_model_id(),
            });
        }
        let settings = self.provider_settings(self.llm_provider);
        let model = &settings.model;
        if model.is_empty() {
            return Err(MissingCleanupField::Model);
        }
        let provider = RemoteProvider::new(self.llm_provider, settings.api_base.clone())
            .ok_or(MissingCleanupField::Endpoint)?;
        Ok(CleanupTarget::Remote(RemoteCleanupTarget {
            provider,
            model: model.clone(),
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
