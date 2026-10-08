use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};
use thiserror::Error;
use url::{Host, Url};

/// Saved by its [`LlmProvider::as_str`] name.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LlmProvider {
    OpenAi,
    Anthropic,
    OpenRouter,
    Ollama,
    /// The small cleanup model trained for this app, run in process by llama.cpp once it is
    /// installed from the model list. It does only Clean Up & Format, and gets a short prompt
    /// made for it.
    LocalCleanup,
    Gemini,
    /// A user-supplied OpenAI-compatible endpoint.
    Custom,
}

impl LlmProvider {
    /// Every provider, in the order the settings list them.
    pub const ALL: [Self; 7] = [
        Self::OpenAi,
        Self::Anthropic,
        Self::OpenRouter,
        Self::Ollama,
        Self::LocalCleanup,
        Self::Gemini,
        Self::Custom,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
            Self::OpenRouter => "openrouter",
            Self::Ollama => "ollama",
            Self::LocalCleanup => "local_cleanup",
            Self::Gemini => "gemini",
            Self::Custom => "other",
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::OpenAi => "OpenAI",
            Self::Anthropic => "Anthropic",
            Self::OpenRouter => "OpenRouter",
            Self::Ollama => "Ollama",
            Self::LocalCleanup => "Local cleanup model",
            Self::Gemini => "Google Gemini",
            Self::Custom => "Custom provider",
        }
    }

    #[must_use]
    pub fn from_name(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "openai" => Some(Self::OpenAi),
            "anthropic" => Some(Self::Anthropic),
            "gemini" | "google" => Some(Self::Gemini),
            "ollama" => Some(Self::Ollama),
            "local_cleanup" => Some(Self::LocalCleanup),
            "openrouter" => Some(Self::OpenRouter),
            "other" => Some(Self::Custom),
            _ => None,
        }
    }

    #[must_use]
    pub fn request_model(self, model: &str) -> &str {
        let trimmed = model.trim();
        let prefix = match self {
            Self::OpenAi => "openai/",
            Self::Anthropic => "anthropic/",
            Self::Gemini => "gemini/",
            Self::Ollama => "ollama/",
            Self::OpenRouter => "openrouter/",
            Self::LocalCleanup | Self::Custom => return trimmed,
        };
        trimmed.strip_prefix(prefix).unwrap_or(trimmed)
    }
}

impl Serialize for LlmProvider {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for LlmProvider {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Self::from_name(&name).ok_or_else(|| D::Error::custom("unknown LLM provider"))
    }
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum EndpointError {
    #[error("Enter a full address without a user name, query or fragment")]
    Invalid,
    #[error("Use an HTTPS address, or HTTP for a service on this computer")]
    InsecureRemote,
}

/// A custom provider base URL: HTTPS, or HTTP to this computer, with no credentials, query or
/// fragment.
#[derive(Clone, Eq, PartialEq)]
pub struct ProviderEndpoint {
    url: String,
}

impl ProviderEndpoint {
    pub fn parse(value: impl Into<String>) -> Result<Self, EndpointError> {
        let value = value.into();
        // The URL parser strips surrounding whitespace and newlines; the saved text keeps them.
        if value
            .chars()
            .any(|character| character.is_whitespace() || character.is_control())
        {
            return Err(EndpointError::Invalid);
        }
        let parsed = Url::parse(&value).map_err(|_| EndpointError::Invalid)?;
        if !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.query().is_some()
            || parsed.fragment().is_some()
        {
            return Err(EndpointError::Invalid);
        }
        require_secure_transport(&parsed)?;
        Ok(Self { url: value })
    }

    #[must_use]
    pub fn expose_url(&self) -> &str {
        &self.url
    }

    /// The scheme, host and port, which say where a transcript goes without its path.
    #[must_use]
    pub fn origin(&self) -> String {
        Url::parse(&self.url)
            .map_or_else(|_| String::new(), |url| url.origin().ascii_serialization())
    }
}

fn require_secure_transport(url: &Url) -> Result<(), EndpointError> {
    let loopback = match url.host() {
        Some(Host::Domain(host)) => host.eq_ignore_ascii_case("localhost"),
        Some(Host::Ipv4(address)) => address.is_loopback(),
        Some(Host::Ipv6(address)) => address.is_loopback(),
        None => return Err(EndpointError::Invalid),
    };
    match url.scheme() {
        "https" => Ok(()),
        "http" if loopback => Ok(()),
        "http" => Err(EndpointError::InsecureRemote),
        _ => Err(EndpointError::Invalid),
    }
}

impl Serialize for ProviderEndpoint {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.url)
    }
}

impl<'de> Deserialize<'de> for ProviderEndpoint {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::parse(String::deserialize(deserializer)?).map_err(D::Error::custom)
    }
}

impl fmt::Debug for ProviderEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderEndpoint")
            .finish_non_exhaustive()
    }
}
