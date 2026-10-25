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
    /// The cleanup models trained for this app, run in process by llama.cpp once one is
    /// installed from the model list. They get the transcript alone, with no instructions.
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

/// A provider's API base URL: HTTPS, or HTTP to this computer, with no credentials, query or
/// fragment.
#[derive(Clone, Eq, PartialEq)]
pub struct ProviderEndpoint {
    text: String,
    url: Url,
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
        Ok(Self {
            text: value,
            url: parsed,
        })
    }

    #[must_use]
    pub fn expose_url(&self) -> &str {
        &self.text
    }

    /// The address without trailing slashes, ready to have a path appended.
    #[must_use]
    pub fn base(&self) -> &str {
        self.text.trim_end_matches('/')
    }

    #[must_use]
    pub fn url(&self) -> &Url {
        &self.url
    }

    /// The scheme, host and port, which say where a transcript goes without its path.
    #[must_use]
    pub fn origin(&self) -> String {
        self.url.origin().ascii_serialization()
    }

    #[must_use]
    pub fn is_on_this_computer(&self) -> bool {
        is_loopback(&self.url.host().expect("validated endpoint has a host"))
    }
}

fn is_loopback(host: &Host<&str>) -> bool {
    match host {
        Host::Domain(host) => host.eq_ignore_ascii_case("localhost"),
        Host::Ipv4(address) => address.is_loopback(),
        Host::Ipv6(address) => address.is_loopback(),
    }
}

fn require_secure_transport(url: &Url) -> Result<(), EndpointError> {
    let loopback = is_loopback(&url.host().ok_or(EndpointError::Invalid)?);
    match url.scheme() {
        "https" => Ok(()),
        "http" if loopback => Ok(()),
        "http" => Err(EndpointError::InsecureRemote),
        _ => Err(EndpointError::Invalid),
    }
}

impl Serialize for ProviderEndpoint {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.text)
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
