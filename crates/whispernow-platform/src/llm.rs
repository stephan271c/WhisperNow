//! Rig-backed LLM provider adapter.
//!
//! Provider request and response bodies are intentionally never logged.

use std::time::Duration;

use futures_util::StreamExt as _;
use rig_core::{
    client::CompletionClient as _,
    completion::{CompletionError, CompletionModel, CompletionRequest},
    providers::{
        anthropic,
        gemini::{self, completion::gemini_api_types::FinishReason as GeminiFinishReason},
        openai,
    },
    streaming::StreamedAssistantContent,
};
use whispernow_core::llm::{
    LlmError, LlmFinishReason, LlmPrompt, LlmRemoteConfig, LlmResponse, ProviderEndpoint,
};
use whispernow_domain::{ApiKey, RemoteProvider};

const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Anthropic requires `max_tokens`. Rig fills in the output maximum of the Claude models it
/// knows; any other model gets this, the smallest maximum among current Claude models.
const ANTHROPIC_FALLBACK_MAX_TOKENS: u64 = 64_000;

const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
const OLLAMA_BASE_URL: &str = "http://localhost:11434/v1";

pub async fn complete(config: LlmRemoteConfig, prompt: LlmPrompt) -> Result<LlmResponse, LlmError> {
    match ProviderModel::connect(&config)? {
        ProviderModel::Anthropic(model) => {
            stream_model(model, &prompt, |_| LlmFinishReason::Unknown).await
        }
        ProviderModel::Gemini(model) => {
            stream_model(model, &prompt, |response| {
                gemini_finish_reason(response.and_then(|response| response.finish_reason.as_ref()))
            })
            .await
        }
        ProviderModel::OpenAiCompatible(model) => {
            stream_model(model, &prompt, |_| LlmFinishReason::Unknown).await
        }
    }
}

/// A configured completion model. Every provider except `Anthropic` and `Gemini` speaks the
/// `OpenAI` chat-completions protocol, so they share one client.
enum ProviderModel {
    Anthropic(anthropic::completion::CompletionModel),
    Gemini(gemini::completion::CompletionModel),
    OpenAiCompatible(openai::completion::CompletionModel),
}

impl ProviderModel {
    fn connect(config: &LlmRemoteConfig) -> Result<Self, LlmError> {
        let http = reqwest::Client::builder()
            .connect_timeout(HTTP_CONNECT_TIMEOUT)
            .build()
            .map_err(|_| LlmError::InvalidRequest)?;
        let key = config.api_key.as_ref();
        let model_id = config.target.model.clone();
        match &config.target.provider {
            RemoteProvider::Anthropic(endpoint) => {
                let mut builder = anthropic::Client::builder()
                    .api_key(required_key(key)?)
                    .http_client(http);
                if let Some(endpoint) = endpoint {
                    builder = builder.base_url(endpoint.expose_url());
                }
                let client = builder.build().map_err(|_| LlmError::InvalidRequest)?;
                let mut model = client.completion_model(model_id);
                model
                    .default_max_tokens
                    .get_or_insert(ANTHROPIC_FALLBACK_MAX_TOKENS);
                Ok(Self::Anthropic(model))
            }
            RemoteProvider::Gemini(endpoint) => {
                let mut builder = gemini::Client::builder()
                    .api_key(required_key(key)?)
                    .http_client(http);
                if let Some(endpoint) = endpoint {
                    builder = builder.base_url(endpoint.expose_url());
                }
                let client = builder.build().map_err(|_| LlmError::InvalidRequest)?;
                Ok(Self::Gemini(client.completion_model(model_id)))
            }
            RemoteProvider::OpenAi(endpoint) => Self::openai_compatible(
                http,
                required_key(key)?,
                model_id,
                openai_compatible_base_url(endpoint.as_ref(), OPENAI_BASE_URL),
            ),
            RemoteProvider::OpenRouter(endpoint) => Self::openai_compatible(
                http,
                required_key(key)?,
                model_id,
                openai_compatible_base_url(endpoint.as_ref(), OPENROUTER_BASE_URL),
            ),
            RemoteProvider::Ollama(endpoint) => {
                // Ollama serves the OpenAI protocol under `/v1`; a bare host means its default path.
                let base_url = match endpoint {
                    Some(endpoint) if endpoint.url().path() == "/" => {
                        format!("{}/v1", endpoint.expose_url().trim_end_matches('/'))
                    }
                    _ => openai_compatible_base_url(endpoint.as_ref(), OLLAMA_BASE_URL),
                };
                Self::openai_compatible(
                    http,
                    key.map(|key| key.expose_secret().to_owned())
                        .unwrap_or_default(),
                    model_id,
                    base_url,
                )
            }
            RemoteProvider::Custom(endpoint) => Self::openai_compatible(
                http,
                key.map(|key| key.expose_secret().to_owned())
                    .unwrap_or_default(),
                model_id,
                endpoint.expose_url().to_owned(),
            ),
        }
    }

    fn openai_compatible(
        http: reqwest::Client,
        key: String,
        model: String,
        base_url: String,
    ) -> Result<Self, LlmError> {
        let client = openai::CompletionsClient::builder()
            .api_key(key)
            .base_url(base_url)
            .http_client(http)
            .build()
            .map_err(|_| LlmError::InvalidRequest)?;
        Ok(Self::OpenAiCompatible(client.completion_model(model)))
    }
}

fn openai_compatible_base_url(endpoint: Option<&ProviderEndpoint>, default: &str) -> String {
    endpoint
        .map_or(default, ProviderEndpoint::expose_url)
        .to_owned()
}

fn required_key(key: Option<&ApiKey>) -> Result<String, LlmError> {
    key.map(|key| key.expose_secret().to_owned())
        .ok_or(LlmError::Authentication)
}

fn gemini_finish_reason(reason: Option<&GeminiFinishReason>) -> LlmFinishReason {
    match reason {
        Some(GeminiFinishReason::Stop) => LlmFinishReason::Completed,
        Some(GeminiFinishReason::MaxTokens) => LlmFinishReason::OutputLimit,
        Some(
            GeminiFinishReason::Safety
            | GeminiFinishReason::Recitation
            | GeminiFinishReason::Blocklist
            | GeminiFinishReason::ProhibitedContent
            | GeminiFinishReason::Spii,
        ) => LlmFinishReason::ContentFilter,
        _ => LlmFinishReason::Unknown,
    }
}

/// API providers get no output
/// limit or temperature of ours, so reasoning models keep their full budget. An empty
/// system prompt sends no system message at all, where rig would otherwise send an empty
/// one.
fn completion_request<M: CompletionModel>(model: &M, prompt: &LlmPrompt) -> CompletionRequest {
    let mut builder = model.completion_request(prompt.user());
    if !prompt.system().is_empty() {
        builder = builder.preamble(prompt.system().to_owned());
    }
    builder.build()
}

/// Streams one completion. Only Gemini reports why a stream stopped in rig's final
/// response; for the other providers the reason is unknown.
async fn stream_model<M, R>(
    model: M,
    prompt: &LlmPrompt,
    reported_finish: R,
) -> Result<LlmResponse, LlmError>
where
    M: CompletionModel,
    R: FnOnce(Option<&M::StreamingResponse>) -> LlmFinishReason,
{
    let provider_request = completion_request(&model, prompt);
    let mut stream = model
        .stream(provider_request)
        .await
        .map_err(|error| map_error(&error))?;
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        if let StreamedAssistantContent::Text(chunk) = item.map_err(|error| map_error(&error))? {
            text.push_str(&chunk.text);
        }
    }
    Ok(LlmResponse {
        text,
        finish_reason: reported_finish(stream.response.as_ref()),
    })
}

fn map_error(error: &CompletionError) -> LlmError {
    if let Some(status) = error.provider_response_status() {
        return match status.as_u16() {
            401 | 403 => LlmError::Authentication,
            408 | 429 => LlmError::RateLimited,
            500..=599 => LlmError::TemporarilyUnavailable,
            400..=499 => LlmError::InvalidRequest,
            _ => LlmError::InvalidResponse,
        };
    }
    match error {
        CompletionError::HttpError(_) => LlmError::Transport,
        CompletionError::UrlError(_) | CompletionError::RequestError(_) => LlmError::InvalidRequest,
        _ => LlmError::InvalidResponse,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ANTHROPIC_FALLBACK_MAX_TOKENS, GeminiFinishReason, OLLAMA_BASE_URL, OPENROUTER_BASE_URL,
        ProviderModel, completion_request, gemini_finish_reason, openai_compatible_base_url,
        required_key,
    };
    use whispernow_core::llm::{
        LlmError, LlmFinishReason, LlmPrompt, LlmRemoteConfig, build_cleanup_prompt,
    };
    use whispernow_domain::{ApiKey, ProviderEndpoint, RemoteCleanupTarget, RemoteProvider};

    fn request(provider: RemoteProvider, model: &str) -> (LlmRemoteConfig, LlmPrompt) {
        (
            LlmRemoteConfig {
                target: RemoteCleanupTarget {
                    provider,
                    model: model.to_owned(),
                },
                api_key: ApiKey::new("token").ok(),
            },
            build_cleanup_prompt("  um so can you send me the report by friday \n"),
        )
    }

    #[test]
    fn remote_providers_require_a_key() {
        assert_eq!(required_key(None), Err(LlmError::Authentication));
        let key = ApiKey::new("token").expect("a key");
        assert_eq!(required_key(Some(&key)), Ok("token".to_owned()));
    }

    #[test]
    fn gemini_reports_why_it_stopped() {
        for (reason, expected) in [
            (GeminiFinishReason::MaxTokens, LlmFinishReason::OutputLimit),
            (GeminiFinishReason::Safety, LlmFinishReason::ContentFilter),
        ] {
            assert_eq!(gemini_finish_reason(Some(&reason)), expected);
        }
    }

    #[test]
    fn openai_compatible_providers_use_the_override_or_their_default() {
        assert_eq!(
            openai_compatible_base_url(None, OPENROUTER_BASE_URL),
            OPENROUTER_BASE_URL
        );
        assert_eq!(
            openai_compatible_base_url(None, OLLAMA_BASE_URL),
            OLLAMA_BASE_URL
        );
        let endpoint = ProviderEndpoint::parse("http://127.0.0.1:11434/v1").expect("endpoint");
        assert_eq!(
            openai_compatible_base_url(Some(&endpoint), OPENROUTER_BASE_URL),
            endpoint.expose_url()
        );
        let (config, _) = request(RemoteProvider::Custom(endpoint), "custom");
        assert!(matches!(
            ProviderModel::connect(&config),
            Ok(ProviderModel::OpenAiCompatible(_))
        ));
    }

    #[test]
    fn an_api_request_sends_no_temperature_or_output_limit() {
        let (config, prompt) = request(RemoteProvider::Ollama(None), "local");
        let Ok(ProviderModel::OpenAiCompatible(model)) = ProviderModel::connect(&config) else {
            panic!("Ollama speaks the OpenAI protocol");
        };
        let body = rig_core::providers::openai::completion::CompletionRequest::try_from((
            "local".to_owned(),
            completion_request(&model, &prompt),
        ))
        .expect("an OpenAI request");
        let body = serde_json::to_value(body).expect("serializable");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["role"], "user");
        assert!(
            body.get("temperature")
                .is_none_or(serde_json::Value::is_null)
        );
        assert!(
            body.get("max_tokens")
                .is_none_or(serde_json::Value::is_null)
        );
        assert!(
            body.get("max_completion_tokens")
                .is_none_or(serde_json::Value::is_null)
        );
    }

    #[test]
    fn an_anthropic_request_carries_the_model_maximum() {
        for (model_id, maximum) in [
            ("claude-sonnet-4-5", 64_000),
            ("claude-opus-4-7", 128_000),
            ("claude-opus-5-5", ANTHROPIC_FALLBACK_MAX_TOKENS),
        ] {
            let (config, prompt) = request(RemoteProvider::Anthropic(None), model_id);
            let Ok(ProviderModel::Anthropic(model)) = ProviderModel::connect(&config) else {
                panic!("an Anthropic model");
            };
            assert_eq!(model.default_max_tokens, Some(maximum), "{model_id}");
            assert_eq!(completion_request(&model, &prompt).max_tokens, None);
        }
    }
}
