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
    LlmDecoding, LlmError, LlmFinishReason, LlmProvider, LlmProviderConfig, LlmRequest,
    LlmResponse, ProviderEndpoint,
};
use whispernow_domain::ApiKey;

const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Anthropic requires `max_tokens`. Rig fills in the output maximum of the Claude models it
/// knows; any other model gets this, the smallest maximum among current Claude models.
const ANTHROPIC_FALLBACK_MAX_TOKENS: u64 = 64_000;

const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
const OLLAMA_BASE_URL: &str = "http://localhost:11434/v1";

pub async fn complete(request: LlmRequest) -> Result<LlmResponse, LlmError> {
    match ProviderModel::connect(&request.config)? {
        ProviderModel::Anthropic(model) => {
            stream_model(model, &request, |_| LlmFinishReason::Unknown).await
        }
        ProviderModel::Gemini(model) => {
            stream_model(model, &request, |response| {
                gemini_finish_reason(response.and_then(|response| response.finish_reason.as_ref()))
            })
            .await
        }
        ProviderModel::OpenAiCompatible(model) => {
            stream_model(model, &request, |_| LlmFinishReason::Unknown).await
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
    fn connect(config: &LlmProviderConfig) -> Result<Self, LlmError> {
        let http = reqwest::Client::builder()
            .connect_timeout(HTTP_CONNECT_TIMEOUT)
            .build()
            .map_err(|_| LlmError::InvalidRequest)?;
        let endpoint = config.endpoint.as_ref();
        let key = config.api_key.as_ref();
        let model_id = config.model.clone();
        match config.provider {
            LlmProvider::Anthropic => {
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
            LlmProvider::Gemini => {
                let mut builder = gemini::Client::builder()
                    .api_key(required_key(key)?)
                    .http_client(http);
                if let Some(endpoint) = endpoint {
                    builder = builder.base_url(endpoint.expose_url());
                }
                let client = builder.build().map_err(|_| LlmError::InvalidRequest)?;
                Ok(Self::Gemini(client.completion_model(model_id)))
            }
            LlmProvider::OpenAi
            | LlmProvider::OpenRouter
            | LlmProvider::Ollama
            | LlmProvider::Custom => {
                let key = match config.provider {
                    LlmProvider::OpenAi | LlmProvider::OpenRouter => required_key(key)?,
                    _ => key
                        .map(|key| key.expose_secret().to_owned())
                        .unwrap_or_default(),
                };
                let client = openai::CompletionsClient::builder()
                    .api_key(key)
                    .base_url(openai_compatible_base_url(config.provider, endpoint)?)
                    .http_client(http)
                    .build()
                    .map_err(|_| LlmError::InvalidRequest)?;
                Ok(Self::OpenAiCompatible(client.completion_model(model_id)))
            }
            LlmProvider::LocalCleanup => Err(LlmError::InvalidRequest),
        }
    }
}

fn openai_compatible_base_url(
    provider: LlmProvider,
    endpoint: Option<&ProviderEndpoint>,
) -> Result<String, LlmError> {
    let Some(endpoint) = endpoint else {
        return match provider {
            LlmProvider::OpenAi => Ok(OPENAI_BASE_URL.to_owned()),
            LlmProvider::OpenRouter => Ok(OPENROUTER_BASE_URL.to_owned()),
            LlmProvider::Ollama => Ok(OLLAMA_BASE_URL.to_owned()),
            _ => Err(LlmError::InvalidRequest),
        };
    };
    // Ollama serves the OpenAI protocol under `/v1`; a bare host means its default path.
    if provider == LlmProvider::Ollama && endpoint.url().path() == "/" {
        return Ok(format!(
            "{}/v1",
            endpoint.expose_url().trim_end_matches('/')
        ));
    }
    Ok(endpoint.expose_url().to_owned())
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
fn completion_request<M: CompletionModel>(model: &M, request: &LlmRequest) -> CompletionRequest {
    let mut builder = model.completion_request(request.prompt.user());
    if let LlmDecoding::LocalCleanup {
        max_output_tokens, ..
    } = &request.decoding
    {
        builder = builder.max_tokens(*max_output_tokens);
    }
    if !request.prompt.system().is_empty() {
        builder = builder.preamble(request.prompt.system().to_owned());
    }
    builder.build()
}

/// Streams one completion. Only Gemini reports why a stream stopped in rig's final
/// response; for the other providers the reason is unknown.
async fn stream_model<M, R>(
    model: M,
    request: &LlmRequest,
    reported_finish: R,
) -> Result<LlmResponse, LlmError>
where
    M: CompletionModel,
    R: FnOnce(Option<&M::StreamingResponse>) -> LlmFinishReason,
{
    let provider_request = completion_request(&model, request);
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
        ProviderModel, complete, completion_request, gemini_finish_reason,
        openai_compatible_base_url, required_key,
    };
    use whispernow_core::llm::{
        LlmDecoding, LlmError, LlmFinishReason, LlmProvider, LlmProviderConfig, LlmRequest,
        build_cleanup_prompt,
    };
    use whispernow_domain::{ApiKey, ProviderEndpoint};

    fn request(provider: LlmProvider, model: &str) -> LlmRequest {
        LlmRequest {
            config: LlmProviderConfig {
                provider,
                model: model.to_owned(),
                api_key: ApiKey::new("token").ok(),
                endpoint: None,
            },
            prompt: build_cleanup_prompt("  um so can you send me the report by friday \n"),
            decoding: LlmDecoding::default(),
        }
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
    fn openai_compatible_providers_share_one_base_url_rule() {
        assert_eq!(
            openai_compatible_base_url(LlmProvider::OpenRouter, None),
            Ok(OPENROUTER_BASE_URL.to_owned())
        );
        assert_eq!(
            openai_compatible_base_url(LlmProvider::Ollama, None),
            Ok(OLLAMA_BASE_URL.to_owned())
        );
        assert_eq!(
            openai_compatible_base_url(
                LlmProvider::Ollama,
                Some(
                    &ProviderEndpoint::parse("http://127.0.0.1:11434/").expect("loopback endpoint")
                )
            ),
            Ok("http://127.0.0.1:11434/v1".to_owned())
        );
        assert_eq!(
            openai_compatible_base_url(
                LlmProvider::Ollama,
                Some(
                    &ProviderEndpoint::parse("http://127.0.0.1:11434/v1")
                        .expect("loopback endpoint")
                )
            ),
            Ok("http://127.0.0.1:11434/v1".to_owned())
        );
        assert_eq!(
            openai_compatible_base_url(LlmProvider::Custom, None),
            Err(LlmError::InvalidRequest)
        );
    }

    #[test]
    fn an_api_request_sends_no_temperature_or_output_limit() {
        let request = request(LlmProvider::Ollama, "local");
        let Ok(ProviderModel::OpenAiCompatible(model)) = ProviderModel::connect(&request.config)
        else {
            panic!("Ollama speaks the OpenAI protocol");
        };
        let body = rig_core::providers::openai::completion::CompletionRequest::try_from((
            "local".to_owned(),
            completion_request(&model, &request),
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
            let request = request(LlmProvider::Anthropic, model_id);
            let Ok(ProviderModel::Anthropic(model)) = ProviderModel::connect(&request.config)
            else {
                panic!("an Anthropic model");
            };
            assert_eq!(model.default_max_tokens, Some(maximum), "{model_id}");
            assert_eq!(completion_request(&model, &request).max_tokens, None);
        }
    }

    #[tokio::test]
    async fn the_local_cleanup_model_is_rejected_by_the_remote_call() {
        assert!(matches!(
            complete(request(LlmProvider::LocalCleanup, "cleanup")).await,
            Err(LlmError::InvalidRequest)
        ));
    }
}
