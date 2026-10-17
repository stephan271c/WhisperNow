//! Rig-backed LLM provider adapter.
//!
//! Provider request and response bodies are intentionally never logged.

use std::time::Duration;

use futures_util::StreamExt as _;
use rig_core::{
    client::CompletionClient as _,
    completion::{CompletionError, CompletionModel},
    providers::{
        anthropic,
        gemini::{self, completion::gemini_api_types::FinishReason as GeminiFinishReason},
        openai,
    },
    streaming::StreamedAssistantContent,
};
use whispernow_core::llm::{LlmError, LlmFinishReason, LlmPrompt, LlmRemoteConfig, LlmResponse};
use whispernow_domain::{ApiKey, ProviderEndpoint, RemoteProvider};

mod model_list;

pub use model_list::list_models;

const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Anthropic requires `max_tokens`. Rig fills in the output maximum of the Claude models it
/// knows; any other model gets this, the smallest maximum among current Claude models.
const ANTHROPIC_FALLBACK_MAX_TOKENS: u64 = 64_000;

const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
const OLLAMA_BASE_URL: &str = "http://localhost:11434/v1";

pub async fn complete(config: LlmRemoteConfig, prompt: LlmPrompt) -> Result<LlmResponse, LlmError> {
    let http = reqwest::Client::builder()
        .connect_timeout(HTTP_CONNECT_TIMEOUT)
        .build()
        .map_err(|_| LlmError::InvalidRequest)?;
    let key = config.api_key.as_ref();
    let model_id = config.target.model;
    let openai_model = |key: String, base_url: String| {
        let client = openai::CompletionsClient::builder()
            .api_key(key)
            .base_url(base_url)
            .http_client(http.clone())
            .build()
            .map_err(|_| LlmError::InvalidRequest)?;
        Ok::<_, LlmError>(client.completion_model(model_id.clone()))
    };
    match &config.target.provider {
        RemoteProvider::Anthropic(endpoint) => {
            let mut builder = anthropic::Client::builder()
                .api_key(required_key(key)?)
                .http_client(http.clone());
            if let Some(endpoint) = endpoint {
                builder = builder.base_url(endpoint.expose_url());
            }
            let client = builder.build().map_err(|_| LlmError::InvalidRequest)?;
            let mut model = client.completion_model(model_id);
            model
                .default_max_tokens
                .get_or_insert(ANTHROPIC_FALLBACK_MAX_TOKENS);
            stream_model(model, &prompt, |_| LlmFinishReason::Unknown).await
        }
        RemoteProvider::Gemini(endpoint) => {
            let mut builder = gemini::Client::builder()
                .api_key(required_key(key)?)
                .http_client(http.clone());
            if let Some(endpoint) = endpoint {
                builder = builder.base_url(endpoint.expose_url());
            }
            let client = builder.build().map_err(|_| LlmError::InvalidRequest)?;
            stream_model(client.completion_model(model_id), &prompt, |response| {
                gemini_finish_reason(response.and_then(|response| response.finish_reason.as_ref()))
            })
            .await
        }
        RemoteProvider::OpenAi(endpoint) => {
            let model = openai_model(
                required_key(key)?,
                openai_compatible_base_url(endpoint.as_ref(), OPENAI_BASE_URL),
            )?;
            stream_model(model, &prompt, |_| LlmFinishReason::Unknown).await
        }
        RemoteProvider::OpenRouter(endpoint) => {
            let model = openai_model(
                required_key(key)?,
                openai_compatible_base_url(endpoint.as_ref(), OPENROUTER_BASE_URL),
            )?;
            stream_model(model, &prompt, |_| LlmFinishReason::Unknown).await
        }
        RemoteProvider::Ollama(endpoint) => {
            // Ollama serves the OpenAI protocol under `/v1`; a bare host means its default path.
            let base_url = match endpoint {
                Some(endpoint) if endpoint.url().path() == "/" => {
                    format!("{}/v1", endpoint.expose_url().trim_end_matches('/'))
                }
                _ => openai_compatible_base_url(endpoint.as_ref(), OLLAMA_BASE_URL),
            };
            let model = openai_model(
                key.map(|key| key.expose_secret().to_owned())
                    .unwrap_or_default(),
                base_url,
            )?;
            stream_model(model, &prompt, |_| LlmFinishReason::Unknown).await
        }
        RemoteProvider::Custom(endpoint) => {
            let model = openai_model(
                key.map(|key| key.expose_secret().to_owned())
                    .unwrap_or_default(),
                endpoint.expose_url().to_owned(),
            )?;
            stream_model(model, &prompt, |_| LlmFinishReason::Unknown).await
        }
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
    // API providers get no output limit or temperature of ours, so reasoning models keep
    // their full budget. An empty system prompt sends no system message at all.
    let mut builder = model.completion_request(prompt.user());
    if !prompt.system().is_empty() {
        builder = builder.preamble(prompt.system().to_owned());
    }
    let provider_request = builder.build();
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
        if names_missing_model(error) {
            return LlmError::ModelNotFound;
        }
        return match status.as_u16() {
            401 | 403 => LlmError::Authentication,
            404 => LlmError::ModelNotFound,
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

// OpenAI, and servers that copy its error shape, name a missing model in `error.code`; some
// answer 400 rather than 404.
fn names_missing_model(error: &CompletionError) -> bool {
    error
        .provider_response_json()
        .ok()
        .flatten()
        .is_some_and(|body| body["error"]["code"] == "model_not_found")
}

#[cfg(test)]
mod tests {
    use super::{
        ANTHROPIC_FALLBACK_MAX_TOKENS, GeminiFinishReason, OPENROUTER_BASE_URL, complete,
        gemini_finish_reason, map_error, openai_compatible_base_url, required_key,
    };
    use reqwest::StatusCode;
    use rig_core::completion::CompletionError;
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
    fn a_missing_model_is_told_apart_from_other_rejections() {
        for (status, body, expected) in [
            (
                StatusCode::NOT_FOUND,
                r#"{"type":"error","error":{"type":"not_found_error","message":"model: claude-x"}}"#,
                LlmError::ModelNotFound,
            ),
            (
                StatusCode::BAD_REQUEST,
                r#"{"error":{"message":"The model `gpt-x` does not exist","code":"model_not_found"}}"#,
                LlmError::ModelNotFound,
            ),
            (
                StatusCode::BAD_REQUEST,
                r#"{"error":{"message":"Invalid value","code":"invalid_value"}}"#,
                LlmError::InvalidRequest,
            ),
            (
                StatusCode::BAD_REQUEST,
                "not json",
                LlmError::InvalidRequest,
            ),
        ] {
            let error = CompletionError::from_http_response(status, body);
            assert_eq!(map_error(&error), expected, "{status} {body}");
        }
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
        let endpoint = ProviderEndpoint::parse("http://127.0.0.1:11434/v1").expect("endpoint");
        assert_eq!(
            openai_compatible_base_url(Some(&endpoint), OPENROUTER_BASE_URL),
            endpoint.expose_url()
        );
    }

    async fn sent_request(
        provider: impl FnOnce(ProviderEndpoint) -> RemoteProvider,
        model: &str,
    ) -> serde_json::Value {
        use std::{
            io::{Read, Write},
            net::TcpListener,
            thread,
            time::{Duration, Instant},
        };

        let listener = TcpListener::bind("127.0.0.1:0").expect("fixture listener");
        listener
            .set_nonblocking(true)
            .expect("nonblocking fixture listener");
        let endpoint = ProviderEndpoint::parse(format!(
            "http://{}/v1",
            listener.local_addr().expect("fixture address")
        ))
        .expect("fixture endpoint");
        let server = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            let (mut connection, _) = loop {
                match listener.accept() {
                    Ok(connection) => break connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(
                            Instant::now() < deadline,
                            "provider did not connect before the fixture request deadline"
                        );
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(error) => panic!("accept provider request: {error}"),
                }
            };
            connection
                .set_nonblocking(false)
                .expect("blocking fixture connection");
            connection
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("request timeout");
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let read = connection.read(&mut buffer).expect("request bytes");
                assert!(read > 0, "request ended before its body");
                request.extend_from_slice(&buffer[..read]);
                if let Some(end) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                    let header = std::str::from_utf8(&request[..end]).expect("request headers");
                    let length = header
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().expect("body length"))
                        })
                        .expect("request body length");
                    if request.len() >= end + 4 + length {
                        let body = serde_json::from_slice(&request[end + 4..end + 4 + length])
                            .expect("request JSON");
                        connection
                            .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
                            .expect("fixture response");
                        return body;
                    }
                }
            }
        });
        let (config, prompt) = request(provider(endpoint), model);
        let result = complete(config, prompt).await;
        let body = server.join().expect("fixture provider request");
        assert!(result.is_err());
        body
    }

    #[tokio::test]
    async fn an_api_request_sends_no_temperature_or_output_limit() {
        let body = sent_request(|endpoint| RemoteProvider::Ollama(Some(endpoint)), "local").await;
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

    #[tokio::test]
    async fn an_unknown_anthropic_model_uses_our_token_limit() {
        let body = sent_request(
            |endpoint| RemoteProvider::Anthropic(Some(endpoint)),
            "claude-unknown-model",
        )
        .await;
        assert_eq!(body["max_tokens"], ANTHROPIC_FALLBACK_MAX_TOKENS);
    }
}
