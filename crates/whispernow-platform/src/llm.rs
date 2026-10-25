//! Rig-backed LLM provider adapter.
//!
//! Provider request and response bodies are intentionally never logged.

use std::{sync::LazyLock, time::Duration};

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
use whispernow_core::llm::{LlmError, LlmFinishReason, LlmPrompt, LlmResponse};
use whispernow_domain::{ApiKey, ProviderEndpoint, RemoteCleanupTarget, RemoteProvider};

mod model_list;
#[cfg(test)]
mod test_server;

pub use model_list::list_models;

const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Anthropic requires `max_tokens`. Rig fills in the output maximum of the Claude models it
/// knows; any other model gets this, the smallest maximum among current Claude models.
const ANTHROPIC_FALLBACK_MAX_TOKENS: u64 = 64_000;

const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
const OLLAMA_ROOT: &str = "http://localhost:11434";

pub async fn complete(
    target: RemoteCleanupTarget,
    api_key: Option<ApiKey>,
    prompt: LlmPrompt,
) -> Result<LlmResponse, LlmError> {
    let http = http_client()?;
    let key = api_key.as_ref();
    let model = target.model;
    match &target.provider {
        RemoteProvider::Anthropic(endpoint) => {
            anthropic(http, endpoint.as_ref(), required_key(key)?, model, &prompt).await
        }
        RemoteProvider::Gemini(endpoint) => {
            gemini(http, endpoint.as_ref(), required_key(key)?, model, &prompt).await
        }
        RemoteProvider::OpenAi(endpoint) => {
            let base = base_url(endpoint.as_ref(), OPENAI_BASE_URL);
            openai_compatible(http, required_key(key)?, base, model, &prompt).await
        }
        RemoteProvider::OpenRouter(endpoint) => {
            let base = base_url(endpoint.as_ref(), OPENROUTER_BASE_URL);
            openai_compatible(http, required_key(key)?, base, model, &prompt).await
        }
        RemoteProvider::Ollama(endpoint) => {
            let base = format!("{}/v1", ollama_root(endpoint.as_ref()));
            openai_compatible(http, optional_key(key), &base, model, &prompt).await
        }
        RemoteProvider::Custom(endpoint) => {
            openai_compatible(http, optional_key(key), endpoint.base(), model, &prompt).await
        }
    }
}

async fn anthropic(
    http: reqwest::Client,
    endpoint: Option<&ProviderEndpoint>,
    key: String,
    model: String,
    prompt: &LlmPrompt,
) -> Result<LlmResponse, LlmError> {
    let mut builder = anthropic::Client::builder().api_key(key).http_client(http);
    if let Some(endpoint) = endpoint {
        builder = builder.base_url(endpoint.base());
    }
    let client = builder.build().map_err(|_| LlmError::InvalidRequest)?;
    let mut model = client.completion_model(model);
    model
        .default_max_tokens
        .get_or_insert(ANTHROPIC_FALLBACK_MAX_TOKENS);
    stream_model(model, prompt, |_| LlmFinishReason::Unknown).await
}

async fn gemini(
    http: reqwest::Client,
    endpoint: Option<&ProviderEndpoint>,
    key: String,
    model: String,
    prompt: &LlmPrompt,
) -> Result<LlmResponse, LlmError> {
    let mut builder = gemini::Client::builder().api_key(key).http_client(http);
    if let Some(endpoint) = endpoint {
        builder = builder.base_url(endpoint.base());
    }
    let client = builder.build().map_err(|_| LlmError::InvalidRequest)?;
    stream_model(client.completion_model(model), prompt, |response| {
        gemini_finish_reason(response.and_then(|response| response.finish_reason.as_ref()))
    })
    .await
}

async fn openai_compatible(
    http: reqwest::Client,
    key: String,
    base: &str,
    model: String,
    prompt: &LlmPrompt,
) -> Result<LlmResponse, LlmError> {
    let client = openai::CompletionsClient::builder()
        .api_key(key)
        .base_url(base)
        .http_client(http)
        .build()
        .map_err(|_| LlmError::InvalidRequest)?;
    stream_model(client.completion_model(model), prompt, |_| {
        LlmFinishReason::Unknown
    })
    .await
}

/// The one client every provider call uses, so connections and TLS sessions are reused. It
/// follows no redirect: reqwest strips only `Authorization` and cookies when a redirect changes
/// host, so following one would resend `x-api-key`, `x-goog-api-key` and the request body to
/// whatever host the endpoint names. It has no total timeout, since a completion streams for as
/// long as the model generates; a call that needs one sets it on its request.
static HTTP_CLIENT: LazyLock<Result<reqwest::Client, reqwest::Error>> = LazyLock::new(|| {
    reqwest::Client::builder()
        .connect_timeout(HTTP_CONNECT_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .build()
});

fn http_client() -> Result<reqwest::Client, LlmError> {
    HTTP_CLIENT
        .as_ref()
        .cloned()
        .map_err(|_| LlmError::InvalidRequest)
}

fn base_url<'a>(endpoint: Option<&'a ProviderEndpoint>, default: &'a str) -> &'a str {
    endpoint.map_or(default, ProviderEndpoint::base)
}

/// The Ollama server's root. It serves the `OpenAI` protocol under `/v1` and its own API under
/// `/api`, so an endpoint given with or without `/v1` names the same server.
fn ollama_root(endpoint: Option<&ProviderEndpoint>) -> &str {
    let base = base_url(endpoint, OLLAMA_ROOT);
    base.strip_suffix("/v1").unwrap_or(base)
}

fn required_key(key: Option<&ApiKey>) -> Result<String, LlmError> {
    key.map(|key| key.expose_secret().to_owned())
        .ok_or(LlmError::Authentication)
}

/// For the providers that also run without a key.
fn optional_key(key: Option<&ApiKey>) -> String {
    key.map_or_else(String::new, |key| key.expose_secret().to_owned())
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
    // their full budget.
    let provider_request = model
        .completion_request(prompt.user())
        .preamble(prompt.system().to_owned())
        .build();
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
        if status.as_u16() == 404 || names_missing_model(error) {
            return LlmError::ModelNotFound;
        }
        return status_error(status.as_u16());
    }
    match error {
        CompletionError::HttpError(_) => LlmError::Transport,
        CompletionError::UrlError(_) | CompletionError::RequestError(_) => LlmError::InvalidRequest,
        _ => LlmError::InvalidResponse,
    }
}

fn status_error(status: u16) -> LlmError {
    match status {
        401 | 403 => LlmError::Authentication,
        408 | 429 => LlmError::RateLimited,
        500..=599 => LlmError::TemporarilyUnavailable,
        300..=399 => {
            tracing::warn!(
                status,
                "the provider redirected; set the endpoint it redirects to"
            );
            LlmError::InvalidRequest
        }
        // Rig also reports an error envelope that came with a success status.
        200..=299 => LlmError::InvalidResponse,
        _ => LlmError::InvalidRequest,
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
        ANTHROPIC_FALLBACK_MAX_TOKENS, GeminiFinishReason, OLLAMA_ROOT, OPENROUTER_BASE_URL,
        base_url, complete, gemini_finish_reason, map_error, ollama_root, required_key,
        status_error,
        test_server::{serve_keep_alive, serve_once},
    };
    use std::{net::TcpListener, time::Duration};

    use reqwest::StatusCode;
    use rig_core::completion::CompletionError;
    use whispernow_core::llm::{
        LlmError, LlmFinishReason, LlmPrompt, LlmResponse, build_cleanup_prompt,
    };
    use whispernow_domain::{ApiKey, ProviderEndpoint, RemoteCleanupTarget, RemoteProvider};

    fn request(
        provider: RemoteProvider,
        model: &str,
    ) -> (RemoteCleanupTarget, Option<ApiKey>, LlmPrompt) {
        (
            RemoteCleanupTarget {
                provider,
                model: model.to_owned(),
            },
            ApiKey::new("token").ok(),
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
    fn a_provider_uses_the_override_or_its_default() {
        assert_eq!(base_url(None, OPENROUTER_BASE_URL), OPENROUTER_BASE_URL);
        let endpoint = ProviderEndpoint::parse("http://127.0.0.1:11434/v1/").expect("endpoint");
        assert_eq!(
            base_url(Some(&endpoint), OPENROUTER_BASE_URL),
            "http://127.0.0.1:11434/v1"
        );
    }

    #[test]
    fn an_ollama_endpoint_names_its_server_root_with_or_without_v1() {
        assert_eq!(ollama_root(None), OLLAMA_ROOT);
        for (endpoint, root) in [
            ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
            ("http://127.0.0.1:11434/v1/", "http://127.0.0.1:11434"),
            (
                "https://proxy.example/ollama",
                "https://proxy.example/ollama",
            ),
            (
                "https://proxy.example/ollama/v1",
                "https://proxy.example/ollama",
            ),
        ] {
            let endpoint = ProviderEndpoint::parse(endpoint).expect("endpoint");
            assert_eq!(ollama_root(Some(&endpoint)), root);
        }
    }

    #[test]
    fn a_status_is_told_apart_by_what_the_user_can_do_about_it() {
        for (status, expected) in [
            (401, LlmError::Authentication),
            (403, LlmError::Authentication),
            (408, LlmError::RateLimited),
            (429, LlmError::RateLimited),
            (503, LlmError::TemporarilyUnavailable),
            (400, LlmError::InvalidRequest),
            (404, LlmError::InvalidRequest),
            (307, LlmError::InvalidRequest),
            (200, LlmError::InvalidResponse),
        ] {
            assert_eq!(status_error(status), expected, "{status}");
        }
    }

    async fn exchange(
        provider: impl FnOnce(ProviderEndpoint) -> RemoteProvider,
        model: &str,
        status_and_headers: &str,
        body: &str,
    ) -> (Result<LlmResponse, LlmError>, serde_json::Value) {
        let (address, server) = serve_once(status_and_headers, body);
        let endpoint = ProviderEndpoint::parse(format!("{address}/v1")).expect("fixture endpoint");
        let (target, api_key, prompt) = request(provider(endpoint), model);
        let result =
            tokio::time::timeout(Duration::from_secs(5), complete(target, api_key, prompt))
                .await
                .expect("complete did not return; the redirect was probably followed");
        let received = server.join().expect("fixture provider request");
        let body = serde_json::from_slice(&received.body).expect("request JSON");
        (result, body)
    }

    async fn sent_request(
        provider: impl FnOnce(ProviderEndpoint) -> RemoteProvider,
        model: &str,
    ) -> serde_json::Value {
        let (result, body) = exchange(provider, model, "400 Bad Request", "{}").await;
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

    #[tokio::test]
    async fn two_calls_reuse_one_connection() {
        let (address, server) = serve_keep_alive("400 Bad Request", "{}", 2);
        for _ in 0..2 {
            let endpoint = ProviderEndpoint::parse(format!("{address}/v1")).expect("endpoint");
            let (target, api_key, prompt) = request(RemoteProvider::Custom(endpoint), "local");
            let result = complete(target, api_key, prompt).await;
            assert_eq!(result, Err(LlmError::InvalidRequest));
        }
        assert_eq!(server.join().expect("fixture provider"), 1);
    }

    #[tokio::test]
    async fn a_redirect_is_not_followed() {
        let target = TcpListener::bind("127.0.0.1:0").expect("redirect target");
        target
            .set_nonblocking(true)
            .expect("nonblocking redirect target");
        let redirect = format!(
            "307 Temporary Redirect\r\nLocation: http://{}/",
            target.local_addr().expect("redirect target address")
        );
        let (result, _) = exchange(
            |endpoint| RemoteProvider::Anthropic(Some(endpoint)),
            "claude-x",
            &redirect,
            "",
        )
        .await;
        assert_eq!(result, Err(LlmError::InvalidRequest));
        assert!(
            matches!(target.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock),
            "the redirect target was contacted"
        );
    }
}
