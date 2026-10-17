use std::time::Duration;

use rig_core::providers::anthropic::{
    client::normalize_anthropic_base_url, completion::ANTHROPIC_VERSION_LATEST,
};
use serde::Deserialize;
use whispernow_core::llm::LlmError;
use whispernow_domain::{ApiKey, ProviderEndpoint, RemoteProvider};

use super::{
    HTTP_CONNECT_TIMEOUT, OLLAMA_BASE_URL, OPENAI_BASE_URL, OPENROUTER_BASE_URL,
    openai_compatible_base_url, required_key,
};

const LIST_TIMEOUT: Duration = Duration::from_secs(20);
const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";
const GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com";

#[derive(Deserialize)]
struct IdList {
    data: Vec<Id>,
}

#[derive(Deserialize)]
struct Id {
    id: String,
}

#[derive(Deserialize)]
struct OllamaTags {
    models: Vec<OllamaTag>,
}

#[derive(Deserialize)]
struct OllamaTag {
    name: String,
}

#[derive(Deserialize)]
struct GeminiModels {
    #[serde(default)]
    models: Vec<GeminiModel>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiModel {
    name: String,
    #[serde(default)]
    supported_generation_methods: Vec<String>,
}

/// The model names `provider` offers, sorted, as its cleanup requests name them.
pub async fn list_models(
    provider: &RemoteProvider,
    api_key: Option<&ApiKey>,
) -> Result<Vec<String>, LlmError> {
    // Redirects are not followed, so a key sent in a provider header never reaches another host.
    let http = reqwest::Client::builder()
        .connect_timeout(HTTP_CONNECT_TIMEOUT)
        .timeout(LIST_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_| LlmError::InvalidRequest)?;
    let optional_bearer = |request: reqwest::RequestBuilder| match api_key {
        Some(key) => request.bearer_auth(key.expose_secret()),
        None => request,
    };
    let mut models = match provider {
        RemoteProvider::OpenAi(endpoint) => {
            let base = openai_compatible_base_url(endpoint.as_ref(), OPENAI_BASE_URL);
            let request = http
                .get(format!("{}/models", base.trim_end_matches('/')))
                .bearer_auth(required_key(api_key)?);
            ids(fetch::<IdList>(request).await?)
        }
        RemoteProvider::OpenRouter(endpoint) => {
            let base = openai_compatible_base_url(endpoint.as_ref(), OPENROUTER_BASE_URL);
            let request = http
                .get(format!("{}/models", base.trim_end_matches('/')))
                .bearer_auth(required_key(api_key)?);
            ids(fetch::<IdList>(request).await?)
        }
        RemoteProvider::Custom(endpoint) => {
            let request = http.get(format!(
                "{}/models",
                endpoint.expose_url().trim_end_matches('/')
            ));
            ids(fetch::<IdList>(optional_bearer(request)).await?)
        }
        RemoteProvider::Ollama(endpoint) => {
            let base = endpoint
                .as_ref()
                .map_or(OLLAMA_BASE_URL, ProviderEndpoint::expose_url)
                .trim_end_matches('/');
            let root = base.strip_suffix("/v1").unwrap_or(base);
            let request = http.get(format!("{root}/api/tags"));
            fetch::<OllamaTags>(optional_bearer(request))
                .await?
                .models
                .into_iter()
                .map(|tag| tag.name)
                .collect()
        }
        RemoteProvider::Anthropic(endpoint) => {
            let base = normalize_anthropic_base_url(
                endpoint
                    .as_ref()
                    .map_or(ANTHROPIC_BASE_URL, ProviderEndpoint::expose_url),
            );
            let request = http
                .get(format!("{base}/v1/models?limit=1000"))
                .header("x-api-key", required_key(api_key)?)
                .header("anthropic-version", ANTHROPIC_VERSION_LATEST);
            ids(fetch::<IdList>(request).await?)
        }
        RemoteProvider::Gemini(endpoint) => {
            let base = endpoint
                .as_ref()
                .map_or(GEMINI_BASE_URL, ProviderEndpoint::expose_url)
                .trim_end_matches('/');
            let request = http
                .get(format!("{base}/v1beta/models?pageSize=1000"))
                .header("x-goog-api-key", required_key(api_key)?);
            fetch::<GeminiModels>(request)
                .await?
                .models
                .into_iter()
                .filter(|model| {
                    model
                        .supported_generation_methods
                        .iter()
                        .any(|method| method == "generateContent")
                })
                .map(|model| {
                    model
                        .name
                        .strip_prefix("models/")
                        .unwrap_or(&model.name)
                        .to_owned()
                })
                .collect()
        }
    };
    models.sort();
    models.dedup();
    Ok(models)
}

fn ids(list: IdList) -> Vec<String> {
    list.data.into_iter().map(|model| model.id).collect()
}

async fn fetch<T: serde::de::DeserializeOwned>(
    request: reqwest::RequestBuilder,
) -> Result<T, LlmError> {
    let response = request.send().await.map_err(|_| LlmError::Transport)?;
    match response.status().as_u16() {
        200..=299 => {}
        401 | 403 => return Err(LlmError::Authentication),
        429 => return Err(LlmError::RateLimited),
        500..=599 => return Err(LlmError::TemporarilyUnavailable),
        _ => return Err(LlmError::InvalidRequest),
    }
    let body = response.bytes().await.map_err(|_| LlmError::Transport)?;
    serde_json::from_slice(&body).map_err(|_| LlmError::InvalidResponse)
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
        time::Duration,
    };

    use whispernow_core::llm::LlmError;
    use whispernow_domain::{ApiKey, ProviderEndpoint, RemoteProvider};

    use super::list_models;

    /// Answers one request with `status` and `body`, and returns the request's head.
    fn serve_once(
        status: &'static str,
        body: &'static str,
    ) -> (String, thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("fixture listener");
        let address = format!("http://{}", listener.local_addr().expect("fixture address"));
        let server = thread::spawn(move || {
            let (mut connection, _) = listener.accept().expect("provider request");
            connection
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("request timeout");
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = connection.read(&mut buffer).expect("request bytes");
                assert!(read > 0, "request ended before its headers");
                request.extend_from_slice(&buffer[..read]);
            }
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            connection
                .write_all(response.as_bytes())
                .expect("fixture response");
            String::from_utf8(request).expect("request head")
        });
        (address, server)
    }

    fn endpoint(url: String) -> ProviderEndpoint {
        ProviderEndpoint::parse(url).expect("fixture endpoint")
    }

    fn key() -> ApiKey {
        ApiKey::new("secret-key").expect("a key")
    }

    #[tokio::test]
    async fn an_openai_compatible_endpoint_lists_its_model_ids_sorted() {
        let (address, server) = serve_once(
            "200 OK",
            r#"{"object":"list","data":[{"id":"qwen3"},{"id":"llama-3.2","object":"model"}]}"#,
        );
        let provider = RemoteProvider::Custom(endpoint(format!("{address}/v1")));
        let models = list_models(&provider, Some(&key())).await;
        let request = server.join().expect("fixture");
        assert_eq!(models, Ok(vec!["llama-3.2".to_owned(), "qwen3".to_owned()]));
        assert!(request.starts_with("GET /v1/models "), "{request}");
        assert!(
            request.contains("authorization: Bearer secret-key"),
            "{request}"
        );
    }

    #[tokio::test]
    async fn ollama_lists_its_tags_from_the_server_root() {
        let (address, server) = serve_once(
            "200 OK",
            r#"{"models":[{"name":"mistral:latest","model":"mistral:latest"},{"name":"gemma3:4b"}]}"#,
        );
        let provider = RemoteProvider::Ollama(Some(endpoint(format!("{address}/v1/"))));
        let models = list_models(&provider, None).await;
        let request = server.join().expect("fixture");
        assert_eq!(
            models,
            Ok(vec!["gemma3:4b".to_owned(), "mistral:latest".to_owned()])
        );
        assert!(request.starts_with("GET /api/tags "), "{request}");
        assert!(!request.contains("authorization"), "{request}");
    }

    #[tokio::test]
    async fn anthropic_sends_its_key_and_version_headers() {
        let (address, server) = serve_once(
            "200 OK",
            r#"{"data":[{"id":"claude-haiku-5-5","type":"model"}],"has_more":false}"#,
        );
        let provider = RemoteProvider::Anthropic(Some(endpoint(address)));
        let models = list_models(&provider, Some(&key())).await;
        let request = server.join().expect("fixture");
        assert_eq!(models, Ok(vec!["claude-haiku-5-5".to_owned()]));
        assert!(
            request.starts_with("GET /v1/models?limit=1000 "),
            "{request}"
        );
        assert!(request.contains("x-api-key: secret-key"), "{request}");
        assert!(request.contains("anthropic-version: "), "{request}");
    }

    #[tokio::test]
    async fn gemini_lists_only_models_that_generate_content() {
        let (address, server) = serve_once(
            "200 OK",
            r#"{"models":[
                {"name":"models/gemini-2.5-flash","supportedGenerationMethods":["generateContent","countTokens"]},
                {"name":"models/text-embedding-004","supportedGenerationMethods":["embedContent"]}
            ]}"#,
        );
        let provider = RemoteProvider::Gemini(Some(endpoint(address)));
        let models = list_models(&provider, Some(&key())).await;
        let request = server.join().expect("fixture");
        assert_eq!(models, Ok(vec!["gemini-2.5-flash".to_owned()]));
        assert!(
            request.starts_with("GET /v1beta/models?pageSize=1000 "),
            "{request}"
        );
        assert!(request.contains("x-goog-api-key: secret-key"), "{request}");
        assert!(
            !request.contains("key=secret-key"),
            "the key stays out of the URL"
        );
    }

    #[tokio::test]
    async fn a_hosted_provider_without_a_key_is_not_asked() {
        for provider in [
            RemoteProvider::OpenAi(None),
            RemoteProvider::OpenRouter(None),
            RemoteProvider::Anthropic(None),
            RemoteProvider::Gemini(None),
        ] {
            assert_eq!(
                list_models(&provider, None).await,
                Err(LlmError::Authentication)
            );
        }
    }

    #[tokio::test]
    async fn a_redirect_is_not_followed() {
        // Following it would reach the fixture again after it closed, a transport failure.
        let (address, server) = serve_once("307 Temporary Redirect\r\nLocation: /elsewhere", "");
        let provider = RemoteProvider::OpenAi(Some(endpoint(address)));
        let models = list_models(&provider, Some(&key())).await;
        server.join().expect("fixture");
        assert_eq!(models, Err(LlmError::InvalidRequest));
    }
}
