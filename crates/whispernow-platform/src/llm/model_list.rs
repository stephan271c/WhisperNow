use std::time::Duration;

use rig_core::providers::anthropic::{
    client::normalize_anthropic_base_url, completion::ANTHROPIC_VERSION_LATEST,
};
use serde::Deserialize;
use whispernow_core::llm::LlmError;
use whispernow_domain::{ApiKey, RemoteProvider};

use super::{
    OPENAI_BASE_URL, OPENROUTER_BASE_URL, base_url, http_client, ollama_root, required_key,
    status_error,
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
    let http = http_client()?;
    let optional_bearer = |request: reqwest::RequestBuilder| match api_key {
        Some(key) => request.bearer_auth(key.expose_secret()),
        None => request,
    };
    let mut models = match provider {
        RemoteProvider::OpenAi(endpoint) => {
            let base = base_url(endpoint.as_ref(), OPENAI_BASE_URL);
            let request = http
                .get(format!("{base}/models"))
                .bearer_auth(required_key(api_key)?);
            ids(fetch::<IdList>(request).await?)
        }
        // OpenRouter lists its models without a key.
        RemoteProvider::OpenRouter(endpoint) => {
            let base = base_url(endpoint.as_ref(), OPENROUTER_BASE_URL);
            let request = http.get(format!("{base}/models"));
            ids(fetch::<IdList>(optional_bearer(request)).await?)
        }
        RemoteProvider::Custom(endpoint) => {
            let request = http.get(format!("{}/models", endpoint.base()));
            ids(fetch::<IdList>(optional_bearer(request)).await?)
        }
        RemoteProvider::Ollama(endpoint) => {
            let root = ollama_root(endpoint.as_ref());
            let request = http.get(format!("{root}/api/tags"));
            fetch::<OllamaTags>(optional_bearer(request))
                .await?
                .models
                .into_iter()
                .map(|tag| tag.name)
                .collect()
        }
        RemoteProvider::Anthropic(endpoint) => {
            let base =
                normalize_anthropic_base_url(base_url(endpoint.as_ref(), ANTHROPIC_BASE_URL));
            let request = http
                .get(format!("{base}/v1/models?limit=1000"))
                .header("x-api-key", required_key(api_key)?)
                .header("anthropic-version", ANTHROPIC_VERSION_LATEST);
            ids(fetch::<IdList>(request).await?)
        }
        RemoteProvider::Gemini(endpoint) => {
            let base = base_url(endpoint.as_ref(), GEMINI_BASE_URL);
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
    let response = request
        .timeout(LIST_TIMEOUT)
        .send()
        .await
        .map_err(|_| LlmError::Transport)?;
    let status = response.status();
    if !status.is_success() {
        return Err(status_error(status.as_u16()));
    }
    let body = response.bytes().await.map_err(|_| LlmError::Transport)?;
    serde_json::from_slice(&body).map_err(|_| LlmError::InvalidResponse)
}

#[cfg(test)]
mod tests {
    use whispernow_core::llm::LlmError;
    use whispernow_domain::{ApiKey, ProviderEndpoint, RemoteProvider};

    use super::{super::test_server::serve_once, list_models};

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
        let request = server.join().expect("fixture").head;
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
        let request = server.join().expect("fixture").head;
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
        let request = server.join().expect("fixture").head;
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
        let request = server.join().expect("fixture").head;
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
    async fn openrouter_is_asked_without_a_key() {
        let (address, server) = serve_once("200 OK", r#"{"data":[{"id":"openai/gpt-6-mini"}]}"#);
        let provider = RemoteProvider::OpenRouter(Some(endpoint(format!("{address}/api/v1"))));
        let models = list_models(&provider, None).await;
        let request = server.join().expect("fixture").head;
        assert_eq!(models, Ok(vec!["openai/gpt-6-mini".to_owned()]));
        assert!(request.starts_with("GET /api/v1/models "), "{request}");
        assert!(!request.contains("authorization"), "{request}");
    }

    #[tokio::test]
    async fn a_cloud_provider_without_a_key_is_not_asked() {
        for provider in [
            RemoteProvider::OpenAi(None),
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
