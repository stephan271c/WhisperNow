use whispernow_domain::{EndpointError, LlmProvider, ProviderEndpoint};

#[test]
fn provider_api_base_requires_a_valid_url_and_exact_loopback_host_for_http() {
    for allowed in [
        "https://api.example.com/v1",
        "HTTPS://api.example.com:443/v1",
        "http://localhost:11434/v1",
        "http://127.0.0.1:11434/v1",
        "http://127.255.255.254/v1",
        "http://[::1]:11434/v1",
    ] {
        assert!(
            ProviderEndpoint::parse(allowed).is_ok(),
            "expected {allowed:?} to be allowed"
        );
    }

    for rejected in [
        "http://127.evil.com/v1",
        "http://localhost.evil.com/v1",
        "http://192.168.1.2/v1",
        "https://",
        "https://example.com:not-a-port/v1",
        "https://[::1/v1",
        "https://example.com/v1#credentials",
        "https://example.com/v1?api_key=credentials",
        "http://localhost:11434/v1?token=credentials",
        "https://example.com/v1\n",
        "https://user:password@example.com/v1",
        "http://user@localhost/v1",
    ] {
        assert!(
            ProviderEndpoint::parse(rejected).is_err(),
            "expected {rejected:?} to be rejected"
        );
    }
}

#[test]
fn a_plain_http_remote_endpoint_is_refused_as_insecure() {
    assert_eq!(
        ProviderEndpoint::parse("http://remote.example/v1").map(|_| ()),
        Err(EndpointError::InsecureRemote)
    );
    assert_eq!(
        ProviderEndpoint::parse("http://[::1]:11434/v1")
            .map(|endpoint| endpoint.expose_url().to_owned()),
        Ok("http://[::1]:11434/v1".to_owned())
    );
}

#[test]
fn provider_names_round_trip_and_accept_the_google_alias() {
    for provider in LlmProvider::ALL {
        assert_eq!(LlmProvider::from_name(provider.as_str()), Some(provider));
    }
    assert_eq!(
        LlmProvider::from_name(" Google "),
        Some(LlmProvider::Gemini)
    );
    assert_eq!(LlmProvider::from_name("future-provider"), None);
}

#[test]
fn only_a_loopback_host_is_on_this_computer() {
    for (url, local) in [
        ("https://localhost:8443/v1", true),
        ("http://127.0.0.1:11434/v1", true),
        ("http://[::1]:11434/v1", true),
        ("https://api.example.com/v1", false),
        ("https://localhost.example.com/v1", false),
    ] {
        let endpoint = ProviderEndpoint::parse(url).expect("a valid endpoint");
        assert_eq!(endpoint.is_on_this_computer(), local, "{url}");
    }
}
