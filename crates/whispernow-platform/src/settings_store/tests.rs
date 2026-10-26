use std::{fs, sync::Arc};

use serde_json::{Value, json};
use tempfile::{TempDir, tempdir};
use whispernow_domain::{
    ApiKey, HistoryLimit, LlmProvider, Settings, SetupCheckpoint, SetupProgress, SetupStep,
};

use crate::{
    AppPaths, CredentialStore, CredentialStoreUnavailable, MemoryCredentialStore, SettingsLoad,
    SettingsNotice, SettingsStore,
};

struct Fixture {
    _root: TempDir,
    paths: AppPaths,
    store: Arc<MemoryCredentialStore>,
    persistence: SettingsStore,
}

fn fixture() -> Fixture {
    let root = tempdir().expect("temporary directory");
    let paths = AppPaths::under(root.path());
    fs::create_dir_all(paths.config_dir()).expect("config directory");
    let store = Arc::new(MemoryCredentialStore::default());
    let persistence = SettingsStore::with_store(paths.clone(), Arc::clone(&store) as _);
    Fixture {
        _root: root,
        paths,
        store,
        persistence,
    }
}

impl Fixture {
    fn write(&self, document: &str) {
        fs::write(self.paths.settings_file(), document).expect("settings fixture");
    }

    fn saved(&self) -> Value {
        serde_json::from_slice(&fs::read(self.paths.settings_file()).expect("settings file"))
            .expect("saved settings are JSON")
    }
}

#[test]
fn settings_round_trip() {
    let fixture = fixture();
    assert_eq!(
        fixture.persistence.load().expect("no file"),
        SettingsLoad::from(Settings::default())
    );

    let settings = Settings {
        keep_audio: true,
        history_limit: HistoryLimit::Last100,
        setup: SetupProgress::Unfinished(SetupCheckpoint {
            step: SetupStep::Shortcuts,
            ..SetupCheckpoint::default()
        }),
        ..Settings::default()
    };
    fixture.persistence.save(&settings).expect("save");
    assert_eq!(
        fixture.persistence.load().expect("load"),
        SettingsLoad::from(settings)
    );
    assert_eq!(fixture.saved()["settings_version"], 1);
}

#[test]
fn an_unreadable_file_is_renamed_and_the_defaults_load() {
    for document in [
        "{ not json",
        r#"{"settings_version":1,"keep_audio":"fast"}"#,
    ] {
        let fixture = fixture();
        fixture.write(document);

        let SettingsLoad {
            settings: loaded,
            notice,
            ..
        } = fixture.persistence.load().expect("load");
        assert_eq!(loaded, Settings::default());
        let moved = fixture.paths.settings_backup_files().expect("list");
        assert_eq!(moved.len(), 1, "{document}");
        assert!(
            moved[0]
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("settings.json.bad-"))
        );
        assert_eq!(fs::read_to_string(&moved[0]).expect("moved"), document);
        assert!(!fixture.paths.settings_file().exists());
        assert_eq!(
            notice,
            Some(SettingsNotice::Unreadable {
                path: moved[0].clone()
            })
        );
        assert_eq!(
            fixture.persistence.load().expect("reload"),
            SettingsLoad::from(Settings::default())
        );
    }
}

#[test]
fn python_settings_are_imported_once_with_their_keys_moved_to_the_store() {
    let fixture = fixture();
    let secret = ["test", "-only", "-key"].concat();
    let python = json!({
        "sample_rate": 44100,
        "first_run_complete": true,
        "auto_start_on_login": true,
        "hotkey": { "modifiers": ["ctrl"], "key": "r" },
        "llm_provider": "openai",
        "llm_provider_settings": { "openai": { "model": "gpt-test", "api_key": secret } }
    })
    .to_string();
    fixture.write(&python);

    let SettingsLoad {
        settings: imported,
        notice,
        enable_autostart,
    } = fixture.persistence.load().expect("import");
    assert!(imported.setup.is_complete());
    assert!(enable_autostart);
    assert_eq!(
        imported.recording_shortcut.key,
        whispernow_domain::ShortcutKey::R
    );
    assert!(fixture.saved().get("auto_start_on_login").is_none());
    assert_eq!(imported.api_keys, [LlmProvider::OpenAi].into());
    assert_eq!(
        fixture
            .persistence
            .api_key(LlmProvider::OpenAi)
            .expect("read key"),
        Some(ApiKey::new(secret.clone()).expect("a key"))
    );
    let saved = fs::read_to_string(fixture.paths.settings_file()).expect("settings");
    assert!(!saved.contains(&secret), "the new file holds no key");
    let copy = fs::read_to_string(fixture.paths.python_settings_file()).expect("kept");
    assert!(!copy.contains(&secret), "the copy holds no accepted key");
    assert_eq!(
        serde_json::from_str::<Value>(&copy).expect("kept settings are JSON"),
        json!({
            "sample_rate": 44100,
            "first_run_complete": true,
            "auto_start_on_login": true,
            "hotkey": { "modifiers": ["ctrl"], "key": "r" },
            "llm_provider": "openai",
            "llm_provider_settings": { "openai": { "model": "gpt-test" } }
        })
    );
    assert_eq!(notice, None);
    assert_eq!(
        fixture.persistence.load().expect("reload"),
        SettingsLoad::from(imported)
    );
}

#[test]
fn a_python_key_the_store_refuses_stays_in_the_kept_file() {
    let fixture = fixture();
    fixture.store.set_available(false);
    fixture.write(r#"{"llm_provider_settings":{"gemini":{"model":"g","api_key":"k"}}}"#);

    let SettingsLoad {
        settings: imported,
        notice,
        ..
    } = fixture.persistence.load().expect("import");
    assert!(imported.api_keys.is_empty());
    assert_eq!(
        imported
            .provider_settings(LlmProvider::Gemini)
            .model
            .as_deref(),
        Some("g")
    );
    assert_eq!(
        serde_json::from_slice::<Value>(
            &fs::read(fixture.paths.python_settings_file()).expect("kept")
        )
        .expect("kept settings are JSON"),
        json!({ "llm_provider_settings": { "gemini": { "model": "g", "api_key": "k" } } })
    );
    assert_eq!(
        notice,
        Some(SettingsNotice::ApiKeysNotImported {
            path: fixture.paths.python_settings_file()
        })
    );
}

struct RefusingGeminiStore(MemoryCredentialStore);

impl CredentialStore for RefusingGeminiStore {
    fn read(&self, provider: LlmProvider) -> Result<Option<ApiKey>, CredentialStoreUnavailable> {
        self.0.read(provider)
    }

    fn write(&self, provider: LlmProvider, key: &ApiKey) -> Result<(), CredentialStoreUnavailable> {
        if provider == LlmProvider::Gemini {
            Err(CredentialStoreUnavailable)
        } else {
            self.0.write(provider, key)
        }
    }

    fn delete(&self, provider: LlmProvider) -> Result<(), CredentialStoreUnavailable> {
        self.0.delete(provider)
    }
}

#[test]
fn a_partial_python_import_keeps_only_the_refused_key() {
    let fixture = fixture();
    let persistence = SettingsStore::with_store(
        fixture.paths.clone(),
        Arc::new(RefusingGeminiStore(MemoryCredentialStore::default())),
    );
    let accepted = "accepted-openai-key";
    let refused = "refused-gemini-key";
    fixture.write(
        &json!({
            "sample_rate": 44100,
            "llm_provider_settings": {
                "openai": { "model": "gpt-test", "api_key": accepted },
                "gemini": { "model": "gemini-test", "api_key": refused }
            },
            "unknown": { "values": [true, null, 42] }
        })
        .to_string(),
    );

    let SettingsLoad {
        settings: imported,
        notice,
        ..
    } = persistence.load().expect("partial import");
    assert_eq!(imported.api_keys, [LlmProvider::OpenAi].into());
    assert_eq!(
        persistence
            .api_key(LlmProvider::OpenAi)
            .expect("stored key"),
        Some(ApiKey::new(accepted).expect("a key"))
    );
    let copy = fs::read_to_string(fixture.paths.python_settings_file()).expect("kept");
    assert!(!copy.contains(accepted));
    assert!(copy.contains(refused));
    assert_eq!(
        serde_json::from_str::<Value>(&copy).expect("kept settings are JSON"),
        json!({
            "sample_rate": 44100,
            "llm_provider_settings": {
                "openai": { "model": "gpt-test" },
                "gemini": { "model": "gemini-test", "api_key": refused }
            },
            "unknown": { "values": [true, null, 42] }
        })
    );
    assert_eq!(
        notice,
        Some(SettingsNotice::ApiKeysNotImported {
            path: fixture.paths.python_settings_file()
        })
    );
    assert_eq!(
        persistence.load().expect("reload"),
        SettingsLoad::from(imported)
    );
}
