use std::{fs, sync::Arc};

use serde_json::{Value, json};
use tempfile::{TempDir, tempdir};
use whispernow_domain::{ApiKey, HistoryLimit, LlmProvider, Settings, SetupCheckpoint, SetupStep};

use crate::{AppPaths, MemoryCredentialStore, SettingsNotice, SettingsStore};

struct Fixture {
    _root: TempDir,
    paths: AppPaths,
    store: Arc<MemoryCredentialStore>,
    persistence: SettingsStore,
}

fn fixture() -> Fixture {
    let root = tempdir().expect("temporary directory");
    let paths = AppPaths::from_base_dirs(root.path().join("config"), root.path().join("data"));
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
        Settings::default()
    );

    let settings = Settings {
        keep_audio: true,
        history_limit: HistoryLimit::Last100,
        setup: Some(SetupCheckpoint {
            step: SetupStep::Shortcuts,
            ..SetupCheckpoint::default()
        }),
        ..Settings::default()
    };
    fixture.persistence.save(&settings).expect("save");
    assert_eq!(fixture.persistence.load().expect("load"), settings);
    assert_eq!(fixture.saved()["settings_version"], 2);
    assert_eq!(fixture.persistence.take_notice(), None);
}

#[test]
fn an_unreadable_file_is_renamed_and_the_defaults_load() {
    for document in [
        "{ not json",
        r#"{"settings_version":2,"keep_audio":"fast"}"#,
    ] {
        let fixture = fixture();
        fixture.write(document);

        assert_eq!(
            fixture.persistence.load().expect("load"),
            Settings::default()
        );
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
            fixture.persistence.take_notice(),
            Some(SettingsNotice::Unreadable {
                path: moved[0].clone()
            })
        );
        assert_eq!(fixture.persistence.take_notice(), None, "told once");
        assert_eq!(
            fixture.persistence.load().expect("reload"),
            Settings::default()
        );
        assert_eq!(fixture.persistence.take_notice(), None);
    }
}

#[test]
fn python_settings_are_imported_once_with_their_keys_moved_to_the_store() {
    let fixture = fixture();
    let secret = ["test", "-only", "-key"].concat();
    let python = json!({
        "sample_rate": 44100,
        "first_run_complete": true,
        "llm_provider": "openai",
        "llm_provider_settings": { "openai": { "model": "gpt-test", "api_key": secret } }
    })
    .to_string();
    fixture.write(&python);

    let imported = fixture.persistence.load().expect("import");
    assert!(imported.first_run_complete);
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
    assert_eq!(
        fs::read_to_string(fixture.paths.python_settings_file()).expect("kept"),
        python
    );
    assert_eq!(fixture.persistence.take_notice(), None);
    assert_eq!(fixture.persistence.load().expect("reload"), imported);
}

#[test]
fn a_python_key_the_store_refuses_stays_in_the_kept_file() {
    let fixture = fixture();
    fixture.store.set_available(false);
    fixture.write(r#"{"llm_provider_settings":{"gemini":{"model":"g","api_key":"k"}}}"#);

    let imported = fixture.persistence.load().expect("import");
    assert!(imported.api_keys.is_empty());
    assert_eq!(imported.provider_settings(LlmProvider::Gemini).model, "g");
    assert_eq!(
        fixture.persistence.take_notice(),
        Some(SettingsNotice::ApiKeysNotImported {
            path: fixture.paths.python_settings_file()
        })
    );
}

#[test]
fn an_api_key_is_set_read_back_and_cleared() {
    let fixture = fixture();
    let key = ApiKey::new(["test", "-only", "-key"].concat()).expect("a key");
    let persistence = &fixture.persistence;

    persistence
        .store_api_key(LlmProvider::Anthropic, &key)
        .expect("store key");
    let mut settings = Settings::default();
    settings.api_keys.insert(LlmProvider::Anthropic);
    persistence.save(&settings).expect("save");
    assert_eq!(
        persistence.api_key(LlmProvider::Anthropic).expect("read"),
        Some(key)
    );

    persistence
        .delete_api_key(LlmProvider::Anthropic)
        .expect("delete key");
    settings.api_keys.clear();
    persistence.save(&settings).expect("save");
    assert_eq!(
        persistence.api_key(LlmProvider::Anthropic).expect("read"),
        None
    );

    fixture.store.set_available(false);
    assert!(persistence.api_key(LlmProvider::Anthropic).is_err());
    assert!(persistence.delete_api_key(LlmProvider::Anthropic).is_err());
}
