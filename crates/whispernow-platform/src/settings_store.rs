//! `settings.json`, with every API key kept in the OS credential store instead.

use std::{fs, io::ErrorKind, path::PathBuf, sync::Arc};
use whispernow_domain::AppError;

use serde::Serialize;
use serde_json::Value;
use time::OffsetDateTime;
use whispernow_domain::{ApiKey, ErrorCategory, LlmProvider, PythonSettings, Settings};

use crate::{
    AppPaths,
    atomic_file::write_atomically,
    credential_store::{CredentialStore, CredentialStoreUnavailable, OsCredentialStore},
};

/// Written into every saved document. The Python release's documents lack it, so a
/// document without it is imported once.
const VERSION_FIELD: &str = "settings_version";
const SETTINGS_VERSION: u32 = 2;

#[derive(Serialize)]
struct Stored<'a> {
    settings_version: u32,
    #[serde(flatten)]
    settings: &'a Settings,
}

/// Something that happened to `settings.json` which the user is told about once.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SettingsNotice {
    /// The file could not be read. It was renamed to `path` and the defaults are in effect.
    Unreadable { path: PathBuf },
    /// The Python release's settings were imported, but the OS credential store refused
    /// their API keys, which remain in plain text in `path`.
    ApiKeysNotImported { path: PathBuf },
}

pub struct SettingsStore {
    paths: AppPaths,
    keys: Arc<dyn CredentialStore>,
}

impl SettingsStore {
    #[must_use]
    pub fn new(paths: AppPaths) -> Self {
        Self::with_store(paths, Arc::new(OsCredentialStore))
    }

    #[must_use]
    pub fn with_store(paths: AppPaths, keys: Arc<dyn CredentialStore>) -> Self {
        Self { paths, keys }
    }

    /// Loads the saved settings; without a file they are the defaults. The Python
    /// release's file is imported and replaced. A file that cannot be read is renamed to
    /// `settings.json.bad-<timestamp>` and the defaults are loaded with a notice.
    pub fn load(&self) -> Result<(Settings, Option<SettingsNotice>), AppError> {
        let path = self.paths.settings_file();
        let document = match fs::read(&path) {
            Ok(document) => document,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return Ok((Settings::default(), None));
            }
            Err(_) => return Err(redacted("settings_read", "could not read settings")),
        };
        match serde_json::from_slice::<Value>(&document) {
            Ok(value) if value.is_object() && value.get(VERSION_FIELD).is_none() => {
                return self.import_python(&value, &document);
            }
            Ok(value) => {
                if let Ok(settings) = serde_json::from_value(value) {
                    return Ok((settings, None));
                }
            }
            Err(_) => {}
        }
        let bad = self.paths.unreadable_settings_file(&timestamp());
        fs::rename(&path, &bad).map_err(|_| {
            redacted(
                "settings_unreadable",
                "Settings could not be read, and the file could not be moved aside",
            )
        })?;
        tracing::warn!(
            code = "settings_unreadable",
            "unreadable settings were moved aside and the defaults loaded"
        );
        Ok((
            Settings::default(),
            Some(SettingsNotice::Unreadable { path: bad }),
        ))
    }

    pub fn save(&self, settings: &Settings) -> Result<(), AppError> {
        let document = serde_json::to_vec_pretty(&Stored {
            settings_version: SETTINGS_VERSION,
            settings,
        })
        .map_err(|_| redacted("settings_encode", "could not encode settings"))?;
        write_atomically(&self.paths.settings_file(), &document)
            .map_err(|_| redacted("settings_write", "could not save settings"))
    }

    /// The API key of `provider` in the credential store. Callers ask only when
    /// [`Settings::api_keys`] says it holds one.
    pub fn api_key(&self, provider: LlmProvider) -> Result<Option<ApiKey>, AppError> {
        self.keys.read(&account(provider)).map_err(|_| {
            redacted(
                "credential_store_unavailable",
                "The OS credential store is unavailable, so the API key cannot be used",
            )
        })
    }

    /// Saves the API key of `provider` in the credential store. The caller records it in
    /// [`Settings::api_keys`].
    pub fn store_api_key(
        &self,
        provider: LlmProvider,
        key: &ApiKey,
    ) -> Result<(), CredentialStoreUnavailable> {
        self.keys.write(&account(provider), key)
    }

    /// Removes the API key of `provider` from the credential store; an absent key is removed.
    pub fn delete_api_key(&self, provider: LlmProvider) -> Result<(), CredentialStoreUnavailable> {
        self.keys.delete(&account(provider))
    }

    /// Imports the Python release's settings: its API keys move into the credential
    /// store, the file is kept as `settings.json.python`, and the imported settings
    /// replace it.
    fn import_python(
        &self,
        document: &Value,
        original: &[u8],
    ) -> Result<(Settings, Option<SettingsNotice>), AppError> {
        let PythonSettings {
            mut settings,
            api_keys,
        } = PythonSettings::import(document);
        let kept = self.paths.python_settings_file();
        write_atomically(&kept, original).map_err(|_| {
            redacted(
                "settings_import",
                "The earlier version's settings could not be imported",
            )
        })?;
        let mut refused = false;
        for (provider, key) in &api_keys {
            if self.store_api_key(*provider, key).is_ok() {
                settings.api_keys.insert(*provider);
            } else {
                refused = true;
            }
        }
        self.save(&settings)?;
        tracing::info!(
            keys = settings.api_keys.len(),
            "imported the earlier version's settings"
        );
        let notice = refused.then_some(SettingsNotice::ApiKeysNotImported { path: kept });
        Ok((settings, notice))
    }
}

fn account(provider: LlmProvider) -> String {
    format!("llm-provider:{}", provider.as_str())
}

fn timestamp() -> String {
    let now = OffsetDateTime::now_utc();
    format!(
        "{:04}{:02}{:02}T{:02}{:02}{:02}Z",
        now.year(),
        u8::from(now.month()),
        now.day(),
        now.hour(),
        now.minute(),
        now.second()
    )
}

fn redacted(code: &'static str, summary: &'static str) -> AppError {
    AppError::new(ErrorCategory::General, code, summary)
}

#[cfg(test)]
mod tests;
