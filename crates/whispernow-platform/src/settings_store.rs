//! `settings.json`, with every API key kept in the OS credential store instead.

use std::{
    collections::BTreeSet,
    fs,
    io::{self, ErrorKind},
    path::PathBuf,
    sync::Arc,
};

use serde::Serialize;
use serde_json::Value;
use whispernow_domain::{ApiKey, AppError, ErrorCategory, LlmProvider, PythonSettings, Settings};

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

#[derive(Debug, Eq, PartialEq)]
pub struct SettingsLoad {
    pub settings: Settings,
    pub notice: Option<SettingsNotice>,
    pub enable_autostart: bool,
}

impl From<Settings> for SettingsLoad {
    fn from(settings: Settings) -> Self {
        Self {
            settings,
            notice: None,
            enable_autostart: false,
        }
    }
}

pub struct SettingsStore {
    paths: AppPaths,
    credentials: Arc<dyn CredentialStore>,
}

impl SettingsStore {
    #[must_use]
    pub fn new(paths: AppPaths) -> Self {
        Self::with_store(paths, Arc::new(OsCredentialStore))
    }

    #[must_use]
    pub fn with_store(paths: AppPaths, credentials: Arc<dyn CredentialStore>) -> Self {
        Self { paths, credentials }
    }

    /// Loads the saved settings; without a file they are the defaults. The Python
    /// release's file is imported and replaced. A file that cannot be read is renamed to
    /// `settings.json.bad-<timestamp>` and the defaults are loaded with a notice.
    pub fn load(&self) -> Result<SettingsLoad, AppError> {
        let path = self.paths.settings_file();
        let document = match fs::read(&path) {
            Ok(document) => document,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return Ok(Settings::default().into());
            }
            Err(error) => {
                return Err(io_failed("settings_read", "could not read settings")(error));
            }
        };
        match serde_json::from_slice::<Value>(&document) {
            Ok(value) if value.is_object() && value.get(VERSION_FIELD).is_none() => {
                return self.import_python(&value);
            }
            Ok(value) => {
                if let Ok(settings) = serde_json::from_value::<Settings>(value) {
                    return Ok(settings.into());
                }
            }
            Err(_) => {}
        }
        let bad = self
            .paths
            .unreadable_settings_file(&AppPaths::unreadable_stamp());
        fs::rename(&path, &bad).map_err(io_failed(
            "settings_unreadable",
            "Settings could not be read, and the file could not be moved aside",
        ))?;
        tracing::warn!(
            code = "settings_unreadable",
            "unreadable settings were moved aside and the defaults loaded"
        );
        Ok(SettingsLoad {
            notice: Some(SettingsNotice::Unreadable { path: bad }),
            ..Settings::default().into()
        })
    }

    pub fn save(&self, settings: &Settings) -> Result<(), AppError> {
        let document = serde_json::to_vec_pretty(&Stored {
            settings_version: SETTINGS_VERSION,
            settings,
        })
        .map_err(|_| redacted("settings_encode", "could not encode settings"))?;
        write_atomically(&self.paths.settings_file(), &document)
            .map_err(io_failed("settings_write", "could not save settings"))
    }

    /// The API key of `provider` in the credential store. Callers ask only when
    /// [`Settings::api_keys`] says it holds one.
    pub fn api_key(&self, provider: LlmProvider) -> Result<Option<ApiKey>, AppError> {
        self.credentials.read(provider).map_err(|_| {
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
        self.credentials.write(provider, key)
    }

    pub fn delete_api_key(&self, provider: LlmProvider) -> Result<(), CredentialStoreUnavailable> {
        self.credentials.delete(provider)
    }

    /// Imports the Python release's settings: its API keys move into the credential
    /// store, the copy in `settings.json.python` retains only refused keys, and the
    /// imported settings replace it.
    fn import_python(&self, document: &Value) -> Result<SettingsLoad, AppError> {
        let mut imported = PythonSettings::import(document);
        let mut refused = BTreeSet::new();
        for (provider, key) in &imported.api_keys {
            if self.store_api_key(*provider, key).is_ok() {
                imported.settings.api_keys.insert(*provider);
            } else {
                refused.insert(*provider);
            }
        }
        let copy = serde_json::to_vec_pretty(&imported.copy_with_refused_keys(document, &refused))
            .map_err(|_| redacted("settings_encode", "could not encode settings"))?;
        let kept = self.paths.python_settings_file();
        write_atomically(&kept, &copy).map_err(io_failed(
            "settings_import",
            "The earlier version's settings could not be imported",
        ))?;
        self.save(&imported.settings)?;
        tracing::info!(
            keys = imported.settings.api_keys.len(),
            "imported the earlier version's settings"
        );
        let notice =
            (!refused.is_empty()).then_some(SettingsNotice::ApiKeysNotImported { path: kept });
        Ok(SettingsLoad {
            settings: imported.settings,
            notice,
            enable_autostart: imported.enable_autostart,
        })
    }
}

fn redacted(code: &'static str, summary: &'static str) -> AppError {
    AppError::new(ErrorCategory::General, code, summary)
}

/// Only the kind is logged: the error text can name the settings directory.
fn io_failed(code: &'static str, summary: &'static str) -> impl FnOnce(io::Error) -> AppError {
    move |error| {
        tracing::warn!(code, kind = ?error.kind(), "settings could not be read or written");
        redacted(code, summary)
    }
}

#[cfg(test)]
mod tests;
