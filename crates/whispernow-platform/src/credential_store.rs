//! Operating-system credential store for LLM provider API keys.

#[cfg(any(test, feature = "test-support"))]
use std::{
    collections::BTreeMap,
    sync::{Mutex, PoisonError},
};

#[cfg(target_os = "macos")]
use apple_native_keyring_store::keychain::Store;
#[cfg(target_os = "linux")]
use dbus_secret_service_keyring_store::Store;
use keyring_core::{Entry, api::CredentialStoreApi as _};
use thiserror::Error;
use whispernow_domain::{ApiKey, LlmProvider};
#[cfg(target_os = "windows")]
use windows_native_keyring_store::Store;

/// Failure to reach or use the credential store, such as a
/// headless Linux session without a Secret Service provider or a locked keychain.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[error("the OS credential store is unavailable")]
pub struct CredentialStoreUnavailable;

pub trait CredentialStore: Send + Sync {
    /// Reads a stored key; `Ok(None)` means the store has no entry.
    fn read(&self, provider: LlmProvider) -> Result<Option<ApiKey>, CredentialStoreUnavailable>;

    fn write(&self, provider: LlmProvider, key: &ApiKey) -> Result<(), CredentialStoreUnavailable>;

    /// Removes a stored key; removing an absent entry succeeds.
    fn delete(&self, provider: LlmProvider) -> Result<(), CredentialStoreUnavailable>;
}

/// The platform keychain: macOS Keychain, Windows Credential Manager, or the
/// Linux Secret Service over D-Bus.
pub(crate) struct OsCredentialStore;

const SERVICE: &str = "WhisperNow";

impl OsCredentialStore {
    fn entry(provider: LlmProvider) -> Result<Entry, CredentialStoreUnavailable> {
        // Saved keys are found again by this account name; changing it loses them.
        let account = format!("llm-provider:{}", provider.as_str());
        // A store per call opens a fresh Secret Service session, so a keyring daemon that
        // started or restarted after launch is still reached.
        let store = Store::new().map_err(|error| unavailable("store", &error))?;
        store
            .build(SERVICE, &account, None)
            .map_err(|error| unavailable("entry", &error))
    }
}

impl CredentialStore for OsCredentialStore {
    fn read(&self, provider: LlmProvider) -> Result<Option<ApiKey>, CredentialStoreUnavailable> {
        match Self::entry(provider)?.get_password() {
            // A key that is no longer one line reads as no key.
            Ok(secret) => Ok(ApiKey::new(secret).ok()),
            Err(keyring_core::Error::NoEntry) => Ok(None),
            Err(error) => Err(unavailable("read", &error)),
        }
    }

    fn write(&self, provider: LlmProvider, key: &ApiKey) -> Result<(), CredentialStoreUnavailable> {
        Self::entry(provider)?
            .set_password(key.expose_secret())
            .map_err(|error| unavailable("write", &error))
    }

    fn delete(&self, provider: LlmProvider) -> Result<(), CredentialStoreUnavailable> {
        match Self::entry(provider)?.delete_credential() {
            Ok(()) | Err(keyring_core::Error::NoEntry) => Ok(()),
            Err(error) => Err(unavailable("delete", &error)),
        }
    }
}

/// Platform errors can name items or collections, and `BadEncoding` holds the secret's
/// bytes, so only the operation and the variant are logged.
fn unavailable(operation: &'static str, error: &keyring_core::Error) -> CredentialStoreUnavailable {
    let variant = match error {
        keyring_core::Error::PlatformFailure(_) => "PlatformFailure",
        keyring_core::Error::NoStorageAccess(_) => "NoStorageAccess",
        keyring_core::Error::NoEntry => "NoEntry",
        keyring_core::Error::BadEncoding(_) => "BadEncoding",
        keyring_core::Error::TooLong(..) => "TooLong",
        keyring_core::Error::Invalid(..) => "Invalid",
        keyring_core::Error::Ambiguous(_) => "Ambiguous",
        keyring_core::Error::NoDefaultStore => "NoDefaultStore",
        keyring_core::Error::BadDataFormat(..) => "BadDataFormat",
        keyring_core::Error::BadStoreFormat(..) => "BadStoreFormat",
        keyring_core::Error::NotSupportedByStore(..) => "NotSupportedByStore",
        _ => "Other",
    };
    tracing::warn!(
        code = "credential_store_unavailable",
        operation,
        variant,
        "the OS credential store could not be used"
    );
    CredentialStoreUnavailable
}

#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Default)]
pub struct MemoryCredentialStore {
    entries: Mutex<BTreeMap<LlmProvider, ApiKey>>,
    unavailable: std::sync::atomic::AtomicBool,
}

#[cfg(any(test, feature = "test-support"))]
impl MemoryCredentialStore {
    pub fn set_available(&self, available: bool) {
        self.unavailable
            .store(!available, std::sync::atomic::Ordering::SeqCst);
    }

    fn entries(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, BTreeMap<LlmProvider, ApiKey>>, CredentialStoreUnavailable>
    {
        if self.unavailable.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(CredentialStoreUnavailable);
        }
        Ok(self.entries.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

#[cfg(any(test, feature = "test-support"))]
impl CredentialStore for MemoryCredentialStore {
    fn read(&self, provider: LlmProvider) -> Result<Option<ApiKey>, CredentialStoreUnavailable> {
        Ok(self.entries()?.get(&provider).cloned())
    }

    fn write(&self, provider: LlmProvider, key: &ApiKey) -> Result<(), CredentialStoreUnavailable> {
        self.entries()?.insert(provider, key.clone());
        Ok(())
    }

    fn delete(&self, provider: LlmProvider) -> Result<(), CredentialStoreUnavailable> {
        self.entries()?.remove(&provider);
        Ok(())
    }
}
