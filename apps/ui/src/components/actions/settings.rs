use serde::Serialize;
use whispernow_domain::ipc::{SetApiKeyRequest, SettingsView};
use whispernow_domain::{ApiKey, AppError, LlmProvider, SettingChange, VocabularyReplacement};

use crate::UiContext;
use crate::client;

use crate::management::SettingKey;
use crate::notice::NoticeScope;
use crate::operations::UiOperation;
use dioxus::core::spawn_forever;
use dioxus::prelude::*;

impl UiContext {
    pub(crate) fn change_setting(mut self, change: SettingChange) {
        let shown = change.clone();
        let begun = self
            .management
            .write()
            .begin_settings_update(SettingKey::from(&change), |settings| shown.apply(settings));
        if begun {
            self.save_setting("update_settings", change);
        }
    }

    pub(crate) fn change_vocabulary(self, edit: impl FnOnce(&mut Vec<VocabularyReplacement>)) {
        let shown = self
            .management
            .read()
            .shown_settings()
            .map(|settings| settings.vocabulary_replacements.clone());
        if let Some(mut rules) = shown {
            edit(&mut rules);
            self.change_setting(SettingChange::Vocabulary(rules));
        }
    }

    pub(crate) fn set_start_on_login(mut self, enabled: bool) {
        let begun = self
            .management
            .write()
            .begin_settings_update(SettingKey::StartOnLogin, |_| {});
        if begun {
            self.save_setting("set_start_on_login", enabled);
        }
    }

    fn save_setting(mut self, command: &'static str, request: impl Serialize + 'static) {
        spawn_forever(async move {
            let answer = client::invoke(command, request).await;
            self.management.write().finish_settings_update(answer);
        });
    }

    pub(crate) fn set_credential(self, provider: LlmProvider, key: ApiKey) {
        self.run(
            UiOperation::CredentialMutation,
            client::invoke("set_api_key", SetApiKeyRequest { provider, key }),
            Self::finish_credential_change,
        );
    }

    pub(crate) fn clear_credential(self, provider: LlmProvider) {
        self.run(
            UiOperation::CredentialMutation,
            client::invoke("clear_api_key", provider),
            Self::finish_credential_change,
        );
    }

    fn finish_credential_change(mut self, answer: Result<SettingsView, AppError>) {
        match answer {
            Ok(settings) => self.management.write().install_settings(settings),
            Err(error) => self.report(NoticeScope::Settings, &error),
        }
    }
}
