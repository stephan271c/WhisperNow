use whispernow_domain::AppError;
use whispernow_domain::ipc::{SetApiKeyRequest, SettingsView};
use whispernow_domain::{ApiKey, LlmProvider, Settings};

use super::UiActions;
use crate::client;

use crate::{SettingKey, UiOperation};
use dioxus::core::spawn_forever;
use dioxus::prelude::*;

impl UiActions {
    /// Shows one control's change at once and saves it; `false` means it was refused and the
    /// control keeps the shown value.
    pub(crate) fn change_setting(self, key: SettingKey, edit: impl FnOnce(&mut Settings)) -> bool {
        if !self.management().write().change_setting(key, edit) {
            return false;
        }
        self.send_settings_changes();
        true
    }

    fn send_settings_changes(self) {
        let Some(request) = self.management().write().next_settings_update() else {
            return;
        };
        spawn_forever(async move {
            let answer = client::update_settings(&request).await;
            self.management()
                .write()
                .finish_settings_update(answer.map_err(|error| error.summary.clone()));
            self.send_settings_changes();
        });
    }

    pub(crate) fn set_credential(self, provider: LlmProvider, key: ApiKey) {
        let request = SetApiKeyRequest { provider, key };
        self.run(
            UiOperation::CredentialMutation,
            async move { client::set_api_key(&request).await },
            Self::finish_credential_mutation,
        );
    }

    pub(crate) fn clear_credential(self, provider: LlmProvider) {
        self.run(
            UiOperation::CredentialMutation,
            async move { client::clear_api_key(&provider).await },
            Self::finish_credential_mutation,
        );
    }

    fn finish_credential_mutation(self, answer: Result<SettingsView, AppError>) {
        match answer {
            Ok(settings) => self
                .management()
                .write()
                .finish_credential_mutation(settings),
            Err(error) => self.report(UiOperation::CredentialMutation, &error),
        }
    }
}
