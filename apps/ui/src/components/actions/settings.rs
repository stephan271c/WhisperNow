use whispernow_domain::ipc::SetApiKeyRequest;
use whispernow_domain::{ApiKey, LlmProvider, Settings};

use crate::UiContext;
use crate::client;

use crate::{SettingKey, UiOperation};
use dioxus::core::spawn_forever;
use dioxus::prelude::*;

impl UiContext {
    pub(crate) fn change_setting(mut self, key: SettingKey, edit: impl FnOnce(&mut Settings)) {
        self.management.write().change_setting(key, edit);
        self.send_settings_changes();
    }

    fn send_settings_changes(mut self) {
        let Some(request) = self.management.write().next_settings_update() else {
            return;
        };
        spawn_forever(async move {
            let answer = client::invoke("update_settings", Some(&request)).await;
            self.management
                .write()
                .finish_settings_update(answer.map_err(|error| error.summary.clone()));
            self.send_settings_changes();
        });
    }

    pub(crate) fn set_credential(self, provider: LlmProvider, key: ApiKey) {
        let request = SetApiKeyRequest { provider, key };
        self.run(
            UiOperation::CredentialMutation,
            async move { client::invoke("set_api_key", Some(&request)).await },
            |mut actions, answer| match answer {
                Ok(settings) => actions.management.write().settings = Some(settings),
                Err(error) => actions.report(UiOperation::CredentialMutation, &error),
            },
        );
    }

    pub(crate) fn clear_credential(self, provider: LlmProvider) {
        self.run(
            UiOperation::CredentialMutation,
            async move { client::invoke("clear_api_key", Some(&provider)).await },
            |mut actions, answer| match answer {
                Ok(settings) => actions.management.write().settings = Some(settings),
                Err(error) => actions.report(UiOperation::CredentialMutation, &error),
            },
        );
    }
}
