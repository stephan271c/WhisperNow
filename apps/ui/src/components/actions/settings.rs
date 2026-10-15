use whispernow_domain::ipc::SetApiKeyRequest;
use whispernow_domain::{ApiKey, LlmProvider, Settings};

use crate::UiContext;
use crate::client;

use crate::{NoticeScope, SettingKey, UiOperation};
use dioxus::core::spawn_forever;
use dioxus::prelude::*;

impl UiContext {
    pub(crate) fn change_setting(mut self, key: SettingKey, edit: impl FnOnce(&mut Settings)) {
        let Some(request) = self.management.write().begin_settings_update(key, edit) else {
            return;
        };
        spawn_forever(async move {
            let answer = client::invoke("update_settings", Some(&request)).await;
            self.management
                .write()
                .finish_settings_update(answer.map_err(|error| error.summary.clone()));
        });
    }

    pub(crate) fn set_start_on_login(mut self, enabled: bool) {
        if self
            .management
            .write()
            .begin_settings_update(SettingKey::StartOnLogin, |_| {})
            .is_none()
        {
            return;
        }
        spawn_forever(async move {
            let answer = client::invoke("set_start_on_login", Some(&enabled)).await;
            self.management
                .write()
                .finish_settings_update(answer.map_err(|error| error.summary));
        });
    }

    pub(crate) fn set_credential(self, provider: LlmProvider, key: ApiKey) {
        let request = SetApiKeyRequest { provider, key };
        self.run(
            UiOperation::CredentialMutation,
            async move { client::invoke("set_api_key", Some(&request)).await },
            |mut actions, answer| match answer {
                Ok(settings) => actions.management.write().install_settings(settings),
                Err(error) => actions.report(NoticeScope::Settings, &error),
            },
        );
    }

    pub(crate) fn clear_credential(self, provider: LlmProvider) {
        self.run(
            UiOperation::CredentialMutation,
            async move { client::invoke("clear_api_key", Some(&provider)).await },
            |mut actions, answer| match answer {
                Ok(settings) => actions.management.write().install_settings(settings),
                Err(error) => actions.report(NoticeScope::Settings, &error),
            },
        );
    }
}
