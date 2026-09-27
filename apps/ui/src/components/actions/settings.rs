//! Settings edits, API keys and prompt tests.

use whispernow_api::{
    ApiKey, ClearApiKeyRequest, LlmProvider, SetApiKeyRequest, Settings, SettingsView,
};

use super::UiActions;
use crate::client::{self, ClientError};
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

    /// Sends the changes made since the last save, unless a save is already with the desktop;
    /// changes made meanwhile go out once it answers.
    fn send_settings_changes(self) {
        let Some(request) = self.management().write().next_settings_update() else {
            return;
        };
        spawn_forever(async move {
            let answer = client::update_settings(&request).await;
            self.management()
                .write()
                .finish_settings_update(answer.map_err(|error| error.summary()));
            self.send_settings_changes();
        });
    }

    /// Runs the saved prompt `prompt_id` on `sample_text`.
    /// Sample text that cannot be sent is explained beside the test's result instead.
    pub(crate) fn test_prompt(self, prompt_id: &str, sample_text: &str) {
        let request = self
            .management()
            .write()
            .begin_prompt_test(prompt_id, sample_text);
        let Some(request) = request else {
            return;
        };
        let sent = request.clone();
        self.run(
            UiOperation::TestPrompt,
            async move { client::test_prompt(&sent).await },
            move |actions, answer| {
                actions
                    .management()
                    .write()
                    .finish_prompt_test(&request, answer.map_err(|error| error.summary()));
            },
        );
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
        let request = ClearApiKeyRequest { provider };
        self.run(
            UiOperation::CredentialMutation,
            async move { client::clear_api_key(&request).await },
            Self::finish_credential_mutation,
        );
    }

    fn finish_credential_mutation(self, answer: Result<SettingsView, ClientError>) {
        match answer {
            Ok(settings) => self
                .management()
                .write()
                .finish_credential_mutation(settings),
            Err(error) => self.report(UiOperation::CredentialMutation, &error),
        }
    }
}
