use whispernow_domain::ModelId;

use super::UiActions;
use crate::client;
use crate::components::ui::Tone;
use crate::{NoticeScope, UiOperation};
use dioxus::core::spawn_forever;
use dioxus::prelude::*;

impl UiActions {
    pub(crate) fn download_model(self, model_id: ModelId) {
        let operation = UiOperation::DownloadModel;
        let name = self
            .management()
            .peek()
            .model_display_name(model_id.as_str());
        self.run(
            operation,
            async move { client::download_model(&model_id).await },
            move |actions, answer| match answer {
                Ok(()) => actions.management().write().notify(
                    Tone::Info,
                    NoticeScope::Models,
                    format!("Downloading {name}"),
                ),
                Err(error) => actions.report(operation, &error),
            },
        );
    }

    pub(crate) fn cancel_download(self, model_id: ModelId) {
        let operation = UiOperation::CancelDownload;
        self.run(
            operation,
            async move { client::cancel_download(&model_id).await },
            move |actions, answer| {
                if let Err(error) = answer {
                    actions.report(operation, &error);
                }
            },
        );
    }

    /// Makes `model_id` the model in use, or loads it again after it failed to load.
    pub(crate) fn switch_model(self, model_id: ModelId) {
        let operation = UiOperation::ModelSwitch;
        let sent = model_id.clone();
        self.run(
            operation,
            async move { client::select_model(&sent).await },
            move |actions, answer| match answer {
                Ok(()) => actions.management().write().finish_model_switch(&model_id),
                Err(error) => actions.report(operation, &error),
            },
        );
    }

    pub(crate) fn delete_model(self, model_id: ModelId) {
        let operation = UiOperation::ModelDelete;
        self.run(
            operation,
            async move { client::delete_model(&model_id).await },
            move |actions, answer| match answer {
                Ok(()) => actions.reload_models(),
                Err(error) => actions.report(operation, &error),
            },
        );
    }

    pub(super) fn reload_models(self) {
        spawn_forever(async move {
            match client::list_models().await {
                Ok(models) => self.management().write().install_models(models),
                Err(error) => self.load_failed("Speech models", &error),
            }
        });
    }
}
