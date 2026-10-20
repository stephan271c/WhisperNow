use whispernow_domain::ModelId;

use crate::UiContext;
use crate::client;
use crate::components::ui::Tone;
use crate::{NoticeScope, UiOperation};
use dioxus::core::spawn_forever;
use dioxus::prelude::*;

impl UiContext {
    pub(crate) fn download_model(self, model_id: ModelId) {
        let operation = UiOperation::DownloadModel;
        let name = self.management.peek().model_display_name(&model_id);
        self.run(
            operation,
            client::invoke("download_model", model_id),
            move |mut actions, answer| match answer {
                Ok(()) => actions.management.write().notify(
                    Tone::Info,
                    NoticeScope::Models,
                    format!("Downloading {name}"),
                ),
                Err(error) => actions.report(NoticeScope::Models, &error),
            },
        );
    }

    pub(crate) fn cancel_download(self, model_id: ModelId) {
        let operation = UiOperation::CancelDownload;
        self.run(
            operation,
            client::invoke::<_, ()>("cancel_download", model_id),
            move |actions, answer| {
                if let Err(error) = answer {
                    actions.report(NoticeScope::Models, &error);
                }
            },
        );
    }

    /// Makes `model_id` the model in use, or loads it again after it failed to load.
    pub(crate) fn switch_model(self, model_id: ModelId) {
        let operation = UiOperation::ModelSwitch;
        self.run(
            operation,
            client::invoke::<_, ()>("select_model", model_id),
            move |actions, answer| {
                if let Err(error) = answer {
                    actions.report(NoticeScope::Models, &error);
                }
            },
        );
    }

    pub(crate) fn delete_model(self, model_id: ModelId) {
        let operation = UiOperation::ModelDelete;
        self.run(
            operation,
            client::invoke("delete_model", model_id),
            move |actions, answer| match answer {
                Ok(()) => actions.reload_models(),
                Err(error) => actions.report(NoticeScope::Models, &error),
            },
        );
    }

    pub(super) fn reload_models(mut self) {
        spawn_forever(async move {
            match client::call("list_models").await {
                Ok(models) => {
                    let mut management = self.management.write();
                    management.models = models;
                    management.models_loaded = true;
                }
                Err(error) => self.load_failed("Speech models", &error),
            }
        });
    }
}
