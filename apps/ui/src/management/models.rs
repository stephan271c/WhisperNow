//! Model catalogue, selection, deletion, and download projection.

use whispernow_api::{AppStateDto, ListModelsResponse, ModelFamily, ModelId};

use super::ManagementState;
use crate::components::ui::Tone;

impl ManagementState {
    /// The models that transcribe speech, in catalogue order.
    pub fn speech_models(&self) -> impl Iterator<Item = &whispernow_api::ModelDto> {
        self.models.iter().filter(|model| model.family.is_speech())
    }

    /// The CUDA pack, where this computer's platform has one.
    #[must_use]
    pub fn cuda_pack(&self) -> Option<&whispernow_api::ModelDto> {
        self.models
            .iter()
            .find(|model| model.family == ModelFamily::CudaPack)
    }

    /// The local cleanup models, in catalogue order, once the catalogue has loaded.
    pub fn cleanup_models(&self) -> impl Iterator<Item = &whispernow_api::ModelDto> {
        self.models
            .iter()
            .filter(|model| model.family == ModelFamily::Cleanup)
    }

    /// The model's human name; the id only before the catalogue loads.
    #[must_use]
    pub fn model_display_name(&self, model_id: &str) -> String {
        self.models
            .iter()
            .find(|model| model.id.as_str() == model_id)
            .map_or_else(
                || "the speech model".to_owned(),
                |model| model.display_name.clone(),
            )
    }

    /// Installs a finished microphone test and explains a silent one.
    pub fn finish_microphone_test(&mut self, result: whispernow_api::MicrophoneTestResponse) {
        const SILENT_PEAK: f32 = 0.01;
        if result.peak < SILENT_PEAK {
            self.notify(
                Tone::Warning,
                crate::NoticeScope::Home,
                "No sound reached the microphone. Check that it is connected and not muted.",
            );
        } else {
            self.notify(
                Tone::Success,
                crate::NoticeScope::Home,
                "Your microphone is working",
            );
        }
        self.microphone_test = Some(result);
    }

    pub fn install_models(&mut self, models: ListModelsResponse) {
        self.models = models.models;
        self.models_loaded = true;
    }

    /// Installs a process snapshot; returns whether a download has ended since the last one,
    /// so the models must be read again.
    pub fn install_snapshot(&mut self, snapshot: &AppStateDto) -> bool {
        self.observe_selected_model(&snapshot.selected_model_id);
        let ended = self.downloads.iter().any(|download| {
            !snapshot
                .active_downloads
                .iter()
                .any(|active| active.model_id == download.model_id)
        });
        self.downloads.clone_from(&snapshot.active_downloads);
        self.observe_setup_test(snapshot);
        ended
    }

    /// The download of `model_id`, while one runs.
    #[must_use]
    pub fn download(&self, model_id: &ModelId) -> Option<&whispernow_api::ActiveDownloadDto> {
        self.downloads
            .iter()
            .find(|download| download.model_id == *model_id)
    }

    fn observe_selected_model(&mut self, model_id: &ModelId) {
        if let Some(settings) = &mut self.settings {
            settings.settings.model_id.clone_from(model_id);
        }
    }

    pub fn finish_model_switch(&mut self, model_id: &ModelId) {
        self.observe_selected_model(model_id);
    }
}
