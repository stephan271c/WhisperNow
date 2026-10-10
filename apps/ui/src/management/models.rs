use whispernow_domain::ModelFamily;
use whispernow_domain::ipc::ModelSummary;

use super::ManagementState;
use crate::components::ui::Tone;

impl ManagementState {
    pub fn speech_models(&self) -> impl Iterator<Item = &ModelSummary> {
        self.models.iter().filter(|model| model.family.is_speech())
    }

    #[must_use]
    pub fn cuda_pack(&self) -> Option<&ModelSummary> {
        self.models
            .iter()
            .find(|model| model.family == ModelFamily::CudaPack)
    }

    pub fn cleanup_models(&self) -> impl Iterator<Item = &ModelSummary> {
        self.models
            .iter()
            .filter(|model| model.family == ModelFamily::Cleanup)
    }

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

    pub fn finish_microphone_test(
        &mut self,
        result: whispernow_domain::ipc::MicrophoneTestResponse,
    ) {
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

    pub fn install_models(&mut self, models: Vec<ModelSummary>) {
        self.models = models;
        self.models_loaded = true;
    }
}
