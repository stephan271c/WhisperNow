use whispernow_domain::ModelId;
use whispernow_domain::ipc::ModelSummary;

use super::ManagementState;
use crate::notice::{NoticeScope, Tone};

impl ManagementState {
    pub(crate) fn speech_models(&self) -> impl Iterator<Item = &ModelSummary> {
        self.models
            .iter()
            .flatten()
            .filter(|model| model.family.is_speech())
    }

    #[must_use]
    pub(crate) fn model_display_name(&self, model_id: &ModelId) -> String {
        self.models
            .iter()
            .flatten()
            .find(|model| model.id == *model_id)
            .map_or_else(
                || "the speech model".to_owned(),
                |model| model.display_name.clone(),
            )
    }

    pub(crate) fn finish_microphone_test(
        &mut self,
        result: whispernow_domain::ipc::MicrophoneTestResponse,
    ) {
        const SILENT_PEAK: f32 = 0.01;
        if result.peak < SILENT_PEAK {
            self.notify(
                Tone::Warning,
                NoticeScope::Home,
                "No sound reached the microphone. Check that it is connected and not muted.",
            );
        } else {
            self.notify(
                Tone::Success,
                NoticeScope::Home,
                "Your microphone is working",
            );
        }
        self.microphone_test = Some(result);
    }
}
