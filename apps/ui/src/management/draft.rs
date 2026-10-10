use whispernow_domain::{ModelId, RecordingMode, Shortcut};

use super::ManagementState;

#[derive(Clone, Debug, PartialEq)]
pub enum SetupEdit {
    Model(ModelId),
    InputDevice(Option<String>),
    RecordingMode(RecordingMode),
    Shortcut(Shortcut),
}

impl ManagementState {
    pub fn edit_setup(&mut self, edit: SetupEdit) {
        let Some(setup) = &mut self.setup else {
            return;
        };
        let checkpoint = &mut setup.checkpoint;
        match edit {
            SetupEdit::Model(model_id) => checkpoint.model_id = model_id,
            SetupEdit::InputDevice(device) => checkpoint.input_device = device,
            SetupEdit::RecordingMode(mode) => checkpoint.recording_mode = mode,
            SetupEdit::Shortcut(shortcut) => checkpoint.recording_shortcut = shortcut,
        }
    }
}
