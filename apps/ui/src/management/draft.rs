use whispernow_api::{ModelId, RecordingMode, Shortcut};

use super::ManagementState;

#[derive(Clone, Debug, PartialEq)]
pub enum SetupEdit {
    Model(ModelId),
    InputDevice(Option<String>),
    HoldEnabled(bool),
    /// Turns toggle-to-record off; it is on once a shortcut is captured for it.
    ToggleOff,
    Shortcut(RecordingMode, Shortcut),
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
            SetupEdit::HoldEnabled(enabled) => checkpoint.hold_shortcut_enabled = enabled,
            SetupEdit::ToggleOff => checkpoint.toggle_shortcut = None,
            SetupEdit::Shortcut(RecordingMode::Hold, shortcut) => {
                checkpoint.hold_shortcut = shortcut;
            }
            SetupEdit::Shortcut(RecordingMode::Toggle, shortcut) => {
                checkpoint.toggle_shortcut = Some(shortcut);
            }
        }
    }
}
