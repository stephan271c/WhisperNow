use crate::components::ui::Tone;

/// The part of the window a notice is about. A page that hosts the scope shows the notice
/// next to its controls; elsewhere it appears in the window's toast region.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NoticeScope {
    Global,
    Home,
    History,
    Setup,
    Settings,
    Models,
    Data,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Notice {
    /// Distinguishes a repeated message so a new notice restarts its dismissal timer.
    pub id: u64,
    pub tone: Tone,
    pub message: String,
    pub scope: NoticeScope,
}

impl Notice {
    pub const AUTO_DISMISS_MS: u32 = 3_000;

    /// Toasts and success notices leave on their own; other notices beside their controls wait
    /// to be dismissed.
    #[must_use]
    pub fn auto_dismisses(&self, inline: bool) -> bool {
        !inline || self.tone == Tone::Success
    }
}
