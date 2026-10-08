use crate::Route;
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
    Updates,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NoticeAction {
    pub label: &'static str,
    pub route: Route,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Notice {
    /// Distinguishes a repeated message so a new notice restarts its dismissal timer.
    pub id: u64,
    pub tone: Tone,
    pub message: String,
    pub action: Option<NoticeAction>,
    pub scope: NoticeScope,
}

impl Notice {
    pub const AUTO_DISMISS_MS: u32 = 4_000;

    /// Success notices confirm and leave; the others wait to be read and dismissed.
    #[must_use]
    pub fn auto_dismisses(&self) -> bool {
        self.tone == Tone::Success
    }
}
