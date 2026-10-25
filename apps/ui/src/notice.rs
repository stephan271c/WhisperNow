/// Severity of a banner or toast. Warnings and errors are announced assertively.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Tone {
    Info,
    Success,
    Warning,
    Error,
}

/// The part of the window a notice is about. A page that hosts the scope shows the notice
/// next to its controls; elsewhere it appears in the window's toast region.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NoticeScope {
    Global,
    Home,
    History,
    Setup,
    Settings,
    Models,
    Data,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Notice {
    /// Distinguishes a repeated message so a new notice restarts its dismissal timer.
    pub(crate) id: u64,
    pub(crate) tone: Tone,
    pub(crate) message: String,
    pub(crate) scope: NoticeScope,
}

impl Notice {
    /// How long every notice, and a failure without a button, stays shown.
    pub(crate) const AUTO_DISMISS_MS: u32 = 3_000;
}
