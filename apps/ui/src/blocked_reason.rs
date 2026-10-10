#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockedReason {
    /// No state snapshot has arrived from the desktop yet.
    Connecting,
    /// A dictation is recording or being processed.
    Dictating,
    Paused,
    UpdateInstalling,
    /// A change this window sent is still in flight.
    CommandPending,
}

impl BlockedReason {
    #[must_use]
    pub const fn message(self) -> &'static str {
        match self {
            Self::Connecting => "WhisperNow is still connecting",
            Self::Dictating => "Wait for the current dictation to finish",
            Self::Paused => "Resume WhisperNow first",
            Self::UpdateInstalling => "Wait for the current update to finish installing",
            Self::CommandPending => "Wait for the current change to finish",
        }
    }
}
