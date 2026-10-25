//! The focused window a paste is aimed at, captured immediately before a delivery; nothing is
//! retained from the start of the recording.

use crate::{PasteKeys, paste_delivery::NativePasteError};

/// Opaque identity held only for the duration of one output operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NativeDestination {
    window: u64,
    process_id: u32,
    focused_element: u64,
    keys: PasteKeys,
}

impl NativeDestination {
    pub(crate) const fn new(
        window: u64,
        process_id: u32,
        focused_element: u64,
        keys: PasteKeys,
    ) -> Self {
        Self {
            window,
            process_id,
            focused_element,
            keys,
        }
    }

    #[cfg(target_os = "windows")]
    pub(crate) const fn process_id(self) -> u32 {
        self.process_id
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(crate) const fn keys(self) -> PasteKeys {
        self.keys
    }
}

/// `found`, unless nothing has focus or this application, `own_process_id`,
/// has it: a transcript is never pasted into the app itself.
pub(crate) fn foreign(
    found: Option<NativeDestination>,
    own_process_id: u32,
) -> Result<NativeDestination, NativePasteError> {
    found
        .filter(|destination| destination.process_id != own_process_id)
        .ok_or(NativePasteError::Unavailable)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn destination(process_id: u32) -> NativeDestination {
        NativeDestination::new(10, process_id, 30, PasteKeys::ControlV)
    }

    #[test]
    fn another_application_with_focus_is_the_target() {
        assert_eq!(foreign(Some(destination(7)), 8), Ok(destination(7)));
    }

    #[test]
    fn this_application_or_no_focus_is_not_a_target() {
        assert_eq!(
            foreign(Some(destination(8)), 8),
            Err(NativePasteError::Unavailable)
        );
        assert_eq!(foreign(None, 8), Err(NativePasteError::Unavailable));
    }
}
