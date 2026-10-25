//! Wayland paste and shortcut support through desktop portals, AT-SPI and the
//! Wayland protocol. Nothing here opens an X11 connection, runs a clipboard
//! helper command or reads a raw input device.

mod accessibility;
mod clipboard;
mod portal;
mod shortcut_keymap;
mod token;

use thiserror::Error;

pub use accessibility::active_paste_keys;
pub use portal::PortalOutput;
pub use shortcut_keymap::shortcut_input_chord;
pub use token::RemoteDesktopRestoreTokenStore;

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum PortalOutputError {
    #[error("desktop portal operation failed")]
    Backend,
    #[error("the paste was cancelled")]
    Cancelled,
    #[error("clipboard snapshot exceeds safe limits")]
    ClipboardSnapshotTooLarge,
    #[error("clipboard portal is unavailable")]
    ClipboardUnavailable,
    #[error("desktop portal disconnected")]
    Disconnected,
    #[error("remote keyboard capability was not granted")]
    KeyboardUnavailable,
    #[error("desktop portal permission was denied")]
    PermissionDenied,
    #[error("required desktop portal is unavailable")]
    PortalUnavailable,
    #[error("remote desktop restore token could not be stored safely")]
    TokenStore,
}
