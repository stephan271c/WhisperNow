//! Wayland paste and shortcut support through desktop portals, AT-SPI and the
//! Wayland protocol. Nothing here opens an X11 connection, runs a clipboard
//! helper command or reads a raw input device.

mod accessibility;
mod clipboard;
mod portal;
mod shortcut_keymap;
mod token;

pub use accessibility::active_paste_keys;
pub use portal::{WaylandOutput, WaylandPortalError};
pub use shortcut_keymap::shortcut_input_chord;
pub use token::RemoteDesktopRestoreTokenStore;
