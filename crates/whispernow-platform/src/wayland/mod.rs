//! Wayland desktop integration.
//!
//! This module is intentionally disjoint from the X11 adapters. Nothing in
//! this path opens an X11 connection, invokes a command-line clipboard helper,
//! or reads a raw input device.

mod accessibility;
mod clipboard;
mod portal;
mod shortcut_keymap;
mod token;

pub use accessibility::active_paste_keys;
pub use portal::{WaylandOutput, WaylandPortalError};
pub use shortcut_keymap::shortcut_input_chord;
pub use token::RemoteDesktopRestoreTokenStore;
