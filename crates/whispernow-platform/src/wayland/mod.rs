//! Native Wayland desktop integration through XDG portals.
//!
//! This module is intentionally disjoint from the X11 adapters. Nothing in
//! this path opens an X11 connection, invokes a command-line clipboard helper,
//! or reads a raw input device.

mod accessibility;
mod clipboard;
mod portal;
mod token;

pub use accessibility::active_paste_keys;
pub use portal::{WaylandOutput, WaylandPasteOutcome, WaylandPortalError};
pub use token::RemoteDesktopRestoreTokenStore;
