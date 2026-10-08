//! Each primitive has one section in `public/ui.css`; colour, spacing, type, radius and
//! elevation values come from the tokens in `public/tokens.css`.

mod buttons;
mod feedback;
mod fields;
pub(crate) mod focus;
mod icon;
mod keys;
mod layout;

pub(crate) use buttons::{Button, ButtonSize, ButtonVariant, ConfirmButton, IconButton};
pub(crate) use feedback::{Banner, ProgressBar, Toast, ToastRegion, Tone};
pub(crate) use fields::{DraftText, Field, Segmented, Select, SettingRow, TextInput, Toggle};
pub(crate) use icon::{BrandMark, Icon, IconName};
pub(crate) use keys::{Kbd, ShortcutLabel, shortcut_parts};
pub(crate) use layout::{Card, EmptyState, Section};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
