//! One serialized clipboard paste-and-restore transaction.
//!
//! The platform supplies plain clipboard reads and writes plus the paste
//! keystroke; this module owns the ordering, the compare-and-restore rule, and
//! the one-at-a-time queue so the rules are testable without a desktop.

mod paste;

pub use paste::{KeystrokeOutcome, PasteDelays, PasteOutcome, PastePlatform, PasteQueue};
