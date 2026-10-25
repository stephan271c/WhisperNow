use std::sync::atomic::{AtomicU8, Ordering};

#[repr(u8)]
enum Stage {
    Running,
    Begun,
    Complete,
}

/// Shutdown advances through [`Stage`]s in order, so it cannot complete without having begun.
pub(super) struct Shutdown(AtomicU8);

impl Shutdown {
    pub(super) const fn new() -> Self {
        Self(AtomicU8::new(Stage::Running as u8))
    }

    /// Returns false when shutdown had already begun.
    pub(super) fn begin(&self) -> bool {
        self.0
            .compare_exchange(
                Stage::Running as u8,
                Stage::Begun as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub(super) fn complete(&self) {
        let advanced = self.0.compare_exchange(
            Stage::Begun as u8,
            Stage::Complete as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        debug_assert!(advanced.is_ok(), "shutdown completes once, after it begins");
    }

    pub(super) fn has_begun(&self) -> bool {
        self.0.load(Ordering::Acquire) >= Stage::Begun as u8
    }

    pub(super) fn is_complete(&self) -> bool {
        self.0.load(Ordering::Acquire) == Stage::Complete as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shutdown_begins_once_and_completes_after_beginning() {
        let shutdown = Shutdown::new();
        assert!(!shutdown.has_begun());
        assert!(!shutdown.is_complete());

        assert!(shutdown.begin());
        assert!(shutdown.has_begun());
        assert!(!shutdown.is_complete());
        assert!(!shutdown.begin());

        shutdown.complete();
        assert!(shutdown.has_begun());
        assert!(shutdown.is_complete());
        assert!(!shutdown.begin());
    }
}
