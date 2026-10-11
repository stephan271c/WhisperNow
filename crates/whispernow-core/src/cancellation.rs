use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use tokio::sync::Notify;

/// Cooperative cancellation shared by blocking workers, which check it, and async work,
/// which awaits it.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken(Arc<Cancellation>);

#[derive(Debug, Default)]
struct Cancellation {
    requested: AtomicBool,
    waiters: Notify,
}

impl CancellationToken {
    pub fn cancel(&self) {
        self.0.requested.store(true, Ordering::Release);
        self.0.waiters.notify_waiters();
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.requested.load(Ordering::Acquire)
    }

    pub async fn cancelled(&self) {
        // Registered before the check, so a cancel between the two still wakes it.
        let notified = self.0.waiters.notified();
        if self.is_cancelled() {
            return;
        }
        notified.await;
    }
}
