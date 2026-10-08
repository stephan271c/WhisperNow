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

#[cfg(test)]
mod tests {
    use super::CancellationToken;

    #[test]
    fn cancellation_is_clone_shared_and_monotonic() {
        let first = CancellationToken::default();
        let second = first.clone();
        assert!(!second.is_cancelled());
        first.cancel();
        assert!(second.is_cancelled());
    }

    #[tokio::test]
    async fn waiting_for_cancellation_ends_when_it_is_requested() {
        let token = CancellationToken::default();
        let waiting = tokio::spawn({
            let token = token.clone();
            async move { token.cancelled().await }
        });
        tokio::task::yield_now().await;
        token.cancel();
        waiting.await.expect("the waiter ends");
        token.cancelled().await;
    }
}
