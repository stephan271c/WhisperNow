use std::{
    fmt,
    sync::{Mutex, PoisonError},
    thread,
    time::{Duration, Instant},
};

use crate::CancellationToken;

/// `propagation` lets the clipboard change reach the focused application
/// before the paste keystroke. `restore` is how long the target gets to read
/// the transcript before the previous clipboard is put back; too short a wait
/// makes slow targets paste the restored value instead.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PasteDelays {
    propagation: Duration,
    restore: Duration,
}

impl PasteDelays {
    #[must_use]
    pub const fn restore(self) -> Duration {
        self.restore
    }
}

impl Default for PasteDelays {
    fn default() -> Self {
        Self {
            propagation: Duration::from_millis(80),
            restore: Duration::from_millis(400),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KeystrokeOutcome<E> {
    Sent,
    NotSent(E),
    /// Some input may have reached the target before the failure.
    Uncertain(E),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PasteOutcome<E> {
    PastedAndRestored,
    /// The transcript was pasted but stays on the clipboard: the previous
    /// clipboard could not be read or held nothing capturable, or putting it
    /// back failed.
    PastedWithoutRestore,
    /// The transcript was pasted, then something newer replaced it, so the
    /// newer clipboard was kept.
    RestoreSkippedClipboardChanged,
    /// Cancelled before the paste keystroke was sent.
    Cancelled,
    Failed(E),
}

pub trait PastePlatform {
    /// The captured previous clipboard: text, HTML with its plain-text
    /// alternative, or an image.
    type Snapshot;
    type Error: fmt::Debug;

    /// Captures the clipboard, or `Ok(None)` when it holds nothing capturable.
    /// `None` leaves the transcript on the clipboard afterwards rather than
    /// clearing it: a clipboard of only formats the platform cannot capture
    /// (file lists, RTF, private formats) must not be emptied, and platforms
    /// cannot reliably tell that apart from an empty clipboard.
    fn snapshot_clipboard(&mut self) -> Result<Option<Self::Snapshot>, Self::Error>;

    /// Writes the transcript, excluded from clipboard history where the
    /// platform supports it.
    fn write_transcript(&mut self, text: &str) -> Result<(), Self::Error>;

    /// Reports whether the clipboard's text is exactly `text`.
    fn clipboard_holds(&mut self, text: &str) -> Result<bool, Self::Error>;

    fn restore_clipboard(&mut self, snapshot: Self::Snapshot) -> Result<(), Self::Error>;

    /// Returns `false` if cancellation ended the wait.
    fn wait(&mut self, duration: Duration, cancellation: &CancellationToken) -> bool {
        let deadline = Instant::now() + duration;
        while !cancellation.is_cancelled() {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return true;
            }
            thread::sleep(remaining.min(Duration::from_millis(5)));
        }
        false
    }

    fn send_paste_shortcut(&mut self) -> KeystrokeOutcome<Self::Error>;
}

/// Runs one paste transaction at a time. A second paste waits until the first
/// has restored the clipboard, so it snapshots the user's clipboard rather
/// than the first transcript.
pub struct PasteQueue<P> {
    platform: Mutex<P>,
    delays: PasteDelays,
}

impl<P: PastePlatform> PasteQueue<P> {
    #[must_use]
    pub const fn new(platform: P, delays: PasteDelays) -> Self {
        Self {
            platform: Mutex::new(platform),
            delays,
        }
    }

    pub fn paste(&self, text: &str, cancellation: &CancellationToken) -> PasteOutcome<P::Error> {
        self.paste_reporting_keystroke(text, cancellation, |_| {})
    }

    /// Pastes like [`Self::paste`], and calls `sent` as soon as the keystroke
    /// has been sent, with whether the previous clipboard will be put back.
    /// The restore still happens before this returns.
    pub fn paste_reporting_keystroke(
        &self,
        text: &str,
        cancellation: &CancellationToken,
        sent: impl FnOnce(bool),
    ) -> PasteOutcome<P::Error> {
        let mut platform = self.platform.lock().unwrap_or_else(PoisonError::into_inner);
        paste_and_restore(&mut *platform, text, cancellation, self.delays, sent)
    }

    /// Leaves `text` on the clipboard, excluded from clipboard history, for the user to paste.
    /// Queued behind any paste so it never interleaves with a compare-and-restore.
    pub fn copy(&self, text: &str) -> Result<(), P::Error> {
        let mut platform = self.platform.lock().unwrap_or_else(PoisonError::into_inner);
        platform.write_transcript(text)
    }
}

enum Previous<S> {
    /// Unreadable, or nothing capturable: the transcript is left in place.
    NotCaptured,
    Contents(S),
}

/// Snapshots the clipboard, writes and pastes `text`, waits, then restores the
/// snapshot only if the clipboard still holds `text`.
///
/// A snapshot failure never blocks the paste; it only skips the restore. Every
/// path after the write, including cancellation, ends in the same
/// compare-and-restore. Cancellation only shortens the wait before the
/// keystroke: once the keystroke may have been sent, the target always gets
/// the whole (bounded) restore delay to read the transcript.
fn paste_and_restore<P: PastePlatform>(
    platform: &mut P,
    text: &str,
    cancellation: &CancellationToken,
    delays: PasteDelays,
    sent: impl FnOnce(bool),
) -> PasteOutcome<P::Error> {
    if cancellation.is_cancelled() {
        return PasteOutcome::Cancelled;
    }
    let previous = match platform.snapshot_clipboard() {
        Ok(Some(snapshot)) => Previous::Contents(snapshot),
        Ok(None) => Previous::NotCaptured,
        Err(error) => {
            tracing::warn!(?error, "clipboard snapshot failed; pasting without restore");
            Previous::NotCaptured
        }
    };

    if let Err(error) = platform.write_transcript(text) {
        // A failed write may still have replaced the clipboard.
        restore_if_transcript(platform, text, previous);
        return PasteOutcome::Failed(error);
    }
    if !platform.wait(delays.propagation, cancellation) {
        restore_if_transcript(platform, text, previous);
        return PasteOutcome::Cancelled;
    }
    let failure = match platform.send_paste_shortcut() {
        KeystrokeOutcome::Sent => {
            sent(matches!(previous, Previous::Contents(_)));
            None
        }
        KeystrokeOutcome::Uncertain(error) => Some(error),
        KeystrokeOutcome::NotSent(error) => {
            restore_if_transcript(platform, text, previous);
            return PasteOutcome::Failed(error);
        }
    };
    let _elapsed = platform.wait(delays.restore, &CancellationToken::default());
    let restored = restore_if_transcript(platform, text, previous);
    failure.map_or(restored, PasteOutcome::Failed)
}

fn restore_if_transcript<P: PastePlatform>(
    platform: &mut P,
    transcript: &str,
    previous: Previous<P::Snapshot>,
) -> PasteOutcome<P::Error> {
    let Previous::Contents(snapshot) = previous else {
        return PasteOutcome::PastedWithoutRestore;
    };
    match platform.clipboard_holds(transcript) {
        Ok(true) => {}
        Ok(false) => return PasteOutcome::RestoreSkippedClipboardChanged,
        Err(error) => {
            tracing::warn!(?error, "clipboard check failed; leaving the transcript");
            return PasteOutcome::PastedWithoutRestore;
        }
    }
    match platform.restore_clipboard(snapshot) {
        Ok(()) => PasteOutcome::PastedAndRestored,
        Err(error) => {
            tracing::warn!(?error, "clipboard restore failed");
            PasteOutcome::PastedWithoutRestore
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{
            Arc, Mutex,
            mpsc::{self, Receiver, RecvTimeoutError, Sender},
        },
        thread,
    };

    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum Event {
        Snapshot,
        Write(String),
        Paste(Option<String>),
        Restore(String),
    }

    #[derive(Clone, Copy, Debug)]
    enum WaitAction {
        Elapse,
        UserCopies(&'static str),
        Cancel,
    }

    /// A one-slot text clipboard and a paste target that records what it read.
    struct FakeDesktop {
        clipboard: Option<String>,
        snapshot_fails: bool,
        write_fails: bool,
        shortcut: KeystrokeOutcome<&'static str>,
        waits: VecDeque<WaitAction>,
        /// Each wait's duration and whether it ran to its end.
        wait_log: Vec<(Duration, bool)>,
        /// Cancelled by the caller as soon as the keystroke is sent.
        cancel_on_paste: Option<CancellationToken>,
        events: Arc<Mutex<Vec<Event>>>,
        restore_wait_gate: Option<(Sender<()>, Receiver<()>)>,
    }

    impl FakeDesktop {
        fn holding(clipboard: Option<&str>) -> Self {
            Self {
                clipboard: clipboard.map(str::to_owned),
                snapshot_fails: false,
                write_fails: false,
                shortcut: KeystrokeOutcome::Sent,
                waits: VecDeque::new(),
                wait_log: Vec::new(),
                cancel_on_paste: None,
                events: Arc::default(),
                restore_wait_gate: None,
            }
        }

        fn record(&self, event: Event) {
            self.events.lock().expect("event log").push(event);
        }

        fn events(&self) -> Vec<Event> {
            self.events.lock().expect("event log").clone()
        }
    }

    impl PastePlatform for FakeDesktop {
        type Snapshot = String;
        type Error = &'static str;

        fn snapshot_clipboard(&mut self) -> Result<Option<String>, &'static str> {
            self.record(Event::Snapshot);
            if self.snapshot_fails {
                return Err("snapshot failed");
            }
            Ok(self.clipboard.clone())
        }

        fn write_transcript(&mut self, text: &str) -> Result<(), &'static str> {
            self.record(Event::Write(text.to_owned()));
            self.clipboard = Some(text.to_owned());
            if self.write_fails {
                return Err("write failed");
            }
            Ok(())
        }

        fn clipboard_holds(&mut self, text: &str) -> Result<bool, &'static str> {
            Ok(self.clipboard.as_deref() == Some(text))
        }

        fn restore_clipboard(&mut self, snapshot: String) -> Result<(), &'static str> {
            self.record(Event::Restore(snapshot.clone()));
            self.clipboard = Some(snapshot);
            Ok(())
        }

        fn wait(&mut self, duration: Duration, cancellation: &CancellationToken) -> bool {
            if duration == PasteDelays::default().restore()
                && let Some((entered, release)) = &self.restore_wait_gate
            {
                entered.send(()).expect("test observes the restore wait");
                release.recv().expect("test releases the restore wait");
            }
            match self.waits.pop_front().unwrap_or(WaitAction::Elapse) {
                WaitAction::Elapse => {}
                WaitAction::UserCopies(text) => self.clipboard = Some(text.to_owned()),
                WaitAction::Cancel => cancellation.cancel(),
            }
            let completed = !cancellation.is_cancelled();
            self.wait_log.push((duration, completed));
            completed
        }

        fn send_paste_shortcut(&mut self) -> KeystrokeOutcome<&'static str> {
            if let Some(cancellation) = &self.cancel_on_paste {
                cancellation.cancel();
            }
            if self.shortcut == KeystrokeOutcome::Sent {
                self.record(Event::Paste(self.clipboard.clone()));
            }
            self.shortcut
        }
    }

    #[test]
    fn copy_writes_the_transcript_without_pasting_or_restoring() {
        let desktop = FakeDesktop::holding(Some("previous"));
        let events = Arc::clone(&desktop.events);
        let queue = PasteQueue::new(desktop, PasteDelays::default());

        assert_eq!(queue.copy("transcript"), Ok(()));
        assert_eq!(
            *events.lock().expect("event log"),
            [Event::Write("transcript".to_owned())]
        );
    }

    fn run(desktop: &mut FakeDesktop) -> PasteOutcome<&'static str> {
        paste_and_restore(
            desktop,
            "transcript",
            &CancellationToken::default(),
            PasteDelays::default(),
            |_| {},
        )
    }

    fn pasted(text: &str) -> Event {
        Event::Paste(Some(text.to_owned()))
    }

    #[test]
    fn a_normal_paste_restores_the_previous_clipboard() {
        let mut desktop = FakeDesktop::holding(Some("previous"));

        assert_eq!(run(&mut desktop), PasteOutcome::PastedAndRestored);
        assert_eq!(desktop.clipboard.as_deref(), Some("previous"));
        assert!(desktop.events().contains(&pasted("transcript")));
    }

    #[test]
    fn the_sent_keystroke_is_reported_with_whether_a_restore_follows() {
        let report = |desktop: &mut FakeDesktop| {
            let mut reported = None;
            paste_and_restore(
                desktop,
                "transcript",
                &CancellationToken::default(),
                PasteDelays::default(),
                |restoring| reported = Some(restoring),
            );
            reported
        };
        assert_eq!(
            report(&mut FakeDesktop::holding(Some("previous"))),
            Some(true)
        );
        assert_eq!(report(&mut FakeDesktop::holding(None)), Some(false));
        let mut refused = FakeDesktop {
            shortcut: KeystrokeOutcome::NotSent("focus changed"),
            ..FakeDesktop::holding(Some("previous"))
        };
        assert_eq!(report(&mut refused), None);
    }

    #[test]
    fn a_snapshot_error_still_pastes_and_leaves_the_transcript() {
        let mut desktop = FakeDesktop {
            snapshot_fails: true,
            ..FakeDesktop::holding(Some("previous"))
        };

        assert_eq!(run(&mut desktop), PasteOutcome::PastedWithoutRestore);
        assert!(desktop.events().contains(&pasted("transcript")));
        assert_eq!(desktop.clipboard.as_deref(), Some("transcript"));
    }

    #[test]
    fn a_user_copy_during_either_wait_is_not_overwritten() {
        for waits in [
            vec![WaitAction::UserCopies("newer")],
            vec![WaitAction::Elapse, WaitAction::UserCopies("newer")],
        ] {
            let mut desktop = FakeDesktop {
                waits: waits.into(),
                ..FakeDesktop::holding(Some("previous"))
            };

            assert_eq!(
                run(&mut desktop),
                PasteOutcome::RestoreSkippedClipboardChanged
            );
            assert_eq!(desktop.clipboard.as_deref(), Some("newer"));
            assert!(
                !desktop
                    .events()
                    .iter()
                    .any(|event| matches!(event, Event::Restore(_)))
            );
        }
    }

    #[test]
    fn cancel_before_the_keystroke_skips_it_and_restores() {
        let mut before = FakeDesktop {
            waits: [WaitAction::Cancel].into(),
            ..FakeDesktop::holding(Some("previous"))
        };
        assert_eq!(run(&mut before), PasteOutcome::Cancelled);
        assert!(!before.events().contains(&pasted("transcript")));
        assert_eq!(before.clipboard.as_deref(), Some("previous"));
    }

    #[test]
    fn cancel_after_the_keystroke_still_gives_the_target_the_whole_restore_delay() {
        let delays = PasteDelays::default();
        for shortcut in [
            KeystrokeOutcome::Sent,
            KeystrokeOutcome::Uncertain("uncertain"),
        ] {
            let cancellation = CancellationToken::default();
            let mut desktop = FakeDesktop {
                shortcut,
                cancel_on_paste: Some(cancellation.clone()),
                ..FakeDesktop::holding(Some("previous"))
            };

            let outcome =
                paste_and_restore(&mut desktop, "transcript", &cancellation, delays, |_| {});

            assert!(cancellation.is_cancelled());
            assert_ne!(outcome, PasteOutcome::Cancelled);
            assert_eq!(
                desktop.wait_log,
                [(delays.propagation, true), (delays.restore, true)]
            );
            assert_eq!(desktop.clipboard.as_deref(), Some("previous"));
        }
    }

    #[test]
    fn nothing_capturable_is_never_cleared_and_reports_no_restore() {
        let mut desktop = FakeDesktop::holding(None);

        assert_eq!(run(&mut desktop), PasteOutcome::PastedWithoutRestore);
        assert!(desktop.events().contains(&pasted("transcript")));
        assert_eq!(desktop.clipboard.as_deref(), Some("transcript"));
    }

    #[test]
    fn failed_writes_and_keystrokes_restore_and_report_the_error() {
        let mut write = FakeDesktop {
            write_fails: true,
            ..FakeDesktop::holding(Some("previous"))
        };
        assert_eq!(run(&mut write), PasteOutcome::Failed("write failed"));
        assert_eq!(write.clipboard.as_deref(), Some("previous"));

        for (shortcut, error) in [
            (KeystrokeOutcome::NotSent("not sent"), "not sent"),
            (KeystrokeOutcome::Uncertain("uncertain"), "uncertain"),
        ] {
            let mut desktop = FakeDesktop {
                shortcut,
                ..FakeDesktop::holding(Some("previous"))
            };
            assert_eq!(run(&mut desktop), PasteOutcome::Failed(error));
            assert_eq!(desktop.clipboard.as_deref(), Some("previous"));
        }
    }

    #[test]
    fn cancellation_before_the_transaction_leaves_the_clipboard_untouched() {
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let mut desktop = FakeDesktop::holding(Some("previous"));

        assert_eq!(
            paste_and_restore(
                &mut desktop,
                "text",
                &cancelled,
                PasteDelays::default(),
                |_| {}
            ),
            PasteOutcome::Cancelled
        );
        assert!(desktop.events().is_empty());
    }

    #[test]
    fn two_rapid_pastes_serialize() {
        let (entered_tx, entered) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        let desktop = FakeDesktop {
            restore_wait_gate: Some((entered_tx, release_rx)),
            ..FakeDesktop::holding(Some("previous"))
        };
        let events = Arc::clone(&desktop.events);
        let queue = Arc::new(PasteQueue::new(desktop, PasteDelays::default()));

        let first_queue = Arc::clone(&queue);
        let first =
            thread::spawn(move || first_queue.paste("first", &CancellationToken::default()));
        entered
            .recv()
            .expect("first paste reaches its restore wait");

        let (second_started_tx, second_started) = mpsc::channel();
        let second_queue = Arc::clone(&queue);
        let second = thread::spawn(move || {
            second_started_tx
                .send(())
                .expect("test observes the second paste");
            second_queue.paste("second", &CancellationToken::default())
        });
        second_started.recv().expect("second paste starts");
        // The second paste must not snapshot while the first holds the queue.
        assert_eq!(
            entered.recv_timeout(Duration::from_millis(100)),
            Err(RecvTimeoutError::Timeout)
        );
        assert_eq!(
            events
                .lock()
                .expect("event log")
                .iter()
                .filter(|event| **event == Event::Snapshot)
                .count(),
            1
        );

        release.send(()).expect("release the first paste");
        assert_eq!(
            first.join().expect("first paste"),
            PasteOutcome::PastedAndRestored
        );
        entered
            .recv()
            .expect("second paste reaches its restore wait");
        release.send(()).expect("release the second paste");
        assert_eq!(
            second.join().expect("second paste"),
            PasteOutcome::PastedAndRestored
        );

        let restore = |text: &str| Event::Restore(text.to_owned());
        let write = |text: &str| Event::Write(text.to_owned());
        assert_eq!(
            *events.lock().expect("event log"),
            vec![
                Event::Snapshot,
                write("first"),
                pasted("first"),
                restore("previous"),
                Event::Snapshot,
                write("second"),
                pasted("second"),
                restore("previous"),
            ]
        );
    }
}
