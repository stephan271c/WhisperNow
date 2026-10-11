use std::path::PathBuf;
#[cfg(target_os = "linux")]
use std::sync::Arc;

use whispernow_core::{
    CancellationToken,
    text_output::{PasteOutcome, PastePlatform, PasteQueue},
};
use whispernow_domain::ipc::PermissionStatus;
use whispernow_platform::{NativePasteError, NativePastePlatform};

#[cfg(target_os = "linux")]
use crate::input_helper::HelperPastePlatform;
#[cfg(target_os = "linux")]
use std::{
    sync::{Mutex, mpsc as std_mpsc},
    thread,
    time::Duration,
};
#[cfg(target_os = "linux")]
use whispernow_platform::wayland::{
    RemoteDesktopRestoreTokenStore, WaylandOutput, WaylandPortalError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OutputNotice {
    Pasted,
    PermissionDenied,
    CapabilityUnavailable,
    PasteFailed,
    Cancelled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CopyUnavailable;

pub(crate) trait TranscriptOutput: Send + Sync {
    fn output(&self, text: &str, cancellation: &CancellationToken) -> OutputNotice;
}

/// The [`PasteQueue`]s run one paste at a time, so a paste never snapshots another's
/// transcript, and the Wayland portal runs on one joined worker that handles one command at
/// a time.
pub(crate) struct OutputCoordinator {
    operation: Option<OutputOperation>,
    stopping: CancellationToken,
}

enum OutputOperation {
    Native(PasteQueue<NativePastePlatform>),
    /// Wayland with the paste helper; preferred over the portal.
    #[cfg(target_os = "linux")]
    Helper(Arc<PasteQueue<HelperPastePlatform>>),
    #[cfg(target_os = "linux")]
    Wayland(WaylandOutputPortal),
}

impl OutputCoordinator {
    pub(crate) fn native(wayland_token_path: Option<PathBuf>) -> Self {
        let stopping = CancellationToken::default();
        #[cfg(target_os = "linux")]
        let operation = wayland_token_path.map(|path| {
            HelperPastePlatform::connect().map_or_else(
                || {
                    tracing::info!("the paste helper is unavailable; pasting through the portal");
                    OutputOperation::Wayland(WaylandOutputPortal::spawn(path, stopping.clone()))
                },
                |platform| OutputOperation::Helper(Arc::new(PasteQueue::new(platform))),
            )
        });
        #[cfg(not(target_os = "linux"))]
        let operation = {
            let _ = wayland_token_path;
            None
        };
        let operation = operation.or_else(|| {
            NativePastePlatform::connect()
                .inspect_err(|error| tracing::warn!(?error, "native paste output is unavailable"))
                .ok()
                .map(|platform| OutputOperation::Native(PasteQueue::new(platform)))
        });
        Self {
            operation,
            stopping,
        }
    }

    pub(crate) fn output(&self, text: &str, cancellation: &CancellationToken) -> OutputNotice {
        if cancellation.is_cancelled() {
            return OutputNotice::Cancelled;
        }
        if text.is_empty() || self.stopping.is_cancelled() {
            return OutputNotice::PasteFailed;
        }
        let Some(operation) = &self.operation else {
            return OutputNotice::PasteFailed;
        };
        match operation {
            OutputOperation::Native(queue) => native_notice(queue.paste(text, cancellation)),
            #[cfg(target_os = "linux")]
            OutputOperation::Helper(queue) => helper_output(queue, text, cancellation),
            #[cfg(target_os = "linux")]
            OutputOperation::Wayland(portal) => portal.output(text, cancellation),
        }
    }

    pub(crate) const fn can_copy(&self) -> bool {
        match self.operation {
            Some(OutputOperation::Native(_)) => true,
            #[cfg(target_os = "linux")]
            Some(OutputOperation::Helper(_)) => true,
            _ => false,
        }
    }

    /// Leaves `text` on the clipboard for a manual paste, excluded from clipboard history.
    /// The portal's clipboard lives only in its paste session, so there it is unavailable.
    pub(crate) fn copy(&self, text: &str) -> Result<(), CopyUnavailable> {
        let copied = match &self.operation {
            Some(OutputOperation::Native(queue)) => queue.copy(text),
            #[cfg(target_os = "linux")]
            Some(OutputOperation::Helper(queue)) => queue.copy(text),
            #[cfg(target_os = "linux")]
            Some(OutputOperation::Wayland(_)) => return Err(CopyUnavailable),
            None => return Err(CopyUnavailable),
        };
        copied.map_err(|error| {
            tracing::warn!(?error, "transcript could not be copied");
            CopyUnavailable
        })
    }

    pub(crate) const fn native_wayland(&self) -> bool {
        match self.operation {
            #[cfg(target_os = "linux")]
            Some(OutputOperation::Helper(_) | OutputOperation::Wayland(_)) => true,
            _ => false,
        }
    }

    /// The `RemoteDesktop` portal consent the paste keystroke needs, or `None`
    /// outside a native Wayland session, where no portal is involved.
    pub(crate) fn remote_desktop_consent(&self) -> Option<PermissionStatus> {
        match &self.operation {
            #[cfg(target_os = "linux")]
            Some(OutputOperation::Wayland(portal)) => Some(portal.consent()),
            _ => None,
        }
    }

    /// Asks for the paste keystroke's portal consent now, while setup shows why,
    /// instead of at the first paste. It waits for the user's answer.
    pub(crate) fn request_remote_desktop_consent(&self) -> Option<PermissionStatus> {
        match &self.operation {
            #[cfg(target_os = "linux")]
            Some(OutputOperation::Wayland(portal)) => Some(portal.prepare()),
            _ => None,
        }
    }

    pub(crate) fn request_shutdown(&self) {
        self.stopping.cancel();
    }

    pub(crate) fn shutdown(&self) {
        self.request_shutdown();
        #[cfg(target_os = "linux")]
        if let Some(OutputOperation::Wayland(portal)) = &self.operation {
            portal.shutdown();
        }
    }
}

impl TranscriptOutput for OutputCoordinator {
    fn output(&self, text: &str, cancellation: &CancellationToken) -> OutputNotice {
        Self::output(self, text, cancellation)
    }
}

const fn native_notice(outcome: PasteOutcome<NativePasteError>) -> OutputNotice {
    match outcome {
        PasteOutcome::PastedAndRestored
        | PasteOutcome::PastedWithoutRestore
        | PasteOutcome::RestoreSkippedClipboardChanged => OutputNotice::Pasted,
        PasteOutcome::Failed(NativePasteError::PermissionDenied) => OutputNotice::PermissionDenied,
        PasteOutcome::Cancelled => OutputNotice::Cancelled,
        PasteOutcome::Failed(_) => OutputNotice::PasteFailed,
    }
}

/// Pastes on its own thread and answers as soon as the keystroke is sent, so
/// the dictation ends, and the next can start, while the target reads the
/// transcript; the queue holds a later paste until the clipboard is restored.
/// The notice assumes the restore succeeds: one skipped because the user copied
/// something within that delay is only logged.
#[cfg(target_os = "linux")]
fn helper_output<P>(
    queue: &Arc<PasteQueue<P>>,
    text: &str,
    cancellation: &CancellationToken,
) -> OutputNotice
where
    P: PastePlatform<Error = NativePasteError> + Send + 'static,
{
    enum Progress {
        Sent,
        Finished(PasteOutcome<NativePasteError>),
    }
    let (progress, reports) = std_mpsc::channel();
    let queue = Arc::clone(queue);
    let text = text.to_owned();
    let cancellation = cancellation.clone();
    let spawned = thread::Builder::new()
        .name("whispernow-paste".to_owned())
        .spawn(move || {
            let sent = progress.clone();
            let outcome = queue.paste_reporting_keystroke(&text, &cancellation, move |_| {
                let _ = sent.send(Progress::Sent);
            });
            tracing::debug!(?outcome, "paste finished");
            let _ = progress.send(Progress::Finished(outcome));
        });
    if spawned.is_err() {
        return OutputNotice::PasteFailed;
    }
    match reports.recv() {
        Ok(Progress::Sent) => OutputNotice::Pasted,
        Ok(Progress::Finished(outcome)) => native_notice(outcome),
        Err(_) => OutputNotice::PasteFailed,
    }
}

#[cfg(target_os = "linux")]
enum WaylandOutputCommand {
    Paste {
        text: String,
        cancellation: CancellationToken,
        reply: std_mpsc::SyncSender<PasteOutcome<WaylandPortalError>>,
    },
    Prepare {
        reply: std_mpsc::SyncSender<Result<(), WaylandPortalError>>,
    },
    Shutdown {
        reply: std_mpsc::SyncSender<()>,
    },
}

#[cfg(target_os = "linux")]
struct WaylandOutputPortal {
    commands: tokio::sync::mpsc::UnboundedSender<WaylandOutputCommand>,
    available: bool,
    worker: Mutex<Option<thread::JoinHandle<()>>>,
    /// A saved restore token means the user already consented.
    restore_token: RemoteDesktopRestoreTokenStore,
    /// The consent the last portal answer showed, which outranks the token.
    consent: Mutex<Option<PermissionStatus>>,
}

/// The consent a portal answer shows, when it shows one.
#[cfg(target_os = "linux")]
const fn consent_after<T>(result: &Result<T, WaylandPortalError>) -> Option<PermissionStatus> {
    match result {
        Ok(_) => Some(PermissionStatus::Granted),
        Err(WaylandPortalError::PermissionDenied | WaylandPortalError::KeyboardUnavailable) => {
            Some(PermissionStatus::Denied)
        }
        Err(WaylandPortalError::PortalUnavailable | WaylandPortalError::Disconnected) => {
            Some(PermissionStatus::Unavailable)
        }
        Err(_) => None,
    }
}

#[cfg(target_os = "linux")]
impl WaylandOutputPortal {
    const PORTAL_TIMEOUT: Duration = Duration::from_mins(2);

    fn spawn(restore_token_path: PathBuf, stopping: CancellationToken) -> Self {
        let restore_token = RemoteDesktopRestoreTokenStore::new(restore_token_path.clone());
        let (commands, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let worker = thread::Builder::new()
            .name("whispernow-wayland-output".to_owned())
            .spawn(move || {
                let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                else {
                    return;
                };
                runtime.block_on(async move {
                    let mut portal = WaylandOutput::new(restore_token_path);
                    while let Some(command) = receiver.recv().await {
                        match command {
                            WaylandOutputCommand::Paste {
                                text,
                                cancellation,
                                reply,
                            } => {
                                // The paste observes cancellation and always ends in
                                // its compare-and-restore, so stopping the application
                                // only cancels it; only a hung portal call is dropped.
                                let paste = tokio::time::timeout(
                                    Self::PORTAL_TIMEOUT,
                                    portal.paste(&text, &cancellation),
                                );
                                tokio::pin!(paste);
                                let result = tokio::select! {
                                    result = &mut paste => result,
                                    () = stopping.cancelled() => {
                                        cancellation.cancel();
                                        paste.await
                                    }
                                };
                                let _ =
                                    reply.send(result.unwrap_or(PasteOutcome::Failed(
                                        WaylandPortalError::Backend,
                                    )));
                            }
                            WaylandOutputCommand::Prepare { reply } => {
                                // Stopping the application drops the wait for the
                                // user's answer.
                                let result = tokio::select! {
                                    result = tokio::time::timeout(
                                        Self::PORTAL_TIMEOUT,
                                        portal.prepare_output(),
                                    ) => result.unwrap_or(Err(WaylandPortalError::Backend)),
                                    () = stopping.cancelled() => Err(WaylandPortalError::Cancelled),
                                };
                                let _ = reply.send(result);
                            }
                            WaylandOutputCommand::Shutdown { reply } => {
                                portal.shutdown().await;
                                let _ = reply.send(());
                                break;
                            }
                        }
                    }
                });
            })
            .ok();
        Self {
            commands,
            available: worker.is_some(),
            worker: Mutex::new(worker),
            restore_token,
            consent: Mutex::new(None),
        }
    }

    fn record_consent<T>(&self, result: &Result<T, WaylandPortalError>) {
        if let Some(consent) = consent_after(result) {
            *self.consent.lock().expect("output state lock poisoned") = Some(consent);
        }
    }

    fn consent(&self) -> PermissionStatus {
        if !self.available {
            return PermissionStatus::Unavailable;
        }
        let observed = *self.consent.lock().expect("output state lock poisoned");
        observed.unwrap_or_else(|| {
            if self.restore_token.has_token() {
                PermissionStatus::Granted
            } else {
                PermissionStatus::NotDetermined
            }
        })
    }

    /// Opens the portal's output session, which asks for consent unless the
    /// restore token covers it. Stopping the application abandons the wait.
    fn prepare(&self) -> PermissionStatus {
        if !self.available {
            return PermissionStatus::Unavailable;
        }
        let (reply, receiver) = std_mpsc::sync_channel(1);
        if self
            .commands
            .send(WaylandOutputCommand::Prepare { reply })
            .is_err()
        {
            return PermissionStatus::Unavailable;
        }
        let result = receiver
            .recv()
            .unwrap_or(Err(WaylandPortalError::Disconnected));
        self.record_consent(&result);
        self.consent()
    }

    fn output(&self, text: &str, cancellation: &CancellationToken) -> OutputNotice {
        if !self.available {
            return OutputNotice::CapabilityUnavailable;
        }
        let (reply, receiver) = std_mpsc::sync_channel(1);
        if self
            .commands
            .send(WaylandOutputCommand::Paste {
                text: text.to_owned(),
                cancellation: cancellation.clone(),
                reply,
            })
            .is_err()
        {
            return OutputNotice::PasteFailed;
        }
        // A cancelled paste still reports what happened: after the keystroke
        // the transcript was pasted, and the worker bounds the wait.
        let Ok(response) = receiver.recv() else {
            return OutputNotice::CapabilityUnavailable;
        };
        let consent = match response {
            PasteOutcome::Failed(error) => Err(error),
            PasteOutcome::Cancelled => Err(WaylandPortalError::Cancelled),
            _ => Ok(()),
        };
        self.record_consent(&consent);
        match response {
            PasteOutcome::PastedAndRestored
            | PasteOutcome::PastedWithoutRestore
            | PasteOutcome::RestoreSkippedClipboardChanged => OutputNotice::Pasted,
            PasteOutcome::Failed(
                WaylandPortalError::PermissionDenied | WaylandPortalError::KeyboardUnavailable,
            ) => OutputNotice::PermissionDenied,
            PasteOutcome::Failed(
                WaylandPortalError::PortalUnavailable | WaylandPortalError::Disconnected,
            ) => OutputNotice::CapabilityUnavailable,
            PasteOutcome::Cancelled => OutputNotice::Cancelled,
            PasteOutcome::Failed(_) if cancellation.is_cancelled() => OutputNotice::Cancelled,
            PasteOutcome::Failed(_) => OutputNotice::PasteFailed,
        }
    }

    fn shutdown(&self) {
        let worker = self
            .worker
            .lock()
            .expect("output state lock poisoned")
            .take();
        let Some(worker) = worker else {
            return;
        };
        let (reply, receiver) = std_mpsc::sync_channel(1);
        let _ = self.commands.send(WaylandOutputCommand::Shutdown { reply });
        let _ = receiver.recv_timeout(Duration::from_secs(5));
        let _ = worker.join();
    }
}

#[cfg(test)]
mod tests {
    use whispernow_core::text_output::PasteOutcome;
    use whispernow_platform::NativePasteError;

    use super::{OutputNotice, native_notice};

    #[test]
    fn native_outcomes_map_to_notices_and_only_permission_failures_ask_for_permission() {
        for (outcome, expected) in [
            (PasteOutcome::PastedAndRestored, OutputNotice::Pasted),
            (
                PasteOutcome::Failed(NativePasteError::PermissionDenied),
                OutputNotice::PermissionDenied,
            ),
            (
                PasteOutcome::Failed(NativePasteError::InjectionFailed),
                OutputNotice::PasteFailed,
            ),
        ] {
            assert_eq!(native_notice(outcome), expected, "{outcome:?}");
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_helper_paste_answers_before_the_clipboard_is_restored() {
        use std::{
            sync::{Arc, mpsc},
            thread,
            time::Duration,
        };
        use whispernow_core::{
            CancellationToken,
            text_output::{KeystrokeOutcome, PastePlatform, PasteQueue},
        };

        struct Desktop {
            release_restore: mpsc::Receiver<()>,
            restored: mpsc::Sender<()>,
        }

        impl PastePlatform for Desktop {
            type Snapshot = ();
            type Error = NativePasteError;

            fn snapshot_clipboard(&mut self) -> Result<Option<()>, NativePasteError> {
                Ok(Some(()))
            }

            fn write_transcript(&mut self, _: &str) -> Result<(), NativePasteError> {
                Ok(())
            }

            fn clipboard_holds(&mut self, _: &str) -> Result<bool, NativePasteError> {
                Ok(true)
            }

            fn restore_clipboard(&mut self, (): ()) -> Result<(), NativePasteError> {
                self.release_restore
                    .recv_timeout(Duration::from_secs(10))
                    .expect("release restore");
                self.restored.send(()).expect("restoration observer");
                Ok(())
            }

            fn send_paste_shortcut(&mut self) -> KeystrokeOutcome<NativePasteError> {
                KeystrokeOutcome::Sent
            }
        }

        let (release_restore, restore_release) = mpsc::channel();
        let (restored, restoration) = mpsc::channel();
        let queue = Arc::new(PasteQueue::new(Desktop {
            release_restore: restore_release,
            restored,
        }));
        let (answer, response) = mpsc::channel();
        let worker = thread::spawn(move || {
            let notice =
                super::helper_output(&queue, "dictated words", &CancellationToken::default());
            answer.send(notice).expect("paste response");
        });
        let answered_before_restore = response.recv_timeout(Duration::from_secs(5));
        release_restore.send(()).expect("release restore");
        worker.join().expect("paste caller");
        restoration
            .recv_timeout(Duration::from_secs(5))
            .expect("clipboard restored");
        assert_eq!(answered_before_restore, Ok(OutputNotice::Pasted));
    }

    #[cfg(target_os = "linux")]
    mod wayland {
        use std::{
            sync::{
                Arc, Mutex,
                atomic::{AtomicUsize, Ordering},
            },
            thread,
        };

        use whispernow_core::{CancellationToken, text_output::PasteOutcome};
        use whispernow_domain::ipc::PermissionStatus;
        use whispernow_platform::wayland::{RemoteDesktopRestoreTokenStore, WaylandPortalError};

        use super::super::{
            OutputCoordinator, OutputNotice, OutputOperation, WaylandOutputCommand,
            WaylandOutputPortal,
        };

        /// A coordinator whose portal worker answers every paste with `reply`
        /// and counts the pastes it was asked to perform.
        fn scripted(
            reply: PasteOutcome<WaylandPortalError>,
        ) -> (OutputCoordinator, Arc<AtomicUsize>) {
            let pastes = Arc::new(AtomicUsize::new(0));
            let (commands, mut receiver) = tokio::sync::mpsc::unbounded_channel();
            let worker = {
                let pastes = Arc::clone(&pastes);
                thread::spawn(move || {
                    while let Some(command) = receiver.blocking_recv() {
                        match command {
                            WaylandOutputCommand::Paste { reply: sender, .. } => {
                                pastes.fetch_add(1, Ordering::SeqCst);
                                sender.send(reply).expect("paste reply");
                            }
                            WaylandOutputCommand::Prepare { reply: sender, .. } => {
                                let result = match reply {
                                    PasteOutcome::Failed(error) => Err(error),
                                    PasteOutcome::Cancelled => Err(WaylandPortalError::Cancelled),
                                    _ => Ok(()),
                                };
                                sender.send(result).expect("prepare reply");
                            }
                            WaylandOutputCommand::Shutdown { reply } => {
                                reply.send(()).expect("shutdown reply");
                                break;
                            }
                        }
                    }
                })
            };
            let coordinator = OutputCoordinator {
                operation: Some(OutputOperation::Wayland(WaylandOutputPortal {
                    commands,
                    available: true,
                    worker: Mutex::new(Some(worker)),
                    restore_token: RemoteDesktopRestoreTokenStore::new(
                        std::env::temp_dir().join("whispernow-no-restore-token"),
                    ),
                    consent: Mutex::new(None),
                })),
                stopping: CancellationToken::default(),
            };
            (coordinator, pastes)
        }

        #[test]
        fn only_portal_permission_failures_ask_for_permission() {
            for (reply, expected) in [
                (
                    PasteOutcome::Failed(WaylandPortalError::PermissionDenied),
                    OutputNotice::PermissionDenied,
                ),
                (
                    PasteOutcome::Failed(WaylandPortalError::KeyboardUnavailable),
                    OutputNotice::PermissionDenied,
                ),
                (
                    PasteOutcome::Failed(WaylandPortalError::Backend),
                    OutputNotice::PasteFailed,
                ),
            ] {
                let (coordinator, pastes) = scripted(reply);
                assert_eq!(
                    coordinator.output("transcript", &CancellationToken::default()),
                    expected,
                    "{reply:?}"
                );
                assert_eq!(pastes.load(Ordering::SeqCst), 1);
                coordinator.shutdown();
            }
        }

        #[test]
        fn empty_stopping_and_cancelled_output_never_reach_the_portal() {
            let (coordinator, pastes) = scripted(PasteOutcome::PastedAndRestored);
            assert_eq!(
                coordinator.output("", &CancellationToken::default()),
                OutputNotice::PasteFailed
            );
            let cancelled = CancellationToken::default();
            cancelled.cancel();
            assert_eq!(
                coordinator.output("transcript", &cancelled),
                OutputNotice::Cancelled
            );
            coordinator.request_shutdown();
            assert_eq!(
                coordinator.output("transcript", &CancellationToken::default()),
                OutputNotice::PasteFailed
            );
            assert_eq!(pastes.load(Ordering::SeqCst), 0);
            coordinator.shutdown();
        }

        #[test]
        fn a_portal_without_a_worker_reports_the_capability_unavailable() {
            let (commands, _receiver) = tokio::sync::mpsc::unbounded_channel();
            let coordinator = OutputCoordinator {
                operation: Some(OutputOperation::Wayland(WaylandOutputPortal {
                    commands,
                    available: false,
                    worker: Mutex::new(None),
                    restore_token: RemoteDesktopRestoreTokenStore::new(
                        std::env::temp_dir().join("whispernow-no-restore-token"),
                    ),
                    consent: Mutex::new(None),
                })),
                stopping: CancellationToken::default(),
            };
            assert_eq!(
                coordinator.output("transcript", &CancellationToken::default()),
                OutputNotice::CapabilityUnavailable
            );
            assert_eq!(
                coordinator.remote_desktop_consent(),
                Some(PermissionStatus::Unavailable)
            );
            assert_eq!(
                coordinator.request_remote_desktop_consent(),
                Some(PermissionStatus::Unavailable)
            );
        }

        #[test]
        fn the_last_portal_answer_decides_the_paste_consent() {
            for (reply, expected) in [
                (PasteOutcome::PastedAndRestored, PermissionStatus::Granted),
                (
                    PasteOutcome::Failed(WaylandPortalError::PermissionDenied),
                    PermissionStatus::Denied,
                ),
                (
                    PasteOutcome::Failed(WaylandPortalError::PortalUnavailable),
                    PermissionStatus::Unavailable,
                ),
                (
                    PasteOutcome::Failed(WaylandPortalError::Backend),
                    PermissionStatus::NotDetermined,
                ),
            ] {
                let (coordinator, pastes) = scripted(reply);
                assert_eq!(
                    coordinator.remote_desktop_consent(),
                    Some(PermissionStatus::NotDetermined),
                    "without a restore token or an answer, consent is not known"
                );
                assert_eq!(
                    coordinator.request_remote_desktop_consent(),
                    Some(expected),
                    "prepare {reply:?}"
                );
                assert_eq!(pastes.load(Ordering::SeqCst), 0, "asking never pastes");
                coordinator.output("transcript", &CancellationToken::default());
                assert_eq!(
                    coordinator.remote_desktop_consent(),
                    Some(expected),
                    "paste {reply:?}"
                );
                coordinator.shutdown();
            }
        }
    }
}
