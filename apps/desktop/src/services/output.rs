use std::path::PathBuf;
#[cfg(target_os = "linux")]
use std::sync::Arc;

#[cfg(target_os = "linux")]
use whispernow_core::text_output::PastePlatform;
use whispernow_core::{
    CancellationToken,
    text_output::{PasteOutcome, PasteQueue},
};
use whispernow_domain::ipc::PermissionStatus;
use whispernow_domain::{AppError, ErrorCategory};
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
use whispernow_platform::{
    NativeClipboard,
    wayland::{RemoteDesktopRestoreTokenStore, WaylandOutput, WaylandPortalError},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OutputNotice {
    Pasted,
    PermissionDenied,
    #[cfg_attr(not(target_os = "linux"), expect(dead_code))]
    CapabilityUnavailable,
    PasteFailed,
    Cancelled,
}

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
    Helper(HelperOutput<HelperPastePlatform>),
    /// The portal's clipboard lives only in its paste session, so copies go to the `XWayland`
    /// clipboard instead.
    #[cfg(target_os = "linux")]
    Wayland {
        portal: WaylandOutputPortal,
        clipboard: Option<Mutex<NativeClipboard>>,
    },
}

impl OutputCoordinator {
    #[cfg_attr(not(target_os = "linux"), expect(clippy::needless_pass_by_value))]
    pub(crate) fn native(wayland_token_path: Option<PathBuf>) -> Self {
        let stopping = CancellationToken::default();
        #[cfg(target_os = "linux")]
        let operation = wayland_token_path.map(|path| {
            HelperPastePlatform::connect().map_or_else(
                || {
                    tracing::info!("the paste helper is unavailable; pasting through the portal");
                    let clipboard = NativeClipboard::connect()
                        .inspect_err(|error| {
                            tracing::warn!(?error, "the XWayland clipboard is unavailable");
                        })
                        .ok()
                        .map(Mutex::new);
                    OutputOperation::Wayland {
                        portal: WaylandOutputPortal::spawn(path, stopping.clone()),
                        clipboard,
                    }
                },
                |platform| OutputOperation::Helper(HelperOutput::new(platform, stopping.clone())),
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
            OutputOperation::Native(queue) => {
                let outcome = queue.paste(text, cancellation);
                tracing::debug!(?outcome, "paste finished");
                native_notice(outcome)
            }
            #[cfg(target_os = "linux")]
            OutputOperation::Helper(helper) => helper.output(text, cancellation),
            #[cfg(target_os = "linux")]
            OutputOperation::Wayland { portal, .. } => portal.output(text, cancellation),
        }
    }

    pub(crate) const fn can_copy(&self) -> bool {
        match self.operation {
            Some(OutputOperation::Native(_)) => true,
            #[cfg(target_os = "linux")]
            Some(
                OutputOperation::Helper(_)
                | OutputOperation::Wayland {
                    clipboard: Some(_), ..
                },
            ) => true,
            _ => false,
        }
    }

    /// Leaves `text` on the clipboard for a manual paste, excluded from clipboard history.
    pub(crate) fn copy(&self, text: &str) -> Result<(), AppError> {
        let copied = match &self.operation {
            Some(OutputOperation::Native(queue)) => queue.copy(text),
            #[cfg(target_os = "linux")]
            Some(OutputOperation::Helper(helper)) => helper.queue.copy(text),
            #[cfg(target_os = "linux")]
            Some(OutputOperation::Wayland {
                clipboard: Some(clipboard),
                ..
            }) => clipboard
                .lock()
                .expect("clipboard lock poisoned")
                .write_text(text),
            #[cfg(target_os = "linux")]
            Some(OutputOperation::Wayland {
                clipboard: None, ..
            }) => return Err(copy_unavailable()),
            None => return Err(copy_unavailable()),
        };
        copied.map_err(|error| {
            tracing::warn!(?error, "text could not be copied");
            copy_unavailable()
        })
    }

    #[cfg_attr(not(target_os = "linux"), expect(dead_code))]
    pub(crate) const fn native_wayland(&self) -> bool {
        match self.operation {
            #[cfg(target_os = "linux")]
            Some(OutputOperation::Helper(_) | OutputOperation::Wayland { .. }) => true,
            _ => false,
        }
    }

    /// The `RemoteDesktop` portal consent the paste keystroke needs, or `None`
    /// outside a native Wayland session, where no portal is involved.
    #[cfg_attr(not(target_os = "linux"), expect(dead_code))]
    pub(crate) fn remote_desktop_consent(&self) -> Option<PermissionStatus> {
        match &self.operation {
            #[cfg(target_os = "linux")]
            Some(OutputOperation::Wayland { portal, .. }) => Some(portal.consent()),
            _ => None,
        }
    }

    /// Asks for the paste keystroke's portal consent now, while setup shows why,
    /// instead of at the first paste. It waits for the user's answer.
    pub(crate) fn request_remote_desktop_consent(&self) -> Option<PermissionStatus> {
        match &self.operation {
            #[cfg(target_os = "linux")]
            Some(OutputOperation::Wayland { portal, .. }) => Some(portal.prepare()),
            _ => None,
        }
    }

    pub(crate) fn request_shutdown(&self) {
        self.stopping.cancel();
    }

    pub(crate) fn shutdown(&self) {
        self.request_shutdown();
        #[cfg(target_os = "linux")]
        match &self.operation {
            Some(OutputOperation::Helper(helper)) => helper.shutdown(),
            Some(OutputOperation::Wayland { portal, .. }) => portal.shutdown(),
            _ => {}
        }
    }
}

impl TranscriptOutput for OutputCoordinator {
    fn output(&self, text: &str, cancellation: &CancellationToken) -> OutputNotice {
        Self::output(self, text, cancellation)
    }
}

fn copy_unavailable() -> AppError {
    AppError::new(
        ErrorCategory::Output,
        "copy_unavailable",
        "The text could not be copied",
    )
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

#[cfg(target_os = "linux")]
struct HelperOutput<P> {
    queue: Arc<PasteQueue<P>>,
    stopping: CancellationToken,
    workers: Mutex<Vec<thread::JoinHandle<()>>>,
}

#[cfg(target_os = "linux")]
impl<P> HelperOutput<P>
where
    P: PastePlatform<Error = NativePasteError> + Send + 'static,
{
    fn new(platform: P, stopping: CancellationToken) -> Self {
        Self {
            queue: Arc::new(PasteQueue::new(platform)),
            stopping,
            workers: Mutex::new(Vec::new()),
        }
    }

    fn output(&self, text: &str, cancellation: &CancellationToken) -> OutputNotice {
        enum Progress {
            Sent,
            Finished(PasteOutcome<NativePasteError>),
        }
        let (progress, reports) = std_mpsc::channel();
        {
            // Admission and worker registration share the drain lock, so a producer
            // that passed the coordinator's earlier check cannot escape shutdown.
            let mut workers = self.workers.lock().expect("paste workers lock poisoned");
            if self.stopping.is_cancelled() {
                return OutputNotice::PasteFailed;
            }
            workers.retain(|worker| !worker.is_finished());
            let queue = Arc::clone(&self.queue);
            let text = text.to_owned();
            let cancellation = cancellation.clone();
            let spawned = thread::Builder::new()
                .name("whispernow-paste".to_owned())
                .spawn(move || {
                    let sent = progress.clone();
                    let outcome =
                        queue.paste_reporting_keystroke(&text, &cancellation, move || {
                            let _ = sent.send(Progress::Sent);
                        });
                    tracing::debug!(?outcome, "paste finished");
                    let _ = progress.send(Progress::Finished(outcome));
                });
            let Ok(worker) = spawned else {
                return OutputNotice::PasteFailed;
            };
            workers.push(worker);
        }
        // Dictation ends at the keystroke; shutdown still owns the restoration.
        match reports.recv() {
            Ok(Progress::Sent) => OutputNotice::Pasted,
            Ok(Progress::Finished(outcome)) => native_notice(outcome),
            Err(_) => OutputNotice::PasteFailed,
        }
    }

    fn shutdown(&self) {
        self.stopping.cancel();
        let workers =
            std::mem::take(&mut *self.workers.lock().expect("paste workers lock poisoned"));
        for worker in workers {
            let _ = worker.join();
        }
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

#[cfg(target_os = "linux")]
const fn failure_consent(error: WaylandPortalError) -> Option<PermissionStatus> {
    match error {
        WaylandPortalError::PermissionDenied | WaylandPortalError::KeyboardUnavailable => {
            Some(PermissionStatus::Denied)
        }
        WaylandPortalError::PortalUnavailable | WaylandPortalError::Disconnected => {
            Some(PermissionStatus::Unavailable)
        }
        _ => None,
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

    fn record_consent(&self, consent: Option<PermissionStatus>) {
        if consent.is_some() {
            *self.consent.lock().expect("output state lock poisoned") = consent;
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
        let consent = match receiver.recv() {
            Ok(Ok(())) => Some(PermissionStatus::Granted),
            Ok(Err(error)) => failure_consent(error),
            Err(_) => failure_consent(WaylandPortalError::Disconnected),
        };
        self.record_consent(consent);
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
        let (consent, notice) = match response {
            PasteOutcome::PastedAndRestored
            | PasteOutcome::PastedWithoutRestore
            | PasteOutcome::RestoreSkippedClipboardChanged => {
                (Some(PermissionStatus::Granted), OutputNotice::Pasted)
            }
            PasteOutcome::Cancelled => (None, OutputNotice::Cancelled),
            PasteOutcome::Failed(error) => {
                let consent = failure_consent(error);
                let notice = match consent {
                    Some(PermissionStatus::Denied) => OutputNotice::PermissionDenied,
                    Some(PermissionStatus::Unavailable) => OutputNotice::CapabilityUnavailable,
                    _ if cancellation.is_cancelled() => OutputNotice::Cancelled,
                    _ => OutputNotice::PasteFailed,
                };
                (consent, notice)
            }
        };
        self.record_consent(consent);
        notice
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
    fn helper_shutdown_waits_for_restoration_after_the_early_paste_answer() {
        use std::{
            sync::{Arc, Mutex, mpsc},
            thread,
            time::Duration,
        };
        use whispernow_core::{
            CancellationToken,
            text_output::{KeystrokeOutcome, PastePlatform},
        };

        struct Desktop {
            clipboard: Arc<Mutex<String>>,
            restore_entered: mpsc::Sender<()>,
            release_restore: mpsc::Receiver<()>,
        }

        impl PastePlatform for Desktop {
            type Snapshot = String;
            type Error = NativePasteError;

            fn snapshot_clipboard(&mut self) -> Result<Option<String>, NativePasteError> {
                Ok(Some(self.clipboard.lock().expect("clipboard").clone()))
            }

            fn write_transcript(&mut self, text: &str) -> Result<(), NativePasteError> {
                *self.clipboard.lock().expect("clipboard") = text.to_owned();
                Ok(())
            }

            fn clipboard_holds(&mut self, text: &str) -> Result<bool, NativePasteError> {
                Ok(*self.clipboard.lock().expect("clipboard") == text)
            }

            fn restore_clipboard(&mut self, snapshot: String) -> Result<(), NativePasteError> {
                self.restore_entered.send(()).expect("restoration observer");
                self.release_restore
                    .recv_timeout(Duration::from_secs(10))
                    .expect("release restore");
                *self.clipboard.lock().expect("clipboard") = snapshot;
                Ok(())
            }

            fn send_paste_shortcut(&mut self) -> KeystrokeOutcome<NativePasteError> {
                KeystrokeOutcome::Sent
            }
        }

        let clipboard = Arc::new(Mutex::new("previous clipboard".to_owned()));
        let (restore_entered, restoration) = mpsc::channel();
        let (release_restore, restore_release) = mpsc::channel();
        let stopping = CancellationToken::default();
        let helper = Arc::new(super::HelperOutput::new(
            Desktop {
                clipboard: Arc::clone(&clipboard),
                restore_entered,
                release_restore: restore_release,
            },
            stopping.clone(),
        ));
        let (answer, response) = mpsc::channel();
        let caller = {
            let helper = Arc::clone(&helper);
            thread::spawn(move || {
                let notice = helper.output("dictated words", &CancellationToken::default());
                let _ = answer.send(notice);
            })
        };
        let answered_before_restore = response.recv_timeout(Duration::from_secs(5));
        let reached_restore = restoration.recv_timeout(Duration::from_secs(5));
        let clipboard_during_restore = clipboard.lock().expect("clipboard").clone();
        stopping.cancel();
        let refused_after_shutdown = helper.output("late words", &CancellationToken::default());
        let (started, shutdown_started) = mpsc::channel();
        let (finished, shutdown_finished) = mpsc::channel();
        let shutdown = {
            let helper = Arc::clone(&helper);
            thread::spawn(move || {
                let _ = started.send(());
                helper.shutdown();
                let _ = finished.send(());
            })
        };
        let shutdown_began = shutdown_started.recv_timeout(Duration::from_secs(5));
        let finished_while_held = shutdown_finished.recv_timeout(Duration::from_millis(100));

        // Release and join before asserting so a failed observation cannot strand workers.
        let _ = release_restore.send(());
        let caller_result = caller.join();
        let shutdown_result = shutdown.join();
        let finished_after_release = shutdown_finished.recv_timeout(Duration::from_secs(5));

        assert_eq!(answered_before_restore, Ok(OutputNotice::Pasted));
        assert_eq!(reached_restore, Ok(()));
        assert_eq!(clipboard_during_restore, "dictated words");
        assert_eq!(refused_after_shutdown, OutputNotice::PasteFailed);
        assert_eq!(shutdown_began, Ok(()));
        assert_eq!(finished_while_held, Err(mpsc::RecvTimeoutError::Timeout));
        caller_result.expect("paste caller");
        shutdown_result.expect("shutdown caller");
        assert_eq!(finished_after_release, Ok(()));
        assert_eq!(*clipboard.lock().expect("clipboard"), "previous clipboard");
    }

    #[cfg(target_os = "linux")]
    mod wayland {
        use std::{
            collections::VecDeque,
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

        enum PortalAnswer {
            Prepare(Result<(), WaylandPortalError>),
            Paste(PasteOutcome<WaylandPortalError>),
            PasteAfterCancel(PasteOutcome<WaylandPortalError>),
        }

        fn scripted(
            answers: impl IntoIterator<Item = PortalAnswer>,
        ) -> (OutputCoordinator, Arc<AtomicUsize>) {
            let mut answers: VecDeque<_> = answers.into_iter().collect();
            let pastes = Arc::new(AtomicUsize::new(0));
            let restore_dir = tempfile::tempdir().expect("restore token directory");
            let restore_token_path = restore_dir.path().join("restore-token");
            let (commands, mut receiver) = tokio::sync::mpsc::unbounded_channel();
            let worker = {
                let pastes = Arc::clone(&pastes);
                thread::spawn(move || {
                    let _restore_dir = restore_dir;
                    while let Some(command) = receiver.blocking_recv() {
                        match command {
                            WaylandOutputCommand::Paste {
                                reply: sender,
                                cancellation,
                                ..
                            } => {
                                let outcome = match answers.pop_front() {
                                    Some(PortalAnswer::Paste(outcome)) => outcome,
                                    Some(PortalAnswer::PasteAfterCancel(outcome)) => {
                                        cancellation.cancel();
                                        outcome
                                    }
                                    _ => panic!("expected a scripted paste answer"),
                                };
                                pastes.fetch_add(1, Ordering::SeqCst);
                                sender.send(outcome).expect("paste reply");
                            }
                            WaylandOutputCommand::Prepare { reply: sender, .. } => {
                                let Some(PortalAnswer::Prepare(result)) = answers.pop_front()
                                else {
                                    panic!("expected a scripted preparation answer");
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
                operation: Some(OutputOperation::Wayland {
                    portal: WaylandOutputPortal {
                        commands,
                        available: true,
                        worker: Mutex::new(Some(worker)),
                        restore_token: RemoteDesktopRestoreTokenStore::new(restore_token_path),
                        consent: Mutex::new(None),
                    },
                    clipboard: None,
                }),
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
                let (coordinator, pastes) = scripted([PortalAnswer::Paste(reply)]);
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
        fn a_paste_that_fails_after_a_cancel_reports_the_cancel() {
            let (coordinator, _) = scripted([PortalAnswer::PasteAfterCancel(
                PasteOutcome::Failed(WaylandPortalError::Backend),
            )]);
            assert_eq!(
                coordinator.output("transcript", &CancellationToken::default()),
                OutputNotice::Cancelled
            );
            coordinator.shutdown();
        }

        #[test]
        fn empty_stopping_and_cancelled_output_never_reach_the_portal() {
            let (coordinator, pastes) = scripted([]);
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
            let restore_dir = tempfile::tempdir().expect("restore token directory");
            let (commands, _receiver) = tokio::sync::mpsc::unbounded_channel();
            let coordinator = OutputCoordinator {
                operation: Some(OutputOperation::Wayland {
                    portal: WaylandOutputPortal {
                        commands,
                        available: false,
                        worker: Mutex::new(None),
                        restore_token: RemoteDesktopRestoreTokenStore::new(
                            restore_dir.path().join("restore-token"),
                        ),
                        consent: Mutex::new(None),
                    },
                    clipboard: None,
                }),
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
            assert!(!coordinator.can_copy());
            assert_eq!(
                coordinator
                    .copy("transcript")
                    .expect_err("no clipboard")
                    .code,
                "copy_unavailable"
            );
        }

        #[test]
        fn the_last_portal_answer_decides_the_paste_consent() {
            let (coordinator, pastes) = scripted([
                PortalAnswer::Prepare(Ok(())),
                PortalAnswer::Paste(PasteOutcome::Failed(WaylandPortalError::PermissionDenied)),
                PortalAnswer::Paste(PasteOutcome::Failed(WaylandPortalError::Backend)),
                PortalAnswer::Paste(PasteOutcome::Failed(WaylandPortalError::PortalUnavailable)),
            ]);
            let initial = coordinator.remote_desktop_consent();
            let prepared = coordinator.request_remote_desktop_consent();
            let pastes_after_preparation = pastes.load(Ordering::SeqCst);
            let denied = coordinator.output("transcript", &CancellationToken::default());
            let consent_after_denial = coordinator.remote_desktop_consent();
            let backend_failed = coordinator.output("transcript", &CancellationToken::default());
            let consent_after_backend = coordinator.remote_desktop_consent();
            let unavailable = coordinator.output("transcript", &CancellationToken::default());
            let consent_after_unavailable = coordinator.remote_desktop_consent();
            coordinator.shutdown();

            assert_eq!(initial, Some(PermissionStatus::NotDetermined));
            assert_eq!(prepared, Some(PermissionStatus::Granted));
            assert_eq!(
                pastes_after_preparation, 0,
                "asking for consent never pastes"
            );
            assert_eq!(denied, OutputNotice::PermissionDenied);
            assert_eq!(consent_after_denial, Some(PermissionStatus::Denied));
            assert_eq!(backend_failed, OutputNotice::PasteFailed);
            assert_eq!(consent_after_backend, Some(PermissionStatus::Denied));
            assert_eq!(unavailable, OutputNotice::CapabilityUnavailable);
            assert_eq!(
                consent_after_unavailable,
                Some(PermissionStatus::Unavailable)
            );
        }

        #[test]
        fn answers_that_show_no_consent_leave_it_unchanged() {
            let (coordinator, _) = scripted([
                PortalAnswer::Prepare(Err(WaylandPortalError::PermissionDenied)),
                PortalAnswer::Prepare(Err(WaylandPortalError::Backend)),
                PortalAnswer::Paste(PasteOutcome::Cancelled),
                PortalAnswer::Paste(PasteOutcome::Failed(WaylandPortalError::Disconnected)),
                PortalAnswer::Paste(PasteOutcome::PastedAndRestored),
            ]);
            let denied = coordinator.request_remote_desktop_consent();
            let after_backend_failure = coordinator.request_remote_desktop_consent();
            let cancelled = coordinator.output("transcript", &CancellationToken::default());
            let after_cancel = coordinator.remote_desktop_consent();
            let disconnected = coordinator.output("transcript", &CancellationToken::default());
            let after_disconnect = coordinator.remote_desktop_consent();
            let pasted = coordinator.output("transcript", &CancellationToken::default());
            let after_paste = coordinator.remote_desktop_consent();
            coordinator.shutdown();

            assert_eq!(denied, Some(PermissionStatus::Denied));
            assert_eq!(after_backend_failure, Some(PermissionStatus::Denied));
            assert_eq!(cancelled, OutputNotice::Cancelled);
            assert_eq!(after_cancel, Some(PermissionStatus::Denied));
            assert_eq!(disconnected, OutputNotice::CapabilityUnavailable);
            assert_eq!(after_disconnect, Some(PermissionStatus::Unavailable));
            assert_eq!(pasted, OutputNotice::Pasted);
            assert_eq!(after_paste, Some(PermissionStatus::Granted));
        }
    }
}
