use std::{
    os::fd::OwnedFd,
    path::PathBuf,
    sync::{Arc, Mutex, PoisonError},
    time::Duration,
};

use ashpd::{
    Error as AshpdError,
    desktop::{
        CreateSessionOptions, PersistMode, Session,
        clipboard::{Clipboard, RequestClipboardOptions, SetSelectionOptions},
        remote_desktop::{
            DeviceType, KeyState, NotifyKeyboardKeysymOptions, RemoteDesktop, SelectDevicesOptions,
            StartOptions,
        },
    },
};
use futures_util::{FutureExt as _, StreamExt as _, future::BoxFuture};
use thiserror::Error;
use tokio::{
    io::{AsyncReadExt as _, AsyncWriteExt as _},
    net::unix::pipe,
    runtime::Handle,
    sync::mpsc::{self, error::TryRecvError},
    task::JoinHandle,
};
use whispernow_core::{
    CancellationToken,
    text_output::{PASTE_RESTORE_DELAY, PasteOutcome},
};

use crate::PasteKeys;

use super::{
    accessibility::active_paste_keys,
    clipboard::{
        CLIPBOARD_TIMEOUT, ClipboardOffer, MAX_FORMAT_BYTES, PortalClipboardState,
        SNAPSHOT_MIME_TYPES, SecretBytes,
    },
    token::RemoteDesktopRestoreTokenStore,
};

const XK_CONTROL_L: i32 = 0xffe3;
const XK_SHIFT_L: i32 = 0xffe1;
const XK_V_LOWER: i32 = 0x0076;
const BEFORE_PASTE_DELAY: Duration = Duration::from_millis(20);

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum WaylandPortalError {
    #[error("desktop portal operation failed")]
    Backend,
    #[error("the paste was cancelled")]
    Cancelled,
    #[error("clipboard snapshot exceeds safe limits")]
    ClipboardSnapshotTooLarge,
    #[error("clipboard portal is unavailable")]
    ClipboardUnavailable,
    #[error("desktop portal disconnected")]
    Disconnected,
    #[error("remote keyboard capability was not granted")]
    KeyboardUnavailable,
    #[error("desktop portal permission was denied")]
    PermissionDenied,
    #[error("required desktop portal is unavailable")]
    PortalUnavailable,
    #[error("remote desktop restore token could not be stored safely")]
    TokenStore,
}

enum ClipboardSignal {
    OwnerChanged { mime_types: Vec<String> },
    Closed,
}

/// Stops the listener and closes the portal session when dropped, so an output
/// abandoned half-way (a cancelled consent wait, a failed setup) leaves neither
/// behind. The portal does not emit `Closed` for a close we ask for.
struct CloseOnDrop {
    listener: JoinHandle<()>,
    close: Option<BoxFuture<'static, ()>>,
}

impl CloseOnDrop {
    fn new(listener: JoinHandle<()>, close: impl Future<Output = ()> + Send + 'static) -> Self {
        Self {
            listener,
            close: Some(close.boxed()),
        }
    }

    /// Waits for the listener to stop and the session to close.
    async fn close(mut self) {
        self.listener.abort();
        let _ignored = (&mut self.listener).await;
        if let Some(close) = self.close.take() {
            close.await;
        }
    }
}

impl Drop for CloseOnDrop {
    fn drop(&mut self) {
        self.listener.abort();
        // Without a runtime the portal closes the session when the process
        // leaves the bus.
        if let Some(close) = self.close.take()
            && let Ok(runtime) = Handle::try_current()
        {
            runtime.spawn(close);
        }
    }
}

struct OutputSession {
    remote: RemoteDesktop,
    clipboard: Arc<Clipboard>,
    session: Arc<Session<RemoteDesktop>>,
    signals: mpsc::UnboundedReceiver<ClipboardSignal>,
    cleanup: CloseOnDrop,
    clipboard_state: Arc<Mutex<PortalClipboardState>>,
    advertised_mime_types: Option<Vec<String>>,
}

impl OutputSession {
    /// Records the clipboard types a signal advertises; false when the signal
    /// says the session closed.
    fn note(&mut self, signal: ClipboardSignal) -> bool {
        match signal {
            ClipboardSignal::OwnerChanged { mime_types } => {
                self.advertised_mime_types = Some(mime_types);
                true
            }
            ClipboardSignal::Closed => false,
        }
    }

    /// Takes in the signals queued so far; true when the session closed or the
    /// listener is gone.
    fn poll_closed(&mut self) -> bool {
        loop {
            match self.signals.try_recv() {
                Ok(signal) => {
                    if !self.note(signal) {
                        return true;
                    }
                }
                Err(TryRecvError::Empty) => return false,
                Err(TryRecvError::Disconnected) => return true,
            }
        }
    }

    /// Asks for the clipboard and keyboard access, which prompts for consent
    /// unless the saved restore token covers it.
    async fn start(
        &self,
        token_store: &RemoteDesktopRestoreTokenStore,
    ) -> Result<(), WaylandPortalError> {
        self.clipboard
            .request(&self.session, RequestClipboardOptions::default())
            .await
            .map_err(map_portal_error)?;
        let restore_token = token_store.load();
        self.remote
            .select_devices(
                &self.session,
                SelectDevicesOptions::default()
                    .set_devices(Some(DeviceType::Keyboard.into()))
                    .set_persist_mode(PersistMode::ExplicitlyRevoked)
                    .set_restore_token(restore_token.as_deref()),
            )
            .await
            .map_err(map_portal_error)?
            .response()
            .map_err(map_portal_error)?;
        let selected = self
            .remote
            .start(&self.session, None, StartOptions::default())
            .await
            .map_err(map_portal_error)?
            .response()
            .map_err(map_portal_error)?;
        if !selected.devices().contains(DeviceType::Keyboard) {
            return Err(WaylandPortalError::KeyboardUnavailable);
        }
        if !selected.is_clipboard_enabled() {
            return Err(WaylandPortalError::ClipboardUnavailable);
        }
        if let Some(token) = selected.restore_token() {
            token_store.replace(token)?;
        }
        Ok(())
    }
}

/// Pastes through the `RemoteDesktop` and clipboard portals: offers the
/// transcript on the clipboard, sends one paste chord, then puts the previous
/// clipboard back. One owner drives it, one paste at a time.
pub struct WaylandOutput {
    token_store: RemoteDesktopRestoreTokenStore,
    output: Option<OutputSession>,
}

impl WaylandOutput {
    #[must_use]
    pub fn new(restore_token_path: PathBuf) -> Self {
        Self {
            token_store: RemoteDesktopRestoreTokenStore::new(restore_token_path),
            output: None,
        }
    }

    async fn create_output_session(&self) -> Result<OutputSession, WaylandPortalError> {
        let remote = RemoteDesktop::new().await.map_err(map_portal_error)?;
        if remote.version() < 2 {
            return Err(WaylandPortalError::ClipboardUnavailable);
        }
        let clipboard = Arc::new(Clipboard::new().await.map_err(map_portal_error)?);
        let session = Arc::new(
            remote
                .create_session(CreateSessionOptions::default())
                .await
                .map_err(map_portal_error)?,
        );

        let (signal_tx, signals) = mpsc::unbounded_channel();
        let clipboard_state = Arc::new(Mutex::new(PortalClipboardState::default()));
        let listener = tokio::spawn({
            let clipboard = Arc::clone(&clipboard);
            let session = Arc::clone(&session);
            let clipboard_state = Arc::clone(&clipboard_state);
            async move {
                listen(&clipboard, &session, &clipboard_state, &signal_tx).await;
                let _ignored = signal_tx.send(ClipboardSignal::Closed);
            }
        });
        let close_session = Arc::clone(&session);
        let output = OutputSession {
            remote,
            clipboard,
            session,
            signals,
            cleanup: CloseOnDrop::new(listener, async move {
                let _ignored = close_session.close().await;
            }),
            clipboard_state,
            advertised_mime_types: None,
        };
        output.start(&self.token_store).await?;
        Ok(output)
    }

    async fn close_output(&mut self) {
        if let Some(output) = self.output.take() {
            output.cleanup.close().await;
        }
    }

    /// Reuses the open session, or opens one. It stays open between pastes:
    /// the clipboard it offers (the transcript or the restored snapshot) is
    /// gone once it closes.
    async fn ensure_output(&mut self) -> Result<&mut OutputSession, WaylandPortalError> {
        if self.output.as_mut().is_some_and(OutputSession::poll_closed) {
            self.close_output().await;
        }
        let output = match self.output.take() {
            Some(output) => output,
            None => self.create_output_session().await?,
        };
        Ok(self.output.insert(output))
    }

    /// Pastes `transcript` into the focused window and restores the clipboard
    /// it replaced when nothing else has taken the clipboard since.
    ///
    /// Cancellation may abandon the paste only before the paste keystroke;
    /// once the transcript is offered, every path ends in the conditional
    /// restore before this returns.
    pub async fn paste(
        &mut self,
        transcript: &str,
        cancellation: &CancellationToken,
    ) -> PasteOutcome<WaylandPortalError> {
        let result = async {
            // Until the transcript is offered, cancellation abandons the work.
            let output = tokio::select! {
                output = self.ensure_output() => output?,
                () = cancellation.cancelled() => return Err(WaylandPortalError::Cancelled),
            };
            // A snapshot failure is never fatal: paste anyway and leave the
            // transcript on the clipboard.
            let snapshot = tokio::select! {
                snapshot = snapshot_clipboard(output) => snapshot
                    .inspect_err(|error| tracing::warn!(%error, "clipboard snapshot failed"))
                    .ok(),
                () = cancellation.cancelled() => return Err(WaylandPortalError::Cancelled),
            };
            set_offer(output, ClipboardOffer::transcript(transcript)).await?;
            let delivered = async {
                // Cancellation may shorten only the wait before the keystroke.
                let keys = tokio::select! {
                    keys = async {
                        serve_for(output, BEFORE_PASTE_DELAY).await?;
                        Ok(active_paste_keys().await)
                    } => keys?,
                    () = cancellation.cancelled() => return Err(WaylandPortalError::Cancelled),
                };
                inject_balanced_paste(output, keys).await?;
                // Once the keystroke is sent, the target always gets the whole
                // restore delay; restoring earlier can make it paste the old value.
                serve_for(output, PASTE_RESTORE_DELAY).await
            }
            .await;
            // A failed delivery still puts the previous clipboard back.
            let restored = restore_if_current(output, snapshot).await;
            delivered.map(|()| restored)
        }
        .await;
        match result {
            Ok(outcome) => outcome,
            Err(WaylandPortalError::Cancelled) => PasteOutcome::Cancelled,
            Err(error) => PasteOutcome::Failed(error),
        }
    }

    /// Opens the `RemoteDesktop` output session the paste keystroke uses, asking
    /// the user for consent unless a saved restore token covers it. Dropping the
    /// future abandons the wait for the user's answer and closes the session.
    pub async fn prepare_output(&mut self) -> Result<(), WaylandPortalError> {
        self.ensure_output().await.map(drop)
    }

    pub async fn shutdown(&mut self) {
        self.close_output().await;
    }
}

async fn restore_if_current(
    output: &mut OutputSession,
    snapshot: Option<ClipboardOffer>,
) -> PasteOutcome<WaylandPortalError> {
    let Some(snapshot) = snapshot else {
        return PasteOutcome::PastedWithoutRestore;
    };
    // The portal won't read back the session's own selection, so ownership
    // tracking stands in for comparing the clipboard with the transcript.
    if !output
        .clipboard_state
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .can_restore()
    {
        return PasteOutcome::RestoreSkippedClipboardChanged;
    }
    match set_offer(output, snapshot).await {
        Ok(()) => PasteOutcome::PastedAndRestored,
        Err(error) => {
            tracing::warn!(%error, "clipboard restore failed");
            PasteOutcome::PastedWithoutRestore
        }
    }
}

async fn snapshot_clipboard(
    output: &mut OutputSession,
) -> Result<ClipboardOffer, WaylandPortalError> {
    let advertised = loop {
        if let Some(advertised) = output.advertised_mime_types.clone() {
            break advertised;
        }
        let signal = tokio::time::timeout(CLIPBOARD_TIMEOUT, output.signals.recv())
            .await
            .map_err(|_| WaylandPortalError::ClipboardUnavailable)?
            .ok_or(WaylandPortalError::Disconnected)?;
        if !output.note(signal) {
            return Err(WaylandPortalError::Disconnected);
        }
    };
    let mut formats = Vec::new();
    for mime_type in SNAPSHOT_MIME_TYPES {
        if advertised
            .iter()
            .any(|advertised| advertised.eq_ignore_ascii_case(mime_type))
            && let Ok(bytes) = read_selection(output, mime_type).await
        {
            formats.push((mime_type.to_owned(), bytes));
        }
    }
    if !advertised.is_empty() && formats.is_empty() {
        return Err(WaylandPortalError::ClipboardUnavailable);
    }
    ClipboardOffer::snapshot(formats)
}

async fn read_selection(
    output: &OutputSession,
    mime_type: &str,
) -> Result<SecretBytes, WaylandPortalError> {
    let fd = output
        .clipboard
        .selection_read(&output.session, mime_type)
        .await
        .map_err(map_portal_error)?;
    let fd: OwnedFd = fd.into();
    let pipe =
        pipe::Receiver::from_owned_fd(fd).map_err(|_| WaylandPortalError::ClipboardUnavailable)?;
    let mut bytes = SecretBytes::default();
    tokio::time::timeout(
        CLIPBOARD_TIMEOUT,
        pipe.take(MAX_FORMAT_BYTES as u64 + 1)
            .read_to_end(&mut bytes),
    )
    .await
    .map_err(|_| WaylandPortalError::ClipboardUnavailable)?
    .map_err(|_| WaylandPortalError::ClipboardUnavailable)?;
    if bytes.len() > MAX_FORMAT_BYTES {
        Err(WaylandPortalError::ClipboardSnapshotTooLarge)
    } else {
        Ok(bytes)
    }
}

async fn set_offer(
    output: &mut OutputSession,
    offer: ClipboardOffer,
) -> Result<(), WaylandPortalError> {
    let mime_types = offer.mime_types();
    output
        .clipboard
        .set_selection(
            &output.session,
            SetSelectionOptions::default().set_mime_types(&mime_types),
        )
        .await
        .map_err(map_portal_error)?;
    output
        .clipboard_state
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .replace_offer(offer);
    Ok(())
}

async fn serve_for(
    output: &mut OutputSession,
    duration: Duration,
) -> Result<(), WaylandPortalError> {
    let deadline = tokio::time::Instant::now() + duration;
    loop {
        match tokio::time::timeout_at(deadline, output.signals.recv()).await {
            Err(_) => return Ok(()),
            Ok(None) => return Err(WaylandPortalError::Disconnected),
            Ok(Some(signal)) => {
                if !output.note(signal) {
                    return Err(WaylandPortalError::Disconnected);
                }
            }
        }
    }
}

/// Forwards clipboard owner changes and serves selection transfers until the
/// session closes or a signal stream ends.
async fn listen(
    clipboard: &Clipboard,
    session: &Session<RemoteDesktop>,
    state: &Mutex<PortalClipboardState>,
    signals: &mpsc::UnboundedSender<ClipboardSignal>,
) {
    let Ok(owner_stream) = clipboard
        .receive_selection_owner_changed::<RemoteDesktop>()
        .await
    else {
        return;
    };
    let Ok(transfer_stream) = clipboard
        .receive_selection_transfer::<RemoteDesktop>()
        .await
    else {
        return;
    };
    let Ok(closed_stream) = session.receive_closed().await else {
        return;
    };
    let owner_stream = owner_stream.fuse();
    let transfer_stream = transfer_stream.fuse();
    let closed_stream = closed_stream.fuse();
    futures_util::pin_mut!(owner_stream, transfer_stream, closed_stream);
    loop {
        tokio::select! {
            owner = owner_stream.next() => {
                let Some((_session, owner)) = owner else { break };
                state
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .owner_changed(owner.session_is_owner());
                let owner_changed = ClipboardSignal::OwnerChanged {
                    mime_types: owner.mime_types().to_vec(),
                };
                if signals.send(owner_changed).is_err() {
                    break;
                }
            }
            transfer = transfer_stream.next() => {
                let Some((_session, mime_type, serial)) = transfer else { break };
                serve_transfer(clipboard, session, state, &mime_type, serial).await;
            }
            _ = closed_stream.next() => break,
        }
    }
}

async fn serve_transfer(
    clipboard: &Clipboard,
    session: &Session<RemoteDesktop>,
    state: &Mutex<PortalClipboardState>,
    mime_type: &str,
    serial: u32,
) {
    let bytes = state
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .transfer(mime_type);
    let Some(bytes) = bytes else {
        let _ignored = clipboard.selection_write_done(session, serial, false).await;
        return;
    };
    let success = match clipboard.selection_write(session, serial).await {
        Ok(fd) => match pipe::Sender::from_owned_fd(fd.into()) {
            Ok(pipe) => write_selection(pipe, &bytes).await,
            Err(_) => false,
        },
        Err(_) => false,
    };
    let _ignored = clipboard
        .selection_write_done(session, serial, success)
        .await;
}

/// Gives up on a reader that does not drain the pipe: transfers run inside the
/// listener, which must keep serving the session's end.
async fn write_selection(mut pipe: pipe::Sender, bytes: &[u8]) -> bool {
    matches!(
        tokio::time::timeout(CLIPBOARD_TIMEOUT, pipe.write_all(bytes)).await,
        Ok(Ok(()))
    )
}

async fn inject_balanced_paste(
    output: &OutputSession,
    keys: PasteKeys,
) -> Result<(), WaylandPortalError> {
    let mut pressed = Vec::with_capacity(3);
    for keysym in match keys {
        PasteKeys::ControlV => vec![XK_CONTROL_L, XK_V_LOWER],
        PasteKeys::ControlShiftV => vec![XK_CONTROL_L, XK_SHIFT_L, XK_V_LOWER],
    } {
        if output
            .remote
            .notify_keyboard_keysym(
                &output.session,
                keysym,
                KeyState::Pressed,
                NotifyKeyboardKeysymOptions::default(),
            )
            .await
            .is_err()
        {
            if !release_pressed(output, &pressed).await {
                let _ignored = output.session.close().await;
            }
            return Err(WaylandPortalError::Backend);
        }
        pressed.push(keysym);
    }
    if release_pressed(output, &pressed).await {
        Ok(())
    } else {
        // Closing the session is the only safe recovery after a partial key
        // release; it prevents a virtual modifier remaining logically held.
        let _ignored = output.session.close().await;
        Err(WaylandPortalError::Backend)
    }
}

async fn release_pressed(output: &OutputSession, pressed: &[i32]) -> bool {
    let mut released = true;
    for keysym in pressed.iter().rev().copied() {
        released &= output
            .remote
            .notify_keyboard_keysym(
                &output.session,
                keysym,
                KeyState::Released,
                NotifyKeyboardKeysymOptions::default(),
            )
            .await
            .is_ok();
    }
    released
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "Result::map_err supplies the owned ashpd error"
)]
fn map_portal_error(error: AshpdError) -> WaylandPortalError {
    match error {
        AshpdError::Response(ashpd::desktop::ResponseError::Cancelled)
        | AshpdError::Portal(
            ashpd::PortalError::NotAllowed(_) | ashpd::PortalError::Cancelled(_),
        ) => WaylandPortalError::PermissionDenied,
        AshpdError::PortalNotFound(_) | AshpdError::RequiresVersion(_, _) => {
            WaylandPortalError::PortalUnavailable
        }
        _ => WaylandPortalError::Backend,
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    struct Counted(Arc<AtomicUsize>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct Parked {
        listener_drops: Arc<AtomicUsize>,
        closes: Arc<AtomicUsize>,
        guard: CloseOnDrop,
    }

    fn parked() -> Parked {
        let listener_drops = Arc::new(AtomicUsize::new(0));
        let closes = Arc::new(AtomicUsize::new(0));
        let held = Counted(Arc::clone(&listener_drops));
        let listener = tokio::spawn(async move {
            let _held = held;
            future::pending::<()>().await;
        });
        let counter = Arc::clone(&closes);
        let guard = CloseOnDrop::new(listener, async move {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        Parked {
            listener_drops,
            closes,
            guard,
        }
    }

    #[tokio::test]
    async fn dropping_a_half_prepared_session_stops_its_listener_and_closes_it() {
        let parked = parked();
        tokio::task::yield_now().await;
        drop(parked.guard);
        tokio::task::yield_now().await;
        assert_eq!(parked.listener_drops.load(Ordering::SeqCst), 1);
        assert_eq!(parked.closes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn closing_explicitly_closes_once() {
        let parked = parked();
        parked.guard.close().await;
        assert_eq!(parked.listener_drops.load(Ordering::SeqCst), 1);
        assert_eq!(parked.closes.load(Ordering::SeqCst), 1);
        tokio::task::yield_now().await;
        assert_eq!(parked.closes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn dropping_outside_a_runtime_stops_the_listener_without_closing() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime");
        let parked = runtime.block_on(async { parked() });
        drop(parked.guard);
        runtime.block_on(tokio::task::yield_now());
        assert_eq!(parked.listener_drops.load(Ordering::SeqCst), 1);
        assert_eq!(parked.closes.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn a_reader_that_never_drains_the_pipe_is_given_up_on() {
        let (sender, _receiver) = pipe::pipe().expect("pipe");
        let bytes = vec![0_u8; 1 << 20];
        let written = tokio::time::timeout(CLIPBOARD_TIMEOUT * 4, write_selection(sender, &bytes))
            .await
            .expect("the write gave up on its own");
        assert!(!written);
    }

    #[tokio::test]
    async fn a_draining_reader_receives_the_whole_selection() {
        let (sender, mut reader) = pipe::pipe().expect("pipe");
        let bytes = vec![7_u8; 1 << 20];
        let mut read_back = Vec::new();
        let (written, read) = tokio::join!(
            write_selection(sender, &bytes),
            reader.read_to_end(&mut read_back),
        );
        assert!(written);
        read.expect("read");
        assert_eq!(read_back, bytes);
    }
}
