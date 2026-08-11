use std::{
    fmt,
    os::fd::OwnedFd,
    path::PathBuf,
    sync::{Arc, Mutex},
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
use futures_util::StreamExt as _;
use tokio::{
    io::{AsyncReadExt as _, AsyncWriteExt as _},
    sync::mpsc,
    task::JoinHandle,
};
use whispernow_core::{CancellationToken, text_output::PasteDelays};

use crate::PasteKeys;

use super::{
    accessibility::active_paste_keys,
    clipboard::{
        MAX_FORMAT_BYTES, PortalClipboardSnapshot, PortalClipboardState, READ_TIMEOUT,
        SNAPSHOT_MIME_TYPES, SecretBytes, TRANSCRIPT_MIME_TYPES,
    },
    token::RemoteDesktopRestoreTokenStore,
};

const XK_CONTROL_L: i32 = 0xffe3;
const XK_SHIFT_L: i32 = 0xffe1;
const XK_V_LOWER: i32 = 0x0076;
const BEFORE_PASTE_DELAY: Duration = Duration::from_millis(20);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaylandPasteOutcome {
    PastedAndRestored,
    /// The previous clipboard could not be read or put back.
    PastedWithoutRestore,
    RestoreSkippedClipboardChanged,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaylandPortalError {
    Backend,
    Cancelled,
    ClipboardSnapshotTooLarge,
    ClipboardUnavailable,
    Disconnected,
    KeyboardUnavailable,
    PermissionDenied,
    PortalUnavailable,
    TokenStore,
}

impl fmt::Display for WaylandPortalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Backend => "desktop portal operation failed",
            Self::Cancelled => "the paste was cancelled",
            Self::ClipboardSnapshotTooLarge => "clipboard snapshot exceeds safe limits",
            Self::ClipboardUnavailable => "clipboard portal is unavailable",
            Self::Disconnected => "desktop portal disconnected",
            Self::KeyboardUnavailable => "remote keyboard capability was not granted",
            Self::PermissionDenied => "desktop portal permission was denied",
            Self::PortalUnavailable => "required desktop portal is unavailable",
            Self::TokenStore => "remote desktop restore token could not be stored safely",
        })
    }
}

impl std::error::Error for WaylandPortalError {}

enum ClipboardSignal {
    OwnerChanged { mime_types: Vec<String> },
    Closed,
}

struct OutputSession {
    remote: Arc<RemoteDesktop>,
    clipboard: Arc<Clipboard>,
    session: Arc<Session<RemoteDesktop>>,
    signals: mpsc::UnboundedReceiver<ClipboardSignal>,
    listener: JoinHandle<()>,
    clipboard_state: Arc<Mutex<PortalClipboardState>>,
    advertised_mime_types: Option<Vec<String>>,
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

    #[allow(
        clippy::too_many_lines,
        reason = "portal consent setup and listener ownership stay together for auditability"
    )]
    async fn create_output_session(&self) -> Result<OutputSession, WaylandPortalError> {
        let remote = Arc::new(RemoteDesktop::new().await.map_err(map_portal_error)?);
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
        let listener_clipboard = Arc::clone(&clipboard);
        let listener_session = Arc::clone(&session);
        let clipboard_state = Arc::new(Mutex::new(PortalClipboardState::default()));
        let listener_state = Arc::clone(&clipboard_state);
        let listener = tokio::spawn(async move {
            let Ok(owner_stream) = listener_clipboard
                .receive_selection_owner_changed::<RemoteDesktop>()
                .await
            else {
                let _ignored = signal_tx.send(ClipboardSignal::Closed);
                return;
            };
            let Ok(transfer_stream) = listener_clipboard
                .receive_selection_transfer::<RemoteDesktop>()
                .await
            else {
                let _ignored = signal_tx.send(ClipboardSignal::Closed);
                return;
            };
            let Ok(closed_stream) = listener_session.receive_closed().await else {
                let _ignored = signal_tx.send(ClipboardSignal::Closed);
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
                        if let Ok(mut state) = listener_state.lock() {
                            state.owner_changed(owner.session_is_owner());
                        }
                        if signal_tx.send(ClipboardSignal::OwnerChanged {
                            mime_types: owner.mime_types().to_vec(),
                        }).is_err() { break; }
                    }
                    transfer = transfer_stream.next() => {
                        let Some((_session, mime_type, serial)) = transfer else { break };
                        serve_transfer(
                            &listener_clipboard,
                            &listener_session,
                            &listener_state,
                            &mime_type,
                            serial,
                        ).await;
                    }
                    closed = closed_stream.next() => {
                        let _ignored = closed;
                        let _ignored = signal_tx.send(ClipboardSignal::Closed);
                        break;
                    }
                }
            }
        });

        let setup = async {
            clipboard
                .request(&session, RequestClipboardOptions::default())
                .await
                .map_err(map_portal_error)?;
            let restore_token = self.token_store.load()?;
            remote
                .select_devices(
                    &session,
                    SelectDevicesOptions::default()
                        .set_devices(Some(DeviceType::Keyboard.into()))
                        .set_persist_mode(PersistMode::ExplicitlyRevoked)
                        .set_restore_token(restore_token.as_deref()),
                )
                .await
                .map_err(map_portal_error)?
                .response()
                .map_err(map_portal_error)?;
            let selected = remote
                .start(&session, None, StartOptions::default())
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
                self.token_store.replace(token)?;
            }
            Ok(())
        }
        .await;
        if let Err(error) = setup {
            listener.abort();
            let _ignored = listener.await;
            let _ignored = session.close().await;
            return Err(error);
        }
        Ok(OutputSession {
            remote,
            clipboard,
            session,
            signals,
            listener,
            clipboard_state,
            advertised_mime_types: None,
        })
    }

    async fn close_output(&mut self) {
        if let Some(output) = self.output.take() {
            output.listener.abort();
            let _ignored = output.listener.await;
            let _ignored = output.session.close().await;
        }
    }

    /// The session stays open: the clipboard it offers, the transcript or the
    /// restored snapshot, is gone once it closes.
    async fn ensure_output(&mut self) -> Result<&mut OutputSession, WaylandPortalError> {
        let closed = self.output.as_mut().is_some_and(|output| {
            let mut closed = false;
            while let Ok(signal) = output.signals.try_recv() {
                match signal {
                    ClipboardSignal::Closed => closed = true,
                    owner @ ClipboardSignal::OwnerChanged { .. } => {
                        apply_non_transfer_signal(output, owner);
                    }
                }
            }
            closed
        });
        if closed {
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
    ///
    /// # Errors
    ///
    /// Returns a content-free portal, clipboard, or permission error when the
    /// paste cannot complete, or [`WaylandPortalError::Cancelled`].
    pub async fn paste(
        &mut self,
        transcript: &str,
        cancellation: &CancellationToken,
    ) -> Result<WaylandPasteOutcome, WaylandPortalError> {
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
        let transcript_offer = TRANSCRIPT_MIME_TYPES
            .iter()
            .map(|mime| (*mime, transcript.as_bytes()))
            .collect();
        set_offer(output, transcript_offer).await?;
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
            serve_for(output, PasteDelays::default().restore()).await
        }
        .await;
        // A failed delivery still puts the previous clipboard back.
        let restored = restore_if_current(output, snapshot).await;
        delivered.map(|()| restored)
    }

    /// Opens the `RemoteDesktop` output session the paste keystroke uses, asking
    /// the user for consent unless a saved restore token covers it. Dropping the
    /// future abandons the wait for the user's answer.
    ///
    /// # Errors
    ///
    /// Returns a content-free portal or permission error.
    pub async fn prepare_output(&mut self) -> Result<(), WaylandPortalError> {
        self.ensure_output().await.map(drop)
    }

    pub async fn shutdown(&mut self) {
        self.close_output().await;
    }
}

async fn restore_if_current(
    output: &mut OutputSession,
    snapshot: Option<PortalClipboardSnapshot>,
) -> WaylandPasteOutcome {
    let Some(snapshot) = snapshot else {
        return WaylandPasteOutcome::PastedWithoutRestore;
    };
    // While the session still owns the selection it holds the transcript; the
    // portal refuses to read a session's own selection back.
    if !output
        .clipboard_state
        .lock()
        .is_ok_and(|state| state.can_restore())
    {
        return WaylandPasteOutcome::RestoreSkippedClipboardChanged;
    }
    match set_offer(output, snapshot.as_offer()).await {
        Ok(()) => WaylandPasteOutcome::PastedAndRestored,
        Err(error) => {
            tracing::warn!(%error, "clipboard restore failed");
            WaylandPasteOutcome::PastedWithoutRestore
        }
    }
}

async fn snapshot_clipboard(
    output: &mut OutputSession,
) -> Result<PortalClipboardSnapshot, WaylandPortalError> {
    if output.advertised_mime_types.is_none() {
        let signal = tokio::time::timeout(READ_TIMEOUT, output.signals.recv())
            .await
            .map_err(|_| WaylandPortalError::ClipboardUnavailable)?
            .ok_or(WaylandPortalError::Disconnected)?;
        apply_non_transfer_signal(output, signal);
    }
    let advertised = output
        .advertised_mime_types
        .clone()
        .ok_or(WaylandPortalError::ClipboardUnavailable)?;
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
    PortalClipboardSnapshot::bounded(formats)
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
    let file = tokio::fs::File::from_std(std::fs::File::from(fd));
    let mut bytes = SecretBytes::default();
    tokio::time::timeout(
        READ_TIMEOUT,
        file.take(u64::try_from(MAX_FORMAT_BYTES).unwrap_or(u64::MAX) + 1)
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
    offer: Vec<(&str, &[u8])>,
) -> Result<(), WaylandPortalError> {
    let mime_types: Vec<&str> = offer.iter().map(|(mime, _)| *mime).collect();
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
        .map(|mut state| state.replace_offer(offer))
        .map_err(|_| WaylandPortalError::Backend)
}

async fn serve_for(
    output: &mut OutputSession,
    duration: Duration,
) -> Result<(), WaylandPortalError> {
    let deadline = tokio::time::Instant::now() + duration;
    loop {
        let now = tokio::time::Instant::now();
        if now >= deadline {
            return Ok(());
        }
        match tokio::time::timeout(deadline - now, output.signals.recv()).await {
            Err(_) => return Ok(()),
            Ok(None | Some(ClipboardSignal::Closed)) => {
                return Err(WaylandPortalError::Disconnected);
            }
            Ok(Some(owner @ ClipboardSignal::OwnerChanged { .. })) => {
                apply_non_transfer_signal(output, owner);
            }
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
        .ok()
        .and_then(|state| state.transfer(mime_type));
    let Some(bytes) = bytes else {
        let _ignored = clipboard.selection_write_done(session, serial, false).await;
        return;
    };
    let success = match clipboard.selection_write(session, serial).await {
        Ok(fd) => {
            let fd: OwnedFd = fd.into();
            let mut file = tokio::fs::File::from_std(std::fs::File::from(fd));
            file.write_all(&bytes).await.is_ok() && file.shutdown().await.is_ok()
        }
        Err(_) => false,
    };
    let _ignored = clipboard
        .selection_write_done(session, serial, success)
        .await;
}

fn apply_non_transfer_signal(output: &mut OutputSession, signal: ClipboardSignal) {
    if let ClipboardSignal::OwnerChanged { mime_types } = signal {
        output.advertised_mime_types = Some(mime_types);
    }
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
