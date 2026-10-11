use std::{
    sync::{Arc, Mutex, mpsc as std_mpsc},
    thread,
    time::Duration,
};

use ashpd::{
    Error as AshpdError, PortalError,
    desktop::{
        CreateSessionOptions, Session,
        global_shortcuts::{BindShortcutsOptions, GlobalShortcuts, NewShortcut},
    },
};
use async_trait::async_trait;
use futures_util::{FutureExt as _, Stream, StreamExt as _, future, stream};
use tokio::{sync::mpsc, time::Instant};

use crate::{
    input_helper::ReleaseWatcher,
    portal_identity::PortalIdentity,
    shortcuts::{
        KeyTransition, ShortcutBinding, ShortcutEvent, ShortcutId, ShortcutRegistrar,
        ShortcutRegistrationError,
    },
};

const COMMAND_TIMEOUT: Duration = Duration::from_mins(2);
const STARTUP_TIMEOUT: Duration = Duration::from_secs(5);
const PORTAL_ID_PREFIX: &str = "whispernow-ptt-";
/// How long a repeating shortcut may go without another repeat before it is
/// taken as released: several repeat intervals, well under the repeat delay.
const REPEAT_RELEASE_TIMEOUT: Duration = Duration::from_millis(250);
/// The compositor can time a last key repeat just after the key's release.
const STALE_REPEAT_SLACK: Duration = Duration::from_millis(50);

pub(crate) fn native_wayland_session() -> bool {
    classify_wayland_session(
        std::env::var("XDG_SESSION_TYPE").ok().as_deref(),
        std::env::var_os("WAYLAND_DISPLAY").is_some(),
    )
}

fn classify_wayland_session(xdg_session_type: Option<&str>, has_wayland_display: bool) -> bool {
    if has_wayland_display {
        return true;
    }
    let Some(session_type) = xdg_session_type else {
        return false;
    };
    session_type.eq_ignore_ascii_case("wayland")
}

enum Command {
    Register {
        binding: ShortcutBinding,
        reply: std_mpsc::SyncSender<Result<(), ShortcutRegistrationError>>,
    },
    Unregister {
        id: ShortcutId,
        reply: std_mpsc::SyncSender<Result<(), ShortcutRegistrationError>>,
    },
    Shutdown {
        reply: std_mpsc::SyncSender<()>,
    },
}

/// One portal key transition with the compositor timestamp that orders it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TimedTransition {
    timestamp: Duration,
    id: ShortcutId,
    transition: KeyTransition,
}

impl TimedTransition {
    /// Timestamp order, with a press before a release at the same instant.
    fn order(&self) -> (Duration, bool) {
        (self.timestamp, self.transition == KeyTransition::Released)
    }
}

// A chord id can be reused while its old session's close signal is still queued.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SessionToken(u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PortalSignal {
    Transition(TimedTransition),
    /// The key release helper saw the shortcut's keys released at this time.
    KeyReleased(ShortcutId, Duration),
    SessionClosed(SessionToken),
    Disconnected,
}

#[async_trait]
trait ShortcutPortal: Send {
    async fn register(
        &mut self,
        binding: &ShortcutBinding,
    ) -> Result<(), ShortcutRegistrationError>;
    async fn unregister(&mut self, id: ShortcutId) -> Result<(), ShortcutRegistrationError>;
    fn session_closed(&mut self, token: SessionToken) -> Option<ShortcutId>;
    async fn shutdown(&mut self);
}

struct BoundSession {
    id: ShortcutId,
    token: SessionToken,
    session: Arc<Session<GlobalShortcuts>>,
}

struct AshpdShortcutPortal {
    proxy: Arc<GlobalShortcuts>,
    session: Option<BoundSession>,
    next_session_token: u64,
    signals: mpsc::UnboundedSender<PortalSignal>,
    releases: ReleaseWatcher,
}

impl AshpdShortcutPortal {
    async fn connect()
    -> Result<(Self, mpsc::UnboundedReceiver<PortalSignal>), ShortcutRegistrationError> {
        let proxy = Arc::new(GlobalShortcuts::new().await.map_err(map_portal_error)?);
        let (signals, receiver) = mpsc::unbounded_channel();
        let transition_signals = signals.clone();
        let transition_proxy = Arc::clone(&proxy);
        tokio::spawn(async move {
            let (Ok(activated), Ok(deactivated)) = (
                transition_proxy.receive_activated().await,
                transition_proxy.receive_deactivated().await,
            ) else {
                let _ = transition_signals.send(PortalSignal::Disconnected);
                return;
            };
            let presses = activated.filter_map(|event| {
                future::ready(
                    parse_portal_id(event.shortcut_id()).map(|id| TimedTransition {
                        timestamp: event.timestamp(),
                        id,
                        transition: KeyTransition::Pressed,
                    }),
                )
            });
            let releases = deactivated.filter_map(|event| {
                future::ready(
                    parse_portal_id(event.shortcut_id()).map(|id| TimedTransition {
                        timestamp: event.timestamp(),
                        id,
                        transition: KeyTransition::Released,
                    }),
                )
            });
            forward_ordered_transitions(presses, releases, &transition_signals).await;
        });
        let release_signals = signals.clone();
        let releases = ReleaseWatcher::new(Arc::new(move |id, at| {
            let _ = release_signals.send(PortalSignal::KeyReleased(id, at));
        }));
        Ok((
            Self {
                proxy,
                session: None,
                next_session_token: 1,
                signals,
                releases,
            },
            receiver,
        ))
    }
}

#[async_trait]
impl ShortcutPortal for AshpdShortcutPortal {
    async fn register(
        &mut self,
        binding: &ShortcutBinding,
    ) -> Result<(), ShortcutRegistrationError> {
        if let Some(session) = &self.session {
            assert_eq!(
                session.id, binding.id,
                "one Wayland shortcut may be registered"
            );
            return Ok(());
        }
        let session = self
            .proxy
            .create_session(CreateSessionOptions::default())
            .await
            .map_err(map_portal_error)?;
        let portal_id = portal_id(binding.id);
        let shortcut = NewShortcut::new(&portal_id, "Record with WhisperNow")
            .preferred_trigger(Some(binding.portal_trigger().as_str()));
        let request = match self
            .proxy
            .bind_shortcuts(&session, &[shortcut], None, BindShortcutsOptions::default())
            .await
        {
            Ok(request) => request,
            Err(error) => {
                let _ = session.close().await;
                return Err(map_portal_error(error));
            }
        };
        let response = match request.response() {
            Ok(response) => response,
            Err(error) => {
                let _ = session.close().await;
                return Err(map_portal_error(error));
            }
        };
        if !response
            .shortcuts()
            .iter()
            .any(|bound| bound.id() == portal_id)
        {
            let _ = session.close().await;
            return Err(ShortcutRegistrationError::PermissionDenied);
        }
        let session = Arc::new(session);
        let closed_session = Arc::clone(&session);
        let closed_signals = self.signals.clone();
        let token = SessionToken(self.next_session_token);
        self.next_session_token += 1;
        tokio::spawn(async move {
            let Ok(mut closed) = closed_session.receive_closed().await else {
                let _ = closed_signals.send(PortalSignal::Disconnected);
                return;
            };
            if closed.next().await.is_some() {
                let _ = closed_signals.send(PortalSignal::SessionClosed(token));
            }
        });
        self.session = Some(BoundSession {
            id: binding.id,
            token,
            session,
        });
        self.releases.watch(binding);
        Ok(())
    }

    async fn unregister(&mut self, id: ShortcutId) -> Result<(), ShortcutRegistrationError> {
        let Some(bound) = self.session.take_if(|bound| bound.id == id) else {
            return Ok(());
        };
        self.releases.unwatch(id);
        bound.session.close().await.map_err(map_portal_error)
    }

    fn session_closed(&mut self, token: SessionToken) -> Option<ShortcutId> {
        let bound = self.session.take_if(|bound| bound.token == token)?;
        self.releases.unwatch(bound.id);
        Some(bound.id)
    }

    async fn shutdown(&mut self) {
        self.releases.stop();
        if let Some(bound) = self.session.take() {
            let _ = bound.session.close().await;
        }
    }
}

/// Synchronous registrar façade over the long-lived portal actor.
pub(crate) struct WaylandShortcutRegistrar {
    commands: mpsc::UnboundedSender<Command>,
    worker: Mutex<Option<thread::JoinHandle<()>>>,
}

impl WaylandShortcutRegistrar {
    /// Registers the portal identity first: the shortcut actor makes this
    /// process's first portal call, and the output portal waits for a paste or
    /// a setup consent request, both of which follow startup.
    pub(crate) fn spawn(
        identity: PortalIdentity,
        events: mpsc::UnboundedSender<ShortcutEvent>,
    ) -> (Arc<Self>, Option<ShortcutRegistrationError>) {
        let (commands, receiver) = mpsc::unbounded_channel();
        let (ready, ready_receiver) = std_mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("whispernow-wayland-shortcuts".to_owned())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build();
                let Ok(runtime) = runtime else {
                    let error = ShortcutRegistrationError::PortalUnavailable;
                    let _ = ready.send(Err(error));
                    return;
                };
                runtime.block_on(async move {
                    identity.register().await;
                    match AshpdShortcutPortal::connect().await {
                        Ok((portal, signals)) => {
                            let _ = ready.send(Ok(()));
                            actor_loop(portal, receiver, signals, events).await;
                        }
                        Err(error) => {
                            let _ = ready.send(Err(error));
                        }
                    }
                });
            })
            .ok();
        let startup_failure = match ready_receiver.recv_timeout(STARTUP_TIMEOUT) {
            Ok(result) => result.err(),
            Err(_) => Some(ShortcutRegistrationError::PortalUnavailable),
        };
        (
            Arc::new(Self {
                commands,
                worker: Mutex::new(worker),
            }),
            startup_failure,
        )
    }

    fn request(
        &self,
        make_command: impl FnOnce(
            std_mpsc::SyncSender<Result<(), ShortcutRegistrationError>>,
        ) -> Command,
    ) -> Result<(), ShortcutRegistrationError> {
        let (reply, receiver) = std_mpsc::sync_channel(1);
        self.commands
            .send(make_command(reply))
            .map_err(|_| ShortcutRegistrationError::PortalUnavailable)?;
        receiver
            .recv_timeout(COMMAND_TIMEOUT)
            .unwrap_or(Err(ShortcutRegistrationError::PortalUnavailable))
    }
}

impl ShortcutRegistrar for WaylandShortcutRegistrar {
    fn register(&self, binding: &ShortcutBinding) -> Result<(), ShortcutRegistrationError> {
        self.request(|reply| Command::Register {
            binding: binding.clone(),
            reply,
        })
    }

    fn unregister(&self, binding: &ShortcutBinding) -> Result<(), ShortcutRegistrationError> {
        self.request(|reply| Command::Unregister {
            id: binding.id,
            reply,
        })
    }

    /// Each registration opens a portal session the compositor may ask the user
    /// to approve, so the cancel shortcut is not registered for every dictation;
    /// the overlay's cancel button and the tray cancel instead. A second Wayland shortcut
    /// would require restoring support for multiple portal sessions and held shortcuts.
    fn supports_transient_shortcuts(&self) -> bool {
        false
    }

    fn shutdown(&self) {
        let mut worker = self.worker.lock().expect("shortcut worker lock poisoned");
        let Some(worker) = worker.take() else {
            return;
        };
        let (reply, receiver) = std_mpsc::sync_channel(1);
        let _ = self.commands.send(Command::Shutdown { reply });
        let _ = receiver.recv_timeout(Duration::from_secs(5));
        let _ = worker.join();
    }
}

impl Drop for WaylandShortcutRegistrar {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// GNOME reports each key repeat of a held shortcut as another `Activated`,
/// the first after the repeat delay and then at the repeat interval, and it
/// sometimes drops the `Deactivated` that ends the hold. Only the first
/// activation is forwarded as a press. The key release helper, when installed,
/// reports the release itself; otherwise, once a shortcut has repeated, a
/// pause in its repeats is taken as its release.
#[derive(Default)]
struct HeldShortcut {
    held: Option<Hold>,
    key_release: Option<KeyRelease>,
}

struct Hold {
    /// The portal timestamp of the press.
    pressed_at: Duration,
    /// The release deadline, set once the shortcut repeats.
    deadline: Option<Instant>,
}

struct KeyRelease {
    at: Duration,
    /// Whether it ended a hold, whose activations still queued are then stale.
    ended_hold: bool,
}

impl HeldShortcut {
    fn key_signal(&mut self, signal: PortalSignal, now: Instant) -> Vec<KeyTransition> {
        let released = match signal {
            PortalSignal::Transition(TimedTransition {
                timestamp,
                transition: KeyTransition::Pressed,
                ..
            }) => return self.activated(timestamp, now),
            PortalSignal::Transition(_) => self.deactivated(),
            PortalSignal::KeyReleased(_, at) => self.key_released(at),
            PortalSignal::SessionClosed(_) | PortalSignal::Disconnected => false,
        };
        if released {
            vec![KeyTransition::Released]
        } else {
            Vec::new()
        }
    }

    /// The transitions an activation at portal time `at` stands for: a press,
    /// nothing for a repeat, or a press and its release when the helper saw a
    /// tap released before the portal reported its press.
    fn activated(&mut self, at: Duration, now: Instant) -> Vec<KeyTransition> {
        if let Some(hold) = &mut self.held {
            hold.deadline = Some(now + REPEAT_RELEASE_TIMEOUT);
            return Vec::new();
        }
        match &mut self.key_release {
            Some(release) if release.ended_hold && at <= release.at + STALE_REPEAT_SLACK => {
                Vec::new()
            }
            Some(release) if at <= release.at => {
                release.ended_hold = true;
                vec![KeyTransition::Pressed, KeyTransition::Released]
            }
            _ => {
                self.held = Some(Hold {
                    pressed_at: at,
                    deadline: None,
                });
                vec![KeyTransition::Pressed]
            }
        }
    }

    /// Whether the helper's release at `at` ends a hold, so it is news.
    fn key_released(&mut self, at: Duration) -> bool {
        let ended_hold = self.held.as_ref().is_some_and(|hold| hold.pressed_at <= at);
        if ended_hold {
            self.held = None;
        }
        self.key_release = Some(KeyRelease { at, ended_hold });
        ended_hold
    }

    /// Whether the shortcut was still held, so the portal's release is news.
    fn deactivated(&mut self) -> bool {
        self.held.take().is_some()
    }

    fn forget(&mut self) {
        self.held = None;
        self.key_release = None;
    }

    fn next_deadline(&self) -> Option<Instant> {
        self.held.as_ref().and_then(|hold| hold.deadline)
    }

    fn expire(&mut self, now: Instant) -> bool {
        self.held
            .take_if(|hold| hold.deadline.is_some_and(|deadline| deadline <= now))
            .is_some()
    }
}

async fn sleep_until(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => future::pending().await,
    }
}

async fn actor_loop<P: ShortcutPortal>(
    mut portal: P,
    mut commands: mpsc::UnboundedReceiver<Command>,
    mut signals: mpsc::UnboundedReceiver<PortalSignal>,
    events: mpsc::UnboundedSender<ShortcutEvent>,
) {
    let mut active = None;
    let mut held = HeldShortcut::default();
    loop {
        tokio::select! {
            () = sleep_until(held.next_deadline()) => {
                if held.expire(Instant::now()) && let Some(id) = active {
                    let _ = events.send(ShortcutEvent::Transition {
                        id,
                        transition: KeyTransition::Released,
                    });
                }
            }
            command = commands.recv() => {
                let Some(command) = command else { break; };
                match command {
                    Command::Register { binding, reply } => {
                        let id = binding.id;
                        let result = portal.register(&binding).await;
                        if result.is_ok() && active != Some(id) {
                            held.forget();
                            active = Some(id);
                        }
                        let _ = reply.send(result);
                    }
                    Command::Unregister { id, reply } => {
                        let result = portal.unregister(id).await;
                        if result.is_ok() && active == Some(id) {
                            active = None;
                            held.forget();
                        }
                        let _ = reply.send(result);
                    }
                    Command::Shutdown { reply } => {
                        portal.shutdown().await;
                        let _ = reply.send(());
                        break;
                    }
                }
            }
            signal = signals.recv() => {
                let Some(signal) = signal else {
                    notify_capability_lost(
                        &events,
                        ShortcutRegistrationError::PortalUnavailable,
                    );
                    break;
                };
                match signal {
                    PortalSignal::Transition(TimedTransition { id, .. })
                    | PortalSignal::KeyReleased(id, _)
                        if active == Some(id) =>
                    {
                        for transition in held.key_signal(signal, Instant::now()) {
                            let _ = events.send(ShortcutEvent::Transition { id, transition });
                        }
                    }
                    PortalSignal::SessionClosed(token) => {
                        if let Some(id) = portal.session_closed(token) && active == Some(id) {
                            active = None;
                            held.forget();
                            notify_capability_lost(
                                &events,
                                ShortcutRegistrationError::SessionClosed,
                            );
                        }
                    }
                    PortalSignal::Transition(_)
                    | PortalSignal::KeyReleased(_, _) => {}
                    PortalSignal::Disconnected => {
                        notify_capability_lost(
                            &events,
                            ShortcutRegistrationError::PortalUnavailable,
                        );
                        portal.shutdown().await;
                        break;
                    }
                }
            }
        }
    }
    portal.shutdown().await;
}

/// Forwards the portal's separate `Activated` and `Deactivated` signal streams,
/// given in either order, as one ordered sequence. Every transition already
/// queued is sorted by its compositor timestamp before it is forwarded, so a
/// fast tap cannot deliver its release ahead of its press. Reports the portal
/// as disconnected when either stream ends.
async fn forward_ordered_transitions(
    first: impl Stream<Item = TimedTransition>,
    second: impl Stream<Item = TimedTransition>,
    signals: &mpsc::UnboundedSender<PortalSignal>,
) {
    // `None` marks the end of either signal stream.
    let merged = stream::select(
        first.map(Some).chain(stream::once(future::ready(None))),
        second.map(Some).chain(stream::once(future::ready(None))),
    );
    let mut merged = std::pin::pin!(merged);
    let mut batch = Vec::new();
    while let Some(Some(next)) = merged.next().await {
        batch.push(next);
        let mut ended = false;
        while let Some(queued) = merged.next().now_or_never() {
            let Some(transition) = queued.flatten() else {
                ended = true;
                break;
            };
            batch.push(transition);
        }
        batch.sort_by_key(TimedTransition::order);
        for transition in batch.drain(..) {
            if signals.send(PortalSignal::Transition(transition)).is_err() {
                return;
            }
        }
        if ended {
            break;
        }
    }
    let _ = signals.send(PortalSignal::Disconnected);
}

fn notify_capability_lost(
    events: &mpsc::UnboundedSender<ShortcutEvent>,
    error: ShortcutRegistrationError,
) {
    let _ = events.send(ShortcutEvent::CapabilityLost(error));
}

fn portal_id(id: ShortcutId) -> String {
    format!("{PORTAL_ID_PREFIX}{:08x}", id.0)
}

fn parse_portal_id(value: &str) -> Option<ShortcutId> {
    value
        .strip_prefix(PORTAL_ID_PREFIX)
        .and_then(|value| u32::from_str_radix(value, 16).ok())
        .map(ShortcutId)
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "map_err supplies the owned portal error; it is logged here and only its category crosses the boundary"
)]
fn map_portal_error(error: AshpdError) -> ShortcutRegistrationError {
    tracing::warn!(%error, "shortcut portal request failed");
    match error {
        AshpdError::Response(_)
        | AshpdError::Portal(PortalError::NotAllowed(_) | PortalError::Cancelled(_)) => {
            ShortcutRegistrationError::PermissionDenied
        }
        _ => ShortcutRegistrationError::PortalUnavailable,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    #[derive(Default)]
    struct FakePortal {
        registered: Arc<Mutex<Option<(ShortcutId, SessionToken)>>>,
        next_session_token: u64,
        unregister_error: Option<ShortcutRegistrationError>,
        deny: Arc<AtomicBool>,
        stopped: Arc<AtomicBool>,
    }

    #[async_trait]
    impl ShortcutPortal for FakePortal {
        async fn register(
            &mut self,
            binding: &ShortcutBinding,
        ) -> Result<(), ShortcutRegistrationError> {
            if self.deny.load(Ordering::Acquire) {
                return Err(ShortcutRegistrationError::PermissionDenied);
            }
            let mut registered = self.registered.lock().expect("registered");
            if let Some((id, _)) = *registered {
                assert_eq!(id, binding.id, "one Wayland shortcut may be registered");
                return Ok(());
            }
            self.next_session_token += 1;
            *registered = Some((binding.id, SessionToken(self.next_session_token)));
            Ok(())
        }

        async fn unregister(&mut self, id: ShortcutId) -> Result<(), ShortcutRegistrationError> {
            let _ = self
                .registered
                .lock()
                .expect("registered")
                .take_if(|(registered, _)| *registered == id);
            self.unregister_error.map_or(Ok(()), Err)
        }

        fn session_closed(&mut self, token: SessionToken) -> Option<ShortcutId> {
            self.registered
                .lock()
                .expect("registered")
                .take_if(|(_, registered)| *registered == token)
                .map(|(id, _)| id)
        }

        async fn shutdown(&mut self) {
            *self.registered.lock().expect("registered") = None;
            self.stopped.store(true, Ordering::Release);
        }
    }

    fn binding(id: u32) -> ShortcutBinding {
        ShortcutBinding {
            id: ShortcutId(id),
            ..crate::shortcuts::binding(&whispernow_domain::Shortcut::default())
        }
    }

    #[test]
    fn a_stopped_actor_reports_the_portal_unavailable() {
        let (commands, receiver) = mpsc::unbounded_channel();
        drop(receiver);
        let registrar = WaylandShortcutRegistrar {
            commands,
            worker: Mutex::new(None),
        };
        assert_eq!(
            registrar.register(&binding(1)),
            Err(ShortcutRegistrationError::PortalUnavailable)
        );
        assert_eq!(
            registrar.unregister(&binding(1)),
            Err(ShortcutRegistrationError::PortalUnavailable)
        );

        let coordinator = crate::shortcuts::ShortcutCoordinator::new(
            Arc::new(registrar),
            Ok(crate::shortcuts::RecordingBinding {
                mode: whispernow_domain::RecordingMode::Hold,
                binding: binding(1),
            }),
            Arc::new(|| true),
            Some(ShortcutRegistrationError::PermissionDenied),
        );
        assert!(!coordinator.registration_active());
        assert_eq!(
            coordinator.failure(),
            Some(ShortcutRegistrationError::PermissionDenied)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn denial_and_session_replacement_are_explicit() {
        let deny = Arc::new(AtomicBool::new(true));
        let portal = FakePortal {
            deny: Arc::clone(&deny),
            ..FakePortal::default()
        };
        let (commands, command_rx) = mpsc::unbounded_channel();
        let (signals, signal_rx) = mpsc::unbounded_channel();
        let (events, mut event_rx) = mpsc::unbounded_channel();
        let actor = tokio::spawn(actor_loop(portal, command_rx, signal_rx, events));

        let (reply, response) = std_mpsc::sync_channel(1);
        commands
            .send(Command::Register {
                binding: binding(1),
                reply,
            })
            .expect("register");
        assert_eq!(
            tokio::task::spawn_blocking(move || response.recv().expect("reply"))
                .await
                .expect("join"),
            Err(ShortcutRegistrationError::PermissionDenied)
        );
        deny.store(false, Ordering::Release);
        let (reply, response) = std_mpsc::sync_channel(1);
        commands
            .send(Command::Register {
                binding: binding(1),
                reply,
            })
            .expect("retry");
        tokio::task::spawn_blocking(move || response.recv().expect("reply"))
            .await
            .expect("join")
            .expect("registered");
        signals
            .send(PortalSignal::SessionClosed(SessionToken(1)))
            .expect("closed");
        assert_eq!(
            event_rx.recv().await,
            Some(ShortcutEvent::CapabilityLost(
                ShortcutRegistrationError::SessionClosed
            ))
        );
        for (id, register) in [(2, true), (2, false), (2, true)] {
            let (reply, response) = std_mpsc::sync_channel(1);
            let command = if register {
                Command::Register {
                    binding: binding(id),
                    reply,
                }
            } else {
                Command::Unregister {
                    id: ShortcutId(id),
                    reply,
                }
            };
            commands.send(command).expect("replace binding");
            tokio::task::spawn_blocking(move || response.recv().expect("binding reply"))
                .await
                .expect("reply worker")
                .expect("binding changed");
        }
        signals
            .send(PortalSignal::Transition(timed(
                1000,
                2,
                KeyTransition::Pressed,
            )))
            .expect("new session press");
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), event_rx.recv())
                .await
                .expect("press in time"),
            Some(ShortcutEvent::Transition {
                id: ShortcutId(2),
                transition: KeyTransition::Pressed,
            })
        );
        signals
            .send(PortalSignal::SessionClosed(SessionToken(2)))
            .expect("old session closed");
        signals
            .send(PortalSignal::Transition(timed(
                1100,
                2,
                KeyTransition::Released,
            )))
            .expect("new session release");
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), event_rx.recv())
                .await
                .expect("release in time"),
            Some(ShortcutEvent::Transition {
                id: ShortcutId(2),
                transition: KeyTransition::Released,
            })
        );
        drop(commands);
        drop(signals);
        actor.await.expect("actor");
    }

    #[tokio::test]
    async fn a_release_from_the_old_binding_cannot_end_the_new_hold() {
        let stopped = Arc::new(AtomicBool::new(false));
        let portal = FakePortal {
            registered: Arc::new(Mutex::new(None)),
            deny: Arc::new(AtomicBool::new(false)),
            stopped: Arc::clone(&stopped),
            ..FakePortal::default()
        };
        let (commands, command_rx) = mpsc::unbounded_channel();
        let (signals, signal_rx) = mpsc::unbounded_channel();
        let (events, mut event_rx) = mpsc::unbounded_channel();
        let actor = tokio::spawn(actor_loop(portal, command_rx, signal_rx, events));
        for (id, register) in [(1, true), (1, false), (2, true)] {
            let (reply, response) = std_mpsc::sync_channel(1);
            let command = if register {
                Command::Register {
                    binding: binding(id),
                    reply,
                }
            } else {
                Command::Unregister {
                    id: ShortcutId(id),
                    reply,
                }
            };
            commands.send(command).expect("binding command");
            tokio::task::spawn_blocking(move || response.recv().expect("binding reply"))
                .await
                .expect("reply worker")
                .expect("binding changed");
        }
        signals
            .send(PortalSignal::Transition(timed(
                1000,
                2,
                KeyTransition::Pressed,
            )))
            .expect("new press");
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), event_rx.recv())
                .await
                .expect("press in time"),
            Some(ShortcutEvent::Transition {
                id: ShortcutId(2),
                transition: KeyTransition::Pressed
            })
        );
        signals
            .send(PortalSignal::KeyReleased(
                ShortcutId(1),
                Duration::from_millis(1100),
            ))
            .expect("old helper release");
        signals
            .send(PortalSignal::SessionClosed(SessionToken(1)))
            .expect("old session closed");
        signals
            .send(PortalSignal::Transition(timed(
                1200,
                2,
                KeyTransition::Released,
            )))
            .expect("new release");
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), event_rx.recv())
                .await
                .expect("release in time"),
            Some(ShortcutEvent::Transition {
                id: ShortcutId(2),
                transition: KeyTransition::Released
            })
        );
        drop(commands);
        actor.await.expect("actor");
        assert!(stopped.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn a_failed_unregister_does_not_carry_the_old_hold_into_the_new_binding() {
        let portal = FakePortal {
            unregister_error: Some(ShortcutRegistrationError::PortalUnavailable),
            ..FakePortal::default()
        };
        let (commands, command_rx) = mpsc::unbounded_channel();
        let (signals, signal_rx) = mpsc::unbounded_channel();
        let (events, mut event_rx) = mpsc::unbounded_channel();
        let actor = tokio::spawn(actor_loop(portal, command_rx, signal_rx, events));
        for id in [1, 2] {
            let (reply, response) = std_mpsc::sync_channel(1);
            commands
                .send(Command::Register {
                    binding: binding(id),
                    reply,
                })
                .expect("register");
            tokio::task::spawn_blocking(move || response.recv().expect("register reply"))
                .await
                .expect("reply worker")
                .expect("registered");
            signals
                .send(PortalSignal::Transition(timed(
                    1000 * u64::from(id),
                    id,
                    KeyTransition::Pressed,
                )))
                .expect("press");
            assert_eq!(
                tokio::time::timeout(Duration::from_secs(2), event_rx.recv())
                    .await
                    .expect("press in time"),
                Some(ShortcutEvent::Transition {
                    id: ShortcutId(id),
                    transition: KeyTransition::Pressed,
                })
            );
            if id == 1 {
                let (reply, response) = std_mpsc::sync_channel(1);
                commands
                    .send(Command::Unregister {
                        id: ShortcutId(id),
                        reply,
                    })
                    .expect("unregister");
                assert_eq!(
                    tokio::task::spawn_blocking(move || response.recv().expect("unregister reply"))
                        .await
                        .expect("reply worker"),
                    Err(ShortcutRegistrationError::PortalUnavailable)
                );
            }
        }
        drop(commands);
        actor.await.expect("actor");
    }

    #[test]
    fn repeats_are_not_presses_and_a_pause_in_them_releases() {
        let start = Instant::now();
        let at = Duration::from_millis;
        let mut held = HeldShortcut::default();

        assert_eq!(held.activated(at(0), start), [KeyTransition::Pressed]);
        // A press alone waits for a release: there is no repeat yet.
        assert_eq!(held.next_deadline(), None);
        let repeat = start + Duration::from_millis(500);
        assert!(held.activated(at(500), repeat).is_empty());
        assert_eq!(held.next_deadline(), Some(repeat + REPEAT_RELEASE_TIMEOUT));
        assert!(!held.expire(repeat + Duration::from_millis(30)));

        assert!(held.expire(repeat + REPEAT_RELEASE_TIMEOUT));
        // The portal's late release changes nothing, and the next press is new.
        assert!(!held.deactivated());
        assert_eq!(held.activated(at(2000), repeat), [KeyTransition::Pressed]);
        assert!(held.deactivated());
    }

    #[test]
    fn the_helper_ends_a_hold_and_its_queued_repeats_are_stale() {
        let now = Instant::now();
        let at = Duration::from_millis;
        let mut held = HeldShortcut::default();

        assert_eq!(held.activated(at(1000), now), [KeyTransition::Pressed]);
        assert!(held.activated(at(1500), now).is_empty());
        assert!(held.key_released(at(1600)));
        assert_eq!(held.next_deadline(), None);
        // Repeats timed before, or just after, the release were already queued.
        assert!(held.activated(at(1590), now).is_empty());
        assert!(held.activated(at(1620), now).is_empty());
        assert!(!held.deactivated());
        assert_eq!(held.activated(at(3000), now), [KeyTransition::Pressed]);
    }

    #[test]
    fn a_tap_released_before_its_press_arrives_is_a_press_and_a_release() {
        let now = Instant::now();
        let at = Duration::from_millis;
        let mut held = HeldShortcut::default();

        assert!(!held.key_released(at(1080)));
        assert_eq!(
            held.activated(at(1000), now),
            [KeyTransition::Pressed, KeyTransition::Released]
        );
        assert!(!held.deactivated());
        assert_eq!(held.activated(at(2000), now), [KeyTransition::Pressed]);
        // A release reported for an earlier hold does not end this one.
        assert!(!held.key_released(at(1990)));
        assert!(held.key_released(at(2100)));
    }

    #[tokio::test]
    async fn a_hold_whose_release_is_dropped_still_ends_when_its_repeats_stop() {
        let portal = FakePortal {
            registered: Arc::new(Mutex::new(None)),
            deny: Arc::new(AtomicBool::new(false)),
            stopped: Arc::new(AtomicBool::new(false)),
            ..FakePortal::default()
        };
        let (commands, command_rx) = mpsc::unbounded_channel();
        let (signals, signal_rx) = mpsc::unbounded_channel();
        let (events, mut event_rx) = mpsc::unbounded_channel();
        let actor = tokio::spawn(actor_loop(portal, command_rx, signal_rx, events));
        let (reply, response) = std_mpsc::sync_channel(1);
        commands
            .send(Command::Register {
                binding: binding(1),
                reply,
            })
            .expect("register");
        tokio::task::spawn_blocking(move || response.recv().expect("reply"))
            .await
            .expect("join")
            .expect("registered");

        for millis in [0, 500, 530] {
            signals
                .send(PortalSignal::Transition(timed(
                    millis,
                    1,
                    KeyTransition::Pressed,
                )))
                .expect("activation");
        }
        let mut received = Vec::new();
        for _ in 0..2 {
            received.push(
                tokio::time::timeout(Duration::from_secs(2), event_rx.recv())
                    .await
                    .expect("event in time"),
            );
        }
        assert_eq!(
            received,
            [
                Some(ShortcutEvent::Transition {
                    id: ShortcutId(1),
                    transition: KeyTransition::Pressed,
                }),
                Some(ShortcutEvent::Transition {
                    id: ShortcutId(1),
                    transition: KeyTransition::Released,
                }),
            ]
        );
        drop(commands);
        drop(signals);
        actor.await.expect("actor");
    }

    fn timed(millis: u64, id: u32, transition: KeyTransition) -> TimedTransition {
        TimedTransition {
            timestamp: Duration::from_millis(millis),
            id: ShortcutId(id),
            transition,
        }
    }

    #[tokio::test]
    async fn queued_portal_transitions_are_forwarded_in_timestamp_order() {
        let presses = stream::iter([
            timed(10, 1, KeyTransition::Pressed),
            timed(30, 1, KeyTransition::Pressed),
        ]);
        let releases = stream::iter([
            timed(20, 1, KeyTransition::Released),
            timed(40, 1, KeyTransition::Released),
        ]);
        let (signals, mut received) = mpsc::unbounded_channel();

        // The release stream is serviced first, as an independent task could be.
        forward_ordered_transitions(releases, presses, &signals).await;

        let mut forwarded = Vec::new();
        while let Ok(signal) = received.try_recv() {
            forwarded.push(signal);
        }
        assert_eq!(
            forwarded,
            [
                PortalSignal::Transition(timed(10, 1, KeyTransition::Pressed)),
                PortalSignal::Transition(timed(20, 1, KeyTransition::Released)),
                PortalSignal::Transition(timed(30, 1, KeyTransition::Pressed)),
                PortalSignal::Transition(timed(40, 1, KeyTransition::Released)),
                PortalSignal::Disconnected,
            ]
        );
    }

    #[tokio::test]
    async fn a_release_with_the_same_timestamp_follows_its_press() {
        let (signals, mut received) = mpsc::unbounded_channel();

        forward_ordered_transitions(
            stream::iter([timed(7, 3, KeyTransition::Released)]),
            stream::iter([timed(7, 3, KeyTransition::Pressed)]),
            &signals,
        )
        .await;

        assert_eq!(
            received.try_recv(),
            Ok(PortalSignal::Transition(timed(
                7,
                3,
                KeyTransition::Pressed
            )))
        );
        assert_eq!(
            received.try_recv(),
            Ok(PortalSignal::Transition(timed(
                7,
                3,
                KeyTransition::Released
            )))
        );
    }

    #[test]
    fn portal_ids_are_bounded_and_round_trip() {
        let id = ShortcutId(u32::MAX);
        assert_eq!(parse_portal_id(&portal_id(id)), Some(id));
        assert_eq!(parse_portal_id("another-app"), None);
    }

    #[test]
    fn native_wayland_wins_even_when_xwayland_is_also_available() {
        assert!(classify_wayland_session(Some("wayland"), true));
        assert!(classify_wayland_session(Some("x11"), true));
        assert!(!classify_wayland_session(Some("x11"), false));
        assert!(!classify_wayland_session(None, false));
    }
}
