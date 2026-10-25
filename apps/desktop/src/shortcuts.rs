use std::{
    collections::HashSet,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
};

use tauri_plugin_global_shortcut::{GlobalShortcutExt as _, Shortcut as GlobalShortcut};
use whispernow_domain::{RecordingMode, Settings, Shortcut, ShortcutModifier};

use crate::services::RecordingTrigger;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ShortcutId(pub(crate) u32);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ShortcutBinding {
    pub(crate) id: ShortcutId,
    pub(crate) global: GlobalShortcut,
    pub(crate) shortcut: Shortcut,
}

impl ShortcutBinding {
    /// The trigger a desktop portal is asked for, such as `CTRL+space`.
    #[cfg_attr(not(target_os = "linux"), expect(dead_code))]
    pub(crate) fn portal_trigger(&self) -> String {
        self.shortcut
            .modifiers
            .iter()
            .map(|modifier| modifier.portal_name())
            .chain([self.shortcut.key.keysym()])
            .collect::<Vec<_>>()
            .join("+")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KeyTransition {
    Pressed,
    Released,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RecordingBinding {
    pub(crate) mode: RecordingMode,
    pub(crate) binding: ShortcutBinding,
}

fn mode_for(recording: Option<&RecordingBinding>, id: ShortcutId) -> Option<RecordingMode> {
    recording
        .filter(|recording| recording.binding.id == id)
        .map(|recording| recording.mode)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ShortcutEvent {
    Transition {
        id: ShortcutId,
        transition: KeyTransition,
    },
    #[cfg_attr(not(target_os = "linux"), expect(dead_code))]
    CapabilityLost(ShortcutRegistrationError),
}

/// The worker decides from its own recording whether a press starts or stops one and whether a
/// release stops it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ShortcutAction {
    Pressed(RecordingTrigger),
    HoldReleased,
    CancelDictation,
}

pub(crate) struct StartedRegistrar {
    pub(crate) registrar: Arc<dyn ShortcutRegistrar>,
    pub(crate) failure: Option<ShortcutRegistrationError>,
}

pub(crate) trait ShortcutRegistrar: Send + Sync {
    fn register(&self, binding: &ShortcutBinding) -> Result<(), ShortcutRegistrationError>;
    fn unregister(&self, binding: &ShortcutBinding) -> Result<(), ShortcutRegistrationError>;

    /// False for a desktop portal that asks the user to approve every registration.
    fn supports_transient_shortcuts(&self) -> bool {
        true
    }

    fn shutdown(&self) {}
}

#[derive(Clone)]
pub(crate) struct TauriShortcutRegistrar(tauri::AppHandle);

impl TauriShortcutRegistrar {
    pub(crate) fn new(app: tauri::AppHandle) -> Self {
        Self(app)
    }
}

impl ShortcutRegistrar for TauriShortcutRegistrar {
    fn register(&self, binding: &ShortcutBinding) -> Result<(), ShortcutRegistrationError> {
        self.0
            .global_shortcut()
            .register(binding.global)
            .map_err(|error| {
                tracing::warn!(%error, "a global shortcut could not be registered");
                ShortcutRegistrationError::ConflictOrUnsupported
            })
    }

    fn unregister(&self, binding: &ShortcutBinding) -> Result<(), ShortcutRegistrationError> {
        self.0
            .global_shortcut()
            .unregister(binding.global)
            .map_err(|error| {
                tracing::warn!(%error, "a global shortcut could not be unregistered");
                ShortcutRegistrationError::ConflictOrUnsupported
            })
    }
}

/// The native binding of a shortcut that breaks no rule.
pub(crate) fn binding(shortcut: &Shortcut) -> ShortcutBinding {
    let accelerator = shortcut
        .modifiers
        .iter()
        .map(|modifier| match modifier {
            ShortcutModifier::Control => "control",
            ShortcutModifier::Alt => "alt",
            ShortcutModifier::Shift => "shift",
            ShortcutModifier::Meta => "super",
        })
        .chain([shortcut.key.name()])
        .collect::<Vec<_>>()
        .join("+");
    let parsed = accelerator
        .parse::<GlobalShortcut>()
        .expect("every shortcut key has an accelerator");
    ShortcutBinding {
        id: ShortcutId(parsed.id()),
        global: parsed,
        shortcut: shortcut.clone(),
    }
}

/// The recording shortcut `settings` turn on; none while the shortcuts break a rule.
pub(crate) fn recording_binding(settings: &Settings) -> Option<RecordingBinding> {
    settings.shortcuts_valid().then(|| RecordingBinding {
        mode: settings.recording_mode,
        binding: binding(&settings.recording_shortcut),
    })
}

/// The cancel shortcut `settings` turn on; none while the shortcuts break a rule.
pub(crate) fn cancel_binding(settings: &Settings) -> Option<ShortcutBinding> {
    if !settings.shortcuts_valid() {
        tracing::warn!("the saved shortcuts break a rule; the cancel shortcut is off");
        return None;
    }
    settings.cancel_shortcut.as_ref().map(binding)
}

/// True while shortcuts may be registered: dictation is not paused, updating or clearing data,
/// and the application is not closing.
pub(crate) type LifecycleProbe = Arc<dyn Fn() -> bool + Send + Sync>;

/// The native registrations, changed only while the `native` lock is held.
#[derive(Default)]
struct Native {
    recording: Option<RecordingBinding>,
    live: Option<ShortcutBinding>,
    capturing: bool,
    cancel: CancelShortcut,
}

/// What input routes by and settings report, published after each native change.
#[derive(Default)]
struct Published {
    /// The recording shortcut, while it is registered.
    route: Option<RecordingBinding>,
    cancel: Option<ShortcutId>,
    /// Whether the configured shortcuts are registered, or only held back for a capture.
    active: bool,
    failure: Option<ShortcutRegistrationError>,
    cancel_failure: Option<ShortcutRegistrationError>,
    /// The persisted recording shortcut has not been registered yet, which is not a failure.
    pending: bool,
}

/// Lock order is `native -> pressed -> published`. `native` is the only lock held while the
/// registrar runs, which may wait on the event-loop thread or a desktop portal; input and
/// settings reads take only `pressed` and `published`, so they never wait for it.
///
/// The cancel shortcut is registered only while a dictation runs, so a key such as Escape is
/// not taken from other applications the rest of the time. Published snapshots report whether
/// one runs ([`Self::observe_dictation`]); the registration then follows on a worker
/// ([`Self::sync_cancel_shortcut`]).
pub(crate) struct Shortcuts {
    native: Mutex<Native>,
    /// The shortcuts held down, so the platform's key repeat is not taken as new presses.
    pressed: Mutex<HashSet<ShortcutId>>,
    published: Mutex<Published>,
    dictation_active: AtomicBool,
    registrar: Arc<dyn ShortcutRegistrar>,
    enabled: LifecycleProbe,
    startup_failure: Option<ShortcutRegistrationError>,
}

impl Shortcuts {
    /// Holds the configured recording shortcut without registering it: the application state the
    /// probe reads does not exist yet, and a desktop portal may hold the registration for the
    /// user's approval. [`Self::register_persisted`] registers it. Without a recording shortcut
    /// (the saved shortcuts break a rule, which the settings report), nothing is registered and
    /// no registration failure is recorded.
    pub(crate) fn new(
        registrar: Arc<dyn ShortcutRegistrar>,
        recording: Option<RecordingBinding>,
        enabled: LifecycleProbe,
        startup_failure: Option<ShortcutRegistrationError>,
    ) -> Self {
        Self {
            published: Mutex::new(Published {
                failure: startup_failure,
                pending: recording.is_some(),
                ..Published::default()
            }),
            native: Mutex::new(Native {
                recording,
                ..Native::default()
            }),
            pressed: Mutex::new(HashSet::new()),
            dictation_active: AtomicBool::new(false),
            registrar,
            enabled,
            startup_failure,
        }
    }

    /// Registers the recording shortcut [`Self::new`] holds. Runs on a worker once the
    /// application state exists, never on the event-loop thread: a portal's approval dialog can
    /// hold it for minutes.
    pub(crate) fn register_persisted(&self) {
        let mut native = self.native();
        if native.recording.is_none() {
            return;
        }
        if self.sync_locked(&mut native, (self.enabled)())
            == Err(ShortcutRegistrationError::PortalUnavailable)
            && let Some(error) = self.startup_failure
        {
            self.published().failure = Some(error);
        }
    }

    fn native(&self) -> MutexGuard<'_, Native> {
        self.native.lock().expect("native shortcut lock poisoned")
    }

    fn pressed(&self) -> MutexGuard<'_, HashSet<ShortcutId>> {
        self.pressed.lock().expect("shortcut press lock poisoned")
    }

    fn published(&self) -> MutexGuard<'_, Published> {
        self.published
            .lock()
            .expect("shortcut routes lock poisoned")
    }

    pub(crate) fn registration_active(&self) -> bool {
        let published = self.published();
        published.active && published.failure.is_none()
    }

    pub(crate) fn registration_pending(&self) -> bool {
        self.published().pending
    }

    pub(crate) fn failure(&self) -> Option<ShortcutRegistrationError> {
        self.published().failure
    }

    /// Why the cancel shortcut could not be registered for the last dictation.
    pub(crate) fn cancel_registration_failure(&self) -> Option<ShortcutRegistrationError> {
        self.published().cancel_failure
    }

    pub(crate) fn input(
        &self,
        id: ShortcutId,
        transition: KeyTransition,
    ) -> Option<ShortcutAction> {
        let enabled = (self.enabled)();
        let mut pressed = self.pressed();
        let new_press = match transition {
            KeyTransition::Pressed => pressed.insert(id),
            KeyTransition::Released => {
                pressed.remove(&id);
                false
            }
        };
        if !enabled {
            return None;
        }
        let published = self.published();
        let mode = mode_for(published.route.as_ref(), id);
        if published.cancel == Some(id) && mode.is_none() {
            return new_press.then_some(ShortcutAction::CancelDictation);
        }
        match (mode?, transition) {
            (RecordingMode::Hold, KeyTransition::Pressed) if new_press => {
                Some(ShortcutAction::Pressed(RecordingTrigger::Hold))
            }
            (RecordingMode::Hold, KeyTransition::Released) => Some(ShortcutAction::HoldReleased),
            (RecordingMode::Toggle, KeyTransition::Pressed) if new_press => {
                Some(ShortcutAction::Pressed(RecordingTrigger::Toggle))
            }
            _ => None,
        }
    }

    /// Records whether a published snapshot has a dictation running, and returns whether it
    /// changed. The caller then runs [`Self::sync_cancel_shortcut`] on a worker, because
    /// registration may wait on the event-loop thread.
    pub(crate) fn observe_dictation(&self, active: bool) -> bool {
        self.dictation_active.swap(active, Ordering::AcqRel) != active
    }

    /// Registers the cancel shortcut while a dictation runs and unregisters it otherwise.
    /// Concurrent callers converge on the latest dictation because each reads it under
    /// `native`. Runs on a worker, never on the main thread.
    pub(crate) fn sync_cancel_shortcut(&self) {
        let mut native = self.native();
        self.sync_cancel_locked(&mut native);
    }

    /// `None` turns the cancel shortcut off. During a dictation the new shortcut replaces the
    /// registered one immediately.
    pub(crate) fn set_cancel_binding(&self, binding: Option<ShortcutBinding>) {
        let mut native = self.native();
        if native.cancel.binding != binding {
            native.cancel.binding = binding;
            native.cancel.failure = None;
        }
        self.sync_cancel_locked(&mut native);
    }

    /// Releases every native registration for good. The application is already shutting
    /// down, so no input routes afterwards.
    pub(crate) fn shutdown(&self) {
        let mut native = self.native();
        if let Some(binding) = native.live.take()
            && let Err(error) = self.registrar.unregister(&binding)
        {
            tracing::warn!(?error, "a shortcut could not be unregistered at shutdown");
        }
        native.cancel.sync(self.registrar.as_ref(), false, None);
        *self.published() = Published::default();
        self.pressed().clear();
        self.registrar.shutdown();
    }

    /// The shortcut backend stopped delivering events: nothing is registered any more.
    pub(crate) fn capability_lost(&self, error: ShortcutRegistrationError) {
        let mut native = self.native();
        native.live = None;
        native.cancel.live = None;
        self.pressed().clear();
        let mut published = self.published();
        published.route = None;
        published.cancel = None;
        published.active = false;
        published.pending = false;
        published.failure = Some(error);
    }

    pub(crate) fn sync_lifecycle(&self) -> Result<(), ShortcutRegistrationError> {
        let mut native = self.native();
        self.sync_locked(&mut native, (self.enabled)())
    }

    /// Unregisters the shortcuts while the settings UI captures a new combination, so pressing
    /// the current shortcut reaches the capture control instead of starting a recording.
    pub(crate) fn suspend_for_capture(&self) -> Result<(), ShortcutRegistrationError> {
        let mut native = self.native();
        native.capturing = true;
        self.sync_locked(&mut native, (self.enabled)())
    }

    /// Ends a capture suspension. Shortcuts stay unregistered while dictation is disabled.
    pub(crate) fn resume_after_capture(&self) -> Result<(), ShortcutRegistrationError> {
        let mut native = self.native();
        native.capturing = false;
        self.sync_locked(&mut native, (self.enabled)())
    }

    /// Replaces the recording shortcut. When the new one cannot be registered, the previous
    /// one is registered again and the error is returned.
    pub(crate) fn apply_update(
        &self,
        candidate: RecordingBinding,
    ) -> Result<(), ShortcutRegistrationError> {
        let mut native = self.native();
        let enabled = (self.enabled)();
        let previous = native.recording.replace(candidate);
        let result = self.sync_locked(&mut native, enabled);
        if result.is_err() {
            native.recording = previous;
            if let Err(error) = self.sync_locked(&mut native, enabled) {
                tracing::warn!(
                    ?error,
                    "the previous shortcuts could not be registered again"
                );
            }
        }
        result
    }

    /// An unregistration failure is only logged.
    fn sync_locked(
        &self,
        native: &mut Native,
        enabled: bool,
    ) -> Result<(), ShortcutRegistrationError> {
        let wanted = native
            .recording
            .as_ref()
            .filter(|_| enabled && !native.capturing)
            .map(|recording| &recording.binding);
        if let Some(live) = native
            .live
            .take_if(|live| wanted.is_none_or(|binding| binding.id != live.id))
            && let Err(error) = self.registrar.unregister(&live)
        {
            tracing::warn!(?error, "a shortcut could not be unregistered");
        }
        let mut result = Ok(());
        if let Some(binding) = wanted
            && native.live.is_none()
        {
            match self.registrar.register(binding) {
                Ok(()) => native.live = Some(binding.clone()),
                Err(error) => result = Err(error),
            }
        }
        let registered = native.live.is_some();
        let route = native.recording.clone().filter(|_| registered);
        {
            let mut pressed = self.pressed();
            let mut published = self.published();
            // A key held while its registration changed may never report its release, and a
            // stale press would swallow the next real one.
            if published.route != route {
                pressed.clear();
            }
            published.route = route;
            published.pending = false;
            published.active = native.capturing || registered || native.recording.is_none();
            // A failure stays reported until registration is tried again.
            if enabled && !native.capturing {
                published.failure = result.err();
            }
        }
        self.sync_cancel_locked(native);
        result
    }

    fn sync_cancel_locked(&self, native: &mut Native) {
        let wanted = (self.enabled)()
            && !native.capturing
            && self.dictation_active.load(Ordering::Acquire)
            && self.registrar.supports_transient_shortcuts();
        let recording = native.live.as_ref().map(|live| live.id);
        let previous = native.cancel.live_id();
        native
            .cancel
            .sync(self.registrar.as_ref(), wanted, recording);
        let current = native.cancel.live_id();
        let mut pressed = self.pressed();
        if previous != current {
            for id in [previous, current].into_iter().flatten() {
                if recording != Some(id) {
                    pressed.remove(&id);
                }
            }
        }
        let mut published = self.published();
        published.cancel = current;
        published.cancel_failure = native.cancel.failure;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ShortcutRegistrationError {
    ConflictOrUnsupported,
    #[cfg_attr(all(not(test), not(target_os = "linux")), expect(dead_code))]
    PermissionDenied,
    PortalUnavailable,
    #[cfg_attr(all(not(test), not(target_os = "linux")), expect(dead_code))]
    SessionClosed,
}

#[derive(Clone, Debug, Default)]
struct CancelShortcut {
    binding: Option<ShortcutBinding>,
    live: Option<ShortcutBinding>,
    /// The last registration failure, cleared by a success or a new binding.
    failure: Option<ShortcutRegistrationError>,
}

impl CancelShortcut {
    fn live_id(&self) -> Option<ShortcutId> {
        self.live.as_ref().map(|live| live.id)
    }

    /// Registers the binding when `wanted` and unregisters it otherwise. A
    /// binding a recording shortcut already holds is never registered twice.
    /// A binding that could not be unregistered stays live, so the next sync
    /// tries again.
    fn sync(
        &mut self,
        registrar: &dyn ShortcutRegistrar,
        wanted: bool,
        recording: Option<ShortcutId>,
    ) {
        let desired = self
            .binding
            .clone()
            .filter(|binding| wanted && recording != Some(binding.id));
        if self.live_id() == desired.as_ref().map(|binding| binding.id) {
            return;
        }
        if let Some(live) = self.live.take()
            && let Err(error) = registrar.unregister(&live)
        {
            tracing::warn!(?error, "the cancel shortcut could not be unregistered");
            self.live = Some(live);
            self.failure = Some(error);
            return;
        }
        let Some(desired) = desired else {
            return;
        };
        match registrar.register(&desired) {
            Ok(()) => {
                self.live = Some(desired);
                self.failure = None;
            }
            Err(error) => {
                tracing::warn!(?error, "the cancel shortcut could not be registered");
                self.failure = Some(error);
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod memory_registrar;
#[cfg(test)]
mod tests;
