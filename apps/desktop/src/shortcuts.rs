use std::{
    collections::HashSet,
    sync::{
        Arc, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
};

use tauri_plugin_global_shortcut::{GlobalShortcutExt as _, Shortcut as GlobalShortcut};
use whispernow_domain::{RecordingMode, Settings, Shortcut, ShortcutModifier};

use crate::services::RecordingTrigger;

/// Derived by the native shortcut library.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ShortcutId(pub(crate) u32);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ShortcutBinding {
    pub(crate) id: ShortcutId,
    pub(crate) canonical: String,
    pub(crate) shortcut: Shortcut,
}

impl ShortcutBinding {
    /// The trigger a desktop portal is asked for, such as `CTRL+space`.
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

pub(crate) trait ShortcutRegistrar: Send + Sync {
    fn register(&self, binding: &ShortcutBinding) -> Result<(), ShortcutRegistrationError>;
    fn unregister(&self, binding: &ShortcutBinding) -> Result<(), ShortcutRegistrationError>;

    fn diagnostic(&self) -> Option<ShortcutRegistrationError> {
        None
    }

    /// Whether a shortcut may be registered for moments, such as the cancel
    /// shortcut for each dictation. A desktop portal that asks the user to
    /// approve every registration cannot.
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
            .register(binding.canonical.as_str())
            .map_err(|_| ShortcutRegistrationError::ConflictOrUnsupported)
    }

    fn unregister(&self, binding: &ShortcutBinding) -> Result<(), ShortcutRegistrationError> {
        self.0
            .global_shortcut()
            .unregister(binding.canonical.as_str())
            .map_err(|_| ShortcutRegistrationError::ConflictOrUnsupported)
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
        canonical: parsed.to_string(),
        shortcut: shortcut.clone(),
    }
}

/// The recording shortcut `settings` turn on; none while the shortcuts break a rule.
pub(crate) fn recording_binding(
    settings: &Settings,
) -> Result<RecordingBinding, ShortcutRegistrationError> {
    if !settings.shortcut_problems().is_empty() {
        return Err(ShortcutRegistrationError::ConflictOrUnsupported);
    }
    Ok(RecordingBinding {
        mode: settings.recording_mode,
        binding: binding(&settings.recording_shortcut),
    })
}

/// The cancel shortcut `settings` turn on; none while the shortcuts break a rule.
pub(crate) fn cancel_binding(settings: &Settings) -> Option<ShortcutBinding> {
    if !settings.shortcut_problems().is_empty() {
        tracing::warn!("the saved shortcuts break a rule; the cancel shortcut is off");
        return None;
    }
    settings.cancel_shortcut.as_ref().map(binding)
}

/// Reports whether shortcuts may be registered: dictation is neither paused
/// nor updating and the application is not closing. The application state is
/// read, never mirrored.
pub(crate) type LifecycleProbe = Arc<dyn Fn() -> bool + Send + Sync>;

/// The native registrations, changed only while the `native` lock is held.
#[derive(Default)]
struct Native {
    recording: Option<RecordingBinding>,
    /// The recording shortcut while it is registered.
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
}

/// Lock order is `native -> pressed -> published`. `native` is the only lock held while the
/// registrar runs, which may wait on the event-loop thread or a desktop portal; input and
/// settings reads take only `pressed` and `published`, so they never wait for it.
///
/// The cancel shortcut is registered only while a dictation runs, so a key such as Escape is
/// not taken from other applications the rest of the time. Published snapshots report whether
/// one runs ([`Self::observe_dictation`]); the registration then follows on a worker
/// ([`Self::sync_cancel_shortcut`]).
pub(crate) struct ShortcutCoordinator {
    native: Mutex<Native>,
    /// The shortcuts held down, so the platform's key repeat is not taken as new presses.
    pressed: Mutex<HashSet<ShortcutId>>,
    published: Mutex<Published>,
    /// Whether the latest published snapshot has a dictation running.
    dictation_active: AtomicBool,
    registrar: Arc<dyn ShortcutRegistrar>,
    enabled: LifecycleProbe,
}

impl ShortcutCoordinator {
    /// Registers the configured shortcuts at once; the application state the probe reads does
    /// not exist yet. Invalid configured shortcuts leave nothing registered and are reported.
    pub(crate) fn new(
        registrar: Arc<dyn ShortcutRegistrar>,
        recording: Result<RecordingBinding, ShortcutRegistrationError>,
        enabled: LifecycleProbe,
    ) -> Self {
        let coordinator = Self {
            native: Mutex::new(Native::default()),
            pressed: Mutex::new(HashSet::new()),
            published: Mutex::new(Published::default()),
            dictation_active: AtomicBool::new(false),
            registrar,
            enabled,
        };
        match recording {
            Ok(recording) => {
                let mut native = coordinator.native();
                native.recording = Some(recording);
                let _reported = coordinator.sync_locked(&mut native, true);
            }
            Err(error) => coordinator.published().failure = Some(error),
        }
        coordinator
    }

    /// Serializes native work. The guarded registrations are rebuilt by every sync, so a panic
    /// in an earlier registrar call does not disable later native work.
    fn native(&self) -> MutexGuard<'_, Native> {
        self.native.lock().unwrap_or_else(PoisonError::into_inner)
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

    pub(crate) fn registration_diagnostic(&self) -> Option<ShortcutRegistrationError> {
        self.published()
            .failure
            .or_else(|| self.registrar.diagnostic())
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

    /// Replaces the cancel shortcut; `None` turns it off. During a dictation the new shortcut
    /// replaces the registered one immediately.
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
        native
            .cancel
            .sync(self.registrar.as_ref(), false, &HashSet::new());
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
        published.failure = Some(error);
    }

    /// Registers the configured shortcuts while dictation is enabled and unregisters them
    /// otherwise.
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
        candidate: Result<RecordingBinding, ShortcutRegistrationError>,
    ) -> Result<(), ShortcutRegistrationError> {
        let candidate = candidate?;
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

    /// Registers the configured shortcut when `enabled` and no capture runs, and releases it
    /// otherwise. An unregistration failure is only logged.
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
        let recording: HashSet<ShortcutId> = native.live.iter().map(|live| live.id).collect();
        let previous = native.cancel.live_id();
        native
            .cancel
            .sync(self.registrar.as_ref(), wanted, &recording);
        let current = native.cancel.live_id();
        let mut pressed = self.pressed();
        if previous != current {
            for id in [previous, current].into_iter().flatten() {
                if !recording.contains(&id) {
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
    PermissionDenied,
    PortalUnavailable,
    SessionClosed,
}

/// The cancel shortcut: its configured binding and its native registration,
/// which exists only while a dictation runs.
#[derive(Clone, Debug, Default)]
struct CancelShortcut {
    /// `None` while the cancel shortcut is turned off.
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
    fn sync<R: ShortcutRegistrar + ?Sized>(
        &mut self,
        registrar: &R,
        wanted: bool,
        recording_shortcuts: &HashSet<ShortcutId>,
    ) {
        let desired = self
            .binding
            .clone()
            .filter(|binding| wanted && !recording_shortcuts.contains(&binding.id));
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
mod tests {
    include!("shortcuts/tests.rs");
}
