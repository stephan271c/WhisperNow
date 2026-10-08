use std::{
    collections::BTreeSet,
    sync::{
        OnceLock, Weak,
        atomic::{AtomicUsize, Ordering},
    },
};

use whispernow_api::ShortcutKey;

use super::*;

fn lifecycle_switch() -> (Arc<AtomicBool>, LifecycleProbe) {
    let running = Arc::new(AtomicBool::new(true));
    let probe = Arc::clone(&running);
    (running, Arc::new(move || probe.load(Ordering::SeqCst)))
}

fn always_running() -> LifecycleProbe {
    lifecycle_switch().1
}

#[derive(Default)]
struct MemoryRegistrar {
    registered: Mutex<BTreeSet<u32>>,
    fail_register: Mutex<Option<u32>>,
    fail_unregister: Mutex<Option<u32>>,
    /// Behaves like a desktop portal that asks before every registration.
    refuses_transient: AtomicBool,
}

impl MemoryRegistrar {
    fn registered(&self) -> BTreeSet<u32> {
        self.registered.lock().expect("registered").clone()
    }

    fn holds(&self, binding: &ShortcutBinding) -> bool {
        self.registered().contains(&binding.id.0)
    }
}

impl ShortcutRegistrar for MemoryRegistrar {
    fn register(&self, value: &ShortcutBinding) -> Result<(), ShortcutRegistrationError> {
        if *self.fail_register.lock().expect("fail register") == Some(value.id.0) {
            return Err(ShortcutRegistrationError::ConflictOrUnsupported);
        }
        self.registered
            .lock()
            .expect("registered")
            .insert(value.id.0);
        Ok(())
    }

    fn unregister(&self, value: &ShortcutBinding) -> Result<(), ShortcutRegistrationError> {
        if *self.fail_unregister.lock().expect("fail unregister") == Some(value.id.0) {
            return Err(ShortcutRegistrationError::ConflictOrUnsupported);
        }
        self.registered
            .lock()
            .expect("registered")
            .remove(&value.id.0);
        Ok(())
    }

    fn supports_transient_shortcuts(&self) -> bool {
        !self.refuses_transient.load(Ordering::SeqCst)
    }
}

fn control(key: &str) -> Shortcut {
    Shortcut {
        modifiers: [ShortcutModifier::Control].into(),
        key: ShortcutKey::from_name(key).expect("a shortcut key"),
    }
}

fn binding_id(key: &str) -> ShortcutId {
    binding(&control(key)).id
}

fn escape() -> ShortcutBinding {
    binding(&Shortcut::default_cancel())
}

/// Ctrl+`hold` and Ctrl+`toggle`, as the coordinator receives them.
fn bindings(hold: Option<&str>, toggle: Option<&str>) -> ShortcutBindings {
    ShortcutBindings {
        hold: hold.map(|key| binding(&control(key))),
        toggle: toggle.map(|key| binding(&control(key))),
    }
}

/// A coordinator with Ctrl+`hold` (and Ctrl+`toggle`) registered and a switch for whether
/// dictation is enabled.
fn switchable_memory_coordinator(
    hold: &str,
    toggle: Option<&str>,
) -> (Arc<MemoryRegistrar>, Arc<AtomicBool>, ShortcutCoordinator) {
    let bindings = Ok(bindings(Some(hold), toggle));
    let registrar = Arc::new(MemoryRegistrar::default());
    let (running, probe) = lifecycle_switch();
    let coordinator = ShortcutCoordinator::new(registrar.clone(), bindings, probe);
    (registrar, running, coordinator)
}

fn memory_coordinator(
    hold: &str,
    toggle: Option<&str>,
) -> (Arc<MemoryRegistrar>, ShortcutCoordinator) {
    let (registrar, _running, coordinator) = switchable_memory_coordinator(hold, toggle);
    (registrar, coordinator)
}

/// A coordinator with a registered Ctrl+F1 hold shortcut and the Escape cancel shortcut.
fn cancel_fixture() -> (
    Arc<MemoryRegistrar>,
    Arc<AtomicBool>,
    ShortcutCoordinator,
    ShortcutBinding,
) {
    let (registrar, running, coordinator) = switchable_memory_coordinator("F1", None);
    let cancel = escape();
    coordinator.set_cancel_binding(Some(cancel.clone()));
    (registrar, running, coordinator, cancel)
}

#[test]
fn cancel_shortcut_is_registered_while_a_dictation_runs_and_unregistered_when_idle() {
    let (registrar, _running, coordinator, cancel) = cancel_fixture();
    assert!(!registrar.holds(&cancel), "Escape is free while idle");
    assert_eq!(coordinator.input(cancel.id, KeyTransition::Pressed), None);
    coordinator.input(cancel.id, KeyTransition::Released);

    // Recording, then processing: both snapshots have a dictation.
    assert!(coordinator.observe_dictation(true));
    coordinator.sync_cancel_shortcut();
    assert!(registrar.holds(&cancel));
    assert!(!coordinator.observe_dictation(true), "processing keeps it");
    assert_eq!(
        coordinator.input(cancel.id, KeyTransition::Pressed),
        Some(ShortcutAction::CancelDictation)
    );
    assert_eq!(
        coordinator.input(cancel.id, KeyTransition::Pressed),
        None,
        "a key repeat does not cancel again"
    );
    assert_eq!(coordinator.input(cancel.id, KeyTransition::Released), None);

    assert!(coordinator.observe_dictation(false));
    coordinator.sync_cancel_shortcut();
    assert!(!registrar.holds(&cancel), "Escape is returned when idle");
    assert_eq!(coordinator.input(cancel.id, KeyTransition::Pressed), None);
    assert_eq!(coordinator.cancel_registration_failure(), None);
}

#[test]
fn cancel_shortcut_stays_unregistered_while_paused_capturing_or_behind_a_portal() {
    let (registrar, running, coordinator, cancel) = cancel_fixture();
    coordinator.observe_dictation(true);

    running.store(false, Ordering::SeqCst);
    coordinator.sync_lifecycle().expect("pause");
    assert!(!registrar.holds(&cancel));
    running.store(true, Ordering::SeqCst);
    coordinator.sync_lifecycle().expect("resume");
    assert!(registrar.holds(&cancel));

    coordinator.suspend_for_capture().expect("capture");
    assert!(!registrar.holds(&cancel));
    coordinator.resume_after_capture().expect("end capture");
    assert!(registrar.holds(&cancel));

    coordinator.set_cancel_binding(None);
    assert!(!registrar.holds(&cancel), "turning it off unregisters");
    registrar.refuses_transient.store(true, Ordering::SeqCst);
    coordinator.set_cancel_binding(Some(cancel.clone()));
    assert!(!registrar.holds(&cancel));

    registrar.refuses_transient.store(false, Ordering::SeqCst);
    coordinator.sync_cancel_shortcut();
    assert!(registrar.holds(&cancel));
    coordinator.shutdown();
    assert!(registrar.registered().is_empty());
}

#[test]
fn a_refused_cancel_shortcut_is_reported_quietly_and_recording_shortcuts_keep_working() {
    let (registrar, _running, coordinator, cancel) = cancel_fixture();
    *registrar.fail_register.lock().expect("fail register") = Some(cancel.id.0);
    coordinator.observe_dictation(true);
    coordinator.sync_cancel_shortcut();

    assert!(!registrar.holds(&cancel));
    assert_eq!(
        coordinator.cancel_registration_failure(),
        Some(ShortcutRegistrationError::ConflictOrUnsupported)
    );
    assert!(coordinator.registration_active());
    assert_eq!(coordinator.registration_diagnostic(), None);
    assert_eq!(
        coordinator.input(binding_id("F1"), KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Hold))
    );

    *registrar.fail_register.lock().expect("fail register") = None;
    coordinator.observe_dictation(false);
    coordinator.sync_cancel_shortcut();
    coordinator.observe_dictation(true);
    coordinator.sync_cancel_shortcut();
    assert!(registrar.holds(&cancel), "the next dictation retries");
    assert_eq!(coordinator.cancel_registration_failure(), None);
}

#[test]
fn a_cancel_press_whose_release_was_lost_does_not_swallow_the_next_cancel() {
    let (registrar, _running, coordinator, cancel) = cancel_fixture();
    coordinator.observe_dictation(true);
    coordinator.sync_cancel_shortcut();
    assert_eq!(
        coordinator.input(cancel.id, KeyTransition::Pressed),
        Some(ShortcutAction::CancelDictation)
    );
    // The cancel ends the dictation, and Escape is unregistered before its
    // release arrives, so the release is never delivered.
    coordinator.observe_dictation(false);
    coordinator.sync_cancel_shortcut();
    assert!(!registrar.holds(&cancel));

    coordinator.observe_dictation(true);
    coordinator.sync_cancel_shortcut();
    assert_eq!(
        coordinator.input(cancel.id, KeyTransition::Pressed),
        Some(ShortcutAction::CancelDictation),
        "the next dictation's first press cancels"
    );
}

#[test]
fn a_cancel_shortcut_that_could_not_be_released_is_released_by_the_next_sync() {
    let (registrar, _running, coordinator, cancel) = cancel_fixture();
    coordinator.observe_dictation(true);
    coordinator.sync_cancel_shortcut();
    *registrar.fail_unregister.lock().expect("fail unregister") = Some(cancel.id.0);

    coordinator.observe_dictation(false);
    coordinator.sync_cancel_shortcut();
    assert!(registrar.holds(&cancel));
    assert_eq!(
        coordinator.cancel_registration_failure(),
        Some(ShortcutRegistrationError::ConflictOrUnsupported)
    );

    *registrar.fail_unregister.lock().expect("fail unregister") = None;
    coordinator.sync_lifecycle().expect("a later sync");
    assert!(!registrar.holds(&cancel));
}

#[test]
fn a_changed_shortcut_replaces_the_old_one_and_keeps_the_unchanged_one() {
    let (registrar, coordinator) = memory_coordinator("F1", Some("F2"));
    assert_eq!(
        registrar.registered(),
        BTreeSet::from([binding_id("F1").0, binding_id("F2").0])
    );
    // Registering the unchanged shortcut again would fail.
    *registrar.fail_register.lock().expect("fail register") = Some(binding_id("F1").0);

    coordinator
        .apply_update(Ok(bindings(Some("F1"), Some("F3"))))
        .expect("replace the toggle");
    assert_eq!(
        registrar.registered(),
        BTreeSet::from([binding_id("F1").0, binding_id("F3").0])
    );
    assert_eq!(coordinator.input(binding_id("F2"), KeyTransition::Pressed), None);
    assert_eq!(
        coordinator.input(binding_id("F3"), KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Toggle))
    );
}

#[test]
fn a_refused_shortcut_change_registers_the_previous_shortcuts_again() {
    let (registrar, coordinator) = memory_coordinator("F1", None);
    *registrar.fail_register.lock().expect("fail register") = Some(binding_id("F3").0);
    assert_eq!(
        coordinator.apply_update(Ok(bindings(Some("F2"), Some("F3")))),
        Err(ShortcutRegistrationError::ConflictOrUnsupported)
    );
    assert_eq!(registrar.registered(), BTreeSet::from([binding_id("F1").0]));
    assert!(coordinator.registration_active());
    assert_eq!(coordinator.registration_diagnostic(), None);
    assert_eq!(
        coordinator.input(binding_id("F1"), KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Hold))
    );
    assert_eq!(coordinator.input(binding_id("F2"), KeyTransition::Pressed), None);
}

#[test]
fn unavailable_shortcuts_are_reported_and_retried_without_restarting() {
    let registrar = Arc::new(MemoryRegistrar::default());
    *registrar.fail_register.lock().expect("fail register") = Some(binding_id("F1").0);
    let coordinator = ShortcutCoordinator::new(
        registrar.clone(),
        Ok(bindings(Some("F1"), None)),
        always_running(),
    );
    assert!(!coordinator.registration_active());
    assert_eq!(
        coordinator.registration_diagnostic(),
        Some(ShortcutRegistrationError::ConflictOrUnsupported)
    );
    assert_eq!(coordinator.input(binding_id("F1"), KeyTransition::Pressed), None);

    *registrar.fail_register.lock().expect("fail register") = None;
    coordinator.sync_lifecycle().expect("retry");
    assert!(coordinator.registration_active());
    assert_eq!(coordinator.registration_diagnostic(), None);

    let invalid = ShortcutCoordinator::new(
        registrar,
        Err(ShortcutRegistrationError::ConflictOrUnsupported),
        always_running(),
    );
    assert_eq!(
        invalid.registration_diagnostic(),
        Some(ShortcutRegistrationError::ConflictOrUnsupported)
    );
}

#[test]
fn paused_dictation_releases_the_shortcuts_and_resuming_restores_them() {
    let (registrar, running, coordinator) = switchable_memory_coordinator("F1", None);
    running.store(false, Ordering::SeqCst);
    coordinator.sync_lifecycle().expect("pause");
    assert!(registrar.registered().is_empty());

    coordinator
        .apply_update(Ok(bindings(Some("F2"), None)))
        .expect("a change while paused");
    assert!(registrar.registered().is_empty(), "nothing registers while paused");
    assert_eq!(coordinator.input(binding_id("F2"), KeyTransition::Pressed), None);

    running.store(true, Ordering::SeqCst);
    coordinator.sync_lifecycle().expect("resume");
    assert_eq!(registrar.registered(), BTreeSet::from([binding_id("F2").0]));
    assert!(coordinator.registration_active());
    coordinator.input(binding_id("F2"), KeyTransition::Released);
    assert_eq!(
        coordinator.input(binding_id("F2"), KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Hold))
    );
}

#[test]
fn capture_suspension_ignores_presses_and_forgets_held_keys() {
    let (registrar, running, coordinator) = switchable_memory_coordinator("F1", None);
    let hold = binding_id("F1");
    assert_eq!(
        coordinator.input(hold, KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Hold))
    );

    coordinator.suspend_for_capture().expect("suspend");
    assert!(registrar.registered().is_empty());
    assert_eq!(coordinator.input(hold, KeyTransition::Pressed), None);
    assert!(coordinator.registration_active());

    // A pause and resume during the capture keep the shortcuts released.
    running.store(false, Ordering::SeqCst);
    coordinator.sync_lifecycle().expect("pause");
    running.store(true, Ordering::SeqCst);
    coordinator.sync_lifecycle().expect("resume");
    assert!(registrar.registered().is_empty());

    coordinator.resume_after_capture().expect("resume");
    assert!(!registrar.registered().is_empty());
    coordinator.input(hold, KeyTransition::Released);
    assert_eq!(
        coordinator.input(hold, KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Hold))
    );
}

#[test]
fn a_hold_press_and_release_each_reach_the_worker_once() {
    let (_registrar, coordinator) = memory_coordinator("F1", Some("F2"));
    let (hold, toggle) = (binding_id("F1"), binding_id("F2"));
    assert_eq!(
        coordinator.input(hold, KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Hold))
    );
    assert_eq!(
        coordinator.input(hold, KeyTransition::Pressed),
        None,
        "a key repeat is not a new press"
    );
    assert_eq!(
        coordinator.input(hold, KeyTransition::Released),
        Some(ShortcutAction::HoldReleased)
    );

    assert_eq!(
        coordinator.input(toggle, KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Toggle))
    );
    assert_eq!(coordinator.input(toggle, KeyTransition::Pressed), None);
    assert_eq!(
        coordinator.input(toggle, KeyTransition::Released),
        None,
        "a toggle release asks for nothing"
    );
    assert_eq!(
        coordinator.input(toggle, KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Toggle))
    );
}

#[test]
fn a_key_released_while_paused_presses_again_after_resuming() {
    let (_registrar, running, coordinator) = switchable_memory_coordinator("F1", None);
    let hold = binding_id("F1");
    let set_running = |value: bool| {
        running.store(value, Ordering::SeqCst);
        coordinator.sync_lifecycle().expect("lifecycle sync");
    };

    coordinator.input(hold, KeyTransition::Pressed);
    set_running(false);
    assert_eq!(coordinator.input(hold, KeyTransition::Released), None);
    assert_eq!(coordinator.input(hold, KeyTransition::Pressed), None);
    set_running(true);
    coordinator.input(hold, KeyTransition::Released);
    assert_eq!(
        coordinator.input(hold, KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Hold))
    );
}

#[test]
fn a_release_before_its_press_leaves_nothing_stuck() {
    let (_registrar, coordinator) = memory_coordinator("F1", None);
    let hold = binding_id("F1");
    assert_eq!(
        coordinator.input(hold, KeyTransition::Released),
        Some(ShortcutAction::HoldReleased)
    );
    assert_eq!(
        coordinator.input(hold, KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Hold))
    );
}

#[test]
fn a_lost_shortcut_backend_unregisters_everything_and_forgets_held_keys() {
    let (_registrar, coordinator) = memory_coordinator("F1", None);
    coordinator.input(binding_id("F1"), KeyTransition::Pressed);
    coordinator.capability_lost(ShortcutRegistrationError::SessionClosed);
    assert!(!coordinator.registration_active());
    assert_eq!(
        coordinator.registration_diagnostic(),
        Some(ShortcutRegistrationError::SessionClosed)
    );
    assert_eq!(coordinator.input(binding_id("F1"), KeyTransition::Pressed), None);
}

/// Records every native call and whether `pressed` or `published` was held when it ran.
/// Native registration may wait on the event-loop thread, which delivers input, so it must
/// never run under a lock input takes.
#[derive(Default)]
struct LockProbingRegistrar {
    registered: Mutex<BTreeSet<u32>>,
    coordinator: OnceLock<Weak<ShortcutCoordinator>>,
    calls: AtomicUsize,
    calls_under_lock: AtomicUsize,
}

impl LockProbingRegistrar {
    fn probe(&self) {
        let Some(coordinator) = self.coordinator.get().and_then(Weak::upgrade) else {
            return;
        };
        self.calls.fetch_add(1, Ordering::SeqCst);
        if coordinator.pressed.try_lock().is_err() || coordinator.published.try_lock().is_err() {
            self.calls_under_lock.fetch_add(1, Ordering::SeqCst);
        }
    }
}

impl ShortcutRegistrar for LockProbingRegistrar {
    fn register(&self, value: &ShortcutBinding) -> Result<(), ShortcutRegistrationError> {
        self.probe();
        self.registered
            .lock()
            .expect("registered")
            .insert(value.id.0);
        Ok(())
    }

    fn unregister(&self, value: &ShortcutBinding) -> Result<(), ShortcutRegistrationError> {
        self.probe();
        self.registered
            .lock()
            .expect("registered")
            .remove(&value.id.0);
        Ok(())
    }
}

#[test]
fn native_registration_never_runs_under_the_input_locks() {
    let registrar = Arc::new(LockProbingRegistrar::default());
    let (running, probe) = lifecycle_switch();
    let coordinator = Arc::new(ShortcutCoordinator::new(
        registrar.clone(),
        Ok(bindings(Some("F1"), None)),
        probe,
    ));
    assert!(
        registrar
            .coordinator
            .set(Arc::downgrade(&coordinator))
            .is_ok()
    );
    coordinator.set_cancel_binding(Some(escape()));
    let calls = || registrar.calls.load(Ordering::SeqCst);
    let observed = std::cell::Cell::new(calls());
    let native_step = |step: &str| {
        assert!(calls() > observed.get(), "{step} must reach the registrar");
        observed.set(calls());
    };

    coordinator
        .apply_update(Ok(bindings(Some("F2"), None)))
        .expect("apply update");
    native_step("apply update");
    coordinator.observe_dictation(true);
    coordinator.sync_cancel_shortcut();
    native_step("cancel shortcut");
    running.store(false, Ordering::SeqCst);
    coordinator.sync_lifecycle().expect("pause");
    native_step("pause");
    running.store(true, Ordering::SeqCst);
    coordinator.sync_lifecycle().expect("resume");
    native_step("resume");
    coordinator.suspend_for_capture().expect("suspend for capture");
    native_step("capture suspension");
    coordinator.resume_after_capture().expect("resume after capture");
    native_step("capture resumption");
    coordinator.shutdown();
    native_step("shutdown");

    assert_eq!(registrar.calls_under_lock.load(Ordering::SeqCst), 0);
    assert!(registrar.registered.lock().expect("registered").is_empty());
}

#[test]
fn every_shortcut_key_is_captured_and_registrable() {
    for &key in ShortcutKey::ALL {
        assert_eq!(ShortcutKey::from_code(key.code()), Some(key));
        let shortcut = Shortcut {
            modifiers: [ShortcutModifier::Control].into(),
            key,
        };
        // Panics when the key has no accelerator.
        let _ = binding(&shortcut);
    }
}

#[test]
fn saved_shortcuts_that_break_a_rule_register_nothing() {
    let settings = Settings {
        toggle_shortcut: Some(Settings::default().hold_shortcut),
        ..Settings::default()
    };
    assert_eq!(
        recording_bindings(&settings),
        Err(ShortcutRegistrationError::ConflictOrUnsupported)
    );
    assert_eq!(cancel_binding(&settings), None);
    assert!(recording_bindings(&Settings::default()).is_ok());
}
