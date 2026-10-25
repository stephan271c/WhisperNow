use std::{collections::BTreeSet, sync::atomic::AtomicUsize};

use whispernow_domain::ShortcutKey;

use super::{memory_registrar::MemoryRegistrar, *};

fn lifecycle_switch() -> (Arc<AtomicBool>, LifecycleProbe) {
    let running = Arc::new(AtomicBool::new(true));
    let probe = Arc::clone(&running);
    (running, Arc::new(move || probe.load(Ordering::SeqCst)))
}

fn always_running() -> LifecycleProbe {
    lifecycle_switch().1
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

fn recording(key: &str, mode: RecordingMode) -> RecordingBinding {
    RecordingBinding {
        mode,
        binding: binding(&control(key)),
    }
}

fn hold(key: &str) -> RecordingBinding {
    recording(key, RecordingMode::Hold)
}

/// A shortcuts with Ctrl+`key` registered for hold-to-record and a switch for whether
/// dictation is enabled.
fn switchable_memory_shortcuts(key: &str) -> (Arc<MemoryRegistrar>, Arc<AtomicBool>, Shortcuts) {
    let registrar = Arc::new(MemoryRegistrar::default());
    let (running, probe) = lifecycle_switch();
    let shortcuts = Shortcuts::new(registrar.clone(), Some(hold(key)), probe, None);
    (registrar, running, shortcuts)
}

fn memory_shortcuts(key: &str) -> (Arc<MemoryRegistrar>, Shortcuts) {
    let (registrar, _running, shortcuts) = switchable_memory_shortcuts(key);
    (registrar, shortcuts)
}

/// A shortcuts with a registered Ctrl+F1 hold shortcut and the Escape cancel shortcut.
fn cancel_fixture() -> (
    Arc<MemoryRegistrar>,
    Arc<AtomicBool>,
    Shortcuts,
    ShortcutBinding,
) {
    let (registrar, running, shortcuts) = switchable_memory_shortcuts("F1");
    let cancel = escape();
    shortcuts.set_cancel_binding(Some(cancel.clone()));
    (registrar, running, shortcuts, cancel)
}

#[test]
fn cancel_shortcut_is_registered_while_a_dictation_runs_and_unregistered_when_idle() {
    let (registrar, _running, shortcuts, cancel) = cancel_fixture();
    assert!(!registrar.holds(&cancel), "Escape is free while idle");
    assert_eq!(shortcuts.input(cancel.id, KeyTransition::Pressed), None);
    shortcuts.input(cancel.id, KeyTransition::Released);

    // Recording, then processing: both snapshots have a dictation.
    assert!(shortcuts.observe_dictation(true));
    shortcuts.sync_cancel_shortcut();
    assert!(registrar.holds(&cancel));
    assert!(!shortcuts.observe_dictation(true), "processing keeps it");
    assert_eq!(
        shortcuts.input(cancel.id, KeyTransition::Pressed),
        Some(ShortcutAction::CancelDictation)
    );
    assert_eq!(
        shortcuts.input(cancel.id, KeyTransition::Pressed),
        None,
        "a key repeat does not cancel again"
    );
    assert_eq!(shortcuts.input(cancel.id, KeyTransition::Released), None);

    assert!(shortcuts.observe_dictation(false));
    shortcuts.sync_cancel_shortcut();
    assert!(!registrar.holds(&cancel), "Escape is returned when idle");
    assert_eq!(shortcuts.input(cancel.id, KeyTransition::Pressed), None);
    assert_eq!(shortcuts.cancel_registration_failure(), None);
}

#[test]
fn cancel_shortcut_stays_unregistered_while_paused_capturing_or_behind_a_portal() {
    let (registrar, running, shortcuts, cancel) = cancel_fixture();
    shortcuts.observe_dictation(true);

    running.store(false, Ordering::SeqCst);
    shortcuts.sync_lifecycle().expect("pause");
    assert!(!registrar.holds(&cancel));
    running.store(true, Ordering::SeqCst);
    shortcuts.sync_lifecycle().expect("resume");
    assert!(registrar.holds(&cancel));

    shortcuts.suspend_for_capture().expect("capture");
    assert!(!registrar.holds(&cancel));
    shortcuts.resume_after_capture().expect("end capture");
    assert!(registrar.holds(&cancel));

    shortcuts.set_cancel_binding(None);
    assert!(!registrar.holds(&cancel), "turning it off unregisters");
    registrar.refuses_transient.store(true, Ordering::SeqCst);
    shortcuts.set_cancel_binding(Some(cancel.clone()));
    assert!(!registrar.holds(&cancel));

    registrar.refuses_transient.store(false, Ordering::SeqCst);
    shortcuts.sync_cancel_shortcut();
    assert!(registrar.holds(&cancel));
    shortcuts.shutdown();
    assert!(registrar.registered().is_empty());
}

#[test]
fn a_refused_cancel_shortcut_is_reported_quietly_and_recording_shortcuts_keep_working() {
    let (registrar, _running, shortcuts, cancel) = cancel_fixture();
    *registrar.fail_register.lock().expect("fail register") = Some(cancel.id.0);
    shortcuts.observe_dictation(true);
    shortcuts.sync_cancel_shortcut();

    assert!(!registrar.holds(&cancel));
    assert_eq!(
        shortcuts.cancel_registration_failure(),
        Some(ShortcutRegistrationError::ConflictOrUnsupported)
    );
    assert!(shortcuts.registration_active());
    assert_eq!(shortcuts.failure(), None);
    assert_eq!(
        shortcuts.input(binding_id("F1"), KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Hold))
    );

    *registrar.fail_register.lock().expect("fail register") = None;
    shortcuts.observe_dictation(false);
    shortcuts.sync_cancel_shortcut();
    shortcuts.observe_dictation(true);
    shortcuts.sync_cancel_shortcut();
    assert!(registrar.holds(&cancel), "the next dictation retries");
    assert_eq!(shortcuts.cancel_registration_failure(), None);
}

#[test]
fn a_cancel_press_whose_release_was_lost_does_not_swallow_the_next_cancel() {
    let (registrar, _running, shortcuts, cancel) = cancel_fixture();
    shortcuts.observe_dictation(true);
    shortcuts.sync_cancel_shortcut();
    assert_eq!(
        shortcuts.input(cancel.id, KeyTransition::Pressed),
        Some(ShortcutAction::CancelDictation)
    );
    // The cancel ends the dictation, and Escape is unregistered before its
    // release arrives, so the release is never delivered.
    shortcuts.observe_dictation(false);
    shortcuts.sync_cancel_shortcut();
    assert!(!registrar.holds(&cancel));

    shortcuts.observe_dictation(true);
    shortcuts.sync_cancel_shortcut();
    assert_eq!(
        shortcuts.input(cancel.id, KeyTransition::Pressed),
        Some(ShortcutAction::CancelDictation),
        "the next dictation's first press cancels"
    );
}

#[test]
fn a_cancel_shortcut_that_could_not_be_released_is_released_by_the_next_sync() {
    let (registrar, _running, shortcuts, cancel) = cancel_fixture();
    shortcuts.observe_dictation(true);
    shortcuts.sync_cancel_shortcut();
    *registrar.fail_unregister.lock().expect("fail unregister") = Some(cancel.id.0);

    shortcuts.observe_dictation(false);
    shortcuts.sync_cancel_shortcut();
    assert!(registrar.holds(&cancel));
    assert_eq!(
        shortcuts.cancel_registration_failure(),
        Some(ShortcutRegistrationError::ConflictOrUnsupported)
    );

    *registrar.fail_unregister.lock().expect("fail unregister") = None;
    shortcuts.sync_lifecycle().expect("a later sync");
    assert!(!registrar.holds(&cancel));
}

#[test]
fn a_changed_mode_keeps_the_shortcut_registered_and_a_changed_key_replaces_it() {
    let (registrar, shortcuts) = memory_shortcuts("F1");
    assert_eq!(registrar.registered(), BTreeSet::from([binding_id("F1").0]));
    // Registering the unchanged shortcut again would fail.
    *registrar.fail_register.lock().expect("fail register") = Some(binding_id("F1").0);

    shortcuts
        .apply_update(recording("F1", RecordingMode::Toggle))
        .expect("switch to toggle");
    assert_eq!(registrar.registered(), BTreeSet::from([binding_id("F1").0]));
    assert_eq!(
        shortcuts.input(binding_id("F1"), KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Toggle))
    );

    shortcuts
        .apply_update(recording("F3", RecordingMode::Toggle))
        .expect("replace the key");
    assert_eq!(registrar.registered(), BTreeSet::from([binding_id("F3").0]));
    assert_eq!(
        shortcuts.input(binding_id("F1"), KeyTransition::Pressed),
        None
    );
}

#[test]
fn a_refused_shortcut_change_registers_the_previous_shortcut_again() {
    let (registrar, shortcuts) = memory_shortcuts("F1");
    *registrar.fail_register.lock().expect("fail register") = Some(binding_id("F3").0);
    assert_eq!(
        shortcuts.apply_update(recording("F3", RecordingMode::Toggle)),
        Err(ShortcutRegistrationError::ConflictOrUnsupported)
    );
    assert_eq!(registrar.registered(), BTreeSet::from([binding_id("F1").0]));
    assert!(shortcuts.registration_active());
    assert_eq!(shortcuts.failure(), None);
    assert_eq!(
        shortcuts.input(binding_id("F1"), KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Hold))
    );
    assert_eq!(
        shortcuts.input(binding_id("F3"), KeyTransition::Pressed),
        None
    );
}

#[test]
fn unavailable_shortcuts_are_reported_and_retried_without_restarting() {
    let registrar = Arc::new(MemoryRegistrar::default());
    *registrar.fail_register.lock().expect("fail register") = Some(binding_id("F1").0);
    let shortcuts = Shortcuts::new(registrar.clone(), Some(hold("F1")), always_running(), None);
    assert!(!shortcuts.registration_active());
    assert_eq!(
        shortcuts.failure(),
        Some(ShortcutRegistrationError::ConflictOrUnsupported)
    );
    assert_eq!(
        shortcuts.input(binding_id("F1"), KeyTransition::Pressed),
        None
    );

    *registrar.fail_register.lock().expect("fail register") = None;
    shortcuts.sync_lifecycle().expect("retry");
    assert!(shortcuts.registration_active());
    assert_eq!(shortcuts.failure(), None);

    let registered = registrar.registered();
    let invalid = Shortcuts::new(registrar.clone(), None, always_running(), None);
    assert_eq!(
        invalid.failure(),
        None,
        "shortcuts that break a rule are the settings' to report"
    );
    assert!(!invalid.registration_active());
    assert_eq!(
        registrar.registered(),
        registered,
        "nothing new is registered"
    );
}

#[test]
fn registration_after_a_backend_startup_timeout_reports_its_actual_result() {
    let registrar = Arc::new(MemoryRegistrar::default());
    let shortcuts = Shortcuts::new(
        registrar.clone(),
        Some(hold("F1")),
        always_running(),
        Some(ShortcutRegistrationError::PortalUnavailable),
    );
    assert!(shortcuts.registration_active());
    assert_eq!(shortcuts.failure(), None);
    assert!(registrar.holds(&binding(&control("F1"))));

    *registrar.fail_register.lock().expect("fail register") = Some(binding_id("F2").0);
    let conflict = Shortcuts::new(
        registrar,
        Some(hold("F2")),
        always_running(),
        Some(ShortcutRegistrationError::PortalUnavailable),
    );
    assert!(!conflict.registration_active());
    assert_eq!(
        conflict.failure(),
        Some(ShortcutRegistrationError::ConflictOrUnsupported)
    );
}

#[test]
fn paused_dictation_releases_the_shortcuts_and_resuming_restores_them() {
    let (registrar, running, shortcuts) = switchable_memory_shortcuts("F1");
    running.store(false, Ordering::SeqCst);
    shortcuts.sync_lifecycle().expect("pause");
    assert!(registrar.registered().is_empty());

    shortcuts
        .apply_update(hold("F2"))
        .expect("a change while paused");
    assert!(
        registrar.registered().is_empty(),
        "nothing registers while paused"
    );
    assert_eq!(
        shortcuts.input(binding_id("F2"), KeyTransition::Pressed),
        None
    );

    running.store(true, Ordering::SeqCst);
    shortcuts.sync_lifecycle().expect("resume");
    assert_eq!(registrar.registered(), BTreeSet::from([binding_id("F2").0]));
    assert!(shortcuts.registration_active());
    shortcuts.input(binding_id("F2"), KeyTransition::Released);
    assert_eq!(
        shortcuts.input(binding_id("F2"), KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Hold))
    );
}

#[test]
fn capture_suspension_ignores_presses_and_forgets_held_keys() {
    let (registrar, running, shortcuts) = switchable_memory_shortcuts("F1");
    let hold = binding_id("F1");
    assert_eq!(
        shortcuts.input(hold, KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Hold))
    );

    shortcuts.suspend_for_capture().expect("suspend");
    assert!(registrar.registered().is_empty());
    assert_eq!(shortcuts.input(hold, KeyTransition::Pressed), None);
    assert!(shortcuts.registration_active());

    // A pause and resume during the capture keep the shortcuts released.
    running.store(false, Ordering::SeqCst);
    shortcuts.sync_lifecycle().expect("pause");
    running.store(true, Ordering::SeqCst);
    shortcuts.sync_lifecycle().expect("resume");
    assert!(registrar.registered().is_empty());

    shortcuts.resume_after_capture().expect("resume");
    assert!(!registrar.registered().is_empty());
    shortcuts.input(hold, KeyTransition::Released);
    assert_eq!(
        shortcuts.input(hold, KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Hold))
    );
}

#[test]
fn presses_ignore_repeats_and_only_hold_releases_reach_the_worker() {
    for (mode, trigger, release) in [
        (
            RecordingMode::Hold,
            RecordingTrigger::Hold,
            Some(ShortcutAction::HoldReleased),
        ),
        (RecordingMode::Toggle, RecordingTrigger::Toggle, None),
    ] {
        let (_registrar, shortcuts) = memory_shortcuts("F1");
        shortcuts
            .apply_update(recording("F2", mode))
            .expect("recording shortcut");
        let binding = binding_id("F2");
        let press = Some(ShortcutAction::Pressed(trigger));
        assert_eq!(
            shortcuts.input(binding, KeyTransition::Pressed),
            press,
            "{mode:?}"
        );
        assert_eq!(
            shortcuts.input(binding, KeyTransition::Pressed),
            None,
            "{mode:?} repeat"
        );
        assert_eq!(
            shortcuts.input(binding, KeyTransition::Released),
            release,
            "{mode:?}"
        );
        assert_eq!(
            shortcuts.input(binding, KeyTransition::Pressed),
            press,
            "{mode:?} next press"
        );
    }
}

#[test]
fn a_key_released_while_paused_presses_again_after_resuming() {
    let (_registrar, running, shortcuts) = switchable_memory_shortcuts("F1");
    let hold = binding_id("F1");
    let set_running = |value: bool| {
        running.store(value, Ordering::SeqCst);
        shortcuts.sync_lifecycle().expect("lifecycle sync");
    };

    shortcuts.input(hold, KeyTransition::Pressed);
    set_running(false);
    assert_eq!(shortcuts.input(hold, KeyTransition::Released), None);
    assert_eq!(shortcuts.input(hold, KeyTransition::Pressed), None);
    set_running(true);
    shortcuts.input(hold, KeyTransition::Released);
    assert_eq!(
        shortcuts.input(hold, KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Hold))
    );
}

#[test]
fn a_release_before_its_press_leaves_nothing_stuck() {
    let (_registrar, shortcuts) = memory_shortcuts("F1");
    let hold = binding_id("F1");
    assert_eq!(
        shortcuts.input(hold, KeyTransition::Released),
        Some(ShortcutAction::HoldReleased)
    );
    assert_eq!(
        shortcuts.input(hold, KeyTransition::Pressed),
        Some(ShortcutAction::Pressed(RecordingTrigger::Hold))
    );
}

#[test]
fn a_lost_shortcut_backend_unregisters_everything_and_forgets_held_keys() {
    let (_registrar, shortcuts) = memory_shortcuts("F1");
    shortcuts.input(binding_id("F1"), KeyTransition::Pressed);
    shortcuts.capability_lost(ShortcutRegistrationError::SessionClosed);
    assert!(!shortcuts.registration_active());
    assert_eq!(
        shortcuts.failure(),
        Some(ShortcutRegistrationError::SessionClosed)
    );
    assert_eq!(
        shortcuts.input(binding_id("F1"), KeyTransition::Pressed),
        None
    );
}

/// Native registration may wait on the event-loop thread, which delivers input, so it must
/// never run under a lock input takes.
#[test]
fn native_registration_never_runs_under_the_input_locks() {
    let registrar = Arc::new(MemoryRegistrar::default());
    let (running, probe) = lifecycle_switch();
    let shortcuts = Arc::new(Shortcuts::new(
        registrar.clone(),
        Some(hold("F1")),
        probe,
        None,
    ));
    let native_calls = Arc::new(AtomicUsize::new(0));
    let locked_calls = Arc::new(AtomicUsize::new(0));
    let (counted, locked) = (Arc::clone(&native_calls), Arc::clone(&locked_calls));
    let probed = Arc::downgrade(&shortcuts);
    assert!(
        registrar
            .probe
            .set(Box::new(move || {
                let Some(shortcuts) = probed.upgrade() else {
                    return;
                };
                counted.fetch_add(1, Ordering::SeqCst);
                if shortcuts.pressed.try_lock().is_err() || shortcuts.published.try_lock().is_err()
                {
                    locked.fetch_add(1, Ordering::SeqCst);
                }
            }))
            .is_ok()
    );
    shortcuts.set_cancel_binding(Some(escape()));
    let calls = || native_calls.load(Ordering::SeqCst);
    let observed = std::cell::Cell::new(calls());
    let native_step = |step: &str| {
        assert!(calls() > observed.get(), "{step} must reach the registrar");
        observed.set(calls());
    };

    shortcuts.apply_update(hold("F2")).expect("apply update");
    native_step("apply update");
    shortcuts.observe_dictation(true);
    shortcuts.sync_cancel_shortcut();
    native_step("cancel shortcut");
    running.store(false, Ordering::SeqCst);
    shortcuts.sync_lifecycle().expect("pause");
    native_step("pause");
    running.store(true, Ordering::SeqCst);
    shortcuts.sync_lifecycle().expect("resume");
    native_step("resume");
    shortcuts
        .suspend_for_capture()
        .expect("suspend for capture");
    native_step("capture suspension");
    shortcuts
        .resume_after_capture()
        .expect("resume after capture");
    native_step("capture resumption");
    shortcuts.shutdown();
    native_step("shutdown");

    assert_eq!(locked_calls.load(Ordering::SeqCst), 0);
    assert!(registrar.registered().is_empty());
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
        cancel_shortcut: Some(Settings::default().recording_shortcut),
        ..Settings::default()
    };
    assert_eq!(recording_binding(&settings), None);
    assert_eq!(cancel_binding(&settings), None);
    assert!(recording_binding(&Settings::default()).is_some());
}
