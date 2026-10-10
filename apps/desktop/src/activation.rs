use std::sync::Mutex;

use whispernow_domain::ipc::Phase;

/// Focus requests from a second launch. A request made before setup finishes, or while a
/// dictation runs, waits until the next snapshot that is not busy.
#[derive(Debug, Default)]
pub(crate) struct ActivationGate {
    activation_deferred: Mutex<bool>,
}

impl ActivationGate {
    /// A second launch must not steal focus from the current transcription target. `phase` is
    /// read under the gate's lock and is `None` until setup has managed the application state,
    /// so a request racing setup is never lost.
    pub(crate) fn request_activation(&self, phase: impl FnOnce() -> Option<Phase>) -> bool {
        let mut deferred = self
            .activation_deferred
            .lock()
            .expect("activation lock poisoned");
        if phase().as_ref().is_none_or(Phase::is_busy) {
            *deferred = true;
            false
        } else {
            true
        }
    }

    pub(crate) fn release_deferred_activation(&self, phase: &Phase) -> bool {
        if phase.is_busy() {
            return false;
        }
        std::mem::take(
            &mut *self
                .activation_deferred
                .lock()
                .expect("activation lock poisoned"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_instance_focus_waits_until_recording_and_processing_finish() {
        let gate = ActivationGate::default();
        assert!(!gate.request_activation(|| Some(Phase::Recording {
            trigger: whispernow_domain::ipc::RecordingTrigger::Hold,
            session: 1,
        })));
        assert!(!gate.release_deferred_activation(&Phase::Processing { cleaning_up: false }));
        assert!(gate.release_deferred_activation(&Phase::Idle));
        assert!(!gate.release_deferred_activation(&Phase::Idle));
        assert!(gate.request_activation(|| Some(Phase::Idle)));
    }

    #[test]
    fn a_launch_before_setup_finishes_is_activated_once_setup_is_done() {
        let gate = ActivationGate::default();
        assert!(!gate.request_activation(|| None));
        assert!(gate.release_deferred_activation(&Phase::Idle));
    }
}
