use std::{
    collections::BTreeSet,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};

use super::{ShortcutBinding, ShortcutRegistrar, ShortcutRegistrationError};

#[derive(Default)]
pub(crate) struct MemoryRegistrar {
    registered: Mutex<Vec<ShortcutBinding>>,
    pub(crate) fail_register: Mutex<Option<u32>>,
    pub(crate) fail_unregister: Mutex<Option<u32>>,
    /// Behaves like a desktop portal that asks before every registration.
    pub(crate) refuses_transient: AtomicBool,
    pub(crate) probe: OnceLock<Box<dyn Fn() + Send + Sync>>,
    pub(crate) gate: Mutex<Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>>,
}

impl MemoryRegistrar {
    pub(crate) fn bindings(&self) -> Vec<ShortcutBinding> {
        self.registered.lock().expect("registered").clone()
    }

    pub(crate) fn registered(&self) -> BTreeSet<u32> {
        self.bindings().iter().map(|binding| binding.id.0).collect()
    }

    pub(crate) fn holds(&self, binding: &ShortcutBinding) -> bool {
        self.registered().contains(&binding.id.0)
    }

    fn probe(&self) {
        if let Some(probe) = self.probe.get() {
            probe();
        }
    }
}

impl ShortcutRegistrar for MemoryRegistrar {
    fn register(&self, value: &ShortcutBinding) -> Result<(), ShortcutRegistrationError> {
        self.probe();
        let gate = self.gate.lock().expect("registration gate").take();
        if let Some((entered, release)) = gate {
            entered.send(()).expect("registration observer");
            release
                .recv_timeout(Duration::from_secs(10))
                .expect("registration released");
        }
        if *self.fail_register.lock().expect("fail register") == Some(value.id.0) {
            return Err(ShortcutRegistrationError::ConflictOrUnsupported);
        }
        self.registered
            .lock()
            .expect("registered")
            .push(value.clone());
        Ok(())
    }

    fn unregister(&self, value: &ShortcutBinding) -> Result<(), ShortcutRegistrationError> {
        self.probe();
        if *self.fail_unregister.lock().expect("fail unregister") == Some(value.id.0) {
            return Err(ShortcutRegistrationError::ConflictOrUnsupported);
        }
        self.registered
            .lock()
            .expect("registered")
            .retain(|registered| registered.id != value.id);
        Ok(())
    }

    fn supports_transient_shortcuts(&self) -> bool {
        !self.refuses_transient.load(Ordering::SeqCst)
    }
}
