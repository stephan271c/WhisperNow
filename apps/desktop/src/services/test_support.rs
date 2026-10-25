use std::{
    fs,
    sync::{Arc, Mutex},
};

use tempfile::TempDir;
use whispernow_core::audio::{
    AudioBlock, AudioCapture, AudioDeviceChoice, AudioInput, AudioInputError, AudioLevelSummary,
    AudioSink,
};
use whispernow_domain::ipc::AudioDevice;
use whispernow_domain::{ModelDefinition, ModelId, Settings};
use whispernow_platform::{AppPaths, MemoryCredentialStore, SettingsStore};

use super::{
    Backend, Events,
    settings::SavedSettings,
    state::{State, StateCell},
};

enum Start {
    Never,
    Samples(Vec<f32>),
    Levels,
    Refuse(AudioInputError),
}

pub(crate) struct FakeAudioInput {
    pub(crate) calls: Arc<Mutex<Vec<&'static str>>>,
    devices: Result<Vec<AudioDevice>, AudioInputError>,
    start: Start,
    before_start: Option<Box<dyn Fn() + Send + Sync>>,
}

impl FakeAudioInput {
    fn with(start: Start) -> Self {
        Self {
            calls: Arc::default(),
            devices: Ok(Vec::new()),
            start,
            before_start: None,
        }
    }

    pub(crate) fn recording(samples: Vec<f32>) -> Self {
        Self::with(Start::Samples(samples))
    }

    pub(crate) fn levels() -> Self {
        Self::with(Start::Levels)
    }

    pub(crate) fn refusing(error: AudioInputError) -> Self {
        Self::with(Start::Refuse(error))
    }

    pub(crate) fn listing(devices: Result<Vec<AudioDevice>, AudioInputError>) -> Self {
        Self {
            devices,
            ..Self::with(Start::Never)
        }
    }

    pub(crate) fn before_start(mut self, hook: impl Fn() + Send + Sync + 'static) -> Self {
        self.before_start = Some(Box::new(hook));
        self
    }
}

impl AudioInput for FakeAudioInput {
    fn list_input_devices(&self) -> Result<Vec<AudioDevice>, AudioInputError> {
        self.devices.clone()
    }

    fn start(
        &self,
        _device: AudioDeviceChoice,
        sink: Arc<dyn AudioSink>,
    ) -> Result<Box<dyn AudioCapture>, AudioInputError> {
        if let Some(hook) = &self.before_start {
            hook();
        }
        self.calls.lock().expect("calls").push("start");
        match &self.start {
            Start::Never => panic!("this test does not record audio"),
            Start::Refuse(error) => return Err(error.clone()),
            Start::Samples(samples) => sink.samples(AudioBlock::new(samples.clone())),
            Start::Levels => {
                for rms in (0_u8..100).map(|step| f32::from(step) / 99.0) {
                    sink.observe(AudioLevelSummary {
                        peak: rms,
                        rms,
                        sample_count: 800,
                    });
                }
            }
        }
        Ok(Box::new(FakeCapture(Arc::clone(&self.calls))))
    }
}

struct FakeCapture(Arc<Mutex<Vec<&'static str>>>);

impl AudioCapture for FakeCapture {
    fn stop(self: Box<Self>) -> Result<(), AudioInputError> {
        self.0.lock().expect("calls").push("stop");
        Ok(())
    }
}

pub(crate) struct SettingsHarness {
    pub(crate) root: TempDir,
    pub(crate) paths: AppPaths,
    pub(crate) credentials: Arc<MemoryCredentialStore>,
    pub(crate) persistence: Arc<SettingsStore>,
    pub(crate) state: Arc<StateCell>,
    pub(crate) settings: Arc<SavedSettings>,
}

impl SettingsHarness {
    pub(crate) fn new(saved: Settings) -> Self {
        let root = tempfile::tempdir().expect("temporary root");
        let paths = AppPaths::under(root.path());
        fs::create_dir_all(paths.config_dir()).expect("config directory");
        let credentials = Arc::new(MemoryCredentialStore::default());
        let persistence = Arc::new(SettingsStore::with_store(
            paths.clone(),
            credentials.clone(),
        ));
        let state = Arc::new(StateCell::new(
            State::new(ModelId::default()),
            Events::channel().0,
        ));
        let settings = Arc::new(SavedSettings::new(
            Arc::clone(&persistence),
            Arc::clone(&state),
            saved,
        ));
        Self {
            root,
            paths,
            credentials,
            persistence,
            state,
            settings,
        }
    }
}

pub(crate) fn install_artifacts(paths: &AppPaths, model: &ModelDefinition) {
    for artifact in model.layout.required_artifacts() {
        let path = paths.models_dir().join(model.id.as_str()).join(artifact);
        fs::create_dir_all(path.parent().expect("artifact parent")).expect("model directory");
        fs::write(path, b"fixture").expect("model artifact");
    }
}

pub(crate) fn shut_down(backend: &Backend) {
    backend.request_shutdown();
    backend.shutdown();
}
