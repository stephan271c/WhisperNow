//! The local cleanup provider: the cleanup model run in process on llama.cpp.
//!
//! One dedicated thread owns the loaded model and decodes one request at a time, so the
//! async runtime's threads never block on it.

mod engine;

use std::{
    ops::ControlFlow,
    path::PathBuf,
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::JoinHandle,
};

use tokio::sync::oneshot;
use whispernow_core::llm::{
    DraftText, LlmError, LlmFinishReason, LlmResponse,
    local::{LocalDecodeError, LocalDecodeOutput, LocalFinish},
};

use self::engine::{EngineError, LocalCleanupEngine, default_threads};
use crate::{llama::LlamaError, models::ModelStore};
use whispernow_domain::{ModelId, ipc::CleanupDevice};

enum Command {
    Load(ModelId),
    Unload(Option<mpsc::Sender<()>>),
    Decode(Job),
}

struct Job {
    model_id: ModelId,
    transcript: String,
    max_output_tokens: usize,
    draft: DraftText,
    /// Closed when the caller stops waiting, as on cancellation or timeout.
    reply: oneshot::Sender<Result<LocalDecodeOutput, EngineError>>,
}

struct Worker {
    commands: mpsc::Sender<Command>,
    thread: JoinHandle<()>,
}

pub struct LocalCleanupService {
    shared: Arc<Shared>,
    worker: Mutex<Option<Worker>>,
}

impl LocalCleanupService {
    /// Nothing loads until the first cleanup or [`Self::preload`]. `observer` hears each change
    /// of the device the loaded model runs on, and `None` when it is freed.
    #[must_use]
    pub fn new(
        library_directory: PathBuf,
        store: Arc<ModelStore>,
        observer: impl Fn(Option<CleanupDevice>) + Send + Sync + 'static,
    ) -> Self {
        Self {
            shared: Arc::new(Shared {
                library_directory,
                store,
                use_gpu: AtomicBool::new(true),
                shutdown: AtomicBool::new(false),
                held: AtomicBool::new(false),
                observer: Box::new(observer),
            }),
            worker: Mutex::new(None),
        }
    }

    pub fn preload(&self, model_id: ModelId) {
        let _ = self.send(Command::Load(model_id));
    }

    /// Whether models load on a GPU; off keeps them on the CPU. A change frees the loaded
    /// model, so the next load follows it.
    pub fn set_use_gpu(&self, use_gpu: bool) {
        if self.shared.use_gpu.swap(use_gpu, Ordering::AcqRel) != use_gpu {
            self.unload();
        }
    }

    pub fn unload(&self) {
        if self.lock_worker().is_some() {
            let _ = self.send(Command::Unload(None));
        }
    }

    /// Frees the model, waiting at most `timeout`, and keeps it from loading again until the
    /// returned guard is dropped, so its file can be removed. `None` if it was not freed in
    /// time, as while a long cleanup runs.
    pub fn hold_unloaded(&self, timeout: std::time::Duration) -> Option<HeldUnloaded<'_>> {
        self.shared.held.store(true, Ordering::Release);
        let held = HeldUnloaded(self);
        if self.lock_worker().is_none() {
            return Some(held);
        }
        let (freed, waiting) = mpsc::channel();
        (self.send(Command::Unload(Some(freed))).is_ok() && waiting.recv_timeout(timeout).is_ok())
            .then_some(held)
    }

    fn lock_worker(&self) -> std::sync::MutexGuard<'_, Option<Worker>> {
        self.worker.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// After the decode thread panicked every command fails.
    fn send(&self, command: Command) -> Result<(), LlmError> {
        let mut worker = self.lock_worker();
        let worker = match worker.as_ref() {
            Some(running) => running,
            None => self.spawn(&mut worker)?,
        };
        worker
            .commands
            .send(command)
            .map_err(|_| LlmError::InvalidResponse)
    }

    fn spawn<'worker>(
        &self,
        worker: &'worker mut Option<Worker>,
    ) -> Result<&'worker Worker, LlmError> {
        let (commands, commands_rx) = mpsc::channel();
        let shared = Arc::clone(&self.shared);
        let thread = std::thread::Builder::new()
            .name("whispernow-cleanup".to_owned())
            .spawn(move || run(&commands_rx, &shared))
            .map_err(|_| LlmError::TemporarilyUnavailable)?;
        Ok(worker.insert(Worker { commands, thread }))
    }
}

pub struct HeldUnloaded<'service>(&'service LocalCleanupService);

impl Drop for HeldUnloaded<'_> {
    fn drop(&mut self) {
        self.0.shared.held.store(false, Ordering::Release);
    }
}

struct Shared {
    library_directory: PathBuf,
    store: Arc<ModelStore>,
    use_gpu: AtomicBool,
    shutdown: AtomicBool,
    held: AtomicBool,
    observer: Box<dyn Fn(Option<CleanupDevice>) + Send + Sync>,
}

impl Drop for LocalCleanupService {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        if let Some(worker) = self.lock_worker().take() {
            drop(worker.commands);
            let _ = worker.thread.join();
        }
    }
}

impl LocalCleanupService {
    pub async fn complete(
        &self,
        model: ModelId,
        transcript: String,
        max_output_tokens: usize,
        draft: DraftText,
    ) -> Result<LlmResponse, LlmError> {
        let (reply, replied) = oneshot::channel();
        self.send(Command::Decode(Job {
            model_id: model,
            transcript,
            max_output_tokens,
            draft,
            reply,
        }))?;
        match replied.await {
            Ok(Ok(output)) => Ok(response(&output)),
            Ok(Err(EngineError::Decode(LocalDecodeError::Stopped))) => Err(LlmError::Cancelled),
            Ok(Err(
                EngineError::Decode(LocalDecodeError::PromptTooLong) | EngineError::Load(_),
            )) => Err(LlmError::InvalidRequest),
            Ok(Err(EngineError::Decode(_))) | Err(_) => Err(LlmError::InvalidResponse),
        }
    }
}

fn response(output: &LocalDecodeOutput) -> LlmResponse {
    LlmResponse {
        text: String::from_utf8_lossy(&output.text).into_owned(),
        finish_reason: match output.finish {
            LocalFinish::EndOfTurn => LlmFinishReason::Completed,
            LocalFinish::OutputLimit | LocalFinish::ContextFull => LlmFinishReason::OutputLimit,
        },
    }
}

/// The engine for `model_id`: the loaded one, or loaded now. Only one model is kept, so
/// another is freed first.
fn load<'engine>(
    shared: &Shared,
    engine: &'engine mut Option<(ModelId, LocalCleanupEngine)>,
    model_id: &ModelId,
) -> Result<&'engine mut LocalCleanupEngine, LlamaError> {
    let (_, loaded) = match engine.take().filter(|(loaded, _)| loaded == model_id) {
        Some(kept) => engine.insert(kept),
        None => engine.insert((model_id.to_owned(), load_engine(shared, model_id)?)),
    };
    Ok(loaded)
}

fn load_engine(shared: &Shared, model_id: &ModelId) -> Result<LocalCleanupEngine, LlamaError> {
    if shared.held.load(Ordering::Acquire) {
        return Err(LlamaError::Held);
    }
    let path = shared
        .store
        .cleanup_model_path(model_id)?
        .ok_or(LlamaError::NotInstalled)?;
    let threads = default_threads();
    let loaded = LocalCleanupEngine::load(
        &shared.library_directory,
        &path,
        threads,
        shared.use_gpu.load(Ordering::Acquire),
    )
    .inspect_err(|error| {
        tracing::warn!(target: "whispernow::local_cleanup", %error, model = model_id.as_str(), "the cleanup model did not load");
    })?;
    tracing::info!(target: "whispernow::local_cleanup", model = model_id.as_str(), threads, "cleanup model loaded");
    Ok(loaded)
}

/// Tells the observer where the loaded model runs, when that changes. Called before each wait
/// for a command, so every command's effect is reported.
fn report_device(
    shared: &Shared,
    engine: Option<&(ModelId, LocalCleanupEngine)>,
    reported: &mut Option<CleanupDevice>,
) {
    let device = engine.map(|(_, loaded)| loaded.device().clone());
    if device != *reported {
        (shared.observer)(device.clone());
        *reported = device;
    }
}

/// The decode thread: loads, frees and decodes in the order commands arrive.
fn run(commands: &mpsc::Receiver<Command>, shared: &Shared) {
    let shutdown = || shared.shutdown.load(Ordering::Acquire);
    let mut engine: Option<(ModelId, LocalCleanupEngine)> = None;
    let mut reported = None;
    loop {
        report_device(shared, engine.as_ref(), &mut reported);
        let Ok(command) = commands.recv() else {
            break;
        };
        match command {
            Command::Load(model_id) => {
                if !shutdown()
                    && let Ok(loaded) = load(shared, &mut engine, &model_id)
                    && let Err(error) = loaded.warm_up()
                {
                    tracing::warn!(target: "whispernow::local_cleanup", %error, "the cleanup model did not warm up");
                }
            }
            Command::Unload(freed) => {
                engine = None;
                if let Some(freed) = freed {
                    let _ = freed.send(());
                }
            }
            Command::Decode(job) => {
                let stopped = || shutdown() || job.reply.is_closed();
                if stopped() {
                    continue;
                }
                let result = match load(shared, &mut engine, &job.model_id) {
                    Ok(loaded) => {
                        loaded.decode(&job.transcript, job.max_output_tokens, &job.draft, |_| {
                            if stopped() {
                                ControlFlow::Break(())
                            } else {
                                ControlFlow::Continue(())
                            }
                        })
                    }
                    Err(error) => Err(EngineError::Load(error)),
                };
                if let Err(EngineError::Decode(
                    error @ (LocalDecodeError::Decode(_)
                    | LocalDecodeError::Logits
                    | LocalDecodeError::Memory),
                )) = &result
                {
                    // The context may be unusable: load afresh next time.
                    tracing::warn!(target: "whispernow::local_cleanup", %error, "the cleanup model failed to decode");
                    engine = None;
                }
                let _ = job.reply.send(result);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, time::Duration};

    use tempfile::TempDir;
    use whispernow_domain::{ModelDefinition, ModelLayout, ModelRegistry, ModelSource};

    use super::*;
    use crate::{AppPaths, DownloadTimeouts, ModelStoreError};

    const MODEL_FILE: &str = "model.gguf";

    fn cleanup_model(id: &str) -> ModelDefinition {
        ModelDefinition {
            id: ModelId::new(id).expect("model ID"),
            display_name: id.to_owned(),
            source: ModelSource {
                url: String::new(),
                archive_size_bytes: 0,
                sha256: String::new(),
            },
            layout: ModelLayout::Gguf {
                model: MODEL_FILE.to_owned(),
            },
            languages: Vec::new(),
            benchmark: None,
        }
    }

    fn service(installed: &[&str]) -> (TempDir, LocalCleanupService) {
        let root = tempfile::tempdir().expect("temporary root");
        let paths = AppPaths::under(root.path());
        for id in installed {
            let directory = paths.models_dir().join(id);
            fs::create_dir_all(&directory).expect("model directory");
            fs::write(directory.join(MODEL_FILE), b"GGUF").expect("model file");
        }
        let registry = ModelRegistry::new(["small", "large"].map(cleanup_model).to_vec());
        let store =
            ModelStore::with_registry_and_timeouts(registry, paths, DownloadTimeouts::default());
        let service =
            LocalCleanupService::new(PathBuf::from("/nonexistent/llama"), Arc::new(store), |_| {});
        (root, service)
    }

    fn model_id(model: &str) -> ModelId {
        ModelId::new(model).expect("model ID")
    }

    fn complete(service: &LocalCleanupService, model: &str) -> Result<LlmResponse, LlmError> {
        let transcript = "um so the meeting is at three";
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime")
            .block_on(service.complete(
                model_id(model),
                transcript.to_owned(),
                64,
                DraftText::from_transcript(transcript),
            ))
    }

    #[test]
    fn a_missing_model_or_library_is_an_invalid_request_and_never_a_hang() {
        for installed in [&[][..], &["small"][..]] {
            let (_root, service) = service(installed);
            let result = complete(&service, "small");
            assert!(matches!(result, Err(LlmError::InvalidRequest)));
        }
    }

    #[test]
    fn nothing_loads_while_the_model_is_held_unloaded() {
        let (_root, service) = service(&["small"]);
        service.preload(model_id("small"));
        let held = service
            .hold_unloaded(Duration::from_secs(5))
            .expect("an idle model is freed at once");
        let load = || load_engine(&service.shared, &model_id("small")).err();
        assert!(matches!(load(), Some(LlamaError::Held)));
        let result = complete(&service, "small");
        assert!(matches!(result, Err(LlmError::InvalidRequest)));
        drop(held);
        // Past the model lookup; llama.cpp is not at the library directory.
        assert!(matches!(load(), Some(LlamaError::Library { .. })));
    }

    #[test]
    fn a_model_is_found_in_the_store_by_the_id_it_names() {
        let (_root, service) = service(&["small"]);
        let load = |model| load_engine(&service.shared, &model_id(model)).err();
        assert!(matches!(load("small"), Some(LlamaError::Library { .. })));
        assert!(matches!(load("large"), Some(LlamaError::NotInstalled)));
        assert!(matches!(
            load("unknown"),
            Some(LlamaError::Store(ModelStoreError::UnknownModel))
        ));
    }
}
