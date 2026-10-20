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
    DraftText, LlmError, LlmFinishReason, LlmFuture, LlmPrompt, LlmResponse,
    local::{LocalDecodeError, LocalDecodeOutput, LocalFinish},
};

use self::engine::{EngineError, LocalCleanupEngine, default_threads};
use crate::llama::{Backend, LlamaError};
use whispernow_domain::ModelId;

/// Finds a cleanup model's GGUF by its registry id when it is needed; `None` while it is not
/// installed, or the id is not a cleanup model's.
pub type CleanupModelLocator = Arc<dyn Fn(&ModelId) -> Option<PathBuf> + Send + Sync>;

/// Finds the installed CUDA pack's directory when a model loads; `None` while there is none.
pub type CudaPackLocator = Arc<dyn Fn() -> Option<PathBuf> + Send + Sync>;

enum Command {
    Load(ModelId),
    Reload,
    /// Frees the model, then signals the sender.
    Unload(Option<mpsc::Sender<()>>),
    Decode(Job),
}

struct Job {
    model_id: ModelId,
    prompt: LlmPrompt,
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
    /// The engine with llama.cpp's libraries from `library_directory`, the CUDA pack found by
    /// `cuda` and the model found by `model`. Nothing loads until the first cleanup or
    /// [`Self::preload`].
    #[must_use]
    pub fn new(
        library_directory: PathBuf,
        model: CleanupModelLocator,
        cuda: CudaPackLocator,
    ) -> Self {
        Self {
            shared: Arc::new(Shared {
                library_directory,
                model_locator: model,
                cuda_locator: cuda,
                use_gpu: AtomicBool::new(true),
                shutdown: AtomicBool::new(false),
                held: AtomicBool::new(false),
            }),
            worker: Mutex::new(None),
        }
    }

    /// Loads the model `model_id` names in the background, so the first cleanup does not
    /// wait for it.
    pub fn preload(&self, model_id: ModelId) {
        let _ = self.send(Command::Load(model_id));
    }

    /// Loads the loaded model again in the background, so a newly installed CUDA pack takes
    /// it over; nothing happens while no model is loaded.
    pub fn reload(&self) {
        if self.lock_worker().is_some() {
            let _ = self.send(Command::Reload);
        }
    }

    /// Whether models load on a GPU; off keeps them on the CPU. A change frees the loaded
    /// model, so the next load follows it.
    pub fn set_use_gpu(&self, use_gpu: bool) {
        if self.shared.use_gpu.swap(use_gpu, Ordering::AcqRel) != use_gpu {
            self.unload();
        }
    }

    /// Frees the model, if it is loaded, in the background.
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

    /// Sends a command to the decode thread, starting it if needed. After the thread
    /// panicked every command fails.
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
    model_locator: CleanupModelLocator,
    cuda_locator: CudaPackLocator,
    use_gpu: AtomicBool,
    shutdown: AtomicBool,
    /// Prevents a model from loading while its file is being removed.
    held: AtomicBool,
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
    pub fn complete(
        &self,
        model: ModelId,
        prompt: LlmPrompt,
        max_output_tokens: usize,
        draft: DraftText,
    ) -> LlmFuture<'_> {
        Box::pin(async move {
            let (reply, replied) = oneshot::channel();
            self.send(Command::Decode(Job {
                model_id: model,
                prompt,
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
        })
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
        return Err(LlamaError::NotInstalled);
    }
    let path = (shared.model_locator)(model_id).ok_or(LlamaError::NotInstalled)?;
    let use_gpu = shared.use_gpu.load(Ordering::Acquire);
    let cuda = use_gpu.then(|| (shared.cuda_locator)()).flatten();
    let threads = default_threads();
    let loaded = LocalCleanupEngine::load(
        &shared.library_directory,
        cuda.as_deref(),
        &path,
        threads,
        use_gpu,
    )
    .inspect_err(|error| {
        tracing::warn!(target: "whispernow::local_cleanup", %error, model = model_id.as_str(), "the cleanup model did not load");
    })?;
    let (kind, device) = match loaded.backend() {
        Backend::Gpu(device) => ("gpu", device.as_str()),
        Backend::Cpu => ("cpu", ""),
    };
    tracing::info!(target: "whispernow::local_cleanup", model = model_id.as_str(), backend = kind, device, threads, "cleanup model loaded");
    Ok(loaded)
}

/// The decode thread: loads, frees and decodes in the order commands arrive.
fn run(commands: &mpsc::Receiver<Command>, shared: &Shared) {
    let shutdown = || shared.shutdown.load(Ordering::Acquire);
    let mut engine: Option<(ModelId, LocalCleanupEngine)> = None;
    while let Ok(command) = commands.recv() {
        match command {
            Command::Load(model_id) => {
                if !shutdown() {
                    let _ = load(shared, &mut engine, &model_id);
                }
            }
            Command::Reload => {
                if let Some((model_id, _)) = engine.take()
                    && !shutdown()
                {
                    let _ = load(shared, &mut engine, &model_id);
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
                        loaded.decode(&job.prompt, job.max_output_tokens, &job.draft, |_| {
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
    use std::time::Duration;

    use super::*;

    fn service(installed: bool) -> LocalCleanupService {
        LocalCleanupService::new(
            PathBuf::from("/nonexistent/llama"),
            Arc::new(move |_| installed.then(|| PathBuf::from("/nonexistent/model.gguf"))),
            Arc::new(|| None),
        )
    }

    async fn complete(service: &LocalCleanupService, model: &str) -> Result<LlmResponse, LlmError> {
        let transcript = "um so the meeting is at three";
        service
            .complete(
                ModelId::new(model).expect("model ID"),
                whispernow_core::llm::build_local_cleanup_prompt(transcript),
                64,
                DraftText::from_transcript(transcript),
            )
            .await
    }

    #[tokio::test]
    async fn a_missing_model_or_library_is_an_invalid_request_and_never_a_hang() {
        for installed in [false, true] {
            let result = complete(&service(installed), "cleanup").await;
            assert!(matches!(result, Err(LlmError::InvalidRequest)));
        }
    }

    #[tokio::test]
    async fn nothing_loads_while_the_model_is_held_unloaded() {
        let loads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = Arc::clone(&loads);
        let service = LocalCleanupService::new(
            PathBuf::from("/nonexistent/llama"),
            Arc::new(move |_| {
                counted.fetch_add(1, Ordering::SeqCst);
                None
            }),
            Arc::new(|| None),
        );
        service.preload(ModelId::new("cleanup").expect("model ID"));
        let held = service
            .hold_unloaded(Duration::from_secs(5))
            .expect("an idle model is freed at once");
        let before = loads.load(Ordering::SeqCst);
        let result = complete(&service, "cleanup").await;
        assert!(matches!(result, Err(LlmError::InvalidRequest)));
        assert_eq!(loads.load(Ordering::SeqCst), before);
        drop(held);
        let _ = complete(&service, "cleanup").await;
        assert_eq!(loads.load(Ordering::SeqCst), before + 1);
    }

    #[tokio::test]
    async fn each_request_and_preload_looks_up_the_model_it_names() {
        let looked_up = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&looked_up);
        let service = LocalCleanupService::new(
            PathBuf::from("/nonexistent/llama"),
            Arc::new(move |model_id: &ModelId| {
                recorded
                    .lock()
                    .expect("not poisoned")
                    .push(model_id.as_str().to_owned());
                None
            }),
            Arc::new(|| None),
        );
        service.preload(ModelId::new("small").expect("model ID"));
        for model in ["large", "small"] {
            let result = complete(&service, model).await;
            assert!(matches!(result, Err(LlmError::InvalidRequest)));
        }
        assert_eq!(
            *looked_up.lock().expect("not poisoned"),
            ["small", "large", "small"]
        );
    }
}
