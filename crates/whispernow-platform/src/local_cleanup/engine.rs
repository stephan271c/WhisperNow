//! The loaded cleanup model and its contexts, used from one thread.

use std::{ops::ControlFlow, path::Path, rc::Rc};

use whispernow_core::llm::{
    DraftText, LlmPrompt,
    local::{self, DecodeContext, LocalDecodeError, LocalDecodeOutput, LocalDecodeRequest},
};

use crate::llama::{Backend, Llama, LlamaContext, LlamaError, LlamaModel};

/// The server's context per slot; contexts are never smaller, so the app decodes in the
/// same memory layout as the evaluation did.
const MIN_CONTEXT: usize = 8_192;
const CONTEXT_STEP: usize = 256;

/// The CPU threads to decode with: half the logical CPUs. On machines with two threads per
/// core that is one per physical core, `llama-server`'s default; it leaves the rest of the
/// machine usable during dictation.
#[must_use]
pub(crate) fn default_threads() -> i32 {
    let logical = std::thread::available_parallelism().map_or(2, std::num::NonZero::get);
    i32::try_from((logical / 2).max(1)).unwrap_or(1)
}

pub(crate) struct LocalCleanupEngine {
    model: Rc<LlamaModel>,
    context: Option<LlamaContext>,
    threads: i32,
}

impl LocalCleanupEngine {
    /// Loads llama.cpp from `library_directory`, with the CUDA pack in `cuda` when given, and
    /// the model from `model_path`, on the CPU unless `use_gpu`. A CUDA pack that does not
    /// load leaves the other backends.
    pub(crate) fn load(
        library_directory: &Path,
        cuda: Option<&Path>,
        model_path: &Path,
        threads: i32,
        use_gpu: bool,
    ) -> Result<Self, LlamaError> {
        let llama = Llama::load(library_directory)?;
        if let Some(cuda) = cuda
            && let Err(error) = llama.load_cuda(cuda)
        {
            tracing::warn!(target: "whispernow::llama", %error, "the CUDA pack did not load; using the other backends");
        }
        let model = LlamaModel::load(llama, model_path, use_gpu)?;
        Ok(Self {
            model: Rc::new(model),
            context: None,
            threads,
        })
    }

    #[must_use]
    pub(crate) fn backend(&self) -> &Backend {
        self.model.backend()
    }

    /// Decodes the reply to `prompt` speculatively from the draft.
    /// `between_passes` sees the reply so far and stops the decode by breaking.
    pub(crate) fn decode(
        &mut self,
        prompt: &LlmPrompt,
        max_output_tokens: usize,
        draft: &DraftText,
        between_passes: impl FnMut(&[u8]) -> ControlFlow<()>,
    ) -> Result<LocalDecodeOutput, EngineError> {
        let rendered = local::render_prompt(&self.model.bos_text(), prompt);
        let tokens = self
            .model
            .tokenize(rendered.as_bytes(), true)
            .map_err(EngineError::Decode)?;
        let context_size = tokens
            .len()
            .saturating_add(max_output_tokens)
            .saturating_add(2)
            .max(MIN_CONTEXT)
            .next_multiple_of(CONTEXT_STEP);
        let context_size = u32::try_from(context_size)
            .map_err(|_| EngineError::Decode(LocalDecodeError::PromptTooLong))?;
        let context = self.context(context_size)?;
        local::decode(
            context,
            LocalDecodeRequest {
                prompt: tokens,
                max_tokens: max_output_tokens,
                draft: Some(draft.as_str()),
            },
            between_passes,
        )
        .map_err(EngineError::Decode)
    }

    /// The kept context when it holds `context_size` tokens, else a new one that does.
    fn context(&mut self, context_size: u32) -> Result<&mut LlamaContext, EngineError> {
        let context = match self.context.take() {
            Some(context) if context.context_size() >= context_size as usize => context,
            // The old context is freed before the larger one is created.
            _ => LlamaContext::new(Rc::clone(&self.model), context_size, self.threads)
                .map_err(EngineError::Load)?,
        };
        Ok(self.context.insert(context))
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum EngineError {
    #[error(transparent)]
    Load(LlamaError),
    #[error(transparent)]
    Decode(LocalDecodeError),
}
