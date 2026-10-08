//! The loaded cleanup model and its contexts, used from one thread.

use std::{ops::ControlFlow, path::Path, rc::Rc};

use whispernow_core::llm::{
    DraftText, LlmDecoding, LlmPrompt,
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
    /// the model from `model_path`. A CUDA pack that does not load leaves the other backends.
    pub(crate) fn load(
        library_directory: &Path,
        cuda: Option<&Path>,
        model_path: &Path,
        threads: i32,
    ) -> Result<Self, LlamaError> {
        let llama = Llama::load(library_directory)?;
        if let Some(cuda) = cuda
            && let Err(error) = llama.load_cuda(cuda)
        {
            tracing::warn!(target: "whispernow::llama", %error, "the CUDA pack did not load; using the other backends");
        }
        let model = LlamaModel::load(llama, model_path)?;
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

    /// Decodes the reply to `prompt`: speculatively when `decoding` carries a draft.
    /// `between_passes` sees the reply so far and stops the decode by breaking.
    pub(crate) fn decode(
        &mut self,
        prompt: &LlmPrompt,
        decoding: &LlmDecoding,
        between_passes: impl FnMut(&[u8]) -> ControlFlow<()>,
    ) -> Result<LocalDecodeOutput, EngineError> {
        let rendered = local::render_prompt(&self.model.bos_text(), prompt);
        // Core always gives this model a budget; without one the context cannot be sized.
        let max_tokens = decoding
            .max_output_tokens
            .and_then(|tokens| usize::try_from(tokens).ok())
            .unwrap_or(usize::MAX);
        let prompt_tokens = self
            .model
            .tokenize(rendered.as_bytes(), true)
            .map_err(EngineError::Decode)?
            .len();
        let context_size = prompt_tokens
            .saturating_add(max_tokens)
            .saturating_add(2)
            .max(MIN_CONTEXT)
            .next_multiple_of(CONTEXT_STEP);
        let context_size = u32::try_from(context_size)
            .map_err(|_| EngineError::Decode(LocalDecodeError::PromptTooLong))?;
        let context = self.context(context_size)?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "llama.cpp keeps the penalty as f32"
        )]
        let repeat_penalty = local::REPEAT_PENALTY as f32;
        local::decode(
            context,
            &LocalDecodeRequest {
                prompt: &rendered,
                max_tokens,
                repeat_penalty,
                repeat_last_n: local::REPEAT_LAST_N,
                draft: decoding.draft.as_ref().map(DraftText::as_str),
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
