//! The loaded cleanup model and its contexts, used from one thread.

use std::{ops::ControlFlow, path::Path, rc::Rc};

use whispernow_core::llm::{
    DraftText,
    local::{self, DecodeContext, LocalDecodeError, LocalDecodeOutput, LocalDecodeRequest},
};
use whispernow_domain::ipc::CleanupDevice;

use crate::llama;

/// The server's context per slot; contexts are never smaller, so the app decodes in the
/// same memory layout as the evaluation did.
const MIN_CONTEXT: usize = 8_192;
const CONTEXT_STEP: usize = 256;

const WARM_UP_TRANSCRIPT: &str = "um so the meeting is at three no wait at four";
const WARM_UP_TOKENS: usize = 8;

/// The CPU threads to decode with: half the logical CPUs. On machines with two threads per
/// core that is one per physical core, `llama-server`'s default; it leaves the rest of the
/// machine usable during dictation.
#[must_use]
pub(crate) fn default_threads() -> i32 {
    let logical = std::thread::available_parallelism().map_or(2, std::num::NonZero::get);
    i32::try_from((logical / 2).max(1)).unwrap_or(1)
}

pub(crate) struct LocalCleanupEngine {
    model: Rc<llama::Model>,
    context: Option<llama::Context>,
    threads: i32,
}

impl LocalCleanupEngine {
    pub(crate) fn load(
        library_directory: &Path,
        model_path: &Path,
        threads: i32,
        use_gpu: bool,
    ) -> Result<Self, llama::LoadError> {
        let libraries = llama::Libraries::load(library_directory)?;
        let model = llama::Model::load(libraries, model_path, use_gpu)?;
        Ok(Self {
            model: Rc::new(model),
            context: None,
            threads,
        })
    }

    pub(crate) fn device(&self) -> &CleanupDevice {
        self.model.device()
    }

    /// Builds the context and decodes a few tokens, unless a cleanup already has, so the first
    /// cleanup does not. On an RTX 5070 with a cold shader cache that cleanup took 4.9 s,
    /// against 65 ms for the next.
    pub(crate) fn warm_up(&mut self) -> Result<(), EngineError> {
        if self.context.is_some() {
            return Ok(());
        }
        self.decode(
            WARM_UP_TRANSCRIPT,
            WARM_UP_TOKENS,
            &DraftText::from_transcript(WARM_UP_TRANSCRIPT),
            |_| ControlFlow::Continue(()),
        )
        .map(drop)
    }

    /// `between_passes` sees the reply so far and stops the decode by breaking.
    pub(crate) fn decode(
        &mut self,
        transcript: &str,
        max_output_tokens: usize,
        draft: &DraftText,
        between_passes: impl FnMut(&[u8]) -> ControlFlow<()>,
    ) -> Result<LocalDecodeOutput, EngineError> {
        let rendered = local::render_prompt(&self.model.bos_text(), transcript);
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

    fn context(&mut self, context_size: u32) -> Result<&mut llama::Context, EngineError> {
        keep_or_replace(
            &mut self.context,
            |context| context.context_size() >= context_size as usize,
            || {
                llama::Context::new(Rc::clone(&self.model), context_size, self.threads)
                    .map_err(EngineError::Load)
            },
        )
    }
}

/// Keeps the value in `slot` if `fits`, else replaces it with `create()`. A rejected value is
/// dropped before `create` runs: a context holds the KV cache and compute buffers (VRAM on the
/// GPU backend), so creating first would hold two at once.
fn keep_or_replace<T, E>(
    slot: &mut Option<T>,
    fits: impl FnOnce(&T) -> bool,
    create: impl FnOnce() -> Result<T, E>,
) -> Result<&mut T, E> {
    let value = match slot.take().filter(fits) {
        Some(kept) => kept,
        None => create()?,
    };
    Ok(slot.insert(value))
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum EngineError {
    #[error(transparent)]
    Load(llama::LoadError),
    #[error(transparent)]
    Decode(LocalDecodeError),
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    type Events = Rc<RefCell<Vec<&'static str>>>;

    struct Tracked {
        size: usize,
        events: Events,
    }

    impl Tracked {
        fn new(size: usize, events: &Events) -> Self {
            Self {
                size,
                events: Rc::clone(events),
            }
        }
    }

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.events.borrow_mut().push("dropped");
        }
    }

    #[test]
    fn a_rejected_value_is_dropped_before_its_replacement_is_created() {
        let events = Events::default();
        let mut slot = Some(Tracked::new(100, &events));

        let replacement = keep_or_replace(
            &mut slot,
            |old| old.size >= 200,
            || {
                events.borrow_mut().push("created");
                Ok::<_, ()>(Tracked::new(200, &events))
            },
        )
        .expect("replacement");

        assert_eq!(replacement.size, 200);
        assert_eq!(*events.borrow(), ["dropped", "created"]);
    }

    #[test]
    fn a_fitting_value_is_kept_without_creating_another() {
        let events = Events::default();
        let mut slot = Some(Tracked::new(300, &events));

        let kept = keep_or_replace(&mut slot, |old| old.size >= 200, || Err("created"))
            .expect("kept value");

        assert_eq!(kept.size, 300);
        assert!(events.borrow().is_empty());
    }
}
