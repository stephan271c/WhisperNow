//! The local cleanup model's decode loop, independent of the inference library.
//!
//! The platform crate runs it on llama.cpp; tests run it on a fake model. It decodes as
//! `llama-server` b11259 does for one request with `cache_prompt: false`, so the app's
//! output equals the server's on the same GGUF and backend:
//!
//! - The prompt is the chat template rendered as text and tokenized with special tokens
//!   parsed, so its leading BOS text is the model's BOS token.
//! - The prompt is decoded in the server's chunks: when the model's memory cannot be rolled
//!   back freely, the server breaks the batch at the start of the user turn and 4 + `n_ubatch`
//!   and 4 tokens before the end of the prompt, to place checkpoints there.
//! - Sampling is greedy over logits with llama.cpp's repeat penalty, whose window holds the
//!   prompt and every sampled token.
//! - With speculation on, each pass drafts up to 4 tokens: llama.cpp's `ngram-simple`
//!   lookup, replaced by the pre-cleaned transcript's next words when the word aligner finds
//!   them (the training side's `serving/llama-server-draft-text.patch`). The pass decodes the
//!   last sampled token with the draft, keeps the draft tokens the model samples itself, and
//!   removes the rest from the model's memory.
//! - Generation stops at an end-of-generation token, or when the reply reaches `max_tokens`
//!   or fills the context, which count as cut off.

use std::{
    fmt,
    ops::{ControlFlow, Range},
    time::{Duration, Instant},
};

use thiserror::Error;

use super::{LlmPrompt, align::WordAligner};

pub type Token = i32;

/// The chat template's turn opener. The server's message spans start at it, so the prompt's
/// first one marks the user turn.
const TURN_START: &str = "<|im_start|>";
/// `ngram-simple`'s n-gram (`--spec-ngram-simple-size-n 4`).
const NGRAM_SIZE: usize = 4;
/// llama.cpp's `repeat_penalty`, as the cleanup model's evaluation ran it.
pub const REPEAT_PENALTY: f64 = 1.05;
/// The repeat penalty's window (`repeat_last_n`), which covers the whole prompt and reply.
/// llama.cpp's default is the last 64 tokens; the model was chosen with the whole context
/// penalized, which keeps spoken numbers as digits more often.
pub const REPEAT_LAST_N: usize = 4_096;
/// `ngram-simple`'s draft length (`--spec-ngram-simple-size-m 4`), which is also the most
/// tokens a text draft carries.
pub const DRAFT_TOKENS: usize = 4;
/// Generated tokens re-tokenized in front of a text draft, so the draft tokenizes as it
/// would in context.
const DRAFT_TAIL_TOKENS: usize = 6;
/// The server's checkpoint offsets before the end of the prompt, besides `n_ubatch`.
const CHECKPOINT_OFFSET: usize = 4;

/// Renders the local cleanup prompt as `training/serving/chat_template.jinja` does with a
/// generation prompt. `bos` is the model's BOS token as text.
#[must_use]
pub fn render_prompt(bos: &str, prompt: &LlmPrompt) -> String {
    format!(
        "{bos}{TURN_START}user\n{}<|im_end|>\n{TURN_START}assistant\n",
        prompt.user()
    )
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum LocalDecodeError {
    #[error("the prompt could not be tokenized")]
    Tokenize,
    #[error("the prompt does not fit the context")]
    PromptTooLong,
    #[error("the model failed to decode a batch (status {0})")]
    Decode(i32),
    #[error("the model returned no logits")]
    Logits,
    #[error("the model's memory could not be rolled back")]
    Memory,
    #[error("the decode was stopped")]
    Stopped,
}

/// Which positions of a decoded batch produce logits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outputs {
    None,
    Last,
    All,
}

/// A loaded model with one sequence of memory: what the decode loop needs from llama.cpp.
pub trait DecodeContext {
    /// Tokenizes without adding BOS, parsing special-token text when `parse_special`.
    fn tokenize(&self, text: &[u8], parse_special: bool) -> Result<Vec<Token>, LocalDecodeError>;
    /// A token's text, with special tokens rendered empty.
    fn piece(&self, token: Token) -> Vec<u8>;
    fn is_end_of_generation(&self, token: Token) -> bool;
    fn vocabulary_size(&self) -> usize;
    /// Tokens the model never samples (`tokenizer.ggml.suppress_tokens`).
    fn suppressed_tokens(&self) -> Vec<Token>;
    fn context_size(&self) -> usize;
    fn batch_size(&self) -> usize;
    fn micro_batch_size(&self) -> usize;
    /// Whether the server would checkpoint this model's memory, and so split the prompt:
    /// when it cannot remove a sequence's tail freely, or only a bounded one.
    fn checkpoints_prompt(&self) -> bool;
    fn clear(&mut self);
    /// Decodes `tokens` at positions `first_position..`.
    fn decode(
        &mut self,
        tokens: &[Token],
        first_position: usize,
        outputs: Outputs,
    ) -> Result<(), LocalDecodeError>;
    /// The logits at `batch_index` of the last decode.
    fn logits(&mut self, batch_index: usize) -> Result<&[f32], LocalDecodeError>;
    /// Removes positions `from..` from the memory.
    fn remove_from(&mut self, from: usize) -> Result<(), LocalDecodeError>;
}

pub struct LocalDecodeRequest<'request> {
    /// The rendered prompt (`render_prompt`).
    pub prompt: &'request str,
    pub max_tokens: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    /// `Some` decodes speculatively with this text as the transcript draft, which may be
    /// empty; `None` decodes one token at a time.
    pub draft: Option<&'request str>,
}

impl fmt::Debug for LocalDecodeRequest<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalDecodeRequest")
            .field("prompt_byte_count", &self.prompt.len())
            .field("max_tokens", &self.max_tokens)
            .field("repeat_penalty", &self.repeat_penalty)
            .field("repeat_last_n", &self.repeat_last_n)
            .field("draft_byte_count", &self.draft.map(str::len))
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalFinish {
    EndOfTurn,
    /// The reply reached `max_tokens`: cut off.
    OutputLimit,
    /// The reply filled the context: cut off.
    ContextFull,
}

/// A decoded reply. The text is captured output, so `Debug` shows only its length.
#[derive(Clone, PartialEq)]
pub struct LocalDecodeOutput {
    /// The reply's bytes: every sampled token's text, the end-of-turn token's being empty.
    pub text: Vec<u8>,
    /// Every sampled token, the end-of-turn token included.
    pub tokens: Vec<Token>,
    pub prompt_tokens: usize,
    pub finish: LocalFinish,
    /// Draft tokens decoded for verification (the server's `timings.draft_n`).
    pub drafted: usize,
    /// Draft tokens the model kept (`timings.draft_n_accepted`).
    pub accepted: usize,
    /// Batches decoded after the prompt.
    pub passes: usize,
    /// Prompt decoding through the first sampled token.
    pub prompt_time: Duration,
    /// Everything after the first sampled token.
    pub generation_time: Duration,
}

impl fmt::Debug for LocalDecodeOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalDecodeOutput")
            .field("text_byte_count", &self.text.len())
            .field("token_count", &self.tokens.len())
            .field("prompt_tokens", &self.prompt_tokens)
            .field("finish", &self.finish)
            .field("drafted", &self.drafted)
            .field("accepted", &self.accepted)
            .field("passes", &self.passes)
            .field("prompt_time", &self.prompt_time)
            .field("generation_time", &self.generation_time)
            .finish_non_exhaustive()
    }
}

/// Decodes one reply. `between_passes` sees the reply so far after the prompt and after
/// every pass, and stops the decode by breaking.
///
/// # Errors
/// When the prompt cannot be tokenized or does not fit, the model fails, or
/// `between_passes` stops the decode.
pub fn decode<C: DecodeContext>(
    context: &mut C,
    request: &LocalDecodeRequest<'_>,
    mut between_passes: impl FnMut(&[u8]) -> ControlFlow<()>,
) -> Result<LocalDecodeOutput, LocalDecodeError> {
    let started = Instant::now();
    context.clear();
    let prompt = context.tokenize(request.prompt.as_bytes(), true)?;
    let n_ctx = context.context_size();
    if prompt.is_empty() || prompt.len() + 1 >= n_ctx {
        return Err(LocalDecodeError::PromptTooLong);
    }
    let prompt_tokens = prompt.len();
    let mut sampler = Sampler::new(
        context.vocabulary_size(),
        request.repeat_penalty,
        request.repeat_last_n,
        &context.suppressed_tokens(),
    );
    for &token in &prompt {
        sampler.accept(token);
    }

    let user_turn = if context.checkpoints_prompt() {
        let turn_start = context.tokenize(TURN_START.as_bytes(), true)?;
        Some(
            prompt
                .windows(turn_start.len().max(1))
                .position(|window| window == turn_start.as_slice()),
        )
    } else {
        None
    };
    let chunks = prompt_chunks(
        prompt.len(),
        context.batch_size(),
        context.micro_batch_size(),
        user_turn,
    );
    let last_chunk = chunks.len() - 1;
    let mut last_length = 0;
    for (index, chunk) in chunks.into_iter().enumerate() {
        if between_passes(&[]).is_break() {
            return Err(LocalDecodeError::Stopped);
        }
        let outputs = if index == last_chunk {
            Outputs::Last
        } else {
            Outputs::None
        };
        last_length = chunk.len();
        context.decode(&prompt[chunk.clone()], chunk.start, outputs)?;
    }

    let first = sampler.sample(context.logits(last_length - 1)?);
    let prompt_time = started.elapsed();
    let generating = Instant::now();
    let mut state = State {
        cached: prompt,
        sampled: first,
        reply: Reply::default(),
    };
    let mut finish = state.reply.push(context, first, &state.cached, request);
    let mut aligner = request.draft.map(WordAligner::new);
    let mut passes = 0;
    let finish = loop {
        if let Some(finish) = finish {
            break finish;
        }
        if between_passes(&state.reply.text).is_break() {
            return Err(LocalDecodeError::Stopped);
        }
        passes += 1;
        finish = state.pass(context, &mut sampler, aligner.as_mut(), request)?;
    };
    let reply = state.reply;
    Ok(LocalDecodeOutput {
        text: reply.text,
        tokens: reply.tokens,
        prompt_tokens,
        finish,
        drafted: reply.drafted,
        accepted: reply.accepted,
        passes,
        prompt_time,
        generation_time: generating.elapsed(),
    })
}

struct State {
    /// The tokens in the model's memory, prompt first.
    cached: Vec<Token>,
    /// The last sampled token, not yet decoded.
    sampled: Token,
    reply: Reply,
}

#[derive(Default)]
struct Reply {
    text: Vec<u8>,
    tokens: Vec<Token>,
    drafted: usize,
    accepted: usize,
}

impl Reply {
    /// Adds a sampled token to the reply and reports why generation stops, if it does
    /// (the server's `process_token`).
    fn push<C: DecodeContext>(
        &mut self,
        context: &C,
        token: Token,
        cached: &[Token],
        request: &LocalDecodeRequest<'_>,
    ) -> Option<LocalFinish> {
        self.text.extend(context.piece(token));
        self.tokens.push(token);
        if context.is_end_of_generation(token) {
            Some(LocalFinish::EndOfTurn)
        } else if cached.len() + 1 >= context.context_size() {
            Some(LocalFinish::ContextFull)
        } else if self.tokens.len() >= request.max_tokens {
            Some(LocalFinish::OutputLimit)
        } else {
            None
        }
    }
}

impl State {
    /// Decodes the last sampled token, with a draft when there is one, and adds the tokens
    /// the model samples.
    fn pass<C: DecodeContext>(
        &mut self,
        context: &mut C,
        sampler: &mut Sampler,
        aligner: Option<&mut WordAligner>,
        request: &LocalDecodeRequest<'_>,
    ) -> Result<Option<LocalFinish>, LocalDecodeError> {
        let draft = match aligner {
            Some(aligner) => self.draft(context, aligner, request)?,
            None => Vec::new(),
        };
        let position = self.cached.len();
        if draft.is_empty() {
            context.decode(&[self.sampled], position, Outputs::Last)?;
            self.cached.push(self.sampled);
            let next = sampler.sample(context.logits(0)?);
            self.sampled = next;
            return Ok(self.reply.push(context, next, &self.cached, request));
        }

        self.reply.drafted += draft.len();
        let mut batch = Vec::with_capacity(draft.len() + 1);
        batch.push(self.sampled);
        batch.extend_from_slice(&draft);
        context.decode(&batch, position, Outputs::All)?;
        // The server's `common_sampler_sample_and_accept_n`.
        let mut ids = Vec::with_capacity(draft.len() + 1);
        let mut matched = true;
        for (index, &drafted) in draft.iter().enumerate() {
            let next = sampler.sample(context.logits(index)?);
            ids.push(next);
            // Draft tokens after an end of generation are never kept.
            if drafted != next || (context.is_end_of_generation(next) && index + 1 < draft.len()) {
                matched = false;
                break;
            }
        }
        if matched {
            ids.push(sampler.sample(context.logits(draft.len())?));
        }
        self.reply.accepted += ids.len() - 1;
        self.cached.push(self.sampled);
        self.cached.extend_from_slice(&ids[..ids.len() - 1]);
        context.remove_from(self.cached.len())?;
        self.sampled = ids[ids.len() - 1];
        for &token in &ids {
            if let Some(finish) = self.reply.push(context, token, &self.cached, request) {
                return Ok(Some(finish));
            }
        }
        Ok(None)
    }

    /// The pass's draft: `ngram-simple`'s, replaced by the transcript's next words when the
    /// aligner has any. Empty when the budget leaves no room for one.
    fn draft<C: DecodeContext>(
        &self,
        context: &C,
        aligner: &mut WordAligner,
        request: &LocalDecodeRequest<'_>,
    ) -> Result<Vec<Token>, LocalDecodeError> {
        // Room for the sampled token and one more, and the reply's last token sampled
        // after the draft.
        let room = context.context_size().saturating_sub(self.cached.len() + 2);
        let remaining = request.max_tokens.saturating_sub(self.reply.tokens.len());
        let limit = if remaining > 0 {
            room.min(remaining - 1)
        } else {
            room
        };
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut draft = ngram_draft(&self.cached, self.sampled);
        draft.truncate(limit);
        if aligner.is_empty() {
            return Ok(draft);
        }
        let words = aligner.draft(&self.reply.text);
        if words.is_empty() {
            return Ok(draft);
        }
        let mut text_draft = self.tokenize_draft(context, &words)?;
        text_draft.truncate(limit.min(DRAFT_TOKENS));
        Ok(if text_draft.is_empty() {
            draft
        } else {
            text_draft
        })
    }

    /// Tokenizes draft words after the reply's last tokens, so they tokenize as they would
    /// in context; on its own when the reply's tokens do not survive re-tokenization.
    fn tokenize_draft<C: DecodeContext>(
        &self,
        context: &C,
        words: &[u8],
    ) -> Result<Vec<Token>, LocalDecodeError> {
        let tail_length = DRAFT_TAIL_TOKENS
            .min(self.reply.tokens.len().max(1))
            .min(self.cached.len() + 1);
        let mut tail = self.cached[self.cached.len() + 1 - tail_length..].to_vec();
        tail.push(self.sampled);
        let mut text: Vec<u8> = tail
            .iter()
            .flat_map(|&token| context.piece(token))
            .collect();
        text.extend_from_slice(words);
        let joined = context.tokenize(&text, false)?;
        if joined.len() > tail.len() && joined.starts_with(&tail) {
            Ok(joined[tail.len()..].to_vec())
        } else {
            context.tokenize(words, false)
        }
    }
}

/// llama.cpp's `common_ngram_simple_draft` with n = m = 4: the 4 tokens that followed the
/// latest earlier occurrence of the last 4 tokens (the cached tail and `sampled`).
#[must_use]
pub fn ngram_draft(cached: &[Token], sampled: Token) -> Vec<Token> {
    let length = cached.len();
    if length <= NGRAM_SIZE + DRAFT_TOKENS + 1 {
        return Vec::new();
    }
    let mut pattern = cached[length + 1 - NGRAM_SIZE..].to_vec();
    pattern.push(sampled);
    // Position 0 never counts as a match.
    let found = (1..length - NGRAM_SIZE)
        .rev()
        .find(|&start| cached[start..start + NGRAM_SIZE] == *pattern);
    let Some(start) = found else {
        return Vec::new();
    };
    let copy = DRAFT_TOKENS.min(length - (start + NGRAM_SIZE));
    if copy < NGRAM_SIZE {
        return Vec::new();
    }
    cached[start + NGRAM_SIZE..start + NGRAM_SIZE + copy].to_vec()
}

/// The server's prompt batches: at most `n_batch` tokens each, and with `user_turn` set
/// (memory that is checkpointed), broken after the token before the user turn's start and
/// 4 + `n_ubatch` and 4 tokens before the end, each capped at `n_batch`.
#[must_use]
pub fn prompt_chunks(
    length: usize,
    batch_size: usize,
    micro_batch_size: usize,
    user_turn: Option<Option<usize>>,
) -> Vec<Range<usize>> {
    let n_batch = batch_size.max(1);
    let offsets = [
        n_batch.min(CHECKPOINT_OFFSET + micro_batch_size),
        n_batch.min(CHECKPOINT_OFFSET),
    ];
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut end = 0;
    while end < length {
        end += 1;
        let full = end - start >= n_batch;
        let breaks = user_turn.is_some_and(|user_turn| {
            user_turn == Some(end) || offsets.iter().any(|&offset| length == end + offset)
        });
        if full || breaks || end == length {
            chunks.push(start..end);
            start = end;
        }
    }
    chunks
}

/// llama.cpp's repeat penalty followed by greedy choice. The window holds the last
/// `last_n` accepted tokens; a token in it has its logit divided by the penalty when
/// positive and multiplied by it otherwise.
struct Sampler {
    penalty: f32,
    last_n: usize,
    window: std::collections::VecDeque<Token>,
    counts: Vec<u32>,
    suppressed: Vec<usize>,
}

impl Sampler {
    fn new(vocabulary: usize, penalty: f32, last_n: usize, suppressed: &[Token]) -> Self {
        let mut suppressed: Vec<usize> = suppressed
            .iter()
            .filter_map(|&token| index(token))
            .filter(|&token| token < vocabulary)
            .collect();
        suppressed.sort_unstable();
        suppressed.dedup();
        Self {
            penalty,
            last_n,
            window: std::collections::VecDeque::with_capacity(last_n),
            counts: vec![0; vocabulary],
            suppressed,
        }
    }

    fn accept(&mut self, token: Token) {
        if self.last_n == 0 {
            return;
        }
        if let Some(count) = index(token).and_then(|token| self.counts.get_mut(token)) {
            *count += 1;
        }
        if self.window.len() >= self.last_n
            && let Some(oldest) = self.window.pop_front()
            && let Some(count) = index(oldest).and_then(|token| self.counts.get_mut(token))
        {
            *count = count.saturating_sub(1);
        }
        self.window.push_back(token);
    }

    fn penalized(&self, token: usize, logit: f32) -> f32 {
        if self.counts.get(token).is_none_or(|&count| count == 0) {
            logit
        } else if logit <= 0.0 {
            logit * self.penalty
        } else {
            logit / self.penalty
        }
    }

    /// Chooses the token with the highest penalized logit, the lowest id on a tie, and
    /// accepts it into the window.
    fn sample(&mut self, logits: &[f32]) -> Token {
        let mut best = (0_usize, f32::NEG_INFINITY);
        let mut suppressed = self.suppressed.iter().copied().peekable();
        for (token, &logit) in logits.iter().enumerate() {
            let value = if suppressed.peek() == Some(&token) {
                suppressed.next();
                f32::NEG_INFINITY
            } else {
                self.penalized(token, logit)
            };
            if value > best.1 {
                best = (token, value);
            }
        }
        let token = Token::try_from(best.0).unwrap_or_default();
        self.accept(token);
        token
    }
}

fn index(token: Token) -> Option<usize> {
    usize::try_from(token).ok()
}

#[cfg(test)]
mod tests;
