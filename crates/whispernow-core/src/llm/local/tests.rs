use std::ops::ControlFlow;

use super::{
    DRAFT_TOKENS, DecodeContext, LocalDecodeError, LocalDecodeOutput, LocalDecodeRequest,
    LocalFinish, Outputs, PromptBatching, Reply, State, Token, decode, ngram_draft, prompt_chunks,
    render_prompt,
};
use crate::llm::build_local_cleanup_prompt;

const SPECIAL: [&str; 3] = ["<s>", "<|im_start|>", "<|im_end|>"];
const END: Token = 2;
const WORDS: [&str; 12] = [
    "user",
    "assistant",
    "\n",
    " the",
    " cat",
    " sat",
    " on",
    " mat",
    " dog",
    " a",
    "The",
    ".",
];

/// Next-token scores for a reply so far; every other token scores 0.
type Model = Box<dyn Fn(&[Token]) -> Vec<(Token, f32)>>;

/// A model over a small vocabulary: the specials, some words, and every printable ASCII
/// character, tokenized by greedy longest match.
struct Fake {
    vocabulary: Vec<String>,
    model: Model,
    memory: Vec<Token>,
    outputs: Vec<Option<Vec<f32>>>,
    batches: Vec<(usize, usize, Outputs)>,
    checkpoints: bool,
    context_size: usize,
}

impl Fake {
    fn new(model: Model) -> Self {
        let vocabulary = SPECIAL
            .iter()
            .chain(WORDS.iter())
            .map(|&piece| piece.to_owned())
            .chain((32_u8..127).map(|byte| char::from(byte).to_string()))
            .collect();
        Self {
            vocabulary,
            model,
            memory: Vec::new(),
            outputs: Vec::new(),
            batches: Vec::new(),
            checkpoints: true,
            context_size: 512,
        }
    }

    fn token(&self, piece: &str) -> Token {
        let index = self
            .vocabulary
            .iter()
            .position(|entry| entry == piece)
            .expect("piece in the vocabulary");
        Token::try_from(index).expect("small vocabulary")
    }

    fn tokens(&self, text: &str) -> Vec<Token> {
        self.tokenize(text.as_bytes(), false).expect("tokenizes")
    }

    /// The reply in `history`: everything after the assistant turn's opening.
    fn reply(&self, history: &[Token]) -> Vec<Token> {
        let opening = [self.token("assistant"), self.token("\n")];
        let start = history
            .windows(2)
            .position(|window| window == opening)
            .expect("an assistant turn")
            + 2;
        history[start..].to_vec()
    }
}

impl DecodeContext for Fake {
    fn tokenize(&self, text: &[u8], parse_special: bool) -> Result<Vec<Token>, LocalDecodeError> {
        let mut tokens = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let (index, piece) = self
                .vocabulary
                .iter()
                .enumerate()
                .filter(|(index, _)| parse_special || *index >= SPECIAL.len())
                .filter(|(_, piece)| rest.starts_with(piece.as_bytes()))
                .max_by_key(|(_, piece)| piece.len())
                .ok_or(LocalDecodeError::Tokenize)?;
            tokens.push(Token::try_from(index).expect("small vocabulary"));
            rest = &rest[piece.len()..];
        }
        Ok(tokens)
    }

    fn piece(&self, token: Token) -> Vec<u8> {
        let index = usize::try_from(token).expect("a vocabulary token");
        if index < SPECIAL.len() {
            Vec::new()
        } else {
            self.vocabulary[index].as_bytes().to_vec()
        }
    }

    fn is_end_of_generation(&self, token: Token) -> bool {
        token == END
    }

    fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }

    fn suppressed_tokens(&self) -> Vec<Token> {
        Vec::new()
    }

    fn context_size(&self) -> usize {
        self.context_size
    }

    fn batch_size(&self) -> usize {
        2_048
    }

    fn micro_batch_size(&self) -> usize {
        512
    }

    fn checkpoints_prompt(&self) -> bool {
        self.checkpoints
    }

    fn clear(&mut self) {
        self.memory.clear();
    }

    fn decode(
        &mut self,
        tokens: &[Token],
        first_position: usize,
        outputs: Outputs,
    ) -> Result<(), LocalDecodeError> {
        assert_eq!(
            first_position,
            self.memory.len(),
            "decodes continue the memory"
        );
        self.memory.extend_from_slice(tokens);
        self.batches.push((first_position, tokens.len(), outputs));
        self.outputs = (0..tokens.len())
            .map(|index| {
                let wanted = match outputs {
                    Outputs::None => false,
                    Outputs::Last => index + 1 == tokens.len(),
                    Outputs::All => true,
                };
                wanted.then(|| {
                    let history = &self.memory[..=first_position + index];
                    let mut logits = vec![0.0; self.vocabulary.len()];
                    for (token, score) in (self.model)(&self.reply(history)) {
                        logits[usize::try_from(token).expect("a vocabulary token")] = score;
                    }
                    logits
                })
            })
            .collect();
        Ok(())
    }

    fn logits(&mut self, batch_index: usize) -> Result<&[f32], LocalDecodeError> {
        self.outputs
            .get(batch_index)
            .and_then(Option::as_deref)
            .ok_or(LocalDecodeError::Logits)
    }

    fn remove_from(&mut self, from: usize) -> Result<(), LocalDecodeError> {
        self.memory.truncate(from);
        Ok(())
    }
}

/// A model that writes `target` and then ends its turn; `rivals` score next to the target
/// token at a reply position. Off the target it ends its turn.
fn follow(target: Vec<Token>, rivals: Vec<(usize, Token, f32)>) -> Model {
    Box::new(move |reply| {
        let position = reply.len();
        if reply != &target[..position.min(target.len())] || position >= target.len() {
            return vec![(END, 10.0)];
        }
        let mut scores = vec![(target[position], 10.0)];
        scores.extend(
            rivals
                .iter()
                .filter(|(at, _, _)| *at == position)
                .map(|&(_, token, score)| (token, score)),
        );
        scores
    })
}

fn fake_following(target: &str, rivals: &[(usize, &str, f32)]) -> Fake {
    let mut fake = Fake::new(Box::new(|_| Vec::new()));
    let target = fake.tokens(target);
    let rivals = rivals
        .iter()
        .map(|&(at, piece, score)| (at, fake.token(piece), score))
        .collect();
    fake.model = follow(target, rivals);
    fake
}

fn run(
    fake: &mut Fake,
    transcript: &str,
    draft: Option<&str>,
    max_tokens: usize,
) -> Result<LocalDecodeOutput, LocalDecodeError> {
    let prompt = render_prompt("<s>", &build_local_cleanup_prompt(transcript));
    let tokens = fake.tokenize(prompt.as_bytes(), true)?;
    decode(
        fake,
        LocalDecodeRequest {
            prompt: tokens,
            max_tokens,
            draft,
        },
        |_| ControlFlow::Continue(()),
    )
}

fn text(output: &LocalDecodeOutput) -> &str {
    std::str::from_utf8(&output.text).expect("UTF-8 reply")
}

#[test]
fn the_prompt_is_split_where_the_server_checkpoints_it() {
    assert_eq!(
        prompt_chunks(
            710,
            2_048,
            512,
            PromptBatching::Checkpointed { user_turn: Some(1) }
        ),
        [0..1, 1..194, 194..706, 706..710]
    );
    assert_eq!(
        prompt_chunks(
            3_000,
            2_048,
            512,
            PromptBatching::Checkpointed { user_turn: Some(1) }
        ),
        [0..1, 1..2_049, 2_049..2_484, 2_484..2_996, 2_996..3_000]
    );
    assert_eq!(
        prompt_chunks(
            3,
            2_048,
            512,
            PromptBatching::Checkpointed { user_turn: Some(1) }
        ),
        [0..1, 1..3]
    );
    assert_eq!(
        prompt_chunks(
            600,
            2_048,
            512,
            PromptBatching::Checkpointed { user_turn: None }
        ),
        [0..84, 84..596, 596..600]
    );
    // Without checkpoints only the batch size splits it.
    assert_eq!(
        prompt_chunks(5_000, 2_048, 512, PromptBatching::Plain),
        [0..2_048, 2_048..4_096, 4_096..5_000]
    );
    assert_eq!(
        prompt_chunks(10, 4, 512, PromptBatching::Checkpointed { user_turn: None }),
        [0..4, 4..6, 6..10]
    );
}

#[test]
fn ngram_drafts_copy_what_followed_the_latest_earlier_match() {
    let cached = [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4];
    assert_eq!(ngram_draft(&cached, 5), [6, 7, 8, 9]);
    // The latest earlier match wins.
    let cached = [9, 2, 3, 4, 5, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 2, 3, 4];
    assert_eq!(ngram_draft(&cached, 5), [6, 6, 6, 6]);
    assert!(ngram_draft(&cached, 7).is_empty(), "no match");
    // A match too close to the end leaves fewer than 4 tokens to copy.
    let cached = [9, 9, 9, 9, 9, 9, 1, 2, 3, 4, 1, 2, 3];
    assert!(ngram_draft(&cached, 4).is_empty());
    // Position 0 never matches, and short histories never draft.
    let cached = [2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 2, 3, 4];
    assert!(ngram_draft(&cached, 5).is_empty());
    assert!(ngram_draft(&[2, 3, 4, 2, 3, 4, 2, 3, 4], 2).is_empty());
}

#[test]
fn drafts_tokenize_after_the_reply_tail() {
    let fake = fake_following("", &[]);
    let mut state = State {
        cached: fake.tokens("x the"),
        sampled: fake.token(" cat"),
        reply: Reply {
            tokens: fake.tokens(" the cat"),
            ..Reply::default()
        },
    };
    assert_eq!(
        state.tokenize_draft(&fake, b" sat on").expect("tokenizes"),
        fake.tokens(" sat on")
    );
    // A tail that merges with the draft is dropped: the words tokenize alone.
    state.sampled = fake.token(" ");
    state.reply.tokens = vec![fake.token(" the"), fake.token(" ")];
    assert_eq!(
        state.tokenize_draft(&fake, b"cat").expect("tokenizes"),
        fake.tokens("cat")
    );
}

#[test]
fn plain_decoding_follows_the_model_to_its_end_of_turn() {
    let mut fake = fake_following(" the cat sat.", &[]);
    let output = run(&mut fake, "q", None, 64).expect("decodes");
    assert_eq!(text(&output), " the cat sat.");
    assert_eq!(output.finish, LocalFinish::EndOfTurn);
    assert_eq!(output.tokens.last(), Some(&END));
    assert_eq!((output.drafted, output.accepted), (0, 0));
    // The prompt: BOS, the user turn up to 4 + 512 tokens before the end, then the last 4.
    let prompt_batches: Vec<_> = fake.batches[..3]
        .iter()
        .map(|&(start, length, outputs)| (start, length, outputs))
        .collect();
    let prompt_len = output.prompt_tokens;
    assert_eq!(
        prompt_batches,
        [
            (0, 1, Outputs::None),
            (1, prompt_len - 5, Outputs::None),
            (prompt_len - 4, 4, Outputs::Last)
        ]
    );
}

#[test]
fn speculative_decoding_writes_what_plain_decoding_writes_in_fewer_passes() {
    let reply = " The cat sat on the mat. The dog sat on the mat. The cat sat on a mat.";
    let mut plain = fake_following(reply, &[]);
    let expected = run(&mut plain, "the cat sat on the mat", None, 64).expect("decodes");
    assert_eq!(text(&expected), reply);

    let mut drafted = fake_following(reply, &[]);
    let output = run(
        &mut drafted,
        "the cat sat on the mat",
        Some("The cat sat on the mat. The dog sat on the mat. The cat sat on a mat."),
        64,
    )
    .expect("decodes");
    assert_eq!(output.text, expected.text);
    assert_eq!(output.tokens, expected.tokens);
    assert!(output.accepted > 0);
    assert!(output.accepted <= output.drafted);
    assert!(output.passes < expected.passes);
    assert!(
        drafted.batches[3..]
            .iter()
            .all(|&(_, length, _)| length <= 1 + DRAFT_TOKENS)
    );
}

#[test]
fn an_empty_draft_text_still_drafts_from_ngrams() {
    let reply = " the cat sat on the mat the cat sat on the mat the cat sat on the mat";
    let mut plain = fake_following(reply, &[]);
    let expected = run(&mut plain, "q", None, 64).expect("decodes");
    let mut drafted = fake_following(reply, &[]);
    let output = run(&mut drafted, "q", Some(""), 64).expect("decodes");
    assert_eq!(output.tokens, expected.tokens);
    assert!(
        output.accepted > 0,
        "the repeated phrase is drafted by lookup"
    );
}

#[test]
fn the_penalty_counts_the_prompt_and_every_sampled_token() {
    // The reply repeats " the"; penalized, it scores 10 / 1.05 < 9.8.
    let mut fake = fake_following(" the the cat", &[(1, " dog", 9.8)]);
    let output = run(&mut fake, "q", None, 64).expect("decodes");
    assert_eq!(text(&output), " the dog");
    // A word from the prompt is penalized from the start.
    let mut fake = fake_following(" cat", &[(0, " dog", 9.8)]);
    let output = run(&mut fake, "a cat", None, 64).expect("decodes");
    assert_eq!(text(&output), " dog");
}

#[test]
fn drafted_tokens_count_toward_the_penalty_as_they_are_verified() {
    // One pass verifies " the the": the first " the" must be penalized before the second
    // is sampled, or the draft would be kept whole.
    let reply = " a the the cat";
    let rivals = [(2, " dog", 9.8)];
    let mut plain = fake_following(reply, &rivals);
    let expected = run(&mut plain, "q", None, 64).expect("decodes");
    assert_eq!(text(&expected), " a the dog");

    let mut drafted = fake_following(reply, &rivals);
    let output = run(&mut drafted, "q", Some("a the the cat"), 64).expect("decodes");
    assert_eq!(output.tokens, expected.tokens);
    assert_eq!((output.drafted, output.accepted), (3, 1));
}

#[test]
fn the_output_limit_cuts_the_reply_off() {
    for draft in [None, Some("the cat sat on the mat")] {
        let mut fake = fake_following(" the cat sat on the mat", &[]);
        let output = run(&mut fake, "q", draft, 3).expect("decodes");
        assert_eq!(text(&output), " the cat sat");
        assert_eq!(output.finish, LocalFinish::OutputLimit);
        assert_eq!(output.tokens.len(), 3);
    }
}

#[test]
fn an_end_of_turn_at_the_limit_still_ends_the_turn() {
    let mut fake = fake_following(" the cat", &[]);
    let output = run(&mut fake, "q", None, 3).expect("decodes");
    assert_eq!(output.finish, LocalFinish::EndOfTurn);
}

#[test]
fn a_full_context_cuts_the_reply_off_and_a_long_prompt_is_refused() {
    let mut fake = fake_following(" the cat sat on the mat", &[]);
    fake.context_size = 14;
    let output = run(&mut fake, "q", Some("the cat sat on the mat"), 64).expect("decodes");
    assert_eq!(output.finish, LocalFinish::ContextFull);
    assert_eq!(text(&output), " the cat");
    assert_eq!(fake.memory.len(), 13);

    fake.context_size = 8;
    assert_eq!(
        run(&mut fake, "q", None, 64),
        Err(LocalDecodeError::PromptTooLong)
    );
}

#[test]
fn breaking_between_passes_stops_the_decode() {
    let mut fake = fake_following(" the cat sat on the mat", &[]);
    let prompt = render_prompt("<s>", &build_local_cleanup_prompt("q"));
    let tokens = fake.tokenize(prompt.as_bytes(), true).expect("tokenizes");
    let mut seen = Vec::new();
    let result = decode(
        &mut fake,
        LocalDecodeRequest {
            prompt: tokens,
            max_tokens: 64,
            draft: None,
        },
        |reply| {
            seen.push(reply.to_vec());
            if reply.len() >= 8 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        },
    );
    assert_eq!(result, Err(LocalDecodeError::Stopped));
    assert_eq!(seen.last().map(Vec::as_slice), Some(&b" the cat"[..]));
}

#[test]
fn without_checkpoints_the_prompt_is_one_batch() {
    let mut fake = fake_following(" the", &[]);
    fake.checkpoints = false;
    let output = run(&mut fake, "q", None, 64).expect("decodes");
    assert_eq!(fake.batches[0], (0, output.prompt_tokens, Outputs::Last));
}
