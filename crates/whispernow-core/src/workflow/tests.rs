use std::collections::VecDeque;

use whispernow_domain::{ModelId, UtcMillis, VocabularyReplacement};

use crate::asr::{AsrEngine, AsrError};
use crate::{CancellationToken, ResolvedModel, llm::CleanupFallback};

use super::*;

struct FakeAsr {
    results: VecDeque<Result<String, AsrError>>,
    sample_lengths: Vec<usize>,
}

impl FakeAsr {
    fn with_texts(texts: &[&str]) -> Self {
        Self::with_results(texts.iter().map(|text| Ok((*text).to_owned())).collect())
    }

    fn with_results(results: VecDeque<Result<String, AsrError>>) -> Self {
        Self {
            results,
            sample_lengths: Vec::new(),
        }
    }
}

impl AsrEngine for FakeAsr {
    fn loaded_model_id(&self) -> Option<&ModelId> {
        None
    }

    fn load(&mut self, _model: &ResolvedModel) -> Result<(), AsrError> {
        Err(AsrError::LoadFailed)
    }

    fn transcribe(&mut self, samples: &[f32]) -> Result<String, AsrError> {
        self.sample_lengths.push(samples.len());
        self.results
            .pop_front()
            .unwrap_or(Err(AsrError::TranscriptionFailed))
    }

    fn unload(&mut self) {}
}

const TIMESTAMP: UtcMillis = UtcMillis(1_754_828_800_000);

fn prepare(
    asr: &mut dyn AsrEngine,
    vocabulary: &[VocabularyReplacement],
) -> Result<PreparedTranscription, PipelineFailureKind> {
    let text = recognize_segment(&[0.1; 1_600], asr, &CancellationToken::default())?;
    prepare_segmented(TIMESTAMP, &[text], vocabulary)
}

#[test]
fn cancelled_segments_are_not_submitted() {
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let mut asr = FakeAsr::with_texts(&["must not be used"]);

    let result = recognize_segment(&[0.1; 1_600], &mut asr, &cancellation);

    assert_eq!(result, Err(PipelineFailureKind::Cancelled));
    assert!(asr.sample_lengths.is_empty());
}

#[test]
fn segments_are_joined_before_deterministic_processing() {
    let vocabulary = [VocabularyReplacement::from((
        "hello world".to_owned(),
        "Hello, World".to_owned(),
    ))];
    let segments = ["So hello", "World. Next", "Thanks for watching!"].map(str::to_owned);

    let result = prepare_segmented(TIMESTAMP, &segments, &vocabulary).expect("prepared");

    assert_eq!(result.raw_text, "So hello world. Next");
    assert_eq!(result.final_text, "So Hello, World. Next");
}

#[test]
fn joined_unknown_tokens_blank_text_and_missing_segments_fail() {
    let joined = |texts: &[&str]| {
        let texts = texts
            .iter()
            .map(|text| (*text).to_owned())
            .collect::<Vec<_>>();
        prepare_segmented(TIMESTAMP, &texts, &[]).expect_err("refused")
    };

    assert_eq!(
        joined(&["fine", "bad <unk>"]),
        PipelineFailureKind::UnknownTokens
    );
    assert_eq!(
        joined(&[" ", "Thank you for watching."]),
        PipelineFailureKind::EmptyTranscription
    );
    assert_eq!(joined(&[]), PipelineFailureKind::NoSpeech);
}

#[test]
fn recognition_and_empty_text_fail_before_optional_processing() {
    let mut failed = FakeAsr::with_results(VecDeque::from([Err(AsrError::TranscriptionFailed)]));
    let mut empty = FakeAsr::with_texts(&[""]);

    assert_eq!(
        prepare(&mut failed, &[]),
        Err(PipelineFailureKind::RecognitionFailed)
    );
    assert_eq!(
        prepare(&mut empty, &[]),
        Err(PipelineFailureKind::EmptyTranscription)
    );
}

#[test]
fn vocabulary_rules_are_literal_sequential_and_case_sensitive() {
    let rules = [
        VocabularyReplacement::from(("cat".to_owned(), "dog".to_owned())),
        VocabularyReplacement::from(("dog".to_owned(), "fox".to_owned())),
        VocabularyReplacement::from((String::new(), "x".to_owned())),
    ];

    assert_eq!(
        apply_vocabulary_replacements("cat Cat concatenate", &rules),
        "fox Cat confoxenate"
    );
}

#[test]
fn failed_cleanup_keeps_the_vocabulary_output_and_successful_cleanup_replaces_it() {
    let vocabulary = [VocabularyReplacement::from((
        "wrong".to_owned(),
        "right".to_owned(),
    ))];
    let mut asr = FakeAsr::with_texts(&["wrong word", "wrong word"]);

    let mut failed = prepare(&mut asr, &vocabulary).expect("prepared");
    failed.apply_cleanup(Err(CleanupFallback::EmptyOutput));
    assert_eq!(failed.raw_text, "wrong word");
    assert_eq!(failed.final_text, "right word");
    assert_eq!(failed.fallback, Some(CleanupFallback::EmptyOutput));

    let mut cleaned = prepare(&mut asr, &vocabulary).expect("prepared");
    cleaned.apply_cleanup(Ok("Right word.".to_owned()));
    assert_eq!(cleaned.final_text, "Right word.");
    assert_eq!(cleaned.fallback, None);
}
