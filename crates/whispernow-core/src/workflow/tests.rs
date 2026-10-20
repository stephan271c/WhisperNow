use whispernow_domain::VocabularyReplacement;

use crate::llm::CleanupFallback;

use super::*;

#[test]
fn segments_are_joined_before_deterministic_processing() {
    let vocabulary = [VocabularyReplacement::from((
        "hello world".to_owned(),
        "Hello, World".to_owned(),
    ))];
    let segments = ["So hello", "World. Next", "Thanks for watching!"].map(str::to_owned);

    let result = prepare_segmented(&segments, &vocabulary).expect("prepared");

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
        prepare_segmented(&texts, &[]).expect_err("refused")
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
fn blank_vocabulary_sources_are_skipped() {
    let rules = ["", " ", " \t"].map(|source| VocabularyReplacement::new(source, "x"));

    assert_eq!(
        apply_vocabulary_replacements("two words\tapart", &rules),
        "two words\tapart"
    );
}

#[test]
fn failed_cleanup_keeps_the_vocabulary_output_and_successful_cleanup_replaces_it() {
    let vocabulary = [VocabularyReplacement::from((
        "wrong".to_owned(),
        "right".to_owned(),
    ))];
    let mut failed = prepare_segmented(&["wrong word".to_owned()], &vocabulary).expect("prepared");
    failed.apply_cleanup(Err(CleanupFallback::EmptyOutput));
    assert_eq!(failed.raw_text, "wrong word");
    assert_eq!(failed.final_text, "right word");
    assert_eq!(failed.fallback, Some(CleanupFallback::EmptyOutput));

    let mut cleaned = prepare_segmented(&["wrong word".to_owned()], &vocabulary).expect("prepared");
    cleaned.apply_cleanup(Ok("Right word.".to_owned()));
    assert_eq!(cleaned.final_text, "Right word.");
    assert_eq!(cleaned.fallback, None);
}
