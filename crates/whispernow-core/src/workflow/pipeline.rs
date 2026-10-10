use std::time::Instant;

use crate::CancellationToken;
use crate::asr::AsrEngine;
use whispernow_domain::{UtcMillis, VocabularyReplacement};

use super::{PipelineFailureKind, PreparedTranscription, join_segment_texts};

/// Blank text is not a failure here: a segment may hold only noise, and the joined transcript
/// decides whether the recording produced text.
pub fn recognize_segment(
    samples: &[f32],
    asr: &mut dyn AsrEngine,
    cancellation: &CancellationToken,
) -> Result<String, PipelineFailureKind> {
    if cancellation.is_cancelled() {
        return Err(PipelineFailureKind::Cancelled);
    }
    let asr_started = Instant::now();
    let result = asr
        .transcribe(samples)
        .map_err(|_| PipelineFailureKind::RecognitionFailed);
    tracing::info!(
        stage = "segment_asr",
        duration_ms = asr_started.elapsed().as_secs_f64() * 1_000.0,
        sample_count = samples.len(),
        outcome = if result.is_ok() {
            "completed"
        } else {
            "failed"
        },
        "batch pipeline stage completed"
    );
    result
}

/// Joins the decoded segment texts and applies the vocabulary once to the whole transcript.
/// A recording without segments contained no speech.
pub fn prepare_segmented(
    timestamp: UtcMillis,
    segments: &[String],
    vocabulary: &[VocabularyReplacement],
) -> Result<PreparedTranscription, PipelineFailureKind> {
    if segments.is_empty() {
        return Err(PipelineFailureKind::NoSpeech);
    }
    let raw_text = join_segment_texts(segments.iter().map(String::as_str));
    if raw_text.is_empty() {
        return Err(PipelineFailureKind::EmptyTranscription);
    }
    if raw_text.contains("<unk>") {
        return Err(PipelineFailureKind::UnknownTokens);
    }
    let deterministic_started = Instant::now();
    let deterministic_text = apply_vocabulary_replacements(&raw_text, vocabulary);
    tracing::info!(
        stage = "deterministic_processing",
        duration_ms = deterministic_started.elapsed().as_secs_f64() * 1_000.0,
        outcome = "completed",
        "batch pipeline stage completed"
    );
    Ok(PreparedTranscription {
        timestamp,
        raw_text,
        final_text: deterministic_text.clone(),
        deterministic_text,
        fallback: None,
    })
}

/// Applies literal, case-sensitive vocabulary rules in order.
#[must_use]
pub fn apply_vocabulary_replacements(text: &str, replacements: &[VocabularyReplacement]) -> String {
    replacements.iter().fold(text.to_owned(), |current, rule| {
        // `str::replace` with an empty pattern inserts the replacement between every character.
        if rule.source.is_empty() {
            current
        } else {
            current.replace(&rule.source, &rule.replacement)
        }
    })
}
