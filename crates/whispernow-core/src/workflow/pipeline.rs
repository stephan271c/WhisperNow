use std::time::Instant;

use whispernow_domain::VocabularyReplacement;

use super::join_segment_texts;
use crate::{audio::AudioLevelAccumulator, llm::CleanupFallback};

const NO_AUDIO_PEAK_THRESHOLD: f32 = 0.000_5;
const NO_AUDIO_RMS_THRESHOLD: f32 = 0.000_1;

/// Has not yet caused history or output side effects.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedTranscription {
    pub raw_text: String,
    /// The raw text after vocabulary replacements, which AI cleanup edits and then replaces.
    pub final_text: String,
    /// Why AI cleanup kept the deterministic text.
    pub fallback: Option<CleanupFallback>,
}

impl PreparedTranscription {
    pub fn apply_cleanup(&mut self, cleanup: Result<String, CleanupFallback>) {
        match cleanup {
            Ok(text) => self.final_text = text,
            Err(fallback) => self.fallback = Some(fallback),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineFailureKind {
    NoAudio,
    NoSpeech,
    RecognitionFailed,
    EmptyTranscription,
    UnknownTokens,
}

/// Joins the decoded segment texts and applies the vocabulary once to the whole transcript.
/// A recording without segments contained no speech.
pub fn prepare_segmented(
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
        raw_text,
        final_text: deterministic_text,
        fallback: None,
    })
}

/// Applies literal, case-sensitive vocabulary rules in order.
#[must_use]
pub fn apply_vocabulary_replacements(text: &str, replacements: &[VocabularyReplacement]) -> String {
    replacements.iter().fold(text.to_owned(), |current, rule| {
        // `str::replace` with an empty pattern inserts the replacement between every character.
        // Editing refuses a blank source, but a hand-edited one stays in the saved settings.
        if rule.has_blank_source() {
            current
        } else {
            current.replace(&rule.source, &rule.replacement)
        }
    })
}

/// Whether a normalized recording holds more than flat device noise. No speech detector runs.
#[must_use]
pub fn has_microphone_audio(samples: &[f32]) -> bool {
    let mut levels = AudioLevelAccumulator::default();
    levels.observe(samples);
    let summary = levels.summary();
    summary.peak >= NO_AUDIO_PEAK_THRESHOLD || summary.rms >= NO_AUDIO_RMS_THRESHOLD
}
