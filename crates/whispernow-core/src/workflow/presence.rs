#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

const NO_AUDIO_PEAK_THRESHOLD: f32 = 0.000_5;

const NO_AUDIO_RMS_THRESHOLD: f32 = 0.000_1;

/// Whether a normalized recording holds more than flat device noise. No speech detector runs.
#[must_use]
pub fn has_microphone_audio(samples: &[f32]) -> bool {
    if samples.is_empty() {
        return false;
    }
    let mut peak = 0.0_f32;
    let mut square_sum = 0.0_f64;
    for sample in samples.iter().copied() {
        peak = peak.max(sample.abs());
        square_sum += f64::from(sample) * f64::from(sample);
    }
    let rms = (square_sum / samples.len() as f64).sqrt() as f32;
    peak >= NO_AUDIO_PEAK_THRESHOLD || rms >= NO_AUDIO_RMS_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_and_flat_signals_are_no_audio() {
        assert!(!has_microphone_audio(&[]));
        assert!(!has_microphone_audio(&vec![0.0; 16_000]));
        assert!(!has_microphone_audio(&vec![0.000_05; 16_000]));
        assert!(has_microphone_audio(&[0.01; 160]));
    }

    #[test]
    fn an_impulse_above_the_peak_threshold_counts_as_audio() {
        let mut impulse = vec![0.0; 16_000];
        impulse[8_000] = 0.01;

        assert!(has_microphone_audio(&impulse));
    }
}
