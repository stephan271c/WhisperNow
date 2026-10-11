const NO_AUDIO_PEAK_THRESHOLD: f32 = 0.000_5;

const NO_AUDIO_RMS_THRESHOLD: f32 = 0.000_1;

/// Whether a normalized recording holds more than flat device noise. No speech detector runs.
#[must_use]
pub fn has_microphone_audio(samples: &[f32]) -> bool {
    let mut levels = crate::audio::AudioLevelAccumulator::default();
    levels.observe(samples);
    let summary = levels.summary();
    summary.peak >= NO_AUDIO_PEAK_THRESHOLD || summary.rms >= NO_AUDIO_RMS_THRESHOLD
}

#[cfg(test)]
mod tests {
    use crate::audio::TARGET_SAMPLE_RATE;

    use super::*;

    #[test]
    fn absent_and_flat_signals_are_no_audio() {
        assert!(!has_microphone_audio(&[]));
        assert!(!has_microphone_audio(&vec![
            0.0;
            TARGET_SAMPLE_RATE as usize
        ]));
        assert!(!has_microphone_audio(&vec![
            0.000_05;
            TARGET_SAMPLE_RATE as usize
        ]));
        assert!(has_microphone_audio(&[0.01; 160]));
    }

    #[test]
    fn an_impulse_above_the_peak_threshold_counts_as_audio() {
        let mut impulse = vec![0.0; TARGET_SAMPLE_RATE as usize];
        impulse[8_000] = 0.01;

        assert!(has_microphone_audio(&impulse));
    }
}
