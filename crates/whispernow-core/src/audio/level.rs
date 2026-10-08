/// Holds no audio samples, so it is safe to emit to diagnostics or a recording indicator.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AudioLevelSummary {
    pub peak: f32,
    pub rms: f32,
    pub sample_count: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct AudioLevelAccumulator {
    peak: f32,
    square_sum: f64,
    sample_count: u64,
}

impl AudioLevelAccumulator {
    pub fn observe(&mut self, samples: &[f32]) {
        for sample in samples.iter().copied() {
            self.peak = self.peak.max(sample.abs());
            self.square_sum += f64::from(sample) * f64::from(sample);
            self.sample_count = self.sample_count.saturating_add(1);
        }
    }

    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn summary(&self) -> AudioLevelSummary {
        let rms = if self.sample_count == 0 {
            0.0
        } else {
            (self.square_sum / self.sample_count as f64).sqrt() as f32
        };
        AudioLevelSummary {
            peak: self.peak,
            rms,
            sample_count: self.sample_count,
        }
    }

    pub fn take_summary(&mut self) -> AudioLevelSummary {
        let result = self.summary();
        *self = Self::default();
        result
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn summary_is_constant_memory_and_resettable() {
        let mut levels = AudioLevelAccumulator::default();
        levels.observe(&[0.0, 1.0, -1.0, 0.0]);
        let summary = levels.take_summary();
        assert_eq!(summary.peak, 1.0);
        assert!((summary.rms - (0.5_f32).sqrt()).abs() < 1.0e-6);
        assert_eq!(summary.sample_count, 4);
        assert_eq!(levels.summary(), AudioLevelSummary::default());
    }
}
