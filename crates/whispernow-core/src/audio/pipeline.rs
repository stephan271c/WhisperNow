use super::{
    AudioBlock, AudioLevelAccumulator, AudioLevelSummary, AudioProcessingError, StreamingResampler,
    convert_interleaved_to_mono,
};

/// Complete worker-side normalized-f32 path: equal-weight downmix, stateful
/// resampling, bounded delivery framing, and constant-memory levels.
///
/// A native callback may copy/convert into pooled `f32` storage, but it must
/// hand that storage to a worker before calling this allocation-heavy type.
pub struct AudioProcessor {
    channels: usize,
    output_block_samples: usize,
    resampler: StreamingResampler,
    delivery_pending: AudioBlock,
    levels: AudioLevelAccumulator,
}

impl AudioProcessor {
    /// Creates one per-recording worker path.
    ///
    /// # Errors
    ///
    /// Returns a typed error for zero channels, sample rate, or output block size.
    pub fn new(
        input_sample_rate: u32,
        channels: usize,
        output_block_samples: usize,
    ) -> Result<Self, AudioProcessingError> {
        if channels == 0 {
            return Err(AudioProcessingError::InvalidChannelCount);
        }
        if output_block_samples == 0 {
            return Err(AudioProcessingError::InvalidOutputBlockSize);
        }
        Ok(Self {
            channels,
            output_block_samples,
            resampler: StreamingResampler::new(input_sample_rate)?,
            delivery_pending: AudioBlock::default(),
            levels: AudioLevelAccumulator::default(),
        })
    }

    /// Accepts interleaved `f32` from a bounded worker queue. It may return zero
    /// blocks while the resampler accumulates frames.
    ///
    /// # Errors
    ///
    /// Returns conversion, lifecycle, or resampling failures.
    pub fn push_interleaved(
        &mut self,
        input: &[f32],
    ) -> Result<Vec<AudioBlock>, AudioProcessingError> {
        let mono = convert_interleaved_to_mono(input, self.channels)?;
        let resampled = self.resampler.push(&mono)?;
        Ok(self.process_resampled(resampled))
    }

    /// Flushes the resampler and any final partial delivery block.
    ///
    /// # Errors
    ///
    /// Returns lifecycle or resampling failures.
    pub fn finish(&mut self) -> Result<Vec<AudioBlock>, AudioProcessingError> {
        let tail = self.resampler.finish()?;
        let mut blocks = self.process_resampled(tail);
        self.emit_delivery_blocks(&mut blocks, true);
        Ok(blocks)
    }

    /// Returns level aggregates since the previous call and starts a fresh
    /// observation window. No captured samples are retained by this summary.
    pub fn take_level_summary(&mut self) -> AudioLevelSummary {
        self.levels.take_summary()
    }

    fn process_resampled(&mut self, block: AudioBlock) -> Vec<AudioBlock> {
        self.levels.observe(block.samples());
        let mut blocks = Vec::new();
        self.delivery_pending.append(block);
        self.emit_delivery_blocks(&mut blocks, false);
        blocks
    }

    fn emit_delivery_blocks(&mut self, output: &mut Vec<AudioBlock>, include_partial: bool) {
        while self.delivery_pending.samples().len() >= self.output_block_samples {
            output.push(
                self.delivery_pending
                    .drain_prefix(self.output_block_samples),
            );
        }
        if include_partial && !self.delivery_pending.is_empty() {
            output.push(std::mem::take(&mut self.delivery_pending));
        }
    }
}

impl std::fmt::Debug for AudioProcessor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AudioProcessor")
            .field("channels", &self.channels)
            .field("output_block_samples", &self.output_block_samples)
            .field("resampler", &self.resampler)
            .field(
                "delivery_pending_samples",
                &self.delivery_pending.samples().len(),
            )
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delivery_framing_preserves_normalized_samples_across_blocks_and_partial_tail() {
        let mut processor = AudioProcessor::new(16_000, 1, 3).expect("processor");
        let samples = [-1.0, -0.0, f32::MIN_POSITIVE, 0.25, 1.0, 0.0, -0.75];
        let mut output = Vec::new();
        for chunk in samples.chunks(2) {
            output.extend(processor.process_resampled(AudioBlock::new(chunk.to_vec())));
        }
        output.extend(processor.finish().expect("finish"));
        assert_eq!(
            output
                .iter()
                .map(|block| block.samples().len())
                .collect::<Vec<_>>(),
            [3, 3, 1]
        );
        assert_eq!(
            output
                .iter()
                .flat_map(AudioBlock::samples)
                .map(|sample| sample.to_bits())
                .collect::<Vec<_>>(),
            samples.map(f32::to_bits)
        );
    }

    #[test]
    fn worker_path_downmixes_resamples_and_frames() {
        let mut processor = AudioProcessor::new(48_000, 2, 320).expect("processor");
        let stereo = vec![0.25_f32; 96_000];
        let mut output = processor.push_interleaved(&stereo).expect("push");
        output.extend(processor.finish().expect("finish"));
        assert_eq!(output.len(), 50);
        assert!(output.iter().all(|block| block.samples().len() == 320));
        assert!(
            output
                .iter()
                .flat_map(AudioBlock::samples)
                .all(|sample| sample.is_finite() && sample.abs() <= 1.0)
        );
        let levels = processor.take_level_summary();
        assert_eq!(levels.sample_count, 16_000);
        assert!(levels.peak <= 1.0);
        assert_eq!(processor.take_level_summary().sample_count, 0);
    }
}
