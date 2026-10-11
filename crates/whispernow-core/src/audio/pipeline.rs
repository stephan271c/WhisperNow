use std::num::NonZeroUsize;

use super::{
    AudioBlock, AudioLevelAccumulator, AudioLevelSummary, AudioProcessingError, Resampling,
    convert_interleaved_to_mono,
};

const OUTPUT_BLOCK_SAMPLES: usize = 320;

/// A native callback may copy/convert into pooled `f32` storage, but it must
/// hand that storage to a worker before calling this allocation-heavy type.
pub struct AudioProcessor {
    channels: NonZeroUsize,
    resampler: Resampling,
    delivery_pending: AudioBlock,
    levels: AudioLevelAccumulator,
}

impl AudioProcessor {
    pub fn new(
        input_sample_rate: u32,
        channels: NonZeroUsize,
    ) -> Result<Self, AudioProcessingError> {
        Ok(Self {
            channels,
            resampler: Resampling::new(input_sample_rate)?,
            delivery_pending: AudioBlock::default(),
            levels: AudioLevelAccumulator::default(),
        })
    }

    /// Accepts interleaved `f32` from a bounded worker queue. It may return zero
    /// blocks while the resampler accumulates frames.
    pub fn push_interleaved(
        &mut self,
        input: &[f32],
    ) -> Result<Vec<AudioBlock>, AudioProcessingError> {
        let mono = convert_interleaved_to_mono(input, self.channels)?;
        let resampled = self.resampler.push(&mono)?;
        Ok(self.process_resampled(resampled))
    }

    /// Flushes the resampler and any final partial delivery block.
    pub fn finish(mut self) -> Result<(Vec<AudioBlock>, AudioLevelSummary), AudioProcessingError> {
        let tail = self.resampler.finish()?;
        let mut blocks = self.process_resampled(tail);
        self.emit_delivery_blocks(&mut blocks, true);
        Ok((blocks, self.levels.take_summary()))
    }

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
        while self.delivery_pending.samples().len() >= OUTPUT_BLOCK_SAMPLES {
            output.push(self.delivery_pending.drain_prefix(OUTPUT_BLOCK_SAMPLES));
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
    use super::super::TARGET_SAMPLE_RATE;
    use super::*;

    #[test]
    fn delivery_framing_preserves_normalized_samples_across_blocks_and_partial_tail() {
        let mut processor =
            AudioProcessor::new(TARGET_SAMPLE_RATE, NonZeroUsize::MIN).expect("processor");
        let samples = [-1.0, -0.0, f32::MIN_POSITIVE, 0.25, 1.0, 0.0, -0.75]
            .into_iter()
            .cycle()
            .take(OUTPUT_BLOCK_SAMPLES * 2 + 1)
            .collect::<Vec<_>>();
        let mut output = Vec::new();
        for chunk in samples.chunks(2) {
            output.extend(processor.process_resampled(AudioBlock::new(chunk.to_vec())));
        }
        output.extend(processor.finish().expect("finish").0);
        assert_eq!(
            output
                .iter()
                .map(|block| block.samples().len())
                .collect::<Vec<_>>(),
            [OUTPUT_BLOCK_SAMPLES, OUTPUT_BLOCK_SAMPLES, 1]
        );
        assert_eq!(
            output
                .iter()
                .flat_map(AudioBlock::samples)
                .map(|sample| sample.to_bits())
                .collect::<Vec<_>>(),
            samples
                .iter()
                .copied()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn worker_path_downmixes_resamples_and_frames() {
        let mut processor = AudioProcessor::new(48_000, NonZeroUsize::new(2).expect("channels"))
            .expect("processor");
        let stereo = vec![0.25_f32; 96_000];
        let mut output = processor.push_interleaved(&stereo).expect("push");
        let (tail, levels) = processor.finish().expect("finish");
        output.extend(tail);
        assert_eq!(output.len(), 50);
        assert!(
            output
                .iter()
                .all(|block| block.samples().len() == OUTPUT_BLOCK_SAMPLES)
        );
        assert!(
            output
                .iter()
                .flat_map(AudioBlock::samples)
                .all(|sample| sample.is_finite() && sample.abs() <= 1.0)
        );
        assert_eq!(levels.sample_count, u64::from(TARGET_SAMPLE_RATE));
        assert!(levels.peak <= 1.0);
    }
}
