#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use rubato::{
    Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction, audioadapter_buffers::direct::InterleavedSlice,
};
use thiserror::Error;

/// The only sample rate accepted by downstream ASR consumers.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;
const RESAMPLER_CHUNK_FRAMES: usize = 1_024;

/// A mono, normalized, finite 16 kHz block.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AudioBlock {
    samples: Vec<f32>,
}

impl AudioBlock {
    /// Sanitizes non-finite values and clamps all samples to `[-1.0, 1.0]`.
    #[must_use]
    pub fn new(samples: impl Into<Vec<f32>>) -> Self {
        Self {
            samples: samples.into().into_iter().map(normalize_float).collect(),
        }
    }

    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        TARGET_SAMPLE_RATE
    }

    #[must_use]
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Combines normalized blocks without reopening the raw-sample boundary.
    pub(super) fn append(&mut self, mut block: Self) {
        self.samples.append(&mut block.samples);
    }

    /// Reframes a prefix while preserving the samples' normalization invariant.
    pub(super) fn drain_prefix(&mut self, sample_count: usize) -> Self {
        Self {
            samples: self.samples.drain(..sample_count).collect(),
        }
    }
}

/// Downmixes interleaved samples with equal channel weights (`sum / channel_count`).
///
/// # Errors
///
/// Returns a typed error for zero channels or an incomplete interleaved frame.
pub(crate) fn convert_interleaved_to_mono(
    input: &[f32],
    channels: usize,
) -> Result<Vec<f32>, AudioProcessingError> {
    if channels == 0 {
        return Err(AudioProcessingError::InvalidChannelCount);
    }
    if !input.len().is_multiple_of(channels) {
        return Err(AudioProcessingError::IncompleteInterleavedFrame);
    }

    Ok(input
        .chunks_exact(channels)
        .map(|frame| {
            let sum = frame.iter().copied().map(f64::from).sum::<f64>();
            (sum / channels as f64) as f32
        })
        .collect())
}

/// Stateful mono streaming sample-rate converter.
///
/// Arbitrary worker-side chunks are buffered until Rubato can process a full
/// input block. `finish` supplies the final partial block and pumps the filter
/// delay, returning exactly `ceil(input_frames * 16_000 / input_rate)` samples
/// across the complete stream. The instance cannot be reused after finish.
pub(crate) struct StreamingResampler {
    input_sample_rate: u32,
    resampler: Option<Box<dyn Resampler<f32>>>,
    pending: Vec<f32>,
    delay_remaining: usize,
    total_input_frames: u64,
    total_output_frames: usize,
    finished: bool,
}

impl StreamingResampler {
    /// Creates a clean stateful resampler for one recording.
    ///
    /// # Errors
    ///
    /// Returns a typed error for a zero sample rate or construction failure.
    pub fn new(input_sample_rate: u32) -> Result<Self, AudioProcessingError> {
        if input_sample_rate == 0 {
            return Err(AudioProcessingError::InvalidSampleRate);
        }
        if input_sample_rate == TARGET_SAMPLE_RATE {
            return Ok(Self {
                input_sample_rate,
                resampler: None,
                pending: Vec::new(),
                delay_remaining: 0,
                total_input_frames: 0,
                total_output_frames: 0,
                finished: false,
            });
        }

        let parameters = SincInterpolationParameters::new(128, WindowFunction::Blackman2)
            .oversampling_factor(128)
            .interpolation(SincInterpolationType::Quadratic);
        let resampler = Async::<f32>::new_sinc(
            f64::from(TARGET_SAMPLE_RATE) / f64::from(input_sample_rate),
            1.0,
            &parameters,
            RESAMPLER_CHUNK_FRAMES,
            1,
            FixedAsync::Input,
        )
        .map_err(|_| AudioProcessingError::ResamplerConstruction)?;
        let delay_remaining = resampler.output_delay();

        Ok(Self {
            input_sample_rate,
            resampler: Some(Box::new(resampler)),
            pending: Vec::with_capacity(RESAMPLER_CHUNK_FRAMES * 2),
            delay_remaining,
            total_input_frames: 0,
            total_output_frames: 0,
            finished: false,
        })
    }

    /// Adds arbitrary mono input and returns currently available 16 kHz samples. Processing is intended for a worker, not an audio callback.
    ///
    /// # Errors
    ///
    /// Returns `AlreadyFinished` after `finish`, or a resampling failure.
    pub fn push(&mut self, input: &[f32]) -> Result<AudioBlock, AudioProcessingError> {
        if self.finished {
            return Err(AudioProcessingError::AlreadyFinished);
        }
        self.total_input_frames = self
            .total_input_frames
            .saturating_add(u64::try_from(input.len()).unwrap_or(u64::MAX));

        if self.resampler.is_none() {
            self.total_output_frames = self.total_output_frames.saturating_add(input.len());
            return Ok(AudioBlock::new(input.to_vec()));
        }

        self.pending.extend_from_slice(input);
        let mut output = Vec::new();
        loop {
            let needed = self
                .resampler
                .as_ref()
                .map_or(0, |resampler| resampler.input_frames_next());
            if self.pending.len() < needed {
                break;
            }
            let chunk: Vec<f32> = self.pending.drain(..needed).collect();
            let produced = self.process_chunk(&chunk, None)?;
            self.append_after_delay(&mut output, &produced);
        }
        self.total_output_frames = self.total_output_frames.saturating_add(output.len());
        Ok(AudioBlock::new(output))
    }

    /// Flushes the final partial chunk and filter delay. Calling it twice is an
    /// error, preventing accidental duplicate tail delivery.
    ///
    /// # Errors
    ///
    /// Returns `AlreadyFinished` or a resampling failure.
    pub fn finish(&mut self) -> Result<AudioBlock, AudioProcessingError> {
        if self.finished {
            return Err(AudioProcessingError::AlreadyFinished);
        }
        self.finished = true;
        if self.resampler.is_none() {
            return Ok(AudioBlock::new(Vec::new()));
        }

        let expected = self.expected_output_frames();
        let mut output = Vec::new();
        let partial = std::mem::take(&mut self.pending);
        if !partial.is_empty() {
            let produced = self.process_chunk(&partial, Some(partial.len()))?;
            self.append_after_delay(&mut output, &produced);
        }

        while self.total_output_frames.saturating_add(output.len()) < expected {
            let produced = self.process_chunk(&[], Some(0))?;
            if produced.is_empty() {
                return Err(AudioProcessingError::Resampling);
            }
            self.append_after_delay(&mut output, &produced);
        }

        let remaining = expected.saturating_sub(self.total_output_frames);
        output.truncate(remaining);
        self.total_output_frames = self.total_output_frames.saturating_add(output.len());
        Ok(AudioBlock::new(output))
    }

    fn expected_output_frames(&self) -> usize {
        let numerator = u128::from(self.total_input_frames) * u128::from(TARGET_SAMPLE_RATE);
        let denominator = u128::from(self.input_sample_rate);
        usize::try_from(numerator.div_ceil(denominator)).unwrap_or(usize::MAX)
    }

    fn process_chunk(
        &mut self,
        input: &[f32],
        partial_len: Option<usize>,
    ) -> Result<Vec<f32>, AudioProcessingError> {
        let adapter = InterleavedSlice::new(input, 1, input.len())
            .map_err(|_| AudioProcessingError::Resampling)?;
        let indexing = partial_len.map(|length| Indexing::new().partial_len(length));
        self.resampler
            .as_mut()
            .ok_or(AudioProcessingError::Resampling)?
            .process(&adapter, indexing.as_ref())
            .map(rubato::audioadapter_buffers::owned::InterleavedOwned::take_data)
            .map_err(|_| AudioProcessingError::Resampling)
    }

    fn append_after_delay(&mut self, output: &mut Vec<f32>, produced: &[f32]) {
        let skip = self.delay_remaining.min(produced.len());
        self.delay_remaining -= skip;
        output.extend_from_slice(&produced[skip..]);
    }
}

impl std::fmt::Debug for StreamingResampler {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StreamingResampler")
            .field("input_sample_rate", &self.input_sample_rate)
            .field("pending_frames", &self.pending.len())
            .field("total_input_frames", &self.total_input_frames)
            .field("total_output_frames", &self.total_output_frames)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum AudioProcessingError {
    #[error("sample rate must be non-zero")]
    InvalidSampleRate,
    #[error("channel count must be non-zero")]
    InvalidChannelCount,
    #[error("interleaved input ends in an incomplete frame")]
    IncompleteInterleavedFrame,
    #[error("audio resampler could not be constructed")]
    ResamplerConstruction,
    #[error("audio resampling failed")]
    Resampling,
    #[error("audio stream has already been finished")]
    AlreadyFinished,
    #[error("output block size must be non-zero")]
    InvalidOutputBlockSize,
}

fn normalize_float(sample: f32) -> f32 {
    if sample.is_nan() {
        0.0
    } else {
        sample.clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn audio_block_enforces_its_invariants() {
        let block = AudioBlock::new(vec![f32::NAN, f32::NEG_INFINITY, -0.5, f32::INFINITY]);
        assert_eq!(block.sample_rate(), TARGET_SAMPLE_RATE);
        assert_eq!(block.samples(), &[0.0, -1.0, -0.5, 1.0]);
        assert!(block.samples().iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn stereo_downmix_uses_equal_weight() {
        let mono =
            convert_interleaved_to_mono(&[1.0_f32, -1.0, 0.75, 0.25], 2).expect("valid stereo");
        assert_eq!(mono, vec![0.0, 0.5]);
        assert_eq!(
            convert_interleaved_to_mono(&[1.0], 0),
            Err(AudioProcessingError::InvalidChannelCount)
        );
        assert_eq!(
            convert_interleaved_to_mono(&[1.0, 2.0, 3.0], 2),
            Err(AudioProcessingError::IncompleteInterleavedFrame)
        );
    }

    fn resample_in_chunks(rate: u32, chunk_sizes: &[usize]) -> Vec<f32> {
        let input_len = usize::try_from(rate).expect("rate fits usize");
        let input: Vec<f32> = (0..input_len)
            .map(|index| ((index as f32) * 0.01).sin() * 0.5)
            .collect();
        let mut resampler = StreamingResampler::new(rate).expect("construct resampler");
        let mut output = Vec::new();
        let mut offset = 0;
        let mut size_index = 0;
        while offset < input.len() {
            let size = chunk_sizes[size_index % chunk_sizes.len()].min(input.len() - offset);
            output.extend_from_slice(
                resampler
                    .push(&input[offset..offset + size])
                    .expect("push")
                    .samples(),
            );
            offset += size;
            size_index += 1;
        }
        output.extend_from_slice(resampler.finish().expect("finish").samples());
        output
    }

    #[test]
    fn resamples_common_rates_across_arbitrary_boundaries() {
        for rate in [44_100, 48_000] {
            let one_shape = resample_in_chunks(rate, &[97, 211, 1_337]);
            let other_shape = resample_in_chunks(rate, &[4_096, 13]);
            assert_eq!(one_shape.len(), TARGET_SAMPLE_RATE as usize);
            assert_eq!(other_shape.len(), TARGET_SAMPLE_RATE as usize);
            let max_difference = one_shape
                .iter()
                .zip(other_shape.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f32, f32::max);
            assert!(max_difference < 1.0e-5, "rate {rate}: {max_difference}");
        }
    }

    #[test]
    fn identity_rate_is_stateful_and_finish_is_one_shot() {
        let mut resampler = StreamingResampler::new(TARGET_SAMPLE_RATE).expect("identity");
        assert_eq!(
            resampler.push(&[0.25, -0.25]).expect("push").samples(),
            &[0.25, -0.25]
        );
        assert!(resampler.finish().expect("finish").is_empty());
        assert_eq!(
            resampler.finish(),
            Err(AudioProcessingError::AlreadyFinished)
        );
        assert_eq!(
            resampler.push(&[0.0]),
            Err(AudioProcessingError::AlreadyFinished)
        );
    }
}
