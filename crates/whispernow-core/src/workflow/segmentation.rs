//! Pause-based speech segmentation for per-segment batch decoding.
//!
//! A recording is split at pauses so each finished segment can be decoded as an ordinary offline
//! batch request while capture continues. This is not streaming recognition: every segment is a
//! complete buffer and no decoder state crosses a segment boundary. The per-frame speech decision
//! is an input, so this module has no native detector types; production plugs in Silero and tests
//! use an energy detector.

use std::{collections::VecDeque, ops::Range};

use super::BATCH_SAMPLE_RATE_HZ;

/// Per-frame speech decision carrying one recording's recurrent detector state.
pub trait VoiceActivity: Send {
    /// Classifies one frame of normalized mono 16-kHz audio. Every frame is
    /// [`FRAME_SAMPLES`] long; a final partial frame is zero-padded.
    fn is_speech(&mut self, frame: &[f32]) -> bool;
}

/// Creates fresh detector state for each recording or retry.
pub trait VoiceActivitySource: Send + Sync {
    fn create(&self) -> Box<dyn VoiceActivity>;
}

/// Treats every frame as speech. Used when no detector is available: segments then end only at
/// forced cuts, and no recording is rejected as silent.
#[derive(Clone, Copy, Debug, Default)]
pub struct AlwaysSpeech;

impl VoiceActivity for AlwaysSpeech {
    fn is_speech(&mut self, _frame: &[f32]) -> bool {
        true
    }
}

impl VoiceActivitySource for AlwaysSpeech {
    fn create(&self) -> Box<dyn VoiceActivity> {
        Box::new(Self)
    }
}

const fn millis(milliseconds: usize) -> usize {
    milliseconds * BATCH_SAMPLE_RATE_HZ as usize / 1_000
}

// Segmentation constants, in samples at 16 kHz. These are tuning constants, not user settings.

/// Detector frame length.
pub const FRAME_SAMPLES: usize = 512;
/// Silence after speech that ends a segment.
const END_SILENCE_SAMPLES: usize = millis(600);
/// Audio kept before the first and after the last speech frame of a segment. It fits inside the
/// pause that ends a segment.
const PADDING_SAMPLES: usize = millis(200);
/// Speech a segment needs before a pause may end it; shorter segments merge into the next.
const MIN_SEGMENT_SPEECH_SAMPLES: usize = millis(1_000);
/// Speech below which a segment is dropped instead of decoded.
const MIN_KEPT_SPEECH_SAMPLES: usize = millis(250);
/// Longest pause a short segment waits across to merge into the next one.
const MAX_MERGE_GAP_SAMPLES: usize = millis(3_000);
/// Length at which a segment is cut even without a pause, below Whisper's 30 s window.
const MAX_SEGMENT_SAMPLES: usize = millis(25_000);
/// Trailing window searched for the quietest frame when forcing a cut. Shorter than the maximum
/// segment, so a forced cut always leaves a non-empty segment.
const CUT_SEARCH_SAMPLES: usize = millis(2_000);

/// One contiguous span of a recording, decoded as one ordinary batch request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AudioSegment {
    /// First sample, in recording coordinates.
    pub start: usize,
    /// One past the last sample.
    pub end: usize,
    /// Samples the detector classified as speech.
    pub speech_samples: usize,
}

impl AudioSegment {
    #[must_use]
    pub const fn range(&self) -> Range<usize> {
        self.start..self.end
    }
}

#[derive(Clone, Copy)]
struct FrameStats {
    start: usize,
    len: usize,
    energy: f32,
    speech: bool,
}

struct OpenSegment {
    start: usize,
    speech_end: usize,
    speech_samples: usize,
    /// Paused before reaching the minimum speech; waiting to merge into the next speech.
    held: bool,
    recent: VecDeque<FrameStats>,
}

/// Incremental segmenter. Audio may arrive in blocks of any size; segment boundaries depend only
/// on the samples, so segmenting retained audio offline reproduces the live segments exactly.
pub struct SpeechSegmenter {
    activity: Box<dyn VoiceActivity>,
    partial: Vec<f32>,
    position: usize,
    emitted_end: usize,
    open: Option<OpenSegment>,
}

impl SpeechSegmenter {
    #[must_use]
    pub fn new(activity: Box<dyn VoiceActivity>) -> Self {
        Self {
            activity,
            partial: Vec::with_capacity(FRAME_SAMPLES),
            position: 0,
            emitted_end: 0,
            open: None,
        }
    }

    /// Classifies complete frames of `samples` and returns the segments they finished.
    pub fn push(&mut self, mut samples: &[f32]) -> Vec<AudioSegment> {
        let mut finished = Vec::new();
        if !self.partial.is_empty() {
            let take = (FRAME_SAMPLES - self.partial.len()).min(samples.len());
            self.partial.extend_from_slice(&samples[..take]);
            samples = &samples[take..];
            if self.partial.len() < FRAME_SAMPLES {
                return finished;
            }
            let mut frame = std::mem::take(&mut self.partial);
            self.process_frame(&frame, FRAME_SAMPLES, &mut finished);
            frame.clear();
            self.partial = frame;
        }
        let mut frames = samples.chunks_exact(FRAME_SAMPLES);
        for frame in &mut frames {
            self.process_frame(frame, FRAME_SAMPLES, &mut finished);
        }
        self.partial.extend_from_slice(frames.remainder());
        finished
    }

    /// Classifies the final partial frame and closes any open segment.
    #[must_use]
    pub fn finish(mut self) -> Vec<AudioSegment> {
        let mut finished = Vec::new();
        if !self.partial.is_empty() {
            let len = self.partial.len();
            let mut frame = std::mem::take(&mut self.partial);
            frame.resize(FRAME_SAMPLES, 0.0);
            self.process_frame(&frame, len, &mut finished);
        }
        if let Some(open) = self.open.take() {
            self.close(&open, &mut finished);
        }
        finished
    }

    fn process_frame(&mut self, frame: &[f32], len: usize, finished: &mut Vec<AudioSegment>) {
        let start = self.position;
        let end = start + len;
        self.position = end;
        let speech = self.activity.is_speech(frame);
        if self.open.is_none() {
            if !speech {
                return;
            }
            self.open = Some(OpenSegment {
                start: start.saturating_sub(PADDING_SAMPLES).max(self.emitted_end),
                speech_end: start,
                speech_samples: 0,
                held: false,
                recent: VecDeque::new(),
            });
        }
        let Some(open) = self.open.as_mut() else {
            return;
        };
        if speech {
            open.speech_end = end;
            open.speech_samples += len;
            open.held = false;
        }
        open.recent.push_back(FrameStats {
            start,
            len,
            energy: mean_square(&frame[..len]),
            speech,
        });
        let window_start = end.saturating_sub(CUT_SEARCH_SAMPLES);
        while open
            .recent
            .front()
            .is_some_and(|frame| frame.start < window_start)
        {
            open.recent.pop_front();
        }

        let pause = end - open.speech_end;
        if !speech && !open.held && pause >= END_SILENCE_SAMPLES {
            if open.speech_samples >= MIN_SEGMENT_SPEECH_SAMPLES {
                self.close_open(finished);
                return;
            }
            open.held = true;
        }
        if open.held && pause >= MAX_MERGE_GAP_SAMPLES {
            self.close_open(finished);
            return;
        }
        if end - open.start >= MAX_SEGMENT_SAMPLES {
            self.force_cut(finished);
        }
    }

    fn close_open(&mut self, finished: &mut Vec<AudioSegment>) {
        if let Some(open) = self.open.take() {
            self.close(&open, finished);
        }
    }

    fn close(&mut self, open: &OpenSegment, finished: &mut Vec<AudioSegment>) {
        let end = (open.speech_end + PADDING_SAMPLES).min(self.position);
        self.emit(open.start, end, open.speech_samples, finished);
    }

    /// Cuts the open segment at the quietest frame of its trailing window, preferring the latest
    /// such frame, and keeps the remainder open.
    fn force_cut(&mut self, finished: &mut Vec<AudioSegment>) {
        let Some(mut open) = self.open.take() else {
            return;
        };
        let cut = open
            .recent
            .iter()
            .rev()
            .filter(|frame| frame.start > open.start)
            .min_by(|left, right| left.energy.total_cmp(&right.energy))
            .map_or(self.position, |frame| frame.start);
        let remainder_speech: usize = open
            .recent
            .iter()
            .filter(|frame| frame.speech && frame.start >= cut)
            .map(|frame| frame.len)
            .sum();
        self.emit(
            open.start,
            cut,
            open.speech_samples - remainder_speech,
            finished,
        );
        if cut < self.position {
            open.recent.retain(|frame| frame.start >= cut);
            open.start = cut;
            open.speech_end = open.speech_end.max(cut);
            open.speech_samples = remainder_speech;
            self.open = Some(open);
        }
    }

    fn emit(
        &mut self,
        start: usize,
        end: usize,
        speech_samples: usize,
        finished: &mut Vec<AudioSegment>,
    ) {
        self.emitted_end = end;
        if speech_samples >= MIN_KEPT_SPEECH_SAMPLES {
            finished.push(AudioSegment {
                start,
                end,
                speech_samples,
            });
        }
    }
}

fn mean_square(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "frame lengths are far below f32's exact integer range"
    )]
    let count = samples.len() as f32;
    samples.iter().map(|sample| sample * sample).sum::<f32>() / count
}

/// Segments a complete recording with fresh detector state, as a live recording did.
#[must_use]
pub fn segment_recording(samples: &[f32], activity: Box<dyn VoiceActivity>) -> Vec<AudioSegment> {
    let mut segmenter = SpeechSegmenter::new(activity);
    let mut segments = segmenter.push(samples);
    segments.extend(segmenter.finish());
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic stand-in for Silero: a frame is speech when its RMS exceeds 0.01.
    struct EnergyActivity;

    impl VoiceActivity for EnergyActivity {
        fn is_speech(&mut self, frame: &[f32]) -> bool {
            mean_square(frame) > 0.01 * 0.01
        }
    }

    const SECOND: usize = BATCH_SAMPLE_RATE_HZ as usize;

    fn speech(seconds: f64) -> Vec<f32> {
        tone(seconds, 0.25)
    }

    fn silence(seconds: f64) -> Vec<f32> {
        tone(seconds, 0.0)
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "test durations are small positive values"
    )]
    fn tone(seconds: f64, amplitude: f32) -> Vec<f32> {
        let count = (seconds * f64::from(BATCH_SAMPLE_RATE_HZ)) as usize;
        (0..count)
            .map(|index| {
                if index % 2 == 0 {
                    amplitude
                } else {
                    -amplitude
                }
            })
            .collect()
    }

    fn concat(parts: &[Vec<f32>]) -> Vec<f32> {
        parts.concat()
    }

    fn segment(samples: &[f32]) -> Vec<AudioSegment> {
        segment_recording(samples, Box::new(EnergyActivity))
    }

    #[test]
    fn silence_and_isolated_noise_produce_zero_segments() {
        assert!(segment(&silence(5.0)).is_empty());
        assert!(segment(&[]).is_empty());

        let mut click = silence(2.0);
        click[SECOND..SECOND + 512].copy_from_slice(&speech(0.032));
        assert!(
            segment(&click).is_empty(),
            "speech shorter than the kept minimum is dropped"
        );
    }

    #[test]
    fn a_long_pause_splits_speech_into_padded_segments() {
        let audio = concat(&[
            silence(1.0),
            speech(2.0),
            silence(1.0),
            speech(1.5),
            silence(1.0),
        ]);
        let segments = segment(&audio);

        assert_eq!(segments.len(), 2);
        let first_speech = SECOND..3 * SECOND;
        let second_speech = 4 * SECOND..11 * SECOND / 2;
        for (segment, speech) in segments.iter().zip([first_speech, second_speech]) {
            assert!(segment.start <= speech.start - PADDING_SAMPLES);
            assert!(segment.start >= speech.start - PADDING_SAMPLES - FRAME_SAMPLES);
            assert!(segment.end >= speech.end + PADDING_SAMPLES - FRAME_SAMPLES);
            assert!(segment.end <= speech.end + PADDING_SAMPLES + FRAME_SAMPLES);
        }
        assert!(segments[0].end <= segments[1].start);
    }

    #[test]
    fn padding_is_clipped_to_the_recording_edges() {
        let audio = speech(1.5);
        assert_eq!(
            segment(&audio),
            [AudioSegment {
                start: 0,
                end: audio.len(),
                speech_samples: audio.len(),
            }]
        );
    }

    #[test]
    fn short_pauses_do_not_split_a_segment() {
        let audio = concat(&[speech(1.5), silence(0.4), speech(1.5)]);
        let segments = segment(&audio);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].range(), 0..audio.len());
    }

    #[test]
    fn short_segments_merge_into_the_next_one() {
        let audio = concat(&[speech(0.5), silence(1.0), speech(1.5), silence(1.0)]);
        let segments = segment(&audio);
        assert_eq!(segments.len(), 1, "{segments:?}");
        assert_eq!(segments[0].start, 0);
        assert!(segments[0].end > 3 * SECOND);
        assert!(segments[0].speech_samples >= 2 * SECOND - 1_024);
    }

    #[test]
    fn a_short_segment_is_kept_alone_after_a_long_pause_or_at_the_end() {
        let audio = concat(&[speech(0.5), silence(4.0), speech(1.5)]);
        let segments = segment(&audio);
        assert_eq!(segments.len(), 2);
        assert!(segments[0].end <= SECOND / 2 + PADDING_SAMPLES + FRAME_SAMPLES);

        let trailing = concat(&[silence(1.0), speech(0.5), silence(0.5)]);
        assert_eq!(segment(&trailing).len(), 1);
    }

    #[test]
    fn continuous_speech_is_cut_before_the_maximum_at_the_quietest_frame() {
        let mut audio = speech(40.0);
        let quiet = 24 * SECOND;
        for sample in &mut audio[quiet..quiet + 512] {
            *sample *= 0.5;
        }
        let segments = segment(&audio);

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].range(), 0..quiet);
        assert_eq!(segments[1].range(), quiet..audio.len());
        assert!(
            segments
                .iter()
                .all(|segment| segment.end - segment.start <= MAX_SEGMENT_SAMPLES)
        );
        assert_eq!(
            segments
                .iter()
                .map(|segment| segment.speech_samples)
                .sum::<usize>(),
            audio.len()
        );
    }

    #[test]
    fn uniform_speech_is_cut_as_late_as_possible() {
        let audio = speech(60.0);
        let segments = segment(&audio);

        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].start, 0);
        for pair in segments.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "forced cuts drop no audio");
        }
        assert_eq!(segments[2].end, audio.len());
        for segment in &segments[..2] {
            let len = segment.end - segment.start;
            assert!(len <= MAX_SEGMENT_SAMPLES);
            assert!(len > MAX_SEGMENT_SAMPLES - FRAME_SAMPLES * 2);
        }
    }

    #[test]
    fn block_boundaries_do_not_change_segments() {
        let audio = concat(&[
            silence(0.7),
            speech(1.2),
            silence(0.8),
            speech(0.3),
            silence(0.9),
            speech(2.1),
            silence(0.3),
        ]);
        let expected = segment(&audio);
        let mut segmenter = SpeechSegmenter::new(Box::new(EnergyActivity));
        let mut incremental = Vec::new();
        for block in audio.chunks(333) {
            incremental.extend(segmenter.push(block));
        }
        incremental.extend(segmenter.finish());

        assert!(!expected.is_empty());
        assert_eq!(incremental, expected);
    }

    #[test]
    fn always_speech_segments_everything_at_forced_cuts() {
        let audio = silence(30.0);
        let segments = segment_recording(&audio, Box::new(AlwaysSpeech));
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].start, 0);
        assert_eq!(segments[1].end, audio.len());
    }
}
