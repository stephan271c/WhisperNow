//! A recording is split at long pauses, and cut before it outgrows the recognizer, so each
//! finished segment can be decoded as an ordinary offline batch request while capture continues.
//! Every segment is a complete buffer and no decoder state crosses a segment boundary.

use std::{collections::VecDeque, ops::Range};

use whispernow_domain::ModelFamily;

use crate::audio::TARGET_SAMPLE_RATE;

/// Per-frame speech decision carrying one recording's recurrent detector state.
pub trait VoiceActivity: Send {
    /// Classifies one frame of normalized mono 16-kHz audio. Every frame is
    /// [`FRAME_SAMPLES`] long; a final partial frame is zero-padded.
    fn poll_speech(&mut self, frame: &[f32]) -> bool;
}

/// Treats every frame as speech. Used when no detector is available: segments then end only at
/// forced cuts, and no recording is rejected as silent.
#[derive(Clone, Copy, Debug, Default)]
pub struct AlwaysSpeech;

impl VoiceActivity for AlwaysSpeech {
    fn poll_speech(&mut self, _frame: &[f32]) -> bool {
        true
    }
}

const fn millis(milliseconds: usize) -> usize {
    milliseconds * TARGET_SAMPLE_RATE as usize / 1_000
}

pub const FRAME_SAMPLES: usize = 512;
/// About the 95th percentile of pauses in recorded dictations. Recognizers often end a sentence at
/// a cut, so the shorter pauses inside a thought must not end a segment.
const END_SILENCE_SAMPLES: usize = millis(2_000);
/// Audio kept before the first and after the last speech frame of a segment. It fits inside the
/// pause that ends a segment.
const PADDING_SAMPLES: usize = millis(200);
/// Speech a segment needs before a pause may end it; shorter segments merge into the next.
const MIN_SEGMENT_SPEECH_SAMPLES: usize = millis(1_000);
/// Speech below which a segment is dropped instead of decoded.
const MIN_KEPT_SPEECH_SAMPLES: usize = millis(250);
/// Longest pause a short segment waits across to merge into the next one.
const MAX_MERGE_GAP_SAMPLES: usize = millis(3_000);

/// Length at which a segment is cut even without a long pause.
const fn max_segment_samples(family: ModelFamily) -> usize {
    match family {
        // Parakeet decodes minutes in one pass; the cap bounds what is left to decode when the
        // user stops, about 2 s for 45 s of audio on the CPU.
        ModelFamily::Transducer => millis(45_000),
        // sherpa-onnx keeps only the first 30 s of Whisper input, SenseVoice's upstream pipeline
        // feeds it clips of at most 30 s, and the Qwen3-ASR token budget is sized for 28 s.
        _ => millis(28_000),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AudioSegment {
    /// First sample, in recording coordinates.
    pub start: usize,
    /// One past the last sample.
    pub end: usize,
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
    max_samples: usize,
    partial: Vec<f32>,
    position: usize,
    emitted_end: usize,
    open: Option<OpenSegment>,
}

impl SpeechSegmenter {
    #[must_use]
    pub fn new(activity: Box<dyn VoiceActivity>, family: ModelFamily) -> Self {
        Self {
            activity,
            max_samples: max_segment_samples(family),
            partial: Vec::with_capacity(FRAME_SAMPLES),
            position: 0,
            emitted_end: 0,
            open: None,
        }
    }

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
            finished.extend(self.process_frame(&frame, FRAME_SAMPLES));
            frame.clear();
            self.partial = frame;
        }
        let mut frames = samples.chunks_exact(FRAME_SAMPLES);
        for frame in &mut frames {
            finished.extend(self.process_frame(frame, FRAME_SAMPLES));
        }
        self.partial.extend_from_slice(frames.remainder());
        finished
    }

    #[must_use]
    pub fn finish(mut self) -> Vec<AudioSegment> {
        let mut finished = Vec::new();
        if !self.partial.is_empty() {
            let len = self.partial.len();
            let mut frame = std::mem::take(&mut self.partial);
            frame.resize(FRAME_SAMPLES, 0.0);
            finished.extend(self.process_frame(&frame, len));
        }
        finished.extend(self.open.take().and_then(|open| self.close(&open)));
        finished
    }

    fn process_frame(&mut self, frame: &[f32], len: usize) -> Option<AudioSegment> {
        let start = self.position;
        let end = start + len;
        self.position = end;
        let speech = self.activity.poll_speech(frame);
        if self.open.is_none() && !speech {
            return None;
        }
        let mut open = self.open.take().unwrap_or_else(|| OpenSegment {
            start: start.saturating_sub(PADDING_SAMPLES).max(self.emitted_end),
            speech_end: start,
            speech_samples: 0,
            held: false,
            recent: VecDeque::new(),
        });
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
        let window_start = end.saturating_sub(self.max_samples / 2);
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
                return self.close(&open);
            }
            open.held = true;
        }
        if open.held && pause >= MAX_MERGE_GAP_SAMPLES {
            return self.close(&open);
        }
        if end - open.start >= self.max_samples {
            return self.force_cut(open);
        }
        self.open = Some(open);
        None
    }

    fn close(&mut self, open: &OpenSegment) -> Option<AudioSegment> {
        let end = (open.speech_end + PADDING_SAMPLES).min(self.position);
        self.emit(open.start, end, open.speech_samples)
    }

    /// Cuts the open segment in the middle of the longest pause in its second half, or without a
    /// pause there at the quietest frame, preferring the latest on ties. The remainder stays open.
    fn force_cut(&mut self, mut open: OpenSegment) -> Option<AudioSegment> {
        let window = || open.recent.iter().filter(|frame| frame.start > open.start);
        let cut = longest_pause_middle(window()).unwrap_or_else(|| {
            window()
                .rev()
                .min_by(|left, right| left.energy.total_cmp(&right.energy))
                .map_or(self.position, |frame| frame.start)
        });
        let remainder_speech: usize = open
            .recent
            .iter()
            .filter(|frame| frame.speech && frame.start >= cut)
            .map(|frame| frame.len)
            .sum();
        let segment = self.emit(open.start, cut, open.speech_samples - remainder_speech);
        if cut < self.position {
            open.recent.retain(|frame| frame.start >= cut);
            open.start = cut;
            open.speech_end = open.speech_end.max(cut);
            open.speech_samples = remainder_speech;
            self.open = Some(open);
        }
        segment
    }

    fn emit(&mut self, start: usize, end: usize, speech_samples: usize) -> Option<AudioSegment> {
        self.emitted_end = end;
        (speech_samples >= MIN_KEPT_SPEECH_SAMPLES).then_some(AudioSegment {
            start,
            end,
            speech_samples,
        })
    }
}

fn longest_pause_middle<'frame>(frames: impl Iterator<Item = &'frame FrameStats>) -> Option<usize> {
    let mut longest: Option<Range<usize>> = None;
    let mut run: Option<Range<usize>> = None;
    for frame in frames {
        if frame.speech {
            run = None;
            continue;
        }
        let pause = run.get_or_insert(frame.start..frame.start);
        pause.end = frame.start + frame.len;
        if longest
            .as_ref()
            .is_none_or(|longest| pause.len() >= longest.len())
        {
            longest = Some(pause.clone());
        }
    }
    longest.map(|pause| pause.start.midpoint(pause.end))
}

fn mean_square(samples: &[f32]) -> f32 {
    #[allow(
        clippy::cast_precision_loss,
        reason = "frame lengths are far below f32's exact integer range"
    )]
    let count = samples.len() as f32;
    samples.iter().map(|sample| sample * sample).sum::<f32>() / count
}

#[cfg(test)]
fn segment_recording(
    samples: &[f32],
    activity: Box<dyn VoiceActivity>,
    family: ModelFamily,
) -> Vec<AudioSegment> {
    let mut segmenter = SpeechSegmenter::new(activity, family);
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
        fn poll_speech(&mut self, frame: &[f32]) -> bool {
            mean_square(frame) > 0.01 * 0.01
        }
    }

    const SECOND: usize = TARGET_SAMPLE_RATE as usize;
    const WHISPER_MAX_SAMPLES: usize = max_segment_samples(ModelFamily::Whisper);

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
        let count = (seconds * f64::from(TARGET_SAMPLE_RATE)) as usize;
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
        segment_recording(samples, Box::new(EnergyActivity), ModelFamily::Whisper)
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
            silence(2.5),
            speech(1.5),
            silence(2.5),
        ]);
        let segments = segment(&audio);

        assert_eq!(segments.len(), 2);
        let first_speech = SECOND..3 * SECOND;
        let second_speech = 11 * SECOND / 2..7 * SECOND;
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
    fn pauses_shorter_than_two_seconds_do_not_split_a_segment() {
        let audio = concat(&[speech(1.5), silence(1.8), speech(1.5)]);
        let segments = segment(&audio);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].range(), 0..audio.len());
    }

    #[test]
    fn short_segments_merge_into_the_next_one() {
        let audio = concat(&[speech(0.5), silence(2.5), speech(1.5), silence(2.5)]);
        let segments = segment(&audio);
        assert_eq!(segments.len(), 1, "{segments:?}");
        assert_eq!(segments[0].start, 0);
        assert!(segments[0].end > 9 * SECOND / 2);
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
    fn a_forced_cut_takes_the_middle_of_the_longest_pause_in_the_second_half() {
        // Pauses at 4 s (1.5 s, before the window), 15.5 s (0.4 s), 20.9 s (1 s) and 25.9 s
        // (0.5 s); the cut is due at 28 s.
        let audio = concat(&[
            speech(4.0),
            silence(1.5),
            speech(10.0),
            silence(0.4),
            speech(5.0),
            silence(1.0),
            speech(4.0),
            silence(0.5),
            speech(8.0),
        ]);
        let segments = segment(&audio);

        assert_eq!(segments.len(), 2, "{segments:?}");
        assert_eq!(segments[0].start, 0);
        assert!(segments[0].end.abs_diff(107 * SECOND / 5) <= FRAME_SAMPLES);
        assert_eq!(segments[0].end, segments[1].start);
        assert_eq!(segments[1].end, audio.len());
    }

    #[test]
    fn continuous_speech_is_cut_before_the_maximum_at_the_quietest_frame() {
        let mut audio = speech(40.0);
        let quiet = 20 * SECOND;
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
                .all(|segment| segment.end - segment.start <= WHISPER_MAX_SAMPLES)
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
    fn uniform_speech_is_cut_as_late_as_each_family_allows() {
        let audio = speech(100.0);
        for family in [ModelFamily::Whisper, ModelFamily::Transducer] {
            let max = max_segment_samples(family);
            let segments = segment_recording(&audio, Box::new(EnergyActivity), family);

            assert_eq!(segments.len(), audio.len().div_ceil(max), "{family:?}");
            assert_eq!(segments[0].start, 0);
            for pair in segments.windows(2) {
                assert_eq!(pair[0].end, pair[1].start, "forced cuts drop no audio");
            }
            assert_eq!(segments[segments.len() - 1].end, audio.len());
            for segment in &segments[..segments.len() - 1] {
                let len = segment.end - segment.start;
                assert!(len <= max);
                assert!(len > max - FRAME_SAMPLES * 2);
            }
        }
    }

    #[test]
    fn block_boundaries_do_not_change_segments() {
        let audio = concat(&[
            silence(0.7),
            speech(1.2),
            silence(2.4),
            speech(0.3),
            silence(0.9),
            speech(20.0),
            silence(0.6),
            speech(12.0),
            silence(0.3),
        ]);
        let expected = segment(&audio);
        let mut segmenter = SpeechSegmenter::new(Box::new(EnergyActivity), ModelFamily::Whisper);
        let mut incremental = Vec::new();
        for block in audio.chunks(333) {
            incremental.extend(segmenter.push(block));
        }
        incremental.extend(segmenter.finish());

        assert_eq!(
            expected.len(),
            3,
            "a pause end and a forced cut: {expected:?}"
        );
        assert_eq!(incremental, expected);
    }

    #[test]
    fn always_speech_segments_everything_at_forced_cuts() {
        let audio = silence(30.0);
        let segments = segment_recording(&audio, Box::new(AlwaysSpeech), ModelFamily::Whisper);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].start, 0);
        assert_eq!(segments[1].end, audio.len());
    }
}
