use std::{
    sync::{
        Mutex, PoisonError,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use whispernow_api::EventPayload;
use whispernow_core::{
    audio::{
        AudioBlock, AudioEvent, AudioObservation, AudioSink, AudioSinkError, TARGET_SAMPLE_RATE,
    },
    workflow::{AudioSegment, SpeechSegmenter},
};

use super::Message;
use crate::services::Events;

/// Level events drive the overlay's bars at roughly 30 frames per second.
const LEVEL_EVENT_INTERVAL: Duration = Duration::from_millis(33);

/// Why capture ended before the user stopped the recording. The samples
/// captured up to that point are kept and transcribed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CaptureInterruption {
    MaximumDuration,
    DeviceLost,
    StreamError,
}

struct CapturedAudio {
    samples: Vec<f32>,
    segmenter: Option<SpeechSegmenter>,
    segments: Vec<AudioSegment>,
}

pub(super) struct CapturedSpeech {
    pub(super) samples: Vec<f32>,
    pub(super) segments: Vec<AudioSegment>,
}

/// Append-only buffer of one recording's 16 kHz samples. Capture faults never discard what
/// was already captured; they end capture and ask the worker to stop. Each speech segment a
/// pause ends is announced to the worker, which decodes it while capture continues.
pub(super) struct RecordingSink {
    session: u64,
    audio: Mutex<CapturedAudio>,
    max_samples: usize,
    dropped_frames: AtomicUsize,
    /// Samples accepted so far, published with each level as elapsed time.
    captured_samples: AtomicUsize,
    interruption: Mutex<Option<CaptureInterruption>>,
    worker: mpsc::Sender<Message>,
    events: Events,
    last_level_event: Mutex<Option<Instant>>,
}

impl RecordingSink {
    pub(super) fn new(
        session: u64,
        worker: mpsc::Sender<Message>,
        events: Events,
        max_samples: usize,
        segmenter: SpeechSegmenter,
    ) -> Self {
        Self {
            session,
            audio: Mutex::new(CapturedAudio {
                samples: Vec::new(),
                segmenter: Some(segmenter),
                segments: Vec::new(),
            }),
            max_samples,
            dropped_frames: AtomicUsize::new(0),
            captured_samples: AtomicUsize::new(0),
            interruption: Mutex::new(None),
            worker,
            events,
            last_level_event: Mutex::new(None),
        }
    }

    /// A copy of the samples of each segment found after the first `from`.
    pub(super) fn segments_after(&self, from: usize) -> Vec<Vec<f32>> {
        let audio = self.audio.lock().unwrap_or_else(PoisonError::into_inner);
        audio
            .segments
            .iter()
            .skip(from)
            .map(|segment| audio.samples[segment.range()].to_vec())
            .collect()
    }

    /// Takes everything captured so far, ending the last segment.
    pub(super) fn take(&self) -> CapturedSpeech {
        let dropped_frames = self.dropped_frames.load(Ordering::Relaxed);
        if dropped_frames > 0 {
            tracing::warn!(dropped_frames, "audio capture dropped buffers under load");
        }
        let mut audio = self.audio.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(segmenter) = audio.segmenter.take() {
            let last = segmenter.finish();
            audio.segments.extend(last);
        }
        CapturedSpeech {
            samples: std::mem::take(&mut audio.samples),
            segments: std::mem::take(&mut audio.segments),
        }
    }

    pub(super) fn interruption(&self) -> Option<CaptureInterruption> {
        *self
            .interruption
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn interrupt(&self, interruption: CaptureInterruption) {
        {
            let mut current = self
                .interruption
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if current.is_some() {
                return;
            }
            *current = Some(interruption);
        }
        tracing::info!(
            ?interruption,
            "audio capture ended before the recording stopped"
        );
        let _worker_stopped = self.worker.send(Message::Stop {
            session: self.session,
        });
    }
}

impl AudioSink for RecordingSink {
    fn samples(&self, block: AudioBlock) -> Result<(), AudioSinkError> {
        let mut audio = self.audio.lock().unwrap_or_else(PoisonError::into_inner);
        let remaining = self.max_samples.saturating_sub(audio.samples.len());
        let accepted = &block.samples()[..block.samples().len().min(remaining)];
        audio.samples.extend_from_slice(accepted);
        let segments = audio
            .segmenter
            .as_mut()
            .map(|segmenter| segmenter.push(accepted))
            .unwrap_or_default();
        let segment_ended = !segments.is_empty();
        audio.segments.extend(segments);
        self.captured_samples
            .store(audio.samples.len(), Ordering::Relaxed);
        let accepted = accepted.len();
        drop(audio);
        if segment_ended {
            let _worker_stopped = self.worker.send(Message::Segments {
                session: self.session,
            });
        }
        if accepted < block.samples().len() {
            self.interrupt(CaptureInterruption::MaximumDuration);
        }
        Ok(())
    }

    fn observe(&self, observation: AudioObservation) {
        let summary = match observation {
            AudioObservation::StreamReady {
                request_to_stream_ready,
            } => {
                tracing::info!(
                    stage = "capture_stream_ready",
                    duration_ms = request_to_stream_ready.as_secs_f64() * 1_000.0,
                    "workflow stage completed"
                );
                return;
            }
            AudioObservation::FirstSamples {
                request_to_first_sample,
            } => {
                tracing::info!(
                    stage = "shortcut_to_first_sample",
                    duration_ms = request_to_first_sample.as_secs_f64() * 1_000.0,
                    "workflow stage completed"
                );
                return;
            }
            AudioObservation::Level { summary } => summary,
        };
        let now = Instant::now();
        let Ok(mut last) = self.last_level_event.lock() else {
            return;
        };
        if last.is_some_and(|previous| now.duration_since(previous) < LEVEL_EVENT_INTERVAL) {
            return;
        }
        *last = Some(now);
        drop(last);
        let samples = self.captured_samples.load(Ordering::Relaxed) as u64;
        self.events.publish(EventPayload::AudioLevel {
            rms: summary.rms,
            peak: summary.peak,
            elapsed_ms: samples * 1_000 / u64::from(TARGET_SAMPLE_RATE),
        });
    }

    fn event(&self, event: AudioEvent) {
        match event {
            AudioEvent::Backpressure { dropped_frames } => {
                self.dropped_frames
                    .fetch_add(dropped_frames, Ordering::Relaxed);
            }
            AudioEvent::DeviceLost => self.interrupt(CaptureInterruption::DeviceLost),
            AudioEvent::StreamError => self.interrupt(CaptureInterruption::StreamError),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, thread};

    use whispernow_core::workflow::VoiceActivity;

    use super::*;

    struct Silence;

    impl VoiceActivity for Silence {
        fn is_speech(&mut self, _frame: &[f32]) -> bool {
            false
        }
    }

    #[test]
    fn samples_wait_for_a_busy_buffer_instead_of_dropping() {
        let (worker, _messages) = mpsc::channel();
        let sink = Arc::new(RecordingSink::new(
            1,
            worker,
            Events::channel().0,
            16_000,
            SpeechSegmenter::new(Box::new(Silence)),
        ));
        let busy = sink.audio.lock().expect("audio");
        let delivery = {
            let sink = Arc::clone(&sink);
            thread::spawn(move || sink.samples(AudioBlock::new(vec![0.5; 320])))
        };
        thread::sleep(Duration::from_millis(50));
        drop(busy);
        assert_eq!(delivery.join().expect("delivery thread"), Ok(()));
        assert_eq!(sink.take().samples.len(), 320);
    }
}
