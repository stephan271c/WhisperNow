use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};

use whispernow_core::{
    audio::{AudioBlock, AudioEvent, AudioLevelSummary, AudioSink, TARGET_SAMPLE_RATE},
    workflow::{AudioSegment, SpeechSegmenter},
};
use whispernow_domain::ipc::AudioLevel;

use super::Message;
use crate::services::Events;

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
    interruption: OnceLock<CaptureInterruption>,
    worker: mpsc::Sender<Message>,
    events: Events,
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
            interruption: OnceLock::new(),
            worker,
            events,
        }
    }

    /// A copy of the samples of each segment found after the first `from`.
    pub(super) fn segments_after(&self, from: usize) -> Vec<Vec<f32>> {
        let audio = self.audio.lock().expect("recording audio lock poisoned");
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
        let mut audio = self.audio.lock().expect("recording audio lock poisoned");
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
        self.interruption.get().copied()
    }

    fn interrupt(&self, interruption: CaptureInterruption) {
        if self.interruption.set(interruption).is_err() {
            return;
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
    fn samples(&self, block: AudioBlock) {
        let mut audio = self.audio.lock().expect("recording audio lock poisoned");
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
        let accepted_count = accepted.len();
        drop(audio);
        if segment_ended {
            let _worker_stopped = self.worker.send(Message::Segments {
                session: self.session,
            });
        }
        if accepted_count < block.samples().len() {
            self.interrupt(CaptureInterruption::MaximumDuration);
        }
    }

    fn observe(&self, summary: AudioLevelSummary) {
        let samples = self.captured_samples.load(Ordering::Relaxed) as u64;
        self.events.level(AudioLevel {
            rms: summary.rms,
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
    use std::{sync::Arc, thread, time::Duration};

    use whispernow_core::workflow::VoiceActivity;
    use whispernow_domain::ModelFamily;

    use super::*;

    struct Silence;

    impl VoiceActivity for Silence {
        fn poll_speech(&mut self, _frame: &[f32]) -> bool {
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
            whispernow_core::audio::TARGET_SAMPLE_RATE as usize,
            SpeechSegmenter::new(Box::new(Silence), ModelFamily::Transducer),
        ));
        let busy = sink.audio.lock().expect("audio");
        let delivery = {
            let sink = Arc::clone(&sink);
            thread::spawn(move || sink.samples(AudioBlock::new(vec![0.5; 320])))
        };
        thread::sleep(Duration::from_millis(50));
        drop(busy);
        delivery.join().expect("delivery thread");
        assert_eq!(sink.take().samples.len(), 320);
    }

    #[test]
    fn a_final_level_summary_is_published_after_the_previous_level() {
        let (worker, _messages) = mpsc::channel();
        let (events, levels) = Events::channel();
        let sink = RecordingSink::new(
            1,
            worker,
            events,
            TARGET_SAMPLE_RATE as usize,
            SpeechSegmenter::new(Box::new(Silence), ModelFamily::Transducer),
        );
        sink.samples(AudioBlock::new(vec![0.5; 320]));
        sink.observe(AudioLevelSummary {
            rms: 0.5,
            peak: 0.5,
            sample_count: 320,
        });
        sink.samples(AudioBlock::new(vec![0.25; 80]));
        sink.observe(AudioLevelSummary {
            rms: 0.25,
            peak: 0.25,
            sample_count: 80,
        });

        let levels: Vec<_> = levels
            .try_iter()
            .map(|published| match published {
                crate::services::Published::Level(level) => level.elapsed_ms,
                crate::services::Published::Event(event) => panic!("an audio level: {event:?}"),
            })
            .collect();
        assert_eq!(levels, [20, 25]);
    }
}
