use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
        mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use whispernow_core::audio::{
    AudioBlock, AudioCapture, AudioDeviceChoice, AudioDeviceDescriptor, AudioEvent, AudioInput,
    AudioInputError, AudioObservation, AudioProcessor, AudioSink, AudioSinkError,
};

use super::{CaptureBackend, CaptureStream, CpalBackend, OpenedCapture, StreamFault};

const RAW_QUEUE_CAPACITY: usize = 8;
const WORKER_POLL_INTERVAL: Duration = Duration::from_millis(20);
const DEFAULT_DEVICE_POLL_INTERVAL: Duration = Duration::from_millis(250);
const LEVEL_OBSERVATION_INTERVAL: Duration = Duration::from_millis(50);

/// Every start creates a dedicated worker that owns downmixing, resampling,
/// level aggregation, and sink delivery. The CPAL callback only normalizes into
/// a pooled buffer and calls `try_send` on a bounded queue; a pooled buffer
/// grows only when the host delivers a larger period than it negotiated.
pub struct CpalAudioInput {
    backend: Arc<dyn CaptureBackend>,
    default_device_poll: Duration,
}

impl CpalAudioInput {
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: Arc::new(CpalBackend),
            default_device_poll: DEFAULT_DEVICE_POLL_INTERVAL,
        }
    }
}

impl Default for CpalAudioInput {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioInput for CpalAudioInput {
    fn list_input_devices(&self) -> Result<Vec<AudioDeviceDescriptor>, AudioInputError> {
        self.backend.list_input_devices()
    }

    fn start(
        &self,
        device: AudioDeviceChoice,
        sink: Arc<dyn AudioSink>,
    ) -> Result<Box<dyn AudioCapture>, AudioInputError> {
        let request_started = Instant::now();
        let (chunks_tx, chunks_rx) = sync_channel(RAW_QUEUE_CAPACITY);
        let OpenedCapture {
            stream,
            resolved_device_id,
            sample_rate_hz,
            channels,
            recycle,
            fault,
            dropped_frames,
        } = self.backend.open(&device, chunks_tx)?;
        let processor = AudioProcessor::new(sample_rate_hz, channels)
            .map_err(|_| AudioInputError::UnsupportedConfiguration)?;
        stream.start()?;
        sink.observe(AudioObservation::StreamReady {
            request_to_stream_ready: request_started.elapsed(),
        });
        let stop_requested = Arc::new(AtomicBool::new(false));
        let worker = Worker {
            backend: Arc::clone(&self.backend),
            stream,
            choice: device,
            resolved_device_id,
            sink,
            request_started,
            fault,
            dropped_frames,
            recycle,
            stop_requested: Arc::clone(&stop_requested),
            default_device_poll: self.default_device_poll,
        };
        let worker = thread::Builder::new()
            .name("whispernow-audio".to_owned())
            .spawn(move || worker.run(&chunks_rx, processor))
            .map_err(|_| backend("spawn audio worker"))?;
        Ok(Box::new(CpalCapture {
            stop_requested,
            worker: Some(worker),
        }))
    }
}

struct CpalCapture {
    stop_requested: Arc<AtomicBool>,
    worker: Option<JoinHandle<Result<(), AudioInputError>>>,
}

impl CpalCapture {
    fn finish(&mut self) -> Result<(), AudioInputError> {
        self.stop_requested.store(true, Ordering::Release);
        self.worker.take().map_or(Ok(()), |worker| {
            worker.join().map_err(|_| backend("join audio worker"))?
        })
    }
}

impl AudioCapture for CpalCapture {
    fn stop(mut self: Box<Self>) -> Result<(), AudioInputError> {
        self.finish()
    }
}

impl Drop for CpalCapture {
    fn drop(&mut self) {
        let _stopped = self.finish().is_ok();
    }
}

struct Worker {
    backend: Arc<dyn CaptureBackend>,
    stream: Box<dyn CaptureStream>,
    choice: AudioDeviceChoice,
    resolved_device_id: String,
    sink: Arc<dyn AudioSink>,
    request_started: Instant,
    fault: Arc<AtomicU8>,
    dropped_frames: Arc<AtomicUsize>,
    recycle: SyncSender<Vec<f32>>,
    stop_requested: Arc<AtomicBool>,
    default_device_poll: Duration,
}

impl Worker {
    fn run(
        self,
        chunks: &Receiver<Vec<f32>>,
        mut processor: AudioProcessor,
    ) -> Result<(), AudioInputError> {
        let mut first_samples = false;
        let mut last_default_check = Instant::now();
        let mut last_level = Instant::now();
        loop {
            if self.stop_requested.load(Ordering::Acquire) {
                let stream_result = self.stream.stop();
                let processing_result =
                    self.drain_queued_chunks(&mut processor, chunks, &mut first_samples);
                let finish_result = self.finish_processor(processor);
                if finish_result.is_err() {
                    self.sink.event(AudioEvent::StreamError);
                }
                return stream_result.and(processing_result).and(finish_result);
            }
            self.poll_default_device(&mut last_default_check);
            let fault = StreamFault::load(&self.fault);
            if fault != StreamFault::None {
                let _stream_was_stopped = self.stream.stop().is_ok();
                // Audio the device delivered before the fault is still the user's speech.
                let drained = self.drain_queued_chunks(&mut processor, chunks, &mut first_samples);
                let finished = self.finish_processor(processor);
                let event =
                    if fault == StreamFault::DeviceLost && drained.is_ok() && finished.is_ok() {
                        AudioEvent::DeviceLost
                    } else {
                        AudioEvent::StreamError
                    };
                self.sink.event(event);
                return Ok(());
            }
            self.emit_callback_backpressure();
            match chunks.recv_timeout(WORKER_POLL_INTERVAL) {
                Ok(chunk) => {
                    if self
                        .process_native_chunk(&mut processor, chunk, &mut first_samples)
                        .is_err()
                    {
                        StreamFault::StreamError.store(&self.fault);
                        continue;
                    }
                    if last_level.elapsed() >= LEVEL_OBSERVATION_INTERVAL {
                        self.sink.observe(AudioObservation::Level {
                            summary: processor.take_level_summary(),
                        });
                        last_level = Instant::now();
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    StreamFault::StreamError.store(&self.fault);
                }
            }
        }
    }

    fn drain_queued_chunks(
        &self,
        processor: &mut AudioProcessor,
        chunks: &Receiver<Vec<f32>>,
        first_samples: &mut bool,
    ) -> Result<(), AudioInputError> {
        for _ in 0..RAW_QUEUE_CAPACITY {
            let Ok(chunk) = chunks.try_recv() else {
                break;
            };
            self.process_native_chunk(processor, chunk, first_samples)?;
        }
        Ok(())
    }

    fn process_native_chunk(
        &self,
        processor: &mut AudioProcessor,
        chunk: Vec<f32>,
        first_samples: &mut bool,
    ) -> Result<(), AudioInputError> {
        if !*first_samples {
            *first_samples = true;
            self.sink.observe(AudioObservation::FirstSamples {
                request_to_first_sample: self.request_started.elapsed(),
            });
        }
        let processed = processor.push_interleaved(&chunk);
        let _recycled = self.recycle.try_send(chunk).is_ok();
        let output = processed.map_err(|_| backend("process native audio chunk"))?;
        self.emit_processor_output(output)
    }

    fn poll_default_device(&self, last_check: &mut Instant) {
        if self.choice != AudioDeviceChoice::SystemDefault
            || last_check.elapsed() < self.default_device_poll
        {
            return;
        }
        *last_check = Instant::now();
        match self.backend.current_default_id() {
            // A changed default ends the session exactly like a lost device.
            Ok(Some(current)) if current == self.resolved_device_id => {}
            Ok(_) => StreamFault::DeviceLost.store(&self.fault),
            Err(_) => StreamFault::StreamError.store(&self.fault),
        }
    }

    fn emit_callback_backpressure(&self) {
        let dropped = self.dropped_frames.swap(0, Ordering::AcqRel);
        if dropped > 0 {
            self.sink.event(AudioEvent::Backpressure {
                dropped_frames: dropped,
            });
        }
    }

    fn emit_processor_output(&self, blocks: Vec<AudioBlock>) -> Result<(), AudioInputError> {
        for block in blocks {
            if self.sink.samples(block) == Err(AudioSinkError::Closed) {
                return Err(backend("deliver audio to closed sink"));
            }
        }
        Ok(())
    }

    fn finish_processor(&self, processor: AudioProcessor) -> Result<(), AudioInputError> {
        let (output, summary) = processor
            .finish()
            .map_err(|_| backend("flush audio processor"))?;
        self.emit_processor_output(output)?;
        if summary.sample_count > 0 {
            self.sink.observe(AudioObservation::Level { summary });
        }
        Ok(())
    }
}

fn backend(operation: &'static str) -> AudioInputError {
    AudioInputError::Backend { operation }
}

#[cfg(test)]
mod tests {
    use std::{collections::VecDeque, sync::Mutex};

    use super::*;

    struct FakeStream {
        _chunks: SyncSender<Vec<f32>>,
        _recycle: Receiver<Vec<f32>>,
        stops: Arc<AtomicUsize>,
    }

    impl CaptureStream for FakeStream {
        fn start(&self) -> Result<(), AudioInputError> {
            Ok(())
        }

        fn stop(&self) -> Result<(), AudioInputError> {
            self.stops.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct FakeBackend {
        default_id: Mutex<Option<String>>,
        chunks: Mutex<VecDeque<Vec<f32>>>,
        stops: Arc<AtomicUsize>,
        fault_on_open: Mutex<Option<StreamFault>>,
    }

    impl FakeBackend {
        fn new(chunks: impl IntoIterator<Item = Vec<f32>>) -> Self {
            Self {
                default_id: Mutex::new(Some("device-1".to_owned())),
                chunks: Mutex::new(chunks.into_iter().collect()),
                stops: Arc::new(AtomicUsize::new(0)),
                fault_on_open: Mutex::new(None),
            }
        }
    }

    impl CaptureBackend for FakeBackend {
        fn list_input_devices(&self) -> Result<Vec<AudioDeviceDescriptor>, AudioInputError> {
            Ok(vec![AudioDeviceDescriptor {
                choice: AudioDeviceChoice::SystemDefault,
                display_name: "System default".to_owned(),
                is_current_default: true,
            }])
        }

        fn open(
            &self,
            choice: &AudioDeviceChoice,
            chunks: SyncSender<Vec<f32>>,
        ) -> Result<OpenedCapture, AudioInputError> {
            if matches!(choice, AudioDeviceChoice::Device(id) if id.as_str() == "missing") {
                return Err(AudioInputError::InvalidDevice);
            }
            if let Some(samples) = self.chunks.lock().expect("chunks lock").pop_front() {
                chunks.send(samples).expect("fixture chunk");
            }
            let (recycle, recycle_rx) = sync_channel(8);
            let fault = Arc::new(AtomicU8::new(StreamFault::None as u8));
            if let Some(initial) = *self.fault_on_open.lock().expect("fault lock") {
                initial.store(&fault);
            }
            Ok(OpenedCapture {
                stream: Box::new(FakeStream {
                    _chunks: chunks,
                    _recycle: recycle_rx,
                    stops: Arc::clone(&self.stops),
                }),
                resolved_device_id: self
                    .default_id
                    .lock()
                    .expect("default lock")
                    .clone()
                    .unwrap_or_else(|| "device-1".to_owned()),
                sample_rate_hz: 48_000,
                channels: std::num::NonZeroUsize::new(2).expect("channels"),
                recycle,
                fault,
                dropped_frames: Arc::new(AtomicUsize::new(0)),
            })
        }

        fn current_default_id(&self) -> Result<Option<String>, AudioInputError> {
            Ok(self.default_id.lock().expect("default lock").clone())
        }
    }

    #[derive(Default)]
    struct RecordingSink {
        blocks: Mutex<Vec<AudioBlock>>,
        observations: Mutex<Vec<AudioObservation>>,
        events: Mutex<Vec<AudioEvent>>,
        delivery_order: Mutex<Vec<&'static str>>,
        faulted: Mutex<Option<SyncSender<AudioEvent>>>,
    }

    impl RecordingSink {
        fn signalling_fault() -> (Arc<Self>, Receiver<AudioEvent>) {
            let (faulted, faulted_rx) = sync_channel(1);
            let sink = Self {
                faulted: Mutex::new(Some(faulted)),
                ..Self::default()
            };
            (Arc::new(sink), faulted_rx)
        }

        fn delivered_samples(&self) -> usize {
            self.blocks
                .lock()
                .expect("blocks lock")
                .iter()
                .map(|block| block.samples().len())
                .sum()
        }
    }

    impl AudioSink for RecordingSink {
        fn samples(&self, block: AudioBlock) -> Result<(), AudioSinkError> {
            self.delivery_order
                .lock()
                .expect("delivery order lock")
                .push("samples");
            self.blocks.lock().expect("blocks lock").push(block);
            Ok(())
        }

        fn observe(&self, observation: AudioObservation) {
            let label = match observation {
                AudioObservation::StreamReady { .. } => "ready",
                AudioObservation::FirstSamples { .. } => "first-samples",
                AudioObservation::Level { .. } => "level",
            };
            self.delivery_order
                .lock()
                .expect("delivery order lock")
                .push(label);
            self.observations
                .lock()
                .expect("observations lock")
                .push(observation);
        }

        fn event(&self, event: AudioEvent) {
            if matches!(event, AudioEvent::DeviceLost | AudioEvent::StreamError)
                && let Some(faulted) = self.faulted.lock().expect("faulted lock").take()
            {
                faulted.send(event.clone()).expect("report capture fault");
            }
            self.events.lock().expect("events lock").push(event);
        }
    }

    #[derive(Default)]
    struct ClosedSink {
        events: Mutex<Vec<AudioEvent>>,
    }

    impl AudioSink for ClosedSink {
        fn samples(&self, _block: AudioBlock) -> Result<(), AudioSinkError> {
            Err(AudioSinkError::Closed)
        }

        fn observe(&self, _observation: AudioObservation) {}

        fn event(&self, event: AudioEvent) {
            self.events.lock().expect("events lock").push(event);
        }
    }

    fn service(backend: Arc<FakeBackend>) -> CpalAudioInput {
        CpalAudioInput {
            backend,
            default_device_poll: Duration::from_millis(10),
        }
    }

    fn start(
        service: &CpalAudioInput,
        sink: Arc<dyn AudioSink>,
    ) -> Result<Box<dyn AudioCapture>, AudioInputError> {
        service.start(AudioDeviceChoice::SystemDefault, sink)
    }

    /// Waits for the capture worker to reach `condition`, bounded so a broken
    /// worker fails the test instead of hanging it.
    fn wait_until(condition: impl Fn() -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !condition() {
            if Instant::now() >= deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(2));
        }
        true
    }

    #[test]
    fn a_stopped_capture_can_be_followed_by_another() {
        let backend = Arc::new(FakeBackend::new([]));
        let service = service(Arc::clone(&backend));
        let first = start(&service, Arc::new(RecordingSink::default())).expect("start");
        first.stop().expect("stop");
        let second = start(&service, Arc::new(RecordingSink::default())).expect("restart");
        second.stop().expect("second stop");
        assert_eq!(backend.stops.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn dropping_a_capture_stops_the_native_stream() {
        let backend = Arc::new(FakeBackend::new([]));
        let service = service(Arc::clone(&backend));
        let capture = start(&service, Arc::new(RecordingSink::default())).expect("start");

        drop(capture);

        assert_eq!(backend.stops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn worker_emits_normalized_audio_levels_and_monotonic_timings() {
        let backend = Arc::new(FakeBackend::new([vec![0.5; 96_000]]));
        let service = service(backend);
        let sink = Arc::new(RecordingSink::default());
        let capture = start(&service, Arc::clone(&sink) as Arc<dyn AudioSink>).expect("start");
        assert!(wait_until(|| {
            !sink.blocks.lock().expect("blocks lock").is_empty()
        }));
        capture.stop().expect("stop");
        let blocks = sink.blocks.lock().expect("blocks lock");
        assert!(blocks.iter().all(|block| {
            block
                .samples()
                .iter()
                .all(|sample| sample.is_finite() && sample.abs() <= 1.0)
        }));
        let observations = sink.observations.lock().expect("observations lock");
        assert!(
            observations
                .iter()
                .any(|observation| matches!(observation, AudioObservation::StreamReady { .. }))
        );
        assert!(
            observations
                .iter()
                .any(|observation| matches!(observation, AudioObservation::FirstSamples { .. }))
        );
        assert!(
            observations
                .iter()
                .any(|observation| matches!(observation, AudioObservation::Level { .. }))
        );
        let delivery_order = sink.delivery_order.lock().expect("delivery order lock");
        let ready = delivery_order
            .iter()
            .position(|item| *item == "ready")
            .expect("stream-ready delivery");
        let first_samples = delivery_order
            .iter()
            .position(|item| *item == "first-samples")
            .expect("first-samples delivery");
        let samples = delivery_order
            .iter()
            .position(|item| *item == "samples")
            .expect("audio delivery");
        assert!(ready < first_samples);
        assert!(ready < samples);
    }

    #[test]
    fn a_changed_default_device_ends_the_session_as_device_loss() {
        let backend = Arc::new(FakeBackend::new([]));
        let service = service(Arc::clone(&backend));
        let (sink, faulted) = RecordingSink::signalling_fault();
        let capture = start(&service, Arc::clone(&sink) as Arc<dyn AudioSink>).expect("start");
        *backend.default_id.lock().expect("default lock") = Some("device-2".to_owned());
        assert_eq!(
            faulted
                .recv_timeout(Duration::from_secs(1))
                .expect("capture stops after the default device changes"),
            AudioEvent::DeviceLost
        );
        capture.stop().expect("stop after change");
    }

    #[test]
    fn device_fault_delivers_audio_queued_before_the_fault() {
        let requested_backend = Arc::new(FakeBackend::new([vec![0.5; 96_000]]));
        let requested_service = service(requested_backend);
        let requested_sink = Arc::new(RecordingSink::default());
        start(
            &requested_service,
            Arc::clone(&requested_sink) as Arc<dyn AudioSink>,
        )
        .expect("start reference capture")
        .stop()
        .expect("stop reference capture");
        let expected = requested_sink.delivered_samples();
        assert!(expected > 0);

        let backend = Arc::new(FakeBackend::new([vec![0.5; 96_000]]));
        *backend.fault_on_open.lock().expect("fault lock") = Some(StreamFault::DeviceLost);
        let service = service(backend);
        let (sink, faulted) = RecordingSink::signalling_fault();
        let capture = start(&service, Arc::clone(&sink) as Arc<dyn AudioSink>)
            .expect("start faulting capture");

        assert_eq!(
            faulted.recv_timeout(Duration::from_secs(5)),
            Ok(AudioEvent::DeviceLost)
        );
        assert_eq!(sink.delivered_samples(), expected);
        capture.stop().expect("release faulted capture");
    }

    #[test]
    fn invalid_device_does_not_strand_the_service() {
        let backend = Arc::new(FakeBackend::new([]));
        let service = service(backend);
        let missing = whispernow_core::audio::AudioDeviceId::new("missing").expect("ID");
        assert_eq!(
            service
                .start(
                    AudioDeviceChoice::Device(missing),
                    Arc::new(RecordingSink::default()),
                )
                .err(),
            Some(AudioInputError::InvalidDevice)
        );
        start(&service, Arc::new(RecordingSink::default()))
            .expect("restart after invalid device")
            .stop()
            .expect("stop");
    }

    #[test]
    fn closed_sink_stops_native_capture_instead_of_reporting_backpressure() {
        let backend = Arc::new(FakeBackend::new([vec![0.5; 96_000]]));
        let service = service(Arc::clone(&backend));
        let sink = Arc::new(ClosedSink::default());
        let capture = start(&service, Arc::clone(&sink) as Arc<dyn AudioSink>).expect("start");
        assert!(wait_until(|| backend.stops.load(Ordering::Relaxed) > 0));
        capture.stop().expect("release faulted capture");
        let events = sink.events.lock().expect("events lock");
        assert!(events.contains(&AudioEvent::StreamError));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AudioEvent::Backpressure { .. }))
        );
    }
}
