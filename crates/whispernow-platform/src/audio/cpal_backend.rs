use std::{
    num::NonZeroUsize,
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicU8, AtomicUsize, Ordering},
        mpsc::{Receiver, SyncSender, TryRecvError, TrySendError, sync_channel},
    },
    time::Duration,
};

use cpal::{
    BufferSize, Device, DeviceId, ErrorKind, FromSample, Sample, SampleFormat, Stream,
    StreamConfig, SupportedBufferSize, SupportedStreamConfig, SupportedStreamConfigRange,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use whispernow_core::audio::{AudioDeviceChoice, AudioInputError, TARGET_SAMPLE_RATE};

use whispernow_domain::AudioDeviceId;
use whispernow_domain::ipc::AudioDevice;

const CALLBACK_BUFFER_POOL: usize = 8;
const UNKNOWN_BUFFER_SAMPLES: usize = 65_536;
const PREFERRED_BUFFER_FRAMES: u32 = 1_024;
const STREAM_BUILD_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum StreamFault {
    None = 0,
    DeviceLost = 1,
    StreamError = 2,
}

impl StreamFault {
    pub(super) fn load(source: &AtomicU8) -> Self {
        match source.load(Ordering::Acquire) {
            1 => Self::DeviceLost,
            2 => Self::StreamError,
            _ => Self::None,
        }
    }

    pub(super) fn store(self, destination: &AtomicU8) {
        destination.store(self as u8, Ordering::Release);
    }
}

pub(crate) trait CaptureStream: Send {
    fn start(&self) -> Result<(), AudioInputError>;
    fn stop(&self) -> Result<(), AudioInputError>;
}

impl CaptureStream for Stream {
    fn start(&self) -> Result<(), AudioInputError> {
        self.play()
            .map_err(|error| map_device_error(&error, "start input stream"))
    }

    fn stop(&self) -> Result<(), AudioInputError> {
        self.pause()
            .map_err(|error| map_device_error(&error, "stop input stream"))
    }
}

pub(crate) struct OpenedCapture {
    pub stream: Box<dyn CaptureStream>,
    pub resolved_device_id: String,
    pub sample_rate_hz: u32,
    pub channels: NonZeroUsize,
    pub recycle: SyncSender<Vec<f32>>,
    pub fault: Arc<AtomicU8>,
    pub dropped_frames: Arc<AtomicUsize>,
}

pub(crate) trait CaptureBackend: Send + Sync + 'static {
    fn list_input_devices(&self) -> Result<Vec<AudioDevice>, AudioInputError>;

    fn open(
        &self,
        choice: &AudioDeviceChoice,
        chunks: SyncSender<Vec<f32>>,
    ) -> Result<OpenedCapture, AudioInputError>;

    fn current_default_id(&self) -> Result<Option<String>, AudioInputError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CpalBackend;

impl CaptureBackend for CpalBackend {
    fn list_input_devices(&self) -> Result<Vec<AudioDevice>, AudioInputError> {
        let host = cpal::default_host();
        let default_id = host
            .default_input_device()
            .and_then(|device| device.id().ok())
            .map(|id| id.to_string());
        let mut devices = Vec::new();
        let inputs = host
            .input_devices()
            .map_err(|error| map_device_error(&error, "enumerate input devices"))?;
        for device in inputs {
            let Some(id) = device
                .id()
                .ok()
                .and_then(|id| AudioDeviceId::new(id.to_string()))
            else {
                continue;
            };
            let display_name = device.description().map_or_else(
                |_| device.to_string(),
                |description| description.to_string(),
            );
            let is_default = default_id.as_deref() == Some(id.as_str());
            devices.push(AudioDevice {
                id,
                display_name,
                is_default,
            });
        }
        Ok(devices)
    }

    fn open(
        &self,
        choice: &AudioDeviceChoice,
        chunks: SyncSender<Vec<f32>>,
    ) -> Result<OpenedCapture, AudioInputError> {
        let host = cpal::default_host();
        let device = resolve_device(&host, choice)?;
        let resolved_device_id = device
            .id()
            .map_err(|error| map_device_error(&error, "resolve input device ID"))?
            .to_string();
        let supported = device
            .supported_input_configs()
            .map_err(|error| map_device_error(&error, "query input configurations"))?
            .collect::<Vec<_>>();
        let selected =
            negotiate_input_config(&supported).ok_or(AudioInputError::UnsupportedConfiguration)?;
        let channels = NonZeroUsize::new(usize::from(selected.channels()))
            .ok_or(AudioInputError::UnsupportedConfiguration)?;
        let sample_rate_hz = selected.sample_rate();
        let mut config = selected.config();
        let buffer_frames = configure_buffer_size(&mut config, selected.buffer_size());
        let callback_capacity = buffer_frames.map_or(UNKNOWN_BUFFER_SAMPLES, |frames| {
            frames as usize * channels.get()
        });
        let (recycle, available) = sync_channel(CALLBACK_BUFFER_POOL);
        for _ in 0..CALLBACK_BUFFER_POOL {
            recycle
                .send(Vec::with_capacity(callback_capacity))
                .map_err(|_| backend("prepare callback buffer pool"))?;
        }
        let fault = Arc::new(AtomicU8::new(StreamFault::None as u8));
        let dropped_frames = Arc::new(AtomicUsize::new(0));
        let callback = CallbackPorts {
            channels,
            chunks,
            available,
            recycle: recycle.clone(),
            dropped_frames: Arc::clone(&dropped_frames),
        };
        let stream = build_stream(
            &device,
            selected.sample_format(),
            config,
            callback,
            Arc::clone(&fault),
        )?;
        Ok(OpenedCapture {
            stream: Box::new(stream),
            resolved_device_id,
            sample_rate_hz,
            channels,
            recycle,
            fault,
            dropped_frames,
        })
    }

    fn current_default_id(&self) -> Result<Option<String>, AudioInputError> {
        cpal::default_host()
            .default_input_device()
            .map(|device| {
                device
                    .id()
                    .map(|id| id.to_string())
                    .map_err(|error| map_device_error(&error, "resolve default input device"))
            })
            .transpose()
    }
}

fn resolve_device(
    host: &cpal::Host,
    choice: &AudioDeviceChoice,
) -> Result<Device, AudioInputError> {
    match choice {
        AudioDeviceChoice::SystemDefault => host
            .default_input_device()
            .ok_or(AudioInputError::DeviceUnavailable),
        AudioDeviceChoice::Device(id) => {
            let native_id =
                DeviceId::from_str(id.as_str()).map_err(|_| AudioInputError::InvalidDevice)?;
            host.device_by_id(&native_id)
                .ok_or(AudioInputError::InvalidDevice)
        }
    }
}

fn negotiate_input_config(
    supported: &[SupportedStreamConfigRange],
) -> Option<SupportedStreamConfig> {
    supported
        .iter()
        .filter_map(|range| {
            let format_rank = sample_format_rank(range.sample_format())?;
            let sample_rate = preferred_sample_rate(range);
            let score = (
                format_rank,
                channel_rank(range.channels()),
                sample_rate.abs_diff(48_000),
            );
            Some((score, (*range).with_sample_rate(sample_rate)))
        })
        .min_by_key(|(score, _)| *score)
        .map(|(_, config)| config)
}

fn preferred_sample_rate(range: &SupportedStreamConfigRange) -> u32 {
    for preferred in [48_000, 44_100, TARGET_SAMPLE_RATE] {
        if range.min_sample_rate() <= preferred && preferred <= range.max_sample_rate() {
            return preferred;
        }
    }
    48_000_u32.clamp(range.min_sample_rate(), range.max_sample_rate())
}

fn sample_format_rank(format: SampleFormat) -> Option<u8> {
    match format {
        SampleFormat::F32 => Some(0),
        SampleFormat::I16 => Some(1),
        SampleFormat::I24 => Some(2),
        SampleFormat::I32 => Some(3),
        SampleFormat::F64 => Some(4),
        SampleFormat::I8 => Some(5),
        SampleFormat::I64 => Some(6),
        SampleFormat::U8 => Some(7),
        SampleFormat::U16 => Some(8),
        SampleFormat::U24 => Some(9),
        SampleFormat::U32 => Some(10),
        SampleFormat::U64 => Some(11),
        _ => None,
    }
}

fn channel_rank(channels: u16) -> (u8, u16) {
    match channels {
        1 => (0, 1),
        2 => (1, 2),
        value => (2, value),
    }
}

fn configure_buffer_size(
    config: &mut StreamConfig,
    supported: &SupportedBufferSize,
) -> Option<u32> {
    match *supported {
        SupportedBufferSize::Range { min, max } => {
            let frames = PREFERRED_BUFFER_FRAMES.clamp(min, max);
            config.buffer_size = BufferSize::Fixed(frames);
            Some(frames)
        }
        SupportedBufferSize::Unknown => None,
    }
}

struct CallbackPorts {
    channels: NonZeroUsize,
    chunks: SyncSender<Vec<f32>>,
    available: Receiver<Vec<f32>>,
    recycle: SyncSender<Vec<f32>>,
    dropped_frames: Arc<AtomicUsize>,
}

impl CallbackPorts {
    /// Converts one callback's samples into a pooled buffer and sends it, never blocking or
    /// allocating more than the pool holds; frames that find no buffer or no room are counted
    /// as dropped.
    fn deliver<T>(&self, data: &[T])
    where
        T: Sample + Copy,
        f32: FromSample<T>,
    {
        let mut buffer = match self.available.try_recv() {
            Ok(buffer) => buffer,
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => {
                add_dropped(&self.dropped_frames, data.len() / self.channels.get());
                return;
            }
        };
        buffer.clear();
        // Hosts such as WASAPI shared mode or PipeWire may ignore the requested period;
        // growing keeps the whole callback and the grown buffer returns to the pool.
        buffer.reserve(data.len());
        buffer.extend(data.iter().map(|sample| {
            let normalized = f32::from_sample(*sample);
            if normalized.is_finite() {
                normalized.clamp(-1.0, 1.0)
            } else {
                0.0
            }
        }));
        if let Err(TrySendError::Full(buffer) | TrySendError::Disconnected(buffer)) =
            self.chunks.try_send(buffer)
        {
            add_dropped(&self.dropped_frames, buffer.len() / self.channels.get());
            let _returned = self.recycle.try_send(buffer).is_ok();
        }
    }
}

fn build_stream(
    device: &Device,
    format: SampleFormat,
    config: StreamConfig,
    callback: CallbackPorts,
    fault: Arc<AtomicU8>,
) -> Result<Stream, AudioInputError> {
    macro_rules! typed_stream {
        ($sample:ty) => {
            build_typed_stream::<$sample>(device, config, callback, fault)
        };
    }
    match format {
        SampleFormat::I8 => typed_stream!(i8),
        SampleFormat::I16 => typed_stream!(i16),
        SampleFormat::I24 => typed_stream!(cpal::I24),
        SampleFormat::I32 => typed_stream!(i32),
        SampleFormat::I64 => typed_stream!(i64),
        SampleFormat::U8 => typed_stream!(u8),
        SampleFormat::U16 => typed_stream!(u16),
        SampleFormat::U24 => typed_stream!(cpal::U24),
        SampleFormat::U32 => typed_stream!(u32),
        SampleFormat::U64 => typed_stream!(u64),
        SampleFormat::F32 => typed_stream!(f32),
        SampleFormat::F64 => typed_stream!(f64),
        _ => Err(AudioInputError::UnsupportedConfiguration),
    }
}

fn build_typed_stream<T>(
    device: &Device,
    config: StreamConfig,
    callback: CallbackPorts,
    fault: Arc<AtomicU8>,
) -> Result<Stream, AudioInputError>
where
    T: cpal::SizedSample + Copy,
    f32: FromSample<T>,
{
    device
        .build_input_stream::<T, _, _>(
            config,
            move |data, _| callback.deliver(data),
            move |error| map_stream_fault(error.kind()).store(&fault),
            Some(STREAM_BUILD_TIMEOUT),
        )
        .map_err(|error| map_device_error(&error, "build input stream"))
}

fn add_dropped(counter: &AtomicUsize, frames: usize) {
    let _previous = counter.fetch_add(frames, Ordering::Relaxed);
}

fn map_stream_fault(kind: ErrorKind) -> StreamFault {
    match kind {
        ErrorKind::DeviceChanged | ErrorKind::DeviceNotAvailable | ErrorKind::StreamInvalidated => {
            StreamFault::DeviceLost
        }
        _ => StreamFault::StreamError,
    }
}

fn map_device_error(error: &cpal::Error, operation: &'static str) -> AudioInputError {
    match error.kind() {
        ErrorKind::PermissionDenied => AudioInputError::PermissionDenied,
        ErrorKind::DeviceNotAvailable => AudioInputError::DeviceUnavailable,
        ErrorKind::DeviceChanged | ErrorKind::StreamInvalidated => AudioInputError::DeviceLost,
        ErrorKind::UnsupportedConfig | ErrorKind::UnsupportedOperation => {
            AudioInputError::UnsupportedConfiguration
        }
        _ => {
            tracing::warn!(%error, operation, "the audio backend failed");
            backend(operation)
        }
    }
}

pub(super) fn backend(operation: &'static str) -> AudioInputError {
    AudioInputError::Backend { operation }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(
        channels: u16,
        min: u32,
        max: u32,
        format: SampleFormat,
    ) -> SupportedStreamConfigRange {
        SupportedStreamConfigRange::new(
            channels,
            min,
            max,
            SupportedBufferSize::Range {
                min: 128,
                max: 2_048,
            },
            format,
        )
    }

    #[test]
    fn negotiation_prefers_float_mono_and_a_standard_rate() {
        let selected = negotiate_input_config(&[
            range(2, 44_100, 44_100, SampleFormat::I16),
            range(1, 8_000, 96_000, SampleFormat::F32),
        ])
        .expect("supported configuration");
        assert_eq!(selected.sample_format(), SampleFormat::F32);
        assert_eq!(selected.channels(), 1);
        assert_eq!(selected.sample_rate(), 48_000);
    }

    #[test]
    fn negotiation_rejects_dsd_and_clamps_unusual_rates() {
        assert!(
            negotiate_input_config(&[range(1, 2_822_400, 2_822_400, SampleFormat::DsdU8)])
                .is_none()
        );
        let selected = negotiate_input_config(&[range(2, 22_050, 32_000, SampleFormat::I16)])
            .expect("PCM configuration");
        assert_eq!(selected.sample_rate(), 32_000);
    }

    fn ports(channels: usize, buffer: Option<Vec<f32>>) -> (CallbackPorts, Receiver<Vec<f32>>) {
        let (chunks, chunks_rx) = sync_channel(1);
        let (recycle, available) = sync_channel(1);
        if let Some(buffer) = buffer {
            recycle.send(buffer).expect("seed buffer pool");
        }
        let ports = CallbackPorts {
            channels: NonZeroUsize::new(channels).expect("channels"),
            chunks,
            available,
            recycle,
            dropped_frames: Arc::new(AtomicUsize::new(0)),
        };
        (ports, chunks_rx)
    }

    #[test]
    fn callback_conversion_is_nonblocking_bounded_and_normalized() {
        let (ports, chunks_rx) = ports(2, Some(Vec::with_capacity(6)));
        ports.deliver(&[i16::MIN, -16_384, 0, 16_384, i16::MAX, 0]);
        let chunk = chunks_rx.recv().expect("callback chunk");
        assert_eq!(chunk.len(), 6);
        // The positive i16 maximum is one 1/32768 step below full scale.
        let expected = [-1.0, -0.5, 0.0, 0.5, 0.999_969_5, 0.0];
        for (index, (&actual, expected)) in chunk.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < f32::EPSILON,
                "sample {index}: expected {expected}, got {actual}"
            );
        }
        assert_eq!(ports.dropped_frames.load(Ordering::Relaxed), 0);
    }

    fn converted<T>(samples: &[T]) -> Vec<f32>
    where
        T: Sample + Copy,
        f32: FromSample<T>,
    {
        let (ports, chunks_rx) = ports(1, Some(Vec::with_capacity(samples.len())));
        ports.deliver(samples);
        chunks_rx.recv().expect("converted chunk")
    }

    #[test]
    fn callback_larger_than_the_negotiated_period_is_fully_captured() {
        const FRAMES: usize = 2_048;
        const CHANNELS: usize = 2;
        let (ports, chunks_rx) = ports(
            CHANNELS,
            Some(Vec::with_capacity(
                PREFERRED_BUFFER_FRAMES as usize * CHANNELS,
            )),
        );
        let callback = (0..FRAMES * CHANNELS)
            .map(|index| i16::try_from(index % 1_000).expect("sample value"))
            .collect::<Vec<_>>();

        ports.deliver(&callback);

        let chunk = chunks_rx.recv().expect("oversized callback chunk");
        assert_eq!(chunk.len(), FRAMES * CHANNELS);
        assert_eq!(chunk, converted(&callback));
        assert_eq!(ports.dropped_frames.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn callback_drops_instead_of_allocating_or_blocking_when_pool_is_empty() {
        let (ports, _chunks_rx) = ports(2, None);
        ports.deliver(&[0.0_f32; 8]);
        assert_eq!(ports.dropped_frames.load(Ordering::Relaxed), 4);
    }
}
