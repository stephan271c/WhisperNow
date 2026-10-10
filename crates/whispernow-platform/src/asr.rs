use std::path::Path;

use sherpa_onnx::{
    OfflineQwen3ASRModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineSenseVoiceModelConfig, OfflineTransducerModelConfig, OfflineWhisperModelConfig,
};
use whispernow_core::{
    ResolvedModel,
    asr::{AsrEngine, AsrError},
    audio::TARGET_SAMPLE_RATE,
};
use whispernow_domain::{ModelId, ModelLayout};

const THREAD_COUNT: i32 = 4;
const TRANSDUCER_FEATURE_DIM: i32 = 80;
const GREEDY_SEARCH: &str = "greedy_search";
/// A 25 s segment is about 325 audio tokens, more than sherpa-onnx's default 512-token budget
/// leaves room for with the text.
const QWEN3_ASR_MAX_TOTAL_TOKENS: i32 = 1024;
const QWEN3_ASR_MAX_NEW_TOKENS: i32 = 256;

#[derive(Default)]
pub struct SherpaAsrEngine {
    loaded: Option<(ModelId, OfflineRecognizer)>,
}

impl SherpaAsrEngine {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl AsrEngine for SherpaAsrEngine {
    fn loaded_model_id(&self) -> Option<&ModelId> {
        self.loaded.as_ref().map(|(model_id, _)| model_id)
    }

    fn load(&mut self, model: &ResolvedModel) -> Result<(), AsrError> {
        let recognizer =
            OfflineRecognizer::create(&recognizer_config(model)?).ok_or(AsrError::LoadFailed)?;
        self.loaded = Some((model.definition.id.clone(), recognizer));
        Ok(())
    }

    fn transcribe(&mut self, samples: &[f32]) -> Result<String, AsrError> {
        let (_, recognizer) = self.loaded.as_ref().ok_or(AsrError::TranscriptionFailed)?;
        let stream = recognizer.create_stream();
        stream.accept_waveform(TARGET_SAMPLE_RATE.cast_signed(), samples);
        recognizer.decode(&stream);
        Ok(stream
            .get_result()
            .ok_or(AsrError::TranscriptionFailed)?
            .text)
    }

    fn unload(&mut self) {
        self.loaded = None;
    }
}

fn recognizer_config(model: &ResolvedModel) -> Result<OfflineRecognizerConfig, AsrError> {
    let path = |artifact: &str| artifact_path(&model.root, artifact);
    let mut config = OfflineRecognizerConfig::default();
    config.model_config.num_threads = THREAD_COUNT;
    config.model_config.provider = Some("cpu".to_owned());
    config.decoding_method = Some(GREEDY_SEARCH.to_owned());
    match &model.definition.layout {
        ModelLayout::Transducer {
            encoder,
            decoder,
            joiner,
            tokens,
        } => {
            config.model_config.tokens = Some(path(tokens)?);
            config.model_config.transducer = OfflineTransducerModelConfig {
                encoder: Some(path(encoder)?),
                decoder: Some(path(decoder)?),
                joiner: Some(path(joiner)?),
            };
            config.model_config.model_type = Some("nemo_transducer".to_owned());
            config.feat_config.sample_rate = TARGET_SAMPLE_RATE.cast_signed();
            config.feat_config.feature_dim = TRANSDUCER_FEATURE_DIM;
        }
        ModelLayout::Whisper {
            encoder,
            decoder,
            tokens,
        } => {
            config.model_config.tokens = Some(path(tokens)?);
            config.model_config.whisper = OfflineWhisperModelConfig {
                encoder: Some(path(encoder)?),
                decoder: Some(path(decoder)?),
                // Empty detects the spoken language.
                language: Some(String::new()),
                task: Some("transcribe".to_owned()),
                tail_paddings: 0,
                enable_token_timestamps: false,
                enable_segment_timestamps: false,
            };
        }
        ModelLayout::SenseVoice {
            model: network,
            tokens,
        } => {
            config.model_config.tokens = Some(path(tokens)?);
            config.model_config.sense_voice = OfflineSenseVoiceModelConfig {
                model: Some(path(network)?),
                language: Some("auto".to_owned()),
                // Inverse text normalization: punctuation and digits.
                use_itn: true,
            };
        }
        ModelLayout::Qwen3Asr {
            conv_frontend,
            encoder,
            decoder,
            vocab,
            ..
        } => {
            let tokenizer = Path::new(vocab).parent().ok_or(AsrError::LoadFailed)?;
            config.model_config.qwen3_asr = OfflineQwen3ASRModelConfig {
                conv_frontend: Some(path(conv_frontend)?),
                encoder: Some(path(encoder)?),
                decoder: Some(path(decoder)?),
                tokenizer: Some(artifact_path(&model.root, tokenizer)?),
                max_total_len: QWEN3_ASR_MAX_TOTAL_TOKENS,
                max_new_tokens: QWEN3_ASR_MAX_NEW_TOKENS,
                ..OfflineQwen3ASRModelConfig::default()
            };
        }
        ModelLayout::Gguf { .. } => return Err(AsrError::LoadFailed),
    }
    Ok(config)
}

fn artifact_path(root: &Path, artifact: impl AsRef<Path>) -> Result<String, AsrError> {
    root.join(artifact)
        .into_os_string()
        .into_string()
        .map_err(|_| AsrError::LoadFailed)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use whispernow_core::ResolvedModel;
    use whispernow_domain::{ModelAvailability, ModelRegistry};

    use super::recognizer_config;
    use whispernow_core::audio::TARGET_SAMPLE_RATE;

    fn model(id: &str) -> ResolvedModel {
        ResolvedModel {
            definition: ModelRegistry::builtin().get(id).expect("model").clone(),
            root: PathBuf::from("/models").join(id),
            availability: ModelAvailability::Downloaded,
        }
    }

    #[test]
    fn each_family_gets_its_own_cpu_greedy_config() {
        let transducer = recognizer_config(&model("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16"))
            .expect("transducer config");
        assert_eq!(transducer.model_config.provider.as_deref(), Some("cpu"));
        assert_eq!(transducer.model_config.num_threads, 4);
        assert_eq!(
            transducer.feat_config.sample_rate,
            TARGET_SAMPLE_RATE.cast_signed()
        );
        assert_eq!(transducer.feat_config.feature_dim, 80);
        assert_eq!(
            transducer.model_config.model_type.as_deref(),
            Some("nemo_transducer")
        );
        assert!(
            transducer
                .model_config
                .transducer
                .joiner
                .as_deref()
                .is_some_and(|joiner| joiner.starts_with("/models/sherpa-onnx-nemo-parakeet"))
        );
        assert_eq!(transducer.decoding_method.as_deref(), Some("greedy_search"));

        let whisper =
            recognizer_config(&model("sherpa-onnx-whisper-turbo")).expect("whisper config");
        assert_eq!(whisper.model_config.whisper.language.as_deref(), Some(""));
        assert_eq!(
            whisper.model_config.whisper.task.as_deref(),
            Some("transcribe")
        );
        assert!(!whisper.model_config.whisper.enable_token_timestamps);
        assert_eq!(whisper.model_config.transducer.joiner, None);

        let qwen3 = recognizer_config(&model("sherpa-onnx-qwen3-asr-0.6b-int8-2026-03-25"))
            .expect("qwen3 config");
        assert_eq!(
            qwen3.model_config.qwen3_asr.tokenizer.as_deref(),
            Some("/models/sherpa-onnx-qwen3-asr-0.6b-int8-2026-03-25/tokenizer")
        );
        assert_eq!(qwen3.model_config.qwen3_asr.max_total_len, 1024);

        let sense_voice = recognizer_config(&model(
            "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
        ))
        .expect("sense voice config");
        assert_eq!(
            sense_voice.model_config.sense_voice.language.as_deref(),
            Some("auto")
        );
        assert!(sense_voice.model_config.sense_voice.use_itn);
    }
}
