//! Capabilities and artifact layouts are data owned by the registry. Consumers
//! must not infer either from a model identifier, display name, or filename.

use std::fmt;

use crate::ipc::ModelSummary;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};

const RELEASE_BASE: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models";

const DEFAULT_SPEECH_MODEL_ID: &str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8";

// Retrained cleanup models get new ids so installed weights cannot outlive their revision.
pub const DEFAULT_CLEANUP_MODEL_ID: &str = "whispernow-cleanup-lfm2.5-350m-r112-q8_0";
/// The larger local cleanup model (a `MiniCPM5-1B` fine-tune): slower, but keeps more facts.
const MINICPM_CLEANUP_MODEL_ID: &str = "whispernow-cleanup-minicpm5-1b-r112-q8_0";
const LFM_1_2B_CLEANUP_MODEL_ID: &str = "whispernow-cleanup-lfm2.5-1.2b-v2-r112-q8_0";
/// Every local cleanup model, the default first.
pub const CLEANUP_MODEL_IDS: [&str; CLEANUP_MODELS.len()] = {
    let mut ids = [""; CLEANUP_MODELS.len()];
    let mut index = 0;
    while index < CLEANUP_MODELS.len() {
        ids[index] = CLEANUP_MODELS[index].id;
        index += 1;
    }
    ids
};
const CLEANUP_MODEL_FILE: &str = "model-Q8_0.gguf";
/// Where the local cleanup models are published: pinned release assets. The URLs are
/// placeholders until the models are uploaded.
const CLEANUP_MODEL_RELEASE: &str =
    "https://github.com/stephan271c/WhisperNow/releases/download/cleanup-model-r112";

struct CleanupModel {
    id: &'static str,
    display_name: &'static str,
    size_bytes: u64,
    sha256: &'static str,
}

const CLEANUP_MODELS: [CleanupModel; 3] = [
    // `training/runs/r112-full-inst-1e-5-e3/gguf/model-Q8_0.gguf`: LFM2.5-350M
    // fine-tuned for cleanup, under the
    // LFM Open License v1.0.
    CleanupModel {
        id: DEFAULT_CLEANUP_MODEL_ID,
        display_name: "Local cleanup model (LFM2.5 350M)",
        size_bytes: 379_215_072,
        sha256: "a46d6f79213615cd2faa960c24e8fb2ace09c28e3c1a07ef15f6e9cad974d53b",
    },
    // `training/runs/r112-minicpm-full-inst-1e-5-e3/gguf/model-Q8_0.gguf`:
    // MiniCPM5-1B fine-tuned the same way, under Apache-2.0.
    CleanupModel {
        id: MINICPM_CLEANUP_MODEL_ID,
        display_name: "Local cleanup model (MiniCPM5 1B)",
        size_bytes: 1_153_520_640,
        sha256: "9c28d3e0c029907624ad0c2dac0adda77d485ddbcfbfebd67658aceb434e3fbb",
    },
    // `training/runs/r112-lfm25-1.2b-full-base-1e-5-e3-v2-no-heard/gguf/model-Q8_0.gguf`:
    // LFM2.5-1.2B-Base fine-tuned on the v2 targeted data, under the LFM Open License v1.0.
    CleanupModel {
        id: LFM_1_2B_CLEANUP_MODEL_ID,
        display_name: "Local cleanup model (LFM2.5 1.2B)",
        size_bytes: 1_246_252_256,
        sha256: "0fbc2316640539df16f99f2ae4cddfb47bbc7674754ce0711a246be1c8ed507e",
    },
];

/// A speech or cleanup model identifier, safe as one filesystem path component.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ModelId(String);

impl ModelId {
    pub fn new(value: impl Into<String>) -> Result<Self, &'static str> {
        let value = value.into();
        if value.is_empty() {
            return Err("model identifier must not be empty");
        }
        if value == "." || value == ".." {
            return Err("model identifier must not be a filesystem path component");
        }
        if !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(&byte)
        }) {
            return Err(
                "model identifier must contain only lowercase ASCII letters, digits, '.', '_', or '-'",
            );
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Default for ModelId {
    fn default() -> Self {
        Self(DEFAULT_SPEECH_MODEL_ID.to_owned())
    }
}

impl<'de> Deserialize<'de> for ModelId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(D::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    Transducer,
    Whisper,
    SenseVoice,
    Qwen3Asr,
    /// The local cleanup model: one GGUF that llama.cpp runs. It does not transcribe.
    Cleanup,
}

impl ModelFamily {
    #[must_use]
    pub const fn is_speech(self) -> bool {
        matches!(
            self,
            Self::Transducer | Self::Whisper | Self::SenseVoice | Self::Qwen3Asr
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadPhase {
    Downloading,
    ValidatingArchive,
    Extracting,
    Installing,
    Complete,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAvailability {
    Downloaded,
    NotInstalled,
}

/// The files a model needs, as slash-separated paths relative to its directory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelLayout {
    Transducer {
        encoder: String,
        decoder: String,
        joiner: String,
        tokens: String,
    },
    Gguf {
        model: String,
    },
    Whisper {
        encoder: String,
        decoder: String,
        tokens: String,
    },
    SenseVoice {
        model: String,
        tokens: String,
    },
    /// sherpa-onnx reads the tokenizer as a directory; its three files sit in one.
    Qwen3Asr {
        conv_frontend: String,
        encoder: String,
        decoder: String,
        vocab: String,
        merges: String,
        tokenizer_config: String,
    },
}

impl ModelLayout {
    #[must_use]
    pub const fn family(&self) -> ModelFamily {
        match self {
            Self::Transducer { .. } => ModelFamily::Transducer,
            Self::Whisper { .. } => ModelFamily::Whisper,
            Self::SenseVoice { .. } => ModelFamily::SenseVoice,
            Self::Qwen3Asr { .. } => ModelFamily::Qwen3Asr,
            Self::Gguf { .. } => ModelFamily::Cleanup,
        }
    }

    #[must_use]
    pub fn required_artifacts(&self) -> Vec<&str> {
        match self {
            Self::Transducer {
                encoder,
                decoder,
                joiner,
                tokens,
            } => vec![encoder, decoder, joiner, tokens],
            Self::Gguf { model } => vec![model],
            Self::Whisper {
                encoder,
                decoder,
                tokens,
            } => vec![encoder, decoder, tokens],
            Self::SenseVoice { model, tokens } => vec![model, tokens],
            Self::Qwen3Asr {
                conv_frontend,
                encoder,
                decoder,
                vocab,
                merges,
                tokenizer_config,
            } => vec![
                conv_frontend,
                encoder,
                decoder,
                vocab,
                merges,
                tokenizer_config,
            ],
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelSource {
    pub url: String,
    /// Exact expected compressed response size.
    pub archive_size_bytes: u64,
    /// Lowercase SHA-256 digest of the compressed archive.
    pub sha256: String,
}

impl ModelSource {
    /// The directory a `.tar.bz2` archive holds the files under: the URL's file name without
    /// the extension. It can differ from the model id in case.
    #[must_use]
    pub fn archive_root(&self) -> &str {
        let name = self
            .url
            .rsplit('/')
            .next()
            .expect("rsplit always yields an item");
        name.strip_suffix(".tar.bz2").unwrap_or(name)
    }
}

/// A score from one to [`Rating::MAX`]; more is better.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(transparent)]
pub struct Rating(u8);

impl Rating {
    pub const MAX: u8 = 5;

    #[must_use]
    pub const fn new(score: u8) -> Self {
        assert!(score >= 1 && score <= Self::MAX);
        Self(score)
    }

    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// Where a speech model's English word error rate comes from.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WerSource {
    /// The average over the Open ASR Leaderboard's English test sets.
    Leaderboard,
    /// Not on the leaderboard: placed among the models that are by its rate on the app's
    /// dictation test (`datagen asr-speed`).
    Estimated,
}

/// A speech model's measured accuracy and speed.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechBenchmark {
    /// In percent.
    pub english_wer: f32,
    pub wer_source: WerSource,
    /// Times faster than real time on the CPU, with the app's recognizer settings.
    pub speed: f32,
}

impl SpeechBenchmark {
    #[must_use]
    pub fn accuracy(&self) -> Rating {
        Rating::new(match self.english_wer {
            wer if wer < 5.5 => 5,
            wer if wer <= 6.25 => 4,
            wer if wer <= 7.25 => 3,
            wer if wer <= 9.0 => 2,
            _ => 1,
        })
    }

    #[must_use]
    pub fn speed(&self) -> Rating {
        Rating::new(match self.speed {
            speed if speed >= 50.0 => 5,
            speed if speed >= 24.0 => 4,
            speed if speed >= 10.0 => 3,
            speed if speed >= 2.0 => 2,
            _ => 1,
        })
    }
}

fn leaderboard(english_wer: f32, speed: f32) -> SpeechBenchmark {
    SpeechBenchmark {
        english_wer,
        wer_source: WerSource::Leaderboard,
        speed,
    }
}

fn estimated(english_wer: f32, speed: f32) -> SpeechBenchmark {
    SpeechBenchmark {
        english_wer,
        wer_source: WerSource::Estimated,
        speed,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelDefinition {
    pub id: ModelId,
    pub display_name: String,
    pub source: ModelSource,
    pub layout: ModelLayout,
    /// The BCP-47 tags of the languages a speech model transcribes; empty for other models.
    pub languages: Vec<String>,
    pub benchmark: Option<SpeechBenchmark>,
}

impl ModelDefinition {
    #[must_use]
    pub const fn family(&self) -> ModelFamily {
        self.layout.family()
    }

    #[must_use]
    pub fn is_recommended(&self) -> bool {
        self.id == ModelId::default()
    }

    #[must_use]
    pub fn summary(self, availability: ModelAvailability) -> ModelSummary {
        ModelSummary {
            recommended: self.is_recommended(),
            family: self.family(),
            size_bytes: self.source.archive_size_bytes,
            id: self.id,
            display_name: self.display_name,
            availability,
            languages: self.languages,
            benchmark: self.benchmark,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelRegistry {
    models: Vec<ModelDefinition>,
}

impl ModelRegistry {
    /// A registry of `models`; tests use it to point at fixture servers.
    #[must_use]
    pub fn new(models: Vec<ModelDefinition>) -> Self {
        Self { models }
    }

    /// Speech models in display order. Accuracy is the average English WER on the Open ASR
    /// Leaderboard's 24 Jul 2026 snapshot; Parakeet Unified and `SenseVoice` are not listed and
    /// are placed by their WER relative to the listed models in `datagen asr-speed`. Speed is
    /// times real time with the app's recognizer settings on a Ryzen 9 8945HS, measured by
    /// `datagen asr-speed` (training/README.md).
    #[must_use]
    pub fn builtin() -> Self {
        let mut models = vec![
            transducer(
                "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
                "Parakeet TDT v2 0.6B (Int8)",
                Precision::Int8,
                482_468_385,
                "157c157bc51155e03e37d2466522a3a737dd9c72bb25f36eb18912964161e1ad",
                ENGLISH,
                leaderboard(5.39, 27.5),
            ),
            transducer(
                "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16",
                "Parakeet TDT v2 0.6B (FP16)",
                Precision::Fp16,
                1_120_982_957,
                "37f67a1a6c942dae27d345ee395fbd19e25ee48996faf70fca25779026054cf0",
                ENGLISH,
                leaderboard(5.39, 20.9),
            ),
            transducer(
                "sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming",
                "Parakeet Unified 0.6B",
                Precision::Int8,
                501_350_460,
                "99f63605b3a85a54c250c0869670a687b7d6598a47bf2421515e1f839a76e150",
                ENGLISH,
                estimated(5.5, 25.0),
            ),
            transducer(
                "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
                "Parakeet TDT v3 0.6B",
                Precision::Int8,
                487_170_055,
                "5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf",
                PARAKEET_V3_LANGUAGES,
                leaderboard(5.66, 25.9),
            ),
            transducer(
                "sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000-int8",
                "Parakeet TDT 110M",
                Precision::Int8,
                108_035_095,
                "f628312e9fdf8686374cb01a69425c41732529d540860311f16f37cbc32cfe9b",
                ENGLISH,
                leaderboard(6.61, 84.4),
            ),
            sense_voice(
                "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
                "SenseVoice Small",
                163_002_883,
                "7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e",
                estimated(11.0, 55.6),
            ),
            qwen3_asr(
                "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
                "Qwen3-ASR 0.6B",
                878_702_423,
                "393f8a14e2f5fb96746aaab342997a40641001fbd5bf9592a080a8329178ee96",
                leaderboard(5.61, 5.9),
            ),
            whisper(
                "sherpa-onnx-whisper-distil-large-v3.5",
                "Whisper Distil Large v3.5",
                "distil-large-v3.5",
                528_659_059,
                "ec874c7346d24ef8063e05430ede616d66d80a410360283099d0bdf659187b1d",
                ENGLISH,
                leaderboard(6.10, 4.9),
            ),
            whisper(
                "sherpa-onnx-whisper-turbo",
                "Whisper Large v3 Turbo",
                "turbo",
                563_790_207,
                "b11acbbcd660b44a8e0df33724feb5aaa709cf65668f2823d59f656312544f22",
                WHISPER_LANGUAGES,
                leaderboard(7.01, 4.1),
            ),
        ];
        models.extend(CLEANUP_MODELS.iter().map(local_cleanup));
        Self::new(models)
    }

    #[must_use]
    pub fn get(&self, id: &str) -> Option<&ModelDefinition> {
        self.models.iter().find(|model| model.id.as_str() == id)
    }

    pub fn speech_models(&self) -> impl Iterator<Item = &ModelDefinition> {
        self.models
            .iter()
            .filter(|model| model.family().is_speech())
    }

    /// Returns the supported models in display order.
    #[must_use]
    pub fn models(&self) -> &[ModelDefinition] {
        &self.models
    }
}

fn source(archive: &str, archive_size_bytes: u64, sha256: &str) -> ModelSource {
    ModelSource {
        url: format!("{RELEASE_BASE}/{archive}.tar.bz2"),
        archive_size_bytes,
        sha256: sha256.to_owned(),
    }
}

const ENGLISH: &[&str] = &["en"];

const PARAKEET_V3_LANGUAGES: &[&str] = &[
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt",
    "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
];

/// Whisper's tokenizer languages, with `jw` written as its BCP-47 tag `jv`.
const WHISPER_LANGUAGES: &[&str] = &[
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da",
    "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw", "he",
    "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "la",
    "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl",
    "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so",
    "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz",
    "vi", "yi", "yo", "yue", "zh",
];

const SENSE_VOICE_LANGUAGES: &[&str] = &["en", "ja", "ko", "yue", "zh"];

const QWEN3_ASR_LANGUAGES: &[&str] = &[
    "ar", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fil", "fr", "hi", "hu", "id", "it",
    "ja", "ko", "mk", "ms", "nl", "pl", "pt", "ro", "ru", "sv", "th", "tr", "vi", "yue", "zh",
];

fn tags(languages: &[&str]) -> Vec<String> {
    languages.iter().map(|&tag| tag.to_owned()).collect()
}

#[derive(Clone, Copy)]
enum Precision {
    Int8,
    Fp16,
}

impl Precision {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Fp16 => "fp16",
        }
    }
}

fn transducer(
    id: &str,
    display_name: &str,
    precision: Precision,
    archive_size_bytes: u64,
    sha256: &str,
    languages: &[&str],
    benchmark: SpeechBenchmark,
) -> ModelDefinition {
    let precision = precision.as_str();
    ModelDefinition {
        id: ModelId::new(id).expect("built-in identifiers are non-empty"),
        display_name: display_name.to_owned(),
        source: source(id, archive_size_bytes, sha256),
        layout: ModelLayout::Transducer {
            encoder: format!("encoder.{precision}.onnx"),
            decoder: format!("decoder.{precision}.onnx"),
            joiner: format!("joiner.{precision}.onnx"),
            tokens: "tokens.txt".to_owned(),
        },
        languages: tags(languages),
        benchmark: Some(benchmark),
    }
}

fn whisper(
    id: &str,
    display_name: &str,
    artifact_prefix: &str,
    archive_size_bytes: u64,
    sha256: &str,
    languages: &[&str],
    benchmark: SpeechBenchmark,
) -> ModelDefinition {
    ModelDefinition {
        id: ModelId::new(id).expect("built-in identifiers are non-empty"),
        display_name: display_name.to_owned(),
        source: source(id, archive_size_bytes, sha256),
        layout: ModelLayout::Whisper {
            encoder: format!("{artifact_prefix}-encoder.int8.onnx"),
            decoder: format!("{artifact_prefix}-decoder.int8.onnx"),
            tokens: format!("{artifact_prefix}-tokens.txt"),
        },
        languages: tags(languages),
        benchmark: Some(benchmark),
    }
}

fn sense_voice(
    id: &str,
    display_name: &str,
    archive_size_bytes: u64,
    sha256: &str,
    benchmark: SpeechBenchmark,
) -> ModelDefinition {
    ModelDefinition {
        id: ModelId::new(id).expect("built-in identifiers are non-empty"),
        display_name: display_name.to_owned(),
        source: source(id, archive_size_bytes, sha256),
        layout: ModelLayout::SenseVoice {
            model: "model.int8.onnx".to_owned(),
            tokens: "tokens.txt".to_owned(),
        },
        languages: tags(SENSE_VOICE_LANGUAGES),
        benchmark: Some(benchmark),
    }
}

fn qwen3_asr(
    archive: &str,
    display_name: &str,
    archive_size_bytes: u64,
    sha256: &str,
    benchmark: SpeechBenchmark,
) -> ModelDefinition {
    ModelDefinition {
        id: ModelId::new(archive.to_ascii_lowercase()).expect("built-in identifiers are valid"),
        display_name: display_name.to_owned(),
        source: source(archive, archive_size_bytes, sha256),
        layout: ModelLayout::Qwen3Asr {
            conv_frontend: "conv_frontend.onnx".to_owned(),
            encoder: "encoder.int8.onnx".to_owned(),
            decoder: "decoder.int8.onnx".to_owned(),
            vocab: "tokenizer/vocab.json".to_owned(),
            merges: "tokenizer/merges.txt".to_owned(),
            tokenizer_config: "tokenizer/tokenizer_config.json".to_owned(),
        },
        languages: tags(QWEN3_ASR_LANGUAGES),
        benchmark: Some(benchmark),
    }
}

fn local_cleanup(model: &CleanupModel) -> ModelDefinition {
    let id = model.id;
    ModelDefinition {
        id: ModelId::new(id).expect("built-in identifiers are non-empty"),
        display_name: model.display_name.to_owned(),
        source: ModelSource {
            url: format!("{CLEANUP_MODEL_RELEASE}/{id}.gguf"),
            archive_size_bytes: model.size_bytes,
            sha256: model.sha256.to_owned(),
        },
        layout: ModelLayout::Gguf {
            model: CLEANUP_MODEL_FILE.to_owned(),
        },
        languages: Vec::new(),
        benchmark: None,
    }
}
