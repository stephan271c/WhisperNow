//! Model identifiers, and metadata for the supported speech models and the local cleanup
//! models.
//!
//! Capabilities and artifact layouts are data owned by the registry. Consumers
//! must not infer either from a model identifier, display name, or filename.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, de::Error as _};

const RELEASE_BASE: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models";

/// The default speech model, the Python release's default.
const DEFAULT_SPEECH_MODEL_ID: &str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16";

/// The local cleanup model AI cleanup uses unless another is chosen: LFM2.5 350M.
pub const DEFAULT_CLEANUP_MODEL_ID: &str = "whispernow-cleanup-lfm2.5-350m-q8_0";
/// The larger local cleanup model (a `MiniCPM5-1B` fine-tune): slower, but keeps more facts.
pub const MINICPM_CLEANUP_MODEL_ID: &str = "whispernow-cleanup-minicpm5-1b-q8_0";
/// Every local cleanup model, the default first.
pub const CLEANUP_MODEL_IDS: [&str; 2] = [DEFAULT_CLEANUP_MODEL_ID, MINICPM_CLEANUP_MODEL_ID];
/// The CUDA pack's id among downloads: llama.cpp's CUDA backend and NVIDIA's CUDA runtime,
/// which run the cleanup models on NVIDIA GPUs. Not a model, and not in the registry.
pub const CUDA_PACK_ID: &str = "llama-cuda-13.4";
/// The GGUF inside a local cleanup model's directory.
pub const CLEANUP_MODEL_FILE: &str = "model-Q8_0.gguf";
/// Where the local cleanup models are published: pinned release assets. They are
/// placeholders until the models are uploaded, which waits for the user's go-ahead.
const CLEANUP_MODEL_RELEASE: &str =
    "https://github.com/stephan271c/WhisperNow/releases/download/cleanup-model-r112";

/// A speech or cleanup model identifier, safe as one filesystem path component.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ModelId(String);

impl ModelId {
    /// Creates a validated model identifier.
    ///
    /// # Errors
    ///
    /// Returns an error unless the identifier is one safe filesystem component
    /// containing only lowercase ASCII letters, digits, `.`, `_`, and `-`.
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

    /// Returns the identifier as text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for ModelId {
    fn as_ref(&self) -> &str {
        self.as_str()
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

/// Artifact/configuration family understood by the model registry.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    /// Encoder/decoder/joiner transducer models, including Parakeet TDT.
    Transducer,
    /// Whisper encoder/decoder models.
    Whisper,
    /// The local cleanup model: one GGUF that llama.cpp runs. It does not transcribe.
    Cleanup,
    /// Not a model: llama.cpp's CUDA backend, downloaded and listed with the models it runs.
    CudaPack,
}

impl ModelFamily {
    /// Whether models of this family transcribe speech.
    #[must_use]
    pub const fn is_speech(self) -> bool {
        matches!(self, Self::Transducer | Self::Whisper)
    }
}

/// Where a model install has got to.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadPhase {
    Downloading,
    ValidatingArchive,
    Extracting,
    Installing,
    Complete,
}

/// Whether a model is on this computer, and how it got there.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAvailability {
    Downloaded,
    Bundled,
    NotInstalled,
}

/// The languages an ASR model is known to recognize.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AsrCapabilities {
    /// BCP-47 tags tested with the model. Empty means unknown.
    pub languages: Vec<String>,
    /// Whether the model detects the spoken language itself.
    pub automatic_language_detection: bool,
}

/// The files a model needs, as slash-separated paths relative to its directory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelLayout {
    /// Offline encoder/decoder/joiner transducer artifacts.
    Transducer {
        encoder: String,
        decoder: String,
        joiner: String,
        tokens: String,
    },
    /// One GGUF for llama.cpp: the local cleanup model.
    Gguf { model: String },
    /// Offline Whisper encoder/decoder artifacts.
    Whisper {
        encoder: String,
        decoder: String,
        tokens: String,
    },
}

impl ModelLayout {
    /// The recognizer configuration family this layout belongs to.
    #[must_use]
    pub const fn family(&self) -> ModelFamily {
        match self {
            Self::Transducer { .. } => ModelFamily::Transducer,
            Self::Whisper { .. } => ModelFamily::Whisper,
            Self::Gguf { .. } => ModelFamily::Cleanup,
        }
    }

    /// Returns every file that must exist before the model can be installed.
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
        }
    }
}

/// Download source and integrity metadata for one model archive.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelSource {
    /// HTTPS URL for the release asset.
    pub url: String,
    /// Container/compression format expected independently of the URL suffix.
    pub archive_format: ArchiveFormat,
    /// Exact expected compressed response size.
    pub archive_size_bytes: u64,
    /// Lowercase SHA-256 digest of the compressed archive.
    pub sha256: String,
}

/// Archive formats supported by the safe model installer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ArchiveFormat {
    /// A bzip2-compressed tar archive.
    TarBz2,
    /// The model's one required artifact itself, stored under the archive root.
    SingleFile,
}

/// A coarse rating that compares one model with the other built-in models.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RelativeRating {
    Low,
    Medium,
    High,
}

/// How a model trades transcription speed for accuracy, relative to the other built-in models.
///
/// A hint for choosing between models on the same computer, not a measurement.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ModelProfile {
    pub speed: RelativeRating,
    pub accuracy: RelativeRating,
}

impl ModelProfile {
    const fn new(speed: RelativeRating, accuracy: RelativeRating) -> Self {
        Self { speed, accuracy }
    }
}

/// One supported speech model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelDefinition {
    /// Stable model identifier used by settings and on-disk paths.
    pub id: ModelId,
    /// User-facing model name.
    pub display_name: String,
    /// Download and integrity metadata. An archive holds the files under a directory
    /// named after the id.
    pub source: ModelSource,
    /// Required family-specific files; they determine the model family.
    pub layout: ModelLayout,
    pub capabilities: AsrCapabilities,
    /// Relative speed and accuracy among the built-in models.
    pub profile: ModelProfile,
}

impl ModelDefinition {
    /// Recognizer configuration family, derived from the artifact layout.
    #[must_use]
    pub const fn family(&self) -> ModelFamily {
        self.layout.family()
    }
}

/// The fixed set of supported models.
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

    /// Builds the registry of the six ASR models and two local cleanup models.
    #[must_use]
    pub fn builtin() -> Self {
        Self::new(vec![
            transducer(
                "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16",
                "Parakeet TDT v2 0.6B (FP16)",
                1_120_982_957,
                "37f67a1a6c942dae27d345ee395fbd19e25ee48996faf70fca25779026054cf0",
                "fp16",
                ModelProfile::new(RelativeRating::Medium, RelativeRating::High),
            ),
            transducer(
                "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
                "Parakeet TDT v2 0.6B (Int8)",
                482_468_385,
                "157c157bc51155e03e37d2466522a3a737dd9c72bb25f36eb18912964161e1ad",
                "int8",
                ModelProfile::new(RelativeRating::High, RelativeRating::High),
            ),
            whisper(
                "sherpa-onnx-whisper-distil-large-v3.5",
                "Whisper Distil Large v3.5",
                "distil-large-v3.5",
                528_659_059,
                "ec874c7346d24ef8063e05430ede616d66d80a410360283099d0bdf659187b1d",
                ModelProfile::new(RelativeRating::Medium, RelativeRating::High),
            ),
            whisper(
                "sherpa-onnx-whisper-small",
                "Whisper Small",
                "small",
                639_387_718,
                "486a46afbb7ba798507190ffe02fea2dd726049af212e774537efac6afb210a6",
                ModelProfile::new(RelativeRating::Medium, RelativeRating::Medium),
            ),
            whisper(
                "sherpa-onnx-whisper-tiny",
                "Whisper Tiny",
                "tiny",
                116_204_861,
                "c46116994e539aa165266d96b325252728429c12535eb9d8b6a2b10f129e66b1",
                ModelProfile::new(RelativeRating::High, RelativeRating::Low),
            ),
            whisper(
                "sherpa-onnx-whisper-large-v3",
                "Whisper Large v3",
                "large-v3",
                1_068_482_488,
                "2d0e134b3b5fc4a0533baf24a0c9d473b629aa47f030af0a165a05f461df7a03",
                ModelProfile::new(RelativeRating::Low, RelativeRating::High),
            ),
            // `training/runs/r112-full-inst-1e-5-e3/gguf/model-Q8_0.gguf`: LFM2.5-350M
            // fine-tuned for cleanup, under the
            // LFM Open License v1.0.
            local_cleanup(
                DEFAULT_CLEANUP_MODEL_ID,
                "Local cleanup model (LFM2.5 350M)",
                379_215_072,
                "a46d6f79213615cd2faa960c24e8fb2ace09c28e3c1a07ef15f6e9cad974d53b",
                ModelProfile::new(RelativeRating::High, RelativeRating::Medium),
            ),
            // `training/runs/r112-minicpm-full-inst-1e-5-e3/gguf/model-Q8_0.gguf`:
            // MiniCPM5-1B fine-tuned the same way, under Apache-2.0.
            local_cleanup(
                MINICPM_CLEANUP_MODEL_ID,
                "Local cleanup model (MiniCPM5 1B)",
                1_153_520_640,
                "9c28d3e0c029907624ad0c2dac0adda77d485ddbcfbfebd67658aceb434e3fbb",
                ModelProfile::new(RelativeRating::Medium, RelativeRating::High),
            ),
        ])
    }

    /// Finds a model by its exact stable identifier.
    #[must_use]
    pub fn get(&self, id: impl AsRef<str>) -> Option<&ModelDefinition> {
        let id = id.as_ref();
        self.models.iter().find(|model| model.id.as_str() == id)
    }

    /// The models that transcribe speech, in display order.
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

fn source(id: &str, archive_size_bytes: u64, sha256: &str) -> ModelSource {
    ModelSource {
        url: format!("{RELEASE_BASE}/{id}.tar.bz2"),
        archive_format: ArchiveFormat::TarBz2,
        archive_size_bytes,
        sha256: sha256.to_owned(),
    }
}

fn transducer(
    id: &str,
    display_name: &str,
    archive_size_bytes: u64,
    sha256: &str,
    precision: &str,
    profile: ModelProfile,
) -> ModelDefinition {
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
        capabilities: AsrCapabilities::default(),
        profile,
    }
}

fn whisper(
    id: &str,
    display_name: &str,
    artifact_prefix: &str,
    archive_size_bytes: u64,
    sha256: &str,
    profile: ModelProfile,
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
        // Whisper detects the language. Only English is tested; tag claims stay
        // conservative until each language is.
        capabilities: AsrCapabilities {
            languages: vec!["en".to_owned()],
            automatic_language_detection: true,
        },
        profile,
    }
}

/// A local cleanup model: one GGUF, downloaded as is.
fn local_cleanup(
    id: &str,
    display_name: &str,
    size_bytes: u64,
    sha256: &str,
    profile: ModelProfile,
) -> ModelDefinition {
    ModelDefinition {
        id: ModelId::new(id).expect("built-in identifiers are non-empty"),
        display_name: display_name.to_owned(),
        source: ModelSource {
            url: format!("{CLEANUP_MODEL_RELEASE}/{id}.gguf"),
            archive_format: ArchiveFormat::SingleFile,
            archive_size_bytes: size_bytes,
            sha256: sha256.to_owned(),
        },
        layout: ModelLayout::Gguf {
            model: CLEANUP_MODEL_FILE.to_owned(),
        },
        capabilities: AsrCapabilities::default(),
        profile,
    }
}
