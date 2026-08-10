use whispernow_domain::{
    ArchiveFormat, CLEANUP_MODEL_FILE, CLEANUP_MODEL_IDS, DEFAULT_CLEANUP_MODEL_ID,
    MINICPM_CLEANUP_MODEL_ID, ModelFamily, ModelId, ModelRegistry, RelativeRating,
};

const RELEASE_BASE: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models";

#[test]
fn builtin_registry_matches_python_order_and_verified_sources() {
    let registry = ModelRegistry::builtin();
    let expected = [
        (
            "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16",
            "Parakeet TDT v2 0.6B (FP16)",
            1_120_982_957,
            "37f67a1a6c942dae27d345ee395fbd19e25ee48996faf70fca25779026054cf0",
        ),
        (
            "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
            "Parakeet TDT v2 0.6B (Int8)",
            482_468_385,
            "157c157bc51155e03e37d2466522a3a737dd9c72bb25f36eb18912964161e1ad",
        ),
        (
            "sherpa-onnx-whisper-distil-large-v3.5",
            "Whisper Distil Large v3.5",
            528_659_059,
            "ec874c7346d24ef8063e05430ede616d66d80a410360283099d0bdf659187b1d",
        ),
        (
            "sherpa-onnx-whisper-small",
            "Whisper Small",
            639_387_718,
            "486a46afbb7ba798507190ffe02fea2dd726049af212e774537efac6afb210a6",
        ),
        (
            "sherpa-onnx-whisper-tiny",
            "Whisper Tiny",
            116_204_861,
            "c46116994e539aa165266d96b325252728429c12535eb9d8b6a2b10f129e66b1",
        ),
        (
            "sherpa-onnx-whisper-large-v3",
            "Whisper Large v3",
            1_068_482_488,
            "2d0e134b3b5fc4a0533baf24a0c9d473b629aa47f030af0a165a05f461df7a03",
        ),
    ];

    let speech = registry.speech_models().collect::<Vec<_>>();
    assert_eq!(speech.len(), expected.len());
    for (model, (id, display_name, size, sha256)) in speech.into_iter().zip(expected) {
        assert_eq!(model.id.as_str(), id);
        assert_eq!(model.display_name, display_name);
        assert_eq!(model.source.url, format!("{RELEASE_BASE}/{id}.tar.bz2"));
        assert_eq!(model.source.archive_format, ArchiveFormat::TarBz2);
        assert_eq!(model.source.archive_size_bytes, size);
        assert_eq!(model.source.sha256, sha256);
        assert_eq!(model.source.sha256.len(), 64);
        assert!(
            model
                .source
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        );
    }
}

#[test]
fn each_cleanup_model_is_one_pinned_gguf_and_not_a_speech_model() {
    let registry = ModelRegistry::builtin();
    let expected = [
        (
            DEFAULT_CLEANUP_MODEL_ID,
            "Local cleanup model (LFM2.5 350M)",
            379_215_072,
            "a46d6f79213615cd2faa960c24e8fb2ace09c28e3c1a07ef15f6e9cad974d53b",
        ),
        (
            MINICPM_CLEANUP_MODEL_ID,
            "Local cleanup model (MiniCPM5 1B)",
            1_153_520_640,
            "9c28d3e0c029907624ad0c2dac0adda77d485ddbcfbfebd67658aceb434e3fbb",
        ),
    ];
    assert_eq!(
        CLEANUP_MODEL_IDS,
        expected.map(|(id, ..)| id),
        "the default comes first"
    );
    for (id, display_name, size, sha256) in expected {
        assert_layout(&registry, id, ModelFamily::Cleanup, &[CLEANUP_MODEL_FILE]);
        let model = registry.get(id).expect("the cleanup model is registered");
        assert!(!model.family().is_speech());
        assert_eq!(model.display_name, display_name);
        assert_eq!(model.source.archive_format, ArchiveFormat::SingleFile);
        assert_eq!(model.source.archive_size_bytes, size);
        assert_eq!(model.source.sha256, sha256);
        assert!(model.source.url.starts_with("https://"));
        assert!(model.source.url.ends_with(&format!("/{id}.gguf")));
        assert!(
            registry
                .speech_models()
                .all(|model| model.id.as_str() != id)
        );
    }
    let [small, large] = CLEANUP_MODEL_IDS.map(|id| registry.get(id).expect("registered").profile);
    assert_eq!(
        (small.speed, small.accuracy),
        (RelativeRating::High, RelativeRating::Medium)
    );
    assert_eq!(
        (large.speed, large.accuracy),
        (RelativeRating::Medium, RelativeRating::High)
    );
}

#[test]
fn transducer_layouts_are_precision_specific_and_complete() {
    let registry = ModelRegistry::builtin();

    assert_layout(
        &registry,
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16",
        ModelFamily::Transducer,
        &[
            "encoder.fp16.onnx",
            "decoder.fp16.onnx",
            "joiner.fp16.onnx",
            "tokens.txt",
        ],
    );
    assert_layout(
        &registry,
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
        ModelFamily::Transducer,
        &[
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "joiner.int8.onnx",
            "tokens.txt",
        ],
    );
}

#[test]
fn whisper_layouts_use_self_contained_int8_artifacts() {
    let registry = ModelRegistry::builtin();
    let fixtures = [
        (
            "sherpa-onnx-whisper-distil-large-v3.5",
            vec![
                "distil-large-v3.5-encoder.int8.onnx",
                "distil-large-v3.5-decoder.int8.onnx",
                "distil-large-v3.5-tokens.txt",
            ],
        ),
        (
            "sherpa-onnx-whisper-small",
            vec![
                "small-encoder.int8.onnx",
                "small-decoder.int8.onnx",
                "small-tokens.txt",
            ],
        ),
        (
            "sherpa-onnx-whisper-tiny",
            vec![
                "tiny-encoder.int8.onnx",
                "tiny-decoder.int8.onnx",
                "tiny-tokens.txt",
            ],
        ),
        (
            "sherpa-onnx-whisper-large-v3",
            vec![
                "large-v3-encoder.int8.onnx",
                "large-v3-decoder.int8.onnx",
                "large-v3-tokens.txt",
            ],
        ),
    ];

    for (id, artifacts) in fixtures {
        assert_layout(&registry, id, ModelFamily::Whisper, &artifacts);
    }
}

#[test]
fn only_whisper_models_claim_languages() {
    let registry = ModelRegistry::builtin();
    for model in registry.models() {
        let capabilities = &model.capabilities;
        if model.family() == ModelFamily::Whisper {
            assert!(capabilities.automatic_language_detection);
            assert_eq!(capabilities.languages, ["en"]);
        } else {
            assert!(!capabilities.automatic_language_detection);
            assert!(capabilities.languages.is_empty());
        }
    }
}

#[test]
fn lookup_accepts_typed_and_text_ids_and_rejects_unknown_models() {
    let registry = ModelRegistry::builtin();
    let id = ModelId::new("sherpa-onnx-whisper-tiny").expect("built-in identifier is valid");

    assert_eq!(registry.get(&id).map(|model| &model.id), Some(&id));
    assert_eq!(
        registry
            .get("sherpa-onnx-whisper-tiny")
            .map(|model| &model.id),
        Some(&id)
    );
    assert!(registry.get("unknown-model").is_none());
}

#[test]
fn identifiers_reject_filesystem_components() {
    for invalid in [
        "",
        ".",
        "..",
        "../model",
        "nested/model",
        r"nested\model",
        "C:model",
        "UPPERCASE",
        "model name",
    ] {
        assert!(ModelId::new(invalid).is_err(), "accepted ID {invalid:?}");
    }
}

fn assert_layout(registry: &ModelRegistry, id: &str, family: ModelFamily, expected_paths: &[&str]) {
    let model = registry.get(id).expect("fixture model is registered");
    assert_eq!(model.family(), family);
    assert_eq!(model.layout.required_artifacts(), expected_paths);
}
