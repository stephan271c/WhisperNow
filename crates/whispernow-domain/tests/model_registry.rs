use std::collections::BTreeSet;

use whispernow_domain::{
    ArchiveFormat, CLEANUP_MODEL_FILE, CLEANUP_MODEL_IDS, DEFAULT_CLEANUP_MODEL_ID, ModelFamily,
    ModelId, ModelRegistry,
};

const RELEASE_BASE: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models";

#[test]
fn every_model_has_a_unique_id_and_a_pinned_source_named_after_it() {
    let registry = ModelRegistry::builtin();
    let mut ids = BTreeSet::new();
    for model in registry.models() {
        let id = model.id.as_str();
        assert!(ids.insert(id), "{id} is registered twice");
        let source = &model.source;
        assert_eq!(source.sha256.len(), 64, "{id}");
        assert!(
            source.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "{id}"
        );
        assert!(source.archive_size_bytes > 0, "{id}");
        match model.family() {
            ModelFamily::Transducer | ModelFamily::Whisper => {
                assert_eq!(source.url, format!("{RELEASE_BASE}/{id}.tar.bz2"));
                assert_eq!(source.archive_format, ArchiveFormat::TarBz2, "{id}");
            }
            ModelFamily::Cleanup => {
                assert!(source.url.starts_with("https://"), "{id}");
                assert!(source.url.ends_with(&format!("/{id}.gguf")), "{id}");
                assert_eq!(source.archive_format, ArchiveFormat::SingleFile, "{id}");
            }
            ModelFamily::CudaPack => {}
        }
    }
    assert!(registry.speech_models().count() > 1);
}

#[test]
fn each_model_has_the_artifacts_its_family_needs() {
    for model in ModelRegistry::builtin().models() {
        let id = model.id.as_str();
        let artifacts = model.layout.required_artifacts();
        let has = |suffix: &str| artifacts.iter().any(|path| path.ends_with(suffix));
        match model.family() {
            ModelFamily::Transducer => {
                // The id names the precision every network file is in.
                let precision = id.rsplit('-').next().expect("a precision suffix");
                for part in ["encoder", "decoder", "joiner"] {
                    assert!(has(&format!("{part}.{precision}.onnx")), "{id} {part}");
                }
                assert!(has("tokens.txt"), "{id}");
            }
            ModelFamily::Whisper => {
                for suffix in ["encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"] {
                    assert!(has(suffix), "{id} {suffix}");
                }
            }
            ModelFamily::Cleanup => assert_eq!(artifacts, [CLEANUP_MODEL_FILE], "{id}"),
            ModelFamily::CudaPack => {}
        }
    }
}

#[test]
fn cleanup_models_start_with_the_default_and_are_not_speech_models() {
    let registry = ModelRegistry::builtin();
    assert_eq!(CLEANUP_MODEL_IDS[0], DEFAULT_CLEANUP_MODEL_ID);
    for id in CLEANUP_MODEL_IDS {
        let model = registry.get(id).expect("the cleanup model is registered");
        assert_eq!(model.family(), ModelFamily::Cleanup);
        assert!(
            registry
                .speech_models()
                .all(|model| model.id.as_str() != id)
        );
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
