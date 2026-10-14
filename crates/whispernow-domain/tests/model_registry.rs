use std::collections::BTreeSet;

use whispernow_domain::{
    CLEANUP_MODEL_IDS, DEFAULT_CLEANUP_MODEL_ID, ModelFamily, ModelId, ModelRegistry,
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
            ModelFamily::Cleanup => {
                assert!(source.url.starts_with("https://"), "{id}");
                assert!(source.url.ends_with(&format!("/{id}.gguf")), "{id}");
            }
            ModelFamily::CudaPack => {}
            _ => {
                assert_eq!(
                    source.url,
                    format!("{RELEASE_BASE}/{}.tar.bz2", source.archive_root())
                );
                assert!(source.archive_root().eq_ignore_ascii_case(id), "{id}");
            }
        }
    }
    assert!(registry.speech_models().count() > 1);
}

#[test]
fn speech_models_name_their_languages_and_other_models_none() {
    for model in ModelRegistry::builtin().models() {
        let id = model.id.as_str();
        if model.family().is_speech() {
            assert!(model.languages.iter().any(|tag| tag == "en"), "{id}");
            let mut sorted = model.languages.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted, model.languages, "{id} lists each tag once, sorted");
        } else {
            assert!(model.languages.is_empty(), "{id}");
        }
    }
}

#[test]
fn each_model_has_the_artifacts_its_family_needs() {
    for model in ModelRegistry::builtin().models() {
        let id = model.id.as_str();
        let artifacts = model.layout.required_artifacts();
        let has_artifact = |suffix: &str| artifacts.iter().any(|path| path.ends_with(suffix));
        match model.family() {
            ModelFamily::Transducer => {
                // The id names the precision every network file is in.
                let precision = id
                    .rsplit('-')
                    .find(|part| ["fp16", "int8"].contains(part))
                    .expect("a precision in the id");
                for part in ["encoder", "decoder", "joiner"] {
                    assert!(
                        has_artifact(&format!("{part}.{precision}.onnx")),
                        "{id} {part}"
                    );
                }
                assert!(has_artifact("tokens.txt"), "{id}");
            }
            ModelFamily::Whisper => {
                for suffix in ["encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"] {
                    assert!(has_artifact(suffix), "{id} {suffix}");
                }
            }
            ModelFamily::SenseVoice => {
                assert!(
                    has_artifact("model.int8.onnx") && has_artifact("tokens.txt"),
                    "{id}"
                );
            }
            ModelFamily::Qwen3Asr => {
                for suffix in [
                    "conv_frontend.onnx",
                    "tokenizer/vocab.json",
                    "tokenizer/merges.txt",
                ] {
                    assert!(has_artifact(suffix), "{id} {suffix}");
                }
            }
            ModelFamily::Cleanup => assert_eq!(artifacts, ["model-Q8_0.gguf"], "{id}"),
            ModelFamily::CudaPack => {}
        }
    }
}

#[test]
fn cleanup_models_start_with_the_default() {
    assert_eq!(CLEANUP_MODEL_IDS[0], DEFAULT_CLEANUP_MODEL_ID);
}

#[test]
fn lookup_accepts_typed_and_text_ids_and_rejects_unknown_models() {
    let registry = ModelRegistry::builtin();
    let id = ModelId::new("sherpa-onnx-whisper-turbo").expect("built-in identifier is valid");

    assert_eq!(registry.get(&id).map(|model| &model.id), Some(&id));
    assert_eq!(
        registry
            .get("sherpa-onnx-whisper-turbo")
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

#[test]
fn every_speech_model_is_measured() {
    for model in ModelRegistry::builtin().models() {
        if model.family() != ModelFamily::Cleanup {
            assert!(model.profile.benchmark().is_some(), "{}", model.id);
        }
    }
}
