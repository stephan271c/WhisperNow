use std::collections::BTreeSet;

use whispernow_domain::{
    CLEANUP_MODEL_IDS, DEFAULT_CLEANUP_MODEL_ID, InvalidModelId, ModelBenchmark, ModelFamily,
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
        if model.family() == ModelFamily::Cleanup {
            assert!(source.url.starts_with("https://"), "{id}");
        } else {
            assert_eq!(
                source.url,
                format!("{RELEASE_BASE}/{}.tar.bz2", source.archive_root())
            );
            assert!(source.archive_root().eq_ignore_ascii_case(id), "{id}");
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
        }
    }
}

#[test]
fn cleanup_models_start_with_the_default_and_download_from_a_pinned_commit() {
    assert_eq!(CLEANUP_MODEL_IDS[0], DEFAULT_CLEANUP_MODEL_ID);
    let registry = ModelRegistry::builtin();
    for id in CLEANUP_MODEL_IDS {
        let model = registry.get(id).expect("registered cleanup model");
        let repository = id
            .strip_suffix("-v1-q8_0")
            .unwrap_or_else(|| panic!("{id} names its revision"));
        let prefix = format!("https://huggingface.co/category271/{repository}/resolve/");
        let commit = model
            .source
            .url
            .strip_prefix(&prefix)
            .and_then(|rest| rest.strip_suffix("/model-Q8_0.gguf"))
            .unwrap_or_else(|| panic!("{id}: {}", model.source.url));
        assert!(
            commit.len() == 40 && commit.bytes().all(|b| b.is_ascii_hexdigit()),
            "{id} downloads from {commit}, not a commit"
        );
    }
}

#[test]
fn lookup_finds_registered_models_and_rejects_unknown_ones() {
    let registry = ModelRegistry::builtin();

    assert_eq!(
        registry
            .get("sherpa-onnx-whisper-turbo")
            .map(|model| model.id.as_str()),
        Some("sherpa-onnx-whisper-turbo")
    );
    assert!(registry.get("unknown-model").is_none());
}

#[test]
fn identifiers_reject_filesystem_components() {
    for (invalid, error) in [
        ("", InvalidModelId::Empty),
        (".", InvalidModelId::PathComponent),
        ("..", InvalidModelId::PathComponent),
        ("../model", InvalidModelId::Character),
        ("nested/model", InvalidModelId::Character),
        (r"nested\model", InvalidModelId::Character),
        ("C:model", InvalidModelId::Character),
        ("UPPERCASE", InvalidModelId::Character),
        ("model name", InvalidModelId::Character),
    ] {
        assert_eq!(ModelId::new(invalid).err(), Some(error), "ID {invalid:?}");
    }
}

#[test]
fn every_model_is_measured_for_its_family() {
    for model in ModelRegistry::builtin().models() {
        let measured = match model.benchmark {
            Some(ModelBenchmark::Speech(_)) => model.family().is_speech(),
            Some(ModelBenchmark::Cleanup(_)) => model.family() == ModelFamily::Cleanup,
            None => false,
        };
        assert!(measured, "{}", model.id);
    }
}
