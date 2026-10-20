use std::path::PathBuf;

use whispernow_core::{ResolvedModel, asr::AsrEngine};
use whispernow_domain::ModelRegistry;
use whispernow_platform::SherpaAsrEngine;

#[test]
#[ignore = "manual native model test; requires the official extracted Parakeet v2 FP16 archive and fixture (see docs/development.md)"]
fn manual_parakeet_fixture_matches_documented_text() {
    const EXPECTED: &str = "Well, I don't wish to see it any more, observed Phebe, turning away her eyes. It is certainly very like the old portrait.";

    let root =
        PathBuf::from(std::env::var_os("WHISPERNOW_ASR_MODEL_ROOT").expect(
            "WHISPERNOW_ASR_MODEL_ROOT must name the extracted Parakeet v2 FP16 archive root",
        ));
    let definition = ModelRegistry::builtin()
        .get("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16")
        .expect("Parakeet v2 FP16")
        .clone();
    assert_eq!(
        root.file_name().and_then(|name| name.to_str()),
        Some(definition.id.as_str()),
        "WHISPERNOW_ASR_MODEL_ROOT must point at the archive root, not its parent"
    );
    // The official archive includes this fixture, and the pinned model's documentation records
    // the transcript asserted below.
    let fixture = root.join("test_wavs/0.wav");
    let wave = sherpa_onnx::Wave::read(fixture.to_str().expect("UTF-8 fixture path"))
        .expect("readable fixture");
    assert_eq!(
        wave.sample_rate(),
        whispernow_core::audio::TARGET_SAMPLE_RATE.cast_signed()
    );

    let mut engine = SherpaAsrEngine::default();
    engine
        .load(&ResolvedModel { definition, root })
        .expect("load Parakeet");
    let transcript = engine
        .transcribe(wave.samples())
        .expect("transcribe fixture");
    assert_eq!(transcript.trim(), EXPECTED);
}
