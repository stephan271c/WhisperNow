from whispernow_training.datagen.references import review
from whispernow_training.datagen.synthetic import second_sample


def test_capitalized_literal_flags_command_lines_and_file_names():
    assert review.capitalized_literal("1. Docker compose up -d.\n2. Check it.") == "Docker compose up -d."
    assert review.capitalized_literal("Source .venv/bin/activate and run it.") == "Source .venv/bin/activate and run it."
    assert review.capitalized_literal("Cargo.toml needs the new crate.") == "Cargo.toml needs the new crate."


def test_capitalized_literal_ignores_lowercase_commands_and_plain_prose():
    assert review.capitalized_literal("docker compose up -d") is None
    assert review.capitalized_literal("Docker is a container tool.") is None
    assert review.capitalized_literal("Git is fine.\ngit push origin main") is None


def test_two_samples_disagree_only_on_content_words_the_speaker_said():
    transcript = "we can meet on wednesday or uh sunday whichever suits you"
    assert second_sample.two_sample_disagreement(transcript, "We can meet on Wednesday or Sunday.", "We can meet on Sunday.") == ["wednesday"]
    assert second_sample.two_sample_disagreement(transcript, "We can meet on Wednesday or Sunday.", "We can meet Wednesday or Sunday!") == []
