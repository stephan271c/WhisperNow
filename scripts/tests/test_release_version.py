import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "release_version.py"
SPEC = importlib.util.spec_from_file_location("release_version", SCRIPT)
release_version = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(release_version)


class ReleaseVersionTests(unittest.TestCase):
    def test_stable_tag_strips_one_leading_v(self):
        self.assertEqual(release_version.parse_stable_tag("v0.2.0"), "0.2.0")

    def test_invalid_tags_and_non_stable_versions_are_rejected(self):
        for tag in (
            "0.2.0",
            "vv0.2.0",
            "v01.2.3",
            "v0.2",
            "v0.2.0-rc.1",
            "v0.2.0+rebuilt",
        ):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                release_version.parse_stable_tag(tag)

    def test_mismatches_across_every_version_surface_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "Cargo.toml"
            manifest.write_text(
                '[workspace]\nmembers = []\n[workspace.package]\nversion = "0.2.0"\n',
                encoding="utf-8",
            )
            tauri = root / "tauri.conf.json"
            tauri.write_text(json.dumps({"version": "0.2.1"}), encoding="utf-8")
            updater = root / "latest.json"
            updater.write_text(json.dumps({"version": "0.2.2"}), encoding="utf-8")

            errors = release_version.validate(
                manifest=manifest,
                tauri_configs=[tauri],
                tag="v0.2.3",
                metadata_versions={"whispernow": "0.2.4"},
                updater_metadata=[updater],
                artifacts=["WhisperNow_0.2.5_amd64.deb"],
            )

        self.assertEqual(len(errors), 5)
        self.assertTrue(any("Cargo metadata" in error for error in errors))
        self.assertTrue(any("release tag" in error for error in errors))
        self.assertTrue(any("Tauri version" in error for error in errors))
        self.assertTrue(any("updater version" in error for error in errors))
        self.assertTrue(any("artifact version" in error for error in errors))

    def test_only_tauri_may_inherit_the_workspace_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "Cargo.toml"
            manifest.write_text(
                '[workspace]\nmembers = []\n[workspace.package]\nversion = "0.2.0"\n',
                encoding="utf-8",
            )
            tauri = root / "tauri.conf.json"
            tauri.write_text("{}", encoding="utf-8")
            updater = root / "latest.json"
            for document in ({"version": "0.2.0"}, {}):
                with self.subTest(updater=document):
                    updater.write_text(json.dumps(document), encoding="utf-8")
                    errors = release_version.validate(
                        manifest=manifest,
                        tauri_configs=[tauri],
                        tag="v0.2.0",
                        metadata_versions={"desktop": "0.2.0", "core": "0.2.0"},
                        updater_metadata=[updater],
                        artifacts=[
                            "WhisperNow_0.2.0_amd64.deb",
                            "WhisperNow-0.2.0-1.x86_64.rpm",
                            "WhisperNow_0.2.0_amd64.AppImage.sig",
                        ],
                    )
                    if document:
                        self.assertEqual(errors, [])
                    else:
                        self.assertEqual(len(errors), 1)
                        self.assertIn("updater version is None", errors[0])

    def test_missing_or_ambiguous_artifact_version_is_rejected(self):
        errors = release_version.validate_artifacts(
            ["WhisperNow.AppImage", "WhisperNow-0.2.0-from-0.1.0.AppImage"], "0.2.0"
        )
        self.assertEqual(len(errors), 2)


if __name__ == "__main__":
    unittest.main()
