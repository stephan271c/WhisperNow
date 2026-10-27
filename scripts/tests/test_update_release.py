import base64
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "update_release.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("update_release", SCRIPT)
update_release = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = update_release
SPEC.loader.exec_module(update_release)

REPOSITORY = "stephan271c/WhisperNow"
TAG = "v0.2.0"
VERSION = "0.2.0"
PUB_DATE = "2026-08-11T12:34:56Z"


def sign(path: Path) -> None:
    """Write a Tauri-shaped envelope whose trusted comment binds the payload hash."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    envelope = "\n".join(
        (
            "untrusted comment: signature from disposable test key",
            base64.b64encode(b"S" * 74).decode("ascii"),
            f"trusted comment: sha256:{digest}",
            base64.b64encode(b"G" * 64).decode("ascii"),
        )
    )
    Path(f"{path}.sig").write_text(
        base64.b64encode(envelope.encode("utf-8")).decode("ascii"), encoding="utf-8"
    )


def verify(payload: Path, signature: Path) -> bool:
    envelope = base64.b64decode(signature.read_text(encoding="utf-8"))
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    return f"sha256:{digest}".encode() in envelope


PACKAGES = {
    "windows-x86_64": ("setup.exe",),
    "linux-x86_64": ("app.deb", "app.rpm", "app.AppImage"),
    "macos-aarch64": ("WhisperNow.app.tar.gz", "WhisperNow.dmg"),
    "macos-x86_64": ("WhisperNow.app.tar.gz", "WhisperNow.dmg"),
}


class UpdateReleaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def bundle(self, platform: str) -> Path:
        root = self.root / "bundle" / platform
        root.mkdir(parents=True)
        for name in PACKAGES[platform]:
            path = root / name
            path.write_bytes(f"{platform}/{name}".encode())
            if path.suffix != ".dmg":
                sign(path)
        return root

    def stage(self, platform: str) -> Path:
        return update_release.stage(
            root=self.bundle(platform),
            output=self.root / "staged" / platform,
            platform=platform,
            version=VERSION,
            signature_verifier=verify,
        )

    def generate(self, manifests: list[Path]) -> Path:
        return update_release.generate(
            manifests=manifests,
            output=self.root / "publish",
            repository=REPOSITORY,
            tag=TAG,
            version=VERSION,
            pub_date=PUB_DATE,
            notes="Release notes",
            signature_verifier=verify,
        )

    def test_stage_copies_every_linux_format_under_canonical_names(self):
        manifest = json.loads(self.stage("linux-x86_64").read_text(encoding="utf-8"))
        self.assertEqual(
            {entry["name"] for entry in manifest["assets"]},
            {f"WhisperNow_{VERSION}_linux_x86_64.{suffix}" for suffix in ("deb", "rpm", "AppImage")},
        )
        self.assertEqual(
            {entry["target"]: entry["bundle_type"] for entry in manifest["updates"]},
            {
                "linux-x86_64-deb": "deb",
                "linux-x86_64-rpm": "rpm",
                "linux-x86_64-appimage": "appimage",
            },
        )

    def test_stage_rejects_a_signature_for_different_bytes(self):
        root = self.bundle("windows-x86_64")
        (root / "setup.exe").write_bytes(b"swapped after signing")
        with self.assertRaisesRegex(ValueError, "signature verification failed"):
            update_release.stage(
                root=root,
                output=self.root / "staged",
                platform="windows-x86_64",
                version=VERSION,
                signature_verifier=verify,
            )

    def test_stage_rejects_a_duplicate_bundle(self):
        root = self.bundle("linux-x86_64")
        (root / "second.deb").write_bytes(b"deb")
        with self.assertRaisesRegex(ValueError, "exactly one Linux deb"):
            update_release.select_platform(root, "linux-x86_64")

    def test_latest_json_lists_every_target_with_immutable_urls_and_signatures(self):
        latest_path = self.generate([self.stage(platform) for platform in PACKAGES])
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        self.assertEqual(latest["version"], VERSION)
        self.assertEqual(
            set(latest["platforms"]),
            {
                "windows-x86_64",
                "linux-x86_64-deb",
                "linux-x86_64-rpm",
                "linux-x86_64-appimage",
                "darwin-aarch64",
                "darwin-x86_64",
            },
        )
        entry = latest["platforms"]["darwin-aarch64"]
        self.assertEqual(
            entry["url"],
            f"https://github.com/{REPOSITORY}/releases/download/{TAG}/"
            f"WhisperNow_{VERSION}_darwin_aarch64.app.tar.gz",
        )
        signature = self.root / "bundle" / "macos-aarch64" / "WhisperNow.app.tar.gz.sig"
        self.assertEqual(entry["signature"], signature.read_text(encoding="utf-8").strip())

    def test_generate_publishes_only_the_packages_and_latest_json(self):
        latest_path = self.generate([self.stage(platform) for platform in PACKAGES])
        self.assertEqual(
            sorted(path.name for path in latest_path.parent.iterdir()),
            [
                f"WhisperNow_{VERSION}_darwin_aarch64.app.tar.gz",
                f"WhisperNow_{VERSION}_darwin_aarch64.dmg",
                f"WhisperNow_{VERSION}_darwin_x86_64.app.tar.gz",
                f"WhisperNow_{VERSION}_darwin_x86_64.dmg",
                f"WhisperNow_{VERSION}_linux_x86_64.AppImage",
                f"WhisperNow_{VERSION}_linux_x86_64.deb",
                f"WhisperNow_{VERSION}_linux_x86_64.rpm",
                f"WhisperNow_{VERSION}_windows_x86_64_nsis.exe",
                "latest.json",
            ],
        )

    def test_generate_rejects_a_missing_platform(self):
        manifests = [self.stage(platform) for platform in PACKAGES if platform != "macos-x86_64"]
        with self.assertRaisesRegex(ValueError, "platforms are incomplete"):
            self.generate(manifests)
        self.assertFalse((self.root / "publish" / "latest.json").exists())

    def test_generate_rejects_packages_changed_after_staging(self):
        manifests = [self.stage(platform) for platform in PACKAGES]
        package = json.loads(manifests[0].read_text(encoding="utf-8"))["assets"][0]["name"]
        (manifests[0].parent / package).write_bytes(b"swapped")
        with self.assertRaisesRegex(ValueError, "bytes do not match"):
            self.generate(manifests)

    def test_generate_rejects_signatures_changed_after_staging(self):
        manifests = [self.stage(platform) for platform in PACKAGES]
        staged = manifests[0].parent
        next(path for path in staged.iterdir() if path.suffix == ".sig").write_text(
            "tampered", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "signature verification failed"):
            self.generate(manifests)

    def test_stable_identity_rejects_prerelease_and_version_mismatch(self):
        for tag, version in (("v0.2.0-rc.1", "0.2.0-rc.1"), ("v0.2.0", "0.2.1")):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                update_release.validate_identity(REPOSITORY, tag, version)


if __name__ == "__main__":
    unittest.main()
