import ast
import base64
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import yaml

SCRIPTS = Path(__file__).parents[1]
sys.path.insert(0, str(SCRIPTS))
SIGNING_SECRETS = {
    "TAURI_SIGNING_PRIVATE_KEY",
    "TAURI_SIGNING_PRIVATE_KEY_PASSWORD",
    "MACOS_CERTIFICATE",
    "MACOS_CERTIFICATE_PASSWORD",
    "WINDOWS_CERTIFICATE",
    "WINDOWS_CERTIFICATE_PASSWORD",
}


def action_expression(value: str) -> str:
    value = value.strip()
    if value.startswith("${{") and value.endswith("}}"):
        value = value[3:-2].strip()
    while value.startswith("(") and value.endswith(")"):
        value = value[1:-1].strip()
    return value


def signing_secret_references(value) -> set[str]:
    if isinstance(value, dict):
        return set().union(*(signing_secret_references(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(signing_secret_references(item) for item in value))
    if not isinstance(value, str):
        return set()
    return {
        (attribute or subscript).upper()
        for expression in re.findall(r"\$\{\{(.*?)\}\}", value, re.DOTALL)
        for attribute, subscript in re.findall(
            r"\bsecrets(?:\.([A-Za-z_][A-Za-z0-9_]*)|\[\s*'([^']+)'\s*\])",
            expression,
            re.IGNORECASE,
        )
    } & SIGNING_SECRETS


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


model_resources = load("stage_model_resources")
windows_imports = load("check_windows_imports")
macos_signature = load("check_macos_signature")


class NativeStagingTests(unittest.TestCase):
    def fixture_archive(self, root: Path, platform: str) -> Path:
        archive_name = "fixture-static-lib.tar.bz2"
        source_root = root / archive_name.removesuffix(".tar.bz2")
        lib_dir = source_root / "lib"
        lib_dir.mkdir(parents=True)
        (lib_dir / "libsherpa-onnx-c-api.a").write_bytes(b"fixture sherpa")
        (lib_dir / "libonnxruntime.a").write_bytes(b"fixture onnxruntime")
        archive = root / archive_name
        with tarfile.open(archive, "w:bz2") as bundle:
            bundle.add(source_root, arcname=source_root.name)
        return archive

    def fixture_allowlist(self, root: Path, archive: Path, platform: str) -> Path:
        source = Path(__file__).parents[2] / "release" / "native-allowlist.json"
        document = json.loads(source.read_text(encoding="utf-8"))
        document["source_archives"] = {
            platform: {
                "name": archive.name,
                "bytes": archive.stat().st_size,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "source": f"https://example.invalid/{archive.name}",
            }
        }
        source_root = root / archive.name.removesuffix(".tar.bz2") / "lib"
        paths = sorted(source_root.iterdir()) if source_root.is_dir() else [archive]
        document["libraries"] = {
            platform: [
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in paths
            ]
        }
        path = root / "fixture-native-allowlist.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def run_stage(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "stage_native_dependencies.py"), "--platform", "linux-x86_64", *arguments],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_static_archive_is_verified_extracted_and_licenses_zipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self.fixture_archive(root, "linux-x86_64")
            allowlist = self.fixture_allowlist(root, archive, "linux-x86_64")
            output = root / "output"
            process = self.run_stage("--archive", str(archive), "--output", str(output), "--allowlist", str(allowlist))
            self.assertEqual(process.returncode, 0, process.stderr)
            lib_dir = Path(process.stdout.strip())
            self.assertEqual(
                {path.name for path in lib_dir.iterdir()},
                {"libsherpa-onnx-c-api.a", "libonnxruntime.a"},
            )
            document = json.loads(allowlist.read_text(encoding="utf-8"))
            with zipfile.ZipFile(output / "native-compliance.zip") as bundle:
                names = set(bundle.namelist())
            self.assertEqual(
                {name for name in names if name.startswith("licenses/")},
                {f"licenses/{name}" for name in document["license_evidence"]},
            )

    def test_production_allowlist_rejects_fake_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self.fixture_archive(root, "linux-x86_64")
            output = root / "output"
            process = self.run_stage("--archive", str(archive), "--output", str(output))
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("does not match the allowlist", process.stderr)

    def test_valid_archive_with_matching_size_and_wrong_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self.fixture_archive(root, "linux-x86_64")
            allowlist = self.fixture_allowlist(root, archive, "linux-x86_64")
            document = json.loads(allowlist.read_text(encoding="utf-8"))
            document["source_archives"]["linux-x86_64"]["sha256"] = "0" * 64
            allowlist.write_text(json.dumps(document), encoding="utf-8")
            output = root / "output"
            process = self.run_stage("--archive", str(archive), "--output", str(output), "--allowlist", str(allowlist))
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("archive size or SHA-256 does not match the allowlist", process.stderr)
            self.assertFalse((output / archive.name.removesuffix(".tar.bz2")).exists())

    def test_extractor_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "fixture-static-lib.tar.bz2"
            escaped = root / "escaped"
            escaped.write_bytes(b"escape")
            with tarfile.open(archive, "w:bz2") as bundle:
                bundle.add(escaped, arcname="../escaped")
            allowlist = self.fixture_allowlist(root, archive, "linux-x86_64")
            output = root / "output"
            process = self.run_stage("--archive", str(archive), "--output", str(output), "--allowlist", str(allowlist))
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("unsafe entry", process.stderr)


class LlamaLibraryTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path]:
        source = root / "llama-b11259"
        source.mkdir()
        (source / "libllama.so.0.5.0").write_bytes(b"fixture llama")
        (source / "libggml-cpu-x64.so").write_bytes(b"fixture cpu")
        (source / "llama-server").write_bytes(b"not shipped")
        (source / "libllama.so.0").symlink_to("libllama.so.0.5.0")
        archive = root / "llama-b11259-bin-ubuntu-vulkan-x64.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(source, arcname=source.name)
        document = json.loads(
            (Path(__file__).parents[2] / "release" / "native-allowlist.json").read_text(
                encoding="utf-8"
            )
        )
        pin = lambda data: {  # noqa: E731
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        document["shared_libraries"]["llama"] = {
            "component": "llama.cpp",
            "archives": {
                "linux-x86_64": {
                    "name": archive.name,
                    **pin(archive.read_bytes()),
                    "source": f"https://example.invalid/{archive.name}",
                }
            },
            "libraries": {
                "linux-x86_64": [
                    {"name": "libggml-cpu-x64.so", "member": "llama-b11259/libggml-cpu-x64.so", **pin(b"fixture cpu")},
                    {"name": "libllama.so.0", "member": "llama-b11259/libllama.so.0.5.0", **pin(b"fixture llama")},
                ]
            },
        }
        allowlist = root / "fixture-native-allowlist.json"
        allowlist.write_text(json.dumps(document), encoding="utf-8")
        return archive, allowlist

    def run_stage(self, *arguments: str, platform: str = "linux-x86_64") -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "stage_llama.py"), "--platform", platform, *arguments],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_pinned_libraries_are_extracted_under_their_sonames_and_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, allowlist = self.fixture(root)
            output = root / "llama"
            staged = self.run_stage("--archive", str(archive), "--output", str(output), "--allowlist", str(allowlist))
            self.assertEqual(staged.returncode, 0, staged.stderr)
            self.assertEqual(Path(staged.stdout.strip()), output.resolve())
            self.assertEqual(
                {path.name: path.read_bytes() for path in output.iterdir()},
                {"libllama.so.0": b"fixture llama", "libggml-cpu-x64.so": b"fixture cpu"},
            )
            verified = self.run_stage("--verify", str(output), "--allowlist", str(allowlist))
            self.assertEqual(verified.returncode, 0, verified.stderr)
            (output / "libggml-rpc.so").write_bytes(b"extra")
            extra = self.run_stage("--verify", str(output), "--allowlist", str(allowlist))
            self.assertIn("not the exact pinned set", extra.stderr)
            (output / "libggml-rpc.so").unlink()
            (output / "libllama.so.0").write_bytes(b"Fixture llama")
            tampered = self.run_stage("--verify", str(output), "--allowlist", str(allowlist))
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("does not match the allowlist", tampered.stderr)

    def test_on_macos_verify_checks_the_names_because_signing_rewrites_the_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, allowlist = self.fixture(root)
            document = json.loads(allowlist.read_text(encoding="utf-8"))
            llama = document["shared_libraries"]["llama"]
            for key in ("archives", "libraries"):
                llama[key]["macos-aarch64"] = llama[key].pop("linux-x86_64")
            allowlist.write_text(json.dumps(document), encoding="utf-8")
            frameworks = root / "Frameworks"
            common = ("--allowlist", str(allowlist))
            staged = self.run_stage(
                "--archive", str(archive), "--output", str(frameworks), *common, platform="macos-aarch64"
            )
            self.assertEqual(staged.returncode, 0, staged.stderr)
            (frameworks / "libllama.so.0").write_bytes(b"fixture llama, signed")
            verified = self.run_stage("--verify", str(frameworks), *common, platform="macos-aarch64")
            self.assertEqual(verified.returncode, 0, verified.stderr)
            (frameworks / "libggml-cpu-x64.so").unlink()
            missing = self.run_stage("--verify", str(frameworks), *common, platform="macos-aarch64")
            self.assertIn("not the exact pinned set", missing.stderr)

    def test_an_archive_that_is_not_the_pinned_one_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, allowlist = self.fixture(root)
            with archive.open("ab") as file:
                file.write(b"\0")
            staged = self.run_stage(
                "--archive", str(archive), "--output", str(root / "llama"), "--allowlist", str(allowlist)
            )
            self.assertNotEqual(staged.returncode, 0)
            self.assertIn("does not match the allowlist", staged.stderr)
            self.assertFalse((root / "llama").exists())

    def test_the_visual_cpp_runtime_is_copied_in_and_is_part_of_the_exact_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, allowlist = self.fixture(root)
            document = json.loads(allowlist.read_text(encoding="utf-8"))
            document["shared_libraries"]["llama"]["msvc_runtime"] = {
                "linux-x86_64": ["msvcp140.dll", "vcruntime140.dll"]
            }
            allowlist.write_text(json.dumps(document), encoding="utf-8")
            redist = root / "Microsoft.VC145.CRT"
            redist.mkdir()
            (redist / "MSVCP140.dll").write_bytes(b"msvcp")
            (redist / "vcruntime140.dll").write_bytes(b"vcruntime")
            (redist / "concrt140.dll").write_bytes(b"not needed")
            output = root / "llama"
            common = ("--archive", str(archive), "--output", str(output), "--allowlist", str(allowlist))
            without = self.run_stage(*common)
            self.assertIn("needs --msvc-runtime", without.stderr)
            staged = self.run_stage(*common, "--msvc-runtime", str(redist))
            self.assertEqual(staged.returncode, 0, staged.stderr)
            self.assertEqual(
                {path.name: path.read_bytes() for path in output.iterdir()},
                {
                    "libllama.so.0": b"fixture llama",
                    "libggml-cpu-x64.so": b"fixture cpu",
                    "msvcp140.dll": b"msvcp",
                    "vcruntime140.dll": b"vcruntime",
                },
            )
            verified = self.run_stage("--verify", str(output), "--allowlist", str(allowlist))
            self.assertEqual(verified.returncode, 0, verified.stderr)
            (output / "vcruntime140.dll").unlink()
            missing = self.run_stage("--verify", str(output), "--allowlist", str(allowlist))
            self.assertIn("not the exact pinned set", missing.stderr)
            (redist / "vcruntime140.dll").unlink()
            incomplete = self.run_stage(*common, "--msvc-runtime", str(redist))
            self.assertIn("lacks vcruntime140.dll", incomplete.stderr)

    def test_production_pinned_libraries_and_license_evidence_exist(self):
        document = json.loads(
            (Path(__file__).parents[2] / "release" / "native-allowlist.json").read_text(encoding="utf-8")
        )
        llama = document["shared_libraries"]["llama"]
        self.assertTrue(llama["archives"])
        self.assertEqual(set(llama["libraries"]), set(llama["archives"]))
        for platform, libraries in llama["libraries"].items():
            self.assertTrue(libraries, platform)
        for component in document["components"].values():
            for name in component["license_files"]:
                self.assertIn(name, document["license_evidence"])


def portable_executable(imports=(), delay_imports=()) -> bytes:
    """A minimal PE32+ image whose one section holds its import and delay-import tables."""
    section_rva, section_raw = 0x1000, 0x200
    body = bytearray()
    name_rvas = {}
    for name in (*imports, *delay_imports):
        name_rvas[name] = section_rva + len(body)
        body += name.encode("ascii") + b"\0"
    body += bytes(-len(body) % 4)
    import_rva = section_rva + len(body)
    for name in imports:
        body += struct.pack("<5I", 0, 0, 0, name_rvas[name], 0)
    body += bytes(20)
    delay_rva = section_rva + len(body)
    for name in delay_imports:
        body += struct.pack("<8I", 1, name_rvas[name], 0, 0, 0, 0, 0, 0)
    body += bytes(32)
    optional = bytearray(112 + 16 * 8)
    struct.pack_into("<H", optional, 0, 0x20B)
    struct.pack_into("<I", optional, 108, 16)
    struct.pack_into("<II", optional, 112 + 8 * 1, import_rva, 20 * (len(imports) + 1))
    if delay_imports:
        struct.pack_into("<II", optional, 112 + 8 * 13, delay_rva, 32 * (len(delay_imports) + 1))
    coff = struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, len(optional), 0x22)
    section = struct.pack(
        "<8sIIIIIIHHI", b".idata", len(body), section_rva, len(body), section_raw, 0, 0, 0, 0, 0x40000040
    )
    header = b"MZ" + bytes(0x3A) + struct.pack("<I", 0x40) + b"PE\0\0" + coff + bytes(optional) + section
    return header.ljust(section_raw, b"\0") + bytes(body)


class WindowsImportTests(unittest.TestCase):
    def test_import_and_delay_import_tables_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "app.exe"
            path.write_bytes(portable_executable(["KERNEL32.dll", "VCRUNTIME140.dll"], ["vulkan-1.dll"]))
            self.assertEqual(
                windows_imports.imported_dlls(path), ["KERNEL32.dll", "VCRUNTIME140.dll", "vulkan-1.dll"]
            )
            path.write_bytes(b"not a portable executable")
            with self.assertRaisesRegex(ValueError, "not a PE file"):
                windows_imports.imported_dlls(path)

    def test_imports_resolve_beside_the_importer_and_never_from_an_installed_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "llama").mkdir()
            (root / "$PLUGINSDIR").mkdir()
            files = {
                "app.exe": ["KERNEL32.dll", "api-ms-win-crt-runtime-l1-1-0.dll", "VCRUNTIME140.dll"],
                "helper.dll": ["KERNEL32.dll"],
                "llama/llama.dll": ["ggml.dll", "MSVCP140.dll", "helper.dll"],
                "llama/ggml.dll": ["vulkan-1.dll", "msvcp140.dll"],
                "llama/msvcp140.dll": ["KERNEL32.dll"],
                "$PLUGINSDIR/nsDialogs.dll": ["missing.dll"],
            }
            for name, imports in files.items():
                (root / name).write_bytes(portable_executable(imports))
            checked, problems = windows_imports.unresolved_imports(
                root, {"kernel32.dll", "vcruntime140.dll"}
            )
            self.assertEqual(checked, 5)
            self.assertEqual(
                problems,
                [
                    "app.exe imports VCRUNTIME140.dll, which is not beside it",
                    "llama/llama.dll imports helper.dll, which is not beside it",
                ],
            )


@unittest.skipUnless(shutil.which("openssl"), "needs openssl")
class MacSignatureTests(unittest.TestCase):
    def make_certificate(self, root: Path) -> Path:
        (root / "scripts").mkdir()
        (root / "release").mkdir()
        script = root / "scripts" / "make_macos_signing_certificate.sh"
        shutil.copy2(SCRIPTS / "make_macos_signing_certificate.sh", script)
        subprocess.run([str(script), str(root / "private")], check=True, capture_output=True)
        return script

    def test_the_certificate_script_pins_a_code_signing_certificate_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = self.make_certificate(root)
            private = root / "private"
            pinned = macos_signature.pinned_certificate(root / "release" / "macos-signing-certificate.pem")
            text = subprocess.run(
                ["openssl", "x509", "-inform", "DER", "-noout", "-ext", "extendedKeyUsage"],
                input=pinned, check=True, capture_output=True,
            ).stdout.decode()
            self.assertIn("Code Signing", text)
            self.assertEqual(
                base64.b64decode((private / "signing.p12.base64").read_text()),
                (private / "signing.p12").read_bytes(),
            )
            bundled = subprocess.run(
                ["openssl", "pkcs12", "-in", str(private / "signing.p12"), "-nokeys", "-clcerts",
                 "-passin", f"file:{private / 'password.txt'}"],
                check=True, capture_output=True,
            ).stdout.decode()
            self.assertEqual(macos_signature.ssl.PEM_cert_to_DER_cert(bundled[bundled.index("-----BEGIN"):]), pinned)
            self.assertEqual((private / "signing.p12").stat().st_mode & 0o077, 0)
            again = subprocess.run([str(script), str(root / "other")], capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("resets every user's macOS permissions", again.stderr)

    def test_the_designated_requirement_must_name_the_pinned_certificate(self):
        certificate = b"certificate bytes"
        digest = hashlib.sha1(certificate).hexdigest()
        executable = "Executable=/Applications/WhisperNow.app/Contents/MacOS/whispernow-desktop\n"
        for requirement, pins in [
            (f'identifier "com.whispernow.desktop" and certificate root = H"{digest}"', True),
            (f'identifier "com.whispernow.desktop" and certificate leaf = H"{digest.upper()}"', True),
            (f'identifier "com.whispernow.desktop" and certificate root = H"{"0" * 40}"', False),
            (f'cdhash H"{digest}"', False),
        ]:
            display = f"{executable}designated => {requirement}\n"
            self.assertEqual(macos_signature.requirement_pins(display, certificate), pins, requirement)

    def test_the_bundle_its_executables_and_its_frameworks_need_the_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory) / "WhisperNow.app"
            for name in ["MacOS/whispernow-desktop", "Frameworks/libllama.0.dylib", "Frameworks/libggml.0.dylib"]:
                (app / "Contents" / name).parent.mkdir(parents=True, exist_ok=True)
                (app / "Contents" / name).write_bytes(b"code")
            (app / "Contents" / "Resources").mkdir()
            (app / "Contents" / "Resources" / "icon.icns").write_bytes(b"icon")
            certificate = b"pinned"
            signers = {"libggml.0.dylib": None}
            display = f'designated => identifier "x" and certificate root = H"{hashlib.sha1(certificate).hexdigest()}"'
            with patch.object(macos_signature, "signer", lambda path: signers.get(path.name, certificate)), \
                    patch.object(macos_signature.subprocess, "run") as run:
                run.return_value.stdout, run.return_value.stderr = "", display
                self.assertEqual(
                    macos_signature.problems(app, certificate),
                    ["WhisperNow.app/Contents/Frameworks/libggml.0.dylib is not signed with the pinned certificate"],
                )
            self.assertEqual(
                [path.relative_to(app).as_posix() for path in macos_signature.signed_code(app)],
                [".", "Contents/MacOS/whispernow-desktop", "Contents/Frameworks/libggml.0.dylib",
                 "Contents/Frameworks/libllama.0.dylib"],
            )


class ReleaseWorkflowTests(unittest.TestCase):
    def setUp(self):
        workflows = Path(__file__).parents[2] / ".github" / "workflows"
        # Actions' `on` is a string; PyYAML's YAML 1.1 boolean resolver coerces it.
        self.workflow = yaml.load(
            (workflows / "release.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader
        )

    def assert_protected_push_jobs(self, workflow):
        jobs = workflow["jobs"]
        for job_id in ("bundle", "macos-gatekeeper", "release"):
            condition = jobs[job_id].get("if")
            self.assertIsInstance(condition, str, job_id)
            self.assertRegex(
                action_expression(condition), r"\Agithub\.event_name\s*==\s*'push'\Z",
                f"{job_id} must only run for pushes",
            )
        for job_id, required in (
            ("bundle", {"supply-chain"}),
            ("release", {"bundle", "macos-gatekeeper"}),
        ):
            job = jobs[job_id]
            environment = job.get("environment")
            if isinstance(environment, dict):
                environment = environment.get("name")
            self.assertEqual(environment, "release", job_id)
            needs = job.get("needs", [])
            self.assertEqual({needs} if isinstance(needs, str) else set(needs), required, job_id)

    def assert_step_scoped_signing_secrets(self, workflow):
        workflow_fields = {name: value for name, value in workflow.items() if name != "jobs"}
        self.assertFalse(
            (set(workflow.get("env", {})) & SIGNING_SECRETS)
            | signing_secret_references(workflow_fields),
            "workflow environment must not receive signing secrets",
        )
        found = set()
        for job_id, job in workflow["jobs"].items():
            job_fields = {name: value for name, value in job.items() if name != "steps"}
            self.assertFalse(
                (set(job.get("env", {})) & SIGNING_SECRETS) | signing_secret_references(job_fields),
                f"{job_id} job environment must not receive signing secrets",
            )
            for step in job["steps"]:
                step_fields = {name: value for name, value in step.items() if name != "env"}
                self.assertFalse(
                    signing_secret_references(step_fields),
                    "signing secrets must be passed through step environments",
                )
                env = step.get("env", {})
                names = (set(env) & SIGNING_SECRETS) | signing_secret_references(env)
                if not names:
                    continue
                self.assertEqual(job_id, "bundle")
                for name in names:
                    self.assertIn(name, env)
                    self.assertRegex(
                        action_expression(env[name]),
                        re.compile(
                            rf"\Asecrets(?:\.{name}|\[\s*'{name}'\s*\])\Z", re.IGNORECASE
                        ),
                    )
                for prefix, platform in (
                    ("TAURI_", None), ("MACOS_", "macOS"), ("WINDOWS_", "Windows")
                ):
                    pair = {name for name in SIGNING_SECRETS if name.startswith(prefix)}
                    if names & pair:
                        self.assertEqual(names & pair, pair)
                        if platform:
                            self.assertRegex(
                                action_expression(step.get("if", "")),
                                rf"\Arunner\.os\s*==\s*'{platform}'\Z",
                            )
                found.update(names)
        self.assertEqual(found, SIGNING_SECRETS)

    def test_release_triggers_allow_only_tags_and_manual_dry_runs(self):
        self.assertEqual(set(self.workflow["on"]), {"push", "workflow_dispatch"})
        self.assertEqual(self.workflow["on"]["push"], {"tags": ["v*.*.*"]})

    def test_signing_and_publication_are_protected_push_jobs(self):
        self.assert_protected_push_jobs(self.workflow)

    def test_only_publication_can_write_repository_contents(self):
        self.assertEqual(self.workflow["permissions"], {"contents": "read"})
        for job_id, job in self.workflow["jobs"].items():
            self.assertEqual(
                job.get("permissions", self.workflow["permissions"]),
                {"contents": "write" if job_id == "release" else "read"},
                job_id,
            )

    def test_signing_secrets_are_step_scoped(self):
        self.assert_step_scoped_signing_secrets(self.workflow)

    def test_remote_actions_are_sha_pinned(self):
        references = [
            step["uses"] for job in self.workflow["jobs"].values()
            for step in job["steps"] if "uses" in step
        ]
        self.assertTrue(references)
        for reference in references:
            self.assertRegex(reference, r"\A[^@\s]+@[0-9a-fA-F]{40}\Z")

    def test_guards_reject_allowing_manual_dispatch_to_sign_or_publish(self):
        for job_id in ("bundle", "release"):
            with self.subTest(job=job_id):
                mutated = copy.deepcopy(self.workflow)
                mutated["jobs"][job_id]["if"] = (
                    "${{ github.event_name == 'push' || github.event_name == 'workflow_dispatch' }}"
                )
                with self.assertRaisesRegex(AssertionError, "must only run for pushes"):
                    self.assert_protected_push_jobs(mutated)

    def test_secrets_reject_promoting_a_signing_key_to_workflow_environment(self):
        for variable in ("TAURI_SIGNING_PRIVATE_KEY", "SHARED_SIGNING_KEY"):
            with self.subTest(variable=variable):
                mutated = copy.deepcopy(self.workflow)
                signing_step = next(
                    step for step in mutated["jobs"]["bundle"]["steps"]
                    if "TAURI_SIGNING_PRIVATE_KEY" in step.get("env", {})
                )
                mutated["env"][variable] = signing_step["env"].pop("TAURI_SIGNING_PRIVATE_KEY")
                with self.assertRaisesRegex(AssertionError, "workflow environment"):
                    self.assert_step_scoped_signing_secrets(mutated)

    def test_equivalent_expression_wrappers_and_yaml_formats_are_accepted(self):
        for flow_style in (False, True):
            for condition in (
                "${{ github.event_name == 'push' }}",
                "${{ (\n github.event_name=='push'\n) }}",
                "((github.event_name == 'push'))",
            ):
                with self.subTest(flow_style=flow_style, condition=condition):
                    equivalent = copy.deepcopy(self.workflow)
                    for job_id in ("bundle", "macos-gatekeeper", "release"):
                        equivalent["jobs"][job_id]["if"] = condition
                    equivalent["jobs"]["bundle"]["environment"] = {"name": "release"}
                    equivalent["jobs"]["bundle"]["needs"] = ["supply-chain"]
                    text = yaml.safe_dump(equivalent, default_flow_style=flow_style, indent=4)
                    parsed = yaml.load(text, Loader=yaml.BaseLoader)
                    self.assertEqual(parsed, equivalent)
                    self.assert_protected_push_jobs(parsed)
                    self.assert_step_scoped_signing_secrets(parsed)

    def test_macos_generated_config_uses_the_identity_without_hardened_runtime(self):
        build_step = next(
            step for step in self.workflow["jobs"]["bundle"]["steps"]
            if step.get("working-directory") == "apps/desktop"
            and re.fullmatch(
                r"runner\.os\s*==\s*'macOS'", action_expression(step.get("if", ""))
            )
        )
        run = build_step["run"]
        heredoc = re.search(r"(?m)^python3\s+-[^\n]*<<['\"]?(\w+)['\"]?\s*\n", run)
        self.assertIsNotNone(heredoc, "macOS config generator must be executable Python")
        code, remainder = run[heredoc.end():].split(f"\n{heredoc.group(1)}\n", 1)
        self.assertTrue(remainder)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            desktop = root / "apps" / "desktop"
            desktop.mkdir(parents=True)
            llama = root / "release" / "llama"
            llama.mkdir(parents=True)
            for name in ("libllama.0.dylib", "libggml.0.dylib", "not-a-framework.txt"):
                (llama / name).write_bytes(b"fixture")
            config = root / "signing-config.json"
            for program in (code, ast.unparse(ast.parse(code))):
                with self.subTest(program=program):
                    config.unlink(missing_ok=True)
                    subprocess.run(
                        [sys.executable, "-", str(config)], input=program,
                        cwd=desktop, text=True, check=True, capture_output=True,
                        env={
                            **os.environ,
                            "WHISPERNOW_UPDATER_PUBLIC_KEY": "fixture-updater-key",
                            "MACOS_SIGNING_IDENTITY": "fixture-certificate-fingerprint",
                        },
                    )
                    document = json.loads(config.read_text(encoding="utf-8"))
                    self.assertEqual(document["plugins"]["updater"]["pubkey"], "fixture-updater-key")
                    signing = document["bundle"]["macOS"]
                    self.assertEqual(signing["signingIdentity"], "fixture-certificate-fingerprint")
                    self.assertIs(signing["hardenedRuntime"], False)
                    self.assertEqual(signing["frameworks"], [
                        str(llama / "libggml.0.dylib"), str(llama / "libllama.0.dylib")
                    ])


class ModelResourceTests(unittest.TestCase):
    def setUp(self):
        self.resources = Path(__file__).parents[2] / "resources" / "vad"

    def test_tauri_bundles_the_exact_model_compliance_resource_set(self):
        config_path = Path(__file__).parents[2] / "apps" / "desktop" / "tauri.conf.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        resources = config["bundle"]["resources"]
        self.assertEqual(
            {
                destination: (config_path.parent / source).resolve()
                for source, destination in resources.items()
                if destination.startswith("vad/")
            },
            {
                f"vad/{name}": self.resources / name
                for name in model_resources.RESOURCE_NAMES
            },
        )

    def test_pinned_model_and_license_are_verified(self):
        model_resources.load_and_validate(self.resources)

    def test_verifier_rejects_wrong_size_model_and_extra_files(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "payload"
            shutil.copytree(self.resources, payload)
            (payload / "rogue.onnx").write_bytes(b"rogue")
            with self.assertRaisesRegex(ValueError, "exact pinned file set"):
                model_resources.load_and_validate(payload)
            (payload / "rogue.onnx").unlink()
            (payload / "silero_vad.onnx").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "model size"):
                model_resources.load_and_validate(payload)

    def test_verifier_rejects_same_size_model_and_license_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "payload"
            shutil.copytree(self.resources, payload)
            for filename, kind in (("silero_vad.onnx", "model"), ("LICENSE-MIT", "license")):
                with self.subTest(filename=filename):
                    path = payload / filename
                    original = path.read_bytes()
                    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                    try:
                        with self.assertRaisesRegex(ValueError, f"{kind} size or SHA-256"):
                            model_resources.load_and_validate(payload)
                    finally:
                        path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
