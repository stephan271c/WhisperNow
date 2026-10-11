import importlib.util
import json
import shlex
import socket
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "smoke_dev_frontend.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("smoke_dev_frontend", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = load_module()


def unused_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class DevelopmentFrontendSmokeTests(unittest.TestCase):
    def test_runs_configured_server_and_inspects_js_and_wasm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            assets.mkdir()
            (root / "index.html").write_text(
                """<!doctype html><html><head><title>WhisperNow</title></head>
                <body><div id=\"main\"></div>
                <script type=\"module\" src=\"/assets/app.js\"></script></body></html>""",
                encoding="utf-8",
            )
            (assets / "app.js").write_text(
                'const binary = new URL("app_bg.wasm", import.meta.url);',
                encoding="utf-8",
            )
            (assets / "app_bg.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")
            port = unused_port()
            command = (
                f"{shlex.quote(sys.executable)} -m http.server {port} "
                "--bind 127.0.0.1"
            )
            config_path = root / "tauri.conf.json"
            config_path.write_text(
                json.dumps(
                    {
                        "productName": "WhisperNow",
                        "build": {
                            "beforeDevCommand": command,
                            "devUrl": f"http://127.0.0.1:{port}",
                        },
                        "app": {
                            "windows": [
                                {
                                    "label": "settings",
                                    "url": "/index.html?window=settings",
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            frontend = smoke.load_dev_frontend(config_path)
            result = smoke.run_smoke(frontend, timeout_seconds=5)

            self.assertEqual(
                frontend.request_url,
                f"http://127.0.0.1:{port}/index.html?window=settings",
            )
            self.assertEqual(
                result["module_url"], f"http://127.0.0.1:{port}/assets/app.js"
            )
            self.assertEqual(
                result["wasm_url"], f"http://127.0.0.1:{port}/assets/app_bg.wasm"
            )

    def test_rejects_non_loopback_dev_url(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tauri.conf.json"
            path.write_text(
                json.dumps(
                    {
                        "productName": "WhisperNow",
                        "build": {
                            "beforeDevCommand": "dx serve",
                            "devUrl": "https://example.com",
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(smoke.SmokeError, "loopback"):
                smoke.load_dev_frontend(path)

    def test_resolves_hook_object_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frontend_directory = root / "frontend"
            frontend_directory.mkdir()
            config_directory = root / "desktop"
            config_directory.mkdir()
            path = config_directory / "tauri.conf.json"
            path.write_text(
                json.dumps(
                    {
                        "productName": "WhisperNow",
                        "build": {
                            "beforeDevCommand": {
                                "script": "dx serve",
                                "cwd": "../frontend",
                            },
                            "devUrl": "http://127.0.0.1:1420",
                        },
                    }
                ),
                encoding="utf-8",
            )

            frontend = smoke.load_dev_frontend(path)

            self.assertEqual(frontend.command, "dx serve")
            self.assertEqual(frontend.command_directory, frontend_directory)


if __name__ == "__main__":
    unittest.main()
