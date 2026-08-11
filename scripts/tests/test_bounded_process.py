import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "bounded_process.py"


class BoundedProcessTests(unittest.TestCase):
    def invoke(
        self, *command: str, timeout: float = 2.0
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--timeout-seconds",
                str(timeout),
                "--grace-seconds",
                "0.2",
                "--label",
                "fixture",
                "--",
                *command,
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )

    def test_success_is_content_free(self):
        result = self.invoke(sys.executable, "-c", "print('private transcript')")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "bounded process fixture: completed\n")
        self.assertNotIn("private transcript", result.stdout + result.stderr)

    def test_timeout_is_deterministic(self):
        result = self.invoke(
            sys.executable, "-c", "import time; time.sleep(10)", timeout=0.1
        )
        self.assertEqual(result.returncode, 124)
        self.assertEqual(result.stderr, "bounded process fixture: deadline exceeded\n")

    def test_nonzero_exit_is_reported_without_child_output(self):
        result = self.invoke(
            sys.executable, "-c", "import sys; print('secret'); sys.exit(7)"
        )
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stderr, "bounded process fixture: exited nonzero (7)\n")

    def test_missing_executable_is_deterministic(self):
        result = self.invoke("whispernow-definitely-missing-executable")
        self.assertEqual(result.returncode, 127)
        self.assertEqual(
            result.stderr, "bounded process fixture: executable unavailable\n"
        )

    def test_signal_termination_is_reported(self):
        result = self.invoke(
            sys.executable,
            "-c",
            "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
        )
        if os.name == "nt":
            self.assertEqual(result.returncode, signal.SIGTERM)
            self.assertEqual(
                result.stderr,
                f"bounded process fixture: exited nonzero ({signal.SIGTERM})\n",
            )
        else:
            self.assertEqual(result.returncode, 143)
            self.assertEqual(
                result.stderr, "bounded process fixture: terminated by signal 15\n"
            )

    def test_timeout_terminates_child_process_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "child.pid"
            code = (
                "import pathlib, subprocess, sys, time; "
                "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)"
            )
            result = self.invoke(sys.executable, "-c", code, str(pid_file), timeout=0.3)
            self.assertEqual(result.returncode, 124)
            child_pid = int(pid_file.read_text(encoding="utf-8"))
            for _attempt in range(20):
                if not self.process_exists(child_pid):
                    break
                time.sleep(0.05)
            else:
                self.fail("timed-out child process remained alive")

    @staticmethod
    def process_exists(pid: int) -> bool:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                text=True,
                capture_output=True,
                check=False,
            )
            return f'"{pid}"' in result.stdout
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True


if __name__ == "__main__":
    unittest.main()
