import unittest
from unittest.mock import ANY, MagicMock, patch

from whispernow.core.deps.dependency_manager import DependencyManager, InstallProgress


class TestDependencyManagerProgress(unittest.TestCase):
    @patch("whispernow.core.deps.dependency_manager.subprocess.Popen")
    @patch("whispernow.core.deps.dependency_manager.sys.executable", "/usr/bin/python3")
    def test_install_dependencies_progress(self, mock_popen):
        # Setup mock process
        mock_process = MagicMock()
        mock_process.poll.side_effect = [
            None,
            None,
            None,
            0,
        ]  # Running, Running, Running, Done
        mock_process.returncode = 0

        # Simulate pip output
        # dependency_manager will read stdout.readline()
        # We need to simulate an iterable that yields lines
        mock_process.stdout = [
            b"Collecting torch",
            b"Downloading torch-2.0.0-cp310-cp310-linux_x86_64.whl (600.0 MB)",
            b"   10% |###                             | 60.0 MB 10.0 MB/s",
            b"   50% |###############                 | 300.0 MB 12.0 MB/s",
            b"  100% |################################| 600.0 MB 15.0 MB/s",
            b"Successfully installed torch",
        ]
        # Make stdout an iterator so existing code that might loop over it works,
        # or if we change to readline() loop it works.
        # For now, let's assume the implementation will iterate over mock_process.stdout
        # But subprocess.Popen.stdout is a file-like object.
        # Let's mock it better to support readline() which is common for real-time output

        # Reset stdout to a proper mock
        mock_stdout = MagicMock()
        mock_stdout.readline.side_effect = [
            "Collecting torch\n",
            "Downloading torch... (600 MB)\n",
            "   10% |###                             | 60.0 MB 10.0 MB/s\n",
            "   50% |###############                 | 300.0 MB 12.0 MB/s\n",
            "ERROR: this line should not break things\n",
            "",  # End of stream
        ]
        mock_process.stdout = mock_stdout

        # When readline returns "", code checks poll(). It should be done.
        # Check happens once inside loop (when "") and once outside.
        mock_process.poll.side_effect = [0, 0]
        mock_process.returncode = 0

        mock_popen.return_value = mock_process

        # Setup manager
        manager = DependencyManager()
        # Mock get_missing_dependencies to return just one
        manager.get_missing_dependencies = MagicMock(return_value=["torch"])
        manager._ensure_install_dir = MagicMock(return_value=True)  # Avoid FS touch

        # progress callback
        progress_calls = []

        def on_progress(p: InstallProgress):
            progress_calls.append(p)

        # Run
        success, msg = manager.install_dependencies(
            use_gpu=False, progress_callback=on_progress
        )

        # Assertions
        self.assertTrue(success)
        self.assertTrue(len(progress_calls) >= 2)  # At least start and finish

        # Verify we got called with package name
        self.assertEqual(progress_calls[0].package, "torch")

        # Verify progress update from percentage line
        # element 2 corresponds to "10%" line
        # But we get callbacks for every line.
        # "Collecting" -> progress 0
        # "Downloading" -> progress 0
        # "10%" -> progress 0.1
        # Find the call with 10% progress
        progress_values = [p.progress for p in progress_calls]
        self.assertIn(0.1, progress_values)
        self.assertIn(0.5, progress_values)
        self.assertIn(1.0, progress_values)

        # Verify Popen was called
        mock_popen.assert_called_once()
