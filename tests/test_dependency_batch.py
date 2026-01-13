import sys
from unittest.mock import MagicMock, patch

from whispernow.core.deps.dependency_manager import DependencyManager, InstallProgress


def test_install_dependencies_batched_cpu():
    """Verify batched command construction for CPU."""
    manager = DependencyManager()

    with (
        patch("subprocess.Popen") as mock_popen,
        patch.object(
            manager, "get_missing_dependencies", return_value=["pkg1", "pkg2"]
        ),
        patch.object(manager, "_ensure_install_dir", return_value=True),
    ):

        # Mock process to return success immediately
        process_mock = MagicMock()
        process_mock.stdout.readline.side_effect = ["", ""]
        process_mock.poll.return_value = 0
        mock_popen.return_value = process_mock

        manager.install_dependencies(use_gpu=False)

        # Check that Popen was called once with all packages
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]

        # Verify command structure
        assert args[0] == sys.executable
        assert args[1:4] == ["-m", "pip", "install"]
        assert "--target" in args
        assert "pkg1" in args
        assert "pkg2" in args
        # Ensure no index-url/extra-index-url for CPU
        assert "--index-url" not in args
        assert "--extra-index-url" not in args


def test_install_dependencies_batched_gpu():
    """Verify batched command construction for GPU."""
    manager = DependencyManager()

    with (
        patch("subprocess.Popen") as mock_popen,
        patch.object(
            manager, "get_missing_dependencies", return_value=["torch", "scipy"]
        ),
        patch.object(manager, "_ensure_install_dir", return_value=True),
    ):

        process_mock = MagicMock()
        process_mock.stdout.readline.side_effect = ["", ""]
        process_mock.poll.return_value = 0
        mock_popen.return_value = process_mock

        manager.install_dependencies(use_gpu=True)

        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]

        assert "torch" in args
        assert "scipy" in args
        # Ensure extra-index-url is used
        assert "--extra-index-url" in args
        assert "https://download.pytorch.org/whl/cu121" in args
