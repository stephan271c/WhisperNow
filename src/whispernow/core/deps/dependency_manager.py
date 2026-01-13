"""
Dependency Manager for WhisperNow.

Handles detection and installation of heavy ML dependencies (PyTorch, NeMo, etc.)
into a local site-packages directory at first run.

NOTE: We use `pip install --target` instead of a venv because AppImage bundles
Python in a way that creates broken venvs with symlinks to temp extraction paths.
"""

import subprocess
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

import platformdirs


class DependencyStatus(Enum):
    NOT_INSTALLED = auto()
    INSTALLING = auto()
    INSTALLED = auto()
    FAILED = auto()


@dataclass
class InstallProgress:
    package: str
    progress: float  # 0.0 to 1.0
    status: str
    total_packages: int
    current_package: int


# Heavy dependencies that are downloaded on first run
HEAVY_DEPS = [
    "torch",
    "torchaudio",
    "nemo-toolkit[asr]",
    "litellm",
    "scipy",
    "numpy",
]

# PyTorch CUDA index for GPU installations
CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu121"

# Packages that need the CUDA index
CUDA_PACKAGES = {"torch", "torchaudio", "torchvision"}


class DependencyManager:
    def __init__(self):
        self._install_dir = Path(platformdirs.user_data_dir("WhisperNow")) / "packages"
        self._status = DependencyStatus.NOT_INSTALLED
        self._cancel_requested = False

    @property
    def install_dir(self) -> Path:
        return self._install_dir

    # Keep legacy property for compatibility with bootstrap.py
    @property
    def site_packages_dir(self) -> Path:
        return self._install_dir

    @property
    def status(self) -> DependencyStatus:
        return self._status

    def add_to_path(self) -> None:
        install_path = str(self._install_dir)
        if self._install_dir.exists() and install_path not in sys.path:
            sys.path.insert(0, install_path)

    # Legacy alias
    add_venv_to_path = add_to_path

    def check_dependencies_installed(self) -> bool:
        self.add_to_path()
        try:
            import importlib.util

            # Check core heavy dependencies
            required = ["torch", "nemo"]
            for module in required:
                if importlib.util.find_spec(module) is None:
                    return False
            return True
        except Exception:
            return False

    def get_missing_dependencies(self) -> list[str]:
        self.add_to_path()
        missing = []
        try:
            import importlib.util

            module_map = {
                "torch": "torch",
                "torchaudio": "torchaudio",
                "nemo-toolkit[asr]": "nemo",
                "litellm": "litellm",
                "scipy": "scipy",
                "numpy": "numpy",
            }
            for pkg, module in module_map.items():
                if importlib.util.find_spec(module) is None:
                    missing.append(pkg)
        except Exception:
            return HEAVY_DEPS.copy()
        return missing

    def detect_gpu_available(self) -> bool:
        try:
            # Check for NVIDIA GPU via nvidia-smi
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _ensure_install_dir(self) -> bool:
        try:
            self._install_dir.mkdir(parents=True, exist_ok=True)
            return True
        except OSError:
            return False

    def cancel_installation(self) -> None:
        self._cancel_requested = True

    def install_dependencies(
        self,
        use_gpu: bool = False,
        progress_callback: Optional[Callable[[InstallProgress], None]] = None,
    ) -> tuple[bool, str]:
        """
        Install heavy dependencies into the local site-packages.

        Args:
            use_gpu: If True, install CUDA-enabled PyTorch
            progress_callback: Called with progress updates

        Returns:
            Tuple of (success, message)
        """
        self._status = DependencyStatus.INSTALLING
        self._cancel_requested = False

        # Ensure install dir exists
        if not self._ensure_install_dir():
            self._status = DependencyStatus.FAILED
            return False, "Failed to create installation directory"

        missing = self.get_missing_dependencies()
        if not missing:
            self._status = DependencyStatus.INSTALLED
            return True, "All dependencies already installed"

        # Batch installation
        success, msg = self._install_packages_batched(
            missing,
            use_gpu,
            progress_callback=progress_callback,
        )

        if not success:
            self._status = DependencyStatus.FAILED
            return False, msg

        self._status = DependencyStatus.INSTALLED
        return True, "Dependencies installed successfully"

    def _install_packages_batched(
        self,
        packages: list[str],
        use_gpu: bool,
        progress_callback: Optional[Callable[[InstallProgress], None]] = None,
    ) -> tuple[bool, str]:
        # Use sys.executable (AppImage python) and --target
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(self._install_dir),
            "--no-input",
        ]

        # Use extra-index-url for CUDA packages so we can still find standard packages on PyPI
        if use_gpu:
            cmd.extend(["--extra-index-url", CUDA_INDEX_URL])

        cmd.extend(packages)

        try:
            # use Popen to capture output in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )

            last_line = ""
            total_packages = len(packages)

            # Simple progress tracking variables
            current_stage = "Initializing..."

            # Read stdout line by line
            while True:
                # Check for cancellation
                if self._cancel_requested:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return False, "Installation cancelled"

                output_line = process.stdout.readline()
                if output_line == "" and process.poll() is not None:
                    break

                if output_line:
                    line = output_line.strip()
                    last_line = line

                    # Heuristics for progress description
                    if line.startswith("Collecting"):
                        pkg = line.split("Collecting")[-1].strip()
                        current_stage = f"Downloading {pkg}..."
                    elif line.startswith("Installing"):
                        current_stage = f"Installing..."
                    elif "Downloading" in line:
                        # Keep the downloading details, maybe shorten
                        if len(line) > 50:
                            current_stage = f"{line[:47]}..."
                        else:
                            current_stage = line

                    if progress_callback:
                        # Since we don't have accurate per-package progress in batched mode without elaborate parsing,
                        # we'll use a "pulse" or indefinite style progress, or just 50% fixed.
                        # However, the UI expects 0-100 float.
                        # We can try to approximate based on "Collecting" counts if we really wanted to,
                        # but for now, let's just show activity.

                        # Let's send a "busy" progress which might just cycle or stay at a point.
                        # Or rely on the UI's existing implementation which might just show a bar.
                        # The existing UI sets value(percent).

                        # Better approach: Just show the text update and keep progress at 50%
                        # until finished, or slowly increment if we could count lines (not reliable).
                        # Let's just forward the status.
                        progress_callback(
                            InstallProgress(
                                package="Batched Install",
                                progress=0.5,  # Indeterminate state
                                status=current_stage,
                                total_packages=total_packages,
                                current_package=1,
                            )
                        )

            return_code = process.poll()
            if return_code != 0:
                return False, f"Failed to install packages: {last_line}"

            if progress_callback:
                progress_callback(
                    InstallProgress(
                        package="",
                        progress=1.0,
                        status="Installation complete!",
                        total_packages=total_packages,
                        current_package=total_packages,
                    )
                )

            return True, ""

        except Exception as e:
            return False, f"Error installing packages: {e}"

    def get_estimated_download_size(self, use_gpu: bool) -> str:
        if use_gpu:
            return "~2.5 GB (CUDA-enabled)"
        else:
            return "~400 MB (CPU-only)"
