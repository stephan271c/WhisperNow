"""
Dependency Manager for WhisperNow.

Handles detection and installation of heavy ML dependencies (PyTorch, NeMo, etc.)
into an isolated virtual environment at first run.
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
        self._venv_dir = Path(platformdirs.user_data_dir("WhisperNow")) / "python"
        self._status = DependencyStatus.NOT_INSTALLED
        self._cancel_requested = False

    @property
    def venv_dir(self) -> Path:
        return self._venv_dir

    @property
    def site_packages_dir(self) -> Path:
        if sys.platform == "win32":
            return self._venv_dir / "Lib" / "site-packages"
        else:
            return (
                self._venv_dir
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
            )

    @property
    def python_executable(self) -> Path:
        if sys.platform == "win32":
            return self._venv_dir / "Scripts" / "python.exe"
        else:
            return self._venv_dir / "bin" / "python"

    @property
    def status(self) -> DependencyStatus:
        return self._status

    def add_venv_to_path(self) -> None:
        site_pkg = str(self.site_packages_dir)
        if self.site_packages_dir.exists() and site_pkg not in sys.path:
            sys.path.insert(0, site_pkg)

    def check_dependencies_installed(self) -> bool:
        self.add_venv_to_path()
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
        self.add_venv_to_path()
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

    def _create_venv(self) -> bool:
        if self._venv_dir.exists():
            return True
        try:
            self._venv_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [sys.executable, "-m", "venv", str(self._venv_dir)],
                check=True,
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def _upgrade_pip(self) -> bool:
        try:
            subprocess.run(
                [
                    str(self.python_executable),
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "pip",
                ],
                check=True,
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def cancel_installation(self) -> None:
        self._cancel_requested = True

    def install_dependencies(
        self,
        use_gpu: bool = False,
        progress_callback: Optional[Callable[[InstallProgress], None]] = None,
    ) -> tuple[bool, str]:
        """
        Install heavy dependencies into the local venv.

        Args:
            use_gpu: If True, install CUDA-enabled PyTorch
            progress_callback: Called with progress updates

        Returns:
            Tuple of (success, message)
        """
        self._status = DependencyStatus.INSTALLING
        self._cancel_requested = False

        # Create venv if needed
        if not self._create_venv():
            self._status = DependencyStatus.FAILED
            return False, "Failed to create virtual environment"

        if not self._upgrade_pip():
            self._status = DependencyStatus.FAILED
            return False, "Failed to upgrade pip"

        missing = self.get_missing_dependencies()
        if not missing:
            self._status = DependencyStatus.INSTALLED
            return True, "All dependencies already installed"

        total = len(missing)

        for i, package in enumerate(missing):
            if self._cancel_requested:
                self._status = DependencyStatus.FAILED
                return False, "Installation cancelled"

            if progress_callback:
                progress_callback(
                    InstallProgress(
                        package=package,
                        progress=(i / total),
                        status=f"Installing {package}...",
                        total_packages=total,
                        current_package=i + 1,
                    )
                )

            success, msg = self._install_package(package, use_gpu)
            if not success:
                self._status = DependencyStatus.FAILED
                return False, msg

        if progress_callback:
            progress_callback(
                InstallProgress(
                    package="",
                    progress=1.0,
                    status="Installation complete!",
                    total_packages=total,
                    current_package=total,
                )
            )

        self._status = DependencyStatus.INSTALLED
        return True, "Dependencies installed successfully"

    def _install_package(self, package: str, use_gpu: bool) -> tuple[bool, str]:
        cmd = [str(self.python_executable), "-m", "pip", "install", "--no-input"]

        # Handle CUDA packages
        base_pkg = package.split("[")[0]
        if use_gpu and base_pkg in CUDA_PACKAGES:
            cmd.extend(["--index-url", CUDA_INDEX_URL])

        cmd.append(package)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout per package
            )
            if result.returncode != 0:
                return False, f"Failed to install {package}: {result.stderr}"
            return True, ""
        except subprocess.TimeoutExpired:
            return False, f"Timeout installing {package}"
        except Exception as e:
            return False, f"Error installing {package}: {e}"

    def get_estimated_download_size(self, use_gpu: bool) -> str:
        if use_gpu:
            return "~2.5 GB (CUDA-enabled)"
        else:
            return "~400 MB (CPU-only)"
