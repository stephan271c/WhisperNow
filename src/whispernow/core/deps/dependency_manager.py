import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable, List, Optional, Set

import platformdirs

# Configuration Constants
CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu121"
ESTIMATED_SIZE_GB = 8.7
ESTIMATED_SIZE_BYTES = int(ESTIMATED_SIZE_GB * 1024 * 1024 * 1024)

HEAVY_DEPS = [
    "torch",
    "torchaudio",
    "nemo-toolkit[asr]",
    "litellm",
    "scipy",
    "numpy",
]

CUDA_PACKAGES = {"torch", "torchaudio", "torchvision"}

MODULE_MAP = {
    "torch": "torch",
    "torchaudio": "torchaudio",
    "nemo-toolkit[asr]": "nemo",
    "litellm": "litellm",
    "scipy": "scipy",
    "numpy": "numpy",
}


class DependencyStatus(Enum):
    NOT_INSTALLED = auto()
    INSTALLING = auto()
    INSTALLED = auto()
    FAILED = auto()


@dataclass
class InstallProgress:
    package: str
    progress: float
    status: str
    total_packages: int
    current_package: int


class DependencyManager:
    def __init__(self):
        self._install_dir = Path(platformdirs.user_data_dir("WhisperNow")) / "packages"
        self._status = DependencyStatus.NOT_INSTALLED
        self._cancel_requested = False

    @property
    def install_dir(self) -> Path:
        return self._install_dir

    @property
    def site_packages_dir(self) -> Path:
        return self._install_dir

    @property
    def status(self) -> DependencyStatus:
        return self._status

    def add_to_path(self) -> None:
        """Adds the custom install directory to sys.path if it exists."""
        install_path = str(self._install_dir)
        if self._install_dir.exists() and install_path not in sys.path:
            sys.path.insert(0, install_path)

    add_venv_to_path = add_to_path

    def check_dependencies_installed(self) -> bool:
        self.add_to_path()
        try:
            required = ["torch", "nemo"]
            for module in required:
                if importlib.util.find_spec(module) is None:
                    return False
            return True
        except Exception:
            return False

    def get_missing_dependencies(self) -> List[str]:
        self.add_to_path()
        missing = []
        try:
            for pkg, module in MODULE_MAP.items():
                if importlib.util.find_spec(module) is None:
                    missing.append(pkg)
        except Exception:
            # Fallback to reinstalling everything if detection fails
            return HEAVY_DEPS.copy()
        return missing

    def detect_gpu_available(self) -> bool:
        """Checks for NVIDIA GPU availability via nvidia-smi."""
        try:
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
        progress_callback: Optional[Callable[[InstallProgress], None]] = None,
    ) -> tuple[bool, str]:
        self._status = DependencyStatus.INSTALLING
        self._cancel_requested = False

        if not self._ensure_install_dir():
            self._status = DependencyStatus.FAILED
            return False, "Failed to create installation directory"

        missing = self.get_missing_dependencies()
        if not missing:
            self._status = DependencyStatus.INSTALLED
            return True, "All dependencies already installed"

        success, msg = self._install_packages_batched(
            missing,
            progress_callback=progress_callback,
        )

        if not success:
            self._status = DependencyStatus.FAILED
            return False, msg

        self._status = DependencyStatus.INSTALLED
        return True, "Dependencies installed successfully"

    def _get_directory_size(self, path: Path) -> int:
        total = 0
        try:
            for entry in os.scandir(path):
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += self._get_directory_size(Path(entry.path))
        except OSError:
            pass
        return total

    def _install_packages_batched(
        self,
        packages: List[str],
        progress_callback: Optional[Callable[[InstallProgress], None]] = None,
    ) -> tuple[bool, str]:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(self._install_dir),
            "--no-input",
            "--extra-index-url",
            CUDA_INDEX_URL,
        ]
        cmd.extend(packages)

        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        if pythonpath:
            env["PYTHONPATH"] = f"{str(self._install_dir)}:{pythonpath}"
        else:
            env["PYTHONPATH"] = str(self._install_dir)

        initial_size = self._get_directory_size(self._install_dir)

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )

            last_line = ""
            current_stage = "Initializing..."
            seen_packages: Set[str] = set()

            last_size_check = 0
            current_size = initial_size

            while True:
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

                    if line.startswith("Collecting"):
                        pkg = line.split("Collecting")[-1].strip()
                        current_stage = f"Downloading {pkg}..."
                        if pkg not in seen_packages:
                            seen_packages.add(pkg)
                    elif line.startswith("Installing"):
                        current_stage = "Installing..."
                    elif "Downloading" in line:
                        if len(line) > 50:
                            current_stage = f"{line[:47]}..."
                        else:
                            current_stage = line

                    if progress_callback:
                        now = time.time()
                        if now - last_size_check > 1.0:
                            current_size = self._get_directory_size(self._install_dir)
                            last_size_check = now

                        downloaded_bytes = max(0, current_size - initial_size)

                        # Progress Calculation Strategy:
                        # 5% reserved for initialization/metadata
                        # 94% based on size (simulating progress up to near-completion)
                        # 1% reserved for final completion
                        base_progress = 0.05
                        size_progress = min(
                            (downloaded_bytes / ESTIMATED_SIZE_BYTES) * 0.95, 0.94
                        )
                        total_progress = base_progress + size_progress

                        display_total = max(len(packages), len(seen_packages))
                        display_current = len(seen_packages)

                        progress_callback(
                            InstallProgress(
                                package=f"Size: {downloaded_bytes / 1024 / 1024:.1f} MB",
                                progress=total_progress,
                                status=current_stage,
                                total_packages=display_total,
                                current_package=display_current,
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
                        total_packages=max(len(packages), len(seen_packages)),
                        current_package=max(len(packages), len(seen_packages)),
                    )
                )

            return True, ""

        except Exception as e:
            return False, f"Error installing packages: {e}"

    def get_estimated_download_size(self) -> str:
        return f"~{ESTIMATED_SIZE_GB} GB (CUDA-enabled)"
