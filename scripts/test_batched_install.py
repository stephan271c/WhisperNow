import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp

from whispernow.core.deps import DependencyManager, InstallProgress


class TestDependencyManager(DependencyManager):
    def __init__(self, temp_dir):
        self._temp_dir = Path(temp_dir)
        super().__init__()
        self._install_dir = self._temp_dir

    def get_missing_dependencies(self) -> list[str]:
        # Return some small, safe packages to test installation
        return ["colorama", "six"]


def main():
    temp_dir = mkdtemp()
    print(f"Testing batched install in: {temp_dir}")

    try:
        manager = TestDependencyManager(temp_dir)

        def progress_callback(p: InstallProgress):
            print(f"PROGRESS: {p.status} ({p.progress*100:.1f}%)")

        print("Starting installation...")
        success, msg = manager.install_dependencies(
            use_gpu=False, progress_callback=progress_callback
        )

        if success:
            print("\nSUCCESS: Installation completed.")
            # Verify files exist
            pkg_dir = Path(temp_dir)
            if (pkg_dir / "colorama").exists() and (pkg_dir / "six.py").exists():
                print("VERIFIED: Packages found in target directory.")
            else:
                print("FAILED: Packages not found in target directory.")
        else:
            print(f"\nFAILED: {msg}")

    finally:
        shutil.rmtree(temp_dir)
        print("Cleaned up temp directory.")


if __name__ == "__main__":
    main()
