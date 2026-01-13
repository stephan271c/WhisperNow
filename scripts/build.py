"""Build WhisperNow using Briefcase for all platforms."""

import platform
import subprocess
import sys
from pathlib import Path


def get_platform_target() -> tuple[str, str]:
    """Return (platform, format) tuple for the current OS."""
    system = platform.system().lower()
    if system == "linux":
        return ("linux", "appimage")
    elif system == "darwin":
        return ("macos", "app")
    elif system == "windows":
        return ("windows", "app")
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def build():
    """Run the Briefcase build process for the current platform."""
    project_root = Path(__file__).parent.parent
    plat, fmt = get_platform_target()

    print("=" * 60)
    print(f"Briefcase Build - {plat} ({fmt})")
    print("=" * 60)

    try:
        # Step 1: Create the application scaffold
        print(f"\n[1/2] Creating application scaffold for {plat}/{fmt}...")
        subprocess.run(
            [sys.executable, "-m", "briefcase", "create", plat, fmt],
            cwd=project_root,
            check=True,
        )

        # Step 2: Build the application
        print(f"\n[2/2] Building application for {plat}/{fmt}...")
        subprocess.run(
            [sys.executable, "-m", "briefcase", "build", plat, fmt],
            cwd=project_root,
            check=True,
        )

        print("\n" + "=" * 60)
        print("Build successful!")
        print("=" * 60)

        # Show output location based on platform
        if plat == "linux":
            build_dir = project_root / "build" / "whispernow" / "linux" / "appimage"
        elif plat == "darwin":
            build_dir = project_root / "build" / "whispernow" / "macos" / "app"
        elif plat == "windows":
            build_dir = project_root / "build" / "whispernow" / "windows" / "app"

        if build_dir.exists():
            print(f"Build output: {build_dir}")
        else:
            print(f"Note: Build directory may be at {build_dir}")

    except subprocess.CalledProcessError as e:
        print(f"\nBuild failed with exit code {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    build()
