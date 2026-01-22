import os
import platform
import re
import subprocess
import sys
from pathlib import Path


def inject_version(project_root: Path, version: str) -> None:
    print(f"Injecting version: {version}")

    pyproject_path = project_root / "pyproject.toml"
    content = pyproject_path.read_text()
    content = re.sub(
        r'^version = "[^"]+"', f'version = "{version}"', content, flags=re.MULTILINE
    )
    pyproject_path.write_text(content)
    print(f"  Updated {pyproject_path}")

    init_path = project_root / "src" / "whispernow" / "__init__.py"
    content = init_path.read_text()
    content = re.sub(
        r'^__version__ = "[^"]+"',
        f'__version__ = "{version}"',
        content,
        flags=re.MULTILINE,
    )
    init_path.write_text(content)
    print(f"  Updated {init_path}")


def get_platform_target() -> tuple[str, str]:
    system = platform.system().lower()
    if system == "linux":
        return ("linux", "appimage")
    elif system == "windows":
        return ("windows", "app")
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def build():
    project_root = Path(__file__).parent.parent
    plat, fmt = get_platform_target()

    app_version = os.environ.get("APP_VERSION")
    if app_version and app_version.startswith("v"):
        app_version = app_version[1:]  # Strip leading 'v' from tag
    if app_version and app_version != "0.1.0":
        inject_version(project_root, app_version)

    print("=" * 60)
    print(f"Briefcase Build - {plat} ({fmt})")
    print("=" * 60)

    try:
        print(f"\n[1/2] Creating application scaffold for {plat}/{fmt}...")
        subprocess.run(
            [sys.executable, "-m", "briefcase", "create", plat, fmt],
            cwd=project_root,
            check=True,
        )

        print(f"\n[2/2] Building application for {plat}/{fmt}...")
        subprocess.run(
            [sys.executable, "-m", "briefcase", "build", plat, fmt],
            cwd=project_root,
            check=True,
        )

        print("\n" + "=" * 60)
        print("Build successful!")
        print("=" * 60)

        if plat == "linux":
            build_dir = project_root / "build" / "whispernow" / "linux" / "appimage"
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
