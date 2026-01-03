
import subprocess
import shutil
from pathlib import Path
import sys

def build():
    """Run the PyInstaller build process."""
    project_root = Path(__file__).parent.parent
    dist_dir = project_root / "dist"
    build_dir = project_root / "build"
    
    # Clean previous builds
    print("Cleaning previous builds...")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    if build_dir.exists():
        shutil.rmtree(build_dir)
        
    print("Running PyInstaller...")
    try:
        subprocess.run(
            ["pyinstaller", "transcribe.spec", "--clean"],
            cwd=project_root,
            check=True
        )
        print("\nBuild successful! Output is in dist/transcribe/")
        
        # Verify
        executable = dist_dir / "transcribe" / "transcribe"
        if executable.exists():
             print(f"Executable created: {executable}")
        else:
             print("Warning: Executable not found at expected path.")
             
    except subprocess.CalledProcessError as e:
        print(f"\nBuild failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    build()
