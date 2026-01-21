import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

from .logger import get_logger

logger = get_logger(__name__)


def _is_safe_path(path: Path) -> bool:

    try:
        resolved = path.resolve()

        if resolved == Path(resolved.root):
            return False

        if resolved == Path.home():
            return False

        return True
    except Exception:
        return False


def generate_cleanup_script(paths_to_delete: List[Path]) -> Path:
    system = platform.system()
    pid = os.getpid()

    valid_paths = []
    for p in paths_to_delete:
        if not p.exists():
            continue

        if not _is_safe_path(p):
            logger.warning(f"SKIPPING UNSAFE PATH cleanup request for: {p}")
            continue

        valid_paths.append(p)

    if not valid_paths:
        logger.warning(
            "No valid, safe paths to delete provided to cleanup script generator"
        )

    if system == "Windows":
        return _generate_windows_script(valid_paths, pid)
    else:
        return _generate_linux_script(valid_paths, pid)


def run_cleanup_script(script_path: Path) -> None:
    logger.info(f"Launching cleanup script: {script_path}")

    system = platform.system()

    try:
        if system == "Windows":
            subprocess.Popen(
                [str(script_path)],
                creationflags=subprocess.CREATE_NO_WINDOW,
                shell=True,
                close_fds=True,
            )
        else:
            subprocess.Popen(
                ["/bin/bash", str(script_path)],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
    except Exception as e:
        logger.error(f"Failed to launch cleanup script: {e}")
        raise


def _generate_windows_script(paths: List[Path], pid: int) -> Path:
    fd, path = tempfile.mkstemp(suffix=".bat", prefix="whispernow_cleanup_")
    os.close(fd)
    script_path = Path(path)

    commands = [
        "@echo off",
        ":WAIT_LOOP",
        f'tasklist /FI "PID eq {pid}" 2>nul | find /i "{pid}" >nul',
        "if errorlevel 1 (",
        "    goto :DELETE_FILES",
        ") else (",
        "    timeout /t 1 /nobreak >nul",
        "    goto :WAIT_LOOP",
        ")",
        "",
        ":DELETE_FILES",
        "timeout /t 2 /nobreak >nul",
    ]

    for p in paths:
        win_path = str(p).replace("/", "\\")
        commands.append(f'if exist "{win_path}\\" rmdir /s /q "{win_path}" >nul 2>&1')
        commands.append(f'if exist "{win_path}" del /f /q "{win_path}" >nul 2>&1')

    commands.append(f'del "%~f0" >nul 2>&1')

    try:
        script_path.write_text("\n".join(commands), encoding="utf-8")
        return script_path
    except Exception as e:
        logger.error(f"Failed to write Windows cleanup script: {e}")
        raise


def _generate_linux_script(paths: List[Path], pid: int) -> Path:
    fd, path = tempfile.mkstemp(suffix=".sh", prefix="whispernow_cleanup_")
    os.close(fd)
    script_path = Path(path)

    commands = [
        "#!/bin/bash",
        "",
        f"tail --pid={pid} -f /dev/null",
        "sleep 2",
        "",
    ]

    for p in paths:
        commands.append(f'rm -rf "{p}"')

    commands.append("")
    commands.append(f'rm -- "$0"')

    try:
        script_path.write_text("\n".join(commands), encoding="utf-8")
        script_path.chmod(0o755)
        return script_path
    except Exception as e:
        logger.error(f"Failed to write Linux cleanup script: {e}")
        raise
