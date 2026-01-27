import os
import platform
import shlex
import sys
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)


def get_executable_path() -> str:
    system = get_platform()

    if system == "linux":
        appimage_path = os.environ.get("APPIMAGE")
        if appimage_path and Path(appimage_path).exists():
            logger.debug(f"Running as AppImage: {appimage_path}")
            return appimage_path

    if system == "windows":
        if getattr(sys, "frozen", False):
            logger.debug(f"Running as frozen Windows exe: {sys.executable}")
            return sys.executable

    logger.debug(f"Running in development mode: {sys.executable}")
    return sys.executable


def is_packaged() -> bool:

    system = get_platform()

    if system == "linux":
        return bool(os.environ.get("APPIMAGE"))
    elif system == "windows":
        return getattr(sys, "frozen", False)

    return False


def get_platform() -> str:
    return platform.system().lower()


def set_autostart(enabled: bool, app_name: str = "WhisperNow") -> bool:

    system = get_platform()

    if system == "windows":
        return _set_autostart_windows(enabled, app_name)
    else:
        return _set_autostart_linux(enabled, app_name)


def _set_autostart_windows(enabled: bool, app_name: str) -> bool:
    try:
        import winreg

        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE
        ) as key:
            if enabled:
                exe_path = get_executable_path()
                if is_packaged():
                    startup_cmd = f'"{exe_path}"'
                else:
                    startup_cmd = f'"{exe_path}" -m whispernow'
                winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, startup_cmd)
                logger.info(f"Windows autostart enabled: {startup_cmd}")
            else:
                try:
                    winreg.DeleteValue(key, app_name)
                    logger.info("Windows autostart disabled")
                except FileNotFoundError:
                    pass

        return True
    except Exception as e:
        logger.error(f"Failed to set autostart: {e}", exc_info=True)
        return False


def _set_autostart_linux(enabled: bool, app_name: str) -> bool:
    autostart_dir = Path.home() / ".config" / "autostart"
    desktop_file = autostart_dir / f"{app_name.lower()}.desktop"

    if enabled:
        exe_path = get_executable_path()
        packaged = is_packaged()

        quoted_exe = shlex.quote(str(exe_path))

        if packaged:
            exec_cmd = quoted_exe
        else:
            exec_cmd = f"{quoted_exe} -m whispernow"

        desktop_content = f"""[Desktop Entry]
Type=Application
Name={app_name}
Exec={exec_cmd}
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
Comment=Voice-to-text transcription with AI enhancement
"""
        try:
            autostart_dir.mkdir(parents=True, exist_ok=True)
            desktop_file.write_text(desktop_content)
            logger.info(f"Linux autostart enabled: {desktop_file} -> {exec_cmd}")
        except OSError as e:
            logger.error(f"Failed to write autostart file: {e}")
            return False
    else:
        try:
            desktop_file.unlink(missing_ok=True)
            logger.info("Linux autostart disabled")
        except OSError as e:
            logger.error(f"Failed to remove autostart file: {e}")
            return False

    return True


def get_app_icon_path() -> Path | None:
    return None
