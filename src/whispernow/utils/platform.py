"""Platform-specific utilities for cross-platform compatibility."""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .logger import get_logger

logger = get_logger(__name__)


def get_executable_path() -> str:

    system = get_platform()

    if system == "linux":
        appimage_path = os.environ.get("APPIMAGE")
        if appimage_path and Path(appimage_path).exists():
            logger.debug(f"Running as AppImage: {appimage_path}")
            return appimage_path

        exe_path = Path(sys.executable).resolve()
        # Look for .app in the path (e.g., /Applications/WhisperNow.app/Contents/MacOS/python)
        for parent in exe_path.parents:
            if parent.suffix == ".app":
                logger.debug(f"Running as macOS .app bundle: {parent}")
                return str(parent)

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
    elif system == "macos":
        exe_path = Path(sys.executable).resolve()
        return any(p.suffix == ".app" for p in exe_path.parents)
    elif system == "windows":
        return getattr(sys, "frozen", False)

    return False


def get_platform() -> str:
    system = platform.system()
    if system == "Darwin":
        return "macos"
    return system.lower()


def check_accessibility_permissions() -> bool:

    if get_platform() != "macos":
        return True

    try:
        # Attempt a minimal System Events interaction to test accessibility
        result = subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to keystroke ""'],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning("Accessibility permission check timed out")
        return False
    except Exception as e:
        logger.warning(f"Failed to check accessibility permissions: {e}")
        return False


def request_accessibility_permissions() -> None:

    if get_platform() != "macos":
        return

    subprocess.run(
        [
            "open",
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
        ],
        check=False,
    )


def set_autostart(enabled: bool, app_name: str = "WhisperNow") -> bool:

    system = get_platform()

    if system == "windows":
        return _set_autostart_windows(enabled, app_name)
    elif system == "macos":
        return _set_autostart_macos(enabled, app_name)
    else:  # Linux
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


def _set_autostart_macos(enabled: bool, app_name: str) -> bool:
    plist_path = (
        Path.home() / "Library" / "LaunchAgents" / f"com.{app_name.lower()}.plist"
    )

    if enabled:
        exe_path = get_executable_path()
        packaged = is_packaged()

        if packaged:
            plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.{app_name.lower()}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/open</string>
        <string>{exe_path}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
"""
        else:
            plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.{app_name.lower()}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exe_path}</string>
        <string>-m</string>
        <string>whispernow</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
"""
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_path.write_text(plist_content)
        logger.info(f"macOS autostart enabled: {plist_path}")
    else:
        plist_path.unlink(missing_ok=True)
        logger.info("macOS autostart disabled")

    return True


def _set_autostart_linux(enabled: bool, app_name: str) -> bool:
    autostart_dir = Path.home() / ".config" / "autostart"
    desktop_file = autostart_dir / f"{app_name.lower()}.desktop"

    if enabled:
        exe_path = get_executable_path()
        packaged = is_packaged()

        if packaged:
            exec_cmd = exe_path
        else:
            exec_cmd = f"{exe_path} -m whispernow"

        desktop_content = f"""[Desktop Entry]
Type=Application
Name={app_name}
Exec={exec_cmd}
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
Comment=Voice-to-text transcription with AI enhancement
"""
        autostart_dir.mkdir(parents=True, exist_ok=True)
        desktop_file.write_text(desktop_content)
        logger.info(f"Linux autostart enabled: {desktop_file} -> {exec_cmd}")
    else:
        desktop_file.unlink(missing_ok=True)
        logger.info("Linux autostart disabled")

    return True


def get_app_icon_path() -> Optional[Path]:
    # TODO: Implement icon path resolution
    return None


def check_and_request_permissions(settings: "Settings") -> bool:

    if get_platform() != "macos":
        return True

    if settings.accessibility_permissions_granted:
        if check_accessibility_permissions():
            return True
        else:
            logger.warning("Accessibility permission was revoked, prompting user")

    if check_accessibility_permissions():
        settings.accessibility_permissions_granted = True
        settings.save()
        logger.info("Accessibility permissions already granted")
        return True

    from ..ui.permissions_dialog import PermissionsDialog

    logger.info("Showing accessibility permissions dialog")
    dialog = PermissionsDialog()
    dialog.exec()

    granted = check_accessibility_permissions()
    settings.accessibility_permissions_granted = granted
    settings.save()

    if granted:
        logger.info("User granted accessibility permissions")
    else:
        logger.warning("User continued without accessibility permissions")

    return granted


if TYPE_CHECKING:
    from ..core.settings import Settings
