"""Platform-specific utilities for cross-platform compatibility."""

import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .logger import get_logger

logger = get_logger(__name__)


def get_platform() -> str:
    system = platform.system()
    if system == "Darwin":
        return "macos"
    return system.lower()


def check_accessibility_permissions() -> bool:
    """
    Check if the app has accessibility permissions (macOS only).

    On macOS, accessibility permissions are required for keyboard listening.
    On other platforms, always returns True.
    """
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
    """Guide the user to grant accessibility permissions (macOS only)."""
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
    """
    Enable or disable auto-start on login.

    Args:
        enabled: Whether to enable auto-start
        app_name: Name of the application

    Returns:
        True if successful, False otherwise.
    """
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
                import sys

                exe_path = sys.executable
                winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, f'"{exe_path}"')
            else:
                try:
                    winreg.DeleteValue(key, app_name)
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
        import sys

        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.{app_name.lower()}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>-m</string>
        <string>transcribe</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
"""
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_path.write_text(plist_content)
    else:
        plist_path.unlink(missing_ok=True)

    return True


def _set_autostart_linux(enabled: bool, app_name: str) -> bool:
    autostart_dir = Path.home() / ".config" / "autostart"
    desktop_file = autostart_dir / f"{app_name.lower()}.desktop"

    if enabled:
        import sys

        desktop_content = f"""[Desktop Entry]
Type=Application
Name={app_name}
Exec={sys.executable} -m transcribe
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
"""
        autostart_dir.mkdir(parents=True, exist_ok=True)
        desktop_file.write_text(desktop_content)
    else:
        desktop_file.unlink(missing_ok=True)

    return True


def get_app_icon_path() -> Optional[Path]:
    # TODO: Implement icon path resolution
    return None


def check_and_request_permissions(settings: "Settings") -> bool:
    """
    Check for required permissions and request them if needed (macOS only).

    On macOS, accessibility permissions are required for keyboard listening.
    Shows a dialog explaining permissions if not granted.
    Updates settings.accessibility_permissions_granted.

    Args:
        settings: Settings object to update with permission status

    Returns:
        True if permissions are granted, False otherwise.
    """
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
