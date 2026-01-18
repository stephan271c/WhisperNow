import ctypes
import ctypes.util
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from .logger import get_logger

if TYPE_CHECKING:
    from ..core.settings import Settings

logger = get_logger(__name__)


def get_subprocess_kwargs(**extra: Any) -> Dict[str, Any]:
    kwargs = extra.copy()
    if platform.system() == "Windows":
        # Bitwise OR to preserve any existing creationflags
        kwargs["creationflags"] = (
            kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
        )

        if "stdin" not in kwargs:
            kwargs["stdin"] = subprocess.DEVNULL

        # Only suppress output if capture_output is NOT set
        if not kwargs.get("capture_output", False):
            kwargs.setdefault("stdout", subprocess.DEVNULL)
            kwargs.setdefault("stderr", subprocess.DEVNULL)
    return kwargs


def get_executable_path() -> str:
    system = get_platform()

    if system == "linux":
        appimage_path = os.environ.get("APPIMAGE")
        if appimage_path and Path(appimage_path).exists():
            logger.debug(f"Running as AppImage: {appimage_path}")
            return appimage_path

    if system == "macos":
        if getattr(sys, "frozen", False):
            exe_path = Path(sys.executable).resolve()
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


def _check_macos_accessibility(prompt: bool = False) -> bool:
    """
    Check if the app has accessibility permissions on macOS using ctypes.
    This avoids subprocess issues with frozen apps where sys.executable
    points to the app binary instead of a Python interpreter.

    Args:
        prompt: If True, triggers the system dialog requesting permission if not already granted.
    """
    if get_platform() != "macos":
        return True

    try:
        # Load ApplicationServices framework (contains AXIsProcessTrustedWithOptions)
        app_services_path = ctypes.util.find_library("ApplicationServices")
        if not app_services_path:
            logger.warning("ApplicationServices framework not found")
            return False

        app_services = ctypes.cdll.LoadLibrary(app_services_path)

        # AXIsProcessTrustedWithOptions(CFDictionaryRef options) -> Boolean
        # Passing NULL (None) checks without prompting
        app_services.AXIsProcessTrustedWithOptions.argtypes = [ctypes.c_void_p]
        app_services.AXIsProcessTrustedWithOptions.restype = ctypes.c_bool

        if not prompt:
            # Simple check without prompt - pass NULL
            return app_services.AXIsProcessTrustedWithOptions(None)
        else:
            # Check first without prompting
            trusted = app_services.AXIsProcessTrustedWithOptions(None)
            if not trusted:
                # Pure ctypes prompting requires CoreFoundation dict creation which is complex.
                # Fall back to opening System Preferences (handled by request_accessibility_permissions)
                pass
            return trusted

    except Exception as e:
        logger.warning(f"Accessibility check failed: {e}")
        return False


def check_accessibility_permissions() -> bool:
    return _check_macos_accessibility(prompt=False)


def get_permission_status() -> Dict[str, bool]:
    """
    Returns a dictionary of permission statuses for cleaner UI separation.
    The caller can use this to decide whether to show a permissions dialog.
    """
    return {
        "accessibility": check_accessibility_permissions(),
        "input_monitoring": check_input_monitoring_permissions(),
        "microphone": check_microphone_permissions(),
    }


def request_accessibility_permissions() -> None:
    if get_platform() != "macos":
        return

    # Trigger the prompt
    _check_macos_accessibility(prompt=True)

    # Also open System Settings as a backup/convenience helper
    subprocess.run(
        [
            "open",
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
        ],
        check=False,
    )


def check_input_monitoring_permissions() -> bool:
    """
    Check if the app has Input Monitoring permissions on macOS using ctypes.
    This avoids subprocess issues with frozen apps.
    """
    if get_platform() != "macos":
        return True

    # Constants from IOKit
    kIOHIDRequestTypeListenEvent = 1
    kIOHIDAccessTypeGranted = 0

    try:
        # Load IOKit framework
        iokit_path = ctypes.util.find_library("IOKit")
        if not iokit_path:
            logger.warning("IOKit framework not found")
            return False

        iokit = ctypes.cdll.LoadLibrary(iokit_path)

        # IOHIDCheckAccess(IOHIDRequestType requestType) -> IOHIDAccessType
        iokit.IOHIDCheckAccess.argtypes = [ctypes.c_uint32]
        iokit.IOHIDCheckAccess.restype = ctypes.c_uint32

        access_type = iokit.IOHIDCheckAccess(kIOHIDRequestTypeListenEvent)
        return access_type == kIOHIDAccessTypeGranted

    except Exception as e:
        logger.debug(f"Input monitoring check failed: {e}")
        return False


def request_input_monitoring_permissions() -> None:

    if get_platform() != "macos":
        return

    subprocess.run(
        [
            "open",
            "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent",
        ],
        check=False,
    )


def check_microphone_permissions() -> bool:
    if get_platform() != "macos":
        return True

    try:
        import sounddevice as sd

        # Check for physical devices first
        devices = sd.query_devices()
        has_inputs = any(d.get("max_input_channels", 0) > 0 for d in devices)
        if not has_inputs:
            logger.warning("Microphone permission check: no input devices found")
            return False

        # Attempt to open a stream briefly to verify actual permission.
        # On macOS, querying devices doesn't always reflect privacy settings.
        try:
            with sd.InputStream(channels=1, samplerate=16000):
                pass
            logger.debug("Microphone permission check: stream opened successfully")
            return True
        except Exception:
            logger.warning(
                "Input devices found but failed to open stream (permission denied?)"
            )
            return False

    except ImportError:
        logger.error("sounddevice library not found")
        return False
    except Exception as e:
        logger.warning(f"Microphone permission check failed: {e}")
        return False


def request_microphone_permissions() -> None:

    if get_platform() != "macos":
        return

    subprocess.run(
        [
            "open",
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
        ],
        check=False,
    )


def set_autostart(enabled: bool, app_name: str = "WhisperNow") -> bool:

    system = get_platform()

    if system == "windows":
        return _set_autostart_windows(enabled, app_name)
    elif system == "macos":
        return _set_autostart_macos(enabled, app_name)
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
        try:
            plist_path.parent.mkdir(parents=True, exist_ok=True)
            plist_path.write_text(plist_content)
            logger.info(f"macOS autostart enabled: {plist_path}")
        except OSError as e:
            logger.error(f"Failed to write LaunchAgent: {e}")
            return False
    else:
        try:
            plist_path.unlink(missing_ok=True)
            logger.info("macOS autostart disabled")
        except OSError as e:
            logger.error(f"Failed to remove LaunchAgent: {e}")
            return False

    return True


def _set_autostart_linux(enabled: bool, app_name: str) -> bool:
    autostart_dir = Path.home() / ".config" / "autostart"
    desktop_file = autostart_dir / f"{app_name.lower()}.desktop"

    if enabled:
        exe_path = get_executable_path()
        packaged = is_packaged()

        # Quote paths to handle spaces in directory names
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


def get_app_icon_path() -> Optional[Path]:
    return None


def check_and_request_permissions(settings: "Settings") -> bool:

    if get_platform() != "macos":
        return True

    if settings.accessibility_permissions_granted:
        if check_accessibility_permissions():
            pass
        else:
            logger.warning("Accessibility permission was revoked, prompting user")

    accessibility_granted = False
    input_monitoring_granted = False
    if check_accessibility_permissions() and check_input_monitoring_permissions():
        settings.accessibility_permissions_granted = True
        settings.save()
        logger.info("Accessibility and Input Monitoring permissions already granted")
        accessibility_granted = True
        input_monitoring_granted = True
    else:
        from ..ui.permissions_dialog import PermissionsDialog

        logger.info("Showing permissions dialog")
        dialog = PermissionsDialog()
        dialog.exec()

        accessibility_granted = check_accessibility_permissions()
        input_monitoring_granted = check_input_monitoring_permissions()
        settings.accessibility_permissions_granted = accessibility_granted
        settings.save()

        if accessibility_granted and input_monitoring_granted:
            logger.info("User granted Accessibility and Input Monitoring permissions")
        else:
            if not accessibility_granted:
                logger.warning("User continued without accessibility permissions")
            if not input_monitoring_granted:
                logger.warning("User continued without Input Monitoring permissions")

    if not check_microphone_permissions():
        logger.warning(
            "Microphone permission not granted. Recording may fail. "
            "Grant access in System Settings > Privacy & Security > Microphone"
        )

    return accessibility_granted and input_monitoring_granted
