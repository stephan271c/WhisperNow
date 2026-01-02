"""
Platform-specific utilities for cross-platform compatibility.
"""

import platform
import subprocess
from pathlib import Path
from typing import Optional


def get_platform() -> str:
    """Get the current platform name."""
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
    
    # TODO: Implement macOS accessibility check
    # Could use: osascript -e 'tell application "System Events" to keystroke ""'
    # and check for permission error
    return True


def request_accessibility_permissions() -> None:
    """
    Guide the user to grant accessibility permissions (macOS only).
    
    Opens System Preferences to the correct pane.
    """
    if get_platform() != "macos":
        return
    
    # Open System Preferences > Security & Privacy > Privacy > Accessibility
    subprocess.run([
        "open",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
    ], check=False)


def set_autostart(enabled: bool, app_name: str = "Transcribe") -> bool:
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
    """Set autostart on Windows via registry."""
    try:
        import winreg
        
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
            if enabled:
                # Get the path to the current executable
                import sys
                exe_path = sys.executable
                winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, f'"{exe_path}"')
            else:
                try:
                    winreg.DeleteValue(key, app_name)
                except FileNotFoundError:
                    pass  # Already not set
        
        return True
    except Exception as e:
        print(f"Failed to set autostart: {e}")
        return False


def _set_autostart_macos(enabled: bool, app_name: str) -> bool:
    """Set autostart on macOS via LaunchAgents."""
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"com.{app_name.lower()}.plist"
    
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
    """Set autostart on Linux via XDG autostart."""
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
    """Get the path to the application icon."""
    # TODO: Implement icon path resolution
    # Should check for bundled resources in PyInstaller, etc.
    return None
