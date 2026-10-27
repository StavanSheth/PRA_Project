"""
Creates or updates the Windows Desktop shortcut for Hand Gesture Studio.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VBS_PATH = os.path.join(BASE_DIR, "launcher.vbs")
ICON_PATH = os.path.join(BASE_DIR, "assets", "app_icon.ico")
DESKTOP_DIR = os.path.expandvars(r"%USERPROFILE%\Desktop")
SHORTCUT_PATH = os.path.join(DESKTOP_DIR, "Hand Gesture Studio.lnk")

# PowerShell command to create Windows .lnk shortcut via WScript.Shell
ps_script = f"""
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut('{SHORTCUT_PATH}')
$Shortcut.TargetPath = 'wscript.exe'
$Shortcut.Arguments = '"{VBS_PATH}"'
$Shortcut.WorkingDirectory = '{BASE_DIR}'
$Shortcut.IconLocation = '{ICON_PATH},0'
$Shortcut.Description = 'Live Hand Gesture Recognition Studio'
$Shortcut.Save()
"""

# Execute via subprocess using powershell
import subprocess

try:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        capture_output=True,
        text=True,
        check=True,
    )
    if os.path.exists(SHORTCUT_PATH):
        print(f"SUCCESS: Shortcut created at: {SHORTCUT_PATH}")
    else:
        print("ERROR: Shortcut file was not created.")
        sys.exit(1)
except Exception as e:
    print(f"Failed to create shortcut: {e}")
    sys.exit(1)
