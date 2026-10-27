"""
Creates or updates the Windows Desktop shortcut for Hand Gesture Studio across all active Desktop locations.
"""

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VBS_PATH = os.path.join(BASE_DIR, "launcher.vbs")
ICON_PATH = os.path.join(BASE_DIR, "assets", "app_icon.ico")

# Find true Windows Desktop folders (including OneDrive redirection and standard Desktop)
ps_get_desktop = """
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$dirs = @(
    [Environment]::GetFolderPath('Desktop'),
    "$env:USERPROFILE\\OneDrive\\Desktop",
    "$env:USERPROFILE\\Desktop",
    [Environment]::GetFolderPath('CommonDesktopDirectory')
)
$dirs | Where-Object { Test-Path $_ } | Select-Object -Unique
"""

res = subprocess.run(
    ["powershell", "-NoProfile", "-Command", ps_get_desktop],
    capture_output=True,
    text=True,
    check=True,
)

desktop_dirs = [line.strip() for line in res.stdout.splitlines() if line.strip() and os.path.exists(line.strip())]

if not desktop_dirs:
    desktop_dirs = [os.path.expandvars(r"%USERPROFILE%\Desktop")]

created_paths = []
for d in desktop_dirs:
    shortcut_path = os.path.join(d, "Hand Gesture Studio.lnk")
    ps_create = f"""
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut('{shortcut_path}')
    $Shortcut.TargetPath = 'wscript.exe'
    $Shortcut.Arguments = '"{VBS_PATH}"'
    $Shortcut.WorkingDirectory = '{BASE_DIR}'
    $Shortcut.IconLocation = '{ICON_PATH},0'
    $Shortcut.Description = 'Live Hand Gesture Recognition Studio'
    $Shortcut.Save()
    """
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_create], check=True)
        if os.path.exists(shortcut_path):
            created_paths.append(shortcut_path)
    except Exception as e:
        print(f"Warning: Could not create shortcut in {d}: {e}")

print("SUCCESS: Shortcut placed in active desktop folders:")
for p in created_paths:
    print(f" - {p}")
