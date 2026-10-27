"""
Desktop Application Launcher for Hand Gesture Studio.
Handles silent background server lifecycle, Chrome --app mode launch, and automatic webcam release on exit.
"""

import os
import sys
import time
import subprocess
import urllib.request
import webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_URL = "http://127.0.0.1:5000"
STATUS_URL = f"{SERVER_URL}/api/status"
PYTHON_EXE = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")
if not os.path.exists(PYTHON_EXE):
    PYTHON_EXE = sys.executable

CHROME_PATHS = [
    os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
]


def is_server_running():
    """Checks if Flask server is already healthy and responsive."""
    try:
        with urllib.request.urlopen(STATUS_URL, timeout=1) as response:
            return response.status == 200
    except Exception:
        return False


def start_server():
    """Starts server.py as a completely detached background process with no console window."""
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW

    server_script = os.path.join(BASE_DIR, "server.py")
    proc = subprocess.Popen(
        [PYTHON_EXE, server_script],
        cwd=BASE_DIR,
        creationflags=creation_flags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def find_chrome():
    """Returns path to chrome.exe if installed."""
    for path in CHROME_PATHS:
        if os.path.exists(path):
            return path
    return None


def main():
    server_proc = None

    # 1. Start server if not already running
    if not is_server_running():
        server_proc = start_server()

        # Wait up to 10 seconds for server to report ready
        for _ in range(50):
            time.sleep(0.2)
            if is_server_running():
                break

    # 2. Launch in Google Chrome App Mode
    chrome_exe = find_chrome()
    if chrome_exe:
        profile_dir = os.path.join(BASE_DIR, ".chrome_profile")
        os.makedirs(profile_dir, exist_ok=True)

        cmd = [
            chrome_exe,
            f"--app={SERVER_URL}",
            f"--user-data-dir={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-translate",
        ]

        # Launch Chrome and wait for user to close the app window
        try:
            chrome_proc = subprocess.Popen(cmd)
            chrome_proc.wait()
        except Exception:
            webbrowser.open(SERVER_URL)
    else:
        # Fallback to default browser
        webbrowser.open(SERVER_URL)
        if server_proc:
            # If opened in browser tab, keep server alive
            return

    # 3. Clean up server when Chrome window closes to release webcam
    if server_proc:
        try:
            server_proc.terminate()
            server_proc.wait(timeout=3)
        except Exception:
            try:
                server_proc.kill()
            except Exception:
                pass


if __name__ == "__main__":
    main()
