#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent

HTML_FILE = (
    SCRIPT_DIR
    / "outputs"
    / "interactive"
    / "cross_check_weight0_interactive.html"
)

if not HTML_FILE.is_file():
    print("Interactive HTML file not found:")
    print(f"  {HTML_FILE}")
    print()
    print("Run cross_check_weight0.py first.")
    sys.exit(1)

# Convert the WSL path to a Windows path.
result = subprocess.run(
    ["wslpath", "-w", str(HTML_FILE)],
    check=True,
    capture_output=True,
    text=True,
)

windows_path = result.stdout.strip()

print("Opening interactive cross-check plot:")
print(f"  {HTML_FILE}")

# Open in the normal Windows default browser.
subprocess.run(
    ["cmd.exe", "/C", "start", "", windows_path],
    check=True,
)
