#!/usr/bin/env python3

# ============================================================
# run_etalepton_interactive.py
#
# Purpose:
#   1. Regenerate BOTH interactive eta-lepton dashboards
#   2. Open BOTH resulting HTML files automatically
#
# Outputs opened:
#
#   etalepton_correction_factor_interactive_2x2.html
#   etalepton_uncertainties_interactive_2x2.html
#
# Run with:
#
#   python3 lepton_pseudorapidity_work/run_etalepton_interactive.py
#
# ============================================================

from pathlib import Path
import subprocess
import sys


# ============================================================
# Locations
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PLOT_SCRIPT = (
    SCRIPT_DIR
    / "plot_csv_etalepton_interactive.py"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "plot_csv_etalepton_interactive_outputs"
)

CORRECTION_HTML = (
    OUTPUT_DIR
    / "etalepton_correction_factor_interactive_2x2.html"
)

UNCERTAINTY_HTML = (
    OUTPUT_DIR
    / "etalepton_uncertainties_interactive_2x2.html"
)


# ============================================================
# Run the plotting script
# ============================================================

def generate_dashboards():

    print()
    print("==============================================")
    print(" Generating interactive eta dashboards")
    print("==============================================")
    print()

    if not PLOT_SCRIPT.exists():

        raise FileNotFoundError(
            "Could not find plotting script:\n"
            f"  {PLOT_SCRIPT}"
        )

    result = subprocess.run(
        [
            sys.executable,
            str(PLOT_SCRIPT),
        ]
    )

    if result.returncode != 0:

        raise RuntimeError(
            "The interactive plotting script failed.\n"
            "The browser will not be opened."
        )


    # --------------------------------------------------------
    # Check both output files exist
    # --------------------------------------------------------

    missing_files = []

    if not CORRECTION_HTML.exists():

        missing_files.append(
            CORRECTION_HTML
        )

    if not UNCERTAINTY_HTML.exists():

        missing_files.append(
            UNCERTAINTY_HTML
        )


    if missing_files:

        message = (
            "The plotting script finished, but the following "
            "HTML file(s) could not be found:\n"
        )

        for path in missing_files:

            message += (
                f"\n  {path}"
            )

        raise FileNotFoundError(
            message
        )


# ============================================================
# Convert WSL path -> Windows path
# ============================================================

def get_windows_path(path):

    result = subprocess.run(
        [
            "wslpath",
            "-w",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


# ============================================================
# Open one HTML file in Windows
# ============================================================

def open_html_file(html_path):

    windows_path = get_windows_path(
        html_path
    )

    safe_path = windows_path.replace(
        "'",
        "''",
    )

    command = (
        f"Start-Process -FilePath '{safe_path}'"
    )

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            command,
        ]
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Windows could not automatically open:\n"
            f"  {html_path}"
        )


# ============================================================
# Open both dashboards
# ============================================================

def open_dashboards():

    print()
    print("==============================================")
    print(" Opening dashboards in browser")
    print("==============================================")
    print()

    print("Opening correction-factor dashboard:")
    print(f"  {CORRECTION_HTML}")
    print()

    open_html_file(
        CORRECTION_HTML
    )

    print("Opening uncertainty dashboard:")
    print(f"  {UNCERTAINTY_HTML}")
    print()

    open_html_file(
        UNCERTAINTY_HTML
    )


# ============================================================
# Main
# ============================================================

def main():

    generate_dashboards()

    open_dashboards()

    print()
    print("==============================================")
    print(" Ready")
    print("==============================================")
    print()

    print(
        "Both interactive dashboards should now be "
        "open in your default browser."
    )

    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()