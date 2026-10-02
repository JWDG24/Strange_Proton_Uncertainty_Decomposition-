#!/usr/bin/env python3

# ============================================================
# MTWcut_run_etalepton_interactive.py
#
# Purpose:
#   1. Regenerate BOTH interactive eta-lepton dashboards
#      using the m_T^W > 40 GeV correction-factor dataset
#   2. Open BOTH resulting HTML files automatically
#
# Outputs opened:
#
#   MTWcut_etalepton_correction_factor_interactive_2x2.html
#   MTWcut_etalepton_uncertainties_interactive_2x2.html
#
# Run with:
#
#   python3 lepton_pseudorapidity_work/MTWcut_run_etalepton_interactive.py
#
# ============================================================

from pathlib import Path
import subprocess
import sys


# ============================================================
# Locations
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent


# Dedicated transverse-mass-cut plotting script.
PLOT_SCRIPT = (
    SCRIPT_DIR
    / "MTWcut_plot_csv_etalepton_interactive.py"
)


# Dedicated transverse-mass-cut interactive output folder.
#
# This keeps the m_T^W > 40 GeV plots completely separate
# from the original no-m_T-cut dashboards.
OUTPUT_DIR = (
    SCRIPT_DIR
    / "MTWcut_plot_csv_etalepton_interactive_outputs"
)


# ============================================================
# Expected transverse-mass-cut HTML outputs
# ============================================================

CORRECTION_HTML = (
    OUTPUT_DIR
    / "MTWcut_etalepton_correction_factor_interactive_2x2.html"
)

UNCERTAINTY_HTML = (
    OUTPUT_DIR
    / "MTWcut_etalepton_uncertainties_interactive_2x2.html"
)


# ============================================================
# Run the transverse-mass-cut plotting script
# ============================================================

def generate_dashboards():

    print()
    print("============================================================")
    print(" Generating interactive eta dashboards: m_T^W > 40 GeV")
    print("============================================================")
    print()

    print("Plotting script:")
    print(f"  {PLOT_SCRIPT}")
    print()

    print("Output directory:")
    print(f"  {OUTPUT_DIR}")
    print()

    if not PLOT_SCRIPT.exists():

        raise FileNotFoundError(
            "Could not find MTW-cut plotting script:\n"
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
            "The MTW-cut interactive plotting script failed.\n"
            "The browser will not be opened."
        )


    # --------------------------------------------------------
    # Check both expected output files exist
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
            "MTW-cut HTML file(s) could not be found:\n"
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
# Open both transverse-mass-cut dashboards
# ============================================================

def open_dashboards():

    print()
    print("============================================================")
    print(" Opening m_T^W > 40 GeV dashboards in browser")
    print("============================================================")
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
    print("============================================================")
    print(" Ready")
    print("============================================================")
    print()

    print(
        "Both m_T^W > 40 GeV interactive dashboards "
        "should now be open in your default browser."
    )

    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
