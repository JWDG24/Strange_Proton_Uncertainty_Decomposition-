#!/usr/bin/env python3

"""
apply_relative_uncertainties_to_new_nominal.py

Supervisor part (b):

Use the NEW high-statistics weight-0 Pythia correction factors as the
nominal correction and apply the RELATIVE systematic uncertainties
extracted in part (a) from the OLD weighted samples.

This script does not reprocess the ROOT event files. It combines the
outputs of the two analyses already completed:

1. New high-statistics nominal correction and statistical uncertainty:
   lepton_pseudorapidity_work/
   └── cross_check_weight0/
       └── outputs/
           └── csv/
               ├── cross_check_Pythia_plus.csv
               └── cross_check_Pythia_minus.csv

2. Relative systematic uncertainties from the old weighted samples:
   lepton_pseudorapidity_work/
   └── weights_relative_uncertainties/
       └── outputs/
           └── csv/
               ├── Pythia_plus_relative_uncertainties.csv
               └── Pythia_minus_relative_uncertainties.csv

For each eta bin the final nominal correction is

    C_final = C_new

The statistical uncertainty is taken directly from the new
high-statistics correction:

    sigma_stat = sigma_stat,new

The systematic uncertainties are transferred from the relative
uncertainties obtained in part (a):

    sigma_scale  = |C_new| * delta_scale
    sigma_PDF    = |C_new| * delta_PDF
    sigma_shower = |C_new| * delta_shower
    sigma_model  = |C_new| * delta_model

The total systematic uncertainty is

    sigma_syst =
        sqrt(
            sigma_scale^2
            + sigma_PDF^2
            + sigma_shower^2
            + sigma_model^2
        )

and the total uncertainty is

    sigma_total =
        sqrt(
            sigma_stat^2
            + sigma_syst^2
        )

Only Pythia W+ and Pythia W- are produced because the new high-statistics
cross-check samples are Pythia-only.

Expected location:

uncertainty_decomposition/
└── lepton_pseudorapidity_work/
    └── weights_relative_uncertainties/
        ├── build_relative_uncertainties.py
        ├── plot_relative_uncertainties_interactive.py
        └── apply_relative_uncertainties_to_new_nominal.py

Outputs are written to:

weights_relative_uncertainties/
└── outputs/
    └── final_new_nominal/
        ├── csv/
        ├── root/
        └── interactive/
"""

import argparse
import csv
import math
import subprocess
from array import array
from pathlib import Path

import ROOT


# ============================================================
# ROOT settings
# ============================================================

ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
LEPTON_WORK_DIR = SCRIPT_DIR.parent

# Part (a) output
RELATIVE_CSV_DIR = (
    SCRIPT_DIR
    / "outputs"
    / "csv"
)

# New high-statistics nominal output
CROSS_CHECK_CSV_DIR = (
    LEPTON_WORK_DIR
    / "cross_check_weight0"
    / "outputs"
    / "csv"
)

# Final part (b) outputs
FINAL_OUTPUT_DIR = (
    SCRIPT_DIR
    / "outputs"
    / "final_new_nominal"
)

FINAL_CSV_DIR = (
    FINAL_OUTPUT_DIR
    / "csv"
)

FINAL_ROOT_DIR = (
    FINAL_OUTPUT_DIR
    / "root"
)

FINAL_INTERACTIVE_DIR = (
    FINAL_OUTPUT_DIR
    / "interactive"
)

for directory in (
    FINAL_CSV_DIR,
    FINAL_ROOT_DIR,
    FINAL_INTERACTIVE_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Channels
# ============================================================

CHANNELS = {
    "Pythia_plus": {
        "display": "Pythia W+",
        "new_csv": (
            CROSS_CHECK_CSV_DIR
            / "cross_check_Pythia_plus.csv"
        ),
        "relative_csv": (
            RELATIVE_CSV_DIR
            / "Pythia_plus_relative_uncertainties.csv"
        ),
    },
    "Pythia_minus": {
        "display": "Pythia W-",
        "new_csv": (
            CROSS_CHECK_CSV_DIR
            / "cross_check_Pythia_minus.csv"
        ),
        "relative_csv": (
            RELATIVE_CSV_DIR
            / "Pythia_minus_relative_uncertainties.csv"
        ),
    },
}


# ============================================================
# CSV helpers
# ============================================================

NEW_REQUIRED_COLUMNS = {
    "bin",
    "bin_low_edge",
    "bin_up_edge",
    "C_new_crosscheck",
    "new_crosscheck_stat_unc",
}

RELATIVE_REQUIRED_COLUMNS = {
    "bin",
    "bin_low_edge",
    "bin_up_edge",
    "relative_scale_unc",
    "relative_pdf_unc",
    "relative_shower_unc",
    "relative_model_unc",
    "relative_total_systematic_unc",
}


def read_csv_rows(
    path,
    required_columns,
):
    """
    Read a CSV into a list of dictionaries and verify the schema.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Required CSV file does not exist:\n"
            f"  {path}"
        )

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        reader = csv.DictReader(
            csvfile
        )

        fieldnames = set(
            reader.fieldnames
            or []
        )

        missing = (
            required_columns
            - fieldnames
        )

        if missing:
            raise RuntimeError(
                f"CSV file is missing required columns:\n"
                f"  {path}\n"
                f"Missing: {sorted(missing)}"
            )

        return list(
            reader
        )


def convert_new_rows(
    rows,
):
    """
    Convert relevant fields from the new-sample CSV to numeric values.
    """

    converted = []

    for row in rows:

        converted.append({
            "bin": int(
                row[
                    "bin"
                ]
            ),
            "bin_low_edge": float(
                row[
                    "bin_low_edge"
                ]
            ),
            "bin_up_edge": float(
                row[
                    "bin_up_edge"
                ]
            ),
            "C_new": float(
                row[
                    "C_new_crosscheck"
                ]
            ),
            "stat_unc_new": float(
                row[
                    "new_crosscheck_stat_unc"
                ]
            ),
        })

    return converted


def convert_relative_rows(
    rows,
):
    """
    Convert relevant fields from the part (a) CSV to numeric values.
    """

    converted = []

    for row in rows:

        converted.append({
            "bin": int(
                row[
                    "bin"
                ]
            ),
            "bin_low_edge": float(
                row[
                    "bin_low_edge"
                ]
            ),
            "bin_up_edge": float(
                row[
                    "bin_up_edge"
                ]
            ),
            "delta_scale": float(
                row[
                    "relative_scale_unc"
                ]
            ),
            "delta_pdf": float(
                row[
                    "relative_pdf_unc"
                ]
            ),
            "delta_shower": float(
                row[
                    "relative_shower_unc"
                ]
            ),
            "delta_model": float(
                row[
                    "relative_model_unc"
                ]
            ),
            "delta_total_syst": float(
                row[
                    "relative_total_systematic_unc"
                ]
            ),
        })

    return converted


# ============================================================
# Binning validation
# ============================================================

def validate_matching_binning(
    new_rows,
    relative_rows,
):
    """
    Ensure the new nominal result and the relative uncertainties refer
    to exactly the same eta bins.
    """

    if len(
        new_rows
    ) != len(
        relative_rows
    ):
        raise RuntimeError(
            "The new nominal and relative uncertainty files have "
            "different numbers of eta bins."
        )

    tolerance = 1.0e-10

    for new_row, relative_row in zip(
        new_rows,
        relative_rows,
    ):

        if (
            new_row[
                "bin"
            ]
            != relative_row[
                "bin"
            ]
        ):
            raise RuntimeError(
                "Bin index mismatch between new nominal and "
                "relative uncertainty files."
            )

        if (
            abs(
                new_row[
                    "bin_low_edge"
                ]
                - relative_row[
                    "bin_low_edge"
                ]
            ) > tolerance
            or
            abs(
                new_row[
                    "bin_up_edge"
                ]
                - relative_row[
                    "bin_up_edge"
                ]
            ) > tolerance
        ):
            raise RuntimeError(
                "Eta bin-edge mismatch between new nominal and "
                "relative uncertainty files."
            )


# ============================================================
# Final uncertainty calculation
# ============================================================

def build_final_rows(
    channel,
    new_rows,
    relative_rows,
):
    """
    Apply the relative systematic uncertainties from part (a) to the
    new high-statistics nominal correction.

    All returned uncertainties are absolute positive magnitudes.
    """

    validate_matching_binning(
        new_rows,
        relative_rows,
    )

    final_rows = []

    for new_row, relative_row in zip(
        new_rows,
        relative_rows,
    ):

        correction = float(
            new_row[
                "C_new"
            ]
        )

        correction_magnitude = abs(
            correction
        )

        stat_unc = abs(
            float(
                new_row[
                    "stat_unc_new"
                ]
            )
        )

        delta_scale = abs(
            relative_row[
                "delta_scale"
            ]
        )

        delta_pdf = abs(
            relative_row[
                "delta_pdf"
            ]
        )

        delta_shower = abs(
            relative_row[
                "delta_shower"
            ]
        )

        delta_model = abs(
            relative_row[
                "delta_model"
            ]
        )

        scale_unc = (
            correction_magnitude
            * delta_scale
        )

        pdf_unc = (
            correction_magnitude
            * delta_pdf
        )

        shower_unc = (
            correction_magnitude
            * delta_shower
        )

        model_unc = (
            correction_magnitude
            * delta_model
        )

        systematic_unc = math.sqrt(
            scale_unc * scale_unc
            + pdf_unc * pdf_unc
            + shower_unc * shower_unc
            + model_unc * model_unc
        )

        total_unc = math.sqrt(
            stat_unc * stat_unc
            + systematic_unc * systematic_unc
        )

        if correction_magnitude > 0.0:

            stat_percent = (
                100.0
                * stat_unc
                / correction_magnitude
            )

            systematic_percent = (
                100.0
                * systematic_unc
                / correction_magnitude
            )

            total_percent = (
                100.0
                * total_unc
                / correction_magnitude
            )

        else:

            stat_percent = 0.0
            systematic_percent = 0.0
            total_percent = 0.0

        final_rows.append({
            "channel": channel,
            "bin": (
                new_row[
                    "bin"
                ]
            ),
            "bin_low_edge": (
                new_row[
                    "bin_low_edge"
                ]
            ),
            "bin_up_edge": (
                new_row[
                    "bin_up_edge"
                ]
            ),
            "nominal_correction_new": (
                correction
            ),
            "stat_unc_new": (
                stat_unc
            ),
            "relative_scale_unc": (
                delta_scale
            ),
            "relative_pdf_unc": (
                delta_pdf
            ),
            "relative_shower_unc": (
                delta_shower
            ),
            "relative_model_unc": (
                delta_model
            ),
            "scale_unc": (
                scale_unc
            ),
            "pdf_unc": (
                pdf_unc
            ),
            "shower_unc": (
                shower_unc
            ),
            "model_unc": (
                model_unc
            ),
            "systematic_unc": (
                systematic_unc
            ),
            "total_unc": (
                total_unc
            ),
            "stat_unc_percent": (
                stat_percent
            ),
            "scale_unc_percent": (
                100.0
                * delta_scale
            ),
            "pdf_unc_percent": (
                100.0
                * delta_pdf
            ),
            "shower_unc_percent": (
                100.0
                * delta_shower
            ),
            "model_unc_percent": (
                100.0
                * delta_model
            ),
            "systematic_unc_percent": (
                systematic_percent
            ),
            "total_unc_percent": (
                total_percent
            ),
        })

    return final_rows


# ============================================================
# CSV output
# ============================================================

FINAL_COLUMNS = [
    "channel",
    "bin",
    "bin_low_edge",
    "bin_up_edge",
    "nominal_correction_new",
    "stat_unc_new",
    "relative_scale_unc",
    "relative_pdf_unc",
    "relative_shower_unc",
    "relative_model_unc",
    "scale_unc",
    "pdf_unc",
    "shower_unc",
    "model_unc",
    "systematic_unc",
    "total_unc",
    "stat_unc_percent",
    "scale_unc_percent",
    "pdf_unc_percent",
    "shower_unc_percent",
    "model_unc_percent",
    "systematic_unc_percent",
    "total_unc_percent",
]


def write_final_csv(
    channel,
    final_rows,
):
    """
    Write the final correction and complete uncertainty decomposition.
    """

    output_path = (
        FINAL_CSV_DIR
        / f"{channel}_final_new_nominal.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=FINAL_COLUMNS,
        )

        writer.writeheader()

        writer.writerows(
            final_rows
        )

    return output_path


# ============================================================
# ROOT output
# ============================================================

def make_histogram(
    name,
    title,
    bin_edges,
):
    """
    Create a detached variable-bin-width ROOT histogram.
    """

    edge_array = array(
        "d",
        bin_edges,
    )

    histogram = ROOT.TH1F(
        name,
        title,
        len(
            edge_array
        ) - 1,
        edge_array,
    )

    histogram.SetDirectory(0)

    return histogram


def write_root_output(
    channel,
    display,
    final_rows,
):
    """
    Store the final nominal correction and each uncertainty component.
    """

    bin_edges = [
        final_rows[
            0
        ][
            "bin_low_edge"
        ]
    ]

    bin_edges.extend(
        row[
            "bin_up_edge"
        ]
        for row in final_rows
    )

    h_nominal = make_histogram(
        f"{channel}_nominal",
        (
            f"{display};"
            f"|#eta_{{#ell}}|;"
            f"Correction factor"
        ),
        bin_edges,
    )

    h_stat = make_histogram(
        f"{channel}_stat",
        "Statistical uncertainty",
        bin_edges,
    )

    h_scale = make_histogram(
        f"{channel}_scale",
        "Scale uncertainty",
        bin_edges,
    )

    h_pdf = make_histogram(
        f"{channel}_pdf",
        "PDF uncertainty",
        bin_edges,
    )

    h_shower = make_histogram(
        f"{channel}_shower",
        "Shower uncertainty",
        bin_edges,
    )

    h_model = make_histogram(
        f"{channel}_model",
        "Model uncertainty",
        bin_edges,
    )

    h_systematic = make_histogram(
        f"{channel}_systematic",
        "Total systematic uncertainty",
        bin_edges,
    )

    h_total = make_histogram(
        f"{channel}_total",
        "Total uncertainty",
        bin_edges,
    )

    for row in final_rows:

        ibin = row[
            "bin"
        ]

        h_nominal.SetBinContent(
            ibin,
            row[
                "nominal_correction_new"
            ],
        )

        # Put the FINAL total uncertainty on the nominal histogram.
        h_nominal.SetBinError(
            ibin,
            row[
                "total_unc"
            ],
        )

        h_stat.SetBinContent(
            ibin,
            row[
                "stat_unc_new"
            ],
        )

        h_scale.SetBinContent(
            ibin,
            row[
                "scale_unc"
            ],
        )

        h_pdf.SetBinContent(
            ibin,
            row[
                "pdf_unc"
            ],
        )

        h_shower.SetBinContent(
            ibin,
            row[
                "shower_unc"
            ],
        )

        h_model.SetBinContent(
            ibin,
            row[
                "model_unc"
            ],
        )

        h_systematic.SetBinContent(
            ibin,
            row[
                "systematic_unc"
            ],
        )

        h_total.SetBinContent(
            ibin,
            row[
                "total_unc"
            ],
        )

    output_path = (
        FINAL_ROOT_DIR
        / f"{channel}_final_new_nominal.root"
    )

    output_file = ROOT.TFile(
        str(
            output_path
        ),
        "RECREATE",
    )

    ROOT.TNamed(
        "Channel",
        channel,
    ).Write()

    ROOT.TNamed(
        "NominalDefinition",
        (
            "new high-statistics weight-0 Pythia "
            "parton-to-particle correction"
        ),
    ).Write()

    ROOT.TNamed(
        "SystematicDefinition",
        (
            "relative systematics from old weighted samples "
            "multiplied by new nominal correction"
        ),
    ).Write()

    h_nominal.Write(
        "final_nominal_correction"
    )

    h_stat.Write(
        "unc_stat"
    )

    h_scale.Write(
        "unc_scale"
    )

    h_pdf.Write(
        "unc_pdf"
    )

    h_shower.Write(
        "unc_shower"
    )

    h_model.Write(
        "unc_model"
    )

    h_systematic.Write(
        "unc_systematic_total"
    )

    h_total.Write(
        "unc_total"
    )

    output_file.Write()
    output_file.Close()

    return output_path


# ============================================================
# Optional interactive plot
# ============================================================

def save_interactive_plot(
    completed_channels,
):
    """
    Build one interactive Plotly HTML with a dropdown for Pythia W+
    and Pythia W-.

    Top panel:
        new nominal correction with total uncertainty
        new nominal correction with statistical uncertainty only

    Bottom panel:
        statistical, scale, PDF, shower, model and total uncertainty
        in percent.
    """

    if not completed_channels:
        return None

    try:

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

    except ImportError:

        print()
        print(
            "Plotly is not installed, so the interactive HTML "
            "will be skipped."
        )

        return None

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Final correction factor",
            "Final uncertainty decomposition",
        ),
        row_heights=[
            0.58,
            0.42,
        ],
    )

    channel_keys = list(
        completed_channels.keys()
    )

    # 8 traces per channel:
    # 2 top
    # 6 bottom
    traces_per_channel = 8

    for channel_index, channel in enumerate(
        channel_keys
    ):

        result = completed_channels[
            channel
        ]

        rows = result[
            "rows"
        ]

        visible = (
            channel_index == 0
        )

        x = [
            0.5
            * (
                row[
                    "bin_low_edge"
                ]
                + row[
                    "bin_up_edge"
                ]
            )
            for row in rows
        ]

        xerr = [
            0.5
            * (
                row[
                    "bin_up_edge"
                ]
                - row[
                    "bin_low_edge"
                ]
            )
            for row in rows
        ]

        nominal = [
            row[
                "nominal_correction_new"
            ]
            for row in rows
        ]

        # ----------------------------------------------------
        # Top: statistical-only errors
        # ----------------------------------------------------

        figure.add_trace(
            go.Scatter(
                x=x,
                y=nominal,
                mode="lines+markers",
                name="New nominal · stat only",
                visible=visible,
                error_x=dict(
                    type="data",
                    array=xerr,
                    visible=True,
                ),
                error_y=dict(
                    type="data",
                    array=[
                        row[
                            "stat_unc_new"
                        ]
                        for row in rows
                    ],
                    visible=True,
                ),
                hovertemplate=(
                    "|ηℓ| = %{x:.3f}"
                    "<br>C = %{y:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        # ----------------------------------------------------
        # Top: final total errors
        # ----------------------------------------------------

        figure.add_trace(
            go.Scatter(
                x=x,
                y=nominal,
                mode="markers",
                name="New nominal · total uncertainty",
                visible=visible,
                error_y=dict(
                    type="data",
                    array=[
                        row[
                            "total_unc"
                        ]
                        for row in rows
                    ],
                    visible=True,
                ),
                marker=dict(
                    symbol="circle-open",
                    size=9,
                ),
                hovertemplate=(
                    "|ηℓ| = %{x:.3f}"
                    "<br>C = %{y:.6f}"
                    "<br>Total uncertainty = "
                    "%{error_y.array:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        # ----------------------------------------------------
        # Bottom: uncertainty components in percent
        # ----------------------------------------------------

        component_definitions = [
            (
                "stat_unc_percent",
                "Statistical",
            ),
            (
                "scale_unc_percent",
                "Scale",
            ),
            (
                "pdf_unc_percent",
                "PDF",
            ),
            (
                "shower_unc_percent",
                "Shower",
            ),
            (
                "model_unc_percent",
                "Model",
            ),
            (
                "total_unc_percent",
                "Total",
            ),
        ]

        for column, label in component_definitions:

            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=[
                        row[
                            column
                        ]
                        for row in rows
                    ],
                    mode="lines+markers",
                    name=label,
                    visible=visible,
                    hovertemplate=(
                        "|ηℓ| = %{x:.3f}"
                        f"<br>{label}: %{{y:.4f}}%"
                        "<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

    total_traces = (
        traces_per_channel
        * len(
            channel_keys
        )
    )

    buttons = []

    for channel_index, channel in enumerate(
        channel_keys
    ):

        visible_mask = [
            False
        ] * total_traces

        first = (
            channel_index
            * traces_per_channel
        )

        for trace_index in range(
            first,
            first + traces_per_channel,
        ):
            visible_mask[
                trace_index
            ] = True

        display = (
            completed_channels[
                channel
            ][
                "display"
            ]
        )

        buttons.append(
            dict(
                label=display,
                method="update",
                args=[
                    {
                        "visible": visible_mask,
                    },
                    {
                        "title.text": (
                            "Final correction using new high-statistics nominal"
                            f"<br><sup>{display}</sup>"
                        ),
                    },
                ],
            )
        )

    first_display = (
        completed_channels[
            channel_keys[
                0
            ]
        ][
            "display"
        ]
    )

    figure.update_layout(
        title=dict(
            text=(
                "Final correction using new high-statistics nominal"
                f"<br><sup>{first_display}</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        height=900,
        width=1100,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.16,
                yanchor="top",
            )
        ],
        margin=dict(
            l=90,
            r=40,
            t=150,
            b=70,
        ),
    )

    figure.update_yaxes(
        title_text="Correction factor",
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text="Uncertainty (%)",
        rangemode="tozero",
        row=2,
        col=1,
    )

    figure.update_xaxes(
        range=[
            0.0,
            2.5,
        ],
        row=1,
        col=1,
    )

    figure.update_xaxes(
        title_text="Lepton |η|",
        range=[
            0.0,
            2.5,
        ],
        row=2,
        col=1,
    )

    output_path = (
        FINAL_INTERACTIVE_DIR
        / "final_new_nominal_uncertainties.html"
    )

    figure.write_html(
        str(
            output_path
        ),
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )

    return output_path


# ============================================================
# Optional WSL browser opening
# ============================================================

def open_in_windows_browser(
    path,
):
    """
    Open the final HTML in the default Windows browser from WSL.
    """

    try:

        result = subprocess.run(
            [
                "wslpath",
                "-w",
                str(
                    path
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        windows_path = (
            result.stdout.strip()
        )

        subprocess.run(
            [
                "cmd.exe",
                "/C",
                "start",
                "",
                windows_path,
            ],
            check=True,
        )

    except Exception as exc:

        print()
        print(
            "Could not automatically open the browser:"
        )
        print(
            f"  {exc}"
        )
        print()
        print(
            "Open the HTML file manually."
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Apply relative systematics from the old weighted "
            "samples to the new high-statistics nominal correction."
        )
    )

    parser.add_argument(
        "--open",
        action="store_true",
        help=(
            "Open the final interactive HTML in the Windows "
            "default browser."
        ),
    )

    args = parser.parse_args()

    print()
    print(
        "============================================================"
    )
    print(
        " Part (b): new nominal + relative systematic uncertainties"
    )
    print(
        "============================================================"
    )
    print()

    completed_channels = {}

    for channel, definition in CHANNELS.items():

        print(
            "------------------------------------------------------------"
        )
        print(
            f"Processing {definition['display']}"
        )
        print(
            "------------------------------------------------------------"
        )

        print(
            "New nominal input:"
        )
        print(
            f"  {definition['new_csv']}"
        )

        print(
            "Relative systematic input:"
        )
        print(
            f"  {definition['relative_csv']}"
        )

        new_rows = convert_new_rows(
            read_csv_rows(
                definition[
                    "new_csv"
                ],
                NEW_REQUIRED_COLUMNS,
            )
        )

        relative_rows = convert_relative_rows(
            read_csv_rows(
                definition[
                    "relative_csv"
                ],
                RELATIVE_REQUIRED_COLUMNS,
            )
        )

        final_rows = build_final_rows(
            channel=channel,
            new_rows=new_rows,
            relative_rows=relative_rows,
        )

        csv_path = write_final_csv(
            channel,
            final_rows,
        )

        root_path = write_root_output(
            channel=channel,
            display=definition[
                "display"
            ],
            final_rows=final_rows,
        )

        completed_channels[
            channel
        ] = {
            "display": (
                definition[
                    "display"
                ]
            ),
            "rows": final_rows,
            "csv_path": csv_path,
            "root_path": root_path,
        }

        print()
        print(
            f"Final CSV:\n  {csv_path}"
        )

        print(
            f"Final ROOT:\n  {root_path}"
        )

        print()

    interactive_path = save_interactive_plot(
        completed_channels
    )

    if interactive_path is not None:

        print(
            "Interactive final plot:"
        )
        print(
            f"  {interactive_path}"
        )
        print()

    print(
        "============================================================"
    )
    print(
        " Finished"
    )
    print(
        "============================================================"
    )
    print()

    print(
        "Final definition:"
    )
    print(
        "  nominal correction = new high-statistics weight-0 sample"
    )
    print(
        "  statistical uncertainty = new sample"
    )
    print(
        "  systematic uncertainties = relative old weighted-sample"
    )
    print(
        "  uncertainties multiplied by the new nominal correction"
    )
    print()

    print(
        f"All final outputs:\n  {FINAL_OUTPUT_DIR}"
    )
    print()

    if (
        args.open
        and interactive_path is not None
    ):

        open_in_windows_browser(
            interactive_path
        )


if __name__ == "__main__":
    main()
