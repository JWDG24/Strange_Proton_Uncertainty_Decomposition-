#!/usr/bin/env python3

"""
display_all_findings.py

TO OPEN: python3 lepton_pseudorapidity_work/display_all_findings.py --open

Create one self-contained interactive HTML dashboard containing the main
findings from the strange-proton uncertainty-decomposition project.

Recommended location:

uncertainty_decomposition/
└── lepton_pseudorapidity_work/
    └── display_all_findings.py

Run from the project root:

    python3 lepton_pseudorapidity_work/display_all_findings.py --open

The script automatically looks for the CSV outputs produced by the
analysis already in the repository. It does not rerun the event loop or
change any physics results.

Dashboard screens
-----------------
1. Final result
   New high-statistics Pythia nominal correction with statistical,
   systematic and total uncertainties.

2. Original mT-selected corrections
   The original Pythia and Herwig W+ and W- correction factors.

3. Statistical validation
   Odd/even split-sample comparison against the propagated ROOT
   statistical uncertainty.

4. New weight-0 cross check
   Original nominal Pythia correction compared with the new
   high-statistics weight-0 sample.

5. Relative systematic uncertainties
   Four synchronized panels for Pythia W+, Pythia W-, Herwig W+
   and Herwig W-.

6. Individual weight responses
   Four synchronized panels with a dropdown for scale, PDF, shower
   and model variations.

7. Pythia versus Herwig
   Generator comparison for W+ and W-.

8. Method and conclusions
   A compact summary of the selection, correction definition and
   interpretation of the checks.

All mathematical labels use Unicode or HTML subscripts rather than
LaTeX-style variable names such as C_new.
"""

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def locate_work_dir():
    """
    Allow the script to live either directly inside
    lepton_pseudorapidity_work or one directory above it.
    """

    direct_markers = [
        SCRIPT_DIR / "MTWcut_build_corrections_etalepton_csv",
        SCRIPT_DIR / "weights_relative_uncertainties",
        SCRIPT_DIR / "cross_check_weight0",
    ]

    if any(path.exists() for path in direct_markers):
        return SCRIPT_DIR

    candidate = SCRIPT_DIR / "lepton_pseudorapidity_work"

    if candidate.is_dir():
        return candidate

    # Fall back to the script directory. Missing-data messages in the
    # dashboard will make any placement mistake obvious.
    return SCRIPT_DIR


WORK_DIR = locate_work_dir()

OUTPUT_DIR = (
    WORK_DIR
    / "all_findings_dashboard"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_HTML = (
    OUTPUT_DIR
    / "strange_proton_uncertainty_dashboard.html"
)


# ============================================================
# Labels and ordering
# ============================================================

PAIR_ORDER = [
    "Pythia_plus",
    "Pythia_minus",
    "Herwig_plus",
    "Herwig_minus",
]

PAIR_LABELS = {
    "Pythia_plus": "Pythia W⁺",
    "Pythia_minus": "Pythia W⁻",
    "Herwig_plus": "Herwig W⁺",
    "Herwig_minus": "Herwig W⁻",
}

PAIR_POSITIONS = {
    "Pythia_plus": (1, 1),
    "Pythia_minus": (1, 2),
    "Herwig_plus": (2, 1),
    "Herwig_minus": (2, 2),
}

SOURCE_ORDER = [
    "scale",
    "pdf",
    "shower",
    "model",
]

SOURCE_LABELS = {
    "scale": "Scale",
    "pdf": "PDF",
    "shower": "Shower",
    "model": "Model",
}

SOURCE_SUMMARY_COLUMNS = {
    "scale": "scale_unc_percent",
    "pdf": "pdf_unc_percent",
    "shower": "shower_unc_percent",
    "model": "model_unc_percent",
}

SOURCE_METHODS = {
    "scale": "Envelope",
    "pdf": "RMS",
    "shower": "Envelope",
    "model": "Envelope",
}


# ============================================================
# General helpers
# ============================================================

def read_csv_safely(path):
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(
            f"Warning: could not read {path}: {exc}"
        )
        return None


def add_eta_columns(dataframe):
    if dataframe is None:
        return None

    required = {
        "bin_low_edge",
        "bin_up_edge",
    }

    if not required.issubset(
        dataframe.columns
    ):
        return dataframe

    result = dataframe.copy()

    result["eta_centre"] = (
        0.5
        * (
            result["bin_low_edge"]
            + result["bin_up_edge"]
        )
    )

    result["eta_half_width"] = (
        0.5
        * (
            result["bin_up_edge"]
            - result["bin_low_edge"]
        )
    )

    return result


def finite_values(values):
    array = np.asarray(
        values,
        dtype=float,
    )

    return array[
        np.isfinite(
            array
        )
    ]


def padded_range(
    values,
    fraction=0.08,
    include_zero=False,
):
    """
    Return a sensible plot range with padding.
    """

    values = finite_values(
        values
    )

    if len(values) == 0:
        return None

    low = float(
        np.min(
            values
        )
    )

    high = float(
        np.max(
            values
        )
    )

    if include_zero:
        low = min(
            low,
            0.0,
        )
        high = max(
            high,
            0.0,
        )

    span = high - low

    if span <= 0.0:
        span = max(
            abs(
                high
            ),
            1.0,
        )

    padding = (
        fraction
        * span
    )

    return [
        low - padding,
        high + padding,
    ]


def correction_range(
    dataframes,
    value_columns,
    error_columns=None,
):
    """
    Common y range for correction-factor plots.
    """

    values = []

    for dataframe in dataframes:

        if dataframe is None:
            continue

        for value_column in value_columns:

            if value_column not in dataframe.columns:
                continue

            central = pd.to_numeric(
                dataframe[
                    value_column
                ],
                errors="coerce",
            ).to_numpy()

            if (
                error_columns
                and value_column
                in error_columns
                and error_columns[
                    value_column
                ]
                in dataframe.columns
            ):

                errors = pd.to_numeric(
                    dataframe[
                        error_columns[
                            value_column
                        ]
                    ],
                    errors="coerce",
                ).fillna(
                    0.0
                ).to_numpy()

                values.extend(
                    central - errors
                )

                values.extend(
                    central + errors
                )

            else:

                values.extend(
                    central
                )

    return padded_range(
        values,
        fraction=0.10,
        include_zero=False,
    )


def ensure_eta(dataframe):
    if dataframe is None:
        return None

    if "eta_centre" not in dataframe.columns:
        return add_eta_columns(
            dataframe
        )

    return dataframe


def source_file_note(paths):
    existing = [
        str(
            path.relative_to(
                WORK_DIR
            )
        )
        for path in paths
        if path is not None
        and path.exists()
    ]

    if not existing:
        return "No matching source CSV was found."

    return (
        "Source data: "
        + ", ".join(
            existing
        )
    )


# ============================================================
# Locate and load original correction CSVs
# ============================================================

def load_original_corrections():
    """
    Original mT-selected correction-factor CSVs produced by
    MTWcut_build_corrections_etalepton.py.
    """

    directory = (
        WORK_DIR
        / "MTWcut_build_corrections_etalepton_csv"
    )

    results = {}

    paths = {}

    if not directory.is_dir():
        return results, paths

    for path in sorted(
        directory.glob(
            "*.csv"
        )
    ):

        dataframe = read_csv_safely(
            path
        )

        if (
            dataframe is None
            or dataframe.empty
            or "pair_label"
            not in dataframe.columns
        ):
            continue

        pair = str(
            dataframe.iloc[
                0
            ][
                "pair_label"
            ]
        )

        if pair not in PAIR_LABELS:
            continue

        results[
            pair
        ] = ensure_eta(
            dataframe
        )

        paths[
            pair
        ] = path

    return (
        results,
        paths,
    )


# ============================================================
# Locate and load odd/even outputs
# ============================================================

def find_even_odd_csvs():
    """
    Search recursively so either even_odd_uncertainties or an older
    capitalisation of the folder works.
    """

    found = {}

    paths = {}

    for path in WORK_DIR.rglob(
        "even_odd_*.csv"
    ):

        dataframe = read_csv_safely(
            path
        )

        if (
            dataframe is None
            or dataframe.empty
            or "pair_label"
            not in dataframe.columns
        ):
            continue

        required = {
            "C_all",
            "C_odd",
            "C_even",
            "split_stat_uncertainty_half_difference",
            "all_ratio_ROOT_stat_error",
        }

        if not required.issubset(
            dataframe.columns
        ):
            continue

        pair = str(
            dataframe.iloc[
                0
            ][
                "pair_label"
            ]
        )

        if pair not in PAIR_LABELS:
            continue

        found[
            pair
        ] = ensure_eta(
            dataframe
        )

        paths[
            pair
        ] = path

    return (
        found,
        paths,
    )


# ============================================================
# Locate and load new weight-0 cross-check outputs
# ============================================================

def load_cross_check():
    directory = (
        WORK_DIR
        / "cross_check_weight0"
        / "outputs"
        / "csv"
    )

    results = {}

    paths = {}

    if not directory.is_dir():
        return results, paths

    for path in sorted(
        directory.glob(
            "cross_check_*.csv"
        )
    ):

        dataframe = read_csv_safely(
            path
        )

        if (
            dataframe is None
            or dataframe.empty
            or "pair_label"
            not in dataframe.columns
        ):
            continue

        required = {
            "C_reference",
            "reference_stat_unc",
            "C_new_crosscheck",
            "new_crosscheck_stat_unc",
            "new_minus_reference",
        }

        if not required.issubset(
            dataframe.columns
        ):
            continue

        pair = str(
            dataframe.iloc[
                0
            ][
                "pair_label"
            ]
        )

        if pair not in {
            "Pythia_plus",
            "Pythia_minus",
        }:
            continue

        results[
            pair
        ] = ensure_eta(
            dataframe
        )

        paths[
            pair
        ] = path

    return (
        results,
        paths,
    )


# ============================================================
# Locate and load relative systematic outputs
# ============================================================

def load_relative_uncertainties():
    directory = (
        WORK_DIR
        / "weights_relative_uncertainties"
        / "outputs"
        / "csv"
    )

    summary = {}
    per_weight = {}
    summary_paths = {}
    per_weight_paths = {}

    if not directory.is_dir():
        return (
            summary,
            per_weight,
            summary_paths,
            per_weight_paths,
        )

    for pair in PAIR_ORDER:

        summary_path = (
            directory
            / f"{pair}_relative_uncertainties.csv"
        )

        weights_path = (
            directory
            / f"{pair}_per_weight_relative_shifts.csv"
        )

        if summary_path.is_file():

            dataframe = read_csv_safely(
                summary_path
            )

            if (
                dataframe is not None
                and not dataframe.empty
            ):
                summary[
                    pair
                ] = ensure_eta(
                    dataframe
                )

                summary_paths[
                    pair
                ] = summary_path

        if weights_path.is_file():

            dataframe = read_csv_safely(
                weights_path
            )

            if (
                dataframe is not None
                and not dataframe.empty
            ):
                per_weight[
                    pair
                ] = ensure_eta(
                    dataframe
                )

                per_weight_paths[
                    pair
                ] = weights_path

    return (
        summary,
        per_weight,
        summary_paths,
        per_weight_paths,
    )


# ============================================================
# Locate and load final part-(b) outputs
# ============================================================

def load_final_results():
    directory = (
        WORK_DIR
        / "weights_relative_uncertainties"
        / "outputs"
        / "final_new_nominal"
        / "csv"
    )

    results = {}

    paths = {}

    if not directory.is_dir():
        return results, paths

    for path in sorted(
        directory.glob(
            "*_final_new_nominal.csv"
        )
    ):

        dataframe = read_csv_safely(
            path
        )

        if (
            dataframe is None
            or dataframe.empty
            or "channel"
            not in dataframe.columns
        ):
            continue

        channel = str(
            dataframe.iloc[
                0
            ][
                "channel"
            ]
        )

        if channel not in {
            "Pythia_plus",
            "Pythia_minus",
        }:
            continue

        results[
            channel
        ] = ensure_eta(
            dataframe
        )

        paths[
            channel
        ] = path

    return (
        results,
        paths,
    )


# ============================================================
# Generic empty-state figure
# ============================================================

def empty_figure(
    title,
    message,
):
    figure = go.Figure()

    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="center",
        font=dict(
            size=16,
        ),
    )

    figure.update_layout(
        title=dict(
            text=title,
            x=0.5,
        ),
        template="plotly_white",
        height=520,
        margin=dict(
            l=60,
            r=40,
            t=90,
            b=60,
        ),
        xaxis=dict(
            visible=False,
        ),
        yaxis=dict(
            visible=False,
        ),
    )

    return figure


# ============================================================
# Figure: final result
# ============================================================

def build_final_figure(
    final_results,
    cross_check,
):
    if not final_results:
        return empty_figure(
            "Final correction",
            (
                "No final-new-nominal CSV files were found.<br>"
                "Run apply_relative_uncertainties_to_new_nominal.py first."
            ),
        )

    channels = [
        channel
        for channel in [
            "Pythia_plus",
            "Pythia_minus",
        ]
        if channel in final_results
    ]

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Nominal correction",
            "Uncertainty decomposition",
        ),
        row_heights=[
            0.58,
            0.42,
        ],
    )

    trace_groups = {}

    correction_values = []

    uncertainty_values = []

    traces_per_channel = None

    for channel_index, channel in enumerate(
        channels
    ):

        dataframe = final_results[
            channel
        ]

        visible = (
            channel_index == 0
        )

        trace_groups[
            channel
        ] = []

        x = dataframe[
            "eta_centre"
        ]

        correction = dataframe[
            "nominal_correction_new"
        ]

        correction_values.extend(
            correction
        )

        if (
            "total_unc"
            in dataframe.columns
        ):

            correction_values.extend(
                correction
                - dataframe[
                    "total_unc"
                ]
            )

            correction_values.extend(
                correction
                + dataframe[
                    "total_unc"
                ]
            )

        # New nominal with statistical uncertainty
        trace_groups[
            channel
        ].append(
            len(
                figure.data
            )
        )

        figure.add_trace(
            go.Scatter(
                x=x,
                y=correction,
                mode="lines+markers",
                name=(
                    "C<sub>new</sub> with "
                    "σ<sub>stat</sub>"
                ),
                visible=visible,
                error_y=dict(
                    type="data",
                    array=dataframe[
                        "stat_unc_new"
                    ],
                    visible=True,
                ),
                customdata=np.column_stack(
                    [
                        dataframe[
                            "bin_low_edge"
                        ],
                        dataframe[
                            "bin_up_edge"
                        ],
                        dataframe[
                            "stat_unc_new"
                        ],
                    ]
                ),
                hovertemplate=(
                    "|η<sub>ℓ</sub>| bin: "
                    "%{customdata[0]:.2f}–"
                    "%{customdata[1]:.2f}"
                    "<br>C<sub>new</sub> = %{y:.6f}"
                    "<br>σ<sub>stat</sub> = "
                    "%{customdata[2]:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        # Same central values with total uncertainty
        trace_groups[
            channel
        ].append(
            len(
                figure.data
            )
        )

        figure.add_trace(
            go.Scatter(
                x=x,
                y=correction,
                mode="markers",
                name=(
                    "C<sub>new</sub> with "
                    "σ<sub>total</sub>"
                ),
                visible=visible,
                marker=dict(
                    symbol="circle-open",
                    size=10,
                ),
                error_y=dict(
                    type="data",
                    array=dataframe[
                        "total_unc"
                    ],
                    visible=True,
                ),
                customdata=np.column_stack(
                    [
                        dataframe[
                            "bin_low_edge"
                        ],
                        dataframe[
                            "bin_up_edge"
                        ],
                        dataframe[
                            "total_unc"
                        ],
                    ]
                ),
                hovertemplate=(
                    "|η<sub>ℓ</sub>| bin: "
                    "%{customdata[0]:.2f}–"
                    "%{customdata[1]:.2f}"
                    "<br>C<sub>new</sub> = %{y:.6f}"
                    "<br>σ<sub>total</sub> = "
                    "%{customdata[2]:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        # Optional old reference on final screen
        if (
            channel
            in cross_check
        ):

            reference = cross_check[
                channel
            ]

            trace_groups[
                channel
            ].append(
                len(
                    figure.data
                )
            )

            figure.add_trace(
                go.Scatter(
                    x=reference[
                        "eta_centre"
                    ],
                    y=reference[
                        "C_reference"
                    ],
                    mode="lines+markers",
                    name="C<sub>ref</sub> old nominal",
                    visible=visible,
                    line=dict(
                        dash="dot",
                    ),
                    error_y=dict(
                        type="data",
                        array=reference[
                            "reference_stat_unc"
                        ],
                        visible=True,
                    ),
                    hovertemplate=(
                        "|η<sub>ℓ</sub>| = %{x:.3f}"
                        "<br>C<sub>ref</sub> = %{y:.6f}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

            correction_values.extend(
                reference[
                    "C_reference"
                ]
            )

        components = [
            (
                "stat_unc_percent",
                "σ<sub>stat</sub>",
            ),
            (
                "scale_unc_percent",
                "σ<sub>scale</sub>",
            ),
            (
                "pdf_unc_percent",
                "σ<sub>PDF</sub>",
            ),
            (
                "shower_unc_percent",
                "σ<sub>shower</sub>",
            ),
            (
                "model_unc_percent",
                "σ<sub>model</sub>",
            ),
            (
                "systematic_unc_percent",
                "σ<sub>syst</sub>",
            ),
            (
                "total_unc_percent",
                "σ<sub>total</sub>",
            ),
        ]

        for column, label in components:

            if column not in dataframe.columns:
                continue

            trace_groups[
                channel
            ].append(
                len(
                    figure.data
                )
            )

            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=dataframe[
                        column
                    ],
                    mode="lines+markers",
                    name=label,
                    visible=visible,
                    hovertemplate=(
                        "|η<sub>ℓ</sub>| = %{x:.3f}"
                        f"<br>{label} = %{{y:.4f}}%"
                        "<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

            uncertainty_values.extend(
                dataframe[
                    column
                ]
            )

        if traces_per_channel is None:
            traces_per_channel = len(
                trace_groups[
                    channel
                ]
            )

    correction_y_range = padded_range(
        correction_values,
        fraction=0.08,
    )

    uncertainty_y_range = padded_range(
        uncertainty_values,
        fraction=0.10,
        include_zero=True,
    )

    buttons = []

    for channel in channels:

        visible_mask = [
            False
        ] * len(
            figure.data
        )

        for trace_index in trace_groups[
            channel
        ]:

            visible_mask[
                trace_index
            ] = True

        buttons.append(
            dict(
                label=PAIR_LABELS[
                    channel
                ],
                method="update",
                args=[
                    {
                        "visible": visible_mask,
                    },
                    {
                        "title.text": (
                            "Final correction and uncertainty budget"
                            "<br><sup>"
                            + PAIR_LABELS[
                                channel
                            ]
                            + "</sup>"
                        ),
                    },
                ],
            )
        )

    first_channel = channels[
        0
    ]

    figure.update_layout(
        title=dict(
            text=(
                "Final correction and uncertainty budget"
                "<br><sup>"
                + PAIR_LABELS[
                    first_channel
                ]
                + "</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        height=900,
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
                y=1.15,
                yanchor="top",
            )
        ],
        margin=dict(
            l=80,
            r=40,
            t=155,
            b=70,
        ),
    )

    figure.update_xaxes(
        range=[
            0.0,
            2.5,
        ],
        matches="x",
    )

    figure.update_xaxes(
        title_text="Lepton |η<sub>ℓ</sub>|",
        row=2,
        col=1,
    )

    figure.update_yaxes(
        title_text="Correction factor C",
        range=correction_y_range,
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text="Uncertainty (%)",
        range=uncertainty_y_range,
        row=2,
        col=1,
    )

    return figure


# ============================================================
# Figure: original mT-selected corrections
# ============================================================

def build_original_figure(
    original,
):
    available = [
        pair
        for pair in PAIR_ORDER
        if pair in original
    ]

    if not available:
        return empty_figure(
            "Original mT-selected correction factors",
            (
                "No original correction CSV files were found in "
                "MTWcut_build_corrections_etalepton_csv."
            ),
        )

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            PAIR_LABELS[
                pair
            ]
            for pair in PAIR_ORDER
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
    )

    all_values = []

    for pair in PAIR_ORDER:

        if pair not in original:
            continue

        dataframe = original[
            pair
        ]

        row, col = PAIR_POSITIONS[
            pair
        ]

        customdata = np.column_stack(
            [
                dataframe[
                    "bin_low_edge"
                ],
                dataframe[
                    "bin_up_edge"
                ],
                dataframe[
                    "stat_unc"
                ],
                dataframe[
                    "total_unc"
                ],
            ]
        )

        figure.add_trace(
            go.Scatter(
                x=dataframe[
                    "eta_centre"
                ],
                y=dataframe[
                    "correction_factor"
                ],
                mode="lines+markers",
                name="Nominal correction",
                legendgroup="nominal",
                showlegend=(
                    pair
                    == available[
                        0
                    ]
                ),
                error_y=dict(
                    type="data",
                    array=dataframe[
                        "total_unc"
                    ],
                    visible=True,
                ),
                customdata=customdata,
                hovertemplate=(
                    "|η<sub>ℓ</sub>| bin: "
                    "%{customdata[0]:.2f}–"
                    "%{customdata[1]:.2f}"
                    "<br>C = %{y:.6f}"
                    "<br>σ<sub>stat</sub> = "
                    "%{customdata[2]:.6f}"
                    "<br>σ<sub>total</sub> = "
                    "%{customdata[3]:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        all_values.extend(
            dataframe[
                "correction_factor"
            ]
            - dataframe[
                "total_unc"
            ]
        )

        all_values.extend(
            dataframe[
                "correction_factor"
            ]
            + dataframe[
                "total_unc"
            ]
        )

    y_range = padded_range(
        all_values,
        fraction=0.08,
    )

    figure.update_layout(
        title=dict(
            text=(
                "Original m<sub>T</sub><sup>W</sup>-selected "
                "parton-to-particle corrections"
            ),
            x=0.5,
        ),
        template="plotly_white",
        height=850,
        hovermode="closest",
        margin=dict(
            l=80,
            r=40,
            t=120,
            b=70,
        ),
    )

    for pair in PAIR_ORDER:

        row, col = PAIR_POSITIONS[
            pair
        ]

        figure.update_xaxes(
            title_text="Lepton |η<sub>ℓ</sub>|",
            range=[
                0.0,
                2.5,
            ],
            matches="x",
            row=row,
            col=col,
        )

        figure.update_yaxes(
            title_text="Correction factor C",
            range=y_range,
            matches="y",
            row=row,
            col=col,
        )

    return figure


# ============================================================
# Figure: odd/even statistical validation
# ============================================================

def build_even_odd_figure(
    even_odd,
):
    channels = [
        pair
        for pair in PAIR_ORDER
        if pair in even_odd
    ]

    if not channels:
        return empty_figure(
            "Odd/even statistical validation",
            (
                "No odd/even CSV outputs were found. "
                "Run even_odd_uncertainty.py first."
            ),
        )

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Correction-factor stability",
            "Statistical uncertainty comparison",
        ),
        row_heights=[
            0.58,
            0.42,
        ],
    )

    trace_groups = {}

    all_corrections = []

    all_uncertainties = []

    for channel_index, pair in enumerate(
        channels
    ):

        dataframe = even_odd[
            pair
        ]

        visible = (
            channel_index == 0
        )

        trace_groups[
            pair
        ] = []

        top_series = [
            (
                "C_all",
                "C<sub>all</sub>",
                None,
            ),
            (
                "C_odd",
                "C<sub>odd</sub>",
                "odd_ratio_ROOT_stat_error",
            ),
            (
                "C_even",
                "C<sub>even</sub>",
                "even_ratio_ROOT_stat_error",
            ),
        ]

        for column, label, error_column in top_series:

            trace_groups[
                pair
            ].append(
                len(
                    figure.data
                )
            )

            kwargs = {}

            if (
                error_column
                and error_column
                in dataframe.columns
            ):
                kwargs[
                    "error_y"
                ] = dict(
                    type="data",
                    array=dataframe[
                        error_column
                    ],
                    visible=True,
                )

            figure.add_trace(
                go.Scatter(
                    x=dataframe[
                        "eta_centre"
                    ],
                    y=dataframe[
                        column
                    ],
                    mode="lines+markers",
                    name=label,
                    visible=visible,
                    hovertemplate=(
                        "|η<sub>ℓ</sub>| = %{x:.3f}"
                        f"<br>{label} = %{{y:.6f}}"
                        "<extra></extra>"
                    ),
                    **kwargs,
                ),
                row=1,
                col=1,
            )

            all_corrections.extend(
                dataframe[
                    column
                ]
            )

        bottom_series = [
            (
                "split_stat_uncertainty_half_difference",
                "½|C<sub>odd</sub> − C<sub>even</sub>|",
            ),
            (
                "all_ratio_ROOT_stat_error",
                "Propagated σ<sub>stat</sub>",
            ),
        ]

        for column, label in bottom_series:

            trace_groups[
                pair
            ].append(
                len(
                    figure.data
                )
            )

            figure.add_trace(
                go.Scatter(
                    x=dataframe[
                        "eta_centre"
                    ],
                    y=dataframe[
                        column
                    ],
                    mode="lines+markers",
                    name=label,
                    visible=visible,
                    hovertemplate=(
                        "|η<sub>ℓ</sub>| = %{x:.3f}"
                        f"<br>{label} = %{{y:.6f}}"
                        "<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

            all_uncertainties.extend(
                dataframe[
                    column
                ]
            )

    buttons = []

    for pair in channels:

        visible_mask = [
            False
        ] * len(
            figure.data
        )

        for trace_index in trace_groups[
            pair
        ]:

            visible_mask[
                trace_index
            ] = True

        buttons.append(
            dict(
                label=PAIR_LABELS[
                    pair
                ],
                method="update",
                args=[
                    {
                        "visible": visible_mask,
                    },
                    {
                        "title.text": (
                            "Odd/even statistical validation"
                            "<br><sup>"
                            + PAIR_LABELS[
                                pair
                            ]
                            + "</sup>"
                        ),
                    },
                ],
            )
        )

    figure.update_layout(
        title=dict(
            text=(
                "Odd/even statistical validation"
                "<br><sup>"
                + PAIR_LABELS[
                    channels[
                        0
                    ]
                ]
                + "</sup>"
            ),
            x=0.5,
        ),
        template="plotly_white",
        height=900,
        hovermode="x unified",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(
            l=90,
            r=40,
            t=155,
            b=70,
        ),
    )

    figure.update_xaxes(
        range=[
            0.0,
            2.5,
        ],
        matches="x",
    )

    figure.update_xaxes(
        title_text="Lepton |η<sub>ℓ</sub>|",
        row=2,
        col=1,
    )

    figure.update_yaxes(
        title_text="Correction factor C",
        range=padded_range(
            all_corrections,
            fraction=0.08,
        ),
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text="Absolute uncertainty",
        range=padded_range(
            all_uncertainties,
            fraction=0.10,
            include_zero=True,
        ),
        row=2,
        col=1,
    )

    return figure


# ============================================================
# Figure: new weight-0 cross check
# ============================================================

def build_cross_check_figure(
    cross_check,
):
    channels = [
        pair
        for pair in [
            "Pythia_plus",
            "Pythia_minus",
        ]
        if pair in cross_check
    ]

    if not channels:
        return empty_figure(
            "New weight-0 cross check",
            (
                "No cross-check CSV files were found. "
                "Run cross_check_weight0.py first."
            ),
        )

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Reference and new correction factors",
            "Difference",
            "Difference in units of combined statistical uncertainty",
        ),
        row_heights=[
            0.50,
            0.27,
            0.23,
        ],
    )

    trace_groups = {}

    all_corrections = []

    all_differences = []

    all_pulls = []

    for channel_index, pair in enumerate(
        channels
    ):

        dataframe = cross_check[
            pair
        ]

        visible = (
            channel_index == 0
        )

        trace_groups[
            pair
        ] = []

        for column, error_column, label in [
            (
                "C_reference",
                "reference_stat_unc",
                "C<sub>ref</sub> original",
            ),
            (
                "C_new_crosscheck",
                "new_crosscheck_stat_unc",
                "C<sub>new</sub> high statistics",
            ),
        ]:

            trace_groups[
                pair
            ].append(
                len(
                    figure.data
                )
            )

            figure.add_trace(
                go.Scatter(
                    x=dataframe[
                        "eta_centre"
                    ],
                    y=dataframe[
                        column
                    ],
                    mode="lines+markers",
                    name=label,
                    visible=visible,
                    error_y=dict(
                        type="data",
                        array=dataframe[
                            error_column
                        ],
                        visible=True,
                    ),
                    hovertemplate=(
                        "|η<sub>ℓ</sub>| = %{x:.3f}"
                        f"<br>{label} = %{{y:.6f}}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

            all_corrections.extend(
                dataframe[
                    column
                ]
                - dataframe[
                    error_column
                ]
            )

            all_corrections.extend(
                dataframe[
                    column
                ]
                + dataframe[
                    error_column
                ]
            )

        trace_groups[
            pair
        ].append(
            len(
                figure.data
            )
        )

        figure.add_trace(
            go.Scatter(
                x=dataframe[
                    "eta_centre"
                ],
                y=dataframe[
                    "new_minus_reference"
                ],
                mode="lines+markers",
                name="C<sub>new</sub> − C<sub>ref</sub>",
                visible=visible,
                error_y=(
                    dict(
                        type="data",
                        array=dataframe[
                            "combined_stat_unc_if_independent"
                        ],
                        visible=True,
                    )
                    if "combined_stat_unc_if_independent"
                    in dataframe.columns
                    else None
                ),
                hovertemplate=(
                    "|η<sub>ℓ</sub>| = %{x:.3f}"
                    "<br>C<sub>new</sub> − C<sub>ref</sub> "
                    "= %{y:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

        all_differences.extend(
            dataframe[
                "new_minus_reference"
            ]
        )

        if (
            "difference_over_combined_stat_if_independent"
            in dataframe.columns
        ):

            pull_values = dataframe[
                "difference_over_combined_stat_if_independent"
            ]

            trace_groups[
                pair
            ].append(
                len(
                    figure.data
                )
            )

            figure.add_trace(
                go.Scatter(
                    x=dataframe[
                        "eta_centre"
                    ],
                    y=pull_values,
                    mode="lines+markers",
                    name=(
                        "(C<sub>new</sub> − C<sub>ref</sub>) / "
                        "σ<sub>comb</sub>"
                    ),
                    visible=visible,
                    hovertemplate=(
                        "|η<sub>ℓ</sub>| = %{x:.3f}"
                        "<br>Difference / σ<sub>comb</sub> "
                        "= %{y:.3f}"
                        "<extra></extra>"
                    ),
                ),
                row=3,
                col=1,
            )

            all_pulls.extend(
                pull_values
            )

    buttons = []

    for pair in channels:

        visible_mask = [
            False
        ] * len(
            figure.data
        )

        for trace_index in trace_groups[
            pair
        ]:

            visible_mask[
                trace_index
            ] = True

        buttons.append(
            dict(
                label=PAIR_LABELS[
                    pair
                ],
                method="update",
                args=[
                    {
                        "visible": visible_mask,
                    },
                    {
                        "title.text": (
                            "New high-statistics weight-0 cross check"
                            "<br><sup>"
                            + PAIR_LABELS[
                                pair
                            ]
                            + "</sup>"
                        ),
                    },
                ],
            )
        )

    figure.update_layout(
        title=dict(
            text=(
                "New high-statistics weight-0 cross check"
                "<br><sup>"
                + PAIR_LABELS[
                    channels[
                        0
                    ]
                ]
                + "</sup>"
            ),
            x=0.5,
        ),
        template="plotly_white",
        height=1050,
        hovermode="x unified",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.12,
                yanchor="top",
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(
            l=100,
            r=40,
            t=160,
            b=70,
        ),
    )

    figure.update_xaxes(
        range=[
            0.0,
            2.5,
        ],
        matches="x",
    )

    figure.update_xaxes(
        title_text="Lepton |η<sub>ℓ</sub>|",
        row=3,
        col=1,
    )

    figure.update_yaxes(
        title_text="Correction factor C",
        range=padded_range(
            all_corrections,
            fraction=0.08,
        ),
        row=1,
        col=1,
    )

    symmetric_difference = finite_values(
        all_differences
    )

    if len(
        symmetric_difference
    ):

        maximum = max(
            abs(
                float(
                    np.min(
                        symmetric_difference
                    )
                )
            ),
            abs(
                float(
                    np.max(
                        symmetric_difference
                    )
                )
            ),
        )

        difference_range = [
            -1.15 * maximum,
            1.15 * maximum,
        ]

    else:
        difference_range = None

    figure.update_yaxes(
        title_text=(
            "C<sub>new</sub> − C<sub>ref</sub>"
        ),
        range=difference_range,
        zeroline=True,
        zerolinewidth=2,
        row=2,
        col=1,
    )

    pull_array = finite_values(
        all_pulls
    )

    if len(
        pull_array
    ):
        pull_max = max(
            1.0,
            abs(
                float(
                    np.min(
                        pull_array
                    )
                )
            ),
            abs(
                float(
                    np.max(
                        pull_array
                    )
                )
            ),
        )

        pull_range = [
            -1.15 * pull_max,
            1.15 * pull_max,
        ]

    else:
        pull_range = None

    figure.update_yaxes(
        title_text="Difference / σ<sub>comb</sub>",
        range=pull_range,
        zeroline=True,
        zerolinewidth=2,
        row=3,
        col=1,
    )

    return figure


# ============================================================
# Figure: relative systematics, 2x2 synchronized
# ============================================================

def build_relative_summary_figure(
    relative_summary,
):
    available = [
        pair
        for pair in PAIR_ORDER
        if pair in relative_summary
    ]

    if not available:
        return empty_figure(
            "Relative systematic uncertainties",
            (
                "No relative-uncertainty CSV files were found. "
                "Run build_relative_uncertainties.py first."
            ),
        )

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            PAIR_LABELS[
                pair
            ]
            for pair in PAIR_ORDER
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
    )

    all_values = []

    for pair in PAIR_ORDER:

        if pair not in relative_summary:
            continue

        dataframe = relative_summary[
            pair
        ]

        row, col = PAIR_POSITIONS[
            pair
        ]

        for source in SOURCE_ORDER:

            column = SOURCE_SUMMARY_COLUMNS[
                source
            ]

            if column not in dataframe.columns:
                continue

            figure.add_trace(
                go.Scatter(
                    x=dataframe[
                        "eta_centre"
                    ],
                    y=dataframe[
                        column
                    ],
                    mode="lines+markers",
                    name=SOURCE_LABELS[
                        source
                    ],
                    legendgroup=source,
                    showlegend=(
                        pair
                        == available[
                            0
                        ]
                    ),
                    hovertemplate=(
                        "|η<sub>ℓ</sub>| = %{x:.3f}"
                        "<br>"
                        + SOURCE_LABELS[
                            source
                        ]
                        + " = %{y:.4f}%"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

            all_values.extend(
                dataframe[
                    column
                ]
            )

        if (
            "total_systematic_unc_percent"
            in dataframe.columns
        ):

            figure.add_trace(
                go.Scatter(
                    x=dataframe[
                        "eta_centre"
                    ],
                    y=dataframe[
                        "total_systematic_unc_percent"
                    ],
                    mode="lines+markers",
                    name="Total systematic",
                    legendgroup="total_systematic",
                    showlegend=(
                        pair
                        == available[
                            0
                        ]
                    ),
                    line=dict(
                        width=3,
                    ),
                    marker=dict(
                        symbol="circle-open",
                        size=9,
                    ),
                    hovertemplate=(
                        "|η<sub>ℓ</sub>| = %{x:.3f}"
                        "<br>δ<sub>syst</sub> = %{y:.4f}%"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

            all_values.extend(
                dataframe[
                    "total_systematic_unc_percent"
                ]
            )

    y_range = padded_range(
        all_values,
        fraction=0.08,
        include_zero=True,
    )

    figure.update_layout(
        title=dict(
            text=(
                "Relative systematic uncertainty decomposition"
                "<br><sup>"
                "All variations relative to weight 0 from the same event sample"
                "</sup>"
            ),
            x=0.5,
        ),
        template="plotly_white",
        height=880,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(
            l=90,
            r=40,
            t=145,
            b=70,
        ),
    )

    for pair in PAIR_ORDER:

        row, col = PAIR_POSITIONS[
            pair
        ]

        figure.update_xaxes(
            title_text="Lepton |η<sub>ℓ</sub>|",
            range=[
                0.0,
                2.5,
            ],
            matches="x",
            row=row,
            col=col,
        )

        figure.update_yaxes(
            title_text="Relative uncertainty (%)",
            range=y_range,
            matches="y",
            row=row,
            col=col,
        )

    return figure


# ============================================================
# Figure: individual weight responses, 2x2 synchronized
# ============================================================

def build_weight_response_figure(
    relative_summary,
    per_weight,
):
    available = [
        pair
        for pair in PAIR_ORDER
        if pair in per_weight
        and pair in relative_summary
    ]

    if not available:
        return empty_figure(
            "Individual systematic-weight responses",
            (
                "No per-weight relative-shift CSV files were found. "
                "Run build_relative_uncertainties.py first."
            ),
        )

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            PAIR_LABELS[
                pair
            ]
            for pair in PAIR_ORDER
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
    )

    source_trace_indices = {
        source: []
        for source in SOURCE_ORDER
    }

    source_ranges = {}

    initial_source = "scale"

    for source in SOURCE_ORDER:

        source_all_values = []

        for pair in PAIR_ORDER:

            if pair not in available:
                continue

            dataframe = per_weight[
                pair
            ]

            summary = relative_summary[
                pair
            ]

            subset = dataframe[
                dataframe[
                    "category"
                ].astype(
                    str
                ).str.lower()
                == source
            ].copy()

            row, col = PAIR_POSITIONS[
                pair
            ]

            if subset.empty:
                continue

            for weight_index in sorted(
                subset[
                    "weight_index"
                ].unique()
            ):

                weight_data = subset[
                    subset[
                        "weight_index"
                    ] == weight_index
                ].sort_values(
                    "bin"
                )

                if weight_data.empty:
                    continue

                weight_name = str(
                    weight_data.iloc[
                        0
                    ][
                        "weight_name"
                    ]
                )

                trace_index = len(
                    figure.data
                )

                source_trace_indices[
                    source
                ].append(
                    trace_index
                )

                figure.add_trace(
                    go.Scatter(
                        x=weight_data[
                            "eta_centre"
                        ],
                        y=weight_data[
                            "signed_relative_shift_percent"
                        ],
                        mode="lines+markers",
                        visible=(
                            source
                            == initial_source
                        ),
                        showlegend=False,
                        opacity=0.28,
                        line=dict(
                            width=1,
                        ),
                        marker=dict(
                            size=4,
                        ),
                        hovertemplate=(
                            f"Weight {int(weight_index)}"
                            f"<br>{weight_name}"
                            "<br>|η<sub>ℓ</sub>| = %{x:.3f}"
                            "<br>(C<sub>k</sub> − C₀) / C₀ "
                            "= %{y:.4f}%"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

                source_all_values.extend(
                    weight_data[
                        "signed_relative_shift_percent"
                    ]
                )

            summary_column = SOURCE_SUMMARY_COLUMNS[
                source
            ]

            if summary_column not in summary.columns:
                continue

            boundary = summary[
                summary_column
            ]

            for sign in [
                1.0,
                -1.0,
            ]:

                trace_index = len(
                    figure.data
                )

                source_trace_indices[
                    source
                ].append(
                    trace_index
                )

                figure.add_trace(
                    go.Scatter(
                        x=summary[
                            "eta_centre"
                        ],
                        y=(
                            sign
                            * boundary
                        ),
                        mode="lines",
                        visible=(
                            source
                            == initial_source
                        ),
                        name=(
                            SOURCE_METHODS[
                                source
                            ]
                            + " boundary"
                        ),
                        legendgroup=(
                            source
                            + "_boundary"
                        ),
                        showlegend=(
                            pair
                            == available[
                                0
                            ]
                            and sign > 0
                        ),
                        line=dict(
                            width=3,
                            dash="dash",
                        ),
                        hovertemplate=(
                            "|η<sub>ℓ</sub>| = %{x:.3f}"
                            "<br>"
                            + SOURCE_METHODS[
                                source
                            ]
                            + " boundary = %{y:.4f}%"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

                source_all_values.extend(
                    sign
                    * boundary
                )

        finite = finite_values(
            source_all_values
        )

        if len(
            finite
        ):

            limit = max(
                abs(
                    float(
                        np.min(
                            finite
                        )
                    )
                ),
                abs(
                    float(
                        np.max(
                            finite
                        )
                    )
                ),
            )

            if limit == 0.0:
                limit = 1.0

            source_ranges[
                source
            ] = [
                -1.12 * limit,
                1.12 * limit,
            ]

        else:

            source_ranges[
                source
            ] = [
                -1.0,
                1.0,
            ]

    buttons = []

    for source in SOURCE_ORDER:

        visible = [
            False
        ] * len(
            figure.data
        )

        for trace_index in source_trace_indices[
            source
        ]:

            visible[
                trace_index
            ] = True

        axis_updates = {
            "title.text": (
                "Individual correlated weight responses"
                "<br><sup>"
                + SOURCE_LABELS[
                    source
                ]
                + " · "
                + SOURCE_METHODS[
                    source
                ]
                + " uncertainty"
                + "</sup>"
            ),
        }

        # Plotly subplot axis keys:
        # yaxis, yaxis2, yaxis3, yaxis4
        for index in range(
            1,
            5,
        ):

            key = (
                "yaxis"
                if index == 1
                else f"yaxis{index}"
            )

            axis_updates[
                f"{key}.range"
            ] = source_ranges[
                source
            ]

        buttons.append(
            dict(
                label=SOURCE_LABELS[
                    source
                ],
                method="update",
                args=[
                    {
                        "visible": visible,
                    },
                    axis_updates,
                ],
            )
        )

    figure.update_layout(
        title=dict(
            text=(
                "Individual correlated weight responses"
                "<br><sup>"
                "Scale · Envelope uncertainty"
                "</sup>"
            ),
            x=0.5,
        ),
        template="plotly_white",
        height=900,
        hovermode="closest",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ],
        margin=dict(
            l=100,
            r=40,
            t=160,
            b=70,
        ),
    )

    for pair in PAIR_ORDER:

        row, col = PAIR_POSITIONS[
            pair
        ]

        figure.update_xaxes(
            title_text="Lepton |η<sub>ℓ</sub>|",
            range=[
                0.0,
                2.5,
            ],
            matches="x",
            row=row,
            col=col,
        )

        figure.update_yaxes(
            title_text=(
                "(C<sub>k</sub> − C₀) / C₀ (%)"
            ),
            range=source_ranges[
                initial_source
            ],
            matches="y",
            zeroline=True,
            zerolinewidth=2,
            row=row,
            col=col,
        )

    return figure


# ============================================================
# Figure: Pythia versus Herwig
# ============================================================

def build_generator_comparison_figure(
    original,
):
    charge_pairs = {
        "plus": (
            "Pythia_plus",
            "Herwig_plus",
            "W⁺",
        ),
        "minus": (
            "Pythia_minus",
            "Herwig_minus",
            "W⁻",
        ),
    }

    available = {
        charge: values
        for charge, values
        in charge_pairs.items()
        if values[
            0
        ] in original
        and values[
            1
        ] in original
    }

    if not available:
        return empty_figure(
            "Pythia versus Herwig",
            (
                "Both Pythia and Herwig original correction CSVs "
                "are needed for this comparison."
            ),
        )

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.11,
        subplot_titles=(
            "Generator correction factors",
            "Generator difference",
        ),
        row_heights=[
            0.62,
            0.38,
        ],
    )

    trace_groups = {}

    correction_values = []

    difference_values = []

    for charge_index, (
        charge,
        (
            pythia_key,
            herwig_key,
            charge_label,
        ),
    ) in enumerate(
        available.items()
    ):

        pythia = original[
            pythia_key
        ].copy()

        herwig = original[
            herwig_key
        ].copy()

        merged = pd.merge(
            pythia,
            herwig,
            on=[
                "bin",
                "bin_low_edge",
                "bin_up_edge",
                "eta_centre",
            ],
            suffixes=(
                "_pythia",
                "_herwig",
            ),
        )

        visible = (
            charge_index == 0
        )

        trace_groups[
            charge
        ] = []

        for suffix, label in [
            (
                "pythia",
                "Pythia",
            ),
            (
                "herwig",
                "Herwig",
            ),
        ]:

            trace_groups[
                charge
            ].append(
                len(
                    figure.data
                )
            )

            figure.add_trace(
                go.Scatter(
                    x=merged[
                        "eta_centre"
                    ],
                    y=merged[
                        f"correction_factor_{suffix}"
                    ],
                    mode="lines+markers",
                    name=label,
                    visible=visible,
                    error_y=dict(
                        type="data",
                        array=merged[
                            f"total_unc_{suffix}"
                        ],
                        visible=True,
                    ),
                    hovertemplate=(
                        "|η<sub>ℓ</sub>| = %{x:.3f}"
                        f"<br>{label} C = %{{y:.6f}}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

            correction_values.extend(
                merged[
                    f"correction_factor_{suffix}"
                ]
                - merged[
                    f"total_unc_{suffix}"
                ]
            )

            correction_values.extend(
                merged[
                    f"correction_factor_{suffix}"
                ]
                + merged[
                    f"total_unc_{suffix}"
                ]
            )

        difference = (
            merged[
                "correction_factor_pythia"
            ]
            - merged[
                "correction_factor_herwig"
            ]
        )

        trace_groups[
            charge
        ].append(
            len(
                figure.data
            )
        )

        figure.add_trace(
            go.Scatter(
                x=merged[
                    "eta_centre"
                ],
                y=difference,
                mode="lines+markers",
                name="C<sub>Pythia</sub> − C<sub>Herwig</sub>",
                visible=visible,
                hovertemplate=(
                    "|η<sub>ℓ</sub>| = %{x:.3f}"
                    "<br>C<sub>Pythia</sub> − C<sub>Herwig</sub> "
                    "= %{y:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

        difference_values.extend(
            difference
        )

    buttons = []

    for charge, (
        _,
        _,
        charge_label,
    ) in available.items():

        visible = [
            False
        ] * len(
            figure.data
        )

        for trace_index in trace_groups[
            charge
        ]:

            visible[
                trace_index
            ] = True

        buttons.append(
            dict(
                label=charge_label,
                method="update",
                args=[
                    {
                        "visible": visible,
                    },
                    {
                        "title.text": (
                            "Pythia versus Herwig"
                            "<br><sup>"
                            + charge_label
                            + "</sup>"
                        ),
                    },
                ],
            )
        )

    difference_array = finite_values(
        difference_values
    )

    if len(
        difference_array
    ):

        maximum = max(
            abs(
                float(
                    np.min(
                        difference_array
                    )
                )
            ),
            abs(
                float(
                    np.max(
                        difference_array
                    )
                )
            ),
        )

        difference_range = [
            -1.15 * maximum,
            1.15 * maximum,
        ]

    else:
        difference_range = None

    first_charge = next(
        iter(
            available.values()
        )
    )[
        2
    ]

    figure.update_layout(
        title=dict(
            text=(
                "Pythia versus Herwig"
                "<br><sup>"
                + first_charge
                + "</sup>"
            ),
            x=0.5,
        ),
        template="plotly_white",
        height=850,
        hovermode="x unified",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(
            l=100,
            r=40,
            t=155,
            b=70,
        ),
    )

    figure.update_xaxes(
        range=[
            0.0,
            2.5,
        ],
        matches="x",
    )

    figure.update_xaxes(
        title_text="Lepton |η<sub>ℓ</sub>|",
        row=2,
        col=1,
    )

    figure.update_yaxes(
        title_text="Correction factor C",
        range=padded_range(
            correction_values,
            fraction=0.08,
        ),
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text=(
            "C<sub>Pythia</sub> − C<sub>Herwig</sub>"
        ),
        range=difference_range,
        zeroline=True,
        zerolinewidth=2,
        row=2,
        col=1,
    )

    return figure


# ============================================================
# Data-derived summary
# ============================================================

def build_key_findings_html(
    final_results,
    cross_check,
    relative_summary,
    even_odd,
):
    """
    Create concise data-derived findings without hard-coding numerical
    results that may change when the analysis is rerun.
    """

    cards = []

    if cross_check:

        all_new_stat = []
        all_old_stat = []
        all_abs_diff = []

        for dataframe in cross_check.values():

            if (
                "new_crosscheck_stat_unc"
                in dataframe.columns
            ):
                all_new_stat.extend(
                    dataframe[
                        "new_crosscheck_stat_unc"
                    ]
                )

            if (
                "reference_stat_unc"
                in dataframe.columns
            ):
                all_old_stat.extend(
                    dataframe[
                        "reference_stat_unc"
                    ]
                )

            if (
                "abs_new_minus_reference"
                in dataframe.columns
            ):
                all_abs_diff.extend(
                    dataframe[
                        "abs_new_minus_reference"
                    ]
                )

        if (
            all_new_stat
            and all_old_stat
        ):

            new_median = float(
                np.median(
                    finite_values(
                        all_new_stat
                    )
                )
            )

            old_median = float(
                np.median(
                    finite_values(
                        all_old_stat
                    )
                )
            )

            if old_median > 0.0:

                ratio = (
                    new_median
                    / old_median
                )

                cards.append(
                    (
                        "High-statistics cross check",
                        (
                            "The median propagated statistical uncertainty "
                            f"in the new Pythia samples is {ratio:.2f} times "
                            "the median value in the reference samples. "
                            "The detailed bin-by-bin agreement is shown in "
                            "the cross-check screen."
                        ),
                    )
                )

    if even_odd:

        split_values = []
        propagated_values = []

        for dataframe in even_odd.values():

            split_values.extend(
                dataframe[
                    "split_stat_uncertainty_half_difference"
                ]
            )

            propagated_values.extend(
                dataframe[
                    "all_ratio_ROOT_stat_error"
                ]
            )

        split_array = finite_values(
            split_values
        )

        propagated_array = finite_values(
            propagated_values
        )

        if (
            len(
                split_array
            )
            and len(
                propagated_array
            )
        ):

            split_median = float(
                np.median(
                    split_array
                )
            )

            propagated_median = float(
                np.median(
                    propagated_array
                )
            )

            if propagated_median > 0.0:

                ratio = (
                    split_median
                    / propagated_median
                )

                cards.append(
                    (
                        "Statistical validation",
                        (
                            "Across the available odd/even studies, the "
                            "median half-difference proxy is "
                            f"{ratio:.2f} times the median propagated ROOT "
                            "statistical uncertainty. The split is used as "
                            "a stability check rather than as a replacement "
                            "for the propagated uncertainty."
                        ),
                    )
                )

    if relative_summary:

        source_medians = {}

        for source, column in SOURCE_SUMMARY_COLUMNS.items():

            values = []

            for dataframe in relative_summary.values():

                if column in dataframe.columns:
                    values.extend(
                        dataframe[
                            column
                        ]
                    )

            finite = finite_values(
                values
            )

            if len(
                finite
            ):
                source_medians[
                    source
                ] = float(
                    np.median(
                        finite
                    )
                )

        if source_medians:

            largest_source = max(
                source_medians,
                key=source_medians.get,
            )

            cards.append(
                (
                    "Relative systematic treatment",
                    (
                        "Systematic weights are expressed as correlated "
                        "variations relative to weight 0. Across the full "
                        "set of available bins and channels, "
                        f"{SOURCE_LABELS[largest_source]} has the largest "
                        "median relative contribution of the four reported "
                        "sources in the current outputs."
                    ),
                )
            )

    if final_results:

        total_percent = []

        for dataframe in final_results.values():

            if (
                "total_unc_percent"
                in dataframe.columns
            ):

                total_percent.extend(
                    dataframe[
                        "total_unc_percent"
                    ]
                )

        finite = finite_values(
            total_percent
        )

        if len(
            finite
        ):

            cards.append(
                (
                    "Final Pythia result",
                    (
                        "The final central values use the new "
                        "high-statistics weight-0 Pythia samples. "
                        "The total relative uncertainty across the final "
                        "bins ranges from "
                        f"{np.min(finite):.2f}% to "
                        f"{np.max(finite):.2f}% in the current outputs."
                    ),
                )
            )

    if not cards:

        cards.append(
            (
                "Analysis outputs",
                (
                    "The dashboard did not find enough generated CSV "
                    "outputs to calculate data-derived summary statements. "
                    "Run the analysis scripts, then rebuild this dashboard."
                ),
            )
        )

    html = ""

    for title, text in cards:

        html += (
            '<div class="finding-card">'
            f"<h3>{title}</h3>"
            f"<p>{text}</p>"
            "</div>"
        )

    return html


# ============================================================
# Figure to HTML
# ============================================================

def figure_div(
    figure,
):
    return pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs=False,
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
        },
    )


# ============================================================
# Dashboard HTML
# ============================================================

def build_dashboard_html(
    figures,
    key_findings_html,
    source_notes,
):
    plotly_js = get_plotlyjs()

    screens = [
        (
            "final",
            "Final result",
            figures[
                "final"
            ],
            (
                "<strong>What this shows.</strong> "
                "The central correction is C<sub>new</sub>, obtained from "
                "the new high-statistics Pythia weight-0 samples. "
                "σ<sub>stat</sub> comes directly from those new samples. "
                "Scale, PDF, shower and model uncertainties are the "
                "relative systematic uncertainties from the old weighted "
                "samples multiplied by C<sub>new</sub>. "
                "σ<sub>total</sub> combines statistical and systematic "
                "components in quadrature. Use the dropdown to switch "
                "between W⁺ and W⁻. "
                + source_notes[
                    "final"
                ]
            ),
        ),
        (
            "original",
            "Original corrections",
            figures[
                "original"
            ],
            (
                "<strong>What this shows.</strong> "
                "These are the original m<sub>T</sub><sup>W</sup>-selected "
                "parton-to-particle correction factors before the new "
                "high-statistics nominal samples were adopted. "
                "All four panels use the same x and y ranges so Pythia, "
                "Herwig, W⁺ and W⁻ can be compared directly. "
                "The error bars are the total uncertainties stored by the "
                "original correction builder. "
                + source_notes[
                    "original"
                ]
            ),
        ),
        (
            "evenodd",
            "Statistical validation",
            figures[
                "evenodd"
            ],
            (
                "<strong>What this shows.</strong> "
                "The event sample is split into odd and even tree entries "
                "without changing the physics selection. The upper panel "
                "compares C<sub>all</sub>, C<sub>odd</sub> and "
                "C<sub>even</sub>. The lower panel compares the split "
                "proxy ½|C<sub>odd</sub> − C<sub>even</sub>| with the "
                "propagated ROOT statistical uncertainty. The half-"
                "difference is a stability diagnostic, not an independent "
                "replacement for the statistical uncertainty. "
                + source_notes[
                    "evenodd"
                ]
            ),
        ),
        (
            "crosscheck",
            "New weight-0 cross check",
            figures[
                "crosscheck"
            ],
            (
                "<strong>What this shows.</strong> "
                "The reference Pythia correction is compared with the new "
                "high-statistics weight-0 correction. The second panel "
                "shows C<sub>new</sub> − C<sub>ref</sub>. The third panel "
                "expresses the difference in units of the quadrature "
                "combination of the two statistical uncertainties, which "
                "is only interpreted as an ordinary pull if the two samples "
                "are statistically independent. The new sample is used to "
                "test consistency and reduce the statistical uncertainty. "
                + source_notes[
                    "crosscheck"
                ]
            ),
        ),
        (
            "relative",
            "Relative systematics",
            figures[
                "relative"
            ],
            (
                "<strong>What this shows.</strong> "
                "For every systematic weight k, the analysis forms "
                "δ<sub>k</sub> = "
                "(C<sub>k</sub> − C₀) / C₀ using weight 0 from the same "
                "Monte Carlo event sample. This uses the statistical "
                "correlation between the nominal and varied weights. "
                "Scale, shower and model use envelopes of the relative "
                "variations while PDF uses an RMS. All four panels have "
                "matched axes. Click legend entries to isolate individual "
                "uncertainty sources. "
                + source_notes[
                    "relative"
                ]
            ),
        ),
        (
            "weights",
            "Individual weight responses",
            figures[
                "weights"
            ],
            (
                "<strong>What this shows.</strong> "
                "Each faint curve is one individual systematic weight "
                "expressed as (C<sub>k</sub> − C₀) / C₀. The dashed "
                "boundaries show the final envelope or RMS magnitude used "
                "for that source. Use the dropdown to switch between "
                "scale, PDF, shower and model variations. The four panels "
                "share the same x range and, for each selected source, the "
                "same y range so the channels can be compared directly. "
                + source_notes[
                    "weights"
                ]
            ),
        ),
        (
            "generator",
            "Pythia vs Herwig",
            figures[
                "generator"
            ],
            (
                "<strong>What this shows.</strong> "
                "The upper panel directly compares the original Pythia and "
                "Herwig correction factors for the selected W charge. The "
                "lower panel shows "
                "C<sub>Pythia</sub> − C<sub>Herwig</sub>. "
                "The dropdown switches between W⁺ and W⁻ while preserving "
                "the same axis definitions. "
                + source_notes[
                    "generator"
                ]
            ),
        ),
    ]

    nav_buttons = ""

    screen_html = ""

    for index, (
        screen_id,
        label,
        figure,
        explanation,
    ) in enumerate(
        screens
    ):

        active = (
            " active"
            if index == 0
            else ""
        )

        nav_buttons += (
            f'<button type="button" '
            f'class="nav-button{active}" '
            f'data-target="{screen_id}">'
            f"{label}"
            "</button>"
        )

        screen_html += (
            f'<section id="screen-{screen_id}" '
            f'class="screen{active}">'
            f'<div class="plot-wrap">'
            f"{figure_div(figure)}"
            f"</div>"
            f'<div class="explanation-box">'
            f"{explanation}"
            f"</div>"
            "</section>"
        )

    methodology = """
    <section id="screen-method" class="screen">
        <div class="method-grid">
            <div class="method-card">
                <h2>Fiducial selection</h2>
                <div class="equation-line">E<sub>T</sub><sup>miss</sup> &gt; 25 GeV</div>
                <div class="equation-line">p<sub>T</sub><sup>ℓ</sup> &gt; 20 GeV</div>
                <div class="equation-line">|η<sub>ℓ</sub>| &lt; 2.5</div>
                <div class="equation-line">m<sub>T</sub><sup>W</sup> &gt; 40 GeV</div>
            </div>

            <div class="method-card">
                <h2>Correction factor</h2>
                <div class="equation-large">
                    C<sub>i</sub> =
                    N<sub>parton,i</sub> /
                    N<sub>particle,i</sub>
                </div>
                <p>
                    The observable is the absolute lepton pseudorapidity
                    |η<sub>ℓ</sub>|. The same binning is used throughout the
                    original analysis, statistical checks and final result.
                </p>
            </div>

            <div class="method-card">
                <h2>Relative systematic variation</h2>
                <div class="equation-large">
                    δ<sub>i,k</sub> =
                    (C<sub>i,k</sub> − C<sub>i,0</sub>) /
                    C<sub>i,0</sub>
                </div>
                <p>
                    Weight k and weight 0 are evaluated on the same events,
                    so common statistical fluctuations largely cancel in the
                    relative variation.
                </p>
            </div>

            <div class="method-card">
                <h2>Final Pythia result</h2>
                <div class="equation-large">
                    C<sub>final</sub> = C<sub>new</sub>
                </div>
                <p>
                    The new high-statistics weight-0 Pythia sample supplies
                    the central value and statistical uncertainty. Relative
                    systematics from the old weighted samples are transferred
                    to that new nominal correction.
                </p>
            </div>
        </div>

        <div class="findings-grid">
            KEY_FINDINGS_PLACEHOLDER
        </div>

        <div class="explanation-box">
            <strong>How the analysis fits together.</strong>
            The transverse-mass selection defines the fiducial event sample.
            The original weighted samples provide both the nominal weight-0
            correction and all generator-weight variations. The odd/even
            split checks the size of the propagated statistical uncertainty.
            The new Pythia weight-0 samples then provide an independent
            high-statistics cross check. Systematic uncertainties are finally
            extracted as correlated relative variations around weight 0 and
            transferred onto the new Pythia nominal correction. New
            high-statistics Herwig weight-0 samples are not part of the current
            workflow, so the final nominal replacement is Pythia-only.
        </div>
    </section>
    """

    methodology = methodology.replace(
        "KEY_FINDINGS_PLACEHOLDER",
        key_findings_html,
    )

    nav_buttons += (
        '<button type="button" '
        'class="nav-button" '
        'data-target="method">'
        "Method & conclusions"
        "</button>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strange Proton Uncertainty Decomposition</title>
<script>{plotly_js}</script>
<style>
    :root {{
        --page: #f6f8fb;
        --panel: #ffffff;
        --text: #182230;
        --muted: #5f6b7a;
        --border: #dce3eb;
        --control: #edf2f7;
        --control-active: #dbe5ef;
    }}

    * {{
        box-sizing: border-box;
    }}

    body {{
        margin: 0;
        padding: 0;
        font-family: Arial, Helvetica, sans-serif;
        background: var(--page);
        color: var(--text);
    }}

    .page {{
        max-width: 1480px;
        margin: 0 auto;
        padding: 24px;
    }}

    .hero {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 16px;
    }}

    .hero h1 {{
        margin: 0 0 6px 0;
        font-size: 27px;
        font-weight: 650;
    }}

    .hero p {{
        margin: 0;
        color: var(--muted);
        line-height: 1.5;
    }}

    .navigation {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 16px;
    }}

    .nav-button {{
        min-height: 42px;
        padding: 9px 13px;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: var(--panel);
        color: var(--text);
        cursor: pointer;
        font-size: 14px;
    }}

    .nav-button.active {{
        background: var(--control-active);
        border-color: #aab7c4;
        font-weight: 650;
    }}

    .nav-button:focus-visible {{
        outline: 3px solid #9fb8d2;
        outline-offset: 2px;
    }}

    .screen {{
        display: none;
    }}

    .screen.active {{
        display: block;
    }}

    .plot-wrap {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 8px;
        overflow: hidden;
    }}

    .explanation-box {{
        margin-top: 14px;
        background: var(--panel);
        border: 1px solid var(--border);
        border-left: 5px solid #8798aa;
        border-radius: 10px;
        padding: 15px 17px;
        line-height: 1.55;
        color: #354354;
    }}

    .method-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
    }}

    .method-card {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 18px;
        min-width: 0;
    }}

    .method-card h2 {{
        margin: 0 0 13px 0;
        font-size: 18px;
    }}

    .method-card p {{
        margin: 12px 0 0 0;
        color: var(--muted);
        line-height: 1.5;
    }}

    .equation-line {{
        font-size: 17px;
        margin: 7px 0;
    }}

    .equation-large {{
        font-size: 21px;
        padding: 10px 0;
        word-break: break-word;
    }}

    .findings-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
        margin-top: 14px;
    }}

    .finding-card {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 17px;
    }}

    .finding-card h3 {{
        margin: 0 0 8px 0;
        font-size: 17px;
    }}

    .finding-card p {{
        margin: 0;
        line-height: 1.5;
        color: var(--muted);
    }}

    .footer {{
        margin-top: 18px;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.5;
        text-align: center;
    }}

    @media (max-width: 760px) {{
        .page {{
            padding: 12px;
        }}

        .hero h1 {{
            font-size: 22px;
        }}

        .method-grid,
        .findings-grid {{
            grid-template-columns: 1fr;
        }}

        .navigation {{
            display: grid;
            grid-template-columns: 1fr 1fr;
        }}

        .nav-button {{
            width: 100%;
        }}
    }}
</style>
</head>
<body>
<div class="page">

    <header class="hero">
        <h1>Uncertainty Decomposition of the Strange Proton</h1>
        <p>
            Interactive summary of the lepton-pseudorapidity W+c
            parton-to-particle correction-factor analysis. All plots are
            generated directly from the CSV outputs already produced by the
            repository.
        </p>
    </header>

    <nav class="navigation" aria-label="Dashboard sections">
        {nav_buttons}
    </nav>

    {screen_html}

    {methodology}

    <div class="footer">
        Generated by display_all_findings.py from the analysis outputs found
        in {WORK_DIR}
    </div>

</div>

<script>
(function () {{
    const buttons = Array.from(
        document.querySelectorAll(".nav-button")
    );

    const screens = Array.from(
        document.querySelectorAll(".screen")
    );

    function activate(target) {{
        buttons.forEach(function (button) {{
            button.classList.toggle(
                "active",
                button.dataset.target === target
            );
        }});

        screens.forEach(function (screen) {{
            screen.classList.toggle(
                "active",
                screen.id === "screen-" + target
            );
        }});

        window.setTimeout(function () {{
            const visible = document.getElementById(
                "screen-" + target
            );

            if (!visible) {{
                return;
            }}

            visible.querySelectorAll(
                ".plotly-graph-div"
            ).forEach(function (plot) {{
                if (
                    window.Plotly
                    && Plotly.Plots
                ) {{
                    Plotly.Plots.resize(plot);
                }}
            }});
        }}, 40);
    }}

    buttons.forEach(function (button) {{
        button.addEventListener(
            "click",
            function () {{
                activate(
                    button.dataset.target
                );
            }}
        );
    }});

    window.addEventListener(
        "resize",
        function () {{
            const active = document.querySelector(
                ".screen.active"
            );

            if (!active) {{
                return;
            }}

            active.querySelectorAll(
                ".plotly-graph-div"
            ).forEach(function (plot) {{
                if (
                    window.Plotly
                    && Plotly.Plots
                ) {{
                    Plotly.Plots.resize(plot);
                }}
            }});
        }}
    );
}})();
</script>

</body>
</html>
"""


# ============================================================
# Browser opening
# ============================================================

def open_in_browser(
    path,
):
    """
    Open in the normal Windows browser when running in WSL. Fall back
    to Python's webbrowser module outside WSL.
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

        return

    except Exception:
        pass

    try:

        import webbrowser

        webbrowser.open(
            path.as_uri()
        )

    except Exception as exc:

        print(
            f"Could not open browser automatically: {exc}"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build one interactive dashboard containing the "
            "main findings of the uncertainty-decomposition project."
        )
    )

    parser.add_argument(
        "--open",
        action="store_true",
        help=(
            "Open the generated HTML dashboard after creating it."
        ),
    )

    args = parser.parse_args()

    print()
    print(
        "============================================================"
    )
    print(
        " Strange Proton Uncertainty Decomposition Dashboard"
    )
    print(
        "============================================================"
    )
    print()

    print(
        f"Working directory:\n  {WORK_DIR}"
    )
    print()

    (
        original,
        original_paths,
    ) = load_original_corrections()

    (
        even_odd,
        even_odd_paths,
    ) = find_even_odd_csvs()

    (
        cross_check,
        cross_check_paths,
    ) = load_cross_check()

    (
        relative_summary,
        per_weight,
        relative_paths,
        per_weight_paths,
    ) = load_relative_uncertainties()

    (
        final_results,
        final_paths,
    ) = load_final_results()

    print(
        f"Original correction channels: {len(original)}"
    )
    print(
        f"Odd/even channels:            {len(even_odd)}"
    )
    print(
        f"Weight-0 cross-check channels:{len(cross_check):2d}"
    )
    print(
        f"Relative-systematic channels: {len(relative_summary)}"
    )
    print(
        f"Per-weight response channels: {len(per_weight)}"
    )
    print(
        f"Final nominal channels:       {len(final_results)}"
    )
    print()

    figures = {
        "final": build_final_figure(
            final_results,
            cross_check,
        ),
        "original": build_original_figure(
            original
        ),
        "evenodd": build_even_odd_figure(
            even_odd
        ),
        "crosscheck": build_cross_check_figure(
            cross_check
        ),
        "relative": build_relative_summary_figure(
            relative_summary
        ),
        "weights": build_weight_response_figure(
            relative_summary,
            per_weight,
        ),
        "generator": build_generator_comparison_figure(
            original
        ),
    }

    key_findings_html = build_key_findings_html(
        final_results=final_results,
        cross_check=cross_check,
        relative_summary=relative_summary,
        even_odd=even_odd,
    )

    source_notes = {
        "final": source_file_note(
            list(
                final_paths.values()
            )
        ),
        "original": source_file_note(
            list(
                original_paths.values()
            )
        ),
        "evenodd": source_file_note(
            list(
                even_odd_paths.values()
            )
        ),
        "crosscheck": source_file_note(
            list(
                cross_check_paths.values()
            )
        ),
        "relative": source_file_note(
            list(
                relative_paths.values()
            )
        ),
        "weights": source_file_note(
            list(
                per_weight_paths.values()
            )
        ),
        "generator": source_file_note(
            list(
                original_paths.values()
            )
        ),
    }

    html = build_dashboard_html(
        figures=figures,
        key_findings_html=key_findings_html,
        source_notes=source_notes,
    )

    OUTPUT_HTML.write_text(
        html,
        encoding="utf-8",
    )

    print(
        "Created dashboard:"
    )
    print(
        f"  {OUTPUT_HTML}"
    )
    print()

    print(
        "The dashboard contains:"
    )
    print(
        "  1. Final Pythia result"
    )
    print(
        "  2. Original four-panel correction factors"
    )
    print(
        "  3. Odd/even statistical validation"
    )
    print(
        "  4. New high-statistics weight-0 cross check"
    )
    print(
        "  5. Four-panel relative systematic decomposition"
    )
    print(
        "  6. Four-panel individual weight responses"
    )
    print(
        "  7. Pythia/Herwig generator comparison"
    )
    print(
        "  8. Method and data-derived conclusions"
    )
    print()

    if args.open:

        print(
            "Opening dashboard..."
        )

        open_in_browser(
            OUTPUT_HTML
        )


if __name__ == "__main__":
    main()
