#!/usr/bin/env python3

"""
display_all_findings.py

Final presentation dashboard for the Strange Proton Uncertainty
Decomposition analysis.

Run from the project root:

    python3 lepton_pseudorapidity_work_FINAL/display_all_findings.py

or:

    python3 lepton_pseudorapidity_work_FINAL/display_all_findings.py --open

The dashboard is generated from the completed analysis CSV outputs and
contains:

    1. Summary
    2. Final result
    3. Correction overview
    4. Statistical validation
    5. Relative systematics
    6. Individual weight responses
    7. Pythia versus Herwig
    8. Method and conclusions

The presentation deliberately uses proper mathematical symbols and HTML
subscripts/superscripts throughout.
"""

from __future__ import annotations

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
WORK_DIR = SCRIPT_DIR

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

VALIDATION_TEXT = (
    OUTPUT_DIR
    / "dashboard_validation_summary.txt"
)

FINAL_CSV_DIR = (
    WORK_DIR
    / "weights_relative_uncertainties"
    / "outputs"
    / "final_conservative"
    / "csv"
)

RELATIVE_CSV_DIR = (
    WORK_DIR
    / "weights_relative_uncertainties"
    / "outputs"
    / "csv"
)

EVEN_ODD_DIR = (
    WORK_DIR
    / "even_odd_uncertainties"
)

CORRECTION_CSV_DIR = (
    WORK_DIR
    / "MTWcut_build_corrections_etalepton_csv"
)


# ============================================================
# Analysis labels
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

CHARGE_GROUPS = {
    "plus": {
        "label": "W⁺",
        "pythia": "Pythia_plus",
        "herwig": "Herwig_plus",
    },
    "minus": {
        "label": "W⁻",
        "pythia": "Pythia_minus",
        "herwig": "Herwig_minus",
    },
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

SOURCE_METHODS = {
    "scale": "Envelope",
    "pdf": "RMS",
    "shower": "Envelope",
    "model": "Envelope",
}

SOURCE_PERCENT_COLUMNS = {
    "scale": "scale_unc_percent",
    "pdf": "pdf_unc_percent",
    "shower": "shower_unc_percent",
    "model": "model_unc_percent",
}

EXPECTED_VARIATION_COUNTS = {
    "Pythia_plus": {
        "scale": 8,
        "pdf": 282,
        "shower": 22,
        "model": 4,
    },
    "Pythia_minus": {
        "scale": 8,
        "pdf": 282,
        "shower": 22,
        "model": 4,
    },
    "Herwig_plus": {
        "scale": 8,
        "pdf": 275,
        "shower": 0,
        "model": 0,
    },
    "Herwig_minus": {
        "scale": 8,
        "pdf": 275,
        "shower": 0,
        "model": 0,
    },
}


# ============================================================
# Required schemas
# ============================================================

FINAL_REQUIRED = {
    "channel",
    "display",
    "generator",
    "charge",
    "bin",
    "bin_low_edge",
    "bin_up_edge",
    "nominal_correction",
    "stat_unc",
    "relative_scale_unc",
    "relative_pdf_unc",
    "relative_shower_unc",
    "relative_model_unc",
    "relative_total_systematic_unc",
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
    "high_stat_nominal_used",
}

RELATIVE_REQUIRED = {
    "pair_label",
    "bin",
    "bin_low_edge",
    "bin_up_edge",
    "nominal_correction_weight0",
    "relative_scale_unc",
    "relative_pdf_unc",
    "relative_shower_unc",
    "relative_model_unc",
    "relative_total_systematic_unc",
    "scale_unc_percent",
    "pdf_unc_percent",
    "shower_unc_percent",
    "model_unc_percent",
    "total_systematic_unc_percent",
}

WEIGHT_REQUIRED = {
    "pair_label",
    "weight_index",
    "weight_name",
    "category",
    "bin",
    "bin_low_edge",
    "bin_up_edge",
    "C_nominal_weight0",
    "C_weight",
    "C_weight_over_C_nominal",
    "signed_relative_shift",
    "absolute_relative_shift",
    "signed_relative_shift_percent",
}

EVEN_ODD_REQUIRED = {
    "pair_label",
    "bin",
    "bin_low_edge",
    "bin_up_edge",
    "C_all",
    "C_odd",
    "C_even",
    "split_stat_uncertainty_half_difference",
    "all_ratio_ROOT_stat_error",
}

CORRECTION_REQUIRED = {
    "pair_label",
    "observable",
    "bin",
    "bin_low_edge",
    "bin_up_edge",
    "correction_factor",
    "stat_unc",
    "scale_unc",
    "pdf_unc",
    "shower_unc",
    "model_unc",
    "total_unc",
}


# ============================================================
# Numerical tolerances
# ============================================================

ATOL = 5.0e-8
RTOL = 5.0e-6
BIN_ATOL = 1.0e-12


# ============================================================
# Generic helpers
# ============================================================

def read_csv_checked(
    path,
    required_columns,
):
    if not path.is_file():
        raise FileNotFoundError(
            "Required CSV file was not found:\n"
            f"  {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    if dataframe.empty:
        raise RuntimeError(
            f"CSV contains no rows:\n  {path}"
        )

    missing = (
        required_columns
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise RuntimeError(
            "CSV is missing required columns:\n"
            f"  {path}\n"
            f"Missing: {sorted(missing)}"
        )

    return add_eta_columns(
        dataframe
    )


def add_eta_columns(
    dataframe,
):
    result = dataframe.copy()

    result[
        "eta_centre"
    ] = (
        0.5
        * (
            pd.to_numeric(
                result[
                    "bin_low_edge"
                ]
            )
            + pd.to_numeric(
                result[
                    "bin_up_edge"
                ]
            )
        )
    )

    result[
        "eta_half_width"
    ] = (
        0.5
        * (
            pd.to_numeric(
                result[
                    "bin_up_edge"
                ]
            )
            - pd.to_numeric(
                result[
                    "bin_low_edge"
                ]
            )
        )
    )

    return result


def finite_values(
    values,
):
    array_values = np.asarray(
        values,
        dtype=float,
    )

    return array_values[
        np.isfinite(
            array_values
        )
    ]


def padded_range(
    values,
    fraction=0.08,
    include_zero=False,
):
    values = finite_values(
        values
    )

    if len(
        values
    ) == 0:
        return [
            0.0,
            1.0,
        ]

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

    span = (
        high
        - low
    )

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


def symmetric_range(
    values,
    fraction=0.10,
    minimum=1.0,
):
    values = finite_values(
        values
    )

    if len(
        values
    ) == 0:
        limit = minimum

    else:
        limit = max(
            minimum,
            float(
                np.max(
                    np.abs(
                        values
                    )
                )
            ),
        )

    limit *= (
        1.0
        + fraction
    )

    return [
        -limit,
        limit,
    ]


def allclose(
    left,
    right,
    atol=ATOL,
    rtol=RTOL,
):
    return np.allclose(
        np.asarray(
            left,
            dtype=float,
        ),
        np.asarray(
            right,
            dtype=float,
        ),
        atol=atol,
        rtol=rtol,
        equal_nan=False,
    )


def max_abs_difference(
    left,
    right,
):
    return float(
        np.max(
            np.abs(
                np.asarray(
                    left,
                    dtype=float,
                )
                - np.asarray(
                    right,
                    dtype=float,
                )
            )
        )
    )


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
# Data loading
# ============================================================

def load_final_results():
    results = {}

    for pair in (
        PAIR_ORDER
    ):
        path = (
            FINAL_CSV_DIR
            / f"{pair}_final_conservative.csv"
        )

        dataframe = read_csv_checked(
            path,
            FINAL_REQUIRED,
        )

        dataframe = (
            dataframe
            .sort_values(
                "bin"
            )
            .reset_index(
                drop=True
            )
        )

        results[
            pair
        ] = dataframe

    return results


def load_relative_results():
    summaries = {}
    per_weight = {}

    for pair in (
        PAIR_ORDER
    ):
        summary_path = (
            RELATIVE_CSV_DIR
            / f"{pair}_relative_uncertainties.csv"
        )

        weight_path = (
            RELATIVE_CSV_DIR
            / f"{pair}_per_weight_relative_shifts.csv"
        )

        summaries[
            pair
        ] = (
            read_csv_checked(
                summary_path,
                RELATIVE_REQUIRED,
            )
            .sort_values(
                "bin"
            )
            .reset_index(
                drop=True
            )
        )

        per_weight[
            pair
        ] = (
            read_csv_checked(
                weight_path,
                WEIGHT_REQUIRED,
            )
            .sort_values(
                [
                    "category",
                    "weight_index",
                    "bin",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    return (
        summaries,
        per_weight,
    )


def load_correction_results():
    results = {}

    for pair in (
        PAIR_ORDER
    ):
        path = (
            CORRECTION_CSV_DIR
            / f"{pair}_etalepton.csv"
        )

        results[
            pair
        ] = (
            read_csv_checked(
                path,
                CORRECTION_REQUIRED,
            )
            .sort_values(
                "bin"
            )
            .reset_index(
                drop=True
            )
        )

    return results


def find_even_odd_csvs():
    candidates = []

    if EVEN_ODD_DIR.is_dir():
        candidates.extend(
            EVEN_ODD_DIR.rglob(
                "even_odd_*.csv"
            )
        )

    # Also search within the final work directory so the dashboard remains
    # robust to a slightly different output subfolder name.
    candidates.extend(
        WORK_DIR.rglob(
            "even_odd_*.csv"
        )
    )

    unique_paths = []

    seen = set()

    for path in candidates:
        resolved = path.resolve()

        if resolved in seen:
            continue

        seen.add(
            resolved
        )

        unique_paths.append(
            path
        )

    results = {}

    for path in sorted(
        unique_paths
    ):
        try:
            dataframe = read_csv_checked(
                path,
                EVEN_ODD_REQUIRED,
            )

        except Exception:
            continue

        pair = str(
            dataframe.iloc[
                0
            ][
                "pair_label"
            ]
        )

        if (
            pair in PAIR_ORDER
            and pair not in results
        ):
            results[
                pair
            ] = (
                dataframe
                .sort_values(
                    "bin"
                )
                .reset_index(
                    drop=True
                )
            )

    missing = [
        pair
        for pair in PAIR_ORDER
        if pair not in results
    ]

    if missing:
        raise RuntimeError(
            "Could not find completed odd/even CSV output for:\n  "
            + "\n  ".join(
                missing
            )
        )

    return results


# ============================================================
# Pipeline validation
# ============================================================

def validate_common_binning(
    dataframes,
    label,
):
    first_key = next(
        iter(
            dataframes
        )
    )

    reference = dataframes[
        first_key
    ]

    for key, dataframe in (
        dataframes.items()
    ):
        if len(
            dataframe
        ) != len(
            reference
        ):
            raise RuntimeError(
                f"{label}: {key} has a different number of bins."
            )

        if not np.array_equal(
            dataframe[
                "bin"
            ].to_numpy(
                dtype=int
            ),
            reference[
                "bin"
            ].to_numpy(
                dtype=int
            ),
        ):
            raise RuntimeError(
                f"{label}: {key} has different bin indices."
            )

        for column in (
            "bin_low_edge",
            "bin_up_edge",
        ):
            if not allclose(
                dataframe[
                    column
                ],
                reference[
                    column
                ],
                atol=BIN_ATOL,
                rtol=0.0,
            ):
                raise RuntimeError(
                    f"{label}: {key} has different eta binning."
                )


def validate_pipeline(
    final_results,
    relative_summaries,
    per_weight,
    correction_results,
    even_odd,
):
    validate_common_binning(
        final_results,
        "Final result",
    )

    validate_common_binning(
        relative_summaries,
        "Relative systematics",
    )

    validate_common_binning(
        correction_results,
        "Correction factors",
    )

    validate_common_binning(
        even_odd,
        "Odd/even validation",
    )

    summary_lines = [
        "Dashboard input validation",
        "==========================",
        "",
    ]

    for pair in (
        PAIR_ORDER
    ):
        final = final_results[
            pair
        ]

        relative = relative_summaries[
            pair
        ]

        correction = correction_results[
            pair
        ]

        weights = per_weight[
            pair
        ]

        split = even_odd[
            pair
        ]

        if not (
            final[
                "channel"
            ].astype(
                str
            )
            == pair
        ).all():
            raise RuntimeError(
                f"{pair}: channel label mismatch in final CSV."
            )

        high_stat_flags = (
            final[
                "high_stat_nominal_used"
            ]
            .astype(
                str
            )
            .str.strip()
            .str.lower()
        )

        if not (
            high_stat_flags
            == "false"
        ).all():
            raise RuntimeError(
                f"{pair}: final CSV does not use the expected final treatment."
            )

        if not allclose(
            final[
                "nominal_correction"
            ],
            correction[
                "correction_factor"
            ],
        ):
            raise RuntimeError(
                f"{pair}: final and correction-builder central values differ."
            )

        for source in (
            SOURCE_ORDER
        ):
            final_absolute = final[
                f"{source}_unc"
            ]

            correction_absolute = correction[
                f"{source}_unc"
            ]

            if not allclose(
                final_absolute,
                correction_absolute,
            ):
                raise RuntimeError(
                    f"{pair}: {source} uncertainty differs between "
                    "final and correction-builder outputs."
                )

            final_relative = final[
                f"relative_{source}_unc"
            ]

            relative_source = relative[
                f"relative_{source}_unc"
            ]

            if not allclose(
                final_relative,
                relative_source,
            ):
                raise RuntimeError(
                    f"{pair}: {source} relative uncertainty differs "
                    "between final and relative-systematic outputs."
                )

        if not allclose(
            final[
                "total_unc"
            ],
            correction[
                "total_unc"
            ],
        ):
            raise RuntimeError(
                f"{pair}: total uncertainty differs between final and "
                "correction-builder outputs."
            )

        recomputed_systematic = np.sqrt(
            final[
                "scale_unc"
            ].to_numpy(
                dtype=float
            ) ** 2
            + final[
                "pdf_unc"
            ].to_numpy(
                dtype=float
            ) ** 2
            + final[
                "shower_unc"
            ].to_numpy(
                dtype=float
            ) ** 2
            + final[
                "model_unc"
            ].to_numpy(
                dtype=float
            ) ** 2
        )

        if not allclose(
            recomputed_systematic,
            final[
                "systematic_unc"
            ],
        ):
            raise RuntimeError(
                f"{pair}: systematic quadrature closure failed."
            )

        recomputed_total = np.sqrt(
            final[
                "stat_unc"
            ].to_numpy(
                dtype=float
            ) ** 2
            + recomputed_systematic ** 2
        )

        if not allclose(
            recomputed_total,
            final[
                "total_unc"
            ],
        ):
            raise RuntimeError(
                f"{pair}: total quadrature closure failed."
            )

        expected_counts = (
            EXPECTED_VARIATION_COUNTS[
                pair
            ]
        )

        counts = {}

        for source in (
            SOURCE_ORDER
        ):
            subset = weights[
                weights[
                    "category"
                ].astype(
                    str
                ).str.lower()
                == source
            ]

            counts[
                source
            ] = int(
                subset[
                    "weight_index"
                ].nunique()
            )

            if (
                counts[
                    source
                ]
                != expected_counts[
                    source
                ]
            ):
                raise RuntimeError(
                    f"{pair}: expected {expected_counts[source]} "
                    f"{source} variations, found {counts[source]}."
                )

        if not allclose(
            split[
                "C_all"
            ],
            correction[
                "correction_factor"
            ],
        ):
            raise RuntimeError(
                f"{pair}: odd/even all-event correction does not match "
                "the final correction-builder central value."
            )

        summary_lines.extend([
            f"{pair}",
            "-" * len(
                pair
            ),
            (
                "  max final/correction central-value difference: "
                f"{max_abs_difference(final['nominal_correction'], correction['correction_factor']):.3e}"
            ),
            (
                "  max total-uncertainty closure difference: "
                f"{max_abs_difference(recomputed_total, final['total_unc']):.3e}"
            ),
            (
                "  variations: "
                f"scale={counts['scale']}, "
                f"PDF={counts['pdf']}, "
                f"shower={counts['shower']}, "
                f"model={counts['model']}"
            ),
            "",
        ])

    VALIDATION_TEXT.write_text(
        "\n".join(
            summary_lines
        )
        + "\n",
        encoding="utf-8",
    )


# ============================================================
# Shared plot ranges
# ============================================================

def final_correction_range(
    final_results,
):
    values = []

    for dataframe in (
        final_results.values()
    ):
        central = dataframe[
            "nominal_correction"
        ].to_numpy(
            dtype=float
        )

        total = dataframe[
            "total_unc"
        ].to_numpy(
            dtype=float
        )

        values.extend(
            central
            - total
        )

        values.extend(
            central
            + total
        )

    return padded_range(
        values,
        fraction=0.10,
    )


def absolute_uncertainty_range(
    final_results,
):
    values = []

    for dataframe in (
        final_results.values()
    ):
        for column in (
            "stat_unc",
            "scale_unc",
            "pdf_unc",
            "shower_unc",
            "model_unc",
            "systematic_unc",
            "total_unc",
        ):
            values.extend(
                dataframe[
                    column
                ].to_numpy(
                    dtype=float
                )
            )

    return padded_range(
        values,
        fraction=0.08,
        include_zero=True,
    )


def relative_systematic_range(
    relative_summaries,
):
    values = []

    for dataframe in (
        relative_summaries.values()
    ):
        for column in (
            "scale_unc_percent",
            "pdf_unc_percent",
            "shower_unc_percent",
            "model_unc_percent",
            "total_systematic_unc_percent",
        ):
            values.extend(
                dataframe[
                    column
                ].to_numpy(
                    dtype=float
                )
            )

    return padded_range(
        values,
        fraction=0.08,
        include_zero=True,
    )


# ============================================================
# Figure: summary
# ============================================================

def build_summary_figure(
    final_results,
):
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "W⁺",
            "W⁻",
        ],
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    y_range = final_correction_range(
        final_results
    )

    for column_index, charge in enumerate(
        [
            "plus",
            "minus",
        ],
        start=1,
    ):
        for generator_key, generator_label in (
            (
                "pythia",
                "Pythia",
            ),
            (
                "herwig",
                "Herwig",
            ),
        ):
            pair = CHARGE_GROUPS[
                charge
            ][
                generator_key
            ]

            dataframe = final_results[
                pair
            ]

            figure.add_trace(
                go.Scatter(
                    x=dataframe[
                        "eta_centre"
                    ],
                    y=dataframe[
                        "nominal_correction"
                    ],
                    mode="lines+markers",
                    name=generator_label,
                    legendgroup=generator_label,
                    showlegend=(
                        column_index
                        == 1
                    ),
                    error_y=dict(
                        type="data",
                        array=dataframe[
                            "total_unc"
                        ],
                        visible=True,
                    ),
                    customdata=np.column_stack([
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
                            "systematic_unc"
                        ],
                        dataframe[
                            "total_unc"
                        ],
                    ]),
                    hovertemplate=(
                        "|η<sub>ℓ</sub>| bin: "
                        "%{customdata[0]:.2f}–"
                        "%{customdata[1]:.2f}"
                        "<br>C = %{y:.6f}"
                        "<br>σ<sub>stat</sub> = %{customdata[2]:.6f}"
                        "<br>σ<sub>syst</sub> = %{customdata[3]:.6f}"
                        "<br>σ<sub>total</sub> = %{customdata[4]:.6f}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=column_index,
            )

    figure.update_layout(
        title=dict(
            text=(
                "Final parton-to-particle correction factors"
                "<br><sup>"
                "Total uncertainty shown · synchronized generator comparison"
                "</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        height=650,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(
            l=85,
            r=40,
            t=135,
            b=70,
        ),
    )

    for column_index in (
        1,
        2,
    ):
        figure.update_xaxes(
            title_text="Lepton |η<sub>ℓ</sub>|",
            range=[
                0.0,
                2.5,
            ],
            matches="x",
            row=1,
            col=column_index,
        )

        figure.update_yaxes(
            title_text=(
                "Correction factor C"
                if column_index == 1
                else None
            ),
            range=y_range,
            matches="y",
            row=1,
            col=column_index,
        )

    return figure


# ============================================================
# Figure: detailed final result
# ============================================================

def build_final_result_figure(
    final_results,
):
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=[
            "Correction factor",
            "Uncertainty decomposition",
        ],
        row_heights=[
            0.58,
            0.42,
        ],
    )

    trace_groups = {}

    correction_range = final_correction_range(
        final_results
    )

    percent_values = []

    for dataframe in (
        final_results.values()
    ):
        for column in (
            "stat_unc_percent",
            "scale_unc_percent",
            "pdf_unc_percent",
            "shower_unc_percent",
            "model_unc_percent",
            "systematic_unc_percent",
            "total_unc_percent",
        ):
            percent_values.extend(
                dataframe[
                    column
                ]
            )

    percent_range = padded_range(
        percent_values,
        fraction=0.08,
        include_zero=True,
    )

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
            "systematic_unc_percent",
            "Total systematic",
        ),
        (
            "total_unc_percent",
            "Total",
        ),
    ]

    for pair_index, pair in enumerate(
        PAIR_ORDER
    ):
        dataframe = final_results[
            pair
        ]

        visible = (
            pair_index
            == 0
        )

        trace_groups[
            pair
        ] = []

        x = dataframe[
            "eta_centre"
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
                x=x,
                y=dataframe[
                    "nominal_correction"
                ],
                mode="lines+markers",
                name="Statistical uncertainty",
                visible=visible,
                error_y=dict(
                    type="data",
                    array=dataframe[
                        "stat_unc"
                    ],
                    visible=True,
                ),
                customdata=np.column_stack([
                    dataframe[
                        "bin_low_edge"
                    ],
                    dataframe[
                        "bin_up_edge"
                    ],
                    dataframe[
                        "stat_unc"
                    ],
                ]),
                hovertemplate=(
                    "|η<sub>ℓ</sub>| bin: "
                    "%{customdata[0]:.2f}–"
                    "%{customdata[1]:.2f}"
                    "<br>C = %{y:.6f}"
                    "<br>σ<sub>stat</sub> = %{customdata[2]:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
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
                x=x,
                y=dataframe[
                    "nominal_correction"
                ],
                mode="markers",
                name="Total uncertainty",
                visible=visible,
                marker=dict(
                    symbol="circle-open",
                    size=9,
                ),
                error_y=dict(
                    type="data",
                    array=dataframe[
                        "total_unc"
                    ],
                    visible=True,
                ),
                customdata=np.column_stack([
                    dataframe[
                        "total_unc"
                    ],
                    dataframe[
                        "systematic_unc"
                    ],
                ]),
                hovertemplate=(
                    "|η<sub>ℓ</sub>| = %{x:.3f}"
                    "<br>C = %{y:.6f}"
                    "<br>σ<sub>syst</sub> = %{customdata[1]:.6f}"
                    "<br>σ<sub>total</sub> = %{customdata[0]:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        for column, label in (
            component_definitions
        ):
            trace_groups[
                pair
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

    buttons = []

    for pair in (
        PAIR_ORDER
    ):
        visible = [
            False
        ] * len(
            figure.data
        )

        for trace_index in (
            trace_groups[
                pair
            ]
        ):
            visible[
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
                        "visible": visible,
                    },
                    {
                        "title.text": (
                            "Final correction and uncertainty decomposition"
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
                "Final correction and uncertainty decomposition"
                "<br><sup>Pythia W⁺</sup>"
            ),
            x=0.5,
            xanchor="center",
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(
            l=95,
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
        range=correction_range,
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text="Uncertainty (%)",
        range=percent_range,
        row=2,
        col=1,
    )

    return figure


# ============================================================
# Figure: correction overview
# ============================================================

def build_correction_overview_figure(
    final_results,
):
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            PAIR_LABELS[
                pair
            ]
            for pair in (
                PAIR_ORDER
            )
        ],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
    )

    y_range = final_correction_range(
        final_results
    )

    for pair_index, pair in enumerate(
        PAIR_ORDER
    ):
        dataframe = final_results[
            pair
        ]

        row, col = PAIR_POSITIONS[
            pair
        ]

        figure.add_trace(
            go.Scatter(
                x=dataframe[
                    "eta_centre"
                ],
                y=dataframe[
                    "nominal_correction"
                ],
                mode="lines+markers",
                name="Correction factor",
                showlegend=(
                    pair_index
                    == 0
                ),
                error_y=dict(
                    type="data",
                    array=dataframe[
                        "total_unc"
                    ],
                    visible=True,
                ),
                customdata=np.column_stack([
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
                        "systematic_unc"
                    ],
                    dataframe[
                        "total_unc"
                    ],
                ]),
                hovertemplate=(
                    "|η<sub>ℓ</sub>| bin: "
                    "%{customdata[0]:.2f}–"
                    "%{customdata[1]:.2f}"
                    "<br>C = %{y:.6f}"
                    "<br>σ<sub>stat</sub> = %{customdata[2]:.6f}"
                    "<br>σ<sub>syst</sub> = %{customdata[3]:.6f}"
                    "<br>σ<sub>total</sub> = %{customdata[4]:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    figure.update_layout(
        title=dict(
            text=(
                "Correction-factor overview"
                "<br><sup>"
                "All four channels · total uncertainty · synchronized axes"
                "</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        height=850,
        hovermode="closest",
        margin=dict(
            l=85,
            r=40,
            t=135,
            b=70,
        ),
    )

    for pair in (
        PAIR_ORDER
    ):
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
                "Correction factor C"
                if col == 1
                else None
            ),
            range=y_range,
            matches="y",
            row=row,
            col=col,
        )

    return figure


# ============================================================
# Figure: statistical validation
# ============================================================

def build_statistical_validation_figure(
    even_odd,
):
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            PAIR_LABELS[
                pair
            ]
            for pair in (
                PAIR_ORDER
            )
        ],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
    )

    mode_trace_indices = {
        "split": [],
        "uncertainty": [],
    }

    correction_values = []
    uncertainty_values = []

    for pair_index, pair in enumerate(
        PAIR_ORDER
    ):
        dataframe = even_odd[
            pair
        ]

        row, col = PAIR_POSITIONS[
            pair
        ]

        for column, label, marker_symbol in (
            (
                "C_all",
                "All events",
                "circle",
            ),
            (
                "C_odd",
                "Odd entries",
                "diamond",
            ),
            (
                "C_even",
                "Even entries",
                "square",
            ),
        ):
            trace_index = len(
                figure.data
            )

            mode_trace_indices[
                "split"
            ].append(
                trace_index
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
                    legendgroup=label,
                    showlegend=(
                        pair_index
                        == 0
                    ),
                    visible=True,
                    marker=dict(
                        symbol=marker_symbol,
                    ),
                    hovertemplate=(
                        "|η<sub>ℓ</sub>| = %{x:.3f}"
                        f"<br>{label}: %{{y:.6f}}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

            correction_values.extend(
                dataframe[
                    column
                ]
            )

        for column, label in (
            (
                "split_stat_uncertainty_half_difference",
                "½|C<sub>odd</sub> − C<sub>even</sub>|",
            ),
            (
                "all_ratio_ROOT_stat_error",
                "Propagated σ<sub>stat</sub>",
            ),
        ):
            trace_index = len(
                figure.data
            )

            mode_trace_indices[
                "uncertainty"
            ].append(
                trace_index
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
                    legendgroup=label,
                    showlegend=(
                        pair_index
                        == 0
                    ),
                    visible=False,
                    hovertemplate=(
                        "|η<sub>ℓ</sub>| = %{x:.3f}"
                        f"<br>{label}: %{{y:.6f}}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

            uncertainty_values.extend(
                dataframe[
                    column
                ]
            )

    split_range = padded_range(
        correction_values,
        fraction=0.08,
    )

    uncertainty_range = padded_range(
        uncertainty_values,
        fraction=0.10,
        include_zero=True,
    )

    buttons = []

    for mode_key, label, y_range, y_title in (
        (
            "split",
            "Odd/even correction split",
            split_range,
            "Correction factor C",
        ),
        (
            "uncertainty",
            "Statistical uncertainty comparison",
            uncertainty_range,
            "Statistical uncertainty",
        ),
    ):
        visible = [
            False
        ] * len(
            figure.data
        )

        for trace_index in (
            mode_trace_indices[
                mode_key
            ]
        ):
            visible[
                trace_index
            ] = True

        layout_updates = {
            "title.text": (
                "Statistical validation"
                "<br><sup>"
                + label
                + " · synchronized across all four channels"
                "</sup>"
            ),
        }

        for axis_index in range(
            1,
            5,
        ):
            axis_name = (
                "yaxis"
                if axis_index
                == 1
                else f"yaxis{axis_index}"
            )

            layout_updates[
                f"{axis_name}.range"
            ] = y_range

            layout_updates[
                f"{axis_name}.title.text"
            ] = (
                y_title
                if axis_index in (
                    1,
                    3,
                )
                else ""
            )

        buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {
                        "visible": visible,
                    },
                    layout_updates,
                ],
            )
        )

    figure.update_layout(
        title=dict(
            text=(
                "Statistical validation"
                "<br><sup>"
                "Odd/even correction split · synchronized across all four channels"
                "</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        height=880,
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(
            l=95,
            r=40,
            t=160,
            b=70,
        ),
    )

    for pair in (
        PAIR_ORDER
    ):
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
                "Correction factor C"
                if col == 1
                else None
            ),
            range=split_range,
            matches="y",
            row=row,
            col=col,
        )

    return figure


# ============================================================
# Figure: relative systematics
# ============================================================

def build_relative_systematics_figure(
    relative_summaries,
):
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            PAIR_LABELS[
                pair
            ]
            for pair in (
                PAIR_ORDER
            )
        ],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
    )

    y_range = relative_systematic_range(
        relative_summaries
    )

    for pair_index, pair in enumerate(
        PAIR_ORDER
    ):
        dataframe = relative_summaries[
            pair
        ]

        row, col = PAIR_POSITIONS[
            pair
        ]

        for source in (
            SOURCE_ORDER
        ):
            figure.add_trace(
                go.Scatter(
                    x=dataframe[
                        "eta_centre"
                    ],
                    y=dataframe[
                        SOURCE_PERCENT_COLUMNS[
                            source
                        ]
                    ],
                    mode="lines+markers",
                    name=SOURCE_LABELS[
                        source
                    ],
                    legendgroup=source,
                    showlegend=(
                        pair_index
                        == 0
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
                    pair_index
                    == 0
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
                    "<br>σ<sub>syst</sub> / |C| = %{y:.4f}%"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    figure.update_layout(
        title=dict(
            text=(
                "Relative systematic uncertainty decomposition"
                "<br><sup>"
                "All variations relative to nominal weight 0 · synchronized axes"
                "</sup>"
            ),
            x=0.5,
            xanchor="center",
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

    for pair in (
        PAIR_ORDER
    ):
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
                "Relative uncertainty (%)"
                if col == 1
                else None
            ),
            range=y_range,
            matches="y",
            row=row,
            col=col,
        )

    return figure


# ============================================================
# Figure: individual weight responses
# ============================================================

def source_ranges_for_weight_responses(
    relative_summaries,
    per_weight,
):
    ranges = {}

    for source in (
        SOURCE_ORDER
    ):
        values = []

        for pair in (
            PAIR_ORDER
        ):
            subset = per_weight[
                pair
            ][
                per_weight[
                    pair
                ][
                    "category"
                ].astype(
                    str
                ).str.lower()
                == source
            ]

            values.extend(
                subset[
                    "signed_relative_shift_percent"
                ].to_numpy(
                    dtype=float
                )
            )

            boundary = relative_summaries[
                pair
            ][
                SOURCE_PERCENT_COLUMNS[
                    source
                ]
            ].to_numpy(
                dtype=float
            )

            values.extend(
                boundary
            )

            values.extend(
                -boundary
            )

        ranges[
            source
        ] = symmetric_range(
            values,
            fraction=0.08,
            minimum=0.2,
        )

    return ranges


def build_weight_response_figure(
    relative_summaries,
    per_weight,
):
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            PAIR_LABELS[
                pair
            ]
            for pair in (
                PAIR_ORDER
            )
        ],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
    )

    source_trace_indices = {
        source: []
        for source in (
            SOURCE_ORDER
        )
    }

    source_ranges = (
        source_ranges_for_weight_responses(
            relative_summaries,
            per_weight,
        )
    )

    initial_source = "scale"

    for source in (
        SOURCE_ORDER
    ):
        for pair in (
            PAIR_ORDER
        ):
            dataframe = per_weight[
                pair
            ]

            summary = relative_summaries[
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

            weight_indices = sorted(
                subset[
                    "weight_index"
                ].unique()
            )

            if not weight_indices:
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
                        x=[
                            1.25
                        ],
                        y=[
                            0.0
                        ],
                        mode="text",
                        text=[
                            "No common "
                            + SOURCE_LABELS[
                                source
                            ]
                            + " variations<br>available for this channel"
                        ],
                        showlegend=False,
                        visible=(
                            source
                            == initial_source
                        ),
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

            for weight_index in (
                weight_indices
            ):
                weight_data = (
                    subset[
                        subset[
                            "weight_index"
                        ]
                        == weight_index
                    ]
                    .sort_values(
                        "bin"
                    )
                )

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
                        customdata=np.column_stack([
                            weight_data[
                                "bin_low_edge"
                            ],
                            weight_data[
                                "bin_up_edge"
                            ],
                        ]),
                        hovertemplate=(
                            f"Weight {int(weight_index)}"
                            f"<br>{weight_name}"
                            "<br>|η<sub>ℓ</sub>| bin: "
                            "%{customdata[0]:.2f}–"
                            "%{customdata[1]:.2f}"
                            "<br>(C<sub>k</sub> − C₀) / C₀ "
                            "= %{y:.4f}%"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

            if weight_indices:
                boundary = summary[
                    SOURCE_PERCENT_COLUMNS[
                        source
                    ]
                ]

                for sign in (
                    1.0,
                    -1.0,
                ):
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
                                == "Pythia_plus"
                                and sign
                                > 0
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

    buttons = []

    total_traces = len(
        figure.data
    )

    for source in (
        SOURCE_ORDER
    ):
        visible = [
            False
        ] * total_traces

        for trace_index in (
            source_trace_indices[
                source
            ]
        ):
            visible[
                trace_index
            ] = True

        layout_updates = {
            "title.text": (
                "Individual systematic-weight responses"
                "<br><sup>"
                + SOURCE_LABELS[
                    source
                ]
                + " · "
                + SOURCE_METHODS[
                    source
                ]
                + " · synchronized y-scale"
                "</sup>"
            ),
        }

        for axis_index in range(
            1,
            5,
        ):
            axis_name = (
                "yaxis"
                if axis_index
                == 1
                else f"yaxis{axis_index}"
            )

            layout_updates[
                f"{axis_name}.range"
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
                    layout_updates,
                ],
            )
        )

    figure.update_layout(
        title=dict(
            text=(
                "Individual systematic-weight responses"
                "<br><sup>"
                "Scale · Envelope · synchronized y-scale"
                "</sup>"
            ),
            x=0.5,
            xanchor="center",
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
            t=160,
            b=70,
        ),
    )

    initial_range = source_ranges[
        initial_source
    ]

    for pair in (
        PAIR_ORDER
    ):
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
                if col == 1
                else None
            ),
            range=initial_range,
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
    final_results,
):
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "W⁺ correction factors",
            "W⁻ correction factors",
            "W⁺ generator difference",
            "W⁻ generator difference",
        ],
        shared_xaxes="columns",
        shared_yaxes="rows",
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
        row_heights=[
            0.62,
            0.38,
        ],
    )

    correction_values = []
    difference_values = []

    for column_index, charge in enumerate(
        [
            "plus",
            "minus",
        ],
        start=1,
    ):
        pythia = final_results[
            CHARGE_GROUPS[
                charge
            ][
                "pythia"
            ]
        ]

        herwig = final_results[
            CHARGE_GROUPS[
                charge
            ][
                "herwig"
            ]
        ]

        for dataframe, label in (
            (
                pythia,
                "Pythia",
            ),
            (
                herwig,
                "Herwig",
            ),
        ):
            figure.add_trace(
                go.Scatter(
                    x=dataframe[
                        "eta_centre"
                    ],
                    y=dataframe[
                        "nominal_correction"
                    ],
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    showlegend=(
                        column_index
                        == 1
                    ),
                    error_y=dict(
                        type="data",
                        array=dataframe[
                            "total_unc"
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
                col=column_index,
            )

            correction_values.extend(
                dataframe[
                    "nominal_correction"
                ]
                - dataframe[
                    "total_unc"
                ]
            )

            correction_values.extend(
                dataframe[
                    "nominal_correction"
                ]
                + dataframe[
                    "total_unc"
                ]
            )

        difference = (
            pythia[
                "nominal_correction"
            ].to_numpy(
                dtype=float
            )
            - herwig[
                "nominal_correction"
            ].to_numpy(
                dtype=float
            )
        )

        combined_uncertainty = np.sqrt(
            pythia[
                "total_unc"
            ].to_numpy(
                dtype=float
            ) ** 2
            + herwig[
                "total_unc"
            ].to_numpy(
                dtype=float
            ) ** 2
        )

        figure.add_trace(
            go.Scatter(
                x=pythia[
                    "eta_centre"
                ],
                y=difference,
                mode="lines+markers",
                name="C<sub>Pythia</sub> − C<sub>Herwig</sub>",
                legendgroup="generator_difference",
                showlegend=(
                    column_index
                    == 1
                ),
                error_y=dict(
                    type="data",
                    array=combined_uncertainty,
                    visible=True,
                ),
                customdata=np.column_stack([
                    combined_uncertainty,
                ]),
                hovertemplate=(
                    "|η<sub>ℓ</sub>| = %{x:.3f}"
                    "<br>C<sub>Pythia</sub> − C<sub>Herwig</sub> "
                    "= %{y:.6f}"
                    "<br>combined uncertainty = %{customdata[0]:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=column_index,
        )

        difference_values.extend(
            difference
            - combined_uncertainty
        )

        difference_values.extend(
            difference
            + combined_uncertainty
        )

    correction_y_range = padded_range(
        correction_values,
        fraction=0.08,
    )

    difference_y_range = symmetric_range(
        difference_values,
        fraction=0.08,
        minimum=0.02,
    )

    figure.update_layout(
        title=dict(
            text=(
                "Pythia versus Herwig"
                "<br><sup>"
                "Final correction factors and generator difference · total uncertainties"
                "</sup>"
            ),
            x=0.5,
            xanchor="center",
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
            l=95,
            r=40,
            t=145,
            b=70,
        ),
    )

    for column_index in (
        1,
        2,
    ):
        figure.update_xaxes(
            range=[
                0.0,
                2.5,
            ],
            matches=(
                "x"
                if column_index
                == 1
                else "x2"
            ),
            row=1,
            col=column_index,
        )

        figure.update_xaxes(
            title_text="Lepton |η<sub>ℓ</sub>|",
            range=[
                0.0,
                2.5,
            ],
            matches=(
                "x"
                if column_index
                == 1
                else "x2"
            ),
            row=2,
            col=column_index,
        )

        figure.update_yaxes(
            title_text=(
                "Correction factor C"
                if column_index
                == 1
                else None
            ),
            range=correction_y_range,
            matches="y",
            row=1,
            col=column_index,
        )

        figure.update_yaxes(
            title_text=(
                "C<sub>Pythia</sub> − C<sub>Herwig</sub>"
                if column_index
                == 1
                else None
            ),
            range=difference_y_range,
            matches="y3",
            zeroline=True,
            zerolinewidth=2,
            row=2,
            col=column_index,
        )

    return figure


# ============================================================
# Data-derived summary cards
# ============================================================

def build_summary_cards(
    final_results,
    relative_summaries,
    even_odd,
):
    all_corrections = np.concatenate([
        dataframe[
            "nominal_correction"
        ].to_numpy(
            dtype=float
        )
        for dataframe in (
            final_results.values()
        )
    ])

    all_total_percent = np.concatenate([
        dataframe[
            "total_unc_percent"
        ].to_numpy(
            dtype=float
        )
        for dataframe in (
            final_results.values()
        )
    ])

    pythia_source_medians = {}

    for source in (
        SOURCE_ORDER
    ):
        values = np.concatenate([
            relative_summaries[
                pair
            ][
                SOURCE_PERCENT_COLUMNS[
                    source
                ]
            ].to_numpy(
                dtype=float
            )
            for pair in (
                "Pythia_plus",
                "Pythia_minus",
            )
        ])

        pythia_source_medians[
            source
        ] = float(
            np.median(
                values
            )
        )

    dominant_source = max(
        pythia_source_medians,
        key=pythia_source_medians.get,
    )

    split_values = []
    propagated_values = []

    for dataframe in (
        even_odd.values()
    ):
        split_values.extend(
            dataframe[
                "split_stat_uncertainty_half_difference"
            ].to_numpy(
                dtype=float
            )
        )

        propagated_values.extend(
            dataframe[
                "all_ratio_ROOT_stat_error"
            ].to_numpy(
                dtype=float
            )
        )

    split_median = float(
        np.median(
            finite_values(
                split_values
            )
        )
    )

    propagated_median = float(
        np.median(
            finite_values(
                propagated_values
            )
        )
    )

    split_ratio = (
        split_median
        / propagated_median
        if propagated_median
        > 0.0
        else math.nan
    )

    generator_differences = []

    for charge in (
        "plus",
        "minus",
    ):
        pythia = final_results[
            CHARGE_GROUPS[
                charge
            ][
                "pythia"
            ]
        ]

        herwig = final_results[
            CHARGE_GROUPS[
                charge
            ][
                "herwig"
            ]
        ]

        generator_differences.extend(
            np.abs(
                pythia[
                    "nominal_correction"
                ].to_numpy(
                    dtype=float
                )
                - herwig[
                    "nominal_correction"
                ].to_numpy(
                    dtype=float
                )
            )
        )

    cards = [
        (
            "Correction-factor range",
            (
                f"{np.min(all_corrections):.3f}–"
                f"{np.max(all_corrections):.3f}"
            ),
            (
                "Range of the final central correction factors across "
                "all four generator/charge channels and all |ηℓ| bins."
            ),
        ),
        (
            "Total relative uncertainty",
            (
                f"{np.min(all_total_percent):.2f}%–"
                f"{np.max(all_total_percent):.2f}%"
            ),
            (
                "Range of σ<sub>total</sub>/|C| across the complete "
                "final result."
            ),
        ),
        (
            "Largest median Pythia systematic",
            (
                SOURCE_LABELS[
                    dominant_source
                ]
            ),
            (
                "The source with the largest median relative contribution "
                "across the Pythia W⁺ and W⁻ bins."
            ),
        ),
        (
            "Odd/even stability ratio",
            (
                f"{split_ratio:.2f}"
                if math.isfinite(
                    split_ratio
                )
                else "—"
            ),
            (
                "Median ½|C<sub>odd</sub>−C<sub>even</sub>| divided by "
                "the median propagated σ<sub>stat</sub>; used as a "
                "stability diagnostic."
            ),
        ),
        (
            "Largest generator separation",
            f"{np.max(generator_differences):.3f}",
            (
                "Largest |C<sub>Pythia</sub>−C<sub>Herwig</sub>| "
                "across the W⁺ and W⁻ bins."
            ),
        ),
        (
            "Analysis scope",
            "4 channels × 11 bins",
            (
                "Pythia and Herwig, W⁺ and W⁻, using the common "
                "lepton-|η| binning throughout."
            ),
        ),
    ]

    html = ""

    for title, value, explanation in (
        cards
    ):
        html += f"""
        <div class="summary-card">
            <div class="summary-card-title">{title}</div>
            <div class="summary-card-value">{value}</div>
            <div class="summary-card-text">{explanation}</div>
        </div>
        """

    return html


# ============================================================
# Method/conclusion content
# ============================================================

def build_conclusion_cards(
    final_results,
    relative_summaries,
):
    pythia_total = np.concatenate([
        final_results[
            pair
        ][
            "total_unc_percent"
        ].to_numpy(
            dtype=float
        )
        for pair in (
            "Pythia_plus",
            "Pythia_minus",
        )
    ])

    herwig_total = np.concatenate([
        final_results[
            pair
        ][
            "total_unc_percent"
        ].to_numpy(
            dtype=float
        )
        for pair in (
            "Herwig_plus",
            "Herwig_minus",
        )
    ])

    pythia_shower = np.concatenate([
        relative_summaries[
            pair
        ][
            "shower_unc_percent"
        ].to_numpy(
            dtype=float
        )
        for pair in (
            "Pythia_plus",
            "Pythia_minus",
        )
    ])

    cards = [
        (
            "Final uncertainty treatment",
            (
                "The final central values and propagated statistical "
                "uncertainties are taken from the validated weighted-sample "
                "weight-0 corrections. Systematic responses are evaluated "
                "relative to weight 0 from the same samples."
            ),
        ),
        (
            "Pythia systematics",
            (
                "Pythia contains common scale, PDF, shower and model "
                "variations. The shower term is substantial in several bins; "
                f"its relative contribution reaches {np.max(pythia_shower):.2f}% "
                "in the current output."
            ),
        ),
        (
            "Herwig systematics",
            (
                "The available Herwig weighted files contain common scale "
                "and PDF variations, but no common shower or model variations. "
                "Those two displayed zero components indicate that no common "
                "variation is available in this workflow; they should not be "
                "read as a demonstration of zero physical uncertainty."
            ),
        ),
        (
            "Overall precision",
            (
                "Across the Pythia channels the final relative uncertainty "
                f"spans {np.min(pythia_total):.2f}%–{np.max(pythia_total):.2f}%; "
                "across Herwig it spans "
                f"{np.min(herwig_total):.2f}%–{np.max(herwig_total):.2f}%."
            ),
        ),
        (
            "Statistical validation",
            (
                "The odd/even split is retained as a stability test. "
                "The propagated ROOT/Sumw2 uncertainty remains the statistical "
                "uncertainty used in the final result."
            ),
        ),
        (
            "Exploratory high-statistics study",
            (
                "An exploratory Pythia high-statistics sample is not part of "
                "the final result after validation of its fragment structure. "
                "The presentation therefore uses the weighted-sample result "
                "consistently throughout."
            ),
        ),
    ]

    html = ""

    for title, text in (
        cards
    ):
        html += f"""
        <div class="finding-card">
            <h3>{title}</h3>
            <p>{text}</p>
        </div>
        """

    return html


# ============================================================
# Dashboard HTML
# ============================================================

def build_dashboard_html(
    figures,
    summary_cards_html,
    conclusion_cards_html,
):
    plotly_js = get_plotlyjs()

    screens = [
        {
            "id": "summary",
            "label": "Summary",
            "figure": figures[
                "summary"
            ],
            "before": (
                '<div class="summary-grid">'
                + summary_cards_html
                + "</div>"
            ),
            "explanation": (
                "<strong>Presentation summary.</strong> "
                "This page gives the headline numerical range of the final "
                "corrections, the total relative uncertainty, the dominant "
                "Pythia systematic source, the odd/even stability diagnostic "
                "and the largest Pythia–Herwig separation. The plot below "
                "places Pythia and Herwig side by side for W⁺ and W⁻ using "
                "the same vertical scale and total uncertainties."
            ),
        },
        {
            "id": "final",
            "label": "Final result",
            "figure": figures[
                "final"
            ],
            "before": "",
            "explanation": (
                "<strong>What this shows.</strong> "
                "The upper panel gives the final parton-to-particle "
                "correction factor C. The first error-bar trace shows "
                "σ<sub>stat</sub>, while the open markers show "
                "σ<sub>total</sub>. The lower panel decomposes the uncertainty "
                "as a percentage of |C|. Use the dropdown to move between "
                "Pythia W⁺, Pythia W⁻, Herwig W⁺ and Herwig W⁻. "
                "The total uncertainty is the quadrature of the propagated "
                "statistical term and all available systematic components."
            ),
        },
        {
            "id": "overview",
            "label": "Correction overview",
            "figure": figures[
                "overview"
            ],
            "before": "",
            "explanation": (
                "<strong>What this shows.</strong> "
                "All four final correction-factor channels are displayed "
                "simultaneously. Every panel uses the same |η<sub>ℓ</sub>| "
                "range and the same correction-factor scale, so generator "
                "and charge differences can be compared directly. Error bars "
                "show σ<sub>total</sub>."
            ),
        },
        {
            "id": "statistics",
            "label": "Statistical validation",
            "figure": figures[
                "statistics"
            ],
            "before": "",
            "explanation": (
                "<strong>What this shows.</strong> "
                "The odd/even split uses the original tree-entry parity while "
                "keeping the same physics selection. In the correction-split "
                "view, C<sub>all</sub>, C<sub>odd</sub> and C<sub>even</sub> "
                "are compared on synchronized axes. Switch the dropdown to "
                "compare ½|C<sub>odd</sub>−C<sub>even</sub>| with the "
                "propagated σ<sub>stat</sub>. The half-difference is a "
                "stability diagnostic rather than a replacement uncertainty."
            ),
        },
        {
            "id": "relative",
            "label": "Relative systematics",
            "figure": figures[
                "relative"
            ],
            "before": "",
            "explanation": (
                "<strong>What this shows.</strong> "
                "For every systematic weight k, the response is "
                "(C<sub>k</sub>−C₀)/C₀ using weight 0 from the same event "
                "sample. Scale, shower and model uncertainties use envelopes; "
                "PDF uses an RMS. The four panels use identical axes. "
                "For Herwig, shower/model are unavailable because there are "
                "no common variations for those sources in the weighted files; "
                "their stored zero values are not interpreted as a physical "
                "zero uncertainty."
            ),
        },
        {
            "id": "weights",
            "label": "Individual weights",
            "figure": figures[
                "weights"
            ],
            "before": "",
            "explanation": (
                "<strong>What this shows.</strong> "
                "Each faint curve is one individual systematic-weight "
                "response (C<sub>k</sub>−C₀)/C₀. The dashed lines show the "
                "final envelope or ±RMS magnitude for the selected source. "
                "Use the dropdown to select Scale, PDF, Shower or Model. "
                "For each selection, every channel shares one symmetric "
                "vertical scale so the response sizes can be compared directly."
            ),
        },
        {
            "id": "generator",
            "label": "Pythia vs Herwig",
            "figure": figures[
                "generator"
            ],
            "before": "",
            "explanation": (
                "<strong>What this shows.</strong> "
                "The upper row compares the final Pythia and Herwig "
                "correction factors for W⁺ and W⁻ with total uncertainty bars. "
                "The lower row shows C<sub>Pythia</sub>−C<sub>Herwig</sub>; "
                "its error bars are the quadrature combination of the two "
                "displayed total uncertainties. The W⁺ and W⁻ panels are "
                "synchronized within each row."
            ),
        },
    ]

    nav_html = ""
    screen_html = ""

    for index, screen in enumerate(
        screens
    ):
        active = (
            " active"
            if index
            == 0
            else ""
        )

        nav_html += (
            f'<button type="button" '
            f'class="nav-button{active}" '
            f'data-target="{screen["id"]}">'
            f'{screen["label"]}'
            "</button>"
        )

        screen_html += (
            f'<section id="screen-{screen["id"]}" '
            f'class="screen{active}">'
            f'{screen["before"]}'
            '<div class="plot-wrap">'
            f'{figure_div(screen["figure"])}'
            "</div>"
            '<div class="explanation-box">'
            f'{screen["explanation"]}'
            "</div>"
            "</section>"
        )

    nav_html += (
        '<button type="button" '
        'class="nav-button" '
        'data-target="method">'
        "Method & conclusions"
        "</button>"
    )

    method_html = f"""
    <section id="screen-method" class="screen">

        <div class="section-heading">
            <h2>Analysis method</h2>
            <p>
                Final selection, correction definition and uncertainty
                construction used throughout the presentation.
            </p>
        </div>

        <div class="method-grid">

            <div class="method-card">
                <h3>Lepton and missing-momentum selection</h3>
                <div class="equation-line">
                    E<sub>T</sub><sup>miss</sup> &gt; 25 GeV
                </div>
                <div class="equation-line">
                    p<sub>T</sub><sup>ℓ</sup> &gt; 20 GeV
                </div>
                <div class="equation-line">
                    |η<sub>ℓ</sub>| &lt; 2.5
                </div>
                <div class="equation-line">
                    m<sub>T</sub><sup>W</sup> &gt; 40 GeV
                </div>
            </div>

            <div class="method-card">
                <h3>Jet and charm selection</h3>
                <div class="equation-line">
                    p<sub>T</sub><sup>jet</sup> &gt; 25 GeV
                </div>
                <div class="equation-line">
                    |η<sub>jet</sub>| &lt; 2.5
                </div>
                <div class="equation-line">
                    exactly one fiducial charm-identified jet
                </div>
                <div class="equation-line">
                    jet_charge ≠ 0
                </div>
                <p>
                    Additional fiducial non-charm jets are allowed.
                    Opposite-sign W/charm events carry +1 and same-sign
                    events carry −1, giving the OS−SS contribution.
                </p>
            </div>

            <div class="method-card">
                <h3>Correction factor</h3>
                <div class="equation-large">
                    C<sub>i</sub> =
                    N<sub>parton,i</sub> /
                    N<sub>particle,i</sub>
                </div>
                <p>
                    The observable is |η<sub>ℓ</sub>| and the same 11-bin
                    definition is used throughout all four channels.
                </p>
            </div>

            <div class="method-card">
                <h3>Relative systematic response</h3>
                <div class="equation-large">
                    Δ<sub>i</sub><sup>(k)</sup> =
                    (C<sub>i</sub><sup>(k)</sup> −
                    C<sub>i</sub><sup>(0)</sup>) /
                    C<sub>i</sub><sup>(0)</sup>
                </div>
                <p>
                    Nominal and varied weights are evaluated on the same
                    Monte Carlo sample. Scale, shower and model use the
                    maximum |Δ|; PDF uses the RMS of Δ.
                </p>
            </div>

            <div class="method-card">
                <h3>Absolute systematic uncertainty</h3>
                <div class="equation-large">
                    σ<sub>X,i</sub> =
                    |C<sub>i</sub>| · Δ<sub>X,i</sub>
                </div>
                <p>
                    X denotes scale, PDF, shower or model. The systematic
                    total is their quadrature combination.
                </p>
            </div>

            <div class="method-card">
                <h3>Total uncertainty</h3>
                <div class="equation-large">
                    σ<sub>total</sub> =
                    √(σ<sub>stat</sub><sup>2</sup> +
                    σ<sub>syst</sub><sup>2</sup>)
                </div>
                <p>
                    σ<sub>stat</sub> is the propagated ROOT/Sumw2
                    parton-to-particle ratio uncertainty.
                </p>
            </div>

        </div>

        <div class="section-heading conclusions-heading">
            <h2>Conclusions and interpretation</h2>
        </div>

        <div class="findings-grid">
            {conclusion_cards_html}
        </div>

        <div class="explanation-box">
            <strong>How the pieces fit together.</strong>
            The selected W+c samples define the particle- and parton-level
            |η<sub>ℓ</sub>| distributions. Their ratio gives C. The odd/even
            split checks statistical stability. Systematic generator weights
            are evaluated relative to weight 0 on the same samples, reducing
            common statistical fluctuations in the systematic response.
            Those relative responses are converted to absolute uncertainties
            using |C| and combined with σ<sub>stat</sub> in quadrature. The
            Pythia/Herwig comparison is shown separately as a generator-level
            comparison rather than being folded into the displayed total.
        </div>

    </section>
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Uncertainty Decomposition of the Strange Proton</title>
<script>{plotly_js}</script>
<style>
    :root {{
        --page: #f4f6f9;
        --panel: #ffffff;
        --text: #172033;
        --muted: #5d6877;
        --border: #d9e0e8;
        --nav-active: #e4eaf1;
        --accent: #7c8da1;
        --shadow: 0 8px 24px rgba(31, 45, 61, 0.06);
    }}

    * {{
        box-sizing: border-box;
    }}

    html {{
        scroll-behavior: smooth;
    }}

    body {{
        margin: 0;
        background: var(--page);
        color: var(--text);
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            Arial,
            Helvetica,
            sans-serif;
    }}

    .page {{
        max-width: 1500px;
        margin: 0 auto;
        padding: 24px;
    }}

    .hero {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 24px 26px;
        box-shadow: var(--shadow);
        margin-bottom: 16px;
    }}

    .eyebrow {{
        font-size: 12px;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 8px;
        font-weight: 700;
    }}

    .hero h1 {{
        margin: 0;
        font-size: 30px;
        line-height: 1.15;
        font-weight: 700;
        letter-spacing: -0.02em;
    }}

    .hero p {{
        margin: 9px 0 0 0;
        color: var(--muted);
        line-height: 1.55;
        max-width: 980px;
        font-size: 15px;
    }}

    .navigation {{
        position: sticky;
        top: 0;
        z-index: 50;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 10px 0 14px 0;
        background: linear-gradient(
            to bottom,
            rgba(244,246,249,0.98),
            rgba(244,246,249,0.92),
            rgba(244,246,249,0)
        );
        backdrop-filter: blur(8px);
    }}

    .nav-button {{
        min-height: 42px;
        padding: 9px 14px;
        border-radius: 9px;
        border: 1px solid var(--border);
        background: var(--panel);
        color: var(--text);
        cursor: pointer;
        font-size: 14px;
        box-shadow: 0 2px 8px rgba(31,45,61,0.03);
        transition:
            background 0.15s ease,
            transform 0.15s ease,
            border-color 0.15s ease;
    }}

    .nav-button:hover {{
        transform: translateY(-1px);
        border-color: #b9c4d0;
    }}

    .nav-button.active {{
        background: var(--nav-active);
        border-color: #aab7c4;
        font-weight: 700;
    }}

    .nav-button:focus-visible {{
        outline: 3px solid #b4c5d6;
        outline-offset: 2px;
    }}

    .screen {{
        display: none;
        animation: fadeIn 0.18s ease;
    }}

    .screen.active {{
        display: block;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(3px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .plot-wrap {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 8px;
        overflow: hidden;
        box-shadow: var(--shadow);
    }}

    .explanation-box {{
        margin-top: 14px;
        background: var(--panel);
        border: 1px solid var(--border);
        border-left: 5px solid var(--accent);
        border-radius: 11px;
        padding: 16px 18px;
        line-height: 1.62;
        color: #354354;
        box-shadow: var(--shadow);
        font-size: 14px;
    }}

    .summary-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 14px;
    }}

    .summary-card {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 17px;
        box-shadow: var(--shadow);
        min-width: 0;
    }}

    .summary-card-title {{
        color: var(--muted);
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }}

    .summary-card-value {{
        margin-top: 8px;
        font-size: 25px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text);
    }}

    .summary-card-text {{
        margin-top: 8px;
        color: var(--muted);
        line-height: 1.48;
        font-size: 13px;
    }}

    .section-heading {{
        margin: 2px 0 14px 0;
    }}

    .section-heading h2 {{
        margin: 0;
        font-size: 23px;
    }}

    .section-heading p {{
        margin: 6px 0 0 0;
        color: var(--muted);
        line-height: 1.5;
    }}

    .conclusions-heading {{
        margin-top: 18px;
    }}

    .method-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 13px;
    }}

    .method-card {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 18px;
        box-shadow: var(--shadow);
        min-width: 0;
    }}

    .method-card h3 {{
        margin: 0 0 12px 0;
        font-size: 17px;
    }}

    .method-card p {{
        margin: 11px 0 0 0;
        color: var(--muted);
        line-height: 1.54;
        font-size: 14px;
    }}

    .equation-line {{
        margin: 7px 0;
        font-size: 17px;
        line-height: 1.4;
    }}

    .equation-large {{
        padding: 8px 0;
        font-size: 20px;
        line-height: 1.55;
        word-break: break-word;
    }}

    .findings-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 13px;
    }}

    .finding-card {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 17px;
        box-shadow: var(--shadow);
    }}

    .finding-card h3 {{
        margin: 0 0 8px 0;
        font-size: 16px;
    }}

    .finding-card p {{
        margin: 0;
        color: var(--muted);
        line-height: 1.52;
        font-size: 14px;
    }}

    .footer {{
        margin-top: 18px;
        color: var(--muted);
        font-size: 12px;
        line-height: 1.5;
        text-align: center;
    }}

    code {{
        font-family:
            "SFMono-Regular",
            Consolas,
            "Liberation Mono",
            monospace;
        font-size: 0.94em;
    }}

    @media (max-width: 1050px) {{
        .summary-grid,
        .method-grid,
        .findings-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
    }}

    @media (max-width: 720px) {{
        .page {{
            padding: 12px;
        }}

        .hero {{
            padding: 18px;
        }}

        .hero h1 {{
            font-size: 24px;
        }}

        .summary-grid,
        .method-grid,
        .findings-grid {{
            grid-template-columns: 1fr;
        }}

        .navigation {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
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
        <div class="eyebrow">
            W+c lepton-pseudorapidity analysis
        </div>
        <h1>Uncertainty Decomposition of the Strange Proton</h1>
        <p>
            Interactive presentation of the |η<sub>ℓ</sub>|
            parton-to-particle correction factors, statistical validation,
            systematic-weight responses and Pythia–Herwig comparison.
        </p>
    </header>

    <nav class="navigation" aria-label="Dashboard sections">
        {nav_html}
    </nav>

    {screen_html}

    {method_html}

    <div class="footer">
        Strange Proton Uncertainty Decomposition · interactive analysis dashboard
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
                    Plotly.Plots.resize(
                        plot
                    );
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
                    Plotly.Plots.resize(
                        plot
                    );
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
            "Build the final interactive Strange Proton uncertainty dashboard."
        )
    )

    parser.add_argument(
        "--open",
        action="store_true",
        help=(
            "Open the generated dashboard in the default browser."
        ),
    )

    args = parser.parse_args()

    print()
    print(
        "=" * 76
    )

    print(
        " Strange Proton Uncertainty Decomposition"
    )

    print(
        " Final presentation dashboard"
    )

    print(
        "=" * 76
    )

    print()
    print(
        f"Working directory:\n  {WORK_DIR}"
    )

    final_results = (
        load_final_results()
    )

    (
        relative_summaries,
        per_weight,
    ) = load_relative_results()

    correction_results = (
        load_correction_results()
    )

    even_odd = (
        find_even_odd_csvs()
    )

    print()
    print(
        "Loaded:"
    )

    print(
        f"  final result channels:        {len(final_results)}"
    )

    print(
        f"  relative-systematic channels: {len(relative_summaries)}"
    )

    print(
        f"  per-weight channels:          {len(per_weight)}"
    )

    print(
        f"  odd/even channels:            {len(even_odd)}"
    )

    print()
    print(
        "Validating presentation inputs..."
    )

    validate_pipeline(
        final_results,
        relative_summaries,
        per_weight,
        correction_results,
        even_odd,
    )

    print(
        "  validation passed"
    )

    figures = {
        "summary": build_summary_figure(
            final_results
        ),
        "final": build_final_result_figure(
            final_results
        ),
        "overview": build_correction_overview_figure(
            final_results
        ),
        "statistics": build_statistical_validation_figure(
            even_odd
        ),
        "relative": build_relative_systematics_figure(
            relative_summaries
        ),
        "weights": build_weight_response_figure(
            relative_summaries,
            per_weight,
        ),
        "generator": build_generator_comparison_figure(
            final_results
        ),
    }

    summary_cards_html = (
        build_summary_cards(
            final_results,
            relative_summaries,
            even_odd,
        )
    )

    conclusion_cards_html = (
        build_conclusion_cards(
            final_results,
            relative_summaries,
        )
    )

    html = build_dashboard_html(
        figures,
        summary_cards_html,
        conclusion_cards_html,
    )

    OUTPUT_HTML.write_text(
        html,
        encoding="utf-8",
    )

    print()
    print(
        "=" * 76
    )

    print(
        " Dashboard created successfully"
    )

    print(
        "=" * 76
    )

    print()
    print(
        f"Dashboard:\n  {OUTPUT_HTML}"
    )

    print(
        f"Validation summary:\n  {VALIDATION_TEXT}"
    )

    print()
    print(
        "Presentation sections:"
    )

    print(
        "  1. Summary"
    )

    print(
        "  2. Final result"
    )

    print(
        "  3. Correction overview"
    )

    print(
        "  4. Statistical validation"
    )

    print(
        "  5. Relative systematics"
    )

    print(
        "  6. Individual weights"
    )

    print(
        "  7. Pythia vs Herwig"
    )

    print(
        "  8. Method & conclusions"
    )

    if args.open:
        print()
        print(
            "Opening dashboard..."
        )

        open_in_browser(
            OUTPUT_HTML
        )


if __name__ == "__main__":
    main()
