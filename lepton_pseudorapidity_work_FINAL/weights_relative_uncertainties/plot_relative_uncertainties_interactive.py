#!/usr/bin/env python3

"""
plot_relative_uncertainties_interactive.py

REPAIRED FINAL relative-systematics visualisation.

Purpose
-------
Read the repaired outputs from:

    build_relative_uncertainties.py

and build a self-contained interactive dashboard for the relative
systematic uncertainty study.

The dashboard contains:

1. Relative uncertainty decomposition
   Four synchronized panels:
       Pythia W+
       Pythia W-
       Herwig W+
       Herwig W-

   Each panel shows:
       scale
       PDF
       shower
       model
       total systematic

2. Individual weight responses
   Four synchronized panels with a dropdown selecting:
       scale
       PDF
       shower
       model

   Each individual variation is shown as

       100 * (C_weight - C_weight0) / C_weight0

   together with the final +/- envelope or +/- RMS boundary.

3. Method and interpretation
   An explanation embedded directly in the HTML dashboard.

Validation performed before plotting
------------------------------------
The script checks that:

- all four repaired summary/per-weight CSVs exist;
- eta binning is identical across all four channels;
- every summary total is the quadrature of scale/PDF/shower/model;
- every per-weight signed shift reproduces
      (C_weight - C_weight0) / C_weight0;
- every per-weight ratio reproduces
      C_weight / C_weight0;
- every per-weight nominal agrees with the summary nominal;
- the expected number of systematic variations is present;
- the plotted summary agrees with the official
  outputs/final_conservative/csv results.

The high-stat/event-matching Pythia study is NOT used by this script.

Important interpretation
------------------------
For Pythia:
    scale  = envelope
    PDF    = RMS
    shower = envelope
    model  = envelope

For Herwig:
    the repaired weighted files provide scale and PDF variations, but
    no common shower/model weights.  These components are therefore
    unavailable in this workflow and are displayed as zero in the
    stored summary.  Zero here means "no common variation supplied",
    NOT "the physical uncertainty is known to be exactly zero".

Expected location
-----------------
uncertainty_decomposition/
└── lepton_pseudorapidity_work_FINAL/
    └── weights_relative_uncertainties/
        ├── build_relative_uncertainties.py
        ├── apply_relative_uncertainties_to_new_nominal.py
        └── plot_relative_uncertainties_interactive.py

Outputs
-------
outputs/interactive/
    relative_uncertainty_decomposition.html
    relative_weight_responses.html
    relative_uncertainties_dashboard.html
    plot_validation_summary.txt

Run
---
python3 lepton_pseudorapidity_work_FINAL/weights_relative_uncertainties/plot_relative_uncertainties_interactive.py

Optional:
    --open
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

CSV_DIR = (
    SCRIPT_DIR
    / "outputs"
    / "csv"
)

FINAL_CONSERVATIVE_CSV_DIR = (
    SCRIPT_DIR
    / "outputs"
    / "final_conservative"
    / "csv"
)

INTERACTIVE_DIR = (
    SCRIPT_DIR
    / "outputs"
    / "interactive"
)

INTERACTIVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_HTML = (
    INTERACTIVE_DIR
    / "relative_uncertainty_decomposition.html"
)

WEIGHTS_HTML = (
    INTERACTIVE_DIR
    / "relative_weight_responses.html"
)

DASHBOARD_HTML = (
    INTERACTIVE_DIR
    / "relative_uncertainties_dashboard.html"
)

VALIDATION_SUMMARY = (
    INTERACTIVE_DIR
    / "plot_validation_summary.txt"
)


# ============================================================
# Analysis channels
# ============================================================

PAIR_DEFINITIONS = {
    "Pythia_plus": {
        "title": "Pythia · W+",
        "row": 1,
        "col": 1,
    },
    "Pythia_minus": {
        "title": "Pythia · W-",
        "row": 1,
        "col": 2,
    },
    "Herwig_plus": {
        "title": "Herwig · W+",
        "row": 2,
        "col": 1,
    },
    "Herwig_minus": {
        "title": "Herwig · W-",
        "row": 2,
        "col": 2,
    },
}


# ============================================================
# Source definitions
# ============================================================

SOURCES = {
    "scale": {
        "label": "Scale",
        "summary_column": "scale_unc_percent",
        "method": "Envelope",
    },
    "pdf": {
        "label": "PDF",
        "summary_column": "pdf_unc_percent",
        "method": "RMS",
    },
    "shower": {
        "label": "Shower",
        "summary_column": "shower_unc_percent",
        "method": "Envelope",
    },
    "model": {
        "label": "Model",
        "summary_column": "model_unc_percent",
        "method": "Envelope",
    },
}

TOTAL_SOURCE = {
    "label": "Total systematic",
    "summary_column": "total_systematic_unc_percent",
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
# Required CSV schemas
# ============================================================

SUMMARY_REQUIRED = {
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

PER_WEIGHT_REQUIRED = {
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

FINAL_REQUIRED = {
    "channel",
    "bin",
    "bin_low_edge",
    "bin_up_edge",
    "nominal_correction",
    "relative_scale_unc",
    "relative_pdf_unc",
    "relative_shower_unc",
    "relative_model_unc",
    "relative_total_systematic_unc",
    "scale_unc_percent",
    "pdf_unc_percent",
    "shower_unc_percent",
    "model_unc_percent",
    "systematic_unc_percent",
    "high_stat_nominal_used",
}


# ============================================================
# Validation tolerances
# ============================================================

ATOL = 5.0e-9
RTOL = 5.0e-7
BIN_ATOL = 1.0e-12


# ============================================================
# Loading helpers
# ============================================================

def add_eta_columns(
    dataframe,
):
    dataframe = dataframe.copy()

    dataframe[
        "eta_centre"
    ] = (
        0.5
        * (
            dataframe[
                "bin_low_edge"
            ]
            + dataframe[
                "bin_up_edge"
            ]
        )
    )

    dataframe[
        "eta_half_width"
    ] = (
        0.5
        * (
            dataframe[
                "bin_up_edge"
            ]
            - dataframe[
                "bin_low_edge"
            ]
        )
    )

    return dataframe


def load_csv_checked(
    path,
    required_columns,
):
    if not path.is_file():
        raise FileNotFoundError(
            "Required CSV does not exist:\n"
            f"  {path}"
        )

    dataframe = pd.read_csv(
        path
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

    if dataframe.empty:
        raise RuntimeError(
            f"CSV contains no rows:\n  {path}"
        )

    return add_eta_columns(
        dataframe
    )


def load_all_data():
    all_data = {}

    for pair_label in (
        PAIR_DEFINITIONS
    ):
        summary_path = (
            CSV_DIR
            / f"{pair_label}_relative_uncertainties.csv"
        )

        per_weight_path = (
            CSV_DIR
            / f"{pair_label}_per_weight_relative_shifts.csv"
        )

        final_path = (
            FINAL_CONSERVATIVE_CSV_DIR
            / f"{pair_label}_final_conservative.csv"
        )

        all_data[
            pair_label
        ] = {
            "summary": load_csv_checked(
                summary_path,
                SUMMARY_REQUIRED,
            ),
            "weights": load_csv_checked(
                per_weight_path,
                PER_WEIGHT_REQUIRED,
            ),
            "final": load_csv_checked(
                final_path,
                FINAL_REQUIRED,
            ),
            "paths": {
                "summary": summary_path,
                "weights": per_weight_path,
                "final": final_path,
            },
        }

    return all_data


# ============================================================
# Generic numerical helpers
# ============================================================

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


def max_abs_delta(
    left,
    right,
):
    left = np.asarray(
        left,
        dtype=float,
    )

    right = np.asarray(
        right,
        dtype=float,
    )

    return float(
        np.max(
            np.abs(
                left
                - right
            )
        )
    )


def common_positive_range(
    values,
    padding_fraction=0.08,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(
        values
    ) == 0:
        return [
            0.0,
            1.0,
        ]

    maximum = max(
        float(
            np.max(
                values
            )
        ),
        0.0,
    )

    if maximum == 0.0:
        maximum = 1.0

    return [
        0.0,
        maximum
        * (
            1.0
            + padding_fraction
        ),
    ]


def symmetric_range(
    values,
    padding_fraction=0.08,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(
        values
    ) == 0:
        return [
            -1.0,
            1.0,
        ]

    maximum = float(
        np.max(
            np.abs(
                values
            )
        )
    )

    if maximum == 0.0:
        maximum = 1.0

    maximum *= (
        1.0
        + padding_fraction
    )

    return [
        -maximum,
        maximum,
    ]


# ============================================================
# Validation
# ============================================================

def validate_binning_across_channels(
    all_data,
):
    reference_label = next(
        iter(
            PAIR_DEFINITIONS
        )
    )

    reference = (
        all_data[
            reference_label
        ][
            "summary"
        ]
        .sort_values(
            "bin"
        )
        .reset_index(
            drop=True
        )
    )

    for pair_label in (
        PAIR_DEFINITIONS
    ):
        summary = (
            all_data[
                pair_label
            ][
                "summary"
            ]
            .sort_values(
                "bin"
            )
            .reset_index(
                drop=True
            )
        )

        if len(
            summary
        ) != len(
            reference
        ):
            raise RuntimeError(
                f"{pair_label}: number of eta bins differs "
                f"from {reference_label}."
            )

        if not np.array_equal(
            summary[
                "bin"
            ].to_numpy(),
            reference[
                "bin"
            ].to_numpy(),
        ):
            raise RuntimeError(
                f"{pair_label}: bin indices differ from "
                f"{reference_label}."
            )

        for column in (
            "bin_low_edge",
            "bin_up_edge",
        ):
            if not allclose(
                summary[
                    column
                ],
                reference[
                    column
                ],
                atol=BIN_ATOL,
                rtol=0.0,
            ):
                raise RuntimeError(
                    f"{pair_label}: {column} differs from "
                    f"{reference_label}."
                )


def validate_pair(
    pair_label,
    data,
):
    summary = (
        data[
            "summary"
        ]
        .sort_values(
            "bin"
        )
        .reset_index(
            drop=True
        )
    )

    weights = (
        data[
            "weights"
        ]
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

    final = (
        data[
            "final"
        ]
        .sort_values(
            "bin"
        )
        .reset_index(
            drop=True
        )
    )

    if not (
        summary[
            "pair_label"
        ]
        == pair_label
    ).all():
        raise RuntimeError(
            f"{pair_label}: pair_label mismatch in summary CSV."
        )

    if not (
        weights[
            "pair_label"
        ]
        == pair_label
    ).all():
        raise RuntimeError(
            f"{pair_label}: pair_label mismatch in per-weight CSV."
        )

    if not (
        final[
            "channel"
        ]
        == pair_label
    ).all():
        raise RuntimeError(
            f"{pair_label}: channel mismatch in final CSV."
        )

    expected_bins = np.arange(
        1,
        len(
            summary
        )
        + 1,
    )

    if not np.array_equal(
        summary[
            "bin"
        ].to_numpy(
            dtype=int
        ),
        expected_bins,
    ):
        raise RuntimeError(
            f"{pair_label}: summary bins are not contiguous."
        )

    # --------------------------------------------------------
    # Summary total = quadrature of four sources.
    # --------------------------------------------------------

    component_matrix = np.column_stack([
        summary[
            "relative_scale_unc"
        ].to_numpy(
            dtype=float
        ),
        summary[
            "relative_pdf_unc"
        ].to_numpy(
            dtype=float
        ),
        summary[
            "relative_shower_unc"
        ].to_numpy(
            dtype=float
        ),
        summary[
            "relative_model_unc"
        ].to_numpy(
            dtype=float
        ),
    ])

    relative_total_recomputed = np.sqrt(
        np.sum(
            component_matrix ** 2,
            axis=1,
        )
    )

    if not allclose(
        relative_total_recomputed,
        summary[
            "relative_total_systematic_unc"
        ],
    ):
        raise RuntimeError(
            f"{pair_label}: summary total systematic is not "
            "the quadrature of the four source components."
        )

    percent_total_recomputed = (
        100.0
        * relative_total_recomputed
    )

    if not allclose(
        percent_total_recomputed,
        summary[
            "total_systematic_unc_percent"
        ],
    ):
        raise RuntimeError(
            f"{pair_label}: percent total systematic is inconsistent."
        )

    # --------------------------------------------------------
    # Variation counts.
    # --------------------------------------------------------

    counts = {}

    for source_key in (
        SOURCES
    ):
        subset = weights[
            weights[
                "category"
            ]
            == source_key
        ]

        counts[
            source_key
        ] = int(
            subset[
                "weight_index"
            ].nunique()
        )

        expected = (
            EXPECTED_VARIATION_COUNTS[
                pair_label
            ][
                source_key
            ]
        )

        if (
            counts[
                source_key
            ]
            != expected
        ):
            raise RuntimeError(
                f"{pair_label}: expected {expected} {source_key} "
                f"variations, found {counts[source_key]}."
            )

    # --------------------------------------------------------
    # Per-weight identities and completeness.
    # --------------------------------------------------------

    summary_by_bin = (
        summary
        .set_index(
            "bin"
        )
    )

    max_shift_delta = 0.0
    max_ratio_delta = 0.0
    max_nominal_delta = 0.0
    max_percent_delta = 0.0

    if not weights.empty:
        expected_nbins = len(
            summary
        )

        grouped_counts = (
            weights
            .groupby(
                [
                    "category",
                    "weight_index",
                ]
            )[
                "bin"
            ]
            .nunique()
        )

        bad_groups = grouped_counts[
            grouped_counts
            != expected_nbins
        ]

        if not bad_groups.empty:
            raise RuntimeError(
                f"{pair_label}: one or more systematic weights do "
                "not contain all eta bins."
            )

        c0 = weights[
            "C_nominal_weight0"
        ].to_numpy(
            dtype=float
        )

        ck = weights[
            "C_weight"
        ].to_numpy(
            dtype=float
        )

        if np.any(
            c0
            == 0.0
        ):
            raise RuntimeError(
                f"{pair_label}: zero C_nominal_weight0 in per-weight CSV."
            )

        shift_recomputed = (
            ck
            - c0
        ) / c0

        ratio_recomputed = (
            ck
            / c0
        )

        shift_saved = weights[
            "signed_relative_shift"
        ].to_numpy(
            dtype=float
        )

        ratio_saved = weights[
            "C_weight_over_C_nominal"
        ].to_numpy(
            dtype=float
        )

        abs_saved = weights[
            "absolute_relative_shift"
        ].to_numpy(
            dtype=float
        )

        percent_saved = weights[
            "signed_relative_shift_percent"
        ].to_numpy(
            dtype=float
        )

        max_shift_delta = max_abs_delta(
            shift_recomputed,
            shift_saved,
        )

        max_ratio_delta = max_abs_delta(
            ratio_recomputed,
            ratio_saved,
        )

        max_percent_delta = max_abs_delta(
            100.0
            * shift_recomputed,
            percent_saved,
        )

        if not allclose(
            shift_recomputed,
            shift_saved,
        ):
            raise RuntimeError(
                f"{pair_label}: per-weight signed relative shifts "
                "do not reproduce (C_weight-C0)/C0."
            )

        if not allclose(
            ratio_recomputed,
            ratio_saved,
        ):
            raise RuntimeError(
                f"{pair_label}: C_weight_over_C_nominal is inconsistent."
            )

        if not allclose(
            np.abs(
                shift_recomputed
            ),
            abs_saved,
        ):
            raise RuntimeError(
                f"{pair_label}: absolute_relative_shift is inconsistent."
            )

        if not allclose(
            100.0
            * shift_recomputed,
            percent_saved,
        ):
            raise RuntimeError(
                f"{pair_label}: signed_relative_shift_percent is inconsistent."
            )

        expected_nominal = weights[
            "bin"
        ].map(
            summary_by_bin[
                "nominal_correction_weight0"
            ]
        ).to_numpy(
            dtype=float
        )

        max_nominal_delta = max_abs_delta(
            c0,
            expected_nominal,
        )

        if not allclose(
            c0,
            expected_nominal,
        ):
            raise RuntimeError(
                f"{pair_label}: per-weight nominal corrections do "
                "not match the summary nominal."
            )

    # --------------------------------------------------------
    # Agreement with official conservative final outputs.
    # --------------------------------------------------------

    if len(
        final
    ) != len(
        summary
    ):
        raise RuntimeError(
            f"{pair_label}: final conservative CSV has a different "
            "number of eta bins."
        )

    for column in (
        "bin",
        "bin_low_edge",
        "bin_up_edge",
    ):
        if column == "bin":
            if not np.array_equal(
                final[
                    column
                ].to_numpy(
                    dtype=int
                ),
                summary[
                    column
                ].to_numpy(
                    dtype=int
                ),
            ):
                raise RuntimeError(
                    f"{pair_label}: final/summary bin indices differ."
                )
        else:
            if not allclose(
                final[
                    column
                ],
                summary[
                    column
                ],
                atol=BIN_ATOL,
                rtol=0.0,
            ):
                raise RuntimeError(
                    f"{pair_label}: final/summary {column} differs."
                )

    final_checks = {
        "nominal": (
            final[
                "nominal_correction"
            ],
            summary[
                "nominal_correction_weight0"
            ],
        ),
        "relative_scale": (
            final[
                "relative_scale_unc"
            ],
            summary[
                "relative_scale_unc"
            ],
        ),
        "relative_pdf": (
            final[
                "relative_pdf_unc"
            ],
            summary[
                "relative_pdf_unc"
            ],
        ),
        "relative_shower": (
            final[
                "relative_shower_unc"
            ],
            summary[
                "relative_shower_unc"
            ],
        ),
        "relative_model": (
            final[
                "relative_model_unc"
            ],
            summary[
                "relative_model_unc"
            ],
        ),
        "relative_total": (
            final[
                "relative_total_systematic_unc"
            ],
            summary[
                "relative_total_systematic_unc"
            ],
        ),
        "scale_percent": (
            final[
                "scale_unc_percent"
            ],
            summary[
                "scale_unc_percent"
            ],
        ),
        "pdf_percent": (
            final[
                "pdf_unc_percent"
            ],
            summary[
                "pdf_unc_percent"
            ],
        ),
        "shower_percent": (
            final[
                "shower_unc_percent"
            ],
            summary[
                "shower_unc_percent"
            ],
        ),
        "model_percent": (
            final[
                "model_unc_percent"
            ],
            summary[
                "model_unc_percent"
            ],
        ),
        "systematic_percent": (
            final[
                "systematic_unc_percent"
            ],
            summary[
                "total_systematic_unc_percent"
            ],
        ),
    }

    final_deltas = {}

    for name, (
        final_values,
        summary_values,
    ) in final_checks.items():
        final_deltas[
            name
        ] = max_abs_delta(
            final_values,
            summary_values,
        )

        if not allclose(
            final_values,
            summary_values,
        ):
            raise RuntimeError(
                f"{pair_label}: plotted relative-systematic data "
                f"do not agree with final conservative output for {name}."
            )

    high_stat_flags = (
        final[
            "high_stat_nominal_used"
        ]
        .astype(
            str
        )
        .str.lower()
        .str.strip()
    )

    if not (
        high_stat_flags
        == "false"
    ).all():
        raise RuntimeError(
            f"{pair_label}: final conservative CSV does not explicitly "
            "exclude the high-stat nominal."
        )

    return {
        "counts": counts,
        "max_summary_total_delta": max_abs_delta(
            relative_total_recomputed,
            summary[
                "relative_total_systematic_unc"
            ],
        ),
        "max_shift_delta": (
            max_shift_delta
        ),
        "max_ratio_delta": (
            max_ratio_delta
        ),
        "max_nominal_delta": (
            max_nominal_delta
        ),
        "max_percent_delta": (
            max_percent_delta
        ),
        "final_deltas": (
            final_deltas
        ),
    }


def validate_all_data(
    all_data,
):
    validate_binning_across_channels(
        all_data
    )

    results = {}

    for pair_label in (
        PAIR_DEFINITIONS
    ):
        results[
            pair_label
        ] = validate_pair(
            pair_label,
            all_data[
                pair_label
            ],
        )

    return results


# ============================================================
# Figure 1: relative-systematic decomposition
# ============================================================

def build_summary_figure(
    all_data,
):
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            PAIR_DEFINITIONS[
                pair_label
            ][
                "title"
            ]
            for pair_label in (
                PAIR_DEFINITIONS
            )
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    y_values = []

    for pair_label in (
        PAIR_DEFINITIONS
    ):
        dataframe = (
            all_data[
                pair_label
            ][
                "summary"
            ]
            .sort_values(
                "bin"
            )
        )

        for source_info in (
            SOURCES.values()
        ):
            y_values.extend(
                dataframe[
                    source_info[
                        "summary_column"
                    ]
                ].to_numpy(
                    dtype=float
                )
            )

        y_values.extend(
            dataframe[
                TOTAL_SOURCE[
                    "summary_column"
                ]
            ].to_numpy(
                dtype=float
            )
        )

    common_y_range = (
        common_positive_range(
            y_values
        )
    )

    for pair_index, (
        pair_label,
        pair_info,
    ) in enumerate(
        PAIR_DEFINITIONS.items()
    ):
        dataframe = (
            all_data[
                pair_label
            ][
                "summary"
            ]
            .sort_values(
                "bin"
            )
        )

        row = pair_info[
            "row"
        ]

        col = pair_info[
            "col"
        ]

        customdata = np.column_stack([
            dataframe[
                "bin_low_edge"
            ].to_numpy(
                dtype=float
            ),
            dataframe[
                "bin_up_edge"
            ].to_numpy(
                dtype=float
            ),
        ])

        for source_key, source_info in (
            SOURCES.items()
        ):
            figure.add_trace(
                go.Scatter(
                    x=dataframe[
                        "eta_centre"
                    ],
                    y=dataframe[
                        source_info[
                            "summary_column"
                        ]
                    ],
                    mode="lines+markers",
                    name=source_info[
                        "label"
                    ],
                    legendgroup=source_key,
                    showlegend=(
                        pair_index
                        == 0
                    ),
                    customdata=customdata,
                    hovertemplate=(
                        "|eta_l| bin: "
                        "%{customdata[0]:.2f}-"
                        "%{customdata[1]:.2f}"
                        "<br>"
                        + source_info[
                            "label"
                        ]
                        + ": %{y:.4f}%"
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
                    TOTAL_SOURCE[
                        "summary_column"
                    ]
                ],
                mode="lines+markers",
                name=TOTAL_SOURCE[
                    "label"
                ],
                legendgroup="total_systematic",
                showlegend=(
                    pair_index
                    == 0
                ),
                marker=dict(
                    symbol="circle-open",
                    size=9,
                ),
                line=dict(
                    width=3,
                ),
                customdata=customdata,
                hovertemplate=(
                    "|eta_l| bin: "
                    "%{customdata[0]:.2f}-"
                    "%{customdata[1]:.2f}"
                    "<br>Total systematic: %{y:.4f}%"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    figure.update_layout(
        title=dict(
            text=(
                "Repaired relative systematic uncertainty decomposition"
                "<br><sup>"
                "same-sample weight variations relative to weight 0 · "
                "shared axes across all four channels"
                "</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        height=900,
        width=1250,
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
            t=145,
            b=75,
        ),
    )

    for pair_info in (
        PAIR_DEFINITIONS.values()
    ):
        row = pair_info[
            "row"
        ]

        col = pair_info[
            "col"
        ]

        figure.update_xaxes(
            title_text="Lepton |eta|",
            range=[
                0.0,
                2.5,
            ],
            showgrid=True,
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
            range=common_y_range,
            showgrid=True,
            matches="y",
            row=row,
            col=col,
        )

    return figure


# ============================================================
# Figure 2: individual weight responses
# ============================================================

def source_symmetric_ranges(
    all_data,
):
    ranges = {}

    for source_key, source_info in (
        SOURCES.items()
    ):
        values = []

        for pair_label in (
            PAIR_DEFINITIONS
        ):
            weights = (
                all_data[
                    pair_label
                ][
                    "weights"
                ]
            )

            subset = weights[
                weights[
                    "category"
                ]
                == source_key
            ]

            values.extend(
                subset[
                    "signed_relative_shift_percent"
                ].to_numpy(
                    dtype=float
                )
            )

            summary = (
                all_data[
                    pair_label
                ][
                    "summary"
                ]
            )

            boundary = summary[
                source_info[
                    "summary_column"
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
            source_key
        ] = symmetric_range(
            values
        )

    return ranges


def build_weight_response_figure(
    all_data,
):
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            PAIR_DEFINITIONS[
                pair_label
            ][
                "title"
            ]
            for pair_label in (
                PAIR_DEFINITIONS
            )
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    source_trace_indices = {
        source_key: []
        for source_key in (
            SOURCES
        )
    }

    source_ranges = (
        source_symmetric_ranges(
            all_data
        )
    )

    initial_source = "scale"

    for source_key, source_info in (
        SOURCES.items()
    ):
        for pair_label, pair_info in (
            PAIR_DEFINITIONS.items()
        ):
            dataframe = (
                all_data[
                    pair_label
                ][
                    "weights"
                ]
            )

            summary = (
                all_data[
                    pair_label
                ][
                    "summary"
                ]
                .sort_values(
                    "bin"
                )
            )

            subset = dataframe[
                dataframe[
                    "category"
                ]
                == source_key
            ].copy()

            row = pair_info[
                "row"
            ]

            col = pair_info[
                "col"
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
                    source_key
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
                            + source_info[
                                "label"
                            ]
                            + " variations<br>"
                            "available in this channel"
                        ],
                        textposition="middle center",
                        visible=(
                            source_key
                            == initial_source
                        ),
                        showlegend=False,
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
                    source_key
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
                            source_key
                            == initial_source
                        ),
                        showlegend=False,
                        line=dict(
                            width=1,
                        ),
                        marker=dict(
                            size=4,
                        ),
                        opacity=0.30,
                        customdata=np.column_stack([
                            weight_data[
                                "bin_low_edge"
                            ].to_numpy(
                                dtype=float
                            ),
                            weight_data[
                                "bin_up_edge"
                            ].to_numpy(
                                dtype=float
                            ),
                        ]),
                        hovertemplate=(
                            f"Weight {weight_index}"
                            f"<br>{weight_name}"
                            "<br>|eta_l| bin: "
                            "%{customdata[0]:.2f}-"
                            "%{customdata[1]:.2f}"
                            "<br>Relative shift: %{y:.4f}%"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

            # Only draw a boundary when this source actually exists
            # in the channel.  A zero caused by unavailable weights
            # should not look like a measured zero uncertainty.
            if weight_indices:
                source_uncertainty = (
                    summary[
                        source_info[
                            "summary_column"
                        ]
                    ]
                )

                customdata = (
                    np.column_stack([
                        summary[
                            "bin_low_edge"
                        ].to_numpy(
                            dtype=float
                        ),
                        summary[
                            "bin_up_edge"
                        ].to_numpy(
                            dtype=float
                        ),
                    ])
                )

                for sign in (
                    1.0,
                    -1.0,
                ):
                    trace_index = len(
                        figure.data
                    )

                    source_trace_indices[
                        source_key
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
                                * source_uncertainty
                            ),
                            mode="lines",
                            visible=(
                                source_key
                                == initial_source
                            ),
                            name=(
                                f"{source_info['method']} boundary"
                            ),
                            legendgroup=(
                                f"{source_key}_boundary"
                            ),
                            showlegend=(
                                pair_label
                                == "Pythia_plus"
                                and sign > 0
                            ),
                            line=dict(
                                width=3,
                                dash="dash",
                            ),
                            customdata=customdata,
                            hovertemplate=(
                                "|eta_l| bin: "
                                "%{customdata[0]:.2f}-"
                                "%{customdata[1]:.2f}"
                                "<br>"
                                + source_info[
                                    "method"
                                ]
                                + " boundary: %{y:.4f}%"
                                "<extra></extra>"
                            ),
                        ),
                        row=row,
                        col=col,
                    )

    total_traces = len(
        figure.data
    )

    buttons = []

    axis_names = [
        "yaxis",
        "yaxis2",
        "yaxis3",
        "yaxis4",
    ]

    for source_key, source_info in (
        SOURCES.items()
    ):
        visible = [
            False
        ] * total_traces

        for trace_index in (
            source_trace_indices[
                source_key
            ]
        ):
            visible[
                trace_index
            ] = True

        layout_updates = {
            "title.text": (
                "Individual repaired relative weight responses"
                "<br><sup>"
                + source_info[
                    "label"
                ]
                + " · "
                + source_info[
                    "method"
                ]
                + " · common y-scale across all four channels"
                "</sup>"
            ),
        }

        for axis_name in (
            axis_names
        ):
            layout_updates[
                f"{axis_name}.range"
            ] = (
                source_ranges[
                    source_key
                ]
            )

        buttons.append(
            dict(
                label=source_info[
                    "label"
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
                "Individual repaired relative weight responses"
                "<br><sup>"
                "Scale · Envelope · common y-scale across all four channels"
                "</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        height=900,
        width=1250,
        hovermode="closest",
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
        annotations=[
            *figure.layout.annotations,
            dict(
                text="Uncertainty source:",
                x=0.01,
                y=1.20,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
            ),
        ],
        margin=dict(
            l=90,
            r=40,
            t=165,
            b=75,
        ),
    )

    initial_range = (
        source_ranges[
            initial_source
        ]
    )

    for pair_info in (
        PAIR_DEFINITIONS.values()
    ):
        row = pair_info[
            "row"
        ]

        col = pair_info[
            "col"
        ]

        figure.update_xaxes(
            title_text="Lepton |eta|",
            range=[
                0.0,
                2.5,
            ],
            showgrid=True,
            matches="x",
            row=row,
            col=col,
        )

        figure.update_yaxes(
            title_text=(
                "(C_weight - C_weight0) / C_weight0 (%)"
                if col == 1
                else None
            ),
            range=initial_range,
            zeroline=True,
            zerolinewidth=2,
            showgrid=True,
            matches="y",
            row=row,
            col=col,
        )

    return figure


# ============================================================
# Standalone HTML outputs
# ============================================================

def write_standalone_html(
    figure,
    path,
):
    figure.write_html(
        str(
            path
        ),
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )


# ============================================================
# Combined dashboard
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
        },
    )


def write_dashboard(
    summary_figure,
    weight_figure,
    validation_results,
):
    plotly_js = (
        get_plotlyjs()
    )

    summary_div = figure_div(
        summary_figure
    )

    weight_div = figure_div(
        weight_figure
    )

    validation_rows = ""

    for pair_label, pair_info in (
        PAIR_DEFINITIONS.items()
    ):
        result = validation_results[
            pair_label
        ]

        counts = result[
            "counts"
        ]

        validation_rows += f"""
        <tr>
            <td>{pair_info['title']}</td>
            <td>{counts['scale']}</td>
            <td>{counts['pdf']}</td>
            <td>{counts['shower']}</td>
            <td>{counts['model']}</td>
            <td>{result['max_shift_delta']:.2e}</td>
            <td>{max(result['final_deltas'].values()):.2e}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Repaired relative systematic uncertainty analysis</title>
<script>{plotly_js}</script>
<style>
    :root {{
        --page: #f5f7fa;
        --panel: #ffffff;
        --text: #182230;
        --muted: #5f6b7a;
        --border: #dce3eb;
        --active: #e7edf3;
    }}

    * {{
        box-sizing: border-box;
    }}

    body {{
        margin: 0;
        background: var(--page);
        color: var(--text);
        font-family: Arial, Helvetica, sans-serif;
    }}

    .page {{
        max-width: 1360px;
        margin: 0 auto;
        padding: 22px;
    }}

    .hero,
    .plot-shell,
    .explanation,
    .method-card {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 11px;
    }}

    .hero {{
        padding: 19px 21px;
        margin-bottom: 14px;
    }}

    .hero h1 {{
        margin: 0 0 7px 0;
        font-size: 26px;
    }}

    .hero p {{
        margin: 0;
        color: var(--muted);
        line-height: 1.55;
    }}

    .tabs {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 13px;
    }}

    .tab-button {{
        padding: 9px 13px;
        border: 1px solid var(--border);
        border-radius: 7px;
        background: var(--panel);
        color: var(--text);
        cursor: pointer;
        font-size: 14px;
    }}

    .tab-button.active {{
        background: var(--active);
        font-weight: 650;
    }}

    .panel {{
        display: none;
    }}

    .panel.active {{
        display: block;
    }}

    .plot-shell {{
        padding: 7px;
        overflow: hidden;
    }}

    .explanation {{
        margin-top: 12px;
        padding: 15px 17px;
        line-height: 1.58;
        color: #39495a;
        border-left: 5px solid #9aa8b6;
    }}

    .method-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 13px;
    }}

    .method-card {{
        padding: 17px;
        min-width: 0;
    }}

    .method-card h2 {{
        margin: 0 0 9px 0;
        font-size: 18px;
    }}

    .method-card p {{
        color: var(--muted);
        line-height: 1.55;
    }}

    .equation {{
        font-size: 20px;
        margin: 8px 0;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        font-size: 14px;
    }}

    th,
    td {{
        text-align: right;
        padding: 8px 9px;
        border-bottom: 1px solid var(--border);
    }}

    th:first-child,
    td:first-child {{
        text-align: left;
    }}

    .validation {{
        margin-top: 14px;
        overflow-x: auto;
    }}

    .footer {{
        margin-top: 16px;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.5;
        text-align: center;
    }}

    @media (max-width: 780px) {{
        .page {{
            padding: 12px;
        }}

        .method-grid {{
            grid-template-columns: 1fr;
        }}

        .hero h1 {{
            font-size: 22px;
        }}
    }}
</style>
</head>
<body>
<div class="page">

<div class="hero">
    <h1>Repaired relative systematic uncertainty analysis</h1>
    <p>
        All variations are formed relative to weight 0 from the same repaired
        weighted event sample. The high-stat/event-matching Pythia study is
        not used anywhere in these plots. The four panels use synchronized
        axes to make generator and charge comparisons visually meaningful.
    </p>
</div>

<div class="tabs">
    <button
        type="button"
        class="tab-button active"
        data-target="summary"
    >
        Decomposition
    </button>

    <button
        type="button"
        class="tab-button"
        data-target="weights"
    >
        Individual weight responses
    </button>

    <button
        type="button"
        class="tab-button"
        data-target="method"
    >
        Method & validation
    </button>
</div>

<section
    id="panel-summary"
    class="panel active"
>
    <div class="plot-shell">
        {summary_div}
    </div>

    <div class="explanation">
        <strong>How to read this plot.</strong>
        Each source is a fractional uncertainty on the correction factor.
        Scale, shower and model use the largest absolute excursion from
        weight 0; PDF uses the RMS of its relative variations. The total
        systematic is the quadrature of those four source magnitudes.
        Pythia contains all four sources. Herwig has no common shower/model
        variations in these weighted files; the stored zero values therefore
        mean <em>unavailable in this workflow</em>, not that the physical
        shower/model uncertainty has been demonstrated to vanish.
    </div>
</section>

<section
    id="panel-weights"
    class="panel"
>
    <div class="plot-shell">
        {weight_div}
    </div>

    <div class="explanation">
        <strong>Individual correlated responses.</strong>
        Use the dropdown above the graph to switch source. Every thin response
        is one real common particle/parton weight, evaluated relative to
        nominal weight 0 from the same repaired event sample. The dashed
        boundary is the envelope for scale/shower/model and the +/-RMS for
        PDF. The y-axis automatically changes to a common symmetric range for
        the selected source, while remaining synchronized across all four
        panels.
    </div>
</section>

<section
    id="panel-method"
    class="panel"
>
    <div class="method-grid">

        <div class="method-card">
            <h2>Relative shift</h2>
            <div class="equation">
                delta<sub>i</sub><sup>(k)</sup> =
                (C<sub>i</sub><sup>(k)</sup> -
                C<sub>i</sub><sup>(0)</sup>) /
                C<sub>i</sub><sup>(0)</sup>
            </div>
            <p>
                The varied and nominal corrections come from the same weighted
                Monte Carlo event sample. The quantity plotted is the
                systematic response, not a second independent measurement.
            </p>
        </div>

        <div class="method-card">
            <h2>Envelope sources</h2>
            <div class="equation">
                delta<sub>X,i</sub> =
                max<sub>k in X</sub> |delta<sub>i</sub><sup>(k)</sup>|
            </div>
            <p>
                This definition is used for scale, shower and model
                variations when those common weights exist.
            </p>
        </div>

        <div class="method-card">
            <h2>PDF source</h2>
            <div class="equation">
                delta<sub>PDF,i</sub> =
                sqrt(mean<sub>k</sub>
                [delta<sub>i</sub><sup>(k)2</sup>])
            </div>
            <p>
                PDF uncertainty is the RMS of the included relative PDF
                responses. PDF weight 281 and excluded weight 318 are not
                included by the upstream repaired builder.
            </p>
        </div>

        <div class="method-card">
            <h2>Total systematic</h2>
            <div class="equation">
                delta<sub>syst</sub> =
                sqrt(delta<sub>scale</sub>2 +
                delta<sub>PDF</sub>2 +
                delta<sub>shower</sub>2 +
                delta<sub>model</sub>2)
            </div>
            <p>
                The official conservative final combination later converts
                each relative component to an absolute uncertainty using
                sigma<sub>X</sub> = |C| delta<sub>X</sub>.
            </p>
        </div>

    </div>

    <div class="method-card validation">
        <h2>Automatic validation performed before plotting</h2>
        <p>
            The table below shows the number of real variations found in each
            channel. The final two columns show numerical closure tests:
            the per-weight shift identity and agreement with the official
            final-conservative CSV. Values close to floating-point precision
            indicate exact pipeline consistency.
        </p>

        <table>
            <thead>
                <tr>
                    <th>Channel</th>
                    <th>Scale</th>
                    <th>PDF</th>
                    <th>Shower</th>
                    <th>Model</th>
                    <th>max shift delta</th>
                    <th>max final delta</th>
                </tr>
            </thead>
            <tbody>
                {validation_rows}
            </tbody>
        </table>
    </div>

    <div class="explanation">
        <strong>Final-result boundary.</strong>
        These figures describe the repaired relative weighted-sample
        systematics only. They do not import the exploratory Pythia
        high-statistics fragments. The official conservative result explicitly
        records <code>high_stat_nominal_used = False</code>.
    </div>
</section>

<div class="footer">
    Repaired W+c lepton-|eta| uncertainty decomposition ·
    synchronized interactive plots
</div>

</div>

<script>
(function () {{
    const buttons = Array.from(
        document.querySelectorAll(".tab-button")
    );

    const panels = Array.from(
        document.querySelectorAll(".panel")
    );

    buttons.forEach(function (button) {{
        button.addEventListener(
            "click",
            function () {{
                const target = button.dataset.target;

                buttons.forEach(function (other) {{
                    other.classList.toggle(
                        "active",
                        other === button
                    );
                }});

                panels.forEach(function (panel) {{
                    panel.classList.toggle(
                        "active",
                        panel.id === "panel-" + target
                    );
                }});

                window.setTimeout(function () {{
                    const visible = document.querySelector(
                        ".panel.active"
                    );

                    if (!visible) {{
                        return;
                    }}

                    const plots = visible.querySelectorAll(
                        ".plotly-graph-div"
                    );

                    plots.forEach(function (plot) {{
                        if (
                            window.Plotly
                            && Plotly.Plots
                        ) {{
                            Plotly.Plots.resize(
                                plot
                            );
                        }}
                    }});
                }}, 30);
            }}
        );
    }});
}})();
</script>

</body>
</html>
"""

    DASHBOARD_HTML.write_text(
        html,
        encoding="utf-8",
    )


# ============================================================
# Validation summary text
# ============================================================

def write_validation_summary(
    validation_results,
    all_data,
):
    lines = [
        "Repaired relative-systematics plotting validation",
        "================================================",
        "",
        "Input relative-systematics directory:",
        f"  {CSV_DIR}",
        "",
        "Official conservative comparison directory:",
        f"  {FINAL_CONSERVATIVE_CSV_DIR}",
        "",
        "High-stat/event-matching nominal used: False",
        "",
    ]

    for pair_label, result in (
        validation_results.items()
    ):
        lines.extend([
            pair_label,
            "-" * len(
                pair_label
            ),
            (
                "  variations: "
                f"scale={result['counts']['scale']}, "
                f"pdf={result['counts']['pdf']}, "
                f"shower={result['counts']['shower']}, "
                f"model={result['counts']['model']}"
            ),
            (
                "  max summary quadrature delta: "
                f"{result['max_summary_total_delta']:.3e}"
            ),
            (
                "  max per-weight shift delta: "
                f"{result['max_shift_delta']:.3e}"
            ),
            (
                "  max per-weight ratio delta: "
                f"{result['max_ratio_delta']:.3e}"
            ),
            (
                "  max per-weight nominal delta: "
                f"{result['max_nominal_delta']:.3e}"
            ),
            (
                "  max per-weight percent delta: "
                f"{result['max_percent_delta']:.3e}"
            ),
            (
                "  max final-conservative comparison delta: "
                f"{max(result['final_deltas'].values()):.3e}"
            ),
            "",
        ])

    lines.extend([
        "Outputs:",
        f"  decomposition: {SUMMARY_HTML}",
        f"  weight responses: {WEIGHTS_HTML}",
        f"  dashboard: {DASHBOARD_HTML}",
    ])

    VALIDATION_SUMMARY.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


# ============================================================
# Browser helper
# ============================================================

def open_in_windows_browser(
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

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ) as exc:
        print()
        print(
            "Could not automatically open the Windows browser:"
        )

        print(
            f"  {exc}"
        )

        print()
        print(
            "Open this file manually:"
        )

        print(
            f"  {path}"
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate and plot the repaired relative systematic "
            "uncertainty analysis."
        )
    )

    parser.add_argument(
        "--open",
        action="store_true",
        help=(
            "Open the combined dashboard in the default "
            "Windows browser after creating it."
        ),
    )

    args = parser.parse_args()

    print()
    print(
        "=" * 72
    )

    print(
        " Repaired interactive relative-systematics plots"
    )

    print(
        "=" * 72
    )

    print()
    print(
        f"Relative-systematic CSVs:\n  {CSV_DIR}"
    )

    print()
    print(
        "Official conservative comparison CSVs:"
    )

    print(
        f"  {FINAL_CONSERVATIVE_CSV_DIR}"
    )

    print()
    print(
        "High-stat/event-matching Pythia result:"
    )

    print(
        "  NOT USED"
    )

    all_data = load_all_data()

    print()
    print(
        "Validating all four channels..."
    )

    validation_results = (
        validate_all_data(
            all_data
        )
    )

    for pair_label, result in (
        validation_results.items()
    ):
        counts = result[
            "counts"
        ]

        print()
        print(
            f"  {pair_label}:"
        )

        print(
            "    variations: "
            f"scale={counts['scale']}, "
            f"PDF={counts['pdf']}, "
            f"shower={counts['shower']}, "
            f"model={counts['model']}"
        )

        print(
            "    max summary quadrature delta: "
            f"{result['max_summary_total_delta']:.3e}"
        )

        print(
            "    max per-weight shift delta: "
            f"{result['max_shift_delta']:.3e}"
        )

        print(
            "    max final-conservative delta: "
            f"{max(result['final_deltas'].values()):.3e}"
        )

    print()
    print(
        "Building synchronized 2x2 decomposition..."
    )

    summary_figure = build_summary_figure(
        all_data
    )

    print(
        "Building synchronized 2x2 weight-response figure..."
    )

    weight_figure = (
        build_weight_response_figure(
            all_data
        )
    )

    write_standalone_html(
        summary_figure,
        SUMMARY_HTML,
    )

    write_standalone_html(
        weight_figure,
        WEIGHTS_HTML,
    )

    write_dashboard(
        summary_figure,
        weight_figure,
        validation_results,
    )

    write_validation_summary(
        validation_results,
        all_data,
    )

    print()
    print(
        "=" * 72
    )

    print(
        " Created successfully"
    )

    print(
        "=" * 72
    )

    print()
    print(
        f"Decomposition:\n  {SUMMARY_HTML}"
    )

    print(
        f"Weight responses:\n  {WEIGHTS_HTML}"
    )

    print(
        f"Dashboard:\n  {DASHBOARD_HTML}"
    )

    print(
        f"Validation summary:\n  {VALIDATION_SUMMARY}"
    )

    print()
    print(
        "The combined dashboard is the main file to inspect."
    )

    if args.open:
        open_in_windows_browser(
            DASHBOARD_HTML
        )


if __name__ == "__main__":
    main()
