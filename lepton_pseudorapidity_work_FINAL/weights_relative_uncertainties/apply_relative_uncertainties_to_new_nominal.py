#!/usr/bin/env python3

"""
apply_relative_uncertainties_to_new_nominal.py

REPAIRED CONSERVATIVE FINAL-COMBINATION VERSION.

Why this file changed role
--------------------------
The original version replaced the Pythia nominal/statistical uncertainty
with a supposedly high-statistics weight-0 sample and transferred the
relative systematics onto it.

A later validation found that those high-statistics fragment files contain
extensive duplicated event content.  They therefore cannot be treated as
20 independent high-statistics samples in the final uncertainty calculation.

This repaired version deliberately DOES NOT use the high-statistics
cross-check as the official nominal.

Instead, for all four channels

    Pythia W+
    Pythia W-
    Herwig W+
    Herwig W-

it uses the fully repaired weighted-sample result from

    MTWcut_build_corrections_etalepton.py

as the official nominal and statistical uncertainty, and combines it with
the relative systematic uncertainties from

    build_relative_uncertainties.py

using

    sigma_X = |C_0| * delta_X

for

    X = scale, PDF, shower, model.

It then independently verifies that the resulting absolute systematic
uncertainties reproduce the repaired correction-builder output.

Final definitions
-----------------
Nominal:
    C_final = C_0 from the repaired weighted sample.

Statistical:
    sigma_stat = independent ROOT/Sumw2 ratio uncertainty from the repaired
                 weighted sample.

Relative systematic responses:
    delta_scale
    delta_PDF
    delta_shower
    delta_model

Absolute systematic uncertainties:
    sigma_X = |C_final| * delta_X

Total systematic:
    sigma_syst = sqrt(
        sigma_scale^2
        + sigma_PDF^2
        + sigma_shower^2
        + sigma_model^2
    )

Total:
    sigma_total = sqrt(
        sigma_stat^2
        + sigma_syst^2
    )

High-stat/event-matching study
------------------------------
NOT used in the official final result.

This is deliberate.  The exploratory high-stat Pythia fragments were found
to contain duplicated event content, so neither their apparent increased
statistics nor the event-paired covariance result is adopted here.

Expected location
-----------------
uncertainty_decomposition/
└── lepton_pseudorapidity_work_FINAL/
    └── weights_relative_uncertainties/
        ├── build_relative_uncertainties.py
        └── apply_relative_uncertainties_to_new_nominal.py

Inputs
------
1. Repaired absolute corrections:
   ../MTWcut_build_corrections_etalepton_csv/
       Pythia_plus_etalepton.csv
       Pythia_minus_etalepton.csv
       Herwig_plus_etalepton.csv
       Herwig_minus_etalepton.csv

2. Repaired relative systematics:
   outputs/csv/
       <channel>_relative_uncertainties.csv

Outputs
-------
outputs/final_conservative/
    csv/
    root/
    interactive/
    final_combination_summary.txt

Run
---
python3 lepton_pseudorapidity_work_FINAL/weights_relative_uncertainties/apply_relative_uncertainties_to_new_nominal.py

Optional:
    --open
"""

from __future__ import annotations

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

CORRECTION_CSV_DIR = (
    LEPTON_WORK_DIR
    / "MTWcut_build_corrections_etalepton_csv"
)

RELATIVE_CSV_DIR = (
    SCRIPT_DIR
    / "outputs"
    / "csv"
)

FINAL_OUTPUT_DIR = (
    SCRIPT_DIR
    / "outputs"
    / "final_conservative"
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

SUMMARY_PATH = (
    FINAL_OUTPUT_DIR
    / "final_combination_summary.txt"
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
        "generator": "Pythia",
        "charge": "plus",
        "correction_csv": (
            CORRECTION_CSV_DIR
            / "Pythia_plus_etalepton.csv"
        ),
        "relative_csv": (
            RELATIVE_CSV_DIR
            / "Pythia_plus_relative_uncertainties.csv"
        ),
    },
    "Pythia_minus": {
        "display": "Pythia W-",
        "generator": "Pythia",
        "charge": "minus",
        "correction_csv": (
            CORRECTION_CSV_DIR
            / "Pythia_minus_etalepton.csv"
        ),
        "relative_csv": (
            RELATIVE_CSV_DIR
            / "Pythia_minus_relative_uncertainties.csv"
        ),
    },
    "Herwig_plus": {
        "display": "Herwig W+",
        "generator": "Herwig",
        "charge": "plus",
        "correction_csv": (
            CORRECTION_CSV_DIR
            / "Herwig_plus_etalepton.csv"
        ),
        "relative_csv": (
            RELATIVE_CSV_DIR
            / "Herwig_plus_relative_uncertainties.csv"
        ),
    },
    "Herwig_minus": {
        "display": "Herwig W-",
        "generator": "Herwig",
        "charge": "minus",
        "correction_csv": (
            CORRECTION_CSV_DIR
            / "Herwig_minus_etalepton.csv"
        ),
        "relative_csv": (
            RELATIVE_CSV_DIR
            / "Herwig_minus_relative_uncertainties.csv"
        ),
    },
}


# ============================================================
# CSV schemas
# ============================================================

CORRECTION_REQUIRED_COLUMNS = {
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

RELATIVE_REQUIRED_COLUMNS = {
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
}


# ============================================================
# Numerical tolerances
# ============================================================

BIN_TOLERANCE = 1.0e-10

# The two stages are built from the same ROOT histograms.  Agreement should
# be much better than this; the tolerance only protects against ordinary
# floating-point/CSV formatting differences.
ABSOLUTE_UNCERTAINTY_ATOL = 5.0e-7
ABSOLUTE_UNCERTAINTY_RTOL = 5.0e-6

NOMINAL_ATOL = 5.0e-7
NOMINAL_RTOL = 5.0e-6


# ============================================================
# Utility helpers
# ============================================================

def read_csv_rows(
    path: Path,
    required_columns,
):
    if not path.is_file():
        raise FileNotFoundError(
            "Required CSV does not exist:\n"
            f"  {path}"
        )

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
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
                "CSV is missing required columns:\n"
                f"  {path}\n"
                f"Missing: {sorted(missing)}"
            )

        rows = list(
            reader
        )

    if not rows:
        raise RuntimeError(
            f"CSV contains no data rows:\n  {path}"
        )

    return rows


def is_close(
    a,
    b,
    atol=ABSOLUTE_UNCERTAINTY_ATOL,
    rtol=ABSOLUTE_UNCERTAINTY_RTOL,
):
    return math.isclose(
        float(a),
        float(b),
        rel_tol=rtol,
        abs_tol=atol,
    )


def percent(
    uncertainty,
    correction,
):
    correction_magnitude = abs(
        float(
            correction
        )
    )

    if correction_magnitude == 0.0:
        raise RuntimeError(
            "Cannot form a relative uncertainty for zero correction."
        )

    return (
        100.0
        * abs(
            float(
                uncertainty
            )
        )
        / correction_magnitude
    )


def quadrature(
    *values,
):
    return math.sqrt(
        sum(
            float(value) ** 2
            for value in values
        )
    )


# ============================================================
# Input conversion
# ============================================================

def convert_correction_rows(
    rows,
):
    converted = []

    for row in rows:
        converted.append({
            "pair_label": row[
                "pair_label"
            ],
            "observable": row[
                "observable"
            ],
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
            "correction_factor": float(
                row[
                    "correction_factor"
                ]
            ),
            "stat_unc": abs(
                float(
                    row[
                        "stat_unc"
                    ]
                )
            ),
            "scale_unc_reference": abs(
                float(
                    row[
                        "scale_unc"
                    ]
                )
            ),
            "pdf_unc_reference": abs(
                float(
                    row[
                        "pdf_unc"
                    ]
                )
            ),
            "shower_unc_reference": abs(
                float(
                    row[
                        "shower_unc"
                    ]
                )
            ),
            "model_unc_reference": abs(
                float(
                    row[
                        "model_unc"
                    ]
                )
            ),
            "total_unc_reference": abs(
                float(
                    row[
                        "total_unc"
                    ]
                )
            ),
        })

    return converted


def convert_relative_rows(
    rows,
):
    converted = []

    for row in rows:
        converted.append({
            "pair_label": row[
                "pair_label"
            ],
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
            "nominal_correction_weight0": float(
                row[
                    "nominal_correction_weight0"
                ]
            ),
            "relative_scale_unc": abs(
                float(
                    row[
                        "relative_scale_unc"
                    ]
                )
            ),
            "relative_pdf_unc": abs(
                float(
                    row[
                        "relative_pdf_unc"
                    ]
                )
            ),
            "relative_shower_unc": abs(
                float(
                    row[
                        "relative_shower_unc"
                    ]
                )
            ),
            "relative_model_unc": abs(
                float(
                    row[
                        "relative_model_unc"
                    ]
                )
            ),
            "relative_total_systematic_unc": abs(
                float(
                    row[
                        "relative_total_systematic_unc"
                    ]
                )
            ),
        })

    return converted


# ============================================================
# Binning / nominal validation
# ============================================================

def validate_matching_inputs(
    channel,
    correction_rows,
    relative_rows,
):
    if len(
        correction_rows
    ) != len(
        relative_rows
    ):
        raise RuntimeError(
            f"{channel}: correction and relative CSVs contain "
            "different numbers of eta bins."
        )

    expected_bins = list(
        range(
            1,
            len(
                correction_rows
            )
            + 1,
        )
    )

    actual_bins = [
        row[
            "bin"
        ]
        for row in correction_rows
    ]

    if actual_bins != expected_bins:
        raise RuntimeError(
            f"{channel}: correction CSV bin indices are not contiguous."
        )

    max_nominal_delta = 0.0

    for correction_row, relative_row in zip(
        correction_rows,
        relative_rows,
    ):
        if (
            correction_row[
                "bin"
            ]
            != relative_row[
                "bin"
            ]
        ):
            raise RuntimeError(
                f"{channel}: bin index mismatch between correction "
                "and relative-systematic inputs."
            )

        for edge_name in (
            "bin_low_edge",
            "bin_up_edge",
        ):
            if not math.isclose(
                correction_row[
                    edge_name
                ],
                relative_row[
                    edge_name
                ],
                rel_tol=0.0,
                abs_tol=BIN_TOLERANCE,
            ):
                raise RuntimeError(
                    f"{channel}: eta-bin mismatch for {edge_name}."
                )

        if (
            correction_row[
                "pair_label"
            ]
            != channel
        ):
            raise RuntimeError(
                f"{channel}: pair_label mismatch in correction CSV."
            )

        if (
            relative_row[
                "pair_label"
            ]
            != channel
        ):
            raise RuntimeError(
                f"{channel}: pair_label mismatch in relative CSV."
            )

        if (
            correction_row[
                "observable"
            ]
            != "etalepton"
        ):
            raise RuntimeError(
                f"{channel}: unexpected observable "
                f"{correction_row['observable']!r}."
            )

        nominal_delta = abs(
            correction_row[
                "correction_factor"
            ]
            - relative_row[
                "nominal_correction_weight0"
            ]
        )

        max_nominal_delta = max(
            max_nominal_delta,
            nominal_delta,
        )

        if not math.isclose(
            correction_row[
                "correction_factor"
            ],
            relative_row[
                "nominal_correction_weight0"
            ],
            rel_tol=NOMINAL_RTOL,
            abs_tol=NOMINAL_ATOL,
        ):
            raise RuntimeError(
                f"{channel}: relative-systematic nominal does not "
                "match repaired correction-builder nominal.\n"
                f"  bin: {correction_row['bin']}\n"
                f"  correction builder: "
                f"{correction_row['correction_factor']:.12g}\n"
                f"  relative stage:     "
                f"{relative_row['nominal_correction_weight0']:.12g}"
            )

    return max_nominal_delta


# ============================================================
# Conservative final combination
# ============================================================

def build_final_rows(
    channel,
    definition,
    correction_rows,
    relative_rows,
):
    max_nominal_delta = (
        validate_matching_inputs(
            channel,
            correction_rows,
            relative_rows,
        )
    )

    final_rows = []

    consistency_deltas = {
        "scale": 0.0,
        "pdf": 0.0,
        "shower": 0.0,
        "model": 0.0,
        "total": 0.0,
    }

    for correction_row, relative_row in zip(
        correction_rows,
        relative_rows,
    ):
        correction = float(
            correction_row[
                "correction_factor"
            ]
        )

        correction_magnitude = abs(
            correction
        )

        if correction_magnitude == 0.0:
            raise RuntimeError(
                f"{channel}: zero nominal correction in "
                f"bin {correction_row['bin']}."
            )

        stat_unc = abs(
            correction_row[
                "stat_unc"
            ]
        )

        relative_scale = (
            relative_row[
                "relative_scale_unc"
            ]
        )

        relative_pdf = (
            relative_row[
                "relative_pdf_unc"
            ]
        )

        relative_shower = (
            relative_row[
                "relative_shower_unc"
            ]
        )

        relative_model = (
            relative_row[
                "relative_model_unc"
            ]
        )

        scale_unc = (
            correction_magnitude
            * relative_scale
        )

        pdf_unc = (
            correction_magnitude
            * relative_pdf
        )

        shower_unc = (
            correction_magnitude
            * relative_shower
        )

        model_unc = (
            correction_magnitude
            * relative_model
        )

        systematic_unc = quadrature(
            scale_unc,
            pdf_unc,
            shower_unc,
            model_unc,
        )

        total_unc = quadrature(
            stat_unc,
            systematic_unc,
        )

        source_checks = {
            "scale": (
                scale_unc,
                correction_row[
                    "scale_unc_reference"
                ],
            ),
            "pdf": (
                pdf_unc,
                correction_row[
                    "pdf_unc_reference"
                ],
            ),
            "shower": (
                shower_unc,
                correction_row[
                    "shower_unc_reference"
                ],
            ),
            "model": (
                model_unc,
                correction_row[
                    "model_unc_reference"
                ],
            ),
            "total": (
                total_unc,
                correction_row[
                    "total_unc_reference"
                ],
            ),
        }

        for source, (
            recomputed,
            reference,
        ) in source_checks.items():
            delta = abs(
                recomputed
                - reference
            )

            consistency_deltas[
                source
            ] = max(
                consistency_deltas[
                    source
                ],
                delta,
            )

            if not is_close(
                recomputed,
                reference,
            ):
                raise RuntimeError(
                    f"{channel}: recomposed {source} uncertainty "
                    "does not reproduce the repaired correction-builder "
                    "result.\n"
                    f"  bin:        {correction_row['bin']}\n"
                    f"  recomputed: {recomputed:.12g}\n"
                    f"  reference:  {reference:.12g}\n"
                    f"  |delta|:    {delta:.6g}"
                )

        relative_total_recomputed = quadrature(
            relative_scale,
            relative_pdf,
            relative_shower,
            relative_model,
        )

        if not math.isclose(
            relative_total_recomputed,
            relative_row[
                "relative_total_systematic_unc"
            ],
            rel_tol=ABSOLUTE_UNCERTAINTY_RTOL,
            abs_tol=ABSOLUTE_UNCERTAINTY_ATOL,
        ):
            raise RuntimeError(
                f"{channel}: relative total systematic uncertainty "
                "does not equal the quadrature of its components in "
                f"bin {correction_row['bin']}."
            )

        final_rows.append({
            "channel": (
                channel
            ),
            "display": (
                definition[
                    "display"
                ]
            ),
            "generator": (
                definition[
                    "generator"
                ]
            ),
            "charge": (
                definition[
                    "charge"
                ]
            ),
            "bin": (
                correction_row[
                    "bin"
                ]
            ),
            "bin_low_edge": (
                correction_row[
                    "bin_low_edge"
                ]
            ),
            "bin_up_edge": (
                correction_row[
                    "bin_up_edge"
                ]
            ),
            "nominal_correction": (
                correction
            ),
            "stat_unc": (
                stat_unc
            ),
            "relative_scale_unc": (
                relative_scale
            ),
            "relative_pdf_unc": (
                relative_pdf
            ),
            "relative_shower_unc": (
                relative_shower
            ),
            "relative_model_unc": (
                relative_model
            ),
            "relative_total_systematic_unc": (
                relative_total_recomputed
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
            "stat_unc_percent": percent(
                stat_unc,
                correction,
            ),
            "scale_unc_percent": (
                100.0
                * relative_scale
            ),
            "pdf_unc_percent": (
                100.0
                * relative_pdf
            ),
            "shower_unc_percent": (
                100.0
                * relative_shower
            ),
            "model_unc_percent": (
                100.0
                * relative_model
            ),
            "systematic_unc_percent": percent(
                systematic_unc,
                correction,
            ),
            "total_unc_percent": percent(
                total_unc,
                correction,
            ),
            "official_nominal_source": (
                "repaired weighted-sample weight 0"
            ),
            "official_statistical_method": (
                "independent ROOT Sumw2 ratio propagation"
            ),
            "high_stat_nominal_used": (
                "False"
            ),
        })

    return (
        final_rows,
        max_nominal_delta,
        consistency_deltas,
    )


# ============================================================
# Final CSV
# ============================================================

FINAL_COLUMNS = [
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
    "official_nominal_source",
    "official_statistical_method",
    "high_stat_nominal_used",
]


def write_final_csv(
    channel,
    final_rows,
):
    path = (
        FINAL_CSV_DIR
        / f"{channel}_final_conservative.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=FINAL_COLUMNS,
        )

        writer.writeheader()

        writer.writerows(
            final_rows
        )

    return path


# ============================================================
# ROOT output
# ============================================================

def make_histogram(
    name,
    title,
    bin_edges,
):
    histogram = ROOT.TH1D(
        name,
        title,
        len(
            bin_edges
        ) - 1,
        array(
            "d",
            bin_edges,
        ),
    )

    histogram.SetDirectory(
        0
    )

    return histogram


def write_root_output(
    channel,
    definition,
    final_rows,
    max_nominal_delta,
    consistency_deltas,
):
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
            f"{definition['display']};"
            f"|#eta_{{#ell}}|;"
            "Correction factor"
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
                "nominal_correction"
            ],
        )

        h_nominal.SetBinError(
            ibin,
            row[
                "total_unc"
            ],
        )

        h_stat.SetBinContent(
            ibin,
            row[
                "stat_unc"
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

    path = (
        FINAL_ROOT_DIR
        / f"{channel}_final_conservative.root"
    )

    output = ROOT.TFile(
        str(
            path
        ),
        "RECREATE",
    )

    if (
        not output
        or output.IsZombie()
    ):
        raise RuntimeError(
            f"Could not create ROOT output:\n  {path}"
        )

    ROOT.TNamed(
        "Channel",
        channel,
    ).Write()

    ROOT.TNamed(
        "NominalDefinition",
        "repaired weighted-sample weight-0 parton-to-particle correction",
    ).Write()

    ROOT.TNamed(
        "StatisticalUncertaintyDefinition",
        "independent ROOT Sumw2 ratio propagation",
    ).Write()

    ROOT.TNamed(
        "SystematicTransferDefinition",
        "sigma_X = |C_weight0| * delta_X",
    ).Write()

    ROOT.TNamed(
        "RelativeSystematicDefinition",
        "delta_k = (C_k - C_0) / C_0 from the same repaired weighted sample",
    ).Write()

    ROOT.TNamed(
        "TotalUncertaintyDefinition",
        "quadrature(stat, scale, PDF, shower, model)",
    ).Write()

    ROOT.TNamed(
        "HighStatNominalUsed",
        "False",
    ).Write()

    ROOT.TNamed(
        "HighStatNominalExclusionReason",
        (
            "exploratory Pythia high-stat fragment validation found "
            "duplicated event content; not adopted in official final result"
        ),
    ).Write()

    ROOT.TNamed(
        "NominalCrossCheckMaxAbsDelta",
        f"{max_nominal_delta:.12g}",
    ).Write()

    for source, delta in (
        consistency_deltas.items()
    ):
        ROOT.TNamed(
            f"RecompositionMaxAbsDelta_{source}",
            f"{delta:.12g}",
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

    output.Write()
    output.Close()

    return path


# ============================================================
# Interactive plot
# ============================================================

def save_interactive_plot(
    completed_channels,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

    except ImportError:
        print()
        print(
            "Plotly is not installed; skipping interactive output."
        )

        return None

    figure = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        shared_yaxes="rows",
        vertical_spacing=0.13,
        horizontal_spacing=0.08,
        subplot_titles=[
            CHANNELS[
                key
            ][
                "display"
            ]
            for key in CHANNELS
        ],
    )

    positions = {
        "Pythia_plus": (
            1,
            1,
        ),
        "Pythia_minus": (
            1,
            2,
        ),
        "Herwig_plus": (
            2,
            1,
        ),
        "Herwig_minus": (
            2,
            2,
        ),
    }

    first_channel = True

    for channel, result in (
        completed_channels.items()
    ):
        rows = result[
            "rows"
        ]

        row_index, col_index = (
            positions[
                channel
            ]
        )

        x = [
            0.5
            * (
                item[
                    "bin_low_edge"
                ]
                + item[
                    "bin_up_edge"
                ]
            )
            for item in rows
        ]

        xerr = [
            0.5
            * (
                item[
                    "bin_up_edge"
                ]
                - item[
                    "bin_low_edge"
                ]
            )
            for item in rows
        ]

        correction = [
            item[
                "nominal_correction"
            ]
            for item in rows
        ]

        figure.add_trace(
            go.Scatter(
                x=x,
                y=correction,
                mode="lines+markers",
                name="Statistical uncertainty",
                legendgroup="stat",
                showlegend=first_channel,
                error_x=dict(
                    type="data",
                    array=xerr,
                    visible=True,
                ),
                error_y=dict(
                    type="data",
                    array=[
                        item[
                            "stat_unc"
                        ]
                        for item in rows
                    ],
                    visible=True,
                ),
                hovertemplate=(
                    "|ηℓ| = %{x:.3f}"
                    "<br>C = %{y:.6f}"
                    "<br>stat = %{error_y.array:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=row_index,
            col=col_index,
        )

        figure.add_trace(
            go.Scatter(
                x=x,
                y=correction,
                mode="markers",
                name="Total uncertainty",
                legendgroup="total",
                showlegend=first_channel,
                marker=dict(
                    symbol="circle-open",
                    size=9,
                ),
                error_y=dict(
                    type="data",
                    array=[
                        item[
                            "total_unc"
                        ]
                        for item in rows
                    ],
                    visible=True,
                ),
                hovertemplate=(
                    "|ηℓ| = %{x:.3f}"
                    "<br>C = %{y:.6f}"
                    "<br>total = %{error_y.array:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=row_index,
            col=col_index,
        )

        first_channel = False

    figure.update_layout(
        title=dict(
            text=(
                "Conservative repaired final correction factors"
                "<br><sup>"
                "Weighted-sample nominal/statistical uncertainty · "
                "relative systematics from the same repaired samples · "
                "high-stat duplicate-fragment result excluded"
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
            l=80,
            r=40,
            t=145,
            b=70,
        ),
    )

    for row_index in (
        1,
        2,
    ):
        for col_index in (
            1,
            2,
        ):
            figure.update_xaxes(
                range=[
                    0.0,
                    2.5,
                ],
                title_text="Lepton |η|",
                row=row_index,
                col=col_index,
            )

            figure.update_yaxes(
                title_text=(
                    "Correction factor"
                    if col_index == 1
                    else None
                ),
                row=row_index,
                col=col_index,
            )

    path = (
        FINAL_INTERACTIVE_DIR
        / "final_conservative_corrections.html"
    )

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

    return path


# ============================================================
# Summary output
# ============================================================

def write_summary(
    completed_channels,
    interactive_path,
):
    lines = [
        "Conservative final W+c uncertainty combination",
        "==============================================",
        "",
        "Official final treatment:",
        "  nominal: repaired weighted-sample weight 0",
        "  statistical: independent ROOT Sumw2 ratio uncertainty",
        "  scale: relative envelope transferred as |C| * delta_scale",
        "  PDF: relative RMS transferred as |C| * delta_PDF",
        "  shower: relative envelope transferred as |C| * delta_shower",
        "  model: relative envelope transferred as |C| * delta_model",
        "  total: quadrature(stat, scale, PDF, shower, model)",
        "",
        "High-stat Pythia nominal used: False",
        (
            "Reason: exploratory high-stat fragment validation found "
            "duplicated event content."
        ),
        "",
        "Consistency checks:",
    ]

    for channel, result in (
        completed_channels.items()
    ):
        lines.append(
            f"  {channel}:"
        )

        lines.append(
            "    max nominal delta between correction/relative stages: "
            f"{result['max_nominal_delta']:.3e}"
        )

        for source, delta in (
            result[
                "consistency_deltas"
            ].items()
        ):
            lines.append(
                f"    max recomposition delta {source}: {delta:.3e}"
            )

        lines.append(
            f"    CSV:  {result['csv_path']}"
        )

        lines.append(
            f"    ROOT: {result['root_path']}"
        )

    if interactive_path is not None:
        lines.extend([
            "",
            f"Interactive: {interactive_path}",
        ])

    SUMMARY_PATH.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


# ============================================================
# Browser helper
# ============================================================

def open_in_browser(
    path: Path,
):
    """
    Best-effort opening when running under WSL.
    Failure is non-fatal.
    """

    try:
        subprocess.run(
            [
                "cmd.exe",
                "/c",
                "start",
                "",
                str(
                    path
                ),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    except Exception:
        print(
            "Could not automatically open the browser; "
            f"open manually:\n  {path}"
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the conservative repaired final W+c correction/"
            "uncertainty outputs."
        )
    )

    parser.add_argument(
        "--open",
        action="store_true",
        help=(
            "Open the final interactive HTML after writing it."
        ),
    )

    args = parser.parse_args()

    print()
    print(
        "=" * 76
    )

    print(
        " Conservative repaired final uncertainty combination"
    )

    print(
        "=" * 76
    )

    print()
    print(
        "Official nominal/statistical treatment:"
    )

    print(
        "  repaired OLD weighted-sample weight 0"
    )

    print(
        "  independent ROOT Sumw2 ratio propagation"
    )

    print()
    print(
        "Relative systematics:"
    )

    print(
        "  scale/shower/model = envelope"
    )

    print(
        "  PDF = RMS"
    )

    print()
    print(
        "High-stat Pythia nominal:"
    )

    print(
        "  NOT USED (duplicate-fragment validation issue)"
    )

    print()
    print(
        "Repaired correction inputs:"
    )

    print(
        f"  {CORRECTION_CSV_DIR}"
    )

    print()
    print(
        "Relative systematic inputs:"
    )

    print(
        f"  {RELATIVE_CSV_DIR}"
    )

    print()
    print(
        "Final outputs:"
    )

    print(
        f"  {FINAL_OUTPUT_DIR}"
    )

    completed = {}

    for channel, definition in (
        CHANNELS.items()
    ):
        print()
        print(
            "-" * 76
        )

        print(
            f"{definition['display']}"
        )

        print(
            "-" * 76
        )

        correction_rows = (
            convert_correction_rows(
                read_csv_rows(
                    definition[
                        "correction_csv"
                    ],
                    CORRECTION_REQUIRED_COLUMNS,
                )
            )
        )

        relative_rows = (
            convert_relative_rows(
                read_csv_rows(
                    definition[
                        "relative_csv"
                    ],
                    RELATIVE_REQUIRED_COLUMNS,
                )
            )
        )

        (
            final_rows,
            max_nominal_delta,
            consistency_deltas,
        ) = build_final_rows(
            channel,
            definition,
            correction_rows,
            relative_rows,
        )

        csv_path = write_final_csv(
            channel,
            final_rows,
        )

        root_path = write_root_output(
            channel,
            definition,
            final_rows,
            max_nominal_delta,
            consistency_deltas,
        )

        completed[
            channel
        ] = {
            "display": (
                definition[
                    "display"
                ]
            ),
            "rows": (
                final_rows
            ),
            "csv_path": (
                csv_path
            ),
            "root_path": (
                root_path
            ),
            "max_nominal_delta": (
                max_nominal_delta
            ),
            "consistency_deltas": (
                consistency_deltas
            ),
        }

        print(
            f"  max nominal-stage |delta|: "
            f"{max_nominal_delta:.3e}"
        )

        for source, delta in (
            consistency_deltas.items()
        ):
            print(
                f"  max recomposition |delta| "
                f"{source:>7s}: {delta:.3e}"
            )

        print(
            f"  CSV:  {csv_path}"
        )

        print(
            f"  ROOT: {root_path}"
        )

    interactive_path = (
        save_interactive_plot(
            completed
        )
    )

    write_summary(
        completed,
        interactive_path,
    )

    print()
    print(
        "=" * 76
    )

    print(
        " Finished successfully"
    )

    print(
        "=" * 76
    )

    print()
    print(
        f"Summary:\n  {SUMMARY_PATH}"
    )

    if interactive_path is not None:
        print(
            f"Interactive:\n  {interactive_path}"
        )

    print()
    print(
        "The official final outputs now explicitly exclude the "
        "duplicated high-stat Pythia nominal."
    )

    if (
        args.open
        and interactive_path
        is not None
    ):
        open_in_browser(
            interactive_path
        )


if __name__ == "__main__":
    main()
