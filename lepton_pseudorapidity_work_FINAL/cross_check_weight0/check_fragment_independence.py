#!/usr/bin/env python3

"""
check_fragment_independence.py

Decisive diagnostic for the high-statistics Pythia fragment structure.

Purpose
-------
The previous final audit found that matching particle suffix s to parton
suffix s by EventNumber did NOT outperform the deliberately wrong
comparison particle suffix s -> parton suffix s+1.

This script determines why.

For each W charge it builds the FULL 20 x 20 matrix

    particle fragment i  versus  parton fragment j

using common EventNumber values and measures:

    - EventNumber overlap
    - lepton-pT Pearson correlation
    - MET Pearson correlation
    - mean cos(delta phi_lepton)
    - lepton-charge equality
    - exact equality of weightvec[0]
    - exact equality of a multi-variable event fingerprint

It also checks WITHIN each level (particle-particle and parton-parton)
whether different suffixes contain identical/reused event records.

Interpretation
--------------
1. If the particle-parton diagonal is clearly stronger than off-diagonal
   cells:
       suffix + EventNumber carries real event-pairing information.

2. If all columns/rows are effectively identical:
       EventNumber is probably local/reused and suffix pairing is not
       established by the available variables.

3. If different fragments have identical ordered-content SHA256 hashes:
       those fragments are duplicates for the tested event content.

4. If EventNumber repeats across fragments but the corresponding event
   content differs:
       EventNumber is a local counter and must not be used across files
       without an additional true event identifier.

This script changes no analysis result.

Recommended location
--------------------
lepton_pseudorapidity_work_FINAL/
└── cross_check_weight0/
    ├── check_fragment_independence.py
    └── cross_check_ROOT_files/

Run
---
python3 lepton_pseudorapidity_work_FINAL/cross_check_weight0/check_fragment_independence.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
import ROOT


ROOT.gROOT.SetBatch(True)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "cross_check_ROOT_files"

OUTPUT_DIR = (
    SCRIPT_DIR
    / "outputs"
    / "fragment_independence_check"
)

CSV_DIR = OUTPUT_DIR / "csv"
INTERACTIVE_DIR = OUTPUT_DIR / "interactive"

for directory in (
    CSV_DIR,
    INTERACTIVE_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

REPORT_PATH = OUTPUT_DIR / "fragment_independence_report.txt"

TREE_NAME = "WCharmTree"

CHANNELS = {
    "plus": {
        "display": "Pythia W+",
        "particle_prefix": "WCharm_WCPy8plus",
        "parton_prefix": "WCharm_WCPyPartonplus",
    },
    "minus": {
        "display": "Pythia W-",
        "particle_prefix": "WCharm_WCPy8minus",
        "parton_prefix": "WCharm_WCPyPartonminus",
    },
}

SCALAR_BRANCHES = [
    "EventNumber",
    "leptons_pt",
    "leptons_eta",
    "leptons_phi",
    "leptons_charge",
    "met_et",
    "met_phi",
    "weightvec",
]


REPORT_LINES = []


def report(line=""):
    line = str(line)
    print(line)
    REPORT_LINES.append(line)


def suffix_number(path, prefix):
    name = path.name

    if not (
        name.startswith(prefix)
        and name.endswith(".root")
    ):
        return None

    text = name[
        len(prefix):
        -len(".root")
    ]

    if not text.isdigit():
        return None

    return int(text)


def discover(prefix):
    result = {}

    for path in sorted(
        INPUT_DIR.glob(
            f"{prefix}*.root"
        )
    ):
        suffix = suffix_number(
            path,
            prefix,
        )

        if suffix is not None:
            result[
                suffix
            ] = path

    return result


def open_tree(path):
    root_file = ROOT.TFile.Open(
        str(path),
        "READ",
    )

    if (
        not root_file
        or root_file.IsZombie()
    ):
        raise RuntimeError(
            f"Could not open ROOT file:\n  {path}"
        )

    tree = root_file.Get(
        TREE_NAME
    )

    if not tree:
        root_file.Close()

        raise RuntimeError(
            f"{TREE_NAME} missing in:\n  {path}"
        )

    return root_file, tree


def branch_names(tree):
    return {
        branch.GetName()
        for branch in tree.GetListOfBranches()
    }


def wrap_phi_delta(a, b):
    return np.arctan2(
        np.sin(a - b),
        np.cos(a - b),
    )


def pearson(x, y):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if (
        len(x) < 2
        or len(y) < 2
    ):
        return math.nan

    x_std = float(
        np.std(
            x
        )
    )

    y_std = float(
        np.std(
            y
        )
    )

    if (
        x_std == 0.0
        or y_std == 0.0
    ):
        return math.nan

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def read_fragment(path):
    """
    Read a compact event representation.

    EventNumber must be unique inside a fragment.
    """

    root_file, tree = open_tree(
        path
    )

    names = branch_names(
        tree
    )

    required = {
        "EventNumber",
        "leptons_pt",
        "leptons_eta",
        "leptons_phi",
        "leptons_charge",
        "met_et",
        "met_phi",
        "weightvec",
    }

    missing = (
        required
        - names
    )

    if missing:
        root_file.Close()

        raise RuntimeError(
            f"{path.name}: missing branches "
            f"{sorted(missing)}"
        )

    nentries = int(
        tree.GetEntries()
    )

    event_numbers = np.empty(
        nentries,
        dtype=np.int64,
    )

    pt = np.empty(
        nentries,
        dtype=np.float64,
    )

    eta = np.empty(
        nentries,
        dtype=np.float64,
    )

    phi = np.empty(
        nentries,
        dtype=np.float64,
    )

    charge = np.empty(
        nentries,
        dtype=np.float64,
    )

    met = np.empty(
        nentries,
        dtype=np.float64,
    )

    met_phi = np.empty(
        nentries,
        dtype=np.float64,
    )

    weight0 = np.empty(
        nentries,
        dtype=np.float64,
    )

    for ientry in range(
        nentries
    ):
        tree.GetEntry(
            ientry
        )

        event_numbers[
            ientry
        ] = int(
            tree.EventNumber
        )

        pt[
            ientry
        ] = float(
            tree.leptons_pt
        )

        eta[
            ientry
        ] = float(
            tree.leptons_eta
        )

        phi[
            ientry
        ] = float(
            tree.leptons_phi
        )

        charge[
            ientry
        ] = float(
            tree.leptons_charge
        )

        met[
            ientry
        ] = float(
            tree.met_et
        )

        met_phi[
            ientry
        ] = float(
            tree.met_phi
        )

        weight0[
            ientry
        ] = float(
            tree.weightvec[0]
        )

    root_file.Close()

    if len(
        np.unique(
            event_numbers
        )
    ) != nentries:
        raise RuntimeError(
            f"{path.name}: EventNumber is not unique "
            "within the fragment."
        )

    # Sort by EventNumber so content hashes are independent of ROOT
    # entry ordering.
    order = np.argsort(
        event_numbers
    )

    data = {
        "EventNumber": (
            event_numbers[
                order
            ]
        ),
        "leptons_pt": (
            pt[
                order
            ]
        ),
        "leptons_eta": (
            eta[
                order
            ]
        ),
        "leptons_phi": (
            phi[
                order
            ]
        ),
        "leptons_charge": (
            charge[
                order
            ]
        ),
        "met_et": (
            met[
                order
            ]
        ),
        "met_phi": (
            met_phi[
                order
            ]
        ),
        "weight0": (
            weight0[
                order
            ]
        ),
    }

    data[
        "index"
    ] = {
        int(
            event_number
        ): index
        for index, event_number in enumerate(
            data[
                "EventNumber"
            ]
        )
    }

    return data


def update_hash_array(
    hasher,
    array,
):
    contiguous = np.ascontiguousarray(
        array
    )

    hasher.update(
        contiguous.dtype.str.encode(
            "utf-8"
        )
    )

    hasher.update(
        struct.pack(
            "<Q",
            contiguous.size,
        )
    )

    hasher.update(
        contiguous.tobytes()
    )


def fragment_hash(data):
    """
    SHA256 of the ordered EventNumber + tested event content.
    """

    hasher = hashlib.sha256()

    for key in (
        "EventNumber",
        "leptons_pt",
        "leptons_eta",
        "leptons_phi",
        "leptons_charge",
        "met_et",
        "met_phi",
        "weight0",
    ):
        hasher.update(
            key.encode(
                "utf-8"
            )
        )

        update_hash_array(
            hasher,
            data[
                key
            ],
        )

    return hasher.hexdigest()


def common_indices(
    left,
    right,
):
    common, left_idx, right_idx = np.intersect1d(
        left[
            "EventNumber"
        ],
        right[
            "EventNumber"
        ],
        assume_unique=True,
        return_indices=True,
    )

    return (
        common,
        left_idx,
        right_idx,
    )


def compare_fragments(
    left,
    right,
):
    common, li, ri = common_indices(
        left,
        right,
    )

    n_common = len(
        common
    )

    if n_common == 0:
        return {
            "overlap_count": 0,
            "left_overlap_fraction": 0.0,
            "right_overlap_fraction": 0.0,
            "pt_pearson": math.nan,
            "eta_pearson": math.nan,
            "met_pearson": math.nan,
            "phi_mean_cos_delta": math.nan,
            "charge_equal_fraction": math.nan,
            "weight0_exact_fraction": math.nan,
            "full_record_exact_fraction": math.nan,
        }

    l_pt = left[
        "leptons_pt"
    ][
        li
    ]

    r_pt = right[
        "leptons_pt"
    ][
        ri
    ]

    l_eta = left[
        "leptons_eta"
    ][
        li
    ]

    r_eta = right[
        "leptons_eta"
    ][
        ri
    ]

    l_phi = left[
        "leptons_phi"
    ][
        li
    ]

    r_phi = right[
        "leptons_phi"
    ][
        ri
    ]

    l_charge = left[
        "leptons_charge"
    ][
        li
    ]

    r_charge = right[
        "leptons_charge"
    ][
        ri
    ]

    l_met = left[
        "met_et"
    ][
        li
    ]

    r_met = right[
        "met_et"
    ][
        ri
    ]

    l_met_phi = left[
        "met_phi"
    ][
        li
    ]

    r_met_phi = right[
        "met_phi"
    ][
        ri
    ]

    l_weight = left[
        "weight0"
    ][
        li
    ]

    r_weight = right[
        "weight0"
    ][
        ri
    ]

    phi_delta = wrap_phi_delta(
        l_phi,
        r_phi,
    )

    charge_equal = (
        l_charge
        == r_charge
    )

    weight_equal = (
        l_weight
        == r_weight
    )

    full_exact = (
        (l_pt == r_pt)
        & (l_eta == r_eta)
        & (l_phi == r_phi)
        & (l_charge == r_charge)
        & (l_met == r_met)
        & (l_met_phi == r_met_phi)
        & (l_weight == r_weight)
    )

    return {
        "overlap_count": (
            n_common
        ),
        "left_overlap_fraction": (
            n_common
            / len(
                left[
                    "EventNumber"
                ]
            )
        ),
        "right_overlap_fraction": (
            n_common
            / len(
                right[
                    "EventNumber"
                ]
            )
        ),
        "pt_pearson": pearson(
            l_pt,
            r_pt,
        ),
        "eta_pearson": pearson(
            l_eta,
            r_eta,
        ),
        "met_pearson": pearson(
            l_met,
            r_met,
        ),
        "phi_mean_cos_delta": float(
            np.mean(
                np.cos(
                    phi_delta
                )
            )
        ),
        "charge_equal_fraction": float(
            np.mean(
                charge_equal
            )
        ),
        "weight0_exact_fraction": float(
            np.mean(
                weight_equal
            )
        ),
        "full_record_exact_fraction": float(
            np.mean(
                full_exact
            )
        ),
    }


def read_sample_set(
    mapping,
    label,
):
    result = {}

    report()
    report(
        f"Reading {label}:"
    )

    for suffix in sorted(
        mapping
    ):
        path = mapping[
            suffix
        ]

        data = read_fragment(
            path
        )

        data[
            "sha256"
        ] = fragment_hash(
            data
        )

        result[
            suffix
        ] = data

        report(
            f"  suffix {suffix:2d}: "
            f"{len(data['EventNumber']):7,d} entries, "
            f"SHA256={data['sha256'][:16]}..."
        )

    return result


def duplicate_hash_groups(
    data_by_suffix,
):
    groups = {}

    for suffix, data in (
        data_by_suffix.items()
    ):
        groups.setdefault(
            data[
                "sha256"
            ],
            [],
        ).append(
            suffix
        )

    return [
        suffixes
        for suffixes in (
            groups.values()
        )
        if len(
            suffixes
        ) > 1
    ]


def write_matrix_csv(
    path,
    row_suffixes,
    col_suffixes,
    values,
):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            [
                "particle_suffix"
            ]
            + [
                f"parton_{suffix}"
                for suffix in (
                    col_suffixes
                )
            ]
        )

        for row_index, suffix in enumerate(
            row_suffixes
        ):
            writer.writerow(
                [
                    suffix
                ]
                + [
                    values[
                        row_index,
                        col_index
                    ]
                    for col_index in range(
                        len(
                            col_suffixes
                        )
                    )
                ]
            )


def analyse_channel(
    channel,
    definition,
):
    report()
    report(
        "=" * 80
    )
    report(
        f"{definition['display']} FRAGMENT INDEPENDENCE CHECK"
    )
    report(
        "=" * 80
    )

    particle_map = discover(
        definition[
            "particle_prefix"
        ]
    )

    parton_map = discover(
        definition[
            "parton_prefix"
        ]
    )

    if not particle_map:
        raise RuntimeError(
            f"{channel}: no particle fragments found."
        )

    if not parton_map:
        raise RuntimeError(
            f"{channel}: no parton fragments found."
        )

    if set(
        particle_map
    ) != set(
        parton_map
    ):
        raise RuntimeError(
            f"{channel}: particle/parton suffix sets differ."
        )

    suffixes = sorted(
        particle_map
    )

    particle_data = read_sample_set(
        particle_map,
        f"{definition['display']} particle",
    )

    parton_data = read_sample_set(
        parton_map,
        f"{definition['display']} parton",
    )

    particle_duplicate_groups = (
        duplicate_hash_groups(
            particle_data
        )
    )

    parton_duplicate_groups = (
        duplicate_hash_groups(
            parton_data
        )
    )

    report()
    report(
        "Exact full-fragment content-hash duplicates:"
    )

    report(
        "  particle: "
        + (
            str(
                particle_duplicate_groups
            )
            if particle_duplicate_groups
            else "none"
        )
    )

    report(
        "  parton:   "
        + (
            str(
                parton_duplicate_groups
            )
            if parton_duplicate_groups
            else "none"
        )
    )

    n = len(
        suffixes
    )

    metric_names = [
        "overlap_count",
        "left_overlap_fraction",
        "right_overlap_fraction",
        "pt_pearson",
        "eta_pearson",
        "met_pearson",
        "phi_mean_cos_delta",
        "charge_equal_fraction",
        "weight0_exact_fraction",
        "full_record_exact_fraction",
    ]

    matrices = {
        name: np.full(
            (
                n,
                n,
            ),
            np.nan,
            dtype=float,
        )
        for name in (
            metric_names
        )
    }

    long_rows = []

    report()
    report(
        "Building full particle-fragment × parton-fragment matrix..."
    )

    for i, particle_suffix in enumerate(
        suffixes
    ):
        for j, parton_suffix in enumerate(
            suffixes
        ):
            comparison = (
                compare_fragments(
                    particle_data[
                        particle_suffix
                    ],
                    parton_data[
                        parton_suffix
                    ],
                )
            )

            for metric in (
                metric_names
            ):
                matrices[
                    metric
                ][
                    i,
                    j
                ] = comparison[
                    metric
                ]

            long_rows.append({
                "channel": (
                    channel
                ),
                "particle_suffix": (
                    particle_suffix
                ),
                "parton_suffix": (
                    parton_suffix
                ),
                "is_diagonal": (
                    particle_suffix
                    == parton_suffix
                ),
                **comparison,
            })

    # --------------------------------------------------------
    # Diagonal-vs-off-diagonal summaries
    # --------------------------------------------------------

    diagonal = np.eye(
        n,
        dtype=bool,
    )

    off_diagonal = ~diagonal

    report()
    report(
        "Diagonal versus off-diagonal particle/parton comparison:"
    )

    discriminating_metrics = [
        "pt_pearson",
        "met_pearson",
        "phi_mean_cos_delta",
        "charge_equal_fraction",
        "full_record_exact_fraction",
    ]

    diagonal_advantages = {}

    for metric in (
        discriminating_metrics
    ):
        matrix = matrices[
            metric
        ]

        diag_values = matrix[
            diagonal
        ]

        off_values = matrix[
            off_diagonal
        ]

        diag_mean = float(
            np.nanmean(
                diag_values
            )
        )

        off_mean = float(
            np.nanmean(
                off_values
            )
        )

        advantage = (
            diag_mean
            - off_mean
        )

        diagonal_advantages[
            metric
        ] = advantage

        report(
            f"  {metric:<28s} "
            f"diagonal mean={diag_mean: .6f}  "
            f"off-diagonal mean={off_mean: .6f}  "
            f"advantage={advantage: .6f}"
        )

    # --------------------------------------------------------
    # Within-level cross-fragment comparisons
    # --------------------------------------------------------

    def within_level_summary(
        data_by_suffix,
        level_name,
    ):
        report()
        report(
            f"Within {level_name} cross-fragment EventNumber reuse:"
        )

        overlap_fracs = []
        exact_fracs = []
        pt_corrs = []

        pair_rows = []

        for i, left_suffix in enumerate(
            suffixes
        ):
            for right_suffix in suffixes[
                i + 1:
            ]:
                comparison = (
                    compare_fragments(
                        data_by_suffix[
                            left_suffix
                        ],
                        data_by_suffix[
                            right_suffix
                        ],
                    )
                )

                overlap_fracs.append(
                    comparison[
                        "left_overlap_fraction"
                    ]
                )

                exact_fracs.append(
                    comparison[
                        "full_record_exact_fraction"
                    ]
                )

                pt_corrs.append(
                    comparison[
                        "pt_pearson"
                    ]
                )

                pair_rows.append({
                    "channel": (
                        channel
                    ),
                    "level": (
                        level_name
                    ),
                    "left_suffix": (
                        left_suffix
                    ),
                    "right_suffix": (
                        right_suffix
                    ),
                    **comparison,
                })

        report(
            f"  mean EventNumber overlap fraction: "
            f"{np.nanmean(overlap_fracs):.6f}"
        )

        report(
            f"  max  EventNumber overlap fraction: "
            f"{np.nanmax(overlap_fracs):.6f}"
        )

        report(
            f"  mean exact full-record fraction:   "
            f"{np.nanmean(exact_fracs):.6f}"
        )

        report(
            f"  max  exact full-record fraction:   "
            f"{np.nanmax(exact_fracs):.6f}"
        )

        report(
            f"  mean lepton-pT correlation:        "
            f"{np.nanmean(pt_corrs):.6f}"
        )

        report(
            f"  max  lepton-pT correlation:        "
            f"{np.nanmax(pt_corrs):.6f}"
        )

        return pair_rows

    particle_within_rows = (
        within_level_summary(
            particle_data,
            "particle",
        )
    )

    parton_within_rows = (
        within_level_summary(
            parton_data,
            "parton",
        )
    )

    # --------------------------------------------------------
    # Conservative classification
    # --------------------------------------------------------

    pt_adv = diagonal_advantages[
        "pt_pearson"
    ]

    met_adv = diagonal_advantages[
        "met_pearson"
    ]

    phi_adv = diagonal_advantages[
        "phi_mean_cos_delta"
    ]

    exact_adv = diagonal_advantages[
        "full_record_exact_fraction"
    ]

    clear_diagonal = (
        (
            math.isfinite(
                pt_adv
            )
            and pt_adv > 0.05
        )
        or (
            math.isfinite(
                met_adv
            )
            and met_adv > 0.05
        )
        or (
            math.isfinite(
                phi_adv
            )
            and phi_adv > 0.05
        )
        or (
            math.isfinite(
                exact_adv
            )
            and exact_adv > 0.05
        )
    )

    exact_duplicate_fragments = bool(
        particle_duplicate_groups
        or parton_duplicate_groups
    )

    report()
    report(
        "CHANNEL VERDICT:"
    )

    if exact_duplicate_fragments:
        verdict = (
            "DUPLICATED_FRAGMENT_CONTENT"
        )

        report(
            "  Exact duplicate fragment content was detected for "
            "the tested event fields. The nominal '20 independent "
            "fragments' interpretation requires review."
        )

    elif clear_diagonal:
        verdict = (
            "DIAGONAL_PAIRING_SUPPORTED"
        )

        report(
            "  The same-suffix particle/parton diagonal is "
            "meaningfully stronger than off-diagonal comparisons."
        )

        report(
            "  This supports suffix + EventNumber as carrying "
            "event-pairing information."
        )

    else:
        verdict = (
            "NO_DIAGONAL_PAIRING_EVIDENCE"
        )

        report(
            "  The same-suffix diagonal is not meaningfully stronger "
            "than off-diagonal fragment pairings."
        )

        report(
            "  EventNumber should therefore NOT be treated as a "
            "validated cross-level event key from these ntuples."
        )

    # --------------------------------------------------------
    # Write channel CSVs
    # --------------------------------------------------------

    long_path = (
        CSV_DIR
        / f"{channel}_particle_parton_matrix_long.csv"
    )

    pd_fields = list(
        long_rows[
            0
        ].keys()
    )

    with open(
        long_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=pd_fields,
        )
        writer.writeheader()
        writer.writerows(
            long_rows
        )

    within_path = (
        CSV_DIR
        / f"{channel}_within_level_fragment_pairs.csv"
    )

    all_within = (
        particle_within_rows
        + parton_within_rows
    )

    with open(
        within_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                all_within[
                    0
                ].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            all_within
        )

    for metric, matrix in (
        matrices.items()
    ):
        write_matrix_csv(
            (
                CSV_DIR
                / f"{channel}_{metric}_matrix.csv"
            ),
            suffixes,
            suffixes,
            matrix,
        )

    return {
        "channel": channel,
        "display": (
            definition[
                "display"
            ]
        ),
        "suffixes": (
            suffixes
        ),
        "matrices": (
            matrices
        ),
        "particle_duplicate_groups": (
            particle_duplicate_groups
        ),
        "parton_duplicate_groups": (
            parton_duplicate_groups
        ),
        "diagonal_advantages": (
            diagonal_advantages
        ),
        "verdict": verdict,
    }


def save_interactive(
    channel_results,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

    except ImportError:
        report()
        report(
            "Plotly not installed; interactive heatmap skipped."
        )
        return None

    metrics = [
        (
            "pt_pearson",
            "Lepton-pT Pearson correlation",
        ),
        (
            "met_pearson",
            "MET Pearson correlation",
        ),
        (
            "phi_mean_cos_delta",
            "Mean cos(Δφℓ)",
        ),
        (
            "full_record_exact_fraction",
            "Exact full-record fraction",
        ),
    ]

    figures = []

    for channel, result in (
        channel_results.items()
    ):
        suffixes = result[
            "suffixes"
        ]

        figure = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                title
                for _, title in (
                    metrics
                )
            ],
            horizontal_spacing=0.10,
            vertical_spacing=0.12,
        )

        for index, (
            metric,
            title,
        ) in enumerate(
            metrics
        ):
            row = (
                index // 2
                + 1
            )

            col = (
                index % 2
                + 1
            )

            z = result[
                "matrices"
            ][
                metric
            ]

            figure.add_trace(
                go.Heatmap(
                    x=[
                        str(
                            suffix
                        )
                        for suffix in suffixes
                    ],
                    y=[
                        str(
                            suffix
                        )
                        for suffix in suffixes
                    ],
                    z=z,
                    colorbar=dict(
                        title=(
                            title
                        ),
                    ),
                    hovertemplate=(
                        "particle suffix %{y}"
                        "<br>parton suffix %{x}"
                        "<br>value %{z:.6f}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        figure.update_xaxes(
            title_text="Parton fragment suffix",
        )

        figure.update_yaxes(
            title_text="Particle fragment suffix",
        )

        figure.update_layout(
            title=dict(
                text=(
                    f"{result['display']} · "
                    "particle × parton fragment matrix"
                ),
                x=0.5,
            ),
            template="plotly_white",
            height=1000,
            width=1250,
            margin=dict(
                l=90,
                r=80,
                t=100,
                b=80,
            ),
        )

        figures.append(
            (
                channel,
                figure,
                result,
            )
        )

    # One self-contained HTML with tabs.
    from plotly.offline import get_plotlyjs
    import plotly.io as pio

    plotly_js = get_plotlyjs()

    nav = ""
    sections = ""

    for index, (
        channel,
        figure,
        result,
    ) in enumerate(
        figures
    ):
        active = (
            " active"
            if index == 0
            else ""
        )

        nav += (
            f'<button class="tab-button{active}" '
            f'data-target="{channel}" type="button">'
            f"{result['display']}"
            "</button>"
        )

        div = pio.to_html(
            figure,
            include_plotlyjs=False,
            full_html=False,
            config={
                "responsive": True,
                "displaylogo": False,
            },
        )

        sections += (
            f'<section id="tab-{channel}" '
            f'class="tab-section{active}">'
            f"{div}"
            '<div class="explanation">'
            f"<strong>Automated verdict:</strong> "
            f"{result['verdict']}. "
            "A genuine suffix-based event pairing should normally "
            "appear as a visibly enhanced diagonal in these matrices. "
            "If the heatmaps show no diagonal structure, EventNumber "
            "is behaving like a reused/local identifier rather than a "
            "validated cross-file event key."
            "</div>"
            "</section>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fragment independence check</title>
<script>{plotly_js}</script>
<style>
body {{
    margin: 0;
    background: #f5f7fa;
    color: #182230;
    font-family: Arial, Helvetica, sans-serif;
}}
.page {{
    max-width: 1350px;
    margin: 0 auto;
    padding: 22px;
}}
.hero,
.plot-shell,
.explanation {{
    background: white;
    border: 1px solid #dce3eb;
    border-radius: 11px;
}}
.hero {{
    padding: 18px 20px;
    margin-bottom: 14px;
}}
.hero h1 {{
    margin: 0 0 7px 0;
    font-size: 25px;
}}
.hero p {{
    margin: 0;
    color: #5e6978;
    line-height: 1.5;
}}
.tabs {{
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
}}
.tab-button {{
    padding: 9px 13px;
    border: 1px solid #dce3eb;
    background: white;
    border-radius: 8px;
    cursor: pointer;
}}
.tab-button.active {{
    background: #e3e9ef;
    font-weight: 650;
}}
.tab-section {{
    display: none;
}}
.tab-section.active {{
    display: block;
}}
.tab-section > .plotly-graph-div {{
    background: white;
    border: 1px solid #dce3eb;
    border-radius: 11px;
}}
.explanation {{
    margin-top: 12px;
    padding: 15px 17px;
    line-height: 1.55;
    color: #354354;
}}
</style>
</head>
<body>
<div class="page">
<div class="hero">
<h1>High-stat Pythia fragment-independence diagnostic</h1>
<p>
Every particle fragment is compared with every parton fragment by
EventNumber. The diagonal corresponds to the suffix pairing previously
assumed to represent the same generated events.
</p>
</div>
<div class="tabs">{nav}</div>
{sections}
</div>
<script>
(function() {{
    const buttons = Array.from(
        document.querySelectorAll(".tab-button")
    );

    const sections = Array.from(
        document.querySelectorAll(".tab-section")
    );

    buttons.forEach(function(button) {{
        button.addEventListener(
            "click",
            function() {{
                buttons.forEach(function(other) {{
                    other.classList.toggle(
                        "active",
                        other === button
                    );
                }});

                sections.forEach(function(section) {{
                    section.classList.toggle(
                        "active",
                        section.id ===
                            "tab-" + button.dataset.target
                    );
                }});

                window.setTimeout(function() {{
                    const active = document.querySelector(
                        ".tab-section.active .plotly-graph-div"
                    );

                    if (
                        active
                        && window.Plotly
                    ) {{
                        Plotly.Plots.resize(
                            active
                        );
                    }}
                }}, 30);
            }}
        );
    }});
}})();
</script>
</body>
</html>
"""

    output_path = (
        INTERACTIVE_DIR
        / "fragment_independence_heatmaps.html"
    )

    output_path.write_text(
        html,
        encoding="utf-8",
    )

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check whether high-stat Pythia ROOT fragments are "
            "independent and whether same-suffix particle/parton "
            "pairing is uniquely supported."
        )
    )

    parser.parse_args()

    if not INPUT_DIR.is_dir():
        raise RuntimeError(
            "ROOT input directory not found:\n"
            f"  {INPUT_DIR}"
        )

    report()
    report(
        "============================================================"
    )
    report(
        " HIGH-STAT PYTHIA FRAGMENT INDEPENDENCE CHECK"
    )
    report(
        "============================================================"
    )

    results = {}

    for channel, definition in (
        CHANNELS.items()
    ):
        results[
            channel
        ] = analyse_channel(
            channel,
            definition,
        )

    interactive = (
        save_interactive(
            results
        )
    )

    report()
    report(
        "=" * 80
    )
    report(
        "GLOBAL SUMMARY"
    )
    report(
        "=" * 80
    )

    for channel, result in (
        results.items()
    ):
        report(
            f"  {result['display']}: "
            f"{result['verdict']}"
        )

    REPORT_PATH.write_text(
        "\n".join(
            REPORT_LINES
        )
        + "\n",
        encoding="utf-8",
    )

    report()
    report(
        f"Report:\n  {REPORT_PATH}"
    )

    report(
        f"CSV matrices:\n  {CSV_DIR}"
    )

    if interactive is not None:
        report(
            f"Interactive heatmaps:\n  {interactive}"
        )


if __name__ == "__main__":
    main()
