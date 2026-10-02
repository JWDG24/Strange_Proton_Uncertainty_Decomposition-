#!/usr/bin/env python3

"""
correlated_stat_uncertainty.py

Correlation-aware statistical uncertainty for the NEW high-statistics
Pythia W+c lepton-|eta| correction factors.

Motivation
----------
The high-statistics particle- and parton-level ROOT samples are not
statistically independent.  The same generated event can be identified
between the two samples using the composite key

    (fragment suffix, EventNumber)

while ROOT entry number itself is NOT a valid event-matching key.

This script keeps the FULL particle and parton samples and the same
central correction

    C_i = N_i(parton) / N_i(particle)

but estimates the statistical uncertainty while preserving the
particle/parton event correlation.

Two correlation-aware estimates are produced:

1. Analytic covariance-aware delta-method uncertainty

       Var(C_i)
         =
       Var(P_i)/D_i^2
       + P_i^2 Var(D_i)/D_i^4
       - 2 P_i Cov(P_i,D_i)/D_i^3

   where P_i is the parton yield and D_i the particle yield.

2. Paired Poisson bootstrap

   Each unique composite event key receives one Poisson(1) multiplier
   per bootstrap replica.  If an event exists on both particle and
   parton sides, THE SAME multiplier is used for both representations.
   Unmatched events naturally contribute only on the side where they
   exist.

For comparison, the script also calculates the ordinary independent
ratio uncertainty, i.e. the same formula with Cov(P_i,D_i)=0.

Important
---------
- No matched events are discarded.
- The central correction factor is NOT changed by event matching.
- This script does NOT overwrite the existing cross-check outputs.
- It is intended first as a statistical-correlation cross-check.
- The final analysis should only adopt this uncertainty after the
  EventNumber interpretation and methodology are accepted.

Expected location
-----------------
uncertainty_decomposition/
└── lepton_pseudorapidity_work_FINAL/
    └── cross_check_weight0/
        ├── correlated_stat_uncertainty.py
        └── cross_check_ROOT_files/
            ├── WCharm_WCPy8plus10.root
            ├── WCharm_WCPyPartonplus10.root
            └── ...

Outputs
-------
cross_check_weight0/
└── outputs/
    └── correlated_stat_uncertainty/
        ├── csv/
        ├── root/
        ├── interactive/
        └── run_summary.txt
"""

import argparse
import csv
import math
from array import array
from pathlib import Path

import numpy as np
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

INPUT_DIR = (
    SCRIPT_DIR
    / "cross_check_ROOT_files"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "outputs"
    / "correlated_stat_uncertainty"
)

CSV_DIR = (
    OUTPUT_DIR
    / "csv"
)

ROOT_OUTPUT_DIR = (
    OUTPUT_DIR
    / "root"
)

INTERACTIVE_DIR = (
    OUTPUT_DIR
    / "interactive"
)

for directory in (
    CSV_DIR,
    ROOT_OUTPUT_DIR,
    INTERACTIVE_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Physics selection
# ============================================================

MET_MIN_GEV = 25.0
LEPTON_PT_MIN_GEV = 20.0
LEPTON_ABS_ETA_MAX = 2.5
MTW_MIN_GEV = 40.0

JET_PT_MIN_GEV = 25.0
JET_ABS_ETA_MAX = 2.5

NOMINAL_WEIGHT_INDEX = 0


# ============================================================
# Eta binning
# ============================================================

ETA_BINS = np.array(
    [
        0.00,
        0.21,
        0.42,
        0.63,
        0.84,
        1.05,
        1.37,
        1.52,
        1.74,
        1.95,
        2.18,
        2.50,
    ],
    dtype=float,
)

N_ETA_BINS = (
    len(ETA_BINS) - 1
)

ETA_BINS_ROOT = array(
    "d",
    ETA_BINS.tolist(),
)


# ============================================================
# Sample definitions
# ============================================================

CHANNELS = {
    "Pythia_plus": {
        "display": "Pythia W+",
        "particle_prefix": (
            "WCharm_WCPy8plus"
        ),
        "parton_prefix": (
            "WCharm_WCPyPartonplus"
        ),
    },
    "Pythia_minus": {
        "display": "Pythia W-",
        "particle_prefix": (
            "WCharm_WCPy8minus"
        ),
        "parton_prefix": (
            "WCharm_WCPyPartonminus"
        ),
    },
}


# ============================================================
# Generic helpers
# ============================================================

def suffix_number(
    path,
    prefix,
):
    """
    Extract integer suffix:
        WCharm_WCPy8plus10.root -> 10
    """
    name = path.name

    expected_start = prefix

    if not (
        name.startswith(
            expected_start
        )
        and name.endswith(
            ".root"
        )
    ):
        return None

    suffix_text = name[
        len(expected_start):
        -len(".root")
    ]

    if not suffix_text.isdigit():
        return None

    return int(
        suffix_text
    )


def discover_fragments(prefix):
    """Return suffix -> ROOT path."""

    mapping = {}

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

            if suffix in mapping:
                raise RuntimeError(
                    f"Duplicate fragment suffix {suffix} "
                    f"for prefix {prefix}."
                )

            mapping[
                suffix
            ] = path

    return mapping


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
            f"Could not open ROOT file:\n"
            f"  {path}"
        )

    tree = root_file.Get(
        "WCharmTree"
    )

    if not tree:
        root_file.Close()

        raise RuntimeError(
            f"WCharmTree not found in:\n"
            f"  {path}"
        )

    return (
        root_file,
        tree,
    )


def branch_names(tree):
    return {
        branch.GetName()
        for branch in tree.GetListOfBranches()
    }


def validate_branches(
    tree,
    path,
):
    required = {
        "EventNumber",
        "met_et",
        "met_phi",
        "leptons_pt",
        "leptons_eta",
        "leptons_phi",
        "leptons_charge",
        "jet_pt",
        "jet_eta",
        "jet_charge",
        "weightvec",
    }

    missing = sorted(
        required
        - branch_names(tree)
    )

    if missing:
        raise RuntimeError(
            f"{path.name}: missing required branches: "
            + ", ".join(missing)
        )


def sequence_length(value):
    try:
        return len(value)
    except TypeError:
        return 1


def sequence_value(
    value,
    index,
):
    try:
        return value[index]
    except TypeError:

        if index != 0:
            raise IndexError(index)

        return value


# ============================================================
# Selection helpers
# ============================================================

def calculate_mtw(tree):

    delta_phi = (
        float(tree.leptons_phi)
        - float(tree.met_phi)
    )

    mt2 = (
        2.0
        * float(tree.leptons_pt)
        * float(tree.met_et)
        * (
            1.0
            - math.cos(
                delta_phi
            )
        )
    )

    return math.sqrt(
        max(
            0.0,
            mt2,
        )
    )


def selected_charm_jets(tree):
    """
    Return the fiducial charm-tagged jets.

    In these ntuples:
        jet_charge != 0
    is the charm-identification information.
    """

    n_pt = sequence_length(
        tree.jet_pt
    )

    n_eta = sequence_length(
        tree.jet_eta
    )

    n_charge = sequence_length(
        tree.jet_charge
    )

    if not (
        n_pt == n_eta == n_charge
    ):
        raise RuntimeError(
            "jet_pt / jet_eta / jet_charge "
            "length mismatch."
        )

    selected = []

    for jet_index in range(
        n_pt
    ):

        jet_pt = float(
            sequence_value(
                tree.jet_pt,
                jet_index,
            )
        )

        jet_eta = float(
            sequence_value(
                tree.jet_eta,
                jet_index,
            )
        )

        jet_charge = float(
            sequence_value(
                tree.jet_charge,
                jet_index,
            )
        )

        if (
            jet_pt
            <= JET_PT_MIN_GEV
        ):
            continue

        if (
            abs(jet_eta)
            >= JET_ABS_ETA_MAX
        ):
            continue

        if jet_charge == 0.0:
            continue

        charm_sign = (
            1.0
            if jet_charge > 0.0
            else -1.0
        )

        selected.append(
            (
                jet_index,
                charm_sign,
            )
        )

    return selected


def os_ss_factor(
    lepton_charge,
    charm_sign,
):

    product = (
        float(lepton_charge)
        * float(charm_sign)
    )

    return (
        -1.0
        if product > 0.0
        else 1.0
    )


def eta_bin(abs_eta):

    index = int(
        np.searchsorted(
            ETA_BINS,
            abs_eta,
            side="right",
        )
        - 1
    )

    if (
        index < 0
        or index >= N_ETA_BINS
    ):
        return -1

    return index


def selected_event_contribution(
    tree,
    normalization,
):
    """
    Return:
        (bin_index, final_weight)

    or None if the event fails the complete repaired selection.
    """

    if (
        float(tree.met_et)
        <= MET_MIN_GEV
    ):
        return None

    if (
        float(tree.leptons_pt)
        <= LEPTON_PT_MIN_GEV
    ):
        return None

    lepton_eta = float(
        tree.leptons_eta
    )

    if (
        abs(lepton_eta)
        >= LEPTON_ABS_ETA_MAX
    ):
        return None

    if (
        calculate_mtw(tree)
        <= MTW_MIN_GEV
    ):
        return None

    charm_jets = (
        selected_charm_jets(
            tree
        )
    )

    if len(
        charm_jets
    ) != 1:
        return None

    _, charm_sign = (
        charm_jets[0]
    )

    factor = (
        os_ss_factor(
            tree.leptons_charge,
            charm_sign,
        )
    )

    if (
        len(tree.weightvec)
        <= NOMINAL_WEIGHT_INDEX
    ):
        raise RuntimeError(
            "weightvec[0] is unavailable."
        )

    generator_weight = float(
        tree.weightvec[
            NOMINAL_WEIGHT_INDEX
        ]
    )

    final_weight = (
        factor
        * generator_weight
        * normalization
    )

    bin_index = eta_bin(
        abs(
            lepton_eta
        )
    )

    if bin_index < 0:
        return None

    return (
        bin_index,
        final_weight,
    )


# ============================================================
# Entry counts
# ============================================================

def total_entries(
    fragment_map,
):

    total = 0

    for suffix in sorted(
        fragment_map
    ):

        root_file, tree = (
            open_tree(
                fragment_map[
                    suffix
                ]
            )
        )

        total += int(
            tree.GetEntries()
        )

        root_file.Close()

    return total


# ============================================================
# Read one side of one fragment
# ============================================================

def read_fragment(
    path,
    normalization,
):
    """
    Read one ROOT fragment.

    Returns
    -------
    all_event_numbers : set[int]
        Every raw EventNumber in the fragment.

    selected : dict[int, tuple[int, float]]
        EventNumber -> (eta_bin, signed normalized event weight)
        for events passing the complete repaired selection.

    counters : dict
        Basic selection/OS-SS diagnostics.
    """

    root_file, tree = (
        open_tree(
            path
        )
    )

    validate_branches(
        tree,
        path,
    )

    all_event_numbers = set()

    selected = {}

    counters = {
        "entries": 0,
        "selected": 0,
        "os": 0,
        "ss": 0,
    }

    nentries = int(
        tree.GetEntries()
    )

    counters[
        "entries"
    ] = nentries

    for ientry in range(
        nentries
    ):

        tree.GetEntry(
            ientry
        )

        event_number = int(
            tree.EventNumber
        )

        if (
            event_number
            in all_event_numbers
        ):
            root_file.Close()

            raise RuntimeError(
                f"{path.name}: duplicate EventNumber "
                f"{event_number} within one fragment."
            )

        all_event_numbers.add(
            event_number
        )

        contribution = (
            selected_event_contribution(
                tree,
                normalization,
            )
        )

        if contribution is None:
            continue

        if event_number in selected:
            root_file.Close()

            raise RuntimeError(
                f"{path.name}: duplicate selected EventNumber "
                f"{event_number}."
            )

        bin_index, weight = (
            contribution
        )

        selected[
            event_number
        ] = (
            bin_index,
            weight,
        )

        counters[
            "selected"
        ] += 1

        if weight >= 0.0:
            counters[
                "os"
            ] += 1
        else:
            counters[
                "ss"
            ] += 1

    root_file.Close()

    return (
        all_event_numbers,
        selected,
        counters,
    )


# ============================================================
# Build full channel event representation
# ============================================================

def build_channel_arrays(
    pair_label,
    definition,
):
    """
    Construct one row per unique composite event key.

    For every key we store a possible particle contribution and a
    possible parton contribution.  Matched keys therefore share one
    row and will receive the same Poisson multiplier in bootstrap
    replicas.
    """

    particle_fragments = (
        discover_fragments(
            definition[
                "particle_prefix"
            ]
        )
    )

    parton_fragments = (
        discover_fragments(
            definition[
                "parton_prefix"
            ]
        )
    )

    common_suffixes = sorted(
        set(
            particle_fragments
        )
        & set(
            parton_fragments
        )
    )

    if not common_suffixes:
        raise RuntimeError(
            f"{pair_label}: no paired fragments found."
        )

    if (
        set(
            particle_fragments
        )
        != set(
            parton_fragments
        )
    ):
        raise RuntimeError(
            f"{pair_label}: particle and parton fragment "
            "suffix sets differ."
        )

    particle_total_entries = (
        total_entries(
            particle_fragments
        )
    )

    parton_total_entries = (
        total_entries(
            parton_fragments
        )
    )

    if (
        particle_total_entries <= 0
        or parton_total_entries <= 0
    ):
        raise RuntimeError(
            f"{pair_label}: empty input sample."
        )

    particle_norm = (
        1.0
        / particle_total_entries
    )

    parton_norm = (
        1.0
        / parton_total_entries
    )

    # One row per selected event in the union of the two sides.
    particle_bins = []
    particle_weights = []

    parton_bins = []
    parton_weights = []

    diagnostics = {
        "particle_total_entries": (
            particle_total_entries
        ),
        "parton_total_entries": (
            parton_total_entries
        ),
        "raw_matched_keys": 0,
        "raw_particle_keys": 0,
        "raw_parton_keys": 0,
        "particle_selected": 0,
        "parton_selected": 0,
        "selected_on_both": 0,
        "selected_same_eta_bin": 0,
        "selected_different_eta_bin": 0,
        "particle_os": 0,
        "particle_ss": 0,
        "parton_os": 0,
        "parton_ss": 0,
    }

    print()
    print(
        "=" * 72
    )
    print(
        f"{definition['display']}"
    )
    print(
        "=" * 72
    )
    print(
        f"Particle entries: {particle_total_entries:,}"
    )
    print(
        f"Parton entries:   {parton_total_entries:,}"
    )
    print(
        f"Paired fragments: {len(common_suffixes)}"
    )

    for suffix in (
        common_suffixes
    ):

        (
            particle_all_ids,
            particle_selected,
            particle_counts,
        ) = read_fragment(
            particle_fragments[
                suffix
            ],
            particle_norm,
        )

        (
            parton_all_ids,
            parton_selected,
            parton_counts,
        ) = read_fragment(
            parton_fragments[
                suffix
            ],
            parton_norm,
        )

        raw_overlap = (
            particle_all_ids
            & parton_all_ids
        )

        diagnostics[
            "raw_particle_keys"
        ] += len(
            particle_all_ids
        )

        diagnostics[
            "raw_parton_keys"
        ] += len(
            parton_all_ids
        )

        diagnostics[
            "raw_matched_keys"
        ] += len(
            raw_overlap
        )

        diagnostics[
            "particle_selected"
        ] += particle_counts[
            "selected"
        ]

        diagnostics[
            "parton_selected"
        ] += parton_counts[
            "selected"
        ]

        diagnostics[
            "particle_os"
        ] += particle_counts[
            "os"
        ]

        diagnostics[
            "particle_ss"
        ] += particle_counts[
            "ss"
        ]

        diagnostics[
            "parton_os"
        ] += parton_counts[
            "os"
        ]

        diagnostics[
            "parton_ss"
        ] += parton_counts[
            "ss"
        ]

        selected_union_ids = (
            set(
                particle_selected
            )
            | set(
                parton_selected
            )
        )

        for event_number in (
            selected_union_ids
        ):

            p_info = (
                particle_selected.get(
                    event_number
                )
            )

            a_info = (
                parton_selected.get(
                    event_number
                )
            )

            if p_info is None:
                p_bin = -1
                p_weight = 0.0
            else:
                p_bin, p_weight = (
                    p_info
                )

            if a_info is None:
                a_bin = -1
                a_weight = 0.0
            else:
                a_bin, a_weight = (
                    a_info
                )

            if (
                p_info is not None
                and a_info is not None
            ):

                diagnostics[
                    "selected_on_both"
                ] += 1

                if p_bin == a_bin:

                    diagnostics[
                        "selected_same_eta_bin"
                    ] += 1

                else:

                    diagnostics[
                        "selected_different_eta_bin"
                    ] += 1

            particle_bins.append(
                p_bin
            )

            particle_weights.append(
                p_weight
            )

            parton_bins.append(
                a_bin
            )

            parton_weights.append(
                a_weight
            )

        print(
            f"  suffix {suffix:2d}: "
            f"raw overlap={len(raw_overlap):6d}, "
            f"selected particle={particle_counts['selected']:6d}, "
            f"parton={parton_counts['selected']:6d}"
        )

    particle_bins = np.asarray(
        particle_bins,
        dtype=np.int16,
    )

    particle_weights = np.asarray(
        particle_weights,
        dtype=np.float64,
    )

    parton_bins = np.asarray(
        parton_bins,
        dtype=np.int16,
    )

    parton_weights = np.asarray(
        parton_weights,
        dtype=np.float64,
    )

    if not (
        len(particle_bins)
        == len(parton_bins)
        == len(particle_weights)
        == len(parton_weights)
    ):
        raise RuntimeError(
            "Internal event-union array length mismatch."
        )

    diagnostics[
        "selected_union_size"
    ] = len(
        particle_bins
    )

    particle_match_fraction = (
        diagnostics[
            "raw_matched_keys"
        ]
        / diagnostics[
            "raw_particle_keys"
        ]
    )

    parton_match_fraction = (
        diagnostics[
            "raw_matched_keys"
        ]
        / diagnostics[
            "raw_parton_keys"
        ]
    )

    print()
    print(
        "Raw event matching:"
    )
    print(
        f"  matched composite keys: "
        f"{diagnostics['raw_matched_keys']:,}"
    )
    print(
        f"  particle matched fraction: "
        f"{100.0 * particle_match_fraction:.4f}%"
    )
    print(
        f"  parton matched fraction:   "
        f"{100.0 * parton_match_fraction:.4f}%"
    )
    print()
    print(
        "After the complete W+c selection:"
    )
    print(
        f"  particle selected: "
        f"{diagnostics['particle_selected']:,}"
    )
    print(
        f"  parton selected:   "
        f"{diagnostics['parton_selected']:,}"
    )
    print(
        f"  selected on both:  "
        f"{diagnostics['selected_on_both']:,}"
    )
    print(
        f"  same eta bin:      "
        f"{diagnostics['selected_same_eta_bin']:,}"
    )
    print(
        f"  different eta bin: "
        f"{diagnostics['selected_different_eta_bin']:,}"
    )

    return {
        "particle_bins": (
            particle_bins
        ),
        "particle_weights": (
            particle_weights
        ),
        "parton_bins": (
            parton_bins
        ),
        "parton_weights": (
            parton_weights
        ),
        "diagnostics": (
            diagnostics
        ),
    }


# ============================================================
# Nominal yields and covariance
# ============================================================

def binned_sum(
    bins,
    weights,
):
    mask = (
        bins >= 0
    )

    return np.bincount(
        bins[
            mask
        ],
        weights=weights[
            mask
        ],
        minlength=N_ETA_BINS,
    ).astype(
        float
    )


def binned_sumw2(
    bins,
    weights,
):
    mask = (
        bins >= 0
    )

    return np.bincount(
        bins[
            mask
        ],
        weights=(
            weights[
                mask
            ]
            ** 2
        ),
        minlength=N_ETA_BINS,
    ).astype(
        float
    )


def same_bin_covariance(
    particle_bins,
    particle_weights,
    parton_bins,
    parton_weights,
):
    """
    Cov(P_i, D_i) under shared Poisson event resampling.

    Only a matched event contributing to bin i on BOTH sides contributes
    to the same-bin covariance term for C_i.
    """

    mask = (
        (particle_bins >= 0)
        & (parton_bins >= 0)
        & (
            particle_bins
            == parton_bins
        )
    )

    return np.bincount(
        particle_bins[
            mask
        ],
        weights=(
            particle_weights[
                mask
            ]
            * parton_weights[
                mask
            ]
        ),
        minlength=N_ETA_BINS,
    ).astype(
        float
    )


def safe_ratio(
    numerator,
    denominator,
):
    result = np.full(
        N_ETA_BINS,
        np.nan,
        dtype=float,
    )

    mask = (
        denominator != 0.0
    )

    result[
        mask
    ] = (
        numerator[
            mask
        ]
        / denominator[
            mask
        ]
    )

    return result


def uncertainty_calculations(
    channel_arrays,
):
    """
    Compute nominal correction, ordinary independent uncertainty and
    covariance-aware delta-method uncertainty.
    """

    particle_bins = (
        channel_arrays[
            "particle_bins"
        ]
    )

    particle_weights = (
        channel_arrays[
            "particle_weights"
        ]
    )

    parton_bins = (
        channel_arrays[
            "parton_bins"
        ]
    )

    parton_weights = (
        channel_arrays[
            "parton_weights"
        ]
    )

    denominator = binned_sum(
        particle_bins,
        particle_weights,
    )

    numerator = binned_sum(
        parton_bins,
        parton_weights,
    )

    var_denominator = binned_sumw2(
        particle_bins,
        particle_weights,
    )

    var_numerator = binned_sumw2(
        parton_bins,
        parton_weights,
    )

    covariance = (
        same_bin_covariance(
            particle_bins,
            particle_weights,
            parton_bins,
            parton_weights,
        )
    )

    correction = safe_ratio(
        numerator,
        denominator,
    )

    independent_variance = (
        np.full(
            N_ETA_BINS,
            np.nan,
            dtype=float,
        )
    )

    correlated_variance = (
        np.full(
            N_ETA_BINS,
            np.nan,
            dtype=float,
        )
    )

    correlation_coefficient = (
        np.zeros(
            N_ETA_BINS,
            dtype=float,
        )
    )

    for ibin in range(
        N_ETA_BINS
    ):

        P = numerator[
            ibin
        ]

        D = denominator[
            ibin
        ]

        var_P = var_numerator[
            ibin
        ]

        var_D = var_denominator[
            ibin
        ]

        cov_PD = covariance[
            ibin
        ]

        if D == 0.0:
            continue

        independent_variance[
            ibin
        ] = (
            var_P
            / (
                D * D
            )
            + (
                P * P
                * var_D
                / (
                    D
                    ** 4
                )
            )
        )

        correlated_variance_raw = (
            independent_variance[
                ibin
            ]
            - (
                2.0
                * P
                * cov_PD
                / (
                    D
                    ** 3
                )
            )
        )

        # Tiny negative values from floating-point cancellation should
        # not become NaN uncertainties.
        correlated_variance[
            ibin
        ] = max(
            0.0,
            correlated_variance_raw,
        )

        denominator_rho = math.sqrt(
            max(
                0.0,
                var_P
                * var_D,
            )
        )

        if denominator_rho > 0.0:

            correlation_coefficient[
                ibin
            ] = (
                cov_PD
                / denominator_rho
            )

    independent_unc = np.sqrt(
        independent_variance
    )

    correlated_delta_unc = np.sqrt(
        correlated_variance
    )

    return {
        "particle_yield": denominator,
        "parton_yield": numerator,
        "particle_variance": (
            var_denominator
        ),
        "parton_variance": (
            var_numerator
        ),
        "covariance": covariance,
        "correlation_coefficient": (
            correlation_coefficient
        ),
        "correction": correction,
        "independent_unc": (
            independent_unc
        ),
        "correlated_delta_unc": (
            correlated_delta_unc
        ),
    }


# ============================================================
# Paired Poisson bootstrap
# ============================================================

def paired_bootstrap(
    channel_arrays,
    replicas,
    seed,
):
    """
    Correlation-aware paired Poisson bootstrap.

    One Poisson multiplier is drawn per unique composite event key.
    Therefore matched particle/parton representations fluctuate
    together automatically.
    """

    if replicas <= 1:
        raise ValueError(
            "Bootstrap replicas must be > 1."
        )

    particle_bins = (
        channel_arrays[
            "particle_bins"
        ]
    )

    particle_weights = (
        channel_arrays[
            "particle_weights"
        ]
    )

    parton_bins = (
        channel_arrays[
            "parton_bins"
        ]
    )

    parton_weights = (
        channel_arrays[
            "parton_weights"
        ]
    )

    n_union = len(
        particle_bins
    )

    rng = np.random.default_rng(
        seed
    )

    boot_corrections = np.full(
        (
            replicas,
            N_ETA_BINS,
        ),
        np.nan,
        dtype=float,
    )

    particle_mask = (
        particle_bins >= 0
    )

    parton_mask = (
        parton_bins >= 0
    )

    print()
    print(
        f"Running {replicas} paired Poisson bootstrap replicas "
        f"over {n_union:,} selected-union event keys..."
    )

    progress_step = max(
        1,
        replicas // 10,
    )

    for irep in range(
        replicas
    ):

        multiplicity = rng.poisson(
            1.0,
            size=n_union,
        )

        particle_yield = np.bincount(
            particle_bins[
                particle_mask
            ],
            weights=(
                particle_weights[
                    particle_mask
                ]
                * multiplicity[
                    particle_mask
                ]
            ),
            minlength=N_ETA_BINS,
        )

        parton_yield = np.bincount(
            parton_bins[
                parton_mask
            ],
            weights=(
                parton_weights[
                    parton_mask
                ]
                * multiplicity[
                    parton_mask
                ]
            ),
            minlength=N_ETA_BINS,
        )

        boot_corrections[
            irep
        ] = safe_ratio(
            parton_yield,
            particle_yield,
        )

        if (
            (irep + 1)
            % progress_step == 0
            or irep == 0
            or irep + 1 == replicas
        ):

            print(
                f"  replica "
                f"{irep + 1:4d}/{replicas}"
            )

    bootstrap_unc = np.nanstd(
        boot_corrections,
        axis=0,
        ddof=1,
    )

    bootstrap_mean = np.nanmean(
        boot_corrections,
        axis=0,
    )

    return {
        "bootstrap_unc": (
            bootstrap_unc
        ),
        "bootstrap_mean": (
            bootstrap_mean
        ),
        "boot_corrections": (
            boot_corrections
        ),
    }


# ============================================================
# CSV output
# ============================================================

CSV_COLUMNS = [
    "pair_label",
    "bin",
    "bin_low_edge",
    "bin_up_edge",
    "correction_factor",
    "particle_yield",
    "parton_yield",
    "stat_unc_independent",
    "stat_unc_correlated_delta",
    "stat_unc_paired_bootstrap",
    "bootstrap_mean_correction",
    "parton_particle_covariance",
    "parton_particle_correlation_coefficient",
    "bootstrap_over_independent",
    "delta_over_independent",
    "bootstrap_change_vs_independent_percent",
    "delta_change_vs_independent_percent",
]


def build_rows(
    pair_label,
    calculation,
    bootstrap,
):

    rows = []

    for ibin in range(
        N_ETA_BINS
    ):

        independent = float(
            calculation[
                "independent_unc"
            ][
                ibin
            ]
        )

        correlated_delta = float(
            calculation[
                "correlated_delta_unc"
            ][
                ibin
            ]
        )

        paired_bootstrap_unc = float(
            bootstrap[
                "bootstrap_unc"
            ][
                ibin
            ]
        )

        bootstrap_ratio = (
            paired_bootstrap_unc
            / independent
            if (
                math.isfinite(
                    independent
                )
                and independent > 0.0
            )
            else math.nan
        )

        delta_ratio = (
            correlated_delta
            / independent
            if (
                math.isfinite(
                    independent
                )
                and independent > 0.0
            )
            else math.nan
        )

        rows.append({
            "pair_label": (
                pair_label
            ),
            "bin": (
                ibin + 1
            ),
            "bin_low_edge": float(
                ETA_BINS[
                    ibin
                ]
            ),
            "bin_up_edge": float(
                ETA_BINS[
                    ibin + 1
                ]
            ),
            "correction_factor": float(
                calculation[
                    "correction"
                ][
                    ibin
                ]
            ),
            "particle_yield": float(
                calculation[
                    "particle_yield"
                ][
                    ibin
                ]
            ),
            "parton_yield": float(
                calculation[
                    "parton_yield"
                ][
                    ibin
                ]
            ),
            "stat_unc_independent": (
                independent
            ),
            "stat_unc_correlated_delta": (
                correlated_delta
            ),
            "stat_unc_paired_bootstrap": (
                paired_bootstrap_unc
            ),
            "bootstrap_mean_correction": float(
                bootstrap[
                    "bootstrap_mean"
                ][
                    ibin
                ]
            ),
            "parton_particle_covariance": float(
                calculation[
                    "covariance"
                ][
                    ibin
                ]
            ),
            "parton_particle_correlation_coefficient": float(
                calculation[
                    "correlation_coefficient"
                ][
                    ibin
                ]
            ),
            "bootstrap_over_independent": (
                bootstrap_ratio
            ),
            "delta_over_independent": (
                delta_ratio
            ),
            "bootstrap_change_vs_independent_percent": (
                100.0
                * (
                    bootstrap_ratio
                    - 1.0
                )
                if math.isfinite(
                    bootstrap_ratio
                )
                else math.nan
            ),
            "delta_change_vs_independent_percent": (
                100.0
                * (
                    delta_ratio
                    - 1.0
                )
                if math.isfinite(
                    delta_ratio
                )
                else math.nan
            ),
        })

    return rows


def write_csv(
    pair_label,
    rows,
):

    output_path = (
        CSV_DIR
        / (
            f"{pair_label}_"
            f"correlated_stat_uncertainty.csv"
        )
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    return output_path


# ============================================================
# ROOT output
# ============================================================

def make_histogram(
    name,
    title,
    values,
    errors=None,
):

    histogram = ROOT.TH1D(
        name,
        title,
        N_ETA_BINS,
        ETA_BINS_ROOT,
    )

    histogram.SetDirectory(0)

    for ibin in range(
        N_ETA_BINS
    ):

        histogram.SetBinContent(
            ibin + 1,
            float(
                values[
                    ibin
                ]
            ),
        )

        if errors is not None:

            histogram.SetBinError(
                ibin + 1,
                float(
                    errors[
                        ibin
                    ]
                ),
            )

    return histogram


def write_root(
    pair_label,
    calculation,
    bootstrap,
    diagnostics,
    replicas,
):

    output_path = (
        ROOT_OUTPUT_DIR
        / (
            f"{pair_label}_"
            f"correlated_stat_uncertainty.root"
        )
    )

    output_file = ROOT.TFile(
        str(output_path),
        "RECREATE",
    )

    ROOT.TNamed(
        "PairLabel",
        pair_label,
    ).Write()

    ROOT.TNamed(
        "MatchingKey",
        "(fragment suffix, EventNumber)",
    ).Write()

    ROOT.TNamed(
        "StatisticalMethod",
        (
            "paired Poisson bootstrap with one multiplier per "
            "composite event key; analytic covariance-aware "
            "delta method also stored"
        ),
    ).Write()

    ROOT.TNamed(
        "BootstrapReplicas",
        str(
            replicas
        ),
    ).Write()

    ROOT.TNamed(
        "EventSelection",
        (
            "MET>25; lepton pT>20; abs(eta_l)<2.5; "
            "mTW>40; jet pT>25; abs(eta_jet)<2.5; "
            "exactly one jet with jet_charge!=0; OS=+1, SS=-1"
        ),
    ).Write()

    correction_independent = (
        make_histogram(
            (
                "correction_with_"
                "independent_stat"
            ),
            (
                ";|#eta_{#ell}|;"
                "C = N_{parton}/N_{particle}"
            ),
            calculation[
                "correction"
            ],
            calculation[
                "independent_unc"
            ],
        )
    )

    correction_correlated = (
        make_histogram(
            (
                "correction_with_"
                "paired_bootstrap_stat"
            ),
            (
                ";|#eta_{#ell}|;"
                "C = N_{parton}/N_{particle}"
            ),
            calculation[
                "correction"
            ],
            bootstrap[
                "bootstrap_unc"
            ],
        )
    )

    independent_unc_hist = (
        make_histogram(
            "stat_unc_independent",
            (
                ";|#eta_{#ell}|;"
                "#sigma_{stat}"
            ),
            calculation[
                "independent_unc"
            ],
        )
    )

    correlated_delta_hist = (
        make_histogram(
            "stat_unc_correlated_delta",
            (
                ";|#eta_{#ell}|;"
                "#sigma_{stat}"
            ),
            calculation[
                "correlated_delta_unc"
            ],
        )
    )

    correlated_bootstrap_hist = (
        make_histogram(
            "stat_unc_paired_bootstrap",
            (
                ";|#eta_{#ell}|;"
                "#sigma_{stat}"
            ),
            bootstrap[
                "bootstrap_unc"
            ],
        )
    )

    covariance_hist = (
        make_histogram(
            "parton_particle_covariance",
            (
                ";|#eta_{#ell}|;"
                "Cov(P,D)"
            ),
            calculation[
                "covariance"
            ],
        )
    )

    rho_hist = (
        make_histogram(
            "parton_particle_correlation_coefficient",
            (
                ";|#eta_{#ell}|;"
                "#rho(P,D)"
            ),
            calculation[
                "correlation_coefficient"
            ],
        )
    )

    for histogram in (
        correction_independent,
        correction_correlated,
        independent_unc_hist,
        correlated_delta_hist,
        correlated_bootstrap_hist,
        covariance_hist,
        rho_hist,
    ):

        histogram.Write()

    for key, value in (
        diagnostics.items()
    ):

        ROOT.TNamed(
            f"diagnostic_{key}",
            str(value),
        ).Write()

    output_file.Write()
    output_file.Close()

    return output_path


# ============================================================
# Interactive Plotly output
# ============================================================

def save_interactive(
    results,
):

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

    except ImportError:

        print()
        print(
            "Plotly is not installed; "
            "interactive HTML will be skipped."
        )

        return None

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Correction factor",
            "Statistical uncertainty",
            "Correlation-aware / independent uncertainty",
        ),
        row_heights=[
            0.42,
            0.36,
            0.22,
        ],
    )

    pair_labels = list(
        results.keys()
    )

    traces_per_pair = 6

    for pair_index, pair_label in enumerate(
        pair_labels
    ):

        result = (
            results[
                pair_label
            ]
        )

        rows = result[
            "rows"
        ]

        display = result[
            "display"
        ]

        visible = (
            pair_index == 0
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

        correction = [
            row[
                "correction_factor"
            ]
            for row in rows
        ]

        independent = [
            row[
                "stat_unc_independent"
            ]
            for row in rows
        ]

        paired = [
            row[
                "stat_unc_paired_bootstrap"
            ]
            for row in rows
        ]

        delta = [
            row[
                "stat_unc_correlated_delta"
            ]
            for row in rows
        ]

        ratio = [
            row[
                "bootstrap_over_independent"
            ]
            for row in rows
        ]

        # ----------------------------------------------------
        # Top: same central correction with old/new stat bars
        # ----------------------------------------------------

        figure.add_trace(
            go.Scatter(
                x=x,
                y=correction,
                error_x=dict(
                    type="data",
                    array=xerr,
                    visible=True,
                ),
                error_y=dict(
                    type="data",
                    array=independent,
                    visible=True,
                ),
                mode="lines+markers",
                name=(
                    "Independent-stat treatment"
                ),
                visible=visible,
                hovertemplate=(
                    "|eta_l| = %{x:.3f}"
                    "<br>C = %{y:.6f}"
                    "<br>independent stat = "
                    "%{error_y.array:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=x,
                y=correction,
                error_y=dict(
                    type="data",
                    array=paired,
                    visible=True,
                ),
                mode="markers",
                name=(
                    "Paired-bootstrap stat"
                ),
                visible=visible,
                hovertemplate=(
                    "|eta_l| = %{x:.3f}"
                    "<br>C = %{y:.6f}"
                    "<br>paired stat = "
                    "%{error_y.array:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        # ----------------------------------------------------
        # Middle: absolute statistical uncertainties
        # ----------------------------------------------------

        figure.add_trace(
            go.Scatter(
                x=x,
                y=independent,
                mode="lines+markers",
                name="Independent stat",
                visible=visible,
                hovertemplate=(
                    "|eta_l| = %{x:.3f}"
                    "<br>sigma = %{y:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=x,
                y=paired,
                mode="lines+markers",
                name="Paired bootstrap",
                visible=visible,
                hovertemplate=(
                    "|eta_l| = %{x:.3f}"
                    "<br>sigma = %{y:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=x,
                y=delta,
                mode="lines",
                name=(
                    "Covariance delta method"
                ),
                visible=visible,
                hovertemplate=(
                    "|eta_l| = %{x:.3f}"
                    "<br>sigma = %{y:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

        # ----------------------------------------------------
        # Bottom: paired / independent
        # ----------------------------------------------------

        figure.add_trace(
            go.Scatter(
                x=x,
                y=ratio,
                mode="lines+markers",
                name=(
                    "Paired / independent"
                ),
                visible=visible,
                hovertemplate=(
                    "|eta_l| = %{x:.3f}"
                    "<br>ratio = %{y:.4f}"
                    "<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )

    total_traces = (
        traces_per_pair
        * len(
            pair_labels
        )
    )

    buttons = []

    for pair_index, pair_label in enumerate(
        pair_labels
    ):

        mask = [
            False
        ] * total_traces

        first = (
            pair_index
            * traces_per_pair
        )

        for trace_index in range(
            first,
            first + traces_per_pair,
        ):

            mask[
                trace_index
            ] = True

        display = (
            results[
                pair_label
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
                        "visible": mask,
                    },
                    {
                        "title.text": (
                            "Event-matched statistical "
                            "uncertainty"
                            f"<br><sup>{display}</sup>"
                        ),
                    },
                ],
            )
        )

    first_display = (
        results[
            pair_labels[
                0
            ]
        ][
            "display"
        ]
    )

    figure.update_layout(
        title=dict(
            text=(
                "Event-matched statistical uncertainty"
                f"<br><sup>{first_display}</sup>"
            ),
            x=0.5,
        ),
        template="plotly_white",
        height=1050,
        width=1150,
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
                y=1.14,
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
        title_text="Statistical uncertainty",
        row=2,
        col=1,
        rangemode="tozero",
    )

    figure.update_yaxes(
        title_text="sigma paired / sigma independent",
        row=3,
        col=1,
    )

    figure.add_hline(
        y=1.0,
        line_dash="dash",
        row=3,
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
        range=[
            0.0,
            2.5,
        ],
        row=2,
        col=1,
    )

    figure.update_xaxes(
        title_text="Lepton |eta|",
        range=[
            0.0,
            2.5,
        ],
        row=3,
        col=1,
    )

    output_path = (
        INTERACTIVE_DIR
        / (
            "correlated_stat_uncertainty_"
            "interactive.html"
        )
    )

    figure.write_html(
        str(output_path),
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
    )

    return output_path


# ============================================================
# Summary output
# ============================================================

def write_summary(
    results,
    replicas,
    seed,
):

    output_path = (
        OUTPUT_DIR
        / "run_summary.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as summary:

        summary.write(
            "Correlation-aware statistical uncertainty study\n"
        )

        summary.write(
            "===============================================\n\n"
        )

        summary.write(
            "Event matching key:\n"
        )

        summary.write(
            "  (fragment suffix, EventNumber)\n\n"
        )

        summary.write(
            "Central correction:\n"
        )

        summary.write(
            "  unchanged; all selected particle and parton "
            "events are retained\n\n"
        )

        summary.write(
            "Statistical methods:\n"
        )

        summary.write(
            "  independent ratio uncertainty\n"
        )

        summary.write(
            "  covariance-aware delta method\n"
        )

        summary.write(
            "  paired Poisson bootstrap\n\n"
        )

        summary.write(
            f"Bootstrap replicas: {replicas}\n"
        )

        summary.write(
            f"Bootstrap seed: {seed}\n\n"
        )

        for pair_label, result in (
            results.items()
        ):

            diagnostics = (
                result[
                    "diagnostics"
                ]
            )

            summary.write(
                f"{pair_label}\n"
            )

            summary.write(
                "-" * len(
                    pair_label
                )
                + "\n"
            )

            for key in (
                "particle_total_entries",
                "parton_total_entries",
                "raw_matched_keys",
                "particle_selected",
                "parton_selected",
                "selected_on_both",
                "selected_same_eta_bin",
                "selected_different_eta_bin",
                "particle_os",
                "particle_ss",
                "parton_os",
                "parton_ss",
            ):

                summary.write(
                    f"  {key}: "
                    f"{diagnostics[key]}\n"
                )

            summary.write(
                "\n"
            )

    return output_path


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Calculate correlation-aware statistical "
            "uncertainties for the high-stat Pythia correction."
        )
    )

    parser.add_argument(
        "--bootstrap-replicas",
        type=int,
        default=300,
        help=(
            "Number of paired Poisson bootstrap replicas "
            "(default: 300)."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help=(
            "Random seed for the paired bootstrap "
            "(default: 12345)."
        ),
    )

    args = parser.parse_args()

    if not INPUT_DIR.is_dir():
        raise RuntimeError(
            "Input ROOT directory does not exist:\n"
            f"  {INPUT_DIR}"
        )

    if (
        args.bootstrap_replicas
        <= 1
    ):
        raise RuntimeError(
            "--bootstrap-replicas must be > 1."
        )

    print()
    print(
        "============================================================"
    )
    print(
        " Event-matched statistical uncertainty study"
    )
    print(
        "============================================================"
    )
    print()

    print(
        "Matching key:"
    )
    print(
        "  (fragment suffix, EventNumber)"
    )
    print()

    print(
        "Central correction:"
    )
    print(
        "  uses ALL selected events; no matched-event subset cut"
    )
    print()

    print(
        "Methods:"
    )
    print(
        "  1. independent ratio error"
    )
    print(
        "  2. covariance-aware delta method"
    )
    print(
        "  3. paired Poisson bootstrap"
    )
    print()

    results = {}

    for pair_index, (
        pair_label,
        definition,
    ) in enumerate(
        CHANNELS.items()
    ):

        channel_arrays = (
            build_channel_arrays(
                pair_label,
                definition,
            )
        )

        calculation = (
            uncertainty_calculations(
                channel_arrays
            )
        )

        # Offset seed between charge channels while remaining
        # fully reproducible.
        bootstrap = (
            paired_bootstrap(
                channel_arrays=(
                    channel_arrays
                ),
                replicas=(
                    args.bootstrap_replicas
                ),
                seed=(
                    args.seed
                    + pair_index
                ),
            )
        )

        rows = (
            build_rows(
                pair_label,
                calculation,
                bootstrap,
            )
        )

        csv_path = (
            write_csv(
                pair_label,
                rows,
            )
        )

        root_path = (
            write_root(
                pair_label=(
                    pair_label
                ),
                calculation=(
                    calculation
                ),
                bootstrap=(
                    bootstrap
                ),
                diagnostics=(
                    channel_arrays[
                        "diagnostics"
                    ]
                ),
                replicas=(
                    args.bootstrap_replicas
                ),
            )
        )

        results[
            pair_label
        ] = {
            "display": (
                definition[
                    "display"
                ]
            ),
            "rows": rows,
            "diagnostics": (
                channel_arrays[
                    "diagnostics"
                ]
            ),
            "csv_path": (
                csv_path
            ),
            "root_path": (
                root_path
            ),
        }

        print()
        print(
            f"{definition['display']} outputs:"
        )
        print(
            f"  CSV:  {csv_path}"
        )
        print(
            f"  ROOT: {root_path}"
        )

        print()
        print(
            "Per-bin paired/independent uncertainty ratio:"
        )

        for row in rows:

            print(
                f"  "
                f"{row['bin_low_edge']:.2f}"
                f"-"
                f"{row['bin_up_edge']:.2f}: "
                f"{row['bootstrap_over_independent']:.4f}"
            )

    interactive_path = (
        save_interactive(
            results
        )
    )

    summary_path = (
        write_summary(
            results=(
                results
            ),
            replicas=(
                args.bootstrap_replicas
            ),
            seed=(
                args.seed
            ),
        )
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
        "Outputs:"
    )
    print(
        f"  {OUTPUT_DIR}"
    )
    print()

    if interactive_path is not None:

        print(
            "Interactive comparison:"
        )

        print(
            f"  {interactive_path}"
        )

        print()

    print(
        "Run summary:"
    )

    print(
        f"  {summary_path}"
    )


if __name__ == "__main__":
    main()
