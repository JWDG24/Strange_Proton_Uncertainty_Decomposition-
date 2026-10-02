#!/usr/bin/env python3

"""
final_event_correlation_audit.py

Comprehensive final audit of the high-statistics Pythia particle/parton
event matching and correlation-aware statistical uncertainty.

This is a DIAGNOSTIC / VALIDATION script.  It does not overwrite the
main analysis outputs.

It checks, independently:

A. Input structure
   - all particle/parton fragments are found
   - fragment suffix sets agree
   - required branches exist
   - EventNumber is unique within each fragment
   - (fragment suffix, EventNumber) is globally unique on each side

B. Event-identity evidence
   - raw EventNumber overlap for corresponding particle/parton fragments
   - same-suffix matched-event agreement/correlation in:
       weightvec[0]
       leptons_pt
       leptons_eta
       leptons_phi
       met_et
       leptons_charge
   - WRONG-FRAGMENT control:
       particle suffix s is deliberately compared with parton suffix s+1
       at equal EventNumber
   If EventNumber + suffix identifies the same generated event, the
   correct-pair metrics should be much stronger than the wrong-fragment
   control metrics.

C. Repaired W+c event definition
   - MET > 25 GeV
   - lepton pT > 20 GeV
   - |eta_lepton| < 2.5
   - mT(W) > 40 GeV
   - jet pT > 25 GeV
   - |eta_jet| < 2.5
   - exactly one fiducial charm-identified jet
   - charm ID: jet_charge != 0
   - OS = +1, SS = -1

D. Correction-factor reproduction
   - recomputes the full-sample correction
       C_i = N_parton_i / N_particle_i
   - compares it with cross_check_weight0.py output if present

E. Statistical uncertainty
   - ordinary independent ratio error
   - covariance-aware analytic delta-method error
   - fresh paired Poisson bootstrap using the matched event key
   - compares with the previously saved correlated-stat CSV if present
   - uses a DIFFERENT audit seed from the production script

F. Output
   - detailed terminal audit
   - CSV per charge channel
   - event-identity metric CSV
   - ROOT file
   - interactive Plotly HTML
   - final_audit_report.txt

Expected location
-----------------
uncertainty_decomposition/
└── lepton_pseudorapidity_work_FINAL/
    └── cross_check_weight0/
        ├── final_event_correlation_audit.py
        └── cross_check_ROOT_files/

Recommended final run
---------------------
python3 lepton_pseudorapidity_work_FINAL/cross_check_weight0/final_event_correlation_audit.py

Default fresh audit bootstrap:
    1000 replicas
    seed = 987654
"""

import argparse
import csv
import math
from array import array
from pathlib import Path

import numpy as np
import ROOT


# ============================================================
# ROOT / paths
# ============================================================

ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "cross_check_ROOT_files"

CROSS_CHECK_CSV_DIR = (
    SCRIPT_DIR
    / "outputs"
    / "csv"
)

CORRELATED_CSV_DIR = (
    SCRIPT_DIR
    / "outputs"
    / "correlated_stat_uncertainty"
    / "csv"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "outputs"
    / "final_event_correlation_audit"
)

CSV_DIR = OUTPUT_DIR / "csv"
ROOT_DIR = OUTPUT_DIR / "root"
INTERACTIVE_DIR = OUTPUT_DIR / "interactive"

for directory in (
    CSV_DIR,
    ROOT_DIR,
    INTERACTIVE_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

REPORT_PATH = OUTPUT_DIR / "final_audit_report.txt"


# ============================================================
# Physics definition
# ============================================================

MET_MIN_GEV = 25.0
LEPTON_PT_MIN_GEV = 20.0
LEPTON_ABS_ETA_MAX = 2.5
MTW_MIN_GEV = 40.0
JET_PT_MIN_GEV = 25.0
JET_ABS_ETA_MAX = 2.5
NOMINAL_WEIGHT_INDEX = 0

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

N_ETA_BINS = len(ETA_BINS) - 1
ETA_BINS_ROOT = array(
    "d",
    ETA_BINS.tolist(),
)


CHANNELS = {
    "Pythia_plus": {
        "display": "Pythia W+",
        "particle_prefix": "WCharm_WCPy8plus",
        "parton_prefix": "WCharm_WCPyPartonplus",
        "cross_check_csv": "cross_check_Pythia_plus.csv",
        "correlated_csv": (
            "Pythia_plus_correlated_stat_uncertainty.csv"
        ),
    },
    "Pythia_minus": {
        "display": "Pythia W-",
        "particle_prefix": "WCharm_WCPy8minus",
        "parton_prefix": "WCharm_WCPyPartonminus",
        "cross_check_csv": "cross_check_Pythia_minus.csv",
        "correlated_csv": (
            "Pythia_minus_correlated_stat_uncertainty.csv"
        ),
    },
}


REQUIRED_BRANCHES = {
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


# ============================================================
# Reporting
# ============================================================

REPORT_LINES = []


def report(line=""):
    line = str(line)
    print(line)
    REPORT_LINES.append(line)


# ============================================================
# Basic helpers
# ============================================================

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


def discover_fragments(prefix):
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

        if suffix is None:
            continue

        if suffix in result:
            raise RuntimeError(
                f"Duplicate suffix {suffix} for {prefix}"
            )

        result[suffix] = path

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
        "WCharmTree"
    )

    if not tree:
        root_file.Close()
        raise RuntimeError(
            f"WCharmTree not found in:\n  {path}"
        )

    return root_file, tree


def branch_names(tree):
    return {
        branch.GetName()
        for branch in tree.GetListOfBranches()
    }


def validate_branches(tree, path):
    missing = sorted(
        REQUIRED_BRANCHES
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


def sequence_value(value, index):
    try:
        return value[index]
    except TypeError:
        if index != 0:
            raise IndexError(index)
        return value


def wrapped_delta_phi(a, b):
    return math.atan2(
        math.sin(a - b),
        math.cos(a - b),
    )


# ============================================================
# Selection
# ============================================================

def calculate_mtw(tree):
    delta_phi = wrapped_delta_phi(
        float(tree.leptons_phi),
        float(tree.met_phi),
    )

    mt2 = (
        2.0
        * float(tree.leptons_pt)
        * float(tree.met_et)
        * (
            1.0
            - math.cos(delta_phi)
        )
    )

    return math.sqrt(
        max(
            0.0,
            mt2,
        )
    )


def selected_charm_jets(tree):
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
            "jet_pt / jet_eta / jet_charge length mismatch"
        )

    selected = []

    for index in range(n_pt):
        pt = float(
            sequence_value(
                tree.jet_pt,
                index,
            )
        )

        eta = float(
            sequence_value(
                tree.jet_eta,
                index,
            )
        )

        charge = float(
            sequence_value(
                tree.jet_charge,
                index,
            )
        )

        if pt <= JET_PT_MIN_GEV:
            continue

        if abs(eta) >= JET_ABS_ETA_MAX:
            continue

        if charge == 0.0:
            continue

        if charge not in (-1.0, 1.0):
            raise RuntimeError(
                f"Unexpected nonzero jet_charge={charge}"
            )

        selected.append(
            (
                index,
                1.0
                if charge > 0.0
                else -1.0,
            )
        )

    return selected


def eta_bin_index(abs_eta):
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


def selected_contribution(
    tree,
    normalization,
):
    if float(tree.met_et) <= MET_MIN_GEV:
        return None

    if float(tree.leptons_pt) <= LEPTON_PT_MIN_GEV:
        return None

    eta = float(
        tree.leptons_eta
    )

    if abs(eta) >= LEPTON_ABS_ETA_MAX:
        return None

    if calculate_mtw(tree) <= MTW_MIN_GEV:
        return None

    charm_jets = selected_charm_jets(
        tree
    )

    if len(charm_jets) != 1:
        return None

    _, charm_sign = charm_jets[0]

    lepton_charge = float(
        tree.leptons_charge
    )

    os_ss = (
        -1.0
        if (
            lepton_charge
            * charm_sign
        ) > 0.0
        else 1.0
    )

    if len(tree.weightvec) <= 0:
        raise RuntimeError(
            "weightvec[0] is unavailable"
        )

    weight0 = float(
        tree.weightvec[0]
    )

    ibin = eta_bin_index(
        abs(eta)
    )

    if ibin < 0:
        return None

    return (
        ibin,
        os_ss
        * weight0
        * normalization,
        os_ss,
    )


# ============================================================
# Identity metric accumulator
# ============================================================

class NumericMetric:
    def __init__(self):
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0
        self.sum_abs_diff = 0.0
        self.close = 0

    def add(
        self,
        x,
        y,
        rtol=1.0e-8,
        atol=1.0e-10,
    ):
        x = float(x)
        y = float(y)

        if not (
            math.isfinite(x)
            and math.isfinite(y)
        ):
            return

        self.n += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_x2 += x * x
        self.sum_y2 += y * y
        self.sum_xy += x * y
        self.sum_abs_diff += abs(
            x - y
        )

        if math.isclose(
            x,
            y,
            rel_tol=rtol,
            abs_tol=atol,
        ):
            self.close += 1

    def pearson(self):
        if self.n < 2:
            return math.nan

        n = float(self.n)

        cov_num = (
            self.sum_xy
            - (
                self.sum_x
                * self.sum_y
                / n
            )
        )

        var_x = (
            self.sum_x2
            - (
                self.sum_x
                * self.sum_x
                / n
            )
        )

        var_y = (
            self.sum_y2
            - (
                self.sum_y
                * self.sum_y
                / n
            )
        )

        denom = math.sqrt(
            max(
                0.0,
                var_x * var_y,
            )
        )

        if denom == 0.0:
            return math.nan

        return cov_num / denom

    def mean_abs_diff(self):
        if self.n == 0:
            return math.nan
        return self.sum_abs_diff / self.n

    def close_fraction(self):
        if self.n == 0:
            return math.nan
        return self.close / self.n


class CircularMetric:
    def __init__(self):
        self.n = 0
        self.sum_abs_wrapped_diff = 0.0
        self.close = 0
        self.sum_cos_delta = 0.0

    def add(
        self,
        x,
        y,
        atol=1.0e-8,
    ):
        x = float(x)
        y = float(y)

        if not (
            math.isfinite(x)
            and math.isfinite(y)
        ):
            return

        delta = wrapped_delta_phi(
            x,
            y,
        )

        self.n += 1
        self.sum_abs_wrapped_diff += abs(
            delta
        )
        self.sum_cos_delta += math.cos(
            delta
        )

        if abs(delta) <= atol:
            self.close += 1

    def mean_abs_diff(self):
        if self.n == 0:
            return math.nan
        return (
            self.sum_abs_wrapped_diff
            / self.n
        )

    def close_fraction(self):
        if self.n == 0:
            return math.nan
        return self.close / self.n

    def mean_cos_delta(self):
        if self.n == 0:
            return math.nan
        return self.sum_cos_delta / self.n


class CategoricalMetric:
    def __init__(self):
        self.n = 0
        self.equal = 0

    def add(self, x, y):
        self.n += 1
        if x == y:
            self.equal += 1

    def equal_fraction(self):
        if self.n == 0:
            return math.nan
        return self.equal / self.n


def new_identity_metrics():
    return {
        "weight0": NumericMetric(),
        "leptons_pt": NumericMetric(),
        "leptons_eta": NumericMetric(),
        "met_et": NumericMetric(),
        "leptons_phi": CircularMetric(),
        "leptons_charge": CategoricalMetric(),
    }


def add_identity_pair(
    metrics,
    particle_record,
    parton_record,
):
    metrics[
        "weight0"
    ].add(
        particle_record[
            "weight0"
        ],
        parton_record[
            "weight0"
        ],
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    metrics[
        "leptons_pt"
    ].add(
        particle_record[
            "leptons_pt"
        ],
        parton_record[
            "leptons_pt"
        ],
    )

    metrics[
        "leptons_eta"
    ].add(
        particle_record[
            "leptons_eta"
        ],
        parton_record[
            "leptons_eta"
        ],
    )

    metrics[
        "met_et"
    ].add(
        particle_record[
            "met_et"
        ],
        parton_record[
            "met_et"
        ],
    )

    metrics[
        "leptons_phi"
    ].add(
        particle_record[
            "leptons_phi"
        ],
        parton_record[
            "leptons_phi"
        ],
    )

    metrics[
        "leptons_charge"
    ].add(
        particle_record[
            "leptons_charge"
        ],
        parton_record[
            "leptons_charge"
        ],
    )


# ============================================================
# Read one fragment
# ============================================================

def read_fragment(
    path,
    normalization,
):
    """
    Read one fragment and return raw records plus selected records.
    """

    root_file, tree = open_tree(
        path
    )

    validate_branches(
        tree,
        path,
    )

    raw_records = {}
    selected_records = {}

    counts = {
        "entries": 0,
        "selected": 0,
        "os": 0,
        "ss": 0,
    }

    nentries = int(
        tree.GetEntries()
    )

    counts["entries"] = nentries

    for ientry in range(nentries):
        tree.GetEntry(
            ientry
        )

        event_number = int(
            tree.EventNumber
        )

        if event_number in raw_records:
            root_file.Close()
            raise RuntimeError(
                f"{path.name}: duplicate EventNumber "
                f"{event_number}"
            )

        weight0 = float(
            tree.weightvec[0]
        )

        raw_records[
            event_number
        ] = {
            "weight0": weight0,
            "leptons_pt": float(
                tree.leptons_pt
            ),
            "leptons_eta": float(
                tree.leptons_eta
            ),
            "leptons_phi": float(
                tree.leptons_phi
            ),
            "leptons_charge": float(
                tree.leptons_charge
            ),
            "met_et": float(
                tree.met_et
            ),
        }

        contribution = (
            selected_contribution(
                tree,
                normalization,
            )
        )

        if contribution is None:
            continue

        ibin, final_weight, os_ss = (
            contribution
        )

        selected_records[
            event_number
        ] = (
            ibin,
            final_weight,
            os_ss,
        )

        counts["selected"] += 1

        if os_ss > 0.0:
            counts["os"] += 1
        else:
            counts["ss"] += 1

    root_file.Close()

    return (
        raw_records,
        selected_records,
        counts,
    )


# ============================================================
# CSV readers
# ============================================================

def read_csv(path):
    if not path.is_file():
        return None

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        return list(
            csv.DictReader(
                csv_file
            )
        )


# ============================================================
# Statistical calculations
# ============================================================

def binned_sum(
    bins,
    weights,
):
    mask = bins >= 0

    return np.bincount(
        bins[mask],
        weights=weights[mask],
        minlength=N_ETA_BINS,
    ).astype(float)


def binned_sumw2(
    bins,
    weights,
):
    mask = bins >= 0

    return np.bincount(
        bins[mask],
        weights=(
            weights[mask]
            ** 2
        ),
        minlength=N_ETA_BINS,
    ).astype(float)


def safe_ratio(
    numerator,
    denominator,
):
    out = np.full(
        N_ETA_BINS,
        np.nan,
        dtype=float,
    )

    mask = denominator != 0.0

    out[mask] = (
        numerator[mask]
        / denominator[mask]
    )

    return out


def compute_statistics(
    particle_bins,
    particle_weights,
    parton_bins,
    parton_weights,
):
    particle_yield = binned_sum(
        particle_bins,
        particle_weights,
    )

    parton_yield = binned_sum(
        parton_bins,
        parton_weights,
    )

    particle_var = binned_sumw2(
        particle_bins,
        particle_weights,
    )

    parton_var = binned_sumw2(
        parton_bins,
        parton_weights,
    )

    same_bin_mask = (
        (particle_bins >= 0)
        & (parton_bins >= 0)
        & (
            particle_bins
            == parton_bins
        )
    )

    covariance = np.bincount(
        particle_bins[
            same_bin_mask
        ],
        weights=(
            particle_weights[
                same_bin_mask
            ]
            * parton_weights[
                same_bin_mask
            ]
        ),
        minlength=N_ETA_BINS,
    ).astype(float)

    correction = safe_ratio(
        parton_yield,
        particle_yield,
    )

    independent_var = np.full(
        N_ETA_BINS,
        np.nan,
        dtype=float,
    )

    correlated_var = np.full(
        N_ETA_BINS,
        np.nan,
        dtype=float,
    )

    rho = np.full(
        N_ETA_BINS,
        np.nan,
        dtype=float,
    )

    for ibin in range(
        N_ETA_BINS
    ):
        P = parton_yield[
            ibin
        ]
        D = particle_yield[
            ibin
        ]

        vP = parton_var[
            ibin
        ]
        vD = particle_var[
            ibin
        ]
        cov = covariance[
            ibin
        ]

        if D == 0.0:
            continue

        independent_var[
            ibin
        ] = (
            vP
            / (
                D * D
            )
            + (
                P * P
                * vD
                / (
                    D ** 4
                )
            )
        )

        correlated_raw = (
            independent_var[
                ibin
            ]
            - (
                2.0
                * P
                * cov
                / (
                    D ** 3
                )
            )
        )

        correlated_var[
            ibin
        ] = max(
            0.0,
            correlated_raw,
        )

        denom = math.sqrt(
            max(
                0.0,
                vP * vD,
            )
        )

        if denom > 0.0:
            rho[
                ibin
            ] = cov / denom

    return {
        "particle_yield": particle_yield,
        "parton_yield": parton_yield,
        "particle_var": particle_var,
        "parton_var": parton_var,
        "covariance": covariance,
        "rho": rho,
        "correction": correction,
        "independent_unc": np.sqrt(
            independent_var
        ),
        "correlated_delta_unc": np.sqrt(
            correlated_var
        ),
    }


def paired_bootstrap(
    particle_bins,
    particle_weights,
    parton_bins,
    parton_weights,
    replicas,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    n_events = len(
        particle_bins
    )

    p_mask = particle_bins >= 0
    a_mask = parton_bins >= 0

    boot = np.full(
        (
            replicas,
            N_ETA_BINS,
        ),
        np.nan,
        dtype=float,
    )

    progress = max(
        1,
        replicas // 10,
    )

    report(
        f"  Fresh audit bootstrap: {replicas} replicas "
        f"(seed {seed})"
    )

    for irep in range(
        replicas
    ):
        multiplicity = rng.poisson(
            1.0,
            size=n_events,
        )

        p_yield = np.bincount(
            particle_bins[
                p_mask
            ],
            weights=(
                particle_weights[
                    p_mask
                ]
                * multiplicity[
                    p_mask
                ]
            ),
            minlength=N_ETA_BINS,
        )

        a_yield = np.bincount(
            parton_bins[
                a_mask
            ],
            weights=(
                parton_weights[
                    a_mask
                ]
                * multiplicity[
                    a_mask
                ]
            ),
            minlength=N_ETA_BINS,
        )

        boot[
            irep
        ] = safe_ratio(
            a_yield,
            p_yield,
        )

        if (
            (irep + 1) % progress == 0
            or irep == 0
            or irep + 1 == replicas
        ):
            report(
                f"    replica "
                f"{irep + 1:4d}/{replicas}"
            )

    return {
        "unc": np.nanstd(
            boot,
            axis=0,
            ddof=1,
        ),
        "mean": np.nanmean(
            boot,
            axis=0,
        ),
    }


# ============================================================
# Channel audit
# ============================================================

def audit_channel(
    pair_label,
    definition,
    replicas,
    seed,
):
    report()
    report("=" * 78)
    report(
        f"{definition['display']} FINAL AUDIT"
    )
    report("=" * 78)

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

    particle_suffixes = set(
        particle_fragments
    )
    parton_suffixes = set(
        parton_fragments
    )

    structure_pass = (
        bool(
            particle_suffixes
        )
        and particle_suffixes
        == parton_suffixes
    )

    report(
        f"Particle fragments: {len(particle_fragments)}"
    )
    report(
        f"Parton fragments:   {len(parton_fragments)}"
    )
    report(
        f"Suffix sets identical: {structure_pass}"
    )

    if not structure_pass:
        raise RuntimeError(
            f"{pair_label}: particle/parton suffix sets differ."
        )

    suffixes = sorted(
        particle_suffixes
    )

    # Total entry counts.
    particle_total = 0
    parton_total = 0

    for suffix in suffixes:
        pf, ptree = open_tree(
            particle_fragments[
                suffix
            ]
        )

        af, atree = open_tree(
            parton_fragments[
                suffix
            ]
        )

        validate_branches(
            ptree,
            particle_fragments[
                suffix
            ],
        )

        validate_branches(
            atree,
            parton_fragments[
                suffix
            ],
        )

        particle_total += int(
            ptree.GetEntries()
        )

        parton_total += int(
            atree.GetEntries()
        )

        pf.Close()
        af.Close()

    particle_norm = (
        1.0
        / particle_total
    )

    parton_norm = (
        1.0
        / parton_total
    )

    report(
        f"Particle entries: {particle_total:,}"
    )
    report(
        f"Parton entries:   {parton_total:,}"
    )

    same_metrics = (
        new_identity_metrics()
    )

    control_metrics = (
        new_identity_metrics()
    )

    global_particle_keys = set()
    global_parton_keys = set()

    raw_overlap_total = 0

    particle_selected_total = 0
    parton_selected_total = 0
    particle_os = 0
    particle_ss = 0
    parton_os = 0
    parton_ss = 0

    selected_both = 0
    selected_same_bin = 0
    selected_different_bin = 0

    particle_bins = []
    particle_weights = []
    parton_bins = []
    parton_weights = []

    # Read each corresponding suffix.  A deliberately wrong next-suffix
    # parton file is also read for the identity-control comparison.
    for suffix_index, suffix in enumerate(
        suffixes
    ):
        next_suffix = suffixes[
            (
                suffix_index + 1
            )
            % len(
                suffixes
            )
        ]

        (
            particle_raw,
            particle_selected,
            particle_counts,
        ) = read_fragment(
            particle_fragments[
                suffix
            ],
            particle_norm,
        )

        (
            parton_raw,
            parton_selected,
            parton_counts,
        ) = read_fragment(
            parton_fragments[
                suffix
            ],
            parton_norm,
        )

        (
            control_parton_raw,
            _,
            _,
        ) = read_fragment(
            parton_fragments[
                next_suffix
            ],
            parton_norm,
        )

        # Composite-key uniqueness.
        for event_number in (
            particle_raw
        ):
            key = (
                suffix,
                event_number,
            )

            if key in global_particle_keys:
                raise RuntimeError(
                    f"Duplicate particle composite key: {key}"
                )

            global_particle_keys.add(
                key
            )

        for event_number in (
            parton_raw
        ):
            key = (
                suffix,
                event_number,
            )

            if key in global_parton_keys:
                raise RuntimeError(
                    f"Duplicate parton composite key: {key}"
                )

            global_parton_keys.add(
                key
            )

        overlap = (
            set(
                particle_raw
            )
            & set(
                parton_raw
            )
        )

        raw_overlap_total += len(
            overlap
        )

        # Correct same-suffix identity metrics.
        for event_number in overlap:
            add_identity_pair(
                same_metrics,
                particle_raw[
                    event_number
                ],
                parton_raw[
                    event_number
                ],
            )

        # Wrong-fragment control.
        control_overlap = (
            set(
                particle_raw
            )
            & set(
                control_parton_raw
            )
        )

        for event_number in (
            control_overlap
        ):
            add_identity_pair(
                control_metrics,
                particle_raw[
                    event_number
                ],
                control_parton_raw[
                    event_number
                ],
            )

        particle_selected_total += (
            particle_counts[
                "selected"
            ]
        )

        parton_selected_total += (
            parton_counts[
                "selected"
            ]
        )

        particle_os += (
            particle_counts[
                "os"
            ]
        )

        particle_ss += (
            particle_counts[
                "ss"
            ]
        )

        parton_os += (
            parton_counts[
                "os"
            ]
        )

        parton_ss += (
            parton_counts[
                "ss"
            ]
        )

        selected_union = (
            set(
                particle_selected
            )
            | set(
                parton_selected
            )
        )

        for event_number in (
            selected_union
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
                p_bin = p_info[0]
                p_weight = p_info[1]

            if a_info is None:
                a_bin = -1
                a_weight = 0.0
            else:
                a_bin = a_info[0]
                a_weight = a_info[1]

            if (
                p_info is not None
                and a_info is not None
            ):
                selected_both += 1

                if p_bin == a_bin:
                    selected_same_bin += 1
                else:
                    selected_different_bin += 1

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

        report(
            f"  suffix {suffix:2d}: "
            f"raw overlap={len(overlap):6d}, "
            f"selected particle={particle_counts['selected']:6d}, "
            f"parton={parton_counts['selected']:6d}"
        )

    # --------------------------------------------------------
    # Structural / overlap diagnostics
    # --------------------------------------------------------

    particle_match_fraction = (
        raw_overlap_total
        / len(
            global_particle_keys
        )
    )

    parton_match_fraction = (
        raw_overlap_total
        / len(
            global_parton_keys
        )
    )

    report()
    report(
        "Composite-key structure:"
    )
    report(
        f"  particle keys unique: True "
        f"({len(global_particle_keys):,})"
    )
    report(
        f"  parton keys unique:   True "
        f"({len(global_parton_keys):,})"
    )
    report(
        f"  matched keys:         "
        f"{raw_overlap_total:,}"
    )
    report(
        f"  particle overlap:     "
        f"{100.0 * particle_match_fraction:.4f}%"
    )
    report(
        f"  parton overlap:       "
        f"{100.0 * parton_match_fraction:.4f}%"
    )

    # --------------------------------------------------------
    # Identity metrics and wrong-fragment control
    # --------------------------------------------------------

    report()
    report(
        "Event-identity evidence:"
    )
    report(
        "  metric                  correct pair          "
        "wrong-fragment control"
    )

    identity_rows = []

    numeric_names = [
        "weight0",
        "leptons_pt",
        "leptons_eta",
        "met_et",
    ]

    strong_discriminators = 0

    for name in numeric_names:
        correct = same_metrics[
            name
        ]
        control = control_metrics[
            name
        ]

        correct_r = (
            correct.pearson()
        )
        control_r = (
            control.pearson()
        )

        correct_close = (
            correct.close_fraction()
        )
        control_close = (
            control.close_fraction()
        )

        report(
            f"  {name:<22s} "
            f"r={correct_r:>8.4f}, "
            f"exact={correct_close:>8.4f}    "
            f"r={control_r:>8.4f}, "
            f"exact={control_close:>8.4f}"
        )

        improvement = (
            (
                math.isfinite(
                    correct_r
                )
                and math.isfinite(
                    control_r
                )
                and correct_r
                > control_r
                + 0.25
            )
            or (
                math.isfinite(
                    correct_close
                )
                and math.isfinite(
                    control_close
                )
                and correct_close
                > control_close
                + 0.25
            )
        )

        if improvement:
            strong_discriminators += 1

        identity_rows.append({
            "pair_label": pair_label,
            "metric": name,
            "correct_n": correct.n,
            "control_n": control.n,
            "correct_pearson": (
                correct_r
            ),
            "control_pearson": (
                control_r
            ),
            "correct_exact_fraction": (
                correct_close
            ),
            "control_exact_fraction": (
                control_close
            ),
            "correct_mean_abs_difference": (
                correct.mean_abs_diff()
            ),
            "control_mean_abs_difference": (
                control.mean_abs_diff()
            ),
        })

    correct_phi = (
        same_metrics[
            "leptons_phi"
        ]
    )
    control_phi = (
        control_metrics[
            "leptons_phi"
        ]
    )

    report(
        f"  {'leptons_phi':<22s} "
        f"<cos dphi>="
        f"{correct_phi.mean_cos_delta():>8.4f}, "
        f"exact={correct_phi.close_fraction():>8.4f}    "
        f"<cos dphi>="
        f"{control_phi.mean_cos_delta():>8.4f}, "
        f"exact={control_phi.close_fraction():>8.4f}"
    )

    if (
        math.isfinite(
            correct_phi.mean_cos_delta()
        )
        and math.isfinite(
            control_phi.mean_cos_delta()
        )
        and correct_phi.mean_cos_delta()
        > (
            control_phi.mean_cos_delta()
            + 0.25
        )
    ):
        strong_discriminators += 1

    identity_rows.append({
        "pair_label": pair_label,
        "metric": "leptons_phi",
        "correct_n": correct_phi.n,
        "control_n": control_phi.n,
        "correct_pearson": math.nan,
        "control_pearson": math.nan,
        "correct_exact_fraction": (
            correct_phi.close_fraction()
        ),
        "control_exact_fraction": (
            control_phi.close_fraction()
        ),
        "correct_mean_abs_difference": (
            correct_phi.mean_abs_diff()
        ),
        "control_mean_abs_difference": (
            control_phi.mean_abs_diff()
        ),
        "correct_mean_cos_delta": (
            correct_phi.mean_cos_delta()
        ),
        "control_mean_cos_delta": (
            control_phi.mean_cos_delta()
        ),
    })

    correct_charge = (
        same_metrics[
            "leptons_charge"
        ]
    )

    control_charge = (
        control_metrics[
            "leptons_charge"
        ]
    )

    report(
        f"  {'leptons_charge':<22s} "
        f"equal="
        f"{correct_charge.equal_fraction():>8.4f}                "
        f"equal="
        f"{control_charge.equal_fraction():>8.4f}"
    )

    identity_rows.append({
        "pair_label": pair_label,
        "metric": "leptons_charge",
        "correct_n": correct_charge.n,
        "control_n": control_charge.n,
        "correct_equal_fraction": (
            correct_charge.equal_fraction()
        ),
        "control_equal_fraction": (
            control_charge.equal_fraction()
        ),
    })

    identity_support = (
        strong_discriminators >= 2
    )

    report()
    report(
        "Identity-control verdict:"
    )

    if identity_support:
        report(
            "  STRONG SUPPORT: correct same-suffix matches "
            "outperform deliberately wrong-fragment matches "
            f"in {strong_discriminators} continuous/circular metrics."
        )
    else:
        report(
            "  INCONCLUSIVE: correct matching does not sufficiently "
            "outperform the wrong-fragment control."
        )

    # --------------------------------------------------------
    # Selection
    # --------------------------------------------------------

    report()
    report(
        "Repaired-selection reproduction:"
    )
    report(
        f"  particle selected: "
        f"{particle_selected_total:,} "
        f"(OS={particle_os:,}, SS={particle_ss:,})"
    )
    report(
        f"  parton selected:   "
        f"{parton_selected_total:,} "
        f"(OS={parton_os:,}, SS={parton_ss:,})"
    )
    report(
        f"  selected on both:  "
        f"{selected_both:,}"
    )
    report(
        f"  same eta bin:      "
        f"{selected_same_bin:,}"
    )
    report(
        f"  different eta bin: "
        f"{selected_different_bin:,}"
    )

    # --------------------------------------------------------
    # Statistical calculation
    # --------------------------------------------------------

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

    statistics = compute_statistics(
        particle_bins,
        particle_weights,
        parton_bins,
        parton_weights,
    )

    bootstrap = paired_bootstrap(
        particle_bins,
        particle_weights,
        parton_bins,
        parton_weights,
        replicas=replicas,
        seed=seed,
    )

    # --------------------------------------------------------
    # Compare with existing analysis outputs
    # --------------------------------------------------------

    cross_check_path = (
        CROSS_CHECK_CSV_DIR
        / definition[
            "cross_check_csv"
        ]
    )

    correlated_path = (
        CORRELATED_CSV_DIR
        / definition[
            "correlated_csv"
        ]
    )

    cross_check_rows = (
        read_csv(
            cross_check_path
        )
    )

    saved_correlated_rows = (
        read_csv(
            correlated_path
        )
    )

    report()
    report(
        "Existing-output comparison:"
    )

    cross_check_max_delta = math.nan
    saved_correction_max_delta = math.nan
    saved_independent_max_rel = math.nan
    saved_delta_max_rel = math.nan
    saved_bootstrap_max_rel = math.nan

    if cross_check_rows is None:
        report(
            "  cross_check CSV: not found"
        )
    else:
        deltas = []

        for ibin, row in enumerate(
            cross_check_rows
        ):
            value = float(
                row[
                    "C_new_crosscheck"
                ]
            )

            deltas.append(
                abs(
                    value
                    - statistics[
                        "correction"
                    ][
                        ibin
                    ]
                )
            )

        cross_check_max_delta = max(
            deltas
        )

        report(
            "  central correction vs cross_check_weight0: "
            f"max |delta| = {cross_check_max_delta:.3e}"
        )

    if saved_correlated_rows is None:
        report(
            "  saved correlated-stat CSV: not found"
        )
    else:
        correction_deltas = []
        independent_rel = []
        delta_rel = []
        bootstrap_rel = []

        for ibin, row in enumerate(
            saved_correlated_rows
        ):
            saved_c = float(
                row[
                    "correction_factor"
                ]
            )

            saved_i = float(
                row[
                    "stat_unc_independent"
                ]
            )

            saved_d = float(
                row[
                    "stat_unc_correlated_delta"
                ]
            )

            saved_b = float(
                row[
                    "stat_unc_paired_bootstrap"
                ]
            )

            correction_deltas.append(
                abs(
                    saved_c
                    - statistics[
                        "correction"
                    ][
                        ibin
                    ]
                )
            )

            current_i = (
                statistics[
                    "independent_unc"
                ][
                    ibin
                ]
            )

            current_d = (
                statistics[
                    "correlated_delta_unc"
                ][
                    ibin
                ]
            )

            current_b = (
                bootstrap[
                    "unc"
                ][
                    ibin
                ]
            )

            if saved_i != 0.0:
                independent_rel.append(
                    abs(
                        current_i
                        - saved_i
                    )
                    / abs(
                        saved_i
                    )
                )

            if saved_d != 0.0:
                delta_rel.append(
                    abs(
                        current_d
                        - saved_d
                    )
                    / abs(
                        saved_d
                    )
                )

            if saved_b != 0.0:
                bootstrap_rel.append(
                    abs(
                        current_b
                        - saved_b
                    )
                    / abs(
                        saved_b
                    )
                )

        saved_correction_max_delta = max(
            correction_deltas
        )

        saved_independent_max_rel = max(
            independent_rel
        )

        saved_delta_max_rel = max(
            delta_rel
        )

        saved_bootstrap_max_rel = max(
            bootstrap_rel
        )

        report(
            "  saved central correction: "
            f"max |delta| = "
            f"{saved_correction_max_delta:.3e}"
        )
        report(
            "  saved independent stat:   "
            f"max relative delta = "
            f"{100.0 * saved_independent_max_rel:.2f}%"
        )
        report(
            "  saved analytic stat:      "
            f"max relative delta = "
            f"{100.0 * saved_delta_max_rel:.2f}%"
        )
        report(
            "  saved bootstrap stat:     "
            f"max relative delta = "
            f"{100.0 * saved_bootstrap_max_rel:.2f}% "
            "(fresh independent audit seed)"
        )

    # --------------------------------------------------------
    # Bin-by-bin audit table
    # --------------------------------------------------------

    report()
    report(
        "Bin-by-bin uncertainty audit:"
    )
    report(
        "  eta bin        independent      analytic-cov      "
        "fresh bootstrap    boot/ind    analytic/boot"
    )

    rows = []

    analytic_bootstrap_agreement = []

    for ibin in range(
        N_ETA_BINS
    ):
        independent = float(
            statistics[
                "independent_unc"
            ][
                ibin
            ]
        )

        analytic = float(
            statistics[
                "correlated_delta_unc"
            ][
                ibin
            ]
        )

        boot_unc = float(
            bootstrap[
                "unc"
            ][
                ibin
            ]
        )

        boot_ratio = (
            boot_unc
            / independent
            if independent > 0.0
            else math.nan
        )

        analytic_boot_ratio = (
            analytic
            / boot_unc
            if boot_unc > 0.0
            else math.nan
        )

        if (
            boot_unc > 0.0
            and math.isfinite(
                analytic
            )
        ):
            analytic_bootstrap_agreement.append(
                abs(
                    analytic
                    - boot_unc
                )
                / boot_unc
            )

        report(
            f"  "
            f"{ETA_BINS[ibin]:.2f}-"
            f"{ETA_BINS[ibin + 1]:.2f}   "
            f"{independent:>12.6g}   "
            f"{analytic:>12.6g}   "
            f"{boot_unc:>14.6g}   "
            f"{boot_ratio:>8.4f}   "
            f"{analytic_boot_ratio:>11.4f}"
        )

        rows.append({
            "pair_label": pair_label,
            "bin": ibin + 1,
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
                statistics[
                    "correction"
                ][
                    ibin
                ]
            ),
            "stat_unc_independent": (
                independent
            ),
            "stat_unc_correlated_delta": (
                analytic
            ),
            "stat_unc_fresh_paired_bootstrap": (
                boot_unc
            ),
            "fresh_bootstrap_mean": float(
                bootstrap[
                    "mean"
                ][
                    ibin
                ]
            ),
            "covariance": float(
                statistics[
                    "covariance"
                ][
                    ibin
                ]
            ),
            "correlation_coefficient": float(
                statistics[
                    "rho"
                ][
                    ibin
                ]
            ),
            "fresh_bootstrap_over_independent": (
                boot_ratio
            ),
            "analytic_over_fresh_bootstrap": (
                analytic_boot_ratio
            ),
        })

    max_analytic_bootstrap_rel = max(
        analytic_bootstrap_agreement
    )

    mean_analytic_bootstrap_rel = (
        sum(
            analytic_bootstrap_agreement
        )
        / len(
            analytic_bootstrap_agreement
        )
    )

    report()
    report(
        "Analytic covariance vs fresh paired bootstrap:"
    )
    report(
        f"  mean relative difference: "
        f"{100.0 * mean_analytic_bootstrap_rel:.2f}%"
    )
    report(
        f"  max relative difference:  "
        f"{100.0 * max_analytic_bootstrap_rel:.2f}%"
    )

    # --------------------------------------------------------
    # Conservative audit verdict
    # --------------------------------------------------------

    central_reproduction_pass = (
        (
            not math.isfinite(
                cross_check_max_delta
            )
            or cross_check_max_delta
            < 1.0e-9
        )
        and (
            not math.isfinite(
                saved_correction_max_delta
            )
            or saved_correction_max_delta
            < 1.0e-9
        )
    )

    analytic_reproduction_pass = (
        not math.isfinite(
            saved_delta_max_rel
        )
        or saved_delta_max_rel
        < 1.0e-9
    )

    bootstrap_stability_pass = (
        (
            not math.isfinite(
                saved_bootstrap_max_rel
            )
            or saved_bootstrap_max_rel
            < 0.15
        )
        and max_analytic_bootstrap_rel
        < 0.15
    )

    rho_physical = np.all(
        np.abs(
            statistics[
                "rho"
            ][
                np.isfinite(
                    statistics[
                        "rho"
                    ]
                )
            ]
        )
        <= (
            1.0
            + 1.0e-9
        )
    )

    final_pass = (
        structure_pass
        and identity_support
        and central_reproduction_pass
        and analytic_reproduction_pass
        and bootstrap_stability_pass
        and rho_physical
    )

    report()
    report(
        "FINAL CHANNEL VERDICT:"
    )
    report(
        f"  structure / uniqueness:       "
        f"{'PASS' if structure_pass else 'FAIL'}"
    )
    report(
        f"  event-identity control:        "
        f"{'PASS' if identity_support else 'INCONCLUSIVE'}"
    )
    report(
        f"  central correction reproduce: "
        f"{'PASS' if central_reproduction_pass else 'FAIL'}"
    )
    report(
        f"  analytic covariance reproduce:"
        f" {'PASS' if analytic_reproduction_pass else 'FAIL'}"
    )
    report(
        f"  bootstrap stability:          "
        f"{'PASS' if bootstrap_stability_pass else 'CHECK'}"
    )
    report(
        f"  covariance coefficients sane: "
        f"{'PASS' if rho_physical else 'FAIL'}"
    )
    report(
        f"  OVERALL: "
        f"{'PASS' if final_pass else 'REVIEW REQUIRED'}"
    )

    diagnostics = {
        "pair_label": pair_label,
        "display": (
            definition[
                "display"
            ]
        ),
        "particle_total_entries": (
            particle_total
        ),
        "parton_total_entries": (
            parton_total
        ),
        "raw_overlap_total": (
            raw_overlap_total
        ),
        "particle_match_fraction": (
            particle_match_fraction
        ),
        "parton_match_fraction": (
            parton_match_fraction
        ),
        "particle_selected_total": (
            particle_selected_total
        ),
        "parton_selected_total": (
            parton_selected_total
        ),
        "particle_os": (
            particle_os
        ),
        "particle_ss": (
            particle_ss
        ),
        "parton_os": (
            parton_os
        ),
        "parton_ss": (
            parton_ss
        ),
        "selected_both": (
            selected_both
        ),
        "selected_same_bin": (
            selected_same_bin
        ),
        "selected_different_bin": (
            selected_different_bin
        ),
        "identity_support": (
            identity_support
        ),
        "strong_identity_discriminators": (
            strong_discriminators
        ),
        "central_reproduction_pass": (
            central_reproduction_pass
        ),
        "analytic_reproduction_pass": (
            analytic_reproduction_pass
        ),
        "bootstrap_stability_pass": (
            bootstrap_stability_pass
        ),
        "rho_physical": (
            bool(
                rho_physical
            )
        ),
        "final_pass": (
            final_pass
        ),
        "mean_analytic_bootstrap_rel": (
            mean_analytic_bootstrap_rel
        ),
        "max_analytic_bootstrap_rel": (
            max_analytic_bootstrap_rel
        ),
        "cross_check_max_delta": (
            cross_check_max_delta
        ),
        "saved_bootstrap_max_rel": (
            saved_bootstrap_max_rel
        ),
    }

    return {
        "rows": rows,
        "identity_rows": (
            identity_rows
        ),
        "diagnostics": (
            diagnostics
        ),
    }


# ============================================================
# Output writers
# ============================================================

AUDIT_COLUMNS = [
    "pair_label",
    "bin",
    "bin_low_edge",
    "bin_up_edge",
    "correction_factor",
    "stat_unc_independent",
    "stat_unc_correlated_delta",
    "stat_unc_fresh_paired_bootstrap",
    "fresh_bootstrap_mean",
    "covariance",
    "correlation_coefficient",
    "fresh_bootstrap_over_independent",
    "analytic_over_fresh_bootstrap",
]


def write_audit_csv(
    pair_label,
    rows,
):
    path = (
        CSV_DIR
        / f"{pair_label}_final_audit.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=AUDIT_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(
            rows
        )

    return path


def write_identity_csv(
    all_identity_rows,
):
    path = (
        CSV_DIR
        / "event_identity_metrics.csv"
    )

    fieldnames = sorted(
        {
            key
            for row in all_identity_rows
            for key in row
        }
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            all_identity_rows
        )

    return path


def make_hist(
    name,
    title,
    values,
):
    hist = ROOT.TH1D(
        name,
        title,
        N_ETA_BINS,
        ETA_BINS_ROOT,
    )
    hist.SetDirectory(0)

    for ibin, value in enumerate(
        values,
        start=1,
    ):
        hist.SetBinContent(
            ibin,
            float(value),
        )

    return hist


def write_root_file(
    results,
    replicas,
    seed,
):
    path = (
        ROOT_DIR
        / "final_event_correlation_audit.root"
    )

    root_file = ROOT.TFile(
        str(path),
        "RECREATE",
    )

    ROOT.TNamed(
        "MatchingKey",
        "(fragment suffix, EventNumber)",
    ).Write()

    ROOT.TNamed(
        "AuditBootstrapReplicas",
        str(
            replicas
        ),
    ).Write()

    ROOT.TNamed(
        "AuditBootstrapSeed",
        str(
            seed
        ),
    ).Write()

    for pair_label, result in (
        results.items()
    ):
        rows = result[
            "rows"
        ]

        values = {
            "correction": [
                row[
                    "correction_factor"
                ]
                for row in rows
            ],
            "independent": [
                row[
                    "stat_unc_independent"
                ]
                for row in rows
            ],
            "analytic": [
                row[
                    "stat_unc_correlated_delta"
                ]
                for row in rows
            ],
            "bootstrap": [
                row[
                    "stat_unc_fresh_paired_bootstrap"
                ]
                for row in rows
            ],
            "rho": [
                row[
                    "correlation_coefficient"
                ]
                for row in rows
            ],
        }

        for short_name, data in (
            values.items()
        ):
            hist = make_hist(
                f"{pair_label}_{short_name}",
                f"{pair_label};|#eta_{{#ell}}|;{short_name}",
                data,
            )
            hist.Write()

    root_file.Write()
    root_file.Close()

    return path


# ============================================================
# Interactive output
# ============================================================

def save_interactive(results):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

    except ImportError:
        report(
            "Plotly unavailable; skipping interactive HTML."
        )
        return None

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.09,
        subplot_titles=(
            "Statistical uncertainty comparison",
            "Correlation-aware / independent uncertainty",
            "Event-identity control: correct vs wrong fragment",
        ),
        row_heights=[
            0.38,
            0.25,
            0.37,
        ],
    )

    pair_labels = list(
        results.keys()
    )

    traces_per_pair = 8

    for pair_index, pair_label in enumerate(
        pair_labels
    ):
        result = results[
            pair_label
        ]

        rows = result[
            "rows"
        ]

        identity = result[
            "identity_rows"
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

        independent = [
            row[
                "stat_unc_independent"
            ]
            for row in rows
        ]

        analytic = [
            row[
                "stat_unc_correlated_delta"
            ]
            for row in rows
        ]

        bootstrap = [
            row[
                "stat_unc_fresh_paired_bootstrap"
            ]
            for row in rows
        ]

        ratio_boot = [
            row[
                "fresh_bootstrap_over_independent"
            ]
            for row in rows
        ]

        ratio_analytic = [
            (
                row[
                    "stat_unc_correlated_delta"
                ]
                / row[
                    "stat_unc_independent"
                ]
            )
            for row in rows
        ]

        figure.add_trace(
            go.Scatter(
                x=x,
                y=independent,
                mode="lines+markers",
                name="Independent",
                visible=visible,
            ),
            row=1,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=x,
                y=analytic,
                mode="lines+markers",
                name="Analytic covariance",
                visible=visible,
            ),
            row=1,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=x,
                y=bootstrap,
                mode="lines+markers",
                name="Fresh paired bootstrap",
                visible=visible,
            ),
            row=1,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=x,
                y=ratio_boot,
                mode="lines+markers",
                name="Bootstrap / independent",
                visible=visible,
            ),
            row=2,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=x,
                y=ratio_analytic,
                mode="lines+markers",
                name="Analytic / independent",
                visible=visible,
            ),
            row=2,
            col=1,
        )

        numeric_identity = [
            row
            for row in identity
            if row.get(
                "metric"
            )
            in (
                "weight0",
                "leptons_pt",
                "leptons_eta",
                "met_et",
            )
        ]

        categories = [
            row[
                "metric"
            ]
            for row in numeric_identity
        ]

        correct_r = [
            row[
                "correct_pearson"
            ]
            for row in numeric_identity
        ]

        control_r = [
            row[
                "control_pearson"
            ]
            for row in numeric_identity
        ]

        figure.add_trace(
            go.Bar(
                x=categories,
                y=correct_r,
                name="Correct pair correlation",
                visible=visible,
            ),
            row=3,
            col=1,
        )

        figure.add_trace(
            go.Bar(
                x=categories,
                y=control_r,
                name="Wrong-fragment correlation",
                visible=visible,
            ),
            row=3,
            col=1,
        )

        # Extra trace: difference correct-control.
        figure.add_trace(
            go.Scatter(
                x=categories,
                y=[
                    (
                        correct_r[index]
                        - control_r[index]
                    )
                    for index in range(
                        len(
                            categories
                        )
                    )
                ],
                mode="lines+markers",
                name="Correlation separation",
                visible=visible,
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

        start = (
            pair_index
            * traces_per_pair
        )

        for index in range(
            start,
            start + traces_per_pair,
        ):
            mask[index] = True

        display = (
            results[
                pair_label
            ][
                "diagnostics"
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
                            "Final event-correlation audit"
                            f"<br><sup>{display}</sup>"
                        ),
                    },
                ],
            )
        )

    first_display = (
        results[
            pair_labels[0]
        ][
            "diagnostics"
        ][
            "display"
        ]
    )

    figure.update_layout(
        title=dict(
            text=(
                "Final event-correlation audit"
                f"<br><sup>{first_display}</sup>"
            ),
            x=0.5,
        ),
        template="plotly_white",
        height=1150,
        width=1150,
        hovermode="x unified",
        barmode="group",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.01,
                y=1.12,
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    figure.update_yaxes(
        title_text="Statistical uncertainty",
        row=1,
        col=1,
        rangemode="tozero",
    )

    figure.update_yaxes(
        title_text="Ratio",
        row=2,
        col=1,
    )

    figure.add_hline(
        y=1.0,
        line_dash="dash",
        row=2,
        col=1,
    )

    figure.update_yaxes(
        title_text="Pearson correlation",
        row=3,
        col=1,
    )

    figure.update_xaxes(
        title_text="Lepton |eta|",
        range=[
            0.0,
            2.5,
        ],
        row=2,
        col=1,
    )

    figure.update_xaxes(
        title_text="Matched-event variable",
        row=3,
        col=1,
    )

    path = (
        INTERACTIVE_DIR
        / "final_event_correlation_audit.html"
    )

    figure.write_html(
        str(path),
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
    )

    return path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Final independent audit of event matching and "
            "correlation-aware statistical uncertainties."
        )
    )

    parser.add_argument(
        "--bootstrap-replicas",
        type=int,
        default=1000,
        help=(
            "Fresh paired-bootstrap replicas "
            "(default: 1000)."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=987654,
        help=(
            "Independent audit random seed "
            "(default: 987654)."
        ),
    )

    args = parser.parse_args()

    if not INPUT_DIR.is_dir():
        raise RuntimeError(
            "Input ROOT directory not found:\n"
            f"  {INPUT_DIR}"
        )

    if args.bootstrap_replicas <= 1:
        raise RuntimeError(
            "--bootstrap-replicas must be > 1"
        )

    report()
    report(
        "============================================================"
    )
    report(
        " FINAL EVENT-MATCHING + CORRELATED-STAT AUDIT"
    )
    report(
        "============================================================"
    )
    report()
    report(
        "This script independently validates:"
    )
    report(
        "  - fragment/EventNumber matching"
    )
    report(
        "  - wrong-fragment identity control"
    )
    report(
        "  - repaired W+c event selection"
    )
    report(
        "  - central correction reproduction"
    )
    report(
        "  - analytic covariance uncertainty"
    )
    report(
        "  - fresh paired Poisson bootstrap"
    )
    report(
        "  - agreement with saved analysis outputs"
    )
    report()

    results = {}
    all_identity_rows = []

    for channel_index, (
        pair_label,
        definition,
    ) in enumerate(
        CHANNELS.items()
    ):
        result = audit_channel(
            pair_label=pair_label,
            definition=definition,
            replicas=(
                args.bootstrap_replicas
            ),
            seed=(
                args.seed
                + channel_index
            ),
        )

        results[
            pair_label
        ] = result

        all_identity_rows.extend(
            result[
                "identity_rows"
            ]
        )

        csv_path = write_audit_csv(
            pair_label,
            result[
                "rows"
            ],
        )

        report(
            f"Audit CSV: {csv_path}"
        )

    identity_csv_path = (
        write_identity_csv(
            all_identity_rows
        )
    )

    root_path = write_root_file(
        results,
        replicas=(
            args.bootstrap_replicas
        ),
        seed=(
            args.seed
        ),
    )

    interactive_path = (
        save_interactive(
            results
        )
    )

    all_pass = all(
        result[
            "diagnostics"
        ][
            "final_pass"
        ]
        for result in (
            results.values()
        )
    )

    report()
    report(
        "=" * 78
    )
    report(
        "GLOBAL FINAL AUDIT VERDICT"
    )
    report(
        "=" * 78
    )

    for pair_label, result in (
        results.items()
    ):
        diagnostic = (
            result[
                "diagnostics"
            ]
        )

        report(
            f"  {diagnostic['display']}: "
            f"{'PASS' if diagnostic['final_pass'] else 'REVIEW REQUIRED'}"
        )

    report()
    report(
        f"  OVERALL: "
        f"{'PASS' if all_pass else 'REVIEW REQUIRED'}"
    )

    if all_pass:
        report()
        report(
            "The event-identity interpretation and the "
            "correlation-aware statistical treatment passed "
            "all automated checks in this audit."
        )
        report(
            "This remains a methodological validation, not a "
            "substitute for supervisor/analysis-group sign-off."
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
        "Outputs:"
    )
    report(
        f"  report:      {REPORT_PATH}"
    )
    report(
        f"  identity CSV:{identity_csv_path}"
    )
    report(
        f"  ROOT:        {root_path}"
    )

    if interactive_path is not None:
        report(
            f"  interactive: {interactive_path}"
        )


if __name__ == "__main__":
    main()
