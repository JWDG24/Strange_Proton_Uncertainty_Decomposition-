#!/usr/bin/env python3

# ============================================================
# truth_reco_analysis.py
#
# Purpose:
#   Perform a truth-level vs reconstruction-level study of the
#   lepton pseudorapidity observable, provided that BOTH truth
#   and reconstructed quantities exist in the SAME event tree.
#
# The analysis:
#
#   1. Scans all W+c ROOT samples.
#   2. Searches for truth/reco versions of:
#        - lepton pT
#        - lepton eta
#        - lepton phi
#        - missing ET
#        - missing-ET phi
#   3. Applies the SAME fiducial cuts at truth and reco level:
#
#        pT(l) > 20 GeV
#        |eta(l)| < 2.5
#        MET > 25 GeV
#        mT(W) > 40 GeV
#
#   4. Calculates:
#        - truth |eta| histogram
#        - reco |eta| histogram
#        - truth/reco correction factor
#        - paired-bootstrap statistical uncertainty
#        - truth -> reco migration matrix
#        - truth efficiency by eta bin
#        - reco purity by eta bin
#        - pass/fail migration counts
#
# IMPORTANT:
#   Particle level is NOT silently treated as reco level.
#   A true truth/reco analysis requires explicit truth and reco
#   branches belonging to the same underlying event.
#
# If suitable truth/reco branch pairs are not found, the script
# writes a diagnostic report explaining what was found and does
# not manufacture a truth/reco result.
#
# Expected location:
#
#   uncertainty_decomposition/
#   └── lepton_pseudorapidity_work/
#       └── truth_reco_analysis/
#           └── truth_reco_analysis.py
#
# Run from the project root:
#
#   python3 lepton_pseudorapidity_work/truth_reco_analysis/truth_reco_analysis.py
#
# ============================================================

from pathlib import Path
from array import array
import csv
import math
import re

import numpy as np
import ROOT


# ============================================================
# ROOT settings
# ============================================================

ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)


# ============================================================
# Directories
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = SCRIPT_DIR / "truth_reco_analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "truth_reco_analysis_report.txt"


# ============================================================
# Input samples
# ============================================================

ROOT_FILES = {
    "WCPy8plus": DATA_DIR / "WCharm_WCPy8plus.root",
    "WCPy8minus": DATA_DIR / "WCharm_WCPy8minus.root",
    "WCPyPartonplus": DATA_DIR / "WCharm_WCPyPartonplus.root",
    "WCPyPartonminus": DATA_DIR / "WCharm_WCPyPartonminus.root",
    "WCH7plus": DATA_DIR / "WCharm_WCH7plus.root",
    "WCH7minus": DATA_DIR / "WCharm_WCH7minus.root",
    "WCHPartonplus": DATA_DIR / "WCharm_WCHPartonplus.root",
    "WCHPartonminus": DATA_DIR / "WCharm_WCHPartonminus.root",
}


# ============================================================
# Analysis settings
# ============================================================

LEPTON_PT_MIN_GEV = 20.0
LEPTON_ABS_ETA_MAX = 2.5
MET_MIN_GEV = 25.0
MTW_MIN_GEV = 40.0

ETA_BIN_EDGES = np.array(
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

N_ETA_BINS = len(ETA_BIN_EDGES) - 1

# Paired Poisson-bootstrap replicas.
#
# The same random multiplier is applied to the truth and reco
# representation of an event, preserving their correlation.
N_BOOTSTRAP = 500
BOOTSTRAP_SEED = 12345


# ============================================================
# Branch aliases
# ============================================================

# These are deliberately broad because ntuple naming conventions
# vary. Exact branch names are resolved from the tree itself.

TRUTH_ALIASES = {
    "lep_pt": [
        "truth_leptons_pt",
        "truth_lepton_pt",
        "truthlep_pt",
        "truth_lep_pt",
        "leptons_truth_pt",
        "lepton_truth_pt",
        "mc_leptons_pt",
        "mc_lepton_pt",
        "truth_pt_lepton",
    ],
    "lep_eta": [
        "truth_leptons_eta",
        "truth_lepton_eta",
        "truthlep_eta",
        "truth_lep_eta",
        "leptons_truth_eta",
        "lepton_truth_eta",
        "mc_leptons_eta",
        "mc_lepton_eta",
        "truth_eta_lepton",
    ],
    "lep_phi": [
        "truth_leptons_phi",
        "truth_lepton_phi",
        "truthlep_phi",
        "truth_lep_phi",
        "leptons_truth_phi",
        "lepton_truth_phi",
        "mc_leptons_phi",
        "mc_lepton_phi",
        "truth_phi_lepton",
    ],
    "met_et": [
        "truth_met_et",
        "truth_met",
        "met_truth_et",
        "truth_missing_et",
        "truth_missinget",
        "mc_met_et",
    ],
    "met_phi": [
        "truth_met_phi",
        "met_truth_phi",
        "truth_missing_phi",
        "truth_missinget_phi",
        "mc_met_phi",
    ],
}

RECO_ALIASES = {
    "lep_pt": [
        "reco_leptons_pt",
        "reco_lepton_pt",
        "recolep_pt",
        "reco_lep_pt",
        "leptons_reco_pt",
        "lepton_reco_pt",
        "leptons_pt",
        "lepton_pt",
    ],
    "lep_eta": [
        "reco_leptons_eta",
        "reco_lepton_eta",
        "recolep_eta",
        "reco_lep_eta",
        "leptons_reco_eta",
        "lepton_reco_eta",
        "leptons_eta",
        "lepton_eta",
    ],
    "lep_phi": [
        "reco_leptons_phi",
        "reco_lepton_phi",
        "recolep_phi",
        "reco_lep_phi",
        "leptons_reco_phi",
        "lepton_reco_phi",
        "leptons_phi",
        "lepton_phi",
    ],
    "met_et": [
        "reco_met_et",
        "reco_met",
        "met_reco_et",
        "met_et",
        "missing_et",
        "missinget",
    ],
    "met_phi": [
        "reco_met_phi",
        "met_reco_phi",
        "met_phi",
        "missing_phi",
        "missinget_phi",
    ],
}


# ============================================================
# Reporting helper
# ============================================================

REPORT_LINES = []


def report(line=""):
    print(line)
    REPORT_LINES.append(str(line))


# ============================================================
# Name normalisation
# ============================================================

def normalise_name(name):
    return re.sub(
        r"[^a-z0-9]",
        "",
        name.lower(),
    )


# ============================================================
# ROOT tree helpers
# ============================================================

def choose_tree(root_file, filename):
    trees = []

    for key in root_file.GetListOfKeys():
        obj = key.ReadObj()

        if obj and obj.InheritsFrom("TTree"):
            trees.append(obj)

    if not trees:
        raise RuntimeError(
            f"No TTree found in:\n  {filename}"
        )

    return max(
        trees,
        key=lambda tree: int(tree.GetEntries()),
    )


def branch_names(tree):
    return [
        branch.GetName()
        for branch in tree.GetListOfBranches()
    ]


# ============================================================
# Branch resolution
# ============================================================

def resolve_alias(branches, aliases):
    """
    Resolve aliases after removing case/underscore differences.
    """

    by_normalised = {
        normalise_name(name): name
        for name in branches
    }

    for alias in aliases:
        key = normalise_name(alias)

        if key in by_normalised:
            return by_normalised[key]

    return None


def resolve_level_branches(tree):
    branches = branch_names(tree)

    truth = {
        key: resolve_alias(
            branches,
            aliases,
        )
        for key, aliases in TRUTH_ALIASES.items()
    }

    reco = {
        key: resolve_alias(
            branches,
            aliases,
        )
        for key, aliases in RECO_ALIASES.items()
    }

    return truth, reco, branches


def mapping_is_complete(mapping):
    return all(
        mapping[key] is not None
        for key in [
            "lep_pt",
            "lep_eta",
            "lep_phi",
            "met_et",
            "met_phi",
        ]
    )


# ============================================================
# Event-value helpers
# ============================================================

def as_float(value):
    return float(value)


def calculate_mtw(
    lep_pt,
    lep_phi,
    met_et,
    met_phi,
):
    delta_phi = math.atan2(
        math.sin(lep_phi - met_phi),
        math.cos(lep_phi - met_phi),
    )

    mt2 = (
        2.0
        * lep_pt
        * met_et
        * (
            1.0
            - math.cos(delta_phi)
        )
    )

    return math.sqrt(
        max(mt2, 0.0)
    )


def passes_fiducial(
    lep_pt,
    lep_eta,
    lep_phi,
    met_et,
    met_phi,
):
    if lep_pt <= LEPTON_PT_MIN_GEV:
        return False

    if abs(lep_eta) >= LEPTON_ABS_ETA_MAX:
        return False

    if met_et <= MET_MIN_GEV:
        return False

    mtw = calculate_mtw(
        lep_pt,
        lep_phi,
        met_et,
        met_phi,
    )

    if mtw <= MTW_MIN_GEV:
        return False

    return True


def eta_bin_index(abs_eta):
    index = np.searchsorted(
        ETA_BIN_EDGES,
        abs_eta,
        side="right",
    ) - 1

    if (
        index < 0
        or index >= N_ETA_BINS
    ):
        return -1

    return int(index)


# ============================================================
# Event weight
# ============================================================

def nominal_event_weight(tree):
    """
    Use the nominal generator weight when available.

    For a truth/reco ratio the same event weight is used for both
    levels. No separate particle/parton normalisation is applied.
    """

    if hasattr(tree, "weightvec"):
        try:
            return float(tree.weightvec[0])
        except Exception:
            pass

    if hasattr(tree, "weight"):
        try:
            return float(tree.weight)
        except Exception:
            pass

    return 1.0


# ============================================================
# ROOT histogram helper
# ============================================================

def numpy_to_root_hist(
    name,
    title,
    values,
    errors=None,
):
    edges = array(
        "d",
        ETA_BIN_EDGES.tolist(),
    )

    hist = ROOT.TH1D(
        name,
        title,
        N_ETA_BINS,
        edges,
    )

    hist.SetDirectory(0)

    for ibin in range(N_ETA_BINS):
        hist.SetBinContent(
            ibin + 1,
            float(values[ibin]),
        )

        if errors is not None:
            hist.SetBinError(
                ibin + 1,
                float(errors[ibin]),
            )

    return hist


# ============================================================
# Safe ratio
# ============================================================

def safe_ratio(
    numerator,
    denominator,
):
    out = np.full_like(
        numerator,
        np.nan,
        dtype=float,
    )

    mask = (
        np.abs(denominator) > 0.0
    )

    out[mask] = (
        numerator[mask]
        / denominator[mask]
    )

    return out


# ============================================================
# Paired bootstrap
# ============================================================

def paired_bootstrap_uncertainty(
    truth_bins,
    reco_bins,
    weights,
):
    """
    Estimate the statistical uncertainty on truth/reco using
    paired Poisson resampling.

    Each event receives one Poisson(1) multiplier per bootstrap
    replica. That same multiplier is applied to its truth and
    reco representations, so any truth/reco statistical
    correlation is preserved automatically.
    """

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    n_events = len(weights)

    boot_ratios = np.full(
        (
            N_BOOTSTRAP,
            N_ETA_BINS,
        ),
        np.nan,
        dtype=float,
    )

    # Process replicas one at a time to keep memory usage small.
    for irep in range(N_BOOTSTRAP):

        multiplicity = rng.poisson(
            1.0,
            size=n_events,
        )

        boot_weight = (
            weights
            * multiplicity
        )

        truth_sum = np.zeros(
            N_ETA_BINS,
            dtype=float,
        )

        reco_sum = np.zeros(
            N_ETA_BINS,
            dtype=float,
        )

        truth_mask = (
            truth_bins >= 0
        )

        reco_mask = (
            reco_bins >= 0
        )

        np.add.at(
            truth_sum,
            truth_bins[truth_mask],
            boot_weight[truth_mask],
        )

        np.add.at(
            reco_sum,
            reco_bins[reco_mask],
            boot_weight[reco_mask],
        )

        boot_ratios[irep] = safe_ratio(
            truth_sum,
            reco_sum,
        )

    return np.nanstd(
        boot_ratios,
        axis=0,
        ddof=1,
    )


# ============================================================
# CSV writers
# ============================================================

def write_bin_summary(
    path,
    truth_yield,
    reco_yield,
    correction,
    stat_unc,
    efficiency,
    purity,
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

        writer.writerow([
            "bin",
            "bin_low_edge",
            "bin_up_edge",
            "truth_yield",
            "reco_yield",
            "truth_over_reco",
            "stat_unc_paired_bootstrap",
            "truth_efficiency",
            "reco_purity",
        ])

        for ibin in range(
            N_ETA_BINS
        ):
            writer.writerow([
                ibin + 1,
                ETA_BIN_EDGES[ibin],
                ETA_BIN_EDGES[ibin + 1],
                truth_yield[ibin],
                reco_yield[ibin],
                correction[ibin],
                stat_unc[ibin],
                efficiency[ibin],
                purity[ibin],
            ])


def write_matrix_csv(
    path,
    matrix,
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

        header = [
            "truth_bin",
            "truth_low",
            "truth_high",
        ]

        for jbin in range(
            N_ETA_BINS
        ):
            header.append(
                f"reco_bin_{jbin + 1}"
            )

        writer.writerow(
            header
        )

        for ibin in range(
            N_ETA_BINS
        ):
            writer.writerow(
                [
                    ibin + 1,
                    ETA_BIN_EDGES[ibin],
                    ETA_BIN_EDGES[ibin + 1],
                    *matrix[ibin].tolist(),
                ]
            )


# ============================================================
# Analyse one sample
# ============================================================

def analyse_sample(
    sample_name,
    path,
):
    report()
    report("=" * 76)
    report(f"SAMPLE: {sample_name}")
    report("=" * 76)

    root_file = ROOT.TFile.Open(
        str(path),
        "READ",
    )

    if (
        not root_file
        or root_file.IsZombie()
    ):
        report(
            f"Could not open: {path}"
        )
        return False

    tree = choose_tree(
        root_file,
        path,
    )

    truth_map, reco_map, branches = (
        resolve_level_branches(
            tree
        )
    )

    report(
        f"Tree: {tree.GetName()}"
    )

    report(
        f"Entries: {int(tree.GetEntries()):,}"
    )

    report()
    report("Resolved truth branches:")

    for key, value in truth_map.items():
        report(
            f"  {key:<8} -> {value}"
        )

    report()
    report("Resolved reco branches:")

    for key, value in reco_map.items():
        report(
            f"  {key:<8} -> {value}"
        )

    if (
        not mapping_is_complete(
            truth_map
        )
        or not mapping_is_complete(
            reco_map
        )
    ):
        report()
        report(
            "No complete truth/reco branch pair was found."
        )

        report(
            "This sample is NOT treated as a truth/reco sample."
        )

        interesting = [
            name
            for name in branches
            if (
                "truth" in name.lower()
                or "reco" in name.lower()
                or "lep" in name.lower()
                or "met" in name.lower()
            )
        ]

        report()
        report(
            "Potentially relevant branches present:"
        )

        for name in interesting:
            report(
                f"  {name}"
            )

        root_file.Close()
        return False

    # Guard against accidentally resolving the exact same branch
    # as both truth and reco.
    same_fields = [
        key
        for key in truth_map
        if (
            truth_map[key]
            == reco_map[key]
        )
    ]

    if same_fields:
        report()
        report(
            "Rejected: the same branch was resolved as both truth "
            "and reco for:"
        )

        for key in same_fields:
            report(
                f"  {key}: {truth_map[key]}"
            )

        report(
            "A true truth/reco comparison requires distinct "
            "representations."
        )

        root_file.Close()
        return False

    # --------------------------------------------------------
    # Event storage
    # --------------------------------------------------------

    truth_bins = []
    reco_bins = []
    event_weights = []

    truth_yield = np.zeros(
        N_ETA_BINS,
        dtype=float,
    )

    reco_yield = np.zeros(
        N_ETA_BINS,
        dtype=float,
    )

    truth_sumw2 = np.zeros(
        N_ETA_BINS,
        dtype=float,
    )

    reco_sumw2 = np.zeros(
        N_ETA_BINS,
        dtype=float,
    )

    migration = np.zeros(
        (
            N_ETA_BINS,
            N_ETA_BINS,
        ),
        dtype=float,
    )

    truth_selected_events = np.zeros(
        N_ETA_BINS,
        dtype=float,
    )

    truth_selected_and_reco_selected = np.zeros(
        N_ETA_BINS,
        dtype=float,
    )

    reco_selected_events = np.zeros(
        N_ETA_BINS,
        dtype=float,
    )

    reco_selected_and_same_truth_bin = np.zeros(
        N_ETA_BINS,
        dtype=float,
    )

    both_pass = 0
    truth_only = 0
    reco_only = 0
    neither_pass = 0

    nentries = int(
        tree.GetEntries()
    )

    for ientry in range(
        nentries
    ):
        tree.GetEntry(
            ientry
        )

        weight = nominal_event_weight(
            tree
        )

        truth_pt = as_float(
            getattr(
                tree,
                truth_map["lep_pt"],
            )
        )

        truth_eta = as_float(
            getattr(
                tree,
                truth_map["lep_eta"],
            )
        )

        truth_phi = as_float(
            getattr(
                tree,
                truth_map["lep_phi"],
            )
        )

        truth_met = as_float(
            getattr(
                tree,
                truth_map["met_et"],
            )
        )

        truth_met_phi = as_float(
            getattr(
                tree,
                truth_map["met_phi"],
            )
        )

        reco_pt = as_float(
            getattr(
                tree,
                reco_map["lep_pt"],
            )
        )

        reco_eta = as_float(
            getattr(
                tree,
                reco_map["lep_eta"],
            )
        )

        reco_phi = as_float(
            getattr(
                tree,
                reco_map["lep_phi"],
            )
        )

        reco_met = as_float(
            getattr(
                tree,
                reco_map["met_et"],
            )
        )

        reco_met_phi = as_float(
            getattr(
                tree,
                reco_map["met_phi"],
            )
        )

        truth_pass = passes_fiducial(
            truth_pt,
            truth_eta,
            truth_phi,
            truth_met,
            truth_met_phi,
        )

        reco_pass = passes_fiducial(
            reco_pt,
            reco_eta,
            reco_phi,
            reco_met,
            reco_met_phi,
        )

        truth_bin = (
            eta_bin_index(
                abs(truth_eta)
            )
            if truth_pass
            else -1
        )

        reco_bin = (
            eta_bin_index(
                abs(reco_eta)
            )
            if reco_pass
            else -1
        )

        truth_bins.append(
            truth_bin
        )

        reco_bins.append(
            reco_bin
        )

        event_weights.append(
            weight
        )

        if truth_bin >= 0:
            truth_yield[
                truth_bin
            ] += weight

            truth_sumw2[
                truth_bin
            ] += weight * weight

            truth_selected_events[
                truth_bin
            ] += abs(weight)

        if reco_bin >= 0:
            reco_yield[
                reco_bin
            ] += weight

            reco_sumw2[
                reco_bin
            ] += weight * weight

            reco_selected_events[
                reco_bin
            ] += abs(weight)

        if (
            truth_bin >= 0
            and reco_bin >= 0
        ):
            both_pass += 1

            migration[
                truth_bin,
                reco_bin,
            ] += weight

            truth_selected_and_reco_selected[
                truth_bin
            ] += abs(weight)

            if truth_bin == reco_bin:
                reco_selected_and_same_truth_bin[
                    reco_bin
                ] += abs(weight)

        elif (
            truth_bin >= 0
            and reco_bin < 0
        ):
            truth_only += 1

        elif (
            truth_bin < 0
            and reco_bin >= 0
        ):
            reco_only += 1

        else:
            neither_pass += 1

    root_file.Close()

    truth_bins = np.asarray(
        truth_bins,
        dtype=int,
    )

    reco_bins = np.asarray(
        reco_bins,
        dtype=int,
    )

    event_weights = np.asarray(
        event_weights,
        dtype=float,
    )

    # --------------------------------------------------------
    # Derived quantities
    # --------------------------------------------------------

    correction = safe_ratio(
        truth_yield,
        reco_yield,
    )

    stat_unc = (
        paired_bootstrap_uncertainty(
            truth_bins,
            reco_bins,
            event_weights,
        )
    )

    efficiency = safe_ratio(
        truth_selected_and_reco_selected,
        truth_selected_events,
    )

    purity = safe_ratio(
        reco_selected_and_same_truth_bin,
        reco_selected_events,
    )

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    prefix = OUTPUT_DIR / sample_name

    summary_csv = Path(
        str(prefix)
        + "_truth_reco_summary.csv"
    )

    matrix_csv = Path(
        str(prefix)
        + "_migration_matrix.csv"
    )

    root_output_path = Path(
        str(prefix)
        + "_truth_reco.root"
    )

    write_bin_summary(
        summary_csv,
        truth_yield,
        reco_yield,
        correction,
        stat_unc,
        efficiency,
        purity,
    )

    write_matrix_csv(
        matrix_csv,
        migration,
    )

    # --------------------------------------------------------
    # ROOT output
    # --------------------------------------------------------

    truth_err = np.sqrt(
        truth_sumw2
    )

    reco_err = np.sqrt(
        reco_sumw2
    )

    h_truth = numpy_to_root_hist(
        "etalepton_truth",
        "Truth-level lepton |eta|",
        truth_yield,
        truth_err,
    )

    h_reco = numpy_to_root_hist(
        "etalepton_reco",
        "Reco-level lepton |eta|",
        reco_yield,
        reco_err,
    )

    h_correction = numpy_to_root_hist(
        "etalepton_truth_over_reco",
        "Truth / reco correction factor",
        np.nan_to_num(
            correction,
            nan=0.0,
        ),
        np.nan_to_num(
            stat_unc,
            nan=0.0,
        ),
    )

    h_efficiency = numpy_to_root_hist(
        "etalepton_truth_efficiency",
        "Truth selection efficiency",
        np.nan_to_num(
            efficiency,
            nan=0.0,
        ),
    )

    h_purity = numpy_to_root_hist(
        "etalepton_reco_purity",
        "Reco bin purity",
        np.nan_to_num(
            purity,
            nan=0.0,
        ),
    )

    edges = array(
        "d",
        ETA_BIN_EDGES.tolist(),
    )

    h_migration = ROOT.TH2D(
        "etalepton_migration_matrix",
        "Truth to reco migration matrix",
        N_ETA_BINS,
        edges,
        N_ETA_BINS,
        edges,
    )

    h_migration.SetDirectory(0)

    for i in range(
        N_ETA_BINS
    ):
        for j in range(
            N_ETA_BINS
        ):
            h_migration.SetBinContent(
                i + 1,
                j + 1,
                float(
                    migration[i, j]
                ),
            )

    out_file = ROOT.TFile(
        str(root_output_path),
        "RECREATE",
    )

    out_file.cd()

    h_truth.Write()
    h_reco.Write()
    h_correction.Write()
    h_efficiency.Write()
    h_purity.Write()
    h_migration.Write()

    ROOT.TNamed(
        "CorrectionDefinition",
        "Truth / Reco",
    ).Write()

    ROOT.TNamed(
        "FiducialSelection",
        (
            "pT(l)>20 GeV; |eta(l)|<2.5; "
            "MET>25 GeV; mT(W)>40 GeV"
        ),
    ).Write()

    ROOT.TParameter(
        int
    )(
        "BootstrapReplicas",
        N_BOOTSTRAP,
    ).Write()

    out_file.Write()
    out_file.Close()

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report()
    report(
        "Truth/reco analysis completed."
    )

    report(
        f"  Both pass:   {both_pass:,}"
    )

    report(
        f"  Truth only:  {truth_only:,}"
    )

    report(
        f"  Reco only:   {reco_only:,}"
    )

    report(
        f"  Neither:     {neither_pass:,}"
    )

    report()
    report(
        f"  Summary CSV: {summary_csv}"
    )

    report(
        f"  Matrix CSV:  {matrix_csv}"
    )

    report(
        f"  ROOT output: {root_output_path}"
    )

    return True


# ============================================================
# Main
# ============================================================

def main():

    report()
    report("=" * 76)
    report(" Truth / reco lepton-pseudorapidity analysis")
    report("=" * 76)
    report()

    report(
        f"Project root:\n  {PROJECT_ROOT}"
    )

    report()

    report(
        f"Data directory:\n  {DATA_DIR}"
    )

    report()

    report(
        "Correction definition: Truth / Reco"
    )

    report(
        "Fiducial cuts: pT(l)>20 GeV, |eta(l)|<2.5, "
        "MET>25 GeV, mT(W)>40 GeV"
    )

    report()

    missing = [
        path
        for path in ROOT_FILES.values()
        if not path.is_file()
    ]

    if missing:
        report(
            "Missing ROOT files:"
        )

        for path in missing:
            report(
                f"  {path}"
            )

        report()

    successful = 0

    for (
        sample_name,
        path,
    ) in ROOT_FILES.items():

        if not path.is_file():
            continue

        if analyse_sample(
            sample_name,
            path,
        ):
            successful += 1

    report()
    report("=" * 76)
    report(" FINISHED")
    report("=" * 76)
    report()

    if successful == 0:
        report(
            "No file contained a complete, distinct truth/reco "
            "branch set that this script could identify."
        )

        report(
            "This means the current ntuples cannot yet be used "
            "for a genuine truth/reco correction with the branch "
            "names searched here. Inspect the diagnostic branch "
            "lists in this report before changing the aliases."
        )

        report()

        report(
            "Do NOT substitute the existing particle-level sample "
            "for reco level merely to force a truth/reco result."
        )

    else:
        report(
            f"Completed truth/reco analysis for "
            f"{successful} sample(s)."
        )

        report()

        report(
            "The paired-bootstrap statistical uncertainty preserves "
            "the truth/reco event correlation by resampling each "
            "event once and applying the same bootstrap multiplier "
            "to both representations."
        )

    report()

    report(
        f"Report:\n  {REPORT_PATH}"
    )

    report()

    REPORT_PATH.write_text(
        "\n".join(
            REPORT_LINES
        )
        + "\n",
        encoding="utf-8",
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
