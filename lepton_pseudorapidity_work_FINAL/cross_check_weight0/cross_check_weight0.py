#!/usr/bin/env python3

# ============================================================
# cross_check_weight0.py
#
# High-statistics Pythia weight-0 cross-check / nominal builder
# for the W+c lepton-pseudorapidity correction factor.
#
# REPAIRED EVENT DEFINITION
# -------------------------
# This version uses the SAME W+c event definition as:
#
#   MTWcut_make_histograms_etalepton.py
#   even_odd_uncertainty.py
#
# Selection:
#
#   MET > 25 GeV
#   lepton pT > 20 GeV
#   |eta_lepton| < 2.5
#   m_T^W > 40 GeV
#   jet pT > 25 GeV
#   |eta_jet| < 2.5
#   exactly one charm-identified fiducial jet
#
# Charm identification in these WCharmTree ntuples:
#
#   jet_charge == 0   -> not charm identified
#   jet_charge != 0   -> charm identified
#
# OS-SS event sign:
#
#   opposite lepton/charm signs -> +1
#   same lepton/charm signs     -> -1
#
# Extra non-charm jets are allowed.
#
# The high-statistics files are fragmented into many ROOT files.
# Fragments are combined into one logical sample and normalized
# using the total number of entries across all fragments.
#
# The script ALWAYS writes:
#
#   outputs/csv/cross_check_Pythia_plus.csv
#   outputs/csv/cross_check_Pythia_minus.csv
#
# with the downstream-required columns:
#
#   C_new_crosscheck
#   new_crosscheck_stat_unc
#
# REFERENCE-SAFETY RULE
# ---------------------
# A reference comparison is performed ONLY if the repaired
# reference CSV exists in the SAME FINAL working tree:
#
#   ../MTWcut_build_corrections_etalepton_csv/
#
# This prevents the repaired high-statistics correction from being
# silently compared with an old pre-repair correction.
#
# If the repaired reference is not yet present, the high-statistics
# correction is still produced and saved; reference-only columns are
# written as NaN and comparison plots are deferred.
#
# ============================================================

import csv
import math
from array import array
from pathlib import Path

import ROOT


# ============================================================
# ROOT settings
# ============================================================

ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)


# ============================================================
# Directory structure
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
LEPTON_WORK_DIR = SCRIPT_DIR.parent

CROSS_CHECK_ROOT_DIR = (
    SCRIPT_DIR
    / "cross_check_ROOT_files"
)

# IMPORTANT:
# This intentionally points only inside the SAME working tree.
# When this script is copied under lepton_pseudorapidity_work_FINAL,
# it will never fall back to the old unrepaired folder.
REFERENCE_CSV_DIR = (
    LEPTON_WORK_DIR
    / "MTWcut_build_corrections_etalepton_csv"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "outputs"
)

ROOT_OUTPUT_DIR = (
    OUTPUT_DIR
    / "root"
)

CSV_OUTPUT_DIR = (
    OUTPUT_DIR
    / "csv"
)

PLOT_OUTPUT_DIR = (
    OUTPUT_DIR
    / "plots"
)

INTERACTIVE_OUTPUT_DIR = (
    OUTPUT_DIR
    / "interactive"
)

for directory in (
    ROOT_OUTPUT_DIR,
    CSV_OUTPUT_DIR,
    PLOT_OUTPUT_DIR,
    INTERACTIVE_OUTPUT_DIR,
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
REQUIRED_CHARM_JETS = 1

NOMINAL_WEIGHT_INDEX = 0

JET_CHARGE_BRANCH = "jet_charge"


# ============================================================
# Lepton pseudorapidity binning
# ============================================================

ETA_LEPTON_BINS = [
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
]

ETA_LEPTON_BIN_ARRAY = array(
    "d",
    ETA_LEPTON_BINS,
)


# ============================================================
# High-statistics sample definitions
# ============================================================

NEW_SAMPLE_PATTERNS = {
    "WCPy8plus": (
        "WCharm_WCPy8plus*.root"
    ),
    "WCPy8minus": (
        "WCharm_WCPy8minus*.root"
    ),
    "WCPyPartonplus": (
        "WCharm_WCPyPartonplus*.root"
    ),
    "WCPyPartonminus": (
        "WCharm_WCPyPartonminus*.root"
    ),
}


PAIR_DEFINITIONS = {
    "Pythia_plus": {
        "particle": "WCPy8plus",
        "parton": "WCPyPartonplus",
        "reference_csv": (
            "Pythia_plus_etalepton.csv"
        ),
        "display": "Pythia W+",
    },
    "Pythia_minus": {
        "particle": "WCPy8minus",
        "parton": "WCPyPartonminus",
        "reference_csv": (
            "Pythia_minus_etalepton.csv"
        ),
        "display": "Pythia W-",
    },
}


# ============================================================
# Generic ROOT/vector helpers
# ============================================================

def tree_branch_names(tree):
    """Return all branch names in a TTree."""
    return {
        branch.GetName()
        for branch in tree.GetListOfBranches()
    }


def sequence_length(value):
    """Length helper for ROOT vector-like values."""
    try:
        return len(value)
    except TypeError:
        return 1


def sequence_value(value, index):
    """Read one item from a ROOT vector/array-like value."""
    try:
        return value[index]
    except TypeError:
        if index != 0:
            raise IndexError(index)
        return value


def validate_required_branches(
    tree,
    path,
):
    """
    Fail if a fragment is missing a branch required by the repaired
    W+c selection or by the nominal weighting.
    """
    required = {
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
        - tree_branch_names(tree)
    )

    if missing:
        raise RuntimeError(
            f"{path.name}: WCharmTree is missing required branches: "
            + ", ".join(missing)
        )


def validate_jet_charge_encoding(
    tree,
    path,
    entries_to_check=100,
):
    """
    Check that jet_pt, jet_eta and jet_charge are aligned and that
    the observed charm-tag encoding is {-1, 0, +1}.
    """
    ncheck = min(
        int(tree.GetEntries()),
        int(entries_to_check),
    )

    for ientry in range(ncheck):

        tree.GetEntry(ientry)

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
                f"{path.name}: per-jet vector length mismatch at "
                f"entry {ientry}: "
                f"len(jet_pt)={n_pt}, "
                f"len(jet_eta)={n_eta}, "
                f"len(jet_charge)={n_charge}."
            )

        for jet_index in range(n_charge):

            value = float(
                sequence_value(
                    tree.jet_charge,
                    jet_index,
                )
            )

            if value not in (
                -1.0,
                0.0,
                1.0,
            ):
                raise RuntimeError(
                    f"{path.name}: unexpected jet_charge={value} "
                    f"at entry {ientry}, jet {jet_index}. "
                    "Expected -1, 0 or +1."
                )


# ============================================================
# Physics helpers
# ============================================================

def calculate_mtw(tree):
    """
    Reconstructed W transverse mass:

        m_T^W = sqrt[
            2 pT_lepton MET (1 - cos DeltaPhi)
        ]
    """
    delta_phi = (
        float(tree.leptons_phi)
        - float(tree.met_phi)
    )

    mtw_squared = (
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
            mtw_squared,
        )
    )


def fiducial_charm_jets(tree):
    """
    Return all charm-identified jets in the fiducial jet region.

    Each result is:
        (jet_index, charm_sign)

    charm_sign is +/-1 from jet_charge.
    """
    jet_pts = tree.jet_pt
    jet_etas = tree.jet_eta
    jet_charges = tree.jet_charge

    n_pt = sequence_length(
        jet_pts
    )
    n_eta = sequence_length(
        jet_etas
    )
    n_charge = sequence_length(
        jet_charges
    )

    if not (
        n_pt == n_eta == n_charge
    ):
        raise RuntimeError(
            "Per-jet branch-length mismatch while processing event: "
            f"len(jet_pt)={n_pt}, "
            f"len(jet_eta)={n_eta}, "
            f"len(jet_charge)={n_charge}."
        )

    selected = []

    for jet_index in range(n_pt):

        jet_pt = float(
            sequence_value(
                jet_pts,
                jet_index,
            )
        )

        jet_eta = float(
            sequence_value(
                jet_etas,
                jet_index,
            )
        )

        jet_charge = float(
            sequence_value(
                jet_charges,
                jet_index,
            )
        )

        if jet_pt <= JET_PT_MIN_GEV:
            continue

        if abs(
            jet_eta
        ) >= JET_ABS_ETA_MAX:
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


def os_ss_charge_weight(
    lepton_charge,
    charm_sign,
):
    """
    W+c OS-SS event sign.

      opposite lepton/charm signs -> +1
      same lepton/charm signs     -> -1
    """
    product = (
        float(lepton_charge)
        * float(charm_sign)
    )

    return (
        -1.0
        if product > 0.0
        else 1.0
    )


def event_selection_text():
    return (
        "met_et > 25 GeV; "
        "leptons_pt > 20 GeV; "
        "abs(leptons_eta) < 2.5; "
        "m_T^W > 40 GeV; "
        "jet_pt > 25 GeV; "
        "abs(jet_eta) < 2.5; "
        "exactly one charm-identified fiducial jet "
        "(jet_charge != 0); "
        "OS-SS sign from leptons_charge * jet_charge"
    )


# ============================================================
# Histogram helpers
# ============================================================

def make_eta_histogram(sample):
    """Create one detached weighted |eta_lepton| histogram."""

    histogram = ROOT.TH1F(
        f"etalepton_{sample}",
        (
            f"{sample};"
            f"|#eta_{{#ell}}|;"
            f"Weighted OS-SS events"
        ),
        len(
            ETA_LEPTON_BIN_ARRAY
        ) - 1,
        ETA_LEPTON_BIN_ARRAY,
    )

    histogram.Sumw2()
    histogram.SetDirectory(0)

    return histogram


# ============================================================
# File discovery / input opening
# ============================================================

def discover_files():
    """
    Discover all high-statistics weight-0 fragments belonging to
    each logical Pythia particle/parton sample.
    """

    if not CROSS_CHECK_ROOT_DIR.is_dir():
        raise RuntimeError(
            "Cross-check ROOT directory does not exist:\n"
            f"  {CROSS_CHECK_ROOT_DIR}"
        )

    discovered = {}

    for (
        sample,
        pattern,
    ) in NEW_SAMPLE_PATTERNS.items():

        discovered[
            sample
        ] = sorted(
            CROSS_CHECK_ROOT_DIR.glob(
                pattern
            )
        )

    return discovered


def get_tree_from_file(path):
    """
    Open a ROOT fragment and retrieve WCharmTree.

    Caller owns the TFile and must close it.
    """

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

    return (
        root_file,
        tree,
    )


def count_total_entries(paths):
    """Total entries across all fragments of one logical sample."""

    total_entries = 0

    for path in paths:

        (
            root_file,
            tree,
        ) = get_tree_from_file(
            path
        )

        total_entries += int(
            tree.GetEntries()
        )

        root_file.Close()

    return total_entries


# ============================================================
# Process one combined high-statistics sample
# ============================================================

def process_combined_sample(
    sample,
    paths,
):
    """
    Combine all fragments for one sample into one nominal histogram.

    Event normalization:

        weightvec[0] / N_total_combined

    with OS-SS sign applied after the complete repaired W+c selection.
    """

    if not paths:
        return None

    total_entries = (
        count_total_entries(
            paths
        )
    )

    if total_entries <= 0:
        raise RuntimeError(
            f"Combined sample {sample} has no events."
        )

    histogram = (
        make_eta_histogram(
            sample
        )
    )

    events_after_lepton_met_cuts = 0
    events_after_mtw_cut = 0
    events_after_wcjet_selection = 0

    opposite_sign_events = 0
    same_sign_events = 0

    print()
    print(
        "------------------------------------------------------------"
    )
    print(
        f"Processing combined sample: {sample}"
    )
    print(
        "------------------------------------------------------------"
    )
    print(
        f"  ROOT fragments: {len(paths)}"
    )
    print(
        f"  Total entries:  {total_entries}"
    )

    for file_number, path in enumerate(
        paths,
        start=1,
    ):

        (
            root_file,
            tree,
        ) = get_tree_from_file(
            path
        )

        nentries = int(
            tree.GetEntries()
        )

        if nentries <= 0:
            root_file.Close()
            continue

        validate_required_branches(
            tree,
            path,
        )

        validate_jet_charge_encoding(
            tree,
            path,
        )

        tree.GetEntry(0)

        if (
            len(tree.weightvec)
            <= NOMINAL_WEIGHT_INDEX
        ):
            root_file.Close()
            raise RuntimeError(
                f"{path.name}: weightvec[0] is unavailable."
            )

        print(
            f"  [{file_number:3d}/{len(paths):3d}] "
            f"{path.name}: {nentries} entries"
        )

        for ientry in range(
            nentries
        ):

            tree.GetEntry(
                ientry
            )

            # ------------------------------------------------
            # Lepton / MET fiducial selection
            # ------------------------------------------------

            if (
                float(tree.met_et)
                <= MET_MIN_GEV
            ):
                continue

            if (
                float(tree.leptons_pt)
                <= LEPTON_PT_MIN_GEV
            ):
                continue

            if (
                abs(
                    float(
                        tree.leptons_eta
                    )
                )
                >= LEPTON_ABS_ETA_MAX
            ):
                continue

            events_after_lepton_met_cuts += 1

            # ------------------------------------------------
            # W transverse mass
            # ------------------------------------------------

            if (
                calculate_mtw(tree)
                <= MTW_MIN_GEV
            ):
                continue

            events_after_mtw_cut += 1

            # ------------------------------------------------
            # RESTORED W+c jet requirement
            # ------------------------------------------------

            charm_jets = (
                fiducial_charm_jets(
                    tree
                )
            )

            if (
                len(charm_jets)
                != REQUIRED_CHARM_JETS
            ):
                continue

            events_after_wcjet_selection += 1

            _, charm_sign = (
                charm_jets[0]
            )

            # ------------------------------------------------
            # RESTORED OS-SS sign
            # ------------------------------------------------

            charge_factor = (
                os_ss_charge_weight(
                    tree.leptons_charge,
                    charm_sign,
                )
            )

            if charge_factor > 0.0:
                opposite_sign_events += 1
            else:
                same_sign_events += 1

            eta_lepton = abs(
                float(
                    tree.leptons_eta
                )
            )

            event_weight = (
                float(
                    tree.weightvec[
                        NOMINAL_WEIGHT_INDEX
                    ]
                )
                / float(total_entries)
            )

            final_weight = (
                charge_factor
                * event_weight
            )

            histogram.Fill(
                eta_lepton,
                final_weight,
            )

        root_file.Close()

    if (
        opposite_sign_events
        + same_sign_events
        != events_after_wcjet_selection
    ):
        raise RuntimeError(
            f"{sample}: OS/SS event counts do not sum to the "
            "accepted W+c event count."
        )

    print(
        f"  After lepton/MET cuts: "
        f"{events_after_lepton_met_cuts}"
    )
    print(
        f"  After mT cut:          "
        f"{events_after_mtw_cut}"
    )
    print(
        f"  After W+c jet cuts:    "
        f"{events_after_wcjet_selection}"
    )
    print(
        f"    opposite-sign:       "
        f"{opposite_sign_events}"
    )
    print(
        f"    same-sign:           "
        f"{same_sign_events}"
    )

    return {
        "sample": sample,
        "paths": paths,
        "total_entries": total_entries,
        "events_after_lepton_met_cuts": (
            events_after_lepton_met_cuts
        ),
        "events_after_mtw_cut": (
            events_after_mtw_cut
        ),
        "events_after_wcjet_selection": (
            events_after_wcjet_selection
        ),
        "opposite_sign_events": (
            opposite_sign_events
        ),
        "same_sign_events": (
            same_sign_events
        ),
        "histogram": histogram,
    }


# ============================================================
# Correction factor
# ============================================================

def make_correction_histogram(
    h_particle,
    h_parton,
    name,
):
    """
    Construct:
        C = parton / particle
    """

    ratio = (
        h_parton.Clone(name)
    )

    ratio.SetDirectory(0)

    ratio.Divide(
        h_parton,
        h_particle,
        1.0,
        1.0,
        "",
    )

    return ratio


# ============================================================
# Reference CSV helpers
# ============================================================

def repaired_reference_path(
    definition,
):
    """Path to the repaired reference correction CSV."""

    return (
        REFERENCE_CSV_DIR
        / definition[
            "reference_csv"
        ]
    )


def read_reference_csv(path):
    """
    Read a repaired nominal correction CSV.

    Expected columns match MTWcut_build_corrections_etalepton.py.
    """

    if not path.is_file():
        return None

    rows = []

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        reader = csv.DictReader(
            csvfile
        )

        required = {
            "bin",
            "bin_low_edge",
            "bin_up_edge",
            "correction_factor",
            "stat_unc",
        }

        missing = (
            required
            - set(
                reader.fieldnames
                or []
            )
        )

        if missing:
            raise RuntimeError(
                f"Reference CSV {path} is missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:

            rows.append({
                "bin": int(
                    row["bin"]
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
                "stat_unc": float(
                    row[
                        "stat_unc"
                    ]
                ),
            })

    return rows


def validate_reference_binning(
    reference_rows,
    correction_hist,
):
    """Ensure repaired reference and new correction use same bins."""

    nbins = (
        correction_hist.GetNbinsX()
    )

    if len(
        reference_rows
    ) != nbins:
        raise RuntimeError(
            "Reference/new bin-count mismatch:\n"
            f"  reference: {len(reference_rows)}\n"
            f"  new:       {nbins}"
        )

    tolerance = 1.0e-8

    for ibin in range(
        1,
        nbins + 1,
    ):

        reference = (
            reference_rows[
                ibin - 1
            ]
        )

        low_edge = (
            correction_hist
            .GetXaxis()
            .GetBinLowEdge(
                ibin
            )
        )

        up_edge = (
            correction_hist
            .GetXaxis()
            .GetBinUpEdge(
                ibin
            )
        )

        if (
            abs(
                low_edge
                - reference[
                    "bin_low_edge"
                ]
            ) > tolerance
            or
            abs(
                up_edge
                - reference[
                    "bin_up_edge"
                ]
            ) > tolerance
        ):
            raise RuntimeError(
                "Reference/new binning mismatch in "
                f"bin {ibin}."
            )


# ============================================================
# Output rows
# ============================================================

COMPARISON_COLUMNS = [
    "pair_label",
    "bin",
    "bin_low_edge",
    "bin_up_edge",
    "C_reference",
    "reference_stat_unc",
    "C_new_crosscheck",
    "new_crosscheck_stat_unc",
    "new_minus_reference",
    "abs_new_minus_reference",
    "relative_abs_difference",
    "relative_abs_difference_percent",
    "combined_stat_unc_if_independent",
    "difference_over_combined_stat_if_independent",
]


def build_output_rows(
    pair_label,
    correction_hist,
    reference_rows=None,
):
    """
    Build output rows.

    If reference_rows is None, all new high-statistics quantities are
    still written and reference-dependent fields are NaN.
    """

    if reference_rows is not None:
        validate_reference_binning(
            reference_rows,
            correction_hist,
        )

    output_rows = []

    for ibin in range(
        1,
        correction_hist.GetNbinsX() + 1,
    ):

        low_edge = (
            correction_hist
            .GetXaxis()
            .GetBinLowEdge(
                ibin
            )
        )

        up_edge = (
            correction_hist
            .GetXaxis()
            .GetBinUpEdge(
                ibin
            )
        )

        new_correction = float(
            correction_hist.GetBinContent(
                ibin
            )
        )

        new_stat_unc = float(
            correction_hist.GetBinError(
                ibin
            )
        )

        if reference_rows is None:

            old_correction = math.nan
            old_stat_unc = math.nan
            difference = math.nan
            absolute_difference = math.nan
            relative_difference = math.nan
            combined_stat_unc = math.nan
            difference_over_combined_stat = math.nan

        else:

            reference = (
                reference_rows[
                    ibin - 1
                ]
            )

            old_correction = float(
                reference[
                    "correction_factor"
                ]
            )

            old_stat_unc = float(
                reference[
                    "stat_unc"
                ]
            )

            difference = (
                new_correction
                - old_correction
            )

            absolute_difference = abs(
                difference
            )

            relative_difference = (
                absolute_difference
                / abs(old_correction)
                if old_correction != 0.0
                else 0.0
            )

            combined_stat_unc = math.sqrt(
                old_stat_unc
                * old_stat_unc
                + new_stat_unc
                * new_stat_unc
            )

            difference_over_combined_stat = (
                difference
                / combined_stat_unc
                if combined_stat_unc > 0.0
                else 0.0
            )

        output_rows.append({
            "pair_label": pair_label,
            "bin": ibin,
            "bin_low_edge": low_edge,
            "bin_up_edge": up_edge,
            "C_reference": old_correction,
            "reference_stat_unc": old_stat_unc,
            "C_new_crosscheck": new_correction,
            "new_crosscheck_stat_unc": new_stat_unc,
            "new_minus_reference": difference,
            "abs_new_minus_reference": (
                absolute_difference
            ),
            "relative_abs_difference": (
                relative_difference
            ),
            "relative_abs_difference_percent": (
                100.0 * relative_difference
                if math.isfinite(
                    relative_difference
                )
                else math.nan
            ),
            "combined_stat_unc_if_independent": (
                combined_stat_unc
            ),
            "difference_over_combined_stat_if_independent": (
                difference_over_combined_stat
            ),
        })

    return output_rows


def write_output_csv(
    pair_label,
    rows,
):
    """
    Write the cross-check CSV.

    This filename/schema is retained for compatibility with
    apply_relative_uncertainties_to_new_nominal.py.
    """

    output_path = (
        CSV_OUTPUT_DIR
        / f"cross_check_{pair_label}.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=COMPARISON_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(rows)

    return output_path


# ============================================================
# Reference histogram / ROOT pair output
# ============================================================

def make_reference_histogram(
    reference_rows,
    name,
):
    """Convert repaired reference CSV rows to a ROOT histogram."""

    histogram = ROOT.TH1F(
        name,
        name,
        len(
            ETA_LEPTON_BIN_ARRAY
        ) - 1,
        ETA_LEPTON_BIN_ARRAY,
    )

    histogram.SetDirectory(0)

    for row in reference_rows:

        ibin = int(
            row["bin"]
        )

        histogram.SetBinContent(
            ibin,
            float(
                row[
                    "correction_factor"
                ]
            ),
        )

        histogram.SetBinError(
            ibin,
            float(
                row[
                    "stat_unc"
                ]
            ),
        )

    return histogram


def write_pair_root_file(
    pair_label,
    particle_hist,
    parton_hist,
    new_correction_hist,
    reference_hist=None,
):
    """Save principal high-statistics cross-check objects."""

    output_path = (
        ROOT_OUTPUT_DIR
        / f"cross_check_{pair_label}.root"
    )

    output_file = ROOT.TFile(
        str(output_path),
        "RECREATE",
    )

    if (
        not output_file
        or output_file.IsZombie()
    ):
        raise RuntimeError(
            f"Could not create ROOT file:\n"
            f"  {output_path}"
        )

    ROOT.TNamed(
        "PairLabel",
        pair_label,
    ).Write()

    ROOT.TNamed(
        "Observable",
        "etalepton",
    ).Write()

    ROOT.TNamed(
        "Weight",
        "weightvec[0] only",
    ).Write()

    ROOT.TNamed(
        "EventSelection",
        event_selection_text(),
    ).Write()

    ROOT.TNamed(
        "CharmJetIDBranch",
        JET_CHARGE_BRANCH,
    ).Write()

    ROOT.TNamed(
        "OSSSWeightDefinition",
        (
            "+1 for opposite-sign lepton/charm; "
            "-1 for same-sign lepton/charm"
        ),
    ).Write()

    ROOT.TNamed(
        "ReferenceIncluded",
        (
            "True"
            if reference_hist is not None
            else "False"
        ),
    ).Write()

    particle_hist.Write(
        "new_particle_etalepton"
    )

    parton_hist.Write(
        "new_parton_etalepton"
    )

    new_correction_hist.Write(
        "new_crosscheck_correction"
    )

    if reference_hist is not None:
        reference_hist.Write(
            "reference_correction"
        )

    output_file.Write()
    output_file.Close()

    return output_path


# ============================================================
# Combined sample ROOT output
# ============================================================

def write_combined_sample_root(
    sample_results,
):
    """Save all available high-statistics sample histograms."""

    output_path = (
        ROOT_OUTPUT_DIR
        / "combined_new_weight0_samples.root"
    )

    output_file = ROOT.TFile(
        str(output_path),
        "RECREATE",
    )

    if (
        not output_file
        or output_file.IsZombie()
    ):
        raise RuntimeError(
            f"Could not create ROOT file:\n"
            f"  {output_path}"
        )

    ROOT.TNamed(
        "EventSelection",
        event_selection_text(),
    ).Write()

    ROOT.TNamed(
        "Weight",
        "weightvec[0] only",
    ).Write()

    for sample, result in (
        sample_results.items()
    ):

        if result is None:
            continue

        result[
            "histogram"
        ].Write(
            f"{sample}_etalepton"
        )

    output_file.Write()
    output_file.Close()

    return output_path


# ============================================================
# Static comparison plot
# ============================================================

def save_static_plot(
    pair_label,
    display_label,
    reference_hist,
    new_hist,
    output_rows,
):
    """
    Save repaired reference vs high-statistics correction comparison.

    Called only when the repaired reference CSV exists.
    """

    canvas = ROOT.TCanvas(
        f"canvas_{pair_label}",
        display_label,
        1000,
        850,
    )

    top_pad = ROOT.TPad(
        f"top_{pair_label}",
        "top",
        0.0,
        0.34,
        1.0,
        1.0,
    )

    bottom_pad = ROOT.TPad(
        f"bottom_{pair_label}",
        "bottom",
        0.0,
        0.0,
        1.0,
        0.34,
    )

    top_pad.SetBottomMargin(
        0.03
    )
    bottom_pad.SetTopMargin(
        0.04
    )
    bottom_pad.SetBottomMargin(
        0.28
    )

    top_pad.Draw()
    bottom_pad.Draw()

    # Top panel.
    top_pad.cd()
    top_pad.SetGrid()

    reference_hist.SetLineColor(
        ROOT.kBlack
    )
    reference_hist.SetMarkerColor(
        ROOT.kBlack
    )
    reference_hist.SetMarkerStyle(
        20
    )
    reference_hist.SetLineWidth(
        2
    )

    new_hist.SetLineColor(
        ROOT.kBlue + 1
    )
    new_hist.SetMarkerColor(
        ROOT.kBlue + 1
    )
    new_hist.SetMarkerStyle(
        21
    )
    new_hist.SetLineWidth(
        2
    )

    reference_hist.SetTitle(
        (
            f"{display_label}: repaired weight-0 cross-check;"
            f"|#eta_{{#ell}}|;"
            f"Correction factor"
        )
    )

    values = []

    for histogram in (
        reference_hist,
        new_hist,
    ):

        for ibin in range(
            1,
            histogram.GetNbinsX() + 1,
        ):

            value = float(
                histogram.GetBinContent(
                    ibin
                )
            )

            if math.isfinite(
                value
            ):
                values.append(
                    value
                )

    if values:

        minimum = min(values)
        maximum = max(values)

        spread = max(
            maximum - minimum,
            0.01,
        )

        reference_hist.SetMinimum(
            minimum
            - 0.25 * spread
        )

        reference_hist.SetMaximum(
            maximum
            + 0.25 * spread
        )

    reference_hist.Draw(
        "E1"
    )
    new_hist.Draw(
        "E1 SAME"
    )

    legend = ROOT.TLegend(
        0.55,
        0.74,
        0.88,
        0.88,
    )

    legend.AddEntry(
        reference_hist,
        "Repaired weighted-sample nominal",
        "lep",
    )

    legend.AddEntry(
        new_hist,
        "New high-stat weight-0",
        "lep",
    )

    legend.Draw()

    # Bottom panel.
    bottom_pad.cd()
    bottom_pad.SetGrid()

    difference_hist = ROOT.TH1F(
        f"difference_{pair_label}",
        (
            ";"
            "|#eta_{#ell}|;"
            "C_{new} - C_{ref}"
        ),
        len(
            ETA_LEPTON_BIN_ARRAY
        ) - 1,
        ETA_LEPTON_BIN_ARRAY,
    )

    difference_hist.SetDirectory(0)

    for row in output_rows:

        ibin = int(
            row["bin"]
        )

        difference_hist.SetBinContent(
            ibin,
            float(
                row[
                    "new_minus_reference"
                ]
            ),
        )

        difference_hist.SetBinError(
            ibin,
            float(
                row[
                    "combined_stat_unc_if_independent"
                ]
            ),
        )

    difference_hist.SetLineColor(
        ROOT.kBlack
    )
    difference_hist.SetMarkerColor(
        ROOT.kBlack
    )
    difference_hist.SetMarkerStyle(
        20
    )

    difference_hist.Draw(
        "E1"
    )

    zero_line = ROOT.TLine(
        ETA_LEPTON_BINS[0],
        0.0,
        ETA_LEPTON_BINS[-1],
        0.0,
    )

    zero_line.SetLineStyle(
        2
    )
    zero_line.Draw()

    canvas.cd()

    png_path = (
        PLOT_OUTPUT_DIR
        / f"cross_check_{pair_label}.png"
    )

    pdf_path = (
        PLOT_OUTPUT_DIR
        / f"cross_check_{pair_label}.pdf"
    )

    canvas.SaveAs(
        str(png_path)
    )
    canvas.SaveAs(
        str(pdf_path)
    )

    canvas.Close()

    return (
        png_path,
        pdf_path,
    )


# ============================================================
# Optional Plotly comparison
# ============================================================

def save_interactive_plot(
    completed_reference_comparisons,
):
    """
    Write a simple interactive repaired-reference comparison.

    If no repaired references exist yet, no HTML is produced.
    """

    if not completed_reference_comparisons:
        return None

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

    except ImportError:

        print()
        print(
            "Plotly is not installed; skipping interactive HTML."
        )

        return None

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Repaired reference and high-stat correction",
            "High-stat minus repaired reference",
        ),
        row_heights=[
            0.66,
            0.34,
        ],
    )

    pair_keys = list(
        completed_reference_comparisons.keys()
    )

    traces_per_pair = 3

    for pair_index, pair_label in enumerate(
        pair_keys
    ):

        result = (
            completed_reference_comparisons[
                pair_label
            ]
        )

        rows = result[
            "rows"
        ]

        visible = (
            pair_index == 0
        )

        x = [
            0.5
            * (
                row["bin_low_edge"]
                + row["bin_up_edge"]
            )
            for row in rows
        ]

        xerr = [
            0.5
            * (
                row["bin_up_edge"]
                - row["bin_low_edge"]
            )
            for row in rows
        ]

        figure.add_trace(
            go.Scatter(
                x=x,
                y=[
                    row[
                        "C_reference"
                    ]
                    for row in rows
                ],
                error_x=dict(
                    type="data",
                    array=xerr,
                    visible=True,
                ),
                error_y=dict(
                    type="data",
                    array=[
                        row[
                            "reference_stat_unc"
                        ]
                        for row in rows
                    ],
                    visible=True,
                ),
                mode="lines+markers",
                name=(
                    "Repaired weighted-sample nominal"
                ),
                visible=visible,
            ),
            row=1,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=x,
                y=[
                    row[
                        "C_new_crosscheck"
                    ]
                    for row in rows
                ],
                error_x=dict(
                    type="data",
                    array=xerr,
                    visible=True,
                ),
                error_y=dict(
                    type="data",
                    array=[
                        row[
                            "new_crosscheck_stat_unc"
                        ]
                        for row in rows
                    ],
                    visible=True,
                ),
                mode="lines+markers",
                name="New high-stat weight-0",
                visible=visible,
            ),
            row=1,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=x,
                y=[
                    row[
                        "new_minus_reference"
                    ]
                    for row in rows
                ],
                error_y=dict(
                    type="data",
                    array=[
                        row[
                            "combined_stat_unc_if_independent"
                        ]
                        for row in rows
                    ],
                    visible=True,
                ),
                mode="lines+markers",
                name="New - reference",
                visible=visible,
            ),
            row=2,
            col=1,
        )

    total_traces = (
        traces_per_pair
        * len(pair_keys)
    )

    buttons = []

    for pair_index, pair_label in enumerate(
        pair_keys
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
            completed_reference_comparisons[
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
                            "Repaired high-stat weight-0 cross-check"
                            f"<br><sup>{display}</sup>"
                        ),
                    },
                ],
            )
        )

    first_display = (
        completed_reference_comparisons[
            pair_keys[0]
        ][
            "display"
        ]
    )

    figure.update_layout(
        title=dict(
            text=(
                "Repaired high-stat weight-0 cross-check"
                f"<br><sup>{first_display}</sup>"
            ),
            x=0.5,
        ),
        template="plotly_white",
        height=850,
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
                y=1.16,
            )
        ],
    )

    figure.update_yaxes(
        title_text="Correction factor",
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text="C_new - C_reference",
        row=2,
        col=1,
        zeroline=True,
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
        range=[
            0.0,
            2.5,
        ],
        row=1,
        col=1,
    )

    output_path = (
        INTERACTIVE_OUTPUT_DIR
        / "cross_check_weight0_interactive.html"
    )

    figure.write_html(
        str(output_path),
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
    )

    return output_path


# ============================================================
# Run summary
# ============================================================

def write_run_summary(
    discovered,
    sample_results,
    pair_results,
    completed_reference_comparisons,
):
    """Write a compact record of this repaired production."""

    output_path = (
        OUTPUT_DIR
        / "run_summary.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Repaired high-statistics Pythia weight-0 cross-check\n"
        )
        f.write(
            "==================================================\n\n"
        )

        f.write(
            "Selection:\n"
        )
        f.write(
            f"  MET > {MET_MIN_GEV} GeV\n"
        )
        f.write(
            f"  lepton pT > "
            f"{LEPTON_PT_MIN_GEV} GeV\n"
        )
        f.write(
            f"  |eta_lepton| < "
            f"{LEPTON_ABS_ETA_MAX}\n"
        )
        f.write(
            f"  m_T^W > "
            f"{MTW_MIN_GEV} GeV\n"
        )
        f.write(
            f"  jet pT > "
            f"{JET_PT_MIN_GEV} GeV\n"
        )
        f.write(
            f"  |eta_jet| < "
            f"{JET_ABS_ETA_MAX}\n"
        )
        f.write(
            "  exactly one charm-identified fiducial jet\n"
        )
        f.write(
            "  charm ID: jet_charge != 0\n"
        )
        f.write(
            "  extra non-charm jets allowed\n"
        )
        f.write(
            "  OS = +1, SS = -1 from "
            "leptons_charge * jet_charge\n\n"
        )

        f.write(
            "Weight:\n"
        )
        f.write(
            "  weightvec[0] only\n\n"
        )

        f.write(
            "Reference policy:\n"
        )
        f.write(
            "  only repaired reference CSVs in the same working "
            "tree are accepted\n"
        )
        f.write(
            f"  reference directory: {REFERENCE_CSV_DIR}\n\n"
        )

        f.write(
            "Discovered ROOT fragments:\n"
        )

        for sample in (
            NEW_SAMPLE_PATTERNS
        ):

            f.write(
                f"  {sample}: "
                f"{len(discovered[sample])}\n"
            )

        f.write(
            "\nProcessed samples:\n"
        )

        for sample, result in (
            sample_results.items()
        ):

            if result is None:

                f.write(
                    f"  {sample}: not available\n"
                )

                continue

            f.write(
                f"  {sample}:\n"
            )
            f.write(
                f"    entries="
                f"{result['total_entries']}\n"
            )
            f.write(
                "    after_lepton_met="
                f"{result['events_after_lepton_met_cuts']}\n"
            )
            f.write(
                "    after_mtw="
                f"{result['events_after_mtw_cut']}\n"
            )
            f.write(
                "    after_wcjet="
                f"{result['events_after_wcjet_selection']}\n"
            )
            f.write(
                "    OS="
                f"{result['opposite_sign_events']}, "
                "SS="
                f"{result['same_sign_events']}\n"
            )

        f.write(
            "\nHigh-statistics correction pairs produced:\n"
        )

        if not pair_results:

            f.write(
                "  none\n"
            )

        else:

            for pair_label in (
                pair_results
            ):

                f.write(
                    f"  {pair_label}\n"
                )

        f.write(
            "\nRepaired-reference comparisons completed:\n"
        )

        if not completed_reference_comparisons:

            f.write(
                "  none - repaired MTWcut reference CSVs "
                "are not present yet\n"
            )

        else:

            for pair_label in (
                completed_reference_comparisons
            ):

                f.write(
                    f"  {pair_label}\n"
                )

    return output_path


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "============================================================"
    )
    print(
        " Repaired Pythia high-stat weight-0 cross-check"
    )
    print(
        "============================================================"
    )
    print()

    print(
        "Selection:"
    )
    print(
        f"  MET > {MET_MIN_GEV:.1f} GeV"
    )
    print(
        f"  lepton pT > "
        f"{LEPTON_PT_MIN_GEV:.1f} GeV"
    )
    print(
        f"  |eta_lepton| < "
        f"{LEPTON_ABS_ETA_MAX:.1f}"
    )
    print(
        f"  m_T^W > "
        f"{MTW_MIN_GEV:.1f} GeV"
    )
    print(
        f"  jet pT > "
        f"{JET_PT_MIN_GEV:.1f} GeV"
    )
    print(
        f"  |eta_jet| < "
        f"{JET_ABS_ETA_MAX:.1f}"
    )
    print(
        "  exactly 1 charm-identified fiducial jet"
    )
    print(
        "  charm ID: jet_charge != 0"
    )
    print(
        "  OS-SS: opposite sign +1, same sign -1"
    )
    print()

    print(
        "Cross-check ROOT directory:"
    )
    print(
        f"  {CROSS_CHECK_ROOT_DIR}"
    )
    print()

    print(
        "Repaired reference CSV directory:"
    )
    print(
        f"  {REFERENCE_CSV_DIR}"
    )
    print()

    discovered = (
        discover_files()
    )

    print(
        "Discovered files:"
    )

    for sample in (
        NEW_SAMPLE_PATTERNS
    ):

        print(
            f"  {sample:18s}: "
            f"{len(discovered[sample])}"
        )

    # --------------------------------------------------------
    # Process all available high-statistics samples.
    # --------------------------------------------------------

    sample_results = {}

    for sample in (
        NEW_SAMPLE_PATTERNS
    ):

        paths = (
            discovered[
                sample
            ]
        )

        if not paths:

            print()
            print(
                f"No files found for {sample}."
            )

            sample_results[
                sample
            ] = None

            continue

        sample_results[
            sample
        ] = (
            process_combined_sample(
                sample,
                paths,
            )
        )

    combined_samples_path = (
        write_combined_sample_root(
            sample_results
        )
    )

    print()
    print(
        "Saved combined high-stat sample histograms:"
    )
    print(
        f"  {combined_samples_path}"
    )

    # --------------------------------------------------------
    # Build Pythia W+ / W- high-stat corrections.
    # --------------------------------------------------------

    pair_results = {}
    completed_reference_comparisons = {}

    for (
        pair_label,
        definition,
    ) in PAIR_DEFINITIONS.items():

        particle_sample = (
            definition[
                "particle"
            ]
        )

        parton_sample = (
            definition[
                "parton"
            ]
        )

        particle_result = (
            sample_results[
                particle_sample
            ]
        )

        parton_result = (
            sample_results[
                parton_sample
            ]
        )

        print()
        print(
            "============================================================"
        )
        print(
            f"Pair: {definition['display']}"
        )

        if (
            particle_result is None
            or parton_result is None
        ):

            print(
                "  Cannot build this correction yet."
            )

            if particle_result is None:
                print(
                    f"  Missing particle files: "
                    f"{particle_sample}"
                )

            if parton_result is None:
                print(
                    f"  Missing parton files:   "
                    f"{parton_sample}"
                )

            continue

        new_correction_hist = (
            make_correction_histogram(
                h_particle=(
                    particle_result[
                        "histogram"
                    ]
                ),
                h_parton=(
                    parton_result[
                        "histogram"
                    ]
                ),
                name=(
                    f"{pair_label}_"
                    f"new_weight0_correction"
                ),
            )
        )

        reference_path = (
            repaired_reference_path(
                definition
            )
        )

        reference_rows = (
            read_reference_csv(
                reference_path
            )
        )

        if reference_rows is None:

            print(
                "  Repaired reference CSV not present yet."
            )
            print(
                "  High-stat correction WILL be saved;"
            )
            print(
                "  repaired-reference comparison is deferred."
            )
            print(
                f"  Expected later at: {reference_path}"
            )

        else:

            print(
                "  Repaired reference CSV found."
            )
            print(
                f"  {reference_path}"
            )

        rows = (
            build_output_rows(
                pair_label=(
                    pair_label
                ),
                correction_hist=(
                    new_correction_hist
                ),
                reference_rows=(
                    reference_rows
                ),
            )
        )

        csv_path = (
            write_output_csv(
                pair_label,
                rows,
            )
        )

        reference_hist = (
            make_reference_histogram(
                reference_rows,
                f"{pair_label}_reference",
            )
            if reference_rows is not None
            else None
        )

        root_path = (
            write_pair_root_file(
                pair_label=(
                    pair_label
                ),
                particle_hist=(
                    particle_result[
                        "histogram"
                    ]
                ),
                parton_hist=(
                    parton_result[
                        "histogram"
                    ]
                ),
                new_correction_hist=(
                    new_correction_hist
                ),
                reference_hist=(
                    reference_hist
                ),
            )
        )

        pair_results[
            pair_label
        ] = {
            "display": (
                definition[
                    "display"
                ]
            ),
            "rows": rows,
            "csv_path": csv_path,
            "root_path": root_path,
        }

        print(
            f"  CSV:  {csv_path}"
        )
        print(
            f"  ROOT: {root_path}"
        )

        if (
            reference_rows is not None
        ):

            (
                png_path,
                pdf_path,
            ) = save_static_plot(
                pair_label=(
                    pair_label
                ),
                display_label=(
                    definition[
                        "display"
                    ]
                ),
                reference_hist=(
                    reference_hist
                ),
                new_hist=(
                    new_correction_hist
                ),
                output_rows=(
                    rows
                ),
            )

            completed_reference_comparisons[
                pair_label
            ] = {
                "display": (
                    definition[
                        "display"
                    ]
                ),
                "rows": rows,
                "png_path": png_path,
                "pdf_path": pdf_path,
            }

            print(
                f"  PNG:  {png_path}"
            )
            print(
                f"  PDF:  {pdf_path}"
            )

    interactive_path = (
        save_interactive_plot(
            completed_reference_comparisons
        )
    )

    if interactive_path is not None:

        print()
        print(
            "Interactive HTML:"
        )
        print(
            f"  {interactive_path}"
        )

    summary_path = (
        write_run_summary(
            discovered=(
                discovered
            ),
            sample_results=(
                sample_results
            ),
            pair_results=(
                pair_results
            ),
            completed_reference_comparisons=(
                completed_reference_comparisons
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
        "All outputs:"
    )
    print(
        f"  {OUTPUT_DIR}"
    )
    print()

    print(
        "Run summary:"
    )
    print(
        f"  {summary_path}"
    )
    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
