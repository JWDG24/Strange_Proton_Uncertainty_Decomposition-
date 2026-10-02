#!/usr/bin/env python3

# ============================================================
# even_odd_uncertainty.py
#
# Purpose
# -------
# Odd/even statistical stability study for the W+c
# lepton-pseudorapidity parton-to-particle correction factors.
#
# REPAIRED EVENT DEFINITION
# -------------------------
# This version uses the same W+c selection and OS-SS convention
# as the repaired MTWcut_make_histograms_etalepton.py:
#
#   MET > 25 GeV
#   lepton pT > 20 GeV
#   |eta_lepton| < 2.5
#   m_T^W > 40 GeV
#   jet pT > 25 GeV
#   |eta_jet| < 2.5
#   exactly one charm-identified fiducial jet
#
# The WCharmTree ntuples store the per-jet charm identification
# in jet_charge:
#
#   jet_charge == 0   -> not charm identified
#   jet_charge != 0   -> charm identified
#
# The sign of jet_charge is used with leptons_charge for OS-SS:
#
#   opposite sign -> +1
#   same sign     -> -1
#
# Extra non-charm jets are allowed.
#
# The odd/even method itself is unchanged:
#
#   even = ROOT tree entry indices 0, 2, 4, ...
#   odd  = ROOT tree entry indices 1, 3, 5, ...
#
# Each particle/parton sample is split independently; no
# event-by-event particle/parton matching is assumed.
#
# Correction:
#
#   C_i = N_i(parton) / N_i(particle)
#
# Split diagnostic:
#
#   split_stat_uncertainty = |C_odd - C_even| / 2
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
PROJECT_DIR = LEPTON_WORK_DIR.parent

DATA_DIR = PROJECT_DIR / "data"

OUTPUT_DIR = SCRIPT_DIR / "outputs"
ROOT_OUTPUT_DIR = OUTPUT_DIR / "root"
CSV_OUTPUT_DIR = OUTPUT_DIR / "csv"
PLOT_OUTPUT_DIR = OUTPUT_DIR / "plots"

for directory in (
    ROOT_OUTPUT_DIR,
    CSV_OUTPUT_DIR,
    PLOT_OUTPUT_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


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

ETA_LEPTON_BIN_ARRAY = array("d", ETA_LEPTON_BINS)


# ============================================================
# Samples and correction-factor pairings
# ============================================================

SAMPLE_PAIRS = {
    "Pythia_plus": (
        "WCPy8plus",
        "WCPyPartonplus",
    ),
    "Pythia_minus": (
        "WCPy8minus",
        "WCPyPartonminus",
    ),
    "Herwig_plus": (
        "WCH7plus",
        "WCHPartonplus",
    ),
    "Herwig_minus": (
        "WCH7minus",
        "WCHPartonminus",
    ),
}

ALL_SAMPLES = sorted({
    sample
    for pair in SAMPLE_PAIRS.values()
    for sample in pair
})


# ============================================================
# Tree / branch helpers
# ============================================================

def tree_branch_names(tree):
    """Return all branch names stored in a TTree."""
    return {
        branch.GetName()
        for branch in tree.GetListOfBranches()
    }


def validate_required_branches(tree, sample):
    """
    Fail loudly if a branch needed by the repaired W+c selection
    is missing.
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
        required - tree_branch_names(tree)
    )

    if missing:
        raise RuntimeError(
            f"{sample}: WCharmTree is missing required branches: "
            + ", ".join(missing)
        )


def sequence_length(value):
    """Length helper for ROOT vector-like branch values."""
    try:
        return len(value)
    except TypeError:
        return 1


def sequence_value(value, index):
    """Read one element from a ROOT vector/array-like branch."""
    try:
        return value[index]
    except TypeError:
        if index != 0:
            raise IndexError(index)
        return value


def validate_jet_charge_encoding(
    tree,
    sample,
    entries_to_check=200,
):
    """
    Verify that jet_pt, jet_eta and jet_charge are aligned and
    that jet_charge follows the observed {-1, 0, +1} encoding.

    We stop on anything unexpected rather than silently applying
    the wrong charm-jet definition.
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
                f"{sample}: per-jet branch-length mismatch at "
                f"entry {ientry}: len(jet_pt)={n_pt}, "
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
                    f"{sample}: unexpected jet_charge={value} "
                    f"at entry {ientry}, jet {jet_index}. "
                    "Expected -1, 0 or +1."
                )


# ============================================================
# Physics-selection helpers
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
        max(0.0, mtw_squared)
    )


def fiducial_charm_jets(tree):
    """
    Return charm-identified jets inside the W+c fiducial region.

    Each returned item is:
        (jet_index, charm_sign)

    where charm_sign is +/-1 from jet_charge.

    Additional non-charm jets are allowed.
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
            "Per-jet branch-length mismatch while processing "
            f"event: len(jet_pt)={n_pt}, "
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

        # WCharmTree charm-tag encoding.
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


# ============================================================
# Histogram helpers
# ============================================================

def make_eta_histogram(
    sample,
    subset,
):
    """Create one weighted |eta_lepton| histogram."""

    histogram = ROOT.TH1F(
        f"etalepton_{sample}_{subset}",
        (
            f"{sample} {subset};"
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


def make_sample_histograms(sample):
    """Return all/odd/even histograms for one sample."""

    return {
        "all": make_eta_histogram(
            sample,
            "all",
        ),
        "odd": make_eta_histogram(
            sample,
            "odd",
        ),
        "even": make_eta_histogram(
            sample,
            "even",
        ),
    }


# ============================================================
# Process one sample
# ============================================================

def process_sample(sample):
    """
    Read one ROOT ntuple and fill all/odd/even nominal histograms
    in one pass.

    The odd/even split is based on the ORIGINAL ROOT tree entry
    index. It is not based on the accepted-event index.

    Nominal MC weight:
        weightvec[0] / total_tree_entries

    W+c sign:
        +1 for OS
        -1 for SS
    """

    input_path = (
        DATA_DIR
        / f"WCharm_{sample}.root"
    )

    print()
    print(
        "------------------------------------------------------------"
    )
    print(
        f"Processing sample: {sample}"
    )
    print(
        "------------------------------------------------------------"
    )

    tfile = ROOT.TFile.Open(
        str(input_path),
        "READ",
    )

    if (
        not tfile
        or tfile.IsZombie()
    ):
        raise RuntimeError(
            f"Could not open ROOT file:\n"
            f"  {input_path}"
        )

    tree = tfile.Get(
        "WCharmTree"
    )

    if not tree:
        tfile.Close()
        raise RuntimeError(
            f"WCharmTree not found in:\n"
            f"  {input_path}"
        )

    total_events = int(
        tree.GetEntries()
    )

    if total_events <= 0:
        tfile.Close()
        raise RuntimeError(
            f"No events found in:\n"
            f"  {input_path}"
        )

    validate_required_branches(
        tree,
        sample,
    )

    validate_jet_charge_encoding(
        tree,
        sample,
    )

    tree.GetEntry(0)

    if (
        len(tree.weightvec)
        <= NOMINAL_WEIGHT_INDEX
    ):
        tfile.Close()
        raise RuntimeError(
            f"{sample}: nominal weightvec[0] is unavailable."
        )

    histograms = make_sample_histograms(
        sample
    )

    events_after_lepton_met_cuts = 0
    events_after_mtw_cut = 0
    events_after_wcjet_selection = 0

    opposite_sign_events = 0
    same_sign_events = 0

    accepted_all = 0
    accepted_odd = 0
    accepted_even = 0

    # ========================================================
    # ONE tree pass
    # ========================================================

    for ientry in range(
        total_events
    ):

        tree.GetEntry(
            ientry
        )

        # ----------------------------------------------------
        # Lepton / MET fiducial selection
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # W transverse-mass requirement
        # ----------------------------------------------------

        if (
            calculate_mtw(tree)
            <= MTW_MIN_GEV
        ):
            continue

        events_after_mtw_cut += 1

        # ----------------------------------------------------
        # RESTORED W+c jet requirement
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # RESTORED OS-SS sign
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Observable and nominal event weight
        # ----------------------------------------------------

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
            / float(total_events)
        )

        final_weight = (
            charge_factor
            * event_weight
        )

        # ----------------------------------------------------
        # Full sample
        # ----------------------------------------------------

        histograms[
            "all"
        ].Fill(
            eta_lepton,
            final_weight,
        )

        accepted_all += 1

        # ----------------------------------------------------
        # Odd/even split by ORIGINAL ROOT tree entry index
        # ----------------------------------------------------

        if ientry % 2 == 0:

            histograms[
                "even"
            ].Fill(
                eta_lepton,
                final_weight,
            )

            accepted_even += 1

        else:

            histograms[
                "odd"
            ].Fill(
                eta_lepton,
                final_weight,
            )

            accepted_odd += 1

    tfile.Close()

    # Internal consistency checks.
    if (
        accepted_all
        != events_after_wcjet_selection
    ):
        raise RuntimeError(
            f"{sample}: accepted count does not match "
            "W+c jet-selection count."
        )

    if (
        accepted_odd
        + accepted_even
        != accepted_all
    ):
        raise RuntimeError(
            f"{sample}: odd/even counts do not sum "
            "to accepted total."
        )

    if (
        opposite_sign_events
        + same_sign_events
        != accepted_all
    ):
        raise RuntimeError(
            f"{sample}: OS/SS counts do not sum "
            "to accepted total."
        )

    print(
        f"  Total entries:             {total_events}"
    )
    print(
        f"  After lepton/MET cuts:     "
        f"{events_after_lepton_met_cuts}"
    )
    print(
        f"  After mT cut:              "
        f"{events_after_mtw_cut}"
    )
    print(
        f"  After W+c jet cuts:        "
        f"{events_after_wcjet_selection}"
    )
    print(
        f"    opposite-sign:           "
        f"{opposite_sign_events}"
    )
    print(
        f"    same-sign:               "
        f"{same_sign_events}"
    )
    print(
        f"  Accepted odd entries:      "
        f"{accepted_odd}"
    )
    print(
        f"  Accepted even entries:     "
        f"{accepted_even}"
    )

    return {
        "histograms": histograms,
        "total_events": total_events,
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
        "accepted_all": accepted_all,
        "accepted_odd": accepted_odd,
        "accepted_even": accepted_even,
    }


# ============================================================
# Correction-factor helpers
# ============================================================

def make_correction_histogram(
    h_particle,
    h_parton,
    name,
):
    """
    Construct:
        C = parton / particle

    ROOT propagates the histogram statistical errors.
    """

    correction = (
        h_parton.Clone(name)
    )

    correction.SetDirectory(0)

    correction.Divide(
        h_parton,
        h_particle,
        1.0,
        1.0,
        "",
    )

    return correction


def empty_hist_like(
    reference,
    name,
    title,
):
    """Create an empty histogram with matching binning."""

    output = (
        reference.Clone(name)
    )

    output.Reset()
    output.SetDirectory(0)
    output.SetTitle(title)

    return output


def build_split_uncertainty_histograms(
    h_all,
    h_odd,
    h_even,
    pair_label,
):
    """
    Build:
      C_odd - C_even
      |C_odd - C_even|
      |C_odd - C_even| / 2
      [|C_odd - C_even| / 2] / |C_all|
    """

    h_signed_difference = (
        empty_hist_like(
            h_all,
            f"{pair_label}_odd_minus_even",
            (
                f"{pair_label}: odd - even;"
                f"|#eta_{{#ell}}|;"
                f"C_{{odd}} - C_{{even}}"
            ),
        )
    )

    h_abs_difference = (
        empty_hist_like(
            h_all,
            (
                f"{pair_label}_"
                f"abs_odd_even_difference"
            ),
            (
                f"{pair_label}: |odd - even|;"
                f"|#eta_{{#ell}}|;"
                f"|C_{{odd}} - C_{{even}}|"
            ),
        )
    )

    h_split_uncertainty = (
        empty_hist_like(
            h_all,
            (
                f"{pair_label}_"
                f"split_stat_uncertainty"
            ),
            (
                f"{pair_label}: split "
                f"statistical uncertainty;"
                f"|#eta_{{#ell}}|;"
                f"#sigma_{{split}}"
            ),
        )
    )

    h_relative_split_uncertainty = (
        empty_hist_like(
            h_all,
            (
                f"{pair_label}_"
                f"relative_split_stat_uncertainty"
            ),
            (
                f"{pair_label}: relative "
                f"split uncertainty;"
                f"|#eta_{{#ell}}|;"
                f"#sigma_{{split}}/|C_{{all}}|"
            ),
        )
    )

    for ibin in range(
        1,
        h_all.GetNbinsX() + 1,
    ):

        c_all = float(
            h_all.GetBinContent(
                ibin
            )
        )

        c_odd = float(
            h_odd.GetBinContent(
                ibin
            )
        )

        c_even = float(
            h_even.GetBinContent(
                ibin
            )
        )

        signed_difference = (
            c_odd
            - c_even
        )

        abs_difference = abs(
            signed_difference
        )

        split_uncertainty = (
            0.5
            * abs_difference
        )

        relative_split_uncertainty = (
            split_uncertainty
            / abs(c_all)
            if c_all != 0.0
            else 0.0
        )

        h_signed_difference.SetBinContent(
            ibin,
            signed_difference,
        )

        h_abs_difference.SetBinContent(
            ibin,
            abs_difference,
        )

        h_split_uncertainty.SetBinContent(
            ibin,
            split_uncertainty,
        )

        h_relative_split_uncertainty.SetBinContent(
            ibin,
            relative_split_uncertainty,
        )

        for histogram in (
            h_signed_difference,
            h_abs_difference,
            h_split_uncertainty,
            h_relative_split_uncertainty,
        ):
            histogram.SetBinError(
                ibin,
                0.0,
            )

    return {
        "signed_difference": (
            h_signed_difference
        ),
        "abs_difference": (
            h_abs_difference
        ),
        "split_uncertainty": (
            h_split_uncertainty
        ),
        "relative_split_uncertainty": (
            h_relative_split_uncertainty
        ),
    }


# ============================================================
# ROOT output
# ============================================================

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


def write_sample_histograms(
    sample_results,
):
    """Save raw all/odd/even histograms to one ROOT file."""

    output_path = (
        ROOT_OUTPUT_DIR
        / "even_odd_sample_histograms.root"
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
            f"Could not create {output_path}"
        )

    ROOT.TNamed(
        "Observable",
        "etalepton",
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
        "CharmJetIDDefinition",
        (
            "jet_charge != 0; sign supplies "
            "charm/anticharm for OS-SS"
        ),
    ).Write()

    ROOT.TNamed(
        "OSSSWeightDefinition",
        (
            "+1 for opposite-sign lepton/charm; "
            "-1 for same-sign lepton/charm"
        ),
    ).Write()

    for sample in ALL_SAMPLES:

        sample_directory = (
            output_file.mkdir(
                sample
            )
        )

        sample_directory.cd()

        result = (
            sample_results[sample]
        )

        ROOT.TNamed(
            "TotalEntries",
            str(
                result[
                    "total_events"
                ]
            ),
        ).Write()

        ROOT.TNamed(
            "EventsAfterBaseCuts",
            str(
                result[
                    "events_after_lepton_met_cuts"
                ]
            ),
        ).Write()

        ROOT.TNamed(
            "EventsAfterMTWCut",
            str(
                result[
                    "events_after_mtw_cut"
                ]
            ),
        ).Write()

        ROOT.TNamed(
            "EventsAfterWCJetSelection",
            str(
                result[
                    "events_after_wcjet_selection"
                ]
            ),
        ).Write()

        ROOT.TNamed(
            "OppositeSignEvents",
            str(
                result[
                    "opposite_sign_events"
                ]
            ),
        ).Write()

        ROOT.TNamed(
            "SameSignEvents",
            str(
                result[
                    "same_sign_events"
                ]
            ),
        ).Write()

        for subset in (
            "all",
            "odd",
            "even",
        ):

            histogram = (
                result[
                    "histograms"
                ][
                    subset
                ]
            )

            histogram.Write(
                f"etalepton_{subset}"
            )

        output_file.cd()

    output_file.Write()
    output_file.Close()

    return output_path


def write_pair_root_output(
    pair_label,
    corrections,
    split_hists,
):
    """Save correction and split-uncertainty histograms."""

    output_path = (
        ROOT_OUTPUT_DIR
        / f"even_odd_{pair_label}.root"
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
            f"Could not create {output_path}"
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
        "EventSplitDefinition",
        (
            "odd/even defined by original ROOT "
            "tree entry index; entry 0 is even"
        ),
    ).Write()

    ROOT.TNamed(
        "EventSelection",
        event_selection_text(),
    ).Write()

    ROOT.TNamed(
        "MTWCutDefinition",
        "m_T^W > 40 GeV",
    ).Write()

    ROOT.TNamed(
        "JetSelectionDefinition",
        (
            "jet_pt > 25 GeV; abs(jet_eta) < 2.5; "
            "exactly one jet with jet_charge != 0"
        ),
    ).Write()

    ROOT.TNamed(
        "OSSSWeightDefinition",
        (
            "+1 for opposite-sign lepton/charm; "
            "-1 for same-sign lepton/charm"
        ),
    ).Write()

    corrections[
        "all"
    ].Write(
        "correction_all"
    )

    corrections[
        "odd"
    ].Write(
        "correction_odd"
    )

    corrections[
        "even"
    ].Write(
        "correction_even"
    )

    split_hists[
        "signed_difference"
    ].Write(
        "odd_minus_even"
    )

    split_hists[
        "abs_difference"
    ].Write(
        "abs_odd_even_difference"
    )

    split_hists[
        "split_uncertainty"
    ].Write(
        "split_stat_uncertainty"
    )

    split_hists[
        "relative_split_uncertainty"
    ].Write(
        "relative_split_stat_uncertainty"
    )

    output_file.Write()
    output_file.Close()

    return output_path


# ============================================================
# CSV output
# ============================================================

def write_pair_csv(
    pair_label,
    corrections,
    split_hists,
):
    """Write a bin-by-bin summary."""

    output_path = (
        CSV_OUTPUT_DIR
        / f"even_odd_{pair_label}.csv"
    )

    h_all = corrections[
        "all"
    ]
    h_odd = corrections[
        "odd"
    ]
    h_even = corrections[
        "even"
    ]

    h_signed = split_hists[
        "signed_difference"
    ]
    h_abs = split_hists[
        "abs_difference"
    ]
    h_split = split_hists[
        "split_uncertainty"
    ]
    h_relative = split_hists[
        "relative_split_uncertainty"
    ]

    with open(
        output_path,
        "w",
        newline="",
    ) as csvfile:

        writer = csv.writer(
            csvfile
        )

        writer.writerow([
            "pair_label",
            "bin",
            "bin_low_edge",
            "bin_up_edge",
            "C_all",
            "C_odd",
            "C_even",
            "odd_minus_even",
            "abs_odd_even_difference",
            (
                "split_stat_uncertainty_"
                "half_difference"
            ),
            "relative_split_stat_uncertainty",
            "all_ratio_ROOT_stat_error",
            "odd_ratio_ROOT_stat_error",
            "even_ratio_ROOT_stat_error",
        ])

        for ibin in range(
            1,
            h_all.GetNbinsX() + 1,
        ):

            low_edge = (
                h_all.GetXaxis()
                .GetBinLowEdge(
                    ibin
                )
            )

            up_edge = (
                h_all.GetXaxis()
                .GetBinUpEdge(
                    ibin
                )
            )

            writer.writerow([
                pair_label,
                ibin,
                low_edge,
                up_edge,
                h_all.GetBinContent(
                    ibin
                ),
                h_odd.GetBinContent(
                    ibin
                ),
                h_even.GetBinContent(
                    ibin
                ),
                h_signed.GetBinContent(
                    ibin
                ),
                h_abs.GetBinContent(
                    ibin
                ),
                h_split.GetBinContent(
                    ibin
                ),
                h_relative.GetBinContent(
                    ibin
                ),
                h_all.GetBinError(
                    ibin
                ),
                h_odd.GetBinError(
                    ibin
                ),
                h_even.GetBinError(
                    ibin
                ),
            ])

    return output_path


# ============================================================
# Plotting
# ============================================================

def style_correction_histograms(
    h_all,
    h_odd,
    h_even,
):
    """Simple ROOT styling for diagnostic plots."""

    h_all.SetLineColor(
        ROOT.kBlack
    )
    h_all.SetMarkerColor(
        ROOT.kBlack
    )
    h_all.SetMarkerStyle(
        20
    )
    h_all.SetLineWidth(
        2
    )

    h_odd.SetLineColor(
        ROOT.kBlue + 1
    )
    h_odd.SetMarkerColor(
        ROOT.kBlue + 1
    )
    h_odd.SetMarkerStyle(
        21
    )
    h_odd.SetLineWidth(
        2
    )

    h_even.SetLineColor(
        ROOT.kRed + 1
    )
    h_even.SetMarkerColor(
        ROOT.kRed + 1
    )
    h_even.SetMarkerStyle(
        22
    )
    h_even.SetLineWidth(
        2
    )


def save_pair_plot(
    pair_label,
    corrections,
    split_hists,
):
    """Save one two-panel diagnostic plot per pair."""

    h_all = corrections[
        "all"
    ]
    h_odd = corrections[
        "odd"
    ]
    h_even = corrections[
        "even"
    ]
    h_split = split_hists[
        "split_uncertainty"
    ]

    style_correction_histograms(
        h_all,
        h_odd,
        h_even,
    )

    canvas = ROOT.TCanvas(
        f"canvas_{pair_label}",
        pair_label,
        1000,
        900,
    )

    canvas.Divide(
        1,
        2,
    )

    # Top panel.
    canvas.cd(1)
    ROOT.gPad.SetGrid()

    h_all.SetTitle(
        (
            f"{pair_label}: all / odd / even corrections;"
            f"|#eta_{{#ell}}|;"
            f"C = N_{{parton}}/N_{{particle}}"
        )
    )

    values = []

    for histogram in (
        h_all,
        h_odd,
        h_even,
    ):

        for ibin in range(
            1,
            histogram.GetNbinsX() + 1,
        ):

            value = (
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

        h_all.SetMinimum(
            minimum
            - 0.20 * spread
        )

        h_all.SetMaximum(
            maximum
            + 0.20 * spread
        )

    h_all.Draw("E1")
    h_odd.Draw("E1 SAME")
    h_even.Draw("E1 SAME")

    legend = ROOT.TLegend(
        0.68,
        0.72,
        0.88,
        0.88,
    )

    legend.AddEntry(
        h_all,
        "All events",
        "lep",
    )
    legend.AddEntry(
        h_odd,
        "Odd entries",
        "lep",
    )
    legend.AddEntry(
        h_even,
        "Even entries",
        "lep",
    )

    legend.Draw()

    # Bottom panel.
    canvas.cd(2)
    ROOT.gPad.SetGrid()

    h_split.SetLineColor(
        ROOT.kBlack
    )
    h_split.SetMarkerColor(
        ROOT.kBlack
    )
    h_split.SetMarkerStyle(
        20
    )
    h_split.SetLineWidth(
        2
    )

    h_split.SetTitle(
        (
            f"{pair_label}: odd/even split uncertainty;"
            f"|#eta_{{#ell}}|;"
            f"|C_{{odd}} - C_{{even}}| / 2"
        )
    )

    h_split.SetMinimum(
        0.0
    )

    h_split.Draw(
        "HIST P"
    )

    canvas.Update()

    png_path = (
        PLOT_OUTPUT_DIR
        / f"even_odd_{pair_label}.png"
    )

    pdf_path = (
        PLOT_OUTPUT_DIR
        / f"even_odd_{pair_label}.pdf"
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
# Text summary
# ============================================================

def write_summary_file(
    sample_results,
    pair_results,
):
    """Write a compact record of the repaired run."""

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
            "Odd/even statistical stability study\n"
        )
        f.write(
            "====================================\n\n"
        )

        f.write(
            "Event split:\n"
        )
        f.write(
            "  even = ROOT tree entry indices "
            "0, 2, 4, ...\n"
        )
        f.write(
            "  odd  = ROOT tree entry indices "
            "1, 3, 5, ...\n\n"
        )

        f.write(
            "Physics selection:\n"
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
            f"  m_T^W > {MTW_MIN_GEV} GeV\n"
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
            "  exactly one charm-identified "
            "fiducial jet\n"
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
            "Nominal weight:\n"
        )
        f.write(
            "  weightvec[0]\n\n"
        )

        f.write(
            "Split quantity:\n"
        )
        f.write(
            "  abs_difference = "
            "|C_odd - C_even|\n"
        )
        f.write(
            "  split_stat_uncertainty = "
            "abs_difference / 2\n\n"
        )

        f.write(
            "Cut flow and accepted-event counts:\n"
        )

        for sample in ALL_SAMPLES:

            result = (
                sample_results[
                    sample
                ]
            )

            f.write(
                f"  {sample}:\n"
            )
            f.write(
                f"    entries={result['total_events']}\n"
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
                "    odd="
                f"{result['accepted_odd']}, "
                "even="
                f"{result['accepted_even']}\n"
            )

        f.write(
            "\nGenerated pair outputs:\n"
        )

        for pair_label in SAMPLE_PAIRS:

            if (
                pair_label
                in pair_results
            ):
                f.write(
                    f"  {pair_label}\n"
                )

    return output_path


# ============================================================
# Main analysis
# ============================================================

def main():

    print()
    print(
        "============================================================"
    )
    print(
        " W+c odd/even statistical uncertainty study"
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
        "  exactly 1 charm-identified "
        "fiducial jet"
    )
    print(
        "  charm ID: jet_charge != 0"
    )
    print(
        "  OS-SS: opposite sign +1, "
        "same sign -1"
    )
    print()

    print(
        f"Data directory:   {DATA_DIR}"
    )
    print(
        f"Output directory: {OUTPUT_DIR}"
    )
    print()

    if not DATA_DIR.is_dir():
        raise RuntimeError(
            f"Data directory does not exist:\n"
            f"  {DATA_DIR}"
        )

    # --------------------------------------------------------
    # Process all raw samples once.
    # --------------------------------------------------------

    sample_results = {}

    for sample in ALL_SAMPLES:

        sample_results[
            sample
        ] = process_sample(
            sample
        )

    sample_root_path = (
        write_sample_histograms(
            sample_results
        )
    )

    print()
    print(
        "Saved sample histograms:"
    )
    print(
        f"  {sample_root_path}"
    )

    # --------------------------------------------------------
    # Build all/odd/even correction factors and diagnostics.
    # --------------------------------------------------------

    pair_results = {}

    for (
        pair_label,
        (
            particle_sample,
            parton_sample,
        ),
    ) in SAMPLE_PAIRS.items():

        print()
        print(
            "============================================================"
        )
        print(
            f"Building pair: {pair_label}"
        )
        print(
            f"  particle: {particle_sample}"
        )
        print(
            f"  parton:   {parton_sample}"
        )

        corrections = {}

        for subset in (
            "all",
            "odd",
            "even",
        ):

            h_particle = (
                sample_results[
                    particle_sample
                ][
                    "histograms"
                ][
                    subset
                ]
            )

            h_parton = (
                sample_results[
                    parton_sample
                ][
                    "histograms"
                ][
                    subset
                ]
            )

            corrections[
                subset
            ] = (
                make_correction_histogram(
                    h_particle=(
                        h_particle
                    ),
                    h_parton=(
                        h_parton
                    ),
                    name=(
                        f"{pair_label}_"
                        f"correction_{subset}"
                    ),
                )
            )

        split_hists = (
            build_split_uncertainty_histograms(
                h_all=(
                    corrections["all"]
                ),
                h_odd=(
                    corrections["odd"]
                ),
                h_even=(
                    corrections["even"]
                ),
                pair_label=(
                    pair_label
                ),
            )
        )

        root_path = (
            write_pair_root_output(
                pair_label=(
                    pair_label
                ),
                corrections=(
                    corrections
                ),
                split_hists=(
                    split_hists
                ),
            )
        )

        csv_path = (
            write_pair_csv(
                pair_label=(
                    pair_label
                ),
                corrections=(
                    corrections
                ),
                split_hists=(
                    split_hists
                ),
            )
        )

        (
            png_path,
            pdf_path,
        ) = save_pair_plot(
            pair_label=(
                pair_label
            ),
            corrections=(
                corrections
            ),
            split_hists=(
                split_hists
            ),
        )

        pair_results[
            pair_label
        ] = {
            "corrections": (
                corrections
            ),
            "split_hists": (
                split_hists
            ),
            "root_path": (
                root_path
            ),
            "csv_path": (
                csv_path
            ),
            "png_path": (
                png_path
            ),
            "pdf_path": (
                pdf_path
            ),
        }

        print(
            f"  ROOT: {root_path}"
        )
        print(
            f"  CSV:  {csv_path}"
        )
        print(
            f"  PNG:  {png_path}"
        )
        print(
            f"  PDF:  {pdf_path}"
        )

    summary_path = (
        write_summary_file(
            sample_results=(
                sample_results
            ),
            pair_results=(
                pair_results
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
        "Main odd/even diagnostic:"
    )
    print(
        "  split_stat_uncertainty_half_difference"
    )
    print(
        "    = |C_odd - C_even| / 2"
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
