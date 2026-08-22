#!/usr/bin/env python3

# ============================================================
# make_histograms_etalepton.py
#
# Purpose:
#   Read the WCharm ntuples and create lepton pseudorapidity
#   histograms for every valid (sample, weight index)
#   combination.
#
# Observable:
#       |eta_lepton|
#
# Behaviour:
#   - Missing weights are skipped
#   - No fallback to weight 0
#   - No fake duplicate outputs
#   - One ROOT output file per real (sample, weight) pair
#
# Folder structure expected:
#
#   project_root/
#   ├── data/
#   ├── weights/
#   └── lepton_pseudorapidity_work/
#       ├── make_histograms_etalepton.py
#       └── make_histograms_etalepton_outputs/
#
# ============================================================

import ROOT
import sys
from pathlib import Path
from array import array


# ============================================================
# ROOT settings
# ============================================================

ROOT.gROOT.SetBatch(True)

# Prevent ROOT from automatically owning histograms.
ROOT.TH1.AddDirectory(False)


# ============================================================
# User options
# ============================================================

# True  = process every valid available weight
# False = process nominal weight 0 only
USE_ALL_WEIGHTS = True


# ============================================================
# Directory structure
# ============================================================

# This script should live directly inside:
#     lepton_pseudorapidity_work/

SCRIPT_DIR = Path(__file__).resolve().parent

# Parent directory containing:
#     data/
#     weights/
#     lepton_pseudorapidity_work/
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
WEIGHT_DIR = PROJECT_DIR / "weights"

OUTPUT_DIR = SCRIPT_DIR / "make_histograms_etalepton_outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Lepton pseudorapidity binning
# ============================================================

# Same variable |eta| binning as the original analysis.

ETA_LEPTON_BINS = sorted([
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
])

ETA_LEPTON_BIN_ARRAY = array("d", ETA_LEPTON_BINS)


# ============================================================
# Requested weights
# ============================================================

# Potential weight indices.
#
# For each individual sample, only indices that genuinely exist
# both in its weight mapping and in its weight vector are run.

ALL_REQUESTED_WEIGHTS = list(range(319))


# ============================================================
# Samples
# ============================================================

SIGNAL_SAMPLES_DEFAULT = [
    "WCH7minus",
    "WCH7plus",
    "WCHPartonminus",
    "WCHPartonplus",
    "WCPy8minus",
    "WCPy8plus",
    "WCPyPartonminus",
    "WCPyPartonplus",
]

# If sample names are supplied on the command line, process only
# those samples. Otherwise process all eight.
SIGNAL_SAMPLES = (
    sys.argv[1:]
    if len(sys.argv) > 1
    else SIGNAL_SAMPLES_DEFAULT
)


# ============================================================
# Weight-map helpers
# ============================================================

def load_weight_map_clean(path):
    """
    Read a cleaned weight mapping file.

    Expected format:

        0 Default
        1 MUR0.5_MUF0.5_PDF13300
        ...

    Returns
    -------
    dict
        Mapping:
            weight index -> weight name
    """

    idx_to_name = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            idx_str, name = line.split(None, 1)

            idx_to_name[int(idx_str)] = name.strip()

    return idx_to_name


def mapping_file_for_sample(sample):
    """
    Return the appropriate cleaned weight mapping file
    for a given WCharm sample.
    """

    if sample == "WCPyPartonplus":
        return WEIGHT_DIR / "WCPyPartonplus_weights_clean.txt"

    if sample == "WCPyPartonminus":
        return WEIGHT_DIR / "WCPyPartonminus_weights_clean.txt"

    if sample.startswith("WCPy8"):
        return WEIGHT_DIR / "weights.py.particle_clean.txt"

    if sample.startswith("WCH7"):
        return WEIGHT_DIR / "weights.hw.particle_clean.txt"

    if sample == "WCHPartonplus":
        return WEIGHT_DIR / "WCHPartonplus_weights_clean.txt"

    if sample == "WCHPartonminus":
        return WEIGHT_DIR / "WCHPartonminus_weights_clean.txt"

    raise RuntimeError(
        f"No weight-mapping rule exists for sample '{sample}'"
    )


def safe_name(text):
    """
    Convert a weight name into a filename-safe string.
    """

    return "".join(
        c if c.isalnum() or c in "._-" else "_"
        for c in text
    )


# ============================================================
# Histogram creation
# ============================================================

def make_etalepton_histogram():
    """
    Create a fresh lepton |eta| histogram.

    The histogram uses the same variable binning as the original
    full make_histograms.py analysis.
    """

    hist = ROOT.TH1F(
        "etalepton",
        "etalepton",
        len(ETA_LEPTON_BIN_ARRAY) - 1,
        ETA_LEPTON_BIN_ARRAY,
    )

    # Required for weighted statistical uncertainties.
    hist.Sumw2()

    # Prevent ROOT directory ownership issues.
    hist.SetDirectory(0)

    return hist


# ============================================================
# Event loop
# ============================================================

def fill_etalepton_histogram(
    tree,
    hist,
    weight_index,
    total_events,
):
    """
    Fill the lepton |eta| histogram for one chosen weight.

    Event weight:

        final_weight =
            (-lepton charge)
            * weightvec[weight_index]
            / total number of events

    Event selection:

        MET > 25 GeV
        lepton pT > 20 GeV
        |lepton eta| < 2.5

    Returns
    -------
    tuple
        pos_eta_sum, neg_eta_sum

    These are retained as diagnostics, as in the original
    analysis.
    """

    pos_eta_sum = 0.0
    neg_eta_sum = 0.0

    nentries = tree.GetEntries()

    for ientry in range(nentries):

        tree.GetEntry(ientry)

        # ----------------------------------------------------
        # Event weight
        # ----------------------------------------------------

        weightxs = (
            (1.0 / total_events)
            * tree.weightvec[weight_index]
        )

        # ----------------------------------------------------
        # Event selection
        # ----------------------------------------------------

        if tree.met_et <= 25:
            continue

        if tree.leptons_pt <= 20:
            continue

        if abs(tree.leptons_eta) >= 2.5:
            continue

        # ----------------------------------------------------
        # Charge sign convention
        # ----------------------------------------------------

        corr = -1.0 * tree.leptons_charge

        final_weight = corr * weightxs

        # ----------------------------------------------------
        # Lepton pseudorapidity
        # ----------------------------------------------------

        eta_lepton = abs(tree.leptons_eta)

        # Diagnostics using the original signed eta.
        if tree.leptons_eta >= 0:
            pos_eta_sum += final_weight
        else:
            neg_eta_sum += final_weight

        # Fill only the observable required for this analysis.
        hist.Fill(
            eta_lepton,
            final_weight,
        )

    return pos_eta_sum, neg_eta_sum


# ============================================================
# Main processing
# ============================================================

def main():

    print()
    print("==============================================")
    print(" Lepton pseudorapidity histogram production")
    print("==============================================")
    print()

    print(f"Data directory:")
    print(f"  {DATA_DIR}")

    print(f"Weight directory:")
    print(f"  {WEIGHT_DIR}")

    print(f"Output directory:")
    print(f"  {OUTPUT_DIR}")

    print()

    # --------------------------------------------------------
    # Loop over samples
    # --------------------------------------------------------

    for signal_sample in SIGNAL_SAMPLES:

        print()
        print("----------------------------------------------")
        print(f"Running sample: {signal_sample}")
        print("----------------------------------------------")

        # ----------------------------------------------------
        # Load corresponding weight map
        # ----------------------------------------------------

        try:

            weight_map_path = mapping_file_for_sample(
                signal_sample
            )

            idx_to_name = load_weight_map_clean(
                weight_map_path
            )

        except Exception as exc:

            print(
                f"  Failed to load weight map: {exc}"
            )

            continue

        # ----------------------------------------------------
        # Open input ROOT file
        # ----------------------------------------------------

        input_path = (
            DATA_DIR
            / f"WCharm_{signal_sample}.root"
        )

        tfile = ROOT.TFile.Open(
            str(input_path),
            "READ",
        )

        if not tfile or tfile.IsZombie():

            print(
                f"  Could not open ROOT file:"
                f"\n    {input_path}"
            )

            continue

        # ----------------------------------------------------
        # Retrieve WCharm tree
        # ----------------------------------------------------

        tree = tfile.Get("WCharmTree")

        if not tree:

            print("  Missing WCharmTree")

            tfile.Close()

            continue

        # ----------------------------------------------------
        # Number of events
        # ----------------------------------------------------

        total_events = tree.GetEntries()

        if total_events <= 0:

            print("  Tree has no entries")

            tfile.Close()

            continue

        # ----------------------------------------------------
        # Determine available weight-vector length
        # ----------------------------------------------------

        tree.GetEntry(0)

        weight_vector_length = len(
            tree.weightvec
        )

        # ----------------------------------------------------
        # Determine genuine weights to process
        # ----------------------------------------------------

        if USE_ALL_WEIGHTS:

            weights_to_run = [
                weight_index
                for weight_index
                in ALL_REQUESTED_WEIGHTS
                if (
                    weight_index in idx_to_name
                    and
                    weight_index
                    < weight_vector_length
                )
            ]

        else:

            weights_to_run = (
                [0]
                if (
                    0 in idx_to_name
                    and
                    weight_vector_length > 0
                )
                else []
            )

        print(
            f"  Number of events: "
            f"{total_events}"
        )

        print(
            f"  Weight vector length: "
            f"{weight_vector_length}"
        )

        print(
            f"  Number of valid weights: "
            f"{len(weights_to_run)}"
        )

        # ----------------------------------------------------
        # Loop over valid weights
        # ----------------------------------------------------

        for weight_index in weights_to_run:

            weight_name = idx_to_name[
                weight_index
            ]

            print(
                f"    Weight {weight_index}: "
                f"{weight_name}"
            )

            # Create a completely fresh histogram for
            # this sample/weight combination.
            hist = make_etalepton_histogram()

            # Fill it.
            (
                pos_eta_sum,
                neg_eta_sum,
            ) = fill_etalepton_histogram(
                tree=tree,
                hist=hist,
                weight_index=weight_index,
                total_events=total_events,
            )

            # ------------------------------------------------
            # Output filename
            # ------------------------------------------------

            output_name = (
                f"output_w{weight_index}_"
                f"{safe_name(weight_name)}_"
                f"{signal_sample}.root"
            )

            output_path = (
                OUTPUT_DIR
                / output_name
            )

            # ------------------------------------------------
            # Write output ROOT file
            # ------------------------------------------------

            output_file = ROOT.TFile(
                str(output_path),
                "RECREATE",
            )

            output_file.cd()

            # Metadata
            ROOT.TNamed(
                "RequestedWeightIndex",
                str(weight_index),
            ).Write()

            ROOT.TNamed(
                "EffectiveWeightIndex",
                str(weight_index),
            ).Write()

            ROOT.TNamed(
                "EffectiveWeightName",
                weight_name,
            ).Write()

            ROOT.TNamed(
                "WeightVecLength",
                str(weight_vector_length),
            ).Write()

            ROOT.TNamed(
                "SampleName",
                signal_sample,
            ).Write()

            ROOT.TNamed(
                "Observable",
                "etalepton",
            ).Write()

            ROOT.TParameter(float)(
                "PosEtaWeightSum",
                pos_eta_sum,
            ).Write()

            ROOT.TParameter(float)(
                "NegEtaWeightSum",
                neg_eta_sum,
            ).Write()

            # The single physics histogram.
            hist.Write()

            output_file.Write()
            output_file.Close()

        # ----------------------------------------------------
        # Finished with this sample
        # ----------------------------------------------------

        tfile.Close()

    print()
    print("==============================================")
    print(" Finished")
    print("==============================================")
    print()
    print(
        "All valid lepton pseudorapidity "
        "histogram files were created."
    )
    print()
    print("Output directory:")
    print(f"  {OUTPUT_DIR}")
    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()