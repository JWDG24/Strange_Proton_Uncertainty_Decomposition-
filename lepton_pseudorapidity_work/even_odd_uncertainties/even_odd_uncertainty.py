#!/usr/bin/env python3

# ============================================================
# even_odd_uncertainty.py
#
# Purpose
# -------
# Self-contained odd/even statistical stability study for the
# lepton-pseudorapidity W+c parton-to-particle correction factors.
#
# This script:
#   1. reads the original ROOT ntuples from project_root/data/
#   2. applies exactly the same fiducial selection as the current
#      mT-cut analysis:
#
#         MET > 25 GeV
#         lepton pT > 20 GeV
#         |eta_lepton| < 2.5
#         m_T^W > 40 GeV
#
#   3. fills nominal-weight histograms for:
#         - all events
#         - odd ROOT tree entries only
#         - even ROOT tree entries only
#
#   4. builds the parton / particle correction factors for:
#         - Pythia W+
#         - Pythia W-
#         - Herwig W+
#         - Herwig W-
#
#   5. compares the odd and even correction factors bin-by-bin.
#
# The main split quantities saved are:
#
#     abs_difference = |C_odd - C_even|
#
#     split_uncertainty =
#         |C_odd - C_even| / 2
#
#     relative_split_uncertainty =
#         split_uncertainty / |C_all|
#
# IMPORTANT
# ---------
# "Odd" and "even" here refer to the ROOT TREE ENTRY INDEX:
#
#     ientry = 0, 1, 2, 3, ...
#
# This does NOT assume that parton- and particle-level samples can
# be matched event-by-event. Each sample is split independently.
#
# Folder structure expected:
#
#   project_root/
#   ├── data/
#   ├── weights/
#   └── lepton_pseudorapidity_work/
#       └── even_odd_Uncertainties/
#           └── even_odd_uncertainty.py
#
# All generated files remain inside:
#
#   even_odd_Uncertainties/outputs/
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

# even_odd_Uncertainties/
# -> lepton_pseudorapidity_work/
LEPTON_WORK_DIR = SCRIPT_DIR.parent

# lepton_pseudorapidity_work/
# -> project root
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

NOMINAL_WEIGHT_INDEX = 0


# ============================================================
# Lepton pseudorapidity binning
# ============================================================

# Exactly the same |eta_lepton| binning as the existing analysis.

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
# Histogram helpers
# ============================================================

def make_eta_histogram(
    sample,
    subset,
):
    """
    Create one |eta_lepton| histogram.

    subset:
        "all"
        "odd"
        "even"
    """

    histogram = ROOT.TH1F(
        f"etalepton_{sample}_{subset}",
        f"{sample} {subset};|#eta_{{#ell}}|;Weighted events",
        len(ETA_LEPTON_BIN_ARRAY) - 1,
        ETA_LEPTON_BIN_ARRAY,
    )

    # Essential for weighted-event statistical errors.
    histogram.Sumw2()
    histogram.SetDirectory(0)

    return histogram


def make_sample_histograms(sample):
    """
    Return all/odd/even histograms for one sample.
    """

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
# W transverse mass
# ============================================================

def calculate_mtw(tree):
    """
    Calculate:

        (m_T^W)^2
          =
        2 pT_lepton MET (1 - cos DeltaPhi)
    """

    delta_phi = abs(
        tree.leptons_phi
        - tree.met_phi
    )

    mtw_squared = (
        2.0
        * tree.leptons_pt
        * tree.met_et
        * (
            1.0
            - ROOT.TMath.Cos(
                delta_phi
            )
        )
    )

    if mtw_squared <= 0.0:
        return 0.0

    return float(
        ROOT.TMath.Sqrt(
            mtw_squared
        )
    )


# ============================================================
# Event selection
# ============================================================

def passes_selection(tree):
    """
    Apply exactly the same selection as the mT-cut eta analysis.
    """

    if tree.met_et <= MET_MIN_GEV:
        return False

    if tree.leptons_pt <= LEPTON_PT_MIN_GEV:
        return False

    if abs(
        tree.leptons_eta
    ) >= LEPTON_ABS_ETA_MAX:
        return False

    mtw_value = calculate_mtw(
        tree
    )

    if mtw_value <= MTW_MIN_GEV:
        return False

    return True


# ============================================================
# Process one sample
# ============================================================

def process_sample(sample):
    """
    Read one ROOT ntuple and fill all/odd/even nominal histograms
    in a single pass through WCharmTree.

    The normalization and charge convention deliberately match the
    current MTWcut_make_histograms_etalepton.py analysis:

        weightxs =
            (1 / total_events) * weightvec[0]

        final_weight =
            (-1 * leptons_charge) * weightxs
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
        f"Input: {input_path}"
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

    # Check that nominal weight 0 physically exists.
    tree.GetEntry(0)

    if not hasattr(
        tree,
        "weightvec",
    ):
        tfile.Close()
        raise RuntimeError(
            f"Sample {sample} has no weightvec branch."
        )

    if len(
        tree.weightvec
    ) <= NOMINAL_WEIGHT_INDEX:
        tfile.Close()
        raise RuntimeError(
            f"Sample {sample} has no nominal weight index 0."
        )

    histograms = make_sample_histograms(
        sample
    )

    accepted_all = 0
    accepted_odd = 0
    accepted_even = 0

    # ========================================================
    # ONE event-tree pass
    # ========================================================

    for ientry in range(
        total_events
    ):

        tree.GetEntry(
            ientry
        )

        if not passes_selection(
            tree
        ):
            continue

        eta_lepton = abs(
            tree.leptons_eta
        )

        charge_factor = (
            -1.0
            * tree.leptons_charge
        )

        weightxs = (
            (1.0 / total_events)
            * tree.weightvec[
                NOMINAL_WEIGHT_INDEX
            ]
        )

        final_weight = (
            charge_factor
            * weightxs
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
        # Odd/even split by ROOT tree entry index
        #
        # Entry numbering starts at zero:
        #   even -> 0, 2, 4, ...
        #   odd  -> 1, 3, 5, ...
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

    print(
        f"  Total entries:             {total_events}"
    )
    print(
        f"  Accepted all:              {accepted_all}"
    )
    print(
        f"  Accepted odd entries:      {accepted_odd}"
    )
    print(
        f"  Accepted even entries:     {accepted_even}"
    )

    return {
        "histograms": histograms,
        "total_events": total_events,
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

    using the same ROOT division convention as the current
    correction-building script.
    """

    correction = h_parton.Clone(
        name
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
    """
    Create an empty histogram with the same eta binning.
    """

    output = reference.Clone(
        name
    )

    output.Reset()
    output.SetDirectory(0)
    output.SetTitle(
        title
    )

    return output


def build_split_uncertainty_histograms(
    h_all,
    h_odd,
    h_even,
    pair_label,
):
    """
    Build three comparison histograms:

      1. signed difference
             C_odd - C_even

      2. absolute difference
             |C_odd - C_even|

      3. split statistical uncertainty
             |C_odd - C_even| / 2

      4. relative split uncertainty
             sigma_split / |C_all|
    """

    h_signed_difference = empty_hist_like(
        h_all,
        f"{pair_label}_odd_minus_even",
        (
            f"{pair_label}: odd - even;"
            f"|#eta_{{#ell}}|;"
            f"C_{{odd}} - C_{{even}}"
        ),
    )

    h_abs_difference = empty_hist_like(
        h_all,
        f"{pair_label}_abs_odd_even_difference",
        (
            f"{pair_label}: |odd - even|;"
            f"|#eta_{{#ell}}|;"
            f"|C_{{odd}} - C_{{even}}|"
        ),
    )

    h_split_uncertainty = empty_hist_like(
        h_all,
        f"{pair_label}_split_stat_uncertainty",
        (
            f"{pair_label}: split statistical uncertainty;"
            f"|#eta_{{#ell}}|;"
            f"#sigma_{{split}}"
        ),
    )

    h_relative_split_uncertainty = empty_hist_like(
        h_all,
        f"{pair_label}_relative_split_stat_uncertainty",
        (
            f"{pair_label}: relative split uncertainty;"
            f"|#eta_{{#ell}}|;"
            f"#sigma_{{split}}/|C_{{all}}|"
        ),
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

        if c_all != 0.0:
            relative_split_uncertainty = (
                split_uncertainty
                / abs(c_all)
            )
        else:
            relative_split_uncertainty = 0.0

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
        "signed_difference": h_signed_difference,
        "abs_difference": h_abs_difference,
        "split_uncertainty": h_split_uncertainty,
        "relative_split_uncertainty": (
            h_relative_split_uncertainty
        ),
    }


# ============================================================
# ROOT output
# ============================================================

def write_sample_histograms(
    sample_results,
):
    """
    Save all raw all/odd/even eta histograms into one ROOT file.
    """

    output_path = (
        ROOT_OUTPUT_DIR
        / "even_odd_sample_histograms.root"
    )

    output_file = ROOT.TFile(
        str(output_path),
        "RECREATE",
    )

    for sample in ALL_SAMPLES:

        sample_directory = (
            output_file.mkdir(
                sample
            )
        )

        sample_directory.cd()

        for subset in (
            "all",
            "odd",
            "even",
        ):

            histogram = (
                sample_results[
                    sample
                ][
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
    """
    Save all correction and split-uncertainty histograms for one pair.
    """

    output_path = (
        ROOT_OUTPUT_DIR
        / f"even_odd_{pair_label}.root"
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
        "Observable",
        "etalepton",
    ).Write()

    ROOT.TNamed(
        "EventSplitDefinition",
        (
            "odd/even defined by ROOT tree entry index; "
            "entry 0 is even"
        ),
    ).Write()

    ROOT.TNamed(
        "MTWCutDefinition",
        "m_T^W > 40 GeV",
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
    """
    Write a bin-by-bin summary.

    Both the raw odd-even difference and the half-difference are
    stored so the supervisor's preferred convention can be chosen
    without rerunning the event loop.
    """

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
            "split_stat_uncertainty_half_difference",
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
    """
    Simple ROOT styling for the comparison plot.
    """

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
    """
    Save one two-panel diagnostic plot per generator/charge pair.
    """

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

    # --------------------------------------------------------
    # Top: correction comparison
    # --------------------------------------------------------

    canvas.cd(
        1
    )

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

            value = histogram.GetBinContent(
                ibin
            )

            if math.isfinite(
                value
            ):
                values.append(
                    value
                )

    if values:
        minimum = min(
            values
        )
        maximum = max(
            values
        )

        spread = max(
            maximum - minimum,
            0.01,
        )

        h_all.SetMinimum(
            minimum - 0.20 * spread
        )

        h_all.SetMaximum(
            maximum + 0.20 * spread
        )

    h_all.Draw(
        "E1"
    )

    h_odd.Draw(
        "E1 SAME"
    )

    h_even.Draw(
        "E1 SAME"
    )

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

    # --------------------------------------------------------
    # Bottom: split uncertainty
    # --------------------------------------------------------

    canvas.cd(
        2
    )

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
    """
    Write a compact text record of what was run.
    """

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
            "  even = ROOT tree entry indices 0, 2, 4, ...\n"
        )

        f.write(
            "  odd  = ROOT tree entry indices 1, 3, 5, ...\n\n"
        )

        f.write(
            "Physics selection:\n"
        )

        f.write(
            f"  MET > {MET_MIN_GEV} GeV\n"
        )

        f.write(
            f"  lepton pT > {LEPTON_PT_MIN_GEV} GeV\n"
        )

        f.write(
            f"  |eta_lepton| < {LEPTON_ABS_ETA_MAX}\n"
        )

        f.write(
            f"  m_T^W > {MTW_MIN_GEV} GeV\n\n"
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
            "  abs_difference = |C_odd - C_even|\n"
        )

        f.write(
            "  split_stat_uncertainty = abs_difference / 2\n\n"
        )

        f.write(
            "Accepted-event counts:\n"
        )

        for sample in ALL_SAMPLES:

            result = sample_results[
                sample
            ]

            f.write(
                (
                    f"  {sample}: "
                    f"all={result['accepted_all']}, "
                    f"odd={result['accepted_odd']}, "
                    f"even={result['accepted_even']}\n"
                )
            )

        f.write(
            "\nGenerated pair outputs:\n"
        )

        for pair_label in SAMPLE_PAIRS:

            if pair_label not in pair_results:
                continue

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
        " Odd/even statistical uncertainty study"
    )
    print(
        " Lepton pseudorapidity, m_T^W > 40 GeV"
    )
    print(
        "============================================================"
    )
    print()

    print(
        f"Project directory: {PROJECT_DIR}"
    )
    print(
        f"Data directory:    {DATA_DIR}"
    )
    print(
        f"Output directory:  {OUTPUT_DIR}"
    )
    print()

    if not DATA_DIR.is_dir():
        raise RuntimeError(
            f"Data directory does not exist:\n"
            f"  {DATA_DIR}"
        )

    # ========================================================
    # Process each raw sample ONCE.
    # ========================================================

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
        f"Saved sample histograms:\n  {sample_root_path}"
    )

    # ========================================================
    # Build correction factors and split uncertainties.
    # ========================================================

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
            ] = make_correction_histogram(
                h_particle=h_particle,
                h_parton=h_parton,
                name=(
                    f"{pair_label}_"
                    f"correction_{subset}"
                ),
            )

        split_hists = (
            build_split_uncertainty_histograms(
                h_all=corrections[
                    "all"
                ],
                h_odd=corrections[
                    "odd"
                ],
                h_even=corrections[
                    "even"
                ],
                pair_label=pair_label,
            )
        )

        root_path = (
            write_pair_root_output(
                pair_label=pair_label,
                corrections=corrections,
                split_hists=split_hists,
            )
        )

        csv_path = (
            write_pair_csv(
                pair_label=pair_label,
                corrections=corrections,
                split_hists=split_hists,
            )
        )

        (
            png_path,
            pdf_path,
        ) = save_pair_plot(
            pair_label=pair_label,
            corrections=corrections,
            split_hists=split_hists,
        )

        pair_results[
            pair_label
        ] = {
            "corrections": corrections,
            "split_hists": split_hists,
            "root_path": root_path,
            "csv_path": csv_path,
            "png_path": png_path,
            "pdf_path": pdf_path,
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
            sample_results=sample_results,
            pair_results=pair_results,
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
        "All outputs are contained inside:"
    )
    print(
        f"  {OUTPUT_DIR}"
    )
    print()

    print(
        "The main CSV quantity is:"
    )
    print(
        "  split_stat_uncertainty_half_difference"
    )
    print(
        "    = |C_odd - C_even| / 2"
    )
    print()

    print(
        "The raw |C_odd - C_even| difference is also saved."
    )
    print()

    print(
        f"Run summary:\n  {summary_path}"
    )
    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
