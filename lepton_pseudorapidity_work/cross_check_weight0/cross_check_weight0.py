#!/usr/bin/env python3

# ============================================================
# cross_check_weight0.py
#
# Cross-check of the nominal Pythia W+c lepton-pseudorapidity
# parton-to-particle correction factors using the new ROOT files
# supplied for the weight-0-only validation sample.
#
# Expected location:
#
#   uncertainty_decomposition/
#   └── lepton_pseudorapidity_work/
#       └── cross_check_weight0/
#           ├── cross_check_weight0.py
#           └── cross_check_ROOT_files/
#               ├── WCharm_WCPy8minus10.root
#               ├── WCharm_WCPy8minus11.root
#               ├── ...
#               ├── WCharm_WCPy8plus....root
#               ├── WCharm_WCPyPartonminus....root
#               └── WCharm_WCPyPartonplus....root
#
# The script:
#   1. automatically finds all new Pythia particle/parton ROOT files;
#   2. treats all fragments belonging to one sample as one combined sample;
#   3. uses ONLY weightvec[0];
#   4. applies the same fiducial selection as the MTW-cut analysis:
#
#         MET > 25 GeV
#         lepton pT > 20 GeV
#         |eta_lepton| < 2.5
#         m_T^W > 40 GeV
#
#   5. constructs:
#
#         C_new(bin) = N_parton_new(bin) / N_particle_new(bin)
#
#      for Pythia W+ and Pythia W-;
#
#   6. compares C_new with the existing nominal MTW-cut correction
#      stored in:
#
#         MTWcut_build_corrections_etalepton_csv/
#
#   7. writes CSV, ROOT, PNG/PDF and (if Plotly is installed)
#      an interactive HTML comparison.
#
# IMPORTANT:
#   If only the particle-level files are present, the script will
#   process and save those histograms, but it will state that the
#   correction cross-check cannot yet be completed. Rerun the script
#   once the corresponding parton-level files appear.
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

# The comparison MUST use the original analysis with the same
# m_T^W > 40 GeV selection.
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

NOMINAL_WEIGHT_INDEX = 0


# ============================================================
# Lepton-pseudorapidity binning
# ============================================================

# Identical to the current lepton-pseudorapidity analysis.

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
# New sample definitions
# ============================================================

# The supplied files are split into many ROOT-file fragments.
# These glob patterns intentionally accept either a numbered
# suffix or a single unnumbered file.

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


# Particle / parton correction pairs.

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
# ROOT helpers
# ============================================================

def make_eta_histogram(sample):
    """
    Create one detached weighted |eta_lepton| histogram.
    """

    histogram = ROOT.TH1F(
        f"etalepton_{sample}",
        (
            f"{sample};"
            f"|#eta_{{#ell}}|;"
            f"Weighted events"
        ),
        len(
            ETA_LEPTON_BIN_ARRAY
        ) - 1,
        ETA_LEPTON_BIN_ARRAY,
    )

    # Correct weighted-event errors.
    histogram.Sumw2()

    histogram.SetDirectory(0)

    return histogram


def calculate_mtw(tree):
    """
    Calculate:

        (m_T^W)^2
          =
        2 pT_lepton MET [1 - cos(DeltaPhi)]
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


def passes_selection(tree):
    """
    Apply exactly the same event selection as the current
    MTW-cut lepton-pseudorapidity analysis.
    """

    if tree.met_et <= MET_MIN_GEV:
        return False

    if tree.leptons_pt <= LEPTON_PT_MIN_GEV:
        return False

    if abs(
        tree.leptons_eta
    ) >= LEPTON_ABS_ETA_MAX:
        return False

    if calculate_mtw(
        tree
    ) <= MTW_MIN_GEV:
        return False

    return True


# ============================================================
# File discovery
# ============================================================

def discover_files():
    """
    Discover all weight-0 cross-check files belonging to each
    of the four Pythia particle/parton samples.
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

        paths = sorted(
            CROSS_CHECK_ROOT_DIR.glob(
                pattern
            )
        )

        discovered[
            sample
        ] = paths

    return discovered


# ============================================================
# Input validation
# ============================================================

def get_tree_from_file(path):
    """
    Open a ROOT file and retrieve WCharmTree.

    Caller owns the returned TFile and must close it.
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
    """
    Return total WCharmTree entries across all fragments
    belonging to one logical sample.
    """

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
# Process one combined new sample
# ============================================================

def process_combined_sample(
    sample,
    paths,
):
    """
    Combine all ROOT-file fragments belonging to one logical
    sample into one weighted |eta_lepton| histogram.

    The normalization mirrors the existing analysis:

        final_weight
          =
        (-leptons_charge)
        * (1 / N_total_combined)
        * weightvec[0]

    where N_total_combined is the total number of events across
    all fragments of this new sample.
    """

    if len(paths) == 0:
        return None

    total_entries = count_total_entries(
        paths
    )

    if total_entries <= 0:
        raise RuntimeError(
            f"Combined sample {sample} has no events."
        )

    histogram = make_eta_histogram(
        sample
    )

    events_after_base_cuts = 0
    events_after_mtw_cut = 0

    print()
    print(
        "------------------------------------------------------------"
    )
    print(
        f"Processing combined sample: {sample}"
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

        if nentries == 0:
            root_file.Close()
            continue

        # Check required branches / weight 0 using first entry.
        tree.GetEntry(0)

        required_branches = [
            "met_et",
            "met_phi",
            "leptons_pt",
            "leptons_eta",
            "leptons_phi",
            "leptons_charge",
            "weightvec",
        ]

        missing = [
            branch
            for branch in required_branches
            if not hasattr(
                tree,
                branch
            )
        ]

        if missing:
            root_file.Close()

            raise RuntimeError(
                f"Missing branches in {path.name}: "
                f"{missing}"
            )

        if len(
            tree.weightvec
        ) <= NOMINAL_WEIGHT_INDEX:
            root_file.Close()

            raise RuntimeError(
                f"No weightvec[0] in {path.name}"
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

            # ----------------------------------------------
            # Existing base selection
            # ----------------------------------------------

            if tree.met_et <= MET_MIN_GEV:
                continue

            if tree.leptons_pt <= LEPTON_PT_MIN_GEV:
                continue

            if abs(
                tree.leptons_eta
            ) >= LEPTON_ABS_ETA_MAX:
                continue

            events_after_base_cuts += 1

            # ----------------------------------------------
            # m_T^W > 40 GeV
            # ----------------------------------------------

            if calculate_mtw(
                tree
            ) <= MTW_MIN_GEV:
                continue

            events_after_mtw_cut += 1

            eta_lepton = abs(
                tree.leptons_eta
            )

            charge_factor = (
                -1.0
                * tree.leptons_charge
            )

            weightxs = (
                (1.0 / total_entries)
                * tree.weightvec[
                    NOMINAL_WEIGHT_INDEX
                ]
            )

            final_weight = (
                charge_factor
                * weightxs
            )

            histogram.Fill(
                eta_lepton,
                final_weight,
            )

        root_file.Close()

    print(
        f"  After base cuts: {events_after_base_cuts}"
    )
    print(
        f"  After mT cut:    {events_after_mtw_cut}"
    )

    return {
        "sample": sample,
        "paths": paths,
        "total_entries": total_entries,
        "events_after_base_cuts": (
            events_after_base_cuts
        ),
        "events_after_mtw_cut": (
            events_after_mtw_cut
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

        correction = parton / particle

    using ROOT's ordinary propagated histogram errors.
    """

    ratio = h_parton.Clone(
        name
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
# Reference CSV
# ============================================================

def read_reference_csv(path):
    """
    Read the existing MTW-cut nominal correction and statistical
    uncertainty from the main analysis CSV.
    """

    if not path.is_file():
        raise RuntimeError(
            "Reference MTW-cut CSV not found:\n"
            f"  {path}\n\n"
            "The new cross-check must be compared with the "
            "m_T^W > 40 GeV reference analysis."
        )

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
    """
    Ensure new and reference correction factors use identical
    pseudorapidity bins.
    """

    nbins = correction_hist.GetNbinsX()

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
# Cross-check calculations
# ============================================================

def build_comparison_rows(
    pair_label,
    correction_hist,
    reference_rows,
):
    """
    Build bin-by-bin old/new comparison quantities.

    The combined statistical denominator is useful only if the
    new cross-check sample is statistically independent of the
    original sample. It is therefore saved as a diagnostic,
    not interpreted automatically.
    """

    validate_reference_binning(
        reference_rows,
        correction_hist,
    )

    comparison_rows = []

    for ibin in range(
        1,
        correction_hist.GetNbinsX() + 1,
    ):

        reference = (
            reference_rows[
                ibin - 1
            ]
        )

        old_correction = (
            reference[
                "correction_factor"
            ]
        )

        old_stat_unc = (
            reference[
                "stat_unc"
            ]
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

        difference = (
            new_correction
            - old_correction
        )

        absolute_difference = abs(
            difference
        )

        if old_correction != 0.0:

            relative_difference = (
                absolute_difference
                / abs(
                    old_correction
                )
            )

        else:

            relative_difference = 0.0

        combined_stat_unc = math.sqrt(
            old_stat_unc * old_stat_unc
            + new_stat_unc * new_stat_unc
        )

        if combined_stat_unc > 0.0:

            difference_over_combined_stat = (
                difference
                / combined_stat_unc
            )

        else:

            difference_over_combined_stat = 0.0

        comparison_rows.append({
            "pair_label": pair_label,
            "bin": ibin,
            "bin_low_edge": (
                reference[
                    "bin_low_edge"
                ]
            ),
            "bin_up_edge": (
                reference[
                    "bin_up_edge"
                ]
            ),
            "C_reference": (
                old_correction
            ),
            "reference_stat_unc": (
                old_stat_unc
            ),
            "C_new_crosscheck": (
                new_correction
            ),
            "new_crosscheck_stat_unc": (
                new_stat_unc
            ),
            "new_minus_reference": (
                difference
            ),
            "abs_new_minus_reference": (
                absolute_difference
            ),
            "relative_abs_difference": (
                relative_difference
            ),
            "relative_abs_difference_percent": (
                100.0
                * relative_difference
            ),
            "combined_stat_unc_if_independent": (
                combined_stat_unc
            ),
            "difference_over_combined_stat_if_independent": (
                difference_over_combined_stat
            ),
        })

    return comparison_rows


# ============================================================
# CSV output
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


def write_comparison_csv(
    pair_label,
    comparison_rows,
):
    """
    Save one comparison CSV per charge channel.
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

        writer.writerows(
            comparison_rows
        )

    return output_path


# ============================================================
# ROOT output
# ============================================================

def make_reference_histogram(
    reference_rows,
    name,
):
    """
    Convert the reference CSV into a ROOT histogram for plotting.
    """

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

        ibin = row[
            "bin"
        ]

        histogram.SetBinContent(
            ibin,
            row[
                "correction_factor"
            ],
        )

        histogram.SetBinError(
            ibin,
            row[
                "stat_unc"
            ],
        )

    return histogram


def write_pair_root_file(
    pair_label,
    particle_hist,
    parton_hist,
    new_correction_hist,
    reference_hist,
):
    """
    Save all principal cross-check objects for one charge.
    """

    output_path = (
        ROOT_OUTPUT_DIR
        / f"cross_check_{pair_label}.root"
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
        "Weight",
        "weightvec[0] only",
    ).Write()

    ROOT.TNamed(
        "EventSelection",
        (
            "met_et > 25 GeV; "
            "leptons_pt > 20 GeV; "
            "abs(leptons_eta) < 2.5; "
            "m_T^W > 40 GeV"
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

    reference_hist.Write(
        "reference_correction"
    )

    output_file.Write()
    output_file.Close()

    return output_path


# ============================================================
# Static ROOT plot
# ============================================================

def save_static_plot(
    pair_label,
    display_label,
    reference_hist,
    new_hist,
    comparison_rows,
):
    """
    Save a two-panel ROOT comparison plot:
      top    : reference vs new correction
      bottom : new - reference
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

    # --------------------------------------------------------
    # Top panel
    # --------------------------------------------------------

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
            f"{display_label}: weight-0 cross-check;"
            f"|#eta_{{#ell}}|;"
            f"Correction factor"
        )
    )

    all_values = []

    for histogram in (
        reference_hist,
        new_hist,
    ):

        for ibin in range(
            1,
            histogram.GetNbinsX() + 1,
        ):

            all_values.append(
                histogram.GetBinContent(
                    ibin
                )
            )

    minimum = min(
        all_values
    )

    maximum = max(
        all_values
    )

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
        0.60,
        0.75,
        0.88,
        0.88,
    )

    legend.AddEntry(
        reference_hist,
        "Reference MTW-cut correction",
        "lep",
    )

    legend.AddEntry(
        new_hist,
        "New weight-0 cross-check",
        "lep",
    )

    legend.Draw()

    # --------------------------------------------------------
    # Bottom panel
    # --------------------------------------------------------

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

    for row in comparison_rows:

        difference_hist.SetBinContent(
            row[
                "bin"
            ],
            row[
                "new_minus_reference"
            ],
        )

        # Error on the difference, if the samples are independent.
        difference_hist.SetBinError(
            row[
                "bin"
            ],
            row[
                "combined_stat_unc_if_independent"
            ],
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
# Optional Plotly HTML
# ============================================================

def save_interactive_plot(
    completed_pairs,
):
    """
    Save one interactive Plotly HTML containing a dropdown for
    Pythia W+ / W-. If Plotly is unavailable, simply skip this
    optional output.
    """

    if not completed_pairs:
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
            "Reference and new correction factors",
            "New minus reference",
        ),
        row_heights=[
            0.66,
            0.34,
        ],
    )

    pair_keys = list(
        completed_pairs.keys()
    )

    traces_per_pair = 3

    for pair_index, pair_label in enumerate(
        pair_keys
    ):

        result = completed_pairs[
            pair_label
        ]

        rows = result[
            "comparison_rows"
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
                name="Reference MTW-cut",
                visible=visible,
                hovertemplate=(
                    "|ηℓ| = %{x:.3f}"
                    "<br>C_ref = %{y:.6f}"
                    "<extra></extra>"
                ),
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
                name="New weight-0 cross-check",
                visible=visible,
                hovertemplate=(
                    "|ηℓ| = %{x:.3f}"
                    "<br>C_new = %{y:.6f}"
                    "<extra></extra>"
                ),
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
                name="New − reference",
                visible=visible,
                hovertemplate=(
                    "|ηℓ| = %{x:.3f}"
                    "<br>ΔC = %{y:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    total_traces = (
        traces_per_pair
        * len(
            pair_keys
        )
    )

    buttons = []

    for pair_index, pair_label in enumerate(
        pair_keys
    ):

        visible_mask = [
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
            visible_mask[
                trace_index
            ] = True

        display = (
            completed_pairs[
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
                        "visible": visible_mask,
                    },
                    {
                        "title.text": (
                            "Weight-0 Pythia cross-check"
                            f"<br><sup>{display} · "
                            "m<sub>T</sub><sup>W</sup> &gt; 40 GeV"
                            "</sup>"
                        ),
                    },
                ],
            )
        )

    first_display = (
        completed_pairs[
            pair_keys[0]
        ][
            "display"
        ]
    )

    figure.update_layout(
        title=dict(
            text=(
                "Weight-0 Pythia cross-check"
                f"<br><sup>{first_display} · "
                "m<sub>T</sub><sup>W</sup> &gt; 40 GeV"
                "</sup>"
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
        title_text="C_new − C_reference",
        row=2,
        col=1,
        zeroline=True,
    )

    figure.update_xaxes(
        title_text="Lepton |η|",
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
    completed_pairs,
):
    """
    Save a compact record of the cross-check production.
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
            "Weight-0 Pythia cross-check\n"
        )
        f.write(
            "==========================\n\n"
        )

        f.write(
            "Selection:\n"
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
            "Weight:\n"
        )
        f.write(
            "  weightvec[0] only\n\n"
        )

        f.write(
            "Discovered ROOT fragments:\n"
        )

        for sample in NEW_SAMPLE_PATTERNS:

            f.write(
                f"  {sample}: "
                f"{len(discovered[sample])}\n"
            )

        f.write(
            "\nProcessed samples:\n"
        )

        for sample, result in sample_results.items():

            if result is None:
                f.write(
                    f"  {sample}: not available\n"
                )
                continue

            f.write(
                (
                    f"  {sample}: "
                    f"entries={result['total_entries']}, "
                    f"after_mtw={result['events_after_mtw_cut']}\n"
                )
            )

        f.write(
            "\nCompleted correction pairs:\n"
        )

        if not completed_pairs:

            f.write(
                "  none - corresponding particle and parton "
                "files are not both available yet\n"
            )

        else:

            for pair_label in completed_pairs:

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
        " Pythia weight-0 correction-factor cross-check"
    )
    print(
        "============================================================"
    )
    print()

    print(
        f"Cross-check ROOT directory:\n"
        f"  {CROSS_CHECK_ROOT_DIR}"
    )
    print()

    print(
        f"Reference MTW-cut CSV directory:\n"
        f"  {REFERENCE_CSV_DIR}"
    )
    print()

    discovered = discover_files()

    print(
        "Discovered files:"
    )

    for sample in NEW_SAMPLE_PATTERNS:

        print(
            f"  {sample:18s}: "
            f"{len(discovered[sample])}"
        )

    # ========================================================
    # Build all available new particle/parton histograms.
    # ========================================================

    sample_results = {}

    for sample in NEW_SAMPLE_PATTERNS:

        paths = discovered[
            sample
        ]

        if not paths:

            print()
            print(
                f"No files currently found for {sample}."
            )

            sample_results[
                sample
            ] = None

            continue

        sample_results[
            sample
        ] = process_combined_sample(
            sample,
            paths,
        )

    # Save all currently available combined sample histograms,
    # including particle-only production before parton files arrive.

    combined_samples_path = (
        ROOT_OUTPUT_DIR
        / "combined_new_weight0_samples.root"
    )

    combined_file = ROOT.TFile(
        str(
            combined_samples_path
        ),
        "RECREATE",
    )

    for sample, result in sample_results.items():

        if result is None:
            continue

        result[
            "histogram"
        ].Write(
            f"{sample}_etalepton"
        )

    combined_file.Write()
    combined_file.Close()

    print()
    print(
        "Saved available combined sample histograms:"
    )
    print(
        f"  {combined_samples_path}"
    )

    # ========================================================
    # Complete cross-checks only when both sides of the
    # particle/parton pair are available.
    # ========================================================

    completed_pairs = {}

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
                "  Cross-check not yet complete."
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
                h_particle=particle_result[
                    "histogram"
                ],
                h_parton=parton_result[
                    "histogram"
                ],
                name=(
                    f"{pair_label}_"
                    f"new_weight0_correction"
                ),
            )
        )

        reference_csv_path = (
            REFERENCE_CSV_DIR
            / definition[
                "reference_csv"
            ]
        )

        reference_rows = (
            read_reference_csv(
                reference_csv_path
            )
        )

        comparison_rows = (
            build_comparison_rows(
                pair_label=pair_label,
                correction_hist=(
                    new_correction_hist
                ),
                reference_rows=(
                    reference_rows
                ),
            )
        )

        reference_hist = (
            make_reference_histogram(
                reference_rows,
                f"{pair_label}_reference",
            )
        )

        csv_path = (
            write_comparison_csv(
                pair_label,
                comparison_rows,
            )
        )

        root_path = (
            write_pair_root_file(
                pair_label=pair_label,
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

        (
            png_path,
            pdf_path,
        ) = save_static_plot(
            pair_label=pair_label,
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
            comparison_rows=(
                comparison_rows
            ),
        )

        completed_pairs[
            pair_label
        ] = {
            "display": (
                definition[
                    "display"
                ]
            ),
            "comparison_rows": (
                comparison_rows
            ),
            "csv_path": csv_path,
            "root_path": root_path,
            "png_path": png_path,
            "pdf_path": pdf_path,
        }

        print(
            f"  CSV:  {csv_path}"
        )
        print(
            f"  ROOT: {root_path}"
        )
        print(
            f"  PNG:  {png_path}"
        )
        print(
            f"  PDF:  {pdf_path}"
        )

    interactive_path = (
        save_interactive_plot(
            completed_pairs
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
            discovered=discovered,
            sample_results=sample_results,
            completed_pairs=completed_pairs,
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
        f"All outputs are contained inside:\n"
        f"  {OUTPUT_DIR}"
    )
    print()

    print(
        f"Run summary:\n"
        f"  {summary_path}"
    )
    print()

    if not completed_pairs:

        print(
            "No correction pair could yet be completed."
        )
        print(
            "This is expected if the new parton-level files "
            "have not appeared yet."
        )
        print(
            "Once they are added to cross_check_ROOT_files/, "
            "simply rerun this same script."
        )
        print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
