#!/usr/bin/env python3

# ============================================================
# build_corrections_etalepton.py
#
# Purpose:
#   Read the lepton-pseudorapidity histogram ROOT files produced
#   by make_histograms_etalepton.py and calculate:
#
#       - nominal correction factors
#       - statistical uncertainty
#       - scale uncertainty
#       - PDF uncertainty
#       - shower uncertainty
#       - model uncertainty
#       - total uncertainty
#       - Pythia/Herwig generator difference
#
# Observable:
#       etalepton = |eta_lepton|
#
# Correction factor:
#
#       C(bin) = parton(bin) / particle(bin)
#
# Important:
#   - Only weights existing at BOTH particle and parton level
#     are used for each generator pair.
#   - Missing weights are skipped.
#   - No fake weights are introduced.
#
# Expected folder structure:
#
#   lepton_pseudorapidity_work/
#   ├── make_histograms_etalepton_outputs/
#   ├── build_corrections_etalepton.py
#   ├── build_corrections_etalepton_outputs/
#   └── build_corrections_etalepton_csv/
#
# ============================================================

import ROOT
import re
import math
import csv

from pathlib import Path


# ============================================================
# ROOT settings
# ============================================================

ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)


# ============================================================
# Directory structure
# ============================================================

# This script lives directly inside:
#     lepton_pseudorapidity_work/

SCRIPT_DIR = Path(__file__).resolve().parent

# Input ROOT files from the previous stage.
INPUT_DIR = (
    SCRIPT_DIR
    / "make_histograms_etalepton_outputs"
)

# ROOT correction outputs.
OUTPUT_DIR = (
    SCRIPT_DIR
    / "build_corrections_etalepton_outputs"
)

# CSV correction outputs.
CSV_DIR = (
    SCRIPT_DIR
    / "build_corrections_etalepton_csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Observable
# ============================================================

OBSERVABLE = "etalepton"


# ============================================================
# Particle / parton sample pairings
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


# ============================================================
# Systematic weight groups
# ============================================================

# Scale variations
SCALE_WEIGHTS = [
    1,
    112,
    215,
    226,
    237,
    248,
    259,
    270,
]

# Modelling variations
MODEL_WEIGHTS = [
    292,
    293,
    314,
    315,
]

# Shower variations
SHOWER_WEIGHTS = (
    list(range(294, 314))
    + [316, 317]
)

# Explicitly excluded weight
EXCLUDED_WEIGHTS = [
    318,
]


# ============================================================
# Input filename parsing
# ============================================================

def parse_output_filename(filename):
    """
    Parse histogram files produced by
    make_histograms_etalepton.py.

    Expected filename format:

        output_w<index>_<weight_name>_<sample>.root

    Returns
    -------
    dict or None
        Information extracted from the filename.
    """

    pattern = (
        r"output_w(\d+)_(.+)_"
        r"(WCH7minus|WCH7plus|"
        r"WCHPartonminus|WCHPartonplus|"
        r"WCPy8minus|WCPy8plus|"
        r"WCPyPartonminus|WCPyPartonplus)"
        r"\.root$"
    )

    match = re.match(
        pattern,
        filename,
    )

    if not match:
        return None

    return {

        "weight_index": int(
            match.group(1)
        ),

        "weight_name": match.group(2),

        "sample": match.group(3),

        "path": (
            INPUT_DIR
            / filename
        ),
    }


# ============================================================
# Scan histogram directory
# ============================================================

def scan_input_files():
    """
    Scan all ROOT files from the histogram-production stage.

    Returns
    -------
    dict

        files_by_sample[sample][weight_index] = info
    """

    files_by_sample = {}

    if not INPUT_DIR.is_dir():
        raise RuntimeError(
            f"Input directory does not exist:\n"
            f"  {INPUT_DIR}"
        )

    for path in INPUT_DIR.iterdir():

        if not path.is_file():
            continue

        if path.suffix != ".root":
            continue

        info = parse_output_filename(
            path.name
        )

        if info is None:
            continue

        sample = info["sample"]

        weight_index = (
            info["weight_index"]
        )

        files_by_sample.setdefault(
            sample,
            {},
        )

        files_by_sample[sample][
            weight_index
        ] = info

    return files_by_sample


# ============================================================
# ROOT file helpers
# ============================================================

def open_root_files_for_sample(sample_files):
    """
    Open every weight ROOT file belonging to one sample.

    Returns
    -------
    dict

        opened[weight_index] = ROOT.TFile
    """

    opened = {}

    for weight_index, info in sample_files.items():

        root_file = ROOT.TFile.Open(
            str(info["path"]),
            "READ",
        )

        if (
            not root_file
            or root_file.IsZombie()
        ):

            raise RuntimeError(
                "Could not open ROOT file:\n"
                f"  {info['path']}"
            )

        opened[
            weight_index
        ] = root_file

    return opened


def close_root_files(opened_files):
    """
    Close all ROOT files belonging to a sample.
    """

    for root_file in opened_files.values():
        root_file.Close()


def get_etalepton_histogram(
    root_file,
    clone_name,
):
    """
    Retrieve the etalepton histogram and clone it
    into memory.
    """

    hist = root_file.Get(
        OBSERVABLE
    )

    if not hist:

        raise RuntimeError(
            f"Histogram '{OBSERVABLE}' not found "
            f"in file:\n"
            f"  {root_file.GetName()}"
        )

    cloned = hist.Clone(
        clone_name
    )

    cloned.SetDirectory(0)

    return cloned


# ============================================================
# Correction factor
# ============================================================

def make_ratio_histogram(
    h_particle,
    h_parton,
    out_name,
):
    """
    Construct:

        correction = parton / particle
    """

    ratio = h_parton.Clone(
        out_name
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
# Empty histogram helper
# ============================================================

def empty_hist_like(
    reference_hist,
    name,
):
    """
    Create an empty histogram with exactly the same binning
    as the reference histogram.
    """

    out = reference_hist.Clone(
        name
    )

    out.Reset()
    out.SetDirectory(0)

    return out


# ============================================================
# PDF weight identification
# ============================================================

def is_pdf_weight_name(
    weight_name,
):
    """
    Identify PDF variations from their weight name.

    This deliberately uses the same criterion as the original
    build_corrections.py.
    """

    return weight_name.startswith(
        "MUR1.0_MUF1.0_PDF"
    )


# ============================================================
# Envelope uncertainty
# ============================================================

def compute_envelope_uncertainty(
    nominal_hist,
    varied_hists,
    out_name,
):
    """
    For every eta bin calculate:

        max |C_variation - C_nominal|

    Used for:
        scale
        shower
        model
    """

    out = empty_hist_like(
        nominal_hist,
        out_name,
    )

    for ibin in range(
        1,
        nominal_hist.GetNbinsX() + 1,
    ):

        central = (
            nominal_hist.GetBinContent(
                ibin
            )
        )

        max_deviation = 0.0

        for hist in varied_hists:

            deviation = abs(
                hist.GetBinContent(ibin)
                - central
            )

            if deviation > max_deviation:
                max_deviation = deviation

        out.SetBinContent(
            ibin,
            max_deviation,
        )

        out.SetBinError(
            ibin,
            0.0,
        )

    return out


# ============================================================
# RMS uncertainty
# ============================================================

def compute_rms_uncertainty(
    nominal_hist,
    varied_hists,
    out_name,
):
    """
    Calculate the RMS deviation of the PDF variations
    from the nominal correction factor.
    """

    out = empty_hist_like(
        nominal_hist,
        out_name,
    )

    if len(varied_hists) == 0:
        return out

    for ibin in range(
        1,
        nominal_hist.GetNbinsX() + 1,
    ):

        central = (
            nominal_hist.GetBinContent(
                ibin
            )
        )

        sum_squared = 0.0

        for hist in varied_hists:

            delta = (
                hist.GetBinContent(ibin)
                - central
            )

            sum_squared += (
                delta * delta
            )

        rms = math.sqrt(
            sum_squared
            / len(varied_hists)
        )

        out.SetBinContent(
            ibin,
            rms,
        )

        out.SetBinError(
            ibin,
            0.0,
        )

    return out


# ============================================================
# Statistical uncertainty
# ============================================================

def compute_stat_uncertainty_from_ratio(
    ratio_hist,
    out_name,
):
    """
    Store the propagated ROOT bin error on the nominal
    correction factor as an uncertainty histogram.
    """

    out = empty_hist_like(
        ratio_hist,
        out_name,
    )

    for ibin in range(
        1,
        ratio_hist.GetNbinsX() + 1,
    ):

        out.SetBinContent(
            ibin,
            ratio_hist.GetBinError(
                ibin
            ),
        )

        out.SetBinError(
            ibin,
            0.0,
        )

    return out


# ============================================================
# Total uncertainty
# ============================================================

def compute_total_uncertainty(
    nominal_hist,
    uncertainty_hists,
    out_name,
):
    """
    Add all supplied uncertainties in quadrature:

        sigma_total =
            sqrt(
                sigma_stat^2
                + sigma_scale^2
                + sigma_pdf^2
                + sigma_shower^2
                + sigma_model^2
            )
    """

    out = empty_hist_like(
        nominal_hist,
        out_name,
    )

    for ibin in range(
        1,
        nominal_hist.GetNbinsX() + 1,
    ):

        sum_squared = 0.0

        for hist in uncertainty_hists:

            sigma = (
                hist.GetBinContent(
                    ibin
                )
            )

            sum_squared += (
                sigma * sigma
            )

        out.SetBinContent(
            ibin,
            math.sqrt(sum_squared),
        )

        out.SetBinError(
            ibin,
            0.0,
        )

    return out


# ============================================================
# Generator difference
# ============================================================

def compute_generator_difference(
    hist_a,
    hist_b,
    out_name,
):
    """
    Compute the absolute difference between the nominal
    Pythia and Herwig correction factors:

        |C_Pythia - C_Herwig|
    """

    out = empty_hist_like(
        hist_a,
        out_name,
    )

    for ibin in range(
        1,
        hist_a.GetNbinsX() + 1,
    ):

        difference = abs(
            hist_a.GetBinContent(ibin)
            - hist_b.GetBinContent(ibin)
        )

        out.SetBinContent(
            ibin,
            difference,
        )

        out.SetBinError(
            ibin,
            0.0,
        )

    return out


# ============================================================
# Build systematic correction variations
# ============================================================

def build_variation_histograms(
    weight_list,
    particle_open_files,
    parton_open_files,
    label,
):
    """
    For every supplied systematic weight:

        C_weight =
            parton_weight / particle_weight
    """

    varied = []

    for weight_index in weight_list:

        h_particle = (
            get_etalepton_histogram(
                particle_open_files[
                    weight_index
                ],
                (
                    f"{OBSERVABLE}_particle_"
                    f"{label}_w{weight_index}"
                ),
            )
        )

        h_parton = (
            get_etalepton_histogram(
                parton_open_files[
                    weight_index
                ],
                (
                    f"{OBSERVABLE}_parton_"
                    f"{label}_w{weight_index}"
                ),
            )
        )

        h_corr = make_ratio_histogram(
            h_particle,
            h_parton,
            (
                f"{OBSERVABLE}_corr_"
                f"{label}_w{weight_index}"
            ),
        )

        varied.append(
            h_corr
        )

    return varied


# ============================================================
# ROOT writing helper
# ============================================================

def write_hist_to_file(
    output_file,
    hist,
):
    """
    Ensure the desired ROOT file is active before writing.
    """

    output_file.cd()
    hist.Write()


# ============================================================
# CSV output
# ============================================================

def write_csv_summary(
    csv_path,
    pair_label,
    h_corr,
    h_stat,
    h_scale,
    h_pdf,
    h_shower,
    h_model,
    h_total,
):
    """
    Write the eta correction factor and all uncertainty
    components to a CSV file.

    The column format deliberately matches the original
    build_corrections.py CSV output.
    """

    with open(
        csv_path,
        "w",
        newline="",
    ) as csvfile:

        writer = csv.writer(
            csvfile
        )

        writer.writerow([
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
        ])

        for ibin in range(
            1,
            h_corr.GetNbinsX() + 1,
        ):

            low_edge = (
                h_corr
                .GetXaxis()
                .GetBinLowEdge(ibin)
            )

            up_edge = (
                h_corr
                .GetXaxis()
                .GetBinUpEdge(ibin)
            )

            writer.writerow([

                pair_label,

                OBSERVABLE,

                ibin,

                low_edge,

                up_edge,

                h_corr.GetBinContent(
                    ibin
                ),

                h_stat.GetBinContent(
                    ibin
                ),

                h_scale.GetBinContent(
                    ibin
                ),

                h_pdf.GetBinContent(
                    ibin
                ),

                h_shower.GetBinContent(
                    ibin
                ),

                h_model.GetBinContent(
                    ibin
                ),

                h_total.GetBinContent(
                    ibin
                ),
            ])


# ============================================================
# Main analysis
# ============================================================

def main():

    print()
    print("==============================================")
    print(" Lepton pseudorapidity correction factors")
    print("==============================================")
    print()

    print("Input directory:")
    print(f"  {INPUT_DIR}")

    print()
    print("ROOT output directory:")
    print(f"  {OUTPUT_DIR}")

    print()
    print("CSV output directory:")
    print(f"  {CSV_DIR}")

    print()

    # --------------------------------------------------------
    # Find all input files
    # --------------------------------------------------------

    files_by_sample = (
        scan_input_files()
    )

    # Store nominal correction factors so Pythia and Herwig
    # can be compared after all four pairs have been processed.
    nominal_ratios = {}

    # --------------------------------------------------------
    # Loop over generator / charge pairs
    # --------------------------------------------------------

    for (
        pair_label,
        (
            particle_sample,
            parton_sample,
        ),
    ) in SAMPLE_PAIRS.items():

        print()
        print("----------------------------------------------")
        print(f"Processing pair: {pair_label}")
        print("----------------------------------------------")

        print(
            f"  Particle sample: "
            f"{particle_sample}"
        )

        print(
            f"  Parton sample:   "
            f"{parton_sample}"
        )

        # ----------------------------------------------------
        # Check samples exist
        # ----------------------------------------------------

        if (
            particle_sample
            not in files_by_sample
        ):

            print(
                "  Missing particle-level files"
            )

            continue

        if (
            parton_sample
            not in files_by_sample
        ):

            print(
                "  Missing parton-level files"
            )

            continue

        particle_files = (
            files_by_sample[
                particle_sample
            ]
        )

        parton_files = (
            files_by_sample[
                parton_sample
            ]
        )

        # ----------------------------------------------------
        # Find weights common to both levels
        # ----------------------------------------------------

        common_weights = sorted(
            set(
                particle_files.keys()
            )
            &
            set(
                parton_files.keys()
            )
        )

        common_weights = [
            weight_index
            for weight_index
            in common_weights
            if (
                weight_index
                not in EXCLUDED_WEIGHTS
            )
        ]

        print(
            f"  Common usable weights: "
            f"{len(common_weights)}"
        )

        # Nominal correction requires weight 0.
        if 0 not in common_weights:

            print(
                "  No common nominal weight 0 "
                "found - skipping"
            )

            continue

        # ----------------------------------------------------
        # PDF weights
        # ----------------------------------------------------

        pdf_weights = []

        for weight_index in common_weights:

            weight_name = (
                particle_files[
                    weight_index
                ]["weight_name"]
            )

            if (
                is_pdf_weight_name(
                    weight_name
                )
                and
                weight_index
                not in SCALE_WEIGHTS
                and
                weight_index
                not in MODEL_WEIGHTS
                and
                weight_index
                not in SHOWER_WEIGHTS
                and
                weight_index != 281
            ):

                pdf_weights.append(
                    weight_index
                )

        # ----------------------------------------------------
        # Other systematic weight lists
        # ----------------------------------------------------

        scale_weights_common = [
            weight_index
            for weight_index
            in SCALE_WEIGHTS
            if (
                weight_index
                in common_weights
            )
        ]

        model_weights_common = [
            weight_index
            for weight_index
            in MODEL_WEIGHTS
            if (
                weight_index
                in common_weights
            )
        ]

        shower_weights_common = [
            weight_index
            for weight_index
            in SHOWER_WEIGHTS
            if (
                weight_index
                in common_weights
            )
        ]

        print(
            f"  Scale weights found:  "
            f"{scale_weights_common}"
        )

        print(
            f"  PDF weights found:    "
            f"{len(pdf_weights)}"
        )

        print(
            f"  Shower weights found: "
            f"{shower_weights_common}"
        )

        print(
            f"  Model weights found:  "
            f"{model_weights_common}"
        )

        # ----------------------------------------------------
        # Open ROOT files
        # ----------------------------------------------------

        particle_open_files = (
            open_root_files_for_sample(
                particle_files
            )
        )

        parton_open_files = (
            open_root_files_for_sample(
                parton_files
            )
        )

        # ----------------------------------------------------
        # ROOT output
        # ----------------------------------------------------

        output_root_path = (
            OUTPUT_DIR
            / f"corrections_{pair_label}.root"
        )

        output_file = ROOT.TFile(
            str(output_root_path),
            "RECREATE",
        )

        output_file.cd()

        # Basic metadata
        ROOT.TNamed(
            "PairLabel",
            pair_label,
        ).Write()

        ROOT.TNamed(
            "ParticleSample",
            particle_sample,
        ).Write()

        ROOT.TNamed(
            "PartonSample",
            parton_sample,
        ).Write()

        ROOT.TNamed(
            "Observable",
            OBSERVABLE,
        ).Write()

        # ====================================================
        # Nominal correction factor
        # ====================================================

        print(
            f"    Observable: "
            f"{OBSERVABLE}"
        )

        h_particle_nom = (
            get_etalepton_histogram(
                particle_open_files[0],
                (
                    f"{OBSERVABLE}_"
                    f"particle_nominal"
                ),
            )
        )

        h_parton_nom = (
            get_etalepton_histogram(
                parton_open_files[0],
                (
                    f"{OBSERVABLE}_"
                    f"parton_nominal"
                ),
            )
        )

        h_corr_nom = (
            make_ratio_histogram(
                h_particle_nom,
                h_parton_nom,
                (
                    f"{OBSERVABLE}_"
                    f"corr_nominal"
                ),
            )
        )

        write_hist_to_file(
            output_file,
            h_corr_nom,
        )

        # Keep a detached copy for later generator comparison.
        nominal_copy = (
            h_corr_nom.Clone(
                (
                    f"{pair_label}_"
                    f"{OBSERVABLE}_nominal"
                )
            )
        )

        nominal_copy.SetDirectory(0)

        nominal_ratios[
            pair_label
        ] = nominal_copy

        # ====================================================
        # Statistical uncertainty
        # ====================================================

        h_unc_stat = (
            compute_stat_uncertainty_from_ratio(
                h_corr_nom,
                f"{OBSERVABLE}_unc_stat",
            )
        )

        write_hist_to_file(
            output_file,
            h_unc_stat,
        )

        # ====================================================
        # Build systematic correction variations
        # ====================================================

        varied_scale = (
            build_variation_histograms(
                scale_weights_common,
                particle_open_files,
                parton_open_files,
                "scale",
            )
        )

        varied_pdf = (
            build_variation_histograms(
                pdf_weights,
                particle_open_files,
                parton_open_files,
                "pdf",
            )
        )

        varied_shower = (
            build_variation_histograms(
                shower_weights_common,
                particle_open_files,
                parton_open_files,
                "shower",
            )
        )

        varied_model = (
            build_variation_histograms(
                model_weights_common,
                particle_open_files,
                parton_open_files,
                "model",
            )
        )

        # ====================================================
        # Scale uncertainty
        # ====================================================

        h_unc_scale = (
            compute_envelope_uncertainty(
                h_corr_nom,
                varied_scale,
                f"{OBSERVABLE}_unc_scale",
            )
        )

        write_hist_to_file(
            output_file,
            h_unc_scale,
        )

        # ====================================================
        # PDF uncertainty
        # ====================================================

        h_unc_pdf = (
            compute_rms_uncertainty(
                h_corr_nom,
                varied_pdf,
                f"{OBSERVABLE}_unc_pdf",
            )
        )

        write_hist_to_file(
            output_file,
            h_unc_pdf,
        )

        # ====================================================
        # Shower uncertainty
        # ====================================================

        h_unc_shower = (
            compute_envelope_uncertainty(
                h_corr_nom,
                varied_shower,
                (
                    f"{OBSERVABLE}_"
                    f"unc_shower"
                ),
            )
        )

        write_hist_to_file(
            output_file,
            h_unc_shower,
        )

        # ====================================================
        # Model uncertainty
        # ====================================================

        h_unc_model = (
            compute_envelope_uncertainty(
                h_corr_nom,
                varied_model,
                (
                    f"{OBSERVABLE}_"
                    f"unc_model"
                ),
            )
        )

        write_hist_to_file(
            output_file,
            h_unc_model,
        )

        # ====================================================
        # Total uncertainty
        # ====================================================

        h_unc_total = (
            compute_total_uncertainty(
                h_corr_nom,
                [
                    h_unc_stat,
                    h_unc_scale,
                    h_unc_pdf,
                    h_unc_shower,
                    h_unc_model,
                ],
                (
                    f"{OBSERVABLE}_"
                    f"unc_total"
                ),
            )
        )

        write_hist_to_file(
            output_file,
            h_unc_total,
        )

        # ====================================================
        # CSV output
        # ====================================================

        csv_path = (
            CSV_DIR
            / (
                f"{pair_label}_"
                f"{OBSERVABLE}.csv"
            )
        )

        write_csv_summary(
            csv_path=csv_path,
            pair_label=pair_label,
            h_corr=h_corr_nom,
            h_stat=h_unc_stat,
            h_scale=h_unc_scale,
            h_pdf=h_unc_pdf,
            h_shower=h_unc_shower,
            h_model=h_unc_model,
            h_total=h_unc_total,
        )

        # ----------------------------------------------------
        # Finish pair
        # ----------------------------------------------------

        output_file.Write()
        output_file.Close()

        close_root_files(
            particle_open_files
        )

        close_root_files(
            parton_open_files
        )


    # ========================================================
    # Generator differences
    # ========================================================

    generator_pairs = [

        (
            "plus",
            "Pythia_plus",
            "Herwig_plus",
        ),

        (
            "minus",
            "Pythia_minus",
            "Herwig_minus",
        ),
    ]

    for (
        charge_label,
        pythia_key,
        herwig_key,
    ) in generator_pairs:

        output_root_path = (
            OUTPUT_DIR
            / (
                "generator_difference_"
                f"{charge_label}.root"
            )
        )

        output_file = ROOT.TFile(
            str(output_root_path),
            "RECREATE",
        )

        output_file.cd()

        if (
            pythia_key
            in nominal_ratios
            and
            herwig_key
            in nominal_ratios
        ):

            h_unc_generator = (
                compute_generator_difference(
                    nominal_ratios[
                        pythia_key
                    ],
                    nominal_ratios[
                        herwig_key
                    ],
                    (
                        f"{OBSERVABLE}_"
                        f"unc_generator"
                    ),
                )
            )

            write_hist_to_file(
                output_file,
                h_unc_generator,
            )

        output_file.Write()
        output_file.Close()


    # ========================================================
    # Finished
    # ========================================================

    print()
    print("==============================================")
    print(" Finished")
    print("==============================================")
    print()

    print(
        "Lepton pseudorapidity correction factors, "
        "uncertainties, ROOT files and CSV summaries "
        "have been created."
    )

    print()

    print("ROOT files:")
    print(f"  {OUTPUT_DIR}")

    print()

    print("CSV files:")
    print(f"  {CSV_DIR}")

    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()