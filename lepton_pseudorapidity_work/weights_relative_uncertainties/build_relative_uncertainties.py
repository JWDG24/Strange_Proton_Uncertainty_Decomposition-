#!/usr/bin/env python3

"""
build_relative_uncertainties.py

Supervisor part (a):
extract scale, PDF, shower and model uncertainties from the OLD
weighted samples relative to the nominal weight-0 correction.

For each eta bin:

    C0 = C(weight 0)
    Ck = C(weight k)

    delta_k = (Ck - C0) / C0

Then:
    scale  = max |delta_k|
    shower = max |delta_k|
    model  = max |delta_k|
    PDF    = RMS(delta_k)

This script does NOT use the new high-statistics weight-0 samples.

Expected location:

uncertainty_decomposition/
└── lepton_pseudorapidity_work/
    ├── MTWcut_make_histograms_etalepton_outputs/
    └── weights_relative_uncertainties/
        └── build_relative_uncertainties.py
"""

import csv
import math
import re
from pathlib import Path

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
LEPTON_WORK_DIR = SCRIPT_DIR.parent

INPUT_DIR = (
    LEPTON_WORK_DIR
    / "MTWcut_make_histograms_etalepton_outputs"
)

OUTPUT_DIR = SCRIPT_DIR / "outputs"
CSV_DIR = OUTPUT_DIR / "csv"
ROOT_DIR = OUTPUT_DIR / "root"

CSV_DIR.mkdir(parents=True, exist_ok=True)
ROOT_DIR.mkdir(parents=True, exist_ok=True)

OBSERVABLE = "etalepton"
NOMINAL_WEIGHT = 0

# ------------------------------------------------------------
# Sample pairs
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Weight groups
# ------------------------------------------------------------

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

MODEL_WEIGHTS = [
    292,
    293,
    314,
    315,
]

SHOWER_WEIGHTS = list(range(294, 314)) + [316, 317]

EXCLUDED_WEIGHTS = {318}
EXCLUDED_PDF_WEIGHTS = {281}

# ------------------------------------------------------------
# Input discovery
# ------------------------------------------------------------

def parse_filename(filename):
    pattern = (
        r"output_w(\d+)_(.+)_"
        r"(WCH7minus|WCH7plus|"
        r"WCHPartonminus|WCHPartonplus|"
        r"WCPy8minus|WCPy8plus|"
        r"WCPyPartonminus|WCPyPartonplus)"
        r"\.root$"
    )

    match = re.match(pattern, filename)

    if not match:
        return None

    return {
        "weight_index": int(match.group(1)),
        "weight_name": match.group(2),
        "sample": match.group(3),
        "path": INPUT_DIR / filename,
    }


def scan_inputs():
    if not INPUT_DIR.is_dir():
        raise RuntimeError(
            f"Input directory does not exist:\n  {INPUT_DIR}"
        )

    files_by_sample = {}

    for path in INPUT_DIR.iterdir():
        if not path.is_file() or path.suffix != ".root":
            continue

        info = parse_filename(path.name)

        if info is None:
            continue

        files_by_sample.setdefault(info["sample"], {})
        files_by_sample[info["sample"]][info["weight_index"]] = info

    return files_by_sample


# ------------------------------------------------------------
# ROOT helpers
# ------------------------------------------------------------

def load_hist(file_info, clone_name):
    root_file = ROOT.TFile.Open(
        str(file_info["path"]),
        "READ",
    )

    if not root_file or root_file.IsZombie():
        raise RuntimeError(
            f"Could not open:\n  {file_info['path']}"
        )

    hist = root_file.Get(OBSERVABLE)

    if not hist:
        root_file.Close()
        raise RuntimeError(
            f"Histogram '{OBSERVABLE}' missing in:\n"
            f"  {file_info['path']}"
        )

    clone = hist.Clone(clone_name)
    clone.SetDirectory(0)

    root_file.Close()

    return clone


def make_correction(
    particle_files,
    parton_files,
    weight_index,
    pair_label,
):
    h_particle = load_hist(
        particle_files[weight_index],
        f"{pair_label}_particle_w{weight_index}",
    )

    h_parton = load_hist(
        parton_files[weight_index],
        f"{pair_label}_parton_w{weight_index}",
    )

    correction = h_parton.Clone(
        f"{pair_label}_correction_w{weight_index}"
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


def empty_like(reference, name):
    out = reference.Clone(name)
    out.Reset()
    out.SetDirectory(0)
    return out


# ------------------------------------------------------------
# Weight classification
# ------------------------------------------------------------

def is_pdf_weight_name(name):
    return name.startswith(
        "MUR1.0_MUF1.0_PDF"
    )


def classify_weight(weight_index, weight_name):
    if weight_index in SCALE_WEIGHTS:
        return "scale"

    if weight_index in SHOWER_WEIGHTS:
        return "shower"

    if weight_index in MODEL_WEIGHTS:
        return "model"

    if (
        is_pdf_weight_name(weight_name)
        and weight_index not in EXCLUDED_PDF_WEIGHTS
        and weight_index not in SCALE_WEIGHTS
        and weight_index not in SHOWER_WEIGHTS
        and weight_index not in MODEL_WEIGHTS
    ):
        return "pdf"

    return None


# ------------------------------------------------------------
# Relative variations
# ------------------------------------------------------------

def make_relative_shift(nominal, varied, name):
    """
    delta = (C_varied - C_nominal) / C_nominal
    """

    out = empty_like(nominal, name)

    for ibin in range(
        1,
        nominal.GetNbinsX() + 1,
    ):
        c0 = float(
            nominal.GetBinContent(ibin)
        )

        ck = float(
            varied.GetBinContent(ibin)
        )

        if c0 == 0.0:
            delta = 0.0
        else:
            delta = (ck - c0) / c0

        out.SetBinContent(
            ibin,
            delta,
        )

        # These are correlated weight variations, not independent
        # statistical measurements.
        out.SetBinError(
            ibin,
            0.0,
        )

    return out


def envelope(nominal, variations, name):
    """
    max_k |delta_k|
    """

    out = empty_like(nominal, name)

    for ibin in range(
        1,
        nominal.GetNbinsX() + 1,
    ):
        maximum = 0.0

        for hist in variations:
            maximum = max(
                maximum,
                abs(
                    hist.GetBinContent(ibin)
                ),
            )

        out.SetBinContent(
            ibin,
            maximum,
        )

        out.SetBinError(
            ibin,
            0.0,
        )

    return out


def rms(nominal, variations, name):
    """
    sqrt( mean(delta_k^2) )
    """

    out = empty_like(nominal, name)

    if not variations:
        return out

    for ibin in range(
        1,
        nominal.GetNbinsX() + 1,
    ):
        values = [
            float(
                hist.GetBinContent(ibin)
            )
            for hist in variations
        ]

        value = math.sqrt(
            sum(x * x for x in values)
            / len(values)
        )

        out.SetBinContent(
            ibin,
            value,
        )

        out.SetBinError(
            ibin,
            0.0,
        )

    return out


def quadrature(nominal, hists, name):
    out = empty_like(nominal, name)

    for ibin in range(
        1,
        nominal.GetNbinsX() + 1,
    ):
        total = math.sqrt(
            sum(
                float(
                    hist.GetBinContent(ibin)
                ) ** 2
                for hist in hists
            )
        )

        out.SetBinContent(
            ibin,
            total,
        )

        out.SetBinError(
            ibin,
            0.0,
        )

    return out


# ------------------------------------------------------------
# CSV output
# ------------------------------------------------------------

def write_summary_csv(
    pair_label,
    nominal,
    h_scale,
    h_pdf,
    h_shower,
    h_model,
    h_total,
):
    path = (
        CSV_DIR
        / f"{pair_label}_relative_uncertainties.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "pair_label",
            "bin",
            "bin_low_edge",
            "bin_up_edge",
            "nominal_correction_weight0",
            "relative_scale_unc",
            "relative_pdf_unc",
            "relative_shower_unc",
            "relative_model_unc",
            "relative_total_systematic_unc",
            "scale_unc_percent",
            "pdf_unc_percent",
            "shower_unc_percent",
            "model_unc_percent",
            "total_systematic_unc_percent",
        ])

        for ibin in range(
            1,
            nominal.GetNbinsX() + 1,
        ):
            scale = float(
                h_scale.GetBinContent(ibin)
            )
            pdf = float(
                h_pdf.GetBinContent(ibin)
            )
            shower = float(
                h_shower.GetBinContent(ibin)
            )
            model = float(
                h_model.GetBinContent(ibin)
            )
            total = float(
                h_total.GetBinContent(ibin)
            )

            writer.writerow([
                pair_label,
                ibin,
                nominal.GetXaxis().GetBinLowEdge(ibin),
                nominal.GetXaxis().GetBinUpEdge(ibin),
                nominal.GetBinContent(ibin),
                scale,
                pdf,
                shower,
                model,
                total,
                100.0 * scale,
                100.0 * pdf,
                100.0 * shower,
                100.0 * model,
                100.0 * total,
            ])

    return path


def write_per_weight_csv(
    pair_label,
    nominal,
    records,
):
    path = (
        CSV_DIR
        / f"{pair_label}_per_weight_relative_shifts.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "pair_label",
            "weight_index",
            "weight_name",
            "category",
            "bin",
            "bin_low_edge",
            "bin_up_edge",
            "C_nominal_weight0",
            "C_weight",
            "C_weight_over_C_nominal",
            "signed_relative_shift",
            "absolute_relative_shift",
            "signed_relative_shift_percent",
        ])

        for record in records:
            varied = record["correction"]
            relative = record["relative"]

            for ibin in range(
                1,
                nominal.GetNbinsX() + 1,
            ):
                c0 = float(
                    nominal.GetBinContent(ibin)
                )
                ck = float(
                    varied.GetBinContent(ibin)
                )

                delta = float(
                    relative.GetBinContent(ibin)
                )

                ratio = (
                    ck / c0
                    if c0 != 0.0
                    else 0.0
                )

                writer.writerow([
                    pair_label,
                    record["weight_index"],
                    record["weight_name"],
                    record["category"],
                    ibin,
                    nominal.GetXaxis().GetBinLowEdge(ibin),
                    nominal.GetXaxis().GetBinUpEdge(ibin),
                    c0,
                    ck,
                    ratio,
                    delta,
                    abs(delta),
                    100.0 * delta,
                ])

    return path


# ------------------------------------------------------------
# ROOT output
# ------------------------------------------------------------

def write_root_output(
    pair_label,
    nominal,
    records,
    h_scale,
    h_pdf,
    h_shower,
    h_model,
    h_total,
):
    path = (
        ROOT_DIR
        / f"{pair_label}_relative_uncertainties.root"
    )

    output = ROOT.TFile(
        str(path),
        "RECREATE",
    )

    ROOT.TNamed(
        "PairLabel",
        pair_label,
    ).Write()

    ROOT.TNamed(
        "RelativeShiftDefinition",
        "(C_weight - C_weight0) / C_weight0",
    ).Write()

    nominal.Write(
        "correction_nominal_weight0"
    )

    variation_dir = output.mkdir(
        "relative_weight_variations"
    )

    variation_dir.cd()

    for record in records:
        record["relative"].Write(
            (
                f"{record['category']}_"
                f"w{record['weight_index']}"
            )
        )

    output.cd()

    h_scale.Write(
        "relative_unc_scale"
    )

    h_pdf.Write(
        "relative_unc_pdf"
    )

    h_shower.Write(
        "relative_unc_shower"
    )

    h_model.Write(
        "relative_unc_model"
    )

    h_total.Write(
        "relative_unc_total_systematic"
    )

    output.Write()
    output.Close()

    return path


# ------------------------------------------------------------
# Pair processing
# ------------------------------------------------------------

def process_pair(
    pair_label,
    particle_sample,
    parton_sample,
    files_by_sample,
):
    print()
    print(
        "============================================================"
    )
    print(
        f"Processing {pair_label}"
    )
    print(
        "============================================================"
    )

    if particle_sample not in files_by_sample:
        raise RuntimeError(
            f"Missing sample: {particle_sample}"
        )

    if parton_sample not in files_by_sample:
        raise RuntimeError(
            f"Missing sample: {parton_sample}"
        )

    particle_files = files_by_sample[
        particle_sample
    ]

    parton_files = files_by_sample[
        parton_sample
    ]

    common_weights = sorted(
        (
            set(particle_files)
            & set(parton_files)
        )
        - EXCLUDED_WEIGHTS
    )

    if NOMINAL_WEIGHT not in common_weights:
        raise RuntimeError(
            f"No common weight 0 for {pair_label}"
        )

    nominal = make_correction(
        particle_files,
        parton_files,
        NOMINAL_WEIGHT,
        pair_label,
    )

    nominal.SetName(
        f"{pair_label}_nominal_weight0"
    )

    records = []

    grouped = {
        "scale": [],
        "pdf": [],
        "shower": [],
        "model": [],
    }

    for weight_index in common_weights:
        if weight_index == NOMINAL_WEIGHT:
            continue

        weight_name = (
            particle_files[
                weight_index
            ][
                "weight_name"
            ]
        )

        category = classify_weight(
            weight_index,
            weight_name,
        )

        if category is None:
            continue

        varied = make_correction(
            particle_files,
            parton_files,
            weight_index,
            pair_label,
        )

        relative = make_relative_shift(
            nominal,
            varied,
            (
                f"{pair_label}_"
                f"{category}_w{weight_index}"
            ),
        )

        records.append({
            "weight_index": weight_index,
            "weight_name": weight_name,
            "category": category,
            "correction": varied,
            "relative": relative,
        })

        grouped[
            category
        ].append(
            relative
        )

    print(
        f"  Scale variations:  {len(grouped['scale'])}"
    )
    print(
        f"  PDF variations:    {len(grouped['pdf'])}"
    )
    print(
        f"  Shower variations: {len(grouped['shower'])}"
    )
    print(
        f"  Model variations:  {len(grouped['model'])}"
    )

    h_scale = envelope(
        nominal,
        grouped["scale"],
        f"{pair_label}_relative_unc_scale",
    )

    h_pdf = rms(
        nominal,
        grouped["pdf"],
        f"{pair_label}_relative_unc_pdf",
    )

    h_shower = envelope(
        nominal,
        grouped["shower"],
        f"{pair_label}_relative_unc_shower",
    )

    h_model = envelope(
        nominal,
        grouped["model"],
        f"{pair_label}_relative_unc_model",
    )

    h_total = quadrature(
        nominal,
        [
            h_scale,
            h_pdf,
            h_shower,
            h_model,
        ],
        f"{pair_label}_relative_unc_total_systematic",
    )

    summary_csv = write_summary_csv(
        pair_label,
        nominal,
        h_scale,
        h_pdf,
        h_shower,
        h_model,
        h_total,
    )

    per_weight_csv = write_per_weight_csv(
        pair_label,
        nominal,
        records,
    )

    root_path = write_root_output(
        pair_label,
        nominal,
        records,
        h_scale,
        h_pdf,
        h_shower,
        h_model,
        h_total,
    )

    print(
        f"  Summary CSV:    {summary_csv}"
    )
    print(
        f"  Per-weight CSV: {per_weight_csv}"
    )
    print(
        f"  ROOT output:    {root_path}"
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print()
    print(
        "Relative systematic uncertainty extraction"
    )
    print(
        "Nominal reference = weight 0 from the same old weighted samples"
    )
    print()
    print(
        f"Input:\n  {INPUT_DIR}"
    )
    print(
        f"Output:\n  {OUTPUT_DIR}"
    )

    files_by_sample = scan_inputs()

    for (
        pair_label,
        (
            particle_sample,
            parton_sample,
        ),
    ) in SAMPLE_PAIRS.items():

        process_pair(
            pair_label,
            particle_sample,
            parton_sample,
            files_by_sample,
        )

    print()
    print(
        "Finished."
    )
    print()
    print(
        "Main outputs:"
    )
    print(
        f"  {CSV_DIR}"
    )
    print()
    print(
        "The *_relative_uncertainties.csv files contain the"
    )
    print(
        "final fractional and percentage scale, PDF, shower,"
    )
    print(
        "model and total systematic uncertainties."
    )


if __name__ == "__main__":
    main()
