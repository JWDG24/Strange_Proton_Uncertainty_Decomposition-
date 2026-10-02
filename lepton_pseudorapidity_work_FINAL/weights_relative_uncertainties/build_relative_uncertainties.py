#!/usr/bin/env python3

"""
build_relative_uncertainties.py

REPAIRED FINAL version.

Purpose
-------
Extract relative systematic uncertainties from the repaired OLD weighted
W+c Monte Carlo samples.

For every eta bin:

    C0 = C(weight 0)
    Ck = C(weight k)

    delta_k = (Ck - C0) / C0

The final relative systematic magnitudes are then:

    scale  = max_k |delta_k|
    shower = max_k |delta_k|
    model  = max_k |delta_k|
    PDF    = sqrt(mean_k(delta_k^2))

and

    total systematic
        = sqrt(
            scale^2
            + PDF^2
            + shower^2
            + model^2
        )

Important statistical point
---------------------------
Weight k and weight 0 are evaluated on the SAME weighted event sample.
Therefore the systematic response is formed as a same-sample relative
shift.  No independent statistical uncertainty is added to delta_k.

This script does NOT use the exploratory high-stat/event-matching
covariance analysis.

Required repaired upstream selection
------------------------------------
Every histogram ROOT file is checked for metadata proving that it was
made with:

    MET > 25 GeV
    lepton pT > 20 GeV
    |eta_lepton| < 2.5
    mT(W) > 40 GeV
    jet pT > 25 GeV
    |eta_jet| < 2.5
    exactly one fiducial charm-identified jet
    charm ID: jet_charge != 0
    OS-SS:
        opposite-sign W/charm -> +1
        same-sign W/charm     -> -1

The script refuses stale pre-repair histogram files.

Additional consistency check
----------------------------
The weight-0 correction reconstructed here is compared bin-by-bin with
the repaired output from:

    MTWcut_build_corrections_etalepton.py

This ensures the relative-systematic stage is using exactly the same
nominal repaired histograms and correction definition as the previous
main-chain stage.

Weight definitions
------------------
Scale weights:
    1, 112, 215, 226, 237, 248, 259, 270

Model weights:
    292, 293, 314, 315

Shower weights:
    294-313, 316, 317

Excluded:
    weight 318
    PDF weight 281

Only weights present at BOTH particle and parton level are used.
Missing weights are never fabricated.

Recommended location
--------------------
uncertainty_decomposition/
└── lepton_pseudorapidity_work_FINAL/
    ├── MTWcut_make_histograms_etalepton_outputs/
    ├── MTWcut_build_corrections_etalepton_csv/
    └── weights_relative_uncertainties/
        └── build_relative_uncertainties.py

Run from the project root
-------------------------
python3 lepton_pseudorapidity_work_FINAL/weights_relative_uncertainties/build_relative_uncertainties.py
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

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
LEPTON_WORK_DIR = SCRIPT_DIR.parent

INPUT_DIR = (
    LEPTON_WORK_DIR
    / "MTWcut_make_histograms_etalepton_outputs"
)

REFERENCE_CORRECTION_CSV_DIR = (
    LEPTON_WORK_DIR
    / "MTWcut_build_corrections_etalepton_csv"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "outputs"
)

CSV_DIR = (
    OUTPUT_DIR
    / "csv"
)

ROOT_DIR = (
    OUTPUT_DIR
    / "root"
)

CSV_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ROOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Analysis definition
# ============================================================

OBSERVABLE = "etalepton"
NOMINAL_WEIGHT = 0

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
# Weight definitions
# ============================================================

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

SHOWER_WEIGHTS = (
    list(
        range(
            294,
            314,
        )
    )
    + [
        316,
        317,
    ]
)

EXCLUDED_WEIGHTS = {
    318,
}

EXCLUDED_PDF_WEIGHTS = {
    281,
}


# ============================================================
# Expected upstream metadata
# ============================================================

EXPECTED_MTW_GEV = 40.0
EXPECTED_JET_PT_GEV = 25.0
EXPECTED_JET_ABS_ETA = 2.5
EXPECTED_REQUIRED_CHARM_JETS = 1

FLOAT_METADATA_TOLERANCE = 1.0e-9
REFERENCE_CORRECTION_TOLERANCE = 5.0e-7


# ============================================================
# Input discovery
# ============================================================

def parse_filename(
    filename: str,
):
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
        "weight_name": (
            match.group(2)
        ),
        "sample": (
            match.group(3)
        ),
        "path": (
            INPUT_DIR
            / filename
        ),
    }


def scan_inputs():
    if not INPUT_DIR.is_dir():
        raise RuntimeError(
            "Repaired histogram directory does not exist:\n"
            f"  {INPUT_DIR}"
        )

    files_by_sample = {}

    ignored = []

    for path in sorted(
        INPUT_DIR.iterdir()
    ):
        if (
            not path.is_file()
            or path.suffix != ".root"
        ):
            continue

        info = parse_filename(
            path.name
        )

        if info is None:
            ignored.append(
                path.name
            )
            continue

        sample = info[
            "sample"
        ]

        weight_index = info[
            "weight_index"
        ]

        table = files_by_sample.setdefault(
            sample,
            {},
        )

        if weight_index in table:
            raise RuntimeError(
                "Duplicate histogram file for sample/weight:\n"
                f"  sample: {sample}\n"
                f"  weight: {weight_index}\n"
                f"  first:  {table[weight_index]['path']}\n"
                f"  second: {path}"
            )

        table[
            weight_index
        ] = info

    if ignored:
        print()
        print(
            "Warning: ignored ROOT files that do not match the "
            "expected histogram filename format:"
        )

        for name in ignored:
            print(
                f"  {name}"
            )

    if not files_by_sample:
        raise RuntimeError(
            "No compatible histogram ROOT files were found in:\n"
            f"  {INPUT_DIR}"
        )

    return files_by_sample


# ============================================================
# Metadata validation
# ============================================================

def get_named_title(
    root_file,
    key: str,
):
    obj = root_file.Get(
        key
    )

    if (
        not obj
        or not hasattr(
            obj,
            "GetTitle",
        )
    ):
        return None

    return str(
        obj.GetTitle()
    )


def require_named_title(
    root_file,
    key: str,
    path: Path,
):
    value = get_named_title(
        root_file,
        key,
    )

    if value is None:
        raise RuntimeError(
            "Input ROOT file is missing repaired-analysis metadata:\n"
            f"  file: {path}\n"
            f"  missing key: {key}\n\n"
            "Re-run the repaired MTWcut_make_histograms_etalepton.py "
            "before extracting relative uncertainties."
        )

    return value


def parse_metadata_float(
    text: str,
    key: str,
    path: Path,
):
    try:
        return float(
            text
        )

    except ValueError as exc:
        raise RuntimeError(
            f"Invalid numeric metadata '{key}' in:\n"
            f"  {path}\n"
            f"Value: {text!r}"
        ) from exc


def parse_metadata_int(
    text: str,
    key: str,
    path: Path,
):
    try:
        return int(
            text
        )

    except ValueError as exc:
        raise RuntimeError(
            f"Invalid integer metadata '{key}' in:\n"
            f"  {path}\n"
            f"Value: {text!r}"
        ) from exc


def validate_repaired_metadata(
    root_file,
    info,
):
    path = info[
        "path"
    ]

    if (
        require_named_title(
            root_file,
            "Observable",
            path,
        )
        != OBSERVABLE
    ):
        raise RuntimeError(
            f"Unexpected Observable metadata in:\n  {path}"
        )

    if (
        require_named_title(
            root_file,
            "SampleName",
            path,
        )
        != info[
            "sample"
        ]
    ):
        raise RuntimeError(
            f"SampleName metadata mismatch in:\n  {path}"
        )

    effective_weight = parse_metadata_int(
        require_named_title(
            root_file,
            "EffectiveWeightIndex",
            path,
        ),
        "EffectiveWeightIndex",
        path,
    )

    if (
        effective_weight
        != info[
            "weight_index"
        ]
    ):
        raise RuntimeError(
            f"EffectiveWeightIndex metadata mismatch in:\n  {path}"
        )

    if (
        require_named_title(
            root_file,
            "MTWCutApplied",
            path,
        )
        != "True"
    ):
        raise RuntimeError(
            f"mT(W) cut is not marked as applied in:\n  {path}"
        )

    mtw_cut = parse_metadata_float(
        require_named_title(
            root_file,
            "MTWCutGeV",
            path,
        ),
        "MTWCutGeV",
        path,
    )

    if not math.isclose(
        mtw_cut,
        EXPECTED_MTW_GEV,
        rel_tol=0.0,
        abs_tol=FLOAT_METADATA_TOLERANCE,
    ):
        raise RuntimeError(
            f"Unexpected mT(W) cut in:\n  {path}"
        )

    if (
        require_named_title(
            root_file,
            "JetSelectionApplied",
            path,
        )
        != "True"
    ):
        raise RuntimeError(
            f"Jet selection is not marked as applied in:\n  {path}"
        )

    jet_pt = parse_metadata_float(
        require_named_title(
            root_file,
            "JetPtMinGeV",
            path,
        ),
        "JetPtMinGeV",
        path,
    )

    if not math.isclose(
        jet_pt,
        EXPECTED_JET_PT_GEV,
        rel_tol=0.0,
        abs_tol=FLOAT_METADATA_TOLERANCE,
    ):
        raise RuntimeError(
            f"Unexpected jet pT threshold in:\n  {path}"
        )

    jet_abs_eta = parse_metadata_float(
        require_named_title(
            root_file,
            "JetAbsEtaMax",
            path,
        ),
        "JetAbsEtaMax",
        path,
    )

    if not math.isclose(
        jet_abs_eta,
        EXPECTED_JET_ABS_ETA,
        rel_tol=0.0,
        abs_tol=FLOAT_METADATA_TOLERANCE,
    ):
        raise RuntimeError(
            f"Unexpected jet |eta| threshold in:\n  {path}"
        )

    required_charm_jets = parse_metadata_int(
        require_named_title(
            root_file,
            "RequiredCharmJets",
            path,
        ),
        "RequiredCharmJets",
        path,
    )

    if (
        required_charm_jets
        != EXPECTED_REQUIRED_CHARM_JETS
    ):
        raise RuntimeError(
            f"Unexpected charm-jet multiplicity in:\n  {path}"
        )

    if (
        require_named_title(
            root_file,
            "CharmJetIDBranch",
            path,
        )
        != "jet_charge"
    ):
        raise RuntimeError(
            f"Unexpected charm-ID branch in:\n  {path}"
        )

    charm_definition = require_named_title(
        root_file,
        "CharmJetIDDefinition",
        path,
    )

    if (
        "jet_charge != 0"
        not in charm_definition
    ):
        raise RuntimeError(
            f"Unexpected charm-ID definition in:\n  {path}"
        )

    os_ss_definition = require_named_title(
        root_file,
        "OSSSWeightDefinition",
        path,
    )

    for phrase in (
        "+1 for opposite-sign",
        "-1 for same-sign",
    ):
        if phrase not in os_ss_definition:
            raise RuntimeError(
                f"Unexpected OS-SS definition in:\n  {path}"
            )

    event_selection = require_named_title(
        root_file,
        "EventSelection",
        path,
    )

    required_phrases = (
        "met_et > 25 GeV",
        "leptons_pt > 20 GeV",
        "abs(leptons_eta) < 2.5",
        "m_T^W > 40 GeV",
        "jet_pt > 25 GeV",
        "abs(jet_eta) < 2.5",
        "exactly one charm-identified fiducial jet",
        "jet_charge != 0",
    )

    missing = [
        phrase
        for phrase in required_phrases
        if phrase not in event_selection
    ]

    if missing:
        raise RuntimeError(
            "Input ROOT file does not document the complete repaired "
            "selection:\n"
            f"  {path}\n"
            f"Missing metadata text: {missing}"
        )


# ============================================================
# Histogram loading / validation
# ============================================================

def load_hist(
    file_info,
    clone_name: str,
):
    root_file = ROOT.TFile.Open(
        str(
            file_info[
                "path"
            ]
        ),
        "READ",
    )

    if (
        not root_file
        or root_file.IsZombie()
    ):
        raise RuntimeError(
            "Could not open ROOT file:\n"
            f"  {file_info['path']}"
        )

    try:
        validate_repaired_metadata(
            root_file,
            file_info,
        )

        hist = root_file.Get(
            OBSERVABLE
        )

        if not hist:
            raise RuntimeError(
                f"Histogram '{OBSERVABLE}' missing in:\n"
                f"  {file_info['path']}"
            )

        if not hist.InheritsFrom(
            "TH1"
        ):
            raise RuntimeError(
                f"Object '{OBSERVABLE}' is not a TH1 in:\n"
                f"  {file_info['path']}"
            )

        clone = hist.Clone(
            clone_name
        )

        clone.SetDirectory(
            0
        )

        return clone

    finally:
        root_file.Close()


def validate_binning(
    hist_a,
    hist_b,
    label_a: str,
    label_b: str,
):
    if (
        hist_a.GetNbinsX()
        != hist_b.GetNbinsX()
    ):
        raise RuntimeError(
            "Histogram bin-count mismatch:\n"
            f"  {label_a}\n"
            f"  {label_b}"
        )

    tolerance = 1.0e-12

    for ibin in range(
        1,
        hist_a.GetNbinsX() + 2,
    ):
        edge_a = (
            hist_a
            .GetXaxis()
            .GetBinLowEdge(
                ibin
            )
        )

        edge_b = (
            hist_b
            .GetXaxis()
            .GetBinLowEdge(
                ibin
            )
        )

        if not math.isclose(
            edge_a,
            edge_b,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise RuntimeError(
                "Histogram bin-edge mismatch:\n"
                f"  {label_a}\n"
                f"  {label_b}\n"
                f"  edge {ibin}: {edge_a} != {edge_b}"
            )


def ensure_nonzero_histogram_bins(
    hist,
    label: str,
):
    zero_bins = []

    for ibin in range(
        1,
        hist.GetNbinsX() + 1,
    ):
        if (
            float(
                hist.GetBinContent(
                    ibin
                )
            )
            == 0.0
        ):
            zero_bins.append(
                ibin
            )

    if zero_bins:
        raise RuntimeError(
            f"{label} has zero bin content in bins {zero_bins}; "
            "a relative correction/shift would be undefined."
        )


# ============================================================
# Correction construction
# ============================================================

def make_correction(
    particle_files,
    parton_files,
    weight_index: int,
    pair_label: str,
):
    h_particle = load_hist(
        particle_files[
            weight_index
        ],
        (
            f"{pair_label}_"
            f"particle_w{weight_index}"
        ),
    )

    h_parton = load_hist(
        parton_files[
            weight_index
        ],
        (
            f"{pair_label}_"
            f"parton_w{weight_index}"
        ),
    )

    validate_binning(
        h_particle,
        h_parton,
        (
            f"{pair_label} particle "
            f"weight {weight_index}"
        ),
        (
            f"{pair_label} parton "
            f"weight {weight_index}"
        ),
    )

    ensure_nonzero_histogram_bins(
        h_particle,
        (
            f"{pair_label} particle "
            f"weight {weight_index}"
        ),
    )

    correction = h_parton.Clone(
        (
            f"{pair_label}_"
            f"correction_w{weight_index}"
        )
    )

    correction.SetDirectory(
        0
    )

    correction.Divide(
        h_parton,
        h_particle,
        1.0,
        1.0,
        "",
    )

    return correction


def empty_like(
    reference,
    name: str,
):
    out = reference.Clone(
        name
    )

    out.Reset()
    out.SetDirectory(
        0
    )

    return out


# ============================================================
# Weight classification
# ============================================================

def is_pdf_weight_name(
    name: str,
):
    return name.startswith(
        "MUR1.0_MUF1.0_PDF"
    )


def classify_weight(
    weight_index: int,
    weight_name: str,
):
    if weight_index in SCALE_WEIGHTS:
        return "scale"

    if weight_index in SHOWER_WEIGHTS:
        return "shower"

    if weight_index in MODEL_WEIGHTS:
        return "model"

    if (
        is_pdf_weight_name(
            weight_name
        )
        and weight_index
        not in EXCLUDED_PDF_WEIGHTS
        and weight_index
        not in SCALE_WEIGHTS
        and weight_index
        not in SHOWER_WEIGHTS
        and weight_index
        not in MODEL_WEIGHTS
    ):
        return "pdf"

    return None


# ============================================================
# Relative variations
# ============================================================

def make_relative_shift(
    nominal,
    varied,
    name: str,
):
    """
    delta_i = (C_i^k - C_i^0) / C_i^0
    """

    validate_binning(
        nominal,
        varied,
        nominal.GetName(),
        varied.GetName(),
    )

    ensure_nonzero_histogram_bins(
        nominal,
        (
            "nominal correction used as "
            "relative-shift denominator"
        ),
    )

    out = empty_like(
        nominal,
        name,
    )

    for ibin in range(
        1,
        nominal.GetNbinsX() + 1,
    ):
        c0 = float(
            nominal.GetBinContent(
                ibin
            )
        )

        ck = float(
            varied.GetBinContent(
                ibin
            )
        )

        delta = (
            ck
            - c0
        ) / c0

        out.SetBinContent(
            ibin,
            delta,
        )

        # The nominal and varied weights come from the same MC sample.
        # This histogram stores the systematic response itself, not an
        # independent statistical measurement.
        out.SetBinError(
            ibin,
            0.0,
        )

    return out


def envelope(
    nominal,
    variations,
    name: str,
):
    """
    max_k |delta_k|
    """

    out = empty_like(
        nominal,
        name,
    )

    for ibin in range(
        1,
        nominal.GetNbinsX() + 1,
    ):
        maximum = 0.0

        for hist in variations:
            maximum = max(
                maximum,
                abs(
                    float(
                        hist.GetBinContent(
                            ibin
                        )
                    )
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


def rms(
    nominal,
    variations,
    name: str,
):
    """
    sqrt(mean(delta_k^2))
    """

    out = empty_like(
        nominal,
        name,
    )

    if not variations:
        return out

    for ibin in range(
        1,
        nominal.GetNbinsX() + 1,
    ):
        values = [
            float(
                hist.GetBinContent(
                    ibin
                )
            )
            for hist in variations
        ]

        value = math.sqrt(
            sum(
                x * x
                for x in values
            )
            / len(
                values
            )
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


def quadrature(
    nominal,
    histograms,
    name: str,
):
    out = empty_like(
        nominal,
        name,
    )

    for ibin in range(
        1,
        nominal.GetNbinsX() + 1,
    ):
        total = math.sqrt(
            sum(
                float(
                    hist.GetBinContent(
                        ibin
                    )
                ) ** 2
                for hist in histograms
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


# ============================================================
# Cross-check against repaired correction-builder output
# ============================================================

def read_reference_nominal_csv(
    pair_label: str,
):
    path = (
        REFERENCE_CORRECTION_CSV_DIR
        / f"{pair_label}_{OBSERVABLE}.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Repaired correction-builder CSV is missing:\n"
            f"  {path}\n\n"
            "Run MTWcut_build_corrections_etalepton.py before this "
            "relative-systematic stage."
        )

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )

        required = {
            "bin",
            "bin_low_edge",
            "bin_up_edge",
            "correction_factor",
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
                f"Reference CSV schema is incomplete:\n  {path}\n"
                f"Missing: {sorted(missing)}"
            )

        rows = list(
            reader
        )

    return (
        path,
        rows,
    )


def validate_nominal_against_reference(
    pair_label: str,
    nominal,
):
    path, rows = read_reference_nominal_csv(
        pair_label
    )

    if (
        len(
            rows
        )
        != nominal.GetNbinsX()
    ):
        raise RuntimeError(
            f"{pair_label}: repaired builder/reference has a "
            "different number of bins."
        )

    max_abs_delta = 0.0

    for ibin, row in enumerate(
        rows,
        start=1,
    ):
        if (
            int(
                row[
                    "bin"
                ]
            )
            != ibin
        ):
            raise RuntimeError(
                f"{pair_label}: bin index mismatch against:\n  {path}"
            )

        low = float(
            row[
                "bin_low_edge"
            ]
        )

        high = float(
            row[
                "bin_up_edge"
            ]
        )

        hist_low = (
            nominal
            .GetXaxis()
            .GetBinLowEdge(
                ibin
            )
        )

        hist_high = (
            nominal
            .GetXaxis()
            .GetBinUpEdge(
                ibin
            )
        )

        if (
            not math.isclose(
                low,
                hist_low,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                high,
                hist_high,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise RuntimeError(
                f"{pair_label}: eta binning differs from repaired "
                "correction-builder output."
            )

        reference = float(
            row[
                "correction_factor"
            ]
        )

        current = float(
            nominal.GetBinContent(
                ibin
            )
        )

        delta = abs(
            current
            - reference
        )

        max_abs_delta = max(
            max_abs_delta,
            delta,
        )

        if (
            delta
            > REFERENCE_CORRECTION_TOLERANCE
        ):
            raise RuntimeError(
                f"{pair_label}: nominal weight-0 correction does not "
                "reproduce the repaired correction-builder output.\n"
                f"  bin:       {ibin}\n"
                f"  here:      {current:.12g}\n"
                f"  reference: {reference:.12g}\n"
                f"  |delta|:   {delta:.6g}\n"
                f"  tolerance: {REFERENCE_CORRECTION_TOLERANCE:.6g}"
            )

    return (
        path,
        max_abs_delta,
    )


# ============================================================
# CSV outputs
# ============================================================

SUMMARY_COLUMNS = [
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
]

PER_WEIGHT_COLUMNS = [
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
]


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
    ) as csv_file:
        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            SUMMARY_COLUMNS
        )

        for ibin in range(
            1,
            nominal.GetNbinsX() + 1,
        ):
            scale = float(
                h_scale.GetBinContent(
                    ibin
                )
            )

            pdf = float(
                h_pdf.GetBinContent(
                    ibin
                )
            )

            shower = float(
                h_shower.GetBinContent(
                    ibin
                )
            )

            model = float(
                h_model.GetBinContent(
                    ibin
                )
            )

            total = float(
                h_total.GetBinContent(
                    ibin
                )
            )

            writer.writerow([
                pair_label,
                ibin,
                (
                    nominal
                    .GetXaxis()
                    .GetBinLowEdge(
                        ibin
                    )
                ),
                (
                    nominal
                    .GetXaxis()
                    .GetBinUpEdge(
                        ibin
                    )
                ),
                float(
                    nominal.GetBinContent(
                        ibin
                    )
                ),
                scale,
                pdf,
                shower,
                model,
                total,
                100.0
                * scale,
                100.0
                * pdf,
                100.0
                * shower,
                100.0
                * model,
                100.0
                * total,
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
    ) as csv_file:
        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            PER_WEIGHT_COLUMNS
        )

        for record in records:
            varied = record[
                "correction"
            ]

            relative = record[
                "relative"
            ]

            for ibin in range(
                1,
                nominal.GetNbinsX() + 1,
            ):
                c0 = float(
                    nominal.GetBinContent(
                        ibin
                    )
                )

                ck = float(
                    varied.GetBinContent(
                        ibin
                    )
                )

                delta = float(
                    relative.GetBinContent(
                        ibin
                    )
                )

                ratio = (
                    ck
                    / c0
                )

                writer.writerow([
                    pair_label,
                    record[
                        "weight_index"
                    ],
                    record[
                        "weight_name"
                    ],
                    record[
                        "category"
                    ],
                    ibin,
                    (
                        nominal
                        .GetXaxis()
                        .GetBinLowEdge(
                            ibin
                        )
                    ),
                    (
                        nominal
                        .GetXaxis()
                        .GetBinUpEdge(
                            ibin
                        )
                    ),
                    c0,
                    ck,
                    ratio,
                    delta,
                    abs(
                        delta
                    ),
                    100.0
                    * delta,
                ])

    return path


# ============================================================
# ROOT output
# ============================================================

def write_root_output(
    pair_label,
    nominal,
    records,
    h_scale,
    h_pdf,
    h_shower,
    h_model,
    h_total,
    reference_csv_path,
    reference_max_delta,
):
    path = (
        ROOT_DIR
        / f"{pair_label}_relative_uncertainties.root"
    )

    output = ROOT.TFile(
        str(
            path
        ),
        "RECREATE",
    )

    if (
        not output
        or output.IsZombie()
    ):
        raise RuntimeError(
            f"Could not create ROOT output:\n  {path}"
        )

    ROOT.TNamed(
        "PairLabel",
        pair_label,
    ).Write()

    ROOT.TNamed(
        "Observable",
        OBSERVABLE,
    ).Write()

    ROOT.TNamed(
        "RelativeShiftDefinition",
        "(C_weight - C_weight0) / C_weight0",
    ).Write()

    ROOT.TNamed(
        "ScaleDefinition",
        "max |delta_k|",
    ).Write()

    ROOT.TNamed(
        "PDFDefinition",
        "sqrt(mean(delta_k^2))",
    ).Write()

    ROOT.TNamed(
        "ShowerDefinition",
        "max |delta_k|",
    ).Write()

    ROOT.TNamed(
        "ModelDefinition",
        "max |delta_k|",
    ).Write()

    ROOT.TNamed(
        "TotalSystematicDefinition",
        "quadrature(scale, PDF, shower, model)",
    ).Write()

    ROOT.TNamed(
        "UsesHighStatNominal",
        "False",
    ).Write()

    ROOT.TNamed(
        "UpstreamSelectionValidated",
        "True",
    ).Write()

    ROOT.TNamed(
        "ReferenceNominalCSV",
        str(
            reference_csv_path
        ),
    ).Write()

    ROOT.TNamed(
        "ReferenceNominalMaxAbsDelta",
        f"{reference_max_delta:.12g}",
    ).Write()

    nominal.Write(
        "correction_nominal_weight0"
    )

    variation_dir = output.mkdir(
        "relative_weight_variations"
    )

    variation_dir.cd()

    for record in records:
        record[
            "relative"
        ].Write(
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


# ============================================================
# Pair processing
# ============================================================

def process_pair(
    pair_label,
    particle_sample,
    parton_sample,
    files_by_sample,
):
    print()
    print(
        "=" * 72
    )

    print(
        f"Processing {pair_label}"
    )

    print(
        "=" * 72
    )

    if particle_sample not in files_by_sample:
        raise RuntimeError(
            f"Missing particle sample: {particle_sample}"
        )

    if parton_sample not in files_by_sample:
        raise RuntimeError(
            f"Missing parton sample: {parton_sample}"
        )

    particle_files = files_by_sample[
        particle_sample
    ]

    parton_files = files_by_sample[
        parton_sample
    ]

    common_weights = sorted(
        (
            set(
                particle_files
            )
            & set(
                parton_files
            )
        )
        - EXCLUDED_WEIGHTS
    )

    if NOMINAL_WEIGHT not in common_weights:
        raise RuntimeError(
            f"No common nominal weight 0 for {pair_label}"
        )

    particle_only = sorted(
        set(
            particle_files
        )
        - set(
            parton_files
        )
    )

    parton_only = sorted(
        set(
            parton_files
        )
        - set(
            particle_files
        )
    )

    print(
        f"  particle sample: {particle_sample}"
    )

    print(
        f"  parton sample:   {parton_sample}"
    )

    print(
        f"  common usable weights: {len(common_weights)}"
    )

    if particle_only:
        print(
            f"  particle-only weights skipped: {particle_only}"
        )

    if parton_only:
        print(
            f"  parton-only weights skipped: {parton_only}"
        )

    # --------------------------------------------------------
    # Nominal correction from the SAME old weighted samples.
    # --------------------------------------------------------

    nominal = make_correction(
        particle_files,
        parton_files,
        NOMINAL_WEIGHT,
        pair_label,
    )

    nominal.SetName(
        f"{pair_label}_nominal_weight0"
    )

    (
        reference_csv_path,
        reference_max_delta,
    ) = validate_nominal_against_reference(
        pair_label,
        nominal,
    )

    print(
        "  repaired nominal cross-check:"
    )

    print(
        f"    reference: {reference_csv_path}"
    )

    print(
        f"    max |delta C0|: {reference_max_delta:.3e}"
    )

    # --------------------------------------------------------
    # Build weight responses.
    # --------------------------------------------------------

    records = []

    grouped = {
        "scale": [],
        "pdf": [],
        "shower": [],
        "model": [],
    }

    category_counts = {
        "scale": 0,
        "pdf": 0,
        "shower": 0,
        "model": 0,
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
                f"{category}_"
                f"w{weight_index}"
            ),
        )

        record = {
            "weight_index": (
                weight_index
            ),
            "weight_name": (
                weight_name
            ),
            "category": (
                category
            ),
            "correction": (
                varied
            ),
            "relative": (
                relative
            ),
        }

        records.append(
            record
        )

        grouped[
            category
        ].append(
            relative
        )

        category_counts[
            category
        ] += 1

    print(
        f"  scale variations:  {category_counts['scale']}"
    )

    print(
        f"  PDF variations:    {category_counts['pdf']}"
    )

    print(
        f"  shower variations: {category_counts['shower']}"
    )

    print(
        f"  model variations:  {category_counts['model']}"
    )

    # --------------------------------------------------------
    # Final relative systematic magnitudes.
    # --------------------------------------------------------

    h_scale = envelope(
        nominal,
        grouped[
            "scale"
        ],
        (
            f"{pair_label}_"
            "relative_unc_scale"
        ),
    )

    h_pdf = rms(
        nominal,
        grouped[
            "pdf"
        ],
        (
            f"{pair_label}_"
            "relative_unc_pdf"
        ),
    )

    h_shower = envelope(
        nominal,
        grouped[
            "shower"
        ],
        (
            f"{pair_label}_"
            "relative_unc_shower"
        ),
    )

    h_model = envelope(
        nominal,
        grouped[
            "model"
        ],
        (
            f"{pair_label}_"
            "relative_unc_model"
        ),
    )

    h_total = quadrature(
        nominal,
        [
            h_scale,
            h_pdf,
            h_shower,
            h_model,
        ],
        (
            f"{pair_label}_"
            "relative_unc_total_systematic"
        ),
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
        reference_csv_path,
        reference_max_delta,
    )

    print(
        f"  summary CSV:    {summary_csv}"
    )

    print(
        f"  per-weight CSV: {per_weight_csv}"
    )

    print(
        f"  ROOT output:    {root_path}"
    )

    return {
        "pair_label": (
            pair_label
        ),
        "summary_csv": (
            summary_csv
        ),
        "per_weight_csv": (
            per_weight_csv
        ),
        "root_path": (
            root_path
        ),
        "counts": (
            category_counts
        ),
    }


# ============================================================
# Terminal summary
# ============================================================

def print_summary_table(
    pair_label,
    csv_path,
):
    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(
            csv.DictReader(
                csv_file
            )
        )

    print()
    print(
        f"{pair_label} relative-systematic summary:"
    )

    print(
        "  eta bin       scale[%]    PDF[%]      "
        "shower[%]   model[%]    total syst[%]"
    )

    for row in rows:
        print(
            "  "
            f"{float(row['bin_low_edge']):.2f}-"
            f"{float(row['bin_up_edge']):.2f}  "
            f"{float(row['scale_unc_percent']):9.4f}  "
            f"{float(row['pdf_unc_percent']):9.4f}  "
            f"{float(row['shower_unc_percent']):9.4f}  "
            f"{float(row['model_unc_percent']):9.4f}  "
            f"{float(row['total_systematic_unc_percent']):12.4f}"
        )


# ============================================================
# Main
# ============================================================

def main():
    print()
    print(
        "=" * 72
    )

    print(
        " Repaired relative systematic uncertainty extraction"
    )

    print(
        "=" * 72
    )

    print()
    print(
        "Definition:"
    )

    print(
        "  delta_k = (C_k - C_0) / C_0"
    )

    print(
        "  C_k and C_0 use the SAME repaired weighted sample"
    )

    print()
    print(
        "Final relative uncertainties:"
    )

    print(
        "  scale  = envelope"
    )

    print(
        "  PDF    = RMS"
    )

    print(
        "  shower = envelope"
    )

    print(
        "  model  = envelope"
    )

    print()
    print(
        "Input histograms:"
    )

    print(
        f"  {INPUT_DIR}"
    )

    print()
    print(
        "Nominal-reference CSVs:"
    )

    print(
        f"  {REFERENCE_CORRECTION_CSV_DIR}"
    )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  CSV:  {CSV_DIR}"
    )

    print(
        f"  ROOT: {ROOT_DIR}"
    )

    files_by_sample = scan_inputs()

    expected_samples = {
        sample
        for pair in (
            SAMPLE_PAIRS.values()
        )
        for sample in pair
    }

    missing_samples = sorted(
        expected_samples
        - set(
            files_by_sample
        )
    )

    if missing_samples:
        raise RuntimeError(
            "Required repaired histogram samples are missing:\n  "
            + "\n  ".join(
                missing_samples
            )
        )

    completed = {}

    for (
        pair_label,
        (
            particle_sample,
            parton_sample,
        ),
    ) in SAMPLE_PAIRS.items():
        completed[
            pair_label
        ] = process_pair(
            pair_label,
            particle_sample,
            parton_sample,
            files_by_sample,
        )

    print()
    print(
        "=" * 72
    )

    print(
        " Relative-systematic outputs"
    )

    print(
        "=" * 72
    )

    for pair_label in SAMPLE_PAIRS:
        print_summary_table(
            pair_label,
            completed[
                pair_label
            ][
                "summary_csv"
            ],
        )

    print()
    print(
        "=" * 72
    )

    print(
        " Finished successfully"
    )

    print(
        "=" * 72
    )

    print()
    print(
        "The *_relative_uncertainties.csv schemas remain compatible "
        "with the existing downstream combination/plotting scripts."
    )


if __name__ == "__main__":
    main()
