#!/usr/bin/env python3

"""
MTWcut_build_corrections_etalepton.py

REPAIRED FINAL version.

Purpose
-------
Read the ROOT histograms produced by the repaired

    MTWcut_make_histograms_etalepton.py

and construct the lepton-|eta| W+c parton-to-particle correction factors

    C_i = N_parton,i / N_particle,i

for:

    Pythia W+
    Pythia W-
    Herwig W+
    Herwig W-

The input histograms already contain the complete repaired event
selection.  This builder DOES NOT re-run the event selection.

Required upstream selection
---------------------------
The script validates the metadata stored in every input histogram and
REFUSES to use stale/pre-repair files unless they state that the
following selection was applied:

    MET > 25 GeV
    lepton pT > 20 GeV
    |eta_lepton| < 2.5
    mT(W) > 40 GeV
    jet pT > 25 GeV
    |eta_jet| < 2.5
    exactly one fiducial charm-identified jet
    charm identification: jet_charge != 0
    OS-SS event sign:
        opposite-sign W/charm -> +1
        same-sign W/charm     -> -1

Uncertainty definitions
-----------------------
For the OLD weighted samples used by this correction builder:

Statistical:
    propagated ROOT ratio uncertainty from Sumw2 histograms.

Scale:
    envelope of |C_k - C_0|.

PDF:
    RMS of C_k - C_0.

Shower:
    envelope of |C_k - C_0|.

Model:
    envelope of |C_k - C_0|.

Total:
    quadrature of
        stat, scale, PDF, shower, model.

This is the conservative/original independent statistical treatment.
The exploratory event-matching/covariance study is deliberately NOT
used here.

Important
---------
- Only weights existing at BOTH particle and parton level are used.
- Missing weights are skipped.
- No fake/substitute weights are introduced.
- Weight 318 is explicitly excluded.
- PDF weight 281 is explicitly excluded from the PDF RMS.
- Input histogram binning is checked before every ratio.
- Zero particle-level denominator bins are treated as an error rather
  than silently creating an undefined correction.
- Input metadata is validated so stale pre-jet-selection ROOT files
  cannot silently enter the repaired pipeline.

Recommended location
--------------------
uncertainty_decomposition/
└── lepton_pseudorapidity_work_FINAL/
    ├── MTWcut_build_corrections_etalepton.py
    ├── MTWcut_make_histograms_etalepton_outputs/
    ├── MTWcut_build_corrections_etalepton_outputs/
    └── MTWcut_build_corrections_etalepton_csv/

Run from the project root
-------------------------
python3 lepton_pseudorapidity_work_FINAL/MTWcut_build_corrections_etalepton.py
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

INPUT_DIR = (
    SCRIPT_DIR
    / "MTWcut_make_histograms_etalepton_outputs"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "MTWcut_build_corrections_etalepton_outputs"
)

CSV_DIR = (
    SCRIPT_DIR
    / "MTWcut_build_corrections_etalepton_csv"
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
# Systematic-weight definitions
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
# Repaired-selection metadata expected from upstream
# ============================================================

EXPECTED_MTW_GEV = 40.0
EXPECTED_JET_PT_GEV = 25.0
EXPECTED_JET_ABS_ETA = 2.5
EXPECTED_REQUIRED_CHARM_JETS = 1

FLOAT_METADATA_TOLERANCE = 1.0e-9


# ============================================================
# Input filename parsing
# ============================================================

def parse_output_filename(
    filename: str,
):
    """
    Parse files written by MTWcut_make_histograms_etalepton.py.

    Expected:
        output_w<index>_<weight_name>_<sample>.root
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


# ============================================================
# ROOT metadata helpers
# ============================================================

def get_named_title(
    root_file,
    key: str,
):
    obj = root_file.Get(
        key
    )

    if not obj:
        return None

    if not hasattr(
        obj,
        "GetTitle",
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
            "This usually means the file was produced before the "
            "jet/charm-selection repair. Re-run "
            "MTWcut_make_histograms_etalepton.py first."
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


def validate_repaired_input_metadata(
    root_file,
    info,
):
    """
    Refuse stale histogram files that do not document the repaired
    selection.

    The repaired histogram-maker writes these TNamed objects.
    """

    path = info[
        "path"
    ]

    expected_sample = info[
        "sample"
    ]

    expected_weight = info[
        "weight_index"
    ]

    observable = require_named_title(
        root_file,
        "Observable",
        path,
    )

    if observable != OBSERVABLE:
        raise RuntimeError(
            f"Observable mismatch in:\n  {path}\n"
            f"Expected: {OBSERVABLE}\n"
            f"Found:    {observable}"
        )

    sample_name = require_named_title(
        root_file,
        "SampleName",
        path,
    )

    if sample_name != expected_sample:
        raise RuntimeError(
            f"Sample metadata mismatch in:\n  {path}\n"
            f"Filename sample: {expected_sample}\n"
            f"Metadata sample: {sample_name}"
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

    if effective_weight != expected_weight:
        raise RuntimeError(
            f"Weight-index metadata mismatch in:\n  {path}\n"
            f"Filename weight: {expected_weight}\n"
            f"Metadata weight: {effective_weight}"
        )

    mtw_applied = require_named_title(
        root_file,
        "MTWCutApplied",
        path,
    )

    if mtw_applied != "True":
        raise RuntimeError(
            f"mT(W) cut is not marked as applied in:\n  {path}"
        )

    mtw_value = parse_metadata_float(
        require_named_title(
            root_file,
            "MTWCutGeV",
            path,
        ),
        "MTWCutGeV",
        path,
    )

    if not math.isclose(
        mtw_value,
        EXPECTED_MTW_GEV,
        rel_tol=0.0,
        abs_tol=FLOAT_METADATA_TOLERANCE,
    ):
        raise RuntimeError(
            f"Unexpected mT(W) cut in:\n  {path}\n"
            f"Expected: {EXPECTED_MTW_GEV}\n"
            f"Found:    {mtw_value}"
        )

    jet_selection = require_named_title(
        root_file,
        "JetSelectionApplied",
        path,
    )

    if jet_selection != "True":
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
            f"Unexpected jet pT cut in:\n  {path}\n"
            f"Expected: {EXPECTED_JET_PT_GEV}\n"
            f"Found:    {jet_pt}"
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
            f"Unexpected jet |eta| cut in:\n  {path}\n"
            f"Expected: {EXPECTED_JET_ABS_ETA}\n"
            f"Found:    {jet_abs_eta}"
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
            f"Unexpected charm-jet multiplicity in:\n  {path}\n"
            f"Expected exactly: {EXPECTED_REQUIRED_CHARM_JETS}\n"
            f"Found:            {required_charm_jets}"
        )

    charm_branch = require_named_title(
        root_file,
        "CharmJetIDBranch",
        path,
    )

    if charm_branch != "jet_charge":
        raise RuntimeError(
            f"Unexpected charm-ID branch in:\n  {path}\n"
            f"Expected: jet_charge\n"
            f"Found:    {charm_branch}"
        )

    charm_definition = require_named_title(
        root_file,
        "CharmJetIDDefinition",
        path,
    )

    if "jet_charge != 0" not in charm_definition:
        raise RuntimeError(
            f"Unexpected charm-ID definition in:\n  {path}\n"
            f"Found: {charm_definition}"
        )

    os_ss_definition = require_named_title(
        root_file,
        "OSSSWeightDefinition",
        path,
    )

    required_os_ss_phrases = (
        "+1 for opposite-sign",
        "-1 for same-sign",
    )

    if not all(
        phrase in os_ss_definition
        for phrase in required_os_ss_phrases
    ):
        raise RuntimeError(
            f"Unexpected OS-SS definition in:\n  {path}\n"
            f"Found: {os_ss_definition}"
        )

    event_selection = require_named_title(
        root_file,
        "EventSelection",
        path,
    )

    required_selection_phrases = (
        "met_et > 25 GeV",
        "leptons_pt > 20 GeV",
        "abs(leptons_eta) < 2.5",
        "m_T^W > 40 GeV",
        "jet_pt > 25 GeV",
        "abs(jet_eta) < 2.5",
        "exactly one charm-identified fiducial jet",
        "jet_charge != 0",
    )

    missing_phrases = [
        phrase
        for phrase in required_selection_phrases
        if phrase not in event_selection
    ]

    if missing_phrases:
        raise RuntimeError(
            "Input ROOT metadata does not describe the complete "
            "repaired selection:\n"
            f"  {path}\n"
            f"Missing text: {missing_phrases}\n"
            f"EventSelection metadata: {event_selection}"
        )


# ============================================================
# Scan input directory
# ============================================================

def scan_input_files():
    """
    files_by_sample[sample][weight_index] = info
    """

    if not INPUT_DIR.is_dir():
        raise RuntimeError(
            "Input histogram directory does not exist:\n"
            f"  {INPUT_DIR}\n\n"
            "Run the repaired MTWcut_make_histograms_etalepton.py "
            "first."
        )

    files_by_sample = {}

    ignored_root_files = []

    for path in sorted(
        INPUT_DIR.iterdir()
    ):
        if (
            not path.is_file()
            or path.suffix != ".root"
        ):
            continue

        info = parse_output_filename(
            path.name
        )

        if info is None:
            ignored_root_files.append(
                path.name
            )
            continue

        sample = info[
            "sample"
        ]

        weight_index = info[
            "weight_index"
        ]

        sample_table = files_by_sample.setdefault(
            sample,
            {},
        )

        if weight_index in sample_table:
            raise RuntimeError(
                "Duplicate histogram output for the same sample/weight:\n"
                f"  sample: {sample}\n"
                f"  weight: {weight_index}\n"
                f"  first:  {sample_table[weight_index]['path']}\n"
                f"  second: {path}"
            )

        sample_table[
            weight_index
        ] = info

    if ignored_root_files:
        print()
        print(
            "Warning: ignored ROOT files that do not match the "
            "expected histogram filename format:"
        )

        for name in ignored_root_files:
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
# ROOT file helpers
# ============================================================

def open_root_file(
    info,
):
    path = info[
        "path"
    ]

    root_file = ROOT.TFile.Open(
        str(
            path
        ),
        "READ",
    )

    if (
        not root_file
        or root_file.IsZombie()
    ):
        raise RuntimeError(
            f"Could not open ROOT file:\n  {path}"
        )

    validate_repaired_input_metadata(
        root_file,
        info,
    )

    return root_file


def open_root_files_for_sample(
    sample_files,
):
    """
    Open and validate every weight ROOT file for one sample.
    """

    opened = {}

    try:
        for weight_index, info in (
            sample_files.items()
        ):
            opened[
                weight_index
            ] = open_root_file(
                info
            )

    except Exception:
        for root_file in opened.values():
            root_file.Close()
        raise

    return opened


def close_root_files(
    opened_files,
):
    for root_file in (
        opened_files.values()
    ):
        root_file.Close()


def get_etalepton_histogram(
    root_file,
    clone_name: str,
):
    hist = root_file.Get(
        OBSERVABLE
    )

    if not hist:
        raise RuntimeError(
            f"Histogram '{OBSERVABLE}' not found in:\n"
            f"  {root_file.GetName()}"
        )

    if not hist.InheritsFrom(
        "TH1"
    ):
        raise RuntimeError(
            f"Object '{OBSERVABLE}' is not a TH1 in:\n"
            f"  {root_file.GetName()}"
        )

    cloned = hist.Clone(
        clone_name
    )

    cloned.SetDirectory(
        0
    )

    return cloned


# ============================================================
# Histogram compatibility
# ============================================================

def validate_histogram_binning(
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
            f"  {label_a}: {hist_a.GetNbinsX()}\n"
            f"  {label_b}: {hist_b.GetNbinsX()}"
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
                f"  edge index {ibin}: "
                f"{edge_a} != {edge_b}"
            )


def validate_nonzero_denominator(
    particle_hist,
    label: str,
):
    zero_bins = []

    for ibin in range(
        1,
        particle_hist.GetNbinsX() + 1,
    ):
        value = float(
            particle_hist.GetBinContent(
                ibin
            )
        )

        if value == 0.0:
            zero_bins.append(
                ibin
            )

    if zero_bins:
        raise RuntimeError(
            "Particle-level correction denominator contains zero "
            "bin content, so the correction is undefined:\n"
            f"  {label}\n"
            f"  zero bins: {zero_bins}"
        )


# ============================================================
# Correction factor
# ============================================================

def make_ratio_histogram(
    h_particle,
    h_parton,
    out_name: str,
    context: str,
):
    """
    Construct:
        C = parton / particle

    ROOT propagates the independent Sumw2 uncertainties.
    """

    validate_histogram_binning(
        h_particle,
        h_parton,
        f"{context} particle",
        f"{context} parton",
    )

    validate_nonzero_denominator(
        h_particle,
        f"{context} particle",
    )

    ratio = h_parton.Clone(
        out_name
    )

    ratio.SetDirectory(
        0
    )

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
    name: str,
):
    out = reference_hist.Clone(
        name
    )

    out.Reset()
    out.SetDirectory(
        0
    )

    return out


# ============================================================
# Systematic-weight classification
# ============================================================

def is_pdf_weight_name(
    weight_name: str,
):
    return weight_name.startswith(
        "MUR1.0_MUF1.0_PDF"
    )


def classify_common_weights(
    common_weights,
    particle_files,
):
    """
    Preserve the established project weight definitions.
    """

    scale = [
        index
        for index in SCALE_WEIGHTS
        if index in common_weights
    ]

    shower = [
        index
        for index in SHOWER_WEIGHTS
        if index in common_weights
    ]

    model = [
        index
        for index in MODEL_WEIGHTS
        if index in common_weights
    ]

    pdf = []

    for index in common_weights:
        if index == NOMINAL_WEIGHT:
            continue

        if index in SCALE_WEIGHTS:
            continue

        if index in SHOWER_WEIGHTS:
            continue

        if index in MODEL_WEIGHTS:
            continue

        if index in EXCLUDED_PDF_WEIGHTS:
            continue

        weight_name = (
            particle_files[
                index
            ][
                "weight_name"
            ]
        )

        if is_pdf_weight_name(
            weight_name
        ):
            pdf.append(
                index
            )

    return {
        "scale": scale,
        "pdf": pdf,
        "shower": shower,
        "model": model,
    }


# ============================================================
# Uncertainty calculations
# ============================================================

def compute_envelope_uncertainty(
    nominal_hist,
    varied_hists,
    out_name: str,
):
    """
    max |C_variation - C_nominal|
    """

    out = empty_hist_like(
        nominal_hist,
        out_name,
    )

    for ibin in range(
        1,
        nominal_hist.GetNbinsX() + 1,
    ):
        central = float(
            nominal_hist.GetBinContent(
                ibin
            )
        )

        max_deviation = 0.0

        for hist in varied_hists:
            deviation = abs(
                float(
                    hist.GetBinContent(
                        ibin
                    )
                )
                - central
            )

            max_deviation = max(
                max_deviation,
                deviation,
            )

        out.SetBinContent(
            ibin,
            max_deviation,
        )

        out.SetBinError(
            ibin,
            0.0,
        )

    return out


def compute_rms_uncertainty(
    nominal_hist,
    varied_hists,
    out_name: str,
):
    """
    RMS(C_variation - C_nominal)
    """

    out = empty_hist_like(
        nominal_hist,
        out_name,
    )

    if not varied_hists:
        return out

    for ibin in range(
        1,
        nominal_hist.GetNbinsX() + 1,
    ):
        central = float(
            nominal_hist.GetBinContent(
                ibin
            )
        )

        squared = []

        for hist in varied_hists:
            delta = (
                float(
                    hist.GetBinContent(
                        ibin
                    )
                )
                - central
            )

            squared.append(
                delta * delta
            )

        rms = math.sqrt(
            sum(
                squared
            )
            / len(
                squared
            )
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


def compute_stat_uncertainty_from_ratio(
    ratio_hist,
    out_name: str,
):
    """
    Store the propagated independent ROOT ratio error.
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
            abs(
                float(
                    ratio_hist.GetBinError(
                        ibin
                    )
                )
            ),
        )

        out.SetBinError(
            ibin,
            0.0,
        )

    return out


def compute_total_uncertainty(
    nominal_hist,
    uncertainty_hists,
    out_name: str,
):
    out = empty_hist_like(
        nominal_hist,
        out_name,
    )

    for ibin in range(
        1,
        nominal_hist.GetNbinsX() + 1,
    ):
        sum_squared = sum(
            float(
                hist.GetBinContent(
                    ibin
                )
            ) ** 2
            for hist in uncertainty_hists
        )

        out.SetBinContent(
            ibin,
            math.sqrt(
                sum_squared
            ),
        )

        out.SetBinError(
            ibin,
            0.0,
        )

    return out


def compute_generator_difference(
    hist_a,
    hist_b,
    out_name: str,
):
    validate_histogram_binning(
        hist_a,
        hist_b,
        hist_a.GetName(),
        hist_b.GetName(),
    )

    out = empty_hist_like(
        hist_a,
        out_name,
    )

    for ibin in range(
        1,
        hist_a.GetNbinsX() + 1,
    ):
        difference = abs(
            float(
                hist_a.GetBinContent(
                    ibin
                )
            )
            - float(
                hist_b.GetBinContent(
                    ibin
                )
            )
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
# Build systematic variations
# ============================================================

def build_variation_histograms(
    pair_label,
    weight_list,
    particle_open_files,
    parton_open_files,
    category: str,
):
    varied = []

    for weight_index in weight_list:
        h_particle = get_etalepton_histogram(
            particle_open_files[
                weight_index
            ],
            (
                f"{pair_label}_"
                f"{OBSERVABLE}_particle_"
                f"{category}_w{weight_index}"
            ),
        )

        h_parton = get_etalepton_histogram(
            parton_open_files[
                weight_index
            ],
            (
                f"{pair_label}_"
                f"{OBSERVABLE}_parton_"
                f"{category}_w{weight_index}"
            ),
        )

        h_corr = make_ratio_histogram(
            h_particle,
            h_parton,
            (
                f"{pair_label}_"
                f"{OBSERVABLE}_corr_"
                f"{category}_w{weight_index}"
            ),
            context=(
                f"{pair_label} "
                f"{category} weight {weight_index}"
            ),
        )

        varied.append(
            h_corr
        )

    return varied


# ============================================================
# CSV output
# ============================================================

CSV_COLUMNS = [
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
]


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
    Preserve the existing downstream CSV schema.
    """

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            CSV_COLUMNS
        )

        for ibin in range(
            1,
            h_corr.GetNbinsX() + 1,
        ):
            writer.writerow([
                pair_label,
                OBSERVABLE,
                ibin,
                (
                    h_corr
                    .GetXaxis()
                    .GetBinLowEdge(
                        ibin
                    )
                ),
                (
                    h_corr
                    .GetXaxis()
                    .GetBinUpEdge(
                        ibin
                    )
                ),
                float(
                    h_corr.GetBinContent(
                        ibin
                    )
                ),
                float(
                    h_stat.GetBinContent(
                        ibin
                    )
                ),
                float(
                    h_scale.GetBinContent(
                        ibin
                    )
                ),
                float(
                    h_pdf.GetBinContent(
                        ibin
                    )
                ),
                float(
                    h_shower.GetBinContent(
                        ibin
                    )
                ),
                float(
                    h_model.GetBinContent(
                        ibin
                    )
                ),
                float(
                    h_total.GetBinContent(
                        ibin
                    )
                ),
            ])


def write_generator_difference_csv(
    csv_path,
    charge_label,
    h_difference,
):
    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(
            csv_file
        )

        writer.writerow([
            "charge",
            "observable",
            "bin",
            "bin_low_edge",
            "bin_up_edge",
            "absolute_pythia_herwig_difference",
        ])

        for ibin in range(
            1,
            h_difference.GetNbinsX() + 1,
        ):
            writer.writerow([
                charge_label,
                OBSERVABLE,
                ibin,
                (
                    h_difference
                    .GetXaxis()
                    .GetBinLowEdge(
                        ibin
                    )
                ),
                (
                    h_difference
                    .GetXaxis()
                    .GetBinUpEdge(
                        ibin
                    )
                ),
                float(
                    h_difference.GetBinContent(
                        ibin
                    )
                ),
            ])


# ============================================================
# ROOT output helpers
# ============================================================

def write_hist(
    output_file,
    hist,
    name=None,
):
    output_file.cd()

    if name is None:
        hist.Write()
    else:
        hist.Write(
            name
        )


def write_pair_metadata(
    output_file,
    pair_label,
    particle_sample,
    parton_sample,
    weight_groups,
):
    output_file.cd()

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

    ROOT.TNamed(
        "CorrectionDefinition",
        "C_i = N_parton,i / N_particle,i",
    ).Write()

    ROOT.TNamed(
        "StatisticalUncertaintyDefinition",
        (
            "Independent ROOT ratio uncertainty propagated "
            "from input Sumw2 histograms"
        ),
    ).Write()

    ROOT.TNamed(
        "ScaleUncertaintyDefinition",
        "Envelope max |C_k - C_0|",
    ).Write()

    ROOT.TNamed(
        "PDFUncertaintyDefinition",
        "RMS(C_k - C_0)",
    ).Write()

    ROOT.TNamed(
        "ShowerUncertaintyDefinition",
        "Envelope max |C_k - C_0|",
    ).Write()

    ROOT.TNamed(
        "ModelUncertaintyDefinition",
        "Envelope max |C_k - C_0|",
    ).Write()

    ROOT.TNamed(
        "TotalUncertaintyDefinition",
        (
            "Quadrature(stat, scale, PDF, shower, model)"
        ),
    ).Write()

    ROOT.TNamed(
        "UpstreamSelectionValidated",
        "True",
    ).Write()

    ROOT.TNamed(
        "UpstreamSelectionSummary",
        (
            "MET>25 GeV; lepton pT>20 GeV; "
            "|eta_l|<2.5; mT(W)>40 GeV; "
            "jet pT>25 GeV; |eta_j|<2.5; "
            "exactly one fiducial charm-ID jet with jet_charge!=0; "
            "OS +1, SS -1"
        ),
    ).Write()

    for category in (
        "scale",
        "pdf",
        "shower",
        "model",
    ):
        ROOT.TNamed(
            f"{category.capitalize()}WeightsUsed",
            ",".join(
                str(
                    index
                )
                for index in (
                    weight_groups[
                        category
                    ]
                )
            ),
        ).Write()


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
            "  particle-only weights skipped: "
            f"{particle_only}"
        )

    if parton_only:
        print(
            "  parton-only weights skipped: "
            f"{parton_only}"
        )

    if NOMINAL_WEIGHT not in common_weights:
        raise RuntimeError(
            f"{pair_label}: common nominal weight 0 is missing."
        )

    weight_groups = classify_common_weights(
        common_weights,
        particle_files,
    )

    print(
        f"  scale weights:  {weight_groups['scale']}"
    )

    print(
        f"  PDF weights:    {len(weight_groups['pdf'])}"
    )

    print(
        f"  shower weights: {weight_groups['shower']}"
    )

    print(
        f"  model weights:  {weight_groups['model']}"
    )

    particle_open_files = {}
    parton_open_files = {}

    output_file = None

    try:
        # Open only common usable weights.  This both avoids unnecessary
        # handles and ensures every file that contributes is metadata-checked.
        particle_subset = {
            index: particle_files[
                index
            ]
            for index in common_weights
        }

        parton_subset = {
            index: parton_files[
                index
            ]
            for index in common_weights
        }

        particle_open_files = (
            open_root_files_for_sample(
                particle_subset
            )
        )

        parton_open_files = (
            open_root_files_for_sample(
                parton_subset
            )
        )

        output_root_path = (
            OUTPUT_DIR
            / f"corrections_{pair_label}.root"
        )

        output_file = ROOT.TFile(
            str(
                output_root_path
            ),
            "RECREATE",
        )

        if (
            not output_file
            or output_file.IsZombie()
        ):
            raise RuntimeError(
                "Could not create ROOT output:\n"
                f"  {output_root_path}"
            )

        write_pair_metadata(
            output_file,
            pair_label,
            particle_sample,
            parton_sample,
            weight_groups,
        )

        # ----------------------------------------------------
        # Nominal
        # ----------------------------------------------------

        h_particle_nom = (
            get_etalepton_histogram(
                particle_open_files[
                    NOMINAL_WEIGHT
                ],
                (
                    f"{pair_label}_"
                    f"{OBSERVABLE}_particle_nominal"
                ),
            )
        )

        h_parton_nom = (
            get_etalepton_histogram(
                parton_open_files[
                    NOMINAL_WEIGHT
                ],
                (
                    f"{pair_label}_"
                    f"{OBSERVABLE}_parton_nominal"
                ),
            )
        )

        h_corr_nom = (
            make_ratio_histogram(
                h_particle_nom,
                h_parton_nom,
                (
                    f"{pair_label}_"
                    f"{OBSERVABLE}_corr_nominal"
                ),
                context=(
                    f"{pair_label} nominal weight 0"
                ),
            )
        )

        write_hist(
            output_file,
            h_corr_nom,
            "correction_nominal",
        )

        # ----------------------------------------------------
        # Statistical uncertainty
        # ----------------------------------------------------

        h_unc_stat = (
            compute_stat_uncertainty_from_ratio(
                h_corr_nom,
                (
                    f"{pair_label}_"
                    f"{OBSERVABLE}_unc_stat"
                ),
            )
        )

        write_hist(
            output_file,
            h_unc_stat,
            "unc_stat",
        )

        # ----------------------------------------------------
        # Systematic variations
        # ----------------------------------------------------

        varied_scale = (
            build_variation_histograms(
                pair_label,
                weight_groups[
                    "scale"
                ],
                particle_open_files,
                parton_open_files,
                "scale",
            )
        )

        varied_pdf = (
            build_variation_histograms(
                pair_label,
                weight_groups[
                    "pdf"
                ],
                particle_open_files,
                parton_open_files,
                "pdf",
            )
        )

        varied_shower = (
            build_variation_histograms(
                pair_label,
                weight_groups[
                    "shower"
                ],
                particle_open_files,
                parton_open_files,
                "shower",
            )
        )

        varied_model = (
            build_variation_histograms(
                pair_label,
                weight_groups[
                    "model"
                ],
                particle_open_files,
                parton_open_files,
                "model",
            )
        )

        h_unc_scale = (
            compute_envelope_uncertainty(
                h_corr_nom,
                varied_scale,
                (
                    f"{pair_label}_"
                    f"{OBSERVABLE}_unc_scale"
                ),
            )
        )

        h_unc_pdf = (
            compute_rms_uncertainty(
                h_corr_nom,
                varied_pdf,
                (
                    f"{pair_label}_"
                    f"{OBSERVABLE}_unc_pdf"
                ),
            )
        )

        h_unc_shower = (
            compute_envelope_uncertainty(
                h_corr_nom,
                varied_shower,
                (
                    f"{pair_label}_"
                    f"{OBSERVABLE}_unc_shower"
                ),
            )
        )

        h_unc_model = (
            compute_envelope_uncertainty(
                h_corr_nom,
                varied_model,
                (
                    f"{pair_label}_"
                    f"{OBSERVABLE}_unc_model"
                ),
            )
        )

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
                    f"{pair_label}_"
                    f"{OBSERVABLE}_unc_total"
                ),
            )
        )

        write_hist(
            output_file,
            h_unc_scale,
            "unc_scale",
        )

        write_hist(
            output_file,
            h_unc_pdf,
            "unc_pdf",
        )

        write_hist(
            output_file,
            h_unc_shower,
            "unc_shower",
        )

        write_hist(
            output_file,
            h_unc_model,
            "unc_model",
        )

        write_hist(
            output_file,
            h_unc_total,
            "unc_total",
        )

        # Save the individual varied correction factors too.  These are
        # diagnostics and do not change downstream CSV compatibility.
        variation_directory = output_file.mkdir(
            "correction_variations"
        )

        variation_directory.cd()

        for category, hists in (
            (
                "scale",
                varied_scale,
            ),
            (
                "pdf",
                varied_pdf,
            ),
            (
                "shower",
                varied_shower,
            ),
            (
                "model",
                varied_model,
            ),
        ):
            for hist in hists:
                hist.Write()

        output_file.cd()

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

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

        # Detached copy used later for Pythia/Herwig comparison.
        nominal_copy = h_corr_nom.Clone(
            (
                f"{pair_label}_"
                f"{OBSERVABLE}_nominal_for_generator_compare"
            )
        )

        nominal_copy.SetDirectory(
            0
        )

        print(
            f"  ROOT: {output_root_path}"
        )

        print(
            f"  CSV:  {csv_path}"
        )

        return nominal_copy

    finally:
        if output_file:
            output_file.Write()
            output_file.Close()

        close_root_files(
            particle_open_files
        )

        close_root_files(
            parton_open_files
        )


# ============================================================
# Generator difference
# ============================================================

def write_generator_differences(
    nominal_ratios,
):
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

    print()
    print(
        "=" * 72
    )
    print(
        "Pythia / Herwig nominal generator differences"
    )
    print(
        "=" * 72
    )

    for (
        charge_label,
        pythia_key,
        herwig_key,
    ) in generator_pairs:
        if (
            pythia_key
            not in nominal_ratios
            or herwig_key
            not in nominal_ratios
        ):
            raise RuntimeError(
                "Cannot build generator difference because a "
                "nominal correction is missing:\n"
                f"  {pythia_key}\n"
                f"  {herwig_key}"
            )

        h_difference = (
            compute_generator_difference(
                nominal_ratios[
                    pythia_key
                ],
                nominal_ratios[
                    herwig_key
                ],
                (
                    f"{OBSERVABLE}_"
                    f"unc_generator_{charge_label}"
                ),
            )
        )

        root_path = (
            OUTPUT_DIR
            / (
                "generator_difference_"
                f"{charge_label}.root"
            )
        )

        output_file = ROOT.TFile(
            str(
                root_path
            ),
            "RECREATE",
        )

        if (
            not output_file
            or output_file.IsZombie()
        ):
            raise RuntimeError(
                "Could not create generator-difference ROOT file:\n"
                f"  {root_path}"
            )

        ROOT.TNamed(
            "Charge",
            charge_label,
        ).Write()

        ROOT.TNamed(
            "Definition",
            "|C_Pythia - C_Herwig|",
        ).Write()

        ROOT.TNamed(
            "UpstreamSelectionValidated",
            "True",
        ).Write()

        h_difference.Write(
            "unc_generator"
        )

        output_file.Write()
        output_file.Close()

        csv_path = (
            CSV_DIR
            / (
                "generator_difference_"
                f"{charge_label}.csv"
            )
        )

        write_generator_difference_csv(
            csv_path,
            charge_label,
            h_difference,
        )

        print(
            f"  {charge_label}:"
        )

        print(
            f"    ROOT: {root_path}"
        )

        print(
            f"    CSV:  {csv_path}"
        )


# ============================================================
# Terminal summary
# ============================================================

def print_pair_bin_summary(
    pair_label,
    csv_path,
):
    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )

        rows = list(
            reader
        )

    print()
    print(
        f"{pair_label} bin summary:"
    )

    print(
        "  eta bin        C             stat          "
        "scale         PDF           shower        "
        "model         total"
    )

    for row in rows:
        print(
            "  "
            f"{float(row['bin_low_edge']):.2f}-"
            f"{float(row['bin_up_edge']):.2f}  "
            f"{float(row['correction_factor']): .7f}  "
            f"{float(row['stat_unc']): .7f}  "
            f"{float(row['scale_unc']): .7f}  "
            f"{float(row['pdf_unc']): .7f}  "
            f"{float(row['shower_unc']): .7f}  "
            f"{float(row['model_unc']): .7f}  "
            f"{float(row['total_unc']): .7f}"
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
        " Repaired W+c lepton-|eta| correction builder"
    )
    print(
        "=" * 72
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
        "Required upstream selection:"
    )

    print(
        "  MET > 25 GeV"
    )

    print(
        "  lepton pT > 20 GeV"
    )

    print(
        "  |eta_lepton| < 2.5"
    )

    print(
        "  mT(W) > 40 GeV"
    )

    print(
        "  jet pT > 25 GeV"
    )

    print(
        "  |eta_jet| < 2.5"
    )

    print(
        "  exactly one charm-ID fiducial jet"
    )

    print(
        "  charm ID: jet_charge != 0"
    )

    print(
        "  OS-SS: opposite sign +1, same sign -1"
    )

    print()
    print(
        "Statistical treatment:"
    )

    print(
        "  independent ROOT ratio error from Sumw2"
    )

    print(
        "  exploratory event-matching covariance is NOT used"
    )

    print()
    print(
        f"ROOT outputs:\n  {OUTPUT_DIR}"
    )

    print(
        f"CSV outputs:\n  {CSV_DIR}"
    )

    files_by_sample = (
        scan_input_files()
    )

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
            "Required histogram samples are missing:\n  "
            + "\n  ".join(
                missing_samples
            )
        )

    nominal_ratios = {}

    for (
        pair_label,
        (
            particle_sample,
            parton_sample,
        ),
    ) in SAMPLE_PAIRS.items():
        nominal_ratios[
            pair_label
        ] = process_pair(
            pair_label,
            particle_sample,
            parton_sample,
            files_by_sample,
        )

    write_generator_differences(
        nominal_ratios
    )

    print()
    print(
        "=" * 72
    )
    print(
        "Final correction summaries"
    )
    print(
        "=" * 72
    )

    for pair_label in (
        SAMPLE_PAIRS
    ):
        csv_path = (
            CSV_DIR
            / (
                f"{pair_label}_"
                f"{OBSERVABLE}.csv"
            )
        )

        print_pair_bin_summary(
            pair_label,
            csv_path,
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

if __name__ == "__main__":
    main()
