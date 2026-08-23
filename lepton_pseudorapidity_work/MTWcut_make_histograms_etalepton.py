#!/usr/bin/env python3

# ============================================================
# MTWcut_make_histograms_etalepton.py
#
# Purpose:
#   Produce ONLY the lepton pseudorapidity histograms needed
#   for the W+c parton-to-particle correction-factor analysis,
#   with the ATLAS-style transverse-mass requirement:
#
#       m_T^W > 40 GeV
#
# Observable produced:
#
#       |eta_lepton|
#
# No other physics histograms are produced.
#
# In particular:
#   - no lepton pT histogram
#   - no jet histograms
#   - no phi histograms
#   - no MET histogram
#   - no mT histogram
#
# The W transverse mass is calculated ONLY as an event-selection
# variable. If an event does not satisfy m_T^W > 40 GeV, it is
# rejected before the |eta_lepton| histogram is filled.
#
# Weights produced:
#   Only weights required by the uncertainty calculation:
#     - nominal weight 0
#     - scale weights
#     - PDF weights
#     - shower weights
#     - model weights
#
#   Weight 318 is excluded because it is excluded downstream.
#   Weight 281 is excluded from the PDF uncertainty, matching
#   the existing build_corrections logic.
#
# IMPORTANT SPEED IMPROVEMENT:
#   The previous eta-only script looped over the whole event tree
#   once for every weight.
#
#   This script instead loops over each sample's event tree ONCE
#   and fills all required weight histograms during that single
#   event pass. This preserves the same weighted histogram result
#   while avoiding hundreds of repeated tree scans.
#
# Folder structure expected:
#
#   project_root/
#   ├── data/
#   ├── weights/
#   └── lepton_pseudorapidity_work/
#       ├── MTWcut_make_histograms_etalepton.py
#       └── MTWcut_make_histograms_etalepton_outputs/
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
# Physics selection
# ============================================================

# Existing fiducial cuts retained from the eta-only analysis.
MET_MIN_GEV = 25.0
LEPTON_PT_MIN_GEV = 20.0
LEPTON_ABS_ETA_MAX = 2.5

# NEW fixed transverse-mass requirement.
#
# This script is specifically the mT-cut version, so there is
# intentionally NO Boolean switch.
#
# Selected events must satisfy:
#
#       m_T^W > 40 GeV
#
MTW_MIN_GEV = 40.0


# ============================================================
# Directory structure
# ============================================================

# This script should live directly inside:
#
#     lepton_pseudorapidity_work/

SCRIPT_DIR = Path(__file__).resolve().parent


# Parent directory containing:
#
#     data/
#     weights/
#     lepton_pseudorapidity_work/

PROJECT_DIR = SCRIPT_DIR.parent


DATA_DIR = PROJECT_DIR / "data"

WEIGHT_DIR = PROJECT_DIR / "weights"


# IMPORTANT:
# Write to a completely separate output folder so the original
# no-transverse-mass-cut histograms are never overwritten.

OUTPUT_DIR = (
    SCRIPT_DIR
    / "MTWcut_make_histograms_etalepton_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Lepton pseudorapidity binning
# ============================================================

# Exactly the same variable |eta_lepton| binning as the existing
# lepton pseudorapidity analysis.

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

ETA_LEPTON_BIN_ARRAY = array(
    "d",
    ETA_LEPTON_BINS,
)


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
#
# Example:
#
#   python3 MTWcut_make_histograms_etalepton.py WCPy8plus
#
# processes only WCPy8plus.

SIGNAL_SAMPLES = (
    sys.argv[1:]
    if len(sys.argv) > 1
    else SIGNAL_SAMPLES_DEFAULT
)


# ============================================================
# Particle / parton pairings
# ============================================================

# These are the exact pairings used by the correction-factor
# calculation.
#
# We use them here to determine the weights that are genuinely
# needed by BOTH members of each correction pair.

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
# Uncertainty weights required downstream
# ============================================================

# These definitions match the existing correction-building logic.

NOMINAL_WEIGHT = 0

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
    list(range(294, 314))
    + [316, 317]
)

# Weight 318 is deliberately excluded downstream.
EXCLUDED_WEIGHTS = {
    318,
}

# Weight 281 is deliberately not included in the PDF uncertainty.
EXCLUDED_PDF_WEIGHTS = {
    281,
}


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
        weight index -> weight name
    """

    idx_to_name = {}

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            idx_str, name = line.split(
                None,
                1,
            )

            idx_to_name[
                int(idx_str)
            ] = name.strip()

    return idx_to_name


def mapping_file_for_sample(sample):
    """
    Return the appropriate cleaned weight mapping file
    for a given WCharm sample.
    """

    if sample == "WCPyPartonplus":

        return (
            WEIGHT_DIR
            / "WCPyPartonplus_weights_clean.txt"
        )

    if sample == "WCPyPartonminus":

        return (
            WEIGHT_DIR
            / "WCPyPartonminus_weights_clean.txt"
        )

    if sample.startswith(
        "WCPy8"
    ):

        return (
            WEIGHT_DIR
            / "weights.py.particle_clean.txt"
        )

    if sample.startswith(
        "WCH7"
    ):

        return (
            WEIGHT_DIR
            / "weights.hw.particle_clean.txt"
        )

    if sample == "WCHPartonplus":

        return (
            WEIGHT_DIR
            / "WCHPartonplus_weights_clean.txt"
        )

    if sample == "WCHPartonminus":

        return (
            WEIGHT_DIR
            / "WCHPartonminus_weights_clean.txt"
        )

    raise RuntimeError(
        f"No weight-mapping rule exists "
        f"for sample '{sample}'"
    )


def safe_name(text):
    """
    Convert a weight name into a filename-safe string.
    """

    return "".join(
        character
        if (
            character.isalnum()
            or character in "._-"
        )
        else "_"
        for character in text
    )


def is_pdf_weight_name(weight_name):
    """
    Match the PDF-weight definition used by the existing
    correction-building script.
    """

    return weight_name.startswith(
        "MUR1.0_MUF1.0_PDF"
    )


# ============================================================
# Determine exactly which weights are required
# ============================================================

def required_weights_for_pair(
    particle_weight_map,
    parton_weight_map,
):
    """
    Return only the weight indices required for the uncertainty
    decomposition for one particle/parton pair.

    A weight must first exist in BOTH mapping files.

    Included:
        nominal
        scale
        PDF
        shower
        model

    Excluded:
        weight 318
        weight 281 from the PDF set
        any unrelated weight not used downstream
    """

    common_indices = (
        set(particle_weight_map.keys())
        & set(parton_weight_map.keys())
    )

    common_indices -= EXCLUDED_WEIGHTS


    required = set()


    # --------------------------------------------------------
    # Nominal
    # --------------------------------------------------------

    if NOMINAL_WEIGHT in common_indices:

        required.add(
            NOMINAL_WEIGHT
        )


    # --------------------------------------------------------
    # Scale variations
    # --------------------------------------------------------

    required.update(
        weight_index
        for weight_index in SCALE_WEIGHTS
        if weight_index in common_indices
    )


    # --------------------------------------------------------
    # Shower variations
    # --------------------------------------------------------

    required.update(
        weight_index
        for weight_index in SHOWER_WEIGHTS
        if weight_index in common_indices
    )


    # --------------------------------------------------------
    # Model variations
    # --------------------------------------------------------

    required.update(
        weight_index
        for weight_index in MODEL_WEIGHTS
        if weight_index in common_indices
    )


    # --------------------------------------------------------
    # PDF variations
    #
    # IMPORTANT:
    # The existing correction builder identifies PDF weights
    # using the PARTICLE-level weight name.
    # We reproduce exactly that logic here.
    # --------------------------------------------------------

    for weight_index in sorted(
        common_indices
    ):

        weight_name = (
            particle_weight_map[
                weight_index
            ]
        )

        if not is_pdf_weight_name(
            weight_name
        ):
            continue

        if weight_index in SCALE_WEIGHTS:
            continue

        if weight_index in MODEL_WEIGHTS:
            continue

        if weight_index in SHOWER_WEIGHTS:
            continue

        if weight_index in EXCLUDED_PDF_WEIGHTS:
            continue

        required.add(
            weight_index
        )


    return sorted(
        required
    )


def build_required_weight_table():
    """
    Build the exact list of weights required for each of the
    eight samples.

    Particle and parton members of a pair receive the same
    required index list so that the downstream correction can
    form the common parton/particle variations.
    """

    required_by_sample = {}

    maps_by_sample = {}


    # Load the mapping file for every sample once.
    for sample in SIGNAL_SAMPLES_DEFAULT:

        maps_by_sample[
            sample
        ] = load_weight_map_clean(
            mapping_file_for_sample(
                sample
            )
        )


    # Determine the required common weights for each pair.
    for (
        pair_label,
        (
            particle_sample,
            parton_sample,
        ),
    ) in SAMPLE_PAIRS.items():

        pair_required = (
            required_weights_for_pair(
                maps_by_sample[
                    particle_sample
                ],
                maps_by_sample[
                    parton_sample
                ],
            )
        )

        required_by_sample[
            particle_sample
        ] = pair_required

        required_by_sample[
            parton_sample
        ] = pair_required


    return (
        required_by_sample,
        maps_by_sample,
    )


# ============================================================
# Histogram creation
# ============================================================

def make_etalepton_histogram(
    sample,
    weight_index,
):
    """
    Create a fresh lepton |eta| histogram for one weight.

    Internally the histogram is given a unique name to avoid ROOT
    name collisions while hundreds of histograms are held in
    memory at the same time.

    When written to its individual ROOT file, it is written with
    the standard name:

        etalepton

    so the existing correction-building code can read it without
    any change to the histogram name.
    """

    internal_name = (
        f"etalepton_"
        f"{sample}_"
        f"w{weight_index}"
    )

    hist = ROOT.TH1F(
        internal_name,
        "etalepton",
        len(
            ETA_LEPTON_BIN_ARRAY
        ) - 1,
        ETA_LEPTON_BIN_ARRAY,
    )

    # Required for correct statistical errors with event weights.
    hist.Sumw2()

    # Prevent ROOT directory ownership issues.
    hist.SetDirectory(0)

    return hist


# ============================================================
# W transverse-mass calculation
# ============================================================

def calculate_mtw(tree):
    """
    Calculate the reconstructed W transverse mass:

        (m_T^W)^2
            =
        2 * pT_lepton * MET * (1 - cos(DeltaPhi))

    where DeltaPhi is the azimuthal separation between the
    charged lepton and the missing transverse momentum.

    Returns
    -------
    float
        m_T^W in GeV.
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

    # Numerical protection against a tiny negative value.
    if mtw_squared <= 0.0:

        return 0.0

    return ROOT.TMath.Sqrt(
        mtw_squared
    )


# ============================================================
# Single-pass event loop
# ============================================================

def fill_all_required_etalepton_histograms(
    tree,
    histograms,
    weights_to_run,
    total_events,
):
    """
    Loop over the event tree ONCE and fill every required weight
    histogram.

    Event selection:

        MET > 25 GeV
        lepton pT > 20 GeV
        |lepton eta| < 2.5
        m_T^W > 40 GeV

    The first three cuts reproduce the existing eta-only analysis.
    The fourth cut is the new transverse-mass requirement.

    Returns
    -------
    tuple
        pos_eta_sums
        neg_eta_sums
        events_after_base_cuts
        events_after_mtw_cut
    """

    pos_eta_sums = {
        weight_index: 0.0
        for weight_index in weights_to_run
    }

    neg_eta_sums = {
        weight_index: 0.0
        for weight_index in weights_to_run
    }


    events_after_base_cuts = 0

    events_after_mtw_cut = 0


    nentries = tree.GetEntries()


    # ========================================================
    # ONE event-tree pass for this sample
    # ========================================================

    for ientry in range(
        nentries
    ):

        tree.GetEntry(
            ientry
        )


        # ====================================================
        # Existing fiducial selection
        # ====================================================

        # Missing transverse momentum:
        #
        #       MET > 25 GeV

        if tree.met_et <= MET_MIN_GEV:

            continue


        # Charged-lepton transverse momentum:
        #
        #       pT_lepton > 20 GeV

        if tree.leptons_pt <= LEPTON_PT_MIN_GEV:

            continue


        # Charged-lepton pseudorapidity:
        #
        #       |eta_lepton| < 2.5

        if abs(
            tree.leptons_eta
        ) >= LEPTON_ABS_ETA_MAX:

            continue


        events_after_base_cuts += 1


        # ====================================================
        # NEW: transverse-mass selection
        # ====================================================

        mtw_value = calculate_mtw(
            tree
        )


        # This script is the dedicated transverse-mass-cut
        # version, so this cut is ALWAYS applied:
        #
        #       m_T^W > 40 GeV

        if mtw_value <= MTW_MIN_GEV:

            continue


        events_after_mtw_cut += 1


        # ====================================================
        # Observable
        # ====================================================

        eta_lepton = abs(
            tree.leptons_eta
        )


        # ====================================================
        # Charge-sign convention
        # ====================================================

        charge_factor = (
            -1.0
            * tree.leptons_charge
        )


        # ====================================================
        # Fill every REQUIRED uncertainty-weight histogram
        #
        # The tree entry has already been loaded once.
        # We now use the different weight-vector elements for
        # this accepted event.
        # ====================================================

        for weight_index in weights_to_run:

            weightxs = (
                (1.0 / total_events)
                * tree.weightvec[
                    weight_index
                ]
            )

            final_weight = (
                charge_factor
                * weightxs
            )


            # Signed-eta diagnostics retained from the
            # previous eta-only script.

            if tree.leptons_eta >= 0:

                pos_eta_sums[
                    weight_index
                ] += final_weight

            else:

                neg_eta_sums[
                    weight_index
                ] += final_weight


            # ------------------------------------------------
            # The ONLY physics histogram produced:
            #
            #       |eta_lepton|
            #
            # after m_T^W > 40 GeV.
            # ------------------------------------------------

            histograms[
                weight_index
            ].Fill(
                eta_lepton,
                final_weight,
            )


    return (
        pos_eta_sums,
        neg_eta_sums,
        events_after_base_cuts,
        events_after_mtw_cut,
    )


# ============================================================
# Weight-category label
# ============================================================

def weight_category(
    weight_index,
    weight_name,
):
    """
    Return a simple metadata label describing why a weight is
    included in this dedicated production.
    """

    if weight_index == NOMINAL_WEIGHT:

        return "nominal"

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
        and
        weight_index
        not in EXCLUDED_PDF_WEIGHTS
    ):

        return "pdf"

    return "other"


# ============================================================
# Write one output ROOT file per sample / weight
# ============================================================

def write_output_file(
    signal_sample,
    weight_index,
    weight_name,
    weight_vector_length,
    hist,
    pos_eta_sum,
    neg_eta_sum,
    events_after_base_cuts,
    events_after_mtw_cut,
):
    """
    Write the output in the same one-file-per-weight format used
    by the existing histogram-production pipeline.
    """

    output_name = (
        f"output_w"
        f"{weight_index}_"
        f"{safe_name(weight_name)}_"
        f"{signal_sample}.root"
    )

    output_path = (
        OUTPUT_DIR
        / output_name
    )


    output_file = ROOT.TFile(
        str(
            output_path
        ),
        "RECREATE",
    )

    output_file.cd()


    # ========================================================
    # Existing-style metadata
    # ========================================================

    ROOT.TNamed(
        "RequestedWeightIndex",
        str(
            weight_index
        ),
    ).Write()

    ROOT.TNamed(
        "EffectiveWeightIndex",
        str(
            weight_index
        ),
    ).Write()

    ROOT.TNamed(
        "EffectiveWeightName",
        weight_name,
    ).Write()

    ROOT.TNamed(
        "WeightVecLength",
        str(
            weight_vector_length
        ),
    ).Write()

    ROOT.TNamed(
        "SampleName",
        signal_sample,
    ).Write()

    ROOT.TNamed(
        "Observable",
        "etalepton",
    ).Write()


    # ========================================================
    # NEW transverse-mass-cut metadata
    # ========================================================

    ROOT.TNamed(
        "MTWCutApplied",
        "True",
    ).Write()

    ROOT.TNamed(
        "MTWCutDefinition",
        "m_T^W > 40 GeV",
    ).Write()

    ROOT.TNamed(
        "MTWCutGeV",
        str(
            MTW_MIN_GEV
        ),
    ).Write()


    # ========================================================
    # Record the complete event selection
    # ========================================================

    ROOT.TNamed(
        "EventSelection",
        (
            "met_et > 25 GeV; "
            "leptons_pt > 20 GeV; "
            "abs(leptons_eta) < 2.5; "
            "m_T^W > 40 GeV"
        ),
    ).Write()


    # ========================================================
    # Weight metadata
    # ========================================================

    ROOT.TNamed(
        "WeightCategory",
        weight_category(
            weight_index,
            weight_name,
        ),
    ).Write()


    # ========================================================
    # Event-count diagnostics
    # ========================================================

    ROOT.TNamed(
        "EventsAfterBaseCuts",
        str(
            events_after_base_cuts
        ),
    ).Write()

    ROOT.TNamed(
        "EventsAfterMTWCut",
        str(
            events_after_mtw_cut
        ),
    ).Write()


    # ========================================================
    # Signed eta diagnostics
    # ========================================================

    ROOT.TParameter(float)(
        "PosEtaWeightSum",
        pos_eta_sum,
    ).Write()

    ROOT.TParameter(float)(
        "NegEtaWeightSum",
        neg_eta_sum,
    ).Write()


    # ========================================================
    # The single physics histogram
    #
    # IMPORTANT:
    # Write it under the standard name "etalepton" so the
    # existing correction code can retrieve it with:
    #
    #       tf.Get("etalepton")
    # ========================================================

    hist.Write(
        "etalepton"
    )


    output_file.Write()

    output_file.Close()


# ============================================================
# Main processing
# ============================================================

def main():

    print()

    print(
        "============================================================"
    )

    print(
        " Lepton pseudorapidity production with m_T^W > 40 GeV"
    )

    print(
        "============================================================"
    )

    print()


    print(
        "Observable:"
    )

    print(
        "  |eta_lepton| only"
    )

    print()


    print(
        "Event selection:"
    )

    print(
        f"  MET > {MET_MIN_GEV:.1f} GeV"
    )

    print(
        f"  lepton pT > {LEPTON_PT_MIN_GEV:.1f} GeV"
    )

    print(
        f"  |eta_lepton| < {LEPTON_ABS_ETA_MAX:.1f}"
    )

    print(
        f"  m_T^W > {MTW_MIN_GEV:.1f} GeV"
    )

    print()


    print(
        "Data directory:"
    )

    print(
        f"  {DATA_DIR}"
    )

    print()


    print(
        "Weight directory:"
    )

    print(
        f"  {WEIGHT_DIR}"
    )

    print()


    print(
        "Output directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print()


    # ========================================================
    # Work out the exact uncertainty weights needed
    # ========================================================

    try:

        (
            required_weights_by_sample,
            weight_maps_by_sample,
        ) = build_required_weight_table()

    except Exception as exc:

        print(
            "Failed while building the required weight table:"
        )

        print(
            f"  {exc}"
        )

        return


    # ========================================================
    # Loop over requested samples
    # ========================================================

    for signal_sample in SIGNAL_SAMPLES:

        print()

        print(
            "------------------------------------------------------------"
        )

        print(
            f"Running sample: {signal_sample}"
        )

        print(
            "------------------------------------------------------------"
        )


        # ----------------------------------------------------
        # Validate sample name
        # ----------------------------------------------------

        if (
            signal_sample
            not in weight_maps_by_sample
        ):

            print(
                f"  Unknown sample: "
                f"{signal_sample}"
            )

            continue


        idx_to_name = (
            weight_maps_by_sample[
                signal_sample
            ]
        )


        requested_weights = (
            required_weights_by_sample[
                signal_sample
            ]
        )


        # ====================================================
        # Open input ROOT file
        # ====================================================

        input_path = (
            DATA_DIR
            / f"WCharm_{signal_sample}.root"
        )

        tfile = ROOT.TFile.Open(
            str(
                input_path
            ),
            "READ",
        )


        if (
            not tfile
            or tfile.IsZombie()
        ):

            print(
                "  Could not open ROOT file:"
            )

            print(
                f"    {input_path}"
            )

            continue


        # ====================================================
        # Retrieve WCharm tree
        # ====================================================

        tree = tfile.Get(
            "WCharmTree"
        )


        if not tree:

            print(
                "  Missing WCharmTree"
            )

            tfile.Close()

            continue


        # ====================================================
        # Number of generated events
        # ====================================================

        total_events = tree.GetEntries()


        if total_events <= 0:

            print(
                "  Tree has no entries"
            )

            tfile.Close()

            continue


        # ====================================================
        # Determine available weight-vector length
        # ====================================================

        tree.GetEntry(0)

        weight_vector_length = len(
            tree.weightvec
        )


        # ====================================================
        # Keep only weights that physically exist in this tree
        # ====================================================

        weights_to_run = [
            weight_index
            for weight_index
            in requested_weights
            if (
                weight_index
                in idx_to_name
                and
                weight_index
                < weight_vector_length
            )
        ]


        print(
            f"  Number of events: "
            f"{total_events}"
        )

        print(
            f"  Weight vector length: "
            f"{weight_vector_length}"
        )

        print(
            f"  Required usable weights: "
            f"{len(weights_to_run)}"
        )


        # ====================================================
        # Report weight categories
        # ====================================================

        nominal_found = [
            w
            for w in weights_to_run
            if w == NOMINAL_WEIGHT
        ]

        scale_found = [
            w
            for w in weights_to_run
            if w in SCALE_WEIGHTS
        ]

        shower_found = [
            w
            for w in weights_to_run
            if w in SHOWER_WEIGHTS
        ]

        model_found = [
            w
            for w in weights_to_run
            if w in MODEL_WEIGHTS
        ]

        pdf_found = [
            w
            for w in weights_to_run
            if (
                is_pdf_weight_name(
                    idx_to_name[w]
                )
                and
                w not in SCALE_WEIGHTS
                and
                w not in SHOWER_WEIGHTS
                and
                w not in MODEL_WEIGHTS
                and
                w not in EXCLUDED_PDF_WEIGHTS
            )
        ]


        print(
            f"  Nominal weights: "
            f"{nominal_found}"
        )

        print(
            f"  Scale weights: "
            f"{scale_found}"
        )

        print(
            f"  PDF weights: "
            f"{len(pdf_found)}"
        )

        print(
            f"  Shower weights: "
            f"{shower_found}"
        )

        print(
            f"  Model weights: "
            f"{model_found}"
        )


        # ====================================================
        # Nominal weight is mandatory
        # ====================================================

        if NOMINAL_WEIGHT not in weights_to_run:

            print(
                "  No usable nominal weight 0 found - "
                "skipping sample."
            )

            tfile.Close()

            continue


        # ====================================================
        # Create one eta histogram for every required weight
        # ====================================================

        histograms = {
            weight_index:
                make_etalepton_histogram(
                    signal_sample,
                    weight_index,
                )
            for weight_index in weights_to_run
        }


        # ====================================================
        # SINGLE event-tree pass
        # ====================================================

        (
            pos_eta_sums,
            neg_eta_sums,
            events_after_base_cuts,
            events_after_mtw_cut,
        ) = fill_all_required_etalepton_histograms(
            tree=tree,
            histograms=histograms,
            weights_to_run=weights_to_run,
            total_events=total_events,
        )


        print()

        print(
            f"  Events after original three cuts: "
            f"{events_after_base_cuts}"
        )

        print(
            f"  Events after m_T^W > "
            f"{MTW_MIN_GEV:.1f} GeV: "
            f"{events_after_mtw_cut}"
        )

        print()


        # ====================================================
        # Write one ROOT output file per required weight
        # ====================================================

        for weight_index in weights_to_run:

            weight_name = (
                idx_to_name[
                    weight_index
                ]
            )

            print(
                f"    Writing weight "
                f"{weight_index}: "
                f"{weight_name}"
            )


            write_output_file(
                signal_sample=(
                    signal_sample
                ),

                weight_index=(
                    weight_index
                ),

                weight_name=(
                    weight_name
                ),

                weight_vector_length=(
                    weight_vector_length
                ),

                hist=(
                    histograms[
                        weight_index
                    ]
                ),

                pos_eta_sum=(
                    pos_eta_sums[
                        weight_index
                    ]
                ),

                neg_eta_sum=(
                    neg_eta_sums[
                        weight_index
                    ]
                ),

                events_after_base_cuts=(
                    events_after_base_cuts
                ),

                events_after_mtw_cut=(
                    events_after_mtw_cut
                ),
            )


        # ====================================================
        # Finished with sample
        # ====================================================

        tfile.Close()


    # ========================================================
    # Finished
    # ========================================================

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
        "Generated ONLY lepton |eta| histograms with:"
    )

    print(
        f"  m_T^W > {MTW_MIN_GEV:.1f} GeV"
    )

    print()

    print(
        "Only uncertainty weights required downstream were produced."
    )

    print()

    print(
        "Output directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()
