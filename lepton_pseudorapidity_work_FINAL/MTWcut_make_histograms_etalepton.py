#!/usr/bin/env python3

"""
MTWcut_make_histograms_etalepton.py
===================================

Replacement for the lepton-|eta| histogram production used by the
Strange Proton uncertainty-decomposition analysis.

The important repair in this version is that the event selection no longer
stops at the W cuts.  It also requires the W+c jet definition used by the
validated ATLAS W+c analysis:

    MET > 25 GeV
    lepton pT > 20 GeV
    |eta_lepton| < 2.5
    mT(W) > 40 GeV
    jet pT > 25 GeV
    |eta_jet| < 2.5
    exactly one charm-identified fiducial jet

The input ROOT ntuples already contain a stored jet collection.  This script
therefore applies the fiducial requirements to that collection rather than
reclustering particles.

CHARM-JET ENCODING
------------------
The project ntuples store one ``jet_charge`` entry for every stored jet.
The checked WCharmTree content shows values -1, 0 and +1, aligned with
``jet_pt`` and ``jet_eta``.  A non-zero value identifies a charm jet and its
sign provides the charm/anticharm sign needed for the ATLAS OS-SS subtraction.
The scalar ``wcharm_charge`` branch is not used for this purpose.
"""

from array import array
from pathlib import Path
import math
import sys

import ROOT


# ============================================================
# ROOT settings
# ============================================================

ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)


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


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
WEIGHT_DIR = PROJECT_DIR / "weights"
OUTPUT_DIR = SCRIPT_DIR / "MTWcut_make_histograms_etalepton_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Observable binning
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

SIGNAL_SAMPLES = (
    sys.argv[1:]
    if len(sys.argv) > 1
    else SIGNAL_SAMPLES_DEFAULT
)

SAMPLE_PAIRS = {
    "Pythia_plus": ("WCPy8plus", "WCPyPartonplus"),
    "Pythia_minus": ("WCPy8minus", "WCPyPartonminus"),
    "Herwig_plus": ("WCH7plus", "WCHPartonplus"),
    "Herwig_minus": ("WCH7minus", "WCHPartonminus"),
}


# ============================================================
# Uncertainty-weight definitions
# ============================================================

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

MODEL_WEIGHTS = [292, 293, 314, 315]
SHOWER_WEIGHTS = list(range(294, 314)) + [316, 317]
EXCLUDED_WEIGHTS = {318}
EXCLUDED_PDF_WEIGHTS = {281}


# ============================================================
# Weight helpers
# ============================================================

def load_weight_map_clean(path):
    idx_to_name = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            idx_str, name = line.split(None, 1)
            idx_to_name[int(idx_str)] = name.strip()

    return idx_to_name


def mapping_file_for_sample(sample):
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

    raise RuntimeError(f"No weight-mapping rule for sample '{sample}'")


def safe_name(text):
    return "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in text
    )


def is_pdf_weight_name(weight_name):
    return weight_name.startswith("MUR1.0_MUF1.0_PDF")


def required_weights_for_pair(particle_weight_map, parton_weight_map):
    common_indices = (
        set(particle_weight_map)
        & set(parton_weight_map)
    )
    common_indices -= EXCLUDED_WEIGHTS

    required = set()

    if NOMINAL_WEIGHT in common_indices:
        required.add(NOMINAL_WEIGHT)

    required.update(
        index for index in SCALE_WEIGHTS
        if index in common_indices
    )

    required.update(
        index for index in SHOWER_WEIGHTS
        if index in common_indices
    )

    required.update(
        index for index in MODEL_WEIGHTS
        if index in common_indices
    )

    for index in sorted(common_indices):
        particle_name = particle_weight_map[index]

        if not is_pdf_weight_name(particle_name):
            continue
        if index in SCALE_WEIGHTS:
            continue
        if index in MODEL_WEIGHTS:
            continue
        if index in SHOWER_WEIGHTS:
            continue
        if index in EXCLUDED_PDF_WEIGHTS:
            continue

        required.add(index)

    return sorted(required)


def build_required_weight_table():
    maps_by_sample = {
        sample: load_weight_map_clean(mapping_file_for_sample(sample))
        for sample in SIGNAL_SAMPLES_DEFAULT
    }

    required_by_sample = {}

    for particle_sample, parton_sample in SAMPLE_PAIRS.values():
        pair_required = required_weights_for_pair(
            maps_by_sample[particle_sample],
            maps_by_sample[parton_sample],
        )

        required_by_sample[particle_sample] = pair_required
        required_by_sample[parton_sample] = pair_required

    return required_by_sample, maps_by_sample


def weight_category(weight_index, weight_name):
    if weight_index == NOMINAL_WEIGHT:
        return "nominal"
    if weight_index in SCALE_WEIGHTS:
        return "scale"
    if weight_index in SHOWER_WEIGHTS:
        return "shower"
    if weight_index in MODEL_WEIGHTS:
        return "model"
    if (
        is_pdf_weight_name(weight_name)
        and weight_index not in EXCLUDED_PDF_WEIGHTS
    ):
        return "pdf"
    return "other"


# ============================================================
# Tree / branch helpers
# ============================================================

def tree_branch_names(tree):
    return {
        branch.GetName()
        for branch in tree.GetListOfBranches()
    }


def validate_base_branches(tree):
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

    names = tree_branch_names(tree)
    missing = sorted(required - names)

    if missing:
        raise RuntimeError(
            "WCharmTree is missing required branches: "
            + ", ".join(missing)
        )


# The ntuples used by this project store the charm-jet identification
# directly in the per-jet ``jet_charge`` vector:
#
#   jet_charge == 0   -> no matched charm hadron
#   jet_charge > 0    -> charm-tagged jet with positive charm sign
#   jet_charge < 0    -> charm-tagged jet with negative charm sign
#
# This was verified against the actual WCharmTree schema.  The vector is
# aligned element-by-element with jet_pt and jet_eta.  The scalar
# ``wcharm_charge`` branch is deliberately NOT used for jet selection.
#
# The sign is also what is needed for the ATLAS OS-SS convention:
# opposite-sign W/charm events have weight +1 and same-sign events -1.
JET_CHARGE_BRANCH = "jet_charge"


def sequence_length(value):
    try:
        return len(value)
    except TypeError:
        return 1


def sequence_value(value, index):
    """Return element ``index`` from a ROOT vector/array-like branch."""
    try:
        return value[index]
    except TypeError:
        if index != 0:
            raise IndexError(index)
        return value


def validate_jet_charge_encoding(tree, entries_to_check=200):
    """
    Sanity-check the stored jet collection before producing outputs.

    We require jet_pt, jet_eta and jet_charge to have identical lengths and
    jet_charge values to be compatible with the observed {-1, 0, +1}
    convention.  Failing loudly is safer than silently misidentifying jets.
    """
    ncheck = min(int(tree.GetEntries()), int(entries_to_check))

    for entry_index in range(ncheck):
        tree.GetEntry(entry_index)

        n_pt = sequence_length(tree.jet_pt)
        n_eta = sequence_length(tree.jet_eta)
        n_charge = sequence_length(tree.jet_charge)

        if not (n_pt == n_eta == n_charge):
            raise RuntimeError(
                "Per-jet branch-length mismatch at entry "
                f"{entry_index}: len(jet_pt)={n_pt}, "
                f"len(jet_eta)={n_eta}, len(jet_charge)={n_charge}."
            )

        for jet_index in range(n_charge):
            value = float(sequence_value(tree.jet_charge, jet_index))
            if value not in (-1.0, 0.0, 1.0):
                raise RuntimeError(
                    "Unexpected jet_charge encoding at entry "
                    f"{entry_index}, jet {jet_index}: {value}. "
                    "Expected -1, 0 or +1."
                )

# ============================================================
# Physics-selection helpers
# ============================================================

def calculate_mtw(tree):
    delta_phi = tree.leptons_phi - tree.met_phi

    mtw_squared = (
        2.0
        * tree.leptons_pt
        * tree.met_et
        * (1.0 - math.cos(delta_phi))
    )

    return math.sqrt(max(0.0, mtw_squared))


def fiducial_charm_jets(tree):
    """
    Return the charm-identified jets inside the W+c fiducial region.

    Each returned item is ``(jet_index, charm_sign)``.  Extra fiducial
    non-charm jets are allowed; the analysis requires exactly one *charm* jet.
    """
    jet_pts = tree.jet_pt
    jet_etas = tree.jet_eta
    jet_charges = tree.jet_charge

    n_pt = sequence_length(jet_pts)
    n_eta = sequence_length(jet_etas)
    n_charge = sequence_length(jet_charges)

    if not (n_pt == n_eta == n_charge):
        raise RuntimeError(
            "Per-jet branch-length mismatch: "
            f"len(jet_pt)={n_pt}, len(jet_eta)={n_eta}, "
            f"len(jet_charge)={n_charge}."
        )

    selected = []

    for jet_index in range(n_pt):
        jet_pt = float(sequence_value(jet_pts, jet_index))
        jet_eta = float(sequence_value(jet_etas, jet_index))
        jet_charge = float(sequence_value(jet_charges, jet_index))

        if jet_pt <= JET_PT_MIN_GEV:
            continue

        if abs(jet_eta) >= JET_ABS_ETA_MAX:
            continue

        # jet_charge == 0 denotes no charm match in these ntuples.
        if jet_charge == 0.0:
            continue

        # Keep only the sign; the current ntuples encode it as +/-1.
        charm_sign = 1.0 if jet_charge > 0.0 else -1.0
        selected.append((jet_index, charm_sign))

    return selected


def passes_w_selection(tree):
    if tree.met_et <= MET_MIN_GEV:
        return False

    if tree.leptons_pt <= LEPTON_PT_MIN_GEV:
        return False

    if abs(tree.leptons_eta) >= LEPTON_ABS_ETA_MAX:
        return False

    if calculate_mtw(tree) <= MTW_MIN_GEV:
        return False

    return True


def os_ss_charge_weight(lepton_charge, charm_sign):
    """
    ATLAS W+c OS-SS sign:

      opposite sign -> +1
      same sign     -> -1

    This is equivalent to the validated Rivet implementation.
    """
    product = float(lepton_charge) * float(charm_sign)
    return -1.0 if product > 0.0 else 1.0


def passes_wcjet_selection(tree):
    if not passes_w_selection(tree):
        return False

    return len(fiducial_charm_jets(tree)) == REQUIRED_CHARM_JETS


# ============================================================
# Histogram creation
# ============================================================

def make_etalepton_histogram(sample, weight_index):
    hist = ROOT.TH1F(
        f"etalepton_{sample}_w{weight_index}",
        "etalepton",
        len(ETA_LEPTON_BIN_ARRAY) - 1,
        ETA_LEPTON_BIN_ARRAY,
    )

    hist.Sumw2()
    hist.SetDirectory(0)
    return hist


# ============================================================
# Single-pass event processing
# ============================================================

def fill_all_required_etalepton_histograms(
    tree,
    histograms,
    weights_to_run,
    total_events,
):
    pos_eta_sums = {index: 0.0 for index in weights_to_run}
    neg_eta_sums = {index: 0.0 for index in weights_to_run}

    events_after_lepton_met_cuts = 0
    events_after_mtw_cut = 0
    events_after_wcjet_selection = 0
    opposite_sign_events = 0
    same_sign_events = 0

    nentries = tree.GetEntries()

    for entry_index in range(nentries):
        tree.GetEntry(entry_index)

        # W preselection diagnostics.
        if tree.met_et <= MET_MIN_GEV:
            continue
        if tree.leptons_pt <= LEPTON_PT_MIN_GEV:
            continue
        if abs(tree.leptons_eta) >= LEPTON_ABS_ETA_MAX:
            continue

        events_after_lepton_met_cuts += 1

        if calculate_mtw(tree) <= MTW_MIN_GEV:
            continue

        events_after_mtw_cut += 1

        # FIX: restore the W+c jet requirement before filling |eta_l|.
        charm_jets = fiducial_charm_jets(tree)

        if len(charm_jets) != REQUIRED_CHARM_JETS:
            continue

        events_after_wcjet_selection += 1

        # Exactly one charm jet survives.  Its stored sign supplies the
        # OS-SS subtraction used by the ATLAS W+c definition.
        _, charm_sign = charm_jets[0]
        charge_factor = os_ss_charge_weight(
            tree.leptons_charge,
            charm_sign,
        )

        if charge_factor > 0.0:
            opposite_sign_events += 1
        else:
            same_sign_events += 1

        eta_lepton = abs(float(tree.leptons_eta))

        for weight_index in weights_to_run:
            event_weight = (
                float(tree.weightvec[weight_index])
                / float(total_events)
            )

            final_weight = charge_factor * event_weight

            if tree.leptons_eta >= 0:
                pos_eta_sums[weight_index] += final_weight
            else:
                neg_eta_sums[weight_index] += final_weight

            histograms[weight_index].Fill(
                eta_lepton,
                final_weight,
            )

    return {
        "pos_eta_sums": pos_eta_sums,
        "neg_eta_sums": neg_eta_sums,
        "events_after_lepton_met_cuts": events_after_lepton_met_cuts,
        "events_after_mtw_cut": events_after_mtw_cut,
        "events_after_wcjet_selection": events_after_wcjet_selection,
        "opposite_sign_events": opposite_sign_events,
        "same_sign_events": same_sign_events,
    }


# ============================================================
# Output
# ============================================================

def write_output_file(
    signal_sample,
    weight_index,
    weight_name,
    weight_vector_length,
    hist,
    diagnostics,
):
    output_name = (
        f"output_w{weight_index}_"
        f"{safe_name(weight_name)}_"
        f"{signal_sample}.root"
    )

    output_path = OUTPUT_DIR / output_name
    output_file = ROOT.TFile(str(output_path), "RECREATE")

    if not output_file or output_file.IsZombie():
        raise RuntimeError(f"Could not create output ROOT file: {output_path}")

    output_file.cd()

    ROOT.TNamed("RequestedWeightIndex", str(weight_index)).Write()
    ROOT.TNamed("EffectiveWeightIndex", str(weight_index)).Write()
    ROOT.TNamed("EffectiveWeightName", weight_name).Write()
    ROOT.TNamed("WeightVecLength", str(weight_vector_length)).Write()
    ROOT.TNamed("SampleName", signal_sample).Write()
    ROOT.TNamed("Observable", "etalepton").Write()
    ROOT.TNamed("WeightCategory", weight_category(weight_index, weight_name)).Write()

    ROOT.TNamed("MTWCutApplied", "True").Write()
    ROOT.TNamed("MTWCutDefinition", "m_T^W > 40 GeV").Write()
    ROOT.TNamed("MTWCutGeV", str(MTW_MIN_GEV)).Write()

    ROOT.TNamed("JetSelectionApplied", "True").Write()
    ROOT.TNamed("JetPtMinGeV", str(JET_PT_MIN_GEV)).Write()
    ROOT.TNamed("JetAbsEtaMax", str(JET_ABS_ETA_MAX)).Write()
    ROOT.TNamed("RequiredCharmJets", str(REQUIRED_CHARM_JETS)).Write()
    ROOT.TNamed("CharmJetIDBranch", JET_CHARGE_BRANCH).Write()
    ROOT.TNamed(
        "CharmJetIDDefinition",
        "jet_charge != 0; sign gives charm/anticharm for OS-SS",
    ).Write()
    ROOT.TNamed(
        "OSSSWeightDefinition",
        "+1 for opposite-sign W/charm; -1 for same-sign W/charm",
    ).Write()

    ROOT.TNamed(
        "EventSelection",
        (
            "met_et > 25 GeV; "
            "leptons_pt > 20 GeV; "
            "abs(leptons_eta) < 2.5; "
            "m_T^W > 40 GeV; "
            "jet_pt > 25 GeV; "
            "abs(jet_eta) < 2.5; "
            "exactly one charm-identified fiducial jet (jet_charge != 0); "
            "OS-SS event sign from lepton_charge * jet_charge"
        ),
    ).Write()

    ROOT.TNamed(
        "EventsAfterBaseCuts",
        str(diagnostics["events_after_lepton_met_cuts"]),
    ).Write()

    ROOT.TNamed(
        "EventsAfterMTWCut",
        str(diagnostics["events_after_mtw_cut"]),
    ).Write()

    ROOT.TNamed(
        "EventsAfterWCJetSelection",
        str(diagnostics["events_after_wcjet_selection"]),
    ).Write()

    ROOT.TNamed(
        "OppositeSignEvents",
        str(diagnostics["opposite_sign_events"]),
    ).Write()

    ROOT.TNamed(
        "SameSignEvents",
        str(diagnostics["same_sign_events"]),
    ).Write()

    ROOT.TParameter(float)(
        "PosEtaWeightSum",
        diagnostics["pos_eta_sums"][weight_index],
    ).Write()

    ROOT.TParameter(float)(
        "NegEtaWeightSum",
        diagnostics["neg_eta_sums"][weight_index],
    ).Write()

    hist.Write("etalepton")

    output_file.Write()
    output_file.Close()


# ============================================================
# Main
# ============================================================

def main():
    print()
    print("============================================================")
    print(" W+c lepton |eta| histogram production")
    print("============================================================")
    print()
    print("Selection:")
    print(f"  MET > {MET_MIN_GEV:.1f} GeV")
    print(f"  lepton pT > {LEPTON_PT_MIN_GEV:.1f} GeV")
    print(f"  |eta_lepton| < {LEPTON_ABS_ETA_MAX:.1f}")
    print(f"  m_T^W > {MTW_MIN_GEV:.1f} GeV")
    print(f"  jet pT > {JET_PT_MIN_GEV:.1f} GeV")
    print(f"  |eta_jet| < {JET_ABS_ETA_MAX:.1f}")
    print(f"  exactly {REQUIRED_CHARM_JETS} charm-identified fiducial jet")
    print("  charm ID: jet_charge != 0")
    print("  OS-SS: opposite sign +1, same sign -1")
    print()
    print(f"Data directory:   {DATA_DIR}")
    print(f"Weight directory: {WEIGHT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    try:
        required_weights_by_sample, weight_maps_by_sample = (
            build_required_weight_table()
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed while building the required weight table"
        ) from exc

    for signal_sample in SIGNAL_SAMPLES:
        print()
        print("------------------------------------------------------------")
        print(f"Running sample: {signal_sample}")
        print("------------------------------------------------------------")

        if signal_sample not in weight_maps_by_sample:
            print(f"  Unknown sample: {signal_sample}")
            continue

        input_path = DATA_DIR / f"WCharm_{signal_sample}.root"
        root_file = ROOT.TFile.Open(str(input_path), "READ")

        if not root_file or root_file.IsZombie():
            print(f"  Could not open ROOT file: {input_path}")
            continue

        tree = root_file.Get("WCharmTree")
        if not tree:
            root_file.Close()
            print("  Missing WCharmTree")
            continue

        total_events = int(tree.GetEntries())
        if total_events <= 0:
            root_file.Close()
            print("  Tree has no entries")
            continue

        tree.GetEntry(0)
        validate_base_branches(tree)

        # Validate the actual ntuple encoding BEFORE creating any output.
        validate_jet_charge_encoding(tree)
        print(
            "  Charm-jet ID: jet_charge != 0 "
            "(sign used for OS-SS)"
        )

        weight_vector_length = len(tree.weightvec)
        idx_to_name = weight_maps_by_sample[signal_sample]
        requested_weights = required_weights_by_sample[signal_sample]

        weights_to_run = [
            index
            for index in requested_weights
            if (
                index in idx_to_name
                and index < weight_vector_length
            )
        ]

        if NOMINAL_WEIGHT not in weights_to_run:
            root_file.Close()
            print("  No usable nominal weight 0; skipping sample.")
            continue

        print(f"  Entries:             {total_events}")
        print(f"  Weight-vector size:  {weight_vector_length}")
        print(f"  Weights to produce:  {len(weights_to_run)}")

        histograms = {
            index: make_etalepton_histogram(signal_sample, index)
            for index in weights_to_run
        }

        diagnostics = fill_all_required_etalepton_histograms(
            tree=tree,
            histograms=histograms,
            weights_to_run=weights_to_run,
            total_events=total_events,
        )

        print(
            "  After lepton/MET cuts: "
            f"{diagnostics['events_after_lepton_met_cuts']}"
        )
        print(
            "  After mT cut:          "
            f"{diagnostics['events_after_mtw_cut']}"
        )
        print(
            "  After W+c jet cuts:    "
            f"{diagnostics['events_after_wcjet_selection']}"
        )
        print(
            "    opposite-sign:       "
            f"{diagnostics['opposite_sign_events']}"
        )
        print(
            "    same-sign:           "
            f"{diagnostics['same_sign_events']}"
        )

        for weight_index in weights_to_run:
            weight_name = idx_to_name[weight_index]

            write_output_file(
                signal_sample=signal_sample,
                weight_index=weight_index,
                weight_name=weight_name,
                weight_vector_length=weight_vector_length,
                hist=histograms[weight_index],
                diagnostics=diagnostics,
            )

        root_file.Close()

    print()
    print("============================================================")
    print(" Finished")
    print("============================================================")
    print(f"Outputs: {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()
