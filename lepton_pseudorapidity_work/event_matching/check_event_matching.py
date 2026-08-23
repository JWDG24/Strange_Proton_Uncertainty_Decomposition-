#!/usr/bin/env python3

# ============================================================
# check_event_matching.py
#
# Purpose:
#   Check whether the particle-level and parton-level ROOT
#   samples can be matched event-by-event using a common event
#   identifier.
#
# The script:
#   1. Finds all eight W+c ROOT files in the project data folder.
#   2. Inspects the TTrees and their branches.
#   3. Searches for plausible event/run identifier branches.
#   4. Tests the four physically corresponding particle/parton
#      pairs:
#
#        Pythia W+ : WCPy8plus      <-> WCPyPartonplus
#        Pythia W- : WCPy8minus     <-> WCPyPartonminus
#        Herwig W+ : WCH7plus       <-> WCHPartonplus
#        Herwig W- : WCH7minus      <-> WCHPartonminus
#
#   5. Checks:
#        - number of entries
#        - number of unique identifiers
#        - duplicate identifiers
#        - number of common identifiers
#        - matching fractions
#
#   6. If a single event identifier is not unique, it also tries
#      sensible compound identifiers such as:
#
#        (runNumber, eventNumber)
#
# Output:
#   A detailed text report is written to:
#
#       lepton_pseudorapidity_work/event_matching/
#       event_matching_report.txt
#
# Run from the project root with:
#
#   python3 lepton_pseudorapidity_work/event_matching/check_event_matching.py
#
# ============================================================

from pathlib import Path
from itertools import combinations
import math
import re

import ROOT


# ============================================================
# ROOT settings
# ============================================================

ROOT.gROOT.SetBatch(True)


# ============================================================
# Directory structure
# ============================================================

# This file lives in:
#
#   uncertainty_decomposition/
#   └── lepton_pseudorapidity_work/
#       └── event_matching/
#           └── check_event_matching.py
#
# Therefore the project root is two directories above this file.

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

REPORT_PATH = SCRIPT_DIR / "event_matching_report.txt"


# ============================================================
# Input ROOT files
# ============================================================

ROOT_FILES = {
    "WCPy8plus": DATA_DIR / "WCharm_WCPy8plus.root",
    "WCPyPartonplus": DATA_DIR / "WCharm_WCPyPartonplus.root",

    "WCPy8minus": DATA_DIR / "WCharm_WCPy8minus.root",
    "WCPyPartonminus": DATA_DIR / "WCharm_WCPyPartonminus.root",

    "WCH7plus": DATA_DIR / "WCharm_WCH7plus.root",
    "WCHPartonplus": DATA_DIR / "WCharm_WCHPartonplus.root",

    "WCH7minus": DATA_DIR / "WCharm_WCH7minus.root",
    "WCHPartonminus": DATA_DIR / "WCharm_WCHPartonminus.root",
}


# ============================================================
# Physically corresponding particle / parton pairs
# ============================================================

SAMPLE_PAIRS = {
    "Pythia W+": (
        "WCPy8plus",
        "WCPyPartonplus",
    ),

    "Pythia W-": (
        "WCPy8minus",
        "WCPyPartonminus",
    ),

    "Herwig W+": (
        "WCH7plus",
        "WCHPartonplus",
    ),

    "Herwig W-": (
        "WCH7minus",
        "WCHPartonminus",
    ),
}


# ============================================================
# Candidate event-identifier branch names
# ============================================================

# These words are used only to decide which branches are worth
# testing as possible event identifiers. The script still checks
# the actual values before deciding whether matching is possible.

IDENTIFIER_KEYWORDS = (
    "event",
    "evt",
    "run",
    "lumi",
    "lumiblock",
    "mcchannel",
    "channelnumber",
    "barcode",
    "identifier",
    "eventid",
    "eventnumber",
    "runnumber",
)


# Branch names that contain "id" are potentially useful, but
# "id" is too broad to use as a substring by itself. Exact/common
# forms are included here separately.

EXACT_ID_NAMES = {
    "id",
    "eventid",
    "eventnumber",
    "evt",
    "evtnumber",
    "runnumber",
    "run",
    "lumiblock",
    "lbn",
    "barcode",
    "mceventnumber",
    "mcchannelnumber",
}


# ============================================================
# Utility: write to terminal and report simultaneously
# ============================================================

REPORT_LINES = []


def report(line=""):
    print(line)
    REPORT_LINES.append(str(line))


# ============================================================
# Normalise branch names
# ============================================================

def normalise_name(name):
    """
    Convert a branch name into a simple comparison form.

    Examples:
        eventNumber  -> eventnumber
        Event_Number -> eventnumber
        event-number -> eventnumber
    """

    return re.sub(
        r"[^a-z0-9]",
        "",
        name.lower(),
    )


# ============================================================
# Check whether a branch name looks like an identifier
# ============================================================

def is_identifier_candidate(branch_name):
    normalised = normalise_name(branch_name)

    if normalised in EXACT_ID_NAMES:
        return True

    for keyword in IDENTIFIER_KEYWORDS:
        if keyword in normalised:
            return True

    return False


# ============================================================
# Find TTrees in a ROOT file
# ============================================================

def find_trees(root_file):
    """
    Return every TTree stored directly in the ROOT file.
    """

    trees = []

    for key in root_file.GetListOfKeys():
        obj = key.ReadObj()

        if obj and obj.InheritsFrom("TTree"):
            trees.append(obj)

    return trees


# ============================================================
# Select the most likely analysis tree
# ============================================================

def choose_tree(root_file, filename):
    """
    Prefer the TTree with the largest number of entries.

    This is a robust default for these ntuples if more than one
    small metadata tree happens to be present.
    """

    trees = find_trees(root_file)

    if not trees:
        raise RuntimeError(
            f"No TTree found in ROOT file:\n  {filename}"
        )

    return max(
        trees,
        key=lambda tree: int(tree.GetEntries()),
    )


# ============================================================
# Get branch names
# ============================================================

def get_branch_names(tree):
    return [
        branch.GetName()
        for branch in tree.GetListOfBranches()
    ]


# ============================================================
# Convert a branch value to a hashable scalar
# ============================================================

def scalarise_value(value):
    """
    Convert common PyROOT scalar types into Python scalar values.

    Return None for vector/container values because those should
    not be used as event identifiers.
    """

    # Strings
    if isinstance(value, str):
        return value

    # Native Python numeric scalars
    if isinstance(value, (int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    # Some PyROOT scalar proxy objects can be converted to int.
    try:
        converted = int(value)
        return converted
    except Exception:
        pass

    # Then try float as a fallback.
    try:
        converted = float(value)

        if math.isfinite(converted):
            return converted
    except Exception:
        pass

    # Vector-like / unsupported values are rejected.
    return None


# ============================================================
# Read one candidate identifier branch
# ============================================================

def read_identifier_values(tree, branch_name):
    """
    Read one scalar branch from every entry.

    Returns:
        values
        invalid_count
    """

    values = []
    invalid_count = 0

    nentries = int(tree.GetEntries())

    for ientry in range(nentries):
        tree.GetEntry(ientry)

        try:
            raw_value = getattr(
                tree,
                branch_name,
            )
        except Exception:
            invalid_count += 1
            continue

        value = scalarise_value(
            raw_value
        )

        if value is None:
            invalid_count += 1
            continue

        values.append(
            value
        )

    return values, invalid_count


# ============================================================
# Branch-pair matching by normalised name
# ============================================================

def find_common_candidate_branches(
    branches_a,
    branches_b,
):
    """
    Match candidate branches even when capitalization or
    underscores differ.

    Example:
        EventNumber <-> event_number
    """

    candidates_a = {
        normalise_name(name): name
        for name in branches_a
        if is_identifier_candidate(name)
    }

    candidates_b = {
        normalise_name(name): name
        for name in branches_b
        if is_identifier_candidate(name)
    }

    common_keys = sorted(
        set(candidates_a)
        & set(candidates_b)
    )

    return [
        (
            candidates_a[key],
            candidates_b[key],
        )
        for key in common_keys
    ]


# ============================================================
# Assess one identifier
# ============================================================

def assess_single_identifier(
    tree_a,
    tree_b,
    branch_a,
    branch_b,
):
    """
    Compare the values of one candidate identifier branch across
    particle and parton samples.
    """

    values_a, invalid_a = (
        read_identifier_values(
            tree_a,
            branch_a,
        )
    )

    values_b, invalid_b = (
        read_identifier_values(
            tree_b,
            branch_b,
        )
    )

    set_a = set(values_a)
    set_b = set(values_b)

    common = set_a & set_b

    unique_a = len(set_a)
    unique_b = len(set_b)

    duplicate_a = (
        len(values_a) - unique_a
    )

    duplicate_b = (
        len(values_b) - unique_b
    )

    overlap_smaller = (
        len(common)
        / min(unique_a, unique_b)
        if min(unique_a, unique_b) > 0
        else 0.0
    )

    overlap_a = (
        len(common) / unique_a
        if unique_a > 0
        else 0.0
    )

    overlap_b = (
        len(common) / unique_b
        if unique_b > 0
        else 0.0
    )

    fully_unique = (
        duplicate_a == 0
        and duplicate_b == 0
        and invalid_a == 0
        and invalid_b == 0
    )

    return {
        "branch_a": branch_a,
        "branch_b": branch_b,
        "values_a": values_a,
        "values_b": values_b,
        "unique_a": unique_a,
        "unique_b": unique_b,
        "duplicates_a": duplicate_a,
        "duplicates_b": duplicate_b,
        "invalid_a": invalid_a,
        "invalid_b": invalid_b,
        "common": len(common),
        "overlap_a": overlap_a,
        "overlap_b": overlap_b,
        "overlap_smaller": overlap_smaller,
        "fully_unique": fully_unique,
    }


# ============================================================
# Assess a compound identifier
# ============================================================

def assess_compound_identifier(
    tree_a,
    tree_b,
    branch_pairs,
):
    """
    Test a compound event key, for example:

        (runNumber, eventNumber)

    branch_pairs contains:
        [(branch_a_1, branch_b_1),
         (branch_a_2, branch_b_2)]
    """

    values_by_branch_a = []
    values_by_branch_b = []

    for branch_a, branch_b in branch_pairs:

        values_a, invalid_a = (
            read_identifier_values(
                tree_a,
                branch_a,
            )
        )

        values_b, invalid_b = (
            read_identifier_values(
                tree_b,
                branch_b,
            )
        )

        if invalid_a != 0 or invalid_b != 0:
            return None

        values_by_branch_a.append(
            values_a
        )

        values_by_branch_b.append(
            values_b
        )

    lengths_a = {
        len(values)
        for values in values_by_branch_a
    }

    lengths_b = {
        len(values)
        for values in values_by_branch_b
    }

    if (
        len(lengths_a) != 1
        or len(lengths_b) != 1
    ):
        return None

    keys_a = list(
        zip(*values_by_branch_a)
    )

    keys_b = list(
        zip(*values_by_branch_b)
    )

    set_a = set(keys_a)
    set_b = set(keys_b)

    common = set_a & set_b

    unique_a = len(set_a)
    unique_b = len(set_b)

    duplicate_a = (
        len(keys_a) - unique_a
    )

    duplicate_b = (
        len(keys_b) - unique_b
    )

    overlap_smaller = (
        len(common)
        / min(unique_a, unique_b)
        if min(unique_a, unique_b) > 0
        else 0.0
    )

    return {
        "branch_pairs": branch_pairs,
        "unique_a": unique_a,
        "unique_b": unique_b,
        "duplicates_a": duplicate_a,
        "duplicates_b": duplicate_b,
        "common": len(common),
        "overlap_smaller": overlap_smaller,
        "fully_unique": (
            duplicate_a == 0
            and duplicate_b == 0
        ),
    }


# ============================================================
# Human-readable assessment
# ============================================================

def classify_result(
    overlap_fraction,
    fully_unique,
):
    """
    Classification is deliberately conservative.

    A high overlap does not by itself prove that two records
    represent the same physical event, but a unique common event
    identifier with near-complete overlap is strong evidence.
    """

    if fully_unique and overlap_fraction >= 0.99:
        return "STRONG EVIDENCE OF EVENT-BY-EVENT MATCHING"

    if fully_unique and overlap_fraction >= 0.90:
        return "LIKELY MATCHABLE, WITH SOME UNMATCHED EVENTS"

    if overlap_fraction >= 0.50:
        return "PARTIAL IDENTIFIER OVERLAP - INVESTIGATE"

    if overlap_fraction > 0.0:
        return "SMALL IDENTIFIER OVERLAP - NOT ENOUGH TO CLAIM MATCHING"

    return "NO IDENTIFIER OVERLAP"


# ============================================================
# Inspect one ROOT file
# ============================================================

def inspect_root_file(
    sample_name,
    path,
):
    """
    Open one ROOT file and report its main TTree and plausible
    identifier branches.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing ROOT file:\n  {path}"
        )

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

    tree = choose_tree(
        root_file,
        path,
    )

    branches = get_branch_names(
        tree
    )

    candidates = [
        name
        for name in branches
        if is_identifier_candidate(name)
    ]

    report(f"Sample: {sample_name}")
    report(f"  File:    {path}")
    report(f"  Tree:    {tree.GetName()}")
    report(f"  Entries: {int(tree.GetEntries()):,}")

    if candidates:
        report(
            "  Candidate ID branches: "
            + ", ".join(candidates)
        )
    else:
        report(
            "  Candidate ID branches: NONE FOUND"
        )

    report()

    return {
        "file": root_file,
        "tree": tree,
        "branches": branches,
        "candidates": candidates,
    }


# ============================================================
# Compare one particle / parton pair
# ============================================================

def compare_pair(
    pair_label,
    particle_name,
    parton_name,
    inspected,
):
    """
    Test all sensible common identifier branches for one
    physically corresponding particle/parton pair.
    """

    particle = inspected[
        particle_name
    ]

    parton = inspected[
        parton_name
    ]

    tree_particle = particle[
        "tree"
    ]

    tree_parton = parton[
        "tree"
    ]

    report("=" * 72)
    report(f"PAIR: {pair_label}")
    report("=" * 72)

    report(
        f"Particle: {particle_name} "
        f"({int(tree_particle.GetEntries()):,} entries)"
    )

    report(
        f"Parton:   {parton_name} "
        f"({int(tree_parton.GetEntries()):,} entries)"
    )

    report()

    common_candidates = (
        find_common_candidate_branches(
            particle["branches"],
            parton["branches"],
        )
    )

    if not common_candidates:

        report(
            "No common event/run identifier-like branch names "
            "were found."
        )

        report(
            "Result: EVENT MATCHING CANNOT BE ESTABLISHED "
            "FROM THE AVAILABLE IDENTIFIER-LIKE BRANCHES."
        )

        report()
        return


    report("Common candidate identifier branches:")

    for branch_particle, branch_parton in common_candidates:
        report(
            f"  {branch_particle}  <->  {branch_parton}"
        )

    report()


    # --------------------------------------------------------
    # Test every single identifier branch
    # --------------------------------------------------------

    single_results = []

    for branch_particle, branch_parton in common_candidates:

        report(
            f"Testing identifier: "
            f"{branch_particle} <-> {branch_parton}"
        )

        result = assess_single_identifier(
            tree_particle,
            tree_parton,
            branch_particle,
            branch_parton,
        )

        single_results.append(
            result
        )

        report(
            f"  Particle unique IDs: "
            f"{result['unique_a']:,}"
        )

        report(
            f"  Parton unique IDs:   "
            f"{result['unique_b']:,}"
        )

        report(
            f"  Particle duplicates: "
            f"{result['duplicates_a']:,}"
        )

        report(
            f"  Parton duplicates:   "
            f"{result['duplicates_b']:,}"
        )

        if (
            result["invalid_a"] > 0
            or result["invalid_b"] > 0
        ):
            report(
                f"  Invalid/non-scalar values: "
                f"particle={result['invalid_a']:,}, "
                f"parton={result['invalid_b']:,}"
            )

        report(
            f"  Common unique IDs:   "
            f"{result['common']:,}"
        )

        report(
            f"  Particle matched:    "
            f"{100.0 * result['overlap_a']:.2f}%"
        )

        report(
            f"  Parton matched:      "
            f"{100.0 * result['overlap_b']:.2f}%"
        )

        classification = classify_result(
            result["overlap_smaller"],
            result["fully_unique"],
        )

        report(
            f"  Assessment: {classification}"
        )

        report()


    # --------------------------------------------------------
    # Pick best single identifier
    # --------------------------------------------------------

    best_single = max(
        single_results,
        key=lambda result: (
            result["fully_unique"],
            result["overlap_smaller"],
            result["common"],
        ),
    )

    if (
        best_single["fully_unique"]
        and best_single["overlap_smaller"] >= 0.99
    ):

        report(
            "BEST RESULT:"
        )

        report(
            f"  A single unique identifier appears sufficient: "
            f"{best_single['branch_a']} <-> "
            f"{best_single['branch_b']}"
        )

        report(
            f"  Match fraction of smaller unique set: "
            f"{100.0 * best_single['overlap_smaller']:.2f}%"
        )

        report(
            "  This is strong evidence that the two samples "
            "can be matched event-by-event."
        )

        report()
        return


    # --------------------------------------------------------
    # If single identifiers are duplicated, try two-field keys
    # --------------------------------------------------------

    if len(common_candidates) >= 2:

        report(
            "No single branch gave a near-complete unique match."
        )

        report(
            "Trying two-branch compound identifiers..."
        )

        report()

        compound_results = []

        # Limit this to candidate pairs. Normally there are only
        # a handful of event/run-like branches.
        for branch_pair_combo in combinations(
            common_candidates,
            2,
        ):

            result = assess_compound_identifier(
                tree_particle,
                tree_parton,
                list(branch_pair_combo),
            )

            if result is None:
                continue

            compound_results.append(
                result
            )

            particle_names = [
                pair[0]
                for pair in branch_pair_combo
            ]

            parton_names = [
                pair[1]
                for pair in branch_pair_combo
            ]

            report(
                "Testing compound key:"
            )

            report(
                "  Particle: ("
                + ", ".join(particle_names)
                + ")"
            )

            report(
                "  Parton:   ("
                + ", ".join(parton_names)
                + ")"
            )

            report(
                f"  Particle unique keys: "
                f"{result['unique_a']:,}"
            )

            report(
                f"  Parton unique keys:   "
                f"{result['unique_b']:,}"
            )

            report(
                f"  Common unique keys:   "
                f"{result['common']:,}"
            )

            report(
                f"  Match fraction:       "
                f"{100.0 * result['overlap_smaller']:.2f}%"
            )

            classification = classify_result(
                result["overlap_smaller"],
                result["fully_unique"],
            )

            report(
                f"  Assessment: {classification}"
            )

            report()


        if compound_results:

            best_compound = max(
                compound_results,
                key=lambda result: (
                    result["fully_unique"],
                    result["overlap_smaller"],
                    result["common"],
                ),
            )

            best_names_particle = [
                pair[0]
                for pair in best_compound[
                    "branch_pairs"
                ]
            ]

            best_names_parton = [
                pair[1]
                for pair in best_compound[
                    "branch_pairs"
                ]
            ]

            report("BEST COMPOUND RESULT:")

            report(
                "  Particle key: ("
                + ", ".join(best_names_particle)
                + ")"
            )

            report(
                "  Parton key:   ("
                + ", ".join(best_names_parton)
                + ")"
            )

            report(
                f"  Match fraction: "
                f"{100.0 * best_compound['overlap_smaller']:.2f}%"
            )

            report(
                "  "
                + classify_result(
                    best_compound[
                        "overlap_smaller"
                    ],
                    best_compound[
                        "fully_unique"
                    ],
                )
            )

            report()
            return


    # --------------------------------------------------------
    # Final fallback summary
    # --------------------------------------------------------

    report("BEST RESULT:")

    report(
        f"  {best_single['branch_a']} <-> "
        f"{best_single['branch_b']}"
    )

    report(
        f"  Match fraction of smaller unique set: "
        f"{100.0 * best_single['overlap_smaller']:.2f}%"
    )

    report(
        "  "
        + classify_result(
            best_single[
                "overlap_smaller"
            ],
            best_single[
                "fully_unique"
            ],
        )
    )

    report()


# ============================================================
# Main
# ============================================================

def main():

    report()
    report("=" * 72)
    report(" W+c particle / parton event-matching check")
    report("=" * 72)
    report()

    report("Project root:")
    report(f"  {PROJECT_ROOT}")
    report()

    report("Data directory:")
    report(f"  {DATA_DIR}")
    report()

    report(
        "This script checks all eight ROOT files and then tests "
        "the four physically corresponding particle/parton pairs."
    )

    report()


    # --------------------------------------------------------
    # Check all expected ROOT files exist first
    # --------------------------------------------------------

    missing_files = [
        path
        for path in ROOT_FILES.values()
        if not path.is_file()
    ]

    if missing_files:

        report("ERROR: Missing ROOT file(s):")

        for path in missing_files:
            report(f"  {path}")

        report()

        raise FileNotFoundError(
            "One or more expected ROOT files are missing."
        )


    # --------------------------------------------------------
    # Inspect all eight ROOT files
    # --------------------------------------------------------

    report("=" * 72)
    report(" INSPECTING ALL ROOT FILES")
    report("=" * 72)
    report()

    inspected = {}

    for sample_name, path in ROOT_FILES.items():

        inspected[
            sample_name
        ] = inspect_root_file(
            sample_name,
            path,
        )


    # --------------------------------------------------------
    # Compare all physically corresponding pairs
    # --------------------------------------------------------

    report()
    report("=" * 72)
    report(" TESTING PARTICLE / PARTON MATCHING")
    report("=" * 72)
    report()

    for (
        pair_label,
        (
            particle_name,
            parton_name,
        ),
    ) in SAMPLE_PAIRS.items():

        compare_pair(
            pair_label,
            particle_name,
            parton_name,
            inspected,
        )


    # --------------------------------------------------------
    # Close ROOT files
    # --------------------------------------------------------

    for info in inspected.values():
        info["file"].Close()


    # --------------------------------------------------------
    # Write detailed report
    # --------------------------------------------------------

    report()
    report("=" * 72)
    report(" FINISHED")
    report("=" * 72)
    report()

    report(
        "Important: a common event identifier with near-complete "
        "unique overlap is strong evidence of event matching. "
        "Equal entry counts alone are NOT sufficient evidence."
    )

    report()

    # Add the report path to the in-memory report before writing.
    report(
        f"Detailed report written to:\n  {REPORT_PATH}"
    )

    report()

    REPORT_PATH.write_text(
        "\n".join(REPORT_LINES) + "\n",
        encoding="utf-8",
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
