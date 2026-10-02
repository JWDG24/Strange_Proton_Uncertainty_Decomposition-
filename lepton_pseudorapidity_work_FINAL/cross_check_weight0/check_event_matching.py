#!/usr/bin/env python3

"""
check_event_matching.py

Diagnostic only: determine whether the NEW high-statistics Pythia
particle- and parton-level ROOT fragments contain enough information
to match the same underlying events one-to-one.

This script DOES NOT modify any analysis output.

What it checks
--------------
1. Pairs particle/parton fragments by their numeric filename suffix.
2. Compares the WCharmTree branch schemas.
3. Searches for plausible common event-identifier branches such as:
      eventNumber, event_number, mcEventNumber, event_id,
      runNumber, lumiBlock, barcode, seed, ...
4. Also scans common scalar integer branches for high-cardinality
   quantities that could be an event identifier even if their name is
   unfamiliar.
5. If a plausible event ID is found, it tests:
      - uniqueness on the particle side
      - uniqueness on the parton side
      - particle/parton ID overlap
      - same-entry alignment within corresponding fragments
6. Prints a conservative verdict.

Important
---------
A matching filename suffix or equal ROOT entry number is NOT by itself
treated as proof that two events correspond.

If no unique common event identifier exists, the script will say that
event-by-event matching cannot be established from these ntuples.

Expected location
-----------------
uncertainty_decomposition/
└── lepton_pseudorapidity_work_FINAL/
    └── cross_check_weight0/
        ├── check_event_matching.py
        └── cross_check_ROOT_files/
            ├── WCharm_WCPy8plus10.root
            ├── WCharm_WCPyPartonplus10.root
            └── ...
"""

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import ROOT


ROOT.gROOT.SetBatch(True)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR / "cross_check_ROOT_files"
REPORT_PATH = SCRIPT_DIR / "event_matching_report.txt"

TREE_NAME = "WCharmTree"

CHANNELS = {
    "plus": {
        "particle_prefix": "WCharm_WCPy8plus",
        "parton_prefix": "WCharm_WCPyPartonplus",
    },
    "minus": {
        "particle_prefix": "WCharm_WCPy8minus",
        "parton_prefix": "WCharm_WCPyPartonminus",
    },
}


# Common names used by ATLAS/ROOT/MC ntuples.
KNOWN_ID_NAMES = [
    "eventNumber",
    "event_number",
    "EventNumber",
    "event",
    "evt",
    "event_id",
    "eventID",
    "mcEventNumber",
    "mc_event_number",
    "MCEventNumber",
    "runNumber",
    "run_number",
    "RunNumber",
    "lumiBlock",
    "lumi_block",
    "luminosityBlock",
    "lb",
    "mcChannelNumber",
    "mc_channel_number",
    "channelNumber",
    "channel_number",
    "barcode",
    "eventBarcode",
    "event_barcode",
    "seed",
    "randomSeed",
    "random_seed",
    "eventIndex",
    "event_index",
    "entry_id",
]


INTEGER_LEAF_TYPES = {
    "Char_t",
    "UChar_t",
    "Short_t",
    "UShort_t",
    "Int_t",
    "UInt_t",
    "Long_t",
    "ULong_t",
    "Long64_t",
    "ULong64_t",
    "Bool_t",
    "char",
    "unsigned char",
    "short",
    "unsigned short",
    "int",
    "unsigned int",
    "long",
    "unsigned long",
    "long long",
    "unsigned long long",
    "bool",
}


def normalise(name):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(name).lower(),
    )


def suffix_number(path, prefix):
    """
    Extract the numeric suffix from:
        WCharm_WCPy8plus10.root -> "10"

    Returns None if there is no numeric suffix.
    """
    match = re.fullmatch(
        re.escape(prefix) + r"(\d+)\.root",
        path.name,
    )
    return match.group(1) if match else None


def discover_fragment_map(prefix):
    """
    Return:
        suffix -> Path
    """
    result = {}

    for path in sorted(
        ROOT_DIR.glob(
            f"{prefix}*.root"
        )
    ):
        suffix = suffix_number(
            path,
            prefix,
        )

        if suffix is not None:
            result[suffix] = path

    return result


def open_tree(path):
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
        TREE_NAME
    )

    if not tree:
        root_file.Close()
        raise RuntimeError(
            f"{TREE_NAME} not found in:\n  {path}"
        )

    return root_file, tree


def branch_info(tree):
    """
    Return:
        name -> {
            class_name,
            leaf_type,
            is_scalar
        }
    """
    info = {}

    for branch in tree.GetListOfBranches():

        name = branch.GetName()
        class_name = (
            branch.GetClassName()
            or ""
        )

        leaves = branch.GetListOfLeaves()

        leaf_type = ""
        is_scalar = False

        # Empty class name + exactly one simple leaf is the normal
        # representation of a scalar primitive ROOT branch.
        if (
            class_name == ""
            and leaves
            and leaves.GetEntries() == 1
        ):
            leaf = leaves.At(0)
            leaf_type = (
                leaf.GetTypeName()
                or ""
            )

            # Reject variable-length C arrays.
            leaf_count = (
                leaf.GetLeafCount()
            )

            is_scalar = (
                leaf_count is None
                and int(
                    leaf.GetLenStatic()
                ) == 1
            )

        info[name] = {
            "class_name": class_name,
            "leaf_type": leaf_type,
            "is_scalar": is_scalar,
        }

    return info


def resolve_by_normalised_name(
    names,
    wanted_names,
):
    by_norm = {
        normalise(name): name
        for name in names
    }

    resolved = []

    for wanted in wanted_names:
        key = normalise(wanted)

        if key in by_norm:
            actual = by_norm[key]

            if actual not in resolved:
                resolved.append(
                    actual
                )

    return resolved


def scalar_value(tree, branch_name):
    value = getattr(
        tree,
        branch_name,
    )

    # Primitive ROOT scalars convert cleanly.
    try:
        return int(value)
    except Exception:
        pass

    try:
        return float(value)
    except Exception as exc:
        raise RuntimeError(
            f"Branch {branch_name} could not be read as a scalar."
        ) from exc


def inspect_candidate_branch(
    tree,
    branch_name,
    max_events,
):
    """
    Characterise one scalar candidate branch on a single tree.
    """
    nentries = int(
        tree.GetEntries()
    )

    nscan = min(
        nentries,
        max_events,
    )

    values = []

    for ientry in range(
        nscan
    ):
        tree.GetEntry(
            ientry
        )

        values.append(
            scalar_value(
                tree,
                branch_name,
            )
        )

    unique = len(
        set(values)
    )

    uniqueness_fraction = (
        unique / len(values)
        if values
        else 0.0
    )

    return {
        "nscan": nscan,
        "unique": unique,
        "uniqueness_fraction": (
            uniqueness_fraction
        ),
        "values": values,
    }


def choose_best_id_candidate(
    particle_tree,
    parton_tree,
    particle_info,
    parton_info,
    sample_scan,
    report,
):
    """
    Find the strongest plausible common event ID.

    Preference:
      1. recognised event-ID-like names;
      2. high-cardinality common scalar integer branches.

    Low-cardinality fields (charge, run constants, etc.) are rejected.
    """

    particle_names = set(
        particle_info
    )
    parton_names = set(
        parton_info
    )

    common_names = (
        particle_names
        & parton_names
    )

    # --------------------------------------------------------
    # Known ID-like names
    # --------------------------------------------------------

    known_common = []

    for actual in resolve_by_normalised_name(
        common_names,
        KNOWN_ID_NAMES,
    ):
        if (
            particle_info[actual][
                "is_scalar"
            ]
            and parton_info[actual][
                "is_scalar"
            ]
        ):
            known_common.append(
                actual
            )

    # --------------------------------------------------------
    # Automatic high-cardinality scalar integer candidates
    # --------------------------------------------------------

    integer_common = []

    for name in sorted(
        common_names
    ):

        p = particle_info[name]
        q = parton_info[name]

        if not (
            p["is_scalar"]
            and q["is_scalar"]
        ):
            continue

        if not (
            p["leaf_type"]
            in INTEGER_LEAF_TYPES
            and q["leaf_type"]
            in INTEGER_LEAF_TYPES
        ):
            continue

        integer_common.append(
            name
        )

    candidates = []

    # Known names first.
    for name in known_common:
        if name not in candidates:
            candidates.append(name)

    # Then other scalar integer branches.
    for name in integer_common:
        if name not in candidates:
            candidates.append(name)

    if not candidates:

        report(
            "  No common scalar integer/event-ID-like branch was found."
        )
        return None, {}

    diagnostics = {}

    report()
    report(
        "  Candidate scalar ID branches:"
    )

    for name in candidates:

        pdiag = inspect_candidate_branch(
            particle_tree,
            name,
            sample_scan,
        )

        qdiag = inspect_candidate_branch(
            parton_tree,
            name,
            sample_scan,
        )

        # Initial same-position overlap is only a diagnostic.
        nalign = min(
            len(pdiag["values"]),
            len(qdiag["values"]),
        )

        same_position = 0

        for index in range(
            nalign
        ):
            if (
                pdiag["values"][index]
                == qdiag["values"][index]
            ):
                same_position += 1

        same_position_fraction = (
            same_position / nalign
            if nalign
            else 0.0
        )

        pset = set(
            pdiag["values"]
        )
        qset = set(
            qdiag["values"]
        )

        intersection = (
            pset & qset
        )

        sample_overlap_particle = (
            len(intersection)
            / len(pset)
            if pset
            else 0.0
        )

        diagnostics[name] = {
            "particle": pdiag,
            "parton": qdiag,
            "same_position_fraction": (
                same_position_fraction
            ),
            "sample_overlap_particle": (
                sample_overlap_particle
            ),
            "known_name": (
                name in known_common
            ),
        }

        report(
            (
                f"    {name}: "
                f"particle uniqueness="
                f"{pdiag['uniqueness_fraction']:.3f}, "
                f"parton uniqueness="
                f"{qdiag['uniqueness_fraction']:.3f}, "
                f"sample ID overlap="
                f"{sample_overlap_particle:.3f}, "
                f"same-entry equality="
                f"{same_position_fraction:.3f}"
            )
        )

    # A useful event key should be close to unique on both sides.
    useful = [
        name
        for name, diag in diagnostics.items()
        if (
            diag[
                "particle"
            ][
                "uniqueness_fraction"
            ] >= 0.90
            and diag[
                "parton"
            ][
                "uniqueness_fraction"
            ] >= 0.90
        )
    ]

    if not useful:

        report()
        report(
            "  No candidate is sufficiently unique to be used "
            "as an event key."
        )
        return None, diagnostics

    # Rank by:
    #   recognised name,
    #   sample overlap,
    #   minimum uniqueness.
    useful.sort(
        key=lambda name: (
            1
            if diagnostics[name][
                "known_name"
            ]
            else 0,
            diagnostics[name][
                "sample_overlap_particle"
            ],
            min(
                diagnostics[name][
                    "particle"
                ][
                    "uniqueness_fraction"
                ],
                diagnostics[name][
                    "parton"
                ][
                    "uniqueness_fraction"
                ],
            ),
        ),
        reverse=True,
    )

    best = useful[0]

    report()
    report(
        f"  Best candidate event key: {best}"
    )

    return best, diagnostics


def read_all_ids(
    paths_by_suffix,
    branch_name,
    report,
    side_label,
):
    """
    Read one scalar ID branch across all fragments.

    Returns:
      set_of_ids,
      duplicate_count,
      total_values
    """
    seen = set()
    duplicates = 0
    total = 0

    for suffix in sorted(
        paths_by_suffix,
        key=lambda x: int(x),
    ):
        path = (
            paths_by_suffix[
                suffix
            ]
        )

        root_file, tree = (
            open_tree(path)
        )

        info = branch_info(
            tree
        )

        # Allow case/underscore naming changes between fragments.
        resolved = (
            resolve_by_normalised_name(
                info.keys(),
                [branch_name],
            )
        )

        if not resolved:
            root_file.Close()
            raise RuntimeError(
                f"{side_label}: branch {branch_name} "
                f"not found in {path.name}"
            )

        actual_name = (
            resolved[0]
        )

        if not info[
            actual_name
        ][
            "is_scalar"
        ]:
            root_file.Close()
            raise RuntimeError(
                f"{side_label}: {actual_name} is not scalar "
                f"in {path.name}"
            )

        nentries = int(
            tree.GetEntries()
        )

        for ientry in range(
            nentries
        ):
            tree.GetEntry(
                ientry
            )

            value = scalar_value(
                tree,
                actual_name,
            )

            total += 1

            if value in seen:
                duplicates += 1
            else:
                seen.add(
                    value
                )

        root_file.Close()

    report(
        f"  {side_label}: read {total:,} IDs, "
        f"{len(seen):,} unique, "
        f"{duplicates:,} duplicates"
    )

    return (
        seen,
        duplicates,
        total,
    )


def same_entry_alignment(
    particle_map,
    parton_map,
    branch_name,
):
    """
    Among corresponding suffix fragments, measure how often the
    candidate ID at entry i is identical on the particle and parton
    side.

    This is NOT used to establish matching; it only tests whether
    naive entry-index pairing happens to be valid.
    """
    equal = 0
    compared = 0

    common_suffixes = sorted(
        set(particle_map)
        & set(parton_map),
        key=lambda x: int(x),
    )

    for suffix in common_suffixes:

        pfile, ptree = open_tree(
            particle_map[suffix]
        )

        qfile, qtree = open_tree(
            parton_map[suffix]
        )

        pinfo = branch_info(
            ptree
        )
        qinfo = branch_info(
            qtree
        )

        presolved = (
            resolve_by_normalised_name(
                pinfo.keys(),
                [branch_name],
            )
        )

        qresolved = (
            resolve_by_normalised_name(
                qinfo.keys(),
                [branch_name],
            )
        )

        if (
            not presolved
            or not qresolved
        ):
            pfile.Close()
            qfile.Close()
            continue

        pname = presolved[0]
        qname = qresolved[0]

        ncompare = min(
            int(
                ptree.GetEntries()
            ),
            int(
                qtree.GetEntries()
            ),
        )

        for ientry in range(
            ncompare
        ):
            ptree.GetEntry(
                ientry
            )
            qtree.GetEntry(
                ientry
            )

            pvalue = scalar_value(
                ptree,
                pname,
            )

            qvalue = scalar_value(
                qtree,
                qname,
            )

            compared += 1

            if pvalue == qvalue:
                equal += 1

        pfile.Close()
        qfile.Close()

    fraction = (
        equal / compared
        if compared
        else 0.0
    )

    return (
        equal,
        compared,
        fraction,
    )


def analyse_channel(
    channel,
    config,
    sample_scan,
    report,
):
    report()
    report("=" * 78)
    report(
        f"CHANNEL: Pythia W{channel}"
    )
    report("=" * 78)

    particle_map = (
        discover_fragment_map(
            config[
                "particle_prefix"
            ]
        )
    )

    parton_map = (
        discover_fragment_map(
            config[
                "parton_prefix"
            ]
        )
    )

    common_suffixes = sorted(
        set(particle_map)
        & set(parton_map),
        key=lambda x: int(x),
    )

    report(
        f"Particle fragments: {len(particle_map)}"
    )
    report(
        f"Parton fragments:   {len(parton_map)}"
    )
    report(
        f"Paired suffixes:     {len(common_suffixes)}"
    )

    if not common_suffixes:
        report(
            "VERDICT: No corresponding filename suffixes exist."
        )
        return

    # Show whether naive fragment sizes already differ.
    report()
    report(
        "Fragment entry counts:"
    )

    any_count_difference = False

    for suffix in common_suffixes:

        pfile, ptree = (
            open_tree(
                particle_map[
                    suffix
                ]
            )
        )

        qfile, qtree = (
            open_tree(
                parton_map[
                    suffix
                ]
            )
        )

        nparticle = int(
            ptree.GetEntries()
        )
        nparton = int(
            qtree.GetEntries()
        )

        if nparticle != nparton:
            any_count_difference = True

        report(
            f"  suffix {suffix}: "
            f"particle={nparticle:,}, "
            f"parton={nparton:,}"
        )

        pfile.Close()
        qfile.Close()

    # Inspect schema on first corresponding fragment.
    suffix0 = (
        common_suffixes[0]
    )

    pfile, ptree = (
        open_tree(
            particle_map[
                suffix0
            ]
        )
    )

    qfile, qtree = (
        open_tree(
            parton_map[
                suffix0
            ]
        )
    )

    pinfo = branch_info(
        ptree
    )

    qinfo = branch_info(
        qtree
    )

    common_branches = sorted(
        set(pinfo)
        & set(qinfo)
    )

    report()
    report(
        f"First paired fragment: suffix {suffix0}"
    )
    report(
        f"Common branch names: {len(common_branches)}"
    )

    # Print ID-looking common branch names, even before quality tests.
    id_looking = [
        name
        for name in common_branches
        if any(
            token in normalise(name)
            for token in (
                "event",
                "run",
                "lumi",
                "barcode",
                "seed",
                "channel",
                "id",
            )
        )
    ]

    report(
        "ID-looking common branches:"
    )

    if id_looking:
        for name in id_looking:
            meta = pinfo[name]
            report(
                f"  {name} "
                f"(class='{meta['class_name']}', "
                f"type='{meta['leaf_type']}', "
                f"scalar={meta['is_scalar']})"
            )
    else:
        report(
            "  none"
        )

    best_id, diagnostics = (
        choose_best_id_candidate(
            particle_tree=ptree,
            parton_tree=qtree,
            particle_info=pinfo,
            parton_info=qinfo,
            sample_scan=sample_scan,
            report=report,
        )
    )

    pfile.Close()
    qfile.Close()

    report()

    if best_id is None:

        if any_count_difference:
            report(
                "Naive entry-number matching is definitely unsafe: "
                "corresponding particle/parton fragments contain "
                "different numbers of entries."
            )

        report(
            "VERDICT: No sufficiently unique common event identifier "
            "was found. Event-by-event particle/parton matching cannot "
            "be established from these ntuples using the available "
            "branches."
        )
        return

    report(
        "Full-sample overlap test using:"
    )
    report(
        f"  {best_id}"
    )

    (
        particle_ids,
        particle_duplicates,
        particle_total,
    ) = read_all_ids(
        particle_map,
        best_id,
        report,
        "particle",
    )

    (
        parton_ids,
        parton_duplicates,
        parton_total,
    ) = read_all_ids(
        parton_map,
        best_id,
        report,
        "parton",
    )

    overlap = (
        particle_ids
        & parton_ids
    )

    overlap_particle_fraction = (
        len(overlap)
        / len(particle_ids)
        if particle_ids
        else 0.0
    )

    overlap_parton_fraction = (
        len(overlap)
        / len(parton_ids)
        if parton_ids
        else 0.0
    )

    report()
    report(
        f"Common unique IDs: {len(overlap):,}"
    )
    report(
        f"Fraction of particle IDs found at parton level: "
        f"{overlap_particle_fraction:.6f}"
    )
    report(
        f"Fraction of parton IDs found at particle level: "
        f"{overlap_parton_fraction:.6f}"
    )

    (
        equal_entries,
        compared_entries,
        alignment_fraction,
    ) = same_entry_alignment(
        particle_map,
        parton_map,
        best_id,
    )

    report()
    report(
        "Naive same-entry alignment diagnostic:"
    )
    report(
        f"  equal IDs at same entry: "
        f"{equal_entries:,} / {compared_entries:,} "
        f"= {alignment_fraction:.6f}"
    )

    unique_enough = (
        particle_duplicates == 0
        and parton_duplicates == 0
    )

    report()

    if (
        unique_enough
        and overlap_particle_fraction >= 0.95
    ):
        report(
            "VERDICT: YES — there is strong evidence that event-by-event "
            f"matching is possible using '{best_id}'."
        )

        if alignment_fraction < 0.99:
            report(
                "However, do NOT match by ROOT entry number. Build an "
                f"ID -> event lookup using '{best_id}'."
            )
        else:
            report(
                "The ordering also appears highly aligned, but matching "
                f"by '{best_id}' is still safer than relying on entry "
                "number."
            )

    elif (
        unique_enough
        and overlap_particle_fraction >= 0.10
    ):
        report(
            "VERDICT: PARTIAL — a unique common identifier exists and a "
            "non-trivial subset overlaps. Event-by-event matching may be "
            "possible for the shared subset, but not for every event."
        )
        report(
            f"Use '{best_id}' as the key; never assume entry i matches "
            "entry i."
        )

    else:
        report(
            "VERDICT: NO useful one-to-one match was demonstrated. "
            "Either the candidate key is duplicated or the particle/"
            "parton overlap is too small."
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check whether high-stat Pythia particle and parton ROOT "
            "files can be matched event-by-event."
        )
    )

    parser.add_argument(
        "--sample-scan",
        type=int,
        default=5000,
        help=(
            "Entries used to assess candidate ID branches before the "
            "full overlap test (default: 5000)."
        ),
    )

    args = parser.parse_args()

    if not ROOT_DIR.is_dir():
        raise RuntimeError(
            "ROOT fragment directory does not exist:\n"
            f"  {ROOT_DIR}"
        )

    report_lines = []

    def report(text=""):
        line = str(text)
        print(line)
        report_lines.append(
            line
        )

    report(
        "High-statistics particle/parton event-matching diagnostic"
    )
    report(
        "=========================================================="
    )
    report()
    report(
        f"ROOT directory: {ROOT_DIR}"
    )
    report()
    report(
        "This script will not treat matching filename suffixes or "
        "matching entry numbers as proof of event identity."
    )

    for channel, config in (
        CHANNELS.items()
    ):
        analyse_channel(
            channel=channel,
            config=config,
            sample_scan=args.sample_scan,
            report=report,
        )

    REPORT_PATH.write_text(
        "\n".join(report_lines)
        + "\n",
        encoding="utf-8",
    )

    report()
    report(
        "=" * 78
    )
    report(
        f"Report saved to: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
