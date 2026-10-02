#!/usr/bin/env python3

"""
check_eventnumber_matching.py

Direct test of whether the NEW high-statistics Pythia particle and
parton ROOT fragments can be matched event-by-event using EventNumber.

Why this exists
---------------
The broader diagnostic found a common branch:

    EventNumber   type=Int_t

but incorrectly rejected it as non-scalar because of an overly strict
ROOT leaf-shape test.

This script therefore reads tree.EventNumber directly and tests:

1. Whether EventNumber exists on both particle and parton trees.
2. Whether EventNumber is unique within each fragment.
3. How much the particle and parton EventNumber sets overlap for each
   matched fragment suffix.
4. Whether the combined key

       (fragment_suffix, EventNumber)

   is unique across the full high-statistics sample.
5. Whether naive entry-index matching happens to align (diagnostic only).

No files are modified.
"""

from pathlib import Path
import ROOT

ROOT.gROOT.SetBatch(True)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR / "cross_check_ROOT_files"
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


def fragment_map(prefix):
    result = {}

    for path in ROOT_DIR.glob(f"{prefix}*.root"):

        stem = path.stem

        suffix = stem.replace(prefix, "", 1)

        if suffix.isdigit():
            result[int(suffix)] = path

    return result


def open_tree(path):
    f = ROOT.TFile.Open(str(path), "READ")

    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open {path}")

    tree = f.Get(TREE_NAME)

    if not tree:
        f.Close()
        raise RuntimeError(f"{TREE_NAME} not found in {path}")

    return f, tree


def has_branch(tree, name):
    return tree.GetBranch(name) is not None


def read_event_numbers(tree):
    if not has_branch(tree, "EventNumber"):
        raise RuntimeError("EventNumber branch not found")

    values = []

    for ientry in range(int(tree.GetEntries())):
        tree.GetEntry(ientry)

        try:
            value = int(tree.EventNumber)
        except Exception as exc:
            raise RuntimeError(
                "EventNumber exists but could not be read as a scalar Int_t."
            ) from exc

        values.append(value)

    return values


def pct(x):
    return f"{100.0 * x:.4f}%"


def analyse_channel(label, config):
    print()
    print("=" * 80)
    print(f"Pythia W{label}")
    print("=" * 80)

    particle = fragment_map(config["particle_prefix"])
    parton = fragment_map(config["parton_prefix"])

    suffixes = sorted(set(particle) & set(parton))

    print(f"Paired fragments: {len(suffixes)}")

    if not suffixes:
        print("No paired fragments.")
        return

    total_particle = 0
    total_parton = 0

    particle_composite = set()
    parton_composite = set()

    particle_composite_duplicates = 0
    parton_composite_duplicates = 0

    all_particle_event_numbers = []
    all_parton_event_numbers = []

    same_entry_equal = 0
    same_entry_compared = 0

    per_fragment_particle_overlap = []
    per_fragment_parton_overlap = []

    print()
    print(
        "suffix   particle   parton    "
        "particle->parton   parton->particle   "
        "unique(P) unique(A)"
    )

    for suffix in suffixes:

        pf, ptree = open_tree(particle[suffix])
        af, atree = open_tree(parton[suffix])

        if not has_branch(ptree, "EventNumber"):
            pf.Close()
            af.Close()
            raise RuntimeError(
                f"EventNumber missing in particle fragment {particle[suffix].name}"
            )

        if not has_branch(atree, "EventNumber"):
            pf.Close()
            af.Close()
            raise RuntimeError(
                f"EventNumber missing in parton fragment {parton[suffix].name}"
            )

        pnums = read_event_numbers(ptree)
        anums = read_event_numbers(atree)

        pf.Close()
        af.Close()

        pset = set(pnums)
        aset = set(anums)

        p_unique = len(pset) == len(pnums)
        a_unique = len(aset) == len(anums)

        overlap = pset & aset

        p_overlap = (
            len(overlap) / len(pset)
            if pset else 0.0
        )

        a_overlap = (
            len(overlap) / len(aset)
            if aset else 0.0
        )

        per_fragment_particle_overlap.append(p_overlap)
        per_fragment_parton_overlap.append(a_overlap)

        print(
            f"{suffix:>4d}   "
            f"{len(pnums):>8,d}   "
            f"{len(anums):>8,d}   "
            f"{pct(p_overlap):>16s}   "
            f"{pct(a_overlap):>16s}   "
            f"{str(p_unique):>9s} {str(a_unique):>9s}"
        )

        total_particle += len(pnums)
        total_parton += len(anums)

        all_particle_event_numbers.extend(pnums)
        all_parton_event_numbers.extend(anums)

        for event_number in pnums:
            key = (suffix, event_number)

            if key in particle_composite:
                particle_composite_duplicates += 1

            particle_composite.add(key)

        for event_number in anums:
            key = (suffix, event_number)

            if key in parton_composite:
                parton_composite_duplicates += 1

            parton_composite.add(key)

        ncompare = min(len(pnums), len(anums))

        for i in range(ncompare):
            same_entry_compared += 1

            if pnums[i] == anums[i]:
                same_entry_equal += 1

    composite_overlap = (
        particle_composite
        & parton_composite
    )

    p_composite_fraction = (
        len(composite_overlap)
        / len(particle_composite)
        if particle_composite else 0.0
    )

    a_composite_fraction = (
        len(composite_overlap)
        / len(parton_composite)
        if parton_composite else 0.0
    )

    global_p_unique = (
        len(set(all_particle_event_numbers))
        == len(all_particle_event_numbers)
    )

    global_a_unique = (
        len(set(all_parton_event_numbers))
        == len(all_parton_event_numbers)
    )

    same_entry_fraction = (
        same_entry_equal / same_entry_compared
        if same_entry_compared else 0.0
    )

    print()
    print("Full-sample results")
    print("-------------------")
    print(f"Particle entries: {total_particle:,}")
    print(f"Parton entries:   {total_parton:,}")
    print()
    print(
        "EventNumber globally unique without fragment suffix:"
    )
    print(f"  particle: {global_p_unique}")
    print(f"  parton:   {global_a_unique}")
    print()
    print(
        "Composite key (fragment suffix, EventNumber):"
    )
    print(
        f"  particle duplicate keys: {particle_composite_duplicates:,}"
    )
    print(
        f"  parton duplicate keys:   {parton_composite_duplicates:,}"
    )
    print(
        f"  common composite keys:   {len(composite_overlap):,}"
    )
    print(
        "  fraction of particle events with a matching parton event: "
        f"{pct(p_composite_fraction)}"
    )
    print(
        "  fraction of parton events with a matching particle event: "
        f"{pct(a_composite_fraction)}"
    )
    print()
    print(
        "Naive same-entry EventNumber equality:"
    )
    print(
        f"  {same_entry_equal:,} / {same_entry_compared:,} "
        f"= {pct(same_entry_fraction)}"
    )
    print()

    composite_unique = (
        particle_composite_duplicates == 0
        and parton_composite_duplicates == 0
    )

    if (
        composite_unique
        and p_composite_fraction >= 0.95
    ):
        print(
            "VERDICT: YES — event-by-event matching is strongly supported."
        )
        print(
            "Use the key (fragment suffix, EventNumber), not the ROOT entry index."
        )

    elif (
        composite_unique
        and p_composite_fraction >= 0.10
    ):
        print(
            "VERDICT: PARTIAL — a real event-level overlap exists."
        )
        print(
            "The shared subset can be matched using "
            "(fragment suffix, EventNumber)."
        )

    else:
        print(
            "VERDICT: NO — EventNumber does not provide a sufficiently "
            "large unique overlap to establish useful event-by-event matching."
        )


def main():

    if not ROOT_DIR.is_dir():
        raise RuntimeError(
            f"Directory not found:\n  {ROOT_DIR}"
        )

    print(
        "Direct EventNumber particle/parton matching test"
    )
    print(
        "================================================"
    )
    print(
        f"ROOT directory: {ROOT_DIR}"
    )

    for label, config in CHANNELS.items():
        analyse_channel(
            label,
            config,
        )


if __name__ == "__main__":
    main()
