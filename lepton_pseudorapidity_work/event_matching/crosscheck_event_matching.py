#!/usr/bin/env python3

from pathlib import Path
import csv
import ROOT

ROOT.gROOT.SetBatch(True)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

REPORT_PATH = SCRIPT_DIR / "event_matching_crosscheck_report.txt"
COUNTS_CSV = SCRIPT_DIR / "event_matching_common_counts.csv"
PARTICLE_FRACTION_CSV = SCRIPT_DIR / "event_matching_particle_fraction.csv"
PARTON_FRACTION_CSV = SCRIPT_DIR / "event_matching_parton_fraction.csv"
SMALLER_FRACTION_CSV = SCRIPT_DIR / "event_matching_smaller_set_fraction.csv"

PARTICLE_FILES = {
    "Pythia W+": DATA_DIR / "WCharm_WCPy8plus.root",
    "Pythia W-": DATA_DIR / "WCharm_WCPy8minus.root",
    "Herwig W+": DATA_DIR / "WCharm_WCH7plus.root",
    "Herwig W-": DATA_DIR / "WCharm_WCH7minus.root",
}

PARTON_FILES = {
    "Pythia W+": DATA_DIR / "WCharm_WCPyPartonplus.root",
    "Pythia W-": DATA_DIR / "WCharm_WCPyPartonminus.root",
    "Herwig W+": DATA_DIR / "WCharm_WCHPartonplus.root",
    "Herwig W-": DATA_DIR / "WCharm_WCHPartonminus.root",
}

LABELS = ["Pythia W+", "Pythia W-", "Herwig W+", "Herwig W-"]
EVENT_BRANCH = "EventNumber"
REPORT_LINES = []

def report(line=""):
    print(line)
    REPORT_LINES.append(str(line))

def choose_tree(root_file, filename):
    trees = []
    for key in root_file.GetListOfKeys():
        obj = key.ReadObj()
        if obj and obj.InheritsFrom("TTree"):
            trees.append(obj)
    if not trees:
        raise RuntimeError(f"No TTree found in {filename}")
    return max(trees, key=lambda t: int(t.GetEntries()))

def load_event_numbers(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing ROOT file: {path}")

    root_file = ROOT.TFile.Open(str(path), "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {path}")

    tree = choose_tree(root_file, path)
    if not tree.GetBranch(EVENT_BRANCH):
        root_file.Close()
        raise RuntimeError(
            f"Branch '{EVENT_BRANCH}' not found in {path}"
        )

    ids = []
    nentries = int(tree.GetEntries())

    for ientry in range(nentries):
        tree.GetEntry(ientry)
        ids.append(int(getattr(tree, EVENT_BRANCH)))

    root_file.Close()

    unique_ids = set(ids)

    return {
        "entries": nentries,
        "unique": len(unique_ids),
        "duplicates": nentries - len(unique_ids),
        "ids": unique_ids,
    }

def compare(particle_info, parton_info):
    p = particle_info["ids"]
    q = parton_info["ids"]
    common = p & q

    np = len(p)
    nq = len(q)
    nc = len(common)

    return {
        "common": nc,
        "particle_fraction": nc / np if np else 0.0,
        "parton_fraction": nc / nq if nq else 0.0,
        "smaller_fraction": nc / min(np, nq) if min(np, nq) else 0.0,
    }

def print_matrix(title, matrix, formatter):
    report()
    report(title)
    report("-" * 90)

    header = f"{'Particle / Parton':<18}"
    for label in LABELS:
        header += f"{label:>17}"
    report(header)
    report("-" * 90)

    for particle_label in LABELS:
        row = f"{particle_label:<18}"
        for parton_label in LABELS:
            row += f"{formatter(matrix[particle_label][parton_label]):>17}"
        report(row)

    report("-" * 90)

def write_matrix_csv(path, matrix):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["particle_sample", *LABELS])
        for particle_label in LABELS:
            writer.writerow(
                [particle_label]
                + [matrix[particle_label][parton_label] for parton_label in LABELS]
            )

def main():
    report()
    report("=" * 90)
    report(" W+c EventNumber cross-check: all particle vs all parton samples")
    report("=" * 90)
    report()
    report(f"Project root: {PROJECT_ROOT}")
    report(f"Data directory: {DATA_DIR}")
    report()
    report("The expected physical matches are the diagonal entries.")
    report()

    particle_data = {}
    parton_data = {}

    report("Loading particle-level EventNumbers...")
    for label in LABELS:
        info = load_event_numbers(PARTICLE_FILES[label])
        particle_data[label] = info
        report(
            f"  {label:<10}: {info['entries']:,} entries, "
            f"{info['unique']:,} unique, {info['duplicates']:,} duplicates"
        )

    report()
    report("Loading parton-level EventNumbers...")
    for label in LABELS:
        info = load_event_numbers(PARTON_FILES[label])
        parton_data[label] = info
        report(
            f"  {label:<10}: {info['entries']:,} entries, "
            f"{info['unique']:,} unique, {info['duplicates']:,} duplicates"
        )

    counts = {}
    particle_fraction = {}
    parton_fraction = {}
    smaller_fraction = {}

    for p_label in LABELS:
        counts[p_label] = {}
        particle_fraction[p_label] = {}
        parton_fraction[p_label] = {}
        smaller_fraction[p_label] = {}

        for q_label in LABELS:
            result = compare(particle_data[p_label], parton_data[q_label])
            counts[p_label][q_label] = result["common"]
            particle_fraction[p_label][q_label] = result["particle_fraction"]
            parton_fraction[p_label][q_label] = result["parton_fraction"]
            smaller_fraction[p_label][q_label] = result["smaller_fraction"]

    print_matrix(
        "COMMON UNIQUE EventNumbers",
        counts,
        lambda x: f"{x:,}",
    )

    print_matrix(
        "PERCENTAGE OF PARTICLE SAMPLE MATCHED",
        particle_fraction,
        lambda x: f"{100*x:.2f}%",
    )

    print_matrix(
        "PERCENTAGE OF PARTON SAMPLE MATCHED",
        parton_fraction,
        lambda x: f"{100*x:.2f}%",
    )

    print_matrix(
        "MATCH FRACTION RELATIVE TO SMALLER UNIQUE SET",
        smaller_fraction,
        lambda x: f"{100*x:.2f}%",
    )

    report()
    report("=" * 90)
    report(" DIAGONAL VS WRONG-PAIR CHECK")
    report("=" * 90)
    report()

    all_diagonals_clearly_larger = True

    for p_label in LABELS:
        correct = smaller_fraction[p_label][p_label]
        wrong_values = [
            smaller_fraction[p_label][q_label]
            for q_label in LABELS
            if q_label != p_label
        ]
        largest_wrong = max(wrong_values)
        difference = correct - largest_wrong

        report(f"{p_label}:")
        report(f"  Correct-pair match:       {100*correct:.2f}%")
        report(f"  Largest wrong-pair match: {100*largest_wrong:.2f}%")
        report(f"  Difference:               {100*difference:.2f} percentage points")
        report()

        if difference < 0.10:
            all_diagonals_clearly_larger = False

    report("=" * 90)
    report(" INTERPRETATION")
    report("=" * 90)
    report()

    if all_diagonals_clearly_larger:
        report(
            "The intended particle/parton pairings stand clearly above the wrong "
            "pairings. This is strong evidence that EventNumber carries genuine "
            "event-by-event correspondence."
        )
    else:
        report(
            "At least one intended pairing does not stand clearly above the wrong "
            "pairings. EventNumber overlap alone is therefore not sufficient to "
            "claim genuine event-by-event matching."
        )

    report()
    report(
        "The 10 percentage-point criterion above is only a diagnostic; inspect "
        "the full matrices before drawing a final conclusion."
    )
    report()

    write_matrix_csv(COUNTS_CSV, counts)
    write_matrix_csv(PARTICLE_FRACTION_CSV, particle_fraction)
    write_matrix_csv(PARTON_FRACTION_CSV, parton_fraction)
    write_matrix_csv(SMALLER_FRACTION_CSV, smaller_fraction)

    report("Output files:")
    report(f"  {REPORT_PATH}")
    report(f"  {COUNTS_CSV}")
    report(f"  {PARTICLE_FRACTION_CSV}")
    report(f"  {PARTON_FRACTION_CSV}")
    report(f"  {SMALLER_FRACTION_CSV}")
    report()

    REPORT_PATH.write_text("\n".join(REPORT_LINES) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
