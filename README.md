# Strange Proton Uncertainty Decomposition

This repository studies uncertainty decomposition in parton-to-particle
correction factors for lepton pseudorapidity in W+c production.

## Contents

- `data/` — ROOT event samples
- `weights/` — uncertainty weight information
- `lepton_pseudorapidity_work/` — analysis and plotting code

## Running the Analysis

All analysis scripts are contained in:

```text
lepton_pseudorapidity_work/
```

The workflow is split into three main stages:

```text
ROOT data
   ↓
make_histograms_etalepton.py
   ↓
build_corrections_etalepton.py
   ↓
run_etalepton_interactive.py
   ↓
Correction-factor and uncertainty plots
```

### 1. Generate the histograms

```bash
python make_histograms_etalepton.py
```

This is the most computationally expensive stage of the analysis and processes the underlying ROOT event samples.

**In most cases, this script does not need to be run.** The outputs generated from this stage are already included in the repository, and rerunning the full event processing can take a considerable amount of time.

The script should mainly be rerun if the underlying ROOT samples, event selection, histogram definitions or uncertainty treatment are being changed.

Generated results are stored in:

```text
make_histograms_etalepton_outputs/
```

### 2. Build the correction factors

```bash
python build_corrections_etalepton.py
```

This stage uses the generated histogram information to construct the lepton-pseudorapidity parton-to-particle correction factors and their associated uncertainty information.

Outputs are written to the corresponding:

```text
build_corrections_etalepton_csv/
build_corrections_etalepton_outputs/
```

directories.

Since the necessary intermediate files are already included in the repository, this stage can also be skipped if the aim is simply to reproduce or inspect the final plots.

### 3. Generate the interactive plots

For the final visualisation, run:

```bash
python run_etalepton_interactive.py
```

This runs the interactive plotting workflow using:

```text
plot_csv_etalepton_interactive.py
```

and produces the correction-factor and uncertainty-decomposition visualisations for the Pythia and Herwig (W^+) and (W^-) samples.

The resulting files are written to the plotting output directory.

### Quick start

If you simply want to reproduce or explore the final uncertainty-decomposition plots, the computationally expensive ROOT-processing stage is unnecessary because the required intermediate files are already provided.

From inside `lepton_pseudorapidity_work/`, simply run:

```bash
python run_etalepton_interactive.py
```

### Full reproduction

To reproduce the complete analysis from the original ROOT samples:

```bash
python make_histograms_etalepton.py
python build_corrections_etalepton.py
python run_etalepton_interactive.py
```

The first command may take significantly longer than the others because it performs the event-level processing of the ROOT samples.
