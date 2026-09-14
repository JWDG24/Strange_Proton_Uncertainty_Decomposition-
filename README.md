# Strange Proton Uncertainty Decomposition

## Overview

This repository contains the lepton pseudorapidity part of a W+c correction factor and uncertainty decomposition study. The analysis compares particle level and parton level Monte Carlo samples, applies the fiducial event selection used in the project and studies the statistical and systematic uncertainty on the correction factor

\[
C_i =
\frac{N^{\mathrm{parton}}_i}
     {N^{\mathrm{particle}}_i}.
\]

The main observable is the absolute lepton pseudorapidity \(|\eta_\ell|\).

The current workflow contains four main stages:

1. Build the original \(m_T^W > 40\) GeV correction factors from the weighted Monte Carlo samples.
2. Validate the propagated statistical uncertainty using an odd and even event split.
3. Cross check the nominal Pythia correction using new high statistics weight 0 samples.
4. Extract systematic uncertainties relative to weight 0 and apply them to the new high statistics Pythia nominal correction.

A final interactive dashboard brings the main results together in one place.

---

# Quick start

If the repository already contains the generated CSV and HTML outputs, you do **not** need to rerun the full ROOT event analysis just to inspect the results.

From the project root:

```bash
cd /mnt/c/Users/dugar/uncertainty_decomposition
```

To generate and open the combined interactive dashboard:

```bash
python3 lepton_pseudorapidity_work/display_all_findings.py --open
```

This creates:

```text
lepton_pseudorapidity_work/
└── all_findings_dashboard/
    └── strange_proton_uncertainty_dashboard.html
```

The dashboard is the easiest way to inspect the complete analysis.

If the dashboard output already exists and has been committed to the repository, it can also be opened directly in a browser without rerunning any analysis scripts.

---

# Repository

GitHub:

```text
https://github.com/JWDG24/Strange_Proton_Uncertainty_Decomposition-
```

Clone with:

```bash
git clone https://github.com/JWDG24/Strange_Proton_Uncertainty_Decomposition-.git
cd Strange_Proton_Uncertainty_Decomposition-
```

The local working directory used during development is:

```text
/mnt/c/Users/dugar/uncertainty_decomposition
```

---

# Software

The analysis is written in Python and uses CERN ROOT.

The development environment used for this project is:

```text
Windows 11
WSL
Ubuntu 24.04
ROOT 6.36.04
VS Code
Python 3
```

ROOT is installed in WSL at:

```text
/opt/root/root
```

The ROOT environment is loaded from `.bashrc` using:

```bash
source /opt/root/root/bin/thisroot.sh
```

Check that ROOT is available with:

```bash
which root
root --version
```

The expected ROOT executable is:

```text
/opt/root/root/bin/root
```

The plotting scripts also use pandas and Plotly.

On Ubuntu:

```bash
sudo apt install python3-pandas python3-plotly -y
```

---

# Physics selection

The lepton pseudorapidity analysis uses the following fiducial cuts:

\[
E_T^{\mathrm{miss}} > 25\ \mathrm{GeV},
\]

\[
p_T^\ell > 20\ \mathrm{GeV},
\]

\[
|\eta_\ell| < 2.5,
\]

\[
m_T^W > 40\ \mathrm{GeV}.
\]

The transverse mass is calculated as

\[
(m_T^W)^2 =
2 p_T^\ell E_T^{\mathrm{miss}}
\left(1-\cos\Delta\phi\right).
\]

The \(|\eta_\ell|\) binning used in the current analysis is

```text
0.00
0.21
0.42
0.63
0.84
1.05
1.37
1.52
1.74
1.95
2.18
2.50
```

---

# Main project structure

The important part of the repository is:

```text
uncertainty_decomposition/
│
├── data/
├── weights/
│
└── lepton_pseudorapidity_work/
    │
    ├── MTWcut_make_histograms_etalepton.py
    ├── MTWcut_build_corrections_etalepton.py
    │
    ├── MTWcut_make_histograms_etalepton_outputs/
    ├── MTWcut_build_corrections_etalepton_csv/
    │
    ├── even_odd_uncertainties/
    │   ├── even_odd_uncertainty.py
    │   ├── plot_even_odd_interactive.py
    │   └── outputs/
    │
    ├── cross_check_weight0/
    │   ├── cross_check_weight0.py
    │   ├── open_cross_check_interactive.py
    │   ├── cross_check_ROOT_files/
    │   └── outputs/
    │
    ├── weights_relative_uncertainties/
    │   ├── build_relative_uncertainties.py
    │   ├── plot_relative_uncertainties_interactive.py
    │   ├── apply_relative_uncertainties_to_new_nominal.py
    │   └── outputs/
    │
    ├── display_all_findings.py
    │
    └── all_findings_dashboard/
        └── strange_proton_uncertainty_dashboard.html
```

Some generated folders may only appear after the corresponding script has been run.

---

# What you actually need to run

## If you only want to inspect the current results

You normally only need:

```bash
python3 lepton_pseudorapidity_work/display_all_findings.py --open
```

This reads the stored CSV outputs from the previous analysis stages and creates one interactive HTML dashboard.

You do **not** need to rerun the ROOT event loops if the required histogram and CSV outputs are already present.

---

# What is already stored and can usually be reused

Several analysis stages write reusable outputs. Once these exist, later scripts can read them directly.

## Original histogram outputs

Produced by:

```text
MTWcut_make_histograms_etalepton.py
```

Stored in:

```text
lepton_pseudorapidity_work/
└── MTWcut_make_histograms_etalepton_outputs/
```

These contain the weighted lepton pseudorapidity histograms.

If these files are already present and unchanged, there is no need to rerun the full event loop before rebuilding correction factors or relative uncertainties.

---

## Original correction CSV files

Produced by:

```text
MTWcut_build_corrections_etalepton.py
```

Stored in:

```text
lepton_pseudorapidity_work/
└── MTWcut_build_corrections_etalepton_csv/
```

These contain the original correction factors and uncertainty decomposition for:

```text
Pythia W+
Pythia W-
Herwig W+
Herwig W-
```

If these CSV files already exist, they can be plotted or compared directly.

---

## Odd and even statistical validation outputs

Produced by:

```text
even_odd_uncertainties/even_odd_uncertainty.py
```

Stored under:

```text
lepton_pseudorapidity_work/
└── even_odd_uncertainties/
    └── outputs/
```

These outputs contain:

```text
C_all
C_odd
C_even
|C_odd - C_even|
|C_odd - C_even| / 2
ROOT propagated statistical uncertainty
relative split uncertainty
```

If the CSV outputs already exist, the event split does not need to be rerun just to reproduce the plots.

---

## New weight 0 Pythia cross check outputs

Produced by:

```text
cross_check_weight0/cross_check_weight0.py
```

Stored under:

```text
lepton_pseudorapidity_work/
└── cross_check_weight0/
    └── outputs/
```

These compare the original nominal Pythia correction with the new high statistics weight 0 Pythia correction.

The analysis includes both:

```text
Pythia W+
Pythia W-
```

The new samples are used as a cross check and, in the final treatment, as the Pythia nominal correction.

The raw cross check ROOT files are stored separately under:

```text
cross_check_weight0/cross_check_ROOT_files/
```

These files can be large and may not be suitable for normal GitHub storage. If they are not present in a fresh clone, they need to be copied or downloaded separately before rerunning `cross_check_weight0.py`.

If the cross check CSV outputs are already present, the raw ROOT files are not needed simply to inspect the existing results.

---

## Relative systematic uncertainty outputs

Produced by:

```text
weights_relative_uncertainties/build_relative_uncertainties.py
```

Stored under:

```text
lepton_pseudorapidity_work/
└── weights_relative_uncertainties/
    └── outputs/
        ├── csv/
        └── root/
```

These are the main outputs for part a) of the revised uncertainty treatment.

For each systematic weight \(k\),

\[
\delta_i^{(k)}
=
\frac{
C_i^{(k)}-C_i^{(0)}
}{
C_i^{(0)}
}.
\]

The nominal correction \(C_i^{(0)}\) and the varied correction \(C_i^{(k)}\) are calculated from the same weighted Monte Carlo event sample.

This means the comparison uses the statistical correlation between the nominal and systematic weights.

The final relative sources are:

\[
\delta_i^{\mathrm{scale}},
\]

\[
\delta_i^{\mathrm{PDF}},
\]

\[
\delta_i^{\mathrm{shower}},
\]

\[
\delta_i^{\mathrm{model}}.
\]

Scale, shower and model use an envelope.

PDF uses an RMS.

The CSV outputs exist for:

```text
Pythia W+
Pythia W-
Herwig W+
Herwig W-
```

If these files already exist, `build_relative_uncertainties.py` does not need to be rerun simply to make the final plots.

---

# Final Pythia correction

The final Pythia treatment uses:

```text
new high statistics weight 0 samples
    -> nominal correction
    -> statistical uncertainty
```

combined with:

```text
old weighted samples
    -> relative scale uncertainty
    -> relative PDF uncertainty
    -> relative shower uncertainty
    -> relative model uncertainty
```

The relative uncertainties are applied to the new nominal correction using

\[
\sigma_{i,X}
=
|C_i^{\mathrm{new}}|
\delta_{i,X}.
\]

The total systematic uncertainty is

\[
\sigma_{i,\mathrm{syst}}
=
\sqrt{
\sigma_{i,\mathrm{scale}}^2
+
\sigma_{i,\mathrm{PDF}}^2
+
\sigma_{i,\mathrm{shower}}^2
+
\sigma_{i,\mathrm{model}}^2
}.
\]

The total uncertainty is

\[
\sigma_{i,\mathrm{total}}
=
\sqrt{
\sigma_{i,\mathrm{stat}}^2
+
\sigma_{i,\mathrm{syst}}^2
}.
\]

This stage is produced by:

```text
weights_relative_uncertainties/
└── apply_relative_uncertainties_to_new_nominal.py
```

Run with:

```bash
python3 lepton_pseudorapidity_work/weights_relative_uncertainties/apply_relative_uncertainties_to_new_nominal.py --open
```

Outputs are stored under:

```text
weights_relative_uncertainties/
└── outputs/
    └── final_new_nominal/
        ├── csv/
        ├── root/
        └── interactive/
```

The final nominal replacement is currently Pythia only because the new high statistics samples are Pythia samples.

The relative systematic study itself still contains both Pythia and Herwig.

---

# Interactive plots

## Odd and even plots

If the odd and even CSV outputs already exist:

```bash
python3 lepton_pseudorapidity_work/even_odd_uncertainties/plot_even_odd_interactive.py
```

The corresponding HTML can also be opened using:

```bash
python3 lepton_pseudorapidity_work/even_odd_uncertainties/open_even_odd_interactive.py
```

---

## Weight 0 cross check

Run the cross check itself only if the raw ROOT files have changed:

```bash
python3 lepton_pseudorapidity_work/cross_check_weight0/cross_check_weight0.py
```

Open the stored interactive output with:

```bash
python3 lepton_pseudorapidity_work/cross_check_weight0/open_cross_check_interactive.py
```

---

## Relative systematic uncertainty plots

If the relative uncertainty CSV files already exist:

```bash
python3 lepton_pseudorapidity_work/weights_relative_uncertainties/plot_relative_uncertainties_interactive.py --open
```

This creates interactive views of:

```text
Pythia W+
Pythia W-
Herwig W+
Herwig W-
```

and allows the individual scale, PDF, shower and model weight responses to be inspected.

---

# Combined dashboard

The recommended entry point for looking through the complete project is:

```bash
python3 lepton_pseudorapidity_work/display_all_findings.py --open
```

This does not rerun the ROOT event processing.

Instead it reads the existing CSV outputs and combines the findings into one interactive HTML file.

The dashboard contains:

```text
1. Final Pythia result
2. Original correction factors
3. Odd/even statistical validation
4. New high statistics weight 0 cross check
5. Relative systematic uncertainty decomposition
6. Individual systematic weight responses
7. Pythia versus Herwig comparison
8. Method and conclusions
```

Dropdown menus are used where appropriate.

Multi-panel plots use synchronised axes to allow direct comparison between channels.

Plot labels use proper rendered subscripts and symbols rather than internal variable names.

Each dashboard screen includes an explanatory text box describing what is being plotted and how it should be interpreted.

---

# Full rerun order

A complete rerun is only needed if the underlying ROOT samples, cuts, weight definitions or binning have changed.

The full order is:

## 1. Build weighted histograms

```bash
python3 lepton_pseudorapidity_work/MTWcut_make_histograms_etalepton.py
```

This is one of the heavier stages because it loops over the event samples.

Do not rerun it unnecessarily if the histogram ROOT outputs already exist and the event selection has not changed.

---

## 2. Build the original corrections

```bash
python3 lepton_pseudorapidity_work/MTWcut_build_corrections_etalepton.py
```

This reads the stored histogram ROOT files.

---

## 3. Run the odd and even statistical check

```bash
python3 lepton_pseudorapidity_work/even_odd_uncertainties/even_odd_uncertainty.py
```

This processes the events again because the samples have to be split into odd and even entries.

Only rerun this if the underlying samples, selection or split definition has changed.

---

## 4. Run the new weight 0 Pythia cross check

```bash
python3 lepton_pseudorapidity_work/cross_check_weight0/cross_check_weight0.py
```

Only rerun this if new cross check ROOT files have been added or the selection has changed.

---

## 5. Build the relative systematic uncertainties

```bash
python3 lepton_pseudorapidity_work/weights_relative_uncertainties/build_relative_uncertainties.py
```

This reads the stored weighted histogram ROOT files.

It does not need to loop over all original events again.

---

## 6. Apply the relative uncertainties to the new Pythia nominal correction

```bash
python3 lepton_pseudorapidity_work/weights_relative_uncertainties/apply_relative_uncertainties_to_new_nominal.py
```

This reads stored CSV outputs.

It is therefore quick to rerun.

---

## 7. Build the final dashboard

```bash
python3 lepton_pseudorapidity_work/display_all_findings.py --open
```

This only reads analysis outputs and builds HTML.

It is also quick to rerun.

---

# Minimal rerun guide

Use this table to decide what needs to be rerun.

| What changed? | What needs to be rerun? |
| --- | --- |
| Nothing, only want to inspect results | `display_all_findings.py` only |
| Only want to remake interactive plots | Plotting script or `display_all_findings.py` |
| New Pythia weight 0 ROOT fragments were added | `cross_check_weight0.py`, then `apply_relative_uncertainties_to_new_nominal.py`, then dashboard |
| Relative uncertainty code changed | `build_relative_uncertainties.py`, then final nominal script, then dashboard |
| Only final combination code changed | `apply_relative_uncertainties_to_new_nominal.py`, then dashboard |
| Histogram correction builder changed | Correction builder and downstream stages |
| Event cuts changed | Histogram generation and every dependent stage |
| Eta binning changed | Histogram generation and every dependent stage |
| Systematic weight definitions changed | Histogram generation if different weights are required, then correction and uncertainty stages |
| Raw Monte Carlo samples changed | Any event-level stages that use those samples |
| Odd/even split definition changed | `even_odd_uncertainty.py` and its plots |
| Only README or documentation changed | No analysis scripts |

---

# Which scripts are expensive?

The scripts that read and loop over event ROOT trees are the ones that should not be rerun unnecessarily.

These include:

```text
MTWcut_make_histograms_etalepton.py
even_odd_uncertainty.py
cross_check_weight0.py
```

The following stages mainly read outputs that have already been generated and are much quicker:

```text
MTWcut_build_corrections_etalepton.py
build_relative_uncertainties.py
plot_relative_uncertainties_interactive.py
apply_relative_uncertainties_to_new_nominal.py
display_all_findings.py
```

The exact running time depends on the local system and the number of input files.

---

# Data and large ROOT files

Large Monte Carlo ROOT files should not normally be duplicated in Git unless there is a specific reason to do so.

The main repository is intended to store:

```text
analysis code
small CSV outputs
plots
documentation
selected lightweight generated outputs
```

Large input ROOT samples may need to be stored externally.

For the new high statistics cross check this especially applies to:

```text
lepton_pseudorapidity_work/
└── cross_check_weight0/
    └── cross_check_ROOT_files/
```

If a fresh clone contains the derived CSV outputs but not the raw ROOT samples, the existing results can still be inspected.

The raw files are only required if the cross check itself needs to be regenerated.

---

# Sample definitions

The current particle and parton pairings are:

```text
Pythia W+
    particle: WCPy8plus
    parton:   WCPyPartonplus

Pythia W-
    particle: WCPy8minus
    parton:   WCPyPartonminus

Herwig W+
    particle: WCH7plus
    parton:   WCHPartonplus

Herwig W-
    particle: WCH7minus
    parton:   WCHPartonminus
```

The correction is always constructed as

\[
C =
\frac{\mathrm{parton}}
     {\mathrm{particle}}.
\]

---

# Weight groups

The nominal weight is:

```text
weight 0
```

The current scale weights are:

```text
1
112
215
226
237
248
259
270
```

The model weights are:

```text
292
293
314
315
```

The shower weights are:

```text
294 to 313
316
317
```

Weight 318 is excluded.

PDF weight 281 is excluded from the PDF uncertainty set.

The exact weight definitions are implemented in the analysis scripts and should be treated as the source of truth if this README and the code ever differ.

---

# Statistical uncertainty

The nominal weighted histogram statistical uncertainty is based on

\[
\sigma_N =
\sqrt{
\sum_j w_j^2
}.
\]

For the correction

\[
C_i =
\frac{
N_i^{\mathrm{parton}}
}{
N_i^{\mathrm{particle}}
},
\]

the original treatment propagates the particle and parton statistical uncertainties through the ratio.

The odd and even event study is an independent stability check on the scale of this propagated statistical uncertainty.

It is not used as the final statistical error estimator.

---

# Relative systematic treatment

The revised systematic uncertainty method uses each varied weight relative to the nominal weight 0 result from the same event sample.

For weight \(k\):

\[
\delta_i^{(k)}
=
\frac{
C_i^{(k)}
-
C_i^{(0)}
}{
C_i^{(0)}
}.
\]

This is useful because the varied and nominal corrections are statistically correlated.

Common fluctuations in the underlying event sample therefore largely cancel when the relative variation is formed.

For scale, shower and model:

\[
\delta_i^X =
\max_k
\left|
\delta_i^{(k)}
\right|.
\]

For PDF:

\[
\delta_i^{\mathrm{PDF}}
=
\sqrt{
\frac{1}{N}
\sum_k
\left(
\delta_i^{(k)}
\right)^2
}.
\]

---

# Git workflow

The normal workflow used for this project is:

```bash
git status
git add .
git commit -m "Describe your changes"
git push
```

Before committing, check `git status` and make sure large raw ROOT files are not being added unintentionally.

---

# Opening the project in VS Code

From PowerShell:

```powershell
code --remote wsl+Ubuntu-24.04 /mnt/c/Users/dugar/uncertainty_decomposition
```

A Windows batch file can also contain:

```bat
@echo off
code --remote wsl+Ubuntu-24.04 /mnt/c/Users/dugar/uncertainty_decomposition
exit
```

---

# Recommended workflow for a reviewer

Someone reviewing the repository does not need to reproduce the whole Monte Carlo analysis immediately.

The recommended order is:

1. Read this README.
2. Open the combined interactive dashboard.
3. Inspect the CSV outputs if exact numerical values are required.
4. Inspect the analysis scripts to check the implementation.
5. Rerun only the relevant downstream script if a plot or final combination needs to be regenerated.
6. Rerun the event-level ROOT processing only if the input samples or physics selection need to be changed.

This keeps the analysis reproducible without requiring every expensive stage to be repeated whenever the repository is opened.

---

# Current interpretation of the analysis

The odd and even split provides a check that the propagated statistical uncertainty is of the correct scale.

The new high statistics Pythia weight 0 samples provide a compatible nominal correction with reduced statistical fluctuations.

The systematic uncertainties are therefore extracted as relative variations around weight 0 from the original weighted samples.

For Pythia, these relative systematic uncertainties are transferred onto the new high statistics nominal correction.

At present the new nominal replacement is Pythia only because equivalent new high statistics Herwig weight 0 samples are not part of the current cross check dataset.

The original and relative Herwig results remain available for comparison.

---

# Main files to look at

For most users, the most useful files are:

```text
README.md

lepton_pseudorapidity_work/
├── display_all_findings.py
│
├── all_findings_dashboard/
│   └── strange_proton_uncertainty_dashboard.html
│
├── MTWcut_build_corrections_etalepton_csv/
│
├── even_odd_uncertainties/
│   └── outputs/
│
├── cross_check_weight0/
│   └── outputs/
│
└── weights_relative_uncertainties/
    └── outputs/
```

If you only want to understand the current results, start with the dashboard and the stored CSV files rather than rerunning the ROOT analysis.
