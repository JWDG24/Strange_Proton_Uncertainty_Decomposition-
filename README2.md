# W+c Lepton-Pseudorapidity Analysis — Final Corrections and Validation

## Purpose of this update

This document describes the changes made to the lepton-pseudorapidity uncertainty-decomposition workflow after a review of the event selection, correction-building chain and statistical treatment.

The final updated analysis is contained in:

```text
lepton_pseudorapidity_work_FINAL/
```

The aim of the update was to make the full W+c correction-factor calculation explicit, internally validated and reproducible from the histogram-building stage through to the final interactive dashboard.

The observable remains the absolute lepton pseudorapidity,

\[
|\eta_\ell|,
\]

with the parton-to-particle correction factor in bin \(i\)

\[
C_i =
\frac{N_i^{\mathrm{parton}}}
     {N_i^{\mathrm{particle}}}.
\]

---

# 1. Main issue identified

During review of the analysis, it was noticed that the version of the histogram-building workflow preserved in the repository did not explicitly apply the jet/charm requirements expected for the W+c selection.

The public repository history did not contain a recoverable earlier implementation of those jet requirements, so the final workflow was rebuilt to make the full selection explicit using the branches available in the ntuples.

The ntuples contain, among others,

```text
jet_pt
jet_eta
jet_phi
jet_E
jet_charge
wcharm_charge
```

and the final implementation uses `jet_charge` as the charm-identification/sign branch.

The corrected event selection is now applied consistently at particle and parton level before any correction factors or uncertainties are built.

---

# 2. Final fiducial selection

The final analysis requires

\[
E_T^{\mathrm{miss}} > 25~\mathrm{GeV},
\]

\[
p_T^\ell > 20~\mathrm{GeV},
\]

\[
|\eta_\ell| < 2.5,
\]

\[
m_T^W > 40~\mathrm{GeV},
\]

together with the jet requirements

\[
p_T^{\mathrm{jet}} > 25~\mathrm{GeV},
\]

\[
|\eta_{\mathrm{jet}}| < 2.5.
\]

Exactly **one fiducial charm-identified jet** is required, with

```text
jet_charge != 0
```

used as the charm-identification condition.

Additional fiducial non-charm jets are allowed.

The OS−SS event sign is defined using the W and charm signs:

```text
opposite-sign W/charm  -> +1
same-sign W/charm      -> -1
```

This definition is now stored in the ROOT metadata so downstream stages can verify that they are reading histograms produced with the intended selection.

---

# 3. Histogram-building stage

The main histogram script is

```text
lepton_pseudorapidity_work_FINAL/
└── MTWcut_make_histograms_etalepton.py
```

This stage was updated to:

- apply the full lepton, missing-momentum, transverse-mass and jet/charm selection;
- use ROOT `Sumw2` for weighted statistical uncertainties;
- retain the original weighted-event information;
- use only real weights present in the samples;
- write output with `RECREATE`;
- store analysis metadata in every output ROOT file.

Important metadata now includes:

```text
Observable
SampleName
EffectiveWeightIndex
MTWCutApplied
MTWCutGeV
JetSelectionApplied
JetPtMinGeV
JetAbsEtaMax
RequiredCharmJets
CharmJetIDBranch
CharmJetIDDefinition
OSSSWeightDefinition
EventSelection
```

This allows later scripts to reject stale histogram files that were not produced with the final selection.

---

# 4. Correction-factor construction

The correction-building stage is

```text
MTWcut_build_corrections_etalepton.py
```

For every common particle/parton weight,

\[
C_i^{(k)} =
\frac{N_{i,\mathrm{parton}}^{(k)}}
     {N_{i,\mathrm{particle}}^{(k)}}.
\]

The nominal correction is weight 0.

The statistical uncertainty is the propagated ROOT/Sumw2 ratio uncertainty.

Before using any histogram, the script validates the stored event-selection metadata. This prevents an old pre-selection output from being mixed into the final chain.

The script also verifies:

- matching particle/parton binning;
- a non-zero particle denominator;
- the presence of weight 0;
- that only weights common to particle and parton samples are used.

---

# 5. Systematic uncertainty definitions

The retained weight groups are

### Scale

```text
1, 112, 215, 226, 237, 248, 259, 270
```

### Model

```text
292, 293, 314, 315
```

### Shower

```text
294-313, 316, 317
```

### Exclusions

```text
weight 318
PDF weight 281
```

PDF variations are identified from weight names beginning with

```text
MUR1.0_MUF1.0_PDF
```

For scale, shower and model uncertainties the absolute correction-factor envelope is used:

\[
\sigma_{X,i}
=
\max_k
\left|
C_i^{(k)} - C_i^{(0)}
\right|.
\]

For PDF variations the RMS is used:

\[
\sigma_{\mathrm{PDF},i}
=
\sqrt{
\frac{1}{N}
\sum_k
\left(
C_i^{(k)} - C_i^{(0)}
\right)^2
}.
\]

The total uncertainty is

\[
\sigma_{\mathrm{total},i}
=
\sqrt{
\sigma_{\mathrm{stat},i}^2
+
\sigma_{\mathrm{scale},i}^2
+
\sigma_{\mathrm{PDF},i}^2
+
\sigma_{\mathrm{shower},i}^2
+
\sigma_{\mathrm{model},i}^2
}.
\]

No missing systematic weights are fabricated.

For the available Herwig weighted samples, common shower/model variations are not present. These components are therefore unavailable in this workflow and remain zero in the stored decomposition rather than being invented.

---

# 6. Relative-systematic treatment

The relative-systematic stage is

```text
weights_relative_uncertainties/
└── build_relative_uncertainties.py
```

Each systematic response is evaluated relative to weight 0 from the **same event sample**:

\[
\Delta_i^{(k)}
=
\frac{
C_i^{(k)} - C_i^{(0)}
}{
C_i^{(0)}
}.
\]

The final relative uncertainties are

\[
\Delta_{\mathrm{scale},i}
=
\max_k
|\Delta_i^{(k)}|,
\]

\[
\Delta_{\mathrm{PDF},i}
=
\sqrt{
\frac{1}{N}
\sum_k
\left(
\Delta_i^{(k)}
\right)^2
},
\]

with equivalent envelope definitions for shower and model.

This same-sample construction avoids treating nominal and varied generator weights as independent statistical measurements.

The relative-systematic stage also reconstructs the weight-0 correction and compares it bin-by-bin with the output of `MTWcut_build_corrections_etalepton.py`.

For all four channels the nominal cross-check closed exactly:

```text
max |ΔC0| = 0
```

in the completed run.

---

# 7. Statistical validation

An independent odd/even split is retained as a statistical stability check:

```text
even_odd_uncertainties/
└── even_odd_uncertainty.py
```

Events are divided by their original ROOT entry number into odd and even subsamples while keeping the same physics selection.

The quantities compared include

\[
C_{\mathrm{all}},
\qquad
C_{\mathrm{odd}},
\qquad
C_{\mathrm{even}},
\]

and the diagnostic

\[
\frac{1}{2}
\left|
C_{\mathrm{odd}} - C_{\mathrm{even}}
\right|.
\]

This split is used as a validation of the statistical behaviour.

It does **not** replace the propagated ROOT/Sumw2 statistical uncertainty used in the final result.

---

# 8. High-statistics Pythia investigation

A separate investigation was performed using the files that had been treated as multiple high-statistics Pythia fragments.

A fragment-independence check found extensive duplicated event content between those files.

Examples from that investigation included:

- repeated identical W+ particle fragments;
- all tested W+ parton fragments having identical content;
- repeated identical W− particle fragments;
- all tested W− parton fragments having identical content.

Because those files are not independent statistical samples, combining them as independent fragments would artificially reduce the apparent statistical uncertainty and can also change the effective particle-level weighting.

For that reason the exploratory high-statistics result is **not used as the official nominal or statistical uncertainty**.

The final result instead remains based on the validated weighted-sample weight-0 correction.

The exploratory scripts are retained only as diagnostic work documenting why that route was not adopted.

---

# 9. Final conservative combination

The combination stage is

```text
weights_relative_uncertainties/
└── apply_relative_uncertainties_to_new_nominal.py
```

Despite the historical filename, the final version does **not** replace the nominal with the duplicated high-statistics result.

For each channel,

\[
C_{\mathrm{final},i}
=
C_i^{(0)}
\]

from the validated weighted sample.

Relative systematic uncertainties are converted back to absolute uncertainties using

\[
\sigma_{X,i}
=
|C_{\mathrm{final},i}|
\Delta_{X,i}.
\]

The final combination script independently checks that the reconstructed absolute scale, PDF, shower, model and total uncertainties reproduce the outputs of the correction-building stage.

The completed run gave agreement at approximately \(10^{-9}\)–\(10^{-10}\) in absolute correction-factor uncertainty, consistent with floating-point precision.

The final result is produced for all four channels:

```text
Pythia W+
Pythia W-
Herwig W+
Herwig W-
```

and is stored under

```text
weights_relative_uncertainties/
└── outputs/
    └── final_conservative/
```

---

# 10. Interactive presentation

The completed results are presented using

```text
display_all_findings.py
```

which creates

```text
all_findings_dashboard/
└── strange_proton_uncertainty_dashboard.html
```

The dashboard contains:

1. Summary
2. Final result
3. Correction overview
4. Statistical validation
5. Relative systematic uncertainties
6. Individual systematic-weight responses
7. Pythia versus Herwig
8. Method and conclusions

The Pythia/Herwig and charge comparisons use synchronized axes where appropriate.

Each page also contains an explanation of what is being shown and how it should be interpreted.

The dashboard is presentation-only: it reads the completed CSV outputs and does not rerun the ROOT event loop.

Run it from the repository root with

```bash
python3 lepton_pseudorapidity_work_FINAL/display_all_findings.py --open
```

---

# 11. Final analysis chain

The final workflow is

```text
1. MTWcut_make_histograms_etalepton.py
        |
        v
2. MTWcut_build_corrections_etalepton.py
        |
        +--------------------------+
        |                          |
        v                          v
3. even_odd_uncertainty.py   4. build_relative_uncertainties.py
                                   |
                                   v
                         5. apply_relative_uncertainties_to_new_nominal.py
                                   |
                                   v
                   6. plot_relative_uncertainties_interactive.py
                                   |
                                   v
                         7. display_all_findings.py
```

The `cross_check_weight0` and event-matching/fragment-independence scripts are retained as diagnostics rather than inputs to the official final result.

---

# 12. Important distinction between the original and final folders

The repository retains

```text
lepton_pseudorapidity_work/
```

as a record of the earlier development workflow.

The completed analysis described in this document is

```text
lepton_pseudorapidity_work_FINAL/
```

and this is the folder that should be used for the final correction factors, uncertainty decomposition and presentation dashboard.

---

# 13. Reproducing the final analysis

From the repository root, the main sequence is:

```bash
python3 lepton_pseudorapidity_work_FINAL/MTWcut_make_histograms_etalepton.py

python3 lepton_pseudorapidity_work_FINAL/MTWcut_build_corrections_etalepton.py

python3 lepton_pseudorapidity_work_FINAL/even_odd_uncertainties/even_odd_uncertainty.py

python3 lepton_pseudorapidity_work_FINAL/weights_relative_uncertainties/build_relative_uncertainties.py

python3 lepton_pseudorapidity_work_FINAL/weights_relative_uncertainties/apply_relative_uncertainties_to_new_nominal.py

python3 lepton_pseudorapidity_work_FINAL/weights_relative_uncertainties/plot_relative_uncertainties_interactive.py

python3 lepton_pseudorapidity_work_FINAL/display_all_findings.py --open
```

If the histogram and CSV outputs already exist, the dashboard can be regenerated without rerunning the ROOT event loop.

---

# 14. Summary of the update

The final update therefore makes the following substantive changes:

- restores explicit W+c jet/charm event selection;
- applies that selection consistently at particle and parton level;
- stores and validates selection metadata in ROOT outputs;
- prevents stale histogram files from entering later stages;
- uses only particle/parton weights that genuinely exist in both samples;
- retains ROOT/Sumw2 propagated statistical uncertainties;
- evaluates systematic variations relative to the same-sample weight-0 correction;
- retains envelope and RMS systematic definitions;
- validates nominal and uncertainty closure between analysis stages;
- uses the odd/even split as a statistical stability diagnostic;
- identifies duplicated content in the exploratory high-statistics fragment set;
- excludes that duplicated high-statistics treatment from the official result;
- produces one internally consistent four-channel final result;
- provides synchronized interactive plots and a presentation-ready final dashboard.

The resulting `lepton_pseudorapidity_work_FINAL` directory is the final documented version of the lepton-pseudorapidity uncertainty-decomposition analysis.
