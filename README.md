# CPTG

## Curvature Polarization Transport Gravity

---

## Contents

- [Core CPTG Papers](#start-here-core-cptg-papers)
- [Available Tools](#available-tools)
- [Overview](#overview)
- [Current Research Status](#current-research-status)
- [Repository Contents](#what-this-repository-contains)
- [Reproducing the Public Benchmarks](#reproducing-the-public-benchmarks)
- [CPTG SPARC Browser Workbench](#galaxy-scale-test-cptg-sparc-browser-workbench)
- [Curvature-Weighted Structural Mode Index](#curvature-weighted-structural-mode-index)
- [Outer-Slope Convergence Test](#outer-slope-convergence-test)
- [Bullet Cluster Benchmark](#cluster-merger-test-bullet-cluster-benchmark)
- [Cluster Active-Gate Test](#cluster-scale-active-gate-test-accept-and-x-cop)
- [Nuclear-Scale Reaction Program](#nuclear-scale-reaction-program-deuterium-proton-radiative-capture)
- [Cosmology and Comparison-Layer Tests](#cosmology-and-comparison-layer-tests)
- [CPTG Research Position](#cptg-research-position)
- [Citation](#citation)

---

## Start Here: Core CPTG Papers

- **[Curvature Polarization Transport Gravity: A Unified Geometric Framework for Cosmic Structure and Expansion](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/CPTG_Unified_Geometric_Framework_Cosmic_Structure_Expansion.pdf)**  
  Primary CPTG theory paper. This manuscript lays out the unified geometric framework: baryon-sourced curvature polarization, curvature transport, the cosmic/structure expansion connection, galaxy and cluster limits, and the broader comparison-layer program.

- **[CPTG Geometric Pi Branch Comparison Coordinates](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/CPTG_Geometric_Pi_Branch_Comparison_Coordinates.pdf)**  
  Comparison-coordinate guide for the locked geometric pi branch. This paper explains how CPTG-native quantities are mapped into observational coordinates for CMB, BAO, BBN, supernova, growth, and DESI-style comparison layers without treating those observational coordinates as the theory itself.

- **[The Science Behind CPTG: A Geometric Alternative to Dark Matter, Dark Energy, and MOND](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/The_Science_Behind_CPTG.pdf)**  
  A public-facing introduction to Curvature Polarization Transport Gravity. This paper explains what makes CPTG different from dark matter, dark energy, and MOND-style approaches: baryon-sourced curvature polarization, curvature transport, structural modes, active gates, and scalable comparison coordinates derived from one geometric framework rather than sector-by-sector tuning.
  
---

## Available Tools

The repository provides two complementary public access paths:

- **CPTG academic package** — the compact reproducibility archive for the core SPARC galaxy and Bullet Cluster reduced-limit benchmarks.
- **CPTG SPARC Browser Workbench v1.11.9** — a larger standalone application for users focused on interactive SPARC rotation-curve analysis.

---

### CPTG SPARC Browser Workbench

The **CPTG SPARC Browser Workbench v1.11.9** is a local browser application available as a Windows release package and as a Python application for supported Windows and Linux environments.

It analyzes the included SPARC galaxy rotation-curve files with CPTG and MOND and supports single-galaxy and batch analysis, SPARC metadata filtering, plots, tables, summaries, and downloadable result bundles.

[View the interface](images/CPTG_SPARC_Browser_Workbench.png) · [Download v1.11.9](https://github.com/CLG2025/CPTG/releases/tag/v1.11.9)

---

### CPTG Pi-Bridge

CPTG Pi-Bridge is a local [research workbench](images/CPTG-Pi-Bridge-Local-Workbench.png) currently in beta development. It is designed to load public astronomy and cosmology datasets, select a CPTG comparison branch, run the audit engine, review results, and export reproducible validation packages.

Its translation layer uses the geometric-π comparison-coordinate method defined in the CPTG π-branch paper. CPTG-native quantities remain in their native geometric branch before being projected into conventional CMB, BAO, BBN, supernova, growth, and DESI-style comparison coordinates.

---

## Overview

*Curvature Polarization Transport Gravity* (CPTG) is an active geometric-gravity framework exploring whether effects commonly attributed to dark matter can arise from the response of spacetime curvature to ordinary baryonic matter.

CPTG is built around two linked mechanisms:

* **Curvature polarization**, which modifies the effective gravitational response according to the strength and structure of the field.
* **Curvature transport**, which allows organized curvature to be redistributed directionally in dynamic systems.

The framework is tested across galaxy rotation curves, cluster-merger lensing, relaxed galaxy clusters, cosmology-facing comparison layers, and a completed nuclear-scale radiative-capture extension to the CPTG lithium solution. The repository provides the current theory papers, CPTG SPARC Browser Workbench, included data, public figures, and reproducible validation materials for inspection, testing, and criticism.

---

## Current Research Status

CPTG is being developed as a geometric framework with reduced-limit tests and comparison-layer audits. The public status is best read by separating reproducible benchmarks, coordinate-layer validations, diagnostic passes, and active theory-development work.

| Area | Current CPTG status | Claim level |
|---|---|---|
| SPARC galaxy rotation curves | Public reduced-limit SPARC test available through the compact academic package and the interactive browser workbench | Reproducible galaxy-scale benchmark |
| Bullet Cluster merger plane | Public reduced merger-plane curvature-transport/lensing reconstruction | Reproducible cluster-merger benchmark |
| Cluster active-gate apertures | Same-aperture cluster-response tests using baryonic loading, support temperature, redshift, and aperture radius | Diagnostic cluster-scale active-gate and X-COP consistency [pass](#Cluster-Scale-Active-Gate-Test-ACCEPT-and-X-COP) |
| Nuclear-scale deuterium-proton radiative capture | Native CPTG `D(p,γ)³He` reaction theory carries source geometry through a coherent amplitude, S-factor normalization, p+D rate branch, BBN-network propagation, and the locked A = 7 lithium transport gate. | Finished reaction-specific [extension to the lithium solution](#nuclear-scale-reaction-program-deuterium-proton-radiative-capture): source-to-amplitude closure, S-factor comparison, network-coupled rate propagation, PRyMordial/AlterBBN validation, and observed D/H, Yp, He3/H, and Li7/H residual ledger |
| Pantheon+ supernova distances | Full-covariance relative distance-shape comparison with marginalized intercept | Distance-shape [pass](#Pantheon-Supernova-Distance-Shape-Test), not an H0 calibration claim |
| BBN abundance and lithium tests | Transported BBN coordinate, locked live A = 7 lithium gate, PRyMordial/AlterBBN source-network validation, and the native D(p,γ)³He reaction-rate extension | Coordinate-layer and source-network [validation](#bbn-abundance-and-lithium-source-network-tests): deuterium and helium remain within their comparison bands while the A = 7 gate maps the raw standard-BBN lithium excess into the observed primordial lithium range |
| Weak-lensing S8 | Compressed comparison against representative weak-lensing and CMB S8 anchors | Diagnostic [pass](#Weak-Lensing-S8-Comparison), not a full shear likelihood |
| CMB comparison-map closure | Locked geometric-pi CMB branch tested against real Planck/WMAP temperature-map products and null controls | Real-map [comparison-map closure](#CMB-Comparison-Map-Closure) pass |
| CMB Route B Option 1 bridge | Fixed amplitude-level curvature-transport bridge tested through CMB spectrum and Planck likelihood-coordinate plumbing | Geometry-first comparison-coordinate bridge [validation](#CMB-Route-B-Option-1-Curvature-Transport-Bridge) |
| DESI compressed ShapeFit and BAO | Compressed-coordinate and ruler-wrapper diagnostics | Coordinate-level [support](#DESI-DR1-Compressed-ShapeFit-and-BAO-Quarter-Ruler), not full raw DESI validation |
| Horizon and Hubble-tension mechanisms | Structural articles mapping CPTG-native branches into observational comparison layers | Theory [mechanism](#Hubble-Tension-Bridge) and derivation-stage interpretation |

Claim levels are used consistently throughout this README:

- **Benchmark** — a reproducible reduced-limit calculation compared with data.
- **Diagnostic pass** — a result compatible with the stated controls.
- **Coordinate-layer validation** — a tested observational mapping or likelihood interface.
- **Closure pass** — agreement within a declared fixed-branch closure protocol.
- **Anchored comparison** — a dimensional comparison whose normalization is explicitly anchored to a stated observable, with independent rows treated as cross-checks rather than as free refits.
- **Theory mechanism** — a derived interpretation connected to a dedicated comparison or audit layer.

---

## What This Repository Contains

This repository contains the public academic package for CPTG, including:

- current CPTG theory manuscripts and research notes,
- reduced benchmark scripts for galaxy and cluster tests,
- supporting public data packages and metadata when included,
- comparison-layer scripts and audit outputs when publicly included,
- nuclear reaction-rate theory material, source-network validation records, and reproducibility material when publicly included,
- CMB source/data availability notes and strict rerun file lists,
- figures, summaries, and reproducibility material.

The recommended compact reproducibility download is **`CPTG_academic_package.zip`**, located in the **`/archive/`** folder. It preserves the public core benchmark environment for the SPARC galaxy and Bullet Cluster reduced-limit tests.

The larger **CPTG SPARC Browser Workbench v1.11.9** is distributed separately for interactive rotation-curve analysis. Additional Python files in the repository should be treated as version upgrades, development variants, or replacement implementations unless a specific package README states otherwise.

---

## Reproducing the Public Benchmarks

The public benchmarks are intended to be inspectable and reproducible.

1. Download or clone the repository.
2. Open the **`/archive/`** folder.
3. Use **`CPTG_academic_package.zip`** for the public benchmark package.
4. Extract the archive into a working folder.
5. Run the benchmark scripts with Python 3.
6. Compare generated outputs against the included figures and summary files.

The package root directory may be renamed freely. Reproducibility depends on preserving the internal relative layout, or on passing explicit input/output paths when running scripts. Planck and WMAP FITS products are not bundled with CMB map-closure packages; they must be placed in the documented data location or supplied by command-line path.

The main public benchmark scripts are:

| Package or tool | Purpose |
|---|---|
| `SPARC_CPTG_MOND_Benchmark.py` | Original galaxy rotation-curve benchmark against SPARC data. |
| `CPTG_Bullet_Cluster_Merger.py` | Reduced merger-plane curvature-transport/lensing benchmark. |
| `CPTG_ClusterActiveGate_IntegratedTool_v0_5.py` | Single-aperture and aperture-ladder cluster-response calculations from baryonic loading, support temperature, redshift, and aperture. Requires the [public X-COP cluster archive](https://drive.switch.ch/index.php/s/j3WUOYXWgv9Jbnz/download). |
| `CPTG_MOND_Upsilon_SPARC_Benchmark.py` | MOND/CPTG comparison with stellar mass-to-light freedom. |
| `CPTG-CMB.zip` | CMB comparison-map closure scripts for Planck/WMAP component maps, split maps, smoothing/mask controls, visual comparisons, summary reports, and null-envelope controls. |

---

## Galaxy-Scale Test: CPTG SPARC Browser Workbench

The **CPTG SPARC Browser Workbench v1.11.9** provides a local browser interface for testing CPTG and MOND against SPARC galaxy rotation-curve data.

The standalone workbench [package](https://github.com/CLG2025/CPTG/releases/tag/v1.11.9) includes the SPARC galaxy data needed to begin running analyses immediately. It supports:

- single-galaxy analysis,
- multi-galaxy analysis,
- full SPARC directory scans,
- SPARC metadata files,
- primary-sample filtering,
- CPTG and MOND fit comparisons,
- averaged normalized rotation curves,
- averaged RAR scatter vs radius,
- galaxy-level fit and structural-mode summaries.

All calculations are performed locally. Each galaxy is solved independently before aggregate results and plots are generated.

For aggregate plots, each galaxy is normalized independently and interpolated onto a shared normalized radial grid. Each galaxy receives equal weight at each grid location, and the outer endpoint at `r / r_max = 1` is retained.

SPARC data source: Lelli, McGaugh, and Schombert, *The Astronomical Journal* 152, 157 (2016), “SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves.”

The figure below summarizes the averaged SPARC results produced by the workbench.

![CPTG SPARC Browser Workbench summary showing the average normalized RAR scatter vs radius and the average normalized SPARC rotation curve.](images/combined_workbench_plots_side_by_side.png)

<sup>Figure: CPTG SPARC Browser Workbench averaged results for the full 175-galaxy SPARC run. Primary-sample metadata filtering was not applied to this figure.</sup>

---

## Curvature-Weighted Structural Mode Index

The Curvature-Weighted Structural Mode Index, **N**, is a CPTG diagnostic derived from the solved acceleration field.

It measures how curvature support is organized inside a galaxy:

**N = R / λ**

where:

* **R** is the outer solved radius.
* **λ** is the curvature-weighted structural scale.

The workbench translates the continuous mode value into a named **CSMI Type**:

| Structural Mode N | CSMI Type             |
| ----------------: | --------------------- |
|          N ≤ 1.45 | Dwarf Irregular       |
|   1.45 < N ≤ 1.75 | Magellanic Irregular  |
|   1.75 < N ≤ 1.95 | LSB Dwarf Disk        |
|   1.95 < N ≤ 2.15 | Transition Dwarf      |
|   2.15 < N ≤ 2.35 | LSB Spiral            |
|   2.35 < N ≤ 2.55 | Very Late Spiral      |
|   2.55 < N ≤ 2.80 | Late Spiral           |
|   2.80 < N ≤ 3.05 | Intermediate Spiral   |
|   3.05 < N ≤ 3.30 | Early Spiral          |
|   3.30 < N ≤ 3.55 | Bulged Spiral         |
|   3.55 < N ≤ 3.72 | Lenticular/Early Disk |
|          N > 3.72 | High-Mode Outlier     |

The CSMI Type is assigned only from the solved mode value. It is not taken from galaxy names, SPARC metadata, catalog morphology, or visual classification. As larger and more diverse galaxy samples are analyzed, future CSMI catalogs will likely expand, subdivide, or refine the named structural types to represent newly resolved mode populations. Any additional categories would remain derived from the solved CPTG mode distribution rather than being imposed from conventional morphological classifications.

Mode-filtered runs allow galaxies with similar CPTG structural organization to be evaluated as subsets of the full database.

In public-facing terms, **N** asks:

> How is the solved curvature structure organized inside this galaxy?

This makes the mode and its CSMI Type theory-derived structural diagnostics rather than conventional galaxy classifications.

---

## Outer-Slope Convergence Test

The CPTG outer-slope convergence test evaluates a theory-defined prediction of the reduced galaxy equation: once the solved field is extended beyond the outermost measured SPARC radius, the rotation-curve response should approach a stable CPTG outer-regime trend rather than drift arbitrarily.

This is an important strength of CPTG because the far-outer behavior follows from the solved curvature-polarization and transport structure. It is not independently fitted to artificial outer data points.

The purpose of the test is not to claim that current observations already measure the entire extended regime. It checks whether the reduced CPTG galaxy equation develops the stable long-range behavior predicted by the theory when continued beyond the observed rotation-curve domain.

In CPTG, weak-field galaxy outskirts should gradually approach a consistent curvature-polarization pattern. The convergence plot visualizes that prediction across the SPARC galaxy sample.

The second benchmark figure shows the stacked CPTG outer-slope convergence trend for the SPARC galaxy sample.

![CPTG outer-slope convergence in the asymptotic extension regime. The plot shows how the extended CPTG rotation-curve behavior evolves beyond the observed SPARC rotation-curve domain. The median trend approaches the predicted CPTG outer-regime behavior, while the shaded region shows the galaxy-to-galaxy spread. This figure illustrates that the extended CPTG solution approaches a stable long-range pattern rather than drifting arbitrarily outside the measured data range.](images/cptg_outer_slope_convergence.png)

<sup>Figure: CPTG outer-slope convergence in the extended galaxy-outskirts regime.</sup>

---

## Cluster-Merger Test: Bullet Cluster Benchmark

The **`CPTG_Bullet_Cluster_Merger.py`** script tests the cluster-merger limit of the theory through a reduced Bullet Cluster merger-plane implementation.

It constructs baryonic gas and galaxy components, builds curvature-polarization background fields, evolves a transported-curvature mode, and produces a normalized convergence/kappa reconstruction.

The model is scored against observed Bullet Cluster gas, galaxy, and lensing separations, including:

- Bullet-side mass-galaxy offset,
- Bullet-side mass-ICM offset,
- main-cluster north mass-ICM offset,
- main-cluster south mass-ICM offset,
- main-subclump separation,
- cluster-scale lensing separation.

The public significance of this test is that CPTG attempts to address not only galaxy rotation curves, but also dissociative cluster mergers, which are often considered strong evidence for collisionless dark matter.

---

The third benchmark figure shows how CPTG reconstructs displaced lensing structure in the Bullet Cluster merger plane.

![Normalized CPTG kappa reconstruction of the Bullet Cluster merger plane. The map shows two main convergence structures: a compact Bullet-side lensing feature on the left, displaced from the Bullet gas peak, and a larger main-cluster lensing structure on the right with north and south substructure. White contours trace the strongest reconstructed convergence regions. Markers identify Bullet and main gas peaks, galaxy peaks, lensing peaks, and main-cluster north/south lens peaks. A scale bar marks 100 kpc.](images/CPTG-Curvature-Transport-Model.png)

<sup>Figure: CPTG Bullet Cluster kappa reconstruction showing gas-lensing separation.</sup>

---

## Cluster-Scale Active-Gate Test: ACCEPT and X-COP

The [cluster-scale active-gate](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/CPTG_Cluster-Scale_Active-Gate_Extension.pdf) work extends CPTG beyond galaxy rotation curves and reduced merger-plane reconstruction into relaxed or approximately coherent galaxy-cluster apertures. The calculation asks whether a cluster aperture can be described by baryonic loading, support temperature, redshift, and aperture radius through one active curvature-response state.

For a selected aperture `R_delta`, the calculation uses gas mass, stellar/intracluster-light support where available, temperature, redshift, and aperture radius to compute the active-gate response and the predicted CPTG mass for the same aperture.

This work provides a same-aperture test of both the accuracy and structural response of the CPTG cluster active-gate law. Across the tested 12-cluster X-COP R₅₀₀ sample, the predicted CPTG masses achieved a median CPTG-to-hydrostatic mass ratio of 0.9892, corresponding to a median absolute fractional difference of approximately 1.08%. The mean absolute difference was approximately 1.19%, and the largest individual difference was approximately 2.68%. The X-COP aperture ladder further reproduced the predicted transition from closure-stable inner apertures to greater active-gate sensitivity at R₅₀₀. Independently, the ACCEPT analysis ordered 379 clean profile rows across 45 clusters into the expected closure-stable, watch, suppressed, and strongly suppressed response states. Together, these results show that the same fixed CPTG law produces percent-level same-aperture mass agreement while also recovering the theory-defined structural ordering of cluster apertures.

This result should be read as a diagnostic cluster-scale active-gate pass and an X-COP same-aperture consistency pass. It is not a claim that one single-aperture formula describes strong cluster mergers without decomposition. Strong mergers are best treated separately unless gas, stellar, temperature, and mass components can be assigned consistently to the same dynamical aperture.

---

## Nuclear-Scale Reaction Program: Deuterium-Proton Radiative Capture

CPTG now includes a native nuclear-scale extension of the [lithium solution](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/CPTG_Cosmological_Lithium_Problem.pdf) built around deuterium-proton radiative capture,

```text
²H + p → ³He + γ
```

together with the reverse photodisintegration channel,

```text
³He + γ → p + ²H
```

The purpose of the extension is to show that the same curvature-polarization and transport framework used in the cosmological lithium solution can also be carried through a reaction-specific nuclear channel. The calculation links a CPTG source-state construction to a coherent transition amplitude, an astrophysical S-factor normalization, a native p+D rate branch, and BBN-network propagation.

In this reading, the nuclear reaction theory is not a separate phenomenological fit. It supplies the D(p,γ)³He reaction-rate ingredient that sits underneath the abundance calculation and connects the lithium gate to a concrete radiative-capture channel.

### Reaction Formula and S-Factor Normalization

The source-to-S-factor layer begins with the closed CPTG reaction intensity,

```text
A_CPTG = 17.5595746398194
I_CPTG = |A_CPTG|² = 308.3386615313886
```

The completed zero-energy normalization is

```text
K_S0 = α π / (33 - 1/(7π)) eV b per dimensionless source intensity
```

or, in the support-split form used by the reaction ledger,

```text
K_S0 = α π / [3(NP3B - 1) - 1/((NP3A/8)π)] eV b
NP3A = 56
NP3B = 12
NP3 = NP3A + NP3B = 68
```

This gives

```text
K_S0 = 0.0006956649414530029 eV b
S12,CPTG(0) = K_S0 I_CPTG = 0.2145003969219307 eV b
S12,CPTG(0) / 0.2145 = 1.0000018504518913
```

The rounded `0.2145 eV b` value is therefore only a display-level shorthand. The operative CPTG default for the reaction theory is the closed-form normalization above, and dependent reaction-rate and abundance comparisons use the corresponding `1.0000018504518913` scale.

The observed comparison row is

```text
S12,obs(0) = 0.216 ± 0.010 eV b
CPTG residual = -0.0014996030780693 eV b
CPTG pull = -0.1499603078069277 sigma
result: PASS
```

The solar-Gamow comparison row is

```text
E0 = 6.64 keV
S12,CPTG(E0) = 0.2516322073981925 eV b
result: PASS within the stated benchmark band
```

### Source-State Structure

The native reaction source uses a resolved A = 3 state with 68 NP3 support indices. The support ledger contains four scalar components per support index, giving 272 support-slot scalar records and 340 coherent amplitude terms once the five current/source coefficients are applied.

```text
NP3 support indices: 68
support-slot scalar records: 68 × 4 = 272
amplitude terms: 68 × 5 = 340
```

The support-slot scalar components are

```text
T_A3 = 27.0897 MeV
V_NN = -34.8757 MeV
V_3NF = 0.0465074 MeV
E_table = -7.73945407385108 MeV
```

with the energy-balance check

```text
T_A3 + V_NN + V_3NF = -7.739492600000001 MeV
absolute residual = 0.000038526148920858816 MeV
structural-zero floor = 0.00010005000000500001 MeV
result: PASS
```

The current/source coefficient set is

```text
c1 = -1.23     GeV^-1
c3 = -4.65     GeV^-1
c4 =  3.28     GeV^-1
cD =  0.8918
cE = -0.38595
```

### Rate Branch and BBN Network Coupling

The S-factor layer is carried into a native CPTG p+D radiative-capture rate branch. That branch is then propagated through the BBN abundance calculation rather than being left as a standalone nuclear observable.

Before applying the lithium transport gate, the raw abundance projection is

```text
D/H     = 2.4794272036813933e-05
He3/H   = 1.0199932320844196e-05
Yp_BBN  = 2.4699580513634747e-01
Li7/H   = 5.3779247321287830e-10
```

The raw lithium value reproduces the standard mass-seven excess. The CPTG lithium solution then applies the locked A = 7 curvature-transport gate to the live 7Li and 7Be abundance channels,

```text
τ7 = ∫ Γ7 dt = ln(π)
survival factor = exp(-τ7) = 1/π
```

implemented as

```text
dY_Li7/dt → dY_Li7/dt - Γ7(T) Y_Li7
dY_Be7/dt → dY_Be7/dt - Γ7(T) Y_Be7
```

The gated current projection is

```text
raw-parent Li7/H = 5.377924732128783e-10
live-gated Li7/H = 1.7118494965634585e-10
live/raw ratio = 0.3183104230404215
target 1/π = 0.3183098861837907
```

Across support and width variations of the live gate, the abundance band remains stable:

```text
Li7/H range = 1.7118266510084370e-10 to 1.7121051910835793e-10
Li7 pull range = +1.04730660403 to +1.04842076433 sigma
D/H pull range = -0.98526877172 to -0.98478538709 sigma
Yp pull range = +0.665269188026 to +0.665272107275 sigma
He3/H: advisory upper-limit pass
```

### PRyMordial and AlterBBN Validation

The nuclear reaction extension is propagated through independent BBN-network environments. Both PRyMordial and AlterBBN preserve the same physical interpretation: the raw mass-seven abundance sits at the standard-BBN excess scale, and the locked CPTG A = 7 gate maps it into the observed primordial lithium band while preserving deuterium and helium consistency.

```text
PRyMordial propagation:
eta10_BBN = 5.998071834744
D/H pull = -0.019 sigma
Yp pull = +0.231 sigma
raw Li7/H = 5.266820229187348e-10
gated Li7/H = 1.6764809477031113e-10
Li7 pull after gate = +0.9059237908124455 sigma
result: PASS
```

```text
AlterBBN propagation:
eta10_BBN = 5.998071834744
D/H = 2.500e-05
D/H pull = -0.27586206896551463 sigma
Yp = 0.2471
Yp pull = +0.699999999999997 sigma
raw Li7/H = 5.291e-10
gated Li7/H = 1.6841776077984365e-10
Li7 pull after gate = +0.9367104311937463 sigma
result: PASS
```

The finished nuclear-scale result is therefore a coupled reaction-and-abundance extension: native D(p,γ)³He source geometry, coherent amplitude, S-factor normalization, p+D rate propagation, live A = 7 lithium transport, and two-code BBN validation under the same locked `1/π` survival rule.

---

## Cosmology and Comparison-Layer Tests

CPTG cosmology-facing work is organized around the distinction between CPTG-native geometric quantities and conventional observational summaries. The goal is not to force CPTG into standard parameter language, but to make controlled comparisons with quantities commonly reported from supernova, CMB, abundance, growth, and large-scale-structure analyses.

### Current Locked Geometric Pi Branch

The current locked CPTG comparison branch uses:

```text
p_C = pi
p_ac = 3 - pi/100 = 2.968584073464
G_T = p_ac / p_C = 0.944929658551
sqrt(G_T) = 0.972074924351
H0^(pi) = 69.4162507897 km s^-1 Mpc^-1
H0_CMB^CPTG = 67.4777967351 km s^-1 Mpc^-1
A_lens = 1
```

The locked CMB comparison-map row used in the current Planck/WMAP map-space closure paper is:

```text
H0 = 67.4777967351 km s^-1 Mpc^-1
omega_b h^2 = 0.022527857494
omega_c h^2 = 0.117685841620526
n_s = 0.968584073464
N_eff = 2.968584073464
A_s = 2.136283004441e-9
tau = 0.058930875934
A_lens = 1
```

These values define the fixed geometric-pi CMB branch used in the current comparison-map closure audits. The branch is locked before the map tests and is not refit to the individual CMB maps.

### CMB Comparison-Map Closure

CPTG CMB map work is organized as a [real-map comparison test](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/CPTG_CMB_Comparison_Map_Closure.pdf) between the locked geometric-pi CMB branch and public CMB map products. The current paper uses real Planck component maps, Planck split maps, and WMAP low-ell support products.

![Observed Planck SMICA vs fitted CPTG comparison map](images/fig_visual_fitted.png)

<sup>Figure: SMICA visual comparison from the CMB comparison-map closure paper. Top: observed Planck SMICA temperature map. Center: fitted CPTG comparison map. Bottom: observed-minus-fitted-CPTG residual.</sup>

The map-space procedure uses the same comparison coordinate for CPTG, the Planck envelope, and controls. It reads the temperature field from the public CMB map product, applies the documented mask, converts to microkelvin, downgrades to `Nside = 256`, removes the monopole and dipole on the valid sky, and evaluates fitted residuals under the same amplitude-plus-offset rule:

```text
T_fit(nhat) = A T_template(nhat) + B
```

The central public result is that the locked CPTG geometric-pi branch reaches near-degenerate CMB comparison-map closure with the Planck comparison envelope across the tested real-map products and controls, while generic null envelopes fail much more strongly under the same map-space procedure. The detailed RMS tables, control ladders, and null-envelope audits are contained in the dedicated CMB comparison-map closure material.

#### Original Planck and WMAP FITS Inputs

The original Planck and WMAP survey FITS maps are **not bundled** because they are large public data products. For the strict CMB comparison-map closure rerun, use only the tested FITS inputs below. Do not add optional masks, alternate survey products, or substitute component maps to the control environment.

| Test layer | Required local filename | Public source |
|---|---|---|
| Planck SMICA full map | [`COM_CMB_IQU-smica_2048_R3.00_full.fits`](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits) | Planck R3 / IRSA |
| Planck SMICA-noSZ full map | [`COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits`](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits) | Planck R3 / IRSA |
| Planck NILC full map | [`COM_CMB_IQU-nilc_2048_R3.00_full.fits`](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-nilc_2048_R3.00_full.fits) | Planck R3 / IRSA |
| Planck SEVEM full map | [`COM_CMB_IQU-sevem_2048_R3.00_full.fits`](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-sevem_2048_R3.00_full.fits) | Planck R3 / IRSA |
| Planck Commander full map | [`COM_CMB_IQU-commander_2048_R3.00_full.fits`](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-commander_2048_R3.00_full.fits) | Planck R3 / IRSA |
| Planck SMICA half-mission 1 | [`COM_CMB_IQU-smica_2048_R3.00_hm1.fits`](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_hm1.fits) | Planck R3 / IRSA |
| Planck SMICA half-mission 2 | [`COM_CMB_IQU-smica_2048_R3.00_hm2.fits`](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_hm2.fits) | Planck R3 / IRSA |
| Planck SMICA odd-ring split | [`COM_CMB_IQU-smica_2048_R3.00_oe1.fits`](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_oe1.fits) | Planck R3 / IRSA |
| Planck SMICA even-ring split | [`COM_CMB_IQU-smica_2048_R3.00_oe2.fits`](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_oe2.fits) | Planck R3 / IRSA |
| WMAP 9-year ILC map | [`wmap_ilc_9yr_v5.fits`](https://lambda.gsfc.nasa.gov/data/map/dr5/dfp/ilc/wmap_ilc_9yr_v5.fits) | NASA LAMBDA |

The Planck rows use the temperature fields and embedded mask fields declared by the CMB scripts. The WMAP row is the tested low-ell cross-mission support product. No optional WMAP masks, SEVEM R3.01 substitute, or other alternate FITS products are part of this strict control list.

### CMB Route B Option 1 Curvature-Transport Bridge

A separate CMB comparison-coordinate branch tests the [Route B Option 1](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/CPTG_Route_B_Option1_Curvature_Transport_Geometry_Bridge_Report_v1.pdf) curvature-transport bridge at the `C_l` and Planck likelihood-coordinate level. This is distinct from the CMB comparison-map closure work above.

The validated bridge applies the CPTG curvature-transport response at the amplitude/potential level:

```text
Phi_pi(a,k) = Phi_0(k) C_T(a,k)
```

Because the response acts at amplitude level, the corresponding power-spectrum bridge is:

```text
P(k) -> P(k) C_T(a,k)^2
```

This result should be understood as a geometry-first CMB comparison-coordinate bridge validation. It demonstrates that the fixed CPTG curvature-transport mapping remains compatible with Planck likelihood coordinates across the tested likelihood families while preserving the underlying geometric branch unchanged. CPTG is not implemented as a movable Boltzmann-source or perturbation-code model; the comparison layer tests the observational reach of its locked geometric relations. Detailed likelihood smoke tests and sector-specific diagnostics are provided in the dedicated Route B Option 1 reports.

### Pantheon+ Supernova Distance-Shape Test

CPTG has been tested against Pantheon+ supernova distance-shape data using a full-covariance comparison with a marginalized intercept. This is a distance-shape test, not a local H0 calibration claim. The purpose is to ask whether the CPTG expansion branch can reproduce the relative supernova distance trend once the absolute calibration is marginalized.

### BBN Abundance and Lithium Source-Network Tests

CPTG abundance work is organized around the transported BBN coordinate and the locked lithium source-network gate. The transported abundance coordinate is

```text
eta10_BBN = 5.998071834744
Omega_b h²_BBN = 0.021898765370
```

The [lithium problem](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/CPTG_Cosmological_Lithium_Problem.pdf) is treated as a surviving mass-seven abundance problem because most final primordial lithium is carried through `7Be` during BBN and later appears as `7Li`. The CPTG gate is

```text
Y7,CPTG = Y7,raw / π
∫ Γ7 dt = ln(π)
```

Operationally, the gate is applied to the live `7Li` and `7Be` channels inside the source network. The network is then re-evolved so the final abundance is produced dynamically rather than by a post-processing label.

The deuterium-proton radiative-capture extension supplies the native D(p,γ)³He reaction branch that connects the source geometry to the BBN abundance calculation. In the combined abundance result, deuterium and helium remain inside their comparison bands, He3/H stays below the advisory upper-limit row, and the locked A = 7 gate moves the raw standard-BBN lithium excess into the observed primordial lithium range.

Representative coupled-network rows are

```text
PRyMordial:
D/H pull = -0.019 sigma
Yp pull = +0.231 sigma
gated Li7/H = 1.6764809477031113e-10
Li7 pull after gate = +0.9059237908124455 sigma
result: PASS
```

```text
AlterBBN source-network validation:
D/H pull = -0.27586206896551463 sigma
Yp pull = +0.699999999999997 sigma
gated Li7/H = 1.6841776077984365e-10
Li7 pull after gate = +0.9367104311937463 sigma
result: PASS
```

The closed-form D(p,γ)³He normalization also propagates through a high-precision AlterBBN reaction-rate comparison. Relative to the rounded anchor, the rate scale is

```text
closed-form/default display scale = 1.0000018504518913
```

The corresponding high-precision abundance response is

```text
Yp:    0.24728924324539345
D/H:   2.4348223639496304e-05
He3/H: 1.0311437909300730e-05
Li7/H: 5.4661777402483630e-10
Li6/H: 1.0726676158966467e-14
Be7/H: 5.1798681429209550e-10
```

The ppm-level shifts relative to the rounded-anchor rate are

```text
Yp:    +0.000501503673 ppm
D/H:   -0.666644711631 ppm
He3/H: +0.730801208411 ppm
Li7/H: +1.142530974942 ppm
Li6/H: -0.655629287414 ppm
Be7/H: +1.217913134845 ppm
```

This establishes the public abundance claim as a source-network result: a transported BBN coordinate, a native D(p,γ)³He reaction-rate extension using the closed-form CPTG normalization, and a locked live A = 7 curvature-transport gate that resolves the mass-seven excess while preserving the light-element comparison rows.

### Weak-Lensing S8 Comparison

CPTG weak-lensing work currently uses compressed S8 comparisons against representative weak-lensing and CMB anchors. These tests are diagnostic: they show whether the CPTG growth/lensing branch lies within representative observational bands, but they are not a substitute for a full shear-correlation likelihood or survey-level weak-lensing pipeline.

### DESI DR1 Compressed ShapeFit and BAO Quarter-Ruler

CPTG large-scale-structure work separates DESI comparisons into layers. The current compressed ShapeFit coordinate comparison uses official DESI DR1 HDF5 likelihood containers and is a compressed-coordinate pass. The BAO quarter-ruler is a strong coordinate-wrapper diagnostic using the CPTG transport relation `G_T^(-1/4)`, but it is not presented as a raw official non-unity-`q` runtime likelihood response.

The full-shape AP/growth work remains an exploratory spectrum-shell diagnostic. It is not a raw DESI full-shape validation claim and should not be described as one. A full raw DESI validation requires nuisance-preserving AP, RSD, tracer-window, covariance, nuisance, counterterm, and stochastic machinery to be wired consistently through the official likelihood path.

### Cosmological Horizon Mechanism

A current CPTG horizon-mechanism article treats the [cosmological horizon problem](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/CPTG_and_the_Cosmological_Horizon_Problem.pdf) as a structural-curvature synchronization problem rather than as a scalar-field inflation mechanism. In this framing, early-universe uniformity is attributed to finite curvature saturation and active geometric transport synchronizing the primordial curvature state before decoupling.

### Hubble-Tension Bridge

A current CPTG article develops a geometric interpretation of the [Hubble tension](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/CPTG_Hubble_Tension_Bridge.pdf). In this framing, the Planck/CMB value and the local distance-ladder value are treated as different comparison-layer projections of one native CPTG geometric branch rather than as two unrelated expansion histories.

The locked working bridge is:

```text
67.48 <- 69.4163 -> 73.04
```

This work does not claim that either Planck or SH0ES is simply wrong. It asks whether the apparent disagreement can be expressed as two observational comparison projections of one native CPTG branch: an acoustic/CMB projection below the native branch and a local luminosity-distance projection above it.

---

## CPTG Research Position

CPTG is presented as a geometric gravity research framework with reproducible reduced-limit implementations and fixed-branch comparison layers. Its public tests emphasize baryon-sourced curvature polarization, curvature transport, structural organization, active gates, and branch-specific observational projection.

The included scripts implement reduced limits of the theory:

- the quasi-static, weak-field, approximately axisymmetric galaxy limit;
- the reduced merger-plane transport/lensing limit for dissociative clusters;
- cluster active-gate aperture response from baryonic loading, support temperature, redshift, and aperture radius;
- comparison-layer cosmology audits that map CPTG-native quantities into conventional observational summaries;
- the completed nuclear and abundance extension for `D(p,γ)³He`, BBN-network integration, the locked live A = 7 transport gate, and PRyMordial/AlterBBN source-network validation.

The repository is organized around reproducibility: fixed branch values, declared comparison coordinates, source manifests, and direct residual tables against observational anchors. External tools such as CAMB, Planck likelihoods, BBN codes, and survey products are used as comparison layers for the locked CPTG branch.

The Route B Option 1 CMB bridge demonstrates this geometry-first approach. A fixed curvature-transport response is applied at the amplitude level and carried through CAMB and Planck likelihood-coordinate plumbing while preserving the underlying geometric branch.

This creates a stricter comparison than a conventional parameter-fitting workflow: CPTG must reproduce the observed signatures while preserving the same locked geometric relations. The comparison layers test the scalability and observational reach of the theory across galaxy, cluster, cosmology, and nuclear/abundance sectors.

---

## Relation to MOND and ΛCDM

The repository compares CPTG to MOND-style galaxy predictions and to the broader dark-matter-halo interpretation associated with ΛCDM, but these comparisons are not identical in type.

- **MOND** modifies the acceleration law and performs well when galaxy behavior follows a nearly universal low-acceleration relation.
- **ΛCDM** explains galaxy and cluster dynamics through non-baryonic dark matter, with individual galaxy rotation curves often modeled through halo fitting and related nuisance parameters.
- **CPTG** tests whether similar observed effects can emerge from baryon-sourced curvature polarization, curvature transport, and theory-derived structural organization.

The CPTG SPARC tools evaluate CPTG directly against observed galaxy rotation data and include a MOND-style comparison under the same loaded galaxy database. The Upsilon benchmark adds stellar mass-to-light freedom as a stricter comparison layer. These tests are included so the comparison can be reproduced rather than treated as a qualitative claim.

Compared with ΛCDM halo fitting, CPTG makes a different kind of test: it asks whether galaxy rotation behavior and dissociative cluster-lensing offsets can be reproduced geometrically through baryon-sourced curvature response rather than by fitting non-baryonic halo components.

---

## Recent CPTG Articles and Research Notes

Recent CPTG writing has expanded beyond the core galaxy and Bullet Cluster benchmarks into focused theory and validation articles, including CMB comparison-map closure, Route B Option 1 CMB comparison-coordinate bridge validation, Pantheon+ distance-shape tests, BBN and lithium source-network validation, weak-lensing S8 diagnostics, DESI compressed-coordinate tests, Hubble-tension bridge work, cosmological horizon-mechanism work, compact high-redshift galaxy stress tests, cluster active-gate extensions, and the completed nuclear geometry-to-reaction-rate extension for deuterium-proton radiative capture.

These articles should be read as part of the active research program. Their claim levels vary by implementation maturity and are identified in the relevant papers and reproducibility notes.

---

## Recent Progress and Active Development

CPTG is being developed as an active research program with reproducible public milestones. Recent progress includes:

- public SPARC and Bullet Cluster reduced-limit benchmarks;
- interactive SPARC analysis through CPTG SPARC Browser Workbench v1.11.9;
- same-aperture X-COP cluster active-gate consistency and ACCEPT profile-state ordering;
- locked geometric-pi CMB comparison-map closure and Route B Option 1 curvature-transport bridge validation;
- transported BBN coordinate control with D/H and helium agreement;
- locked live A = 7 lithium source-network gate with PRyMordial and AlterBBN validation;
- native `D(p,γ)³He` source-to-amplitude closure, S-factor normalization, p+D rate-branch integration, BBN-network coupling, and observed light-element residual comparison;
- compact reproducibility packages, source manifests, and exact-input records across the public materials.

Active development continues by extending the same reproducibility-first approach to broader eta/rate sampling, higher-resolution CMB projections, larger same-aperture cluster samples, and manuscript/report consolidation.

---

## Repository Policies

- [Security policy](SECURITY.md)
- [Citation information](#citation)

---

## Citation

If referencing the CPTG framework, please cite:

Carter L. Glass Jr., *Curvature Polarization Transport Gravity: A Unified Geometric Framework for Cosmic Structure and Expansion*, [DOI: 10.13140/RG.2.2.26030.68164](https://doi.org/10.13140/RG.2.2.26030.68164).

For the repository and supporting code package, cite:

CPTG, Supporting Python Models, Benchmark Implementations, and Research References for Curvature Polarization Transport Gravity, companion resource, available at [https://github.com/CLG2025/CPTG](https://github.com/CLG2025/CPTG).

---

## Summary

CPTG is a geometric gravity framework in which gravitational enhancement, lensing displacement, cosmological comparison quantities, CMB map-space closure, Hubble-tension structure, and nuclear/abundance transport are modeled through curvature polarization, curvature transport, and branch-specific observational projection.

The public repository contains reduced numerical implementations, the compact academic benchmark package, the standalone CPTG SPARC Browser Workbench, figures, manuscripts, and development notes intended for reproduction, criticism, and further theory testing. Galaxy rotation curves, reduced cluster-merger reconstruction, and cluster active-gate aperture tests represent the most direct public-scale benchmarks. Cosmology-facing work is organized through fixed comparison branches. The nuclear-scale extension completes the lithium-solution chain through native `D(p,γ)³He` source closure, S-factor normalization, p+D rate-branch integration, BBN-network propagation, the locked live A = 7 lithium gate, and PRyMordial/AlterBBN abundance validation.
