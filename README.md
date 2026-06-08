# CPTG
## Curvature Polarization Transport Gravity


## Start Here: Core CPTG Papers

- **[Curvature Polarization Transport Gravity: A Unified Geometric Framework for Cosmic Structure and Expansion](https://github.com/CLG2025/CPTG/blob/main/CPTG_Unified_Geometric_Framework_Cosmic_Structure_Expansion.pdf)**  
  Primary CPTG theory paper. This manuscript lays out the unified geometric framework: baryon-sourced curvature polarization, curvature transport, the cosmic/structure expansion connection, galaxy and cluster limits, and the broader comparison-layer program.

- **[CPTG Geometric Pi Branch Comparison Coordinates](https://github.com/CLG2025/CPTG/blob/main/CPTG_Geometric_Pi_Branch_Comparison_Coordinates.pdf)**  
  Comparison-coordinate guide for the locked geometric pi branch. This paper explains how CPTG-native quantities are mapped into observational coordinates for CMB, BAO, BBN, supernova, growth, and DESI-style comparison layers without treating those observational coordinates as the theory itself.

---

## Plain-Language Summary

*Curvature Polarization Transport Gravity* (CPTG) is an active research framework exploring whether effects usually attributed to dark matter can instead arise from the way spacetime curvature responds to ordinary baryonic matter.

In this view, galaxies do not require separate dark matter halos to explain their rotation curves. Instead, weak gravitational fields develop a nonlinear curvature-polarization response. In merging galaxy clusters, organized curvature can also be directionally transported, allowing lensing peaks to separate from hot gas without introducing an additional collisionless matter component.

The purpose of this repository is to make the CPTG theory, benchmark scripts, data package, public figures, and continuing comparison-layer tests available for inspection, reproduction, and criticism.

---

## Technical Overview

CPTG is a unified geometric gravity framework in which baryonic matter sources a nonlinear curvature response rather than requiring a separate non-baryonic dark matter halo. The theory is built around two linked mechanisms:

- **Curvature polarization**, which changes the effective gravitational response according to the strength and structure of the field.
- **Curvature transport**, which allows organized curvature to be redistributed directionally in dynamical systems and comparison-layer projections.

In this framework, compact systems recover the Newtonian limit, galaxies probe the polarization-dominated low-acceleration regime, relaxed cluster apertures test active-gate curvature response from gas loading and support temperature, dissociative cluster mergers probe the regime where polarization and directional transport act together to produce displaced lensing structure, and cosmology-facing tests examine whether acoustic, luminosity-distance, abundance, and growth summaries can be organized through CPTG comparison layers.

CPTG is therefore intended to connect galaxy rotation curves, cluster-merger lensing offsets, and selected cosmological comparison quantities within a single geometric framework.

---

## Current Research Status

CPTG is being developed as a geometric framework with several reduced-limit tests and comparison-layer audits. The current public status is best understood by separating established repository benchmarks from active validation candidates and exploratory extensions.

| Area | Current CPTG status | Claim level |
|---|---|---|
| SPARC galaxy rotation curves | Public reduced-limit benchmark against observed rotation curves and MOND-style comparisons | Reproducible galaxy-scale benchmark |
| Bullet Cluster merger plane | Public reduced merger-plane curvature-transport/lensing reconstruction | Reproducible cluster-merger benchmark |
| Cluster active-gate apertures | Single-aperture cluster-response law tested on ACCEPT diagnostics and X-COP hydrostatic-mass apertures | Diagnostic cluster-scale active-gate pass and same-aperture X-COP hydrostatic-mass consistency pass |
| Pantheon+ supernova distances | Full-covariance distance-shape comparison with marginalized intercept | Distance-shape pass, not an H0 calibration claim |
| BBN table-control abundance comparison | Transported CPTG BBN coordinate checked against PRIMAT/PArthENoPE/CAMB abundance tables | D/H and helium table-control pass under the BBN coordinate; coordinate-layer validation |
| Cosmological lithium problem | Locked CPTG lithium gate applied to live `7Li` and `7Be` source-network channels in PRyMordial and AlterBBN | Two-code source-network validation within the stated background-admissible standard; final AlterBBN `N=200` audit gives `484/484` admitted gated rows passing D/H, helium, and lithium |
| Weak-lensing S8 | Compressed comparison against representative weak-lensing and CMB S8 anchors | Diagnostic pass, not a full shear likelihood |
| CMB comparison-map closure | Locked geometric-pi CMB branch tested against real Planck component maps, Planck split maps, WMAP low-ell support products, smoothing controls, mask and sky-fraction controls, and null-envelope controls | Real-map comparison-map closure pass; CPTG remains near-degenerate with the Planck comparison envelope at sub-microkelvin residual-gap scale while generic null envelopes fail much more strongly |
| CMB Route B Option 1 curvature-transport bridge | Geometry-first amplitude-level bridge tested through CAMB `C_l` plumbing and Planck likelihood families, including NPIPE high-ell TTTEEE, NPIPE split TT/TE/EE sectors, Plik split sectors, Planck low-ell TT/EE, and PR4/NPIPE marginalized lensing smoke integration | Geometry-first CMB likelihood-bridge validation; not native CPTG perturbation-code validation and not a completed Boltzmann/source implementation |
| Cosmological horizon mechanism | Structural horizon-branch article using finite curvature saturation and active geometric transport to address pre-recombination causal uniformity | Theory mechanism and falsifiable CMB-perturbation program; not a full perturbation-code validation |
| DESI DR1 compressed ShapeFit and BAO | Official compressed-coordinate dry run plus BAO quarter-ruler coordinate-wrapper diagnostic | ShapeFit coordinate-level pass and BAO ruler support; not full raw DESI validation |
| Hubble-tension bridge | Native CPTG branch projected into CMB/acoustic and local luminosity-distance comparison layers | Article-stage interpretation and derivation target |

---

## What This Repository Contains

This repository contains the public academic package for CPTG, including:

- the current CPTG theory manuscript and research papers,
- SPARC galaxy rotation-curve benchmark scripts,
- Bullet Cluster reduced merger-plane benchmark scripts,
- cluster-scale active-gate research notes and calculation material when publicly included,
- supporting SPARC data and metadata,
- comparison scripts against MOND-style galaxy predictions,
- cosmology-facing comparison-layer scripts and audit outputs when publicly included,
- CMB comparison-map closure Python scripts, CSV summaries, figures, and source/data availability notes when publicly included,
- Route B Option 1 CMB curvature-transport bridge reports and audit summaries when publicly included,
- BBN table-control and lithium source-network validation notes when publicly included,
- horizon-mechanism and structural-branch articles when publicly included,
- benchmark figures and reconstruction images.

The recommended public download is **`CPTG_academic_package.zip`**, located in the **`/archive/`** folder.

This archive contains the SPARC galaxy benchmark script, the Bullet Cluster merger script, the SPARC rotation-curve database, the SPARC supporting metadata file, the Upsilon benchmark comparison script, and the CPTG research paper.

Additional Python files found in the same directory, if present, should be treated as version upgrades, development variants, or replacement implementations. Each script contains detailed internal comments explaining the equations, implementation choices, benchmark logic, and interpretation of the outputs.

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

| Script | Purpose |
|---|---|
| `SPARC_CPTG_MOND_Benchmark.py` | Galaxy rotation-curve benchmark against SPARC data. |
| `CPTG_Bullet_Cluster_Merger.py` | Reduced merger-plane curvature-transport/lensing benchmark. |
| `CPTG_ClusterActiveGate_IntegratedTool_v0_5.py` | Single-aperture and aperture-ladder cluster-response calculations from baryonic loading, support temperature, redshift, and aperture. Requires cluster archive: https://drive.switch.ch/index.php/s/j3WUOYXWgv9Jbnz/download|
| `CPTG_MOND_Upsilon_SPARC_Benchmark.py` | MOND/CPTG comparison with stellar mass-to-light freedom. |
| `CPTG-CMB.zip` | CMB comparison-map closure scripts for Planck/WMAP component maps, split maps, smoothing/mask controls, visual comparisons, summary reports, and null-envelope controls. |

---

## Galaxy-Scale Test: SPARC Benchmark

The **`SPARC_CPTG_MOND_Benchmark.py`** script tests the CPTG galaxy-limit equation against SPARC rotation-curve data.

The benchmark loads the same galaxy files for CPTG and MOND, builds the corresponding baryonic source fields, solves the CPTG nonlinear acceleration equation, computes the MOND comparison, and reports full-database, metadata-defined primary-sample, and mode-filtered benchmark results.

Its diagnostics include:

- total rotation-curve chi-square,
- chi-square per datapoint,
- conservative effective-DOF audits,
- velocity RMS residuals,
- mean absolute velocity error,
- radial acceleration relation scatter,
- galaxy-level CPTG/MOND win counts,
- Curvature-Weighted Structural Mode Index values.

The benchmark supports the included SPARC database, user-supplied galaxy databases, and targeted single-galaxy or multi-galaxy comparison runs.

The main public significance of this test is that CPTG is evaluated directly against observed galaxy rotation data rather than only being presented as a conceptual theory.

---

The first benchmark figure summarizes how CPTG and MOND compare across the full SPARC galaxy sample.

![Composite CPTG--MOND benchmark summary for the full 175-galaxy SPARC sample. The panels compare RAR scatter, population-averaged normalized rotation curves, all-galaxy velocity medians, and separate CPTG- and MOND-source RAR trends. Across these diagnostics, CPTG more closely tracks the observed SPARC behavior, with lower RAR scatter, closer normalized rotation-curve agreement, and better velocity-median alignment than MOND.](https://github.com/CLG2025/CPTG/blob/main/images/CPTG-MOND-Benchmark.png)

<sup>Figure: Average SPARC Benchmark Across 175 Galaxies</sup>

---

## Outer-Slope Convergence Test

The CPTG galaxy benchmark also includes an outer-slope convergence test. This test extends the solved CPTG rotation-curve behavior beyond the outermost measured SPARC data points to examine how the theory behaves in the far outer regions of galaxies.

The purpose of this test is not to claim that current observations already measure this far-out behavior directly. Instead, it checks whether the reduced CPTG galaxy equation develops a stable long-range trend once the model is continued beyond the observed rotation-curve domain.

In CPTG, this outer behavior is important because the theory predicts that weak-field galaxy outskirts should not drift randomly. They should gradually approach a consistent curvature-polarization pattern. The convergence plot visualizes that prediction across the SPARC galaxy sample.

The second benchmark figure shows the stacked CPTG outer-slope convergence trend for the SPARC galaxy sample.

![CPTG outer-slope convergence in the asymptotic extension regime. The plot shows how the extended CPTG rotation-curve behavior evolves beyond the observed SPARC rotation-curve domain. The median trend approaches the predicted CPTG outer-regime behavior, while the shaded region shows the galaxy-to-galaxy spread. This figure illustrates that the extended CPTG solution approaches a stable long-range pattern rather than drifting arbitrarily outside the measured data range.](https://github.com/CLG2025/CPTG/blob/main/images/cptg_outer_slope_convergence.png)

<sup>Figure: CPTG outer-slope convergence in the extended galaxy-outskirts regime.</sup>

---

## Curvature-Weighted Structural Mode Index

The Curvature-Weighted Structural Mode Index, **N**, is a CPTG diagnostic derived from the solved acceleration field.

It measures how curvature support is organized inside a galaxy. It is not assigned from catalog morphology, galaxy name, or visual classification. Mode-filtered runs allow galaxies with similar CPTG structural organization to be benchmarked as subsets of the full database.

In public-facing terms, **N** is a way of asking:

> How is the solved curvature structure organized inside this galaxy?

This makes the mode index a theory-derived structural diagnostic rather than a conventional galaxy-type label.

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

![Normalized CPTG kappa reconstruction of the Bullet Cluster merger plane. The map shows two main convergence structures: a compact Bullet-side lensing feature on the left, displaced from the Bullet gas peak, and a larger main-cluster lensing structure on the right with north and south substructure. White contours trace the strongest reconstructed convergence regions. Markers identify Bullet and main gas peaks, galaxy peaks, lensing peaks, and main-cluster north/south lens peaks. A scale bar marks 100 kpc.](https://github.com/CLG2025/CPTG/blob/main/images/CPTG-Curvature-Transport-Model.png)

<sup>Figure: CPTG Bullet Cluster kappa reconstruction showing gas-lensing separation.</sup>

---

## Cluster-Scale Active-Gate Test: ACCEPT and X-COP

The cluster-scale active-gate work extends CPTG beyond galaxy rotation curves and reduced merger-plane reconstruction into relaxed or approximately coherent galaxy-cluster apertures. The calculation asks whether a cluster aperture can be described by baryonic loading, support temperature, redshift, and aperture radius through one active curvature-response state.

For a selected aperture `R_delta`, the calculation uses:

```text
z, Delta, R_delta, M_gas(<R_delta), M_star(<R_delta), M_ICL(<R_delta), T_gate
```

and computes the baryonic loading `F_delta`, the support ratio `theta_T`, the critical support threshold `theta_crit`, the active-gate variable `Q_C`, the mass ratio `y_delta`, and the predicted cluster mass:

```text
M_CPTG,delta = y_delta M_ref,delta
```

The active-gate classes are:

```text
Q_C > 1.2        -> closure-stable
0.8 <= Q_C <= 1.2 -> watch
0.5 <= Q_C < 0.8 -> suppressed
Q_C < 0.5       -> strongly suppressed
```

The cluster structural mode is the response-load variable already present in the active-gate equation:

```text
N_C,delta = y_delta N_C0 / Q_C^2
```

This is not the same as the total-to-baryon mass multiplier. `M_CPTG/M_b` is a mass accounting ratio, while `N_C,delta` measures how deeply the selected aperture sits in the active curvature-response branch.

In the current X-COP same-aperture `R500` comparison, the active-gate calculation was evaluated against 12 hydrostatic-mass cluster apertures. The median CPTG/HSE ratio is about `0.989`, the median absolute fractional difference is about `1.08%`, and the maximum absolute difference is about `2.68%`. The strongest X-COP result is the aperture-ladder behavior: the median active-gate variable decreases smoothly from inner closure-stable apertures to the outer active-gate regime, with median `Q_C` moving from about `1.5781` at `R2500` to about `0.9701` at `R500`.

The ACCEPT diagnostic sample provides an independent clean outer-profile ordering test. In the full clean outer-profile pass, closure-stable rows have higher median `Q_C`, suppressed rows have lower median `Q_C`, and strongly suppressed rows show the lowest median `y_delta`. A representative 100-row subset preserves the same ordering: high-`Q_C` rows close, low-`Q_C` rows suppress, and the active gate sorts cluster-profile rows by response state.

This active-gate result should be read as a diagnostic cluster-scale response pass and a same-aperture X-COP hydrostatic-mass consistency pass. It is not a claim that one single-aperture formula describes strong cluster mergers without decomposition. Strong mergers are best treated separately unless gas, stellar, temperature, and mass components can be assigned consistently to the same dynamical aperture.

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

CPTG CMB map work is organized as a real-map comparison test between the locked geometric-pi CMB branch and public CMB map products. The current paper uses real Planck component maps, Planck split maps, and WMAP low-ell support products.

![Observed Planck SMICA vs fitted CPTG comparison map](images/fig_visual_fitted.png)

<sup>Figure: SMICA visual comparison from the CMB comparison-map closure paper. Top: observed Planck SMICA temperature map. Center: fitted CPTG comparison map. Bottom: observed-minus-fitted-CPTG residual.</sup>

The map-space procedure uses the same comparison coordinate for CPTG, the Planck envelope, and controls. It reads the temperature field from the public CMB map product, applies the documented mask, converts to microkelvin, downgrades to `Nside = 256`, uses `ell_max = 767`, removes the monopole and dipole on the valid sky, extracts the observed phase scaffold, and builds phase-locked comparison maps for the locked CPTG branch, the Planck baseline envelope, and null controls. The fitted comparison map is evaluated with the same amplitude-plus-offset residual rule:

```text
T_fit(nhat) = A T_template(nhat) + B
```

The central result is that the locked CPTG geometric-pi branch reaches near-degenerate CMB comparison-map closure with the Planck comparison envelope across real CMB map products and control layers.

Representative map-space values from the SMICA visual comparison are:

```text
Observed RMS = 100.509653 microK
Raw CPTG RMS = 118.452082 microK
Raw residual RMS = 26.784740 microK
Fitted CPTG RMS = 98.844631 microK
Fitted residual RMS = 18.218928 microK
Observed vs fitted correlation = 0.983434
```

Across Planck component maps, the representative CPTG-minus-Planck fitted-residual RMS gap is:

```text
Mean component-map gap = +0.023518 microK
Inpainted component-map gap = about +0.0264 microK
```

Across Planck full-band split maps, the result is highly stable:

```text
Full-band split mean RMS gap = +0.023891 microK
Split-to-split span = 0.000690 microK
```

The full control-scatter audit contains `1159` control rows:

```text
Mean CPTG-minus-Planck RMS gap = +0.034265 microK
Full control span = 0.056006 microK
Planck-ahead fraction = 0.9931
```

Mask and sky-fraction controls show that the residual gap is strongly sky-selection sensitive at high Galactic latitude:

```text
No Galactic cut gap = about +0.02379 microK
|b| > 40 degrees gap = about +0.00146 microK
```

The null-envelope audit shows that the near-degeneracy is not produced automatically by reusing the observed phase scaffold. The Planck and CPTG envelopes remain adjacent, while generic controls fail much more strongly under the same procedure:

```text
Observed self-spectrum control = 11.503 microK
Planck baseline = 18.230 microK
CPTG locked pi-CMB = 18.253 microK
Smoothed observed spectrum control = 19.282 microK
Flat D_ell null = 39.281 microK
Power-law D_ell null = 56.247 microK
Flat C_ell null = 78.236 microK
Shuffled-spectrum nulls = 95.461 microK
```

This is the current public CMB comparison-map closure result: a fixed geometric-pi branch projects into real Planck/WMAP CMB map space at essentially the Planck-envelope residual scale, while generic null envelopes fail under the same map-space test.

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

A separate CMB development branch tests the Route B Option 1 curvature-transport bridge at the `C_l` and Planck-likelihood plumbing level. This is distinct from the CMB comparison-map closure work above.

The validated bridge applies the CPTG curvature-transport response at the amplitude/potential level:

```text
Phi_pi(a,k) = Phi_0(k) C_T(a,k)
```

Because the response acts at amplitude level, the corresponding power-spectrum bridge is:

```text
P(k) -> P(k) C_T(a,k)^2
```

This branch was accepted by CAMB bridge smoke tests, `C_l` residual audits, NPIPE high-ell TTTEEE controlled near-identity ladders, NPIPE split TT/TE/EE/TTTEEE sectors, Plik split TT/TE/EE/TTTEEE sectors, Planck low-ell TT and EE, PR4/NPIPE marginalized lensing, and a combined high-ell + low-ell + lensing smoke stack.

The strongest with-lensing endpoint was:

```text
f_r = 1.000
summed chi2 = 11094.58772933417

f_r = 0.999
summed chi2 = 11094.937838138996

delta chi2 = +0.3501088048269594
```

The Plik TE sector showed a diagnostic inversion rather than a branch rejection. The refined TE diagnostic anchor was:

```text
best f_r = 0.996
best chi2 = 891.3320114884485
best delta chi2 vs identity = -0.4459062018408986
```

This result should be read as geometry-first Route B bridge validation. It validates the curvature-transport mapping and likelihood plumbing across tested Planck likelihood families. It is not native CPTG perturbation-code validation, not a completed Boltzmann/source implementation, and not the earlier locked `f -> A_s` proxy mapping.

### Pantheon+ Supernova Distance-Shape Test

CPTG has been tested against Pantheon+ supernova distance-shape data using a full-covariance comparison with a marginalized intercept. This is a distance-shape test, not a local H0 calibration claim. The purpose is to ask whether the CPTG expansion branch can reproduce the relative supernova distance trend once the absolute calibration is marginalized.

### BBN Abundance and Lithium Source-Network Tests

CPTG abundance work is separated into two layers.

The first layer is the BBN table-control comparison. This layer checks whether the transported CPTG BBN coordinate remains compatible with standard light-element controls, especially deuterium and helium. In this comparison, the acoustic/CMB baryon coordinate is not inserted directly into the nuclear-abundance table. The BBN abundance coordinate is the transported one:

```text
eta10_BBN = 5.998071834744
Omega_b h^2_BBN = 0.021898765370
```

This distinction matters because D/H is a sensitive background control. Rows whose unmodified background already fails D/H or helium are not used to judge a lithium correction.

The second layer is the CPTG lithium source-network test. The cosmological lithium problem is treated as a surviving mass-seven abundance problem, because most final primordial lithium is carried through `7Be` during BBN and later appears as `7Li`. The locked CPTG lithium gate is:

```text
Y_7,CPTG = Y_7,raw / pi
integral Gamma_7 dt = ln(pi)
```

Operationally, the gate is applied to the live `7Li` and `7Be` channels inside the source network, and the network is then re-evolved. It is not a post-processing division applied after the final abundance table is produced.

The source-network validation now uses two independent BBN networks. PRyMordial gives `205/205` passing background-admissible locked-gate cases. AlterBBN gives `24/24` passing thermal-window and shape cases, `33/33` passing eta/rate admitted gated rows, and `484/484` passing background-admissible gated rows in the final `N=200` combined Monte Carlo audit.

The claim level is therefore a two-code source-network validation under the stated background-admissible standard. It should not be read as an assertion that every possible BBN code, every stellar lithium systematic, or every observational systematic has been exhausted.

AlterBBN has an important output convention: its reported final `Li7/H` column already includes the post-BBN `7Be -> 7Li` contribution. The diagnostic `Be7/H` column is therefore not added a second time.

### Weak-Lensing S8 Comparison

CPTG weak-lensing work currently uses compressed S8 comparisons against representative weak-lensing and CMB anchors. These tests are diagnostic: they show whether the CPTG growth/lensing branch lies within representative observational bands, but they are not a substitute for a full shear-correlation likelihood or survey-level weak-lensing pipeline.

### DESI DR1 Compressed ShapeFit and BAO Quarter-Ruler

CPTG large-scale-structure work separates DESI comparisons into layers. The current compressed ShapeFit coordinate comparison uses official DESI DR1 HDF5 likelihood containers and is a compressed-coordinate pass. The BAO quarter-ruler is a strong coordinate-wrapper diagnostic using the CPTG transport relation `G_T^(-1/4)`, but it is not presented as a raw official non-unity-`q` runtime likelihood response.

The full-shape AP/growth work remains an exploratory spectrum-shell diagnostic. It is not a raw DESI full-shape validation claim and should not be described as one. A full raw DESI validation requires nuisance-preserving AP, RSD, tracer-window, covariance, nuisance, counterterm, and stochastic machinery to be wired consistently through the official likelihood path.

### Cosmological Horizon Mechanism

A current CPTG horizon-mechanism article treats the cosmological horizon problem as a structural-curvature synchronization problem rather than as a scalar-field inflation mechanism. The relevant scale in that paper is the structural Hubble-equivalent branch,

```text
H_struct(t) = (16 pi / 3) a_star(t) / c
```

This structural scale is kept distinct from the native pi-branch quantities `H_infinity` and `H0^(pi)`, and from the transported CMB comparison value `H0_CMB^CPTG`. In this framing, early-universe uniformity is not attributed to ordinary thermal contact between widely separated matter regions. It is attributed to finite curvature saturation and active geometric transport synchronizing the primordial curvature state before decoupling.

The claim level is a theory mechanism with falsifiable consequences. The article identifies CMB anisotropy statistics, primordial non-Gaussianity, tensor-background limits, acoustic consistency, and the relation between structural acceleration response and cosmological expansion as the relevant stress tests. It should not be read as a completed Boltzmann-code implementation or a replacement for the locked CMB comparison-layer audits.

### Hubble-Tension Bridge

A current CPTG article develops a geometric interpretation of the Hubble tension. In this framing, the Planck/CMB value and the local distance-ladder value are not treated as two incompatible physical expansion histories. They are treated as different comparison-layer projections of one native CPTG geometric branch.

The locked working bridge is:

```text
67.48 <- 69.4163 -> 73.04
```

The native CPTG branch is:

```text
H0^(pi) = 69.4162507897 km s^-1 Mpc^-1
```

The acoustic/CMB comparison layer maps this branch downward through:

```text
H0_CMB^CPTG = H0^(pi) sqrt(G_T)
```

The local luminosity-distance side maps it upward through:

```text
H0_loc^CPTG = H0^(pi) L_loc
```

For a SH0ES-like endpoint of `H0 = 73.04`, the required local luminosity-transport factor is:

```text
L_loc ~= 1.052203
```

This work does not claim that either Planck or SH0ES is simply wrong. It asks whether the apparent disagreement can be expressed as two observational comparison projections of one native CPTG geometric branch: an acoustic/CMB projection below the native branch and a local luminosity-distance projection above it.

---

## What CPTG Is Not

CPTG is not a dark matter halo-fitting model.

CPTG is not a MOND interpolation-function model.

CPTG is not presented as a completed replacement for general relativity or ΛCDM. The repository contains reduced theoretical and numerical implementations designed to test whether nonlinear curvature polarization and curvature transport can reproduce key galaxy-scale, cluster-scale, and comparison-layer cosmological observations.

The included scripts implement reduced limits of the theory:

- the quasi-static, weak-field, approximately axisymmetric galaxy limit;
- the reduced merger-plane transport/lensing limit for dissociative clusters;
- comparison-layer cosmology audits that map CPTG-native quantities into conventional observational summaries.

They are not full numerical-relativity solvers for the complete covariant CPTG field equations. Cosmology-facing scripts are not automatically full Boltzmann, full DESI, or full weak-lensing likelihood implementations unless those specific pipelines are provided. The Route B Option 1 CMB result is a geometry-first curvature-transport bridge validation through CAMB and Planck likelihood plumbing; it should not be described as native CPTG perturbation-code validation or as a completed CPTG Boltzmann/source solver. Their purpose is to test whether reduced CPTG equations and comparison maps reproduce important observational signatures normally associated with dark matter, dark energy, or parameter tension.

---

## Relation to MOND and ΛCDM

The repository compares CPTG to MOND-style galaxy predictions and to the broader dark-matter-halo interpretation associated with ΛCDM, but these comparisons are not identical in type.

CPTG can produce MOND-like weak-field behavior in galaxies, but it is not constructed as a MOND interpolation law. MOND applies a mostly universal acceleration rule to the baryonic source. CPTG instead solves for a nonlinear curvature response sourced by the galaxy's baryonic structure.

In the included SPARC benchmark, this distinction is tested directly. Across the full 175-galaxy SPARC sample, CPTG produces lower rotation-curve error, lower velocity residuals, lower radial-acceleration scatter, and stronger galaxy-level win counts than the MOND comparison implemented in the script. The same advantage remains in the metadata-defined primary sample, where inclination and data-quality concerns are more tightly controlled.

The Upsilon benchmark adds a stricter comparison by allowing stellar mass-to-light scaling. MOND improves substantially when Upsilon freedom is added, which is expected because the stellar contribution can be rescaled to better match the observed rotation curves. But this improvement is internal to MOND: MOND+Upsilon does better than baseline MOND, yet it does not surpass CPTG in the benchmark comparisons. CPTG remains ahead with or without Upsilon because the changed baryonic source is not merely rescaled; it is allowed to recompute the CPTG structural response. This highlights the central difference between the models: MOND+Upsilon adjusts the baryonic input inside a universal acceleration-law framework, while CPTG directly computes the curvature response of each galaxy.

Compared with ΛCDM, the repository makes a different kind of comparison. The included public galaxy scripts do not fit non-baryonic dark matter halos. Instead, CPTG tests whether galaxy rotation behavior and dissociative cluster-lensing offsets can be reproduced geometrically through nonlinear curvature polarization and curvature transport. In this sense, CPTG is evaluated as a baryon-sourced geometric alternative to halo fitting, rather than as another adjustable halo model.

In short:

- **MOND** modifies the acceleration law and performs well when galaxy behavior follows a nearly universal low-acceleration relation.
- **ΛCDM** explains galaxy and cluster dynamics through non-baryonic dark matter, but individual galaxy rotation curves are usually modeled through halo fitting and related nuisance parameters.
- **CPTG** tests whether the same observed effects can emerge from baryon-sourced curvature polarization, curvature transport, and theory-derived structural organization.

The purpose of the repository is not to declare the issue settled, but to make the CPTG comparison reproducible: the same public scripts load the data, solve the reduced CPTG equations, compute the MOND comparison, and report the resulting diagnostics.

---

## Recent CPTG Articles and Research Notes

Recent CPTG writing has expanded beyond the original galaxy and Bullet Cluster benchmarks into several focused theory and validation articles:

- unified CPTG cosmology and comparison-layer framework,
- CMB comparison-map closure with real Planck component maps, Planck split maps, and WMAP low-ell support products,
- Pantheon+ supernova distance-shape comparison,
- BBN transported-baryon abundance comparison,
- cosmological lithium problem and CPTG lithium solution with two-code source-network validation,
- weak-lensing S8 compressed comparison,
- DESI DR1 compressed ShapeFit coordinate-level dry run and BAO quarter-ruler diagnostic,
- Hubble-tension bridge article,
- cosmological horizon-mechanism article using finite curvature saturation and active geometric transport,
- compact high-redshift galaxy stress tests,
- cluster-scale active-gate extension using ACCEPT diagnostics and X-COP same-aperture/aperture-ladder comparisons.

These articles should be read as part of the active research program.

---

## Continuing Work

CPTG is being developed as an active research program rather than a single fixed software tool or one-time benchmark. Several continuing work streams are currently being explored.

### Structure-Response Distance Refinement

The CPTG-SRD distance calculator explores whether CPTG can provide a local distance refinement around an existing SPARC metadata distance. It does not attempt to solve galaxy distances blindly from raw photometry. Instead, it tests nearby distance branches around the existing metadata value.

For each trial distance branch, the software rebuilds the baryonic source field, solves the CPTG galaxy response, recomputes the structural acceleration scale, and selects the branch with the strongest local structure-response consistency.

The purpose of SRD is to ask whether CPTG contains a useful internal distance-diagnostic signal tied to the galaxy's solved curvature response. This is especially relevant because distance, baryonic source strength, structural scale, and rotation-curve response are linked in any theory attempting to explain galaxy dynamics without fitting dark matter halos.

### Cluster-Scale Development

CPTG cluster-scale work now has two complementary public directions. The reduced Bullet Cluster benchmark tests curvature transport and displaced lensing structure in a dissociative merger plane. The cluster active-gate work tests relaxed or approximately coherent apertures using baryonic loading, support temperature, redshift, and aperture radius. The current ACCEPT and X-COP diagnostics indicate that inner apertures tend toward closure-stable response while outer apertures expose active-gate sensitivity.

The next cluster-scale goals are to expand the X-COP/ACCEPT-style sample, add lensing-mass comparisons where reliable same-aperture products are available, improve treatment of stellar and intracluster-light baryons, and develop merger-aware decompositions for disturbed systems where a single-aperture relaxed interpretation is not adequate.

### Perturbation-Level CMB Development

The locked CMB comparison-map closure paper provides a real-map benchmark for the geometric-pi branch using Planck component maps, Planck split maps, WMAP low-ell support products, control ladders, and null-envelope tests. The long-term CMB development goal is to express early curvature transport, acoustic source structure, optical-depth visibility, scalar-amplitude response, and horizon-mechanism curvature-state synchronization directly in a physical perturbation model comparable in role to CLASS or CAMB.

The Route B Option 1 bridge is an intermediate step in this development path. It shows that an amplitude-level CPTG curvature-transport response can be mapped into the CMB spectrum pipeline as a squared power-spectrum response and remain accepted across tested Planck likelihood families. The next stage remains a native CPTG perturbation/source-equation implementation rather than another bridge or proxy layer.

### Direction of Development

The next development goals are:

- strengthen the mathematical connection between curvature transport, curvature polarization, and observed galaxy structure;
- evaluate whether SRD distance refinements correlate with independent distance-quality indicators;
- extend cluster active-gate tests to larger same-aperture samples and connect them to gas, galaxy, lensing, and merger-aware decompositions;
- carry the locked CMB comparison-map closure result forward into a more physical perturbation-equation model;
- keep all numerical tests reproducible, compact, and open to criticism.

These continuing investigations are included to make the research path transparent. They should be read as active development directions, not settled conclusions.

---

## Citation

If referencing the CPTG framework, please cite:

Carter L. Glass Jr., *Curvature Polarization Transport Gravity: A Unified Geometric Framework for Cosmic Structure and Expansion*, [DOI: 10.13140/RG.2.2.26030.68164](https://doi.org/10.13140/RG.2.2.26030.68164).

For the repository and supporting code package, cite:

CPTG, Supporting Python Models, Benchmark Implementations, and Research References for Curvature Polarization Transport Gravity, companion resource, available at [https://github.com/CLG2025/CPTG](https://github.com/CLG2025/CPTG).

---

## Summary

CPTG is not a dark matter halo fit and is not a MOND interpolation law. It is a geometric gravity framework in which gravitational enhancement, lensing displacement, cosmological comparison quantities, CMB map-space closure, and possible Hubble-tension structure are modeled through curvature polarization, curvature transport, and branch-specific observational projection.

The public repository contains reduced numerical implementations, benchmark scripts, figures, manuscripts, and development notes intended for reproduction, criticism, and further theory testing. Galaxy rotation curves, reduced cluster-merger reconstruction, and cluster active-gate aperture tests represent the most direct public-scale benchmarks. Cosmology-facing work is organized by claim level: comparison-map closure pass, comparison-layer pass, source-network validation, diagnostic pass, validation candidate, theory mechanism, or exploratory extension depending on the maturity of the implementation.

The current CMB result belongs to the comparison-map closure category because the locked geometric-pi branch has been tested against real Planck and WMAP CMB map products under a common phase-scaffold, masking, resolution, and residual-evaluation procedure. The Route B Option 1 CMB result belongs to the separate geometry-first likelihood-bridge category because it validates the amplitude-level curvature-transport mapping and Planck likelihood plumbing, not a native perturbation-code solver. The current lithium result belongs to the source-network validation category because it has passed locked PRyMordial and AlterBBN source-network tests under the stated background-admissible rule.
