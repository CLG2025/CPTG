# CPTG
## Curvature Polarization Transport Gravity

## Start Here: Core CPTG Papers

- **[Curvature Polarization Transport Gravity: A Unified Geometric Framework for Cosmic Structure and Expansion](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/CPTG_Unified_Geometric_Framework_Cosmic_Structure_Expansion.pdf)**  
  Primary CPTG theory paper. This manuscript lays out the unified geometric framework: baryon-sourced curvature polarization, curvature transport, the cosmic/structure expansion connection, galaxy and cluster limits, and the broader comparison-layer program.

- **[CPTG Geometric Pi Branch Comparison Coordinates](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/CPTG_Geometric_Pi_Branch_Comparison_Coordinates.pdf)**  
  Comparison-coordinate guide for the locked geometric pi branch. This paper explains how CPTG-native quantities are mapped into observational coordinates for CMB, BAO, BBN, supernova, growth, and DESI-style comparison layers without treating those observational coordinates as the theory itself.

- **[The Science Behind CPTG: A Geometric Alternative to Dark Matter, Dark Energy, and MOND](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/The_Science_Behind_CPTG.pdf)**  
  A public-facing introduction to Curvature Polarization Transport Gravity. This paper explains what makes CPTG different from dark matter, dark energy, and MOND-style approaches: baryon-sourced curvature polarization, curvature transport, structural modes, active gates, and scalable comparison coordinates derived from one geometric framework rather than sector-by-sector tuning.
  

## CPTG Pi-Bridge

CPTG Pi-Bridge is a local [research workbench](https://raw.githubusercontent.com/CLG2025/CPTG/main/images/CPTG-Pi-Bridge-Local-Workbench.png) planned for introduction in the near future.

It is being developed to give researchers a practical way to use CPTG against real astronomy and cosmology data. The goal is to provide one local interface where users can load public datasets, select a CPTG comparison branch, run the CPTG audit engine, inspect the results, and export a reproducible validation package.

Pi-Bridge is intended to reduce the need for scattered notebooks, one-off validation scripts, and custom comparison pipelines for every separate test. For CPTG work, it is being designed to serve the role that larger modeling portals or solver pipelines often serve in standard cosmology: a repeatable place to load data, run the model-side comparison, review the evidence, and preserve the audit trail.

The translation layer uses the geometric-pi comparison-coordinate method from the CPTG pi-branch paper. CPTG-native quantities are first kept in their native geometric branch, then projected into conventional observational coordinates such as CMB, BAO, BBN, supernova, growth, and DESI-style summaries. Pi-Bridge uses that comparison-coordinate approach so researchers can compare CPTG outputs against public data products without treating the observational coordinate system as the theory itself.

The browser workbench is currently in beta testing.

---

## Plain-Language Summary

*Curvature Polarization Transport Gravity* (CPTG) is an active research framework exploring whether effects usually attributed to dark matter can instead arise from the way spacetime curvature responds to ordinary baryonic matter.

In this view, galaxies do not require separate dark matter halos to explain their rotation curves. Instead, weak gravitational fields develop a nonlinear curvature-polarization response. In merging galaxy clusters, organized curvature can also be directionally transported, allowing lensing peaks to separate from hot gas without introducing an additional collisionless matter component.

The purpose of this repository is to make the CPTG theory, benchmark scripts, data package, public figures, and continuing comparison-layer tests available for inspection, reproduction, and criticism.

---

## Technical Overview

CPTG is a geometric gravity framework in which ordinary baryonic matter sources a nonlinear curvature response rather than requiring a separate non-baryonic dark matter halo. The theory is built around two linked mechanisms:

- **Curvature polarization**, which changes the effective gravitational response according to the strength and structure of the field.
- **Curvature transport**, which allows organized curvature to be redistributed directionally in dynamical systems and comparison-layer projections.

The same framework is tested in reduced limits across galaxy rotation curves, cluster-merger lensing structure, relaxed cluster apertures, and cosmology-facing comparison layers. The repository is organized to keep these tests reproducible and to separate established benchmarks from active validation candidates.

## Current Research Status

CPTG is being developed as a geometric framework with reduced-limit tests and comparison-layer audits. The public status is best read by separating reproducible benchmarks, coordinate-layer validations, diagnostic passes, and active theory-development work.

| Area | Current CPTG status | Claim level |
|---|---|---|
| SPARC galaxy rotation curves | Public reduced-limit benchmark against observed rotation curves and MOND-style comparisons | Reproducible galaxy-scale benchmark |
| Bullet Cluster merger plane | Public reduced merger-plane curvature-transport/lensing reconstruction | Reproducible cluster-merger benchmark |
| Cluster active-gate apertures | Same-aperture cluster-response tests using baryonic loading, support temperature, redshift, and aperture radius | Diagnostic cluster-scale active-gate and X-COP consistency [pass](#Cluster-Scale-Active-Gate-Test--ACCEPT-and-X-COP) |
| Pantheon+ supernova distances | Full-covariance relative distance-shape comparison with marginalized intercept | Distance-shape [pass](#Pantheon+-Supernova-Distance-Shape-Test), not an H0 calibration claim |
| BBN abundance and lithium tests | Transported BBN coordinate and locked lithium source-network gate checked in independent BBN workflows | Coordinate-layer and source-network [validation](#BBN-Abundance-and-Lithium-Source-Network-Tests) under stated controls |
| Weak-lensing S8 | Compressed comparison against representative weak-lensing and CMB S8 anchors | Diagnostic [pass](#Weak-Lensing-S8-Comparison), not a full shear likelihood |
| CMB comparison-map closure | Locked geometric-pi CMB branch tested against real Planck/WMAP temperature-map products and null controls | Real-map [comparison-map closure](#CMB-Comparison-Map-Closure) pass |
| CMB Route B Option 1 bridge | Fixed amplitude-level curvature-transport bridge tested through CMB spectrum and Planck likelihood-coordinate plumbing | Geometry-first comparison-coordinate bridge [validation](#CMB-Route-B-Option-1-Curvature-Transport-Bridge) |
| DESI compressed ShapeFit and BAO | Compressed-coordinate and ruler-wrapper diagnostics | Coordinate-level [support](#DESI-DR1-Compressed-ShapeFit-and-BAO-Quarter-Ruler), not full raw DESI validation |
| Horizon and Hubble-tension mechanisms | Structural articles mapping CPTG-native branches into observational comparison layers | Theory [mechanism](#Hubble-Tension-Bridge) and derivation-stage interpretation |

## What This Repository Contains

This repository contains the public academic package for CPTG, including:

- current CPTG theory manuscripts and research notes,
- reduced benchmark scripts for galaxy and cluster tests,
- supporting public data packages and metadata when included,
- comparison-layer scripts and audit outputs when publicly included,
- CMB source/data availability notes and strict rerun file lists,
- figures, summaries, and reproducibility material.

The recommended public download is **`CPTG_academic_package.zip`**, located in the **`/archive/`** folder. That archive contains the public benchmark package for the original SPARC galaxy and Bullet Cluster reduced-limit tests. Additional Python files in the repository should be treated as version upgrades, development variants, or replacement implementations unless a specific package README states otherwise.

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

The [cluster-scale active-gate](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/CPTG_Cluster-Scale_Active-Gate_Extension.pdf) work extends CPTG beyond galaxy rotation curves and reduced merger-plane reconstruction into relaxed or approximately coherent galaxy-cluster apertures. The calculation asks whether a cluster aperture can be described by baryonic loading, support temperature, redshift, and aperture radius through one active curvature-response state.

For a selected aperture `R_delta`, the calculation uses gas mass, stellar/intracluster-light support where available, temperature, redshift, and aperture radius to compute the active-gate response and the predicted CPTG mass for the same aperture.

This work is intended as a same-aperture cluster-response diagnostic. The current ACCEPT and X-COP tests indicate that inner apertures tend toward closure-stable response while outer apertures expose active-gate sensitivity. The X-COP same-aperture comparison shows close consistency with hydrostatic-mass apertures under the stated inputs, while ACCEPT provides an independent outer-profile ordering diagnostic.

This result should be read as a diagnostic cluster-scale active-gate pass and an X-COP same-aperture consistency pass. It is not a claim that one single-aperture formula describes strong cluster mergers without decomposition. Strong mergers are best treated separately unless gas, stellar, temperature, and mass components can be assigned consistently to the same dynamical aperture.

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

This result should be read as geometry-first CMB comparison-coordinate bridge validation. It validates the fixed curvature-transport mapping and Planck likelihood-coordinate compatibility across tested likelihood families. It is not native CPTG perturbation-code validation, not a movable Boltzmann/source implementation, and not the earlier locked `f -> A_s` proxy mapping. The detailed likelihood smoke tests and sector-specific diagnostics are kept in the dedicated Route B Option 1 reports.

### Pantheon+ Supernova Distance-Shape Test

CPTG has been tested against Pantheon+ supernova distance-shape data using a full-covariance comparison with a marginalized intercept. This is a distance-shape test, not a local H0 calibration claim. The purpose is to ask whether the CPTG expansion branch can reproduce the relative supernova distance trend once the absolute calibration is marginalized.

### BBN Abundance and Lithium Source-Network Tests

CPTG abundance work is separated into two layers.

The first layer is the BBN table-control comparison. This checks whether the transported CPTG BBN coordinate remains compatible with standard light-element controls, especially deuterium and helium. The BBN abundance coordinate is the transported one:

```text
eta10_BBN = 5.998071834744
Omega_b h^2_BBN = 0.021898765370
```

The second layer is the CPTG lithium source-network test. The [cosmological lithium problem](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/CPTG_Cosmological_Lithium_Problem.pdf) is treated as a surviving mass-seven abundance problem because most final primordial lithium is carried through `7Be` during BBN and later appears as `7Li`. The locked CPTG lithium gate is:

```text
Y_7,CPTG = Y_7,raw / pi
integral Gamma_7 dt = ln(pi)
```

Operationally, the gate is applied to the live `7Li` and `7Be` channels inside the source network, and the network is then re-evolved. It is not a post-processing division applied after the final abundance table is produced.

The claim level is a two-code source-network validation under the stated background-admissible standard. It should not be read as an assertion that every possible BBN code, every stellar lithium systematic, or every observational systematic has been exhausted.

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

## What CPTG Is Not

CPTG is not a dark matter halo-fitting model.

CPTG is not a MOND interpolation-function model.

CPTG is not presented as a completed replacement for general relativity or ΛCDM. The repository contains reduced theoretical and numerical implementations designed to test whether nonlinear curvature polarization and curvature transport can reproduce key galaxy-scale, cluster-scale, and comparison-layer cosmological observations.

The included scripts implement reduced limits of the theory:

- the quasi-static, weak-field, approximately axisymmetric galaxy limit;
- the reduced merger-plane transport/lensing limit for dissociative clusters;
- comparison-layer cosmology audits that map CPTG-native quantities into conventional observational summaries.

They are not full numerical-relativity solvers for the complete covariant CPTG field equations. Cosmology-facing scripts are not automatically full Boltzmann, full DESI, or full weak-lensing likelihood implementations unless those specific pipelines are provided. The Route B Option 1 CMB result is a fixed curvature-transport comparison-coordinate bridge carried through CAMB and Planck likelihood plumbing; it should not be described as native CPTG perturbation-code validation or as a movable Boltzmann/source implementation. These reduced implementations test whether CPTG equations and comparison maps reproduce important observational signatures normally associated with dark matter, dark energy, or parameter tension.

---

## Relation to MOND and ΛCDM

The repository compares CPTG to MOND-style galaxy predictions and to the broader dark-matter-halo interpretation associated with ΛCDM, but these comparisons are not identical in type.

- **MOND** modifies the acceleration law and performs well when galaxy behavior follows a nearly universal low-acceleration relation.
- **ΛCDM** explains galaxy and cluster dynamics through non-baryonic dark matter, with individual galaxy rotation curves often modeled through halo fitting and related nuisance parameters.
- **CPTG** tests whether similar observed effects can emerge from baryon-sourced curvature polarization, curvature transport, and theory-derived structural organization.

The public SPARC benchmark evaluates CPTG directly against observed galaxy rotation data and includes a MOND-style comparison under the same loaded galaxy database. The Upsilon benchmark adds stellar mass-to-light freedom as a stricter comparison layer. These tests are included so the comparison can be reproduced rather than treated as a qualitative claim.

Compared with ΛCDM halo fitting, CPTG makes a different kind of test: it asks whether galaxy rotation behavior and dissociative cluster-lensing offsets can be reproduced geometrically through baryon-sourced curvature response rather than by fitting non-baryonic halo components.

## Recent CPTG Articles and Research Notes

Recent CPTG writing has expanded beyond the original galaxy and Bullet Cluster benchmarks into focused theory and validation articles, including CMB comparison-map closure, Route B Option 1 CMB comparison-coordinate bridge validation, Pantheon+ distance-shape tests, BBN and lithium source-network validation, weak-lensing S8 diagnostics, DESI compressed-coordinate tests, Hubble-tension bridge work, cosmological horizon-mechanism work, compact high-redshift galaxy stress tests, and cluster active-gate extensions.

These articles should be read as part of the active research program. Their claim levels vary by implementation maturity and are identified in the relevant papers or audit reports.

## Continuing Work

CPTG is being developed as an active research program rather than a single fixed software tool or one-time benchmark. Current development directions include:

- refining the mathematical connection between curvature transport, curvature polarization, and observed galaxy structure;
- testing whether structure-response distance refinements correlate with independent distance-quality indicators;
- extending cluster active-gate tests to larger same-aperture samples and improving treatment of gas, stellar, intracluster-light, lensing, and merger-aware decompositions;
- carrying the locked CMB comparison-map closure result into higher-resolution geometric projections into CMB observables;
- keeping numerical tests reproducible, compact, and open to criticism.

These continuing investigations are included to make the research path transparent. They should be read as active development directions, not settled conclusions.

## Citation

If referencing the CPTG framework, please cite:

Carter L. Glass Jr., *Curvature Polarization Transport Gravity: A Unified Geometric Framework for Cosmic Structure and Expansion*, [DOI: 10.13140/RG.2.2.26030.68164](https://doi.org/10.13140/RG.2.2.26030.68164).

For the repository and supporting code package, cite:

CPTG, Supporting Python Models, Benchmark Implementations, and Research References for Curvature Polarization Transport Gravity, companion resource, available at [https://github.com/CLG2025/CPTG](https://github.com/CLG2025/CPTG).

---

## Summary

CPTG is not a dark matter halo fit and is not a MOND interpolation law. It is a geometric gravity framework in which gravitational enhancement, lensing displacement, cosmological comparison quantities, CMB map-space closure, and possible Hubble-tension structure are modeled through curvature polarization, curvature transport, and branch-specific observational projection.

The public repository contains reduced numerical implementations, benchmark scripts, figures, manuscripts, and development notes intended for reproduction, criticism, and further theory testing. Galaxy rotation curves, reduced cluster-merger reconstruction, and cluster active-gate aperture tests represent the most direct public-scale benchmarks. Cosmology-facing work is organized by claim level in the relevant sections and dedicated papers.
