# CPTG
## Curvature Polarization Transport Gravity

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
| BBN abundance comparison | Primary abundance comparison using CPTG transported baryon/acoustic quantities | Within representative observational bands |
| Weak-lensing S8 | Compressed comparison against representative weak-lensing and CMB S8 anchors | Diagnostic pass, not a full shear likelihood |
| CMB comparison layer | Locked cleaned comparison-layer recovery across high-ell profile robustness, lowE/lensing diagnostic stability, posterior-proxy checks, combined-stack profile validation, and combined-stack posterior validation | Control-level combined-stack profile and posterior validation under a locked cleaned comparison layer; amplitude/tau gate derivation and full perturbation implementation remain open |
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

The main public benchmark scripts are:

| Script | Purpose |
|---|---|
| `SPARC_CPTG_MOND_Benchmark.py` | Galaxy rotation-curve benchmark against SPARC data. |
| `CPTG_Bullet_Cluster_Merger.py` | Reduced merger-plane curvature-transport/lensing benchmark. |
| Cluster active-gate calculator (Pending) | Single-aperture and aperture-ladder cluster-response calculations from baryonic loading, support temperature, redshift, and aperture. |
| `CPTG_MOND_Upsilon_SPARC_Benchmark.py` | MOND/CPTG comparison with stellar mass-to-light freedom. |

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

These values define the fixed branch used in the current comparison-layer audits.

### CMB Comparison Layer

CPTG CMB work is organized as a comparison layer between CPTG-native geometric quantities and the conventional variables used by CMB likelihoods. The early-universe ordering is not treated as a direct copy of the late-time galaxy regime. The working physical order is:

> early expansion -> curvature transport through the hot plasma -> organized pre-recombination gravitational source structure -> recombination imprint -> later curvature polarization in bound systems.

In this view, curvature transport precedes galaxy-scale curvature polarization. The CMB question is whether the gravitational role usually assigned to cold dark matter can be represented by an early transported-curvature source and a fixed acoustic-transport comparison map.

The current locked cleaned CMB comparison layer reaches control-level combined-stack profile and posterior validation across Plik, decisive full NPIPE CamSpec, and SPT/ACT, with the lensing-amplitude setting fixed at `A_lens = 1`. It also includes high-ell profile robustness, converged Plik/SPT posterior-proxy checks, and low-ell/lensing diagnostic stability.

The current locked CMB comparison map uses:

```text
omega_c,eff h^2 = omega_c,CPTG h^2 D_c
A_s,eff = A_s,CPTG pi/3
tau_eff = tau_CPTG G_T^2
```

with:

```text
D_c = 0.9735353159758136
G_T^2 = 0.8928920596100127
tau_eff = 0.05893087593426084
A_lens = 1
```

This is stronger than a high-ell-only validation candidate. It should be described as a locked cleaned comparison-layer CMB pass through the completed profile, posterior-proxy, lowE/lensing diagnostic, combined-stack profile, and combined-stack posterior tests. It is not the same as a full Boltzmann-code or full perturbation-equation implementation. The combined-stack posterior validation item is now completed under the locked comparison layer. The remaining CMB work is theoretical and implementation-facing: deriving or fully motivating the scalar-amplitude gate `A_s -> A_s pi/3` and the optical-depth gate `tau -> tau G_T^2`, and expressing the comparison layer inside a full perturbation-equation framework.

### Pantheon+ Supernova Distance-Shape Test

CPTG has been tested against Pantheon+ supernova distance-shape data using a full-covariance comparison with a marginalized intercept. This is a distance-shape test, not a local H0 calibration claim. The purpose is to ask whether the CPTG expansion branch can reproduce the relative supernova distance trend once the absolute calibration is marginalized.

### BBN Abundance Comparison

CPTG abundance work compares transported baryon and acoustic quantities against primary BBN abundance summaries. These comparisons are useful because they test whether the CPTG branch remains compatible with early-universe light-element constraints. They should be read as abundance comparisons, not as a complete replacement for a full nuclear reaction-network derivation inside CPTG.

### Weak-Lensing S8 Comparison

CPTG weak-lensing work currently uses compressed S8 comparisons against representative weak-lensing and CMB anchors. These tests are diagnostic: they show whether the CPTG growth/lensing branch lies within representative observational bands, but they are not a substitute for a full shear-correlation likelihood or survey-level weak-lensing pipeline.

### DESI DR1 Compressed ShapeFit and BAO Quarter-Ruler

CPTG large-scale-structure work separates DESI comparisons into layers. The current compressed ShapeFit coordinate comparison uses official DESI DR1 HDF5 likelihood containers and is a compressed-coordinate pass. The BAO quarter-ruler is a strong coordinate-wrapper diagnostic using the CPTG transport relation `G_T^(-1/4)`, but it is not presented as a raw official non-unity-`q` runtime likelihood response.

The full-shape AP/growth work remains an exploratory spectrum-shell diagnostic. It is not a raw DESI full-shape validation claim and should not be described as one. A full raw DESI validation requires nuisance-preserving AP, RSD, tracer-window, covariance, nuisance, counterterm, and stochastic machinery to be wired consistently through the official likelihood path.

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

They are not full numerical-relativity solvers for the complete covariant CPTG field equations. Cosmology-facing scripts are not automatically full Boltzmann, full DESI, or full weak-lensing likelihood implementations unless those specific pipelines are provided. Their purpose is to test whether reduced CPTG equations and comparison maps reproduce important observational signatures normally associated with dark matter, dark energy, or parameter tension.

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

## Claim-Level Guide

CPTG results in this repository should be read at different levels of maturity.

| Label | Meaning |
|---|---|
| Public benchmark | Script and data package are available for reproduction. |
| Comparison-layer pass | A locked CPTG comparison map passes the stated likelihood or compressed-coordinate test without retuning the branch. |
| Validation candidate | A result passes strong locked tests but still needs broader independent confirmation or a deeper physical implementation. |
| Diagnostic pass | A compressed or reduced comparison succeeds, but the full likelihood or full physical implementation remains future work. |
| Exploratory | Used to scope theory behavior, not presented as a settled result. |

This distinction is especially important for cosmology-facing work. The current CMB status is a locked cleaned comparison-layer pass through completed high-ell profile robustness, lowE/lensing diagnostic stability, posterior-proxy checks, combined-stack profile validation, and combined-stack posterior validation. This remains a comparison-layer result rather than a full Boltzmann-code or full perturbation-equation implementation; the remaining CMB theory work is amplitude/tau gate derivation and perturbation-level implementation. DESI ShapeFit and BAO currently sit at compressed-coordinate and coordinate-wrapper level; the DESI full-shape AP/growth spectrum-shell work remains exploratory until the official nuisance-preserving likelihood machinery is used.

CPTG comparison-layer results should not be confused with full Boltzmann-code validation, full raw DESI likelihood validation, or full weak-lensing shear-likelihood validation unless those specific implementations are provided.

---

## Recent CPTG Articles and Drafts

Recent CPTG writing has expanded beyond the original galaxy and Bullet Cluster benchmarks into several focused theory and validation articles:

- unified CPTG cosmology and comparison-layer framework,
- Pantheon+ supernova distance-shape comparison,
- BBN transported-baryon abundance comparison,
- weak-lensing S8 compressed comparison,
- DESI DR1 compressed ShapeFit coordinate-level dry run and BAO quarter-ruler diagnostic,
- Hubble-tension bridge article,
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

The locked CMB comparison layer now provides a strong set of fixed-row likelihood checks, but it is still a comparison-layer construction. A deeper CMB implementation requires a perturbation-equation treatment comparable in role to CLASS or CAMB. The long-term goal is to express early curvature transport, acoustic source structure, optical-depth visibility, and scalar-amplitude response directly in a physical perturbation model rather than relying on comparison-layer maps alone.

### Direction of Development

The next development goals are:

- strengthen the mathematical connection between curvature transport, curvature polarization, and observed galaxy structure;
- evaluate whether SRD distance refinements correlate with independent distance-quality indicators;
- extend cluster active-gate tests to larger same-aperture samples and connect them to gas, galaxy, lensing, and merger-aware decompositions;
- convert the locked CMB acoustic-transport comparison layer into a more physical perturbation-equation model;
- keep all numerical tests reproducible, compact, and open to criticism.

These continuing investigations are included to make the research path transparent. They should be read as active development directions, not settled conclusions.

---

## Citation

If referencing the CPTG framework, please cite:

Carter L. Glass Jr., *Curvature Polarization Transport Gravity: A Unified Geometric Framework for Cosmic Structure and Expansion*,\
[DOI: 10.13140/RG.2.2.26030.68164](https://doi.org/10.13140/RG.2.2.26030.68164).

For the repository and supporting code package, cite:

CPTG, Supporting Python Models, Benchmark Implementations, and Research References for Curvature Polarization Transport Gravity, companion resource, available at [https://github.com/CLG2025/CPTG](https://github.com/CLG2025/CPTG).

---

## Summary

CPTG is not a dark matter halo fit and is not a MOND interpolation law. It is a geometric gravity framework in which gravitational enhancement, lensing displacement, cosmological comparison quantities, and possible Hubble-tension structure are modeled through curvature polarization, curvature transport, and branch-specific observational projection.

The public repository contains reduced numerical implementations, benchmark scripts, figures, manuscripts, and development notes intended for reproduction, criticism, and further theory testing. Galaxy rotation curves, reduced cluster-merger reconstruction, and cluster active-gate aperture tests represent the most direct public-scale benchmarks, while cosmology-facing work is organized by claim level: comparison-layer pass, diagnostic pass, validation candidate, or exploratory extension depending on the maturity of the implementation.
