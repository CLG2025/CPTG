# CPTG
## Curvature Polarization Transport Gravity

## Plain-Language Summary

*Curvature Polarization Transport Gravity* (CPTG) is an active research framework exploring whether the effects usually attributed to dark matter can instead arise from the way spacetime curvature responds to ordinary baryonic matter.

In this view, galaxies do not require separate dark matter halos to explain their rotation curves. Instead, weak gravitational fields develop a nonlinear curvature-polarization response. In merging galaxy clusters, organized curvature can also be directionally transported, allowing lensing peaks to separate from hot gas without introducing an additional collisionless matter component.

The purpose of this repository is to make the CPTG theory, benchmark scripts, data package, and public figures available for inspection, reproduction, and criticism.

---

## Technical Overview

CPTG is a unified geometric gravity framework in which baryonic matter sources a nonlinear curvature response rather than requiring a separate non-baryonic dark matter halo. The theory is built around two linked mechanisms:

- **Curvature polarization**, which changes the effective gravitational response according to the strength and structure of the field.
- **Curvature transport**, which allows organized curvature to be redistributed directionally in dynamical systems.

In this framework, compact systems recover the Newtonian limit, galaxies probe the polarization-dominated low-acceleration regime, and dissociative cluster mergers probe the regime where polarization and directional transport act together to produce displaced lensing structure.

CPTG is therefore intended to connect galaxy rotation curves and cluster-merger lensing offsets within a single geometric framework.

---

## What This Repository Contains

This repository contains the public academic package for CPTG, including:

- the current CPTG theory manuscript,
- the SPARC galaxy rotation-curve benchmark,
- the Bullet Cluster reduced merger-plane benchmark,
- supporting SPARC data and metadata,
- comparison scripts against MOND-style galaxy predictions,
- benchmark figures and reconstruction images.

The recommended public download is **`CPTG_academic_package.zip`**, located in the **`/archive/`** folder.

This archive contains the SPARC galaxy benchmark script, the Bullet Cluster merger script, the SPARC rotation-curve database, the SPARC supporting metadata file, the Upsilon benchmark comparison script, and the CPTG research paper.

Additional Python files found in the same directory, if present, should be treated as version upgrades, development variants, or replacement implementations. Each script contains detailed internal comments explaining the equations, implementation choices, benchmark logic, and interpretation of the outputs.

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

<sup>Figure: CPTG Bullet Cluster kappa reconstruction showing gas–lensing separation.</sup>

---

## What CPTG Is Not

CPTG is not a dark matter halo-fitting model.

CPTG is not a MOND interpolation-function model.

CPTG is not presented as a completed replacement for general relativity or ΛCDM. The repository contains reduced theoretical and numerical implementations designed to test whether nonlinear curvature polarization and curvature transport can reproduce key galaxy-scale and cluster-scale observations.

The included scripts implement reduced limits of the theory:

- the quasi-static, weak-field, approximately axisymmetric galaxy limit;
- the reduced merger-plane transport/lensing limit for dissociative clusters.

They are not full numerical-relativity solvers for the complete covariant CPTG field equations. Their purpose is to test whether the reduced CPTG equations reproduce important observational signatures normally associated with dark matter.

---

## Relation to MOND and ΛCDM

CPTG can produce MOND-like weak-field behavior in galaxies, but it is not constructed as a MOND interpolation law. In CPTG, the enhanced galaxy-scale gravitational response is interpreted as a consequence of nonlinear curvature polarization.

Compared with ΛCDM, CPTG does not introduce a non-baryonic collisionless mass halo in the included galaxy or merger scripts. The apparent excess gravitational response is modeled as nonlinear curvature response sourced by baryonic structure, while dissociative cluster-lensing offsets are modeled through curvature polarization plus directional curvature transport.

In short:

- **MOND** modifies the acceleration law.
- **ΛCDM** adds non-baryonic dark matter.
- **CPTG** explores whether nonlinear curvature polarization and curvature transport can account for the same classes of observations geometrically.

---

## Public Research Status

CPTG is an active theoretical and computational research framework.

The repository is intended for:

- public review,
- theory development,
- numerical testing,
- benchmark reproduction,
- comparison with observational data,
- and criticism of the assumptions, reductions, and implementation choices.

The scripts and manuscript should be read together. The README gives a public overview, while the Python files contain detailed implementation comments and the manuscript contains the formal theoretical development.

---

## Summary

CPTG is not a dark matter halo fit and is not a MOND interpolation law. It is a geometric gravity framework in which gravitational enhancement and lensing displacement are modeled as consequences of nonlinear curvature polarization and curvature transport.

The included scripts are reduced numerical implementations of the galaxy and cluster-merger limits of the theory, designed for reproducible benchmarking, theory auditing, and direct comparison with observational data.
