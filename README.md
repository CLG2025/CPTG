# CPTG

## Curvature Polarization Transport Gravity

---

## Contents

- [Core CPTG Papers](#start-here-core-cptg-papers)
- [Overview](#overview)
- [Current Research Status](#current-research-status)
- [Repository and Tools](#repository-and-tools)
- [Reproducing the Public Benchmarks](#reproducing-the-public-benchmarks)
- [Galaxy-Scale Tests](#galaxy-scale-test-cptg-sparc-browser-workbench)
- [Structural Mode N](#structural-mode-n)
- [Bullet Cluster Benchmark](#cluster-merger-test-bullet-cluster-benchmark)
- [Universal Geometric Nuclear-Reaction Theory](#universal-geometric-nuclear-reaction-theory)
- [Cosmology and Comparison-Layer Tests](#cosmology-and-comparison-layer-tests)
- [Relation to MOND and ΛCDM](#relation-to-mond-and-λcdm)
- [Recent Progress](#recent-progress-and-active-development)
- [Citation](#citation)

---

## Start Here: Core CPTG Papers

- **[Curvature Polarization Transport Gravity: A Unified Geometric Framework for Structure, Cosmology, and Nuclear Reactions](https://doi.org/10.5281/zenodo.22441039)**  
  Primary CPTG theory paper. It establishes the variational framework, curvature-polarization and curvature-transport laws, scale-dependent reductions, the native geometric branch and its observational projections, and the cross-scale structure connecting gravitational, cosmological, and nuclear applications.

- **[CPTG Geometric Pi Branch: Cosmological Comparison Coordinates](https://doi.org/10.5281/zenodo.22426857)**  
  Comparison-coordinate guide for the locked native geometric π branch and its acoustic and luminosity projections.

- **[The Science Behind CPTG: A Framework of Scales for Gravity, Structure, Cosmology, and Nuclear Reactions](https://doi.org/10.5281/zenodo.22421593)**  
  Public-facing introduction to CPTG as a framework of scales spanning gravitational, structural, cosmological, and nuclear sectors.

- **[Structural Mode N in CPTG: Galaxy Structural Classification from the Solved CPTG Field](https://doi.org/10.5281/zenodo.22436682)**  
  Dedicated Structural Mode \(N\) paper. \(N\) is derived from the solved CPTG field as a post-solution measure of polarization/transport organization and provides the basis for downstream CSMI structural labels.

- **[Geometric Nuclear Reaction Theory in CPTG: Deuterium-Proton Capture and Primordial Mass-Seven Transport](https://doi.org/10.5281/zenodo.22425309)**  
  Commissioning-stage nuclear-reaction paper establishing the first complete CPTG source-to-reaction-to-rate-to-network chain.

- **[A Universal Geometric Theory of Nuclear Reactions in CPTG: The Four-Sector Foundation of Vertex, Bridge, Closure, and Saturation](https://doi.org/10.5281/zenodo.22439971)**  
  Universal nuclear-reaction parent paper presenting the four-sector architecture, governing reaction laws, conservation structure, and validation hierarchy.

- **[PRIMAT-Anchored Reduced-Graph Reachability and the Complete A=1–119 Mass-Sector Register](https://doi.org/10.5281/zenodo.22442178)**  
  Computational companion to the universal theory. PRIMAT supplies native trajectory and endpoint authority through A=23; the prescribed-bath reduced-transport continuation carries the register through A=119 and separately qualifies the natural A=338 structural frontier. The external \(Y_A\) values are prescribed-bath reduced-transport populations, not self-consistent final abundances of a coupled PRIMAT heavy-element network and not asserted primordial heavy-element yield predictions.

---

## Repository and Tools

The repository combines public theory papers, reduced-limit benchmark code, interactive workbenches, figures, comparison-layer materials, and reproducibility evidence.

- **CPTG academic package** — the compact reproducibility archive for the core SPARC galaxy and Bullet Cluster reduced-limit benchmarks, distributed as `CPTG_academic_package.zip`.
- **CPTG SPARC Browser Workbench v1.12.0** — standalone local SPARC analysis for Windows, with cross-platform Python source and CPTG/MOND comparison tools. [View the interface](images/CPTG_SPARC_Browser_Workbench.png) · [Download v1.12.0](https://github.com/CLG2025/CPTG/releases/tag/v1.12.0)
- **CPTG Reaction Workbench** — a PC-first nuclear-reaction interface in active development for deterministic calculation, comparison, project execution, evidence capture, replay, and external-model integration.
- **CPTG Pi-Bridge** — a local [research workbench](images/CPTG-Pi-Bridge-Local-Workbench.png) in beta development for loading public astronomy/cosmology datasets, applying CPTG comparison branches, auditing results, and exporting reproducible validation packages.
- **Research and evidence archive** — current manuscripts, protocol documents, validation packages, source-network records, comparison-layer scripts, CMB source/data notes, figures, and supporting reproducibility materials.

The compact academic package is the recommended starting point for the public galaxy and Bullet Cluster benchmarks. The larger workbenches are distributed separately because they serve interactive research workflows rather than the compact benchmark environment.

---

## Overview

*Curvature Polarization Transport Gravity* (CPTG) is a geometric-gravity framework in which ordinary matter sources curvature polarization and curvature transport across different physical scales.

Its two linked mechanisms are:

* **Curvature polarization**, which modifies the effective gravitational response according to field strength and structural organization.
* **Curvature transport**, which carries organized curvature through the geometry according to the dynamical state and symmetry of the system.

CPTG is organized as a framework of scale-specific reductions rather than a single reduced formula applied everywhere. The published theory spans galaxy dynamics, Structural Mode \(N\), dissociative cluster-merger lensing, finite strong-field compact objects, the native geometric π branch and its observational projections, CMB comparison-map tests, the Hubble and horizon sectors, and a universal geometric nuclear-reaction extension.

The public record separates **published theory**, **closed validations**, **reproducible benchmarks**, and **continuing software or publication development**. Reproducibility is organized around fixed laws, declared comparison coordinates, source manifests, direct residual tables, and native-network or evidence-package validation rather than unrestricted parameter fitting.

---

## Current Research Status

The current CPTG record is organized by published theory, closed validations, reproducible benchmarks, comparison-coordinate studies, and continuing software development.

| Area | Current CPTG status | Claim level |
|---|---|---|
| SPARC galaxy rotation curves | Public reduced-limit SPARC tests and the interactive browser workbench | Reproducible galaxy-scale benchmark; see [rotation-curve paper](https://doi.org/10.5281/zenodo.22418189) |
| Structural Mode \(N\) | Post-solution structural measure derived from the solved CPTG field, with downstream CSMI labels in the galaxy workbench | Reproducible structural diagnostic; see [Structural Mode \(N\)](https://doi.org/10.5281/zenodo.22436682) |
| Bullet Cluster merger plane | Public reduced merger-plane curvature-transport/lensing reconstruction | Reproducible cluster-merger benchmark |
| Finite compact objects / black holes | Curvature-bounded strong-field admissibility framework with finite physical interiors and GR exterior recovery; compact boundary-value closure remains active theory work | See [compact-object paper](https://doi.org/10.5281/zenodo.22424445) |
| Universal geometric nuclear-reaction theory | Closed four-sector foundation spanning free nucleons, deuterium, the mass-three pair, and helium-4 | [Universal parent paper](https://doi.org/10.5281/zenodo.22439971); PRIMAT primary authority — **CLOSED/PASS**; PArthENoPE second authority — **CLOSED/PASS**; PRyMordial third authority — **CLOSED/PASS** |
| Computational companion / post-silicon continuation | PRIMAT-native authority through A=23, complete prescribed-bath reduced-transport register through A=119, and separately qualified structural continuation through A=338 | [Computational companion](https://doi.org/10.5281/zenodo.22442178); external \(Y_A\) values are reduced-transport populations, not coupled PRIMAT heavy-element yield predictions |
| Nuclear-reaction interface and exchange layer | PC-first CPTG Reaction Workbench, deterministic evidence/replay architecture, formula-package authority, compiled-runtime boundary, and external scientific-model interface | Active engineering implementation and qualification |
| Pantheon+ supernova distances | Full-covariance relative distance-shape comparison with marginalized intercept | Distance-shape comparison, not a local \(H_0\) calibration claim |
| BBN abundance and lithium tests | Transported BBN coordinate, locked live A=7 transport law, PRIMAT-native mass-seven validation, and the earlier PRyMordial commissioning result | See [cosmological lithium paper](https://doi.org/10.5281/zenodo.22420032) and [geometric nuclear-reaction paper](https://doi.org/10.5281/zenodo.22425309) |
| Weak-lensing \(S_8\) | Compressed comparison against representative weak-lensing and CMB \(S_8\) anchors | Diagnostic comparison, not a full shear likelihood |
| CMB comparison-map closure | Locked geometric-π comparison construction tested against real Planck/WMAP temperature-map products and null controls | [Real-map comparison-map closure](https://doi.org/10.5281/zenodo.22413491) |
| CMB Route B Option 1 | Fixed amplitude-level curvature-transport bridge studied at spectrum and likelihood-coordinate level | Development-stage comparison-coordinate bridge; see [Route B report](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/CPTG_Route_B_Option1_Curvature_Transport_Geometry_Bridge_Report_v1.pdf) |
| DESI compressed ShapeFit and BAO | Compressed-coordinate and ruler-wrapper diagnostics | Coordinate-level support, not full raw DESI validation |
| Horizon and Hubble-tension mechanisms | Native-branch and projection papers relating CPTG geometric time/rate structure to observational comparison layers | [Horizon paper](https://doi.org/10.5281/zenodo.22430125) and [Hubble-tension bridge](https://doi.org/10.5281/zenodo.22429502) |

---

## Reproducing the Public Benchmarks

The public benchmarks are intended to be inspectable and reproducible.

1. Download or clone the repository, or obtain the relevant published/release package.
2. Use **`CPTG_academic_package.zip`** for the compact SPARC and Bullet Cluster benchmark set.
3. Extract the selected package into a working location.
4. Run the documented benchmark scripts with Python 3.
5. Compare generated outputs against the included figures, summaries, and evidence records.

Reproducibility depends on retaining the package contents and declared inputs, or on supplying explicit input/output paths when supported by the scripts. Planck and WMAP FITS products are not bundled with CMB map-closure packages; they must be supplied from the documented public sources.

The main public benchmark scripts are:

| Package or tool | Purpose |
|---|---|
| `SPARC_CPTG_MOND_Benchmark.py` | Original galaxy rotation-curve benchmark against SPARC data. |
| `CPTG_Bullet_Cluster_Merger.py` | Reduced merger-plane curvature-transport/lensing benchmark. |
| `CPTG_MOND_Upsilon_SPARC_Benchmark.py` | MOND/CPTG comparison with stellar mass-to-light freedom. |

---

## Galaxy-Scale Test: CPTG SPARC Browser Workbench

The recently updated **CPTG SPARC Browser Workbench v1.12.0** provides a local browser interface for testing CPTG and MOND against SPARC galaxy rotation-curve data.

The standalone [release package](https://github.com/CLG2025/CPTG/releases/tag/v1.12.0) includes the SPARC galaxy files and metadata needed to begin running analyses immediately. It supports:

- searchable single and multi-galaxy selection,
- individual-galaxy and batch analysis,
- metadata views for all, primary, excluded, and unmatched galaxies,
- primary-sample filtering when a metadata file is active,
- CPTG and MOND rotation-curve comparisons,
- averaged normalized rotation curves,
- averaged RAR scatter versus radius,
- compact result tables reporting total points, total χ², χ² per point, RMS residuals, and mean observed/model velocities,
- galaxy-level fit, Structural Mode N, and downstream CSMI summaries,
- optional CSV, JSON, PNG, and ZIP output saving.

All calculations are performed locally. Each galaxy is solved independently before aggregate statistics and plots are generated. By default, completed analyses are displayed in the browser without retaining output files. Files are written only when the user explicitly selects **Save result files**.

When metadata is enabled, the galaxy list defaults to the primary sample while preserving the user's previous metadata-view selection during the browser session. Primary, excluded, and unmatched galaxies remain separately identifiable, and checked galaxies from any active view can be processed.

For aggregate plots, each galaxy is normalized independently and interpolated onto a shared normalized radial grid. Each galaxy receives equal weight at each grid location, and the outer endpoint at `r / r_max = 1` is retained.

[SPARC data source](https://astroweb.case.edu/SPARC/): Lelli, McGaugh, and Schombert, *The Astronomical Journal* 152, 157 (2016), [“SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves”](https://ui.adsabs.harvard.edu/abs/2016AJ....152..157L/abstract).

The figure below summarizes averaged SPARC results produced by the workbench.

![CPTG SPARC Browser Workbench summary showing the average normalized SPARC rotation curve on the left and the average normalized RAR scatter versus radius on the right.](images/combined_workbench_plots_side_by_side.png)

<sup>Figure: CPTG SPARC Browser Workbench averaged results for the full 175-galaxy SPARC run. Primary-sample metadata filtering was not applied to this figure.</sup>

---

## Structural Mode N

[Structural Mode **N**](https://doi.org/10.5281/zenodo.22436682) is a dimensionless CPTG diagnostic derived from the solved field after the galaxy solution is obtained.

It measures the radial organization of the solved polarization/transport structure:

**N = R / λ**

where:

* **R** is the outer solved radius.
* **λ** is the curvature-weighted structural scale.

The workbench can then translate the continuous mode value into a downstream **CSMI Type**:

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

The continuous mode \(N\) is the theory-derived structural quantity. The CSMI Type is a downstream workbench label assigned from that solved value rather than from catalog morphology.

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

The model is scored against observed Bullet Cluster gas, galaxy, and lensing separations.

### Primary JWST and lensing benchmark reference

The primary high-resolution observational reference used for the Bullet Cluster mass-map interpretation and offset benchmarks in this reduced model is:

> Cha, S., Cho, B. Y., Joo, H., Lee, W., HyeongHan, K., Scofield, Z. P., Finner, K., & Jee, M. J. (2025), “[A High-Caliber View of the Bullet Cluster through JWST Strong and Weak Lensing Analyses](https://arxiv.org/abs/2503.21870),” *The Astrophysical Journal Letters*, **987**, L15.

The CPTG benchmark uses this study for:

- the high-resolution JWST mass-map interpretation;
- the Bullet subcluster mass-galaxy offset benchmark of **17.78 ± 0.66 kpc**;
- the Bullet mass-ICM offset target of approximately **150 kpc**;
- the main-cluster north and south mass-ICM offset targets of approximately **200 kpc** and **400 kpc**;
- the interpretation that the main cluster contains resolved north/south substructure and that the merger geometry is more complex than a simple binary-merger picture.

The public significance of this test is that CPTG attempts to address not only galaxy rotation curves, but also dissociative cluster mergers, which are often considered strong evidence for collisionless dark matter.

---

The third benchmark figure shows how CPTG reconstructs displaced lensing structure in the Bullet Cluster merger plane.

![Normalized CPTG kappa reconstruction of the Bullet Cluster merger plane. The map shows two main convergence structures: a compact Bullet-side lensing feature on the left, displaced from the Bullet gas peak, and a larger main-cluster lensing structure on the right with north and south substructure. White contours trace the strongest reconstructed convergence regions. Markers identify Bullet and main gas peaks, galaxy peaks, lensing peaks, and main-cluster north/south lens peaks. A scale bar marks 100 kpc.](images/CPTG-Curvature-Transport-Model.png)

<sup>Figure: CPTG Bullet Cluster kappa reconstruction showing gas-lensing separation.</sup>

---


## Universal Geometric Nuclear-Reaction Theory

CPTG has transitioned from a commissioned deuterium-proton capture model to a universal geometric nuclear-reaction theory with a validated four-sector foundation:

- **Free nucleons (`n`, `p`) — vertex**
- **Deuterium — bridge**
- **Mass three (`³H`, `³He`) — closure**
- **Helium-4 (`⁴He`) — saturation**

These sectors form one ordered transport-polarization architecture rather than four unrelated reaction constructions. The dynamic vertex contains free neutrons and protons; deuterium supplies the first bound bridge; tritium and helium-3 form the two charge orientations of the mass-three closure sector; and helium-4 is the saturated endpoint.

### Universal paper and computational companion

The [universal parent paper](https://doi.org/10.5281/zenodo.22439971) presents the closed four-sector foundation, governing geometric reaction laws, conservation structure, physical interpretation, and accepted native-network validation chain.

The updated [computational companion](https://doi.org/10.5281/zenodo.22442178) carries that fixed architecture into an explicit reproducible calculation. PRIMAT supplies the native trajectory and endpoint authority through A=23; beyond the native boundary, a declared seed-free prescribed-bath reduced graph defines the continuation topology under frozen reachability, convergence, and source-robustness controls. The resulting external `Y_A` values are reduced-transport populations.

### Architecture and scalability

The theory separates **ordered transport**, which moves baryonic content through vertex, bridge, closure, and saturation, from **internal polarization**, which preserves the neutron-proton and tritium-helium-3 orientation required by charge conservation. Structural closure fixes the physical coordinate, baryon and charge constraints, reaction-source basis, curvature-response hierarchy, and separation between direct source current and final network response.

Fixed-law scalability does not require different network codes to have identical internal currents, source kernels, integration measures, or solver trajectories. It requires the same geometric law, source-coordinate construction, conservation structure, and baryon-density dependence to carry across reaction channels without a separately fitted geometric rule for each reaction or code.

### Commissioning foundation

The published *[Geometric Nuclear Reaction Theory in CPTG: Deuterium-Proton Capture and Primordial Mass-Seven Transport](https://doi.org/10.5281/zenodo.22425309)* remains the commissioning-stage foundation. It carried `D(p,γ)³He` from a native source-state construction through reaction amplitude, astrophysical S-factor, thermonuclear-rate interface, transported BBN coordinate, and live primordial mass-seven transport.

That paper reported a zero-energy S-factor comparison at **−0.149960σ**, a solar-Gamow comparison at approximately **−0.02043σ**, and a PRyMordial gated-lithium result at **+0.90593σ** while preserving deuterium and helium controls.

### Native-network authority

Network results are ranked by native authoritative capability rather than forced into one common reduced framework:

1. **PRIMAT v0.3.2 — primary authority**
2. **PArthENoPE 3.0 — second authority**
3. **PRyMordial — third authority, CLOSED/PASS**

#### PRIMAT primary authority

| PRIMAT headline metric | Result |
|---|---:|
| Committed native rows | **20,550/20,550** |
| Native reactions | **428/428** |
| Matched plus/minus pairs | **10,272/10,272** |
| Complete eight-branch ladders | **2,568/2,568** |
| Frozen native source/Jacobian predictions | **84/84** |
| Primary resolved reaction-anchor tests | **48/48 PASS** |
| Worst primary endpoint-vector discrepancy | **0.0367629%** |
| Minimum primary endpoint direction cosine | **0.9999999921** |

The full campaign also closed its row-integrity and direct native rate-key checks, with 59 finite positive final nuclides on every committed row. The separate source/Jacobian campaign validates the PRIMAT-native first-order source-to-endpoint mechanism over the tested light-sector domain; it does not claim a network-independent static endpoint map, second-order variational closure, or native heavy-nucleus authority.

Primary paper-facing evidence:

```text
CPTG_PRIMAT_PaperReadyValidationEvidence_20260817_r02.zip
SHA-256: b41c8ee49477d8330deb56c85a230d8dcc1d4bba80a1405251a4d6ea7f9b3205
Prediction freeze SHA-256:
679e3e6fc7432c0592004b9bf0985b0756323ceb327050f3303ca6c816f31bff
```

#### PArthENoPE second authority

PArthENoPE 3.0 remains the completed second authority. Its full-network campaign closed **695/695 native rows**, **338/338 matched branch pairs**, and **84/84 complete ladders**; the declared Reaction-20 endpoint direction cosine was **0.999979437501036**. A later clean-room native-physics reconstruction independently recovered the rank-4 source architecture and passed **6/6 held-out rows across 3/3 density anchors under two numerical profiles**, without refitting the frozen construction.

Accepted full-network evidence:

```text
CPTG_v129_r109_PArthENoPE_CPTG_Reaction20_ReplicationAuthorityBundle_20260731_r02.zip
SHA-256: 3d42d1cb1d710248841db8d7b1ceafcc7569f092b9b35c9f4114ce9615074cfc
```

Clean-room evidence:

```text
CPTG_PARTHENOPE_CLEANROOM_FINAL_RESULTS.zip
SHA-256: 8d503dd80036e6917500ca8367d8830d727930a23e8779fa3497bf9c1cf4d903
Construction freeze SHA-256:
770139cfb7b98f5c0cf6e5a038ea32aee4fb8c3430ed6fb02f38292221fe63d3
```

These results establish an independent PArthENoPE-native implementation check without asserting equality of PRIMAT and PArthENoPE internal currents, Jacobians, source kernels, trajectories, or solvers.

#### PRyMordial third authority

PRyMordial is the completed third authority. Candidate C-R production qualification passed, and the fresh **3030/3030-row full-network authority campaign** closed successfully with the final integrity, response-ladder, and locked Reaction-20 decision gates **PASS**. Earlier response and clean-room results remain supporting evidence alongside the completed full-network authority object.

### Computational companion results

The [computational companion](https://doi.org/10.5281/zenodo.22442178) separates native authority from the reduced-graph continuation topology and its reduced-transport populations:

- **Native PRIMAT domain:** A≤23, using the frozen PRIMAT trajectory and endpoint authority.
- **Complete reduced-graph topology / reduced-transport register:** A=24–119, with positive modeled mass-sector support at all six frozen baryon-density anchors and no imported heavy-sector seed.
- **Natural structural frontier:** the same selected reduced-graph rule remains gap-free through A=338; A=339 is disconnected under the declared topology.
- **Frontier numerical qualification:** the accepted 512/1024/2048 two-scheme campaign passes all six anchors under unchanged convergence gates. The earlier 128/256/512 run is retained as a fail-closed numerical non-qualification.
- **Source-tail robustness:** all **18/18** retained-source cases satisfy their applicable preregistered requirements, including all **12/12** hard-gated cases. After restoring unrenormalized source amplitude, the largest absolute departure of the A=338 truncated-to-baseline endpoint ratio from unity is **2.36 × 10⁻¹³**.
- **Native mass-seven validation:** all **7/7** paired PRIMAT coordinates pass the locked transport test; the worst relative survival disagreement from the locked target is **0.018083%**, with negligible non-mass-seven control shifts.

These results describe the declared reduced operator and its numerical behavior. The external \(Y_A\) values are prescribed-bath reduced-transport populations, not self-consistent final abundances of a coupled PRIMAT heavy-element network and not asserted primordial heavy-element yield predictions.

Controlling public evidence:

```text
CPTG_PRIMAT_A1A119_FINAL_EVIDENCE_20260818-071825_AUDITED_SELFCONTAINED_r01.zip
SHA-256: 39549f1eba0201999aa0953d9ed8b36d4526c5df3ef70c0f11cdc9ae57449620

CPTG_PRIMAT_NativeA7_LiveTransport_CoreTheoryValidation_FINAL_EVIDENCE_20260818.zip
SHA-256: 2a1f52eea6020753b08f838be54cb25005fe79506f2d05245e6dbd8bc8d0d315

CPTG_PRIMAT_PostA119_A160A338_FinerFrontierContinuumQualification_FINAL_EVIDENCE_20260818_r01.zip
SHA-256: b0733da182a4a31a810ccd97e23b33389e11099b5b8bfc72e32501157e7dd68c

CPTG_PRIMAT_PostA119_A160A338_SourceTail_FINAL_SELFSEALED_EVIDENCE_20260819_r01.zip
SHA-256: d78aae075d3fe579874d3002fdee6b4d47d98712df9dd1ff85d2f9022e4f246a
```

### Completed validation and continuing public development

The three-network nuclear authority hierarchy is complete within its declared scope:

1. **PRIMAT v0.3.2 — primary authority — CLOSED/PASS**
2. **PArthENoPE 3.0 — second authority — CLOSED/PASS**
3. **PRyMordial — third authority — CLOSED/PASS**

Additional future cross-network or post-silicon studies, if undertaken, are new validation domains rather than unfinished work in the closed authority hierarchy.

The CPTG Reaction Workbench remains under active development as a deterministic interface for approved reaction formula and coefficient packages, comparison projects, evidence capture, replay, and external-model integration.


### BBN software citations

- C. Pitrou, A. Coc, J.-P. Uzan, and E. Vangioni, “Precision big bang nucleosynthesis with improved Helium-4 predictions,” *Physics Reports* **754**, 1–66 (2018), [doi:10.1016/j.physrep.2018.04.005](https://doi.org/10.1016/j.physrep.2018.04.005). Primary citation for PRIMAT.
- S. Gariazzo, P. F. de Salas, O. Pisanti, and R. Consiglio, “PArthENoPE revolutions,” *Computer Physics Communications* **271**, 108205 (2022), [doi:10.1016/j.cpc.2021.108205](https://doi.org/10.1016/j.cpc.2021.108205).
- A.-K. Burns, T. M. P. Tait, and M. Valli, “PRyMordial: the first three minutes, within and beyond the standard model,” *The European Physical Journal C* **84**, 86 (2024), [doi:10.1140/epjc/s10052-024-12442-0](https://doi.org/10.1140/epjc/s10052-024-12442-0).

---

## Cosmology and Comparison-Layer Tests

CPTG cosmology-facing work is organized around the distinction between CPTG-native geometric quantities and conventional observational summaries. The goal is not to force CPTG into standard parameter language, but to make controlled comparisons with quantities commonly reported from supernova, CMB, abundance, growth, and large-scale-structure analyses.

### Current Locked Geometric π Branch

The current locked CPTG comparison branch is defined by:

```text
p_C = pi
p_ac = 3 - pi/100 = 2.968584073464
G_T = p_ac / p_C = 0.944929658551
sqrt(G_T) = 0.972074924351
H0^(pi) = 69.4162507897 km s^-1 Mpc^-1
H0_CMB^CPTG = 67.4777967351 km s^-1 Mpc^-1
A_lens = 1
```

The complete fixed CMB comparison row, including baryon, matter, spectral, amplitude, optical-depth, and effective-radiation coordinates, is maintained in the geometric-π paper and the strict CMB rerun package. Those values are locked before map tests and are not refit to individual CMB maps.

### CMB Comparison-Map Closure

CPTG CMB map work is organized as a [real-map comparison test](https://doi.org/10.5281/zenodo.22413491) between the locked geometric-π CMB branch and public CMB map products. The current paper uses real Planck component maps, Planck split maps, and WMAP low-ell support products.

<p align="center">
  <img src="images/fig_visual_fitted.png" alt="Observed Planck SMICA vs fitted CPTG comparison map" width="70%">
</p>

<sup>Figure: SMICA visual comparison from the CMB comparison-map closure paper. Top: observed Planck SMICA temperature map. Center: fitted CPTG comparison map. Bottom: observed-minus-fitted-CPTG residual.</sup>

The map-space procedure uses the same comparison coordinate for CPTG, the Planck envelope, and controls. It reads the temperature field from the public CMB map product, applies the documented mask, converts to microkelvin, downgrades to `Nside = 256`, removes the monopole and dipole on the valid sky, and evaluates fitted residuals under the same amplitude-plus-offset rule:

```text
T_fit(nhat) = A T_template(nhat) + B
```

The central public result is that the locked CPTG geometric-π branch reaches near-degenerate CMB comparison-map closure with the Planck comparison envelope across the tested real-map products and controls, while generic null envelopes fail much more strongly under the same map-space procedure. The detailed RMS tables, control ladders, and null-envelope audits are contained in the dedicated CMB comparison-map closure material.

#### Original Planck and WMAP FITS Inputs

The original Planck and WMAP survey FITS maps are **not bundled** because they are large public data products. The strict input set used by the [CPTG CMB Comparison-Map Closure](https://doi.org/10.5281/zenodo.22413491) test is listed below so the public rerun environment can be reconstructed directly. Use only these tested FITS products for the strict comparison-map closure rerun; do not add optional masks, alternate survey products, or substitute component maps.

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

A separate CMB comparison-coordinate report studies the [Route B Option 1](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/CPTG_Route_B_Option1_Curvature_Transport_Geometry_Bridge_Report_v1.pdf) curvature-transport bridge at the `C_l` and Planck likelihood-coordinate level. This work is distinct from the real-map comparison-map closure paper above.

The current bridge prescription applies the CPTG curvature-transport response at amplitude/potential level:

```text
Phi_pi(a,k) = Phi_0(k) C_T(a,k)
```

At power-spectrum level the corresponding mapping is written schematically as:

```text
P(k) -> P(k) |C_T(a,k)|^2
```

Route B should presently be read as a geometry-first comparison-coordinate construction and plumbing test. It is not used here as an independent CMB closure result or as a replacement for the locked real-map comparison-map test.

### Pantheon+ Supernova Distance-Shape Test

CPTG has been tested against Pantheon+ supernova distance-shape data using a full-covariance comparison with a marginalized intercept. This is a distance-shape test, not a local H0 calibration claim. The purpose is to ask whether the CPTG expansion branch can reproduce the relative supernova distance trend once the absolute calibration is marginalized.

### BBN Abundance and Lithium Source-Network Tests

CPTG abundance work is organized around the transported BBN coordinate and the locked live A = 7 source-network gate. The transported abundance coordinate is

```text
eta10_BBN = 5.998071834744
Omega_b h²_BBN = 0.021898765370
```

The [lithium problem](https://doi.org/10.5281/zenodo.22420032) is treated as a surviving mass-seven abundance problem because most final primordial lithium is carried through `7Be` during BBN and later appears as `7Li`. The locked CPTG gate is

```text
Y7,CPTG = Y7,raw / pi
∫ Gamma7 dt = ln(pi)
```

Operationally, the gate is applied to the live `7Li` and `7Be` channels inside the source network. The network is then re-evolved so the final abundance is produced dynamically rather than by a post-processing label.

The deuterium-proton radiative-capture extension supplies the native `D(p,γ)³He` reaction branch underneath the abundance calculation. The earlier admitted PRyMordial commissioning row gives

```text
raw Li7/H = 5.2668261732457650e-10
gated Li7/H = 1.6764828397556692e-10
Li7 pull after gate = +0.90593 sigma
D/H and Yp controls: PASS
```

That result remains the published commissioning-stage abundance demonstration. The updated computational companion adds a separate prospective PRIMAT-native implementation of the locked mass-seven transport law. All **7/7 paired coordinates pass**; the worst relative survival disagreement from the locked target is **0.018083%**, the independent-grid optical-depth reconstruction passes, and the largest non-mass-seven disturbances remain negligible under the declared controls. Observational lithium is recorded only as a downstream diagnostic and is not used as an acceptance gate.

Together, the two layers preserve a clear chronology: the commissioning paper first demonstrated the source-network abundance consequence, while the updated computational companion tests the locked mass-seven transport directly inside the current primary PRIMAT authority. Within the completed universal theory, the broader hydrogen-deuterium-helium-3-helium-4 architecture supplies the governing reaction context without altering the accepted commissioning result.


### Weak-Lensing S8 Comparison

CPTG weak-lensing work currently uses compressed S8 comparisons against representative weak-lensing and CMB anchors. These tests are diagnostic: they show whether the CPTG growth/lensing branch lies within representative observational bands, but they are not a substitute for a full shear-correlation likelihood or survey-level weak-lensing pipeline.

### DESI DR1 Compressed ShapeFit and BAO Quarter-Ruler

CPTG large-scale-structure work separates DESI comparisons into layers. The current compressed ShapeFit coordinate comparison uses official DESI DR1 HDF5 likelihood containers and is a compressed-coordinate pass. The BAO quarter-ruler is a strong coordinate-wrapper diagnostic using the CPTG transport relation `G_T^(-1/4)`, but it is not presented as a raw official non-unity-`q` runtime likelihood response.

The full-shape AP/growth work remains an exploratory spectrum-shell diagnostic. It is not a raw DESI full-shape validation claim and should not be described as one. A full raw DESI validation requires nuisance-preserving AP, RSD, tracer-window, covariance, nuisance, counterterm, and stochastic machinery to be wired consistently through the official likelihood path.

### Cosmological Horizon Mechanism

A current CPTG horizon-mechanism article treats the [cosmological horizon problem](https://doi.org/10.5281/zenodo.22430125) as a structural-curvature synchronization problem rather than as a scalar-field inflation mechanism. In this framing, early-universe uniformity is attributed to finite curvature saturation and active geometric transport synchronizing the primordial curvature state before decoupling.

### Hubble-Tension Bridge

The CPTG [Hubble-tension bridge](https://doi.org/10.5281/zenodo.22429502) treats the CMB/acoustic and local luminosity determinations as different observational projections of one native CPTG geometric branch, with each projection carrying its own rate history at the same selected native phase.

The locked working bridge is:

```text
67.48 <- 69.4163 -> 73.04
```

The acoustic/CMB projection lies below the native branch while the local luminosity-distance projection lies above it; both are comparison-layer coordinates of the same native geometric history.

---

## Relation to MOND and ΛCDM

The repository compares CPTG to MOND-style galaxy predictions and to the broader dark-matter-halo interpretation associated with ΛCDM, but these comparisons are not identical in type.

- **MOND** modifies the acceleration law and performs well when galaxy behavior follows a nearly universal low-acceleration relation.
- **ΛCDM** explains galaxy and cluster dynamics through non-baryonic dark matter, with individual galaxy rotation curves often modeled through halo fitting and related nuisance parameters.
- **CPTG** tests whether similar observed effects can emerge from baryon-sourced curvature polarization, curvature transport, and theory-derived structural organization.

The CPTG SPARC tools evaluate CPTG directly against observed galaxy rotation data and include a MOND-style comparison under the same loaded galaxy database. The Upsilon benchmark adds stellar mass-to-light freedom as a stricter comparison layer. These tests are included so the comparison can be reproduced rather than treated as a qualitative claim.

Compared with ΛCDM halo fitting, CPTG makes a different kind of test: it asks whether galaxy rotation behavior and dissociative cluster-lensing offsets can be reproduced geometrically through baryon-sourced curvature response rather than by fitting non-baryonic halo components.

---

## Recent Progress and Active Development

Recent public milestones include:

- the SPARC reduced-limit benchmarks, CPTG SPARC Browser Workbench v1.12.0, and the published [galaxy rotation-curve](https://doi.org/10.5281/zenodo.22418189) and [Structural Mode \(N\)](https://doi.org/10.5281/zenodo.22436682) papers;
- the reduced Bullet Cluster merger-plane benchmark and the published [finite compact-object / black-hole framework](https://doi.org/10.5281/zenodo.22424445);
- the locked geometric-π [CMB comparison-map closure](https://doi.org/10.5281/zenodo.22413491), the [geometric π comparison-coordinate paper](https://doi.org/10.5281/zenodo.22426857), and the [horizon](https://doi.org/10.5281/zenodo.22430125) and [Hubble-tension](https://doi.org/10.5281/zenodo.22429502) papers;
- completion of the three-network nuclear authority hierarchy: **PRIMAT primary — CLOSED/PASS; PArthENoPE second — CLOSED/PASS; PRyMordial third — CLOSED/PASS**;
- publication of the [universal nuclear-reaction parent paper](https://doi.org/10.5281/zenodo.22439971) and its [PRIMAT-anchored computational companion](https://doi.org/10.5281/zenodo.22442178);
- continued development of the CPTG Reaction Workbench, Pi-Bridge, reproducibility packages, and coordinated publication updates.

The publication program is also undergoing a coordinated update so revised papers, newly completed papers, evidence links, and cross-references remain consistent across the full CPTG record.

The fixed CPTG geometry and immutable nuclear coefficient package remain unchanged across the accepted authority hierarchy.

---

## Repository Policies

- [Security policy](SECURITY.md)
- [Citation information](#citation)

---

## Citation

If referencing the CPTG framework, please cite:

Carter L. Glass Jr., *Curvature Polarization Transport Gravity: A Unified Geometric Framework for Structure, Cosmology, and Nuclear Reactions*, [DOI: 10.5281/zenodo.22441039](https://doi.org/10.5281/zenodo.22441039).

For the repository and supporting code package, cite:

CPTG, *Supporting Python Models, Benchmark Implementations, and Research References for Curvature Polarization Transport Gravity*, companion resource, available at [https://github.com/CLG2025/CPTG](https://github.com/CLG2025/CPTG).

---
