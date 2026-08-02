# CPTG Geometric Nuclear-Reaction Theory

This directory contains the papers, source code, validation tools, execution protocols, and evidence packages for the universal, fixed, scalable nuclear-reaction extension of **Curvature Polarization Transport Gravity (CPTG)**.

The validated foundation consists of four coupled nuclear sectors:

- **Free nucleons (`n`, `p`) — vertex**
- **Deuterium — bridge**
- **Mass three (`³H`, `³He`) — closure**
- **Helium-4 (`⁴He`) — saturation**

Together, these sectors form one transport-polarization architecture rather than a collection of unrelated reaction models. They define the validated foundation of the universal theory without imposing helium-4 as an absolute upper mass limit.

**Fixed-law scalability** means that the same CPTG geometric law, source-coordinate construction, conservation structure, and baryon-density dependence are carried across reaction channels and independently developed network implementations without reaction-specific or code-specific geometric refitting. Reaction stoichiometry determines the source direction; it does not replace the underlying geometry.

> **Formula-protection notice**
>
> This public README intentionally omits the protected universal nuclear-reaction equations, source matrices, susceptibility coefficients, normalization identities, and reconstructive formula-package details. Public computational companions and evidence packages may disclose reduced-graph methods, numerical recurrences, verification scripts, and archived outputs when those materials do not reconstruct the protected universal formula authority.

---

## Validation Status

| Component | Status |
|---|---|
| Four-sector nuclear geometry | Closed |
| Baryon and electric-charge constraints | Exact |
| Reaction-source space | Rank four and complete |
| Scalar curvature-response law | Passed |
| Universal network susceptibility | Fixed; full-resolution native authority completed in AlterBBN |
| Independent zero-refit validation basis | Passed without refitting |
| Mirror-polarization and mixed-direction diagnostics | Passed |
| Commissioning `D(p,γ)³He` projection | Passed within its declared scope |
| Cross-network Reaction-20 transfer | Confirmed in AlterBBN, PRyMordial, and PArthENoPE without network-specific refitting |
| Full-network PArthENoPE endpoint validation | 695/695 rows, 338/338 matched pairs, and 84/84 eight-branch ladders passed hard-integrity checks |
| Numerical-rigidity and fail-closed qualification | Passed |
| Post-silicon continuation through `A = 119` | Published computational companion; reduced-graph reachability qualified by boundary, convergence, and source controls |

The native authority campaign used independent construction, zero-refit validation, mirror-polarization, and mixed-direction reaction sets under a fixed geometry. The susceptibility authority was determined from the construction basis, fixed before independent validation, and retained without post-result fitting or reaction-specific correction.

The separate PArthENoPE Reaction-20 campaign is a **preregistered zero-refit cross-network validation of the fixed Reaction-20 law**. It confirms the declared shared-endpoint response in a third independently developed BBN implementation while preserving code-local boundaries for currents, source kernels, solver internals, and normalization conventions.

---

## Universal Four-Sector Architecture

### Vertex

The vertex is the unbound nucleon sector. It contains free neutrons and free protons at the dynamic level. At late network endpoints, the free-neutron contribution becomes negligible and the surviving vertex content is dominated by ordinary hydrogen.

### Bridge

Deuterium is the first stable bound bridge. It connects the free-nucleon vertex to the mass-three closure sector and carries the most direct transition sensitivity in the primordial network.

### Closure

Tritium and helium-3 form two internal charge orientations of the mass-three closure sector. Their distinction is preserved through the polarization coordinate while their shared transport role is represented by the closure state.

### Saturation

Helium-4 is the saturation sector of the validated four-sector coordinate and the principal terminal reservoir for the dominant primordial reaction flow. This role does not imply that nuclear organization universally ends at `A = 4`.

### Transport and polarization

Reaction flow follows three ordered transitions:

- vertex → bridge;
- bridge → closure;
- closure → saturation.

These transitions define the transport backbone of the validated network. One independent polarization direction preserves the neutron-proton and tritium-helium-3 charge orientation while maintaining exact baryon and electric-charge conservation.

The complete reaction-source space therefore contains:

- three transport directions;
- one polarization direction.

This rank-four structure is the universal source space of the four-sector theory.

### Source current, susceptibility, and observable response

The theory distinguishes three levels:

- **reaction source current** — the direct microscopic drive;
- **network susceptibility** — redistribution of that drive by the full reaction system;
- **observable abundance response** — the final projected network result.

The full-resolution native qualification established one fixed source-current-normalized susceptibility over the declared reaction-source basis without reaction-specific refitting. The earlier scalar commissioning result is retained as one projection of this larger transport-polarization operator.

---

## Conservation and Closure

The validated operator preserves:

- total baryon number;
- total electric charge;
- the ordered four-sector topology;
- nonnegative physical abundances;
- the internal polarization constraint;
- the distinction between source current and final network response.

Independent construction and zero-refit validation reaction sets were used to determine and evaluate the universal susceptibility without changing the underlying geometry or introducing reaction-specific corrections.

---

## Relationship to the Commissioning Paper

The earlier article,

> *[Geometric Nuclear Reaction Theory in CPTG: Deuterium-Proton Capture and Primordial Mass-Seven Transport](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/CPTG_Geometric_Nuclear_Reaction_Theory.pdf)*

is the commissioning-stage demonstration of the nuclear-reaction program. It established:

- a native geometric source-state construction;
- a coherent reaction-amplitude framework;
- a reaction-rate connection;
- a primordial abundance-propagation layer;
- the mass-seven transport gate.

The universal theory generalizes that result from one principal capture channel to the complete free-nucleon–deuterium–mass-three–helium-4 architecture. The commissioning paper remains valid within its declared scope.

---

## Validation Program

### Native AlterBBN authority

The fixed reaction construction was propagated through native reaction-network calculations spanning:

- multiple reaction channels;
- independent construction and zero-refit validation reaction bases;
- multiple primordial baryon-density coordinates;
- symmetric rate perturbations;
- exact reaction-current instrumentation;
- first- and second-order response extraction;
- full abundance-vector comparison;
- durable checkpointing and recovery;
- susceptibility fixing before independent validation.

The universal transport-polarization susceptibility authority was completed in the native AlterBBN environment under the fixed full-resolution design. The campaign preserved the construction/validation boundary and introduced no post-result fitting, reaction-specific correction factors, validation exclusions, or changes to the fixed four-sector geometry.

### Cross-network Reaction-20 transfer

The commissioned Reaction-20 response has been evaluated in three independently developed BBN implementations without network-specific refitting:

| Network | Established result | Authority boundary |
|---|---|---|
| **AlterBBN** | Native Reaction-20 response surface, endpoint propagation, code-local current instrumentation, and full-resolution universal-susceptibility authority | Source kernels, accepted-state current definitions, and current normalization remain AlterBBN-local |
| **PRyMordial** | Independent matched-uniform Reaction-20 endpoint and current-normalized response; the first-order species vector closely aligns with AlterBBN | PRyMordial current normalization remains code-local and is not asserted to be numerically identical to AlterBBN |
| **PArthENoPE 3.0** | Official full-network 695-row endpoint campaign; all hard row, pair, ladder, trajectory, and provenance gates passed; the fixed D/H, He-3/H, and Li-7/H gate passed across six density anchors | Zero-refit shared-endpoint validation of the fixed Reaction-20 law; no PArthENoPE internal-current, source-kernel, or five-coordinate susceptibility equivalence claim |

The three codes share portions of the underlying BBN physics and nuclear-rate literature. The result is therefore described as **cross-network transfer without network-specific refitting**, not as three statistically independent trials and not as proof that the codes use identical internal-current normalizations.

### Full-network PArthENoPE Reaction-20 validation

The PArthENoPE campaign used the official 26-nuclide, 100-reaction network; one native process per row; symmetric logarithmic rate branches; retained native trajectories; atomic row commits; matched-pair auditing; and complete eight-branch ladder auditing.

| Validation metric | Measured result | Fixed requirement |
|---|---:|---:|
| Native rows | 695/695 | 695/695 |
| Matched branch pairs | 338/338 | 338/338 |
| Complete eight-branch ladders | 84/84 | 84/84 |
| Reaction-20 direction cosine | 0.999979437501036 | ≥ 0.995 |
| D/H component ratio | 0.998516625 | 0.85–1.15 |
| He-3/H component ratio | 0.993911268 | 0.85–1.15 |
| Li-7/H component ratio | 1.008589280 | 0.85–1.15 |
| D/H six-anchor variation | 2.491% | ≤ 10% |
| He-3/H six-anchor variation | 0.809% | ≤ 10% |
| Li-7/H six-anchor variation | 1.156% | ≤ 10% |

The accepted compact authority bundle is:

```text
CPTG_v129_r109_PArthENoPE_CPTG_Reaction20_ReplicationAuthorityBundle_20260731_r02.zip
SHA-256: 3d42d1cb1d710248841db8d7b1ceafcc7569f092b9b35c9f4114ce9615074cfc
```

It contains the clean campaign source, all 695 accepted native evidence rows, 338 matched-pair audits, 84 ladder audits, recomputation source code, provenance records, six PArthENoPE anchor vectors, threshold calculations, a claim matrix, and one-command audit regeneration. The official PArthENoPE distribution is obtained separately from its published program archive.

After extracting the bundle, regenerate the audit with:

```cmd
RUN_VERIFY_AND_REGENERATE_WINDOWS.cmd
```

This verifies the retained evidence and reconstructs the post-execution audit without rerunning all 695 native rows.

The result establishes the preregistered shared-endpoint response in D/H, He-3/H, and Li-7/H. It does **not** establish equality of code-local currents, AlterBBN source-kernel identity inside PArthENoPE, a complete five-coordinate susceptibility reconstruction, or one universal low-order endpoint law across every reaction.

### Evidence integrity

The accepted evidence chain preserves:

- source, executable, ledger, susceptibility-authority, and package hashes;
- command records and native execution logs;
- endpoint, trajectory, solver, and row-integrity audits;
- matched-pair and complete-ladder audit records;
- checkpoint and recovery state;
- susceptibility-freeze and preregistration records;
- output manifests and append-only event ledgers;
- raw abundance and current data where exposed by the native code;
- independent reconstruction and recomputation scripts;
- clean-extraction replay evidence;
- claim matrices separating established, diagnostic, and untested statements.

The r109 authority bundle consolidates the accepted PArthENoPE campaign, native evidence, recomputation source, reference-vector provenance, software citations, and audit-regeneration entry point into one compact research object while excluding the separately distributed PArthENoPE program archive.

---

## Scope of Closure

The universal CPTG nuclear-reaction theory is closed and full-resolution qualified for the declared free-nucleon–deuterium–mass-three–helium-4 architecture, reaction basis, baryon-density domain, perturbation domain, and native network environment.

Within that scope:

- the four-sector geometry is fixed;
- the reaction-source vectors are fixed;
- the transport and polarization structure is fixed;
- the first- and second-order curvature responses are fixed;
- the full-precision universal network susceptibility is fixed;
- the full-precision susceptibility was fixed before independent validation;
- the independent zero-refit validation basis passed;
- mirror-polarization and mixed-direction diagnostics passed under the fixed susceptibility;
- no reaction-specific fitting or postdecision correction is required.

Formula closure does not by itself establish:

- completed absolute-rate derivations for every nuclear reaction;
- full validation for every nucleus beyond the declared four-sector domain;
- laboratory confirmation in every plasma regime;
- certification for safety-critical reactor control;
- replacement of independent experimental or network replication.

These are downstream validation and application domains, not open terms in the closed formula structure.

---

## Exploratory Nuclear-Chain Continuation Through `A = 119`

The broader program extends beyond the native silicon-30 frontier through a prescribed-trajectory neutron-capture and beta-minus graph. This calculation tests post-silicon reachability while remaining separate from the validated authority of the four-sector universal susceptibility.

The computational companion contains a gap-free register for every integer mass number from `A = 1` through `A = 119`. The `A = 5` row records the unbound helium-5 and lithium-5 resonance states and has no abundance coordinate. Every published post-silicon mass-sector sum from `A = 31` through `A = 119` is positive in the archived prescribed-trajectory result.

The companion paper,

> *[A Universal Geometric Theory of Nuclear Reactions in CPTG: Post-Silicon Reachability, Convergence, and the Continuous A=1–119 Mass-Sector Register](https://raw.githubusercontent.com/CLG2025/CPTG/main/nuclear-reactions/universal-theory/Complete-Processed-Nuclear-Chain.pdf)*

and `Complete-Processed-Nuclear-Chain_A1-A119_ComputationalCompanion_20260721_r02_PACKAGE.zip` preserve three levels of computational support:

- `A = 1–30`: native reaction-network inventory through silicon-30;
- `A = 31–32`: prescribed-trajectory diagnostic continuation beyond the native inventory;
- `A = 33–119`: exploratory prescribed-trajectory neutron-capture and beta-minus graph.

Moving the absorbing boundary from `A = 120` to `A = 140` and `A = 160` leaves the stored mass-sector sums through `A = 119` unchanged. Temporal refinement through 64 substeps preserves positive support, while source-isolation and source-cutoff controls distinguish stable graph reachability from source-sensitive tail magnitude.

This is a reproducible reduced-graph reachability result. It is not native-network coverage beyond silicon-30, a precision-qualified prediction of primordial heavy-element abundances, or a finite physical endpoint.

---

## Repository Structure and Evidence Policy

This directory is the publication and evidence home for the CPTG nuclear-reaction program. Public contents are released incrementally as manuscripts, interfaces, and immutable evidence packages complete their disclosure and validation requirements.

```text
/nuclear-reactions/
├── README.md
├── papers/
│   ├── universal-theory/
│   ├── computational-companion/
│   └── commissioning-paper/
├── packages/
│   ├── theory-development/
│   ├── native-validation/
│   ├── stress-testing/
│   ├── protocol-frameworks/
│   └── audits-and-handoffs/
├── package-index/
└── releases/
```

Many CPTG packages are complete audit objects containing source code, protocols, data, evidence, logs, manifests, reports, checksums, and upload markers. When these components are bound by package-level hashes, the package must remain intact.

Published packages should retain their original versioned filename and internal directory structure. Extracted or convenience copies may be supplied for readability, but they do not replace the hash-authoritative archive.

A package marked **accepted** in the package ledger is controlling scientific evidence unless a later accepted package explicitly supersedes it.

The `package-index/` directory should record each package's contents, scientific role, status, SHA-256 digest, external software requirements, and relationship to earlier or later packages.

The r109 PArthENoPE authority bundle intentionally does not redistribute the official PArthENoPE program ZIP. Researchers obtain that distribution separately and use the bundle to verify the official source-file identities before a full native replication.

---

## Reproducibility Policy

Executable packages should provide, as applicable:

- a single Windows entry point;
- exact Command Prompt instructions;
- explicit dependencies and external-source requirements;
- resumable execution;
- progress records;
- fail-closed error handling;
- an output archive when new execution is performed;
- an upload marker when returned evidence is required;
- a SHA-256 manifest;
- clean-extraction self-validation.

Long-running native calculations must preserve completed rows and resume only from missing or invalid entries.

Independent validation results must never be used to alter the fixed susceptibility authority. Cross-network validation results must retain their declared shared-observable scope and code-local boundaries; they must not be used to introduce network-specific geometric refitting.

Any future extension must begin as a new validation domain. It must not silently refit or rewrite the consumed evidence supporting the closed four-sector theory.

---

## Security, Safety, and Disclosure

The software in this directory is research software. It is designed to fail closed when required authority is missing, hashes do not match, evidence records are malformed, conserved quantities exceed tolerance, states become nonphysical, numerical results are nonfinite, fixed authority boundaries are violated, or execution records are incomplete.

This repository does not claim that research software alone satisfies the certification requirements of reactor-control, medical, aerospace, or other safety-critical deployment environments.

The public README does not reproduce the protected closed-form equations. Formal derivations, normalization conventions, curvature structure, source-current definitions, source matrices, susceptibility coefficients, and reconstructive formula-package details remain in controlled research papers and hash-bound technical packages.

Public computational companions may disclose reduced-graph equations, numerical recurrences, verification scripts, and archived outputs when those materials do not reconstruct the protected universal formula authority. Release of the complete universal formulas and final theory paper remains subject to explicit author authorization.

---

## Current Status

The CPTG geometric nuclear-reaction theory is **formula-closed and full-resolution qualified within its declared four-sector native-authority domain**.

The central structure includes:

- the free-neutron/proton vertex;
- the deuterium bridge;
- the tritium/helium-3 closure sector;
- the helium-4 saturation sector;
- the ordered transport operator;
- the charge-constrained polarization mode;
- the rank-four reaction-source basis;
- the first- and second-order curvature-response hierarchy;
- the fixed full-precision network susceptibility;
- independent construction and zero-refit validation reaction bases;
- no-refit native validation;
- mirror-polarization and mixed-direction diagnostics;
- numerical-rigidity, recovery, and fail-closed evidence controls.

The commissioned Reaction-20 projection additionally has native AlterBBN response and current evidence, matched-uniform PRyMordial transfer evidence, and an official full-network PArthENoPE endpoint validation comprising 695 accepted rows, 338 matched branch pairs, 84 complete ladders, and six-anchor D/H, He-3/H, and Li-7/H gate closure.

The PArthENoPE result is a **completed zero-refit cross-network validation of the fixed Reaction-20 shared-endpoint response**. It confirms the declared endpoint response but does not establish identical code-local currents, source kernels, solvers, or complete susceptibility coordinates across the three networks.

Subsequent work concerns publication, external replication, experimental comparison, broader-domain falsification, higher-mass extension, software deployment, performance qualification, and the intellectual-property decision governing public disclosure.

---

## Citation

**Curvature Polarization Transport Gravity**

Repository: https://github.com/CLG2025/CPTG

Author: **Carter L. Glass Jr.**

For the post-silicon computational continuation, cite:

Carter L. Glass Jr., *[A Universal Geometric Theory of Nuclear Reactions in CPTG: Post-Silicon Reachability, Convergence, and the Continuous A=1–119 Mass-Sector Register](https://raw.githubusercontent.com/CLG2025/CPTG/main/nuclear-reactions/universal-theory/Complete-Processed-Nuclear-Chain.pdf)*, computational companion, 2026.

---

## BBN software citations

- A. Arbey, J. Auffinger, K. P. Hickerson, and E. S. Jenssen, “AlterBBN v2: A public code for calculating Big-Bang nucleosynthesis constraints in alternative cosmologies,” *Computer Physics Communications* **248**, 106982 (2020), [doi:10.1016/j.cpc.2019.106982](https://doi.org/10.1016/j.cpc.2019.106982).
- A.-K. Burns, T. M. P. Tait, and M. Valli, “PRyMordial: the first three minutes, within and beyond the standard model,” *The European Physical Journal C* **84**, 86 (2024), [doi:10.1140/epjc/s10052-024-12442-0](https://doi.org/10.1140/epjc/s10052-024-12442-0).
- S. Gariazzo, P. F. de Salas, O. Pisanti, and R. Consiglio, “PArthENoPE revolutions,” *Computer Physics Communications* **271**, 108205 (2022), [doi:10.1016/j.cpc.2021.108205](https://doi.org/10.1016/j.cpc.2021.108205).
- Official PArthENoPE 3.0 program distribution: Mendeley Data, version 2, [doi:10.17632/wvgr7d8yt9.2](https://doi.org/10.17632/wvgr7d8yt9.2).
