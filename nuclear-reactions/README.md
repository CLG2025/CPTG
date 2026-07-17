# CPTG Nuclear Reactions

This directory contains the research papers, source code, validation tools, execution protocols, and evidence packages for the **universal nuclear-reaction extension of Curvature Polarization Transport Gravity (CPTG)**.

The work collected here generalizes the original deuterium–proton capture result into a **closed four-state geometric theory** for the light-element states:

- **Hydrogen** — vertex
- **Deuterium** — bridge
- **Helium-3** — closure
- **Helium-4** — saturation

The completed theory treats these states as one ordered geometric reaction system rather than as unrelated reaction channels.

---

## Overview

The original CPTG nuclear-reaction program was commissioned through the reaction

> deuterium + proton → helium-3 + photon

and its primordial abundance consequences.

That single-channel result established that a nuclear-reaction calculation could be organized from a geometric source state, through a coherent reaction response, into a network-level abundance prediction.

The extension documented in this directory completes that construction.

The universal architecture identifies a reaction system in which:

1. free nucleons define the initial geometric vertex;
2. deuterium forms the first bound bridge;
3. the mass-three system supplies the closure state;
4. helium-4 forms the saturated endpoint;
5. transport between these states is governed by an ordered geometric current;
6. internal proton–neutron and tritium–helium-3 orientation is retained through a polarization degree of freedom;
7. baryon number and charge remain exact constraints of the full reaction operator;
8. the reaction network is represented by a universal transport-polarization response rather than a collection of unrelated empirical fits.

The theory is closed at the level of state definition, reaction topology, conservation structure, source decomposition, curvature response, and universal network transport within the tested light-element domain.

---

## Final Theory Architecture

The universal CPTG nuclear-reaction theory is built from four physical roles.

### 1. Vertex

The vertex is the unbound nucleon state.

At the dynamic level, the vertex contains both free neutrons and free protons. At late endpoints, the free-neutron contribution becomes negligible and the vertex reduces to ordinary hydrogen.

The vertex is the initial topological degree of freedom from which the light-element reaction ladder develops.

### 2. Bridge

Deuterium is the first stable bound bridge.

It connects the free-nucleon state to the mass-three closure sector and carries the most direct reaction sensitivity in the light-element network. Its role is transitional rather than terminal.

### 3. Closure

The mass-three system is the closure state.

Tritium and helium-3 form two internal orientations of the same mass-three sector. Their distinction is preserved through the polarization coordinate, while their common baryonic role is represented by the closure state.

### 4. Saturation

Helium-4 is the saturated endpoint.

It is the maximally closed light-element state in the four-state system and serves as the terminal reservoir for the dominant reaction flow.

---

## Universal Reaction Architecture

The completed architecture separates the nuclear network into two coupled structures.

### Ordered transport

Reaction flow moves through three geometric transitions:

- vertex → bridge
- bridge → closure
- closure → saturation

These transitions define the universal transport backbone of the light-element network.

The transport operator is fixed by the topology of the reaction chain. It is not fitted separately for each reaction.

### Internal polarization

The transport states alone do not distinguish:

- neutron from proton inside the vertex sector;
- tritium from helium-3 inside the closure sector.

CPTG therefore includes one independent polarization direction that preserves the internal charge orientation of the system while remaining consistent with exact baryon and charge conservation.

The full reaction source space contains:

- three transport directions;
- one polarization direction.

This four-dimensional source structure is the universal input space of the theory.

---

## Conservation and Closure

The final operator is constructed so that the light-element system remains inside a closed baryonic state space.

The theory preserves:

- total baryon number;
- total electric charge;
- the ordered four-state topology;
- nonnegative physical abundances;
- the internal polarization constraint;
- the distinction between source current and final network response.

The reaction-source basis spans the complete transport-polarization space. Independent reaction sets were used for construction and held-out validation without changing the underlying geometry or introducing reaction-specific corrections.

---

## Curvature Response

CPTG does not model nuclear reactions as isolated rate adjustments.

The theory assigns a geometric response to the reaction current and carries that response through the full network.

The completed curvature structure determines:

- the first-order reaction response;
- the second-order curvature response;
- the scale dependence across the tested primordial baryon-density domain;
- the relation between direct reaction current and final abundance displacement;
- the universal network susceptibility connecting the complete reaction-source basis to the full light-element state vector.

The original scalar commissioning result is retained as one projection of the larger transport-polarization operator.

The completed universal theory explains why the original deuterium–proton channel succeeded while showing that the single-channel result was one observable projection of a broader geometric law.

---

## Physical Interpretation

The four light elements form a complete geometric sequence:

> unbound vertex → first bound bridge → mass-three closure → helium-4 saturation

This sequence is not imposed as a descriptive analogy. It is the organizing structure of the reaction operator itself.

Each reaction contributes a specific combination of:

- forward transport;
- reverse transport;
- closure transfer;
- saturation transfer;
- polarization rotation.

The final abundance response is produced by the network acting on that source geometry.

This distinction is essential:

- the **reaction source current** describes the direct microscopic drive;
- the **network susceptibility** describes how the full system redistributes that drive;
- the **observable abundance response** is the final projected result.

CPTG closes these three layers within one geometric framework.

---

## Relationship to the Original Paper

The earlier article,

> *[Geometric Nuclear Reaction Theory in CPTG: Deuterium-Proton Capture and Primordial Mass-Seven Transport](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/CPTG_Geometric_Nuclear_Reaction_Theory.pdf)*

should be understood as the commissioning-stage demonstration of the nuclear-reaction program.

It established:

- a native geometric source-state construction;
- a coherent reaction-amplitude framework;
- a reaction-rate connection;
- a primordial abundance propagation layer;
- the mass-seven transport gate.

The work in this directory generalizes that result from one principal capture channel to the complete hydrogen–deuterium–helium-3–helium-4 reaction architecture.

The original paper remains valid within its stated scope. The universal theory presented here contains it as a special scalar projection.

---

## Validation Program

The CPTG nuclear-reaction framework is evaluated through several independent validation layers designed to separate mathematical closure, native execution, observational comparison, cross-network portability, and held-out testing.

### Native reaction-network validation

The frozen reaction construction has been propagated through native reaction-network calculations spanning:

- multiple light-element reaction channels;
- independent construction and held-out reaction sets;
- multiple primordial baryon-density coordinates;
- symmetric rate perturbations;
- exact reaction-current instrumentation;
- first- and second-order response extraction;
- full abundance-vector comparison;
- durable checkpointing and recovery;
- candidate freezing before held-out exposure.

The commissioned deuterium–proton capture branch has completed native network execution and a no-refit observational comparison. The corrected frozen candidate improves the tested abundance vector primarily through its deuterium response while leaving helium effectively unchanged.

### Independent network replication

The commissioned reaction response has also been tested in an independent BBN network implementation.

The second network reproduces the same qualitative abundance-response structure:

- deuterium increases;
- helium-3 decreases;
- lithium-7 decreases;
- helium-4 remains minimally affected.

The response magnitude differs between network implementations, but the shared direction demonstrates that the result is not specific to a single native network code.

### Numerical rigidity and stress testing

The mathematical and software implementation has been tested under large-scale numerical and adversarial conditions, including:

- randomized valid-state ensembles;
- injected malformed and nonphysical inputs;
- conservation-preserving perturbations;
- boundary and near-singular cases;
- nonfinite and underflow conditions;
- deterministic replay;
- invariant monitoring;
- simulated evidence corruption;
- fail-closed interruption and recovery.

These tests found no unresolved violation of the frozen mathematical contracts, conservation laws, or numerical invariants.

### Full-resolution qualification

The broader universal qualification campaign remains organized around independent discovery, held-out, mirror, and mixed-direction reaction sets.

This program evaluates whether the frozen transport-polarization susceptibility remains valid across the complete tested reaction basis without:

- post-result fitting;
- reaction-specific correction factors;
- held-out exclusions;
- changes to the frozen four-state geometry.

Long-running native calculations preserve committed rows through durable checkpoint and recovery records so that completed evidence is not lost or silently regenerated.

### Independent reconstruction and evidence audit

The validation program preserves:

- exact source and candidate hashes;
- command records;
- native execution logs;
- row-level audits;
- checkpoint and recovery state;
- candidate-freeze records;
- output manifests;
- raw abundance and current data;
- independent reconstruction scripts;
- clean-extraction replay evidence.

The purpose of this structure is not only to report favorable results, but to preserve the evidence required to reproduce, challenge, or falsify them.

---

## Scope of Closure

The universal CPTG nuclear-reaction formulas are closed for the tested hydrogen–deuterium–helium-3–helium-4 architecture, reaction basis, baryon-density domain, perturbation domain, and native network environment.

Within that scope:

- the four-state geometry is fixed;
- the source vectors are fixed;
- the transport and polarization structure is fixed;
- the first- and second-order curvature response is fixed;
- the universal network susceptibility is fixed;
- no reaction-specific fitting is required;
- no held-out result was used to modify the frozen candidate.

Formula closure does not by itself claim:

- completed absolute-rate derivations for every nuclear reaction;
- validation for nuclei beyond the stated light-element architecture;
- laboratory confirmation in every plasma regime;
- certification for safety-critical reactor control;
- replacement of independent experimental or network replication.

Those are downstream validation and application domains, not open terms in the closed formula structure.

---

## Experimental Nuclear Reaction Chain Extension Through (A=119)

### A Post hoc Exploratory Probe of the CPTG Universal Nuclear Reaction Program

This experimental extension is an exploratory probe of the CPTG universal nuclear reaction program. The primary research effort will remain focused on the core criteria established in the original paper: the fixed geometric source construction, coherent reaction amplitude, reaction-rate behavior, network transport, and abundance closure. The higher-mass study tests how far that same geometric framework can be continued without replacing it with independently fitted formulas for successive reaction stages. Current testing shows that the reaction chain can be propagated continuously through every mass number from (A=1) to (A=119).

The [resulting register](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/Complete-Processed-Nuclear-Chain.pdf) contains three levels of computational support. Masses (A=1) through (A=30) belong to the native reaction-network inventory, extending from the light-element foundation through oxygen and the post-oxygen network to silicon-30. Masses (A=31) and (A=32) form a strong prescribed-trajectory diagnostic continuation beyond the native inventory. Masses (A=33) through (A=119) belong to the larger exploratory prescribed-trajectory neutron-capture and beta-decay graph. Together, these categories provide a gap-free higher-mass test of the fixed CPTG geometric framework while the central program remains directed toward completing and validating the original universal-reaction criteria.

---

## Repository Structure

The directory is organized around **immutable, versioned evidence packages** rather than attempting to separate every script, protocol, dataset, manifest, and report into independent folders.

Many CPTG packages are designed as complete audit objects. A single package may contain its executable source, validation protocol, raw or derived evidence, logs, manifests, reports, checksums, and upload markers. When those components are bound together by package-level hashes, the package must remain intact. Splitting or reorganizing its contents would break the preserved evidence chain.

```text
/nuclear-reactions/
├── README.md
├── papers/
│   ├── universal-theory/
│   └── commissioning-paper/
├── packages/
│   ├── theory-development/
│   ├── native-validation/
│   ├── stress-testing/
│   ├── protocol-frameworks/
│   └── audits-and-handoffs/
├── package-index/
│   ├── PACKAGE_LEDGER.md
│   ├── EVIDENCE_LEDGER.md
│   ├── SHA256SUMS.txt
│   └── RELEASE_NOTES.md
└── releases/
```

Each item under `packages/` should retain its original versioned filename and internal directory structure. Packages may be stored as ZIP archives, extracted directories, or both, but the hash-authoritative archive must not be altered.

Where a package contains source code, protocols, data, evidence, logs, manifests, and reports together, that package is the authoritative unit of publication. Separate copies may be published for readability, but they must be labeled as convenience copies and must not replace the original hashed package.

The `package-index/` directory records what each package contains, its scientific role, version, status, SHA-256 digest, and relationship to earlier or later packages. This provides repository navigation without dismantling the evidence objects themselves.

---

## Reproducibility Policy

All executable packages should provide:

- a single Windows entry point;
- exact Command Prompt instructions;
- explicit dependency paths;
- resumable execution;
- progress records;
- fail-closed error handling;
- an output ZIP;
- an `UPLOAD_THIS_FILE.txt` marker;
- a SHA-256 manifest;
- clean-extraction self-validation.

Long-running native calculations must preserve completed rows and resume only from missing or invalid entries.

Held-out results must never be used to alter a frozen candidate.

Any future extension must begin as a new validation domain. It must not silently refit or rewrite the consumed evidence supporting the closed light-element theory.

---

## Security and Safety Policy

The nuclear-reaction software in this directory is research software.

It is designed to fail closed when:

- required source authority is missing;
- package hashes do not match;
- an endpoint record is malformed;
- a conserved quantity exceeds tolerance;
- a state becomes nonphysical;
- a numerical result is nonfinite;
- a held-out boundary is violated;
- a candidate is modified after freezing;
- an execution row is incomplete or internally inconsistent.

The fail-closed architecture is intended to prevent invalid numerical states from propagating into downstream control, comparison, or interpretation layers.

This repository does not claim that research software alone satisfies the certification requirements of a reactor-control, medical, aerospace, or other safety-critical deployment environment.

---

## Formula Disclosure

The public README intentionally does not reproduce the closed-form equations.

Formal derivations, normalization conventions, curvature structure, source-current definitions, and validation details are preserved in controlled research papers and hash-bound technical packages.

Public release of the complete formulas and final paper is intentionally sequenced after appropriate intellectual-property protection. This preserves the scientific record while allowing the formulas to be disclosed in full once that protection is secured.

---

## Status

The CPTG nuclear-reaction framework is **formula-closed within its tested light-element scope**.

The completed theoretical structure includes:

- the four-state light-element geometry;
- the dynamic free-nucleon vertex;
- the deuterium bridge;
- the mass-three closure sector;
- the helium-4 saturation state;
- the ordered transport operator;
- the charge-constrained polarization mode;
- the complete reaction-source basis;
- the first- and second-order curvature response;
- the universal network-susceptibility construction;
- the native construction and held-out validation architecture;
- the numerical-rigidity and fail-closed evidence framework.

The commissioned deuterium–proton capture branch has additionally passed:

- native reaction-network execution;
- frozen baseline-versus-candidate comparison;
- no-refit observational abundance comparison;
- independent second-network directional replication;
- exact rate-transfer and evidence-chain auditing.

The active qualification program continues to test the frozen universal response across independent discovery and held-out reaction sets at full numerical resolution.

Subsequent work consists of:

- completing the active held-out qualification campaign;
- independent experimental comparison;
- broader-domain falsification;
- additional reaction-channel development;
- software implementation and performance qualification;
- intellectual-property protection;
- formal publication.

New evidence may confirm, constrain, or falsify the frozen theory, but consumed validation data must not be reused to refit the formula set after the fact.

---

## Citation

**Curvature Polarization Transport Gravity**

Repository: https://github.com/CLG2025/CPTG

Author: **Carter L. Glass Jr.**
