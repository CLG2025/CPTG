# CPTG Nuclear Reactions

This directory contains the research papers, source code, validation tools, execution protocols, and evidence packages for the **universal nuclear-reaction extension of Curvature Polarization Transport Gravity (CPTG)**.

The work collected here generalizes the original deuterium–proton capture result into a closed geometric theory for the four light-element states:

- **Hydrogen** — vertex
- **Deuterium** — bridge
- **Helium-3** — closure
- **Helium-4** — saturation

The final theory treats these states as one ordered geometric reaction system rather than as unrelated reaction channels.

---

## Overview

The original CPTG nuclear-reaction program was commissioned through the reaction

> deuterium + proton → helium-3 + photon

and its primordial abundance consequences.

That single-channel result established that a nuclear-reaction calculation could be organized from a geometric source state, through a coherent reaction response, into a network-level abundance prediction.

The extension documented in this directory completes that construction.

The final theory identifies a universal reaction architecture in which:

1. free nucleons define the initial geometric vertex;
2. deuterium forms the first bound bridge;
3. the mass-three system supplies the closure state;
4. helium-4 forms the saturated endpoint;
5. transport between these states is governed by an ordered geometric current;
6. internal proton–neutron and tritium–helium-3 orientation is retained through a polarization degree of freedom;
7. baryon number and charge remain exact constraints of the full reaction operator;
8. the reaction network is represented by a universal transport-polarization response rather than a collection of unrelated empirical fits.

The theory is therefore closed at the level of state definition, reaction topology, conservation structure, source decomposition, curvature response, and network transport.

---

## Final Theory

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

The completed theory separates the nuclear network into two coupled structures.

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

The reaction-source basis spans the complete transport-polarization space. Independent reaction sets can therefore be used for theory construction and held-out validation without changing the underlying geometry.

---

## Curvature Response

CPTG does not model nuclear reactions as isolated rate adjustments.

The theory assigns a geometric response to the reaction current and carries that response through the full network.

The completed curvature structure determines:

- the first-order reaction response;
- the second-order curvature response;
- the scale dependence across the primordial baryon-density domain;
- the relation between direct reaction current and final abundance displacement;
- the universal network susceptibility connecting source reactions to observable outcomes.

The scalar commissioning result is retained as a projection of this larger transport-polarization operator.

The universal theory therefore explains why the original deuterium–proton channel succeeded while also showing that the single-channel result was only one observable projection of a broader geometric law.

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

> *Geometric Nuclear Reaction Theory in CPTG: Deuterium-Proton Capture and Primordial Mass-Seven Transport*

should be understood as the commissioning-stage demonstration of the nuclear-reaction program.

It established:

- a native geometric source-state construction;
- a coherent reaction-amplitude framework;
- a reaction-rate connection;
- a primordial abundance propagation layer;
- the mass-seven transport gate.

The work in this directory generalizes that result from one principal capture channel to the complete hydrogen–deuterium–helium-3–helium-4 reaction architecture.

The original paper remains valid within its stated scope. The universal theory presented here contains it as a special projection.

---

## Validation Program

The theory is supported by several independent validation layers.

### Native reaction-network validation

The native network program tests the universal transport-polarization response across:

- multiple reaction channels;
- independent discovery and held-out reaction bases;
- the full light-element state vector;
- multiple baryon-density rows;
- symmetric high-precision perturbation scales;
- first- and second-order response extraction;
- exact integrated reaction-current measurements;
- fail-closed candidate freezing before held-out exposure.

### Numerical rigidity and stress testing

The external stress-test suite evaluates the mathematical and software implementation under extreme conditions, including:

- large randomized state ensembles;
- matrix-exponential propagation;
- underflow and subnormal inputs;
- nonfinite values;
- malformed data;
- simulated memory corruption;
- conservation-preserving adversarial perturbations;
- deterministic replay;
- invariant monitoring;
- fail-closed shutdown behavior.

The stress suite tests whether the implementation remains rigid, conservative, and safe under numerical and adversarial pressure.

### Evidence-chain audit

Every major package is designed to preserve:

- exact source hashes;
- exact commands;
- runtime logs;
- execution ledgers;
- checkpoint state;
- candidate-freeze records;
- output manifests;
- raw endpoint data;
- independent reconstruction scripts;
- clean-extraction replay.

The goal is not merely to report a result, but to preserve the complete chain needed to reproduce and audit it.

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

Formal derivations, normalization conventions, curvature structure, source-current definitions, and validation details are maintained in the versioned research papers and controlled technical packages stored in this directory.

This keeps the repository overview readable while preserving the complete scientific record in its proper context.

---

## Status

The universal CPTG nuclear-reaction framework is presented as complete.

The final theory includes:

- the four-state light-element geometry;
- the dynamic free-nucleon vertex;
- the deuterium bridge;
- the mass-three closure sector;
- the helium-4 saturation state;
- the ordered transport operator;
- the charge-constrained polarization mode;
- the complete reaction-source basis;
- the curvature response;
- the network susceptibility;
- the native held-out validation architecture;
- the numerical rigidity and fail-closed safety framework.

The remaining work in this directory is implementation, publication, replication, and continued independent testing—not alteration of the final theoretical structure.

---

## Citation

**Curvature Polarization Transport Gravity**

Repository: [CLG2025/CPTG](https://github.com/CLG2025/CPTG)

Author: **Carter L. Glass Jr.**
