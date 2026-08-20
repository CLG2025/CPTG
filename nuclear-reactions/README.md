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

## Publication Status

The universal parent manuscript, *[A Universal Geometric Theory of Nuclear Reactions in CPTG](https://raw.githubusercontent.com/CLG2025/CPTG/main/nuclear-reactions/universal-theory/CPTG_Universal_Geometric_Nuclear_Reaction_Theory.pdf)*, is in final publication preparation. It presents the full four-sector derivation, governing geometric reaction laws, physical interpretation, conservation structure, and accepted native-network validation chain.

The updated *[computational companion](https://raw.githubusercontent.com/CLG2025/CPTG/main/nuclear-reactions/universal-theory/Complete-Processed-Nuclear-Chain.pdf)* carries that fixed architecture into an explicit reproducible calculation. PRIMAT supplies native trajectory and endpoint authority through `A = 23`; a declared seed-free prescribed-bath reduced graph supplies the continuation through `A = 119` and separately qualifies the natural `A = 338` structural frontier and source-tail robustness.

---

## Validation Status

| Component | Current status |
|---|---|
| Four-sector nuclear geometry | **Closed** |
| Baryon and electric-charge constraints | **Exact** |
| Reaction-source space | **Rank four and complete** |
| Scalar curvature-response/source-current law | **Closed within the declared theory** |
| Network realization | **Network-native dynamic observable pushforward; no universal static endpoint matrix is assumed** |
| Primary numerical authority | **PRIMAT v0.3.2 — completed and closed within the validated primordial/full-network and first-order native source/Jacobian scope** |
| Second numerical authority | **PArthENoPE 3.0 — completed full-network endpoint authority plus accepted clean-room native-physics reconstruction** |
| Third numerical authority | **PRyMordial — Candidate C-R production solver qualified; fresh 3030-row full-network authority campaign active** |
| Commissioning `D(p,γ)³He` projection | **Passed within its declared scope** |
| Full-network PRIMAT endpoint campaign | **20,550/20,550 rows; 428/428 reactions; 10,272/10,272 matched pairs; 2,568/2,568 eight-branch ladders** |
| PRIMAT native source/Jacobian mechanism validation | **84/84 predictions frozen; 48/48 primary resolved tests PASS; worst endpoint-vector discrepancy 0.0367629%** |
| Full-network PArthENoPE endpoint validation | **695/695 rows; 338/338 matched pairs; 84/84 eight-branch ladders** |
| Clean-room PArthENoPE native-physics reconstruction | **6/6 held-out rows PASS across 3 density anchors under 2 numerical profiles; no inherited numerical responses and no refit** |
| Clean-room PRyMordial native-physics reconstruction | **6/6 held-out rows PASS across 3 density anchors under 2 numerical profiles; zero integrated perturbation rows and no refit** |
| PRyMordial Candidate C-R qualification | **30/30 high-risk rows PASS; 29 scaled-coordinate paths + 1 native-coordinate recovery; worst endpoint component-relative difference 5.8799×10⁻⁸** |
| PRyMordial fresh full-network authority campaign | **Active: 3030 rows, 63 reactions, 6 density anchors, 8 branches; no numerical rows imported from the quarantined earlier campaign** |
| PRIMAT-anchored computational companion through `A = 119` | **Current public result: PRIMAT-native authority through `A = 23`; complete seed-free prescribed-bath reduced-graph register through `A = 119`; all six anchors PASS** |
| Post-`A = 119` structural frontier | **Reachable through `A = 338`; eligible `A = 339` sector disconnected under the selected topology; accepted 512/1024/2048 two-scheme qualification PASS** |
| Source-tail robustness at `A = 338` | **18/18 retained-source cases satisfy applicable preregistered requirements; all 12 hard-gated rows PASS; maximum unrenormalized endpoint departure 2.36×10⁻¹³** |
| PRIMAT-native mass-seven live transport | **7/7 paired coordinates PASS; worst relative survival disagreement from the locked target 0.018083% with negligible non-mass-seven control shifts** |

The present authority hierarchy is intentionally capability-based:

1. **PRIMAT v0.3.2 — primary authority**
2. **PArthENoPE 3.0 — second authority**
3. **PRyMordial — third authority, full-network renewal active**

The networks are not forced into one reduced numerical framework. Their code-local currents, Jacobians, trajectories, source kernels, integration measures, normalization conventions, and solver internals may differ. The common object being tested is the fixed CPTG geometric source architecture and its declared observable consequences through each network's native dynamics.

The current interpretation is therefore:

> **CPTG source geometry → exact reaction-source construction → network-native dynamic propagation → observable endpoint response**

A static endpoint matrix is not treated as universal unless separately derived and validated for that network and observable object.

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

### Source current, native dynamics, and observable response

The theory distinguishes three levels:

- **reaction source current** — the direct microscopic drive selected by reaction stoichiometry and the fixed CPTG geometric law;
- **network-native dynamic propagation** — redistribution of that source through the native reaction system and numerical evolution of the selected BBN implementation;
- **observable abundance response** — the final projected network result.

The fixed CPTG source construction is shared. The network-native dynamic pushforward is not required to be numerically identical across independent codes.

---

## Conservation and Closure

The validated operator preserves:

- total baryon number;
- total electric charge;
- the ordered four-sector topology;
- nonnegative physical abundances;
- the internal polarization constraint;
- the distinction between source current and final network response.

Independent construction, held-out validation, full-network perturbation campaigns, and native source/Jacobian tests are used to evaluate the fixed theory without introducing reaction-specific geometric corrections after results are exposed.

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

### PRIMAT v0.3.2 — primary authority

PRIMAT is the primary numerical authority for the current primordial/light-sector CPTG program.

The completed full-network campaign used PRIMAT's native reaction inventory and supported numerical architecture across six preregistered baryon-density anchors and symmetric logarithmic rate perturbations.

| PRIMAT full-network metric | Result |
|---|---:|
| Committed native rows | **20,550/20,550** |
| Native reactions | **428/428** |
| Matched plus/minus pairs | **10,272/10,272** |
| Complete eight-branch ladders | **2,568/2,568** |
| Direct native rate proofs | **428 reaction identities exercised across the campaign** |
| Final nuclides per committed row | **59 finite positive final nuclides** |

The paper-facing full-network evidence chain is bound in:

```text
CPTG_PRIMAT_PaperReadyValidationEvidence_20260817_r02.zip
SHA-256: b41c8ee49477d8330deb56c85a230d8dcc1d4bba80a1405251a4d6ea7f9b3205
```

#### PRIMAT native source/Jacobian mechanism validation

A separate six-anchor native source/Jacobian campaign tested the source-to-endpoint mechanism without treating the completed endpoint campaign as a fit target.

| PRIMAT mechanism metric | Result |
|---|---:|
| Frozen predictions | **84/84** |
| Primary resolved tests | **48/48 PASS** |
| Primary + secondary resolved tests | **66** at response norm ≥ 10⁻⁴ |
| Active source-history samples | **6,283** |
| Maximum source-direction error | **3.17×10⁻¹⁶** |
| Maximum baryon-conservation residual | **4.44×10⁻¹⁶** |
| Worst primary endpoint-vector discrepancy | **0.0367629%** |
| Minimum primary endpoint direction cosine | **0.9999999921** |
| Maximum BDF/Radau cross-relative difference | **5.8248×10⁻⁷** |

The frozen prediction object is:

```text
Prediction freeze SHA-256:
679e3e6fc7432c0592004b9bf0985b0756323ceb327050f3303ca6c816f31bff
```

The accepted six-anchor result archive is:

```text
CPTG_PRIMAT_SIXANCHOR_NATIVE_SOURCE_JACOBIAN_RESULTS_20260817-202839.zip
SHA-256: 8511f3f79d4298d077a32625859440e01f3b530a3eda6a3e999582e1f938baf8
```

The controlling paper-ready package above binds the full-network and native-mechanism evidence into one audited authority object.

#### PRIMAT interpretation boundary

The PRIMAT result establishes:

- a completed 20,550-row native endpoint campaign;
- a fixed CPTG source construction carried through PRIMAT's native reaction dynamics;
- a separately frozen first-order source/Jacobian prediction layer;
- high-accuracy endpoint-vector recovery over the tested resolved light-sector domain.

It does **not** establish:

- one code-independent static endpoint susceptibility matrix;
- universal equality of PRIMAT, PArthENoPE, and PRyMordial internal currents or Jacobians;
- a universal second-order endpoint gate without a separately derived network-native second-order source kernel;
- native PRIMAT heavy-network authority beyond `A = 23`; the companion's `A ≥ 24` populations remain prescribed-bath reduced-graph transport coordinates.

PRIMAT is **closed** within its completed primordial/full-network and first-order native source/Jacobian scope.

---

### PArthENoPE 3.0 — second authority

PArthENoPE remains the completed second authority.

Its full-network Reaction-20 campaign used the official 26-nuclide, 100-reaction network; one native process per row; symmetric logarithmic rate branches; retained native trajectories; atomic row commits; matched-pair auditing; and complete eight-branch ladder auditing.

| Validation metric | Measured result | Fixed requirement |
|---|---:|---:|
| Native rows | **695/695** | 695/695 |
| Matched branch pairs | **338/338** | 338/338 |
| Complete eight-branch ladders | **84/84** | 84/84 |
| Reaction-20 direction cosine | **0.999979437501036** | ≥ 0.995 |
| D/H component ratio | **0.998516625** | 0.85–1.15 |
| He-3/H component ratio | **0.993911268** | 0.85–1.15 |
| Li-7/H component ratio | **1.008589280** | 0.85–1.15 |
| D/H six-anchor variation | **2.491%** | ≤ 10% |
| He-3/H six-anchor variation | **0.809%** | ≤ 10% |
| Li-7/H six-anchor variation | **1.156%** | ≤ 10% |

The accepted compact authority bundle is:

```text
CPTG_v129_r109_PArthENoPE_CPTG_Reaction20_ReplicationAuthorityBundle_20260731_r02.zip
SHA-256: 3d42d1cb1d710248841db8d7b1ceafcc7569f092b9b35c9f4114ce9615074cfc
```

It contains the clean campaign source, all 695 accepted native evidence rows, 338 matched-pair audits, 84 ladder audits, recomputation source code, provenance records, six PArthENoPE anchor vectors, threshold calculations, a claim matrix, and one-command audit regeneration.

After extracting the bundle, regenerate the audit with:

```cmd
RUN_VERIFY_AND_REGENERATE_WINDOWS.cmd
```

This verifies the retained evidence and reconstructs the post-execution audit without rerunning all 695 native rows.

#### PArthENoPE clean-room native-physics reconstruction

A later clean-room program reconstructed the source-to-observable response from PArthENoPE's own native source/Jacobian history without importing prior numerical response vectors.

| Metric | Result | Requirement |
|---|---:|---:|
| Exact-core source-space rank | **4** | rank 4 |
| Held-out native validation rows | **6/6 PASS** | 6/6 |
| Held-out density anchors | **3/3 PASS** | 3/3 |
| Numerical profiles per anchor | **2/2 PASS** | 2/2 |
| Worst frozen-operator core-4 relative error | **1.1143%** | ≤ 3% |
| Worst frozen-operator core-4 direction error | **0.4965°** | ≤ 1° |
| Worst individual-reaction relative error | **1.7098%** | ≤ 3% |
| Worst mass-seven polarization relative error | **1.6095%** | ≤ 3% |
| Mass-seven polarization sign agreement | **100%** | required |

```text
CPTG_PARTHENOPE_CLEANROOM_FINAL_RESULTS.zip
SHA-256: 8d503dd80036e6917500ca8367d8830d727930a23e8779fa3497bf9c1cf4d903
Construction freeze SHA-256:
770139cfb7b98f5c0cf6e5a038ea32aee4fb8c3430ed6fb02f38292221fe63d3
```

The construction freeze remained byte-identical through held-out validation. The campaign used no inherited PArthENoPE numerical-response vectors and no post-freeze refitting.

---

### PRyMordial — third authority renewal

PRyMordial remains the third authority target. Its earlier clean-room native-physics reconstruction remains supporting evidence, but the controlling full-network authority is now being rebuilt from scratch under the qualified **Candidate C-R** production solver architecture.

#### Earlier clean-room native-physics result

PRyMordial used unperturbed native trajectories, native analytic Jacobians, cancellation-free local reaction-source isolation, and direct variational propagation. No integrated rate-perturbed response rows were used.

| Metric | Result | Requirement |
|---|---:|---:|
| Exact-core source-space rank | **4** | rank 4 |
| Held-out native validation rows | **6/6 PASS** | 6/6 |
| Held-out density anchors | **3/3 PASS** | 3/3 |
| Numerical profiles per anchor | **2/2 PASS** | 2/2 |
| Worst frozen-operator core-4 relative error | **1.184957%** | ≤ 3% |
| Worst frozen-operator core-4 direction error | **0.496262°** | ≤ 1° |
| Worst individual-reaction relative error | **1.801701%** | ≤ 3% |
| Worst individual-reaction direction error | **0.425239°** | ≤ 1° |
| Worst mass-seven polarization relative error | **1.831870%** | ≤ 3% |
| Mass-seven polarization sign agreement | **100%** | required |
| Worst dual-profile core-4 relative difference | **9.71×10⁻⁸** | ≤ 3% |
| Worst dual-profile core-4 direction difference | **5.33×10⁻⁶°** | ≤ 1° |
| Worst local-source stoichiometric alignment error | **1.48×10⁻⁶°** | ≤ 0.01° |

```text
CPTG_PRYMORDIAL_CLEANROOM_FINAL_RESULTS.zip
SHA-256: b1810b1332fbb859437e15bbde3583330ec97456341f690b7f64b2948cfd3ba3
Construction freeze SHA-256:
e617fcbad9ff2db1edbc443c16348f67e7f7c197fa64503971c91dce862f72b0
```

This clean-room result is retained as native-physics support. It is not substituted for the renewed full-network endpoint authority campaign.

#### Candidate C-R production-solver qualification

PRyMordial's native stiff integration exposed a coordinate-conditioning boundary: one difficult row was solvable in scaled/shifted coordinates when the native coordinate realization refused, while another row exhibited the complementary behavior.

Candidate C-R therefore preserves **one physical Candidate-C solver contract** with deterministic numerical coordinate redundancy:

- Candidate-C physical equations and tolerances remain unchanged;
- scaled/shifted low-temperature coordinates are attempted first;
- native Candidate-C coordinates are permitted only after a strict evidence-valid scaled-coordinate solver refusal;
- no endpoint result is available to the coordinate selector;
- no tolerance relaxation is permitted;
- no alternate production model is introduced.

The high-risk qualification result is:

| Candidate C-R metric | Result |
|---|---:|
| Frozen high-risk rows | **30/30 PASS** |
| Scaled-coordinate accepted paths | **29/30** |
| Native-coordinate recovery paths | **1/30** |
| Dual-coordinate failures | **0/30** |
| Worst endpoint component-relative difference | **5.8799193110136354×10⁻⁸** |
| Fixed endpoint gate | **1×10⁻⁶** |
| Repeated `η10 = 6.094` baseline difference | **0.0** |
| Repeatability gate | **1×10⁻⁹** |
| Native-fail / scaled-pass boundary | **PASS** |
| Scaled-fail / native-pass boundary | **PASS** |
| Reference opened before prediction freeze | **False** |

The final qualification evidence object is:

```text
CPTG_PRyMordial_CANDIDATE_CR_FINAL_QUALIFICATION_EVIDENCE_20260817_r01.zip
SHA-256:
a5139f710d1438ddac205ac611c86a9a09c25f81f997e25f5b3ae71b7e4af830

Prediction freeze SHA-256:
5c99f23a998a131c5894da53b35aef55d02a9faf31d54a524af31517e3463bf0
```

Candidate C-R is therefore **qualified for full-network production use**.

#### Fresh 3030-row PRyMordial full-network authority campaign

The current full-network campaign is a fresh authority object:

```text
CPTG_PRyMordial_Full63Reaction_CandidateCR_AuthorityCampaign_20260817_r10.zip
SHA-256:
0a350cde6359d46ef230e390d4bed3f14421e9af9356e6cacd8849cc734e62a7
```

The campaign preserves the same full-network scientific ledger:

```text
6 baselines
63 native PRyMordial reactions
6 eta10 anchors
8 branches per reaction/anchor
3024 perturbed rows
3030 total authority rows
1512 matched +/- pairs
378 complete ladders
```

The campaign starts from a **zero-row production ledger**. No numerical production rows from the earlier quarantined campaign are imported. Candidate C-R is the single production solver authority object for all 3030 rows.

The full-network campaign is **active**, not yet closed. PRyMordial is promoted to completed third authority only after the fresh campaign reaches 3030/3030 strict-valid commits and passes final integrity, response-ladder, and locked Reaction-20 decision gates.

---

## Evidence Integrity

The accepted evidence chain preserves, as applicable:

- source, executable, ledger, construction-freeze, prediction-freeze, and package hashes;
- command records and native execution logs;
- endpoint, trajectory, solver, row-integrity, source-stoichiometry, and direct-variational audits;
- matched-pair and complete-ladder audit records;
- checkpoint and recovery state;
- preregistration records;
- output manifests and append-only event ledgers;
- raw abundance, Jacobian, and source/current data where exposed by the native code;
- independent reconstruction and recomputation scripts;
- clean-extraction replay evidence;
- claim matrices separating established, diagnostic, active, and untested statements.

The PRIMAT paper-ready evidence archive is the controlling primary authority object for the completed PRIMAT work. The PArthENoPE r109 authority bundle and clean-room final result are separate accepted second-authority objects. PRyMordial's Candidate C-R final qualification package is a completed solver-qualification authority object, while the fresh r10 full-network campaign remains active until its own final evidence package closes.

A package marked **accepted** in the package ledger is controlling scientific evidence unless a later accepted package explicitly supersedes it.

---

## Scope of Closure

The universal CPTG nuclear-reaction theory is structurally closed for the declared free-nucleon–deuterium–mass-three–helium-4 architecture.

Within that scope:

- the four-sector geometry is fixed;
- the reaction-source vectors are fixed;
- the transport and polarization structure is fixed;
- baryon and electric-charge constraints are fixed;
- the CPTG source-current hierarchy is fixed;
- PRIMAT has completed full-network endpoint authority and a separate native first-order source/Jacobian mechanism validation;
- PArthENoPE has completed its full-network endpoint authority and clean-room native-physics reconstruction;
- PRyMordial has completed Candidate C-R solver qualification and is executing a fresh full-network authority campaign;
- no reaction-specific geometric refitting is introduced after result exposure;
- network-local currents, Jacobians, source kernels, trajectories, normalization conventions, and numerical solvers remain code-local.

Theory closure does not by itself establish:

- completed absolute-rate derivations for every nuclear reaction;
- one universal static endpoint susceptibility matrix across independent BBN implementations;
- a universal second-order endpoint-response gate without a separately derived network-native second-order source kernel;
- full validation for every nucleus beyond the declared four-sector domain;
- native heavy-nucleus authority through `A = 119`;
- laboratory confirmation in every plasma regime;
- certification for safety-critical reactor control;
- replacement of independent experimental or network replication.

These are downstream validation and application domains, not open terms in the closed four-sector geometry.

---

## Computational Companion: `A = 1–119` Register and `A = 338` Frontier

The current *[computational companion](https://raw.githubusercontent.com/CLG2025/CPTG/main/nuclear-reactions/universal-theory/Complete-Processed-Nuclear-Chain.pdf)* is the paper-facing numerical realization of the universal theory. It preserves a strict distinction between native-network authority and reduced-graph continuation.

Its accepted scope is:

- **Native PRIMAT domain:** `A ≤ 23`, using the frozen PRIMAT trajectory and endpoint authority.
- **Complete mass-sector register:** `A = 1–119`, with `A = 5` retained as a physical-gap annotation and every external sector from `A = 24` through `A = 119` carrying strictly positive modeled support at all six frozen baryon-density anchors.
- **Seed-free continuation:** external states begin at exact zero and are driven only by the declared native-to-external boundary source; no heavy-sector endpoint vector or artificial abundance floor is imported.
- **Natural structural frontier:** the same selected reduced-graph rule remains gap-free through reachable `A = 338`; the eligible `A = 339` sector is disconnected because none of its allowed incoming sources belongs to the reachable component.
- **Frontier numerical qualification:** the accepted 512/1024/2048 primary/backward-Euler campaign passes all six anchors under unchanged convergence gates. The earlier 128/256/512 frontier run is retained as a fail-closed numerical non-qualification rather than being erased or promoted.
- **Source-tail robustness:** all **18/18** retained-source cases satisfy their applicable preregistered requirements, including all **12/12** hard-gated rows. After restoring the unrenormalized source amplitude, the largest absolute departure of the `A = 338` truncated-to-baseline endpoint ratio from unity is **2.36×10⁻¹³**.
- **Native mass-seven validation:** a separate prospective PRIMAT implementation of the locked mass-seven transport law passes **7/7** paired coordinates, with worst relative survival disagreement **0.018083%** and negligible non-mass-seven control shifts.

The external populations beyond `A = 23` are reduced-graph transport coordinates. They are **not** native PRIMAT heavy-network abundance predictions, a self-consistent coupled primordial heavy-element network, or precision primordial heavy-yield predictions.

### Controlling companion evidence

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

Additional cross-network post-silicon replication may be added where technically applicable. Such work would extend the evidence base without changing the present companion's explicit PRIMAT-primary claim boundary.

---

## Repository and Evidence Policy

This directory is the publication and evidence home for the CPTG nuclear-reaction program. Current papers, validation packages, protocol materials, source-network records, and supporting reproducibility evidence are maintained under **[`/nuclear-reactions/`](https://github.com/CLG2025/CPTG/tree/main/nuclear-reactions)**.

Many CPTG packages are complete audit objects containing source code, protocols, data, evidence, logs, manifests, reports, checksums, and upload markers. When these components are bound by package-level hashes, the package must remain intact.

Published packages should retain their original filename and internal directory structure. Extracted or convenience copies may be supplied for readability, but they do not replace the hash-authoritative archive.

The PArthENoPE and PRyMordial clean-room archives should retain their original filenames, published SHA-256 digests, and construction-freeze hashes. PRyMordial's Candidate C-R qualification and full-network production evidence remain separated so solver qualification cannot be mistaken for completed full-network scientific authority.

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

Independent validation results must never be used to alter the fixed CPTG geometry after exposure. Cross-network validation must retain its declared shared-observable scope and code-local boundaries; it must not be used to introduce network-specific geometric refitting.

Clean-room reconstruction packages must preserve the construction/held-out chronology: construction inputs are declared before execution, the construction object is hash-frozen before held-out rows begin, and held-out results may not be used to regenerate, rotate, rescale, or refit the frozen operator.

Candidate C-R coordinate selection must remain solver-success-only. Endpoint values, reference vectors, fitted responses, or post-result agreement may not determine which coordinate realization is accepted for a production row.

Any future extension must begin as a new validation domain. It must not silently refit or rewrite the consumed evidence supporting the closed four-sector theory.

---

## Security, Safety, and Disclosure

The software in this directory is research software. It is designed to fail closed when required authority is missing, hashes do not match, evidence records are malformed, conserved quantities exceed tolerance, states become nonphysical, numerical results are nonfinite, fixed authority boundaries are violated, or execution records are incomplete.

This repository does not claim that research software alone satisfies the certification requirements of reactor-control, medical, aerospace, or other safety-critical deployment environments.

The public README does not reproduce the protected closed-form equations. Formal derivations, normalization conventions, curvature structure, source-current definitions, source matrices, susceptibility coefficients, and reconstructive formula-package details remain in controlled research papers and hash-bound technical packages.

Public computational companions may disclose reduced-graph equations, numerical recurrences, verification scripts, and archived outputs when those materials do not reconstruct the protected universal formula authority. The universal parent paper is in final publication preparation; the protected governing formulas remain omitted from this README until their paper release.

---

## Current Status

The CPTG geometric nuclear-reaction theory is **structurally closed in its four-sector foundation and has completed primary PRIMAT validation**.

Current authority state:

```text
PRIMAT      primary authority    CLOSED/PASS
PArthENoPE  second authority     CLOSED/PASS
PRyMordial  third authority      C-R QUALIFIED; fresh 3030-row full-network campaign ACTIVE
```

The central validated structure includes the free-neutron/proton vertex, deuterium bridge, tritium/helium-3 closure sector, helium-4 saturation sector, ordered transport operator, charge-constrained polarization mode, rank-four reaction-source basis, fixed CPTG source-current hierarchy, and network-native dynamic observable pushforward.

PRIMAT supplies the completed 20,550-row primary full-network authority and the strongest completed first-order native source/Jacobian mechanism validation. PArthENoPE supplies a completed independent 695-row full-network endpoint authority and separate clean-room native-physics reconstruction. PRyMordial has passed Candidate C-R production qualification and is executing the fresh 3030-row campaign required to close the third authority.

The updated [computational companion](https://raw.githubusercontent.com/CLG2025/CPTG/main/nuclear-reactions/universal-theory/Complete-Processed-Nuclear-Chain.pdf) is the current PRIMAT-primary reduced-graph result: native authority through `A = 23`, a complete seed-free mass-sector register through `A = 119`, and a separately qualified natural structural frontier at `A = 338` with source-tail robustness established under the declared operator.

---

## Citation

**Curvature Polarization Transport Gravity**

Repository: https://github.com/CLG2025/CPTG

Author: **Carter L. Glass Jr.**

For the commissioning nuclear-reaction paper, cite:

Carter L. Glass Jr., *[Geometric Nuclear Reaction Theory in CPTG: Deuterium-Proton Capture and Primordial Mass-Seven Transport](https://raw.githubusercontent.com/CLG2025/CPTG/main/research/CPTG_Geometric_Nuclear_Reaction_Theory.pdf)*.

For the computational companion, cite:

Carter L. Glass Jr., *[A Universal Geometric Theory of Nuclear Reactions in CPTG: Computational Companion — PRIMAT-Anchored Reduced-Graph Reachability and the Complete A=1–119 Mass-Sector Register](https://raw.githubusercontent.com/CLG2025/CPTG/main/nuclear-reactions/universal-theory/Complete-Processed-Nuclear-Chain.pdf)*.

---

## BBN Software Citations

- C. Pitrou, A. Coc, J.-P. Uzan, and E. Vangioni, “Precision big bang nucleosynthesis with improved Helium-4 predictions,” *Physics Reports* **754**, 1–66 (2018), [doi:10.1016/j.physrep.2018.04.005](https://doi.org/10.1016/j.physrep.2018.04.005). Primary citation for PRIMAT.
- S. Gariazzo, P. F. de Salas, O. Pisanti, and R. Consiglio, “PArthENoPE revolutions,” *Computer Physics Communications* **271**, 108205 (2022), [doi:10.1016/j.cpc.2021.108205](https://doi.org/10.1016/j.cpc.2021.108205).
- Official PArthENoPE 3.0 program distribution: Mendeley Data, version 2, [doi:10.17632/wvgr7d8yt9.2](https://doi.org/10.17632/wvgr7d8yt9.2).
- A.-K. Burns, T. M. P. Tait, and M. Valli, “PRyMordial: the first three minutes, within and beyond the standard model,” *The European Physical Journal C* **84**, 86 (2024), [doi:10.1140/epjc/s10052-024-12442-0](https://doi.org/10.1140/epjc/s10052-024-12442-0).
