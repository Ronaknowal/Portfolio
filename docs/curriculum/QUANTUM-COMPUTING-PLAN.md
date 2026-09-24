# Quantum computing, information and engineering curriculum

Updated 18 September 2026. This is a catalogue expansion with named concept coverage, prerequisites and starting teaching briefs. It does not complete lesson research/write or implementation. The [handoff](../../LESSON-AUTHORING-HANDOFF.md), teaching standard and two-phase ledger continue to control authoring work.

## Live structure and preservation

The former **Quantum AI & Quantum Computing for ML** module is expanded and displayed as **Quantum Computing, Information & Engineering**. Its stable module ID remains `quantum-ai`; every one of its 48 earlier topics keeps its title, ID, URL and any prior blueprint. There are now **106 topics in 12 substantial sections: 48 retained and 58 new**. No separate duplicate quantum module was created.

The new **Quantum Computing** guided path has stable path ID `quantum-computing`. It includes the expanded module and the shared prerequisites its topics actually require. It does not copy mathematics, Python, ML, RL or finance lessons. The complete catalogue now contains **1,460 unique topics, 29 modules and 10 guided paths**. Publication and both delivery phases are unchanged.

The [detailed ordered syllabus](QUANTUM-COMPUTING-SYLLABUS.md) is generated from the live catalogue and records the exact topic-to-concept mapping. Every quantum topic has searchable named subtopics and reviewed prerequisite links. The 58 additions have individual starting briefs; the earlier mathematical bridge keeps its existing brief. The other 47 retained topics still require detailed design during their eventual authoring request. None of these counts claims that a manuscript or interactive lesson is finished.

| Section | Topics | Learning responsibility |
| --- | ---: | --- |
| Foundations, States & Quantum Reasoning | 10 | Tasks, amplitudes, measurement, relative phase, state spaces, gates, composite systems and dynamics. |
| Quantum Information, Measurement & Protocols | 8 | Teleportation, Bell correlations, channels, generalized measurement, reconstruction, entropy and information limits. |
| Circuit Construction & Quantum Software Engineering | 8 | Frameworks, reversible construction, oracles, synthesis, languages, routing, adaptive execution and reproducibility. |
| Core Algorithms, Complexity & Classical Simulation | 11 | Promise algorithms, search, amplitude estimation, QFT/QPE/Shor, walks and several classical simulation regimes. |
| Advanced Algorithm Design, Simulation & Chemistry | 9 | Hamiltonian access, product formulas/LCU/qubitization, QSP/QSVT, linear systems, fermions, many-body and sampling tasks. |
| Variational Algorithms, Optimization & Quantum Learning | 12 | Data access, parameterized circuits, gradients, VQE/QAOA/annealing, trainability, kernels and generative models. |
| Applied Quantum Learning, Finance & Evidence Limits | 9 | RL, transfer, reservoirs, language, Monte Carlo distinctions, financial tasks, dequantization and advantage claims. |
| Physical Qubits, Hardware & Control Engineering | 9 | Superconducting, ion, atom, photonic, spin/defect and topological approaches, plus control and readout. |
| Noise, Characterization, Benchmarking & Mitigation | 7 | Physical error models, diagnostic experiments, fair benchmarks, mitigation, suppression and open-system dynamics. |
| Error Correction, Fault Tolerance & Resource Engineering | 9 | Stabilizers/CSS, surface and sparse codes, bosonic encodings, decoding, logical gates and physical resource budgets. |
| Quantum Communication, Cryptography & Sensing | 8 | QKD, classical PQC, repeaters, modular computation, delegation, metrology, measurement-based computation and randomness. |
| Professional Practice, Reproducibility & Integrated Capstones | 6 | Research critique, algorithm, logical-memory, chemistry and network projects, and role-specific development. |

Sections are coherent study groups, not one-algorithm modules. The full module reading order is authoritative; prerequisites support review without silently reshuffling it. Named terms such as **QSVT, GKP, BB84, ML-KEM, QIR, Rydberg blockade, Kraus operators and magic-state distillation** should be findable without guessing a broad topic title.

## Specialist routes within the module

| Role or interest | Emphasis | Demonstrable result |
| --- | --- | --- |
| Algorithm / information researcher | Foundations, information, circuit construction, algorithms, simulation and resource analysis | A precisely stated result or independently checked algorithm comparison with access assumptions and end-to-end costs. |
| Quantum software / compiler engineer | State semantics, programming, synthesis, routing, dynamic circuits, simulation, testing and resource estimation | A verified small compiler or experiment workflow with explicit conventions, backend constraints and reproducible artifacts. |
| Hardware / control engineer | States/dynamics, channels, physical platforms, pulse control, noise, characterization and error correction | A modeled or measured calibration/characterization study with units, uncertainty and stated laboratory limits. |
| QEC / architecture researcher | Stabilizer simulation, noise, codes, decoders, logical gates, networks and resource estimation | A reproducible logical-memory or architecture study with circuit-level assumptions and decoder/communication cost. |
| Scientific or financial applications researcher | Algorithms and data access, chemistry/optimization/ML/finance branches, classical baselines and fault-tolerant resources | A bounded problem with reconciled model, algorithm, statistical and hardware error budgets. |
| Networking / cryptography / sensing specialist | Information protocols, channels, networks, security models, metrology and certification | An independently checkable protocol or estimation study with explicit trust and physical assumptions. |

These are optional emphases, not promises that every role requires every frontier branch. Hardware careers need appropriate physics, electronics, optics, cryogenics or laboratory training beyond reading a website. Add contextual prerequisites or destination notes when a future topic design identifies a concrete gap; do not turn a beginner's first circuit into a requirement to finish every physics discipline first.

## Teaching contract for future authors

Run the topic preflight and read `subtopics`, `scopeNote` when present, prerequisites, domain guidance and destination notes. Retained broad titles do not exempt an author from designing their concept-level progression. Named labels are teaching obligations, not decorative search keywords. An author may adjust the sequence or split a genuinely overloaded lesson, but must preserve stable identities and explicitly maintain where every useful concept is owned.

Start with a concrete task and a very small state/circuit. Explain the physical or computational meaning before notation. Define basis order, tensor/register order, units, phase conventions, initial state and what the learner can actually observe. Build from intuition to exact calculation, finite samples, noise, implementation and advanced assumptions. A first-time learner must understand the input, mechanism and output without already knowing quantum vocabulary.

Use the representation that answers the current question: amplitude/phase arrows for interference; a Bloch sphere only for a single qubit; joint probability and reduced-state views for entanglement; circuit and register-lifetime traces for uncomputation; operator maps for fermions; physical diagrams and pulse traces for control; space-time syndromes for decoding; network/memory timelines for repeaters. Add multiple focused diagrams and labs when independent mechanisms need them. A probability bar chart alone cannot teach phase, and a generic slider box is not a substitute for a circuit or syndrome investigation. Preserve the website's visual quality, accessible labels and responsive layout rules.

Each substantial example must distinguish **exact ideal values**, **finite-shot estimates**, **noise-model simulations** and **real-device observations**. Do not silently present a simulator's access to the full statevector as a capability of hardware measurement. Give independent reference values, meaningful uncertainty and changed-case practice with explained solutions. Numerical exercises must state truncation, convergence, random seed and model assumptions where relevant; never generate decorative performance curves or invented hardware measurements.

Software examples need dated SDK versions, installation assumptions, complete inputs/outputs and a small independent semantic check. Keep simulator size and browser work bounded. A future implementation should load only its own circuit models, example data and visuals; it must not eagerly load a quantum SDK or every simulator with navigation. Hardware access must be optional unless specifically requested; simulations and recorded data should remain useful for learning. Do not submit paid cloud jobs merely because they appear in a lesson exercise.

Teach applications through the mechanism and a fair comparison. Account for input loading, QRAM/oracle access, coherent depth, output extraction, compilation, shots, classical optimization, error correction and physical runtime as appropriate. Distinguish theoretical/query advantage, a demonstrated benchmark, practical advantage and speculative future use. Strong classical baselines and comparable target errors matter. Date device and frontier claims and recheck them while authoring; do not turn a vendor roadmap into an established result.

Specific distinctions that must survive simplification:

- Relative phase versus global phase; amplitudes versus probabilities; superposition versus classical mixture; entanglement versus arbitrary correlation; no-signalling versus nonlocal correlations.
- A POVM's probabilities versus the quantum instrument's state update; positivity versus complete positivity; tomography versus estimation of selected properties.
- Physical versus logical qubits; error suppression, mitigation and correction; memory protection versus universal fault-tolerant computation; code-capacity versus circuit-level thresholds.
- Quantum-inspired classical algorithms versus algorithms on quantum hardware. The retained **Quantum Monte Carlo Methods** topic must distinguish classical QMC for quantum systems from amplitude-estimation methods on quantum computers.
- QKD versus post-quantum classical cryptography. In ML-KEM and ML-DSA, “ML” means module lattice, not machine learning. Retain the old topic ID while considering a compatible broader display title during its future authorized rewrite.
- Experimental evidence for topological devices versus proof of a usable topological qubit or fault-tolerant architecture. Frontier interest is not a license to state disputed interpretations as fact.

Practice should move from predictions and hand traces to small implementations, diagnosis and transfer. Capstones need artifacts another person can reproduce, including failures and negative results. Curate annotated alternative resources during each lesson's actual research phase: useful official tutorials, primary papers, articles, lecture videos and playlists. There is no quota and no requirement to copy another course's organization.

## Scope research and sources

The curriculum is an original coverage design informed by primary educational/documentation resources inspected on 18 September 2026. A course index or paper abstract anchors scope; it does not verify all future lesson claims. Historical resources below are valuable foundations, not current hardware scoreboards.

| Resource | Scope use and inspection limits |
| --- | --- |
| [IBM Quantum Learning](https://quantum.cloud.ibm.com/learning/en) and [error-correction course](https://quantum.cloud.ibm.com/learning/en/courses/foundations-of-quantum-error-correction) | Course overviews for information, algorithms and QEC. Individual proofs/exercises require their own research. |
| [Preskill's Caltech lecture notes](https://www.preskill.caltech.edu/ph229/) | Broad theoretical map and primary lecture-note starting points; the full note collection was not re-read in this planning pass. |
| [MIT Quantum Information Science I](https://ocw.mit.edu/courses/8-370x-quantum-information-science-i-spring-2018/) | Structured foundational alternative learning route, including its course materials/videos. A dated course, not a claim about current SDKs. |
| [IBM programming guides](https://quantum.cloud.ibm.com/docs/en/guides), [OpenQASM](https://openqasm.com/language/) and [Cirq noisy simulation](https://quantumai.google/cirq/simulate/noisy_simulation) | Software, language, simulation and execution boundaries. Recheck actual APIs and supported capabilities during authoring. |
| [PennyLane demonstrations](https://www.pennylane.ai/demonstrations) and [adaptive chemistry circuits](https://pennylane.ai/demos/tutorial_adaptive_circuits) | Application and practical experiment starting points; no example was executed as part of curriculum planning. |
| [QSVT original paper record](https://arxiv.org/abs/1806.01838) | Advanced algorithm framework and its assumptions; abstract/record inspected, full technical proof review deferred. |
| [Materials challenges for quantum hardware](https://doi.org/10.1126/science.abb2823) | Platform breadth and physical engineering scope from the primary publication's search excerpt. Direct page retrieval returned 403; do not claim the full paper was reviewed. |
| [Quantum internet: a vision for the road ahead](https://qutech.nl/wp-content/uploads/2018/10/Quantum-internet-A-vision.pdf) | Protocol/functionality and network-resource framing from the retrieved paper, not a current deployment forecast. |
| [Quantum sensing](https://arxiv.org/abs/1611.02427) | Expert-authored review record for the sensing/metrology branch; precise bounds require lesson-level inspection. |
| [Microsoft resource estimation](https://learn.microsoft.com/en-us/azure/quantum/intro-to-resource-estimation) | Conditional mapping of algorithms to physical costs and assumptions. Tool output is not a device delivery prediction. |
| [NIST post-quantum standardization](https://csrc.nist.gov/Projects/Post-Quantum-Cryptography/Post_Quantum_Cryptography-Standardization) | Current official starting point for PQC families, standards and migration; recheck exact versions and guidance during the cryptography lesson. |

This is broad foundational-to-specialist coverage, not a guarantee of every possible future technique, employment outcome or research skill. New discoveries should become explicit scoped additions or destination instructions, following the same ongoing coverage policy as the rest of the site.

## Source ownership and verification

- `src/learn/data/curriculum/quantum-computing.js` owns the ordered expansion and named coverage; the builder refuses to omit or reuse an earlier quantum entry. `quantum-curriculum-sources.js` owns starting references. Later authored topic blueprints still override starting briefs through the existing integration rule.
- Module identity remains `quantum-ai`; guided-path identity is `quantum-computing`. Generated navigation includes compact subtopic labels; detailed starting outlines load independently when selected.
- `quantum-curriculum-baseline.json` records the pre-change 1,402 IDs, earlier module contents and phase/publication hashes. Keep it as conservation evidence rather than a parallel live catalogue.
- Run `node scripts/verify-quantum-curriculum.mjs --write-syllabus` after scope edits. Its optional `--check-planning-boundary` also checks this expansion's untouched modules and phase/publication state; that historical constraint must not block future authorized lesson work.
- Run the normal curriculum, artifact, inventory and application checks for runtime changes. The scoped browser verifier covers the new path, full module order, named search, planned pages, navigation, preserved progress, responsive layout and lazy loading. Verification evidence records the actual tested scope.

## Completed planning verification — 18 September 2026

The expansion preserves all 1,402 earlier catalogue IDs, every earlier quantum topic and the exact section/topic order of the other 28 modules. Publication and the two-phase delivery ledger remain byte-for-byte unchanged. The catalogue now has 1,460 topics across 29 modules and 10 guided paths; the Quantum Computing path resolves to 106 quantum topics plus 19 prerequisite topics across seven modules.

Curriculum integrity, prerequisite ordering/cycles, generated-artifact freshness, named search, inventory reconciliation and the production build passed. The [browser evidence](quantum-curriculum-browser-evidence.json) records 48 passing checks at 1,440px and 390px, including route counts, all module titles in order, representative planned entries, previous/next navigation, retained completion IDs, overflow and on-demand outline loading. Both saved introductory-page screenshots were visually inspected. No browser page errors were observed. This is sampled page coverage, not a visual review of every planned entry or future lesson.

The [source-bound verification record](quantum-curriculum-verification.json) identifies the tested source, build and captures. These results certify the catalogue integration, not completion of research/write or implementation. Individual scientific claims, runnable examples, diagrams, labs and learning resources still require the normal topic-level authoring and review phases. Nothing was deployed.
