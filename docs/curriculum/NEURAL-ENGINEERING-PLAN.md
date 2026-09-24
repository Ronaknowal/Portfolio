# Neural engineering curriculum and authoring plan

Research baseline: **2026-09-09**. Scope confirmed by the user: biological neural engineering and neurotechnology, including brain-computer interfaces, instrumentation, brain and peripheral signals, neuromodulation, prostheses, rehabilitation, neural computation, NeuroAI and their connections to embodied intelligence.

This is a curriculum plan, not a completed course or a claim that a reader will know every discovery in an open research field. Its practical completeness target is that a learner can explain the principal mechanisms, follow a complete data or engineering workflow, choose a specialist direction, evaluate evidence, and identify what remains unknown. Completing web lessons cannot replace supervised experimental training, clinical training, professional qualification or device validation.

## Where a future agent should start

1. Read [the authoring handoff](../../LESSON-AUTHORING-HANDOFF.md) and [the teaching standard](../../LESSON-TEACHING-STANDARD.md). The accepted Linux lesson is the current experience benchmark: clear explanation, several mechanism-specific visuals where useful, guided practice, visible outputs and reasoned solutions.
2. Read this subject plan and the selected topic's machine-readable blueprint in [neural-expansion.js](../../src/learn/data/curriculum/neural-expansion.js).
3. Inspect the actual lesson, existing labs, prerequisite lessons and next lessons. A title being listed or a brief being present does not mean its content meets the new standard.
4. Resolve prerequisite titles through the live catalogue. Follow prerequisite edges rather than trusting the order of the old neuroscience sections. If a prerequisite is only planned, provide a short accurate bridge or improve the prerequisite first within the authorized increment.
5. Read the blueprint's research anchors, then find the specific primary evidence needed for the chosen lesson. Recheck all numerical, clinical, regulatory and current-software claims. The registry records research starting points, not certification that every planned sentence has already been verified.
6. Produce a topic-specific teaching plan, implement only the authorized lesson increment, verify its calculations/code/visual behavior, and update the handoff and topic status honestly.

The metadata exports **66 additional topics in nine sections**, **individual next-pass blueprints for all 26 original neuroscience topics**, **prerequisite mappings for those existing topics**, and a dated source registry. Every brief includes a summary, measurable outcomes, exact prerequisite titles, a 4–6-stage teaching sequence, visual question and interaction, practical task and success criterion, misconceptions, research sources, depth and review focus. The resulting neural module has **92 individually planned topics** before shared prerequisite subjects are counted.

## Scope and boundaries

The guided neural-engineering path should reuse existing programming, mathematical, machine-learning, control and computing lessons. Its neural topics remain in the expanded `computational-neuroscience` track so the existing topic URLs and progress identity are retained. A distinct guided path is useful because a brain-inspired AI learner and a neural-device engineer need different routes through the same catalogue.

The main progression is **biological question → physical mechanism → measurement → data quality → model and uncertainty → interactive system → useful outcome → evidence and stewardship**. It contains specialist branches, not a requirement that every beginner finish all modalities before reaching an initial project.

| Coverage area | Added topics | What the author must make teachable |
| --- | ---: | --- |
| Biological and physical foundations | 9 | Orientation; functional CNS/PNS anatomy; glia, metabolism and barriers; units and circuits; ion gradients and conductance; multicompartment modeling; injury and repair; hemodynamics; comparative models |
| Signals and experimental reasoning | 7 | Psychophysics; LTI systems and convolution; sampling and ADC limitations; spectra and time-frequency estimation; hidden-state estimation and identification; experimental units/power; neural information and usable communication |
| Instrumentation and device engineering | 9 | Electrode electrochemistry; material/tissue mechanics; analog front ends; referencing/noise; embedded and FPGA pipelines; wireless bandwidth/power/thermal constraints; packaging and verification; chemical sensing/delivery concepts; tissue engineering and organoid systems |
| Recording and imaging modalities | 9 | Intracellular clamp modes; EEG/ERP/MEG including OPM concepts; forward/inverse localization; ECoG/depth/high-density probes; EMG and peripheral signals; optical calcium/voltage imaging; fNIRS; structural/diffusion/molecular imaging; targeted perturbations |
| Reproducible data pipelines | 7 | Clock synchronization; artifact handling and leakage; spike sorting/curation; calcium pipelines; NWB/BIDS; open-data provenance; behavioral tracking and annotation |
| Neural models and NeuroAI | 6 | Encoding GLMs and point processes; deployment-matched decoding evaluation; drift/adaptation; causal inference; neural foundation-model evaluation; applied neuromorphic deployment |
| Closed-loop neurotechnology | 8 | Stimulation recruitment concepts; noninvasive modalities; DBS/responsive/adaptive systems; peripheral/spinal/FES systems; sensory prostheses; communication interfaces; control stability/shared autonomy; rehabilitation and neurofeedback transfer |
| Responsible translation | 6 | Clinical evidence; risk/regulation/quality; data privacy and agency; participatory accessible design; responsible animal/tissue research; cybersecurity and long-term software support |
| Integration projects | 5 | Auditable EEG; longitudinal decoder with simulated feedback; acquisition-chain verification; multimodal data/model comparison; translation and user-design dossier |

The 26 retained topics supply neuronal signaling, sensorimotor plasticity, internally generated states, HH/LIF models, Brian2/Nengo, population data/coding/dynamics, neural decoding, fMRI, connectivity/RSA, awake and primate electrophysiology, BCI architecture, SNNs and learning rules, predictive coding, memory, oscillations, neuromorphic foundations and connectomics. Their new briefs clarify scope rather than duplicating their identities.

Related material belongs elsewhere when a shared lesson already owns it: Python and scientific programming; linear algebra and probability; ordinary differential equations; Fourier/Laplace transforms; generic ML evaluation; robot dynamics, controls and embodied simulation; GPU implementation and profiling. Neural lessons apply these tools to biological signals and inference. Add a cross-link and a short bridge instead of cloning the general lesson.

## Routes through the curriculum

**First independent neural-data analysis.** Start with the orientation, neurons and anatomy; learn units/circuits and signals; connect sampling to spikes and fields; learn experimental units and timing; use a small EEG or simulated spike dataset; build a transparent preprocessing and encoding analysis; finish a bounded reproducible project. Do not delay all practical work until the complete device-engineering branch is finished.

**Neural instrumentation and embedded systems.** Follow the physical foundation into membrane/electrode interfaces, analog acquisition and noise. Then add digital conversion, buffers and timing, embedded/FPGA implementation, telemetry and power, materials, packaging and verification. Work toward the simulated acquisition-chain capstone and a documented requirements/evidence file. Shared GPU/C++/systems lessons are extensions when the workload justifies them.

**Neural decoding and BCI.** Start from the data route and movement/state decoding. Add grouped evaluation, state estimation, multidimensional outputs and BCI architecture; then study closed-loop stability, adaptation, communication and/or sensory interfaces. Use the decoder/simulator capstone and include user control, failure recovery and operational outcomes.

**Neural modulation, prostheses and rehabilitation.** Require biological mechanisms and measurement limits before discussing intervention. Compare normalized stimulation models, modality-specific evidence and controller architecture. Branch into motor/peripheral/spinal assistance, hearing/vision/touch, or DBS/responsive systems. Join the evidence, participatory design and stewardship lessons before the translation dossier. Study concepts and evidence; practical activities stay in simulation or openly reusable archived data.

**Computational neuroscience, NeuroAI and embodied intelligence.** Follow membrane models, population coding, state-space dynamics, learning and memory into causal model comparison, NeuroAI and connectomics. Compare predictions with neural and behavioral measurements and clearly separate artificial neural networks, biological models, neuromorphic execution and actual nervous systems. Link to the robotics and fly embodiment paths for body/environment coupling. Use matched baselines and validate neural and task behavior separately.

**Cellular, optical and tissue engineering.** Branch from cell biology and membrane/circuit foundations into clamp modes, optical indicators and imaging pipelines, targeted perturbation, chemical sensors and tissue/organoid models. Connect every measurement to its transfer function and every claimed intervention to controls and evidence. Primary-study appraisal and archived-data modeling are the appropriate independent practical work.

These are recommended routes. The exact prerequisite graph is the machine-readable authority for ordering. Do not introduce cycles by making a foundation lesson depend on its own advanced application. In particular, the oscillations introduction precedes the detailed spectral-analysis application; the BCI overview precedes closed-loop architecture and then stability/shared-control; decoder evaluation follows a basic decoding implementation; consent/stewardship is taught early and revisited in device-specific settings.

## Teaching patterns that fit this subject

Apply the same learning philosophy as Linux, but choose representations for biological and experimental reasoning. Avoid an identical article template or a required number of labs.

### A biological mechanism lesson

Start with a familiar task, then draw the relevant cells or pathway. Introduce measured quantities and timescales before naming every anatomical component. Explain the mechanism through a small causal sequence, then attach notation to quantities the learner already understands. Use one dynamic visual for the mechanism and another when an independent mechanism or scale change deserves it. For membrane models, an integration/threshold lab and a separate conductance/gating lab are more useful than a single overloaded control panel. End with a novel prediction, a misconception and a boundary of the model.

### A measurement-modality lesson

Trace **biological event → transducer or contrast mechanism → instrument → sampled data**. Show the hidden simulated truth beside the measured output. Explain spatial/temporal resolution, mixing, noise, reference and inverse ambiguity. Give one worked trace or image interpretation with units. Use an intervention on the measurement model—sampling rate, sensor position, indicator kinetics or clock delay—to reveal a specific limitation. Practice should distinguish several plausible sources of the same observation.

### A data-analysis lesson

Begin with the scientific question and inspect a small raw dataset. Draw its axes, timestamps, experimental units and train/test boundaries. Show the effect of each transformation with linked raw and processed views. Introduce a simple baseline before a complicated method. Explain every diagnostic plot and every important output. Include a deliberately broken case such as temporal leakage, drift, motion contamination, pseudoreplication or a wrong coordinate transform. Finish with a runnable analysis and an honest conclusion; a clean plot alone is insufficient.

### A model or mathematical lesson

Start with a prediction the model should make. Use a small hand-solvable example, then equations, then a numerical implementation. Name the observation, state, parameter, noise and objective separately. Treat identifiability and uncertainty as part of understanding the method. Compare a limiting case, a violated assumption and an alternative model. Geometric similarity, good prediction and biological mechanism are different claims and need different evidence.

### An engineering-system lesson

Draw the complete system boundary and follow a sample or action through it. Show timing, buffering, units and resource costs at stage boundaries. Use several focused labs when signal fidelity, latency and failure recovery are distinct conceptual problems. Requirements and failure behavior need measurable tests. A software simulation is labeled as such; a bench result does not imply human-connected or implant safety. Make practical tradeoffs reviewable in a small design and test record.

### An evidence, ethics or translation lesson

Begin with a person, intended use and a concrete claim. Use a primary-study evidence table or a fictional case, not decorative sliders. Separate measured results, inference, uncertainty and missing evidence. Compare alternatives and the effects of different design decisions on user agency, accessibility and long-term support. Practice asks the learner to justify a decision and identify evidence that would change it. Explain relevant technical and institutional concepts without presenting medical advice, legal certification or an approved clinical protocol.

## Depth and practical-work contract

Each lesson must define its scope and what a learner can do on completion. `core`, `specialist` and `frontier` are depth/route labels, not publication-quality labels. Advanced material still starts with understandable motivation. Frontier lessons separate established methods, current results, active uncertainty and speculation.

For each difficult concept, identify the learner's likely wrong mental model and provide the visual, example or exercise that corrects it. A visual earns its place by answering a question. Keep raw observations and processed interpretations distinguishable; show axes and units; use legends beyond color; retain an accessible text explanation and keyboard-operable controls. Offer a reset, a known scenario and an interpretation of changed outputs. Use animation only to show a process whose sequence matters.

The practice progression is **predict → inspect → calculate or implement → diagnose → transfer**. A lesson may need several short checkpoints before one integration task. Supply optional hints, full reasoned solutions, expected outputs or tolerances, and at least one unfamiliar case. Explain why an incorrect approach fails. Tests should verify scientifically meaningful invariants, reference values, dimensions, causality and edge cases rather than merely reasserting implementation details.

Every new neural brief specifies practical work using synthetic signals, simulations, licensed open deidentified data, or review of supplied evidence. No learner task should require self-experimentation, human stimulation, implant fabrication, surgical procedures, animal handling or new invasive recording. Hardware education may use isolated bench abstractions with no biological connection. Clinical and regulatory learning is analysis of evidence and design responsibilities, not authority to treat, diagnose or deploy a device.

Avoid giant mandatory downloads. Choose a documented subset with stable identifiers and a small CPU-first baseline. Label any optional GPU route and report compute/memory expectations. Preserve raw files, provenance and versions; do not upload participant data or infer identities. Check licenses and consent conditions before redistribution.

## Research anchors and refresh rules

The following sources were opened or returned in targeted searches on 2026-09-09. Links are starting points for authored lessons; claims must cite the exact page, method or primary study that supports them.

| Research anchor | Curriculum use and limits |
| --- | --- |
| [NIH BRAIN recording and modulation](https://www.braininitiative.nih.gov/research/neural-recording-and-modulation) | Check breadth across electrical, optical, chemical and other neural technologies. This research overview is not an implementation specification or clinical endorsement. |
| [NINDS Brain Basics](https://www.ninds.nih.gov/health-information/public-education/brain-basics) and [Allen brain science](https://alleninstitute.org/brain-science) | Beginner biological orientation and atlas/cell/connectivity resources. Confirm anatomy, species, scale and current dataset details from the linked scientific releases. |
| [Allen student research guide](https://alleninstitute.org/wp-content/uploads/2024/06/Student-Research-Guide.pdf) | Develop bounded open-data investigations rather than requiring new experimental acquisition. |
| [Neuromatch computational neuroscience](https://compneuro.neuromatch.io/tutorials/intro.html) | Inspect educational progression from foundations into models, signals, inference and projects. Create original lesson explanations and exercises. |
| [NEURON cable equations](https://www.neuronsimulator.org/en/latest/nmodl/transpiler/contents/cable_equations.html), [Brian2](https://brian2.readthedocs.io/en/stable/) and [Nengo 4.0.0 documentation](https://www.nengo.ai/nengo/v4.0.0/index.html) | Validate model implementation, units and numerical conventions. Pin the version actually used; Nengo's unversioned landing page was a development version at review. |
| [Intan amplifier documentation](https://www.intantech.com/products_RHD2000.html), [noise application note](https://intantech.com/files/Intan_noise_reduction_techniques.pdf) and [Neuropixels support](https://www.neuropixels.org/support) | Read concrete instrumentation examples and geometry. Distinguish vendor-specific values from general principles and biological safety evidence. |
| [MNE tutorials](https://mne.tools/stable/auto_tutorials/index.html) and [preprocessing](https://mne.tools/stable/auto_tutorials/preprocessing/index.html) | Validate EEG/MEG, fNIRS, source modeling, artifact handling, coordinate and analysis workflows. |
| [SpikeInterface quality metrics](https://spikeinterface.readthedocs.io/en/stable/modules/metrics/quality_metrics.html) | Review contamination, completeness and drift diagnostics. Metric definitions may change between releases. |
| [Suite2p](https://suite2p.readthedocs.io/en/latest/index.html) and [deconvolution FAQ](https://suite2p.readthedocs.io/en/latest/FAQ/) | Guide movie-to-trace analysis and avoid claiming exact spike counts from deconvolved fluorescence. |
| [LSL time synchronization](https://labstreaminglayer.readthedocs.io/info/time_synchronization.html) | Distinguish timestamps, clock correction, jitter, buffering and end-to-end validation. |
| [NWB training](https://nwb.org/training-materials/), [BIDS specification](https://bids-specification.readthedocs.io/en/stable/) and [DANDI documentation](https://docs.dandiarchive.org/) | Design semantically interpretable, reusable datasets. Record the actual specification version and dataset license/consent context. |
| [Neural Latents Benchmark](https://neurallatents.github.io/) and [FALCON](https://snel-repo.github.io/falcon/) | Study reproducible model evaluation and longitudinal decoding. NLB's official site states its EvalAI submissions closed in January 2026; do not write a task requiring an active submission service without checking. |
| [BrainGate neurotechnology](https://www.braingate.org/research-areas/neurotechnology/) and [speech restoration](https://www.braingate.org/research-areas/speech-restoration/) | Discover current primary studies and their specific participants, tasks and endpoints. Cite and inspect the linked study before reporting results. |
| [NIDCD cochlear implants](https://www.nidcd.nih.gov/health/cochlear-implants), [NINDS DBS](https://www.ninds.nih.gov/health-information/disorders/deep-brain-stimulation-dbs) and [NIBIB robotic/bionic devices](https://www.nibib.nih.gov/science-education/science-topics/robotic-bionic-medical-devices) | Ground sensory and motor examples in realistic uses. Verify current device-specific eligibility/authorization from the appropriate official record before making such claims. |
| [FDA implanted BCI guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/implanted-brain-computer-interface-bci-devices-patients-paralysis-or-amputation-non-clinical-testing) | Organize nonclinical and clinical evidence questions for the guidance's defined scope. It is not blanket approval of an interface category. |
| [FDA QMSR](https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr) | QMSR became effective February 2, 2026. Do not present the transition as future work. Check current requirements for the intended device and jurisdiction. |
| [FDA medical-device cybersecurity guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/cybersecurity-medical-devices-quality-management-system-considerations-and-content-premarket) | The February 2026 guidance supersedes the June 2025 version. Recheck current text when writing lifecycle/security lessons. |
| [FDA human-factors guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/applying-human-factors-and-usability-engineering-medical-devices) | The retrieved final guidance page is dated August 2026. Older teaching material citing only 2016 should be checked against it. |
| [FDA neurological standards and guidance](https://www.fda.gov/medical-devices/neurological-devices/standards-and-guidances-neurological-devices) and [software guidance navigator](https://www.fda.gov/medical-devices/regulatory-accelerator/medical-device-software-guidance-navigator) | Find relevant evidence domains and current recognized standards. Overview pages can retain older edition numbers: verify the recognized edition and applicability before making requirements claims. |
| [NIH BRAIN neuroethics roadmap](https://www.braininitiative.nih.gov/vision/nih-brain-initiative-reports/brain-20-neuroethics-enabling-and-enhancing-neuroscience) and [ARRIVE guidelines](https://arriveguidelines.org/arrive-guidelines) | Incorporate agency, consent, support, welfare and rigorous reporting. Ethical guidance, reporting guidance and legal authorization are different things. |
| [NIBIB biomaterials](https://www.nibib.nih.gov/science-education/science-topics/biomaterial-technologies) and [BRAIN human-tissue workshop](https://braininitiative.nih.gov/news-events/events/research-human-tissue) | Frame materials/tissue questions and unresolved organoid issues. Refresh with current primary studies for every substantive frontier claim. |
| [Lava architecture](https://lava-nc.org/lava_architecture_overview.html), [feedback alignment paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5105169/), [equilibrium propagation paper](https://arxiv.org/abs/1602.05179), [predictive coding paper](https://pubmed.ncbi.nlm.nih.gov/10195184/) and [free-energy theory paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2660582/) | Validate computational assumptions and distinguish original claims, later evidence and biological plausibility. |

Clinical or regulatory claims require fresh device-, indication- and jurisdiction-specific research. Software examples require current official documentation and a pinned executable environment. Frontier model or benchmark claims require a dated primary paper, data-overlap checks and a clear statement of what remains untested. Stable biological explanations still need authoritative references and careful species/context qualification.

## Validation and maintenance

Before publishing a lesson, verify its substantive equations, unit conversions, array axes, timestamps, solver assumptions, test splits and expected outputs. Use known synthetic truth for at least one mechanism or failure example where feasible. Test interactive controls against the same underlying calculation used by worked examples. Check keyboard access, responsive layout, reduced-motion behavior and legibility of plots. A visual should show a valid simplified model, not merely animate the expected answer.

Before updating the catalogue, check exact-title resolution, duplicate topics, missing prerequisites and cycles. Shared foundations remain single canonical lessons. Related existing titles that are close in wording must have different scopes documented: BCI overview → closed-loop architecture → stability/shared-control; functional connectivity/RSA methods → decoding/model-comparison RSA; neuromorphic foundations → deployed neural-interface tradeoffs.

This expansion was syntax-loaded in Node and checked for required blueprint fields, resolved prerequisite titles and cycles against the live merged catalogue. At the validation point, no missing neural prerequisite titles or dependency cycles were found. This is planning-data validation, not scientific validation of unwritten lessons or execution of neural experiments.

Future handoffs must record which topic was authored, which sources and versions were checked, which calculations/code/browser checks ran, unresolved limitations, and the next eligible lesson. Do not mark a planned brief as an implemented lesson. Retire stale recommendations by replacing the live source and recording historical context where it remains useful; do not maintain competing active teaching standards.
