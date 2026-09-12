# Domain teaching playbook

Reviewed 10 September 2026. This document implements the [teaching standard](../../LESSON-TEACHING-STANDARD.md). Use its strategies selectively. The common contract is understandable mechanism, meaningful representation, complete example, independent practice, explicit assumptions and a next step. Page structure, lab count, examples and depth differ by topic.

## Match the representation to the question

Domain strategies are starting points, not uniform visual templates. Choose the layout, terms and controls around the particular objects and operation being taught. Introduce subject terminology and explain conventions before relying on them. Keep familiar controls and accessible behavior across the site; reuse a visual form whenever it serves the next concept well. Change form when the learner needs a different relationship made visible, including within one topic. These are possible matches, not required widgets:

| Learner question | A representation to consider |
| --- | --- |
| Which symbols merge, and how does changing the corpus affect that choice? | Token strips with a selected pair, frequency counts and linked segmentations; a guided trace followed by an editable trainer when both help. |
| Why does changing one name's object affect another name? | Names and objects connected by reference arrows, with the changed object and resulting values visible. |
| Which part of an algorithm changes, and what invariant survives? | The actual tree, array or frontier with active elements and before/after structure, rather than only a textual event log. |
| What does a transformation or probability statement mean? | Movable geometric objects or a correctly scaled distribution with the relevant correspondence or probability region identified. These need different views when they teach different mechanisms. |
| Where do execution, memory access or feedback occur? | Worker/address mappings, execution lanes or a closed-loop trajectory chosen for the specific question, with units and timing where relevant. |
| What does a neural sensor measure, and what changes after processing? | A measurement/interface schematic linked to a time trace, artifact or filter effect; distinguish the biological process from the measured signal. |

Keep accurate text, code and tables when those are the clearest representation or necessary companions. Follow the standard's graph-provenance rules: a compelling plot or realistic interface does not establish empirical, biological or performance claims.

## Mathematics, probability and statistics

Begin with a question that a learner can state without symbols. Identify objects, units and what is known versus unknown. Move between a tiny numerical case, a diagram and symbolic notation, naming the correspondence each time. Derive the result in justified steps; explain what each operation preserves. Only then generalize. Offer a formal proof or deeper derivation when it is an outcome, with an accessible route through its key idea.

Use geometric transformations for linear algebra; local change and accumulation for calculus; sample spaces, repeated samples and shaded areas for probability; explicit data-generating models for inference. A matrix picture should connect cells to the operation; a distribution needs labeled axes and a distinction between density and probability. When a topic includes both sampling variation and a decision rule, use separate views for those different mechanisms.

Practice moves from reading a representation to a hand calculation, a changed problem, a mistaken derivation to repair, and an assumption/counterexample check. A statistics solution interprets effect size and uncertainty in context. Distinguish a theorem from a heuristic, an estimator from its realized estimate, confidence from posterior probability, and correlation from intervention. Do not claim the three original pilots already cover every conceptual hurdle.

## Data structures, algorithms and discrete reasoning

Start with an actual operation: maintain the smallest pending job, find a route, update a range, or match a string. Show a simple correct method and the reason it becomes expensive. Give concrete state, then expose how the new representation changes permitted operations. Trace a small input with a cursor/frontier, data structure and output visible together.

Introduce the invariant before relying on it. Explain initialization, preservation, termination and how the invariant implies the answer. Count a meaningful resource under a stated input-size and cost model. Contrast worst-case, average/expected and amortized bounds when relevant; do not equate a runtime plot with an asymptotic proof.

Use separate investigations for structural operations and the algorithm that uses them. A heap explorer and a shortest-path frontier can share labels without becoming one crowded lab. Include empty/singleton inputs, duplicates, skew, disconnected graphs and numerical limits where applicable. Practice includes predicting the next state, implementing from the invariant, constructing a counterexample, comparing approaches and transferring to a changed constraint. Assess correctness as well as performance; show expected output and reasoning.

Every DSA lesson also curates official LeetCode practice following [DSA-PRACTICE-STANDARD.md](DSA-PRACTICE-STANDARD.md), including the evolving pattern-coverage map, staged prerequisites, optional hints and changed-constraint transfer. External submissions supplement fully explained local practice. Assess independent reasoning on unseen and mixed tasks; problem counts and completion marks do not certify readiness for every interview.

## Programming and scientific computing

Begin with inputs and the desired result. Introduce language entities before syntax: a name versus an object, an iterator versus a collection, a row versus an index label, or a shell versus the operating system. Trace execution beside state and output. Give line-by-line commentary only where a line changes the learner's model; avoid paraphrasing obvious syntax throughout.

The approved Linux lesson is the reference for exposing mechanisms and pacing. For Python, arrows between names and objects teach sharing and mutation. For NumPy, align dimensions and highlight the actual operands of one output cell. For Pandas, trace source records through a join and explain multiplicity before summarizing the result. For plotting, connect a scientific question to axes, scales and uncertainty; visual attractiveness does not validate a result.

Use compact inline diagrams at the point of explanation as well as larger investigations. Examples to consider when they resolve a real hurdle: two names reaching one object versus an independent copy; one reusable collection supplying two independent cursors versus two names sharing one cursor; input shapes aligned with an output cell; quoted versus unquoted shell arguments shown as distinct received strings; notebook document, live kernel and saved output shown as separate states. Keep the example values and relationships visible instead of replacing the picture with boxes of descriptive prose. Follow with a lab when changing input, stepping execution or exploring a failure adds understanding. These are options to assess, not required diagrams or a shared template for every programming lesson.

Provide runnable setup, imports, data, commands and expected results, including explained warnings or failures. Verify against the actual language/library. Browser simulations need explicit supported cases. Exercises progress from prediction through repair to an independent small task; require a changed-input result to avoid recipe copying. Treat numerical stability, units, dtype, indexing, randomness and reproducibility as reasoning, not footnotes.

## Classical ML, deep learning and generative models

Start with the prediction/generation question and the unit of data. Establish a baseline and split protocol before model complexity. Trace a single example through representations, tensor shapes, computation, objective and parameter update. Separate training behavior from inference. Translate losses into what errors the system is encouraged to change.

For architecture lessons, a data-flow diagram should reveal dimensions and dependency paths; a separate experiment can show a training or capacity effect. For probabilistic generators, distinguish model distribution, learned approximation, sampling rule and randomness. For an optimizer, show parameter updates separately from the changing predictions. Ablations change one interpretable choice with comparable conditions, and report uncertainty where experiments vary.

Practice includes a manual forward/update step, inspecting a tiny dataset, diagnosing leakage or mismatch, modifying an implementation and interpreting an ablation. Publish a reproducible small CPU route where feasible; state when realistic results require unavailable data or compute. No fabricated benchmark measurements or unsupported claims that one model wins universally.

Use a constructed fixture for the hand-traced update and a small real dataset for the library fit and diagnostics; the real data supplies the question the method is answering and the moment a learner can judge whether the answer is useful. Embed the data so the program runs offline and record its provenance. When a method's canonical treatment includes a hardness result, a worst-case versus typical-case gap or a convention that differs between libraries, teach it briefly with its source; these are the facts a practitioner meets first and a lesson that omits them reads as incomplete to anyone who has opened the standard text.

## LLMs, agents and evaluation

Explain the task boundary first: next-token prediction, retrieval, a tool action, a judged answer or a deployed service. Follow one real miniature request through tokenization, context construction, model output, parsing and any downstream action. Distinguish probabilities, logits, scores, confidence estimates and calibrated reliability.

Agents need explicit state, observation/action boundaries, failure/retry/stop conditions and a limited permission model. Retrieval lessons must show the source records, retrieval choices and answer evidence. Evaluation lessons start with the construct and decision being measured, then sampling, annotation/rubric, scoring, uncertainty, validity threats and the resulting decision. Scores and completion markers are not demonstrations of competence.

Use separate views for model computation and system workflow. Include a failure case, baseline, held-out evaluation and cost/latency or human effort tradeoff. Exercises ask learners to repair a misleading metric, design a discriminating test, detect contamination or diagnose a tool failure. Date model/API/benchmark facts; verify current interfaces rather than copying historical snippets. Harmful capability claims, deployment reliability and clinical use require evidence appropriate to the claim.

## GPU engineering, compilers and distributed systems

Start with a correct small CPU computation or systems contract. Show how elements map to workers and memory addresses. Teach execution groups, communication and synchronization with explicit scope; then implement and compare outputs. A model of a warp or cache is deliberately simplified, not a promise about all devices.

Separate arithmetic correctness, memory safety, concurrency correctness, numerical error and performance. Use independent CPU/reference-library results, edge dimensions, sanitizers and tolerance justified by dtype/operation. A race must be explained even if a sample run happens to pass. For performance, state hardware, software, sizes, warmup, synchronization, timing boundaries and variability. Diagnose with a profile and a cost model before changing a kernel; compare after measurement.

Visuals may include address-to-transaction maps, dependency timelines, occupancy/resource budgets, arithmetic-intensity plots and collective communication flows. Several are necessary for a topic spanning unrelated bottlenecks. Practice includes repairing an indexing/race bug, choosing an optimization from evidence, checking a nonmultiple tile size, and reporting a regression. Provide trace-analysis or CPU exercises when a compatible GPU is unavailable; never label these hardware performance validation.

## Robotics, reinforcement learning and embodied intelligence

Start with task, environment and measurable success. Introduce frames, units, state, observations and actions with a small physical story. Connect sensing → estimation → planning/control → actuation → changed environment → next observation. Show how delay, noise, model error and constraints affect that closed loop.

Separate policy learning, state estimation and low-level control. In RL, expose reward timing, return, bootstrapping and exploration separately when those are distinct hurdles. Use a state-transition view, trajectory plot and a control-response explorer for different questions. In robotics, link coordinate frames to sensor measurements and desired motion; do not leave arrows or transform direction implicit.

Practice includes predicting the effect of delayed feedback, diagnosing a frame error, comparing estimators/controllers, evaluating held-out environments and reproducing a simulated task. Report seeds, simulation timestep, actuator limits, resets and termination/truncation conditions. A simulation benchmark does not establish physical safety or sim-to-real performance. Physical extensions require an appropriate supervised setting; core exercises can use bounded simulation.

## Neural engineering and neurotechnology

Follow a biological question through what a sensor can actually measure. Distinguish cell mechanisms, extracellular/electrical/optical measurements, neural codes and behavioral outcomes. Show biological and engineering time/length scales and units; an electrode signal is not a direct readout of a person's thought.

Connect tissue/interface → analog front end → sampling/reference/timing → artifact handling → features/state estimate → decoder or controller → outcome. Introduce a noise/artifact example before treating a pipeline output as neural evidence. Separate biological plausibility, predictive validation, causal intervention and therapeutic benefit. A decoder evaluation should expose participant/session/trial separation and nonstationarity, not only average accuracy.

Use membrane/equivalent-circuit diagrams, signal chain views, raw-versus-processed traces, spike rasters, latent-state projections and closed-loop timelines as distinct learning tools. Practice uses synthetic signals, suitable open de-identified datasets, simulations or electrically isolated bench phantoms. Human stimulation, invasive procedures and clinical protocol execution are not home exercises. Teach translational design and evidence with qualified supervision and jurisdiction/date-specific sources.

Stimulation lessons need biophysical effect, interface constraints, control goals, monitoring, uncertainty and failure mechanisms. No single charge/voltage limit guarantees safety across tissues, waveforms and interfaces. Clinical claims require appropriate study design and endpoints. Device quality systems, privacy, participant autonomy and long-term support belong in the technical design, not a detached final disclaimer.

## Scientific applications, frontier research and reference pages

For finance, quantum AI, evolutionary methods, fly embodiment and other applications, establish domain assumptions before importing an ML tool. Use the relevant data-generating or physical process, a domain-valid baseline, uncertainty and a reproducibility route. Financial examples must account for temporal leakage and costs; quantum examples must distinguish ideal simulation, noisy hardware and demonstrated advantage; animal models must separate species-specific evidence from extrapolation.

A research lesson asks a precise question, reconstructs the method, locates the evidence, attempts a scaled reproduction and identifies what remains unsettled. Date claims and compare an appropriate alternative. A framework reference teaches one coherent workflow, links concepts to APIs and records the tested release. A glossary gives plain meaning, disambiguation, a miniature example and a link to the full explanation; it should not be inflated into a fake full course.

## Choosing depth and changing a stored plan

The first pass must achieve a scoped useful outcome. Intermediate material introduces realistic variation and diagnosis; advanced branches supply formal justification, difficult edge cases and alternatives. A topic may be split when independent prerequisite chains or different practices would overload one page. Preserve links and compatibility when splitting a bundled topic.

Change the proposed sequence, scope/title, visual type, examples or exercise volume when that makes the learner's causal model clearer, including discoveries made midway through writing. Record the reason in the topic brief, check that useful existing coverage and prerequisites are retained, and keep a core route. Follow the standard's title/identity compatibility rules. Save ideas better taught elsewhere in [destination-topic notes](topic-notes/README.md) for that author's reasoned review. Do not add a lab solely to make the page resemble Linux. Do not remove technical depth solely to shorten the page.

## Choosing interesting connections within a domain

Use these as directions for investigation, not a list of applications that must be inserted. Verify each concrete claim when authoring. Several distinct connections can belong in one lesson; some lessons need no extra branch.

| Domain | Look for a connection that deepens understanding |
| --- | --- |
| Mathematics/statistics | An unexpected equivalence, illuminating counterexample or second setting for the same operation; work through the correspondence and the assumptions that allow it. |
| DSA | A different task solved by the same invariant or representation, or a changed constraint that breaks the familiar approach; trace why the transfer succeeds or fails. |
| Programming/scientific computing | A visible consequence of hidden runtime/data behavior or a useful workflow enabled by it. Keep initial examples accessible; introduce specialised applications after the required language and data concepts. |
| AI/ML/LLMs | A less familiar data/task setting, useful baseline or revealing failure case; map inputs, objective and evaluation rather than simply naming a product or industry. |
| GPU/systems | The same memory, synchronization or scheduling principle under different workloads; explain the work/data mapping and measure any performance claim. |
| Robotics/embodied intelligence | A new environment or sensing/action constraint that reveals the same feedback or estimation mechanism; distinguish the conceptual example, simulation and physical evidence. |
| Neural engineering | A different measurement, interface or analysis question that reveals what a signal can and cannot support; distinguish interesting engineering possibility from demonstrated biological or clinical benefit. |

Explain each chosen connection to the depth needed to follow its mechanism and consequence. A compact fact can clarify a misconception; a substantial application needs a worked chain, interpretation and limits, with a visual or practice where useful. If its main explanation needs another topic's prerequisites and outcomes, teach the local bridge and persist the full suggestion there. Interest should arise from understanding a useful or surprising relationship, not from unsupported novelty claims.

Pedagogical grounding: the [IES practice guide](https://ies.ed.gov/ncee/wwc/PracticeGuide/1) supports combining verbal/graphical representations, connecting concrete and abstract ideas, alternating worked examples and practice, retrieval and explanatory questions, with differing evidence ratings. [PhET's research](https://phet.colorado.edu/en/research) informs focused investigations and iterative learner walkthroughs; its findings do not prove that a simulation replaces equipment skills or that an unguided lab establishes mastery. The domain recipes above are this project's design synthesis, not externally validated curricula.
