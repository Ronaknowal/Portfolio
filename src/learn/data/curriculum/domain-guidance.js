// Shared authoring strategies. These do not masquerade as bespoke topic plans.
// Detailed methods: docs/teaching/DOMAIN-PLAYBOOK.md.
const strategies = {
  mathematics: {
    flow: ["Pose a concrete question and identify known/unknown quantities", "Link a small numerical case, a representation and defined notation", "Derive the mechanism with justified intermediate steps", "Interpret a complete worked result", "Test a changed case, assumption and counterexample", "Connect to a formal or applied deeper branch"],
    visuals: ["Linked geometric and symbolic views", "Labeled distributions and repeated samples", "Stepwise derivation with exact-value inspection"],
    practice: "Hand calculation, explanation of a representation, flawed reasoning to repair, and independent transfer with an explained solution.",
    verification: "Check assumptions, conventions, proof steps, units, numerical approximations and independent reference values.",
  },
  algorithms: {
    flow: ["Specify inputs, outputs and a small task", "Establish a simple correct method and its cost", "Expose the data structure and trace its changing state", "State and justify an invariant and termination", "Analyze resources under a stated cost model", "Implement, diagnose edge cases and transfer to a new constraint"],
    visuals: ["Execution trace linked to code and state", "Pointer/tree/frontier diagram", "Same-input method comparison"],
    practice: "Predict a next state, implement from an invariant, construct a counterexample and compare time/space tradeoffs.",
    verification: "Correctness argument plus independent/oracle outputs for meaningful boundary and adversarial cases; measurements do not prove asymptotics.",
  },
  programming: {
    flow: ["Introduce a practical task and the meaning of inputs", "Name the language/runtime entities before syntax", "Trace execution alongside state and output", "Run a complete small example with explained setup", "Investigate an edge case and repair a failure", "Complete a changed-input task independently"],
    visuals: ["Names, objects and reference arrows", "Data shape and row/cell correspondence", "State, stream and dependency flows"],
    practice: "Prediction, guided variation, diagnosis and a complete independent task with fixtures, expected results, hints and explained solutions.",
    verification: "Execute in the stated language/library/environment; compare browser models against actual behavior and label their limits.",
  },
  models: {
    flow: ["Define task, data unit and a baseline", "Establish splits and what information is available", "Trace a tiny example through representations and computation", "Connect objective, parameter update and inference", "Evaluate a failure and a controlled comparison", "Apply independently and explain assumptions and limits"],
    visuals: ["Data and tensor-shape flow", "Parameter-to-prediction experiment", "Decision regions, latent spaces or sampled trajectories"],
    practice: "Manual calculation, reproducible small implementation, diagnosed failure, and an ablation or changed-data experiment.",
    verification: "Check leakage, shapes, objective, baseline, metrics, seeds, uncertainty and realistic compute requirements; no invented measurements.",
  },
  systems: {
    flow: ["Define a computation or service contract", "Map components, state and resource boundaries", "Trace the normal request/work/data flow", "Explain coordination and the relevant failure mode", "Measure correctness and a diagnosed bottleneck", "Repair, compare and validate under a changed workload"],
    visuals: ["Worker/memory mapping", "Causal timelines and synchronization", "Profiles, resource budgets and communication flows"],
    practice: "Independent reference computation, fault diagnosis, measured optimization and a reproducible engineering report.",
    verification: "Separate output correctness, memory/concurrency safety, numerical tolerance, performance and platform support.",
  },
  evaluation: {
    flow: ["State the construct and decision the evidence will support", "Define sample, unit, rubric and baseline", "Trace an item through annotation and scoring", "Estimate uncertainty and examine validity threats", "Audit a plausible misleading result", "Design a changed evaluation and justify its decision limits"],
    visuals: ["Item-to-score trace", "Sampling and uncertainty views", "Matched comparisons and error breakdowns"],
    practice: "Repair a misleading metric or rubric, compute a small example, audit leakage/bias and design a discriminating held-out test.",
    verification: "Check construct validity, sampling, dependence, calibration, uncertainty, benchmark contamination, cost and scope of conclusions.",
  },
  embodied: {
    flow: ["Specify task, environment and observable success", "Introduce frames, units, state, observations and actions", "Trace sensing through estimation and decision to actuation", "Explain feedback, delay, noise and constraints", "Compare trajectories and diagnose a failure", "Validate in changed environments and bound physical claims"],
    visuals: ["Closed-loop system", "Frames linked to trajectories", "Time response with uncertainty and constraints"],
    practice: "Predict response to a change, repair a frame/delay error, reproduce a bounded simulation and evaluate transfer.",
    verification: "Record seeds, timestep, frames, units, actuator limits and termination rules; distinguish simulation from physical evidence.",
  },
  neural: {
    flow: ["Start from a biological question and measurable quantity", "Connect biological mechanism to sensor/interface and units", "Trace acquisition, reference, sampling, noise and artifacts", "Build and validate an analysis, decoder or controller", "Interpret behavioral/clinical evidence and uncertainty", "Compare conditions and bound translation and safety claims"],
    visuals: ["Biophysics and signal-chain diagrams", "Raw/processed traces, rasters and latent states", "Closed-loop timing and outcome views"],
    practice: "Synthetic signals, appropriate open de-identified data, bounded simulations or bench phantoms; independent analysis and failure diagnosis.",
    verification: "Check measurement assumptions, participant/session splits, confounds, nonstationarity, reproducibility and jurisdiction-specific translational evidence.",
  },
  research: {
    flow: ["Ask a precise research question and establish prerequisites", "Explain the baseline and proposed mechanism", "Reconstruct a small example or experiment", "Locate evidence and check assumptions", "Attempt a bounded reproduction and comparison", "Identify uncertainty, limitations and a concrete next investigation"],
    visuals: ["Mechanism diagram tied to equations/data", "Comparable experimental outcomes", "Evidence and assumption maps"],
    practice: "Reconstruct a result, evaluate an alternative explanation and produce a bounded reproducibility report.",
    verification: "Use original papers/data/docs; distinguish hypothesis, demonstration, generalization and unsettled claims; date moving information.",
  },
  reference: {
    flow: ["Give plain meaning and why the term/tool appears", "Disambiguate related concepts", "Show a miniature example or coherent workflow", "Explain a common misuse", "Link prerequisites and the full mechanism lesson"],
    visuals: ["Small comparison or mapping", "Workflow view only when useful"],
    practice: "Interpret a new occurrence, choose the correct term/tool and explain the distinction.",
    verification: "Verify terminology and current interfaces; never treat a glossary definition as a full mechanism lesson.",
  },
};

// The examples are module anchors. Authors choose a topic-appropriate concrete
// example; they are not fabricated individual plans for every existing title.
const modules = {
  "math-foundations": ["mathematics", "Small vectors, measurements and uncertainty; connect exact operations to an interpreted decision."],
  "data-structures-algorithms": ["algorithms", "Scheduling jobs, finding routes and maintaining changing collections under explicit constraints."],
  "programming-scientific-computing": ["programming", "A small measurement project from files and records through transformations to a reproducible report."],
  "classical-ml": ["models", "A small tabular prediction problem with a baseline, held-out evaluation and inspectable errors."],
  "deep-learning-fundamentals": ["models", "One example and one parameter update before scaling to a trained neural model."],
  "large-language-models": ["models", "One request from tokens and context through computation, output, evidence and serving behavior."],
  "generative-models": ["models", "A tiny data distribution and comparable samples under explicit training and sampling rules."],
  "nlp-cv-multimodal": ["models", "Trace a labeled text, image or audio sample through representation and task-specific evaluation."],
  "self-supervised-learning": ["models", "Compare views, representation objectives, collapse risks and downstream transfer under controlled data."],
  "meta-learning": ["models", "Separate tasks, support/query data, inner adaptation and outer learning before comparing baselines."],
  "reinforcement-learning": ["embodied", "A small state/action environment with visible reward, return, exploration and termination."],
  "robotics-embodied-ai": ["embodied", "A robot senses, estimates, plans and acts under frames, delay, noise and physical limits."],
  "drosophila-fly-embodiment": ["embodied", "A bounded fly behavior linked to biomechanics, sensory feedback and biological measurements."],
  "computational-neuroscience": ["neural", "A biological signal passes through measurement, analysis and a bounded closed-loop interface."],
  "hardware-systems": ["systems", "A correct CPU operation becomes a verified and profiled kernel, then a reliable multi-device workflow."],
  "model-optimization": ["systems", "Compare the same model/task before and after compression using quality, memory and measured latency."],
  "mlops-infrastructure": ["systems", "Follow data/model/request versions through delivery, observation, failure and recovery."],
  "jax-ecosystem": ["programming", "Transform a pure array program, then inspect shapes, compilation, randomness and device placement."],
  "agents-tool-use": ["systems", "A bounded task with explicit state, tool contracts, permissions, retries, stop conditions and held-out evaluation."],
  "llm-evaluation": ["evaluation", "A decision-relevant assessment from construct and sample through scoring, uncertainty and an audit."],
  "ai-safety-alignment": ["evaluation", "Test a precise behavioral/safety claim, compare interventions and state evidence limits."],
  "evolutionary-algorithms": ["models", "Compare optimization on an inspectable landscape using equal evaluation budgets, seeds and constraints."],
  "quantitative-finance": ["research", "A temporally separated financial experiment with costs, risk, data provenance and leakage checks."],
  "quantum-ai": ["research", "A tiny circuit/state example separates ideal calculation, noisy execution and claims of advantage."],
  "frontier-research": ["research", "Reconstruct one narrow claim with a baseline and distinguish published evidence from speculation."],
  "landmark-models": ["research", "Explain the architectural change against its predecessor and inspect the original experimental evidence."],
  "core-frameworks": ["reference", "One complete task from setup and data to output, with concepts mapped to tested APIs."],
  "terminology-glossary": ["reference", "Plain meaning, disambiguation, one miniature example and a linked mechanism lesson."],
};

export function getDomainGuidance(trackId) {
  const entry = modules[trackId];
  if (!entry) throw new Error(`Missing domain guidance: ${trackId}`);
  const [strategy, exampleAnchor] = entry;
  return { strategy, ...strategies[strategy], exampleAnchor, playbook: "docs/teaching/DOMAIN-PLAYBOOK.md", nextDesignStep: "Adapt this domain strategy using the topic's scoped outcomes, exact prerequisites, concept-hurdle map, source checks, complete examples and individual visual/practice contracts." };
}
