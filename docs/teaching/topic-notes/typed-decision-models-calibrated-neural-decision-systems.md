# Authoring notes: Typed Decision Models & Calibrated Neural Decision Systems

Canonical topic ID: typed-decision-models-calibrated-neural-decision-systems

## 2026-09-22 — Connect probability-reporting models to reproducible research

- Status: open
- Origin: user-requested Jev/Laya coverage assessment and [curriculum/project plan](../../curriculum/TYPED-DECISION-MODELS-PLAN.md).
- Destination and ownership rationale: this integrated topic connects encoder representations, typed option heads, scoring objectives, calibration and cost-sensitive action. Earlier owners retain their general mathematical and transformer implementations; the project owns the complete build workflow.
- Idea and learning benefit: learners should construct and modify the mechanism from scratch, connect it to ordinary library use, and distinguish model outputs from evidence justifying a real action.
- Existing coverage: planned here in the [authored blueprint](../../../src/learn/data/curriculum/blueprints/typed-decision-models-calibrated-neural-decision-systems.js); not a manuscript or implementation. Existing Decision Theory and Calibration lessons supply reusable foundations, not a completed integrated typed-model lesson.
- Proposed treatment: follow the blueprint and plan, using an inspectable shared task through architecture, objective, calibration and deployment. Include direct supervised optimization as a baseline before the policy-estimator extension. Add exact prerequisite program links during writing and a library-to-scratch ownership map.
- Explanation/example: a document-routing question produces probabilities over caller-provided candidate descriptions. A changed cost table can change the action without changing the distribution; candidate or context changes can change the distribution itself. Use these as separate live investigations, with untouched test evidence separate from exploratory inputs.
- Prerequisites and boundaries: do not reimplement whole prerequisite systems gratuitously. Do not assume Jev's internals are public or that schema-valid outputs are correct. Preserve train/model-selection/calibration/test roles and distinguish calibrated probability from entropy concentration. The linked project is `/learn/projects/typed-decision-model`.
- Evidence: source URLs, version caveats and inspection limits are recorded in the plan. Recheck current Laya Router/Agent APIs and pin a checkpoint/source revision before writing runnable programs.
- Resolution: planned ownership accepted for the new catalogue addition; research/write and implementation remain not started. Reassess current sources and exact earlier code when authoring.
- Implementation/verification links: no lesson implementation yet; curriculum conservation is performed by the integrating task.
