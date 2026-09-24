# Typed decision model project — content and evidence

Authoring date: 22 September 2026. Authorized scope: first end-to-end research project for the redesigned learning hub. This is a **project build guide**, separate from a canonical depth topic and its lesson-delivery phases. The companion topic is `typed-decision-models-calibrated-neural-decision-systems`; its planned status must remain honest.

## Current revision 2 — explained end-to-end build

The user's follow-up requested the intended explanation depth inside the project and relevant topic links where the mechanisms are used. All eight stages now follow the same refund-routing example through representation, batching and masks, attention and marker gathering, gradients and optimization, calibration, diagnostic interpretation and artifact restoration. Worked arithmetic, code walkthroughs and solved modification exercises supplement the canonical source disclosures. Thirty-one contextual links point to 24 distinct curriculum topics with a reason to follow each; three of those topics are honestly labeled planned outlines.

The guide now has four live investigations: the retained probability/cost explorer, an exact native-tokenizer trace, a constructed masked-attention mixture, and a shared-scoring-head update. These have different representations for different mechanisms. The latter two explain their illustrative/frozen-feature scope and do not imply that a browser is running the trained transformer. Controls update results directly, with no learner-prediction gate.

`project-elements.jsx` owns on-demand canonical source views and contextual links; `mechanism-depth.jsx` and `research-depth.jsx` own the extended reading flow; `mechanism-models.js` and `mechanism-labs.jsx` own the new live investigations. The original `typed_decision.py` remains unchanged. New `research_tools.py` imports it to provide exact traces and executable diagnostics. New `pretrained_decision.py` implements the ordinary Transformers adapter with tokenizer/marker setup, training, calibration, saving and inference. The [native author record](typed-decision-model-native-depth.md) distinguishes executed offline random-encoder checks from the still-unexecuted meaningful pretrained experiment.

The actual unfavorable findings are part of the teaching: confident errors, a harmful calibration fit, realized action costs worse than review-all on this fixture, and strong candidate-order sensitivity despite training-time shuffling. They are supported by measured diagnostics, not invented success examples.

Revision 2 is independently reviewed and integrated: [depth review](typed-decision-model-depth-review.md), [final integration and evidence](TYPED-DECISION-DEPTH-INTEGRATION.md), and [delivery checkpoint](project-delivery-progress.json). User review remains pending. The sections below preserve the original revision-1 scope and evidence; their former outline-only pretrained bridge and single-lab description are superseded by this revision. The companion topic and all lesson delivery counts remain unchanged.

## Original revision 1 record

## Contract and interface

- Stable project ID: `typed-decision-model`.
- Metadata: `src/learn/data/projects/typed-decision-model/metadata.js`.
- Stage map: default export of `content.jsx`, keys `define`, `data`, `baseline`, `architecture`, `training`, `calibration`, `evaluation`, `serving`.
- Pure live model: `decision-model.js`; styles scoped to `.tdp-*`.
- Executable/downloads: `public/learn-projects/typed-decision-model/`.
- Parent reader owns headings, stage navigation, stage progress and prerequisite display. Progress means a learner marked a deliverable, not an independently validated skill or a trained model.

The project promises an original tiny trained decision transformer, explicit costs, an artifact and a research report. It does not promise a Jev reproduction, a trained Laya checkpoint, broad language understanding or a deployed endpoint. The live lab is clearly labelled editable illustrative logits; it runs no language model. It updates probabilities, entropy concentration, expected action cost and policy choice immediately when logits, temperature, candidate count or costs change. There is no learner-prediction feature.

## Learning coverage

| Stage | Mechanism and learner evidence |
| --- | --- |
| Define | Distinguish schema validity, probability calibration and actions; write a task/cost contract. |
| Data | Bind target to current candidate ordering, document ambiguity, preserve grouped splits, distinguish multi-class and multi-label. |
| Baseline | Install/run the program; inspect overlap scoring and appropriate probability metrics. |
| Architecture | Follow marker positions and tensor shapes; inspect explicit attention, mask semantics and equivalent library route. |
| Training | Explain stable CE and gradient, candidate shuffling and saved trace; separate scoring-rule/RL research extensions from executed supervised code. |
| Calibration | Fit on disjoint examples, inspect boundary fits and test losses, derive the cost threshold. |
| Evaluation | Preserve an unfavorable baseline comparison, failure under paraphrase, ablation contract and measurement discipline. |
| Serving | Run saved-model JSON inference, understand validation and complete artifacts, scope pretrained/library/operational extensions. |

The small implementation intentionally owns its full contextual build while linking the existing attention, transformer, loss, calibration and decision-theory prerequisites. The scratch attention is vectorized PyTorch tensor code, uses stable softmax/logsumexp and explicit padding/candidate masks, and is checked against SDPA. Autograd, tensor primitives and AdamW are normal library tools. Large source views fetch the canonical downloadable program only when the learner opens them; there are no manually copied long program variants.

## Source review and accuracy boundaries

Primary sources inspected during this project and its preceding coverage assessment:

- https://docs.typesafe.ai/introduction — Jev's typed question interface. It does not disclose enough internals to establish an exact architectural match with Laya.
- https://typesafe.ai/blog/introducing-system-one-models-and-jev — dated Jev announcement and stated training/interface scope. Vendor performance and “no hallucination” framing are not treated as independent guarantees.
- https://github.com/NandhaKishorM/laya/blob/main/laya/common.py — public sequence construction, candidate scoring, masks and confidence implementation. No code copied into the original tiny model.
- https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb — one concrete public training workflow; optional policy-gradient extension remains unexecuted here.
- https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html — ordinary library API; keep boolean mask convention explicit.
- https://huggingface.co/docs/transformers/model_doc/modernbert — pretrained-encoder API bridge and tokenizer/configuration compatibility.

These are mutable public URLs, not pinned claims of reproducibility of upstream weights. The supplied original reference program has no remote dependency at runtime beyond the installed PyTorch package. No unverified provider benchmark numbers are displayed.

## Actual native verification

Executed locally with Python from `scratch/lesson-tools`, `-X utf8 -B`, PyTorch `2.14.0+cpu`, one CPU thread:

1. `typed_decision.py verify` passed explicit/manual attention versus SDPA equality; invalid-candidate zero probability; padding invariance; scratch CE versus native CE; stable softmax under a common shift; finite parameter gradients; act/review cost cases; split sentence disjointness; token-budget rejection.
2. JSX parsed successfully using the installed Babel parser.
3. Training and saved-checkpoint JSON inference ran successfully. The final displayed run is 600 steps, seed 7, 30,977 parameters. The learner-facing `verified-report.json` records the complete measured trace and metrics.
4. A preliminary 120-step run stayed near chance; its training trace motivated a bounded longer optimization run. The 600-step model fits training but loses to the lexical baseline on test (12/18 vs 18/18). Temperature fitting selects the lower grid boundary, and test NLL worsens from 2.047912 to 39.975479. Content explicitly reports this failure. We did not tune the data or hide unsuccessful results to manufacture a win.
5. Sample saved-artifact inference returns valid probabilities and an explicit decision; the resulting sharp confidence is not claimed trustworthy. Calibration failure is explained before serving.

Independent review found and prompted two corrections: the p=0.9, wrong-cost=10, review-cost=1 floating-point boundary previously acted instead of reviewing; both Python and JS now use an explicit shared 1e-12 scaled tie tolerance, with exact and either-side checks. The ASCII-word tokenizer's scope is now stated, and question/state/descriptions without supported a-z tokens are rejected instead of accepting empty evidence. These changes do not alter the recorded trained weights or metrics. Parent integration owns final browser/keyboard/responsive/build verification and will append exact results. No checkpoints, virtual environments, dataset downloads or GPU models need to be shipped with the source. The two temporary run directories are removable after retaining the final report and source evidence.

The [independent review](typed-decision-model-review.md) rechecked and closed both findings and records additional gradient-parity, corpus and probability-model checks against final source hashes. The final author source identity and native scope are available in `public/learn-projects/typed-decision-model/verification.json`. Both task-owned scratch run directories were removed after preserving the measured report; no unrelated working files were cleaned.

## Acceptance and future author rules

Final production/browser integration is complete: see [the integration record](WORKSPACE-INTEGRATION.md) and [project delivery checkpoint](project-delivery-progress.json). All eight stages, responsive layouts, live controls, progress, source disclosures and loading recovery passed the scoped checks. The user-requested neutral/amber palette supersedes the initial project CSS colour identity; mathematical sources and measured reports are unchanged. The earlier parent-integration handoff above is closed.

The published build guide is distinct from completion of any learner's experiment. Future additions must preserve observed unfavorable outcomes, distinguish constructed fixtures from real data, provide ordinary library control alongside scratch mechanisms, make the data/model/calibration versions travel together, and describe execution limits once at the right location. Schema guarantees must not become factual correctness claims. Expected-cost estimates require suitable probability evidence and stated loss assumptions.

Pretrained fine-tuning, other question types, policy-gradient training, a network API and real support-data evaluation remain optional extensions, explicitly labelled. Implementing them later requires real native and integration evidence; this project's runnable core is not evidence that they have been completed.
