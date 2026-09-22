# Typed decision models: curriculum and project connection

Planned on 22 September 2026, following the user's request for a research-oriented learning section with separate end-to-end projects. This is a scoped curriculum addition and authoring brief, not a completed manuscript, published lesson, implementation checkpoint or claim of model reproduction.

## Placement and identity

- Topic: **Typed Decision Models & Calibrated Neural Decision Systems**.
- Stable ID: `typed-decision-models-calibrated-neural-decision-systems`.
- Module: `large-language-models`.
- Section: **Post-Training & Alignment**, after **RL for Reasoning (DeepSeek-R1 Style)** and before the next section, **Inference Optimization**.
- Level/depth: advanced / specialist.
- Project: [Build a typed decision model](/learn/projects/typed-decision-model), project ID `typed-decision-model`.

This placement integrates an encoder-based decision architecture with training objectives and empirical decision quality. It does not imply that the architecture generates language autoregressively or that every system of this kind is a large generative model. A new one-topic module or a second copy in the BERT module would fragment ownership. Existing section and topic order is preserved; prerequisite review links may point into other modules without reordering the learner's chosen module.

The topic owns mechanisms, derivations, explained scratch programs, the library bridge, and independently transferable practice. The project owns the integrated build sequence, repository artifacts, experimental protocol, operational interface and research report. Link directly to earlier teaching and actual reusable programs; do not repeat a full transformer, calibration or decision-theory lesson in the project. The `projectConnection` field is authoring metadata; the project UI uses its own compact related-topic mapping rather than shipping the full blueprint to the hub.

## Learning requirements

Use the current [teaching standard](../../LESSON-TEACHING-STANDARD.md), especially scratch implementation followed by control of ordinary libraries. The [authored blueprint](../../src/learn/data/curriculum/blueprints/typed-decision-models-calibrated-neural-decision-systems.js) is the concrete scope; its named catalogue subtopics are discoverable scope obligations, not proof of completed teaching.

1. Start from a decision with observable consequences. Explain state, question, candidate meaning, probability report and action before architecture terminology. Reuse the same small task through the head, training, calibration and cost analysis.
2. Build a small, inspectable dynamic candidate scorer. Reuse the existing transformer owner when available and identify the exact imports. Teach token/marker positions, masks, question-type conditioning and shared scoring; contrast fixed-class heads and constrained text generation. State assumptions under which candidate permutations should preserve the intended semantics, then test whether the model actually does so.
3. Derive the head and objective before the library call. Use stable normalization, deliberate shapes and bounded efficient computation. The scratch route must expose gradients and trainable state, not substitute a table of precomputed outputs for training. Explain where a small educational implementation differs from an optimized pretrained encoder.
4. Start with direct supervised optimization of an appropriate probability loss. Extend earlier proper-score teaching to spherical and ordinal ranked scores, including sign conventions, target distributions and category order. Explain the sample space and baseline of any policy-gradient variant; compare its variance and performance with the direct baseline. A named RL method alone is not evidence that reinforcement learning is necessary for the task.
5. Separate training, model selection, calibration and final evaluation with explicit identities. If size requires cross-fitting, derive the scheme and preserve each observation's role. Do not copy a public notebook's reused training/calibration rows into the recommended protocol without disclosure and repair.
6. Evaluate probability quality, task discrimination and action quality separately. Reuse calibration and decision-theory implementations; measure coverage/risk of abstention and a nonideal fallback. Include changed candidate counts, candidate order, distractors, omitted correct choices, paraphrases, context truncation and language/data shift where the chosen task supports them.
7. Bridge to the current pinned library and checkpoint: schema construction, tokenizer/preprocessing, masks, batched forward pass, typed postprocessing, calibration state, model routing and checkpoint replay. Show what an engineer/researcher changes and what a library owns. Measure cold/warm latency and memory under stated conditions rather than extrapolating a vendor number.
8. Preserve meaningful topic-specific visuals: token/marker lanes, distribution and ordinal-CDF comparisons, evidence-partition maps and cost/action regions. Labs show results on first paint and update live where cheap; expensive model execution is explicit and bounded. Never add learner-prediction fields or answer gates.
9. End with changed-data/changed-schema practice and an ablation plan with a falsifiable question. Link the project for integration rather than promising that one finite exercise supplies universal model-building mastery.

This is not a mandate to reproduce an undisclosed commercial system. Teach the public mechanisms and a defensible research workflow; distinguish an educational prototype, a checkpoint adaptation and a verified reproduction.

## Evidence and boundaries

Sources reviewed on 22 September 2026. Public `main` branches and service documentation can change; the author must pin the source/checkpoint/version actually used and revisit relevant claims during writing. Source inspection is not evidence that code was executed locally.

- [TypeSafe announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev) and [API introduction](https://docs.typesafe.ai/introduction): Jev exposes typed probabilistic decisions and describes an RLCD training goal. The inspected pages do not establish a reproducible layer-by-layer architecture or training objective. Similar interfaces do not establish shared internals with Laya. Correct output types do not guarantee correct real-world answers.
- [Laya public repository](https://github.com/NandhaKishorM/laya), [decision-head implementation](https://github.com/NandhaKishorM/laya/blob/main/laya/common.py), [inference implementation](https://github.com/NandhaKishorM/laya/blob/main/laya/agent.py) and [fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb): inspect actual code rather than deriving the implementation from a marketing label. Verify scorer masking, training estimator, score modifications and calibration partition roles. Current repository guidance includes a `Router` and multiple checkpoints, so an older single-agent example must not be presented as the only current library route. Pin actual API contracts before authoring runnable examples.
- [ModernBERT paper](https://arxiv.org/abs/2412.13663) and [mmBERT model card](https://huggingface.co/jhu-clsp/mmBERT-base): backbone detail belongs with the existing encoder owner; the integrated topic needs only the bridge and model-specific constraints. See the destination notes below.
- [Gneiting and Raftery on proper scoring rules](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) and [Guo et al. on neural calibration](https://arxiv.org/abs/1706.04599): teach the assumptions behind probability objectives and the evidence needed for calibration. A mathematically proper score does not guarantee calibrated finite-sample trained outputs; clipping or other implementation changes require separate analysis.

Treat released latency/benchmark tables as source-reported results with their hardware, sample and prompt conditions. Do not compare local and remote systems as a controlled head-to-head experiment without matching the protocol. Keep entropy concentration, a learned act/escalate score, maximum probability and measured probability of correctness distinct.

## Persistent destination notes

- [This topic's authoring notes](../teaching/topic-notes/typed-decision-models-calibrated-neural-decision-systems.md) retain the origin and phase boundary.
- [Contextual embeddings / BERT variants](../teaching/topic-notes/contextual-embeddings-elmo-bert-variants.md) owns the modern bidirectional encoder mechanism and library bridge.
- [BERT & T5 landmarks](../teaching/topic-notes/bert-t5-encoder-encoder-decoder-landmarks.md) owns historical/architectural comparison and links to the mechanism owner.
- [GRPO and related methods](../teaching/topic-notes/grpo-rloo-kto-advanced-preference-methods.md) owns policy-estimator distinctions; this new topic owns the detailed probability-reporting application.

No earlier topic is renamed or removed. Publication stays manifest-based; no manifest entry is added. Both delivery phases remain not started. The integrating task regenerates the deterministic navigation/outlines and inventory and performs the shared curriculum/build checks once the related UI changes are ready.

## Scoped source verification

The authoring-source comparison passed on 22 September 2026: all 1,460 previous stable IDs remain in their original relative order, every previous module/section/title sequence is identical after excluding the one intentional addition, and the catalogue now has 1,461 topics. All six prerequisite IDs resolve. Both `full-curriculum` and `llm-engineer` include the new topic. Its blueprint parses, and the normal topic preflight returns its destination note with publication `planned`, content `not-started`, implementation `not-started`, and `canFinish: false`. No phase ledger or lesson manifest was edited for this addition. These are source/planning checks, not browser or lesson implementation evidence.
