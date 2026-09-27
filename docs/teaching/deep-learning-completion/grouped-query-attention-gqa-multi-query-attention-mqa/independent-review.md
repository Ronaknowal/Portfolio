# Independent GQA/MQA review

Reviewer: memory/energy implementation agent, 27 September 2026. Source and scientific review passed after the corrections below. Root owns rendered/browser and final integration acceptance. This is an expert heuristic learning review, not an observed beginner study.

## Scientific review and independent trust roots

Read the complete prepared manuscript, visual specifications, design/provenance, both complete Python programs, runtime model, labs, diagrams, study history, generator and author check scripts. Checked the written calculation, every practice answer, dimensions, mean conversion, offset masks, loss averaging, split protocol and declared deployment assumptions. The construction and learned forecasting study are consistently distinguished.

`independent-native.py` uses a newly written torch functional forward based on `linear`, `layer_norm`, `gelu` and native SDPA. It does not import the author's CausalForecaster or use its einsum implementation. Nine new full-model cases cover all three deployed checkpoints and constant, alternating extreme-coordinate and curved prefixes. Browser outputs agree to maximum absolute error 2.79e-8; three-point cache chunks and five-step generated rollouts match full native recomputation. A separate random six-query/two-KV test with unequal key/value widths checks outputs and all Q/K/V gradients against explicit tied-head expansion. Large future keys/values leave the legal read unchanged. Earliest validation minima match selected checkpoints, and eight public reproduction files match canonical sources byte-for-byte. Forty checks passed; results retain actual environment and limits.

The new candidate-mean distance equations were independently inspected: sum over two originals equals the mean's minimum plus twice squared candidate displacement. The shared-value gradient is a sum of contributions, without a second group average. Exact byte arithmetic counts unique K/V fields; the distributed example explicitly counts physical replicas separately. No timing is inferred from payload arithmetic.

## Closed findings

1. `GroupedQueryLabs.jsx:GqaReadLab` originally omitted key/query ID and record-count controls required by F2. A separate fixed-vector mask lab could not test these changes on the learner's edited Q/K/V. It now supports both logical query rows, 1–8 records, editable IDs, complete-record permutation, logical/local mask contrast and an explicit undefined all-masked state. The read still computes from edited vectors.
2. `GqaConversionLab` originally showed only outputs, omitting F5's distinction between minimizing parameter distance and preserving prediction. It now shows key/value squared distance, its minimum/excess, per-head output delta and editable candidate displacement from the mean. The tie and reset controls also reset candidate displacement.
3. F1/F3/F4/F7 dependencies were overly dependent on prose/text inventories. `GroupedQueryDiagrams.jsx` adds known-prompt versus sequential-decode availability, unique-key rotation before compact append, the nested payload axes, logical-versus-physical replicated heads, source-fixed versus decoder-growing cache lifetime and attention-versus-expert-FFN ownership. The compact-write join's disconnected 16/20-pixel path gaps were closed at (550,179) with a visible join.
4. Root's readability/invalid-input findings are addressed in source by readable nominal figure widths with bounded focusable scrolling and the opt-in `NeuralNumberControl`. Root must confirm painted output and keyboard/error behavior after integration.

## Learning-experience checklist

| Item | Concrete review |
| --- | --- |
| Route | Manuscript opening and §7 label first-pass and deeper branches. Core path follows cache need, reader/group definition, exact read, storage, code, conversion and real forecast before deployment/practice. |
| Cautions | Each operational caveat has a useful home: masking in §5, evaluation limits beside §6 outcomes, bandwidth/replication in §7. Programs print measured quantities and assertion outputs, not caution paragraphs. |
| Real question | §1 asks how successive predictions can reuse their history; §6 returns to a genuine UCI trajectory forecast. Labels determine split strata only; targets are future coordinates. |
| Live investigations | `GqaReadLab` edits actual vectors/IDs/records; `GqaBudgetLab` changes all payload axes; `GqaMaskLab` isolates legal history; `GqaConversionLab` separates distance and output; `GqaForecastLab` edits actual observed coordinates and generates from its own outputs; `GqaGradientFigure` exposes signed sum-of-use. All show current results without learner prediction gates. |
| Figures | Head-to-group edges differ from token attention. Vector mixtures use signed coordinates; payload diagrams label schematic repetition; measured histories use log10 MSE with interpretation; generated and observed paths have distinct labels/styles. Source widths prevent deliberate label shrinkage. Painted readability remains root's browser check. |
| Connections | §3 repeated tied heads matches compact algebra; §5 NumPy maps to SDPA/gradient contracts; §6 parameter mean is separated from nonlinear behavior; §7 connects memory capacity, kernels, allocation, precision and device replication without multiplying unsupported speed claims. |
| Code | Complete NumPy reference, native SDPA call-site and downloadable full model/training program remain locally explained. The model's input, shapes, projections, rotary convention, residual/FFN/readout, cache and rollout are all inspectable. Validation is small and subordinate to mechanism. |
| Practice | Nine independent changed-value exercises cover 12/3 routing, two-query shared values, unequal-width binary MiB count, offset masks, fair conversion, signed gradient, assumed timing, replicas and rollout leakage. Hints/solutions are closed and arithmetic is correct. |
| Screenshots | Not claimed by this reviewer. Root owns informative desktop/phone screenshots, edited/null/error states and full-width figure inspection. |
| Buildup throughout | §1 persisted projections/availability → cache timeline/prompt diagram; §2 fixed routing versus data-dependent read → wiring; §3 dot/softmax/value sum → worked four-reader example and vector mixture; §4 compact tensor fields → payload/rotation/mask diagrams; §5 repeated/direct implementations → full NumPy and native comparison; §6 mean minimizer versus nonlinear output → changed candidate and measured recovery; §6 causal one-step versus generated rollout → observed boundary, native cached/full equality and actual trajectories; §7 sum-of-use, restricted value span and replica layout → signed gradient, convex-combination argument and eight-device geometry. |

The stable title/scope, all original outcomes and both scratch/library paths are retained. Optional multimodal applications remain prose because their scope is to locate the position/cache contract, not teach a new multimodal architecture. No additional benchmark, GPU implementation or browser training is implied.

## Remaining checks

Browser/painted accessibility and integration remain with root, including the new diagrams, updated editable-memory controls, model-load failure handling, invalid-input reset, 320/390-pixel local scroll and neutral styling. Numerical and manuscript review is closed for the exact hashes in `independent-checks.json`.
