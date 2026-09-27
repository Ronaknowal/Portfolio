# Transfer learning: concept-level intuition review

26 September 2026. Read the complete 12-section production lesson, every code block/practice solution and relevant reuse, freeze, LoRA, adapter, evidence and checkpoint labs. Production JSX is the full-mode manuscript. Independent and rendered review pending.

| Learning transition | Disposition |
| --- | --- |
| Opening and backbone/head | Removed the premature control catalogue. Retained the 0–4→5–9 motivating problem, feature/head definitions and exact dimensions/label-meaning visual. |
| Probe failure versus information loss | Added four-row sign-agreement geometry: all information present but not linearly separable, versus first-coordinate-only features with conflicting labels. Makes two explanations of a weak probe distinguishable. |
| Domain/task and coadaptation | Existing concrete domains, target meanings and experimental boundaries retained. New probe diagram supplements the measuring-instrument analogy. |
| Adaptation strategy and negative transfer | Existing restrictions table, same-target comparison and train/validation interpretation already connect the choices; retained. |
| Three freeze meanings and upstream derivatives | Existing live switches expose optimizer ownership, grad recording, buffers and mode independently; retained with explicit normalization prerequisite. |
| Complete pipeline and data partitions | All actual rows, code, matched initializations, rates, selection and one-test rules retained. |
| LoRA rank / measurements / output span | Added exact rank-one direction diagram for three inputs and a fixed factor pair. Distinguishes restricted correction span from unrestricted base output. |
| LoRA initialization / gradient / update / merge | Existing one-step calculation, factor-edit lab, complete manual/library bridge and measured merge errors already teach these; retained. |
| Adapter bottleneck and nonlinearity | Existing lab explains ownership/budget but lacks a numerical feature transformation. Added (1,2)→difference −1→tanh→two-coordinate correction→residual output trace. Shows why merging is generally unavailable. |
| Frozen values versus old behavior | Existing explicit derivatives and measured retention counterexample already show the distinction; retained. |
| Real comparison / selection / uncertainty | All measured runs, exact denominators, test gap and honest interpretation retained. |
| Checkpoint meaning / cache / serving | Existing contract lab and invalid-cache examples already expose these dependencies; retained. |
| Schedule / gradual unfreezing | Existing calculated schedule figure and optimizer-state policy retained. |
| Optional efficient-method families | Added concrete diagonal feature-scaling contrast and separated quantized storage from restricted update directions. These remain optional bridges; later language-model-specific lessons own their complete architectures. |
| Memory units | Existing per-parameter byte budget and omitted-state boundary retained. |
| Practice / sources / progression | All six changed-input exercises and manual implementation extension retained. |

## Research and representations

Consulted the original LoRA paper (https://arxiv.org/html/2106.09685v2) and adapter paper (https://arxiv.org/html/1902.00751v2), and the fixed-feature/head replacement part of the official PyTorch transfer tutorial (https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). The new constructions derive directly from the already stated factor and adapter formulas; no copied benchmark or video-watch claim. Retain the lesson's train/eval distinction rather than assuming a tutorial's weight freeze freezes all state.

New topic-owned static representations use different forms for different questions: 2D class geometry plus collapsed-feature columns; actual collinear output corrections; a numerical nonlinear detour reunited with the direct feature path. They sit at their concept's transition. Exact values/labels supplement shape and color. Columns stack on narrow viewports; SVG labels must be inspected at phone width. Existing playable labs remain the investigations; no prediction gate or imitation controls.

## Evidence

scripts/verify-transfer-intuition.mjs checks conflicting projection labels, linear-separation impossibility via pair-average identities, factor corrections, nonlinear adapter arithmetic, parsing and unchanged native sources. Original measured fits, mechanisms and downloaded programs remain unchanged. Independent full-reading review and desktop/narrow browser verification are pending.
