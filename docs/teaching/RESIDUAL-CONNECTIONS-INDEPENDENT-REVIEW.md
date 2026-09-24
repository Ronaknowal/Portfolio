# Residual Connections: independent implementation review

Reviewer: `/root/decision_depth_prose`, separate from author `/root/residual_implementation`. Reviewed 22 September 2026. Topic: `residual-connections-skip-connections`. **Source/content and complementary native review passed with no material findings; production browser and visual closure remain the parent integrator's responsibility.** This is not user acceptance or an all-site certification.

## Scope and evidence

Read the complete revision-three manuscript, visual specifications, design/implementation record, canonical Python program, runtime mechanism model, live components, stylesheet, manuscript-to-JSX mapping, blueprint and author evidence. The full reader source was assessed against those inputs, including its 13 sections, seven independent exercises with separate hints/solutions, and scalar-to-feature-gate coding extension. The parent will bind the final build and actually rendered desktop/phone states separately.

- [Reviewed-source hashes and disposition](evidence/residual-connections-independent-review.json).
- [Complementary native checks](evidence/residual-connections-independent-native.json), reproducible with `scratch/lesson-tools/Scripts/python.exe -X utf8 -B scripts/verify-residual-independent.py`.
- [Author numerical evidence](evidence/residual-connections-author.json), reused for the complete 39-fit replay, saved outputs, pure-model fixtures and finite differences. The reviewer did not repeat the training campaign or relabel author tests as independent work.

## Correctness and implementation depth

The explicit scratch boundary is the residual connection, shape/projection contract, gating and their derivatives. Earlier owners supply linear layers, normalization and differentiation. The local canonical program composes ordinary `nn.Module` layers, explicitly returns the same routing equation and executes an actual fitting/evaluation loop. The visible `Refinement.forward` excerpt and following explanation connect each operation, buffer/parameter distinction and optimizer step to that mechanism. The whole program and data are downloadable, with source fetched on demand rather than embedded in every initial page request.

The arithmetic was checked throughout the reading flow: the opening correction/update and signed input gradients, cancellation and ordered Jacobian product, LayerNorm's common-shift null direction, projection transpose, zero-last-linear versus zero-last-ReLU gradients, scalar/channel gate distinction, nonlinear path counterexample, denoising subtraction and Euler stability interval. No universal gradient guarantee, exact independent-model ensemble, invented removal robustness or unsupported speed claim replaces those mechanisms. Real fit and deletion claims stay tied to their actual seed, depth, architecture and split; validation is not presented as a final test estimate.

The additional native review passed five targeted groups:

1. Used changed two-row inputs, nonzero corrections in both features and three initial scales to verify scalar versus feature gate gradients, batch summation and independent finite differences. This checks a risk hidden by the manuscript fixture's zero first correction.
2. Executed an actual zero-gate block's identity Jacobian, first backward/SGD step and next backward pass. Branch matrix gradients are initially zero; the gate gradient is nonzero for the chosen changed example; branch gradients become nonzero after that real gate update.
3. Confirmed fixed scales are saved buffers, trainable zero gates are parameters, and Adam leaves fixed buffers unchanged. This checks the explanation's actual API contract.
4. Replaced the final correction map with a broadcastable width-one map and confirmed the canonical shape guard fails. The check exercises its failure path; matching dimensions still cannot prove semantic feature alignment.
5. Confirmed same-seed shared branch prefixes/stem/head match across configurations, and score/diagnostic calls neither accumulate parameter gradients nor mutate model state in this LayerNorm experiment.

Primary-source spot checks covered the actual [Torchvision 0.26 BasicBlock/Bottleneck forward functions](https://docs.pytorch.org/vision/0.26/_modules/torchvision/models/resnet.html) and [PyTorch 2.14 checkpoint contract](https://docs.pytorch.org/docs/2.14/checkpoint.html). The existing packet's documented paper-review extent remains authoritative for its broader historical coverage; this review does not claim another exhaustive paper reading or a new CNN benchmark.

## Learning-experience review

This is an independent source-based teaching assessment, not an observed beginner session. It separately considered the current teaching checklist:

| Question | Concrete assessment |
| --- | --- |
| Route and depth | The opening explains keeping a representation and learning its correction before notation. The first-pass route leads through a two-value example, backward paths, operation order, shape and real training. Scaling, family connections and Euler are identified as deeper branches. |
| Cautions and load | Qualifications live next to the inference they constrain: cancellation, post-operation identity, recorded validation, deletion and memory. The code remains operation-focused; it does not print cautionary prose. No new warning blocks are needed. |
| A real question | The opening asks whether to replace or refine an existing representation. The actual digit experiment permits a useful counter-result: a smaller stem-only network can win, so depth is a decision to investigate rather than a promised improvement. |
| Live investigations | Source controls edit actual weights, slope, depth, operation placement, shift, projection entries, gate scale/shape and Euler step. Current results appear without prediction submission. Recorded fit/deletion selectors are accurately labeled. Changed and null cases have explicit meaning. |
| Inline visual reading | Direct and correction lanes make the first addition visible; signed gradient contributions expose cancellation at its introduction. The operation-location diagram precedes the architecture discussion. Projection sockets show a genuine incompatible state. Zero-branch versus zero-gate behavior appears before measured fitting. Nonlinear composition and denoising retain separate visible examples. These surfaces serve different mechanisms rather than one interchangeable output box. |
| Quantitative representations | Exact tables accompany signed derivatives, gains and chart points. Training/validation CE uses its stated units and is distinct from correct counts. Recorded curves connect saved points without claiming intermediate measurements. The Euler plot preserves negative values and zero. Actual readability at final rendered sizes is still a browser-review obligation. |
| Connections and ownership | Initialization, backpropagation, normalization and the following dropout topic are explicitly connected. CNN/Transformer/DnCNN/Neural-ODE connections are scoped explanations; the page does not claim trained reproductions. |
| Code and control | The learner can identify and change the exact branch return expression, independently alter scalar to feature gating, use ordinary PyTorch optimizers/modes, and reproduce the full measured route. The transparent implementation does not require duplicating earlier matrix/autograd engines. |
| Practice | Tasks change weights, gradient gain, projection inputs, feature count, deletion interpretation and Euler step. The additional gate extension requires an actual parameter-shape change and separate gradient/output checks. Hints and complete solutions remain separate disclosures. |
| Screenshots and actual operation | Not performed by this reviewer. Parent must run the scoped browser script, check pointer/keyboard/invalid/reset behavior and inspect informative final desktop/phone images before marking implementation complete. |

No blocking source/content issue was found, and no author source was changed by the reviewer. Re-review is needed only for material later changes or a browser finding affecting these conclusions; unchanged native and full-fit evidence can be reused.

Final author polish was checked before integration: number fields restore their declared last-valid value on blur (including when a reset leaves the underlying number unchanged), and an owned stylesheet rule replaces the reused source viewer's older green tint with charcoal, neutral borders and text. The final review hash includes that stylesheet. Numerical sources and the native conclusions are unchanged; actual pointer/reset and final theme appearance still belong to browser closure.
