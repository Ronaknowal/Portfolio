# Recurrent models: concept-level author review

Read the entire production lesson: ten numbered sections, complete code route, all eight exercises and advanced Jacobian/variant branches. The nine existing interactive investigations already give the core mechanisms strong support. New support addresses later conceptual transitions without displacing them. Canonical manuscript and generator are synchronized.

| Concept and location | Disposition |
| --- | --- |
| §1 ordered input, running state, input/output axes, position vs time | Retained: pen-path lab, three representation baselines, sequence arrangements and timestamp limitation. |
| §2 affine recurrence, shared parameters, BPTT, credit accumulation | Retained: full three-step values, loss, per-use gradients, actual successful update and editable scalar trace. |
| §2 saturation, contraction/explosion and clipping limits | Retained: weight 2 at saturated preactivation counterexample and explicit product reasoning. |
| §3 LSTM cell/readout and gates | Retained: four intermediate arithmetic terms, native retain/write/read investigation and gate-affine equations. |
| §3 fixed-gate retention / bias | Retained: exact half-life table, adjustable retention lab and distinction from total gradients. |
| §4 GRU blend / reset placement | Retained two-coordinate reset-placement counterexample and live investigation; added convex-range invariant contrasted with LSTM's independent retain/write coefficients. |
| §5 actual data / matched experiment / reversal | Retained: completed-trace preprocessing contract, writer split, nine fits, baselines, measured live edits and no unmeasured streaming claim. |
| §6 carry/reset/detach, state ownership and cache invalidation | Retained: numerical/gradient boundary lab and session-identity examples. |
| §6 clipping | Retained: exact [3,4] scaling and comparison with coordinate clipping. |
| §7 lengths / backward padding / final state | Retained packed API program and live padding; added four-position dependency table to expose why the last output lacks the complete backward history. |
| §7 stacking, direction availability and loss mask | Retained: next-layer input width, causal contract and nine-valid-target denominator. |
| Scratch/library and implementation task | Retained explicit NumPy gates, native tensor ordering/parity, shared-gradient code and chunk exercise. |
| §8 cost and sequential dependence | Retained exact counts, bias conventions and activation/state memory distinction. |
| §8 full two-state Jacobian | Fixed gap: explicit c→c→c and c→h→c path products explain why the forget term alone is insufficient. The repeated rounded matrix is declared constructed, not a measured trajectory. |
| §8 eigenvalues vs products | Fixed gap: same-scale vector progression e₂→2e₁→4e₂ and direction explanation after the original A/B matrices. |
| §8 initialization/dropout | Retained orthogonality caveat, effective two-bias rule and eval vs no_grad. |
| §8 peephole/projection variants | Fixed name-only transition: cell-to-gate numerical example and explicit cell32/hidden12 projection shapes, parameter and head counts. |
| §9–10 practice and references | Retained all worked exercises and primary/author alternate routes. |

## Research and checks

Read the [PyTorch LSTM shape/projection/endpoint documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html), parameter-shape and direction notes. Read [D2L Deep RNN](https://d2l.ai/chapter_recurrent-modern/deep-rnn.html), opening stacked-state explanation and scratch construction, to assess whether existing layer/time transitions needed replacement (retained). Numerical variants and counterexamples are independently constructed; no new measured performance claim.

`RecurrentDirectionProductFigure` uses the same 20-pixel axis unit in each stage and responsive stacked semantic panels. Text supplies exact values and distinguishes forward sensitivity from reverse credit. Existing nine labs remain unchanged.

`node scripts/verify-structure-learning-intuition.mjs rnns-lstms-grus` checks five arithmetic groups and parses changed JSX, plus preserves four runtime/result identities. An initial rounded path-sum typo was caught by the author check and corrected before receipt. No models were retrained. Independent reading, browser viewports and production integration remain pending with root.
