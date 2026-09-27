# State-space models: whole-lesson concept-transition review

26 September 2026. Read the full learner JSX, including all advanced disclosures, complete implementation explanations, nine solved practices and resource annotations. The previous revision already introduces each major architecture through mechanisms. Two advanced transitions benefit from small explicit calculations.

## Canonical ownership

This directory's `lesson.md` is the active manuscript rendered by `scripts/generate-state-space-lesson.mjs` into the stable topic JSX. Original revision-4 manuscript/evidence remain historical and unchanged. Every code fence is retained verbatim. Live numerical engines, Python programs, datasets, weights and measured studies are unchanged; their existing evidence is retained rather than relabeled as a new run.

| Concept / exact manuscript location | Finding and action | Representation and transfer evidence |
| --- | --- | --- |
| §1 state, recurrence and A/B/C/D | Sufficient 80/20 running summary before formal definitions, shapes and update-before-output indexing; retained. | Retain/write and four-path diagrams. |
| §1 linear and time-invariant scope | Sufficient scaling/superposition versus input-dependent rule distinction; retained. | Small system and explicitly bounded architecture claims. |
| §2 continuous to sampled dynamics | Sufficient held-input story, exact exponential/integral and half-decay example; retained. | Sampling figure and timestep lab. |
| §2 singular and bilinear cases | Sufficient integrator, block exponential/expm1, bilinear versus exact values; retained. | Exact local equations and boundary warnings. |
| §3 recurrent to convolution | Sufficient unrolled impulse paths, kernel taps, initial-state term and direct-feedthrough distinction; retained. | Impulse trails, ledger and two-mode hand calculation. |
| §3 FFT | Gap: circular wrap warning lacked a visible concrete failure. Added input [1,0,1], taps [1,2], four-output trail table, and tail folding into time zero. | Correct causal [1,2,1] versus circular [3,2,1] makes padding a causality invariant. |
| §3 equivalent computations | Sufficient recurrence/convolution/FFT agreement investigation; retained. | Direct controls and full checked CPU implementation. |
| §4 timescales and oscillation | Sufficient decay/half-life, damped real rotation and conjugate-pair readout; retained. | Memory clocks and shrinking spiral. |
| §4 diagonal layer | Sufficient trainable small model and its limited scope; retained. | Explicit state shape and retained gradient route. |
| §4 HiPPO meaning | Sufficient polynomial level/trend with inner product, exact linear reconstruction and time-dependent 1/t rule; retained. | Polynomial memory figure and return-visit disclosure. |
| §4 DPLR and generating function | Sufficient normal correction, finite kernel series and frequency interpretation; retained. | Clear finite-length correction and inverses-exist condition. |
| §4 Woodbury | Gap: scalar correction named without showing what it preserves. Added original 2×2 DPLR inverse table and right-hand-side comparison. | Off-diagonal output survives through the cheap correction; direct identity-product check. |
| §5 fixed delay versus selective memory | Sufficient marked-copy need and exact fixed-delay counterexample; retained. | Marked-event diagram and three-state proof. |
| §5 learned gates and Mamba injection | Sufficient exact scalar ZOH gate versus Mamba ΔB injection distinction, shapes and coefficients; retained. | Live selection task with annotated writes. |
| §5 operator versus full block | Sufficient short convolution, outside SiLU gate and projections; retained. | Mamba block figure and scratch/library boundary. |
| §5 scan | Sufficient affine composition, associativity, work versus dependency depth; retained. | Exact .1h+3.2 example and implementation. |
| §6 SSD state and dual view | Sufficient outer-product matrix writes, four numerical updates and signed causal influence; retained. | Matrix-write figure, SSD figure and chunking lab. |
| §6 chunking | Sufficient local outputs plus incoming state, decay and outgoing summary, final short chunk; retained. | Exact two-route reconciliation and practice 5. |
| §7 trainable real-data task | Sufficient deduplication, folds, baseline, full model/loss pipeline and honest losing results; retained. | Actual trajectory display, trained-model lab and complete programs. |
| §8 engineering and libraries | Sufficient state/KV counts, finite precision, causality versus pooled classification, diagnostics and maintained-kernel limitations; retained. | Deployment practice and CPU/library bridge; no unexecuted GPU claim. |
| §9 S5 | Sufficient multi-input/output shapes and distinction from independent scalar lanes; retained. | Explicit matrix dimensions. |
| §9 Mamba-3 | Sufficient endpoint interpolation example, order condition, complex rotation/parity toy and higher-rank write; retained after primary-paper recheck. | Existing endpoint/rotation/rank figure, no blanket claim that any learned mixing is second order. |
| §10 and references | Nine complete worked solutions and annotated alternatives retained. | Tasks include missing initial state, fixed delay, gate mismatch, SSD, deployment and stronger real-data design. |

## Research actually inspected

- [S4](https://arxiv.org/pdf/2111.00396), §§3.2–3.3, Algorithm 1 and Appendix C (Woodbury Proposition 4, Lemma C.3): finite kernel, inverse correction and frequency-domain construction. New inverse and wraparound numbers are original exact constructions, not measured performance.
- [Mamba-3](https://arxiv.org/pdf/2603.15569), §3.1, Proposition 1 and Remarks 2–3: endpoint weights and the λ=1/2+O(Δ) second-order qualification. Existing explanation is adequate and preserved.
- A fresh tool retrieval of The Annotated S4 failed; this review does not claim to have freshly inspected or run it. Its prior-revision resource annotation remains intact. No video or native fit campaign rerun.

## Representation and checks

The wrap table tracks destination time, the exact hidden relationship an FFT-only formula obscures. The inverse table keeps diagonal part, scalar denominator and correction separate inside the advanced branch. Existing topic-specific figures and live systems remain better than an extra generic control panel.

Root's narrow-screen inspection found that the original wide wrap table hid its numerical outputs beyond the first column. Transposed it into four time rows with three columns (time, linear contribution, circular slot), preserving every value and the surrounding explanation. Root will recheck the rendered layout; this source change alone is not a new browser pass.

Independent arithmetic checks compute linear and circular convolution from input/taps, calculate the rank-one correction, verify the matrix inverse in both multiplication orders, and compare its action on a basis input. Author checks also verify all code fences remain identical, generation is deterministic and the JSX parses. Source-bound records distinguish these checks from pending independent teaching and browser/integration review.
