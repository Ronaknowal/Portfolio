# Entropy, Cross-Entropy & KL Divergence: independent review

Reviewed 10 September 2026 by a different agent from the author. This is a bounded source, mathematical and numerical review, separate from the author's browser sweep and root integration. Final result: no unresolved material defect within the revised documented contracts.

## Scope and reasoning

Read the complete lesson, all eleven native programs, practice prompts/hints/solutions, pure models and the author's validation record. Checked the distinction between individual surprise, average entropy and realized code length; finite support and the direction of KL; weighted conditional risk; stable finite-logit likelihood; coding overhead and decoder framing; coordinate-dependent differential entropy versus invariant measure KL; and empirical point masses versus negative log density. The actual program questions are visible through the topic's own wrapper.

The prefix argument uses the stated prefix/Kraft assumptions and an expectation under the source, without promising a bound for each message. The conditional decomposition uses actual context frequencies and an adequate conditional model. The maximum-entropy certificate fixes the moments and expected log candidate density; positivity and endpoint limits are explicit. For nonuniform positive base weights the certified objective includes the expected log base weight, rather than ordinary Shannon entropy. The continuous and empirical qualifications correctly separate density values from event probabilities.

The reviewer opened the [Stanford maximum-entropy notes](https://cs229.stanford.edu/notes-spring2019/MaxEnt.pdf) and [PyTorch 2.14 CrossEntropyLoss contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html). The finite interior derivation and native logits/class-index/soft-target use match their stated scope. No independent full-video playback or new first-time-learner study is claimed.

## Findings and closure

Two defects were reproduced on the initial frozen pure model `cc259356798fb47d9febaff16fb6fe3694e896a7397620509c50b290027a5356` and sent to the author before any correction:

1. `maxEntropyState(1e-200, 1)` accepted a strictly interior mean but underflowed a positive optimum mass to zero, yielding a false infinite KL gap. `1e-100` also exposed precision loss in the entropy-gap equality. The author narrowed the visual numerical contract to exact means 0/2 or interior means in [1e-6, 2−1e-6], rejecting closer interior means. The mathematical theorem remains unrestricted by this calculator range. Every original selector choice survives.
2. `informationState([1,1,0,0], [100,1e-322,0,0], Math.E)` normalized a positive weight to zero and falsely displayed a support infinity. This was reachable through the editable draft. The normalizer now rejects positive normalization underflow; the text parser additionally rejects nonzero literals such as `1e-9999` that underflow during numeric conversion. Intentional exact zeros still carry the correct infinity, and the representable `1e-310` input remains finite.

The author made the changes. The reviewer independently reran both failure families, accepted boundaries, genuine support zero and a tiny finite case against the final model `519d1fff3ae48c1cd36e11a3bce3a0e31880b8286ca8188faddfe958c0ccf49f`. The lesson body, native examples, CSS and blueprint are unchanged; the model and lab instructions changed. All six final fingerprints were checked against disk and are preserved in the durable evidence.

## Complementary independent execution

Command: `node scripts/verify-entropy-information-independent.mjs`, which executes the separate Python reviewer oracle. Passed at 17:38:15 UTC.

| Independent check | Actual evidence |
| --- | --- |
| Near-equal heterogeneous laws and tiny finite support | 21 comparisons against 90-digit Decimal arithmetic, in bits and nats. Relative error at most 5.32e−14, including KL values so small that a loose absolute tolerance would be meaningless. |
| Repaired input contracts | Six rejected range/underflow inputs; intentional exact-zero support remains infinity. |
| Maximum entropy | 32 states including exact boundaries and both accepted interior range edges; independent moment root and entropy optimization for 24 interior states. Gap checked against independently calculated entropy deficit. |
| Nonuniform-base certificate | 606 feasible laws over two different base-weight vectors and three different means; each also checks the constant shift under scaling all base weights. |
| Changed prefix alphabet | Actual native decoder used on all 780 messages of lengths 1–4 over a different five-symbol code, plus invalid-prefix and incomplete-stream rejection. |
| Continuous native helper | Three changed Normal pairs checked by independent density/log-density quadrature. |
| Changed practice | Unequal-context table independently yields H(Y)=0.9340680554, H(Y\|X)=0.7219280949 and reduction 0.2121399605 bits. |

The independent scalar optimizer was parameterized on [0,1] after its default relative-position termination proved unsuitable for a narrow interval next to t=1. This was a reviewer-oracle adjustment, not a new product defect or a loosened comparison tolerance. The mathematical certificate also checks all selected feasible candidates independently of the optimizer.

## Browser evidence and limits

The author's final font-enabled 1440/390/320 `numeric-boundary-results.json` was read: rejected drafts preserve previous finite state, the tiny finite input and intentional-zero infinity differ, all 18 existing maximum-control combinations per width pass, and module range checks and document geometry pass with no page/console/request errors. This browser work is attributed to the author, not rerun as an independent full sweep. The reviewer opened the actual `numeric-error-390.png` screenshot and confirmed the readable error with retained finite probabilities. Earlier full browser/keyboard evidence remains in the author's verification record.

This finite review does not establish arbitrary-range float64 safety, exhaustive screen-reader behavior, or user approval. It does close the reproduced defects and verify the central proofs and changed numerical cases. Parent/root owns integrated build and publication status.

Durable evidence: [entropy-independent-review.json](evidence/entropy-independent-review.json). Reproducible fixtures and raw outputs: `scratch/entropy-independent-review/`; scripts: `scripts/verify-entropy-information-independent.mjs` and `.py`.
