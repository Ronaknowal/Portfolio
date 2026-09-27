# Depthwise and dilated convolution: conceptual-transition review

26 September 2026. Full generated lesson read, all nine numbered sections plus scratch ownership and seven practice problems. Canonical manuscript is `docs/teaching/drafts/depthwise-separable-dilated-convolutions/lesson.md`; its topic-specific generator has been updated and run. Existing six specialized labs already expose channel gradients, rank probes, stencils, reachable sites, branch context and real frozen inference. Author-only status, pending independent and browser reviews.

| Exact location/concepts | Disposition |
| --- | --- |
| §1 channel axes, dense/spatial/mixing operations, forward values | Retained complete two-channel calculations and live contribution lanes. |
| §1 gradient chain, simultaneous step, shared uses | Retained detailed derivatives and existing lab's complete editable gradient panel. |
| §2 MAC conventions/ratios/latency | Retained exact budget table, cost figure, one-output counterexample and performance conditions. |
| §2 rank/multiplier/spatial factorization/nonlinear distinction | Retained identity counterexample and live effective-kernel probes. |
| §3 dilation/null example/shape/stride/padding | Gap fixed for shape formula: enumerate stencil starts 0,2,4 on padded length 10, explaining floor and +1. Existing live stencil retained. |
| §3 field/jump/coverage/gridding | Retained support enumeration and live lab. Gap fixed inside advanced details: discrete shifted intervals for d=3 versus d=4, no-overlap-versus-no-missing-integer distinction, endpoint derivation d≤2S+1. |
| §4 groups/effective weights/API/reference | Retained complete matched code and reused scratch operator ownership. |
| §4 biases/BN/eval/training singleton | Gap fixed with exact affine folding construction; all state/mode caveats retained. |
| §5 width/resolution/inverted residual/linear bottleneck | Existing cost/shape explanation retained; gap fixed with sign-preserving (x,−x) expansion/ReLU/projection and explicit limit of invertibility intuition. |
| §5 V3 gates/segmentation/output stride/ASPP/global branch/decoder/causality | Retained specific examples, existing context controls, edge handling, code and variant limits. |
| §6 compression study/SVD/factor placement/results | Gap fixed for weight-objective versus task-objective: diag(4,3) compressed on two contrasting patches. All six measured runs and complete training/factor code retained. |
| Scratch section adaptive rank/complexity | Retained exact energy practice, rank-layout tradeoff and SVD-owner link. |
| §7 diagnosis, §8 seven practices, §9 readiness/references | Retained detailed checks, solutions and annotated alternatives. |

## Research and native representation

Read MobileNet V2 [original paper](https://arxiv.org/pdf/1801.04381) §3.1–3.2 and the bottleneck/manifold figure explanation (PDF pages 2–3). The new scalar construction gives an elementary bridge to that geometric argument without asserting universal invertibility. Reopened [Distill receptive fields](https://distill.pub/2019/computing-receptive-fields/); only title/metadata returned in this reading call, so no claim of a fresh full article reading. Existing article remains an annotated alternative from the earlier work.

New `DilationIntervalFigure` shows exact integer support on a common axis, separate shifted copies, visible crosses for holes, and a text equivalent. It appears inside the advanced interval construction. Generator inserts the figure after the same canonical paragraph and imports the semantic component. No asset generation, fitting or benchmark rerun. Author checks cover five new arithmetic groups plus JSX parse and baseline identities for the original model/lab and served factor program/frozen inference. Generated JSX and canonical manuscript both appear in the receipt.
