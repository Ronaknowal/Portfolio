# Independent review: five Classical ML implementation-depth additions

22 September 2026. Reviewer: a separate agent from the Classical author. **Independent review complete: all four findings are closed on the bound sources.** Final production integration belongs to the root agent. The review covers the additions and their connection to the existing mechanism owners, not a repeat certification of the five complete historical lessons.

## What was inspected

- All three complete new canonical programs: UMAP graph/update, CRFsuite score/inference and active-learning query/state bridge.
- Their actual learner-facing sections, setup commands, recorded-output modules, practice/solutions, mathematical/API distinctions and immediate surrounding text.
- Actual JSX strings for GP and logistic appended code, their original complete programs, and the native author verifier/evidence (18 initial groups; 19 passing final groups after the UMAP representability check). The final four-selection replay preserves all reviewed canonical source hashes; its receipt is bound below.
- Installed UMAP 0.5.12 layout source, including attraction, negative repulsion, coordinate clipping, endpoint ownership, skipped self-negatives and coincident-coordinate behavior.
- Primary CRFsuite objective source and scikit-learn objective/prediction contracts, rather than treating package import or similar accuracy as parity.

## Findings and closure

1. **Closed — conflicting CRF execution claim and learning flow.** An older paragraph still said no CRFsuite result was shown, while the new bridge executed it. The bridge also interrupted the original real-data program and its explanatory results. The author moved the complete bridge after that experiment's interpretation and distinguishes the separate invented API fixture from the unchanged treebank benchmark.
2. **Closed — CRF penalty mapping lacked its factor.** The updated paragraph now states that CRFsuite adds `c2 * ||theta||²` to the summed sequence loss, so mean sequence loss plus `lambda * ||theta||² / 2` uses `lambda=2*c2/S` for S unweighted sentences. State/transition parameter inventory still needs to match. The learner paragraph links the actual objective source. This is a convention mapping, not a claim that the existing custom real-data fit and the new small-fixture library fit have identical training optima.
3. **Closed — UMAP sampled-step cost omitted the output copy.** The explanatory text now separates O((1+s)q) touched-coordinate arithmetic from O(nq) copying/storage for this functional helper's complete returned array. Production in-place ownership is described without pretending the teaching function already has that whole-call cost.
4. **Closed — finite coordinates can lose representable distance arithmetic.** The original `exact_neighbors([[1e308],[-1e308],[0]],2)` emitted overflow warnings and returned an invalid repeated-self neighbor `[0,0]` and infinite distances. A finite input is not sufficient for representable squared distances or a float32 distance output. The lower-end companion [[0],[1e-50],[3e-50]] silently cast positive distances to zero. The repaired row calculation and float32 cast now raise on overflow, underflow or invalid arithmetic, converting this into explicit rescaling guidance before returning a graph. Independent checks reject 1e308 squared-distance overflow, 1e45 float32 overflow and 1e-50 positive-distance underflow. This is a conservative representability boundary (including subnormal arithmetic), not a claim that arbitrary finite coordinates are supported. Ordinary fixtures retain their outputs.

## Independent execution, separate from author replay

`scripts/verify-classical-library-review.py` writes [`classical-library-independent-native.json`](../evidence/classical-library-independent-native.json). It does not overwrite author evidence or regenerate program metadata.

| Scope | Complementary check and finding |
| --- | --- |
| GP and logistic | Babel extracts the exact appended `CodeBlock` string from each current JSX body. It is executed after the actual imported original program, without manually transcribing an equivalent. Both pass. GP agrees on full latent covariance, mean and log evidence under the fixed kernel/noise; logistic agrees on the same regularized optimum and gradient residual. |
| UMAP | k=3 versus k=4 changes the graph; multiplying distances by five preserves each declared graph within search precision. A hand-derived negative-update case checks attraction to .08, clipped repulsion to −.32, unchanged negative endpoint and preserved input array; self-negative IDs are skipped. The installed kernel confirms the stated a=b=gamma=1 formulas. This does not certify the full package scheduler as the ideal full-pair gradient. |
| CRF | Add token-wise score offsets [1000,−500,3] to all labels. Stable inference preserves every marginal and Viterbi path while logZ increases by the summed shift; rows remain normalized. This complements the author's exhaustive path/one-token checks and tests a log-space invariant under large offsets. |
| Active Learning | An instrumented oracle confirms each ID was still unlabeled and never repeated. The input label array is unchanged, the returned array gains exactly one acquired label, and each resulting classifier agrees with an independently fitted classifier on those acquired labels, including the last paid acquisition. No hidden-label vector is supplied to the strategy. |

## Learning and contract assessment

- UMAP preserves graph construction, a differentiated ideal objective and one explicit sampled rule as distinct mechanisms. The ordinary fitted estimator remains the owner of scheduling/initialization/annealing; a graph parity fixture does not promise layout parity. Duplicate/unattainable local-mass limits and removed-neighbor semantics are explained.
- CRF uses numeric dictionary features with explicit bias/BOS/EOS, avoiding implicit feature-expansion ambiguity. Marginals, Viterbi and whole-path probability are mapped correctly, and rounded weight-dump precision is disclosed. The use of all-possible transition features is correctly separated elsewhere from a hard legality mask.
- Active learning uses a missing-label sentinel, original pool IDs and identical fitted probabilities for utility parity, with cost/refit responsibility explicit. A library query is not a replacement for the controller, paid oracle or evaluation protocol. The new availability-mask exercise changes a meaningful constraint.
- Logistic's `C=1/(n*lambda)` follows the documented unweighted mean-loss objective with an unpenalized lbfgs intercept; duplicated rows need adjusted C. No trajectory-equivalence claim is made.
- GP fixes kernel, prior mean, target normalization, noise and hyperparameter optimization. Training `alpha` versus WhiteKernel test variance and latent versus new-observation uncertainty are distinguished correctly. Jitter is not silently interpreted as future measurement noise.
- Existing experiments, richer scratch programs and adverse/limited scientific interpretations are preserved. Three large new code views are lazy; the two short append blocks reuse their actual predecessor programs. No generic replacement lab, prediction gate or additional curriculum topic is introduced.

## Primary verification sources

- [CRFsuite L-BFGS objective source](https://github.com/chokkan/crfsuite/blob/master/lib/crf/src/train_lbfgs.c#L91-L98), read 22 September 2026. Raw source: `https://raw.githubusercontent.com/chokkan/crfsuite/master/lib/crf/src/train_lbfgs.c`; 10,754 bytes; SHA-256 `34b7c8519ea24f6533d4d45f57a65cf6364b5fe58ebc523460e4d0d64e98d1d9`. Lines 91–98 explicitly add c2 times squared norm and 2c2 times the weight to the gradient. The API/manual's word “coefficient” alone is insufficient to infer the half-factor.
- [scikit-learn binary logistic objective](https://scikit-learn.org/stable/modules/linear_model.html#binary-case): weighted mean likelihood plus regularizer divided by total weight and C; L2 regularizer includes 1/2.
- [GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html): training-diagonal alpha, disabled kernel optimization, target normalization and returned covariance.
- Installed `scratch/manifold-learning-runtime/Lib/site-packages/umap/layouts.py`, version 0.5.12, source hash retained in independent native evidence. The web-rendered layout source endpoint was unavailable; the actual installed source, not an unseen page, was inspected.

[`classical-remediation-independent.json`](classical-remediation-independent.json) binds final learner/program sources, primary-source bytes and closure status. Browser fetch/download bytes, accessibility/theme, mobile layouts, lazy loading and production build remain integration checks, not results of this review.
