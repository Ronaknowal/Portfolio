# Recommender systems: conceptual-transition review

Author review, 26 September 2026. Full implemented JSX is the manuscript owner; no generator or second prose manuscript owns this lesson. This is an in-scope teaching revision, with independent and rendered-browser review pending separately.

## Complete reading map

| Actual location / concepts | Before assessment and disposition | Support now / transfer |
| --- | --- | --- |
| §1 request, user/item/event/impression, missingness, observation mask, observed SSE | Retained: concrete workshop request, editable evidence matrix and zero-vs-missing practice already establish the objects and loss. | Feedback lab and missing-observation exercise; complete evidence program retained. |
| §2 temporal information, train/validation/test, eligibility, warm/cold, shrinkage | Retained: count-proportional timeline and actual 3.06 calculation explain timing and pseudo-count. | Time-replay and cold-start exercises test different settings. |
| §3 co-rated alignment, raw vs adjusted cosine, overlap/shrinkage, signed residual predictions, fallback, cost | Gap: algebra named centering without a complete visible case of the question changing. Fixed with co-rater table and raw .976 versus centered −1 calculation. | Same people and items remain visible through baseline subtraction. Existing neighbor lab, signed-weight/fallback exercise and complete code remain. |
| §4 shared factors, biases, shapes/rank, ambiguity, rotations, SVD distinction | Retained: original .14 dot product and 3.14 score, ambiguous 2×2 completions, rotation investigation and completion exercise make the distinct limitations visible. | All parameter-count and general invertible-transform depth retained. |
| §5 loss, gradient sign, simultaneous old-value updates, epoch, RMSE, initialization, degree-weighted regularization | Retained: one concrete update, learning-rate failure, live factor update, finite differences and changed-input exercise. | Penalty-counting deeper branch and sum/mean distinction retained. |
| §6 ALS, fixed block, positive definiteness, monotonic block objective, bias and penalty conventions | Gap: “ordinary regularized least squares” was an unexplained switch in viewpoint. Added freeze-one-side bridge and calculated scalar loss curve before the vector solve. | Scalar targets (4,2), fixed factors (1,2), p=4/3 minimum show what a block solve does; existing full ALS and penalty-transfer exercise retained. |
| §7 implicit targets/confidence, all-pair loss, normal system, cached Gram, sparse correction, zero history | Gap: exact identity was present but the surviving missing-pair contribution was mentally hidden. Added five-term matrix decomposition and separate target-vector calculation. | All-item baseline includes q₂q₂ᵀ despite zero target; original lab and changed implicit-block exercise retained. |
| §8 gap, pair probability, softplus, gradient, regularization, sampler/objective/support | Gap: vector equations needed a direction and signal-size bridge. Added preferred/sampled/user directional explanation and g at −2,0,2. | Existing BPR live investigation, finite differences and reversed-pair exercise retained. |
| §9 slate/candidates, precision/recall/RR/AP/NDCG, ideal denominator, retrieval ceiling, aggregate cohort | Retained: movable list and complete [2,0,4,1,3] calculation show each count and position. | Existing denominator-preserving transfer exercise and candidate-recall ceiling make information loss explicit. |
| §10 library semantics, CSR confidence, orientation, stale alternating state, raw/inner IDs, clipping/cold fallback | Retained: each API is mapped to the earlier equation and actual factor shape; unchanged complete library programs and native results. | Explicit confidence transform and final-Q recalculation explain semantic choices rather than treating API names as equivalence. |
| §11 cold metadata map, hybrid, towers, sequence vs counts, eligibility/retrieval/rerank, inner product vs cosine | Retained: metadata ridge derivation and runnable program; pipeline and numeric ranking reversal; maintenance-report application maps evidence to outcome. | Cold-user/item exercise and specialist system-design ownership preserved. |
| §12 logging vs target mixture, assignment propensity, IPS cancellation, variance, missing support | Gap: the two mixtures were only equations, inviting confusion between exposure and click probability. Added proportional mixture bars and explicit contribution reweighting .32/.16→.20/.40. | Existing policy lab, sample vs expectation, support counterexample and changed-policy exercise retained. |
| §13 complete experiment, refit, cohort report, honest losing baseline | Retained unchanged: explicit 20/5/6 and 18/7/6 reports, frozen test and baseline comparison. | No measurements or outcomes regenerated. |
| §14 independent practice and readiness | All eleven end exercises, hints and reasoned solutions retained. | New representations explain prerequisites for their existing changed settings; no learner prediction gate introduced. |

## Research actually read

- [Google's matrix-factorization lesson](https://developers.google.com/machine-learning/recommendation/collaborative/matrix), current page dated 25 August 2025, read 26 September 2026: embeddings, weighted-objective and alternating fixed-factor explanations. Used to reassess the missing fixed-side bridge. Did not adopt its unqualified speed comparison or conflate implicit observed positives with explicit observed ratings.
- [Hu, Koren and Volinsky](https://yifanhu.net/PUB/cf.pdf), original paper's implicit-feedback characteristics and §4 normal-system identity, read during this revision. Inspired showing the common baseline and sparse correction separately; all numbers here are the lesson's own constructed example.
- [Rendle et al. BPR](https://arxiv.org/pdf/1205.2618), original source opened; existing exact derivation and measured evidence preserved rather than claiming a fresh full-paper or video review. New multiplier values follow directly from the displayed sigmoid.

The existing annotated Google/Stanford alternatives and primary/library references remain. No video viewing claimed.

## New representations and checks

`RecommenderIntuitionFigures.jsx` owns three static figures; `recommender-intuition.css` scopes their theme and responsive layout. They load only with this lesson. The centering table is inline at §3. No live simulator or expensive training is added.

- Scalar curve: p-axis 0–3, loss-axis linear; all samples calculated from the declared quadratic. A distinct minimum mark at 4/3 and nearby interpretation connect the curve to the normal equation. SVG scales uniformly, full arithmetic is in the caption.
- Gram decomposition: actual 2×2 cells in wrapping HTML, arithmetic operators and labels; no positional or color-only inference required. Matrices wrap rather than shrinking their numbers on a phone.
- Mixture bars: exact proportional widths, shares labeled, A amber and B neutral charcoal. 100% width means probability one for both rows. Caption states click probabilities held fixed and resulting contributions.
- Exact calculations and JSX parsing recorded in `author-checks.json`. Existing data models, runnable programs, measured outputs, original lab components and practice are unchanged. Prior evidence covers those unchanged computations; it does not certify the new rendered figures.

Pending: independent full reading and rendered desktop/narrow-screen visual review by the integration owner; record actual checks before full closure.
