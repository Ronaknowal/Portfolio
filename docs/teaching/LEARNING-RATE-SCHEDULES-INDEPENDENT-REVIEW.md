# Learning Rate Schedules: bounded independent review

10 September2026. Independent review of the root author's complete rewrite at `learning-rate-schedules-cosine-warmup-onecyclelr`. **No substantive mathematical, model/API or pedagogical blocker found.** No authored lesson/model/example/lab source was edited by this review. Root owns final browser/formatting/integration evidence and acceptance status.

## Actual read scope

Read the full current lesson, design and incoming scope note, all nine complete example programs, all pure model functions, and the four investigations plus calibration/update-clock/resume figures. Examined the scalar derivative and contraction thresholds, finite-budget endpoints and warmup join, noise assumptions and stationary variance, OneCycle phase/momentum convention, actual used versus prepared rates, decay-product approximation, strict plateau predicate/counter/cooldown order, accumulated/committed clocks and complete-state resumption. Read all seven changed-input exercises and their hints/answers/acceptance criteria.

The lesson keeps scalar assumptions adjacent to the equations and does not generalize the independent-noise recurrence to validation-adaptive rates. Its one-cycle shape lab explicitly separates the momentum plot and its noise investigation explicitly models plain SGD only. The noise plot's log10 clipping is disclosed and does not alter exact readouts. The event/state visuals expose mechanisms that a generic rate curve would miss. Optional advanced empirical context is qualified; useful examples do not replace the core derivation.

The retained ten-update warmup example is explicitly presented as a different endpoint convention beside the new finite-budget version. This review checked the two actual formulas/outputs and the preservation rationale in the design; it does not claim an independent byte-for-byte comparison with an unavailable historical lesson snapshot.

## Independent computations

Command: `node scripts/review-learning-rate-schedule-mathematics.mjs`, which invokes `scripts/review-learning-rate-schedule-mathematics.py` using the workspace Python3.12.14 and **PyTorch2.14.0+cpu**. Pass at **2026-09-10 11:37:32.571 UTC**. Source hashes and compact results: [learning-rate-schedule-independent-review.json](evidence/learning-rate-schedule-independent-review.json). Regenerable local fixtures/results: `scratch/learning-rate-schedule-independent/`.

| Independent check | Evidence and conclusion |
| --- | --- |
| OneCycle boundary variants | Ten policies with total3/7/12/22 and permitted early/middle/late aligned rises, 126 actual used rates/momentum values. Fresh CPU SGD/OneCycleLR objects agree with model values before every optimizer update, including very short phases |
| Noise moments | 36 rate/curvature/noise policies and135 states. Python Fraction closed products of future multipliers give the mean and accumulated variance independently of the forward recurrence. Exhaustive ±σ trajectories at each prefix agree with those exact fractions and model outputs. Cases include zero-rate updates, zero noise, q=0, negative multipliers and unstable individual updates |
| Plateau state | Sixteen configurations,192 actual PyTorch observations. Binary-exact threshold0/.125, patience0/2, cooldown0/2, floor0/.025; best, bad count, cooldown and actual rate agree. This tests strict boundary handling and event order, not just the final rate |
| Hand checks | New six-rate warmup/cosine sequence, two distinct equal-sum decay products and the1/16 stationary variance independently calculated. These are four checks in addition to root's broader changed-input practice suite |

The noise computation uses suffix products and complete finite distributions rather than restating the JavaScript recurrence. OneCycle and plateau comparisons use actual PyTorch state, not another handwritten scheduler. These checks corroborate selected implementation contracts; the written invariants/derivations and assumptions remain necessary.

## Primary-source spot check

Independently opened the [SGDR original paper](https://arxiv.org/pdf/1608.03983), section4.4 and Figure4 text (PDF pages8–9). It explicitly describes EEG classification of **actual** right/left hand and foot movements from14 subjects, alongside different budgets and snapshot comparisons. Thus the lesson's specific alternate application is supported; it does not mislabel this as imagined movement or claim clinical efficacy. Only this relevant paper section and surrounding figure discussion were inspected in this cross-review, not every empirical claim or the underlying EEG dataset.

Actual installed PyTorch behavior was inspected through execution. Root's design/source ledger owns the broader official API and original scheduling-paper research. The alternate lecture is correctly described as a verified resource/outline whose playback was not reviewed.

## Limits and handoff

This was a bounded independent source/mathematics/API review. It does not duplicate root's full browser run, claim a novice study, assess GPU throughput or establish cross-version/distributed reproducibility. The toy expected-loss curves are calculated moments, not measured neural-network training results. No universal scheduling winner is asserted.

Root was still completing narrow mobile equation/capture and final evidence work during this read. The design record's provisional research/runtime/status wording should be reconciled with the final verification record during that already planned handoff. This is documentation completion, not a mathematical finding against the reviewed source. If substantive formulas/models change after the stored hashes, assess whether the affected independent checks need repeating.
