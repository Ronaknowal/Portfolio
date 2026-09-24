# Sequential ICA implementation review — 16 September 2026

Scope: Classical ML position 18 only, `independent-component-analysis-ica`. Independent review compared the complete prepared manuscript and visual specifications in `drafts/independent-component-analysis-ica`, the ICA lesson design, current reader, figures, investigations, state/formatting helper, styles and displayed programs. Numerical/data/native evidence was reused only after source-identity checks. The integration owner owns production repairs and current model/build/browser runs. This record does not advance the queue.

## Coverage and learning experience

No missing prepared section, figure, program, practice problem or substantive investigation contract was found. The reader retains all 11 sections, six mechanism figures (M1, D1, W1, F1, P1, E1), the R1 and C1 investigations with gated C1b follow-up, two complete programs and six separately closed hint/solution pairs.

The implementation preserves source versus observed coordinate identities; the invertible mixture and concrete sensor sums; uncorrelated but dependent variables; centering, covariance eigendecomposition and whitening; the whitened diamond's remaining dependence; higher-order rotation evidence; equal-kurtosis and Gaussian nulls; the non-Gaussian zero-kurtosis counterexample; fixed-point updating, normalization and within-iteration deflation; sign/permutation/scale ambiguity; contribution reconstruction; and real-recording limitations. The distinctions between objective, algorithm, extension and application remain explicit. Training, development selection and untouched test evaluation remain separate; the direct reference never enters either fitted decomposition. The previously corrected hand-trace rounding (`0.7050422`) is appropriate and is not an omitted manuscript claim.

R1's proposed marker appears only after a matching commitment; applied readouts retain the prior state during draft changes. C1 gates its first independent task on changing an amplitude from the already-worked example. C1b's source/column/result table stays hidden until commitment and application, and a changed parent reconstruction remounts the follow-up. Historical results are labelled; edited predictions cannot apply an old commitment. Exact Gaussian, excluded-component and zero-amplitude nulls have distinct feedback branches. The four defects below concern precision, a remaining identity null, provenance copy and Reset's gate, rather than missing instructional scope.

Source/CSS inspection found the intended equal-axis plots, finite support labels, origin absence, contribution signs and two-interval source identities intact. The reviewer inspected retained narrow `ica-rotation-320.png` and `ica-w1-390.png` captures as corroboration of actual geometry, not as a fresh production-browser claim. The integration owner independently inspected the new tiny-change feedback captures after the current build.

## Findings and repairs

### I18-1 — graded changes disappeared from numerical feedback (P2, repaired)

The grader retained numerical precision while feedback used six decimal places. Three supported examples:

| Investigation | Reproduction | Actual numerical evidence | Original feedback problem |
| --- | --- | --- | --- |
| R1 | Binary family, applied 45°, proposed 45.001°, predict larger | magnitude `1 → 1.00000000121847`, difference `1.21846999157e−9`, greater than the `1e−10` same threshold | both magnitudes printed 1, both angles printed 45°, difference printed 0 |
| C1 | sources `(2.00000001, 0)`, keep first, predict sensor 1 = 4 | actual `4.00000002`, error `−1.99999998785e−8` | rejected answer and actual both printed 4, error printed 0 |
| C1b | sources `(1e−8, 0)`, keep first, source-only scale 2 on component 1 | reconstruction `(2e−8, 1e−8) → (4e−8, 2e−8)` | result said changed but every displayed comparison coordinate printed 0 |

The repaired shared feedback helpers retain 12 significant digits, with signed nonzero differences; R1 preserves the actual entered angle. All three grading messages use those helpers and prompts state the grading tolerances. Compact ordinary readouts can remain rounded because graded feedback now exposes its actual operands and difference. Extracting the actual helper definitions and evaluating 15,168 comparisons (including 8,400 graded rotation changes across all three families) found every graded change distinguishable. Direct reevaluation of all three examples produces visible evidence. Browser regressions cover all three cases.

### I18-2 — source-only identity rescaling claimed compensation (P2, repaired)

With a nonzero kept component, choose source-only scaling and `c = 1`, then predict no change. The original fallback said the compensated product was fixed by construction, even though this operation never divided the column. An explicit identity branch now explains multiplication by one. A separate fallback accurately describes other differences within `1e−9`; existing compensation, excluded-component and zero-amplitude branches remain intact. The browser regression requires identity wording and rejects the old compensation claim.

### I18-3 — E1 caption misstated evaluation sample identity (P2, repaired)

The caption claimed all displayed correlations used every one of the 20,000 instants. In fact training uses 12,000 samples; development and test correlations each use their own 4,000-sample interval. The caption now says the correlations use all full-resolution samples of their respective development or test interval, not the plotted envelope. This agrees with the unchanged prepared program and data evidence. The browser regression rejects the old claim.

### I18-4 — C1 Reset failed to restore the independent-task gate (P2, repaired)

After completing any C1 trial, Reset returned amplitudes and keep-set to the worked initial example but left `everApplied` true. The learner could grade the already-worked answer and C1b remained open. The parent investigation now resets both the investigation state and `everApplied`. The regression completes C1, resets it, requires the amplitude-change notice and closed C1b, and confirms typing the initial answer alone cannot enable commitment.

### I18-5 — inclusive decimal tolerance boundary (P2, repaired)

A nearby direct model check found sources `(2,0)`, keep first, prediction `4.000000001` rejected: subtraction yields `1.000000082740371e−9`, just above the promised inclusive `1e−9`. This is floating-point subtraction dust, not a meaningful excess over the tolerance. The model now adds `4 * Number.EPSILON * max(1, |prediction|, |actual|)` as subtraction-rounding slack. Independent direct execution of the repaired model accepts both `4 ± 1e−9` and rejects both `4 ± 1.01e−9`. The script exercises those four cases in the browser's same audit group. The larger changes in I18-1 remain unaffected. The owner reports the rebuilt app, 14,978 model assertions and final production-browser run (seven groups, 39 captures) all pass after this delta.

## Evidence and source identity

At entry, all eight topic-owned runtime hashes matched the retained browser evidence. The two navigation hashes had changed through later integration, so old browser success was not treated as proof of the current production bundle. Both displayed programs and expected outputs match the native record (48 oracles); the example module is unchanged. Data evidence records 31 checks, with source identity confirmed by the integration owner. The owner reran the numerical suite (14,978 assertions) and production browser (seven groups, 39 captures) after I18-1–4; those runs predate the newly reported I18-5 boundary.

The reviewer changed only this record and the authorized bounded regression addition in `scripts/verify-ica-browser.cjs`. Its syntax check passes. One initial new assertion accidentally captured a sentence period in a numeric token; trimming terminal punctuation fixed that assertion before the successful run. No production source was edited by this reviewer.

SHA-256 snapshot after I18-1–4 (reader/model/data/examples/styles unchanged):

| Source | SHA-256 |
| --- | --- |
| `src/learn/data/topics/independent-component-analysis-ica.jsx` | `66dfdfab99e1a91e70ce2677fa36840061397a574eaa262006a96da864657794` |
| `src/learn/data/ica-models.js` | `9430d1ee67b56dd013cc3aef15a7576e1dc8b907bc54dd961ef7df5812f58cfa` |
| `src/learn/data/ica-data.js` | `897ca68813a4b55b64677032e9a7df97e25d30d8a5acffdfde610ce9e3152e61` |
| `src/learn/data/ica-examples.js` | `65262b951808f1d3bd2e12e33669c59766ed78c85e90a05224072d7a263e6064` |
| `src/learn/components/lesson-labs/IcaShared.jsx` | `90d5226d9402eede3410df83f35040afae1b583cb3035ea8c616f7a15f31da4f` |
| `src/learn/components/lesson-labs/IcaFigures.jsx` | `92a7acfdd2847928d98dfad207a69f2b65fb1cf8f5e064da9030a2b5ac5070a1` |
| `src/learn/components/lesson-labs/IcaLabs.jsx` | `e340e26d79057c92c0325dead4da3f3ea0806270f28f3299235167b92e1c0880` |
| `src/learn/components/lesson-labs/ica-labs.css` | `1556b44d8add961520b11a689cc6f44fbdc890642bda73ed72f946f8ceb2edcd` |
| `scripts/verify-ica-browser.cjs` | `4780727352cfa63237294114770cdcc09d739f4c276fac73afb276380f9baa45` |

Pre-repair source identity: shared `11b01dc5eefd7ae04d877551a461153713cb27c8f843550fffa69361c6b09cf2`; figures `fa7cc42c66b5d675e4eb015f44474c114c95b37960b791222e49b4f45b97d0aa`; labs `5983bbbd8c8e0b4b1e67b45f453c680ff6f4ea24f5a27fe558dafe2a38aae9a0`. The unchanged model hash above identifies the original I18-5 reproduction.

Final I18-5 delta identity: `src/learn/data/ica-models.js` SHA-256 `f4d48ae6fc021e1ff3574ebf4ce1893205852094e9519ca86b8aee66aa502774`; `scripts/verify-ica-browser.cjs` SHA-256 `65b25b8ae1652d5d339eb1c60077bb1a2d4fca87b508781ceb716c67159b0a49`. Other topic-owned source hashes remain as in the snapshot table. No remaining source-review finding or content omission was identified after this bounded delta.

Closure: all five findings are repaired and covered by the final source-bound browser run. The independent reviewer accepts the bounded fixes; position 18 is ready for the integration owner's checkpoint closure. Review of any subsequent topic requires explicit advancement.
