# Concentration Inequalities — implementation and author verification

10 September 2026. Mathematics19, stable identity `concentration-inequalities-hoeffding-bernstein-chernoff`. The scoped rewrite is complete and author-verified; root owns integrated build/publication review. User acceptance and observation of an actual novice learner remain separate.

## What changed and what was preserved

The previous body introduced the right bounded/variance/count tools but had no topic-native figures or investigations. It now explains the event before the theorem, derives the moment/exponential mechanism, works the unequal-range and Bernstein inversions, contrasts exact tails with guarantees, and connects sampling design, selection and monitoring to the validity of the conclusion. Essential explanations are on the reading route; optional proof disclosures add the tilted-curvature and bounded-moment details.

All three original Python programs and their expected outputs are preserved exactly. The original body and programs are stored in [the preservation record](evidence/concentration-original-content.json), original SHA256 `234855e034120a720d11c2be9055c0e70c2053288fabd1e79e29d76391b56bf4`. The original 185-sample calculation, .136/.106 comparison, .115 relative-count bound and changed 400-observation task remain. The generic bootstrap/permutation recommendation was corrected to explicit assumptions, and the hash application now treats one fixed bucket's load rather than falsely independent occupied-bucket indicators.

[The design](CONCENTRATION-INEQUALITIES-LESSON-DESIGN.md) records the original inspection, preimplementation scope, actual primary-source research and representation contracts. The title and module order are unchanged. The actual next module topic is Monte Carlo Methods & MCMC; the closing bridge does not skip to an arbitrary publication.

## Representations and their contracts

| Representation | Learning work | Actual calculation / interaction | Evidence and limits |
| --- | --- | --- | --- |
| Inclusive finite count tail | Separate an event's actual probability from several upper bounds | Binomial probability bars, highlighted integer count event, independent n/p/threshold controls, named certificate ledger | Exact integer/Fraction tail oracle; p0/1, certain/impossible events, threshold at/below mean, tiny and near-one probabilities. Default20/.25/8 keeps the highlighted event visible. Bars are numerical finite-law masses, not samples. |
| Inline exponential proof chain | Make the pointwise cover, expectation, factorization and optimization steps explicit | Five connected semantic steps alongside the prose | Root source/proof review; mobile ordinary-reading screenshots. Independence belongs to factorization, not to the union bound. |
| Exponential witness investigation | Changing λ changes a bound, not the original event | Fixed Binomial(20,1/4), threshold10, exact log-MGF objective versus Hoeffding envelope and actual tail, exact-optimum control, contribution table | Exact finite expectation sums and 101 witness states; keyboard Home/End/optimum/reset. Slider `step=any` preserves λ=log3 without browser rounding. |
| Precision and variance curves | Distinguish sample size, units, error budget and justified variance information | Analytic Hoeffding/root-Bernstein/relaxed-Bernstein radii, logarithmic n axis, control-driven sample budget | Independent scalar root, 192 budget/inversion cases, unit/range and variance0/.25 states. Variance is a known upper bound, not an estimate. Known-zero degeneracy is explained separately from the generic formula. |
| Rare-variance counterexample | Show why an uncorrected sample variance can invalidate coverage | Forty labelled population tickets versus an explicitly chosen 100-zero sample | Exact (39/40)^100≈.07951729 exceeds .05 while the substituted radius .02459253 misses error .025. This proves actual miscoverage for the proposed procedure, rather than showing one permissible failure. |
| Three sampling laws | Equal marginals do not imply equal information | Independent binomial, one draw copied n times, uniform-subset hypergeometric distributions and mean-variance readouts | 1,600 changed cases checked independently; full population is deterministic. Without replacement uses a separate convex-order theorem, not independence. Each vertical scale is explicit. |
| Family-error budget | Protect the family selected or monitored, rather than one isolated check | Independent versus identical check failures, δ versus δ/K radii, exact finite-law family probabilities and union certificates | 144 independent exact family checks; UI count/relation/keyboard/reset. Integer points joined as guides; conservative boundary-including failure is distinguished from closed-interval coverage. |

The lesson contains five investigations, two inline figures, eleven displayed equations, nine complete Python programs, one short checkpoint and eight substantial independent tasks. These counts describe this lesson's actual conceptual hurdles, not a future template or quota. All eight tasks offer a hidden optional hint before a separately hidden worked solution. All nine programs have a visible investigation question before their code.

## Native and mathematical verification

Command: `scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-concentration-inequalities.py`.

Final native record: `scratch/concentration-verification/native-results.json`, 10 September2026 15:06:11 UTC. Python3.12.14, NumPy2.3.5, SciPy1.18.1 on Windows. Every displayed program was executed in a fresh isolated Python process and stdout matched its stored expected result. The six additional programs use only the Python standard library; NumPy/SciPy support independent author checks, not learner setup.

- 3,555 inclusive binomial-tail cases across counts, probabilities and support endpoints; integer-polynomial/Fraction sums supply an independent finite reference. Maximum absolute log-tail error was 2.2737367544323206e−13. The original three code/output pairs match exactly.
- 192 sample/radius cases; Bernstein inversion is compared with independent scalar root finding, and upward-rounded Hoeffding budgets are checked at n and n−1.
- 1,600 sampling-design cases checked against independent binomial/hypergeometric/copy laws, event membership, probability normalization and sample-mean variance.
- 101 exponential witness states checked by finite expectation sums and pointwise contribution domination.
- 144 family cases checked with exact rational single-check probabilities and exact independent/identical family composition.
- 500 asymmetric, non-Bernoulli bounded-MGF checks complement the written lemma and Bernstein arguments. Numerical checks support the implementation; they are not presented as proofs of the theorems.
- Twelve invalid input cases reject invalid domains. Explicit finite-log/complement cases retain approximately 10^(−3320.036), 1−10^−200 and 1−10^−4000 rather than incorrectly reporting exact zero or one.
- All numerical changed exercises were independently recomputed, including radius .06790508, counts1060/4427/73778, exact37/256, unequal/weighted widths, Bernstein .04504995, finite-population variance and simultaneous radius .08311291. The protocol task is assessed by its stated population/design/selection/stopping obligations.

During oracle development, SciPy's ordinary double-precision survival computation lost some extreme tails or significant digits; converting an extremely small rational to float before taking a log also lost digits. The final reference instead sums integer probability numerators and takes logs before conversion in the small tail, with a complementary calculation near one. Central tails retain an additional SciPy comparison. These were reference-method limitations, not evidence against the exact finite law.

## Actual browser and normal-reading review

Commands:

```text
node scripts/review-concentration-inequalities.cjs
node scripts/review-concentration-reading.cjs
```

Microsoft Edge, headless Playwright, real localhost5173 route. The interaction record at `scratch/concentration-browser/results.json` passed at1440 and390 pixels, final10 September2026 15:10:22 UTC. At each width it checks eight actual keyboard anchor arrivals; ten count-tail edge/threshold states; λ endpoints/exact optimum/reset and contribution disclosure; four changed sample/variance budgets; six changed sampling-law states; four family states; all reset buttons; nine rendered code/output pairs and visible program questions; the short checkpoint and eight independently revealed hints/solutions; all five reference links; all eleven equations; and SVG label bounds. Page errors and horizontal page overflow were absent.

The separate ordinary-reading record, `scratch/concentration-browser/reading-results.json`, passed at1440/390/320 pixels, final10 September2026 15:10:42 UTC. It inspects introductory route, inline proof chain, rare-variance figure, native question/code/output, a closed task then separate hint/solution, references, equations and all plots at320. Long code scrolls within its own block. All eleven equations fit without horizontal equation scrolling; topic containers and SVG labels stay within their bounds. Exact-optimum/reset keyboard behavior was also checked at320.

Actual opened screenshots include the default tail, exponential optimum/contributions, budget curves, all three sampling laws, family curve, inline proof chain, rare-variance figure, repaired equations, program/output and independent hint/solution/source reading. [The durable author evidence](evidence/concentration-author-review.json) records the exact opened-image list, screenshot paths/hashes, test payloads and frozen source fingerprints. Capturing or opening screenshots does not imply an actual user study.

Repairs made from this review: two rounded changed-answer values were corrected before final checks; direct complementary probability evaluation preserves near-one information and support-determined certain events; the λ control no longer rounds the exact optimum; the initial discrete tail is visible without requiring a first interaction; three long equations received meaningful line breaks; mobile graph type was enlarged with endpoint tick alignment. Early browser harness assumptions about exact wrapped-select labels and the shared `OUTPUT` label were corrected. Final tests passed after those repairs.

## Independent source review and remaining boundaries

Root read the complete lesson's proofs, constants, variance counterexample, fixed-bucket application, finite-population comparison and countable-budget reasoning, plus the model contracts. No actionable mathematical defect was reported. Its two pedagogical findings—optional hints before substantial solutions and visible program questions—were implemented and exercised in the final browser record. This is a complementary source review, not an extra claim of numerical case counts.

Research scope is recorded precisely in the design and annotated learner references: Waterloo written moment/concentration notes; Bartlett's selected MGF/concentration pages; Stark's finite-population/Bernstein section; the MIT18.200 official video page and description, with no full playback claim; and Howard et al.'s introductory/time-uniform definitions and stopping equivalence. No chart contains an invented benchmark or simulated measurement disguised as a theorem.

Incoming [concentration notes](topic-notes/concentration-inequalities-hoeffding-bernstein-chernoff.md) are adapted. The MCMC dependence continuation already has its own resolved note. A scoped [PAC/VC destination note](topic-notes/pac-learning-vc-dimension.md) preserves the finite-class selection bridge and asks its future author to reassess the stronger learnability assumptions; that destination was not rewritten. Sharper empirical-Bernstein/martingale/mixture constructions are explicitly deferred with this concentration topic as owner and a theorem/verification revisit condition. The present elementary all-time bound is valid but not claimed to be sharp.

No global module ordering, IDs, shared renderer, publication manifest or root rollout ledger was edited by this author. Runtime imports remain topic-owned, using direct `Math.jsx` and existing small pedagogic UI. No deployment or full integration build was run here; those remain with root. Final source fingerprints and all actual evidence are in [the durable record](evidence/concentration-author-review.json).
