# Bayesian Inference & Conjugate Priors — verification

10 September 2026. Mathematics 18, stable ID `bayesian-inference-conjugate-priors`. Complete source is author-verified with [independent mathematical review](BAYESIAN-INFERENCE-INDEPENDENT-REVIEW.md). Production loading/integration and user acceptance are separate.

## Retained and improved learning

The entire earlier pilot was read and preserved before replacement. Its original visitor question, Beta(2,2)→Beta(10,4), credible area, prior sensitivity, 80/100 comparison, fresh sequential update and zero-success interpretation remain. The original effective density/evidence forms are retained as topic-owned implementations, without importing unrelated pilot models.

Ten connected sections now derive shared-rate Beta-Binomial prediction; compare realized prior/posterior uncertainty; include Gamma exposure and count prediction, Dirichlet compositions and support, Normal precision and outcome prediction, coupled unknown-variance Student-t inference, predictive ordering checks, proper model evidence, explicit losses and nonconjugate grid refinement. Five investigations use different mechanisms and three inline figures explain immediate relationships. Two early checkpoints and eight changed exercises have actual visible prompts; the closing exercises offer separate hints and explained solutions. All eleven program questions are rendered by a local wrapper, since the shared runnable component does not render an example’s question field.

The title and reading order are unchanged: Hypothesis → Bayesian → Concentration → MCMC. No publication count, topic identity or learner completion record changes through this rewrite. [The design](BAYESIAN-INFERENCE-CONJUGATE-DESIGN.md) records inspected primary/alternate resources and boundaries; video metadata inspection is not described as full playback.

## Numerical and native verification

Commands:

```text
scratch/lesson-tools/Scripts/python.exe scripts/prepare-bayesian-inference-examples.py
node scripts/verify-bayesian-inference-models.mjs
scratch/lesson-tools/Scripts/python.exe scripts/verify-bayesian-inference-native.py
```

Final [native results](../../scratch/bayesian-inference-review/native-results.json): 15:09:28 UTC, Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1. All eleven complete programs execute and exactly reproduce displayed stdout. Ten require only the standard library; the marked unknown-variance program uses SciPy t quantiles. Black 26.5.1 formats each program, with normalized Python AST equality asserted against its pre-format source before execution. The formatter is in the workspace verification environment, not a browser or application dependency.

**3,667 independent numeric comparisons** cover 128 Beta density/tail/quantile cases, 40 update configurations with integrated credible mass and density samples, 32 predictive batch cases, 300 Gamma coordinate/quantile cases, nine exposure updates, 27 Normal precision models and six run-reference states. SciPy distributions and continuous density quadrature provide references distinct from the JavaScript implementations. Exact Fraction sequence probabilities check every selected run distribution; 20 invalid-input cases enforce the declared teaching domain. Direct survival calculations retain tiny tails. Gamma rates are bounded to the explicit numerical domain; exposure/curve/batch/sequence work is finite and cached or memoized where appropriate.

Changed practice checks cover sequential Beta results, predictive masses, exposure/unit conversion, Normal precision, decision arithmetic and detector-grid limiting cases. For four reported successes and six failures, independent integration gives mean .4205427998, threshold tail .03629248153 and ordered evidence .0006818201453. Perfect reporting recovers Beta(6,8); a constant report probability leaves the prior unchanged.

A second reviewer read all actual body/model/lab/example files and all eight solutions. Their complementary reference counts success/failure run compositions without bit enumeration, integrates predictive variances and Gamma mixtures, and checks five Normal–Inverse-Gamma predictive densities against Student t. [Independent results](../../scratch/bayesian-independent-review/results.json) and [review](BAYESIAN-INFERENCE-INDEPENDENT-REVIEW.md) report no remaining substantive issue.

## Actual browser and reading

`scripts/review-bayesian-inference.cjs` passed in Edge at 1440/390/320, final interaction run 15:04:31 UTC. All five investigations’ meaningful changes and resets work: prior strength/no data/boundary mass, one-visitor predictive agreement and concentrated batches, zero observed events, more independent measurements and same-total ordering references. Keyboard operation, ten anchors, eleven full code/stdout blocks, eight visible exercise prompts and disclosures were checked. No lesson console/page errors, SVG label escape, display-math overflow or page overflow remained.

The final `scripts/review-bayesian-reading.cjs` pass at 15:12:18 UTC follows the added preserved 80/100 checkpoint and Python formatting. At all three widths, both early checkpoints have nonempty actual questions and keyboard-revealed answers; 20 subsequent Tab steps reach native controls/disclosures. All sixteen display equations fit. The earlier full interaction result remains applicable to unchanged models/lab behavior; the formatted programs passed fresh native execution and AST equality.

Ordinary viewport captures keep the real header/layout. Root opened the final mobile practice route, evidence lanes, shared-parameter figure, original posterior area, batch comparison, Gamma exposure, Normal parameter/prediction intervals, ordering conditional reference, narrow predictive equations and native output. They show visible mathematical consequences and explanatory context, not just a generic slider panel. Captures and structured results are in [interaction evidence](../../scratch/bayesian-inference-review/browser/) and [ordinary reading evidence](../../scratch/bayesian-inference-review/reading/).

The narrow review found long equations and an initially hidden exercise prompt. Equations were split into mathematical continuation lines without reducing text size; prompts now remain outside hint/answer disclosures. JSX comparison text is entity-escaped. Early restricted font requests failed; final authorized runs used the actual public fonts successfully. The intentional blocked development HMR socket is separately recorded; it is not a lesson runtime failure. No failed public-font requests remain in those final runs.

## Boundaries and final source

These are synthetic, bounded analytic/combinatorial calculations and explicitly approximate grid examples. No measured latency, real experiment, universal numerical-error guarantee or observed learning outcome is claimed. Conjugacy is a computation convenience, not a model-adequacy certificate. Credible mass, frequentist coverage, parameter variation, predictive uncertainty, posterior checking and decision losses retain distinct interpretations.

[Owned source evidence](evidence/bayesian-inference-author-review.json) contains the final six production-file hashes, native results and final browser/reading results. The manifest continues to load this existing topic by its stable mapping. The next production integration checks must confirm the new topic-owned dependencies and continued curriculum conservation.
