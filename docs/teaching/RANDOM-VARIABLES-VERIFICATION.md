# Random Variables, Expectation & Covariance — author verification

Mathematical & Statistical Foundations position 48; stable ID `random-variables-expectation-covariance`. The complete scoped lesson is implemented and author-verified. The final evidence packet is [random-variables-author-review.json](evidence/random-variables-author-review.json). Independent review, production integration and user acceptance are separate parent-owned stages.

## Scope and learning route

The initial plan in `scratch/random-variables-initial-plan.json` showed a planned topic without a published source or original program. No old lesson was removed. The inherited two-dice exercise and promised finite/joint/covariance coverage are retained. The approved [design](RANDOM-VARIABLES-LESSON-DESIGN.md) keeps the title, position, prerequisites and next topic Sampling, Measurement & Experimental Design.

The route begins with experiment/outcome/event, a fixed numerical function and its induced law. It develops PMFs/CDFs and endpoint events, expectation/LOTUS/linearity/indicators, variance and optimal constant squared-loss prediction, joint/conditional laws, covariance/correlation/independence, shared-error combinations, covariance matrices and affine prediction, conditional moments and average-risk improvement, two-branch continuous transformations, moment existence, and population-versus-sample quantities. Bernoulli/binomial laws are introduced locally before use. Eleven changed exercises have separate hints and explained answers, plus an early copied-coin checkpoint. Fourteen native programs have a visible question, complete source, exact captured stdout and interpretation.

## Representations and numerical contracts

These seven forms have distinct learning purposes; they are not a quota for other lessons.

| Investigation | Visible mechanism and active contract |
| --- | --- |
| Outcome mapping | Four labelled coin outcomes retain their probabilities as a chosen numerical rule groups them into a PMF. The threshold table computes the CDF, including endpoint mass. Independent head probabilities use integer percentages; actual zero masses remain zero. |
| Mean and squared loss | Geometric residual squares share a value scale. A weighted contribution table links them to the exact loss curve and its mean minimizer. Three populations include a constant law; the mean need not be an observation. |
| Joint covariance | Actual population pairs and their masses appear beside the joint/marginal table. Centered products determine color; the selected joint cell is compared with the product of its marginals. Nonlinear dependence has an explicit zero-versus-2/9 witness. Zero scale produces undefined correlation. |
| Shared noise | A common disturbance feeds both channels; coefficients visibly add before its variance contribution is squared. Independent local contributions and the combined law update together. Baseline, units, and estimand are explicit; a difference is not silently substituted for an average. |
| Conditional moments | Conditional ranges and their means share one number line with the overall mean. Grouped outcomes, signed residuals, conditional statistics, weighted decompositions and risk comparisons remain linked. A zero-mass group has no identified conditional law and contributes no mass. |
| Squared uniform | Two actual preimage intervals feed one output interval. Input lengths, branch masses, CDF difference and the zero-width/zero-atom case agree. Endpoints use hundredths and remain ordered; no fake finite density spike is plotted. |
| Sample mean | Independent and copied sample-mean laws share axes. Probability comes from exact finite convolution, not a sampled histogram. The complete program independently uses binomial coefficients. n=1 and p=0/1 are explicit coincidence/degenerate states. |

Public finite helpers use 1–64 matching values in [-1000,1000], probabilities summing to one within `1e-12`, with positive mass at least `1e-12`. Accepted totals are normalized. Centered sums avoid cancellation in a raw difference of moments; anchoring at a positive-mass value makes constant laws exactly constant. Unrepresentable positive squared contributions and positive variance below `1e-280` are rejected rather than converted into a misleading zero. Correlation divides by the product of separately computed standard deviations, avoiding the potentially underflowing product of variances before taking a square root. No UI input comes near these numerical limits. The binomial helper separately supports its finite tail range, down to approximately `1e-32`. Snapshots are recursively frozen; input arrays remain unchanged. Displayed decimals use six significant digits and probability tables identify that rounding.

## Native and model evidence

Run from the repository root:

- `scratch/lesson-tools/Scripts/python.exe scripts/build-random-variables-examples.py`
- `node scripts/verify-random-variables-models.mjs`
- `node scripts/review-random-variables-lesson.cjs`
- `node scripts/review-random-variables-lesson.cjs --reading-only`

The builder uses installed Black only as development tooling; it verifies normalized Python AST conservation before executing all 14 programs. Learner programs use the Python standard library only. The verifier exports the actual JavaScript states and actual displayed programs, then runs `verify-random-variables-native.py` against independent Fraction laws, closed-form binomial coefficients, high-precision preimage mass and polynomial quadrature, and NumPy covariance propagation. It verifies actual native helper calls on changed values, not a rewritten substitute for those functions.

The current numerical pass contains 43,738 counted comparisons/program checks: 144 outcome/rule states; 87 loss states; 140 joint/scale/shift states; 720 common/local/coefficient combinations; all 101 group weights; all 5,151 ordered endpoint pairs; all 1,616 bounded sample-mean states; 15 additional finite-law boundaries; and complementary native/exercise calculations. The count includes per-state moments/cells rather than claiming 43,738 unrelated proofs. The maximum absolute discrepancy is approximately `5.82e-11` in a large second moment; comparisons use both absolute and relative tolerances. Tiny positive variances and binomial tails receive relative checks rather than a permissive absolute-only test. Twenty-six invalid calls are rejected; constant correlation, an underflow-prone variance-product correlation and frozen nested snapshots have separate regressions.

Native changed cases include 180 iid paired sequences for the covariance estimator, nonuniform group weights and null groups, alternative sensor amplitudes, binomial sizes up to 30, 13 transformed moment integrals, rectangular covariance propagation and stable shifted streams. Changed practice checks include the new finite law/loss, optimal calibrated weight, impossible correlation matrix, conditional variance, mixed linear variance, and equal-moment laws with different tail probabilities. General theorems are justified by the displayed proofs and hypotheses; finite numerical checks do not prove them.

## Browser and reading evidence

The author packet records the final actual-font desktop/390/320 browser results and exact source fingerprints. The suite checks program/questions/stdout identity, all 12 route anchors, state-derived readouts, PMF conservation, zero-mass/zero-correlation cases, group and endpoint transitions, repeated versus copied laws, resets, real range/select/button/disclosure keyboard interactions, intended font loading, page/console/network errors, equation widths and SVG text bounds. Screenshots are captured during ordinary reading and meaningful changed states; final selected images are actually opened and recorded individually.

Initial review found and repaired clipped chart-label margins, overly wide aligned equations, and a long unbroken practice expression. The repair uses line breaks and a two-step explanation, not reduced mathematical font size. The conditional number-line diagram was added when a first visual read showed that the grouped table alone could make within-versus-between spread clearer. Final reading rechecks the local Bernoulli/binomial bridge and final program formatting separately from the earlier behavioral pass. The public variance-floor amendment changes only rejected extreme helper inputs; every displayed interaction state was rerun through the full numerical oracle and remains identical.

The successful complete pass checks 234 interaction states per width. The final ordinary pass independently checks all 17 displayed equations, actual questions/source/output for all 14 programs and the final loaded-model arithmetic boundaries. A final focused three-width pass verifies the four-object legend caption, the local Bernoulli bridge, a precise specialist reference annotation, properly signed noise-only contribution labels and the shorter continuous-branch Mass heading. Twenty selected final or unchanged-region screenshots were actually opened; the packet distinguishes those passes rather than suggesting a late label amendment changed the algorithms or the entire original test run.

## Sources and destination work

Selected primary passages and exact reviewed scope are recorded in the design: MIT18.05 finite expectation/covariance and continuous moments; MIT18.06 covariance matrices; Duke's jointly Gaussian block-independence argument. The Harvard Stat110 official guide and MIT Tsitsiklis total-variance resource provide annotated spoken alternatives. Course/video identity was checked; full recordings were not claimed as watched. The lesson's probabilities, diagrams and synthetic measurement model are independently computed, not empirical benchmark claims or copied resource examples.

[Sampling49's destination note](topic-notes/sampling-measurement-experimental-design.md) carries independent units, repeated readings, shared systematic errors, calibrated estimands and the iid requirements behind the n−1 correction. Its author has assessed the scope. Final destination inclusion/resolution belongs to that author. No shared catalogue, registry, ledger or unrelated lesson was changed by this topic owner.
