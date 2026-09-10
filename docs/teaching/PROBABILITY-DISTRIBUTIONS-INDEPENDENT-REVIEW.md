# Probability Distributions & Bayes' Theorem — bounded independent review

Reviewed 10 September 2026 by the independent teaching/mathematics reviewer while the author ran native and browser checks. This is a finite source review, not final publication, browser or integration verification.

## Inspected scope

Read the complete current lesson body, its nine standalone Python examples, the complete pure probability models and relevant lab rendering/contracts, plus `PROBABILITY-DISTRIBUTIONS-BAYES-DESIGN.md`. Checked event/conditional denominators, total probability and Bayes direction, class-conditional versus marginal independence, copied evidence, discrete count sampling/support, finite-population variance, density units, mixed CDF endpoints, Poisson/exponential hypotheses, normal/CLT qualifications, continuous-density conditioning and all eight changed practice solutions. The author was already replacing visually awkward spelled-out Unicode exponential superscripts; that work was not duplicated here.

## Findings and resolution

1. **Concrete null-state wording:** `BayesPopulationLab` initially said an undefined posterior meant the result had zero probability “under both classes.” Counterexample: `bayesPopulationState(0,1,0)` gives zero total positive evidence, but `P(+|H)=1`. The author confirmed the message was changed to zero total probability under the current prior and class-conditional model. The lesson's separate checkpoint explicitly assuming both class-conditional probabilities are zero is valid.
2. **CDF endpoint precision:** the first CDF description said it “reaches1 in the far right tail.” Requested the limiting statement as x→±∞; many unbounded distributions never reach1 at a finite x. This is a wording precision issue, not a defect in the implemented CDF models. The author confirmed the limiting statement is now applied.
3. **Parameter-domain precision:** requested explicit λ>0 for exponential waits and0<p≤1 for the geometric first-success/finite-mean formula. The interactive models already restrict their relevant parameters appropriately. The author confirmed both domain qualifications are now explicit and owns the final source snapshot.

No additional actionable mathematical defect was identified in the bounded read. In particular, the mixed singleton interval[0,0] correctly includes the atom; the continuous part is not falsely normalized to1; zero evidence is null rather than an invented posterior; repeated evidence retains both class-conditional assumptions; the finite-population covariance calculation yields the stated correction; and density likelihood units cancel in the continuous Bayes example. The changed practice answer at log(3)/2 minutes is correct.

## Complementary independent calculations

Ran `node scratch/review-probability-independent.mjs`; actual results are saved in `scratch/probability-independent-review/results.json`.

- Four changed-prior Bayes tables and the zero-evidence wording counterexample.
- Twelve mixed-distribution interval cases, including atoms0/.2/1, singleton endpoints and seconds/milliseconds transformations.
- Three copied/fresh evidence settings checked by explicit latent-branch enumeration.
- Nine no-replacement count settings checked by enumerating six-object subset bitmasks, including empty/full draws and all/none marked.
- Nine Poisson-count/exponential-wait settings, including zero observation duration, quantile inversion, and displayed mass plus omitted tail normalization.
- Independent equality and side-of-boundary checks for the fast/slow continuous posterior at log(3)/2 minutes =32.9583686600 seconds.

These checks complement the author's larger native oracles; they do not replace that suite or establish every possible floating-point input behavior. No production source was edited by this reviewer. No browser suite was rerun or screenshot inspection claimed here. Final author verification and production integration remain the root author's responsibility.

All three scoped findings are author-confirmed resolved. This closes the bounded source review; it does not claim final author browser/integration completion.
