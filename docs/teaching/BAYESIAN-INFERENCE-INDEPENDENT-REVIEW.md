# Bayesian Inference & Conjugate Priors — independent source review

10 September2026. Bounded independent review of the completed Math18 source while the author performs browser/integration work. No production file was edited by this reviewer. This record does not certify completion of the author's remaining checks.

## Inspected scope

Read the complete actual lesson `src/learn/data/topics/bayesian-inference-conjugate-priors.jsx`, all pure models in `bayesian-inference-models.js`, all11 complete example records and `BayesianInferenceLabs.jsx`. Read the individual design. Reviewed all eight changed closing solutions and the two earlier checkpoints; example prompts use a local wrapper that actually renders their question field.

The finite mathematical read found **no actionable correctness defect** in:

- Conditional independence at fixed θ versus integrated dependence from a shared θ; Beta-Binomial moments, covariance and same-mean plug-in counterexample.
- Prior/likelihood/posterior/evidence distinctions, count-versus-sequence constants, proper support, boundary modes, realized variance increases and average information gain.
- Gamma shape/rate convention and units, exposure likelihood, zero-count information, predictive negative-binomial mass and hours/minutes equivalence.
- Dirichlet mean/covariance, marginal versus simultaneous intervals, and declared-but-unseen versus undeclared categories.
- Normal precision completion, posterior-mean versus new-observation variance; explicit coupled Normal–Inverse-Gamma prior and the Student-t scale/variance distinction.
- Predictive run diagnostics versus conditioning on a fixed total, posterior predictive calibration cautions, proper-model evidence and stated expected-loss decisions.
- Noisy detection likelihood, grid cell-width/evidence convention, numerical refinement boundaries and all eight changed-input exercises.

The source supplies local bridges for the needed conditioning, normalizing integrals, units and new distributions. It does not claim the convenient conjugate families validate the observation design. No essential hidden prerequisite or contradiction was identified within this finite reading.

## Complementary independent calculations

Commands:

```text
node scratch/review-bayesian-independent.mjs
scratch/lesson-tools/Scripts/python.exe scratch/review-bayesian-independent.py
```

Passed [results](../../scratch/bayesian-independent-review/results.json), timestamp2026-09-10T15:12:48.456915+00:00. Python3.12/SciPy1.18.1.

| Calculation | Distinct reference mechanism |
| --- | --- |
|30 run-reference probability states and11 count partitions | Count positive-length success/failure block compositions by binomial coefficients, then integrate over Beta density. This does not enumerate bit masks as the application does. |
|3 predictive count variances | Numerically integrate conditional binomial variance plus variability in conditional mean, including the changed Beta(2,2)/two-visitor exercise. |
|8 Gamma predictive masses | Integrate Poisson probabilities directly against a Gamma density under two unit systems; hour and minute representations agree. |
|5 unknown-variance predictive densities | Integrate the conditional Normal density over inverse-Gamma variance, then compare with the stated Student-t law at five coordinates. |
|Changed grid practice and limiting models | Continuous quadrature independently obtains mean.4205427998, threshold tail.03629248153 and ordered evidence.0006818201453. Perfect reporting recovers mean3/7; constant reported-success probability preserves the prior mean.5. |

Review scripts: [JS actual-model capture](../../scratch/review-bayesian-independent.mjs) and [independent Python calculations](../../scratch/review-bayesian-independent.py). The review used the declared small models and inspected derivations; it did not rerun the author's comprehensive native suite, review external recordings, or conduct a second browser suite. Numerical stress outside the declared teaching bounds and observational model adequacy in a real deployment are outside this scoped check.
