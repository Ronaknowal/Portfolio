# Variational Inference — independent mathematical and source review

Completed 10 September 2026 by a different author from the lesson implementation. This bounded review read the entire current lesson, all seven programs, eight substantial practice solutions, core model functions, and the design/author verification records. It complements the [author verification](VARIATIONAL-INFERENCE-VERIFICATION.md); it does not repeat or claim the author's full browser run. No lesson source was edited by this reviewer.

## Findings and disposition

No unresolved material mathematical or pedagogical defect was found. Two valid-input numerical defects were reported to the author before changes, then repaired and independently rechecked:

| Contract | Observed before repair | Final disposition |
| --- | --- | --- |
| Finite ELBO/KL and actual support | `finiteElbo([1,0], [1e-310,1])` returned finite ELBO −713.8013788281542 but infinite KL: forming `q/posterior` overflowed before its logarithm. Normalizing a still-positive minimum-subnormal joint against a large total could also round the posterior to zero and falsely imply absent support. | The author now computes the log ratio from `log q − log joint + log evidence`, with support failure determined by the original exact zero joint weight. Independent 80-digit Decimal calculations pass both cases, ordinary weights, a mixed q, actual support mismatch, and zero-q/zero-target conventions. A posterior readout can still underflow as a floating-point normalized value; the KL support decision no longer relies on it. |
| Gaussian determinant ratio | With zero means, `C=1e-100 I` and `Σ=1e100 I`, both determinants are positive and finite, but their quotient overflowed, returning infinite KL. The true finite result is approximately 459.51701859880916. | The author replaced `log(detΣ/detC)` by separate logarithms. Independent NumPy solves and signed log determinants confirm the result. This is a bounded teaching implementation; the check does not certify arbitrarily ill-conditioned covariance arithmetic or determinants that themselves overflow/underflow. |

These counterexamples are accepted helper inputs beyond the current fixed interactive fixtures. They do not establish that normal slider states were broken. Pre-fix model SHA256 was `e17c198b9b2f0416aaf7e78044af6931ccc539aa93151e57785ad3e000678257`; repaired model SHA256 is `75401daaae5e5fb3356129baa7e6e23c29b1594d59c8176d43379dbcb2ba1538`. The lesson body, examples, lab component, CSS and blueprint hashes did not change during those repairs.

## Mathematical and teaching review

- The fixed-joint ELBO identity, unknown evidence constant, support direction and cross-family comparison are correct. Exact same-joint objectives are comparable; dropped model-dependent constants, stochastic estimates and absolute KL require separate care. Positive q on absent target support and absent q on positive target support are not conflated.
- The correlated Gaussian derivation gives reverse-KL mean-field variance `1−ρ²`, not the true marginal variance. Sum/difference decisions demonstrate that underestimated individual marginal variances do not imply underestimated variance of every reported quantity. The unit Mahalanobis contour's 39.35% two-dimensional mass and equal coordinate scales match its representation.
- CAVI's normalized exponential of the expected log joint is derived under its finite-normalizer/family assumptions. The shifted Gaussian updates and worked values are consistent. Monotonic exact updates are distinguished from global optimality and from stochastic optimization. The remaining family error is visible even as the mean-coordinate error contracts.
- The two-mode density calculations compare specified candidates rather than claiming a global optimum. The sign probability exposes a practical discrepancy that an objective value or posterior mean alone hides. A trapped MCMC chain is not described as an independent certificate of correctness.
- Reparameterized and score derivatives use the correct mean/log-scale coordinates. Entropy's log-scale derivative is included. The matched-target zero-score example is explicitly restricted to a normalized target; an unknown evidence constant changes the unbaselined score estimator's variance. Fixed-support, derivative-interchange, discrete-latent and moving-boundary cautions are adjacent to the claims.
- Minibatch replication applies to the likelihood terms once per sampled observation, while global prior and entropy contributions remain unreplicated. The global/local conjugate SVI formula is framed as additional structure, not a claim that every minibatch method has that update. Amortization shares an inference mapping across separate cases; it does not require identical posterior parameters or merge the cases into one latent variable's dataset. Posterior predictive variance includes observation noise.
- The eight changed-data/assumption tasks have useful separate hints and explained answers. Restricted-family optima, support mismatch, negative correlation, a coordinate update, missing-mode decisions, changed gradient inputs, minibatch bias and changed measurement noise agree with the lesson contracts. The seven actual programs are complete and interpreted; the finite stochastic optimizer does not claim monotonic progress from its printed results.

The design preserves the old mixture example's `.877/.689/.689` candidate results and develops its former CAVI/reparameterization/minibatch/amortization mentions into actual mechanisms. Topic-specific diagrams reveal probability allocation, covariance geometry, a coordinate path, mixture density and noise-to-gradient transformations. No generic visualization quota was used as a review criterion. This is a finite source/proof review, not a beginner user study or a complete independent replication of all referenced papers.

## Complementary independent calculations

Run:

```powershell
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-variational-inference-independent.py
```

The passing result is saved in [durable numerical evidence](evidence/variational-inference-independent-review.json); the original run output is in `scratch/variational-inference-independent-review/results.json`.

- Six finite log-domain/support cases use Decimal arithmetic at 80 digits, preserving represented float inputs with `Decimal.from_float`, including the smallest positive subnormal. Exact support mismatch stays infinite; finite log evidence equals ELBO plus KL where defined.
- Four Gaussian cases use NumPy linear solves and `slogdet`, including nonunit and correlated covariances, the extreme determinant-ratio case, and a nonorthogonal invertible affine transformation. The transformed and original KLs agree.
- Three changed target Gaussian models use 12-node Gauss–Hermite quadrature and an independent finite-difference objective. Both gradient estimators' integrated values, the model's exact derivatives, and 49 actual seeded per-draw contributions agree. These complement the author's adaptive-integral checks with different target means/scales and a distinct integration method.

An initial independent harness invocation rejected its own transformed covariance because floating-point matrix multiplication produced slightly unequal off-diagonal entries, violating the helper's exact-symmetry input contract. The fixture is now explicitly symmetrized before calling the helper; no lesson source change was needed for that harness issue. The final run passed. The author separately reported rerunning its model/program/SciPy suite and targeted 1440/320 browser checks after the numerical repair; those are attributed author checks, not reviewer-run browser evidence.

Root owns final production integration and curriculum state. User acceptance remains separate.
