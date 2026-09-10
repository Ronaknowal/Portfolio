# Rate-Distortion Theory — bounded independent review

10 September 2026. Read-only mathematical/teaching cross-review with one narrowly scoped numerical-contract finding, corrected by the author and independently retested. No other actionable issue was found within the inspected scope. Author native/browser verification and root production integration remain separately attributed; this is not a full reimplementation, proof of all possible numerical inputs, or user acceptance.

## Inspected source and mathematical reasoning

Read the full scoped design and lesson body, all core pure models, and the complete finite-optimizer, byte-serialization and Gaussian-allocation programs. Reviewed the ten changed practice solutions. The relevant complete source snapshot is fingerprinted in the independent result below; root's final author freeze should be compared with that identity.

- **Finite rate and coding endpoint:** the three-bit example sends a codebook index and loses specific source distinctions. Expected distortion, worst positive-probability block distortion, fixed-width rate and framing overhead are separated. The converse uses index cardinality, data processing, iid single-letterization and convexity. The asymptotic theorem explicitly requires distortion strictly above the minimum endpoint and appropriate integrability. The finite exact-zero-error counterexample for a biased full-support source correctly needs all 2ⁿ reconstructions; it does not confuse expected-length lossless coding with fixed-width finite coding.
- **Binary construction:** the error bit is independent of the reconstruction in the attaining backward channel. The biased forward false-positive/false-negative probabilities differ, as expected. The piecewise function stays at zero past the best-constant threshold rather than rising when binary entropy falls after .5. Budget, actual distortion and the multiplier’s inverse distortion units are kept distinct.
- **Finite BA objective and bound:** the decomposition of the temporary-marginal objective adds KL from the actual output marginal. The two coordinate minimizations are valid for the finite model. Applying Jensen to the ratio of normalizers gives `f(u) ≥ f(r) − log(max g)`, including output symbols excluded by a zero-support initialization. The bound is on the weighted objective, not separately on rate and distortion. A loop cap or unchanged restricted-support iterate is not declared proof of global convergence.
- **Gaussian construction and allocation:** the entropy bound is accompanied by its density/integrability qualifications; a backward independent Gaussian error attains it. The forward conditional mean shrinks toward the source mean, with variance `D(1−D/variance)`. Vector error is a sum, rates are bits per vector, variance caps are explicit and the covariance transform requires joint Gaussianity for independence.
- **Fidelity and actual format:** independent random signs match the marginal law while doubling MSE compared with the conditional-mean output. This is not presented as an unrestricted perception coding theorem. The serializer reconstructs from only its bytes and shared protocol; it counts the four-byte header, checks payload length/padding and distinguishes a smaller payload from a smaller complete tiny file.

Independently cross-checked [MIT 6.441 Chapter 24](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/aaa8d18ddecde45f97134d3f3dcee4a3_MIT6_441S16_chapter_24.pdf), especially Theorems 24.1–24.2 and the random-codebook discussion on the first three pages. Actual parsed text was inspected; extraction loses some formula formatting, so the source equations were also checked by derivation. No full PDF visual review or alternate-video playback is claimed by this reviewer.

## Finding and verified correction

The original exported `gaussianAllocation([1,1], Number.MIN_VALUE)` accepted a strictly positive total budget but its division underflowed the water level to zero. It returned zero component errors and infinite rate, whereas the mathematical positive-budget allocation has finite rate. This was outside the visible slider fixture, but inside the accepted helper domain. The displayed Python allocation also used a fixed-100-iteration bisection whose resolution could overwhelm a very small budget.

The author repaired the JS accepted-range contract to reject an unrepresentable positive allocation/aggregate budget. The displayed Python implementation now solves the active interval directly, computes rates by log differences and applies matching numerical guards. This reviewer did not edit those production files. The updated JS and actual displayed Python both reject the unrepresentable smallest-positive budget and retain finite correct output at a representable `1e-300` budget. Ordinary lesson control choices are unchanged.

## Complementary independent checks

Command: `node scratch/review-rate-distortion-independent.mjs`, invoking `scratch/review-rate-distortion-codec.py`. Actual evidence [`scratch/rate-distortion-independent-review/results.json`](../../scratch/rate-distortion-independent-review/results.json) passed **2026-09-10T17:38:05.669Z**.

| Scope | Independent evidence |
| --- | --- |
| BA whole-alphabet bounds | 135 biased/fair binary problems with analytic optimum, varied cost/multiplier and full/zero initial support. The analytic optimum lies between reported lower/upper values, and the gap covers possible improvement. |
| Objective invariance | 270 complementary relabeling and per-source-row constant-cost shift comparisons. A row constant preserves the conditional optimum while adding its known expected weighted cost. |
| Binary channels | 42 source/budget pairs, including deterministic sources, exact zero distortion, threshold and beyond-threshold cases. Marginals/error are checked; interior cases explicitly verify backward error independence. |
| Gaussian allocation | 35 states checked by enumerating every possible saturated-component subset and solving its remaining equal-error equation, independently of the production sorted sweep. Additional representable-tiny and underflow-rejection cases pass. |
| Actual learner codec | All 2,047 binary strings of lengths 0–10 execute the displayed `compress`/`decompress`. An independent integer-bit assembly checks every encoded byte, header, majority decision, decoded length and partial-block truncation. 2,046 nonzero unused-bit mutations reject. |
| Changed native Gaussian | Five changed variance/budget cases execute the actual displayed allocation helper, including zero variances and a `1e-300` budget; the underflowed positive budget rejects. |

The selected four-word practice code and its index order were checked independently. These checks complement rather than repeat the author's broad numeric, complete-stdout and browser suites. No browser suite was run by this reviewer.

Reviewed body SHA256 `718d928d3fe4f7463b3c9bb8d7fc7ff867944a1220d529be578d7deef2e1629e`; model `c783813d848028f3d218848bbcd7411a54b33ee73c968e53a7d7885835bbe886`; examples `faae6b625572f9609f61fd27b0f5959f8cd47c20a18b8360ea225047fb010a71`. The independent JSON includes the lab and stylesheet hashes as well.

The author confirmed the final **2026-09-10T17:40:30.503Z** freeze in `docs/teaching/evidence/rate-distortion-author-review.json`. Its body, model and example hashes match this inspected snapshot. [RATE-DISTORTION-VERIFICATION.md](RATE-DISTORTION-VERIFICATION.md) owns the author's completed native, three-width browser and final displayed-program checks. The identified numerical-range defect is closed; no further action is requested from this bounded review.
