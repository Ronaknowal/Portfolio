# Exponential Families & Sufficient Statistics — bounded independent review

10 September 2026. Read-only complementary review requested by root after the Measure Theory author freeze. The Exponential Families author owns source changes, native/browser verification and post-freeze amendments; root owns integration.

## Inspected scope and conclusion

Read the complete ten-section lesson, all nine complete Python examples, the complete pure-model file, all four lab implementations and the associated string/spread/feature figures, individual design and source-fingerprint record. Checked every local practice solution, including the final monitoring-service task. This is a finite source/math/teaching review; it does not repeat the author's full browser suite or certify every general exponential-family result.

**No actionable mathematical defect was found in the inspected claims.** The lesson distinguishes sufficiency relative to a model from literal reconstruction; ordered-data probability from a count event; parameter-independent factors from model-comparison constants; full/interior natural-parameter identities from boundary and constrained cases; minimal representation from minimal sufficiency; and mean-parameter geometry from an unconditional MLE-existence theorem. Gaussian natural-coordinate signs and the two sufficient moments are consistent. GLM fixed-design summaries, exposure units, Beta prior Jacobian and properness qualifications are sound in the stated scope.

One teaching-flow improvement was reported: explicit Python3/standard-library/save-and-run instructions first appeared in section9, after the earlier runnable examples. The author agreed to move the existing setup before the first FamilyExample and retain a focused practice introduction. This changes prose location, not program/model semantics. The author's targeted first-program reading check and revised body fingerprint are recorded in its post-freeze amendment; closure passed at 2026-09-10T16:18:34.286242Z. The author checked actual first-program placement at 1440/390/320, all nine program/practice counts and error/overflow behavior, and opened all three screenshots. See `scratch/exponential-family-browser/setup-amendment-results.json` and the author snapshot’s `postFreezeAmendments`. Final body SHA-256: `64a1ae33f30b28d62cf691c12a2ba86bfab637647f3a323e2793ed534320c75d`; numerical source is unchanged.

Reviewed numerical source versions:

- `src/learn/data/exponential-family-models.js`: `b371f7f03adebd0d60f21ab7b4ac3e1568465bd0db605c9a34a5cc58d8132a7c`.
- `src/learn/data/exponential-family-examples.js`: `fc115d8d97394fb82a40931a7bdf55776015fe7d71a7505295cdb9465dd41521`.
- Initial reviewed body: `f4a0a591778f2de011efc3f25198d79f790e45e9ca0cccfbaebc6a84327d3972`, before the author-owned setup relocation.
- Author-owned complete snapshot: `docs/teaching/evidence/exponential-family-author-review.json`.

## Complementary calculations

Scripts: `scratch/review-exponential-families-independent.mjs` captures the actual model states and runnable programs; `scratch/review-exponential-families-independent.py` independently computes references. Evidence: `scratch/exponential-family-independent-review/states.json` and `results.json`, passed at **2026-09-10T16:14:31Z**.

| Changed question | Independent method and result |
| --- | --- |
| Does the Beta prior retain event mass under log-odds? | Seven parameter pairs, including endpoint-mode Beta(1,8)/Beta(8,1), with 49 point checks and21 adaptive integrals. Integrate the independently derived canonical log-density using betaln and logaddexp; both corresponding intervals, displayed window and full normalization agree. |
| Is covariance still the normalizer's curvature with different base weights? | Twelve actual model states using base weights(2,5,3), checked against independent raw first–fourth moment expansion. The covariance matrices agree and have positive eigenvalues on these finite interior states. |
| Can a constrained score vanish without full moment matching? | For eta(theta)=(theta,0), the symmetric observed distribution(.1,.8,.1) has zero first-moment residual at theta0, but second-moment residual .2−2/3. This explicitly supports the projected-score caveat. |
| Does parity preserve Bernoulli parameter information? | Exact conditional probabilities within the even-count fiber for five trials at three distinct rational p values differ. The changed statistic is insufficient. |
| Can moving support still permit a sufficient statistic? | Four Uniform(0,theta) contexts confirm equal maximums give equal likelihoods while different maximums change support. This tests the distinction from the lesson's fixed-support exponential-family form. |
| Does summary merging preserve high-offset spread? | Twenty changed split datasets near2^16, checked against exact Fraction two-pass centered sums rather than another implementation of the merge recurrence. Means and M2 agree within floating-point tolerances. |
| Are Gamma exposure updates invariant to hours versus minutes? | Three changed datasets through the actual runnable update helper, then nine corresponding Gamma interval-probability comparisons via SciPy. Both parameters and complete event probabilities transform consistently. |
| Are the changed numerical exercises sound? | Exact base-weight mean22/15 and variance86/225; four independent Beta(4,2) posterior-kernel coordinate checks; hand verification of the two triangle fits, Gaussian mergedM2=20, parity ratio, feature/exposure retention and boundary explanations. |

The prior derivation was additionally checked against the actually read sections1.1–1.2 of [Geyer's primary lecture notes](https://www.stat.umn.edu/geyer/f22/5421/notes/prior.html), with coordinate notation translated. The lesson appropriately does not adopt the source's philosophical preference for conjugacy as a mathematical necessity, and does not use a general-looking kernel as proof of properness.

## Review limits

These checks supplement, rather than duplicate, the author's execution and responsive/keyboard evidence. No browser screenshot is claimed as independently reviewed here. The review does not certify arbitrary-range floating-point safety, arbitrary invalid inputs to illustrative Python helpers, or general unbounded-family boundary geometry beyond the lesson's explicitly qualified statements. No production or shared source was edited by the reviewer. The bounded mathematical review is complete; the author-owned setup relocation is verified and closed. No outstanding finding remains.
