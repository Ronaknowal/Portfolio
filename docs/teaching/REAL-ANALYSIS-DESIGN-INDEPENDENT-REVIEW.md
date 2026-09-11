# Real Analysis — independent design and draft-model review

Reviewed 11 September 2026 by the Conditioning52 author at the parent's request. This is a bounded assessment of the complete design, individual brief, current pure models and all 13 displayed native programs. It does not review a finished body, lab rendering, final author freeze or production integration. No root-owned source was edited.

The exact inspected hashes, actual execution time and complementary checks are in [results.json](../../scratch/real-analysis-design-independent/results.json). Reproduce with [verify-real-analysis-design-independent.mjs](../../scripts/verify-real-analysis-design-independent.mjs) and its [Python oracle](../../scripts/verify-real-analysis-design-independent.py).

## Scope and mathematical assessment

The proposed ownership is coherent. Real-number completeness supplies the interval theorems deferred by Calculus47; the measure/probability branch has explicit review requirements and does not silently become part of the elementary prerequisites. The lesson distinguishes a theorem's all-tail or all-domain claim from its finite numerical illustration. The planned derivative theorem includes the needed convergent base value, and finite-interval integration does not silently become an infinite-domain assertion.

The finite read found no substantive mathematical defect in the strict sequence index, supremum/maximum distinction, nested-interval argument, subsequence/Cauchy route, compact-interval consequences, harmonic block counterexample, function-series endpoint distinctions, triangular-family integrals, Fourier jump obstruction, derivative-interchange examples, Bernstein moment/error proof, or the typewriter/UI constructions. The changed Bernstein integral report correctly distinguishes an insufficient sufficient bound from an actual failed tolerance. Final prose must still supply the promised proofs; a sound plan is not proof that they have been implemented.

As an external cross-check, the reviewer inspected the relevant definitions and derivative/basepoint, integral and interior power-series statements in [Lebl's pointwise/uniform section](https://www.jirka.org/ra/html/sec_puconv.html) and [interchange section](https://www.jirka.org/ra/html/sec_liminter.html). The design's extension to uniformly small differences of unbounded functions is appropriately distinguished from the bounded-function uniform norm. This is a scoped source comparison, not a claim to have reviewed the complete book or videos.

## Complementary execution

All 13 actual stored Python programs reproduce their displayed stdout. The additional checks use different mathematical constructions where useful:

- Exact squared-endpoint and dyadic-width invariants for 15 changed brackets, including the maximum browser depth and target5; the actual native bracket helper agrees.
- Enumeration of the first valid strict-tail index for 27 rational tolerances, with actual native and chosen-index comparisons.
- Exact piecewise Simpson integration of the actual triangle helper and its square for 12 changed families; Simpson is exact on the linear/quadratic pieces, independently recovering the model's area and squared L2 values.
- Thirty changed Bernstein inputs, including endpoints and degree256, checked with exact binomial weights, first/second moments and the squared Lipschitz inequality.
- Six actual report cases checked by expanding the Bernstein polynomial into monomials and integrating coefficients directly, rather than repeating the report's equal-basis-integral identity.
- Exact dyadic left, midpoint and excluded-right boundaries for blocks1–5 of the actual typewriter helper.

These checks cover the stated finite fixtures; they are not a full final model suite, arbitrary numerical-range certification or browser review.

## Reported arithmetic-range finding

The inspected draft accepts `bernsteinApproximation(256, Number.MIN_VALUE, 0.3, 1)` but returns `pointwiseBound: 0`. The exact mathematical bound is positive (approximately `1.39e-163`); `x*(1-x)/n` underflows before the square root. This is outside ordinary slider increments, but it is inside the exported helper's accepted input range. The design's theorem is sound; the numerical display contract needs a narrow repair.

Reported to root: evaluate the square-root factors before dividing by `sqrt(n)`, or explicitly restrict the supported arithmetic range. Also guard a nonzero slope and an interior x whose final bound is still zero, since an extremely small positive slope can make the mathematical result genuinely unrepresentable. Exact x=0/1 or slope0 should retain their genuine zero result. The original observed state and source hash are preserved in the JSON. Repair verification is pending at the time of this initial record.

Root retains responsibility for the correction, final theorem wording, complete author/browser checks and a separate final independent review. No shared metadata or production file was changed by this reviewer.
