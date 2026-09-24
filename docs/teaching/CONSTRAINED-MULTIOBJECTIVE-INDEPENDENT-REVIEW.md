# Constrained & Multi-Objective Optimization — independent source review

10 September2026. Root reviewed the complete lesson's reasoning and changed practice, and the core pure-model implementations. The separate [author record](CONSTRAINED-MULTIOBJECTIVE-VERIFICATION.md) owns the executed eleven programs, independent numerical oracles and full browser interaction evidence. This cross-review does not claim an independent rerun of all those checks.

The coupled simplex projection and sequential-projection counterexamples agree with the stated feasible set. Existence uses a bounded comparison region for the distance minimization on a closed set; convexity supplies uniqueness. The projection variational inequality gives the stated descent coefficient1/η−L/2; constrained stationarity is correctly distinguished from a zero ordinary gradient.

The quadratic penalty, hinge exactness threshold and rationalized barrier root match their examples. The two-block ADMM signs, scaled multiplier, primal/dual residuals and tolerances use the declared consensus problem. Its convergence discussion includes the required convexity/saddle/exact-subproblem conditions and does not infer convergence from a zero initial primal residual or promise unrestricted primal-iterate convergence.

Pareto dominance distinguishes identical objective vectors from incomparable points. Positive weighted sums, zero-weight weak guarantees, unsupported points, unit changes, epsilon constraints and lexicographic priorities have different contracts. Root clarified one epsilon-tie sentence: bounds taken from an actual Pareto target force all primary-optimum ties to match that target's complete objective vector; general chosen bounds can admit dominated ties. The final source preserves this distinction.

The minimum-norm convex combination of gradients gives the common-descent argument; a zero combination is only a first-order certificate, with the constant-objective and constraint-normal qualifications visible. The continuous extension atx=3 is explicitly outside the earlier bounded decision problem. Reported latency/accuracy metrics and changed exercises retain their intended units and conventions.

A separate source finding was invalid paragraph nesting because Prose renders a paragraph. The author repaired51 groups into80 separate Prose paragraphs, preserving the epsilon amendment and every other text/formula/model value. Tag-only normalized-AST assertions passed. The final source hash is `2dbf3150a4e5fdbef457e0670de54ccc034016f02c6f675f78b39415763a824f`. Actual1440/390/320 console, DOM and reading checks passed; root also opened the final320px epsilon passage and confirmed its reading flow. Evidence is `scratch/optimization-paragraph-repair/{source-results,browser-results}.json`.

No further actionable mathematical defect was found in this bounded review. Source hashes, production loading and route checks remain separate completion requirements; user approval is not implied.
