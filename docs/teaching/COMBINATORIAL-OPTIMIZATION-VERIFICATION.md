# Combinatorial Optimization & Approximation Algorithms — author verification

Stable ID: `combinatorial-optimization-approximation-algorithms`, Mathematics32. This records the implemented lesson under the active 74-topic rollout. Author verification is complete; independent closure and production integration are separate. User acceptance is not claimed.

## Implemented learning experience

The original complete set-cover program and printed A/B/C versus B/C output are retained exactly. The title, identity, module membership and sequence are unchanged. [The design](COMBINATORIAL-OPTIMIZATION-LESSON-DESIGN.md) explains preservation, scope and all three incoming discoveries.

The lesson has eleven connected sections, six concept-specific investigations, three inline explanatory figures, ten complete executed Python programs, eleven independently answerable practice tasks with hints and explained solutions, one early checkpoint and fourteen mathematical blocks. These counts describe this lesson, not a template or quota.

The learner models feasibility and objective units, reads instance certificates, derives the matroid greedy theorem, follows bound-driven exact search, refunds assignments through signed residual paths, derives harmonic set-cover charges, compares LP rounding with primal-dual cover, proves the cardinality submodular bound, implements a value-scaling FPTAS and uses metric parity repair for tours. A scheduling capstone separates modeled optimality from realized operational quality.

Important preserved boundaries: mandatory bases versus arbitrary signed independent sets; feasible incumbents versus optimistic bounds; required matching size versus attained cardinality; exact weighted bipartite cover versus general weighted-cover approximation; dual feasibility versus dual optimality; zero-cost and uncovered requirements; full cover versus cardinality coverage; encoded input size versus numeric magnitude; filtered feasible Vmax; fixed metric assumptions; tiny exhaustive oracles versus polynomial implementations.

## Numerical and actual-program evidence

`scripts/verify-combinatorial-models.mjs` and the independent `scripts/verify-combinatorial-models.py` pass 50,657 comparisons over 90 matroid instances, 240 signed/rectangular/missing assignments, 220 stopped searches, 220 scaled DPs, 240 cover/coverage cases and 180 weighted covers, plus seventeen invalid contracts. The Python reviewer uses separate permutation/subset enumeration, exact Fraction bounds/scaling, NumPy rank and SciPy linear programming, including every reachable DP cell and descendant/pruning bound. This is bounded adversarial coverage, not exhaustive verification of arbitrary inputs.

`scripts/verify-combinatorial-native.py` executes the actual ten displayed code strings and compares every stdout. Changed-input verification passes:

- 768 two-by-two assignment matrices and sizes, exhaustively using costs −2,0,3 or a missing edge.
- 550 exact-search stopping cases and 440 value-scaling cases, independently enumerated for feasibility, optimum and bounds.
- 130 weighted set-cover instances and 581 budgeted-coverage cases, with Fraction charging and approximation inequalities.
- 120 weighted vertex-cover instances, including a separately solved edge-load dual and exact integer oracle.
- 36 metric routing fixtures, four specified changed learner contracts and sixteen rejected invalid inputs.

Execution environment: Python3.12.14, NumPy2.3.5 and SciPy1.18.1. Nine displayed programs use the standard library; the LP program explicitly requires NumPy/SciPy. The LP run is numerical evidence; the exact feasibility/rounding theorem is proved independently. Exact assignment and knapsack programs use integer/Fraction arithmetic; tiny tour/matching enumeration is clearly labeled exponential.

## Actual browser, reading and code quality

`scripts/review-combinatorial-optimization-lesson.cjs` passed at1440,390,320 in headless Edge152 with the actual Space Grotesk, JetBrains Mono and KaTeX fonts. At each width it operated24 meaningful states and22 keyboard-focusable controls, all six resets, every selector/range endpoint and stepper, all eleven route-anchor arrivals, all ten visible pre-code questions/code/output blocks, every practice hint/solution, local horizontal scrolling and all fourteen equations.

It checks absence of page/render errors, failed requests, invalid HTML nesting, math errors, document overflow and clipped/undersized SVG labels. The intentionally closed Vite HMR socket is separately recorded, not counted as an application failure. No other console failure is suppressed. Final minimum SVG text size at320px is14.65 rendered pixels.

Actual screenshots found and closed: hidden crossing-edge weight, overly small mobile graph labels, graph top-label clipping, three wide equations, a rotated chart label and a bound interval wrapping into an unclear arrangement. Shorter selectors and a visible narrow-screen incidence-table scroll instruction improve use without shrinking the content. A JSX literal-set rendering defect was found in the first live load and repaired; full later browser runs check the actual rendered notation.

`scripts/review-combinatorial-final-reading.cjs` checks the final bound layout and equations at all three widths after the last scoped CSS change. The narrow bound is an ordered vertical lower-limit/unknown-optimum/feasible-cost display. Final proof and program screenshots were actually opened. The durable evidence identifies opened image hashes separately from files merely captured.

Semantic topic-owned sources load through the existing individual publication mapping. Models use bounded pure computations and memoized results; there are no background timers, new dependencies, global CSS overrides, navigation reorderings or eager cross-topic imports. `scripts/format-combinatorial-optimization.cjs` verifies normalized JS/JSX AST and string conservation and CSS meaning before/after formatting. Production payload/recovery checks remain root integration work.

Independent mathematical/native review subsequently requested one wording correction: the earliest-finish exercise now correctly says it solves a different, **unweighted** objective. The final three-width reading harness checks that visible answer. Algorithms and all native code remain unchanged; the durable freeze retains the earlier hashes.

## Research and limits

The design's primary-source ledger records the substantive inspected sections: Lee's matroid proof, Roughgarden's matching/flow/cover/submodular notes, MIT's approximation and flow proofs, Cornell/ETH value scaling and MIT's lecture17 transcript. The video is a direct optional alternate route; full playback is not claimed. No performance ranking or current best-known approximation ratio was inferred from older notes.

SciPy's current official `linprog` and HiGHS documentation was additionally inspected for inequality signs, bounds, continuous variables and success/status handling. The retrieved documentation header was1.18.0; execution used1.18.1. The code declares its runtime rather than pretending those are the same version. Mathematical proofs, actual operation counts and finite computed fixtures replace invented benchmark curves.

The complete relevant source-notes are implemented/adapted here. General blossom implementation, production routing, multicommodity optimization and arbitrary non-submodular models are explicitly outside the supplied algorithms. The lesson links existing DSA practice rather than duplicating a LeetCode quota in mathematics. No claim of user acceptance, observed beginner mastery, screen-reader listening or full-goal completion is made.
