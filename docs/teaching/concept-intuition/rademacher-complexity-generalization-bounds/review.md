# Rademacher complexity: conceptual-transition review

26 September 2026. Read all 992 original JSX lines, all eleven sections, eight worked exercises and references. The core already has eighteen specialized figures and four live investigations; its early mechanism/proof sequence is retained. Improvements target compressed advanced branches, not an additional generic introduction. Author-only status; independent reading and rendered browser checks remain pending.

| Location/concepts | Disposition |
| --- | --- |
| §1 fixed-sample noise, inner best response, average, nested classes | Retained full enumeration, matrix figure and interactive class comparison. |
| §2 convention, absolute values, singleton/translation/scaling/duplicate nulls, sample size | Retained explicit counterexamples and qualifications. |
| §3 loss versus score class, theorem, probability scope, binary half identity, vacuity | Retained dedicated class/theorem figures and exact algebra. |
| §4 ghost sample, pair swap, splitting supremum, concentration and confidence allocation | Retained full proof and diagram; factors accounted for explicitly. |
| §5 Euclidean support, Jensen, energy, geometry, kernel Gram, Massart, Sauer and ℓ1 duality | Retained derivations, diagrams and sign enumerations; no fit/engine changes. |
| §6 contraction, range/Lipschitz distinction, ramp, margin/scale tradeoff, finite family | Retained loss slopes, ramp diagram and live bound. |
| §7 exact enumeration, complexity costs, Monte Carlo correction, optimization direction | Retained inspectable programs and finite-simulation accounting. |
| §8 representation/fitting/selection/assessment, norm constraint, vacuity, dataset limits | Retained complete practical pipeline, original coefficients and empirical results. |
| §9 convex hull | Retained weighted-correlation proof and figure. |
| §9 covers/chaining | Gap fixed: nested 3/5/9-point covers of constant prediction vectors, empirical RMS metric, signed telescoping corrections and residual. Native figure uses common axes; it is explicitly not a bound. |
| §9 localization/Gaussian complexity | Gap fixed for local radius: illustrative c√(r/n) fixed point c²/n with assumptions and confidence caveat. Gaussian distinction retained. |
| §9 neural norms | Gap fixed: scalar ReLU parameter rescaling table; identical function, changed individual norms, preserved product; no universal-bound claim. |
| §9 PAC-Bayes/stability | Gap fixed for KL object: three equal-mean distributions with different explicit KL values and randomized-predictor loss distinction. Stability already explains the algorithm perturbation, supremum and conditional theorem. |
| §10 practice, §11 synthesis/provenance | All original eight exercises, solutions, references, dataset/programs and next-topic link preserved. |

## Research and representations

Read Rebeschini's [lecture 3](https://www.stats.ox.ac.uk/~rebeschi/teaching/AFoL/22/material/lecture03.pdf), pages 1–5, on loss contraction, norm constraints, convexity and network recursion. Read Bartlett–Bousquet–Mendelson [local complexity](https://arxiv.org/pdf/math/0508275), section 3 definition/lemma/theorem and accompanying assumptions (PDF pages 9–11), plus the returned introduction/context excerpts; not the entire 43-page paper. This supported the explicit distinction between a fixed-point illustration and a theorem justified by a variance relation. An attempted Mohri lecture 10 fetch failed; no reading claimed for it. Existing longer alternatives remain in the lesson. No videos watched.

The new cover figure, ReLU and KL examples are original finite calculations, not sourced measurements. Covers are of the class fₜ=(t,t), t∈[0,1]; nearest-grid radii are half the spacing. The local envelope is hypothetical, clearly labeled. No recorded fit, simulation, source data or existing diagram was changed. Author script parses changed JSX, checks these arithmetic groups and matches retained original engines/examples/labs/figures to baseline hashes.
