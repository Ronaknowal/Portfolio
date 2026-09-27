# Gaussian mixtures: intuition at each new operation

Author review: 26 September 2026. The complete production lesson was read, including deeper branches, code explanations, eight practices and references. Production JSX is the content checkpoint. Independent and browser review remain pending.

| Transition / local heading | Existing support, gap and action | Representation / practice |
| --- | --- | --- |
| §1–2 overlap → selector → density | Sufficient flower task, SelectorFigure, normalization and area. Retained. | Fixed two-bell example. |
| Reverse selector → responsibility versus density | Sufficient contributions and x=0/2/8 counterexample. Retained. | ResponsibilityLab; changed-weight checkpoint/practice 1. |
| §3 fixed parameters → fitting; missing memberships → E/M | Sufficient likelihood coupling and explicit frozen state. Retained. | Full cycle, AllocationFigure and changed-observation checkpoint. |
| Counts → mean → variance around new mean | Sufficient weighted first/second moments. Retained. | Practice 2. |
| §4 collapse → constrained floor; logs; stopping | Sufficient exact constrained maximizer, underflow and convergence distinctions. Retained. | CollapseFigure, complete program, EmStepLab, practice 3. |
| §5 scalar spread → covariance geometry | Gap: matrix formula before its geometric operation. Added circular-cloud stretch/rotate/undo bridge. | Existing equal-distance counterexample, ellipse lab, outer product and practice 4. |
| Covariance restrictions and parameter counts | Sufficient table and CovarianceGalleryFigure. Retained. | Shape/count reasoning. |
| §6 task and complexity selection | Added why training fit rewards both real modes and accidental narrow components before AIC/BIC. | Exact penalty comparison and held-out selection retained. |
| Starts, renaming, collapse diagnostics | Sufficient objective/identity distinctions. Retained. | Full Iris program and narrow-component diagnosis. |
| §7 split, APIs and results | Sufficient fixed IDs, training-only scaling, output meanings and inconvenient outcome. Retained. | CandidateFigure and practice 5. |
| §8 sampling versus averaging | Gap: verbal distinction without visible consequence. Added variance-budget figure: selection variance 5 versus independent-average variance .5. | Original within/between calculation and sampling code preserved. |
| §8 covariance validity and resource costs | Sufficient negative-logdet counterexample and concrete storage accounting. Retained. | Practice 3. |
| §9 indicators → Q → entropy bound | Added exact row (.3,.1), posterior (.75,.25), equal .4 ratios and a non-touching bound. | Existing full proof/BoundChainFigure; practice 6. |
| §9 raised bound, KL gap and convergence | Sufficient equality chain, frozen q and unchanged E-step likelihood. Retained. | Missing-equality diagnosis in practice 6. |
| §10 k-means limit | Sufficient shared-variance/equal-weight restrictions and ties. Retained. | Sharpening table and practice 7. |
| §10 observe u → predict v | Added two-operation bridge, four-row trace and within/between conditional variance. | Means .5/3.5, within .75, total variance 3; practice 8 changes u. |
| §10 Bayesian mixture | Added responsibility versus parameter-uncertainty distinction and variational approximation purpose. | Existing six-component prior-sensitivity program retained. |
| §11–12 transfer/readiness/references | All eight tasks/solutions and source annotations retained. | No fixed figure quota. |

## Research

Read [Ma and Ng's EM notes](https://cs229.stanford.edu/notes-spring2019/cs229-notes8.pdf), Jensen's inequality and E/M bound/equality chain (pages 1–7). Its construction-first progression motivated an original two-contribution example. No wording or image was copied. Existing Bishop and Stanford learner alternatives remain relevant. New variance and conditional examples are independently derived from the lesson's declared distributions, not purported measurements.

## Visual and verification contract

The mixture-variance-budget figure shares a 0–5 scale: within variance 1, between variance 4, independent-average variance .5. Amber pieces have text equivalents. It is static because the comparison is selecting versus averaging; existing live density, EM and covariance labs remain. Inline SVG text and maximum width are explicit; actual geometry closure belongs to integration.

Author-checks.json independently recomputes bound values and variance decompositions, parses JSX and binds source hashes. Programs, fitted arrays and numeric engines are unchanged. Browser and independent teaching review remain separate.
