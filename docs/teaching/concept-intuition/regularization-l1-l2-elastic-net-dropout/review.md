# Regularization: whole-lesson concept-transition review

26 September 2026. Read the complete production lesson, all eleven sections, ten practice solutions, code explanations and references. The production JSX is the revised content checkpoint. Independent teaching review and browser verification remain pending.

| Concept / exact location in the topic JSX | Finding and action | Representation and transfer evidence |
| --- | --- | --- |
| Intro and §1 objective, intercept, loss normalization, units | Sufficient task-first comparison with an explicit objective convention; retained. | Objective-cost figure and rescaling counterexample; practice 1. |
| §2 scalar ridge, lasso and elastic net | Sufficient derivative/subgradient progression, zero threshold and live movement; retained. | Soft-threshold lab and constraint geometry expose different mechanisms. |
| §3 centering and matrix equation | Sufficient shapes, unpenalized intercept and stable solve; retained. | Full centered derivation, scratch program, solver warnings. |
| §3 coordinate residual to update | Added the missing reason to put the current feature back: target 10, current fit 8, current contribution 3 leaves 5 for this coordinate, not the ordinary residual 2. | A single-row calculation precedes the general partial-residual equation and four-row sweep. |
| §4 duplicate/correlated features | Sufficient coefficient nonuniqueness, equal-split and prediction distinctions; retained. | Existing duplicate-sensor derivation and comparison. |
| §5 curved features, library normalization and assessment | Sufficient basis expansion, objective mapping, train-only scaling and actual airfoil study; retained. | Complete scratch/library programs and recorded fits remain unchanged. |
| §6 dropout scaling, mask averaging and nonlinearity | Sufficient exact enumerated masks, expected squared loss and nonlinear exception; retained. | Existing interactive mask investigation, exact penalty identity and counterexample. |
| §7 SVD shrinkage, paths, computational choice | Sufficient direction-by-direction explanation and solver cost; retained. | Existing SVD representation and lasso path interpretation. |
| §8 early stopping, SGD decay and adaptive updates | Sufficient contracts and counterexamples; retained. | Existing filter factors and update equations. |
| §8 Bayesian priors | Added sample-size interpretation: a fixed prior weakens relative to more independent evidence; duplicating rows is not that experiment. | Existing noise/prior conversion now connects to n rather than leaving normalization abstract. |
| §8 factorization, smoothness and other structured penalties | Existing factor-scale counterexample and three-point difference operator are adequate. Group lasso was named without a usable mechanism; added its radial threshold derivation. | New regularization-group-threshold figure: z=(3,4), norm 5, lambda 4 gives (0.6,0.8); coordinate L1 gives zero. Equal x/y coordinate scales. |
| §8 group assumptions | Identity curvature, unsquared group norm, zero input, group scaling and surviving zero coordinates explicitly distinguished. | Prevents interpreting this special closed form as a general correlated-design solver. |
| §9 AIC, BIC and MDL | Sufficient distinct motivations, sample-size assumptions, regularity limits and worked scores; retained. | Existing examples and complete comparison code. |
| §10–11 practice, readiness and next topics | Ten detailed solutions and progression retained. | New group example extends the existing structured-penalty branch rather than replacing core practice. |

## Research actually inspected

- [Yuan and Lin, Model selection and estimation in regression with grouped variables](https://www.columbia.edu/~my2550/papers/glasso.final.pdf): sections 1–2 and section 5 equation 5.1. Inspected the invariance motivation, group-weight convention and orthonormal closed form. The lesson's unweighted single-group calculation is original; it explicitly fixes identity curvature.
- Existing annotated learning alternatives remain. This pass did not watch videos or rerun the recorded airfoil fits.

## Checks and representation

The new diagram is a static coefficient-space calculation with HTML values and explanation, not a slider. Amber marks the two penalty outcomes, with separate labels and ring/filled marks. It reuses the existing topic figure treatment and contains no browser fitting.

The scoped author verifier checks the partial residual, radial optimum against alternative directions and zero-threshold boundary, the separate-coordinate result, and fixed-prior normalization. JSX parsing is included. Native code, fixtures, numerical engines and recorded evidence are unchanged. Independent teaching review, responsive rendering and browser interaction review are pending.
