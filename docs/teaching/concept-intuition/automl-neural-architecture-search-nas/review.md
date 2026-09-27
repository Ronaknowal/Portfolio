# AutoML and NAS: whole-lesson concept-transition review

26 September 2026. Read all ten sections, ten practice solutions, native and optional library implementation explanations, annotations and limitations. Production JSX is the content checkpoint. Author review only; independent teaching and browser review remain pending.

| Concept / exact location | Finding and action | Representation and transfer evidence |
| --- | --- | --- |
| Intro / §1 search object and held-out boundary | Sufficient recipe rather than estimator-only framing; retained. | Existing grammar and fitting/selection diagrams. |
| §2 random/grid priors | Sufficient log-scale versus linear coverage and conditional spaces; retained. | Existing distribution calculation and sampling discussion. |
| §2 Bayesian selection | Sufficient predictive uncertainty, expected improvement and changed candidate choice; retained. | Existing EI figure and explicit optimization convention. |
| §3 allocation and fidelity | Sufficient slow-starter reversal, halving budgets and full-procedure cost; retained. | Existing live allocation and Hyperband explanation. |
| §4 architecture versus weights | Sufficient actual dense parameter count; retained. Added executable shape example: B×8 and B×12 cannot be added; concatenation gives B×20; affine 12→8 then addition costs 104 parameters. | New table exposes graph validity before evaluating candidates. |
| §5 complete Banknote study | Sufficient protected data roles, actual recorded 35-fit budget and replay limitations; retained. | Complete scratch code and measured outputs unchanged. |
| §6 ensembling and meta-learning | Sufficient separate search/combine/transfer questions and failure limits; retained. | Existing portfolio and symbolic-model examples. |
| §6 deployment constraints | Sufficient measured device constraints rather than FLOPs-only rank; retained. | Existing direct-manipulation deployment tradeoff. |
| §7 DARTS mixture | Added intuition for the softmax derivative as redistribution from the current average, and why equal operation outputs produce no first-order output change. | Existing mixture lab retained. |
| §7 discretization | Added exact identity/negation construction: input 1, target 0, equal mixture has zero loss; either discrete operation has half-squared loss 0.5. | Separates optimized mixture from deployed graph with an original counterexample. |
| §7 bilevel dependence | Existing chain derivative, unrolled step and BilevelFigure are sufficient; retained. | No duplicated proof or gratuitous visual. |
| §7 shared weights and proposal families | Sufficient weight provenance and coupled rankings; retained. | Existing WeightProvenanceFigure, RL/evolution/BO/network-morphism scope. |
| §7 NASWOT | Existing binary codes and determinant lacked a geometric explanation. Added active/inactive concatenation, dot product, Gram matrix and squared-area reasoning. | New naswot-agreement-kernel table: codes 110/101 give diagonal 3, off-diagonal 1, determinant 8; identical codes give log score negative infinity. |
| §7 proxy limits and final comparison | Explicitly retained distinction between activation diversity, nuisance separation and trained accuracy. | No invented finite singular score or benchmark. |
| §8 libraries | Sufficient FLAML/KerasTuner controlled setup, exact executed outputs and state ownership; retained. | Complete optional programs unchanged. |
| §9–10 practice and sources | Ten transfer questions, readiness and annotated alternatives retained. | New constructs explain advanced mechanisms already in their scope. |

## Research actually inspected

- [Liu et al., DARTS](https://arxiv.org/pdf/1806.09055), sections 2.1–2.2 and figure 1: candidate graph, continuous mixed operations and final discrete selection. The identity/negation example is original and deliberately shows a relaxation gap.
- [Mellor et al., Neural Architecture Search without Training](https://proceedings.mlr.press/v139/mellor21a/mellor21a.pdf), section 3 equations 1–2 and adjacent interpretation: activation-code agreement and log determinant. The explicit augmented binary vectors are a local derivation, not a copied experiment.

## Checks and representation

Shape constraints and the NASWOT kernel use responsive HTML tables; their values and interpretation remain readable without SVG. No runtime training or new heavy dependency was added. Existing topic-specific interactive labs already expose the core mechanisms.

The author verifier checks projection parameters, mixture/discrete losses, zero output derivative for equal operation outputs, and the augmented-vector Gram determinant including singular codes. JSX parse included. Native runs, data, labs and engine modules remain untouched. Independent/browser/integration stages pending.
