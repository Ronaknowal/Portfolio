# HMM: whole-lesson concept-transition review

26 September 2026. Read all thirteen sections, complete code/library explanations, ten practices and sources. Inspected TopologyFigure and the numerical model's documented conventions. Production JSX is the revised content checkpoint; this record does not claim independent or browser review.

| Concept / exact source location | Finding and action | Representation and transfer evidence |
| --- | --- | --- |
| Intro / §1 hidden state, observation and time | Sufficient original story, explicit matrices and reversed-conditional warning; retained. | Graph unroll and path-product lab. |
| §1 Markov and emission assumptions | Sufficient conditional versus marginal distinction and T−1 edges; retained. | Existing factorization, indexing and sequence-boundary explanation. |
| §2 six inference/learning questions | Sufficient named conditioning information, outputs and loss distinctions; retained. | Query table. |
| §3 forward, filtering and forecast | Sufficient one-cell mass calculation, normalization and complete-belief propagation; retained. | Forward trellis, exact constructed result, practice 1. |
| §4 backward and smoothing | Added why alpha/beta can be multiplied: conditioned state separates past/future, current report belongs only to alpha. | Existing belief figure and evidence lab now have a double-counting diagnostic. |
| §4 missing observation versus deleted time | Sufficient marginalization and elapsed-time distinction; retained. | Explicit separate computed outcomes. |
| §5 Viterbi, backpointers and pointwise modes | Sufficient dynamic-programming proof and impossible marginal-mode path; retained. | Existing two trellises, PointwiseFigure and LegalPathLab, practices 2–4. |
| §6 Baum–Welch pair counts | Added reuse of the impossible A→A path: product of positive marginal occupancies invents mass on a forbidden edge, while xi preserves it at zero. | Exact existing constrained fixture links decoding to parameter estimation. |
| §6 sequence counts, EM and initialization | Sufficient full boundary-sensitive update, monotonicity scope, symmetry and state permutation; retained. | CountFlowFigure, BoundaryCountLab, measured EM histories, practices 5 and 8. |
| §7 numerical inference and code | Sufficient log-sum-exp, all-impossible branch, scale definitions and original source excerpts; retained. | NumericScaleFigure and complete scratch/native runs. |
| §7 library meaning | Sufficient categorical/multinomial/Gaussian shapes and versioned MAP-return trap; retained. | Actually recorded hmmlearn output untouched. |
| §8 real supervised tagging | Sufficient source extraction, train vocabulary, smoothing and count-fit ownership; retained. | RealTaggingLab, tie-audited comparison and source-preserved data. |
| §8 assessment wording | Replaced an overabsolute claim with precise wording: development-selected output is not an independent assessment. | Exact tie band and untouched reserved data remain unchanged. |
| §9 durations | Sufficient geometric mechanism, constant hazard, absorbing/impossible-conditioning boundaries; retained. | DurationLab and practice 7. |
| §9 Gaussian emissions | Added two equal-mean states with standard deviations 1 and 2: a zero reading yields posterior 2/3; conversion from metres to centimetres changes densities but not posterior ratios. | Concrete bridge from fractional counts to weighted moments; full covariance and autoregression caveats retained. |
| §9 state count | Sufficient row constraints, nonregular criteria and actual decision-time query; retained. | Existing parameter count table. |
| §10 profile, factorial, continuous and conditional models | Existing topology illustrates graph changes. Added a four-world shared-meter table to expose posterior dependence between prior-independent devices. | New factorial-shared-evidence: sum-one evidence retains two worlds, each marginal stays 1/2 but joint on/on becomes zero. |
| §11 computational cost | Sufficient sparse/dense transitions, memory, emission families and beam limitations; retained. | Teaching implementation's retained pair arrays remain honestly identified. |
| §12–13 practice and progression | Ten worked solutions, readiness and next-topic links retained. | New mechanism bridges support existing Bayesian-networks/CRF progression. |

## Research actually inspected

- [Jurafsky and Martin, Appendix A](https://web.stanford.edu/~jurafsky/slp3/A.pdf), August 19, 2026 draft: forward factorization and section A.5 expected pair/state counts, including equations A.23–A.28. No source example was copied.
- [Ghahramani and Jordan, Factorial Hidden Markov Models](https://mlg.eng.cam.ac.uk/pub/pdf/GhaJor97a.pdf), section 2 and section 3.2: factorized prior dynamics, shared emissions, posterior dependence and exact-inference cost. The noiseless two-device table is an original small construction, not their Gaussian experiment.
- Existing Eisner resources and video/spreadsheet qualifications remain; neither media nor spreadsheet was newly executed.

## Checks and representation

The new table complements the existing factorial graph by showing exactly which joint worlds the observation excludes. It has explicit observed evidence, prior and posterior masses and a noiseless limitation. It is a static calculation; existing six topic-specific labs remain interactive.

The verifier independently calculates the two-time constrained marginal-product counterexample, Gaussian density ratio before/after a unit change, and shared-meter posterior dependence. JSX parse included. Native EM/tagging campaigns and their tied-path evidence are unchanged. Independent teaching review and browser/integration checks remain pending.
