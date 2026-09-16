# Problem formulation: content design and continuation

Stable ID: ml-problem-formulation-baselines-data-leakage. Classical ML position 37; authorized batch position 19. Root author, 12 September 2026. Research/write only, revision 1. The central phase ledger owns current status and source hashes.

## Scope and source

Read the full topic preflight, its returned authoring notes, and the complete bespoke plan in src/learn/data/curriculum/cross-domain-expansion.js at baseline 8c5da59f18516be77c29d5aeeafca3decca4f738. This is an existing planned topic, not a published JSX replacement. The source plan teaches target/unit/features/decision, available information, split-before-fitting, baseline, suspicious-score audit, a data-availability timeline and an experiment-contract exercise. All are substantively developed in the manuscript. Runtime, catalogue, blueprint, order and publication are unchanged.

Retain the existing title and identity. During writing, add label maturity, versioned availability, unit dependence, metric/action alignment and the distinction between propensity and intervention effect. These directly serve problem formulation rather than requiring a new topic. Full causal identification, detailed metric derivations and rolling forecast algorithms retain their dedicated existing homes. No whole-catalogue audit or title change was needed.

Read the incoming [destination note](../../topic-notes/ml-problem-formulation-baselines-data-leakage.md) in full. Its event4/arrival8/cutoff5 calibration case is integrated into §3, extended with an earlier eligible value, a later revision, entity matching and an age-limit null. The note remains open for implementation; its prepared-content disposition links this packet. Pandas owns join mechanics; this topic owns the information contract. The next Time-Series lesson must extend these same cutoff/availability semantics.

## Outcomes and learning sequence

Core outcomes: write a target/unit/cutoff contract; trace a feature's provenance; reconstruct a known-time snapshot; choose a meaningful baseline and metric; distinguish temporal leakage, preprocessing leakage, unit contamination, selection leakage and deployment mismatch; report a finding with an appropriately scoped next step.

The opening duration example returns as a measured real-data comparison. First-pass route §§1–7/practices1–5 is separate from §8's deeper causal/selection/loss branch. A novice gets local definitions for target, row/unit, prediction/action, prevalence, precision/recall at capacity, baseline and pipeline. Formal causal estimation and quantile optimization are explicitly deeper connections, not unstated entry requirements.

| Hurdle | Explanation / example | Representation and learner evidence |
| --- | --- | --- |
| A prediction is confused with a business goal | Outcome/prediction/action separation; explicit capacity | F1 flow, experiment contract |
| A row is treated as an independent deployment case | Several hourly rows per parcel | F2 identity lanes; changed equipment exercise |
| An absent outcome becomes a negative label | Seven-day event window and incomplete follow-up | Core explanation and changed practice 7 |
| Old event mistaken for old knowledge | Exact two-time calibration history with revisions | I1 actual eligibility/selection; changed practice 1 |
| A good score substitutes for a valid task | Actual unavailable-duration comparison | F3 boundary, F4 results beside feature contract |
| Accuracy hides ranking and action constraints | Candidate 733 correct versus baseline734 but top50 positives20 versus6 | I2 selected identities and fractions; practice3/4 |
| One software pipeline is treated as universal protection | Separate information paths through features, units and selection | F5 lineage repair and practice2/5 |
| Association is mistaken for action impact | Constructed propensity/impact rank reversal | F6 optional comparison; practice8 |
| Mean prediction is presumed optimal for every cost | Original two-point spare-parts demand | Exact deeper costs15/5 and quantile connection |

Interesting applications are tied to mechanisms: historical calibration revisions, limited call capacity, delayed equipment/parcel labels and spare-parts decisions. They are not disconnected trivia. Static figures clarify relationships; the two investigations have different operations and meaningful entity edits rather than a fixed lab quota.

## Canonical-reference section-list check

Canonical teaching reference: Google's current Problem Framing course, with the actual body and section structures of [Understand the problem](https://developers.google.com/machine-learning/problem-framing/problem) and [Framing an ML problem](https://developers.google.com/machine-learning/problem-framing/ml-framing) inspected. The full course/video experience was not completed.

| Canonical section family | Disposition |
| --- | --- |
| Goal, clear use case, non-ML benchmark, data feasibility | Core §1 contract and §4 baselines; no claim that every use case needs ML |
| Data availability/reliability/representativeness | Core §§2–3 and real-data limits; explicit event versus available-at addition |
| Predictive power | Use controlled baseline/model comparison and provenance, not a universal correlation ranking |
| Predictions versus actions | Central flow and real capacity example |
| Ideal outcome, output type, proxy labels | Core §§1–2, with decision-relevant output/loss distinction in deeper §8 |
| Classification/regression selection | Already taught earlier; locally state meanings and correct the simplistic fixed-threshold rule with expected-cost reasoning |
| Generative customization | Outside this Classical ML contract's modeling scope; later generative/LLM topics own prompting, fine-tuning and distillation |
| Success versus model metrics, constraints, failure analysis | Core §§4–7; full empirical capstone follows the next time-series topic |

The course's advice about monotonic benefit from more data, linear correlation as general predictive power, and fixed-threshold model choice is not reproduced as a theorem. Bias–Variance and feature selection already establish why those are conditional heuristics; this manuscript gives an original decision-cost counterexample. An attractive external explanation is a reference to evaluate, not a template to copy.

## Primary research and evidence actually inspected

All retrievals 12 September 2026.

- Google course above: introductory objectives, both core pages' topic lists and substantive paragraphs/examples on goals, data, outputs, proxies and metrics. Link follows the observed /problem URL; an initially guessed /understand-problem URL failed and was corrected before handoff. No video watched.
- [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing): dataset versions, citation and CC BY 4.0. Downloaded the provider's nested additional-data archive and read its full names/variable-description member, which is retained. This resolves the website's mixed older/newer schema: this packet uses 999, twenty inputs and the actual additional-file duration note.
- [Pandas merge_asof](https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html): backward/forward/nearest semantics, sorting/grouping, tolerance and inclusive matching. Our original loop implements two time predicates, not an unsupported claim that merge_asof alone does so.
- [Feast point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins): event-time retrieval and TTL explanation. [Issue6615](https://github.com/feast-dev/feast/issues/6615) and linked [merged PR6617](https://github.com/feast-dev/feast/pull/6617): substantive problem, opt-in created-time behavior, backend contract and merge date31 July2026. The issue's pre-change claim is not reported as current universal behavior. No Feast runtime/backend was installed or tested.
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): actual inconsistent-preparation and fit-only feature-selection examples, with pipeline boundary. The packet does not inherit an overbroad claim that a pipeline prevents every form of leakage.
- [QuantileRegressor API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html): conditional-quantile and pinball-loss definition checked as a supporting deeper connection. The longer gallery retrieval exposed navigation without the substantive example body, so that example is not claimed read. The spare-parts calculation and CDF derivative are original exact reasoning, not an empirical cost report. The earlier quantile-regression lesson owns full fitting.

No copied model-performance curve, external benchmark numbers or unsupported universal operating threshold is used. Alternate resources are annotated primary educational articles and documentation; a video was not necessary merely to satisfy a format count.

## Calculations and author self-review

The real-data calculation script executed two fixed logistic fits, a constant-prior baseline and six exact timeline fixtures in the existing runtime. All reported probabilities, rankings, confusion counts and means are retained. A JSON-only follow-up computed top25 outcomes without refitting. Full details and limitations are in [provenance](data-provenance.md).

The author reread the full manuscript and specification sequence after writing. The pass corrected practice 3's misleading “same accuracy” heading, supplied the actual top-25 answer and swap IDs, aligned the timeline null edit with the supported value bounds, and added the CDF derivative behind the deeper cost/quantile connection. It checked source-to-question consistency, the double-time rule, measured versus constructed distinctions, practice arithmetic and onward sequence. Formal independent correctness/learning-experience review and displayed-program verbatim execution remain phase two.

Checklist:

1. Route and deeper readiness are distinct; threshold/prior/precision terms receive local explanations.
2. Main cautions have concrete homes: real data's scope in §5, leakage taxonomy in §6, intervention limits in §8. Code prints calculations rather than disclaimers.
3. The opening call-duration question returns with an actual licensed dataset, baseline and measured unfavorable accuracy result.
4. Both investigations specify input-bound recorded predictions, editable entities, actual contrasts and nulls. Timeline math is exact; the action-set view uses saved real identities.
5. Quantitative figures have appropriate common scales, actual counts, baseline and accessible values. Rendered perceptibility is specified but unverified.
6. Event/availability/time joins link to the incoming Pandas note; preparation/CV/metrics remain earlier owners; time-series and capstone follow.
7. Displayed programs emphasize the mechanism. The calibration loop is intentionally bounded and not advertised as a production-scale join.
8. Changed exercises have answers: new timeline, costs, threshold, label maturity and actual top25 count12.
9. No screenshot or browser review occurred. Future contrast/reveal/mobile states are required in the specifications.

## Continuation

Retain all nine topic-owned files: lesson, visual specifications, design, raw CSV, source description, data-source metadata, provenance, author calculation and calculated inputs. Raw data and research inputs are not automatically initial browser payloads.

On an authorized finish request, consume the content checkpoint, implement the topic-specific representations and models, execute displayed programs verbatim, verify independent numerical/behavioral results and rendered accessibility, obtain formal independent review, and integrate publication with updated source hashes. Existing planned state and current runtime stay unchanged during this content request.
