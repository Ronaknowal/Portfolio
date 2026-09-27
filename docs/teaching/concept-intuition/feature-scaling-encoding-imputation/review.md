# Preprocessing: whole-lesson concept-transition review

26 September 2026. Read all eleven sections, complete implementation prose, the implementation exercise plus nine end practices and annotated references. Production JSX is the content checkpoint. Independent and browser review remain pending.

| Concept / exact source location | Finding and action | Representation / transfer evidence |
| --- | --- | --- |
| §1 measured record → feature representation | Sufficient real-record motivation and transform ownership. Retained. | RecordFigure. |
| §2 scaling changes nearest-neighbour geometry | Sufficient numerical distances and editable divisors. Retained. | RulerLab; practices 1–2. |
| §2 learned standard/min–max/robust scales | Sufficient outlier comparison, new value beyond fitted range and zero-spread handling. Retained. | RulerFigure; practice 2. |
| §2 column scaling versus row normalization | Sufficient directional versus magnitude distinction. Retained. | Practice 3. |
| §3 categories and unknown/missing/rare | Sufficient coordinate geometry and invented-order counterexample. Retained. | CategoryFigure; practice 4. |
| §4 simple imputation and missingness mechanisms | Added three concrete conditional mechanisms distinguishing MCAR, MAR and MNAR, with explicit assumptions rather than labels alone. | Independent loss, recorded-station-dependent loss, unobserved-weight-dependent loss table. |
| §4 donor imputation | Sufficient observed-coordinate distances and target-specific donor eligibility. Retained. | DonorLab; practice 5. |
| §4 iterative imputation | Added first-turn mass-from-length calculation and explanation of cycling through currently completed columns. | Distinguishes a model-based conditional fill from obtaining new observations. |
| §5 fitted state and library parity | Sufficient saved means/scales/categories and exact parity check. Retained. | BoundaryFigure, fittedStateParity, missing-indicator extension. |
| §6 real experiment and pipeline | Sufficient protected split, actual outcomes and changed-record tracing. Retained. | ComparisonFigure, PipelineFigure/PipelineLab; practice 6. |
| §7 logs/power/quantile transforms | Sufficient domain restrictions, formulas and changed rank geometry. Retained. | RankFigure; practice 8. |
| §7 cycles | Gap: seam repair was only named. Added equal-scale unit-circle geometry for 350° and 10°, chord 0.3473, and why both sine and cosine are required. | New cyclic-feature-geometry; elapsed-time counterexample prevents indiscriminate reuse. |
| §8 target encoding shrinkage | Added weighted-compromise interpretation, one versus 100-row influence, and explicit difference from cross-fitting. | Existing formula and prior example. |
| §8 target cross-fitting and hashes | Sufficient row ownership, collision and implementation limitations. Retained. | EncodingFigure/TargetEncodingLab, crossFitEncoding; practice 7. |
| §9 multiple imputation | Added noncommuting square example: analyse 0 and 2 then pool gives 2; average first then square gives 1. | Existing PoolingFigure and Rubin variance; practice 9. |
| §10–11 transfer and references | Sufficient nine explanations plus scratch extension and annotated resources. Retained. | No completed rows misrepresented as measured truth. |

## Research actually inspected

- [Stef van Buuren, Flexible Imputation of Missing Data, section 1.2](https://stefvanbuuren.name/fimd/sec-MCAR.html): conditional missingness examples and classification. The author’s conditional-group distinction informed the station example; the assumption is stated locally instead of treating MAR as a visible property of a spreadsheet.

The cyclic seam and pooling counterexamples are independently calculated constructions. Existing alternatives already link the imputation book, so the resource list is retained.

## Representation and checks

The circle has identical horizontal and vertical scales, labelled angles, an amber chord and exact HTML interpretation. It is a static geometry diagram with no slider-like affordance. Both projected coordinates are kept. Browser width/contrast still require integration review.

The scoped verifier checks the circular chord, sine-only collision, target-smoothing weights, noncommutation of pooling and analysis, and JSX parsing. Existing native preprocessing experiments, runnable programs and historical fit outputs were not regenerated.
