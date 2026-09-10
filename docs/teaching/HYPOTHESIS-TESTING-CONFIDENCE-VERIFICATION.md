# Hypothesis Testing & Confidence Intervals — author verification

10 September2026. Stable ID `hypothesis-testing-confidence-intervals`, Mathematical & Statistical Foundations position17. Complete source and individual brief are author-reviewed; root independent review and production integration are separate. No user acceptance or observed beginner study is claimed.

## Scope and preservation

Read the full earlier pilot, its shared coverage model and later interval-decision figure; original copies remain in `scratch/hypothesis-testing-authoring/`. The rewrite preserves the five old/new timings, moving40-interval known-σ mechanism, zero-versus1ms comparison, original shifted-data practice and useful method/design warnings. It replaces the isolated presentation with ten connected sections, eight concept-specific investigations, two inline figures, two early checkpoints, fourteen complete native programs and eight changed closing exercises. These counts document the implementation and are not authoring quotas.

The [design record](HYPOTHESIS-TESTING-CONFIDENCE-DESIGN.md) documents sources, original coverage, scope decisions and representation contracts. The incoming [MLE Bernoulli-boundary note](topic-notes/hypothesis-testing-confidence-intervals.md) is implemented. Scoped discoveries for [sequential analysis](topic-notes/a-b-testing-sequential-analysis-at-scale.md) and [BCa bootstrap](topic-notes/bootstrap-confidence-intervals-bca.md) are saved against verified existing IDs; their planned bodies were not rewritten.

## Numerical and native evidence

Run `node scripts/verify-hypothesis-testing.mjs`, which executes `scripts/verify-hypothesis-testing-native.py` with `scratch/lesson-tools/Scripts/python.exe`. Passed record: [native-results.json](../../scratch/hypothesis-testing-verification/native-results.json), timestamp2026-09-10T14:49:33.147Z. Runtime Python3.12.14, NumPy2.3.5, SciPy1.18.1.

| Checked behavior | Independent evidence |
| --- | --- |
| Original coverage conservation |12,000 known-σ pilot interval rows exactly match the original implementation. |
| t inference and inversion |1,260 changed t states checked against SciPy tests, tails and intervals;15 quantiles and1,205 distribution/density/tail coordinates. |
| Whole coverage models |600 configurations of40 intervals; Python reconstructs the integer RNG and exact Normal mean/variance factorization, with independent SciPy critical values. |
| Prospective power |2,160 states checked by shifted Normal distribution probabilities. |
| Binary intervals |404 finite coverage configurations: all k endpoints and binomial masses, Wilson versus SciPy and exact Clopper–Pearson versus SciPy; endpoint and conservative-coverage checks. |
| Error-control and resampling |300 independent family risks with Fraction arithmetic;36 optional-look cases with exhaustive independent coin paths;96 complete sign-assignment states;80 changed Holm helper cases against the step-down decision definition. |
| Other mechanisms |27 cluster-variance cases,60 mean/prediction comparisons,24 practical-effect/TOST states. |
| Native helpers and invalid inputs |24 changed paired inputs,369 changed Wilson cases,31 invalid model/helper cases including nonfinite values, wrong type/range and unsupported zero-spread t inference. |
| Complete examples |All14 standalone programs execute without warnings and reproduce every displayed output. The finite bootstrap enumerates all3,125 empirical resamples and repeated-looking program all4,096 coin paths. |
| Changed independent practice |Verified shifted confidence endpoints, changed reference/direction p=.8509925941, changed equivalence interval, all-success CP boundary, cluster arithmetic and5-metric family probability. |

Oracles are not merely tests of a copied formula: SciPy distribution/interval APIs, Fraction event probabilities, exhaustive bit-string/sign enumeration and changed native helper inputs provide distinct reference mechanisms. One test-harness refinement used exp(SciPy logsf) because direct SciPy sf underflows before the mathematically representable extreme Normal tail; it did not change the lesson model. The model uses bounded finite inputs, a stable t4 tail and Gaussian quadrature for Normal tails, not an unbounded general-purpose statistics library.

## Actual browser and reading evidence

Local route `http://127.0.0.1:5173/learn/path/full-curriculum/hypothesis-testing-confidence-intervals` in headless Microsoft Edge using the installed Playwright runtime.

- `node scripts/review-hypothesis-testing-lesson.cjs` passed at1440px and390px:366 changed investigation states per width,14 complete code/output blocks,10 live section anchors,8 closing practice sections,16 correctly rendered display equations, all plot labels within their SVG, no document overflow, page errors, console errors/warnings or failed requests. [Full results](../../scratch/hypothesis-testing-browser/results.json).
- `node scripts/review-hypothesis-testing-reading.cjs` passed on final source at1440px,390px and320px: all10 section openings and2 inline figures captured;26 enabled lab controls reached in order by keyboard with visible focus and44px targets; slider/select keyboard changes, practice disclosures, and both early checkpoint questions plus revealed answers verified. Every display equation fits, no invalid paragraph nesting, no clipped plot labels, no page/console errors or failed requests. The long method/state tables scroll locally by keyboard where needed:0,1 and3 tables at the three widths. [Final reading results](../../scratch/hypothesis-testing-browser/reading-results.json).
- Actual opened screenshots include every ordinary section at390px, selected320px derivations and practice, desktop null-tail reading, desktop known coverage, mobile known/estimated coverage, prediction, null tails, equivalence, planned power, binary under-coverage, repeated chances, sign-flip counting, both inline figures, both repaired checkpoints and the final annotated source list. Captures are in [the browser evidence directory](../../scratch/hypothesis-testing-browser/).

Relevant examples: [ordinary paired derivation390](../../scratch/hypothesis-testing-browser/reading-3-390.png), [small-screen derivation320](../../scratch/hypothesis-testing-browser/reading-3-320.png), [finite binary coverage](../../scratch/hypothesis-testing-browser/proportion-undercoverage-390.png), [estimated-spread coverage](../../scratch/hypothesis-testing-browser/coverage-estimated-390.png), [power mechanism](../../scratch/hypothesis-testing-browser/power-390.png), [sign assignments](../../scratch/hypothesis-testing-browser/sign-flip-390.png), [checkpoint1](../../scratch/hypothesis-testing-browser/checkpoint-1-390.png), [checkpoint2](../../scratch/hypothesis-testing-browser/checkpoint-2-320.png), [sources](../../scratch/hypothesis-testing-browser/sources-390.png).

Lab/figure element captures temporarily hide the fixed site header to avoid covering the representation, then restore it. Ordinary reading/checkpoint captures keep normal page layout and header. Screenshots are reviewed for actual labels, numerical consequences and readable flow, not merely generated.

## Findings closed and final source

- Split wide equations into logical lines at390/320px; retained the derivations and introduced explicit pivot/margin/binomial-weight quantities where helpful. All16 final displayed equations fit320px without reducing text size.
- Root independent component-contract read found two Checkpoint calls using question/answer instead of prompt/children. Fixed both without changing their reasoning and added actual nonempty prompt, keyboard reveal and visible-answer assertions at all three widths. All14 example records contain title/code/expected rather than an unused question field.
- Added spacing between annotated references. Sources use the correct children/alternatives contract.
- Corrected the changed-reference practice p to .850993 before numerical verification; all final worked answers match the recorded native values.
- Formatted owned model/render functions conventionally with normalized Babel AST equality, including preserved JSX and template literal values: [formatting evidence](../../scratch/hypothesis-testing-verification/formatting-results.json). Native calculations and lab behavior are unchanged by formatting. The final reading rerun follows the checkpoint and spacing repairs; the prior full interaction result remains applicable to unchanged models/lab behavior.
- Initial restricted browser execution could not load the site's existing public Google Fonts stylesheet. The final reviewed executions used authorized public-font network access and report zero failed requests. No font request was suppressed and no shared typography source changed.

Frozen semantic source fingerprints are recorded in [final-source-hashes.json](../../scratch/hypothesis-testing-verification/final-source-hashes.json). Root owns manifest/index, curriculum conservation, production build, loading/recovery and module-order integration; publication alone is not completion.

## Limits and source review

Numerical plots represent declared analytical/finite/simulated teaching models, not measured latency or guaranteed performance. Forty displayed experiments and4,000 native simulated experiments are finite draws; coverage claims follow stated models, not their empirical percentages. Browser sampling/inputs are deliberately bounded; this is not validation of arbitrary-range statistical software or an assessment of a real experiment's assumptions.

Technical sources and exact inspected video/notes bounds are recorded in the design. MIT's official lecture association/direct video and substantive companion slides were checked; complete video playback/transcript was not available. StatQuest links were verified from the creator's index but full video/transcript inspection is not claimed. Brown's written companion description was read; its external interactions were not retested. References supplement the complete original lesson.
