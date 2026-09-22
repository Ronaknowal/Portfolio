> **Current interaction amendment, 21 September 2026:** Read [live-exploration.md](live-exploration.md). Labs now update directly from valid edits and contain no learner prediction feature, including optional predictions. The older prediction/commit/reveal clauses below are historical design records; their numerical, scope, layout and evidence requirements remain applicable where unchanged.

# Design and content handoff: End-to-End Supervised Learning & Error Analysis

Stable ID: `end-to-end-supervised-learning-error-analysis`. Classical ML position39, batch position21 in [authorized scope](../../CLASSICAL-ML-DEEP-LEARNING-CONTENT.md). Owner: deep_foundations_content. Mode: research and write only, revision1. [Manuscript](lesson.md), [visual specifications](visual-specifications.md), [data](wine.csv), [provenance](data-provenance.md), [author calculations](author-calculations.py), [calculated inputs](calculated-inputs.json). The central ledger alone owns phase status and file hashes. Implementation, publication and formal independent review remain deferred.

## Scope, source conservation and title

The topic preflight `node scripts/build-curriculum-inventory.mjs --topic end-to-end-supervised-learning-error-analysis --work content` returned a planned topic with a complete blueprint, no destination notes and no relevant unresolved inbox work. The entire blueprint was read: dataset/split contract, fitted preprocessing/simple model, controlled stronger model, residual/subgroup analysis and frozen test report. Source owner is `src/learn/data/curriculum/cross-domain-expansion.js`, the `plan("End-to-End Supervised Learning & Error Analysis", ...)` entry. Starting commit: `8c5da59f18516be77c29d5aeeafca3decca4f738`; no original published body was present. The root's manifest-resolved baseline owns the original-source identity; the live blueprint, title, stable ID, order and memberships remain unchanged.

Retain title: it accurately promises a complete supervised-study synthesis with an error-analysis loop. Classification errors, probabilities and paired repairs satisfy the blueprint's residual-explorer purpose; a separate regression implementation would repeat the same information-flow lesson. Regression residuals remain with prior regression/evaluation lessons. The deep-learning module bridge motivates another function class without claiming neural superiority. The predecessor author confirmed a transition from origin/horizon units to specimens; the manuscript explicitly explains why source row order is not a temporal split.

## Coverage and ownership decisions

| Idea | Decision and evidence | Teaching home |
| --- | --- | --- |
| Contract, split and unavailable information | Local refresh before programs; prior topic teaches broader leakage taxonomy | §§1–2; previous formulation/time-series lessons |
| Fitted pipeline and raw/fitted state | Core complete program and information-lane figure | §§2–4 |
| Baseline/controlled stronger method | Majority plus same-two-feature linear/forest; added-feature linear is a predeclared alternative, not an invented chronology | §§3–4 |
| Error slices versus aggregate | Actual fixed specimen IDs, class and feature-range partitions, denominator and opposite subgroup movement | §5 and investigationB |
| Frozen test versus refit | Explicit selected training-only model; alternative refit protocol explained but not silently substituted | §6 |
| Uncertainty and repeated selection | Deeper Wilson scale check, pairing and nested procedure evaluation; no unjustified significance claim | §8; full theory remains prior evaluation/CV/calibration |
| Deferral and total system cost | Constructed acceptance queue with checked reversed cost comparison, no deployed-system claim | §8 investigationC |
| Model report/inference schema | Concrete report and actual reproducibility fields; systems deployment details remain future MLOps | §§6–7 |
| Neural debugging, architecture, gradient details | Preview only; next module owns mechanism and execution | Next Perceptrons/Backprop lessons |

No new topic or title expansion was warranted. No cross-owner uncovered mechanism requires an additional destination note from this capstone.

## Canonical-reference section-list audit

Canonical methodological reference: Goodfellow/Bengio/Courville [Deep Learning, chapter11](https://www.deeplearningbook.org/contents/guidelines.html). Section list and relevant prose were inspected, including11.1PerformanceMetrics,11.2DefaultBaselineModels,11.3DeterminingWhethertoGatherMoreData,11.4SelectingHyperparameters (manual/grid/random/model-based),11.5DebuggingStrategies,11.6Multi-DigitNumberRecognition. This map is a coverage check, not adoption of the chapter's wording or 2016 defaults.

| Reference idea | Scope decision |
| --- | --- |
| Choose metrics from the decision | Core local metric refresh and complete contract |
| Establish baseline chain | Core offline experiment |
| Gather data versus change algorithm | Core diagnostic hypotheses; learning-curve computation belongs to earlier dedicated topic |
| Manual/grid/random/model-based tuning | One finite predeclared comparison here; full algorithms remain earlier CV/AutoML lessons, avoiding a second search survey |
| Check known cases, internal state and gradients | Input/label/schema checks locally; actual network gradients belong to next Backprop lesson |
| End-to-end example and acceptance coverage | Original Wine study and constructed inspection-cost extension |
| Historical architecture/default claims | Not repeated as current recommendations. The chapter's use of “test” during adjustment is rendered here as validation, and zero-accepted conditional accuracy is undefined rather than perfect |

## Hurdles and assessment

| Hurdle/outcome | Local bridge | Mechanism/example | Representation and evidence |
| --- | --- | --- | --- |
| Distinguish learning by fit from selection by a person | Training/validation/test meanings | Scaler state and human-mediated test leak | Lane figureA; practice3 |
| Compare methods for a reason | Weighted sums/classes; balanced accuracy | Same specimens/features for forest versus linear, same family for added feature | Executed comparison; practice1 |
| Interpret aggregate improvement | Counts and proportions | Four repairs/one new error; low/highcolor reversal | FigureD/investigationB; practices2/4 |
| Freeze an interpretable result | Model artifact versus procedure | No-refit final matrix35/36 | Full report; practice6 |
| Judge uncertainty and deferral | Independent-trial formula locally; counts/costs | Wilson[.8583,.9951] illustration and ten-case queue | Optional investigationC; practice5 |

First-pass route: §§1–7 and practices1–4; §§8/practices5–6 are explicitly deeper/synthesis. Useful interest comes from a real counterintuitive subgroup tradeoff and a decision rule where improved conditional accuracy need not lower total cost, not from detached trivia.

## Research record and actual review extent

Reviewed12September2026. Primary evidence used for factual/API claims; the study and constructed exercises are original teaching calculations.

| Source/locator | Checked and used | Extent/limits |
| --- | --- | --- |
| [scikit-learn1.9.1 common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html),12.1–12.3 | Fit/apply separation, preprocessing/selection leakage, reproducibility nuances | Page/code examples and section list read; no copied reported scores |
| [CV guide](https://scikit-learn.org/stable/modules/cross_validation.html),intro and data-transform passages | Validation role and fitting within appropriate partitions | Relevant page text read; broader group/time protocols assigned to preceding lessons |
| [load_wine API](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) and [UCI DOI](https://doi.org/10.24432/C5PC7J) | Dimensions, identity, target coding, local loader and license | API + UCI content/license read; failed direct Wine URL and successful numeric-ID alternate documented in provenance |
| [Google MLCC dividing datasets](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets),page and exercises | Accessible alternate explanation for duplicates and train/validation/test roles | Written page/exercises read. No video was watched or endorsed from metadata |
| [Deep Learning chapter11](https://www.deeplearningbook.org/contents/guidelines.html),section list and11.1–11.6 relevant passages | Canonical coverage and methodology alternative | Historical model defaults not adopted; not a current benchmark source |
| [Model Cards](https://arxiv.org/abs/1810.03993),abstract | Reporting scope including intended use/evaluation/group breakdowns | Abstract read; attempted HTMLfulltext failed, so no claim fullpaper/template reviewed |
| [NIST Wilson intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm),formulas and terminology note | Formula and Wilson versus adjusted-Wald naming | Full short page read; our arithmetic independently calculated |

Alternate resources are annotated in the learner references. Appropriate video candidates were considered through Google course pages, but metadata-only discovery was not treated as substantive review; the selected written alternatives already teach this workflow well. No format quota was imposed.

## Author checks and learning-experience findings

- Bounded CPU author calculation ran the four fitted candidates and only the selected candidate's final test. Python3.12.14, NumPy2.3.5, scikit-learn1.9.1; no dependency changes. Actual split IDs, probabilities, confusion matrices and validation rows retained. The majority scaler in the author helper has no effect on its predictions; displayed code omits that unnecessary operation.
- Derived score reconstruction agrees with actual matrices: linear-twoBA.792857, three-featureBA8/9; paired repairs/regressions4/1; lowcolor2→3 and highcolor5→1. Retained unfavorable slice rather than selecting a fixture that makes every metric improve.
- Wilson35/36 arithmetic checked with scalar formula. Practice1score32/45 and practice5cost100/100 derived independently from changed inputs. Acceptance fixture enumerated for threshold/cost contrasts and equal/no-answer nulls; its exact inputs are retained for future model checks.
- Root content-continuity read found malformed inline math, inconsistent output-metric order, a contradictory confusion-axis header and a misleading practice heading. These were corrected in the manuscript before checkpoint. This is content editing, not the formal independent implementation review.
- Author full reading and inline-visual pass completed at content boundary: local metric and split refreshers precede code; predeclared alternatives are distinguished from exploratory cutoff analysis; paired changes connect to aggregate counts; the neural bridge preserves the actual module sequence. Cautions have a main home in§1population,§2informationboundary,§8uncertainty. Code prints data/results only.
- First-pass/deeper route and corresponding readiness are explicit. Practices1/2/5 change numerical problems, practice3 diagnoses a new scenario, practice4 is guided reaggregation, practice6 requires a documented new hypothesis. Optional hints/solutions are closed disclosures.
- Investigation contracts provide unset input-bound predictions, unsolved cutoff/cost/entity edits, graded consequences, checked opposite/null fixtures and recovery. Actual controls, rendered chart perceptibility and screenshot assessment remain unperformed; author specification review is not browser evidence.

## Phase-two continuation

Consume all seven packet files after the root records final hashes. Implement topic-owned reader content, figures/labs/models/example/download destinations under the code standard; preserve stable identity and sequence. Execute the displayed program verbatim, compare with saved results, and check prospective lab models against independent arithmetic/fixtures. Perform formal independent content/learning-experience review, focused fixes, actual browser/keyboard/mobile/accessibility review, publication and integration. A diagram placement suggestion can be improved for a recorded teaching reason.

No application build, browser interaction, formal independent review, learner study or publication occurred during research/write. No frozen test-based candidate search was performed. Next action: authorized finish work only after content checkpoint validation; otherwise continue the next requested content topic in batch order.

## Phase two: implementation record

Implemented 20 September 2026 from the frozen packet. The manuscript, visual specifications, recorded
calculations, provenance and data were consumed unchanged; this section is the only part of the packet this
phase edited, and it is appended rather than rewritten.

### The packet's numbers were recomputed independently, and agree

Before any lesson code was written, every quantity the packet states was recomputed by a second route in pure
Python — accuracy, balanced accuracy and every confusion matrix in exact rational arithmetic from the stored
label and prediction vectors, log loss from its definition, the slices and paired counts from the per-specimen
rows, the Wilson interval from the scalar formula, the acceptance ledger by exact enumeration, and the practice
arithmetic from its own matrix. The manuscript's displayed program was then executed verbatim beside the
served copy of `wine.csv`.

**No disagreement was found on any quantity.** The displayed program reproduces the manuscript's recorded
output block byte for byte, and `calculated-inputs.json` agrees with an independent recomputation of all 2,214
of its scalar leaves. There is nothing for this section to record as a packet error.

Two implementation findings are worth passing on, because both cost time and neither is a packet defect.

- **The fitted forest compares in single precision.** Re-deriving the forest's probabilities by walking its
  100 trees routed three of the thirty-six validation specimens — 94, 98 and 150 — to different *leaves* than
  the fitted model, until the input row was cast to `float32` before each split comparison. The independent
  review confirmed this and refined it: the disagreement is at the leaf and therefore in the
  **probabilities**, by up to 7.6 × 10⁻²; the **predicted labels agree on all 36 either way**. It also
  confirmed that casting the **input only** is correct and that casting the threshold as well is wrong,
  because not every stored threshold is exactly float32-representable. A fitted tree stores its thresholds in double
  precision but its inference routine casts the input matrix to `float32` first, so at a split where the two
  representations fall on opposite sides of the threshold a specimen routes to a different leaf. Walking with
  the raw double values is a real routing difference, not a rounding blemish, and it looks exactly like a
  packet error until it is diagnosed. The reason is recorded beside the walk in
  `scripts/verify-endtoend-data.py`.
- **One measurement is not on the two-decimal grid.** Specimen 172 carries a colour intensity of `9.899999`,
  written by the packet's `%.8g` export of a value that was never a round hundredth. An early draft of the
  slice explorer assumed every colour value round-tripped through hundredths and would have compared
  inconsistently at that specimen. The cutoff, not the measurement, now carries the integer-hundredths
  encoding, so the browser and the verifier evaluate `colour < cutoff` on bit-identical operands and no
  measurement is rounded.

### The held-out result is not in the document until it is earned

This is the one substantive departure from the manuscript's presentation, and it is deliberate.

As the closing lesson of the module, this page is the easiest place in the hub to let a development selection
score pass as independent evidence. The manuscript states the test numbers in §6 and prints them in §4's
recorded output block. Rendering them that way would put the held-out result on screen before the learner has
made the call it reports on — on the page that exists to teach why that is the error.

So the implementation sequences the reveal without changing a sentence of the teaching:

- the study program is displayed **whole**, exactly as the manuscript displays it, because its final block is
  part of the research sequence a learner must read. Its **output** is split at the line where the study opens
  the test set. The twelve development lines appear beside the program; the four held-out lines appear in §6
  once the decision is frozen. `scripts/verify-endtoend-examples.py` checks that the two halves concatenate
  back to the exact bytes the program printed, that the three held-out scores and the held-out confusion
  matrix are in the held-out half, and that none of them is in the development half;
- §6's report, §6's compact final-report callout and §8's Wilson scale check are all inside the gate, because
  all three quote a held-out quantity;
- the gate **withholds** rather than hides. Its children are not rendered at all. A CSS rule that hid them
  would leave the text in the document, in the accessibility tree and in a copy-paste, and "readable" is the
  property that matters. The source-hygiene checker refuses any attempt to gate the report with `display:
  none`.

A learner who runs the downloaded program locally sees everything at once. That is correct: the discipline is
the researcher's, not the interpreter's, and the lesson says so.

### Every displayed number carries its role

Four roles are defined in the model layer and nothing else is accepted: **training** (computed on the 106
fitting rows), **validation** (the 36 development rows), **selection** (a validation quantity in its role as
the declared choice criterion) and **held-out** (the 36 test rows, once, after the freeze). `scoreRecord`
refuses a role outside the four; `asSelectionCriterion` refuses to re-badge anything but a validation
measurement, and keeps `from: 'validation'` so the promotion is visible rather than hidden.

Every number reaches the reader through `Score` or `Count`, both of which require a role and print it as text
beside the value as well as carrying it as a data attribute. The source-hygiene checker refuses a
four-or-more-decimal literal anywhere in a component, so a score cannot be typed in unlabelled, and refuses
any held-out literal in a component so the gate remains the only thing that decides whether the report
renders.

### Where the implementation departs from the visual specification, and why

The specification is followed except as recorded here. Each departure has a teaching reason.

| Specification | Implementation | Reason |
| --- | --- | --- |
| Figure A, information lanes | As specified, with the forbidden edges asserted in the model layer rather than only drawn | The absent arrows — none from a label into a transform, none from the test lane into selection — are the picture's whole claim, so they are checkable facts rather than a drawing convention |
| Investigation B, development errors | As specified | Cutoff in [0, 14] to two decimals, both sides, any ordered pair of candidates including a candidate against itself, specimen selection, the flavanoid strip as its own axis |
| Investigation C, acceptance and cost | As specified | Threshold [.5, 1.01], wrong cost [0, 50], defer cost [0, 20], single-score edits, all four recorded contrasts and nulls reproduced |
| Figure D, paired repairs and regressions | Implemented and labelled **Figure C** on the page | Page order. The page carries four figures and three investigations, both lettered A–D in reading order; the prefix distinguishes them. **The mapping in full: the manuscript's Figure A is the page's Figure A; the manuscript's Figure D — "a net gain contains two directions" — is the page's Figure C; the page's Figure B and Figure D are additions the manuscript does not letter; the manuscript's Investigation B and Investigation C keep their letters, and Investigation D is an addition.** A reader moving between the frozen packet and the page will otherwise find "Figure D" naming two different pictures |
| — | **Figure B added**: the four candidates' validation balanced accuracy as bars, with the majority baseline in the frame | The specification keeps the baseline in "an aggregate comparison table". A table answers "what did each score"; it does not make "does measuring anything help at all" visible at a glance, and the teaching standard asks for the natural baseline point to anchor a comparison |
| — | **Figure D added**: per-class recall before and after the added measurement | §5's central claim is that a model can improve balanced accuracy while worsening one class. The manuscript supports it with a confusion matrix and a sentence. The claimed feature has to be perceptible, and the one cultivar that goes backwards is the thing the mean hides — so it is drawn, not only tabulated. Which cultivar is derived from the recalls, never typed |
| — | **Investigation D added**: freeze a decision, then open the test once | §6's hurdle — applying a declared selection rule and living with its result — had no representation, and the held-out gate needs something for the learner to earn it with. The learner's own inputs are the eligible candidate set and the deciding measure; the graded prediction is the winner, and a second graded prediction about the held-out direction opens the report |
| "Allow Explore without grading as a separate action if useful" | No separate action | Everything that is not the answer is already ungated: the scatter, the cutoff, the slice size, the specimen inspector and its probabilities. Only the error counts wait for a commitment, so a separate Explore button would duplicate what the page already does |
| "Two model panels can switch with accessible tabs" at mobile width | One scatter with reference and candidate selects | The specimens' coordinates are identical under every candidate — that is the specification's own point. Two panels would draw the same points twice; the selects change which predictions are outlined, with fewer controls to operate on a phone |

Beyond the specification's list, the page also carries a training-against-validation table for the three
fitted candidates. Its gap column is the only place on the page that subtracts one role from another, and it
says so: it is a diagnostic about the fit, not an estimate of anything.

### An honest consequence: the study refuses to answer some of its own controls

Investigation D lets the learner change the eligible candidate set and the deciding measure. Most of those 48
settings select something other than the model this study froze — and this study evaluated exactly one model
on the test rows. Rather than manufacture a held-out estimate for a choice nobody tested, the page refuses,
names the refusal, and says what would be needed instead. Exactly one of the 48 settings may open the report,
and that is asserted rather than assumed.

The slice explorer is honest in the same way. At the lesson's own cutoff of 4 the added measurement makes the
low-colour slice *worse* — 2 errors in 17 becoming 3 — while the high-colour slice improves from 5 in 19 to 1.
Both directions are reachable and both are graded; no cut was chosen to flatter the result.

### Verification produced

All five verifiers run from a clean checkout and write evidence to `docs/teaching/evidence/`.

| Verifier | What it establishes | Result |
| --- | --- | --- |
| `scripts/verify-endtoend-data.py` | Re-derives every scalar leaf of `calculated-inputs.json` from the served dataset by genuinely different routes, and regenerates `endtoend-data.js` byte-identically (read-only without `--write`) | 2,436 checks; **100% of 2,214 trust-root leaves re-derived**, of which 178 split-index leaves are validated rather than re-derived and are listed as such |
| `scripts/verify-endtoend-examples.py` | Extracts the displayed program verbatim from the frozen manuscript, pinned by SHA-256, executes it, and checks the held-out split on the actual output bytes | 57 oracles with `--isolated`; output reproduces the manuscript's recorded block byte for byte, in the shared runtime and in a clean environment built from the lesson's own setup line |
| `scripts/verify-endtoend-models.mjs` | Recomputes every stated number by a second derivation and sweeps every graded rule over the whole enterable grid | 284 grouped checks in 53 groups: 44,832 slice comparisons (every one of 1,401 cutoffs, both sides, all 16 candidate pairs), 55,692 acceptance ledgers, 48 selection settings, 162 tie-rule cases |
| `scripts/verify-endtoend-sources.py` | Source hygiene over every owned file, the five verifiers and the harness, including a JSX parse of every component and behavioural exercises of the harness's recovery gate, heartbeat and liveness probe | 111 checks over 16 files |
| `scripts/verify-endtoend-browser.cjs` | Production build at 1366, 390 and 320 px with screenshots | 14 cases, 29 captures, every digest and size re-checked, all distinct, no orphans |
| `scripts/falsify-endtoend.mjs` | Applies each breakage alone and requires the intended guard to fire *and name itself* | **51 of 51 fired**, 43 offline and 8 browser, none deferred; sources and build directory both restored |

The lesson's own displayed setup line was checked rather than assumed. `--isolated` builds a throwaway
virtual environment under `scratch/endtoend-venv`, installs exactly the two versions §4 tells a learner to
install, and runs the program there. It resolved Python 3.12.14, NumPy 2.3.5, scikit-learn 1.9.1 and SciPy
1.18.1 and reproduced the recorded output byte for byte. The environment is isolated because a sibling
agent's install once resolved a different NumPy against the shared runtime; nothing was installed into
`scratch/lesson-tools`. Without the flag the evidence file records `isolatedRun: null` rather than implying a
run that did not happen.

Three guards were found **inert** by the harness and repaired rather than explained away.

1. Relaxing the selection reducer's comparison broke nothing, because no two candidates tie on these data. The
   reducer is now the exported `bestByRule`, swept over every assignment of three values to four positions in
   both directions — 162 cases, 90 of them with a repeated extremum — so the tie rule is exercised on data
   that contains ties.
2. Giving every bar a minimum length broke nothing, because the floor tried was shorter than every bar. The
   proportionality check now runs across all three metric charts, and records that the shortest bar on the
   page is short enough for a floor to change it.
3. The untagged-SVG breakage could not be applied, because the target text was in the wrong file. It now
   targets the shared canvas, and a second case covers the distinct hazard that a figure is passed *some*
   class that the stylesheet's layout rule does not match.

**Crash safety was proved by an actual kill, not by inspection.** The harness was force-terminated
(`Stop-Process -Force`, which on Windows delivers no catchable signal) at the moment a `.orig` sidecar existed
on disk. The breakage was left applied, as expected; the next run refused to start and named the stale
sidecar; `--recover` restored the file and every one of the fourteen owned sources hashed identically to its
pre-kill state, with no sidecar and no lock remaining.

The KaTeX hazard is live on this page: it renders three SVGs of its own — the radical in §8's Wilson formula
and the two stretchy delimiters of practice 1's matrix. The stylesheet's layout rule is scoped by class, the
hygiene checker refuses a bare descendant `svg` selector at the selector level, and the browser verifier
measures every `.katex svg` in **both** dimensions, because a collapsed radical is also narrow and a
width-only check reads as correct while the defect ships.

### What is not covered

- **Nothing has been rendered in a browser.** The lesson was server-rendered once as a smoke test during
  implementation, which establishes that the component tree does not throw and that no held-out quantity is in
  the markup on first paint. It does not establish that anything is legible, that a label does not collide
  with a line, or that a control can be operated. The five browser falsification cases are deferred with it.
- The split index lists are reproduced by the same two stated `train_test_split` calls; the meaning of a seed
  is the library's own stream. Every property the lesson relies on — exact partition, disjointness, sizes and
  preserved cultivar representation — is checked independently, and the 178 leaves are recorded as validated
  rather than re-derived.
- The fitted coefficients and tree structures are obtained by refitting with the same estimators, because the
  estimators are the object under study. Everything computed from them is independent.
- No independent content or learning-experience review has been performed. The author's own reading is not
  independent review.

### Phase C: browser review

Run 20 September 2026 against a production build (`dist-e2e`) previewed on `127.0.0.1:4197 --strictPort`,
with Playwright 1.59.1 driving Edge. The lesson was registered between phases: 229 manifest entries, position
39 of the Classical Machine Learning module, and `module.topicIds[38]` asserted on the live route.

**13 browser cases pass and 27 captures were taken**, each with its `{file, digest, bytes}` recorded, every
digest and size re-checked afterwards, and an orphan check that finds no stray `endtoend-*.png`. **All five
deferred browser breakages fire and name their guard**, bringing the harness to 37 of 37.

#### What the browser found that nothing offline did

Five defects. Two were in the lesson, two were in the evidence, and one was in a verifier.

- **Two SVG labels overlapped.** In the information-lane figure, each lane carried a full sentence —
  "Changes the model's fitted state" is thirty-one characters — inside a ninety-six-unit box. SVG text does
  not wrap, so it ran across the figure and landed on the pipeline's own labels. The sentences now live in
  the HTML table beside the figure, where they reflow, and the drawing carries one short label per lane. In
  the bar chart the y-axis title overlapped the topmost tick label; `padding.top` is now a measured band
  rather than a margin, on all three plot geometries.
- **The threshold rule crossed a tile's own number.** On the acceptance rail, the case sitting exactly at the
  threshold has its tile at exactly the threshold's x, so a full-height rule ran straight down its number —
  and the boundary case is the most interesting one on the rail. It was invisible only because the tile
  rectangle is opaque and painted afterwards, which is paint order doing the work of layout. The rule is now
  drawn in two segments, above and below the tile band, and a model check asserts it enters neither the tile
  band nor the axis tick labels.
- **A categorical axis title read as part of a data label.** "candidate" sat a few units under
  "linear_three" and "actual cultivar" under "cultivar 2", at every width. The title is now anchored at the
  left for a categorical axis, and the bottom padding holds two rows rather than one.
- **Capture artifacts blanked part of the evidence, three times over.** Chromium stitches an element
  screenshot that exceeds the viewport and the seam painted as an opaque band across two investigation
  captures; growing the viewport fixed that and exposed the next layer, where an element screenshot scrolls
  its target to the top of the viewport, which is exactly where the reader's sticky header sits, so the
  investigation headings came back covered. Tall elements are now captured as a full-page screenshot clipped
  to the element's **page** box — `boundingBox()` is viewport-relative and a full-page clip is measured from
  the document top, so the scroll offset is added rather than assumed to be zero — with the document scrolled
  to the top first so the sticky header sits above the content it belongs to. Evidence that cannot be read is
  not evidence, and these images are what an independent reviewer reads.
- **A verifier threw instead of checking.** The route-link check resolved anchors as CSS selectors, and every
  section id on this page begins with its number; `#1-write-the-question...` is not a valid selector, so the
  check raised a harness error rather than reporting a missing anchor. It resolves with `getElementById` now.

#### A pin that could never match

The browser verifier pinned the Wilson bounds at six decimals, the precision every other computed value on
the page uses. Section 8 prints them at three, as the manuscript states them. The absence assertion therefore
passed trivially and proved nothing — and it was the **paired presence assertion** that caught it, which is
exactly why the two are always written together. The pin is now the bracketed pair the page actually renders,
still computed in the verifier from the model layer rather than read off the page.

#### What was confirmed by looking

Every capture was opened. The held-out gate behaves as designed on the real page: before either commitment,
none of the three held-out scores, the Wilson pair or the held-out confusion matrix is in the document, and
three sealed placeholders stand in their place. After the first commitment the selection criterion appears —
the same number the table above shows as a validation measurement, now badged **selection** — and the
held-out quantities are still absent. After the second, the report opens and prints
**held-out** beside each of its three scores, next to the validation estimate it exceeded. Selecting a
candidate the study never evaluated on the test rows produces the refusal in place of a number.

Before the gate the page prints 31 scores carrying only the training and validation roles; after it, 41
carrying all four. No score anywhere lacks a role.

KaTeX's three SVGs — the radical in section 8 and the two stretchy delimiters in practice 1 — paint
203.6 × 24.8 px and 16.9 × 69.7 px at 1366, 390 and 320 px. The floor is now 4 px in **both** dimensions
rather than "not zero", and the count measured at each narrow width is required to equal the desktop count so
the check cannot go inert by the selector ceasing to match.

#### A limit of the scatter, stated on the page

Two development specimens sit .05 apart in colour intensity on opposite sides of the default cutoff of 4.
That is under a pixel at any width the figure can be rendered at, and no size separates them. Rather than
move quantitative marks apart, the scatter now says so: the slice count and the specimen inspector are the
authority on membership, and the picture shows the shape of the slice rather than its case-by-case
composition. The figure is rendered wider than its viewBox so the rest of the scatter is not needlessly
cramped, and the paired strip is rendered wider too, since thirty-six columns at the viewBox's own width gave
each specimen about seven pixels.

#### Still not covered

- No independent content or learning-experience review. The author's own reading is not independent review.
- Screen-reader behaviour is not simulated; roles, labels and descriptions are asserted structurally.
- The painted-against-declared check catches a stylesheet overriding an attribute, a shape painting black
  with no stroke and a shape painting the page ground. It does not catch a shape painted correctly and then
  covered by a later sibling — which is precisely the class the acceptance rail's threshold rule fell into,
  and it was found by looking at the image rather than by any check.
- The held-out property is asserted on rendered text. It establishes that the quantities are not in the
  document; it does not establish that the JavaScript bundle lacks them, which it does not and cannot, since
  the page computes the report once the gate opens.

## Disposition of the independent review

Reviewed 20 September 2026 against the sources listed in
[the review](../../END-TO-END-INDEPENDENT-REVIEW.md); dispositioned 21 September. The reviewer refit the whole
study from the served CSV in exact rational arithmetic and 50-digit floating point, compared 576 quantities,
attacked the held-out gate through six surfaces and drove all 48 selection settings through the real controls.
**It found no disagreement with any number this lesson states.** One blocking defect, eight should-fix items
and ten observations followed. Every one is dispositioned below. Nothing was declined as wrong; one was
declined as already correct, with the reviewer's agreement.

### B1, blocking — the grader compared a quantity its question did not name

Investigation C asked *"What total cost will the proposed rule have?"* and graded against
`comparison.difference`. At the default state the correct answer is 12 — printed in the page's own comparison
table, in the same viewport — and the page replied that 12 did not match while accepting −2, which is not a
total cost at all. **Fixed**, and fixed as the class rather than the site, because this was the fifth variant
of it across this effort and every previous repair was written against the last instance's surface.

The instance is one word: `value: proposed.cost`. The class is a property now asserted in the browser:

> the quantity a question **names** is the quantity its grader **compares**, and the reference is the value
> the page **itself displays** for that quantity.

Each numeric question declares a `quantityKey`; the element that prints that quantity carries
`data-graded-quantity` with the same key. `assertGradesWhatItDisplays` then commits once to open the reveal,
reads the number the page prints, resets, types exactly that number back in, commits again, and requires the
verdict to say it matches — then types a neighbouring value and requires it to be rejected, so a grader that
accepted everything would not pass either. It runs for every numeric field on the page, and asserts the count
of fields it covered equals the count that exist, so a graded question cannot be added without being covered.

Two falsification cases repoint a grader at a neighbouring quantity — the cost difference in one
investigation, the reference error count in the other — and both fire.

**Why the existing evidence missed it, which is the more useful finding.** The reviewer noticed that
`endtoend-investigation-c-after.png` was captured with the numeric field **empty**, so the verdict in the
evidence carried no numeric clause at all. The one capture that could have exhibited the defect never
exercised the path — the same shape as a breakage that survives because its state is unreachable in the
opening view. The class guard now types into that field on every run.

### The should-fix items

**S1 — a Wilson pin at a precision nothing produces, in an absence-only check.** The models verifier asserted
the absence of `0.858300` and `0.995080` from the *program's stdout*, which computes no interval at any
precision, and neither had a presence partner. This is the identical defect found and fixed in the browser
verifier during Phase C and **never back-ported** — and it survived precisely because that file broke the
paired rule for those two entries. **Fixed.** The leak block now draws its absence and presence assertions
from one list, so an entry cannot be absent-only; the Wilson bounds are not in it, because the program never
produces them, and a separate check says so in as many words. A falsification case adds a quantity the program
never prints and fires on the presence half.

Both remaining verifiers were then swept for the same shape, which found three more in the browser file
(S8): two of three `test 0.966667`-style pins could never match, `[[12` had no presence partner, and a
`.ete-table caption` check could never fail because the `Table` component renders its caption as a sibling
paragraph by deliberate convention. All three now carry a partner, and the caption check asserts the sibling
exists rather than only that the `<caption>` does not.

**Then the same defect appeared a third time, inside this very fix.** Badging the two Wilson bounds put a
role label between them, so `[0.858, 0.995]` stopped being one run of text and the absence assertion began
passing for a reason that had nothing to do with the gate. The paired presence assertion caught it
immediately. That is the point worth keeping: **an absence check silently becomes vacuous whenever the string
it hunts stops existing — and a perfectly correct change can cause that.** It is why absence may never stand
alone, and this is the third independent confirmation of it across this effort. The pins are now the two
bounds separately, each with both halves of the pair.

**S2 — `round()` made an encoding guard unfailable.** `round(h / 100 * 100) != h` is false for every input;
the reviewer measured 0 failures as written against 143 without the `round`. Deleting `round` would have been
wrong too — those 143 are ordinary float behaviour. **Fixed** by asserting the property the control actually
needs: the 1,401 cutoffs are pairwise distinct doubles and strictly increasing, so no two settings silently
mean the same cutoff and moving the control never moves the cutoff backwards.

**S3 — held-out numbers printed with no role badge.** Against the lesson's own headline invariant, `35/36`
in the compact report and the Wilson pair were bare text. **Fixed**, and the guard that missed them was
replaced rather than patched: `collectScoreRoles` enumerates `.ete-score` and so can only inspect numbers
that already carry a badge. The complement now sweeps every score-shaped text node and requires it to sit
inside a badge — and it immediately found **three more** the review had not listed: the §5 balanced accuracy
in a table footnote, practice 1's balanced accuracy, and the ranking scores inside investigation D's own
generated verdict sentence. That is the argument for framing a guard as the complement rather than the
instance.

Its four exclusions are **declared on the element**, never assumed by the sweep: program stdout (marked
explicitly, because the shared `CodeBlock` renders a styled `<div>` rather than a `<pre>` and matching another
component's styling would break silently when that component changes); SVG text; a subtree declared a
constructed fixture; and an individual number carrying a `data-role-exempt` reason. The sweep asserts each
declaration exists and states a reason, so a region it cannot classify is one somebody wrote a sentence about.

**The worst thing this guard found is worth stating plainly: the lesson committed its own subject matter.**
The acceptance queue's counts were badged **validation**. Those are ten invented inspection cases, and the
badge claimed they were measured on the 36 wine development rows. A page whose entire argument is that a
number must be labelled for the evidence it actually is had mislabelled a constructed fixture as this study's
development evidence — the exact error it teaches a learner to catch, committed by the teacher. Practice 1's
invented confusion matrix was the same shape. Both are now declared constructed fixtures, outside the
four-role system, with the declaration carrying its reason on the element. The training-minus-validation gap
column, which belongs to neither role, is a declared exemption rather than a bare number.

A lesson that commits the defect it exists to teach is the most useful kind to document, because it shows
that stating a discipline is not the same as being held to one. What holds a discipline is a check whose
domain is the whole page rather than the cases already conforming.

**A guard that enumerates forms will miss the form nobody thought of.** The falsification case for this
guard was itself inert at first: the breakage removed a badge so the compact report printed `35/36`, and the
sweep's list of held-out forms contained only `35 of 36` — the spelling `<Count>` produces when the badge is
*present*. An omission does not produce the badged spelling; it produces whatever an author types. The bare
`n/m` form is now in the list, and the general lesson is recorded here rather than only fixed: a
form-enumerating guard needs its list built from what a *failure* looks like, not from what success looks
like.

**And it was still inert after that, for a second and different reason.** With the bare form in the list the
case remained inert, because React renders `{heldOut.correct}/{heldOut.total}` as **separate text nodes** —
`35`, `/`, `36` — so the string `35/36` exists in the rendered block and in **no single node**, while the
sweep walked nodes one at a time. The guard was looking at a unit smaller than the thing it hunts. It now
strips the excluded regions and joins what remains of each block before testing, which is sound in one
direction only and that is the direction needed: removing a badged span can break a string apart but can
never fabricate one, while joining sibling nodes is what makes a real omission visible.

Two distinct domain gaps in one guard — which *spelling* the list held, and which *unit* the sweep examined —
found one after the other, each by the falsification case refusing to fire. Neither would have been visible
from reading the guard, and both were invisible to a passing run.

**S4 — stale counts.** The record said 283 checks in 51 groups against 285/52 in the tree; the verifier's own
header comment and its published `scope` string both said **25,218** slice comparisons against the 44,832 it
runs and asserts, and the floor at `sliceCases >= 25000` had been written against that stale figure. All
corrected, and the floor raised to 44,000. The current figures are in the table below.

**S5 — the harness protected sources across a kill but not evidence.** The evidence snapshot lived only in the
process heap and was restored in a `finally`, which covers a throw but not a forced kill — and a `--browser`
case, a production build plus a browser run, is the widest kill window there is. **Fixed** at the root rather
than by adding a second sidecar: the browser verifier now takes `--no-evidence`, sending its report and its
captures to `scratch/`, and the harness passes it. A falsification run no longer writes the record it is
falsifying at all, killed or not. The heap snapshot stays as a second line of defence.

**S6 — 144 self-compared trust-root leaves.** `compare(".../model", key, row["model"])` where `key` *is*
`row["model"]` — 6.5% of the "100% re-derived" total were comparisons that could not fail. **Fixed**: the key
is derived from the row's position in the packet's own block ordering, independently of the row's contents. A
falsification case perturbs that derivation and fires.

**S7 — a cluster of containment checks that read the value they guard, including the `MINIMUM_INSET` floor
written specifically to prevent that.** This is the one worth sitting with: the guard against
self-referential guards was itself self-referential. It asserted `MINIMUM_INSET >= 12` and called it "the 12
units every containment check assumes", and no containment check read it — the three geometries carrying it
were checked against their `padding`, and the other four against their own insets, so `lane.x >= lanes.inset`
compared a value with itself and the rail's tiles sat inside a scale built from the inset for any inset at
all. **Fixed**: every containment check now compares against a literal `HARD_MARGIN`, which the positions are
not derived from. Two falsification cases zero an inset — the lane figure's and the rail's — and both now
fire where neither would have before.

The threshold-segment check (S7d) defined its bound from the same `tileBand` the segments are derived from,
so both disjuncts held by construction for any geometry. It now asserts against the two label positions the
component actually draws text at, including that the gap between the segments is where the tile number is.
The string-length tautology (S7e) was deleted; the property it claimed to test is really tested in the
examples verifier, against the program's actual stdout.

### Observations

**O6** — investigation D's verdict read `draft.metricKey` rather than the committed `shown.inputs.metricKey`.
Safe today because the controls disable on commit, but one edit from the "displayed from the draft, graded
against the applied state" shape. **Fixed.** **O7** — the dead `validated_not_rederived.update()` is removed.
**O8** — `allRestoredExactly` treated a missing field as success, so a case that was never applied reported as
restored; it now requires `applied === false` or `restoredExactly === true` and reports not-applied separately.
**O9** — neither Python verifier floored its own counter; both now do, which matters most for the examples
verifier, whose five pattern-matching branches could all have matched zero lines and still printed PASS.
**O4** — one sentence now maps the figure letterings: **the manuscript's Figure D is the page's Figure C**,
and the page's Figure D is the per-class recall chart, which the manuscript does not letter.

**O1, the declined item, and the reviewer agreed it should be declined.** Specimens 28 (colour 3.95) and 146
(colour 4.00) sit .05 apart on opposite sides of the default cutoff — about one pixel at any renderable
width. Moving quantitative marks apart to make the picture legible would falsify the measurement, which is
the worse error. The slice count and the specimen inspector remain the stated authority, and the caption
beside the figure says so.

**A3's refinement, adopted.** The float32 routing difference is at the *leaf*, so it changes the
probabilities (by up to 7.6 × 10⁻²) but **not the predicted labels**, which agree on all 36 validation
specimens either way. The Phase A record's wording was accurate about routing and could have been read as a
prediction disagreement. The reviewer also confirmed independently that casting the input only is correct and
casting the threshold as well is wrong, which is what `verify-endtoend-data.py` does.

### A harness defect found by two readers reaching opposite wrong conclusions

While the full falsification run was in progress, its on-disk artifacts — a lock file and a `.orig` sidecar —
were read by **two independent readers, who drew opposite conclusions, and both were wrong**. The coordinator
read a task notification that reports only an agent's live children, and concluded the run was dead. The
builder armed a monitor using `kill -0` on the process id, and concluded the run had exited. Acting on the
second, the builder ran `--recover` against a **live** run: it restored a file the harness had deliberately
mutated, mid-case, so that case's verifier ran against unmutated source and would have reported INERT for a
reason having nothing to do with its guard. That run's result was discarded rather than published, because a
falsification record describes a specific state of the tree and this one no longer did.

Two environment findings came out of it, both cases where the obvious POSIX reflex is silently wrong on
Windows, and both now recorded for the handoff:

- **`kill -0 <pid>` from Git Bash against a native Windows pid is unreliable.** It reported an exit that had
  not happened. `tasklist //FI "PID eq <pid>"` is the correct instrument. This sits beside the finding that a
  forced kill delivers no catchable signal here, which is why the sidecar rather than a signal handler is the
  real protection.
- **A lock plus a sidecar cannot distinguish "crashed" from "case in progress."** That ambiguity is the
  actual defect. It is not hypothetical and it is not a one-off: it misled two readers within the same hour,
  in opposite directions, from the same two files.

**Fixed rather than recorded as a limitation** — but the first fix was itself inert, and that is worth more
than the fix.

The lock was given a heartbeat refreshed by `setInterval`, and `--recover` was made to refuse while that
heartbeat was fresh **or** the process id was still in the process table. Measured against a live run, the
heartbeat never advanced: `heartbeatAt` sat 3 ms after `startedAt` for the whole run. **A timer cannot fire
in a process that blocks its event loop with `spawnSync`, and this harness drives every child that way — so
the heartbeat was inert precisely while a case was running, which is essentially the entire lifetime of a
run.** That is the same shape as the two defects it was written alongside: a guard whose domain excludes the
only situation it exists for.

What made the gate sound anyway was redundancy that was designed in for a different reason: the refusal
consults **`tasklist`** as well as the heartbeat, and the process check is reliable here. So the gate worked;
the heartbeat contributed nothing to it. The record says that plainly rather than claiming a working
heartbeat, because a field that looks like liveness and never moves is worse than no field — a reader seeing
a stale `heartbeatAt` beside a live process has to know which to believe.

The repair makes the heartbeat real without a timer. Every blocking child now runs through one wrapper that
writes the lock immediately **before** handing control to the child and immediately **after** getting it
back, which are the only two moments this process can write anything. The staleness window is widened to
exceed the longest single blocking call — a production build followed by a browser run — and, because a
widened window would otherwise make a dead process look live, **the process check is made authoritative and
the heartbeat secondary**: the heartbeat decides only when no process id is recorded. A falsification case
removes the post-child refresh and requires a probe to observe `heartbeatAt` failing to advance across a
blocking call.

The generalisation for the handoff: **a timer-based heartbeat is inert in any harness that uses `spawnSync`,
which is most of them in this repository.** Refreshing synchronously around each blocking call is the fix
that survives that pattern.

### The harness resolved its anchors by position rather than by identity

Found while wiring the liveness gate, and the most consequential tooling defect in this scope.

A falsification case names a `find` string and the harness applies it with `String.replace`, which changes
**the first occurrence**. The gate's breakage used the anchor `if (live && !process.argv...`, and that string
occurred **twice in the harness file — once at the gate and once inside the case's own `find` field**. The
mutation landed on the case definition. The gate stayed intact, the verifier correctly refused, and the case
reported INERT for a reason having nothing to do with the guard it was aiming at. A reader would then have
"fixed" a guard that was never broken.

The harness now **refuses any case whose anchor is not unique in its target file**, rather than silently
resolving to the first match. That guard immediately caught a second latent instance: the untagged-SVG case's
anchor matched both `PlotFrame`'s and `Canvas`'s `<svg>`, so which one it broke was luck. Both anchors were
lengthened until unique.

A harness that mutates the wrong thing by luck is worse than no harness, because it produces confident green
output. **A mutation-based harness is only as trustworthy as the uniqueness of its anchors, and nothing was
checking that.**

This is the same shape as the bare-form gap above, one level up. There, a guard enumerated what **success**
looks like and so could not see a failure. Here, a harness resolved an anchor by **position** rather than by
**identity** and so could not tell which site it had broken. Both are cases where the checking machinery had a
degree of freedom nobody had constrained — which is a more useful way to look for inert guards than
enumerating their known shapes.

Two smaller self-references were fixed alongside it. The source verifier **skipped** its liveness-gate check
whenever a lock existed — which is exactly when the harness runs it, so the check was unreachable from inside
the harness; the lock path is now overridable so the refusal can be exercised in isolation. And the refusal is
asserted **behaviourally**, by performing it against a throwaway lock, rather than by matching the source text
that refuses: matching text would have re-created the very class being eliminated.

### A wrong diagnosis, withdrawn

Recorded because a record that shows a wrong conclusion being retracted is worth more than one that never
mentions it.

While investigating a browser-verifier failure, the builder reported that the harness's **restoring** rebuild
died of `ENOBUFS`: the per-case builds passed `maxBuffer: 64MB` while the restoring one used Node's 1 MB
default, and a stale build directory was demonstrably the cause of the failure. The asymmetry was real and
the stale build was real. The mechanism was invented — and it was reported before it was measured, and
relayed upward on that basis.

The measurement that disproved it:

```
default maxBuffer -> status: 0 | error: null | signal: null
stdout bytes: 180858
```

180,858 bytes against a 1,048,576 default. The restoring rebuild was never close to the limit. **Withdrawn.**

The underlying problem was nonetheless genuine, and is fixed: the restoring rebuild now goes through the same
call as the case builds and **its status is checked** rather than assumed, so a failed restore says so loudly
instead of leaving the last breakage's build on disk. The field that reported success was also renamed —
`allRestoredExactly` reads as a claim about everything a run touched and was only ever about source files; it
reported `true` throughout while a stale build sat on disk. It is now `allSourcesRestoredExactly`, with
`buildDirectoryRestored` beside it, because a field that overstates its scope is the record-contradicts-tree
class.

**The transferable lesson of this session sits here rather than in any single defect.** Across one working
session, this builder produced four significant findings. Two were measured before they were stated — the
anchor-uniqueness hazard and the heartbeat-versus-`spawnSync` inertness — and both held. Two were inferred
and stated before an experiment — the `ENOBUFS` mechanism and a liveness reading taken from `kill -0` — and
both were wrong, one of them costing a discarded falsification run. Same author, same hour, same care. **The
difference was never judgment; it was whether the experiment ran before the sentence.** Every guard in this
lesson is written on that principle: exercise the behaviour, do not match the text that claims it.

### A defect the review could not have caught, found while dispositioning it

While applying these fixes a Python heredoc ate a `\'` escape and left an unescaped apostrophe in
`EndToEndLabs.jsx`, breaking `npx vite build` **for the whole repository**. A sibling lesson found it because
it could not regenerate its own evidence. This is failure class 2 from the brief — heredocs eating
backslashes — recurring in the tooling rather than in the content, and nothing offline saw it: the model and
data verifiers import plain `.js` and never touch a component, and the text scans do not parse.

The source-hygiene verifier now **parses every owned `.jsx`** with esbuild. It is the one assertion in that
file that has already fired in reality, and a falsification case reproduces it. Heredocs are no longer used
for any content containing a backslash.

The generalisation is the part worth keeping. This is the same mechanism that once turned a `\b` into a
literal backspace byte *inside a verifier*, leaving a guard permanently inert — and the lesson is that a
heredoc corrupts **the thing that writes the code as readily as the code**. A verifier is not a safer place
for that hazard than a component; it is a worse one, because a corrupted guard fails silently while a
corrupted component fails loudly. That is why this lesson's hygiene verifier audits its own verifiers and
harness alongside its sources, and why the audit now includes a parse rather than only a regex.

### An evidence file rewritten wholesale cannot be read as cumulative

`endtoend-falsification.json` is replaced by every invocation, and that misled a reader twice in one session.
An offline-only run replaced a 51-case offline-and-browser record with a 43-case one: nothing about the
lesson was wrong, but the record then understated what had been proved and `buildDirectoryRestored` reverted
to `null` — the very field added so that a reader could tell source restoration from build restoration.

The standing advice — run offline-only **before** `--browser`, never after — is correct, and a sibling lesson
recorded it after hitting the same trap from the other direction. But a rule that has to be remembered is not
a fix, and the instinct that collides with it is a reasonable one: re-running the cheap pass last, to re-pin
the record against the bytes that actually shipped.

So the artifact enforces it. A run covering fewer cases than the record already on disk writes itself
**beside** that record rather than over it, and says so on stderr. Verified in both directions: against a
broader record the target redirects to a mode-suffixed path and reports `narrowed: true`; against a narrower
one it replaces as normal.

This is the same conclusion reached about the lock, and it is the third time in this session it has come up:
**when an artifact cannot answer a question a reader will ask of it, put the property in the artifact rather
than in the reader's discipline.** A lock that could not distinguish a crash from a case in progress, a field
named for more than it covered, and a report that silently narrowed are three instances of one thing.

### The crash-safety machinery was exercised by three unplanned events

Not by a contrived case. Over this scope the harness was interrupted three times for reasons nobody
arranged: a spend limit killed it mid-case; `--recover` was run against a live run on a mistaken liveness
reading; and the shared `vite build` failed while a breakage was applied, almost certainly a sibling lesson
mid-edit — the receiving end of the same hazard this lesson's own unescaped apostrophe caused earlier.

Each time the `.orig` sidecar restored the mutated file and the tree came back clean: no source left broken,
no lock stranded after the final event, and `--recover` available and correct for the one case that needed
it. That is stronger evidence that the sidecar was worth building than any falsification case could be,
because none of the three was designed to test it.

The one that cost something was the second, and it cost a whole falsification run rather than a file — the
run had to be discarded because a case had been un-mutated underneath it. That is what the liveness gate now
prevents.
