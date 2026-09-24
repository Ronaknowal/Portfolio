> **Current interaction amendment, 21 September 2026:** Read [live-exploration.md](live-exploration.md). Labs now update directly from valid edits and contain no learner prediction feature, including optional predictions. The older prediction/commit/reveal clauses below are historical design records; their numerical, scope, layout and evidence requirements remain applicable where unchanged.

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

## Phase two: implementation, 20 September 2026

Appended by the implementation author. Everything above this heading is the frozen content checkpoint and was
not edited. The packet's nine files are byte-identical to their recorded state; `lesson.md` still hashes
`1ff63549d7055024cdc8a4a982d1af05b6c575e305b824f88f90d8f0642b3139` and `calculated-inputs.json`
`d694b6546ba00e7bc6d4592d21e973a4f7faa3d81551b252762b5b7d26b405a7`.

This lesson was a `planned` topic with no published body and no manifest entry, so phase two created one rather
than replacing one. Registration in `lesson-manifest.json` and `blueprints/index.js` belongs to the increment
owner and was not performed here.

### Files created

Topic and models: `src/learn/data/topics/ml-problem-formulation-baselines-data-leakage.jsx`,
`src/learn/data/formulation-models.js`, `src/learn/data/formulation-data.js` (generated),
`src/learn/data/formulation-examples.js` (generated),
`src/learn/data/curriculum/blueprints/ml-problem-formulation-baselines-data-leakage.js`.

Components: `FormulationShared.jsx`, `FormulationLabs.jsx`, `FormulationFigures.jsx`, `formulation-labs.css`
under `src/learn/components/lesson-labs/`.

Served assets under `public/learn-assets/problem-formulation/`: `bank-additional.csv` (the packet's file, byte
for byte), `ATTRIBUTION.txt`, `bank-marketing-variable-description.txt` (the provider's own description) and
`formulation-calculations.py` (the packet's `author-calculations.py`, byte for byte).

Verification: `scripts/verify-formulation-{models.mjs,data.py,examples.py,sources.py,browser.cjs}` and
`scripts/falsify-formulation.mjs`, with evidence under `docs/teaching/evidence/formulation-*.json`.

### What was implemented

Seven inline figures and two investigations. F1 is the five-stage decision flow with the feedback channel that
returns only mature outcomes; F2 is nine parcel observations under two allocations, with the identity overlap
derived from the allocation rather than captioned; F3 is the expected-cost crossing; F4 is the fixed partition
and the feature families against the call; F5 is the three measured procedures one metric at a time, with each
procedure's availability contract rendered beside its score; F6 is three information paths with their different
repairs; F7 is the propensity-against-impact reversal with the stocking costs. I1 is the availability timeline
of the destination note; I2 is the 824-row ranked action set.

Every drawn coordinate lives in `formulation-models.js` and is asserted there. Every SVG is created by one
shared `Drawing` wrapper, which applies the class the stylesheet's layout rule is scoped to; the source-hygiene
verifier refuses a hand-written `<svg>` anywhere in the figures or labs, so a new figure cannot omit it. The
stylesheet contains no bare descendant `svg` selector, and that is asserted in two places.

### Departures from the visual specifications, with reasons

1. **One figure added.** The specifications list F1–F6; the implementation has seven. The extra one is the
   expected-cost crossing in section 4. The manuscript derives the threshold algebraically and practice 6 asks
   a learner to compute it; a picture of two straight lines crossing makes "`.5` is the threshold only when the
   costs are equal" visible rather than stated. It is exact arithmetic on declared costs, its crossing is
   placed by the closed form, and the verifier requires that crossing to lie on both drawn lines at all 144
   cost pairs.
2. **F3's two panels share one scale.** The specification asks for the partition and the availability axis as
   separate diagrams, which they are. Within the partition, both bars are drawn on the one
   4,119-row scale, so the second visibly sits inside the development segment of the first instead of being a
   second bar at a second scale that happens to look similar.
3. **I1's arrival control reaches below its own event time.** The specification bounds each arrival edit at
   "its event time through 12". It is bounded at 0 through 12 here, and an arrival earlier than the event it
   measures is refused by name with the reason spelled out. An unreachable guard is an untestable guard: with
   the specified bound, the refusal path could not be exercised in the browser at all.
4. **I2 disables capacity while the action set is edited.** The specification offers two ways to avoid an
   ambiguous interaction between a manual exchange and a later capacity change; this takes the second,
   explicitly restoring the model ranking first. The capacity field is disabled with a named reason and a
   restore button, and switching the score source clears any edits.
5. **I1's numeric guess is optional; I2's is required.** The graded answer in I1 is a record identity, and the
   `none` case has no number to compare against, so a required numeric field would force a meaningless entry.
   I2's graded answer is a precision, which always exists.
6. **The manuscript's section 5 results table is rendered inside F5** rather than as a separate table, with an
   added column naming each procedure's information contract. Same numbers, one place.

### One packet claim that does not hold, recorded rather than repaired silently

`lesson.md` section 8 states: "The earlier quantile-regression lesson owns fitting that conditional quantity."
**There is no quantile-regression lesson in this curriculum.** Exactly one catalogue TITLE matches
`/quantile/i`: `streaming-quantiles-kll-t-digest-reservoir-sampling`, a planned sketching topic in
system-design. None teaches quantile regression.

**Two corrections to this paragraph, made after the independent review, 21 September.** It previously said "a
search of all 1,460 catalogue entries finds three unrelated uses of the word". Both halves were wrong. The
catalogue holds **1,499** titled entries with distinct slugs, not 1,460 — different counting bases could
explain that. What they cannot explain is "three unrelated uses": the word appears in at least seven entries'
titles, outcomes or subtopics, and **two of them are not unrelated**:

- **`calibration-conformal-prediction`** — published, and the lesson immediately before this one — defines
  **pinball loss** by name in its conformalised-quantile-regression section.
- **`decision-theory-risk-cost-sensitive-decisions`** — published, and earlier again — carries the identical
  derivation section 8 gives, under "absolute error requests a median; asymmetric error requests a quantile",
  with `τ = c_u/(c_u+c_o)` displayed, a worked underage and overage order, and the discrete-quantile subtlety
  this section skips.

The adjudication stands: the manuscript's sentence is false and was not rendered. But the first replacement
was materially incomplete — it routed the reader *outward* to scikit-learn while two published lessons behind
them owned the material. Section 8 now names both, and reserves the scikit-learn link for the one thing
neither supplies: the fitting technique. The search that justified the original wording missed the two nearest
neighbours, which is exactly the failure mode the lesson's own §6 is about — a claim resting on a search
nobody re-ran. The manuscript is still unchanged; this is the disagreement record.

**No numerical disagreement was found.** Every number in the packet reproduced. Average precision, log loss,
the confusion cells, the correct counts, the top-50 counts and the full 824-row rankings were recomputed from
the served dataset through implementations written independently of `author-calculations.py` and agree with it
to floating-point noise (worst probability gap and worst metric gap both under 1e-12). The constant baseline's
average precision is exactly the validation positive fraction 90/824, as the manuscript says. The six timeline
cases, the top-25 count of 12, the recorded exchange 589 → 1680 giving 11, and the reorder null all reproduce.
Running the packet's own `author-calculations.py` in a scratch workspace beside the served CSV rewrites
`calculated-inputs.json` byte for byte: 168,884 bytes, SHA-256
`d694b6546ba00e7bc6d4592d21e973a4f7faa3d81551b252762b5b7d26b405a7`.

### The destination note

`docs/teaching/topic-notes/ml-problem-formulation-baselines-data-leakage.md` is OPEN and was not edited. Its
proposal is fully implemented and its worked case is the lesson's own.

The note asks the learner to distinguish when an event happened, when its value became available, and when a
prediction had to be made. Section 3 names those three times in that order and the whole of investigation 1 is
built on them. The note's case — a prediction needed at time 5, a calibration with event time 4 arriving at
time 8, recent enough for a backward match on event time yet unavailable at the cutoff — is the fixture, and
the admissible answer is 10. Moving that arrival from 8 to 4 without touching any event time changes the
admissible record from event 1 to event 4 and the value from 10 to 20; that is a preset button, and it is
graded against a prediction the learner records first. The note's optional extension, a later revision of the
same event, is the third case: at cutoff 7 version 2 of event 1 is known and version 1 is retained rather than
overwritten. The note's warning against equating a database creation timestamp with serving availability is in
section 3's prose and again in the Feast paragraph, with the merged created-time change linked.

**Recommended disposition: close the note once phase C confirms the page renders.** The note itself says its
status remains open only because browser implementation and verification were deferred. The offline half of
that is now done and falsified; what remains is a rendered page. I have not edited the note — it is yours.

### Verification actually run

All four non-browser verifiers and the falsification harness are green on the sources hashed below. The browser
verifier is written but not run: the topic is not registered, so its route does not exist yet.

| Check | Result |
| --- | --- |
| `node scripts/verify-formulation-models.mjs` | 128 grouped checks across 56 groups; 50,700 as-known cases over every cutoff, every age, both entities and 150 arrival configurations (22,672 selections and 28,028 missing calibrations, so both branches are exercised); 2,197 null cases; 1,800 timeline geometries; 300 capacity cases; 58,675 exchanges with all three graded directions seen; 300 membership nulls; 144 cost pairs |
| `verify-formulation-data.py` | 10,163 checks; 100% of the trust root's 10,099 scalar leaves asserted; 35 independent property checks; module regenerates byte-identically |
| `verify-formulation-examples.py` | 40 oracles; both displayed programs extracted verbatim from the frozen manuscript (80 lines, 0 composed) and executed; the downloadable program reproduces the packet's results file exactly; module regenerates byte-identically |
| `verify-formulation-sources.py` | 107 hygiene checks over 16 files, the five verifiers and the harness included |
| `node scripts/falsify-formulation.mjs` | 29 of 29 offline breakages made the intended guard fire and name itself; 5 browser guards deferred and listed |

**Trust-root coverage, measured rather than asserted.** The comparison walks the packet tree and records the
path of every scalar leaf it actually compares; the run fails unless that set is the complete set. All 10,099
leaves are asserted. Of those, **3,506 are re-derived independently** — average precision summed over distinct
score thresholds from its definition, log loss written out, the confusion cells and top-50 counts counted here,
the ranking by an explicit sort rather than `numpy.lexsort`, the one-hot names rebuilt from the sorted training
categories, the as-known results from an independent implementation of the five-step rule, and the feature
lists and the calibration history parsed out of the frozen manuscript rather than retyped. **2,474 are
validated by refitting** the same scikit-learn estimator: the fitted probabilities and the two iteration
counts. The estimator is the object under study, so an independent reimplementation of regularised logistic
regression would be testing something else; everything computed *from* those probabilities is independent.
**4,119 are recorded split inputs** regenerated from their seeds through `StratifiedShuffleSplit`, a different
entry point from the packet's `train_test_split`, and separately checked for disjointness, exhaustiveness, size
and per-class stratification. Their exact order is a property of scikit-learn's random state and is not
independently reproduced. That is the whole of what is not covered.

### Crash safety, proved by an actual kill

The harness was started, allowed to apply its first breakage, and terminated with `Stop-Process -Force`, which
on Windows delivers no catchable signal. Observed:

- `formulation-models.js` was left mutated
  (`d346dd847f00c4e8f1c929a10d2fa5acf22a1481561158a5254320584721903d` against the original
  `f98a8c4bb6f2aac480a5cdc22965f856f207c3912beb2e441416c5e0e0ec5fb8`), with its `.formulation-falsify-orig`
  sidecar and the lock file on disk;
- a second run refused to start, naming the lock;
- `node scripts/falsify-formulation.mjs --recover` restored the file to
  `f98a8c4bb6f2aac480a5cdc22965f856f207c3912beb2e441416c5e0e0ec5fb8`, removed the sidecar and cleared the lock;
- the model verifier passed again immediately afterwards.

Every verifier also writes a **provisional record stamped `passed: false` before its first assertion**, so a
run that throws or is killed leaves a red record rather than the previous green one. That was checked by
breaking the availability predicate, running, and reading `passed: false` off the evidence file.

### Source versions this record covers

- `src/learn/data/formulation-models.js` — SHA-256 `f98a8c4bb6f2aac480a5cdc22965f856f207c3912beb2e441416c5e0e0ec5fb8`
- `src/learn/data/topics/ml-problem-formulation-baselines-data-leakage.jsx` — SHA-256 `e78a0c2b535120027f4a2e2a71731bb79de8a5189670eea9012ec213f38b5206`
- `src/learn/components/lesson-labs/FormulationLabs.jsx` — SHA-256 `dfbfbb09663c1b5fa4be10c6859c371664f40c3a0b9e0e411e56f91ec3a8cad7`
- `src/learn/components/lesson-labs/FormulationFigures.jsx` — SHA-256 `3a85fdcb78afe856e302f375eeea1fa148be65ca4fd1122b48c33699f75f2f84`
- `src/learn/components/lesson-labs/FormulationShared.jsx` — SHA-256 `f1b6a1f91c8389807b05235d3a9eb4818957aa6fd3031504ddc5c73fa715e141`
- `src/learn/components/lesson-labs/formulation-labs.css` — SHA-256 `4d63fc3e8961eacd16866c4f09cf591a2681384a86856d1e143f106d2e2e8d61`

The two generated modules are omitted from this list on purpose: their digests are recorded in
`docs/teaching/evidence/formulation-data.json` and `formulation-native.json` by the scripts that write them.

### What remains

Browser verification, the figure-by-figure and state-by-state visual pass, and independent review. The browser
verifier and its five falsification cases are written and waiting on a registered route; nothing in this record
claims a rendered page has been looked at.

## Phase C: browser verification and the visual pass, 20 September 2026

Appended by the implementation author after the topic was registered. The frozen packet is still untouched:
`lesson.md` hashes `1ff63549d7055024cdc8a4a982d1af05b6c575e305b824f88f90d8f0642b3139` and
`calculated-inputs.json` `d694b6546ba00e7bc6d4592d21e973a4f7faa3d81551b252762b5b7d26b405a7`.

Built to `dist-form`, previewed on `127.0.0.1:4195 --strictPort`, verified at 1366, 1024, 768, 390 and 320 px.
The build directory was deleted and the preview stopped when this record was written.

| Check | Result |
| --- | --- |
| `verify-formulation-browser.cjs` | **14 cases, 37 screenshots, passed** |
| `verify-formulation-models.mjs` | 130 grouped checks across 56 groups |
| `verify-formulation-data.py` | 10,163 checks; 100% of 10,099 trust-root leaves asserted; module byte-identical |
| `verify-formulation-examples.py` | 40 oracles; both programs executed; module byte-identical |
| `verify-formulation-sources.py` | 107 checks over 16 files |
| `falsify-formulation.mjs --browser` | **35 of 35 breakages fired and named their guard, 6 of them browser guards** |

**State of the falsification evidence file, and one episode worth keeping.** The record at
`docs/teaching/evidence/formulation-falsification.json` now reads `mode: offline and browser`, **35 of 35
breakages fired**, 6 browser guards run, none deferred, none inert, every file restored byte for byte.

It did not read that at first. An offline-only invocation of the same harness was run *after* the `--browser`
one and overwrote the record with **29 of 29, 6 deferred** — accurate for that invocation and an understatement
of what had been proven. Regenerating it needed a production build, and the tree did not build: a concurrently
authored position-39 file, `src/learn/components/lesson-labs/EndToEndLabs.jsx`, carried an unescaped apostrophe
inside a single-quoted string (`'…not this study's development or held-out rows…'`). That file is not owned
here and was not touched; all four of this lesson's components were confirmed to parse individually. The number
was left understated and the exact state recorded here rather than edited to what was known to be true — an
evidence file has to describe a run that happened. Once the sibling's fix landed, the build went green and
`node scripts/falsify-formulation.mjs --browser` restored the record on its own.

The ordering lesson is worth carrying: **run the offline-only harness before the `--browser` one, never after**,
because the later run's report replaces the earlier one.

### Nine defects found by looking at the images

None of these were caught by an assertion. Seven were repaired in the page; two were repaired in the verifiers.

1. **The cutoff label sat outside its own viewBox.** Its baseline was at y=5 with a nine-unit ascent, so a whole
   line of glyphs rose past the top edge. Found by the shared layout inspector at 1366 px. The top padding is
   now floored in the model with `LABEL_ASCENT`, both coordinates come from the model, and the offline sweep
   asserts the clearance on 1,800 geometries.
2. **The arrival marker ran through the label of the record it names.** The readable identity is sixteen
   characters, about 82 units, and the lane gutter is 58; the diamond for `A·event 1·v1` sat 28% inside its own
   label. Found by the curve sampler. The drawing now uses a compact identity, the readable one stays in the
   tables, and `timelineGeometry` refuses a label that does not fit its gutter.
3. **The arrival diamond covered the event dot wherever the two times coincide**, which is two of the four
   default records. Both marks drawn, both correct, one invisible: paint order beating colour, the class no
   attribute check can see. The diamond is now outline-only and slightly larger, and the browser verifier
   asserts its computed fill is `none` while its stroke is not.
4. **"An exact answer is required. Optional: leave it blank."** Both sentences were on screen together in the
   numeric prediction caption. Rewritten as one sentence per case.
5. **Three JSX expressions lost the space after them**, rendering `0.25and 0.75`, `is notdeterministic` and
   `in  pdays`. Invisible in the source and in the DOM structure. Two were found by reading screenshots, which
   is not a method that scales, so the third is now covered by a rendered-prose guard described below.
6. **The timeline and the metric bars were pinned at 300 px in a 760 px column**, which made the event-versus-
   arrival separation — the whole point of the centrepiece figure — hard to see. Their maximum widths are now
   480 and 470; the viewBox is unchanged, so the text scales with the drawing and narrow widths are unaffected.
7. **The selected-case strips centred themselves half a column away from their own labels.** They are now a
   two-column grid, which also lines the three strips up with each other so 6 / 20 / 26 reads at a glance.
8. **Figure 1 claimed the feedback channel "is annotated"** with a phrase that was never drawn. The caption and
   the accessible description now describe the channel instead of claiming an annotation the picture lacks.
9. **The legend rendered above a suppressed drawing** in the refusal state, legending nothing; and the edited-
   policy note read "0 exchanges and a reversal". Both gated on what is actually present.

### Two defects found in the verifiers themselves

- **A vacuous KaTeX guard.** This lesson's mathematics emits **no `.katex svg` at all** — no radical, no
  stretchy delimiter, no KaTeX-drawn accent — so a filter over that set passed while looking at nothing, and
  read exactly like a guard that works. The count is now **pinned at 0** rather than filtered, so adding a
  square root fails the assertion and forces whoever adds it to reinstate a dimension floor deliberately. What
  covers the collapse class here is the box measurement: 161 KaTeX boxes carrying text, **both dimensions**,
  floors of 4 and 3 against observed minima of 9 and 4.69, re-measured at 390 and 320 px with the count
  required to equal the desktop count so it cannot go inert.
- **A curve-sampler floor that was a guess.** It required ten shapes and the page draws five. The check now
  names the shapes it must reach — four `form-arrival` diamonds and the `form-arrow is-feedback` channel — so a
  class rename or a `<path>` turned into a `<line>` fails rather than shrinking a number nobody reads.

### Two guards added

- **A rendered-prose scan for lost spaces**, over 471 prose elements, excluding SVG text (whose concatenation is
  legitimate) and hex digests (a SHA-256 is sixty-four characters of digits followed by letters). The first
  pattern matched any word *ending* in a short word — "into", "hand", "This" — eleven false positives to one
  real defect; a guard a reader has to triage is a guard that gets deleted, so it is now anchored on the actual
  signature: a value touching the word after it.
- **A presence partner for every absence assertion.** An absence check alone cannot distinguish "correctly
  withheld" from "looking for a phrase the page never renders". The four phrases investigation 1 must not show
  before commitment are now asserted present after; so are the four outcome words of investigation 2; and the
  suppressed drawing in the refusal state is asserted to return.

### The two browser breakages that had to be rebuilt to fire

Both were INERT on first application, and both were measured rather than assumed.

- A stylesheet collapse aimed at `.katex .mfrac` did nothing, because KaTeX sets those spans inline and `height`
  on an inline box is ignored. Aimed at `.katex-display` it reached the built stylesheet — confirmed by grep —
  and the browser still reported a 16 px box. `font-size: 0` on `.katex` collapses every glyph to 0 by 0,
  measured before the case was written, and is the truer analogue: mathematics present in the DOM, correct in
  its markup, unreadable on screen.
- A timeline-inset breakage could not be constructed at all, because the model floor added in defect 1 refuses
  the value before the page can render, so the browser verifier fails on an unrelated timeout. That is defence
  in depth working, and it is recorded here rather than papered over. It was replaced by a breakage the model
  does not model: the drawing renders the readable identity while the model still computes the compact one's
  width, so no floor fires and the curve sampler has to catch it. It does.

### The availability contract on the rendered page

Verified visually and by assertion. The three clocks are distinct in the drawing — a filled event dot, an
outline arrival diamond, and a dashed line between them — with the cutoff as a separate vertical rule and the
admissible age window as a shaded band. The destination note's fixture behaves as asserted on the page: at
cutoff 5 the admissible record is `A · event 1 · v1` at value `10.000000`, and the preset that moves the
delayed event's arrival from 8 to 4, changing no event time, changes the selected record to `A · event 4 · v1`
at `20.000000`. Nothing states which record is admissible before a prediction is committed: the four reveal
phrases are absent beforehand and present afterwards, and the graded value's six-decimal text is absent
beforehand and present afterwards. Investigation 2 renders **no outcome for any of its 824 cases** until a
prediction is committed — every outcome cell reads "hidden until you apply" — and the starting precision is
stated in prose as `0.4`, never at six decimals, so the pin cannot pass trivially.

### Payload

Measured on the production build, gzipped at level 9. The ~51 KB data module compresses to 21,213 bytes and
leaves the lesson between two of its neighbours rather than above them, so no action was taken.

| Lesson | JS raw | JS gzip | CSS gzip | Total gzip |
| --- | ---: | ---: | ---: | ---: |
| Evaluation Metrics (33) | 131,630 | 41,548 | 2,318 | 43,866 |
| PAC Learning (34) | 175,993 | 57,070 | 3,219 | 60,289 |
| **This lesson (37)** | **170,650** | **63,130** | **3,129** | **66,259** |
| Rademacher (36) | 209,782 | 70,518 | 2,818 | 73,336 |
| Calibration (35) | 254,014 | 83,801 | 3,047 | 86,848 |

The route's full closure is 9 scripts, 1,761,450 raw bytes, 394,819 gzipped — shared chunks included; that is a
built-file measurement, not measured network compression or latency.

### Reading load

Investigation 2's ranking window was narrowed from 20 rows to 12. Below the narrow breakpoint every cell
becomes its own labelled row, so twenty ranked opportunities became a 7,700 px table sitting between the
capacity control and the prediction control — a learner on a phone could not see what they were changing and
what they were predicting at the same time. Twelve keeps six either side of the boundary and the window can be
moved anywhere in the ranking.

### Screenshots

37 captures, each with a unique path, a unique digest and a recorded `{file, digest, bytes}`, with an orphan
check that fails if any `formulation-*.png` on disk is not one the run just wrote. Re-checked after the final
falsification run, which snapshots and restores every capture it could touch: 37 recorded, 37 on disk, no
digest mismatch, nothing recorded-but-missing, no orphan, and the smallest capture 3,021 bytes. Twenty-two were opened and
read during this pass, covering every distinct figure at desktop, both investigations in every informative
state — initial, committed, refused, the membership null, the entity null and the destination note's moved
arrival — the identity split, the threshold figure at three cost settings, the results figure on each of its
three metrics, and 390 and 320 px for the investigations, figure 5 and the tallest formulas. The remaining
fifteen are the same figures at narrow widths, which are uniform scalings of drawings already read.

### Source versions this record covers

- `src/learn/data/formulation-models.js` — SHA-256 `d4b248ba818eb09722dfedc90640ddfb3af77649f9d877cf9c6c65610afb9545`
- `src/learn/data/topics/ml-problem-formulation-baselines-data-leakage.jsx` — SHA-256 `e78a0c2b535120027f4a2e2a71731bb79de8a5189670eea9012ec213f38b5206`
- `src/learn/components/lesson-labs/FormulationLabs.jsx` — SHA-256 `d95ea8ccc00a37d14c482085de528f471696b29c42a042cfd7b19dbe5bf1c1d7`
- `src/learn/components/lesson-labs/FormulationFigures.jsx` — SHA-256 `da1277acc759ed72e8cf2017a47f77d9e7fc4866ced49c1befd982eb90c1569d`
- `src/learn/components/lesson-labs/FormulationShared.jsx` — SHA-256 `cdf25e4cd530bafa581da3d9317bc441edceaf9d40eb93cc132379e2fc5ef95f`
- `src/learn/components/lesson-labs/formulation-labs.css` — SHA-256 `32988e8767bb32b844627f7a29e2a24822b08d5cb4107d4da83bd2f442164722`
- `scripts/verify-formulation-browser.cjs` — SHA-256 `925d99ad50c8bdddc6df53f245063519d559fb5616a75bc6fd28f446dc134a86`
- `scripts/falsify-formulation.mjs` — SHA-256 `26078d1d4f1a6c758a735c99c91bb91c3c0b5d49f8d8923926e46508ca036a59`

### What remains

Independent review. Nothing in this record claims a reviewer has read the lesson; it claims that the author
built it, verified it, falsified its guards, and opened its images.

## Disposition of the independent review, 21 September 2026

Review: [`docs/teaching/PROBLEM-FORMULATION-INDEPENDENT-REVIEW.md`](../../PROBLEM-FORMULATION-INDEPENDENT-REVIEW.md).
It found **no numerical disagreement of any kind** across a full independent refit — design matrix rebuilt by
hand, average precision summed in exact `Fraction` arithmetic, both splits checked for disjointness,
exhaustiveness, size and stratification, all six timeline cases, every §4 and §8 rational. The prior baseline's
average precision is exactly 45/412. It also confirmed all nine Phase C visual repairs in the images.

**Every finding is accepted and fixed. Nothing is declined.** Two blocking, nine should-fix, eleven
observations, plus two defects the coordinator surfaced from a sibling. The frozen packet is untouched;
`lesson.md` still hashes `1ff63549d7055024cdc8a4a982d1af05b6c575e305b824f88f90d8f0642b3139`.

| Check | Before | After |
| --- | --- | --- |
| `verify-formulation-models.mjs` | 130 / 56 groups | **138 grouped checks across 60 groups** |
| `verify-formulation-data.py` | 10,163 checks, 100% of 10,099 leaves | unchanged |
| `verify-formulation-examples.py` | 40 oracles | unchanged |
| `verify-formulation-sources.py` | "107 checks" | **82 failable assertions + 58 scan passes, 11 sources parsed** |
| `verify-formulation-browser.cjs` | 14 cases, 37 captures | **16 cases, 39 captures** |
| `falsify-formulation.mjs --browser` | 35 of 35, 6 browser | **42 of 42, 8 browser** |

### B1 — a tolerance tighter than the printed precision (blocking)

Investigation 2 graded at `1e-9` while printing at six decimals, so at capacity 3 the page rendered *"You
wrote 0.666667 …; the calculation gives 0.666667, **outside** 1.000 × 10⁻⁹ of it"* — two identical strings
declared different, with the guess `required`, at roughly four fifths of the capacities the control reaches.

Fixed as the **property, not the site**. `gradingAllowance({ declared, digits })` in the model layer returns
`max(declared, 0.5 × 10⁻ᵈⁱᵍⁱᵗˢ)`, and `Prediction` routes every numeric comparison through it using the same
`digits` it prints with. A lab can still declare a tighter intent; it can no longer grade tighter than it
displays. The verdict now reads "which matches to 6 decimal places" and the caption states a rule a learner can
act on rather than a raw exponent.

*Evidence it fails when it should.* Offline: a 300-case sweep over every capacity for all three rankings
requires the printed value, read back, to grade as correct — and asserts that **more than 150 of those 300
would fail at the declared `1e-9`**, so the guard cannot quietly become a tautology. Live: a browser case
drives capacity 3, asserts its precision does *not* terminate in six decimals, types the page's own printed
string, and requires acceptance. Falsification case *"the graded allowance drops below the precision the page
prints"* replaces the `max` with `declared` — **fires**.

### B2 — the page instructed one baseline and graded another (blocking)

The pre-commit caption named capacity 50 and precision .4 unconditionally, because it reappears after every
retirement, while the grader compares against the previously committed state. From round two a learner
following the on-screen instruction was marked wrong in a verdict that simultaneously confirmed their number.

The caption now computes from `state.active` through `selectionMetrics` — the same state `answerFor` reads —
and keeps the section-5 wording only on the first round. Baselines print through `round`, never `fixed`, so the
six-decimal form stays reserved for graded values and the leak pin stays sharp.

*Evidence.* A two-round browser case reads the caption off the page: round one must cite section 5 and name
capacity 50; after applying at 25, round two must name **capacity 25 and 0.48** and must not cite section 5;
then a learner reasoning from that caption predicts "It falls" and must be graded correct. Falsification case
*"the baseline caption goes back to naming the section-5 starting point every round"* — **fires**.

### The third inert guard, and the guard that gave it a green tick

`asInput` was `Number.isInteger(value) ? sign(String(value)) : sign(String(value))` — byte-identical branches,
the exact pattern the hygiene scan exists to catch, in a file that scan reads. The regex matched only quoted
string literals, and the falsification case injected a string literal, so the guard collected a green tick on a
shape it could not catch in real code.

`asInput` is now `sign(String(value))`. The scan is a balanced-paren walker over the construct: it skips quotes
and template literals so a colon inside a string cannot end a branch, and declines nested ternaries rather than
guessing at their extent — under-reporting is a stated limit, a wrong finding is not. Probed on a fixture it
reports the expression shape and the literal shape and ignores differing branches, optional chaining and
nullish coalescing.

*Evidence.* **Two** falsification cases now, not one: an expression-valued pair and a string-literal pair. Both
**fire**. Widening a guard can trade one blind spot for another, so both spellings are pinned.

### The remaining should-fix items

| Finding | Fix | Guard evidence |
| --- | --- | --- |
| **S1** figure 5's axis labelled "positives among the selected 50" and drawn as a 0–1 fraction | Relabelled "share of the selected 50 that subscribed"; no number moved | Browser case asserts the label is no longer the count wording and the axis is `[0, 1]`; the metric labels are now **read from the model**, so a rename fails a check rather than timing a selector out |
| **S2** `startsBeforeTheRuler` computed and read by nothing; caption said "−10 to 2" beside a band drawn 0→2 | The figure caption now reports the truncation and names the drawn extent; `drawnFrom` added | 169-case sweep asserts the flag equals `from < domain start`, that a truncated window really is drawn shorter, and that **the component reads the flag**. Falsification replaces the read with `false` — **fires** |
| **S4** the arrive-earlier preset was a no-op on sensor B | `withArrival` takes an explicit `entity` and refuses a record that does not exist; preset renamed to name sensor A | Asserted under both entity selections; falsification reverts to the queried entity — **fires** |
| **S6** the model verifier's header claimed local metric implementations that were imports | `averagePrecisionLocal`, `logLossLocal` and `confusionLocal` now exist; each measured value is checked **three ways** — module, local, scikit-learn. The header's blanket "nothing is compared with itself" is withdrawn and the self-referential geometry assertions are named | Existing metric breakages still fire against all three routes |
| **S7** browser inert guards | The caption check is keyed on the element the scroll region's `aria-labelledby` names, so it has subjects; `/outlines/` and foreign-asset pins are documented deliberate zeros over a floored request log, joined by a reachable check on what the page **links**; `.formulation-reveal` absence gained its presence partner in both investigations; floors raised to the actual totals | The caption breakage **moves** the caption inside the scroll box — deleting it was inert because `Table`'s footnote also carries the caption class and stood in for it — and **fires** |
| **S8** the harness's snapshot restored its own provisional record, cancelling it | `OWN_REPORT` is excluded from the snapshot and both writes route through it | A throw mid-run now leaves the provisional `status: running`, not the previous green report |
| **S9** nothing parsed the JSX it owns | The hygiene verifier parses **11 owned sources** — `.jsx` and `.js` through the repository's esbuild, `.mjs`/`.cjs` through `node --check` — in ~2 s | Falsification injects an unescaped apostrophe inside a single-quoted string, the exact one-character error that broke `vite build` repo-wide — **fires** |
| **49 real checks behind 107** | Counters split: `check()` counts failable assertions, `scanned()` counts scan passes, and both are printed and floored separately. Scans whose subject set must be non-empty now assert that as a real check | The counter-neutering breakage fires against the assertion floor |
| **S5** quantile routing | §8 now routes to **Decision Theory** (the identical `τ = c_u/(c_u+c_o)` derivation) and **Calibration & Conformal Prediction** (pinball loss by name), reserving scikit-learn for the fitting technique neither supplies. The record's "three unrelated uses of the word" claim is corrected above | — |

### Observations, also fixed

**O2** the cutoff label crossed the right edge of its viewBox at cutoff 12 — the label now has its own clamped
`labelX`, so the line stays where the data puts it and only the label slides. **O3** the purple dashed feedback
channel terminated in a green arrowhead — it has its own marker. **O4** `metricAxes[*].digits` was dead
configuration — the figure reads it. **O5** the 8→4 claim was pinned at two points — a 169-case sweep now
asserts the *property*: moving that arrival with no event time changed either leaves the selection alone or
moves it to a later event, and earlier availability never removes anything from the eligible set. **O6** two
unreachable clipping flags deleted. **O9** the one preset that changes two things says so in its name. **O10**
practice 1's age limit is named once. **O8** Phase C's "no outcome for any of its 824 cases" is corrected here:
the window renders 12 rows, so the measured claim is that no outcome is rendered for any **rendered** case.

**O1 is accepted as recorded rather than rewritten.** Several dozen offline geometry assertions compare
`scale(clamp(v))` against the endpoints of that same scale. They still catch a source mutation — which is what
the harness exercises, and case 3 fires on one — and the real containment check is the browser verifier's
against the rendered SVG box. What was wrong was the header claiming otherwise; that claim is withdrawn.

### Two defects the coordinator surfaced from a sibling

**The restoring rebuild had Node's 1 MB default `maxBuffer`** while every per-case build passed 64 MB. On a
full asset listing it could die with ENOBUFS and leave the last breakage's build on disk — sources restored
correctly, the record honest, and any browser run afterwards silently testing sabotaged bytes. It now uses the
same 64 MB, checks the status, warns loudly on failure, and records the outcome: this run reports
`restoringBuild: { attempted: true, status: 0, ok: true }`.

**Two of my own message edits drifted from their falsification patterns**, so the patterns no longer matched
what the code emits — the bare-form defect in another costume. Both realigned. The harness now distinguishes
**PATTERN DRIFT** (the verifier failed, the pattern did not match) from **DID NOT FIRE** (the verifier passed)
and **NOT APPLIED**, prints the pattern that missed, and records `expectSource` per case, so the next occurrence
diagnoses itself instead of reading as an inert guard.

### Final state

All five verifiers green; **42 of 42** falsification breakages fire and name their guard, 8 of them browser
guards rebuilt and run against the live preview, every file restored byte for byte, the restoring build
verified successful. **39 captures: 39 on disk, no digest or size mismatch, nothing recorded-but-missing, no
orphans, unique paths and unique digests, smallest 3,021 bytes.** Harness run offline-only first and
`--browser` last, so the fuller record is the one on disk. Build directory deleted and the preview stopped.
