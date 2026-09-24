> **Current interaction amendment, 21 September 2026:** Read [live-exploration.md](live-exploration.md). Labs now update directly from valid edits and contain no learner prediction feature, including optional predictions. The older prediction/commit/reveal clauses below are historical design records; their numerical, scope, layout and evidence requirements remain applicable where unchanged.

# Active learning — visual and investigation contracts

Content prepared; implementation not started. This file specifies behavior, not delivered UI. Use the canonical `active-learning` topic ID and topic-owned imports. Keep the manuscript's alternative visual forms: an annotation queue, threshold rulers, probability strips, committee decomposition, geometric distance map, measured acquisition curves, and state timeline. Do not replace them with a repeated text-output lab. No chart is a measured benchmark unless its data are in `checked-results.json` under `banknotes`.

## Shared behavior and boundaries

- Every investigation begins with **no prediction selected**. The learner records a specific prediction before revealing the computed result. Record the prediction text/value and a signature of every active input: entities, IDs, coordinates/probabilities, hypotheses, weights, observed answers, pending query/batch and tie policy. Changing an input invalidates both prediction and result and visibly invites a new prediction. A saved old run can remain in an explicitly labeled history, never masquerading as the current answer.
- Ordinary input editing does not silently execute the solution. Display inputs and local explanatory labels first. Use separate Record prediction and Reveal/Acquire actions. Do not preselect the correct answer or show its score before the learner commits.
- Store actual editable entities, not just a preset or difficulty slider. Presets supply baseline, changed and null cases, and remain editable. Reset restores the selected preset's inputs and clears predictions, acquired history, results and validation messages. Reset all selects the default preset.
- Keyboard alternatives for every graphical action; labeled numeric fields and tables are authoritative. Color supports labels and shape. Each result has an accessible text explanation and numeric table; no hover-only information. Live announcements describe a completed action, not every keystroke. Respect reduced motion; animations are optional transitions between exact states.
- On narrow screens stack the figure above the controls and table; avoid shrinking all labels until unreadable. A table may scroll within its labeled container without horizontal page overflow. Buttons remain reachable and histories stay collapsible. Phase two must verify keyboard, focus, reduced motion, narrow layout and screen-reader labels.
- No remote oracle, dataset, model, video or tracking request on lesson load. External resources are ordinary learner-initiated links. Lazy-load the topic's investigation modules; the mathematical exercises are tiny. Do not execute the four-strategy training benchmark in the browser. Supply its complete Python program and measured result asset as explicit optional resources; decode only required data on demand.
- All printed values are rounded display values; comparisons use full precision. Stable ID-order ties are explicit. Validate before computation and explain errors in context without destroying edits. Reset never mutates downloaded source data.

## Inline figures and reading placement

| ID and manuscript placement | Representation and content | Readable fallback / evidence |
|---|---|---|
| `annotation-data-boundary`, §1 | Four lanes L/U/dev/test, pending queue and one selected card crossing the label boundary; acquired count and model-fit version distinct | Ordered loop plus four-column data-access table; conceptual diagram, no invented performance |
| `threshold-version-space`, §2 | Eight aligned rulers, thresholds .5…7.5, query4, labels0/1 and contradicted rulers after answer0 | Exact three-query table from `examples.threshold.bisection`; no hidden result in initial lab |
| `multiclass-uncertainty-strips`, §3 | A/B/C segmented class probabilities, top-two gap and three scores; natural-log units | `examples.uncertainty`; table with values; widths sum1, zero segment still has a class label |
| `committee-entropy-decomposition`, §4 | Member strips → mean strip; H(mean) and mean H as aligned lengths with difference bracket | `examples.committee`; include shared ambiguity and opposing-confidence side by side; fixture probabilities, not trained Bayesian output |
| `geometric-covering-radius`, §5 | Anchor square, candidate named circles, nearest-center segments; longest distance labeled | `examples.diversity`; coordinates and nearest distances; Euclidean units in authored plane; no interpolated class boundary |
| `measured-acquisition-curves`, §6 | 31 actual checkpoints; x total fitting labels6…36, y correct/80 or accuracy with labeled conversion; mean plus toggle individual runs | `banknotes.traces[*].development_correct`; milestone table and five per-run finals; no smoothing, no forced monotonicity, optional min–max explicitly descriptive not CI |
| `annotation-record-timeline`, §7 | Candidate/pending/answered/accepted-or-adjudication/model-fit-version and abstained branch | Text state transitions; no implied automatic oracle correctness |

Keep inline figures near the first mechanism that needs them. The three investigations deepen distinct hurdles and need not become one giant combined dashboard. Small advanced arithmetic works best as annotated equations/tables, not more controls.

## Investigation A: threshold questions and surviving explanations

**Hurdle:** choose a question by its possible answers, rather than reveal labels at arbitrary positions.

Entities: 2–16 unique sorted candidate thresholds in [−10,10], editable positive weights normalized only on explicit confirmation; default uniform eight thresholds .5…7.5. Fixed/editable observed rows default (−1,0),(9,1); candidate query rows 0…8 with stable IDs; allow 1–24 distinct eligible query values in [−12,12]. The input editor can select an oracle threshold from the candidate list for a consistent simulation or choose manual oracle answers. The displayed “true threshold” is concealed during a run; changing it resets all answers. Explain simulated versus manual mode. Do not reveal labels for unqueried rows.

Before reveal: select one unused query; record predicted survivor counts for answers0/1 or the hypothesized retained weight in weighted mode. Show observed rows and hypotheses as inputs, but hide computed prediction columns and score decomposition until commitment. Acquisition obtains exactly that answer, increments acquired-label count, filters consistent thresholds and records the state. Manual mode allows a contradictory answer; the result then diagnoses incompatible evidence instead of silently restoring candidates. No remaining hypotheses disables further acquisition until editing/reset.

Calculation: predictions `1[x>=threshold]`; retain equal observed label. Uniform expected survivor count `(m0²+m1²)/M`; weighted mode compute each answer probability from remaining normalized weights and its corresponding survivor count, making the distinction visible. Unsupported weights zero/negative/nonfinite produce an input error. Display seed, newly acquired and total labels separately.

Checked fixtures:

1. Default consistent oracle threshold5.5: query4→0 leaves4.5,5.5,6.5,7.5;6→1 leaves4.5,5.5;5→0 leaves5.5. Three new plus two seed labels. Default score expected survivors4 for x4,6.25 forx1,8 forx0.
2. Changed threshold family [.5,1.5,4.5,7.5], uniform: queries1/3/8 have expectations2.5/2/4; require learner choice before computed comparison. This is an independent family, not only changing a slider.
3. Null: query0 is unobserved but every candidate predicts0; oracle0 preserves all8. No spurious progress message.
4. Failure: in the default family manual answer0 at query8 eliminates all8; retain answer provenance and “no listed threshold is consistent.” Do not claim the learner found a threshold.

Post-result feedback compares the recorded prediction with actual survivors and points to the eliminated rulers. A balanced query minimizes the displayed one-step criterion under its hypotheses/weights; no universal label-complexity claim. History includes before/after hypothesis IDs and acquired answer. Phase two compare all transitions to `examples.threshold` and `examples.supplementary`; test manually entered contradictions, duplicate query prevention, weight edits/invalidation and empty state.

## Investigation B: average uncertainty versus member disagreement

**Hurdle:** distinguish entropy of the mean from average entropy, without labeling all shared uncertainty as irreducible noise.

Entities: 2–6 committee-member rows and 2–4 class columns; each row holds explicit probabilities in [0,1] summing to1 within validation tolerance. Default two binary opposing rows [.95,.05],[.05,.95]. Add/remove member and edit probabilities with numeric controls plus accessible segmented strips. Preserve stable IDs. Offer a normalize button for nonnegative nonzero rows, never silently normalize a partly typed vector. Equal member weights only in this lab; a weighted extension would require matching formulas/fixtures.

Prediction before reveal: learner records whether the proposed edited committee has larger/same/smaller disagreement than the baseline and a numeric estimate of H(mean) or D. Start unset, including preset switches. Show member probabilities as inputs; defer mean and computed entropies until reveal. Compare two saved runs only when their active configurations are labeled.

Calculate H(p)=−Σp ln p,0ln0=0; mean across members; D=H(mean)−meanH. Clamp only tiny negative floating error around zero, not genuine invalid inputs. Display natural-log units and the fact that these are authored probability vectors. For the optional vote-entropy view use per-member argmax with lowest-class-index ties, counts/M and explicit tie notice; no inferred posterior-sampling claim.

Checked contrasts/nulls from `examples.committee`: opposing [.95,.05]/[.05,.95] Hmean.693147180560 meanH.198515243346 D.494631937214; shared [.5,.5]/[.5,.5] same Hmean, meanH.693147180560 D0; confident identical [.95,.05] rows Hmean=meanH.198515243346,D0. Additional practice deterministic opposing[1,0]/[0,1] Dln2. Identical-member null remains D0 for any valid entered distribution; phase two check extremes exactly and arbitrary normalized fixture by direct equation.

Feedback references changed member rows and decomposition: agreement is a property of these members; shared misspecification remains possible. BALD terminology is explained only for a model posterior approximation. No simulated random predictions described as dropout. Complexity O(MC), tiny bounds. Phase two tests probability validation, add/remove invalidation, stable ties and equation/table agreement.

## Investigation C: choose a batch by its added coverage

**Hurdle:** a high individual score can repeat the same location; geometry and uncertainty answer different questions.

Entities: 1–4 existing labeled anchors and 2–24 candidate rows with stable IDs, x/y coordinates in [−10,10] and independent supplied binary p in [0,1]. Default anchor(0,0), A(0,1,p.5),B(.1,1,p.52),C(0,4,p.7),D(4,0,p.72); budget2 (editable1…min(6,candidate count)). Coordinates use an authored dimensionless plane. Learner may drag or edit table rows, add/remove a candidate, move an anchor or change probabilities; all invalidate prediction/result. Coincident coordinates allowed, duplicate IDs not.

Commit a chosen batch of exactly B distinct eligible IDs plus predicted covering radius before reveal. Then calculate user radius, top-entropy batch and farthest-first batch. Show all nearest-center assignments and distances. Step through farthest-first: initial nearest distances to anchors; choose maximal eligible distance with stable-ID ties; mask selected ID; pointwise min with distances to new center. On each step show a caption describing the change and a table, not only animated circles. Existing anchors must be nonempty; invalid geometry does not fabricate an origin anchor.

Exact checked fixtures (`examples.diversity`): default entropyA,B; farthestC,D with intermediate radii4 then√1.01=1.004987562112; userA,B radius4. Changed C to(0,2) yieldsD,C with radii2 then√1.01. Null three candidates at(0,0) with anchor(0,0),budget2 selects distinct IDs0,1 and zero radius each step. These show changed order with same final radius, and no geometric advantage in the null case. Compute entropy from the displayed probabilities; never attach arbitrary entropy scores to probabilities that disagree.

Feedback compares the learner's radius with the computed geometric result while explicitly limiting the criterion to coverage. Do not certify label accuracy from distance or draw a fake class boundary. Farthest-first guarantee annotation refers to a metric covering radius with fixed centers, not accuracy. Phase two checks ties, duplicate coordinates, selected masking, narrow numeric fields, keyboard point movement, at-mostB eligible distinct output IDs, nearest-segment/table consistency and coordinates after reset.

## Measured experiment explorer, §6

An evidence reader rather than a fourth prediction exercise: select strategy/run seed to inspect stored trajectories and queried source-row IDs. The dataset and result signatures remain visible in an expandable provenance panel; ordinary reading can use the static milestone table. Load only the active topic's small data on opening. Show all31 dev checkpoints and query history30, initial six and final36 labels separately. No controls that imply the stored model retrains after arbitrary feature edits. Such edits belong in the executable Python exercise.

Do not show unacquired oracle labels during a query-step playback; show selected label only at acquisition. Full oracle labels exist in the downloadable dataset for reproducibility, so the interface is a pedagogical reveal boundary, not a security boundary. The real experiment uses four features; a two-feature scatter is at most a labeled projection, never a plot of its true decision boundary or proof that distant 2D points disagree in 4D.

Final result selection uses highest mean final development correct, stable strategy-order ties, then only the selected strategy's five test results. Other strategies' test results do not exist in this packet and must not be invented. No browser training, no synthetic performance lines, no implied CI or statistical dominance from shared-data repetitions. Phase two verifies exact traces/rounding, label counts, split identifiers, download paths and lazy loading. Full browser/accessibility/code checks remain implementation work.
