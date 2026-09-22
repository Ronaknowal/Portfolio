> **Current interaction amendment, 21 September 2026:** Read [live-exploration.md](live-exploration.md). Labs now update directly from valid edits and contain no learner prediction feature, including optional predictions. The older prediction/commit/reveal clauses below are historical design records; their numerical, scope, layout and evidence requirements remain applicable where unchanged.

# Problem formulation: visual and investigation specifications

Written contracts only, 12 September 2026. Consume the full [manuscript](lesson.md), [design](design.md), [provenance](data-provenance.md), raw CSV and calculated inputs. No production visuals, browser interactions or formal independent review have been completed.

## Shared behavior

Represent actual entities and information paths: opportunities, events, versions, selected cases and fitting boundaries. Use a timeline for knowability and a ranked action list for capacity. Do not replace both with generic text-output boxes.

Every scored investigation starts with an unset prediction. Commit saves the prediction and complete old/proposed input state; Apply calculates from that committed state, reveals the result and records feedback. Relevant edits invalidate the prediction. Optional Explore is clearly ungraded. Reset clears edits, result history and grading. Future grading must not accept a retrospective answer after reveal as a new correct prediction.

Use labeled native controls, keyboard alternatives to dragging, text/shape plus color, clear focus and a concise polite result announcement. At 320 CSS px stack lanes/panels; keep readable fonts and an accessible exact-value table. Avoid automatic animation and entire-diagram downscaling. Invalid entries receive local messages without silently changing applied state.

Computed, measured and constructed values must be visibly distinguished. Model comparisons use saved actual outputs; editable human policy changes are labeled policy exploration and never presented as a newly fitted model. No data upload, Python execution or network call is needed for these browser investigations.

## F1. From a desired outcome to a decision — §1

Five connected elements: available records → prediction → eligibility/capacity rule → action → observed outcome. The feedback arrow returns mature outcomes to later training, not the same earlier prediction. Under the prediction display p=.32 as an explicitly illustrative value; do not imply it is an extracted bank record. Under the action show “select at most 50 eligible opportunities.”

A parallel small contract table ties population/unit/cutoff/target/features/action to these elements. The relationship is more important than placing all nine contract fields into a giant horizontal diagram. On mobile use a vertical flow with a feedback annotation beside the final element. Accessible explanation explicitly states prediction, action and business outcome are different objects.

Acceptance checks: no outcome-to-current-feature arrow; prediction value does not itself imply eligibility; capacity affects selection; text identifies the relevant cutoff. This is an explanatory figure, not a mandatory scored lab.

## F2. One entity has many rows — §2

Constructed fixture: parcels P1, P2, P3, each observed at hours 0, 1 and 2. Draw three lanes with stable identity labels. View A assigns alternating observations to train/validation, leaving all parcel identities represented in both. View B uses P1/P2 for training and P3 for validation.

Caption names the question: predictions on observations from represented parcels versus held-out parcel identity. Do not write “row splits always leak” or call three parcels adequate empirical evaluation. Show the number of distinct parcels separately from row count. No invented accuracy bar belongs in this figure.

Mobile: show two compact stacked allocation tables with the same nine IDs. Check that each row appears exactly once per allocation and that identity overlap is as claimed. This geometry prepares the information-path audit.

## I1. Reconstruct what was known — §3

Question: which calibration version is eligible for a prediction after the proposed edit?

Initial data from JSON timelineInput:

| Entity | Event | Available | Version | Value |
| --- | ---: | ---: | ---: | ---: |
| sensor_A | 1 | 1 | 1 | 10 |
| sensor_A | 4 | 8 | 1 | 20 |
| sensor_A | 1 | 6 | 2 | 12 |
| sensor_B | 4 | 4 | 1 | 99 |

Initial selected entity A, cutoff 5, maximum event age 5. A fixed common time ruler spans 0–12. Each record gets an event circle and available-at diamond, joined by a line. A vertical cutoff and shaded eligible-age interval express two different conditions. Label the two versions of event 1 separately; never overwrite version 1's cell when version 2 becomes available.

Controls:

- Edit each row's available-at time from its event time through 12, step 1.
- Edit values from -100 to 100 in whole units; permit arbitrary changes, not only preset scenarios.
- Edit cutoff 0–12 and maximum age 0–12, integer.
- Choose entity A or B.

Event times/version IDs stay fixed in this bounded investigation; this prevents undefined version semantics while leaving meaningful input freedom. If future implementation supports event edits or new rows, specify version identity and ties first.

Prediction is the selected row identity or “none,” not only a value that could coincide across versions. Commit includes rows, entity, cutoff, age and chosen identity. On Apply evaluate:

entity match; cutoff-age ≤ event ≤ cutoff; available ≤ cutoff; maximum lexicographic (event, available, version). Interval boundaries are inclusive. If no eligible row exists, return none.

Reveal an eligibility table with separate reason cells for entity, event age and availability. Use a selected outline on the actual version, and show its value. Rejected rows retain their values and explicit reason instead of disappearing. Feedback traces why the newest event may not be the newest known event.

Executed contrasts/nulls:

- Default A cutoff5/age5 selects event1/version1, value10.
- Move A event4 arrival8→4: select event4/version1, value20.
- Original history cutoff7/age7: select event1/version2, value12.
- Original history cutoff9/age9: select event4/version1, value20.
- Default cutoff5, age2: no eligible row.
- Edit B value99→other allowed value: selection/value for A unchanged.
- Translate every row value by a common amount within bounds: selected identity unchanged, selected numeric value translated. Distinguish identity null from value change.
- Edit only the unavailable event4 value: default selected value stays10; after moving its arrival before cutoff, the edit becomes observable.

The last two properties follow directly from the selection rule and must be checked for arbitrary supported values during implementation. Author calculations establish the first six saved cases; future model tests must add edited values independently.

Keep independent data-record identity, applied state and result table synchronized. Grading compares IDs exactly; no numerical tolerance is needed for integer inputs. Reset restores all records/cutoff/age, not only a selected scenario.

On phones show a readable ruler for one focused record and a persistent four-row status table, with numeric time fields as the keyboard/touch alternative. Do not hide the other versions' existence. The diagram needs both event and arrival markers at different locations in the default state.

Formal later checks: separate reference query versus UI selection, all fixture transitions, boundary inclusivity, none state, stale prediction invalidation, version/value independence, entity filtering, reset and keyboard operation. No browser check is claimed now.

## F3. Fitting boundary is separate from time availability — §§5–6

Use two horizontal dimensions as separate diagrams, not one confusing red line. First show source data 4,119 → development3,295/reserved824, then development → fit2,471/validation824. Imputer/scaler/category vocabulary/model are all learned on fit rows; validation receives the fitted transformations. Reserved rows have no scoring arrow.

Second show the selected feature families relative to the call cutoff: proposed recorded customer/history features on one side with availability contract “must be verified”; final duration after the call. A train-only pipeline containing duration still crosses the temporal boundary.

Annotate the deterministic sentinel transformation pdays999 → contacted_before0 and missing elapsed time. This rule is not a dataset-learned median; the median is learned later only within fitting.

Do not claim target stratification never observes reserved labels: it does use them for the declared partition. The distinction is that no reserved prediction/metric/selection occurs. The source row split does not establish new-client independence.

Acceptance checks: counts, disjoint source IDs, no reserved score, fitted state arrows, explicit history limitation, source-schema sentinel999 rather than older-schema -1.

## F4. Actual scores beside their feature contracts — §5

Use the three real result rows from JSON results, with AP, log loss, correct count and top50 positive count. Keep baseline present. Give each metric its own column and direction label; do not place .46AP next to 738correct on a shared numeric y-axis.

A horizontal AP bar can run 0–.5 with exact values .109223/.253440/.461700 and a table below. Use a neutral unavailable-feature symbol next to duration; the largest score must not receive a “deploy winner” signal. A companion selected-case strip shows 50 equal cells, with6/20/26 positive counts, if it remains legible; label these as aggregate counts, not ordered actual identities.

Caption identifies the fixed row-level validation experiment and 90/824 prevalence. Rendered plot labels must remain readable at desktop and phone sizes. No fabricated error bars, performance estimates on reserved data or money savings. Raw per-case scores stay accessible through the ranked investigation.

## I2. Capacity turns predictions into actions — §5

Question: how does a proposed change to the action set affect its observed precision?

Load only this topic's saved validation row IDs, targets and candidate probabilities. Default candidate ranking sorts probability descending, then source row index ascending for exact ties. Initial capacity50. Show rank, ID, probability and selection status; outcomes are hidden until Apply/Explore for the current question. Hiding is a teaching interaction, not a claim of a cryptographically sealed test.

Controls:

- Capacity is an integer1–100.
- Move a chosen source-row ID across the selection boundary using a keyboard “select instead of…” action with an explicit replacement ID, or the equivalent pointer action.
- Reorder selected rows without changing membership.
- Restore the fixed candidate ranking. Baseline/duration ranking can be inspected separately, with their feature-contract labels; switching model invalidates a pending prediction.

Capacity changes use the chosen ranking. A manual selection swap preserves capacity and sets the policy label to “edited action set”; it does not modify scores, labels, fitting or AP. Reordering within the same set changes no set metric. Avoid ambiguous interactions between manual state and a later capacity edit: either maintain a full edited permutation or explicitly restore the model ranking before editing capacity. Prefer a full permutation of fixed IDs with native up/down/move controls.

Predict precision direction relative to the prior applied state: rise/same/fall. Commit records old/new order or membership, capacity, active score source and answer. Apply counts selected positives, selected total and all validation positives. Compute precision=TP/k, recall=TP/90. Show exact counts and fractions, highlighting which identities entered or left. The same selected ID cannot occur twice.

Checked measured cases: candidate top 50 contains 20 positives, precision .4, recall 20/90; top 25 contains 12, precision .48, recall 12/90. Thus capacity 50→25 increases precision here, while recall falls. Reversing the order within the top 25 is a checked membership null. The executed policy fixture replaces selected positive source row 589 with unselected negative row 1680 at capacity 25, giving 11/25=.44; reversing the swap restores 12/25=.48. These actual IDs and outcomes are in JSON policyFixtures; do not invent a row label.

The exercise intentionally examines already observed development results. Do not suggest choosing the best edited set establishes generalization. A different capacity may reduce precision; the chosen input needs actual calculation. For the constant prior, all scores tie and the displayed top-k is determined by source-row tie order, not predictive discrimination.

Default layout: a ranked list with a visible boundary, selected-case outcomes after reveal, and two fractions whose numerator highlights those same positives. This is an action-membership view, not another generic metric dashboard. Paginate the824-row list or focus on the boundary with a searchable ID selector; a full table may be disclosed. Keep at most a bounded visible page mounted. No browser refitting or entire rawCSV parse is needed.

Later verification: saved per-case scores reproduce counts and ties; edited positive/negative swaps and within-set null; arbitrary supported capacities; no duplicates; membership/table/fraction agreement; stale predictions/reset; reading order and keyboard moves. Preserve actual measured probabilities even if the result is less visually dramatic.

## F5. Trace the source of a shortcut — §6

Three small information paths:

1. Final outcome → status/aggregate → proposed current feature.
2. Validation labels → feature selector → fitted representation.
3. Validation score → repeated candidate choice → reported “final” score.

Use distinct operations and labeled data roles. Repair each path beside it: cutoff-safe definition, fit-only selector, separate development selection/evaluation. This explains why one train/test split cannot repair every issue. Do not mark all future training feedback invalid; mature historical outcomes are legitimate training information.

## F6. Predictive propensity differs from impact — §8

Two paired bars for constructed groups A/B, with the same0–1 probability axis and called/not-called values .80/.75 and .45/.10. Difference brackets .05 and .35 show the ranking reversal. Explicit label: hypothetical potential-outcome probabilities, not estimates from the bank file. Accessible table contains all four values and differences.

A second small cost table for demand10/20 and stock10/20 shows expected costs15/5. Formulas are preferable to a new simulator here. These optional figures must not introduce causal/quantile mastery into core readiness.

## Phase-two handoff

Implement semantic topic-owned models, figures and investigations; retain rawCSV download, source attribution and the displayed complete programs. The source/rawresearch files are handoff inputs, not all required initial browser payloads. Load compact numerical inputs only when this lesson is opened and reveal heavy tables on demand.

Execute complete displayed programs verbatim, validate exported models and arbitrary/null transitions, obtain formal independent correctness and learning-experience review, inspect actual desktop/mobile/keyboard states, and integrate publication only on an authorized finish request. Content corrections discovered then must update current source checkpoints.
