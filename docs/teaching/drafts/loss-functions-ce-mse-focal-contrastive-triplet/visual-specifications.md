# Visual and investigation specifications

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Watch which errors receive influence.** Move observations, change the loss and focal gamma, drag decision thresholds, edit pair/triplet coordinates and change InfoNCE temperature.
**See the consequence.** Update loss, signed gradients, fitted location, confusion counts, eligible negatives and candidate probabilities together. Keep score-based metrics distinct from threshold decisions.
**Decision connection.** Choose an objective or operating threshold from the error tradeoff rather than from a single loss number.


Content preparation only. No visual is implemented or browser-verified in this packet. Preserve the different representations below; do not replace their mechanisms with repeated text-output controls.

## Shared interaction contract

Each investigation follows the live exploration contract above: current results are visible immediately, valid entity edits update all linked views, and comparisons explain the mechanism. Reset restores the declared inputs and recomputes their result. No prediction or answer-submission state is retained.

All exact arithmetic fixtures come from formulas and the accompanying Python program; the measured digit results come only from calculated-inputs.json. Display source type and units next to results. Curves are formula evaluations, not fabricated training traces. Finite inputs only, no autonomous animation, no live browser training, and no network requests for learner data. On small screens stack the views while retaining before/after labels. Every draggable item must also have a labeled numeric input, keyboard adjustment, and textual coordinate table. Never Calculate and explain by color alone. Announce run results and invalidation accessibly; use labels/patterns and adequate contrast for categories. Tooltips cannot be the only explanation.

## A. Prediction, penalty, tangent, update

Inline static/stepped explanation immediately after the opening example. Three coordinated panels: target line with y=5 and prediction=3; squared-loss graph x=prediction, y=squared target units; derivative arrow −4 and parameter path yhat=wx+b. Use x=1,w=3,b=0 for an optional one-step numerical trace, learning rate .1: gradients w=−4,b=−4, new prediction3.8, loss1.44. If exposing both w and b, make clear both move; do not claim the one-parameter-only prediction3.4. This panel teaches correspondence and needs no quiz wrapper.

Text equivalent lists prediction3, residual−2, loss4, slope−4, updated w3.4,b.4. Author derived; independently replay the optional update in phase two before exposing controls.

## B. Which observation moves the fitted constant?

Location plot of all seven measurements; aligned loss curves and a row-per-observation derivative table. Inputs are six zeros plus editable final measurement v in [2,100], step .5; allow a changed-data task with all observations in [−100,100]. Select MSE, MAE, or Huber(delta1). Do not hide the raw observations behind a generic “outlier amount.”

Drag/edit the observation from v10→100 and inspect which fitted constants change. Calculate and explain the three live changes against numerical optima (MSE mean; MAE median with interval handling; Huber solve monotone derivative by bisection). The checked contrast is MSE10/7→100/7 while MAE0→0 and Huber1/6→1/6. Changed investigation: edit a formerly zero observation, inspect direction and find the optimum. Null: all observations3 gives all minima3. MSE derivative uses2r; Huber uses clipped r; axes and gradient scale must not imply identical local curvature. At a nonunique median show the complete optimum interval and the loss at any selected point in it.

For arbitrary changed data, author supplied formula; author did not execute every UI input. Phase two must compare optimizer values and derivative signs at neighboring points, including even observation counts if permitted; simplest bound is fixed seven observations.

## C. Focal loss: loss weight versus gradient contribution

An individual-row ledger linked to a stacked contribution plot, not a single curve with no differentiation. Inputs: negative count 1–2000; each negative's positive-class probability .001–.4; one positive's probability .01–.99; gamma0–5; optional alpha disabled by default. No probabilities0/1. Use stable logit-based formulas in lesson.

At default negative count1000,p_negative=.01,positive p=.1,gamma0, show whether changing gamma to2 reverses the shared additive-bias update direction. The current computed result is visible. User changes gamma; calculate and explain the sign of total gradient before/after, not the sum of absolute magnitudes. Checked CE contributions +10 and−.9; focal +.002989966499 and−1.102018785065. Negative total means gradient descent increases the shared bias. A displayed “importance” value must say loss, signed derivative, or absolute derivative.

Contrasting independent task: edit the number of negatives until the CE net gradient becomes negative too (count<90 for these probabilities; at90 the CE sum is zero within tolerance). Null: gamma0 with alpha disabled agrees with BCE; gamma0 with balanced alpha enabled is explicitly not this null. A second local probe pt=.9,gamma2 shows loss ratio .01 but gradient ratio .0289648928184. If presenting zero cancellation, mark the tie and do not divide by its norm.

Display the consequences of every chosen count immediately; explain the actual count and computed sign; explanations derive count*.01−.9. Reset/input edits follow shared contract; upper bound keeps the visualization aggregated rather than creating2000DOMnodes.

## D. Real digit probabilities and decisions

Use only nine stored runs and their120 validation IDs/probabilities. Link selected row ID to the actual8×8 pixel specimen and label from digits-400.csv; no image generation. Default seed1/BCE. Probability strip and confusion matrix share highlight selection. A threshold line moves across fixed points; AP, Brier, and log loss appear in a separate fixed-probability column.

**Live threshold comparison:** moving .5→.1 updates FP/FN and the fixed-probability Brier value immediately. Checked .5 gives TP10,FP1,FN2,TN107; .1 gives TP12,FP2,FN0,TN106; .9 gives TP8,FP0,FN4,TN108; .01 gives TP12,FP22; .99 gives TP1,FP0. Compute from saved per-example probabilities. Let the learner move the threshold to retain all 12 positives with at most 2 false positives; show both counts and the intervals between observations, without an answer entry or grade.

Null: changing threshold leaves ranking/probability metrics unchanged. Require explicitly selected objective/seed and threshold input in snapshot. Changing run clears previous comparisons. No default “best model” label; show validation status. If multiple points tie, threshold policy is p>=threshold. Bounds threshold[0,1]. A textual table provides equivalent row inspection; render only selected image and bounded chart marks, not all400 images.

## E. Pair geometry and triplet mining

Two-dimensional coordinate board with anchor, positive, and three editable negative entities. Distance ruler switches explicitly between D and D²; keep formula and margin units synchronized. Default a(0,0),p(1,0),negatives(.5,0),(1.2,0),(2,0), squared margin1. Show candidate labels/IDs, distance table, active loss, and selection status. Hard/semihard/easy differ by marker shape as well as hue.

Live observation: highlight the negative selected by the nearest strictly semi-hard policy; show “none; skip” when no candidate qualifies. Recompute candidate distances and eligibility immediately. Checked default selects row1 (zero-based), distance1.44 and loss.56. Other losses1.75 and0. User then edits n1 to(2.5,0) and observe whether any candidate remains semi-hard: none. Independent repair: edit one eligible candidate to satisfy1<D²<2, with no prefilled answer; Calculate and explain predicate and actual first-row tie-break. Bounds coordinates[−3,3],margin[0,4]; ties at D²ap and D²ap+margin are excluded by this policy. Null task: translate all points by same vector; distances, losses and selected identity stay unchanged. Margin1→.3 with original coordinates gives the formerly semi-hard candidate zero loss.

Optional derivative arrows must compute active-region gradients and choose a documented subgradient at the hinge. Exact collapse a=p=n shows positive loss1 and zero squared-distance gradients; this is a deliberate diagnostic fixture. Pair view uses same=1, no one-half factor, margin1: D.2 yields matching.04/nonmatching.64; negativeD1.2 yields0. AtD0 explicitly label norm derivative convention, not an invented push direction.

## F. InfoNCE candidate competition

One query, three editable candidate similarity cards, designated positive index0. Similarities in[−1,1] are direct mathematical inputs, not purported similarities from a trained encoder. Temperature .05–2. Optional geometry mode can construct unit vectors, but do not simultaneously imply arbitrary pairwise cosine matrices are realizable.

Show aligned similarities, temperature-scaled logits, candidate probabilities and loss. Changing tau 1→.2 immediately changes correctly ranked [.8,.2,−.1] loss .670585210653→.059113895273, while [.2,.8,−.1] gives 1.270585210653→3.059113895273. Editing a candidate creates that changed rank directly. Explain why sharpening helps one ranking and harms the other.

Null equal scores[.4,.4,.4]: probabilities1/3,losslog3 for either temperature. Duplicate-positive conflict[.8,.8,−.1]: losses.878202355837→.698686309490, approachinglog2 rather than0. Extra task: identify whether a wrong label, duplicate, or shortcut can be inferred from the displayed information; score only the supported duplicate-score limitation, not unobserved data causes.

Optional B×B mask view shows one-way row positive diagonals and other-row candidates. A separate diagram builds2B SimCLR views with self exclusion and both directions; do not reuse the same mask under a different title. B<=8 for mask inspection. Valid input changes recompute every dependent result and explanation. Temperature zero invalid with explanatory feedback rather than NaN.

## G. Scale and reduction audit

Static area diagram B1024→4096: score matrix side grows4× and cells16×; float32 storage4→64MiB. Include bytes-per-value and exclude activations/gradients in label. Adjacent two-sequence table shows token sum versus per-sequence mean; use actual author calculations if making it interactive. No invented wall-clock axes or universal speed ranking.

## Phase-two checks

Independently replay formula fixtures, numerical corners, saved-data joins and threshold counts; run the full downloadable program in the declared clean environment; verify gamma0 values and gradients against BCE, extreme logits, zero-vector rejection, squared/unsquared margin distinction, and mining ties/no-candidate behavior. Check dataset hashes and source-ID joins. Then implement bounded components, test actual live updates, invalidation, reset, keyboard/touch/text routes, narrow screen layout, mathematical typesetting, loading boundaries and renderer contracts. These are deferred implementation checks, not completed content checks.
