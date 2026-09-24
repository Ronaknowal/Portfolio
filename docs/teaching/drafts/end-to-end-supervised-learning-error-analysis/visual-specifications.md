> **Current interaction amendment, 21 September 2026:** Read [live-exploration.md](live-exploration.md). Labs now update directly from valid edits and contain no learner prediction feature, including optional predictions. The older prediction/commit/reveal clauses below are historical design records; their numerical, scope, layout and evidence requirements remain applicable where unchanged.

# Visual contracts: End-to-End Supervised Learning & Error Analysis

Content-first specification, 12 September 2026. No visual, browser model or React implementation exists for this packet. Read [lesson.md](lesson.md) and [design.md](design.md). Values come from [calculated-inputs.json](calculated-inputs.json), [wine.csv](wine.csv), and the stated exact formulas. `author-calculations.py` creates the numerical handoff. Author calculations are distinct from formal model/native/browser verification.

## A. Specimen and information lanes — §2

Question: why does a fitted pipeline travel to inference, while targets take another route? Use three horizontal lanes labeled Train 106, Validation 36, Test 36, plus an inference specimen below them. Training arrows reach the scaler's stored mean/scale and the classifier's coefficients. Put both objects inside one fitted-pipeline boundary. Validation and test arrows reach the same pipeline's `transform → predict`; their labels reach the metric calculation separately. Validation metrics point to a candidate-selection decision. Test metrics point only to a final report. Use a visibly closed gate on the test lane until the choice is frozen; it is a conceptual workflow, not a browser security guarantee.

No arrow from labels to a transform, and no arrow from test to candidate selection. A dashed counterexample arrow can be paired with one short caption explaining human-mediated leakage. Arrows encode information, not runtime speed. Population validity stays in §1 instead of repeated inside every lane. Each lane has text explaining its source and allowed effect.

Desktop: lanes align on common pipeline/evaluator columns. Mobile at 320 CSS px: stack three compact lane diagrams with identical labels, preserving each label's path into evaluation. Show the fitted pipeline once above them and refer to it by an accessible named marker. Minimum body labels 14 px; keep reading order train→validation→test. No controls or animation needed. Verify arrow directions and counts against saved split IDs; rendered verification is deferred.

## B. Development-error investigation — §5

### Data and spatial form

Use only `validationRows` from the saved JSON, filtered by selected model. Exactly 36 unique specimen IDs per model. Put alcohol on x, color intensity on y; axes use recorded source values and say `source measurement scale` rather than inventing units absent from UCI's column metadata. Fixed shared bounds across models: x [11,15], y [0,14]. Three actual classes use shape plus label. A ring/cross means misclassified. Model choices are two-feature linear (default), two-feature forest, three-feature linear. Majority appears in an aggregate comparison table so the baseline stays visible, but need not crowd the scatter controls. Do not interpolate classifier boundaries from the probability rows; no grid of fitted predictions has been supplied.

A movable horizontal cutoff defines `color intensity < cutoff` or its complement `>= cutoff`. Show selected IDs below the plot and the slice denominator/errors/error rate. Selecting a specimen highlights its unchanged coordinates in both comparison panels and a row containing actual/predicted class, three raw feature values and probability of actual class. Flavanoids appears in the row and in a labeled one-dimensional strip, not by moving the 2D point when the model changes. The model's input dimension changes; the plotting coordinates do not.

### Recorded prediction and meaningful edits

Initial state: compare `linear_two` against `linear_three`, cutoff 4, lower slice, no prediction selected. User can edit cutoff continuously in [0,14] with a number input/slider step .01, choose lower/upper, and select either candidate comparison. These edits create an unsolved slice rather than cycling among narrated presets. Before reveal, require an initially unset choice `fewer errors / same number / more errors` for candidate relative to reference in the active slice. On Commit, store the full input key `(reference,candidate,cutoff,side)` and the choice. On Compare, compute integer error-count difference, grade the committed choice, and explain with changed-case IDs and counts. A changed input invalidates the prior prediction; keep old result only as an explicitly labeled prior comparison, never as current graded evidence.

Allow Explore without grading as a separate action if useful; never preselect a correct answer. Counts are exact; error-rate formatting must keep the denominator. Empty slice returns `0 specimens — choose another cutoff`; do not report 0% error or grade a superiority claim with no data. Selecting the same model on both sides is supported as a null case and yields unchanged counts for every nonempty slice.

### Fixtures checked during writing

All values below were computed from actual saved model predictions:

| Settings | Reference errors | Candidate errors | Correct prediction |
| --- | ---: | ---: | --- |
| linear_two→linear_three, lower cutoff 4, n=17 | 2 | 3 | More |
| same models, upper cutoff 4, n=19 | 5 | 1 | Fewer |
| linear_two→forest_two, upper cutoff 4, n=19 | 5 | 5 | Same |
| linear_two→linear_two, all rows | 7 | 7 | Same |
| linear_two→linear_three, actual-class-1 filter in practice, n=14 | 1 | 0 | Fewer |

The lower/upper reversal is intentional: an aggregate winner can regress within a slice. Overall paired repairs/breaks are 4/1 for three-feature linear and 1/0 for forest versus two-feature linear. Same aggregate count need not mean identical wrong IDs: show the actual paired changes for the selected cases, not a claim of identical behavior.

Reset returns comparison/cutoff/side/selection to defaults, clears prediction, comparison output and history. Edits are applied explicitly with Compare; labels distinguish proposed and active cutoffs if using a staged interaction. Limit to saved models and 36 rows; no fitting on the browser main thread, no arbitrary code, no test rows. Predicted probabilities are saved values with full precision, rounded only for display. Explain score uncertainty once in prose; do not print warnings per case.

### Text, keyboard and mobile

All operations available through named native inputs and specimen-ID selection. No drag-only cutoff. Focused points have the same row details as pointer selection. The table is the full text equivalent; name every class/encoding. At mobile width use one scatter plus a before/after strip of slice counts, keeping the active model label visible. Two model panels can switch with accessible tabs preserving the specimen selection, rather than shrink to unreadable columns. Keep quantitative axes/ticks readable and long chemical names in table rows. Announce concise comparison feedback in a polite status region. Reduced motion has immediate state replacement; no required autoplay.

Phase-two checks: compute all fixtures independently from saved labels/predictions; check `>=` boundary inclusion and IDs at equality; verify empty slices, input invalidation, same-model null and reset. Compare each displayed aggregate to the JSON. Inspect the cutoff, selected point and checked prediction at desktop and narrow size; ensure the contrasting 2→3 versus 5→1 changes are visible. Fixed axes are chosen for raw data; inspect actual layout before claiming perceptibility.

## D. Paired repairs and regressions — immediately after §4 result interpretation

A static strip or small matrix keeps all 36 validation specimens in ID order. Two rows show correctness before and after, linked only where state changes. Label four `repaired` cases and one `new error` for linear_three. A count beside them gives 29→32 correct, net +3. For forest, 29→30 correct, one repaired and zero new errors. Derive categories by actual ID alignment, not probability rank. Put unchanged cases in a muted but readable group. A summary table contains the same counts. This figure teaches why a net score change hides opposite movements.

On mobile group by changed/unchanged categories and list changed IDs with before/after class. Do not lose case identity or misrepresent stripe area as effect size. No controls required; optionally sharing selected specimen with B is an enhancement only if it avoids coupling distant page state. Verify the change sets from predictions and label supports.

## C. Acceptance and cost investigation — optional §8

Use the ten constructed cases in `deferralFixture`, not Wine probabilities. Each tile is one case with stable ID, confidence score and known correct/incorrect automatic outcome. Scores [0.95,.90,.85,.80,.75,.70,.65,.60,.55,.50], correctness [T,T,T,T,T,F,T,T,F,T]. This is an exact teaching fixture; scores are not calibrated population probabilities. Arrange tiles on a score rail and route them to `automatic answer` or `defer` using score `>= threshold`. The accompanying ledger shows accepted count A, wrong accepted W, deferred count D, coverage A/10, conditional error W/A and total cost `wrongCost*W + deferCost*D`.

Default reference threshold .60, wrongCost=10, deferCost=2: A8,W1,D2,coverage.8,error.125,cost14. Start proposed threshold .80 but unset prediction; user chooses lower/same/higher total cost, commits with current threshold/cost/edited-score key, then Applies. At .80: A4,W0,D6,coverage.4,error0,cost12, so lower cost. Offer editable threshold [.5,1.01] step .01, wrongCost[0,50], deferCost[0,20]. In addition let the learner edit one case's score [.5,1] using numeric input, applying to the same outcome; this creates unsolved entity changes. Do not alter the empirical outcome when editing a score. Explain that this is a proposed scoring/decision scenario, not retraining.

Checked contrasts/nulls by exact enumeration: .60→.80 with 10/2 costs14→12; with wrongCost2,deferCost2 costs6→12 (direction reverses); threshold .61 vs .64 gives the same A7,W1,D3,cost16 under10/2. Equal zero costs always tie; threshold1.01 accepts none, coverage0,cost20 under10/2 and conditional error `undefined (no answers)`, not 0 or 100% accuracy. Comparison grading uses integer counts and numeric costs tolerance1e-9. Cost changes invalidate prediction. Preserve baseline state in a separate labeled row; reset all tile scores, costs, cutoffs and prediction to default. Reject nonfinite/out-of-bounds inputs inline without overwriting active state.

Mobile: two vertically stacked destination queues with a readable cost ledger; no tiny axes required. Tiles have numeric scores and words rather than color-only outcomes. Number-input controls provide keyboard alternative to dragging. Predicted/observed cost difference and accepted/deferred IDs form a text equivalent. No animation required, only explicit steps; at most 10 tiles and constant-size arithmetic. Future checks must include tie equality at .80, both reversed cost fixtures, empty accepted queue, modified score ordering, prediction invalidation and reset, plus informative phone/desktop states.

## Implementation ownership and deferred closure

Use topic-owned future `src/learn/data/topics/end-to-end-supervised-learning-error-analysis.jsx`, examples and model/figure/lab files only when finish is authorized. Preserve stable ID/module order and load only this topic's data and controls. The CSV can become a topic-owned download; keep provenance accessible. Do not import author calculations or the phase ledger in browser code. Exact component names are up to phase two under the code standard.

Still deferred: full displayed-program execution, independent content/model review, actual plot geometry/perceptibility, live prediction grading, keyboard/mobile/accessibility checks, component contracts, payload measurement if warranted, publication and integration. These specifications may be improved for a recorded teaching reason; they are not evidence that any browser interaction has passed.
