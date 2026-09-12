# Visual specifications — Anomaly & Outlier Detection

Content revision 1; stages 1–2 only. No visual, lab or browser state in this file is implemented or visually verified. Read [lesson.md](lesson.md), [design/research](../../ANOMALY-DETECTION-LESSON-DESIGN.md) and [data provenance](dataset-provenance.md) before implementation. Preserve the catalogue identity and module sequence.

The representations below are chosen for distinct conceptual hurdles. Implementation may improve their form with a recorded reason, but must preserve or strengthen the learning action and numerical contract. Ordinary explanatory figures and genuinely unsolved investigations serve different purposes.

## Shared interaction and display contract

* Every investigation begins with **no prediction selected**, no silently committed default and no solved numerical answer. Show the question and relevant inputs; require an explicit “Commit prediction” action before “Run/reveal” becomes available. A correct answer is not required to continue. Offer “I'm not sure” as an explicit committed response, then explain the mechanism.
* Store the prediction with its exact input snapshot, not merely a global answered flag. Editing a point, query, parameter, method, threshold mode or selected dataset invalidates that prediction and clears the old answer from the new state. Retain an optional labeled previous attempt for comparison; do not show it as current.
* Presets help orientation but do not replace entity or numeric editing. Numeric inputs and buttons must be keyboard operable. Dragging is optional and has a labeled numeric alternative. Validate drafts atomically before changing the last valid model; associate errors with the field. Empty, nonfinite and out-of-range values do not silently become defaults.
* Mark answer correctness using text and shape as well as color. A wrong prediction gets an explanation connected to the visible quantities, not merely a red cross.
* Labels identify row IDs, quantities and units. Never use implementation node indices as learner labels. Exact zeros and boundary/tie cases are not rendered as fake tiny negative values; displayed rounding is identified. Preserve underlying values for thresholds.
* Figures use native readable labels and meaningful axis scales. No screenshot substitute for text equivalents. At 390/320 px, stack the plot, calculation and controls; avoid shrinking a desktop table to illegible text. Local scrolling is acceptable for a genuinely dense tree/table with an explicit keyboard-operable scroll region, never to conceal the main contrast.
* The visual area must show the promised effect. A small numeric change that looks unchanged must be accompanied by a correctly labeled zoom/residual view; do not exaggerate coordinates without saying so.
* No automatic animation is required. Cut/step transitions are user controlled, and reduced-motion preference must be respected if motion is added.
* In phase two, practice A–J must render questions first, with separate initially closed hint and solution disclosures. The two prose checkpoints likewise reveal explanations only on request. Plain Markdown Hint/Solution headings are content storage, not the intended published interaction.

## 1. Inline task and time-lane figure

Placement: §2 after reference→score→alert.

Entities: one row/observation, frozen transform+detector, calibration scores/threshold and later decision. Three horizontal time intervals on desktop; chronological vertical flow on phone.

Make provenance visible: reference data supply scaler and model, calibration supplies only threshold, later rows supply scores, annotations supply evaluation. Include a crossed-out route from future data back to scaler without suggesting that labels may never be used in appropriately supervised training.

Caption: “The reference defines the comparison; calibration chooses the action; later data test the declared protocol.” Text equivalent is the paragraph plus a compact dependency list. No interactive state is needed.

Use actual NAB split dates only when §10 refers back to this schematic; before that, use generic earlier/later labels so the introductory graphic does not require interpreting an unexplained dataset.

## 2. Isolation: point strip, cut intervals and path tree

Placement: §3 before harmonic normalization, with the terminal-leaf view revisited after c(m).

### Entities and quantities

Five point IDs P0–P4 with editable one-dimensional coordinates. Default [0,1,2,3,12]. Treat duplicates as separate IDs. Native axis length reflects coordinate gaps. At any selected node, show its member IDs, min/max, possible cut intervals and branch membership.

The primary display is a line strip with a vertical cut and a linked small path tree. The current query path is highlighted with depth labels. A leaf badge says “2 rows remain; depth 2” or equivalent, followed by d+c(m), rather than falsely claiming complete isolation.

### Actions and prediction

Input: five coordinates in [-20,30], at most two decimals; editable selected query coordinate in the same range; ID buttons for selecting a reference query. Presets “large gap,” “regular spacing,” “identical records” are shortcuts only. Sorting for display must not relabel the original IDs.

Before the initial reveal, predict the point with the shortest average corrected path, or predict a tie/“not sure.” User can then reveal exact expected lengths or step through an explicitly selected sample cut sequence. Do not call one realized path the exact expectation.

If random cuts are shown, hold a reproducible seed and expose “new tree” after an initial committed prediction. Match splits to drawn geometry; use a cut strictly inside a nonconstant node's span or handle floating endpoint draws as declared leaf stops.

### Checked examples and distinctions

Data and exact expectations are in author-calculations.json. The small author integrator exhausts 1D gap intervals, with max depth3 and exact harmonic correction:

* Default P4=12: first-cut singleton probability3/4; P0 probability1/12. Average paths [31/12,73/22,17/5,17/6,841/660]; c5=77/30. Scores approximately [.497755,.408159,.399239,.465258,.708845].
* Contrast P4=4: endpoint scores tie about .569715, middle .444782; the exceptional gap disappears.
* Null all zero: no cut; all corrected paths c5 and scores .5. Explicitly say no ranking is available.
* Practice reachable cap: four values[0,1,2,3], cuts .5 then1.5, selected query2. Leaf[2,3] at depth2 gives h3 and score2^(-18/13)=.3829915893. This is a separate worked terminal fixture, not a five-row state.

For arbitrary edited five-point states, implement the small exact interval integration or clearly show finite-tree estimates with a recorded sample count. If the exact algorithm cannot handle the admitted edits, narrow and explain the input domain rather than silently switching numerical meaning.

### Geometry, alternatives and checks

Keep isolated12 visibly far from3, but preserve full coordinate scale. Text table gives member IDs, cut interval widths/probabilities, depth, terminal population, corrected path and score. Phone: strip above a vertical path trail; the full tiny tree can be optional.

Verify actual cut endpoints, singleton probabilities, ID conservation, duplicate/all-equal stops, actual psi normalization, finite-score domain and distinct exact/Monte Carlo captions. Compare at least one changed coordinate state with an independent finite cut calculation. No requirement to animate hundreds of trees.

## 3. LOF: a query's neighborhood contains neighbors with their own radii

Placement: §4 across the four calculation steps. The introductory radius table is inline static; the investigation extends it after both query calculations.

### Entities and quantities

Fixed initial references P0–P5 at [0,1,2,20,24,28]. Main line spans all points; below it place a calculation strip for each selected neighbor, showing distance d(p,o), that neighbor's r_k(o), the max, and reference lrd(o). A final reciprocal and ratio panel connects the same values.

Do not draw all six overlapping radius circles simultaneously. Show selected neighbor intervals or aligned distance bars sharing units. The selected radius belongs to neighbor o, with an arrow connecting its row to the max expression.

### Inputs and predictions

Editable query p in [-5,33], k integer1–5, and an advanced editable reference-coordinate table (six values within[-5,33]). Default distinct coordinates. Reject duplicate references in the exact-arithmetic mode with a precise explanation; a separate duplicate explanation mode may show why zero reaches require a stabilization policy, without inventing a finite mathematical ratio.

First compare p4 with p17: predict which has higher LOF, or tie/not sure. Then choose a new p and predict “above, equal to, below reference-like1” before calculation. For k change, ask whether the particular prior ranking will persist, not a generic claim that larger k always raises/lowers scores.

Calculation mode selects exactly k other rows with stable-ID ties. A noninteractive definition note explains original tie-inclusive LOF. Do not implement one rule while labeling it another.

### Checked contrasts and nulls

Default k=2 reference lrd [2/3,1/2,2/3,1/6,1/8,1/6], factors[7/8,4/3,7/8,7/8,4/3,7/8].

* p4: neighbors2,1; reaches2,3; lrd2/5; LOF 35/24.
* p17: neighbors20,24; reaches8,7; lrd2/15; LOF 35/32. Larger nearest distance, lower LOF.
* p6: reaches4,5; lrd2/9; LOF21/8.
* k3 with queries3,6,17,25: all query scores1 under this reference. This is the checked local-contrast disappearance; do not extrapolate to every edited query.
* Common positive scale factor applied to all coordinates preserves fixed-k neighborhoods and LOF, while lrd scales inversely. Supply an optional “change units ×10” button preserving bounds through display-unit conversion, not a new raw-domain limit.

### Training row versus query state

Placement: §5. Two adjacent neighbor strips (stacked on phone) compare training rowP1 at1 with a new queryQ at1. The former excludes P1; the latter can select P1 with zero distance. Prediction starts unset: same factor or different? Training factor4/3 versus new-query7/8. Identify Q separately despite coordinate overlap.

A null comparison at reference endpoint0 gives7/8 in both modes even though contracts still differ. Equality of one number must not be presented as proof that the APIs are interchangeable.

Optional earlier-formula aside: LOF uses neighbor radius; OPTICS uses source core radius; HDBSCAN mutual reachability includes both. DBSCAN includes self in min_samples; this k counts other rows. The main computation needs no earlier hierarchy branch.

### Checks

Independent neighbor identity/ties, chosen radius owner, all distance maxima, lrd units, ratio and mode differences. Text equivalent lists all actual neighbors and values. On phone, each row is a legible card/list; use a small table only when columns remain readable. Test the meaningful advanced coordinate editor, not presets alone.

## 4. Kernel support: two bumps and an offset

Placement: §6 after analytic two-anchor derivation.

Entities: reference anchors at -a,+a, query x, two weighted RBF contributions, sum and horizontal rho. A second aligned plot shows signed g with a zero line. Label the first axis “kernel compatibility sum” and second “signed decision”; neither probability nor input-space distance.

Fixed normalized nu=.5 and alpha=.5,.5. Editable a in[.5,3], gamma in[.05,2], x in[-4,4]. Each field has a number editor; sliders are secondary. Explain that this is an exact symmetric two-reference solution, not a general SVM solver with gamma-only coefficients held incorrectly fixed.

Formulas:

    contribution1 = .5 exp(-gamma (x+a)^2)
    contribution2 = .5 exp(-gamma (x-a)^2)
    rho = .5 (1 + exp(-4 gamma a^2))
    g = contribution1 + contribution2 - rho

Prediction: before reveal, classify selected x as inside, boundary, outside, or not sure. Preserve exact input snapshot. Changing gamma/a/query invalidates the answer. A reference-anchor button sets x=a and explicitly asks whether that boundary will move in this symmetric construction.

Checked contrast a1,x0: gamma.1 gives +.0696773950; gamma 1 gives −.1412783783. Null x=a: g0 for every admitted a/gamma. Changed practice a2,gamma.25 gives earlier a1,gamma 1 midpoint value.

Show separated positive intervals at gamma 1 using actual computed zero crossings or sufficiently sampled sign-consistent curve; no generic closed blob. Plot y range includes both weighted curves, rho and actual extrema, with enough vertical separation to see .0697 versus−.1413. An optional residual zoom must label its scale.

Phone: stack two aligned plots, show selected-query contribution arithmetic underneath. Text equivalent gives both contributions/rho/g and classification. Verify actual anchor labels, no clipping at a3, signed-score orientation, normalized coefficients and geometry across editable states. State a numerical boundary tolerance in UI classification without claiming exact floating values are theorem proofs.

## 5. Score threshold and population workload

Placement: §8. Use two forms because a score ordering and a conditional population count are different objects.

### Inline threshold ruler

Sorted five calibration IDs with scores[1,1,2,4,4]. Show strict greater-than threshold. At tau4 no alerts; lowering to2 selects both4s, not exactlyone. Explain top-B plus tie rule separately. This is a static worked illustration, not the unsolved investigation.

### Population investigation

Inputs: N (integer100–1,000,000), prevalence p in[0,1], sensitivity t in[0,1], false-positive rate f in[0,1], review budget B integer0–N. Accept percentages with explicit percent labels; do not interpret1 as .01 silently. Default N100000,p.001,t.8,f.01,B200.

Prediction: expected useful-alert fraction bucket (<10%,10–50%,>50%, no alerts, not sure) and whether B suffices. Commit before displaying derived answer.

Counts are expected values:

    true alerts = N p t
    false alerts = N (1-p) f
    total = sum
    precision = true / total if total > 0; otherwise undefined

Represent total populations and alert composition in separate aligned counts. A 100000-item icon wall is not required; aggregate bars with clear N and integer/expected labels are better. Rare true alerts must remain visible in a correctly labeled zoom of the alert population; do not falsely enlarge the population share without noting the denominator change.

Checked default true80,false999,total1079,precision80/1079; contrast p.01 true800,false990,total1790,precision80/179. Null f0 yields no false alerts and precision1 when true>0; t=f0 yields undefined precision/no alerts. Changed practice N50000,p.002,t.9,f.005 yields90+249.5=339.5 and26.5096% precision.

Feedback ties the answer to the two denominators; explain fractional expected counts. Real NAB outside-window counts must never be put into this diagram as though they were verified false positives.

## 6. Actual temperature trace and an unsolved threshold investigation

Placement: §10 after the full fixed comparison, or initially before the table with an explicit “worked comparison” reveal. Do not merely reproduce the already-solved .95→.99 values as the learner's only activity.

### Data, scope and units

Use supplied original CSV, event windows and [nab-derived-scores.csv](nab-derived-scores.csv). The latter contains 22,671 feature rows and actual scores, with reference scores blank. Hash/provenance live in dataset-provenance.md and visual-input-calculation.json. All learned models use level+change; baseline uses level. No detector retraining in the browser is needed for the specified threshold task.

Overview: level versus recorded time, reference/calibration/test bands, four annotated windows. Detail panel: chosen interval's actual scores plus threshold and row alerts. Each alert can expose timestamp, level, change, score, period and annotation-window ID. Label recorded units/timezone unknown, not invented Celsius/UTC.

Do not fabricate a smooth curve from the aggregate table. Use actual rows. Downsample only drawing, preserving extrema per pixel/bin and drawing alert/window markers separately; derive counts from every original score. Never connect across missing timestamps as though observations were continuous without indicating gaps. Annotation start/end lines use exact inclusive timestamp comparisons.

### Meaningfully editable task

Choose method and either:

1. calibration quantile q typed anywhere from.90 through1 (at least three decimals), with threshold defined by NumPy higher empirical quantile; or
2. direct anomaly score tau, typed on the selected method's real scale.

Keep these modes explicit. Negative anomaly-oriented OCSVM thresholds are valid. In direct mode, display the corresponding calibration alert count only after commitment. Do not constrain every method to0–1.

Initial q=.975 is an **unsolved** state: no answer shown and no default prediction. User predicts window-hit count 0–4 and an unmatched test-alert range (e.g.0–500,501–2000,2001–5000,>5000, not sure), or enters an exact numerical prediction voluntarily. Commit method+mode+input before revealing.

After revealing, display calibrated threshold, calibration denominator 1152, test denominator 20634, inside/outside counts and window-hit list. Repeat attempts permit user-chosen q or tau beyond prescribed examples; changing them clears the current result/prediction state. The time trace can remain visible as context, while new threshold-specific alert markers/counts are hidden until commitment.

The earlier .95/.99 pair remains an explicitly worked comparison. Show “all four windows still hit” alongside changing row counts; it is a null for this event metric, not an assurance that every larger threshold preserves hits.

### Author-only expected .975 cases

These values are for phase-two verification, not prefilled learner answers:

| Method | Threshold | Calibration alerts | Test alerts | Outside windows | Window hits |
|---|---:|---:|---:|---:|---:|
| Baseline |4.566579723695049|28|2170|1036|4|
| Isolation Forest |.5555652392489324|28|5632|4263|4|
| One-Class SVM |−6.415545934270259|28|9661|8318|4|
| LOF novelty |1.4936764075602469|28|8191|7002|4|

The q=.95/.99 reference rows are in author-calculations.json. q=1 uses max calibration score and strict greater-than; it guarantees zero calibration alerts, not zero later alerts. For a direct tau larger than every test score, zero test alerts/hits is a necessary null and precision must not be fabricated.

### Temporal feedback and comparison

All counts are row alerts unless labeled window hits. Outside-window = unmatched workload, not false positives. A first alert relative to a window start is not lead time relative to an independently established fault onset. Do not implement point-adjusted labels or official NAB scoring under these captions.

Optional alert episode grouping is outside the specified score contract; if added for learner benefit, require an explicit duration rule and show raw row counts alongside it rather than quietly changing the task.

### Mobile, accessibility and required checks

At 1440 use overview over detail with metrics beside detail. At 390/320 stack overview, method/input/prediction, detail and counts. Provide previous/next window buttons and labeled date selection, not pinch-only zoom. Keyboard selection moves between actual observations; a table with bounded visible rows provides numeric access. Dense full-series SVG DOM of tens of thousands of markers is unnecessary; use an appropriate rendering surface with textual equivalents and focusable selection controls.

Phase two must verify record hashes, missing-reference-score semantics, higher quantile index ceil(q*(n−1)), strict ties, negative score thresholds, all event endpoints, imported decimal precision, unchanged full counts under downsampling, actual input edits, prediction invalidation and window-hit null/contrast. Inspect screenshots in normal reading flow and at unsolved/answered/error states in all three widths using intended fonts. Those checks remain deferred.

## 7. Practice, references and phase-two review checklist

Practices A–J have independent changed inputs, hints and complete solutions in lesson.md. Each question precedes closed hint/solution disclosures; do not automatically open solutions when a section is reached. “Try it” explanations also start hidden. Programme questions can be displayed before code; setup appears before the first program and all examples retain their complete input/print context.

Resources remain annotated links, including the actual recording and companion rather than a generic video-search result. Link provided CSV, JSON and license with meaningful download labels. Preserve route module context.

Required later review must distinguish:

* model arithmetic from rendered geometry;
* exact expected cut calculations from finite random trees;
* real observed scores from hypothetical population rates;
* library training factors from new-query scores;
* content-author calculations from independent/runtime verification;
* a specified readable design from actually opened final screenshots.

The current task ends with this buildable content packet. There are no built labs or successful-browser claims to inherit.
