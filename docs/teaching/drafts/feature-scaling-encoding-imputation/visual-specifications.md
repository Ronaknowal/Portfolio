# Feature preparation: visual and investigation contracts

## Current live-exploration contract — 21 September 2026

This dated UX amendment supersedes earlier prediction-entry, grading, commit-to-reveal and prediction-retirement requirements in this document. There is no learner prediction feature, even optional. Historical evidence below records the earlier interface and remains history; it is not the current acceptance contract.

Edit query coordinates and divisors and inspect distance contributions live; clear donor measurements, change target column and neighbour count and inspect eligible donors and imputation; select or edit a held-out record and inspect its transformation under the frozen training-fitted bundle; change categories, targets, folds or smoothing and follow the donor graph without crossing held-out-label boundaries.

Keep separate independent practice, model predictions, scientific validity checks and training/validation/held-out information boundaries. Meaningful valid control changes must reach the visible calculation and topic-specific diagram together. Natural algorithm Step/Back/Run actions remain where they expose a process; they must never require a learner guess. Reset restores a coherent initial state. A graph or number must not silently describe obsolete inputs; invalid inputs show an error and either clear invalid outputs or explicitly retain the last valid result. See [the current migration evidence](../../LIVE-EXPLORATION-CLASSICAL-EARLY.md) for implemented checks and limitations.


Research/write packet, 12 September 2026. These are complete implementation instructions, not a claim that UI or a live lab exists. Read lesson.md in sequence before implementing. All concrete observed results are in calculated-inputs.json; constructed fixtures are explicitly identified below. Preserve the distinction between a training-fitted rule and a changed later record.

## Shared behavior without forcing shared appearance

Use the site's typography, focus treatment and semantic colors, but give each mechanism its appropriate representation: geometric ruler, incomplete-cell donor table, pipeline column trace, and target-dependency graph. No generic slider whose only output is a paragraph. Initial outcome predictions are unset. The learner selects a prediction and optionally gives a reason, then commits it with the current draft input. Reveal the computed outcome and reasoning only after Apply. Store the prediction with an input fingerprint; editing an entity, method, inspected output, or fold invalidates that prediction. Preserve a visibly labeled previous trial for comparison without presenting it as current evidence. Reset restores original entities and an unset prediction. Explanatory static figures remain visible before the labs.

All equations have readable text/table equivalents. Color distinguishes roles but labels, shapes, and connectors carry the same meaning. Keyboard controls must edit every draggable quantity. At narrow widths, stack coordinated views without shrinking text; keep the inspected row and legend adjacent. Numeric output should distinguish raw measurements with units from unitless transformed coordinates. No remote data request, training package, GPU or external service is required to read the lesson. Keep interactions bounded and compute on Apply; cache immutable offline results at topic level. Phase two owns keyboard/mobile/rendered checks, actual result comparison, and lazy-loading integration.

## F1 — One record, three meanings (lesson §1)

Show a single table row with cells `40 mm`, `4000 g`, `female`, and target `Adelie`. Under the first two cells, a branch labeled numerical preparation; under sex, a categorical preparation branch. Beside the row, a second example replaces mass with a visually explicit missing cell. Its branch inserts a labeled estimate rather than changing the missing symbol into a measured number. Target follows a separate labeled training-answer rail into model fitting. At the output show symbolic named coordinates, not made-up fitted numerical statistics; actual numbers arrive in F4/I3. Caption: “Units, categories, and absence require different representation choices.” On mobile, a vertical flow with the target rail still separate is sufficient.

## F2 — Five observations, three rulers (lesson §2)

Use calculated-inputs.json.scale_fixture for all displayed Standard/MinMax/Robust coordinates. Source values are the constructed training column `[1,2,3,4,100]`; the new observation 150 is marked separately. Training-fit statistics are shown beside each number line: mean 22, population-style standard deviation, min 1/max 100, median 3/IQR 2. Use exact formulas for tooltips and four decimals in the visible table. The source and transformed number lines each have actual labeled ticks; avoid normalizing every row into an unlabeled identical width. A clearly outlined inset magnifies the 1–4 observations, with a connector showing the full-range interval it expands. Full range includes the 150 point; it cannot be cropped because it is the min–max out-of-range contrast. Robust outliers remain on the full line, not deleted. A source-row ID/shape follows each point across views. Reading alternative is the manuscript's complete table.

## I1 — Choose the ruler (lesson §2)

Purpose: discover how per-feature units change nearest-neighbor ordering while common positive rescaling does not.

Initial constructed entities: query Q bill 40 mm/mass 4000 g, candidate A 41/4100, candidate B 43/4001. Learners can edit the query and both candidates by number input or point movement. Divisors are positive numeric controls, initially bill 1 mm and mass 1 g; offer named raw-unit and 1 mm/100 g presets as entry points, not as the only experiments. Record an unset choice A nearer/B nearer/tie before applying. Permit finite reasonable table values and positive divisors; show local validation for zero divisors without coercing them silently.

Display the original physical coordinates with proper units and a separate transformed-difference plane centered on Q. Draw the two difference vectors, equal-distance circles in scaled coordinates, and an aligned contribution table: bill difference squared/divisor squared, mass contribution, total squared distance. The result label is determined from the unrounded computed totals. A tie uses an explicit numerical comparison tolerance and remains labeled tie; the lesson does not smuggle in a classifier tie-break rule.

Checked default raw totals: A 10,001, B 10 → B. With divisors 1 and 100: A 2, B 9.0001 → A. Contrast entity edit: under those scaled divisors change A mass to 4,400 g; A total becomes 17 → B. Null: multiply both scaled divisors by 2; totals become .5 and 2.250025, ranking stays A. Another null translates every entity by the same vector while holding divisors fixed. Feedback names the contribution responsible and contrasts the submitted prediction with the actual winner. Explain that axis scaling expresses a metric choice, not physical movement. All geometry is independently reproducible from the contribution table.

Phase-two checks: actual changed inputs change plotted points and totals; circles match transformed units and do not falsely use square aspect ratios on an anisotropic physical plot; tie/invalid-input states readable; no computed result accidentally revealed before committed prediction.

## F3 — Categorical geometry (lesson §3)

Show the exact full one-hot table for red/green/blue and equal pairwise distances sqrt(2). A triangular schematic may represent these pairwise relations; label it a schematic of three points in a three-coordinate representation rather than pretending it is the raw coordinate plot. Beside it show dropped-red 2D coordinates red(0,0), green(1,0), blue(0,1) on a true equal-aspect coordinate plane; label distances 1,1,sqrt(2). Underneath show unknown all-zero vector as a distinct “unknown input” annotation coinciding with the dropped reference only in that representation. Alternative is a complete pairwise-distance table. No implication that biological categories have ordinal numeric values.

## I2 — Donor eligibility and overlap (lesson §4)

Purpose: calculate which cells contribute to a missing value when different rows have different observed coordinates. Initial donor table D1 `[1,10,100]`, D2 `[3,missing,300]`, D3 `[missing,14,500]`; query `[2,12,missing]`. Columns a,b,c are constructed numeric coordinates with no claimed physical unit. Default target is c, k=2, uniform weights. The source table is editable: allow numeric donor cells to become missing and vice versa, edit observed query values, and choose k=1..3. Keep the target query cell absent; selecting another target explicitly makes its query cell absent. Do not fabricate an imputed value and then use it as an observed distance feature.

Prediction records the donor set and numeric estimate or “no eligible data” for the applied input. Reveal a per-donor strip: shared coordinate names, q, m/q with m=3, squared differences, adjusted squared distance, target eligibility. Sort eligible rows with defined distances; explicitly state stable source-row order breaks equal-distance selection ties. Average up to k eligible nearest donor target values. If none has a defined distance but observed donor target values exist, use their column mean and label fallback. If the whole donor target column is missing, show cannot estimate in this investigation rather than quietly inventing a zero; this explicit teaching policy differs from optional keep-empty implementations.

Checked initial squared distances D1=7.5,D2=3,D3=12; target c eligible all three; k2 estimate200. Contrast: remove D2.c → D1/D3 donate300. Separate contrast: original table with D1.c=140 → D2/D1 donate220. Null: change original D3.c from500 to900 → still200 because D3 not selected. Changing c of an eligible selected donor changes the estimate without changing overlap distance because query c is missing. Empty-overlap example query `[missing,missing,missing]`, target c, donors original → fallback mean300. No uncertainty claim is inferred from the spread of three donors.

On mobile, keep original table above the selected donor explanation; horizontally scroll a full table with labeled header if necessary, while the selected-row explanation itself wraps. Phase two must test arbitrary missing masks, changed query target, donor tie, no-overlap/all-missing distinctions and prediction invalidation.

## F4 — Fitting boundary and pipeline branches (lesson §§5–6)

Two connected figures, sharing labels. First shows training rows creating the fitted statistics/vocabulary object; evaluation rows apply the object without an arrow into fitting. A muted crossed backward arrow names fitting on held-out data. This diagram teaches one boundary and does not claim a pipeline fixes target availability or group leakage.

Second uses the actual fitted standard model. Numerical branch: median `[45,17.3,197,4000]`, then saved means/scales in calculated-inputs.json.scaled_inspection. Category branch: missing→not_recorded, then `[female,male,not_recorded]` one-hot order. Join into the seven exact feature_names. Source row309 yields `[1.3091367505,.8772706993,.1645100047,-.1127925780,0,1,0]`. Keep row index zero-based/source origin explicit and not a model feature. Show actual training count258/test86.

## I3 — Inspect a held-out record through the fitted pipeline (lesson §6)

Use the real CSV and calculated-inputs.json.scaled_inspection, including all86 raw held-out rows, fitted objects, transformed rows, truth/predictions. Default source309. Learner chooses any observed held-out row and edits an independent copy of any of its four numeric measurements or recorded category; permit missing values and an explicit unknown category input. Keep a visible badge “edited copy” distinct from “observed source row.” Do not overwrite the observed file or claim an edited record is a measured specimen.

Select one output coordinate to predict, with a numerical prediction or category-vector choice. This selection and the record are part of the prediction fingerprint. Reveal the branch operation in order: original cell, imputed value if absent, subtraction, division, or category activation; then display the joined seven-column result. Prediction feedback reports the exact calculation and deviation in units of the selected output. Training statistics remain frozen through all edits. Unknown categories follow the specified all-zero policy; missing category maps to the fitted not_recorded coordinate.

Checked contrast: replacing source309 mass4100 with missing supplies4000, giving `(4000−4190.988372093023)/806.6875829303058`, approximately−.2368, instead of−.1128. All other coordinates stay unchanged. Category contrast: male→female changes finalblock `[0,1,0]`→`[1,0,0]`; unknown→`[0,0,0]`; missing→`[0,0,1]`. Null: edit body mass while predicting unchanged bill-length output; its value remains1.3091367505. Do not recalculate classification scores for edited rows from the fixed real-experiment result table. An optional classification extension requires implementing the actual saved training preparation/neighbors and new prediction verification; it is not required by this packet.

The fixed real comparison uses correct counts38,67,84,85,85 out of86 for majority/raw/standard/minmax/robust. Use a zero-origin count axis and an exact table, not a cropped score plot exaggerating the one-row difference. Label controlled holdout comparison and manuscript's no-universal-winner interpretation. Standard confusion rows/columns order is Adelie,Chinstrap,Gentoo; counts stored in JSON.

## F5 — Rank versus log map (lesson §7)

Constructed training `[1,2,3,4,100]` maps to illustrative rank coordinates `[0,.25,.5,.75,1]` and natural logs `[0,ln2,ln3,ln4,ln100]`. State this is the explicit rank convention, not a claim to replicate every QuantileTransformer interpolation detail. Show identified observations with connectors, numeric axes, and a table. A small changed-state illustration replaces100 with1000: its rank remains1, its log becomesln1000. This is a null for rank and a contrast for magnitude. Do not draw an invented empirical Gaussian density. The circular angle example can use two vectors at1° and359° on a labeled unit circle, with sine/cosine coordinate formulas and degree→radian conversion shown.

## I4 — The target-dependency graph (lesson §8)

Initial six rows/category/target/fold values exactly match calculated-inputs.json.target_encoding. Internal folds alternate0,1; smoothing2. Editable category labels, binary target cells, fold membership and nonnegative smoothing give meaningful changed entities. Require both folds nonempty; if smoothing0 and a category has no donor, explicitly use the fold prior as the chosen unseen-category policy. The manuscript formula assumes positive denominator; teach this fallback without a zero divide. Keep outer-training boundary around the whole six-row table; no external evaluation rows are present.

Inspect one row. Show directed edges only from the other fold's targets. Different labels/styles distinguish same-category contributions to the numerator from all-donor contributions to the prior. The inspected target has no edge into its own encoding. Before Apply record the predicted encoded value and optional explanation; after Apply show donor count/sum, prior, smoothing contribution, denominator, result. Changing any table cell invalidates the current prediction, even if the computed result will be a null.

Checked base array `[5/9,7/9,2/9,4/9,2/9,7/9]`; changing row0 target1→0 gives `[5/9,2/9,2/9,2/9,2/9,5/9]`. Thus inspectedrow0's own target edit is a null, but inspectingrow1 sees a contrast. Changingrow3 target0→1 makes row0 value7/9 while row3 remains4/9. The graph must update the prior edges, not just category-match edges. Keep predictions/input commits honest and reveal fractions where exact to expose the mechanism. Narrow screen can replace crossing edges with two donor lists; preserve dependency information.

## F6 — Uncertainty through several analyses (lesson §9)

An incomplete table branches into three explicitly symbolic plausible completions; don't invent observed completed datasets. Each passes through a separate analysis box producing estimate9,10,11 and variance4. Combine average estimate10, within variance4, between variance1, correction4/3, total16/3. A labeled variance-contribution bar starts at zero and has units estimate-units squared, not standard-error units. The final standard error sqrt(16/3)≈2.309 is a separate value. A crossed shortcut from averaged completed tables to one analysis shows the wrong order. Caption confines validity to appropriate imputation/analysis assumptions; no repeated warning text in the program.

## Phase-two continuation

Implement all inline figures and four distinct investigations in topic-specific modules with descriptive identifiers, loaded only for this topic. Keep observed-data transformation and derived fixtures numerically traceable to the packet. Execute displayed programs separately from UI fixtures, test changed and null inputs, compare every plotted state to actual values, and inspect desktop/mobile/text alternatives. Verify optional hints/solutions start closed and core readiness does not depend on deeper branches. No formal browser/native campaign has been performed in this content-only phase.
