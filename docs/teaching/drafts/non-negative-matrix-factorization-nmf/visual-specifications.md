# NMF visual and investigation specifications

Content-only packet, 12 September 2026. These are buildable teaching contracts, not implemented or browser-reviewed figures. Read the complete [manuscript](lesson.md) and [design record](design.md). All calculations use observations in rows, components in H rows, and activations in W columns. Reconsider the proposed presentation if it improves comprehension, preserving the mathematical model, exact inputs and evidence category.

## Shared behavior and representation

Use a sequential intensity palette for nonnegative feature/contribution mass and a diverging signed palette for residuals. Component identity has a stable name and index in text as well as a color. Always say which quantity is normalized. Controls operate on draft input until Apply; the applied model, labels, figures and values change atomically. A recorded prediction starts unset, includes its active input key, and is invalidated when the relevant draft changes. A learner can explore without grading, clearly marked as unrecorded; do not retrospectively pretend a prediction was made. Provide Apply, Reset, keyboard focus and numeric input alternatives to dragging. Round only display values. Relative tolerance applies to grading float equalities, with the exact-equality fixtures below avoiding ambiguous boundary scoring.

The existing dark/gold design remains. Match representation to purpose: aligned additive image strips, cone geometry, a factor update worksheet and real image contributions. Do not repeat a generic chart-and-text panel. At 320 CSS px, place the observation above its components and each exact-value table below its visual; retain minimum readable text size rather than scaling a large SVG down. All figures must be meaningful without color or hover. Any animation is optional, reduced-motion compatible and manually stepped. No timers, workers or global data imports are needed for these bounded inputs.

## F1 — A row is a sum of component contributions

- Placement: §1 immediately after the first reconstruction calculation.
- Question: which entries multiply and which resulting contributions add to one observation?
- Data: X row1=[2,1,3], W row1=[2,1], H rows=[1,0,1] and [0,1,1]. Show unscaled patterns, multiplication by 2/1, weighted rows [2,0,2] and [0,1,1], then their sum [2,1,3]. Cell3 highlights 2+1=3.
- Geometry: three aligned columns are features1–3, not time; horizontal strips are vectors. Equal-width cells encode feature identity; intensity and explicit values encode amounts. A brace labels k=2 contributions, not two categories.
- Text equivalent: a semantic three-feature table with each addend and sum, followed by the displayed equation. The learner should see the same multiplication in vector and matrix form.
- Mobile: vertically stack strips with the multiplication label on its own adjacent line; keep feature alignment.
- Checks in phase two: trace each product exactly, cross-highlights and all-zero contributions. No browser evidence yet.

## I1 — Edit an additive mixture

- Placement: directly after F1, opening in its taught example state.
- Goal: predict effect of changing an amount or pattern entry, and construct different observations sharing one feature.
- Model: W=[a,b], H=[[h11,h12,h13],[h21,h22,h23]], prediction x_j=a*h1j+b*h2j. Each amount in [0,4] in .25 steps, each pattern entry [0,2] in .25 steps; support keyboard numeric entry on the same lattice. Default a=2,b=1,H as F1. Patterns and amounts may be edited simultaneously, but before applying show the selected target feature and the previous applied value.
- Recorded prediction: radio choice increase/decrease/unchanged for selected feature1–3, initially unset. Commit binds to old and proposed state; Apply compares proposed minus old using tolerance1e-12. Feedback says which contribution changed and why, with actual old/new scalar values. A direct-input numeric predicted total is an optional secondary mode, not a substitute for mechanism feedback.
- Distinct visible views: contribution strips update together; selected feature's little sum expands into operands. A general prose box alone is insufficient.
- Checked constructed alternatives: default → a=3 gives [3,1,4]; default → b=0 gives [2,0,2]; default → h12=.5 gives [2,2,3]. Null: when a=0, editing h11 from1 to2 leaves all reconstructed values unchanged. Fixed-third-feature synthesis: (a,b)=(1,2) and (2,1) both give x3=3 but x1=1 vs2.
- Open investigation: find three distinct points with x3=3, then edit a pattern so the earlier rule a+b=3 no longer characterizes that feature. The input is not restricted to labeled presets.
- Reset restores defaults and clears recorded answer/history. At most five prior comparisons may be retained; no unbounded history. Invalid/out-of-lattice values show the supported range beside the field without applying them.
- Boundary: an exact user-composed reconstruction, not a learned fit or a classifier. It does not discover physical components.
- Phase-two checks: all alternatives above, edit invalidation, keyboard flow, reset, exact correspondence between selected operands and strip totals at desktop/mobile.

## F2 — Error is a signed difference before it is squared

- Placement: §2 after the observed/reconstructed row.
- Exact inputs: observed[2,1,3], reconstructed[1.5,1,2.5], residual[.5,0,.5], squared residual[.25,0,.25], half sum=.25.
- Layout: observed and reconstruction on same nonnegative0–3 scale; signed residual on−.5 to+.5 scale with center0; the caption states positive residual is missing mass. Connect each residual cell to its squared contribution, using numbers rather than marker area as the sole encoding.
- Add a compact same-input loss comparison from calculated-inputs.json: observed2/reconstructed4 versus20/22. Use a table as primary representation; a tiny bar comparison is optional and must show KL.613706/.093796 and IS.193147/.004401, with separate axes. Do not combine unlike losses on one unlabeled axis or call the table an empirical likelihood study.
- Text/mobile: values and calculations remain present in normal reading order.

## F3 and I2 — An alternating update has two different phases

F3 introduces the process before controls: X is fixed; H changes using oldW, then W changes using newH. Include X3×3,W3×2,H2×3 dimension labels and highlight one H cell's actual dot products. In §3 the first H11 numerator5.5, denominator2.65 and old value1 lead to2.075471698. The post-H state precedes the post-W state; only after both phases is the recorded sweep loss.039447438. Show F0=17.06 before any update.

I2 turns the trace into an investigation:

- Author model: formulas in `author-calculations.py:update`, no added epsilon for the strictly positive supported entry states. X cells [0,6] on .25 grid, initialized matrices positive [0.1,4] on .1 grid for edits. Keep every X row and column with positive sum; reject an edit that makes a full zero row/column in this simplified model. Exact-fit preset is handled separately and is valid for the zero H entries because each required denominator is positive.
- Controls: choose an X cell and an initial H cell to edit; W initialization remains the documented fixed starting matrix. Prediction asks whether selected H entry grows/shrinks/is unchanged on next H phase. Select H target before answering. Start/apply records immutable X and initial factors; manual Update H then Update W buttons; max40 sweeps; Back restores complete previous factor state. Edits restart from the new initial state and clear predictions.
- Fixtures: default trace recorded in JSON for all0…40 sweeps; at1 loss.039447438,2 loss.028233772,10 loss.001806076,40 loss4.9313891e−8. Changing X[0,1] from1 to2 yields H[0,1]=.489795918 and H[1,1]=2.264150943, post-sweep loss.224942552 (JSON). Changing initial H11 from1 to3 while others stay fixed makes numerator5.5, denominator7.15, newH11=30/13≈2.307692308: it decreases. Default H11 increases. Use these checked arithmetic contrasts to avoid a prediction task whose every supported outcome is “increase.”
- Exact-fit null: W1,H1 in §1 and X=W1H1. H and W stay fixed, loss0. Positive ratios are1; zero H entries remain0. This is not the zero-lock failure fixture.
- Selected dot products expose all summands, with matrix-shape alignment. A log10 loss graph may accompany the trace, plotting positive losses and an explicit exact-zero strip rather than log(0). Show iteration0 and small later improvements via log scale; a huge initial linear-axis drop must not flatten every other step. Label total half-squared loss, not norm/MSE.
- Zero-lock demonstration is a separate static contrast, not an invalid unguarded execution in this lab: fixedW=[1],X=[2,1],H=[0,1]; a defined “leave zero unchanged” multiplicative rule leaves H1=0 despite gradient−2, whereas fixed-W NNLS gives[2,1]. Caption specifies the zero convention; do not evaluate0/0 as if it were mathematical evidence.
- Transfer: edit an untaught X entry, predict a different H cell, then explain why loss can drop while a W entry shrinks. Boundary: short pedagogical updates, not production unconstrained input support or a promise of stationarity.
- Phase two: independently derive the default H/W step and counterexample, ensure snapshots use newH inW, grade and clear correct prediction keys; numerical precision and active state consistent across all views.

## F4 — Two explanations inhabit the same measured region

- Placement: §4, after the two exact factorizations.
- Exact geometry: 2D projections of dictionary rays H1:(1,0),(0,1), H2:(1.25,.25),(.25,1.25); observations(2,1),(1,2),(3,3). Axis0…3.5 with equal aspect, feature1 horizontal/feature2 vertical. Draw two patterned translucent cone regions, rays with labels and shared data markers. Show the full three-feature factor/product tables beside/under it: feature3=x1+x2 in this fixture.
- Goal: seeing all observations inside both feasible cones explains ambiguity beyond order and scale. Plot positions are numeric, not decorative arrows. Do not imply these are two local-minimum losses; both exact products equalX.
- A separate small scale-invariance strip depicts Hrow×2,Wcolumn÷2 and unchanged contribution; it is not a third geometry region.
- Mobile: geometry one panel with toggles to isolate ray sets, but initial state includes labeled rays from both; exact text declares that each set contains all three points. No required hover.
- Author evidence: `ambiguity_products` inJSON and hand multiplication in manuscript. Verify both full products in phase two.

## F5 — Training learns a dictionary; transform learns new amounts

- Placement: §5 before program.
- Two data lanes: train180×64 → fit → Wtrain180×k and Hk×64; validation60×64/test60×64 each→transform with sameH→newW60×k→reconstruction60×64. Only train has an update edge toH. H is a shared object, not copied-and-refitted.
- Include a thin scale16 operator on each raw input lane; no learned scaler is implied. source_row/digit columns branch into ID/split/diagnostic labels, not into model features.
- Text equivalent reproduces exact information flow; mobile stacks lanes while keeping sharedH identity visible.

## F6 — Recorded candidate errors, without a manufactured elbow

- Placement: after §5 validation table.
- Provenance: `calculated-inputs.json:runs`, actual author fits on offline data and recorded settings. Plot k1,4,8,16 vs MSE; include both train and validation points for seeds7,19, distinguished by line pattern/marker. Show each run; the seed19 point at16 differs enough to see on an inset0.010–0.030 if needed. Main axis0–.08 preserves the one-component baseline. No timing or uninspected between-k fits.
- Caption: “Both validation fits improve over this inspected component range; the two initializations disagree more at larger k.” Do not label an elbow or true component count. K8/seed19 is predeclared for inspection, not gridwinner; show candidatewinnerK16/seed7 with a text annotation on its validation point only.
- Separate final diagnostic bars/table for mean.069994624,PCA8.019678463,NMF8.025380891. Name test-image MSE on scaled pixel units. PCA uses its learned mean; the comparison is not equal bit budget. Retain its unfavorable-to-NMF outcome.
- Desktop/phone perceptibility remains unverified. During implementation inspect actual plotted separation and inset; exact table always available.

## F7 and I3 — What did a component contribute to this actual image?

F7 is visible directly in §5, no clicking required:

- Offline source: digits-300.csv and JSON `splits.test`, `visual.H`, `visual.W_test`, `visual.reconstructed_test`. First example corresponds to UCI/sklearn source row 242, at test index 0. Fit k = 8, seed = 19. Reshape features in row-major 8×8 order.
- Initial view: observed, reconstructed and signed residual images plus eight small actual dictionary patterns. Each selected contribution image equals W[i,r] times H[r,:]. All intensity panels share scale zero to max(1, maximum current reconstructed/contribution value), with the raw numeric maximum labeled. The initial reconstruction contains a value ≈1.02, so silently clipping at 1 is forbidden. The residual panel has its own symmetric −maxabs to +maxabs scale. Zero residual stays white/neutral with numeric 0.
- Component pattern gallery may offer normalized shapes, but all eight divisors are labeled and reconstruction continues using original values. No reordering of model identities when normalizing; ordering by contribution totals is an explicit display order.

I3 investigation has a different job from I1/I2:

- The learner selects any of 60 held-out image IDs (display digit only as an optional diagnostic), toggles which of eight fitted contributions to include, and records the direction of image MSE change before Apply. Hold fitted H and W fixed; no browser refit. Prediction binds to image ID, old mask and new mask. Offer individual-pixel error prediction by clicking an 8×8 cell using keyboard-accessible cell selection. This reveals that one overpredicted pixel may improve even when total error increases.
- Default exact fixture, source row 242: row MSE .021503642603. Remove component 2 (one-based; coefficient .740160683895) → .068409087403. Remove component 1 → .024686350768. Remove component 8 (coefficient exactly 0) → unchanged .021503642603. All eight removals were calculated; values are recorded in design/checks. Also test the all-zero mask: reconstruction is zero and MSE = mean(Xrow²), calculated from the provided CSV. Multiple mask combinations are generated exactly from supplied W and H, not invented fit outcomes.
- Main comparison: before/after residuals and total MSE; highlight the removed contribution and inspect exact pixel values. Show per-pixel improvements and deteriorations in a signed squared-error-change image with a textual count of each. A low summed error is not evidence a removed component is a physical source.
- Core null: zero-activation component 8 at source row 242. Selecting another image may activate it; the visible coefficient establishes which case is being tested.
- Numerical feedback: at a row NNLS optimum, removing any nonzero coefficient with others fixed cannot reduce total SSE; allow 1e−8 numerical tolerance and explain deviations as solver accuracy to investigate. Do not claim arbitrary MSE improvement is expected from these fitted rows. The pixel task can improve or worsen and is not bound by the summed optimum argument.
- Transfer: choose another ID and the component with greatest **contribution sum**, not greatest raw coefficient; explain three pixels and record total MSE. An optional slider can scale a selected coefficient from zero to twice its fitted value in .1 increments for further entity edits, but its values must be derived when applied and checked. This is not required if it burdens mobile comprehension.
- Controls: Reset restores the full contribution mask and first image, clearing the prediction. Changing the image clears mask history and prediction. No automatic animation. Exact values can be selected without precise touch placement. On a phone, stack before/after views and keep the predicted quantity visible with the Apply button.
- Phase-two checks: CSV row identities, row-major orientation, contribution sum/reconstruction/residual agreement, chart scales, zero-activation null, all-contributions-removed boundary, input invalidation when switching images, keyboard sequence and selected pixel coordinates.

## F8 — A positive rectangle cannot cover crossed zeros

- Placement: §7 after the ordinary/nonnegative-rank matrix.
- Exact 4×4 matrix S = [[0,0,1,1],[1,0,0,1],[1,1,0,0],[0,1,1,0]]. Mark (1,3), (2,4), (3,1), (4,2) with letters A–D. Pair inspection displays the rectangle spanning two chosen positive cells and one of its crossed zero cells. A zero cell stays labeled 0 and visibly excluded from positive support.
- Goal: show why the four designated positive cells cannot be covered by fewer positive outer products. A static set of small panels with representative pairs and text for the other pairs is sufficient. Optional pair selection adds accessible inspection but is not another prediction lab.
- Check all six pairs contain a crossed zero. Ordinary rank 3 can be verified with the row dependency and a 3×3 minor. This is exact combinatorial reasoning, not an optimization run.

## Intended implementation ownership and closure

Use existing topic source `src/learn/data/topics/non-negative-matrix-factorization-nmf.jsx` with topic-owned figures/labs/models/examples named for NMF mechanisms. Proposed `NmfFigures.jsx`, `NmfInvestigations.jsx`, `nmf-models.js` and `non-negative-matrix-factorization-nmf.js` example/data files are semantic suggestions, not files created in this phase. Keep actual dictionary/test inputs with the consuming lesson; do not import the manifold manuscript, other lessons, the full authoring JSON or all topics at runtime. Export only the small selected production data needed, retaining the CSV/provenance download.

Before closure in an authorized finish request, execute displayed programs fully; verify independent numeric/probability/geometry cases; inspect actual diagrams at desktop/320px and zoom; exercise recorded predictions, input edits, meaningful contrast/null states and keyboard behavior; perform formal independent correctness and learning-experience review. Current author-only calculations establish numeric inputs, not these deferred checks.
