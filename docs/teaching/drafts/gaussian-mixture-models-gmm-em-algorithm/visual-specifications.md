# GMM visual and investigation specifications

## Current live-exploration contract — 21 September 2026

This dated UX amendment supersedes earlier prediction-entry, grading, commit-to-reveal and prediction-retirement requirements in this document. There is no learner prediction feature, even optional. Historical evidence below records the earlier interface and remains history; it is not the current acceptance contract.

Edit the measurement, mixing weight and variance and follow densities and responsibility shares immediately; edit observations or initial means to restart the current EM trace, then step E and M phases separately and step back; vary correlation and point positions beside a permanent zero-correlation reference. No learner answer precedes a calculation.

Keep separate independent practice, model predictions, scientific validity checks and training/validation/held-out information boundaries. Meaningful valid control changes must reach the visible calculation and topic-specific diagram together. Natural algorithm Step/Back/Run actions remain where they expose a process; they must never require a learner guess. Reset restores a coherent initial state. A graph or number must not silently describe obsolete inputs; invalid inputs show an error and either clear invalid outputs or explicitly retain the last valid result. See [the current migration evidence](../../LIVE-EXPLORATION-CLASSICAL-EARLY.md) for implemented checks and limitations.


Stable ID: `gaussian-mixture-models-gmm-em-algorithm`. Complete content-first specifications, 12 September 2026. Implementation, browser inspection and independent phase-two review are **not started**. Read [lesson.md](lesson.md) in full and [the design record](../../GMM-LESSON-DESIGN.md) before building. These are teaching contracts, not a claim that a rendered graph exists.

## Shared representation and interaction contract

Use topic-owned production models/components when phase two is authorized. Keep the current calm dark/gold design and semantic reading order. Do not funnel all representations through a text-stepper: the topic needs density curves, a fractional-allocation diagram, covariance ellipses and an exact objective/bound comparison. The six inline figures have distinct jobs; three investigations let the learner change a mechanism. A useful already-visible investigation view can replace a redundant inline figure if the design record explains the choice.

All density curves are **calculated model densities** from stated parameters. Iris scatter points are **observed measurements**, and candidate scores are **computed fits to a fixed observed dataset**, with the environment recorded in `checked-results.json`. Neither the hand fixtures nor their curves are measured benchmarks. Do not draw simulated sample points unless they are explicitly labeled and generated reproducibly. A Gaussian density has units reciprocal to the feature units; two-dimensional densities have units per square measurement area. Standardized coordinates are dimensionless but must be identified as standardized. Negative log-density and log-likelihood use natural logs.

`checked-results.json` holds exact input IDs, parameter arrays, every shown trace state, the 16 real fits, boundary fixtures and supplementary proof/practice values. `author-calculations.py` supplies the reproducible bounded author calculation. Browser models must implement the formulas for edited inputs, not interpolate between the answer fixtures. Numeric tables provide accessible exact quantities; curves provide the relationships. Compare model values with the retained SciPy/NumPy results and an analytic case rather than asserting only that two views sharing one helper agree.

For every investigation:

1. Initial **prediction is unset**. Provide a labeled radio group with no selected option and an optional short explanation field. The target is determinate, so record the choice, never silently infer it from a control or preset.
2. Maintain `draftInputs`, `activeInputs`, `prediction`, and `predictionInputKey`. Display the active values. Editing inputs clears a checked result and returns the prediction to unset, with a polite status such as “Inputs changed; record a new prediction.” Inputs and prediction are committed together by the explicit calculation action. Disable “Check prediction” until a choice exists; an accessible “Explore without a prediction” action may compute without graded feedback if desired, but must not record a correct prediction.
3. The record contains all relevant active parameters, comparison points, iteration and question type. A previous prediction must never be assessed against different inputs or a later state. After a result, state the actual comparison and why using the model quantities. A wrong prediction offers a causal explanation, not only a red mark. A correct choice accompanied by a wrong explanation still invites the learner to inspect the mechanism; free text is not automatically graded.
4. No automatic animation or unconstrained run. Step/back/reset are explicit. Reset restores the fixture, parameters, step0 and unset prediction. Switching a fixture also resets the prediction and clearly changes the data/model. No hidden randomness.
5. Keyboard-operable numeric fields accompany any draggable marks; arrows and a step value offer precise movement. Every control has a persistent label, unit, range and visible focus. Points have text IDs and shapes as well as color. All outputs have table/text equivalents and a concise live result summary. A visual detail cannot rely on hover, color or animation.
6. At narrow widths, stack linked views in causal order, keeping controls and result together. Redraw charts at their available width with fixed readable type (target 14–16px), rather than scaling a large SVG down. Allow tables to scroll within their own labeled region. No page-width overflow at 320px. Verify 200% zoom, reduced-motion behavior, keyboard order and contrast in phase two.
7. Validate finite numbers and the supported bounds before committing. Retain the last valid state and name the invalid field. Never replace an invalid Gaussian with uniform responsibility or silently clip learner data. Numerical limits of the browser model are described once near its controls.

Do not claim “every alternative passed” for arbitrary continuous inputs. The fixture checks below establish the described family comparisons and null cases. Phase-two validation must add boundary/range checks for the final supported controls and exercise every *discrete* mode that is actually exposed.

## F1 — A hidden selector, weighted densities and a visible observation

**Placement:** §2, after the two-stage generative story and before the first density equation. **Question:** why do we add densities, and what is hidden? **Misconception:** drawing from a mixture means averaging two measurements.

Draw a selector marked `z: A or B` with branches labeled `P(A)=0.5`, `P(B)=0.5`. Each branch terminates at a miniature Gaussian curve: A mean−2, variance1; B mean2, variance1. Join them to one common x-axis view containing the two weighted curves and their sum. The drawn measurement at x=2 is an example location, not a new random simulation. Label the selector “hidden” with an outline and the observation “observed” with a filled point. Arrow direction encodes generation; horizontal position in curve panels encodes measurement.

Compute curves on x∈[−6,6], sampling adaptively or with at least 241 points. Weighted peaks are about 0.19947; mixture peaks are slightly higher owing to the other tail. Plot y from0 to0.23. Display separate dashed component curves and a solid mixture curve. Areas, not peak heights, are labeled 0.5, 0.5, total1. A shaded interval is not needed because its exact mass is not taught here.

**Text equivalent:** “Choose A or B with equal probabilities. Given A, draw N(−2,1); given B, draw N(2,1). Hiding the selector gives density 0.5 N(x;−2,1)+0.5 N(x;2,1). At x=2 the density is 0.199538055.”

**Mobile:** selector over two small branch panels; the summed curve is full width beneath. Do not crowd all labels into one long horizontal arrow chain. **Verification:** branch weights sum1; component areas integrate to their weights up to the finite plotting tail; true analytic total integrates1. Curves share axes and add at each plotted x. Inspect the x=2 value, overlapping tails, arrow labels and text at desktop/phone. A finite plotting window must not be labeled as containing exactly all mass.

## I1 — Component responsibility versus total density

**Placement:** §2 after the three-row density/responsibility table. **Learning question:** can a decisive component probability coexist with very low measurement density? **Starting state:** fixed base means−2,+2, variances1, weights0.5/0.5, draft x=1. The manuscript has not solved this starting x for this exact model. Before reveal, show the curves and active parameters, but keep the selected point's exact allocation hidden.

**Prediction:** “At your measurement, which component has greater responsibility?” Options A / B / equal. Save it with the full input state. On Calculate, show A and B weighted-density stems at x, their total and a normalized stacked responsibility bar. Also show log-density and negative log-density, with their names rather than an ambiguous “score.” Feedback compares the recorded choice with the weights after normalization; ties are judged in log weighted-density space at absolute tolerance1e−10. A/B naming is independent of which curve lies left.

**Controls:** x numeric field/slider [−10,10], step0.1 with finer typed input allowed; A's weight [0.05,0.95], step0.05; B weight is1−A; A's variance [0.25,4], step0.25; B variance1 fixed. Two named resets: base model and identical components (both means0, variances1, A weight0.2). Changing variance in the identical reset is allowed but visibly exits “identical components”; it is a starting fixture, not a separate normalization algorithm. The learner edits x and weights to solve an unshown case.

**Connected views:** density y-axis is linear for comparing weighted heights where visible; a companion log-density number/compact axis preserves the difference between x2 and x8 without compressing a 10⁻9 tail into the same apparent zero. Extend the curve domain to include the current x with margin; do not clip the selected mark. No claim that the tiny density at8 is perceptible on the linear plot alone. The responsibility bar is [0,1] with percentages and fixed A/B patterns. Exact weighted densities use scientific notation below10⁻4. Never label a displayed rounded1 as exact certainty.

**Investigation path:** record a choice for x1; calculate; choose a new x yourself; then find two x values with B responsibility above0.99 whose negative log-densities differ by at least5. The app checks this target only after both input snapshots have been recorded, and shows both points on the same frozen model. Changing the model clears the saved pair. This two-point comparison is optional synthesis of the same mechanism, not a new lab.

**Checked contrast:** `responsibility` records base x0/2/8; B responsibility0.5/0.999664650/approximately1, densities0.053990967/0.199538055/3.03794e−9. At x0, A weight0.2 versus0.8 reverses the winner while leaving density unchanged. At x0, A variance4 versus1 changes A responsibility from0.5 to0.691438454. **Null:** identical components with weights0.2/0.8 at x0 andx3 have unchanged responsibility despite changed density. The log-sum-exp fixture (−1000,−1001) gives (0.731058579,0.268941421); denominator clipping is not an alternative mode.

**Accessibility/mobile:** plot, input controls, outcome then table in document order. Weighted density stems have endpoint text in a separate aligned readout to avoid overlap. Stack stems above the normalized bar; retain a visible “sum then divide” connecting annotation. **Phase-two checks:** all values agree with formula and table, editable unsolved point works, prior prediction invalidates on x/weight/variance/fixture edits, tie0 and identical-component null behave, reset clears pair/prediction, log-density remains finite at range extremes, keyboard can complete the same investigation as dragging.

## F2 — Responsibility mass becomes weighted statistics

**Placement:** §3 immediately after the complete first M-step and before the next E-step discussion. **Question:** how can one observation affect two fitted means without moving or duplicating it?

Data IDs A–D at x−2,−1,1,2. Initial Left/Right means−1/+1 and variances1. Use `em[0].responsibilities`. Show four stationary observation markers on the common x ruler. Under each marker, split a unit-height rectangle into Left/Right mass. Two component lanes project those fractions downward, retaining their x location. Rectangle height/area encodes responsibility; horizontal position stays measurement. Lane weighted mean pointers move from−1 to−1.344824658 and+1 to+1.344824658. New SD intervals are mean±sqrt(0.691446639); old intervals are mean±1. These bars denote one-dimensional component standard deviations, not confidence intervals for estimated means.

**Labels:** effective counts2 and2, weights0.5/0.5, weighted sum Left−2.689649316, new variances0.691446639. Provide the exact responsibility table adjacent. Zero-looking tails remain labeled as small positive fractions. Caption from manuscript is mandatory.

**Text equivalent:** list each observation's two fractions, then the numerator/count calculation and new variance. **Mobile:** stationary data ruler first, followed by one full-width component lane at a time; retain A–D under all lanes. A small before/after table preserves comparison if old/new bars overlap. **Verification:** no fractional row sums other than1; both effective counts sum4; the new marker is exactly the weighted mean on the axis, and SD lengths derive from variance rather than using variance as a length. No component label is confused with observation A or B.

## I2 — Step exact constrained EM from your own start

**Placement:** §4 after the compact program and first-run interpretation. **Question:** how do initialization, soft allocations and a fixed variance floor affect the next fit? The introductory displayed state must expose both allocation and geometry before interaction.

**State/model:** four editable observation IDs A–D, initially−2,−1,1,2; two components Left and Right with initial means−1,+1, variances1, weights0.5. Fixed variance floor0.05 except repeated-observation preset below, where0.25 is explicitly stated. The browser evaluates exact one-dimensional constrained EM from §§3–4: E-step via log-sum-exp, M-step weight=N/n, mean=weighted sum/N, variance=max(weighted scatter/N,floor). No additive regularizer. Fresh final responsibilities are recomputed for a final model readout, but must not overwrite the saved E-step matrix shown as the input to an M-step.

**Controls:** four numeric observation fields [−4,4]; two initial mean fields [−3,3]; Apply setup; Record prediction; Compute E-step; Apply M-step; Back one half-step; Reset. Initial variance and weights are displayed fixed inputs, not more sliders. An optional “run up to20 cycles” can be added only after at least one manual E+M cycle, with a bounded loop, cancel/reset and no animation requirement. Presets: standard data, asymmetric means(−2,−1), identical initial components(means0,variances2.5) and repeated data(−2,−2,2,2), means−2/+2, variances1, floor0.25. Each replaces the entire declared setup, with an unset prediction. Users may edit points and means in every preset.

**Recorded prediction:** before E-step, radio options “log-likelihood rises after the next full cycle” / “stays the same to tolerance”. Do not offer a correct “decreases” choice for this supported exact update; diagnosis of a decrease is a separate practice task. Save the prediction against setup+iteration, with the old total log-likelihood. The check occurs after the M-step's new likelihood is evaluated; equality uses absolute total tolerance1e−9. E-step feedback says parameter values and observed likelihood remain unchanged while allocations are recomputed. M-step feedback names the old/new counts, means, variances and objective. If near-equality is due only to threshold, state “change below1e−9,” not mathematical equality.

**Visible consequence:** synchronized density curves on x[−5,5], fixed data rug/IDs, an n×2 fractional-allocation view and an objective history that includes iteration0. A “used for this M-step” badge fixes the time of responsibility values. Two panels distinguish the objective's broad early rise and small later gains: main log-likelihood trace has data-fitted y-range, while a small signed gain strip/readout shows changes near the plateau. Do not use a log y-axis on negative log-likelihood values. Current step is text-labeled. For edited inputs, re-evaluate every output; do not replay a retained JSON trace.

**Checked fixtures:** standard trace in `em` gives log-likelihood−7.158186977 at0,−6.461856301 at1,−5.724277515 at2 and−5.675741839 near the plateau; step1 means±1.344824658/variances0.691446639. Changed D=3 yields Right mean1.844792261, weight0.503878397 and variance1.582910448 after one cycle. Asymmetric means are in `em_variations.asymmetric_initialization` and converge to the separated solution, establishing that both offered non-identical starts are functioning. **Null:** `em_null` remains mean0/variance2.5, all responsibilities0.5 and objective−7.508335597. **Floor contrast:** `em_variations.floor_active` uses repeated observations and floor0.25; both variances hit0.25. The standard case does not hit its floor0.05. All retained transitions passed a nondecrease tolerance1e−10 and row sums within1e−14.

**Invalid states/boundary:** the field ranges and positive floor bound logs, but a component may receive numerically negligible mass. If effective count is below1e−12, pause with a specific “This component received too little weight for this teaching update; choose another start or reset” message before division. Do not silently drop/reseed it. This is a supported-browser boundary, not a new EM theorem. Preserve evidence up to the last valid state. Four rows and two components are intentional to keep the full mechanism inspectable.

**Mobile/text equivalent:** data ruler, two component curves stacked, table, then step controls; old/new parameter table appears immediately under the consequence. Half-step text names exactly what is fixed and what changed. **Phase-two checks:** evaluate supplied and edited fixtures, compare derivative-based constrained optimum on a floor-active case, preserve old E matrix through M, no strict-growth claim at symmetric stationary state, current prediction invalidation on all setup changes, backward stepping restores model and appropriate prediction snapshot without labeling future predictions as already made, loop cap/reset/keyboards work. Check curve labels at standard, edited D3, identical and active-floor states, not only the default.

## F3 — The covariance-collapse counterexample

**Placement:** §4 beside the unbounded-likelihood explanation. **Question:** how can training fit improve for a pathological reason? Fixed data−2,−1,1,2; weights0.5/0.5, means−2 and0, broad variance4. Only the first component's SD changes.

Show aligned small curve panels for SD1,0.1 and0.01. Do not put all peaks on one global y-axis that flattens the broad curve invisibly; label each panel's y-range and add the exact height at−2. Use a separate common-axis inset of the broad component to establish that it remains fixed. Width can be encoded by an expanded local x-view for the spike, with clearly marked scale breaks and matching measurement mark−2. A table is the primary exact comparison. Objective strip plots SD1,0.1,0.01,0.001 on a log x-axis, with log-likelihood−8.122121377,−6.945323529,−4.669586147,−2.369725898.

**Accessible statement:** “One component stays broad. The other narrows at an observed point. Its density peak grows as1/SD; its contribution raises the total data log-likelihood without improving a broad description.” No claim that a plotted finite set proves the infinite limit; the algebra in the prose establishes the limit.

**Mobile:** one panel per row, x/SD labels adjacent and table below. **Verification:** component areas0.5, broad parameters unchanged, density scales disclosed, log SD axis increasing direction clearly labeled, objective values match `collapse`, absence of covariance floor explicitly stated. Rendered peak-width contrast must be visible even for SD0.01; use a zoom panel rather than a one-pixel spike.

## I3 — Correlation changes plausibility

**Placement:** §5 after the worked Mahalanobis comparison. **Question:** why do equally distant points have different density? One Gaussian mean(0,0), covariance[[1,rho],[rho,1]], with rho initially0.75. A second, always-visible null panel uses rho0 and the same two points. This is a comparison of two declared Gaussian models, not two fitted clusters.

**Controls:** correlation rho in[−0.9,0.9], step0.05; point P and Q coordinate fields in[−3,3], plus optional constrained dragging; reset P(1,1), Q(1,−1), rho0.75. Marginal variances stay1, so rho alone changes the covariance. The learner edits either point to investigate an unsolved pair. Changing rho does not redraw random data.

**Prediction:** before calculation choose P higher / Q higher / equal. Save rho and both coordinates. Feedback compares log densities at tolerance1e−10; report squared Mahalanobis distances, determinant and each log-density. Normalize no component probabilities here: it is one component with two measurement locations. The determinant is common to P andQ within a panel, which is why comparing quadratic forms determines the ordering. Between panels the determinant changes, so density magnitudes need both terms.

**Geometry:** equal horizontal/vertical scales with axes[−3.5,3.5]. Draw mean0, P and Q with shapes/text, contour D²1 and optionallyD²4 as labeled constant-density ellipses, covariance eigenvector axes, and projections of each point onto those axes on selection. Eigenvalues1+rho and1−rho, directions(1,1)/sqrt2 and(1,−1)/sqrt2; semiaxes sqrt(eigenvalue) for D²1. At rho0 choose those same two directions by convention to avoid meaningless eigenvector flipping. The convention is layout, not a unique preferred direction of the identity covariance. No “68% ellipse” label. The D²1 mass is39.346934% in2D.

**Checked contrasts:** rho+0.75 gives D²P8/7 andD²Q8, densities0.135882281/0.004407103. rho−0.75 reverses them with equal determinant0.4375. **Null:** rho0 gives both D²2 and density0.058549832; choosing identical P/Q is another exact null. The mean point givesD²0 and densities in `geometry`. Add boundary checks rho±0.9 for the actual model during phase two; positive definiteness follows from both1±rho>0 over the range.

**Transfer:** ask the learner to choose two points whose order reverses after changing rho's sign, record both predictions and explain which squared projection changes its weighting. A second task finds points with equal density away from the original diagonal fixture. Use exact comparison, not color as a grading signal.

**Mobile:** show one common-scale panel with a clear toggle between active and rho0 models and a fixed comparison table underneath, or stack both full-width. Do not compress the two plots into unreadable side-by-side thumbnails. **Phase-two verification:** numeric values versus `scipy.stats.multivariate_normal`, analytic inverse/eigenpairs, equal aspect ratio, no rotated/mirrored ellipse error, controls/prediction keys/reset, text identifies single-density comparison rather than clustering confidence, point label collisions solved without moving data marks.

## F4 — Four covariance freedoms

**Placement:** §5 immediately after covariance-family table. **Question:** what do parameter restrictions remove geometrically? An analytic shape gallery, not fitted data or an empirical model ranking.

Keep means(−2,0),(2,0) in every panel. Use the following positive-definite covariances:

| Family | Left | Right |
| --- | --- | --- |
| full | [[2,0.75],[0.75,1]] | [[0.5,−0.25],[−0.25,1.5]] |
| tied | [[1,0.5],[0.5,1]] | exactly the same matrix |
| diag | [[2,0],[0,1]] | [[0.5,0],[0,1.5]] |
| spherical | 1.5I | 1I |

Draw D²1 contours on common equal-scale axes x[−4,4],y[−2,2]. Connect diagonal entries to spread and off-diagonal cells to orientation; tied's two matrices have a “shared” bracket. Spherical uses separate scalar badges1.5 and1 to make the API distinction unavoidable. Shapes/patterns carry component identity across panels. Add total parameter counts for d2,K3 from the manuscript in a separate table rather than falsely implying the two-component drawings contain17 parameters.

**Text:** describe each allowed covariance family and read its matrix aloud in row order. **Mobile:** 2×2 on medium width, one per row below480px. Preserve equal scale. **Verification:** each determinant positive; contours derive from eigenvectors/eigenvalues rather than arbitrary ellipse rotations; tied shapes congruent, diagonal axis-aligned, spherical circles of radii sqrt1.5 and1. The displayed matrix-label correspondence is checked independently of visual attractiveness.

## F5 — Candidate scores and narrow fitted variance on real data

**Placement:** §7 following the16-candidate table, before the reserved test result. **Question:** how can model-selection criteria disagree, and which component behavior helps explain that? Data are the supplied corrected Iris sepal values; offline provenance and exact split IDs are in the packet. Parameters/scores come from `iris.candidates` in `checked-results.json`; no browser refit required for this static comparison.

First panel: K1–4 against validation mean log-density, four covariance-family series with shape/dash keys; y-range[−3.35,−2.60]. Mark the maximum full K2; full K3 is visibly almost tied. Second panel: K against training BIC with y-range[450,550], mark minimum full K4. Include every K1 baseline, no hand-drawn elbow. If labels overlap, label ends or use an adjacent compact legend. Exact table remains available. Use a filled selection marker for the declared validation winner and an open diagnostic marker for training-BIC winner; explain both symbols.

Third panel: actual training observations in raw centimeters, x=sepal length, y=sepal width, fixed observation IDs. Overlay full-K4 component0 covariance geometry transformed back using train means/scales. This component has weight0.148476513, standardized mean(0.245438856,−0.190069573), covariance diagonal(1.313807522,0.0001) and numerically negligible cross term. Its raw mean width is3.0cm; raw SD width≈0.004209cm, narrower than the recorded0.1cm resolution. Draw a magnified y-strip around3.0cm plus full data context, clearly marking their different y-scales. Do not artificially widen the actual ellipse to make it look less narrow. Text identifies the width regularization level and rounded measurements as a diagnosis to investigate, not an established causal account of every candidate's failure.

Component label0 is the retained fitted run's identity; naming it “the first species” is forbidden. Species colors remain absent until the separately reported ARI diagnostic; fitting and selection must not visually leak labels into a story of perfect class recovery.

**Text equivalent:** the full candidate table and selected/BIC summaries from §7, plus the narrow covariance's values and actual train IDs. **Mobile:** one panel per row with shared legend repeated only where needed; zoom strip below full scatter. Test scores appear only in the later prose/table, not hidden in a hover available before selection discussion. **Phase-two checks:** all16 points andK1 baselines match retained arrays; score sign/direction labels correct; no log axis on negative score; fixed transformation of means/covariance into cm; selected point and narrow strip perceptible at desktop/phone; accessible data includes all source observations with IDs and split membership.

## F6 — Exact old/new bound chain

**Placement:** §9 just after the inequality proof. **Question:** which equality makes the proof valid? Use the four observations and first EM cycle; preserve old responsibilities throughout the M-step evaluation.

Create a compact value diagram with two parameter columns, old and new. At old parameters, show objective and bound touching at−7.158186977. At new parameters, show the old-q bound−6.799547014 below objective−6.461856301. A horizontal labeled arrow from old bound to new bound says “M-step raises the same bound.” A vertical equality at old says “E-step: q is the posterior.” The gap at new is0.337690713, labeled as the posterior mismatch before the next E-step. The next E-step lifts the bound to the new objective while parameters stay fixed.

Retained numbers: `bound.old.q_function`−8.069044223, entropy0.910857246, old ELBO−7.158186977; `bound.new_with_old_q.q_function`−7.710404260, same entropy, new ELBO−6.799547014. `bound.new_objective`−6.461856301. Bound=Q+H. Use a short vertical value axis with actual numeric positions, or clearly present an equation/inequality strip without pretending it is parameter-space geometry. Never invent a univariate likelihood curve whose horizontal axis has no defined parameter.

**Text equivalent:** “−6.461856301 ≥ −6.799547014 ≥ −7.158186977; the last value equals the old objective because the E-step used its posterior. Q alone differs from the bound by entropy0.910857246.” **Mobile:** stack old state, M-step arrow, new state; repeat exact inequality as text. **Verification:** compute the expected log joint and entropy from the old responsibility matrix, verify the touching equality and ordering; reject a fixture that accidentally recomputes q before evaluating the M-step bound. Check the old equality, new gap and unchanged parameter label are visually readable.

## Phase-two implementation and review handoff

Intended ownership: existing topic JSX body; topic-owned `gmm-models.js`, `gmm-examples.js` and `GmmLabs.jsx`/`GmmFigures.jsx` as needed; dedicated blueprint under this stable ID; offline data asset under `public/learn-assets/gmm/iris.csv` with attribution. These are proposed destinations, not files created by this content phase. Keep teaching-program source and outputs separate; preserve the compact algorithm on the page and put browser input validation in the model layer.

Before declaring implementation ready, verify mathematical correspondence, native displayed output, every discrete fixture family and recorded prediction, input-key invalidation, mobile/keyboard/text behavior and selected informative screenshots. Required screenshots include: I1 distant decisive point plus log-density; I2 incorrect prediction feedback, identical-start null and active floor; I3 reversed correlation andrho0 null; F3 spike zoom; F5 whole candidate comparison and narrow-variance strip; F6 old/new bound equality/gap. All are currently deferred. No screenshot or browser success is inferred from the specifications.

The future author runs the full learning-experience checklist independently of numerical checks and rereads the lesson without operating controls: F1 hidden selection, F2 weighted update, F3 collapse, the initial I3 covariance geometry, F4 model restrictions, F5 real-data comparison and F6 touching bound must each be intelligible inline. A new feature or revised dataset may improve the lesson, but record its reason and refresh its source-bound checks.
