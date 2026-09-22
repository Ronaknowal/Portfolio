# ICA visual and investigation specifications

## Current live-exploration contract — 21 September 2026

This dated UX amendment supersedes earlier prediction-entry, grading, commit-to-reveal and prediction-retirement requirements in this document. There is no learner prediction feature, even optional. Historical evidence below records the earlier interface and remains history; it is not the current acceptance contract.

Rotate a population between source families and follow joint geometry, moments and contrasts live; edit source amplitudes and a keep-set and inspect sensor contributions immediately; rescale a source with and without compensating its mixing column. Invalid singular or zero-scale configurations show an error and preserve the last valid output; reset and undo remain available.

Keep separate independent practice, model predictions, scientific validity checks and training/validation/held-out information boundaries. Meaningful valid control changes must reach the visible calculation and topic-specific diagram together. Natural algorithm Step/Back/Run actions remain where they expose a process; they must never require a learner guess. Reset restores a coherent initial state. A graph or number must not silently describe obsolete inputs; invalid inputs show an error and either clear invalid outputs or explicitly retain the last valid result. See [the current migration evidence](../../LIVE-EXPLORATION-CLASSICAL-EARLY.md) for implemented checks and limitations.


Status: complete content-first specifications, 12 September 2026. No visual, lab, website component or formal browser verification is implemented by this packet. Read [the full manuscript](lesson.md) and [the design record](../../ICA-LESSON-DESIGN.md) together with these contracts. IDs below are placement keys, not proposed reader headings.

## Shared mathematical and interaction conventions

The hand model is $x=As$, with $A=[[2,1],[1,2]]$, two arbitrary-unit sources, column-vector equations and row-sample code. Four equiprobable source states in order are A = $(-1,-1)$, B = $(-1,1)$, C = $(1,-1)$, D = $(1,1)$. Their observed points are A = $(-3,-3)$, B = $(-1,1)$, C = $(1,-1)$, D = $(3,3)$. These are an exact constructed distribution. Real recording values have separate measured-data labels and microvolt calibration. Never render constructed points as measurements.

Use text labels and stable point shapes in addition to restrained existing lesson colors. Axes encode actual coordinate values; layout-only arrows do not encode probability, distance or information strength. Use ordinary text for long labels rather than shrinking a large SVG on a phone. Diagrams require a caption and readable exact-value table/description. No interaction is hover-only; focus and selected states need visible cues. All mechanisms can be followed without animation; reduced motion disables interpolated transitions. No statistical conclusion depends on a user noticing motion.

For both investigations, prediction state starts `null`, never preselected. A committed prediction belongs to a serialized active input snapshot and candidate change. Editable staged values are visibly separate from the currently applied values. Changing a prediction-dependent input clears the committed prediction and graded feedback. Run/Apply is unavailable until the staged input is valid and the user explicitly records a prediction. Grading compares the actual model result for that snapshot, not a preset label. Keep old results only as visibly labeled history; they must never appear as feedback on edited inputs. Reset restores the documented initial model and an unset prediction. Undo restores the previous model, but clears any prediction/grade unless its exact state is restored and labeled. Predictions are learning interactions, not assertions of overall mastery.

Suggested topic-owned production destinations, to assess during authorized phase two: `src/learn/components/lesson-labs/IcaFigures.jsx`, `IcaLabs.jsx`, `ica-figures.css`, `ica-labs.css`; pure math under `src/learn/data/ica-models.js`, example records under `ica-examples.js`, real input under `ica-recording-data.js` and a downloadable `public/learn-assets/ica/r01-first20s.csv` with attribution. Keep the CSV/data lazy with the consuming lesson or explicit investigation; do not import it through the hub. The manuscript remains the complete content source until implementation consumes it.

## M1. One sample through two sensors

- **Placement/question:** section 1 immediately after the mixing matrix. Which source contribution does each matrix entry produce, and what does a column mean?
- **Initial state:** $s=(1,-1)^T$. Left nodes Source 1: +1, Source 2: −1. Middle contribution arrows: $2s_1=2$ into sensor 1; $s_2=-1$ into sensor 1; $s_1=1$ into sensor 2; $2s_2=-2$ into sensor 2. Right sum nodes Sensor 1: 1, Sensor 2: −1. Label coefficients on arrows and show the corresponding matrix cells nearby.
- **Encoding:** source identity uses color plus numbered labels; contribution sign remains explicit. Do not vary arrow width, since negative widths and implied power would be ambiguous. Group both contributions entering each actual sum.
- **Second line:** show inverse row 1 $(2,-1)/3$ acting on the two observed values, with operands $2(1)-(-1)=3$, division by 3 giving 1. This introduces why a mixing column and an inverse row have different signs.
- **Accessible text:** “Source 1 contributes 2 and 1 to sensors 1 and 2; source 2 contributes −1 and −2. The sensor totals are 1 and −1. The first inverse row produces $[2\cdot1-(-1)]/3=1$.”
- **Mobile:** stack one sensor sum per row, repeating short source labels; keep the 2×2 matrix above. Do not draw crossing arrows through text.
- **Verification:** exact integer sums; $A^{-1}A=I$; no accidental transpose. Static, so no prediction/controls required.

## D1. Covariance cancellation and conditional support

- **Placement/question:** section 2 after the $u,u^2$ counterexample. How can zero covariance coexist with knowing one variable from the other?
- **Data:** three equiprobable points $(-1,1),(0,0),(1,1)$. Axes $u$ and $v=u^2$; range u[-1.3,1.3], v[-.15,1.3]. Draw separate point markers, with a thin dashed parabola explicitly labeled “relation” if helpful; it is not sampled continuous data.
- **Visible default contrast:** left panel all three points; right panel the event $v=0$, retaining only $0,0$. Beneath show products $uv=-1,0,1$, mean 0; $P(u=0,v=0)=1/3$ versus $P(u=0)P(v=0)=1/9$. This is immediately visible, not hidden behind hover.
- **Alternative accessible form:** a three-row probability/product table and the conditional sentence. Stacked mobile panels preserve the probability comparison and all labels.
- **Verification:** moment and probability sums independently hand-derived in manuscript. No unit variance claim for these variables. This differs from the binary source mixture and must retain its own $u,v$ names.

## W1. Whitening’s unfinished job

- **Placement/question:** section 3 beside exact PCA whitening. What changed at each transformation, and what dependence remains?
- **States:** source points above; observed points above; whitened diamond A = $(-\sqrt2,0)$, B = $(0,-\sqrt2)$, C = $(0,\sqrt2)$, D = $(\sqrt2,0)$; recovery by $Q=[[1,1],[1,-1]]/\sqrt2$ restores the source points. Calculate them from the shared fixture, do not type an independent drawing.
- **Transforms/labels:** Source $s$ → $A$ → Observation $x$ → $K=[[1,1]/(3\sqrt2),[1,-1]/\sqrt2]$ → Whitened $z$ → $Q$ → Recovered $s$. Label the final matrix “orthogonal change (includes reflection)”. A compact PCA-score note states variances $9,1$, then divide by $3,1$ to whiten.
- **Geometry:** each panel has equal x/y scale, real ticks and its own explicitly labeled extent. Source/recovery range±1.5; observed range±3.5; whitened range±1.8. No arrows claiming original sensor-space unmixing rows are orthogonal. Correspondence is the A–D marker identity, not similar scale across panels.
- **Covariance/data:** show $\Sigma_x=[[5,4],[4,5]]$ and $E[zz^T]=I$. Next to the whitened panel state $P(z_1=0)=P(z_2=0)=1/2$, but $P(z_1=0,z_2=0)=0$; hollow mark at the absent joint origin with label “no probability mass here”. This mismatch must be visible on the reading path.
- **Mobile:** one panel per row or two columns at comfortable widths, with transformations above the receiving panel; repeat point identities rather than shrinking. Max panel width about 280–320 CSS px; ordinary text captions at reader font size.
- **Verification:** exact $n=4$ expectations use denominator 4. Values/transform conventions checked by author probe. Phase two verifies marker coordinates, equal axis scales, label collision and the absent-origin distinction on desktop and at 390 px. Do not turn this into an arbitrary number of lab steps.

## R1. Rotation investigation

**Question:** What can a distribution-sensitive statistic distinguish after all directions have equal variance? What happens when the non-Gaussian distinction disappears?

**Placement:** section 4 after the fourth-moment derivation, Gaussian case and zero-kurtosis counterexample; revisit it in practice 2. Introduce projections, unit covariance and the requested statistic before controls. This investigation uses the **source-aligned whitened coordinate frame** $s$; the section-3 diamond is the same binary distribution after a 45° change of axes and a sign convention. State that frame change above the angle control so “30°” has a fixed meaning.

**Active model:** two independent unit-variance sources with the same chosen marginal family. Current projection $y_1=\cos\theta\,s_1+\sin\theta\,s_2$, companion $y_2=-\sin\theta\,s_1+\cos\theta\,s_2$. Default family binary, active angle45°, no staged angle and no prediction. The output covariance is exactly identity in the population model for every angle.

**Meaningful input:** numeric angle entry in degrees [0,180], step0.5 but permit any finite decimal in range. Optional accessible range control mirrors the same staged angle. The learner chooses an angle not resolved beside a preset. Family selector: binary ±1, standardized Laplace, standard Gaussian. Family edits are staged with the angle. Reset restores binary45° and unset prediction; optional suggestions offer starting questions, not answer-named buttons.

**Recorded prediction:** “Compared with the current projection, the new projection’s absolute excess kurtosis will be [smaller / the same / larger].” Native radio group begins unset. Commit stores `{oldFamily,oldAngle,newFamily,newAngle,prediction}`. Prefer hold family fixed per run; when a new family is applied, reset old/new angles to the same currently selected angle and ask a fresh question for the next angle experiment, avoiding a changed-family comparison the prompt does not describe. Keyboard activation required; grade numeric absolute-kurtosis difference using tolerance1e−10. Display actual old/new values and their signed difference alongside the prediction result.

**Exact formulas/data:** $\kappa_s=-2,3,0$ for the three families. At angle θ, $\kappa(y_1)=\kappa_s(\cos^4\theta+\sin^4\theta)$, with equal value for $y_2$. Population variance is 1. Binary joint transformed support consists of the four input states at probability 1/4 each; show those exact nodes. Laplace standardized density is $p(s)=2^{-1/2}e^{-\sqrt2|s|}$; the two-variable density is $p(s_1,s_2)=\tfrac12e^{-\sqrt2(|s_1|+|s_2|)}$. Gaussian joint density is $(2\pi)^{-1}e^{-(s_1^2+s_2^2)/2}$. For the continuous families, use analytic equal-density contours: Laplace level sets $|s_1|+|s_2|=1,2,3$ and Gaussian radii $1,2,3$, transformed by the same angle. Label them density contours, not observations or probability boundaries. No random point clouds are needed.

**Visible consequence:** a joint-support/contour panel beside a curve of $|\kappa|$ against θ over[0,180]. Mark the current and proposed angles only after prediction is committed/applied; do not pre-reveal the proposed value. Use family-specific y extent [0,2.2] binary, [0,3.3] Laplace, and [0,1] Gaussian with zero line and explicit “flat at zero”. Include both pure-source endpoints and45° midpoint. Keep unit-covariance badge/table beside the changing distribution. Curves are analytic predictions, not measured timings or fit performance.

**Feedback:** state what changed geometrically and numerically: “Your direction changed the fourth moment while variance stayed 1”; for Gaussian, “Every angle has the same joint Gaussian distribution; this contrast supplies no source direction.” A neutral wrong-prediction result explains the formula and preserves an opportunity to try a new angle. Do not label a single fourth moment as an independence test. The zero-kurtosis non-Gaussian counterexample remains in the adjacent prose, with a linked exact-value table if needed.

**Checked contrasts:** author-calculations evaluates angles0°,30°,45°,90° for every family. Binary kurtoses -2,-1.25,-1,-2; Laplace3,1.875,1.5,3; Gaussian0,0,0,0. Binary45→30 gives greater absolute kurtosis;0→45 smaller;0→90 equal. Laplace45→30 likewise greater; Gaussian is the null at all angles. Independent algebra uses $a^4+b^4=1-2a^2b^2$, not the plotted sample list. Phase two must test arbitrary decimal angles, sign/period symmetry, exact equal-angle input, invalid values, prediction invalidation and every family. No claim of full parameter-range rendered verification yet.

**Transfer:** reproduce practice2 at30° and propose an angle with the same value without using the same coordinate (e.g.60°); explain symmetry. Learner can act on any allowed angle. This is a population-distribution investigation, not a sample-based ICA solver.

**Accessibility/mobile:** use angle field with degree suffix and a textual current/proposed state. Put plot after controls then exact numerical results, all in one reading group. The population curve needs axis labels large enough to read at narrow width; use a separate non-scaled legend. Use keyboard-operable radios/Apply; announce short feedback through polite status without moving focus. No lab data should exceed a few hundred contour/curve vertices; no asynchronous worker required for this tiny calculation.

## F1. One fixed-point operation, exposed

- **Placement:** section5 after the numerical update. Static worked trace, not an investigation; the necessary result is available while reading.
- **Data/steps:** use whitened diamond A–D. The four y values are $-.8\sqrt2,-.6\sqrt2,.6\sqrt2,.8\sqrt2$. Show columns z, y, $y^3$, $z y^3$, and $3y^2$ in a compact exact companion table. Vector mean $E[z y^3]=(1.024,.432)$; derivative mean 3; correction $3w=(2.4,1.8)$; raw vector $(-1.376,-1.368)$; unit-normalized vector approximately $(-.709165300,-.705042246)$. Sign-aligned vector $(.709165300,.705042246)$ is an equivalent displayed direction, not a different update.
- **Representation:** a short operation pipeline plus unit-circle inset with old $w=(.8,.6)$, actual new $w$ in the negative quadrant, and dashed sign-equivalent positive direction. Draw vector geometry from numbers. Label the sign flip explicitly; otherwise the picture will appear to contradict the equation.
- **Text equivalent:** narrate averages → subtract → normalize → compare direction up to sign. Include $1-|w_{new}^Tw|$ as the convergence concept, without claiming convergence after this single update.
- **Mobile:** one operation per row with vector values in aligned columns; keep circle around240 px rather than full-width SVG. The four-row table may horizontally scroll in its own accessible region, while all essential means remain readable without scrolling.
- **Checks:** author values from independent polynomial averages and NumPy probe agree. Verify source IDs, sign, denominator4, vector length and eigenbasis convention in actual implementation. The programmed NumPy example uses `eigh`’s ascending eigenvalue order, so its internal whitened coordinates can differ by order/sign from this expressly chosen whitener; explain the equivalent convention if shown together.

## P1. Fit, development selection, held-out evaluation

- **Placement/question:** section6 before real code. Which information can influence which decision?
- **Inputs/provenance:** `r01-first20s.csv`, metadata in [data provenance](data-provenance.md). Original sample indexi corresponds to t=i/1000 seconds. Half-open intervals: train[0,12), development[12,16), test[16,20). No hidden record/window search.
- **Diagram:** aligned lanes for four abdominal input channels and one direct reference. First lane has a fit boundary at12 seconds and arrows only from abdominal train to fitted PCA/ICA. Development values plus reference select a coordinate independently in each representation. Test values plus reference evaluate those fixed selections. A lock marker at selection denotes frozen index, accompanied by text.
- **Accessible alternative:** a table with stage, time, available arrays, output and forbidden dependency; explain that the decomposition is blind while coordinate labeling uses development reference.
- **Mobile:** three stacked stage rows retaining a small0–20s ruler, with reference arrows entering only the last two. Do not crisscross several model arrows; write separate PCA and ICA names in the same fit container and raw baseline outside it.
- **Checks:** using test reference or reference as a fifth input is a failing implementation. Fitted means and whitening come from[0,12), no test refit. Provider’s offline filtering is documented; diagram must not imply an online evaluation. All labels derived from declared split lengths, not hardcoded incompatible time bounds.

## E1. The real outcome and waveform agreement

- **Placement/question:** immediately after section6 result interpretation. What does the diagnostic compare, and how does it relate to the signals?
- **Data:** measurements/calibration from CSV; fitted model/settings and selection exactly as code. Numeric result table: raw chosen3/dev.201203/test.119806; PCA chosen4/dev.184580/test.450178; ICA chosen2/dev.169804/test.343966. Values are author calculations, not benchmark claims. Reproduce at phase two and update all linked views if a relevant environment change changes rounding or component identity.
- **Visual:** three labeled horizontal marks/bars on one0–1 absolute-correlation axis, including zero. Put development as a hollow point and test as a solid point per row; legend also says which is which. The jump between the intervals is part of the evidence. Do not truncate axis to imply a dramatic universal advantage.
- **Trace portion:** selected reference, raw, PCA, ICA scores for test time16–18 seconds, each in its own aligned row. For display only, center and divide each shown trace by its own standard deviation in that displayed interval; label “display z-score, not fitted model normalization”. Sign-align candidate traces using their **development** reference-correlation sign, frozen before test; no test-driven sign flip. Keep the reference in its recorded polarity. This makes polarity disagreement visible. No fitted parameter is altered by display scaling. Values for tooltips/table retain original sensor µV or IC arbitrary units; do not label ICs µV.
- **Decimation:** raw computation uses all20,000 rows. Plot at most1,000 min/max envelope columns per trace or a deterministic peak-preserving representation of the selected2,000 samples; never silently downsample the diagnostic. Explain plot sampling in caption. A small exact table can expose a selectable interval of20 rows without rendering20,000 DOM nodes.
- **Reading/accessibility:** result table is primary exact data; legend and brief purpose of correlation precede plot. Mobile stacks result and trace panels; common time labels0.5s apart, no overlapping row titles. No interactive fit is required. Do not add a fake “run ICA” button that merely swaps canned outputs.
- **Required phase-two evidence:** check outputs and component selection against native scikit-learn; plot coordinates against actual values; informative desktop/narrow captures of3-way comparison and representative trace segment. Neither waveform plot nor correlations establish clinical interpretation. No screenshots have been taken in content phase.

## C1. Component contribution investigation

**Question:** What exactly does dropping or rescaling a component change in the observed sensors?

**Placement:** section7 after the exact subtraction example. Uses the known two-source hand model; it is not an interface for choosing clinical exclusions.

**Initial state:** A=[[2,1],[1,2]], source amplitudes $(1,-1)$, both components kept, scaling factor 1. Active observed $x=(1,-1)$. Prediction unset. Existing example is visible as a worked state; meaningful new input is the learner’s amplitude pair in [-4,4] with finite decimals and a keep-set (both, source 1 only, source 2 only, neither). At least one changed amplitude is required for the first independent task, so it cannot be completed merely by selecting the already solved example.

**First investigation mode:** staged amplitudes and keep-set. Prompt “What will sensor1’s reconstructed amplitude be?” Numeric prediction field begins empty, unit arbitrary amplitude; optionally sensor2 in a second field for transfer. Commit snapshots the edited amplitudes and keep-set, then Reveal computes $x=A s$, contributions $a_1s_1,a_2s_2$, reconstruction $\sum_{j\in keep}a_js_j$, and removed difference $x-x_{kept}$. Compare with a tolerance of1e−9 absolute and explain the sum. No preset outcome text beside the controls.

**Scale-ambiguity mode:** hold the active physical contributions fixed; user enters c in [-4,-.25]∪[.25,4] and selects a component to rescale. Replace $s_j$ by $cs_j$ and column $a_j$ by $a_j/c$. Prediction radio “Observed sensors change/stay the same” begins unset for this mode. Result exposes changed source amplitude and mixing cells with unchanged products. A negative c reveals the sign ambiguity. c=0 is invalid with a concise inline reason; no division occurs. Keep-set continues to act on the same component identity. A compact comparison with “scale source only” is a separate staged operation so the learner can investigate why compensating both factors matters.

**Representation:** use per-sensor signed contribution stacks around zero and explicit addition equations; support negative quantities by direction/sign, not negative bar widths. Show the two mixing columns as vertical sensor patterns next to the source amplitudes. Reconstructedsensor positions/values update from the shared state. Avoid a generic many-control dashboard. Present the scale mode as a focused second question after completing an exclusion trial.

**Checked fixtures and nulls:** manuscript initial $(1,-1)$, keep 1 → $(2,1)$, removed $(-1,-2)$. Independent changed sources $(2,-3)$: observed $(1,-4)$, keep 1 → $(4,2)$, keep 2 → $(-3,-6)$, keep both → $(1,-4)$, keep none → $(0,0)$. Sources $(2,0)$ make removing source 2 a null operation; all-zero input remains zero for all keep-sets. c=2 and c=−2 with compensated columns preserve each of these products; source-only c=2 changes the nonzero contribution. These are exact arithmetic fixtures. Phase-two model check must run every keep-set, both modes, zero source, negative/positive c branches and invalid zero; no actual browser state is claimed verified yet.

**Feedback/transfer:** show predicted and actual amplitudes plus the retained and removed summands. Ask the learner to choose source amplitudes where dropping a component makes sensor1 larger; negative source2 is one solution. Then connect to practice5’s wanted-activity subtraction without pretending the simulation identifies a real artifact.

**State/accessibility/mobile:** default predictionempty, staged/active inputs separate, Apply commits one transaction and returns causal feedback. Undo/reset clear prediction as shared contract. Labels use source/sensor names and units; tab order input→keep-set→prediction→Reveal→result. Announce only concise feedback. Stack per-sensor equations on mobile; signs and source identities remain readable at200% zoom. Operations are a handful of scalars, no worker/timer/unbounded loop.

## Phase-two closure plan

Implementation must consume this packet, retain offline provenance/attribution, reproduce displayed code and results, then verify actual models and graphs. Author calculations below support content claims only. Required independent risks include: four-state probability identities; whitening convention and rank handling; deflation inside iterations; same-kurtosis contrast; Gaussian and zero-kurtosis non-Gaussian distinction; sign/permutation/scale matching; component reconstruction and retained-subspace semantics; real fit/reference boundary; stale-prediction grading; column-vs-row spatial meaning; accurate trace units and plot envelope.

Bounds: rotation angle[0,180] finite, at most361 curve samples per family and3 contours; contribution amplitudes[-4,4], nonzero bounded scales; native fixed-point200 iterations for the exact two-dimensional teaching fixture; real native example1,000 iterations/four components/12,000 fit samples. The browser should not run arbitrary Python, fetch a whole ECG database or fit an unbounded real-data model during render. Use retained native results for static teaching views or an explicitly bounded model with independent agreement checks if implementing further interaction.

Actual browser review still needs desktop and narrow layouts, ordinary reading without interactions, keyboard/focus, invalid-input recovery, reduced motion, informative post-prediction states, and figures at intended rendered sizes. Perform the learning-experience checklist independently of correctness. Root/integration owner handles blueprint/manifest/navigation changes, production build and curriculum conservation only during authorized phase two.
