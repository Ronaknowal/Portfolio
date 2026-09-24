# RNN, LSTM and GRU visual and investigation specifications

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Carry and edit recurrent state.** Edit sequence entries, recurrent weights, LSTM gates, GRU reset placement and supported pen-trajectory coordinates.
**See the consequence.** Update state trajectories, retained/injected terms, shared-weight credit and exact learned outputs. Step, rewind and reset state explicitly; padding and request boundaries remain visible.
**Decision connection.** Choose what must persist or reset, and diagnose saturation, reset-order differences and accidental cross-sequence leakage.


Content-first packet, 13 September 2026. These are implementation contracts, not existing UI. Preserve the distinctions in `lesson.md`; do not replace the mechanisms below with one generic parameter panel.

## Shared interaction and evidence contract

Each investigation follows the live exploration contract above: current results are visible immediately, valid entity edits update all linked views, and comparisons explain the mechanism. Reset restores the declared inputs and recomputes their result. No prediction or answer-submission state is retained.

Compute every numeric outcome from the same model that draws the figure. Show baseline/current states and their difference immediately; the explanation traces the responsible recurrence term. All controls are exploration inputs, with no predicted-answer field or grade.

Reset restores the canonical inputs and immediately recomputes every dependent result and explanation. Valid input changes recompute every dependent result and explanation; retained baselines keep their original inputs. Values must be bounded and finite; reject malformed/empty numbers without losing the last valid valid input.

No network call or retraining is needed in a browser. Load this topic's inputs/model weights only when this topic and relevant investigation require them. Fitted-model evaluation is bounded to two traces, eight points, H=32 and K=10. Debounce dragging and provide bounded debounced inference; never run nine fits on page load. Retain provenance when converting JSON for phase two.

Every diagram has a caption, shape/axis labels and keyboard-accessible numeric/table equivalent. Color supplements labels, symbols and line styles. No hover-only essential information or automatic trace animation. Stepping is learner-controlled; reduced motion removes transitions. On narrow screens stack mechanism stages and confine scrolling to bounded equation/table regions. Point selection works through keyboard and numeric fields as well as dragging. Announce a concise settled result, not every pointer movement.

## A. What information survives? Explanatory pen-path diagram

**Placement:** section 1 before recurrence. **Hurdle:** confusing order availability with architecture.

Show the same eight-point source as numbered polyline with arrows, a 2×8 ordered coordinate strip, and orderless mean/standard deviation/minimum/maximum statistics for x and y. Use `pen-trajectories.csv`; original coordinates 0–100 are dimensionless. Computation uses x/50−1. A display-unit toggle must not silently change model input.

This is an **ungraded explanatory diagram**, not a live comparison investigation. Learner reverses all points or swaps adjacent positions and directly sees the changed arrays/statistics. The point set stays fixed, while the connecting path and ordered slots change. Null: inverse permutation recovers original arrays; renaming the label leaves arrays fixed. E is the distinct live investigation that applies this representation to a new real specimen and fitted model.

The orderless baseline's reversal probability difference is 1.44e-15. The ordered baseline's can approach one. Explain that information is lost in the representation before training. For a static alternative show paired indexed coordinate tables and invariant statistics. No timestamps on this equal-arclength sequence; retain the completed-trace availability caption.

## B. Three-step recurrence and shared-weight credit

**Placement:** section 2. **Hurdles:** weights versus states, causality, summing reused-parameter gradients.

Draw three state nodes linked in time, three input nodes, one shared-weight legend and additive components at each step. Expand a selected step into input contribution, old-state contribution, bias, preactivation, tanh and state. This constructed scalar state needs no dense heatmap.

**Worked explanatory fixture, not the exploration default:** x=[.4,−.2,.7], h0=0, wx=.8, wh=.6, b=.1, target .3. Source `scalar_credit()` and `mechanics-results.json.scalar_credit`. States 0,.396930432,.176296951,.644467823; half-square loss .0593290405. Backward trace visits 3,2,1; show parameter contributions before sums. Gradients wx=.1412339561, wh=.0819792658, b=.3776608494. SGD step .1 gives loss .0428389099.

**Fresh exploration default:** x=[.3,−.5,.2], h0=.1, wx=.7, wh=.4, b=−.05, target −.1, learning rate .1. Do not display its outputs before live comparison. Author-only answers in `fresh-investigation-fixtures.json.B`: final h=−.0341690579, loss .00216685647, gradients [wx,wh,b]=[.0040084011,−.0148091731,.0986569810], updated loss .00128482372. A real middle-input edit −.5→.1 preserves h1=.1973753202 and changes h2=.0986284434,h3=.1287330910. Changing only target to .4 preserves every forward state but changes loss/gradients; rate0 preserves parameters and loss. These fresh outcomes are checked separately from the manuscript example. B teaches shared-weight credit; F is a separate later investigation about state boundaries and is not another view of this same scalar problem.

Edit the chosen input in [−1,1], weights/bias in [−2,2], h0/target in [−1,1] and learning rate in [0,.2]. Recompute the recurrence, final h, shared-weight gradients and simultaneous update immediately. Step the chain rule to inspect each contribution; show the full current numerical result alongside it.

Checked contrast: x2 −.2→+.2 leaves h1 fixed and gives h2=.4606674922,h3=.7335640858. Null: changing only target leaves all forward states fixed but may change loss/gradients. Zero learning rate leaves weights/forward values fixed. Show honestly if an allowed nonzero update raises loss.

Backward credit and forward state arrows must have different labels and signed values. Full vector/Jacobian details remain in the advanced branch.

## C. LSTM cell accounting and retention interval

**Placement:** section 3. **Hurdles:** gate versus candidate, multiplication versus addition, hidden versus cell, direct versus full gradient.

The worked explanatory diagram uses c_old=.8,f=.9,i=.2,g=−.5,o=.6: retained .72, written −.10, cell .62, hidden .3306768171. Signed old/write bars meet at addition; output passes through tanh and a separate multiplier. Controls are labeled **constructed gate intervention**, distinct from fitted gates in E.

**Fresh exploration default:** c_old=−.4,f=.85,i=.3,g=.6,o=.7. Author-only answers in `fresh-investigation-fixtures.json.C`: retained −.34, written .18, c=−.16,h=−.1110539530. Candidate edit .6→−.2 changes c to −.4; output-only edit .7→.2 keeps c=−.16. Both contrasts and the i=0 candidate-edit null have computed outputs in the fixture. The live comparison displays these matching values immediately.

Edit c_old [−2,2], f/i/o [0,1], g [−1,1]. inspect new cell and hidden sign in the current live view. Endpoints 0/1 are exact intervention limits; finite sigmoid preactivations yield interior values. Contrast: only output gate changes h but not this step's c; later h-dependent gates can change. Null: i=0 makes a candidate edit irrelevant to this step's c/h. Do not suggest f+i=1.

Compute f=R^(1/T) and half-life log(.5)/log(f) live from the target retention and interval. Show both quantities and the resulting decay curve; changing either input recomputes the required gate value without an answer submission.

**Checked reference:** f=.9946296855 and half-life 128.7232441 steps. Show these results for their matching inputs immediately, with the plotted decay and numeric table. Changing retention/interval updates both results; it does not unlock them.

Optional derivative view uses `lstm_full_state`: show the declared full 2×2 Jacobian and the one direct forget entry. Expose exact gate preactivations from the source program. Do not label the whole derivative as f. Advanced view starts collapsed.

## D. GRU reset before or after mixing

**Placement:** section 4. **Hurdle:** a convention can change the function.

The worked explanatory diagram uses h=[1,2], r=[.2,.8], W=[[1,2],[3,4]], hidden bias=[.5,−.5]. Show four labeled matrix edges. Left scales sources before W and then adds bias; right mixes first then scales output and bias. Checked contributions [3.9,6.5] versus [1.1,8.4] are in `gru_reset_placement`.

**Fresh exploration default:** h=[−.5,1.5],r=[.7,.3],W=[[.4,−1.2],[1.1,.5]],bias=[.2,−.1]. Author-only answers in `fresh-investigation-fixtures.json.D`: before=[−.48,−.26], after=[−1.26,.03]. Edit W[0,1] from −1.2 to −.4: before=[−.12,−.26], after=[−.42,.03]. Reset-ones and diagonal/zero-bias equality nulls, plus diagonal/nonzero-bias contrast, are recomputed with these fresh inputs. Compute the comparison from the complete current inputs.

Learner edits a matrix entry [−3,4], reset coordinate [0,1], or hidden bias [−1,1], then predicts equality/difference and one coordinate. Calculate and explain both explicit formulas. Do not call this a speed benchmark or automatically convert weights.

Nulls: r=[1,1] agrees including biases; diagonal W and bias 0 agree for any r. Nonzero bias can distinguish the placements even with diagonal W. Keep final z-retain blend separate; a zero reset need not clear final state. Author checks calculate these nulls.

## E. Actual learned state follows a changed pen trace

**Placement:** section 5. **Hurdles:** learned gates without invented semantics, probability versus decision, perturbation versus natural evaluation.

Use `calculated-inputs.json.saved_models` seed 1 RNN/LSTM/GRU. The first-two-example records are worked explanatory traces only. **Fresh exploration default:** source `pendigits.tes:6`, actual digit1, model GRU, selected probability class1. Its eight coordinates, full original/edited traces and all three model outputs are retained in `fresh-investigation-fixtures.json.E`. Independent `manual_sequence` is the browser reference. Keep all 32 coordinates/10 class outputs available with selectors; default show a few legible coordinates. Do not label an arbitrary feature “loop detector.”

Link three views: numbered editable pen path; position×selected-state plot with computed gates on demand; ten-class probability bars with actual digit and argmax separately labeled. LSTM adds c/retained/written from `saved_model_traces`; GRU adds reset/retain/candidate. Inputs, gates and states belong to the same specimen/model revision.

Display the original and edited top class, the selected class's probability and its numerical difference. Use the supported frozen model and actual edited trajectory. Include ties and unchanged class with changed probability, since these are different effects.

Checked worked-example point3 x edit: add .2 normalized and clip. Sources `pendigits.tes:1` and `pendigits.tes:2`, actual8. Max final probability differences: RNN .00018666/.00051550; LSTM .00059168/.00005334; GRU .00041014/.00146786. Preserve sufficient precision without visually exaggerating tiny changes.

The fresh source6 applies that actual point3 x+.2 edit to a different trace. Author-only outcomes: GRU stays class1 while p(class1) changes by −.0274250589; RNN stays class5 with p(class1) change −.0283209123; LSTM stays class5 with change +.0100169995. Original/edit argmax, all states/probabilities and unchanged-prefix/repeated-input/reverse-twice nulls were calculated without fitting. Show edited paths' model outputs as soon as valid bounded inference completes. Baseline input may be visible; no pre-revealed direction or answer cue.

Whole-development fixed-weight contrasts: seed 1 original RNN/LSTM/GRU correct 268/273/279; reversal 42/39/37; adjacent swap 262/239/259. Never derive aggregate counts from two visible examples or animate training when changing inference inputs.

Nulls: unchanged input/repeated evaluation, cosmetic label rename, reversal twice, unchanged prefix before edit. Independent/native saved traces agree within 3e−6. Phase two checks all nulls and probability normalization. Display completed/preprocessed-trace scope at the caption. Intermediate readouts are not validated live-prefix classifiers.

## F. State boundaries, padding and directions

**Placement:** sections 6–7. **Hurdles:** carry/detach/reset, state ownership, padding not automatically ignored.

Use the five-observation seed19 width3 GRU fixture in `state_and_padding()`. Two arrows cross a boundary: forward state and backward credit. Boundary2 is the worked explanatory fixture only. **Fresh exploration default is boundary3**, saved in `fresh-investigation-fixtures.json.F` with weights and inputs: carry/detach forward errors0; reset last-state difference .0886393591; detach cuts the first three input gradients, while full gradients are retained. Hide these outcomes in the current live view. Move boundary to positions1–4, select carry/detach/reset, predict forward equality and earlier-input gradient, then compute recurrence and bounded derivative. The old boundary2 answers remain explanatory comparisons.

Identity exercise adds two editable streams A/B and batch reorder. Learner assigns states to stream cards. Correct mapping follows identity; incorrect mapping computes a changed state, not just a wrong-answer label. The executed A/B fixture uses the five-point sequence for A and its reversed order for B, each split after point 2. Swapping their prefix states while keeping suffix owners fixed produces maximum hidden-output difference 0.0426162211. Reordering both input rows and their owned states recovers the correctly reordered output exactly. Inputs, prefix states, correct/wrong final states and native weights are retained in state_and_padding.two_streams. Cosmetic label changes keep ownership/data unchanged; reassigning a state changes the computation.

The fresh state-ownership branch also uses boundary3: incorrectly swapped states change suffix outputs by at most .0188559454; correctly reordered owned states give error0. These computed changed/null outputs, both streams and prefix/final states are in the fresh fixture. The boundary2 ownership result above remains explanatory.

Padding's worked fixture has lengths5,3,1 and edit4. **Fresh gated padding default:** lengths5,4,2 (11 valid positions) on the same five-point sequence with the shorter prefixes; actual padded coordinates change from0 to −3.5. Predict valid forward/backward outputs in the current live view. Fresh checked results: valid forward error0; bidirectional valid error .4424245727; packed versus individual final error0; changing only ignored padding leaves packed states exactly unchanged. Full inputs, weights and outputs/errors are in the fresh fixture. Keep old length5/3/1 results as explanatory comparisons, not the practice defaults.

Rows/time cells say valid/padding rather than relying on color. Backward arrows reverse explicitly. Packing omits invalid updates; final state direction×batch×feature is separate from sequence output. Numeric table includes every checked error/shape. Optional target-loss counter computes the actual valid count (fresh default11) and the changed loss scale from dividing by15.

## Implementation handoff and deferred checks

Phase one executed nine CPU fits, independent NumPy/native gates, nonzero-state/bias cases, scalar/backward arithmetic, changed-coordinate inference and native chunk/padding fixtures. `author-check-results.json` records extra practice/contrast/null calculations. No browser labs or diagrams exist yet.

Phase two must create semantic topic-owned files and bounded models; test input-bound results, actual live updates, contrasts/nulls, invalid input, reset/invalidation and stale-result protection if asynchronous. Match figures to actual states/units; prevent mislabeled full-gradient/live-prefix claims. Complete relevant displayed-program/native checks, independent content/learning review, browser/mobile/keyboard/accessibility and lazy-load/integration checks.

Preserve the complete learner program as a readable, progressively disclosed code section and download. Retain setup, imports, source data, split, full protocol and honest results; do not replace it with a placeholder training loop.

## Precise fixture and scope details retained for implementation

**Arithmetic agreement.** Use absolute 1e−6 for floating equality and 3e−6 for saved float32 versus independent float64 parity. A .005 tolerance was used for three-decimal scalar inspection; it is not a learner-answer field. Class identity and probability change are separate readouts.

**Fixed-gate cell path.** Plot c0 f^t with no writes, labeled “fixed-gate direct cell path, not trained-model gradient.” Editable retained fraction R∈[.05,.99], horizon T∈[1,200] integers or f∈[.01,.99999] give f=R^(1/T) and half-life log(.5)/log(f). Axes are steps and dimensionless fraction; an optional log view is labeled. At 100 steps f=.5,sigmoid(1),.99,.999 give 7.89e−31,2.484e−14,.366032,.904792. Half remaining after 100 needs f=.9930924954 and bias 4.9682153688. The editable default R=.65,T=80,c0=1 gives f=.9946296855 and half-life 128.7232441. Permit one positioned write in [−.5,.5], using c_t=f c_prev+write_t; +.15 at step 25 and zero-write recovery are checked fixtures. Show the computed gate and trajectory immediately.


## Code-to-mechanism implementation contract — 22 September 2026

Place the manuscript's new implementation route next to its stated concept section. Preserve the named axes, state and algorithm steps when implementing figures; the complete teaching programs are content inputs, not a hidden replacement for learner-visible code. Render long source only on demand with keyboard-scrollable code and wrapping download labels. Keep constructed comparison fixtures separate from recorded training experiments. No browser execution of Python, pretrained-model download or GPU experiment is required to operate a lab.

The ownership map in design.md identifies which operations are implemented here and which actual sources are reused. Both paths must be findable: the transparent mechanism and the ordinary package/tool route, followed by the changed-input practice. There is no guess-entry, prediction submission or answer-unlock state. Current outputs remain visible while the learner edits meaningful inputs; separate written practice can retain hints and solutions.
