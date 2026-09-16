# Perceptrons: visual and investigation specifications

Content-only handoff; no UI, model, SVG or publication implemented. Place each object at its marker in lesson.md. Source values: author-calculations.py and calculated-inputs.json, with licensed input in digits-400.csv. These are specification identifiers, not shared component names.

## Shared behavior for the three investigations

Each opens with the example visible but the requested outcome hidden. Prediction inputs are initially unset. Commit binds the actual choices/numeric answer to an immutable snapshot of the editable mathematical inputs and question. Reveal computes from that snapshot, grades against its actual result and explains the contributing terms. A changed input invalidates any prior grade, hides the result and requires a new commitment; never leave “correct” attached to a changed model. A separate read-only worked trace may reveal its known numbers without pretending to assess the learner.

All meaningful entity edits have labeled numeric inputs as well as optional drag controls; presets are conveniences. Invalid/nonfinite values show a nearby error and disable commit, without clamping silently. Apply edits atomically so graph, table and feedback refer to one model. Reset returns original entity values, question and unset prediction. No automatic animation or ongoing timers. Keyboard access, visible focus, sufficient contrast, shapes/labels as well as color, and concise live feedback are required. Use native controls, with text/table equivalents available beside each drawing. On a phone, stack related stages vertically and retain the same labels; never shrink mathematical labels into unreadable diagrams.

Author fixture calculations are complete. Browser models, full numeric implementation tests, rendering, screen-reader operation and phone/desktop readability are deferred to phase2. Until rendered, no claim is made that a plotted contrast is already visibly clear at a particular size.

## A. Weighted evidence, geometry and coefficient scaling

**Question/home:** §1: what does each coefficient do, and why is score not distance? Use a contribution → sum → activation strip coupled to a2D coordinate plane. Space shows input geometry; bar length shows signed product in score units. Color identifies the two features consistently, not “good” and “bad.”

**Initial worked state:** x=(2,−1), w=(1.5,−2), b=−1. Products3,2; score4; norm2.5; distance1.6; threshold output1; sigmoid≈.982014. Mark the closest boundary point and perpendicular distance. Axes are input-coordinate units; no real sensor units are claimed. Draw the clipped line w·x+b=0, not a guessed angle.

**Investigation state:** learner edits x1,x2,w1,w2,b in[−4,4], step.25, and a positive common scale c in[.25,3], step.25. The score may exceed the visual domain; expand or clip with explicit off-screen indications, never clip the value. Compare original and c-scaled coefficients at the same point.

**Prediction:** three independent initially unset choices: hard output same/changed; signed distance same/changed/undefined; sigmoid same/increased/decreased. Show the current action “multiply all weights and bias by c” before commitment. Grade exact hard output and distance status, numerical differences with1e−10 tolerance, and explain the ratio/canceling scale. For c=1, equality is expected for every defined quantity. Negative scale is excluded because it reverses orientation and adds a different question.

**Checked contrasts/nulls:** calculated-inputs.investigationFixtures.geometry: worked→scaled score4→8, distance1.6→1.6, hard1→1, sigmoid.982014→.999665. Tie x=(1,1),w=(1,−1),b0 and its scale2 both score0, distance0, hard0,sigmoid.5. Move x2 .5→1.5 with x1=1 changes score+.5→−.5 and hard1→0. Zero w=(0,0),b1 has no unique boundary and distance=null; show a constant-output plane and textual explanation, never NaN or an artificial line.

**Transfer:** x=(1,1),w=(1,−1),b0; move only x2 to cross the boundary. Prediction asks new hard output0/1; grade from score>0, including tie0. It is an unsolved entity move, not choosing a named answer preset.

**Accessibility:** a table lists coefficients, products, score,norm,distance and outputs. Plane text describes which side and distance, with “distance undefined: zero weight vector” in the degenerate case. Mobile shows contribution strip before plane, never overlays labels on handles.

## B. XOR as a feature transformation and repair task

**Question/home:** §3: what hidden coordinates let the final combination do? Show four input-square points with distinct labels00,01,10,11; two hidden coordinate columns; signed output contributions; four predicted/target rows. Selecting a row is a read-only correspondence aid. No inference that a ramp is a Boolean OR.

**Worked state:** h1=ReLU(x1+x2), h2=ReLU(x1+x2−1), q=h1−2h2. All four exact outputs and hidden values come from calculated-inputs.xor. A separate stepped walkthrough highlights sums, activations, products then output; Prev/Next changes only reveal stage, not the model.

**Investigation initial state:** second output coefficient v=−1, second hidden bias b2=−1; fixed first output coefficient1 and first bias0. All four binary inputs remain in the table; output cells are masked before reveal. Edit v∈[−4,2], b2∈[−3,0], step.25; an optional numeric exact-entry permits −4/3 for the changed problem. Fixed targets(0,1,1,0) are never editable.

**Prediction and grading:** first commit numeric q for selected input11 (absolute tolerance1e−8), then reveal the entire truth-table result. The independent repair asks the learner to edit v, predict whether *all* rows match and commit. Grade each row with1e−8 tolerance and the all-rows decision, showing the offending contribution if incorrect. Do not auto-set v=−2 after the first reveal. b2 edits invalidate every old row/result and prediction.

**Checked fixtures:** b2=−1,v=−1 gives outputs(0,1,1,1); v=−2 repairs(0,1,1,0). b2=−.5,v0 gives(0,1,1,2); v=−4/3 gives(0,1/3,1/3,0), so repairing11 alone breaks01/10. Null: b2=−3 makes h2 zero for all corners; changing v−1→−2 leaves outputs(0,1,1,2). These exact states were calculated and retained; never promise every control edit moves every output.

**Feedback:** identify whether a hidden value is zero, a coefficient multiplies an active feature, or all-row constraints conflict. A final optional text response explains the conflict using equations at sums1 and2; show the manuscript solution only on request. No automatic semantic grade for free prose.

**Rendering:** desktop supports input→hidden→output lanes; mobile vertically ordered lanes and a compact table preserve correspondence. Draw signed contributions with a zero baseline. Rounded screen labels must not drive grading. Do not animate weights as if trained: these are manual representation edits.

## C. Activation values and local sensitivities

**Question/home:** §4: how does value differ from slope, and why do weights still matter? Two graphs share horizontal preactivation z axis; value and slope have independent labeled y axes. Add a small chain strip x → multiply by w → activation with “local sensitivity = w × slope.”

**Functions:** core sigmoid,tanh,ReLU,leakyReLU(.1), with optional GELU exact/tanh,SiLU,ELU(α1),Mish expanded after §7. Value/slope formulas must agree with manuscript and PyTorch2.14 definitions. Exact GELU derivative Φ(z)+zφ(z). SiLU derivative σ(z)+zσ(z)(1−σ(z)). Use stable sigmoid/softplus formulas. At ReLU0 show no ordinary derivative and an explicit framework convention0; never imply calculus uniquely chooses0.

**Inputs:** z∈[−6,6], step.1; scalar weight w∈[−4,4], step.25. z is edited as the current operating point; a compatible bias may hold it fixed while examining another weight. This is local sensitivity, not a complete trained chain, distribution, convergence or gradient-norm experiment.

**Prediction:** initially unset negative / zero / positive below1 / at least1 for s=wφ'(z). Classify using negative<−1e−10, zero absolute≤1e−10, and positive thresholds with explicit tolerance. In addition show signed numeric value after reveal; “at least1” is not absolute magnitude. Contrast−2 and+2 and a changed weight; no default choice.

**Checked fixtures:** ReLU z2,w.5 gives slope1,s.5; leakyReLU same gives identical null. At z−2,w.5,ReLU slope0,s0; leaky slope.1,s.05. Sigmoid z0,w4 gives slope.25,s1. SiLU z−2,w1 gives s≈−.090784. ReLU z0 follows recorded convention0. All in investigationFixtures.localSensitivity. Separate saved sampled curves in activations document values/slopes at−5,−2,−1,0,1,2,5; phase2 must test dense browser calculations against the definitions, not interpolate sparse checkpoints as the true curve.

**Transfer:** make a positive ReLU's sensitivity negative by editing its incoming weight while keeping the operating point positive; or produce the same sensitivity1 using sigmoid at0. Neither is available only as a preset. Text feedback distinguishes activation slope, weight, and their product.

**Plot contract:** calculated curves, not measurements. Include z0, negative region and corresponding values. Avoid smoothing a derivative discontinuity. Exact values/table available; zero and sign encoded by labels/line style as well as color. On phone put value then slope then product; selected z stays synchronized.

## D. Actual image ↔ input tensor

§5 before training code. Select ten actual retained specimens, the first retained source_id for each label, with explicit labels and source IDs. Pixels are row-major pixel_0…pixel_63, displayed8×8 with fixed0–16 grayscale. Show one image's row0 mapped into first8 flattened entries and then into input-feature cells. Source is digits-400.csv, not generated artwork. Caption attribution links data-provenance.md and UCI. Alt text identifies digit/source and supplies numeric8×8 table; no claim of accessibility from grayscale alone. Phone keeps one selected image and a wrapping table. Static image selection is inspection, not a graded investigation.

## E. Measured activation comparison

§5 beside executed results. Source calculated-inputs.digitComparison with seeds1,2,3, six functions and trace samples0,1,10,50,100,200. Data conditions:400 selected UCI images;280train/120validation, split22,64→32→10, default initialization, Adam.01,200fullbatchupdates, CPUfloat32,one thread, versions recorded. No test or time benchmark.

Plot training mean cross-entropy against actual update number; use log y with label and visible update0. Do not place irregular steps at equal spacing without an explicitly categorical axis. A seed selector may reduce clutter, with all seeds accessible; changing it only inspects recorded values. Validation panel uses correct counts out of120; default full0–120 scale with optional clearly labeled116–120 zoom, and a tabular exact-count baseline. Do not use a zoom alone to imply a large practical difference.

Check the saved18final rows against manuscript. Ties118/120 must overlap or be visibly labeled equal, not jittered vertically into fake rankings. Different train losses with tied validation counts are the intended contrast. Optional paired-error view joins validationPredictions to digitSplits.validationSourceIds and CSV actual labels; no model weights were retained, so do not invent neuron saliency, logits, probabilities or inference on editable new images. Disclose interpolating lines connect sampled observations.

Phase2 must confirm late-loss differences are legible on desktop/phone without suppressing the starting point, error counts remain legible, and ties stay honest. A performance benchmark is outside this content.

## F. SwiGLU coordinate correspondence

§7 after its equations. Two learned projections shown as separately labeled matrices; only gate branch passes throughSiLU. Example projected gate(1,−1),value(2,3) →SiLU(.731059,−.268941) →products(1.462117,−.806824). Output projection follows without invented numeric weights. Annotate shapes x[d], gate/value[h], product[h], output[d]. Clearly mark these as *projected values*, not raw x and not probabilities. Numerical values follow sigmoid and multiplication; check with the function checkpoints. On phone stack branches then align coordinate rows. Text table completely states both coordinate calculations.

## G. Ramps make a triangle

§7 after spline example. Calculated continuous curves on x∈[−1,3]: ReLU(x),−2ReLU(x−1),ReLU(x−2),sum. Shared axes and separate small multiples show slope changes at0,1,2. Mark outputs at0,.5,1,1.5,2 as0,.5,1,.5,0. Labels distinguish activation output from negative weighted contribution. Use an adjacent exact table and derivative-by-interval description. This is a static mathematical demonstration; no extra lab is necessary. Phase2 verifies breakpoints and line segment endpoints rather than fitting a smoothing spline.
