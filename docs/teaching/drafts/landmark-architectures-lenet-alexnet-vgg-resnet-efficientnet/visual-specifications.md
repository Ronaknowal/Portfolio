# Landmark architecture visual and investigation specifications

Content-first specification, 2026-09-13. No React, SVG, browser behavior or grader has been implemented. The manuscript explains every mechanism without requiring a visual to supply missing definitions. Produce code-native grids, shape routes and graphs in phase two; no raster imitation of a network diagram. Stable IDs below identify meaning, not a fixed lab count.

## Evidence and shared interaction contract

Exact budgets, scalar gates, algebraic maps and recorded real fits are different evidence categories. Use architecture-experiments.py and calculated-inputs.json. The 12 fits are actual CPU runs; no ImageNet models or hardware timing were run. All quantities called MACs count Conv2d/Linear multiply-accumulates only. Never derive an accuracy or latency from a parameter count.

All investigations start with prediction **unset**. Bind a submitted prediction to the complete relevant input state (numeric entities, selected model/seed/data row, operation and evidence version), then reveal and grade. Changed inputs make a previous prediction stale; retain it only as labeled history, clear current correctness and require a new prediction. A display-only zoom or expanding the text equivalent does not change the model or invalidate it.

Show before, predicted and computed quantities together. Numeric answers use stated tolerances; categorical answers are assessed against computed or saved outcomes, including equal/no-change. Feedback names the responsible operation and displays enough arithmetic to repair an error. A learner should not have to type exact explanatory prose to pass. No default radio choice, prefilled numeric answer, slider masquerading as an unanswered prediction, or success state from merely pressing Next.

Meaningful edits include classifier dimensions, feature-map cells, signed head weights, channel-context cells and a deployment budget. Step/select controls are useful secondary aids. Invalid/nonfinite values keep the last valid state, present a local error, and disable submission. Reset restores the declared starting entities and clears prediction/feedback. No state persists silently across unrelated datasets.

Use visible labels, explicit units, standard keyboard inputs and non-color encodings. Diagrams have short text explanations and exact-data tables. Announce a concise result once through a polite live region after Check; do not read every intermediate grid cell on each keystroke. Keyboard focus stays on the initiating control. Any drag edit has a numeric/table alternative. Reduced-motion mode uses immediate state changes; all movement has a manual equivalent.

Bounds below are for each pedagogical model. They are not general neural-network limitations. Avoid heavy inference/training in the browser. Load recorded model observations only when their panel opens. Exact small arithmetic can run synchronously, with input commits/debouncing where helpful.

## Inline figures: put structure next to its first use

### architecture-feature-route — §1

Question: how can spatial size shrink while channel count grows? Display 1×8×8 → 12×8×8 → 12×4×4 → 16×2×2 → 16 → 10. These correspond to the later experiment's meaningful boundaries; the body preserves 12×4×4. A selected map is a real grid, with channel stack count written explicitly. Show one image and identify N as an optional batch axis rather than drawing N decorative copies.

Width/height of a block may suggest grid resolution, but must not pretend that perspective volume quantitatively encodes memory. Exact tensor element counts belong in labels/table. The stem and tail use different channel weights; do not reuse a colored “edge channel” as if its identity survives every layer unchanged.

Mobile: one operation/shape per row, with a persistent compact preceding→following shape strip. Desktop: horizontal or wrapped flow. All dimensions must be readable at normal zoom.

### architecture-lenet-fidelity — §2

Use the historical route 1×32×32 → 6×28×28 → 6×14×14 → 16×10×10 → 16×5×5 → 120×1×1 → 84 → 10 penalties. Source: LeCun1998 §II.B. Use visual distinction between convolution, learned subsampling and output penalty. A nearby disclosure explains sparse C3, scaled tanh, learned subsampling and RBF output; do not label a modern AvgPool/Linear replica “exact original.”

A selected C1 kernel has 25 weights plus one bias, six copies gives156 total. One visible patch can show reuse across positions. The historical image is schematic, not a newly measured activation.

### architecture-stride-versus-pool — §3

Question: which grid is being sampled? The first panel uses the explicit 227-input/11-kernel/stride4/pad0 teaching variant:55 outputs. A second panel uses 55-input/3-pool/stride2:27 outputs. Selected source supports and stride arrows carry pixel/feature-position units. Never annotate 55 as the result after pooling. Keep channel count96 constant through that pool. Explain the original-paper/library variant boundary locally.

### architecture-kernel-composition — §3

Two successive3×3 neighborhoods, each stride1, have a5×5 interior support; three have7×7. Show the intermediate nodes and paths, not merely nested boxes with no operation. Annotate18C² versus25C² only when all three widths equal C and biases are omitted. A small alternative panel changes middle width to2C and displays36C², showing the condition behind the saving. Nonlinearity between layers is visible.

### architecture-branch-lanes — §4

Inception: common input fans out to four paths, each finishes with matching spatial size, then concatenates. One route shows1×1 reduction before the expensive spatial layer; the pooling route projects afterward. Output-width accounting is a stacked strip 64+128+32+32=256. A same-size output is necessary for concatenation along channels; height mismatch is shown as a shape error rather than silently resized.

Adjacent add-versus-concatenate fixture: x=[1,2], F=[3,−1]; add=[4,1], concatenate=[1,2,3,−1]. A plus merge aligns corresponding slots; a concatenation bracket appends distinct slots. ResNet bottleneck route 256→64→64→256 includes normalization/activation placements from manuscript; bias-free convolution weights69,632 are separate from complete block totals.

### architecture-inverted-gate-route — §5

Narrow x:C bypasses expansion→depthwise→SE→linear projection back to C. Mark ordinary channel mixing, independent spatial filtering and input-dependent gating with different arrow semantics, not just different box colors. Show the wide hidden representation, then the narrow skip boundary. A shape-changing block has no identity skip unless explicitly supplied.

### architecture-family-side-branches — optional §7

Dense growth:8→11→14→17→20 channels, preserving old channel segments and adding three new ones per layer. This is not residual addition. RegNet: an illustrative quantized width sequence may be drawn only from explicitly calculated parameters; no synthetic performance plot is needed. NFNet's residual scale and adaptive gradient clipping belong to distinct diagrams/locations if visualized: forward computation versus optimizer update.

## Investigations

### architecture-head-budget — §3, before the parameter table

Learning question: which architectural change removes a large classifier budget, and what information does it discard?

Entities/state: final feature shape C×H×W; class count K; intermediate dense width D; head type two-hidden-dense versus GAP+linear. Defaults C512,H7,W7,D4096,K1000. Bounds C1–2048,H/W1–32,D1–4096,K1–1000 integers. Show formulas and count biases. Editable entities are actual dimensions/head operations, not a generic difficulty slider.

Model: dense head parameters (CHW+1)D+(D+1)D+(D+1)K. GAP head (C+1)K. Per-head linear MACs CHWD+D²+DK versus CK. Spatial mean computation excluded and named. Add a separate optional float32 array-storage calculator using16 bytes/parameter for weight+gradient+two Adam moments; it is not peak runtime memory.

Prediction initially asks which part is largest (first dense / second dense / classifier / tie), or asks for the difference after a learner changes K or H. Do not show computed totals until Check. Once checked, retain full exact table; visualization can use proportional bars with a linear zero origin plus a separately labeled log toggle. A logarithmic bar cannot imply additive segment lengths.

Fixtures already calculated: default dense head123,642,856; GAP513,000; first layer102,764,544. Changed C128,H7,W7,K7 with a **direct flattened linear head** in practice1 gives43,911 versus903; this is a distinct available direct-linear head mode, not the two-hidden default. Null: direct-linear head with H=W=1 equals GAP+linear for any C,K. Different class counts alter both totals but do not establish a score difference.

Phase-two parity fixtures also include C1,H1,W1,K1 direct mode (2 parameters each), and a newly edited H×W>1 case. A direct-linear mode must use (CHW+1)K, not the two-hidden formula. Checked outcomes and exact mode state belong in feedback. The dense/GAP information-loss explanation is adjacent; no statement that smaller predicts better.

### architecture-channel-context — §5, immediately after the hand-sized gate

Learning question: can a change far away in one channel change another channel's response?

Entities: two2×2 maps with scalar cells in[0,8], step.25; choose which output channel/cell to inspect. Defaults channelA all2, channelB all1. Summary a,b=their means; h=max(a−b,0); gates sigmoid(h),sigmoid(−h). Output cells are input cells multiplied by their channel gate. This toy has declared fixed learned coefficients; it is not an actual trained digit model.

Prediction: after editing a cell, will the **other channel's gate** increase, decrease or stay equal? The proposed edit is staged before applying; baseline and proposed inputs form the state. Grade against computed difference (equal within1e−12). New edit clears current grade. An optional numeric question asks for the inspected output value with tolerance.0005.

Checked contrasts: means2/1 → gates.7310585786/.2689414214; means2/3 →.5/.5; means0/0 →.5/.5. Mean-preserving cell permutation leaves gates unchanged, but can move output cells. Increasing both means equally leaves the difference/gates unchanged. To expose a nontrivial edit, start a new task with A=[0,4;2,2], B=[1,0;2,1] and let the learner alter one chosen B cell. Compute actual means, rather than offering only the already solved uniform example.

Draw grids→two average values→one ReLU unit→two sigmoid gates→broadcast arrows back to map cells. The global summary path must be visible; no spatial-softmax labels. Mobile separates baseline/proposed maps vertically and keeps gate change beside the active cell editor. Arithmetic never invokes a pretrained model.

### architecture-scaling-budget — §5

Learning question: which resource changes when compute growth is assigned to depth, width or resolution?

State: positive continuous multipliers d,w,r within[1,3], target MAC factor1–16. Clearly labeled **idealized dense-convolution approximation**, not exact EfficientNet. Compute P_factor=dw² and M_factor=dw²r². Entity edits allocate a declared design budget. Coefficient presets α1.2,β1.1,γ1.15 with φ0–4 show their actual product, alongside explicitly separate exact named-model documentation links.

Prediction: for two staged allocations, classify parameter increase/equality/decrease at equal MAC budget and enter the new parameter factor. Grade categorical values from computed ratios; numeric tolerance1e−4. Plot is a budget contour or two aligned bars, never an accuracy forecast.

Checked cases: (d2,w1,r1)→P2,M2; (1,√2,1)→P2,M2; (1,1,√2)→P1,M2. Identity(1,1,1)→1,1. Compoundφ1→P1.452,M1.92027;φ2→P2.108304,M3.6874368729. Doubling resolution alone→P1,M4. Store full-precision inputs and label rounding in display.

Changed allocation beyond solved presets: learner can fix a budget and allocate r1.25, then solve w from remaining budget for d1.5 if feasible. At budget2 the required w≈.92376 is below this investigation's lower bound1: report the infeasible allocation rather than clamping it and pretending the budget still holds. At budget3, w≈1.13137 is feasible. Reject unavailable/out-of-bounds allocations locally. No opaque auto-optimization chooses an “optimal accuracy.”

### architecture-recorded-comparison — §8, before full outcome table

Learning question: does adding the skip necessarily improve the same-size shallow model, and which measured candidates meet a task budget?

Read only saved fits. State seed1/2/3, selected body/pair, plotted metric, parameter budget1–6000, MAC budget1–100000, current trace step from0/1/25/100/200/400. Entity edit is the resource contract and candidate selection; changing a hypothetical width is **not offered** because no such fit exists.

Before revealing a selected pair's final counts, record greater/equal/less prediction. Exact grading compares saved correct counts for the same seed/split. Seed1 residual112 versus plain116; seed2 both115; seed3 residual114 versus plain112. These are checked contrasting and null fixtures. Match the baseline in the plot and include count/120, not only rounded percentages.

For budget prediction, present the four declared cost records and an initially unselected eligible-candidate set. The learner edits limits, then predicts the eligible set. Grade every membership against both inequalities. Budget5000/60000 permits parallel and inverted-gated. A budget4650/76192 permits all four. Changing only seed leaves all costs and eligibility unchanged. Budget2502/41920 permits parallel exactly at the boundary; reducing either relevant bound excludes it. This is a decision exercise on recorded candidates, not a new model search.

Learning plots use observed steps as points with line segments clearly labeled as connections; no dense invented learning curve or population uncertainty band. CE and correct/count have separate axes. Show training and development distinction, with all final training counts280/280. A different initialization can change comparisons; one seed isn't a universal ranking.

### architecture-score-map — §9

Learning question: how do signed spatial contributions add up to one class score, and what changes after a meaningful edit?

Two modes with separate explicit evidence labels:

1. **Editable constructed model**: two2×2 feature maps, two signed class weights, bias. Map entries−8…8 step.25, weights−4…4 step.25, bias−4…4 step.25. Default A=[[1,2],[0,3]],B=[[0,1],[2,1]],w=[2,−1],b.5. Compute each per-channel weighted map, total map, spatial mean+bias and average-then-linear result.
2. **Recorded trained examples**: four seed1 bodies, exactly two observations each from JSON. Source251 digit4 and each model's first recorded mistake (plain97 actual8/pred5; residual104 actual3/pred7; parallel379 actual8/pred5; inverted310 actual8/pred0). Select class0–9. Actual image,16 final2×2 feature maps, all10 CAMs/head weights/biases/logits/probabilities are saved. Arbitrary image edits/reruns and other seeds' maps are not available.

In constructed mode, stage a meaningful cell or signed-weight edit, record increase/decrease/same and optionally numeric next logit, then Apply & Check. Grade against exact calculation with numeric tolerance1e−6. Defaultscore2.5; Bbottom-left2→6 yields1.5; joint180° position permutation preserves2.5; zero weights or zero features leaves bias.5. Practice alternative w=[−1,2],b−.5 gives0; changing A00 1→5 gives−1. These contrast, null and reversed-sign fixtures were calculated.

In trained mode, prediction can ask which of two selected classes has larger meanCAM+bias, using visible map contributions while concealing logits until Check. Correct answer uses corresponding saved logits with tolerant tie handling. Never grade an explanation of “what the image really contains” from the CAM. Editing only a displayed identity label must not change any numbers; selecting a different recorded specimen changes the input record and invalidates prior prediction.

Visual: signed feature tiles → weighted contribution tiles → sum tile → mean+bias → logit; probability is a separate softmax across all10 scores. A zero-anchored diverging legend and numeric cells make negative contributions legible. Raw2×2 CAM and exact numbers remain available beside any image overlay. Enlarged overlay must say interpolated display,2×2 source grid; no pixel-precise explanation or causal label.

Real-mode allowed arithmetic can recompute logits/CAMs from saved **features and head** only; it does not reconstruct the whole backbone. Editing learned feature values switches into a clearly labeled constructed intervention and does not assert a new natural-image prediction. Do not store/render all model assets eagerly.

## Quantitative honesty and visual reading pass

Retire the old fictional post2017 ILSVRC line, mixed-protocol Pareto families, repeated fake depth tiers and arbitrary1–5 recommendation heatmap. A concise historical reported-result table may use only verified primary conditions; the current manuscript needs no global historical scoreboard.

The VGG parameter components and small-model costs are exact counts under explicit conventions. The training/development points are actual recorded float32 runs. CAM identity is mathematical with measured floating-point discrepancies. Gate and scaling plots are constructed calculations. All captions should say which category is present.

Author specification reading checked the first introduction and each transformation: axis correspondence, convolution→pool grid, kernel composition, parallel branch concat, residual add, expanded/narrow route, context summary/broadcast, shape/budget tradeoff, signed score decomposition and actual mistake. This provides several complementary forms rather than one reused text-output box.

## Phase-two completion checks still deferred

- Translate all displayed formulas into code with independent small-fixture parity, including biases, head modes, endpoint eligibility, equal predictions, signed CAM and original float32 tolerances.
- Verify actual prediction persistence/invalidation, error handling, reset and back/step behavior; meaningful cell edits must update every linked representation.
- Inspect informative contrast/null/mistake states at desktop and narrow mobile widths,200% zoom, keyboard-only and reduced motion. Check label collisions, focus, exact-data alternatives, heading structure and links.
- Check asset demand loading and that no12-fit training loop, model download, invented interpolation or unbounded activation tensor runs in the browser.
- Bind evidence to final implementation version, complete relevant application/build/integration checks, and conduct formal phase-two review. None has been run or claimed here.
