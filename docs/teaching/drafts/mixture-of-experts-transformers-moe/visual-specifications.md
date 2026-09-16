# MoE visual and investigation specifications

Research/write phase, 13 September 2026. These are implementation specifications, not rendered or browser evidence. The manuscript supplies complete explanations and placements. The retained offline programs, models and results establish arithmetic and learned inputs. Phase two creates the website figures/labs, independent review and integration.

## Interaction and representation contract

Keep expert indices, token identities and selected-slot identities persistent. A token-to-expert arrow means execution; an attention arrow means information exchange across positions. Return arrows terminate in sums. Distinguish logits, affinities, gate weights, assignment counts, class probabilities, parameters, MACs and bytes. Expert indices do not imply semantic professions. The real image model uses bidirectional attention.

Worked examples are ungraded and may show answers. Each investigation starts with fresh inputs, a visible baseline and staged change, an unset prediction and hidden results. The sequence is: edit an actual entity → commit a prediction → Run and compare → explained feedback. Store an immutable snapshot of every answer-relevant input with that prediction. Changing scores, outputs, k, normalization, capacity, order, model, pixels, temperature, expert intervention or resource dimensions invalidates the prediction and hides stale results. Reset restores the fresh fixture and clears prediction/result/progress. Pointer movement stages a proposal rather than revealing hidden outputs. Tooltips and accessible labels must not reveal answers early.

Feedback gives the actual delta, why it follows and how it compares with the prediction. “Unchanged” is a valid answer. Numeric tolerances: hand arithmetic 1e-8, explicitly rounded answers 1e-5; start portable float32 model comparison at 2e-5 and adjust only with numerical evidence. Unchanged class labels cannot certify unchanged probabilities or vectors.

Keyboard tables accompany draggable paths and pixels. Controls have labels, ranges and visible focus. Patterns and indices supplement color. SVG/HTML diagrams have text/table equivalents. Narrow layouts stack stages and display selected-patch details; do not shrink a whole network into illegibility. Reduced motion retains before/after states. All lab code and models load on demand; deduplicate requests, cancel stale computations and do no offscreen training/animation. Never import all author history into the initial lesson bundle.

## Inline figures

| Figure and placement | Exact representation and data | Interpretation and layout |
| --- | --- | --- |
| F1, §1 | Three token lanes through attention/residual and MoE/residual. Four experts, two schematic routes per token. | Distinguish attention edges from assignment edges. Expert names are indices. Mobile stacks the two stages with persistent token labels. |
| F2, §2 | routing.worked_selected and worked_full in calculated-inputs.json: masses [4,2,1], expert outputs [2,0],[0,3],[-1,1], chosen 0/1. Weights 2/3,1/3 versus 4/7,2/7; outputs [4/3,1] versus [8/7,6/7]. | Aligned score bars, selected paths and 2D contribution arrows. Preserve expert 2's denominator arrow only in the full-softmax calculation. A table supplies exact signed components. |
| F3, §3 | Backward paths for selected-normalized top-2, normalized top-1 and full-softmax top-1. Worked scalar sum-output gradient [-2/9,2/9,0]. | Expert-parameter and router-score gradients are different paths. Caption states local derivatives with fixed selection; selected derivatives can vanish for particular outputs/loss directions. |
| F4, §4 | Five schematic tokens carry token/slot IDs through gather, expert processing and weighted scatter-add. | Two contributions returning to one token meet at addition. Table columns are token, slot, expert, gate and output. Arrow length does not imply latency. |
| F5, §4 | Worked T=4,N=3,k=2, routes [[0,1],[0,1],[0,2],[0,2]], capacity 2. Requested loads [4,2,2], retained [2,2,2]. | Rejected token-2/3 routes to expert 0 retain their expert-2 routes. Show 2 rejected assignments, 2 affected tokens, 0 fully dropped tokens. Residual bypass stays visible. |
| F6, §5 | auxiliary fixture: [.51,.49] repeated three times, [.001,.999]; q=[.75,.25], P=[.38275,.61725], loss .88275. Companion peaked p=[.99,.01], logit shift 0→3, z-loss 0→9. | Probability rows, top-1 arrows and separate count/mean histograms. Uniform reference 1 is not a minimum/percentage scale. Shared offset changes z-loss while preserving probabilities. |
| F7, §6 | costs entries (8,128,2) and (16,64,4), d=64,T=32,b=2. All expert matrices versus selected branches; expert storage 393216 bytes/.375 MiB. | Separate parameters, MACs, bytes and shared/router costs. Fine graining keeps expert work/storage but changes router/assignment costs. No synthetic timing chart. |
| F8, §7 | Constructed rows [.7,.2,.1],[.6,.3,.1],[.2,.2,.6],[.1,.8,.1]. Row top-1 chooses [0,0,2,1]; column top-1 chooses token [0,3,2] for experts [0,1,2], leaving token 1 unassigned. Companion selection_bias fixture s=[.8,.7,.6,.1], b=[0,0,.3,0], selected 2/0, weights 3/7,4/7. | Count both row and column assignments. Use a documented tie rule. Bias affects the selection strip; original affinities feed the weighting strip. |
| F9, §8 | Observed training-file row 1762, validation digit 0, selected model moe_001_17. The 8×8 count grid becomes sixteen 2×2 patches, attention, expert paths, pooling and class scores. | Actual clean p0=.9517156482, route counts [0,1,15,16]. Lower-half-zero edit gives prediction 9, p0=.00866820384, counts [15,7,4,6]. Original identity belongs to observed pixels; edits are hypothetical. |
| F10, §8 | All twelve study-results.json runs, clean/stress metrics, selected steps and route histograms. | Keep denominator 300 and all seeds. Counts sum to 9600 assignments per assessment run. Class-conditioned counts use their own denominators. Optional curves label minibatch loss before update versus validation loss after update. |
| F11, §9 | F(x) copied into three identical experts, two selected weights .25/.75, equal original/replacement output. Companion drops the .75 route without reweighting. | Exact equality assumes retained weights sum to one. Not a measured upcycling experiment. Mobile uses formulas and an explicit contribution table. |

F9 additionally exposes a selected patch's actual 16-component normalized expert input, four full router probabilities, two selected weights and 16-component combined vector. F10 may use paired clean/stress points on a 0–1 accuracy axis; do not connect different seeds as a time series. Small probability changes need exact deltas or a labelled zoom, not exaggerated arrows. Every figure remains understandable without operating a lab.

## I1 — Route and recombine new vectors

**Fresh problem.** routing.fresh_selected uses masses [1,3,2,4], meaning their logarithms are logits, outputs [[2,-1],[1,2],[-1,4],[3,0]], and k=2. The staged change raises expert 2's mass to 5. Ask the selected identities and the direction/value of both output coordinates before revealing. Baseline chooses 3/1 with weights 4/7,3/7 and output [15/7,6/7]. Changed input chooses 2/3 with weights 5/9,4/9 and output [7/9,20/9]. These differ from the solved three-expert example.

**Actions and computation.** Edit actual logits in [-6,6] or equivalent positive masses, output components in [-8,8], k and normalization. Allow 2–6 experts with two output coordinates. Switching units preserves represented values. A deterministic teaching tie rule selects lower expert ID first; compare PyTorch away from ties unless its selected tie behavior is explicitly reproduced. Display selection margin and flag a boundary rather than assuming the output varies continuously.

**Contrasts and nulls.** A common logit shift preserves both normalizers' outputs. Changing an unselected score without crossing a boundary is a selected-normalized null. Constant outputs [2,-1] give that output under selected normalization. Normalized top-1 has weight 1, while retained full-softmax top-1 has score sensitivity. In the fresh top-2 case, both selected outputs happen to have coordinate sum 3, so the derivative of their summed coordinates is zero. Use a chosen coordinate or explicitly declared loss direction in derivative mode; never promise every selected score has nonzero gradient.

**View and feedback.** Score bars feed path lanes and signed 2D contribution arrows. Show gained/lost paths and exact products after prediction. A mobile table can display one coordinate at a time. Phase two checks both normalizers, local gradients, ties, constant/common-shift nulls, dense reference agreement and snapshot invalidation. No training is needed.

## I2 — Fill capacity slots and return token contributions

**Fresh problem.** Use five tokens, four experts, k=2 and routes [[0,1],[0,2],[0,3],[1,2],[0,1]]. Every gate initially has weight .5; expert i returns scalar i+1. Capacity begins at 2, with proposed change to 3. Ask rejected assignment count, fully dropped token IDs and token 4's output.

At capacity 2, three assignments are rejected, token 4 loses both routes and outputs are [1.5,2,2,2.5,0]. At capacity 3, only token 4→expert 0 is rejected, no token loses all routes and outputs are [1.5,2,2.5,2.5,1]. Capacity 4 rejects nothing.

**Actions.** Edit actual route IDs, capacity 1–6, dispatch order, expert scalars or the two normalized weights. A token cannot select the same expert twice. Persist token identity when reordering. Reset capacity occupancy before each run. At capacity 2, order [4,0,1,2,3] still rejects three assignments but loses no whole token; outputs in original token order are [1.5,1.5,2,1.5,1.5].

The default drops assignments without reweighting. An optional survivor-renormalization toggle is labelled as a changed model and computes its own answers; if all routes drop, output remains zero. Dropless mode accepts every assignment and is order invariant after returning rows to original IDs. Repeating the same run is a null.

**View.** Editable token cards move into actual slot bins; rejected paths stay visible. Count requested/retained assignments, affected tokens and fully dropped tokens separately. The residual path remains outside the expert bins. Keyboard order buttons and a numerical return table provide full access. Run animation begins only after prediction; reduced motion shows static retained/rejected tables.

**Phase-two checks.** Match saved capacity 2/3/4 and reordered fixtures, dropless order invariance, additive return, disabled duplicate IDs, reset and prediction binding.

## I3 — Separate balance from numerical scale and task quality

**Fresh problem.** Four three-expert probability rows are [.6,.3,.1], [.5,.4,.1], [.2,.7,.1], [.1,.2,.7]. Use top-1 and the q=count/(Tk), P=mean(full-softmax) convention. Baseline q=[.5,.25,.25], P=[.35,.4,.25], balance loss 1.0125. Stage changing the first row to [.2,.7,.1]; q becomes [.25,.5,.25], P becomes [.25,.5,.25], loss 1.125. Ask which counts/means change and whether this establishes an accuracy change. It does not determine task accuracy.

**Actions.** Edit row logits, k=1/2, grouping and common logit offset. Always recompute normalized rows, selected counts and N·dot(q,P). There is no capacity dropping in this investigation. Distinguish per-group aggregation from a global product. A second fresh numerical-scale question starts p=[.2,.5,.3], logits log(p), and stages common shift +2: probabilities/routes/balance stay fixed while z-loss changes 0→4.

**Controls.** Row permutation within one unchanged group preserves aggregate counts/means. A shared logit shift preserves probabilities but not generally z-loss. Peaked and uniform normalized probabilities both have zero z-loss when their logits are log(p). The worked .88275 example is a separate ungraded counterexample.

**View and feedback.** Probability rows and discrete arrows lead to count and mean histograms, then an explicit formula. Keep task quality separate from these objectives; do not invent an accuracy meter. Optional gradient inspection uses a declared loss direction and fixed-selection qualification. Phase two checks formulas, scope, nulls, finite input validation and snapshot feedback.

## I4 — Hold one resource budget fixed

**Fresh problem.** Set d=48,m=80,N=10,k=2,T=24 and four bytes/component, with all assignments remote. One expert has 11,520 parameters; total expert parameters 115,200; router parameters 480; active expert MACs/token 23,040; forward dispatch plus return payload 18,432 bytes. Propose N=20,m=40,k=4. Expert storage/work are unchanged, router parameters become 960 and payload becomes 36,864 bytes.

**Actions and formulas.** Edit matrix widths, expert count, k, routing-group token count, bytes/component and remote assignment fraction r. Calculate 3dm, 3Ndm, 3kdm, dN and 2Tkdbr. The traffic expression is a simple payload accounting model, excluding metadata, backward traffic and shared operations. Storage uses all N experts. An optional shared-expert row adds its own stored and active work.

Bounds: widths 8–4096, N=2–256, k≤N, T=1–4096, bytes 1/2/4/8, r∈[0,1]. Compute arithmetic only, never allocate actual tensors at those dimensions. Exact values remain available beside abbreviated/IEC units.

**Contrasts.** Increasing N at fixed k,m leaves active expert work unchanged but raises router cost. Changing T affects batch traffic, not parameters. Setting r=0 eliminates this remote payload without changing expert computation. Coarse/fine matching does not match dispatch cost.

**View and checks.** Matrix outlines and selected branches precede a resource table with units. No seconds or hardware-speed graph. Phase two checks default/fine fixtures, shared additions, integer bounds, safe arithmetic, units and committed prediction state.

## I5 — Edit a real image and inspect learned paths

**Fresh problem.** Use moe_001_17 and validation digit 7, original optdigits.tra row 2030; trained_fixtures["7"] binds the reference. Show observed pixels and a staged lower-four-rows-zero edit. Hide final class scores, changed routes and outcomes until the learner predicts which route identities, mixture values and class label can change. An optional predicted class is initially unset.

Author results: clean prediction 7, p7=.96999669075, counts [1,12,16,3]. Lower-half-zero predicts 3, p7=.00776563585, counts [16,11,4,1]. This is the actual fixture outcome, not a guarantee for arbitrary edits. The upper-left 2×2 patch is already zero: zeroing it is an exact input null. Doubling temperature preserves routes/counts and class 7 but gives p7=.96980702877. Disabling expert 0 retains class 7 and gives p7=.97009193897.

**Actions.** Edit any of 64 integer counts 0–16 through a pointer/keyboard grid; choose a bounded curated validation image or enter a custom image; set temperature .5–3; disable one expert or none; inspect patch 0–15, head 0/1 and expert 0–3. Disabling an expert zeroes its contribution without changing routes or renormalizing survivors. A selected patch's input vector, full router probabilities, selected weights and combined output are computed from the current edited image. Never substitute a preset animation for inference.

The observed label belongs to the original source. An edited image may no longer depict that class, and a custom image has no observed label. Reset restores fresh digit 7 and clears predictions. Hover before reveal shows only input/position data. After revealing routes, the learner may choose a used or unused expert and commit a new intervention prediction. The worked digit-0 tab is ungraded: expert 0 is unused there, so disabling it is an exact null.

**Exact model.** Divide counts by 16. Reshape to B×4×2×4×2, permute into patch-row, patch-column, local-row, local-column order, then reshape to B×16×4. Apply biased 4→16 projection and learned 16×16 position vectors. LayerNorm uses epsilon 1e-5, population variance and learned weight/bias. A bias-free 16→48 QKV projection forms two heads of width 8; scores divide by √8 and softmax over all 16 keys. Concatenate weighted values, apply bias-free output projection and residual. A second LayerNorm feeds the bias-free 16→4 router, divided by temperature. Top-2 selected scores are softmaxed. Bias-free SwiGLU experts have width 16, SiLU activation x/(1+exp(-x)), weighted gather/scatter-add and FFN residual. Final LayerNorm, mean over sixteen patch vectors and biased 16→10 classifier produce logits. Class softmax is for inspection. Weights are in fitted-models.json. No padding, dropout, capacity dropping, gradients or optimizer runs in the browser.

**Representation.** Link the 8×8 pixel grid to patch lanes, selected expert contributions and class bars. Attention weights and router weights have separate views. Show an actual selected patch's weighted vector calculation, with a selected component and full table available. Small probability differences use numerical deltas or an explicitly labelled zoom. Editing one patch can change other patch representations through bidirectional attention; do not simply recolor the edited patch.

**Performance and verification.** Lazy-load one selected 4,762-parameter model and curated images, not all twelve models and training histories. Run on explicit action, cancel stale work and use a worker if measured need warrants it. Check model logits/probabilities, all patch selections/weights/vectors, patch orientation, unused-expert and zero-patch nulls, temperature contrast, lower-half edit and disabled-expert policy against all saved variants. Single-versus-batch companion agreement permits documented float32 roundoff. Verify keyboard editing, readable mobile views, load/error recovery, pre-reveal privacy and prediction invalidation independently in phase two.

## Continuation

Implement these figures and five investigations where the manuscript places them. The count follows this topic's mechanisms. A later author may improve arrangements while preserving the mathematical operators, actual inputs, learner agency and observed-versus-constructed distinction. All practice hints/solutions remain initially closed.

Phase two separately checks numerical portability, reproduction downloads, rendered perceptibility, accessibility and interactions, resolves independent content/learning review, and integrates publication. Author calculations do not certify these deferred stages. Retain the pending packet until implementation consumes it.

