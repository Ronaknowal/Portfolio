# Vision Transformers: visual and investigation contracts

Content-first revision, 13 September 2026. No figures, browser models or labs have been implemented for this revision. These specifications accompany the complete lesson; mathematical inputs and bounded author calculations are retained in this directory. Do not replace the image-specific representations below with a succession of generic text-output panels.

## Shared contract

Use image coordinates `(row,column)` starting at zero, row-major patch IDs, channels-last labels on displayed feature grids and explicit tensor layout in code. Preserve pixel/patch/window/token distinctions. Intensity is observed input; probability, attention, signed feature coordinates and cosine similarity each need separate legends and scales. No color alone encodes a correct answer, allowed edge or negative value.

The lesson must remain understandable without interacting. Every inline figure has a meaningful initial view, caption and equivalent text/table. Labs supplement the core explanation; they do not conceal a first definition or the only account of a mechanism. Keep worked examples ungraded and distinct from the fresh lab defaults and independent exercises. The answer-bearing JSON is author evidence, not copy to show before an attempt.

For each investigation: initialize the prediction to unset. Show the exact current input and proposed edit, collect the specified prediction, and require Commit before computing/revealing the scored consequence. Bind the attempt to a fingerprint of input, weights/configuration, source ID, operation and prediction target. Editing any of those invalidates a previous prediction and restores the unset state. A stale answer must never be graded against a new input. During an exploratory ungraded mode, make its ungraded status explicit and keep it separate from a fresh attempt. Revealed feedback reports actual values and mechanism, not merely “correct.”

Provide numeric controls and keyboard equivalents for every drag/paint operation. Apply on explicit button or bounded debounced commit; do not run inference on every pointermove. Step, Back and Reset restore coherent input, intermediate state, prediction and history. Buttons have accessible names and visible focus. No automatic animation; offer single steps and respect reduced motion. Plot tooltips also open on focus/tap. Invalid finite ranges, shapes, zero directions and temperatures receive local explanatory feedback.

On narrow screens, preserve an entire small image/window grid without page-wide overflow. Stack connected diagrams vertically with repeated short labels and numbered links. A large matrix uses one selected row/column and a compact overview plus a data table, rather than shrinking labels below readable size. Preserve sign, axis labels and exact value inspection at 320–390 CSS pixels. These are phase-two checks, not claims of already rendered success.

Only lesson-specific modules and compact inputs should load for this topic. The manuscript and initial static figures do not require full model weights. Derive the tiny plain-ViT state and the three illustrated examples from the author packet into a separate lazy-loaded asset for I2. Do not eagerly ship the CNN, distilled model, all split IDs, original dataset or training scripts to the reader. They remain downloadable reproducibility material. Nothing trains in the browser. Dispose workers/listeners on unmount; cancel stale computations; retain a text/data fallback and an actionable retry on load failure.

## Inline figures

### F1 — Image, patch, scalar order, shared projection (§1)

Question: how can a spatial image become vectors without treating color channels as additional image positions? Misconceptions: patch token equals object, flatten means discard ordering inside the patch, each patch has its own independent projection.

Initial state: the exact constructed 4×4 one-channel image 0–15 from the lesson, split 2×2. Show the four patches with coordinates and IDs 0–3, each unfolding to its ordered row. A single shared two-output projection sits to the right. Small colored/symbol-coded connectors map the selected patch's scalars into its two dot products. Show outputs `[5.5,−2]`, `[9.5,−2]`, `[21.5,−2]`, `[25.5,−2]`. Next to this small example, a compact dimension inset states 224×224 RGB →14×14 patches →196×768 raw patch scalars →196×d embeddings. No false depiction that a patch vector is a word or whole object.

Units: arbitrary constructed intensity units in the hand image; embedding coordinates are learned-feature units. Geometry encodes image location, connector color only tracks identity. Always print the signs and ordered scalar row. Text equivalent spells out the four patch vectors and two formulas. Mobile stacks image→selected patch→projection→all output rows; never overlays arrows on numeric labels.

Verification: `vision-mechanisms.py` independently computes unfold+linear and tied Conv2d; recorded exact equality. Phase two verifies every connector/index and output, including RGB channel-major ordering in the explanation. No external image asset required.

### F2 — CLS reads tokens, head reads features (§2)

Question: how does one image-level output arise from spatial token interactions? Show a patch grid with 16 cells and a distinct nonspatial CLS tile. Beneath it, show 17 sequence positions, positional additions, two pre-LN residual blocks, final LN, CLS extraction and 10-class head. A selected attention head can be illustrated from the saved first image, but the initial schematic must work without loading weights.

Separate three relationships: attention weights over 17 tokens, feature vector width 32, class probabilities over 10 labels. If displaying actual saved last-head 0 CLS weights, include the self/CLS mass in a separate box and the 16 patch masses at their true coordinates. Row sum is one including CLS; do not renormalize the patch-only image invisibly. Head 0/layer 2 is labeled as one internal computation, not a saliency truth. The final head scores use the saved state and actual dataset label.

The permutation inset shows two physical patch identities exchanging sequence slots: content-only swap leaves position badges in place; joint swap carries position badges too. This is an explanatory ungraded before/after on source 1. Use arrows and identities, not just “same/different” text. Text equivalent gives the proof assumptions and computed max differences 7.8398 versus 8.35e−7. Mobile stacks the two cases, preserving token identity badges. Verification compares metadata/saved arrays with `author-results.json`; reference NumPy attention agreement is 2.995e−7 and logits 3.921e−6, not exact float32 identity.

### F3 — Resolution and operation budget (§3)

Question: what changes when image resolution changes, and which work is quadratic? Top:14×14 and 24×24 position grids with one CLS prefix outside each; show resizing the spatial grid while retaining prefix identity. Add a 2×3→3×4 rectangular miniature with two prefixes from `mechanism-results.json.position_resize`. Bicubic intermediate vectors may overshoot; do not visualize them as necessarily convex averages. An unchanged grid produces exact identity.

Bottom: calculated MAC curves for P=16, C=3, d=768, depth=12, one CLS, 1,000 classes and FFN ratio 4. Use `count_macs` equations. Plot projections, FFN and pair MACs per block on linear axes (image side in pixels; billions of MACs) over the four table points 224/384/512/1024. Connecting segments are guides between calculated points, not timing observations. Initially use 224–512 for readable separation, with 1024 available as a clearly disclosed extended range; include the exact table so 4.10% is interpretable. Whole-model counts live in the table, not mixed into a per-block plot. Zero baseline, legend, point values and the exclusion/counted-unit caption remain visible. Never title the axis FLOPs without the two-operations convention, or milliseconds.

Mobile uses a modest wide-enough plot with shorter tick labels and the selectable per-operation table. Phase two verifies formula sums, point ordering, linear transform, small 224 pair value visibility and no fabricated runtime interpretation. No interactive grading is required; this is an explanatory calculated figure and values table.

### F4 — Swin two-hop geometry, boundary mask and hierarchy (§4)

Initial main trace: constructed 4×4 grid A–P, window 2, shift 1, selected F=(1,1). Three spatial panels show first-window edges, shifted-window edges, and composed original dependencies. Print 4 direct second-layer states and 16 original contributors separately. For a uniform mean, all 16 coefficients at F are 1/16. Alternate corner A exposes only 4 originals. A connecting-layer stack can use same row/column colors plus different line styles for direct and two-hop paths.

Boundary inset uses 6×6,window 2,shift 1; show the rolled window with original `(5,5),(5,0),(0,5),(0,0)`. Stripe the forbidden pair cells. Value 16 at(5,5), zeros elsewhere: wrapped mean at(0,0)=4, correctly masked result 0. Explain this is an ungraded counterexample, not the fresh scored task below. Padding is a separate hatched cell type, not confused with wrapped valid tokens. A relative-offset inset shows 2×2 query/key coordinates mapping to one of 9 per-head bias entries. The value is added to the score, before softmax.

Hierarchy inset: token-grid shapes 56×56×96→28×28×192→14×14×384→7×7×768. Zoom one merge into TL,BL,TR,BR concatenation→LN 4d→linear 4d→2d. These are shapes and operations, not artificial photographs showing increased image detail. Raw input patch side 4 is distinct from window side measured in tokens.

Text equivalent enumerates source coordinates and coefficients; mobile one layer per panel with a persistent selected query badge. Verify composed Boolean reach and mean matrices against `region_matrix` and `mean_matrix`. The readable rolled implementation was independently checked against a full-grid per-query reference on actual 5×7×4 features with nonzero relative bias and padding, max error 5.552e−17; its relative-bias gradient was nonzero. Phase two still verifies renderer hit targets and matrix/model parity.

### F5 — Supervised teacher, two student tokens (§5)

Show a frozen CNN teacher above and a trainable patch student below. An image is the only model input. Dataset label goes to CLS loss; teacher argmax goes to DIST loss. Shared student sequence contains CLS, DIST and patches; two outgoing heads. Distillation signal arrows terminate at losses, never at token input values. Freeze/gradient labels distinguish teacher from student. Training-only teacher branch fades/collapses in an inference view while both student heads remain.

A compact numerical fusion inset uses the lesson's explicitly constructed three-class logits `[-3,−2,0]` and `[2,3,0]`. Display two calculation orders side by side, mean logits choosing class 1 versus mean probabilities choosing class 2. Do not label one convention as a bug. The caption identifies paper versus released-code convention and our choice. The probability strip has 0–1 scale; logits use signed numbers, not “confidence.” Mobile stacks the two routes with their operation order written. Verify the counterexample from `mechanism-results.json.fusion_counterexample`; no generated accuracy curve is needed.

### F6 — Cross-view self-distillation and two distinct state updates (§6)

Use a small constructed image layout and two visible crop rectangles to explain view identity, without pretending they are actual DINO-training observations. Larger teacher views and student global/local views occupy separate rows; cross-view arrows exclude identical views. The worked prototype example is represented by three-column strips, center subtraction, temperature division and softmax. Student targets detach; the teacher has no gradient arrow.

Beside the distributions, show two independent stored states: teacher parameters updated by EMA and prototype center updated by batch teacher logits. Use the exact parameter 2→2.2 example and clearly name center as a vector of prototype-logit averages. Do not imply an EMA parameter update equals an average of features. The worked single-row probabilities/loss/gradient come from `mechanism-results.json.dino_worked`. The all-equal collapse inset uses three uniform prototypes, loss log3, zero student gradient; no animation claims the recipe always escapes it.

DINOv2 extension is a masked-patch correspondence map: same image coordinates, student masked patch, teacher visible patch, distinct patch head; image/CLS head appears separately. Source-based architecture schematic, not a measured benchmark. A small row/column balancing diagram can show Sinkhorn's alternating normalization on assignments; if numerical iterations are added in phase two, independently check declared marginals/iterations rather than inventing a convergence trace.

Mobile separates probability calculation from temporal state updates, with shared prototype labels. Text alternative states each directed comparison and actual gradient. Phase two checks all cross-view exclusions, no teacher gradient claim inferred just from a drawn icon, and value/temperature invariance against the retained calculations.

### F7 — Relational geometry and registers (§6 deeper)

Main ungraded Gram example: two orthogonal unit feature arrows, their 2×2 identity Gram matrix, a common 90° rotation preserving it, and collapse to equal arrows changing the squared difference to 2. Display “constructed geometry; not measured DINOv3 output.” Show feature-coordinate and pairwise-similarity views together, so the invariant is visible.

Register inset: CLS, four explicitly nonspatial example register tokens and spatial patches enter attention; only the patch segment becomes a 2D output map. The number 4 is an example architecture choice. It must not become a universal instruction or implication that adding random register tokens to any frozen checkpoint reproduces a trained register model. DINOv3's current 2D RoPE bridge refers back to the earlier positional lesson, with no invented pretrained map.

Text includes dot products and Gram matrices; rotate by an explicit button, never continuously by default. Mobile matrix cells remain at readable size. The fresh graded Gram task is I5 below, with four vectors, distinct from this worked pair and Exercise 9's three vectors.

### F8 — Observed study and feature atlas (§§7–8)

Use exact `study-results.json` observations. Top model comparison shows integer correct/denominator and cross-entropy on separate panels. The selected step is based on validation CE; selection marker must not appear on a test curve. Validation trace contains 13 observations, step 0 then 50…600. Do not imply intermediate evaluations or repeated seed estimates. Raw-pixel/frozen-feature readouts are a separate table because their input representations and head fitting differ. No error bars representing absent repeated runs.

Feature atlas uses the first three official test rows with source IDs and labels. Include the wrong classifier case, source 3; preserve all ten probabilities. Next to the grayscale image, show 4×4 patch vectors projected into the *same training-fitted* PCA basis. Fixed channel ranges from `author-results.json.pca.training_min/max`; clipping outside those ranges, if needed, is display-only and disclosed. The three individual variance ratios are 22.716%,16.764%,10.496%; sum 49.976%, subject to rounding. Color is a projection coordinate, not an object class. Print numeric coordinates on selected patch and offer single-PC sequential/diverging views for accessibility. PCA basis/sign alignment is frozen for all examples.

The selected source 1 patch 5 matches against source 2 patches with cosines in the retained array, top 10/2/14. Show linking arrows between actual patch locations and values 0.3620/0.2958/0.2895. Caption states the weak semantic inference: maximum among candidates is not proof of correspondence. This is a measured saved-feature comparison. Avoid adding inferred segmentation contours.

Caption ties data to UCI CC BY 4.0 and its source transformations, separate writer pools and small declared protocol. On phone, stack source→representation→scores, preserving source ID and model. Exact tables supplement graphs. Phase two checks plotted arrays, scales, source images, hidden head-fusion differences and consistency between changed input and downstream views.

## Investigations

### I1 — Build and change a patch token (§1)

Purpose: predict which information a projection keeps and loses. This is an image-patch arithmetic workspace, not a dropdown quiz over named rules.

Fresh initial state: 2×2 patch `[[2,1],[4,0]]`, fixed example projection rows `[1,0,0,1]` and `[0,1,−1,0]`, bias `[.5,1]`. Proposed initial edit changes lower-left 4→2. Show the input/projection and blank output cells. The learner commits which output coordinates change and can optionally enter their predicted new values. Baseline and changed result are hidden until commit. Author-only results are `[2.5,−2]→[2.5,0]`.

Entity controls: select/edit any pixel within 0…16, edit each projection coefficient within −4…4 and biases −8…8, choose one proposed pixel operation or directly construct a replacement patch. Operation preview shows what is held fixed. Reveal shows signed multiply/add contributions next to their source pixels. Prediction grading uses a stated 1e−8 tolerance for entered numbers and exact coordinate-change classification beyond that tolerance. User-created variations are calculated from their actual input, not graded against a preset answer.

Checked null: replacing the baseline by `[[1,1],[4,1]]` changes two pixels but preserves both features. Do not present the null recipe in the initial graded screen. After feedback, ask the learner to find their own changed patch with equal output. Optional hint points to a preserved sum before revealing an example. This differs from Exercise 2 and the solved 0–15 image.

State: one edit per attempt, backward trace restores the preceding dot-product stage; full reset restores fresh input and clears prediction. Reject nonfinite/out-of-range values. Arithmetic remains tiny; main thread is sufficient. Mobile patch and editable coefficient rows stack with same numbered scalar labels. Text mode exposes the entire operation as editable numeric rows. Phase two verifies changed and null fixtures against `author-results.json.fresh_patch`, alternative coefficient edits, keyboard/pointer parity and invalidated predictions.

### I2 — An image edit, a position swap and a frozen model (§§2,7–8)

Purpose: distinguish measured input changes, bookkeeping permutations and internal maps. Use the actual plain 18,218-parameter model, not hand-drawn class probabilities or a constant mock output.

Fresh graded specimen: test source 3, actual digit 2. Initial proposed edit is `(2,3):0.8125→0.1875`. Ask the learner to commit whether the model's probability for the annotated class 2 increases, decreases or stays the same; allow a rationale. The initial baseline probability may be inspected, but changed probability, logits, attention and maps are hidden until commitment. Author-only result is stored in `author-results.json.fresh_image_change`; no fit is needed. The first two specimens are ungraded worked/inspection alternatives, not recycled scored defaults.

Entity controls: editable 8×8 pixels on the original 0–16 grid (display normalized values too), select two of 16 actual patch locations for a swap, and choose whether their position vectors move too. An explicit “Predict this edit” locks the input/proposed operation before reveal. Real edit recomputes patch vectors, attention, all final features, logits and probabilities from the same frozen state. Do not update a heatmap while leaving prediction bars from the old input. Label-only edits do not enter the model.

Checked contrasts: source 3 content-only patch 5↔6 changes logits by up to 6.93287; joint content+position reorder changes by at most 1.1921e−6. The source 3 pixel edit at(2,3) produces a nonzero change as stored. Null cases: unchanged pixel/reset; source 3 reflection at(2,4), whose value is 0.5, leaves input and outputs exactly unchanged; moving positions with content preserves CLS to float tolerance. Explicitly distinguish an unchanged-input null from the nontrivial joint-permutation symmetry.

Prediction comparison uses probability delta tolerance 1e−5 for the three-way classification and reports actual signed delta. Keep softmax probability separate from empirical reliability. The wrong original prediction stays visible after feedback. Source 1's demonstrated 0.997865→0.996650 result does not substitute for the fresh task. After solving, a transfer prompt asks for a distinct editable pixel/patch case and an explanation of whether an observed change tests geometry or only this trained model.

Rendering: main view is image with 2×2 boundaries, edited-cell marker, two selected patch badges, and before/after class-score bars; a details drawer adds chosen head/query attention and common-basis PCA features. Do not show all 17×17×4×2 attention matrices at once. Zero-based coordinates, fixed probability 0–1 axes and explicit head/layer labels. Include complete numeric output accessible on demand. On phone show one before/after image pair or toggled view with a persistent change list, never 64 tiny unlabeled textboxes across the page.

Compute: lazy plain state only; one inference on Apply, with a small worker if measured responsiveness warrants it. Input 17 tokens×32×2 blocks and 3 stored specimens is the bounded default. Cap editable images at 8×8; no arbitrary resolution multiplier or training. Cache only the current source/config state, cancel stale worker results, dispose on exit. Failure leaves image/text lesson available and allows retry. Phase two implements independent browser arithmetic checks against saved baseline/edited/permuted outputs, probability sums, positions, PCA, fresh predictions, reset and keyboard editing. NumPy author checks are evidence inputs, not completed browser tests.

### I3 — Make or break a spatial communication path (§4)

Purpose: expose graph reach, two-layer composition, wrap-around errors and actual score bias. Use a small spatial workspace with visible layers, not just an adjacency table.

Fresh path task: a 6×6 scalar grid, window side 2, first shift 0, second shift 1, source (0,0)=16, all other values 0, destination (2,2). The proposed change toggles the second layer from fixed windows to shifted windows. Prediction asks whether the destination changes and optionally requests its value. Hide the two-layer coefficients and results until commitment. Author-only checked values: fixed windows 0, shifted windows 1. The two-layer coefficient for a source at (5,5) is 0, yielding a nontrivial distant null.

Entity controls: select any source/destination, edit any scalar from −16 to 16, choose window side 2 or 3 on the 6×6 grid, change the valid shift from 0 to M−1, and step exactly one or two layers. The arithmetic mode uses normalized averaging with an explicit mask. A separate diagnostic switch turns the cyclic boundary mask off; show a hatched wrapped edge and its original coordinates when activated. An ungraded padding demonstration can use a 5×7 grid; retain its actual validity mask and do not claim every provider uses this padding convention.

Fresh bias task within the same workspace: one shifted-window layer, 6×6 grid, M=2, shift=1, source (1,1)=10, destination (2,2), zero content scores and zero relative bias. The proposed edit adds log 2 to the bias at offset (+1,+1). Commit whether that source's attention weight and the destination value increase, decrease or stay fixed. Author-only results: destination 2.5→4, source weight 1/4→2/5. A distant edit at (5,5) changes nothing at this query. Both actual PyTorch forward settings and the null were executed and retained in `author-results.json.fresh_window_bias`.

For free bias editing, expose the 3×3 offset table and all allowed-pair contributions. Adding a finite bias to a forbidden pair must not unmask it. Content/query scores remain fixed at zero in this mode, and the caption identifies that simplification. This is not a trained Swin classifier. The standalone program supplies actual learned-projection machinery; there is no need to add another full neural network to this browser lab.

State: step 0 is the raw grid, step 1 contains first-layer states, and step 2 contains second-layer states. Back restores states and highlights consistent with the selected step. Changing input or configuration clears derived steps and the prediction. Reset the fresh path and fresh bias tasks separately. Reject invalid windows/shifts, nonfinite values and queries outside the valid grid. For a custom all-masked configuration, show a local undefined-row explanation; do not run softmax on all negative infinities or substitute plausible zeros.

Feedback names the intermediate cells carrying the path and the exact mask blocking it. Transfer asks the learner to locate an unreached source after two layers, then open a valid path. The full text alternative lists allowed keys at each step and composed coefficients. On mobile, maintain the 6×6 grid at a usable touch size, stack layers and show one selected query's weights. Matrix operations with at most 42 valid tokens run on Apply. Phase two independently checks reach, normalization, roll reversal, padding and the checked value/bias contrasts, plus prediction binding when moving queries.

### I4 — Tell the student which way to move (§6)

Fresh teacher logits `[.2,−.1,.4]`, center `[.1,.1,0]`, student logits `[0,.2,−.2]`, student temperature .5 and teacher temperature .25. The proposed edit changes teacher prototype 2's logit from .4 to 0. Predict whether student prototype 0 should move up, down or stay fixed under gradient descent after that edit. The worked three-prototype example and Exercise 6's two-prototype problem remain separate.

Before commitment, show original and proposed raw scores, center and temperatures, with blank target probabilities and gradients. After commitment, show both calculated paths, actual student probabilities and the gradient arrow. The checked baseline gradient for student 0 is +.1997045, so descent moves it down; the changed gradient is −.3819665, so descent moves it up. Changed loss 1.2261424 versus baseline 1.4125549 uses different teacher targets: do not interpret it as improvement from training. Student parameters were held fixed.

Entity controls edit every teacher/student logit from −4 to 4, every center entry from −4 to 4, and temperatures from .05 to 2. The learner selects the prototype to predict. A global-offset control adds a constant to all teacher logits. The checked +3 null preserves the teacher target; a single-prototype edit changes it. A separate ungraded collapsed case uses all zeros and teacher/student uniform distributions with zero gradient. Introduce it after the fresh attempt, not prefilled as that attempt's answer.

An optional single student step uses a chosen learning rate from 0 to .1 and shows `s_new = s − eta * gradient`, then recalculates loss against the same frozen target. EMA and center updates remain a separate ungraded view. Do not change teacher and student simultaneously and attribute every movement to one gradient. If this view is added, use the exact parameter 2→2.2 fixture and explicit old/new states. Scores, probabilities and parameter states have distinct labels.

Feedback compares the committed direction to the actual gradient sign and explains the target–student probability difference. An optional closed hint precedes the closed answer. A zero learning rate is another obvious null, not the only control. Back/Reset restores parameters and clears the attempt; edits invalidate predictions. These vector operations need no worker. On mobile, stack readable prototype columns and fields; the text alternative gives the full formula and evaluated numbers. Phase two compares the hand formula, finite-difference check and author results; verify the teacher stop-gradient claim in the code example and ensure no state updates before Apply.

### I5 — Preserve feature relationships (§6 deeper)

The fresh graded task has four 2D unit features `[[1,0],[0,1],[−1,0],[0,−1]]`, associated with four constructed patch locations. The proposed edit moves the fourth feature to `[1,0]`. Predict which Gram row/column changes and whether the squared Frobenius difference stays zero or increases. Original and changed Gram values remain hidden until commitment; feature arrows and inputs are visible.

Entity controls edit one feature's angle, select features by patch identity and apply a common rotation. If allowing coordinate edits instead, normalize explicitly and show normalized vectors; zero vectors are invalid cosine directions. The author-only changed loss is 6 under the unnormalized sum reduction; a common 90° rotation has loss 0. Both were checked in `author-results.json.fresh_gram`. Worked pair geometry and Exercise 9's three-vector fixture must not become the fresh graded default.

Show arrows and matrix together. Rows/columns follow patch IDs; color encodes cosine from −1 to 1 using a diverging legend and printed values. Diagonal self-similarity is 1. Highlight changes along both the affected row and column. A feature-coordinate change can preserve the entire Gram matrix under a common rotation; feedback distinguishes that nontrivial null from unchanged inputs. No semantic segmentation claim follows from the constructed geometry.

Transfer asks the learner to construct another common orthogonal transform and compare it with changing one feature. New input invalidates prediction and step state. Provide button/numeric angle alternatives to dragging; motion is optional and respects reduced motion. On mobile, put arrows above the matrix and the selected pair's dot product below it. The tiny calculation runs on Apply. Phase two checks norms, Gram symmetry, diagonal, loss normalization and the fresh contrast/null, together with keyboard operation and a copyable matrix.

## Disclosure, resources and implementation checklist

All nine independent manuscript exercises have initially closed hints/solutions. Their prompts remain outside disclosure. Do not precompute and display lab answers in accessible names or hidden live-region text before commitment: a visually hidden answer still spoils practice. Source/video annotations preserve the actual review extent: Stanford slides read, video identity verified, no claim of recording/transcript review.

Derive initial inline figures from compact topic data. Load I2's weights and model only when needed. Training programs and original data remain optional downloads. Keep author-only answers out of initial teaching text. Preserve source hashes, evidence and offline provenance. Required later work: implement the topic-specific SVG/HTML/canvas representations and models; independent content/model review; execute or correct the optional checkpoint program if it is to be presented as verified code; browser numerical parity; desktop/mobile layout; keyboard/focus/readouts; load failure and retry; exports, lazy entry and payload checks; normal publication/integration. None of those is claimed completed by this packet.
