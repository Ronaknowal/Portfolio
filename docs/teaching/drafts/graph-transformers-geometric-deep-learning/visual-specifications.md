# Graph transformers and geometry: visual contracts

Research/write only. Implement this complete manuscript later; none of the components described here is already built. Use calculated-inputs.json for the actual34-node experiment and geometry-results.json for constructed exact examples. Code/data/provenance meanings are binding; visual styles may adapt to the hurdle. Keep the topology, basis and physical-coordinate illustrations visibly distinct.

## Shared investigation behavior

Each scored activity starts with an unset prediction tied to the current graph, features, transformation, model and question. Require recording it before Check/Reveal, then compare with the current actual result and explain the responsible step. No default correct answer, correctness awarded for pressing Next or fabricated success on unchanged inputs. Mathematical edits invalidate an earlier prediction. Layout-only dragging does not change the mathematical input. Reset restores the full initial state and unset prediction; hints and solutions start closed.

Provide numeric tables/keyboard edge editing as alternatives to dragging. IDs, arrowheads, operation labels and line styles carry meaning independently of color. At narrow widths, stack input/process/output without shrinking text; matrices may scroll locally with their axes retained. Use a concise live announcement on explicit checking, not on every pointer movement. Animation supplements static states; respect reduced motion. Optional trained evidence loads on panel expansion, bounded to34 nodes/two16-wide blocks; exact constructed graphs≤8 nodes, points≤6. No browser training, unbounded loops or external data requests.

## 1. Identity, graph layout and geometry — first use in §1

Three synchronized views use the same named nodes: node/edge table; topology-only drawing; a separate physical-coordinate example. Relabel permutes H and both A axes; a layout drag changes only drawing positions; a physical edit changes X and derived distances. Do not animate social-network layout as a molecular conformation. A task asks which stored mathematical inputs changed, with actual node/table edits and feedback on the selected object. Contrast a rewire with a layout move. This small view establishes semantics before attention heatmaps appear.

## 2. Structural attention route — §2

Inline path0—1—2 with sender values1/2/4; selected receiver0. Separate content score, bias, allowed mask, exponential weight, denominator and weighted value. Default qk scores0, bias−distance·ln2, weights4/7,2/7,1/7, output12/7. Values are constructed, not model weights. A hard one-hop mask renormalizes to2/3,1/3,0 and gives4/3. Remote value4→11 changes global output to19/7 and leaves one-hop output unchanged.

Learner edits actual edges, allowed senders and scalar values, then predicts numerical output or which node can influence it. Recompute shortest paths and show unreachable as its own category, never equal to a capped finite distance by accident. For an entirely masked receiver, show an explicit invalid/no-message policy rather than softmax NaNs. Bound values−12..12 and distance preference0..2; no numeric instability needed. Uniform equal values7 provide a checked null despite changed weights. Negative bias is a finite preference, visually different from a forbidden edge. A graph-layout drag is another null.

Adjacent static diagrams show feature encodings, pair bias and local-message branch as separate inputs, then a GPS local/global parallel-branch schematic with residual/normalization/MLP junctions. Do not call the simplified training model Graphormer/GPS, and do not imply all global operators share exact softmax semantics.

## 3. Return-walk and eigenspace encodings — §3

Show six-cycle versus two triangles with equal scalar node states. Step a three-step random walk; label probabilities at choices and enumerate the two returning triangle paths, each1/8. Exact returns0versus1/4 come from geometry-results.json. A learner adds one edge or changes step count1..5 and predicts a selected return probability. Compute T powers at≤8 nodes; display A-power walk counts in a separate column, with probability/count units distinguished. Isolated zero-row convention matches code; an optional absorbing variant must be labeled as a changed walk. No fake model accuracy from this graph pair.

Basis investigation uses the full two-column eigenvalue2 basis of the four-cycle. Editable angle multiplies both columns by an orthogonal2×2 matrix. Show node-colored columns changing beside a stable projector heatmap and cell table. Require a prediction about same-subspace versus new-graph before Reveal. Sign flips, whole-basis rotations and an actual edge edit have different roles. State complete eigenspace/truncation assumptions. Initial basis and rotated.37-radian case are retained; a direct projector check≤1e−12 supplies expected null. A first-entry sign rule is a deliberately labeled incomplete repair, not the preferred solution. Keep the vector columns visible at first explanation, not only in an end-of-page lab.

## 4. Access versus information — §4

A compact same-value comparison keeps different global/local weights but shows identical weighted outputs7. Let the learner construct a nonconstant value table that exposes the difference. This is a meaningful counterexample task: equality of all selected values is rejected when the prompt asks for a differing output; mere selector changes do not count. Distinguish possible graph access, retained information, learned use and actual prediction in a small four-stage flow. Do not reuse a generic architecture-versus-score chart.

## 5. Square action and weight tying — §5

Display four physical sensor sites with readings1/2/4/8 and the value-routing matrix R. S fixes0/2and swaps1/3. Two horizontal routes: filter→reflect gives8/4/2/1; reflect→filter gives2/1/8/4. A matrix view distinguishes acting on readings from moving drawing positions. Learner edits readings, selects one of the eight square symmetries and predicts route equality. Offer W=R and tied W=(R+R^-1)/2; tied output5/2.5/5/2.5. Constant input is an explicit false-reassurance case for untied W, while the matrix commutator still exposes failure.

Finite-group average can reveal eight conjugate matrices and their mean; do not animate eight samples as all continuous angles. A position-dependent calibration scenario asks whether tying is appropriate for the stated target. The conclusion needs a reason, not a universal “more symmetry is better” response. All exact matrices are in geometry-results.json; numeric edits remain small and computations immediate.

## 6. Physical representations and nonlinearities — §6

Inline coordinate arrows distinguish position Qx+t, displacement Qv, scalar unchanged and axial det(Q)Qv. A90°z rotation plus translation can be chosen from exact matrices; reflect toggle changes determinant. Show the matrix orthogonality/determinant and output-type labels. Give translations only to positions. The learner selects output representation and enters a changed point/vector; feedback compares the actual two actions.

Cartesian ReLU contrast startsv=(1,−2,0). Rotate then ReLU→(2,1,0); ReLU then rotate→(0,1,0). A vector arrow and component bars explain defect2. Scalar norm gate and vector sum are compatible alternatives; they do not convert vectors into invariant scalars. Identity transformation is a null; reflection exposes polar versus axial behavior. Keep the physical assumptions local—external anchored fields are not silently transformed away.

## 7. Geometric message/update workbench — §7

Default constructed triangle(0,0,0)/(1,0,0)/(0,2,0), all pair edges, scalar weight1/(1+r²), step.1. Present relative arrows, squared distances, scalar multipliers and accumulated displacement. No trained status. Direct values from geometry-results.json: updated point0(−.05,−.04,0),point1(1.0666667,−.0333333,0),point2(−.0166667,2.0733333,0). A projection view makes the default planar case readable; a coordinate table and optional3D view preserve z when editing. Never rely on a rotating3D cloud alone to communicate exact values.

Require prediction before applying a point edit or rigid transform. Compare update(transform(X)) with transform(update(X)) side by side; original case uses90°z and translation(3,−2,1). Show maximum absolute discrepancy with scale and tolerance. Reflect is also expected to commute under these invariant scalar/radial choices; derive/verify it on the implemented function. A single moved point changes the geometry; step0is a null; duplicate points remain finite because1+r²is nonzero. Bounds points−4..4,step0..0.3,nodecount≤6,manual one-round updates. Do not infer deep-stack stability from this coefficient range or call the displacement a learned physical force.

Controlled broken modes: apply Cartesian ReLU to directional messages, or make neighbors depend on x-coordinate order. Label them intentionally non-equivariant and require a non-symmetric counterexample. Symmetric/constant inputs can conceal failure. The correct mode uses only the stated invariant inputs and consistent neighbor choices.

## 8. Real graph comparison and fitted inspection — §8

Use all12 measured model rows and raw correct counts/denominators; preserve98versus4546/4616/4610parameter differences and the disclosed .003learning rate. Graph drawing positions are layout only. Legend separates ID,known edge,label role,prediction and true label. Affiliation colors should not give away an initial predicted answer. Inspect a real wrong assessment prediction as well as an arbitrary selected node, with full features/probabilities and two34×34attention head matrices for seed11models.

Seed11bounded forward inference uses saved state_dict, exact LayerNorm epsilon(default1e−5), row-wise layer normalization, two heads of width8, scaled dot products, pre-norm residual ordering, default exact GELU, shared distance embedding across the two blocks and final per-node linear head. GCN uses bias after normalized propagation. Browser translation must match recorded logits/probabilities within explicit tolerances. Recorded other seeds offer only retained probabilities and traces, not arbitrary inference. Epoch traces represent pre-update loss at1/10/50/100/300; do not present them as post-update measurements or add interpolated measurements.

Investigations: record prediction about semantic node relabeling, or edit actual edge0—1and predict whether the set model can react with original features fixed. Its exact unchanged result is meaningful. Other models' tiny changes are displayed numerically, with axes honestly labeled; don't turn.0000008into a dramatic unlabeled bar. If learner chooses recomputed degree/clustering, recompute every affected feature and label this different intervention; saved edit outputs apply only to fixed original structural features. Edits invalidate prior answers. Trained edge weights/attention show mixing and dependence, not causal explanations or assurance that graph structure helps.

No network fitting in the browser. Lazy-load model/evidence only after inspection is opened, cap to34nodes, compute on committed action, discard stale results by input-version token. A small exact path remains the main high-contrast lesson where actual trained sensitivity is visually negligible.

## 9. Chirality, force and advanced representations — §9

Two labeled tetrahedra share pair distances but oriented volume+1/−1. Show three displacement arrows and determinant sign with ordered point roles; swapping identities is not a free geometric reflection. Learner chooses which transformation should preserve a specified target, then explains whether a distance-only input is sufficient. Avoid claiming every molecular target is reflection-sensitive.

Spring example uses exactly E=.5Σpairdistance² and forces from the same energy. Show triangle energy5, force arrows(1,2,0)/(−2,2,0)/(1,−4,0), and sum0. Small point edit recomputes energy/negative gradient; translation null leaves both unchanged, rotation moves forces, step through one central-difference calculation. Units labeled constructed; no predicted molecular energies. Contrast an arbitrary equivariant vector map with an energy-derived field conceptually; don't certify conservation from an arrow rotation.

Representation illustration has scalar1component,vector3and symmetric traceless tensor5; a small symmetric3×3example shows the trace constraint. Explain the general9componentmatrix decomposes into different types rather than calling all tensors ℓ=2. Optional point-attention frame diagram uses two residue-local frames→global query/key points→invariant squared distances→global weighted point→inverse receiver frame. It teaches frame consistency without a fabricated protein fold or simulation.

## 10. Resource decision and finish requirements — §10

An exact tensor-size calculator varies N≤20000,heads1..16,bytes2/4,batch1..4. Show HN²b and decimal units; default8heads,float32,N1000=32millionbytes,N10000=3.2billion. Label this one materialized tensor, not peak memory. Separate pair-work,retained scores,bias preprocessing and graph encodings in the diagram. No universal hardware cutoff or fabricated device benchmark.

Phase two must render the entire inline route, implement prediction/invalidation/reset/null/error behaviors, run final displayed programs and independent correctness/learning reviews, and verify desktop/narrow/keyboard accessibility plus numerical output and proper lazy loading. Current packet has author calculations/specification review only. Complete applicable integration/build checks then record implementation separately; keep the existing content checkpoint intact unless an actual content revision is required.
