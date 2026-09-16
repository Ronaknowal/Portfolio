# Message passing: visual and investigation specifications

Research/write phase only. These are implementation contracts for the complete lesson.md, not implemented browser components. Preserve the three distinct representations: graph topology, node/channel values and numerical aggregation. Use message-passing-study.py, karate-club.json and calculated-inputs.json as the scoped offline evidence. Retained seed 11 model parameters support the bounded inference explicitly described below; other seeds have recorded predictions but no saved weights.

## Common interaction contract

Every scored investigation begins with an unset prediction for the current graph/features/rule/step/model. Save that prediction before Reveal or Run, compare against the actual computed or recorded result, and explain the responsible operation. Input edits invalidate prior correctness and require a new prediction. A layout-only node drag is not a mathematical edit and should not invalidate it. Reset restores the complete starting input, step and unset answers. Hints/solutions are closed initially; no preset correct option, answer-filled field or success just from advancing a trace.

Use keyboard controls and numeric/table alternatives to node/edge dragging. Pair colors with node IDs, role labels, line styles and a text equivalent. Announce a concise result on an explicit check; do not flood a live region while editing. At mobile width stack graph and matrix, retain normal-size labels and allow local matrix scrolling. Only opened interactive/evidence panels load optional data; do not run Python, train networks or fetch external datasets in the browser. Exact fixtures≤8 nodes; retained inference≤34 nodes×16 hidden channels, one manual run per committed edit. Cancel stale results if computation is moved to a worker. No animation is required to understand a step.

## 1. Graph/matrix/direction — §1

Static inline initial path 0—1—2, features 1/2/4, and adjacency with receiver rows/sender columns. Highlight an edge, its matrix cell and source/target names together. Let learner construct a directed three-node chain via an edge table, predict the receiver of one message, then reveal the corresponding multiplication. Edges use distinct arrowheads; undirected display represents two directions. Duplicate edges must be explicitly rejected or assigned a declared weighted meaning; no accidental double counting. A self-loop is visible as a separate access choice.

Task uses actual graph edits, not a model-name selector alone. Null: changing a non-sender’s feature cannot affect a receiver in one local round. Contrast: reversing an edge reverses its permitted flow. Relabeling nodes moves feature rows and both matrix axes together; moving positions on the canvas does not relabel them or create physical coordinates.

## 2. Synchronized rounds and reachability — §2

For a small editable graph show old-state and next-state columns. In synchronized mode all arrows read the old column and writes accumulate into the next; commit all nodes together. Node 0 reachability at rounds 0/1/2 in the path is{0},{0,1},{0,1,2} when the rule includes self. Show dependency support separately from a numeric derivative/change; a zero learned weight can remove actual influence without deleting an edge.

Start the two-node averaging exercise with (1,3), closed-neighbor mean. Require predicted output before revealing synchronized (2,2) versus a deliberately labeled in-place 0-first loop (2,2.5). Reverse processing order to demonstrate the bug. The implementation must keep the faulty algorithm isolated as an educational contrast, never silently use it for the correct-mode graph. A one-node self-only graph is a null case. State L-hop assumptions locally; a virtual/global node changes reachability.

## 3. GCN degree factors, bias and one update — §3

Use exact.adjacency/features/symmetric from calculated-inputs.json. Inline draw the path with closed degrees 2/3/2. A selected receiver has contribution cards 1/√(dᵥdᵤ), feature, product; their sum equals the matrix view. First outputs 1.31649658,2.70790812,2.81649658; second 1.76374715,2.58992343,2.51374715. Use full internal precision and sensible displayed rounding.

Allow nonnegative weighted undirected edges, self-loop toggle and scalar features−8..8. Recompute degrees after edge/loop edits. For symmetric normalization, isolated zero-degree cases without loops need an explicit zero-row convention and a local notice that connected positive-degree spectral claims do not apply. Default loops make all degrees positive. Compare row-normalized and symmetric operators on the same unequal-degree graph; regular graphs are the equality null. Do not label symmetric rows probability distributions.

A small bias-placement comparison uses b1 and the default graph. Before propagation gives 2.22474487/3.85773803/3.72474487; after gives 2.31649658/3.70790812/3.81649658. The learner predicts whether associativity justifies moving the bias; reveal S1 versus 1. Bias 0 is a null case.

The parameter-update rail uses exact.single_update. Inputs one scalar w, target and learning rate, feature/graph current version. Predict gradient sign and next prediction, then execute the actual squared-error step. Default w.5,target 1,rate.1, receiver 0; gradient−.449914957, w′.544991496, prediction′.717479441. Show current supervised node distinctly from context nodes. Rate 0 and zero aggregated feature are useful no-change cases; a larger rate may overshoot. Do not conflate this scalar step with the later 12 fitted classifiers.

## 4. Aggregation rules and attention ranking — §4

Static three-lane comparison keeps graph/features fixed: GCN endpoint-degree coefficients, SAGE separate self transformation plus neighbor mean, GAT projected sender/receiver scores→LeakyReLU→local softmax. Label one head versus concatenation/mean of several heads; width changes must be visible if a second explanatory head is shown.

Editable scalar SAGE fixture own 4/neighbors 1,3,5/Ws 2/Wn 1 outputs 11. Let learner edit own value or a neighbor and predict which contribution changes; empty neighborhood→zero neighbor vector with retained self branch. Mean/max/sum collision task uses actual entered multisets≤6 elements, not prefilled answers. Example duplicated 5 changes sum 9→14 and mean 3→3.5 but leaves max 5. Specify empty mean/max convention rather than NaNs.

GAT rank fixture has sender scores 1/2, receiver 0 versus−3 and slope.2. Compute softmax outputs from current scores. The second sender ranks higher in both shared supports; weights need not agree. Zero score parameters give uniform allowed weights; a forbidden sender has exactly zero contribution. Allow a graph-mask edit as a contrast but distinguish changed support from changed ranking on common support. This is an exact original-GAT scalar score example, not a fabricated trained explanation. The real 34-node trained head is a separate panel below.

## 5. Information boundary construction — §5

Create a small paper graph with timestamped feature observations, edges and labels; all constructed facts are explicitly labeled. Three tasks have different availability contracts: known-graph transductive labels; wholly unseen graph; future-edge prediction. Learner places each item into available-context, fitting-supervision or unavailable-future categories and draws the fitting graph. Predictions begin unset and feedback is based on actual chosen categories/edges, with one concrete leakage route per error. A loss mask that excludes a label must not magically hide its leaked edge/feature. Reset restores the specific task state, not a universally correct graph.

## 6. Real graph, fitted states and controls — §6

Show the 34-node real network from karate-club.json with deterministic layout. Legend distinguishes observed edge, fit/development/assessment role, prediction and true label; do not color every node by true label before the learner has predicted. Zero-based model ID and one-based paper ID are both discoverable. Original edge context counts are retained, but this experiment's adjacency is binary; do not encode original weight by line width and imply the model used it.

Input feature table:1, degree/33, clustering coefficient. Explain triangle-count denominator choose (d,2). Model comparison chart uses all measured raw counts and denominators; do not hide unfavorable runs or add a winner claim. Label propagation 16/18 is a fixed graph baseline, structural MLP 10/18, GCN 8/18, SAGE 9–11/18, GAT 9/18 in this run. Retain train/development context and parameter counts. Saved epoch traces only at 1/10/50/100/300; interpolation is not new measurement. No covariance/confidence interval is fabricated from three seed values.

Recorded mode selects any of 12 model/seed results. Seed 11 inference mode uses actual state_dict keys from the author program and identical layer order, bias-after-aggregation, ReLU, mean isolation convention and GAT slope. All weights/probabilities/hidden vectors come from the saved model; do not combine one seed's weights with another's outputs. GAT's first head weights are 34×34, row-normalized over allowed closed neighborhoods. Clicking one receiver shows only its incoming edge weights and weighted feature contributions; a table provides every coefficient. A high weight is not a causal explanation.

Prediction tasks: choose an assessment node and predict whether a relabeling changes its semantic result; compare exact recomputed logits after applying the same permutation to features/adjacency. A second task changes a feature or removes an edge, predicts a selected probability direction/equality and runs bounded seed 11 inference. For arbitrary edge edits offer two clearly separated interventions: hold the original structural features fixed (isolates propagation) or recompute degree/33/clustering from edited binary graph (changes model input as well). Stored “propagation removed” aggregate counts use the former. Do not present those saved counts under the recomputed-feature label. Same-node edit/relabel/graph restore invalidates answers appropriately.

No retraining, arbitrary network architecture editing or induced performance forecasts. Model/state loading occurs on panel expansion and results are cached by model+input version. A degenerate edgeless graph remains finite: GCN/GAT retain self loops, SAGE retains self branch. Keep max 34 nodes; adding new nodes requires a separate feature and model contract and is outside this investigation.

## 7. Propagation modes and the routed spectral correction — §7

Inline plot of the path's exact fixed-linear iterates at 0/1/2/4/10/40; raw and divided-by-√degree coordinates must be separate labeled views. The limit is 2.12842564/2.60677839/2.12842564; degree-divided values approach 1.50502420. An adjacent gain graph displays signed 1−λ and magnitude|1−λ| over[0,2], with exact axis labels and the path eigenvalues mapped to λ=1−μ. No trained-depth or accuracy interpretation.

Learner predicts constant-raw versus degree-scaled limit before revealing. Change to a two-node no-loop edge with (1,0): show alternation, not convergence. Self-loop toggle restores a different operator and its applicable conditions. Disconnected variant has per-component surviving states. For editable≤8 node undirected nonnegative graphs, compute iterates directly; spectral eigendecomposition is unnecessary for the core slider and can remain a fixed analytical example. Stop after a bounded number of steps and never invent a measured convergence time. General learned weights, residuals, nonlinearities and directed/signed graphs are outside this exact limiting theorem.

Over-squashing gets a different static representation: branching distant inputs crossing one narrow state vector; over-smoothing shows distinctions mixing away. Both link to specific interventions, not a single collapse icon or fixed depth threshold.

## 8. Aggregation collisions and graph expressiveness — §8

Two editable scalar multisets≤6 elements show raw sum/mean/max and optional transformed sum (x,x²). Ask learner to construct unequal multisets colliding under a selected aggregation. Validate unequal multisets and equal outputs numerically; don't mark two identical inputs as a successful counterexample. Initial (1,3)/(2,2) gives equal raw 4 but transformed (4,10)/(4,8). Explain why a post-sum MLP cannot separate equal inputs.

Static/steppable six-cycle versus two disjoint triangles, all initial node features equal. Track identical local summaries at each round and identical six-node sum readout. This is a counterexample for the stated plain local features/model, not an assertion that every graph model fails. Adding a component feature or appropriate structural encoding changes the inputs and belongs to an explicitly separate contrast; do not silently smuggle node IDs into one side.

## 9. Sampling tree and batching repair — §9

Occurrence tree root→s1→s2 makes 1+s1+s1s2 visible; join repeated IDs to show distinct-node count separately. Bound fanouts 0..8, depth≤3. Show optional without-replacement cap by available degree; if sampling with replacement, display multiplicities and define their aggregation effect. A fixed sampled tree does not imply general unbiased gradients.

Small exact distribution uses neighbors 1/3/5/9, all six size-two means 2/3/5/4/6/7. Learner predicts mean 4.5 and whether applying square first preserves equality. Reveal average squares 139/6 versus 20.25. Connect estimation and nonlinearity with actual bars, no pseudo-random decorative trace.

Batch repair uses two graphs with local IDs starting at 0, shows necessary node-index offset for graphB and a node-to-graph membership vector. Learner fixes actual edge indices and predicts graphA's response to a graphB feature edit. Correct block diagonal message passing leaves graphA unchanged; cross-graph normalization would be a different declared operation. All local bounds and text alternative remain available on mobile.

## Finish acceptance

Verify the whole inline visual reading flow as well as interactions; figures belong where the mechanism is first explained. Execute the final displayed program, check recorded cases against offline source IDs and outputs, and compare small loop/matrix/permutation/limit cases against independent arithmetic. Browser checks cover actual keyboard focus, stale predictions, every no-change contrast, reset and narrow layouts. Independent correctness/learning review, implementation fixes, build and route integration remain phase-two work, not claimed here.
