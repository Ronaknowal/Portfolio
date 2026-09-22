# Message passing: visual and investigation specifications

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Edit a graph and follow the message.** Change directed edges/node features, aggregation/normalization, attention scores, masks and supported fitted-graph inputs.
**See the consequence.** Update adjacency, degree factors, synchronized round states, reachability, information boundaries and fitted outputs together.
**Decision connection.** Choose aggregation and sampling from information flow, normalization and expressiveness while avoiding evaluation leakage.


Research/write phase only. These are implementation contracts for the complete lesson.md, not implemented browser components. Preserve the three distinct representations: graph topology, node/channel values and numerical aggregation. Use message-passing-study.py, karate-club.json and calculated-inputs.json as the scoped offline evidence. Retained seed 11 model parameters support the bounded inference explicitly described below; other seeds have Show the current computed result and its contributing terms immediately.

## Common interaction contract

Each investigation follows the live exploration contract above: current results are visible immediately, valid entity edits update all linked views, and comparisons explain the mechanism. Reset restores the declared inputs and recomputes their result. No prediction or answer-submission state is retained.

Use keyboard controls and numeric/table alternatives to node/edge dragging. Pair colors with node IDs, role labels, line styles and a text equivalent. Announce a concise result on an explicit check; do not flood a live region while editing. At mobile width stack graph and matrix, retain normal-size labels and allow local matrix scrolling. Only opened interactive/evidence panels load optional data; do not run Python, train networks or fetch external datasets in the browser. Exact fixtures≤8 nodes; retained inference≤34 nodes×16 hidden channels, one manual run per valid edit. Cancel stale results if computation is moved to a worker. No animation is required to understand a step.

## 1. Graph/matrix/direction — §1

Static inline initial path 0—1—2, features 1/2/4, and adjacency with receiver rows/sender columns. Highlight an edge, its matrix cell and source/target names together. Let learner construct a directed three-node chain via an edge table, inspect the receiver of one message, and show immediately the corresponding multiplication. Edges use distinct arrowheads; undirected display represents two directions. Duplicate edges must be explicitly rejected or assigned a declared weighted meaning; no accidental double counting. A self-loop is visible as a separate access choice.

Task uses actual graph edits, not a model-name selector alone. Null: changing a non-sender’s feature cannot affect a receiver in one local round. Contrast: reversing an edge reverses its permitted flow. Relabeling nodes moves feature rows and both matrix axes together; moving positions on the canvas does not relabel them or create physical coordinates.

## 2. Synchronized rounds and reachability — §2

For a small editable graph show old-state and next-state columns. In synchronized mode all arrows read the old column and writes accumulate into the next; write all nodes together. Node 0 reachability at rounds 0/1/2 in the path is{0},{0,1},{0,1,2} when the rule includes self. Show dependency support separately from a numeric derivative/change; a zero learned weight can remove actual influence without deleting an edge.

Start the two-node averaging exercise with (1,3) and closed-neighbor mean. Show synchronized output (2,2) beside the explicitly faulty in-place 0-first result (2,2.5). Reversing processing order exposes the bug. Keep that faulty algorithm isolated from correct mode. A one-node self-only graph is the null; state L-hop assumptions and the effect of adding a virtual/global node locally.

## 3. GCN degree factors, bias and one update — §3

Use exact.adjacency/features/symmetric from calculated-inputs.json. Inline draw the path with closed degrees 2/3/2. A selected receiver has contribution cards 1/√(dᵥdᵤ), feature, product; their sum equals the matrix view. First outputs 1.31649658,2.70790812,2.81649658; second 1.76374715,2.58992343,2.51374715. Use full internal precision and sensible displayed rounding.

Allow nonnegative weighted undirected edges, self-loop toggle and scalar features−8..8. Recompute degrees after edge/loop edits. For symmetric normalization, isolated zero-degree cases without loops need an explicit zero-row convention and a local notice that connected positive-degree spectral claims do not apply. Default loops make all degrees positive. Compare row-normalized and symmetric operators on the same unequal-degree graph; regular graphs are the equality null. Do not label symmetric rows probability distributions.

A small bias-placement comparison uses b1 and the default graph. Before propagation gives 2.22474487/3.85773803/3.72474487; after gives 2.31649658/3.70790812/3.81649658. The learner can observe whether associativity justifies moving the bias; reveal S1 versus 1. Bias 0 is a null case.

The parameter-update rail uses exact.single_update. Inputs one scalar w, target and learning rate, feature/graph current version. inspect gradient sign and next prediction, then execute the actual squared-error step. Default w.5,target 1,rate.1, receiver 0; gradient−.449914957, w′.544991496, prediction′.717479441. Show current supervised node distinctly from context nodes. Rate 0 and zero aggregated feature are useful no-change cases; a larger rate may overshoot. Do not conflate this scalar step with the later 12 fitted classifiers.

## 4. Aggregation rules and attention ranking — §4

Static three-lane comparison keeps graph/features fixed: GCN endpoint-degree coefficients, SAGE separate self transformation plus neighbor mean, GAT projected sender/receiver scores→LeakyReLU→local softmax. Label one head versus concatenation/mean of several heads; width changes must be visible if a second explanatory head is shown.

Editable scalar SAGE fixture own 4/neighbors 1,3,5/Ws 2/Wn 1 outputs 11. Let learner edit own value or a neighbor and inspect which contribution changes; empty neighborhood→zero neighbor vector with retained self branch. Mean/max/sum collision task uses actual entered multisets≤6 elements, not prefilled answers. Example duplicated 5 changes sum 9→14 and mean 3→3.5 but leaves max 5. Specify empty mean/max convention rather than NaNs.

GAT rank fixture has sender scores 1/2, receiver 0 versus−3 and slope.2. Compute softmax outputs from current scores. The second sender ranks higher in both shared supports; weights need not agree. Zero score parameters give uniform allowed weights; a forbidden sender has exactly zero contribution. Allow a graph-mask edit as a contrast but distinguish changed support from changed ranking on common support. This is an exact original-GAT scalar score example, not a fabricated trained explanation. The real 34-node trained head is a separate panel below.

## 5. Information boundary construction — §5

Each investigation follows the live exploration contract above: current results are visible immediately, valid entity edits update all linked views, and comparisons explain the mechanism. Reset restores the declared inputs and recomputes their result. No prediction or answer-submission state is retained.

## 6. Real graph, fitted states and controls — §6

Show the 34-node real network from karate-club.json with deterministic layout. Legend distinguishes observed edge, fit/development/assessment role, prediction and true label; show true labels separately from the model outputs and training/assessment roles. Zero-based model ID and one-based paper ID are both discoverable. Original edge context counts are retained, but this experiment's adjacency is binary; do not encode original weight by line width and imply the model used it.

Input feature table:1, degree/33, clustering coefficient. Explain triangle-count denominator choose (d,2). Model comparison chart uses all measured raw counts and denominators; do not hide unfavorable runs or add a winner claim. Label propagation 16/18 is a fixed graph baseline, structural MLP 10/18, GCN 8/18, SAGE 9–11/18, GAT 9/18 in this run. Retain train/development context and parameter counts. Saved epoch traces only at 1/10/50/100/300; interpolation is not new measurement. No covariance/confidence interval is fabricated from three seed values.

Recorded mode selects any of 12 model/seed results. Seed 11 inference mode uses actual state_dict keys from the author program and identical layer order, bias-after-aggregation, ReLU, mean isolation convention and GAT slope. All weights/probabilities/hidden vectors come from the saved model; do not combine one seed's weights with another's outputs. GAT's first head weights are 34×34, row-normalized over allowed closed neighborhoods. Clicking one receiver shows only its incoming edge weights and weighted feature contributions; a table provides every coefficient. A high weight is not a causal explanation.

Select an assessment node and relabel the graph, applying the same permutation to features and adjacency. Show its semantic output and exact logits immediately. Then edit a real feature or remove an edge and recompute bounded seed-11 inference, displaying the selected probability difference. Restoring the graph recomputes the original; label identity, feature values and true label remain distinct.

No retraining, arbitrary network architecture editing or induced performance forecasts. Model/state loading occurs on panel expansion and results are cached by model+input version. A degenerate edgeless graph remains finite: GCN/GAT retain self loops, SAGE retains self branch. Keep max 34 nodes; adding new nodes requires a separate feature and model contract and is outside this investigation.

## 7. Propagation modes and the routed spectral correction — §7

Inline plot of the path's exact fixed-linear iterates at 0/1/2/4/10/40; raw and divided-by-√degree coordinates must be separate labeled views. The limit is 2.12842564/2.60677839/2.12842564; degree-divided values approach 1.50502420. An adjacent gain graph displays signed 1−λ and magnitude|1−λ| over[0,2], with exact axis labels and the path eigenvalues mapped to λ=1−μ. No trained-depth or accuracy interpretation.

Display the raw and degree-scaled states at the current step and their respective limiting behavior; changing the graph recomputes both views. Change to a two-node no-loop edge with (1,0): show alternation, not convergence. Self-loop toggle restores a different operator and its applicable conditions. Disconnected variant has per-component surviving states. For editable≤8 node undirected nonnegative graphs, compute iterates directly; spectral eigendecomposition is unnecessary for the core slider and can remain a fixed analytical example. Stop after a bounded number of steps and never invent a measured convergence time. General learned weights, residuals, nonlinearities and directed/signed graphs are outside this exact limiting theorem.

Over-squashing gets a different static representation: branching distant inputs crossing one narrow state vector; over-smoothing shows distinctions mixing away. Both link to specific interventions, not a single collapse icon or fixed depth threshold.

## 8. Aggregation collisions and graph expressiveness — §8

Two editable scalar multisets≤6 elements show raw sum/mean/max and optional transformed sum (x,x²). Ask learner to construct unequal multisets colliding under a selected aggregation. Validate unequal multisets and equal outputs numerically; don't mark two identical inputs as a successful counterexample. Initial (1,3)/(2,2) gives equal raw 4 but transformed (4,10)/(4,8). Explain why a post-sum MLP cannot separate equal inputs.

Static/steppable six-cycle versus two disjoint triangles, all initial node features equal. Track identical local summaries at each round and identical six-node sum readout. This is a counterexample for the stated plain local features/model, not an assertion that every graph model fails. Adding a component feature or appropriate structural encoding changes the inputs and belongs to an explicitly separate contrast; do not silently smuggle node IDs into one side.

## 9. Sampling tree and batching repair — §9

Occurrence tree root→s1→s2 makes 1+s1+s1s2 visible; join repeated IDs to show distinct-node count separately. Bound fanouts 0..8, depth≤3. Show optional without-replacement cap by available degree; if sampling with replacement, display multiplicities and define their aggregation effect. A fixed sampled tree does not imply general unbiased gradients.

Small exact distribution uses neighbors 1/3/5/9, all six size-two means 2/3/5/4/6/7. Display mean 4.5 and show whether applying square first preserves equality. Display average squares 139/6 versus 20.25. Connect estimation and nonlinearity with actual bars, no pseudo-random decorative trace.

Batch repair uses two graphs with local IDs starting at 0, shows necessary node-index offset for graphB and a node-to-graph membership vector. Learner fixes actual edge indices and inspects graphA's response to a graphB feature edit. Correct block diagonal message passing leaves graphA unchanged; cross-graph normalization would be a different declared operation. All local bounds and text alternative remain available on mobile.

## Finish acceptance

Verify the whole inline visual reading flow as well as interactions; figures belong where the mechanism is first explained. Execute the final displayed program, check recorded cases against offline source IDs and outputs, and compare small loop/matrix/permutation/limit cases against independent arithmetic. Browser checks cover actual keyboard focus, stale results, every no-change contrast, reset and narrow layouts. Independent correctness/learning review, implementation fixes, build and route integration remain phase-two work, not claimed here.

## Written implementation route and placement — 22 September 2026

Place a dense receiver-row matrix beside sparse source/target edges directly after the real study. Highlight the same edge in both; show each receive-group's GAT max/sum and GCN endpoint factors. Isolate/no-edge controls have immediate computed outputs. The package mapping table and gradient/update program are shown alongside, without pretending Python/PyG executes in the browser.

Use topic-owned responsive diagrams and local scrolling for code/matrices. Long filenames and links wrap within the reader at 320px. Show source/setup/download dependencies at the relevant explanation; deferred Python programs load only on request. Keep labels outside geometric marks where possible, fixed scale comparisons truthful, and current results visible during edits. No learner prediction field, submit button, answer lock or optional prediction gate is specified. Existing numerical/interaction checks still apply, and optional package/checkpoint routes carry their actual unexecuted status until phase two supplies evidence.
