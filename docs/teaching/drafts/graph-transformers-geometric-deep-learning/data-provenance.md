# Graph-transformer and geometry teaching evidence

Author: root, 13 September 2026. Prepared content only; no runtime implementation or independent phase-two review.

## Observed graph

karate-club.json is the unchanged 34-node/78-edge NetworkX 3.6.1 representation retained in the immediately preceding Message Passing packet. That packet exported the installed generator after checking its [official documentation](https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html), [source](https://networkx.org/documentation/stable/_modules/networkx/generators/social.html#karate_club_graph) and installed BSD 3-Clause license. NETWORKX-LICENSE.txt accompanies the copied representation. This license statement concerns NetworkX's representation; it does not relicense the original paper. Original observation credit: Wayne W. Zachary, “An Information Flow Model for Conflict and Fission in Small Groups,” Journal of Anthropological Research 33(4), 1977, pp.452–473.

Retained fields: zero-based NetworkX ID, one-based paper ID, affiliation club (0=Mr. Hi,1=Officer), unweighted degree, clustering coefficient and edge interaction-context counts. Class labels are historical affiliations, not personal-risk estimates. The classifier uses binary edges; original counts remain available only for provenance. Every node/source ID is distinct. Repeated structural feature vectors are legitimate different nodes, not duplicate observations to delete. The dependent single-graph task does not claim IID people or separate-community generalization.

## Protocol fixed before fitting

Full graph and structural features are available under the declared transductive task. No affiliation appears in an input feature, normalization, distance, random walk or edge. Features are 1, degree/33 and clustering; four positive-step random-walk return probabilities supplement only the walk model. NumPy default_rng(133) independently permutes each class's IDs; five per class fit, three development and nine assessment. This matches the previous lesson's node roles:

- Fit: 1,4,7,12,16,23,28,30,31,33.
- Development: 0,6,10,18,20,22.
- Assessment: 2,3,5,8,9,11,13,14,15,17,19,21,24,25,26,27,29,32.

All 12 fits use 300 epochs, AdamW learning rate .003/weight decay .01, full graph, no dropout/sampling/early stopping, seeds 11/29/47, one CPU thread and deterministic torch operations. Development/assessment labels do not select epochs or hyperparameters. Width16, two heads width8 for the two attention blocks. GCN has 98 parameters; set attention4546; distance4616; walk4610. Parameter counts differ, explicitly acknowledged. The GCN rate differs from the preceding lesson's .02, so its different scores are not presented as a matched change of architecture across lessons.

Actual assessment scores out of18: GCN9/9/10, set10/10/10, distance9/9/9, walk10/9/10. Weak outcomes are kept; no refitting to improve the story. Majority9/18. Previous packet's fixed label propagation16/18 uses the same graph and role split; it is referenced as previously observed, not a newly executed baseline here. Training/development counts, all probabilities, exact sampled history epochs1/10/50/100/300 and seed11 model weights/head matrices are retained in calculated-inputs.json.

Relabeling recomputes distance/return/normalized operators from the permuted adjacency and permutes the feature rows. Maximum observed logit defect across12 models is1.3351440e−5, within the explicit1e−4 threshold. The analytical permutation argument is separate from these floating-point spot checks. The post-fit edge(0,1) removal holds degree/clustering features fixed and recomputes each explicit graph operator; the set model is exactly unchanged. Largest assessed class1-probability difference in seed11: GCN node19 .0243230; distance node5 .000126481; walk node8 .000000834465. Nodes chosen by maximum absolute difference for inspection, not for a favorable accuracy result. The small changes remain small and do not certify causal explanations.

No coordinates are assigned physical meaning to this historical social graph. Its visual layout is only a drawing. Geometry-calculations.py instead uses explicitly constructed points and analytical targets; those are not measured chemistry, fitted forces or a physical benchmark.

## Reproduction and computed geometry

Run graph-transformer-study.py beside its JSON with Python3.12.14, NumPy2.3.5, torch2.14.0+cpu. One complete12-fit campaign was executed. Shared lesson-tools environment was read-only; no installs. Geometry-calculations.py is a separate complete NumPy program executed once. It records exact path weights/output12/7, mask output4/3, remote-value contrast19/7; six-cycle/triangle spectra and three-step returns0/.25; full four-cycle eigenspace projector invariance1.11e−16; all-eight finite group commutation; constant-input false null; rigid coordinate-update calculation and twelve orthogonal probes; vector-ReLU defect2; spring energy5 and force finite-difference discrepancy3.7858e−11; signed tetrahedron volume+1/−1. Analytical derivations in the manuscript explain these properties independently of the numerical check.

Full-precision matrices/points and original role IDs are retained. Approximate plot lines must identify whether they show exact calculated values, saved epochs or interactive recomputation. No theoretical sample-efficiency multiplier, speedup, production leaderboard or hardware cutoff is measured by this packet.

Required phase-two inputs: lesson.md, visual-specifications.md, design.md, this provenance, the two programs, both result JSONs, karate-club.json and NETWORKX-LICENSE.txt. These10 files are pending implementation inputs and must remain. No temporary downloaded sources, screenshots or package changes were retained. Runtime browser inference must be explicitly translated and verified against the seed11 weights; no arbitrary-seed inference is supported by unsaved weights.
