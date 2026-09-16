# Message passing and graph convolutions: author design

Stable ID message-passing-graph-convolutions-gcn-gat-graphsage; deep-learning-fundamentals position29. Author/root,13September2026. Research/write only. Full manuscript, visual contracts, offline licensed graph, program and actual results prepared; implementation remains not started.

## Preflight, reference and scope

Actual --work content preflight read in full, including the open destination note from Spectral Graph Theory and the unrelated resolved inbox history. Published topic has no bespoke blueprint; current handoff/teaching standard/design brief/model playbook/code-retention instructions governed this individual design. Complete original JSX consumed in sequential ranges0–225/225–465/465–765/765–1025/1025–1235/1220–end, with the missing1205–1223 reference window recovered after output truncation.

Original source src/learn/data/topics/message-passing-graph-convolutions-gcn-gat-graphsage.jsx is untouched. Baseline commit8c5da59f18516be77c29d5aeeafca3decca4f738, SHA-256cfc63c969032ce8a17c991bd42e15bce7fdf00a695850d4bbdb575045d25c925. No archive copy or runtime mutation.

Retain topic title/stable identity: the three named layers and their common mechanism are the right coverage. GIN/spectral/sampling/readout branches preserve necessary broader depth without a misleading title listing every GNN. The preceding Cross-Attention packet owns general rectangular attention and masks; refresh a local neighborhood read here. Earlier graph fundamentals and spectral concepts are linked/readiness support, not hidden prerequisites for the first path example. Next topic is Graph Transformers & Geometric Deep Learning, which owns structural encodings and physical symmetries. No module reorder or catalogue change.

Learner contract: enter with vector sums, neural affine layers and supervised loss; learn graph/node/edge task units, receiver/sender convention, synchronous messages, aggregation information loss, actual GCN/SAGE/GAT updates, a parameter step, valid data boundaries and evaluation. First pass §§1–6 and practice1–5; optional §§7–9/practice6–8 explain spectral limits, expressiveness, sampling and applications. One whole example moves from observed graph→features→label roles→model→loss→fit→assessment→decision.

## Scope/coverage decisions

| Original coverage or discovered issue | Final treatment |
| --- | --- |
| Long history/default-model narrative | Short concrete relational motivation; remove unsupported “every/all/dominant” claims and inaccurate history |
| MPNN framework and variants | Local send/aggregate/update semantics, synchronized state and edge roles; no claim all messages are means |
| Permutation and receptive fields | Node equivariance versus graph invariance; at-most-L-hop dependency under stated local assumptions; actual numerical influence separate |
| GCN normalization and spectral origin | Unequal-degree path, full endpoint weights, channel/neighbor axes, exact bias placement, optional correct eigenmode explanation |
| Original GCN bias inside linear projection | Correct bias-after-aggregation convention with direct numerical counterexample; associativity applies to linear products, not arbitrary bias moves |
| GraphSAGE aggregation/sampling | Distinguish self/neighbor roles, exact empty-neighbor convention, sampled versus full-neighbor variant, L2 option and LSTM order variability |
| GAT/multiple heads | Local masked score/softmax/value derivation, one-head actual experiment, multi-head shape choices; static common-neighbor ranking limitation |
| Transductive versus inductive | Evaluation/feature/graph-availability contract; GCN can be applied to new graphs with suitable features and fresh graph state |
| GIN and multiset collision claim | A post-sum MLP cannot separate identical raw sums; use nonlinear pre-map example and qualified WL/injectivity statement |
| Synthetic Cora-like high-score example | Replace with actual observed34-node NetworkX graph, full offline provenance,12 fixed fits and two meaningful baselines; no invented method ordering |
| Code and purported PyG equivalence | Complete executable dense-small reference, manual/loop/matrix and relabeling checks; current API behavior read, no unexecuted parity result claimed |
| Production library recipes/samplers | Preserve sparse operation cost, sampling tree, block-diagonal batching and data contracts; remove fake device cutoffs and fixed timing/library rankings |
| Over-smoothing/over-squashing | Correct separate fixed-linear limit, aperiodicity and normalized coordinates; distinguish bottleneck, optimization and overfit instead of hardcoded depth threshold |
| Charts and attention maps | Actual path iterates, real retained trained weights/states/probabilities and all measured scores; no hand-entered collapse or Cora curves |
| Applications | Bond-conditioned molecular message/readout, relation-specific program analysis and time/direction-aware traffic; enough mechanism to show why graph design matters |
| Exposed-answer self-check | Eight changed exercises with independent closed hints and full solutions, plus scored graph/data construction tasks |

Actual discovery map: normalized propagation note belongs here and is accepted in§7; preserve open implementation status. GAT static ranking is valuable local scope, not a new topic; deeper global attention belongs next. Abstract-algebra/geometric invariants belong next owner, not this chapter. Full graph generation/knowledge graph embeddings/large-scale kernel engineering are specialist material; explain the connection without auditing or rewriting them. No new uncovered curriculum item requires a routing note.

## Canonical section-list audit and research

Canonical [Hamilton, Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf), author-posted prepublication draft released with permission. Its actual contents were inspected: introductory graph/task/statistical/spectral material; node embeddings and multirelational reconstruction; Chapter5 neural message passing (self-loops, normalization, set/attention aggregators, concat/residual/gated/jumping updates, edge/multirelation features, pooling/generalized messages); Chapter6 tasks/losses, implementations, subsampling/minibatching and sharing/regularization; Chapter7 convolution/spectral foundations, graphical-model relationships and WL/isomorphism; subsequent graph generation chapters.

Coverage decisions: Chapter5's core operators/edge/readout distinctions and Chapter6's task/split/efficiency concerns taught locally. Gated/RNN and residual machinery already exists earlier; brief connection suffices here rather than duplicating a full GRU derivation. Chapter7 spectral/expressiveness limits are optional but substantive. Full knowledge-graph embedding, graphical-model mean-field derivations, generative graphs and other book chapters are outside this named layer lesson. This is a contents/coverage audit plus selected primary method reading, not a claim to have reread all141 pages.

Primary method reading on13September2026: [GCN](https://arxiv.org/pdf/1609.02907)§2/normalized/spectral equations; [GraphSAGE](https://arxiv.org/pdf/1706.02216)§3 algorithm, neighborhood/sample and training discussion; [GAT](https://arxiv.org/pdf/1710.10903)§2 score/mask/head definition; [MPNN](https://arxiv.org/pdf/1704.01212)§2/framework and edge-conditioned example; [GIN](https://arxiv.org/pdf/1810.00826)Lemma5/Corollary6 and aggregator limits; [GATv2](https://arxiv.org/abs/2105.14491) primary abstract/static ranking characterization with a directly derived monotonicity example. No experimental universal-ranking claim relies on these papers.

Current implementation source read: [PyG2.9 GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html), source-to-target convention, self-loops/normalization, cached graph and shapes. NeighborLoader attempted deep link unavailable; no unverified API snippet or behavior claim depends on it. The full teaching program does not require PyG. [NetworkX graph generator/documentation](https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html), actual generator source, installed3.6.1 data and license inspected.

Alternate resource: official [Stanford CS224W lecture5.1](https://www.youtube.com/watch?v=6g9vtxUmfwM) identified by indexed title/topic and [instructor course-video page](https://ai.stanford.edu/~jure/teaching.html). Direct video fetch unavailable; full video not watched. Resource annotation accurately states the review extent. No secondary summary supplies a technical claim. Sources inform the author-created examples; source wording and original diagrams are not reproduced.

## Numerical evidence and learning review

message-passing-study.py executed successfully with Python3.12.14, NumPy2.3.5, torch2.14.0+cpu, one torch thread and deterministic operations. Shared environment read-only. Real graph export from installed NetworkX3.6.1 retains34 nodes/78 edges/weights and source IDs; data/label meanings and BSD representation license documented. Program runs offline from the packet.

Twelve fixed fits: structural MLP98parameters, GCN98, mean-SAGE178, one-head GAT134; seeds11/29/47,300epochs, no assessment-driven tuning. Actual assessment counts: MLP10/18 each; GCN8/18 each; SAGE10/11/9; GAT9each; fixed label propagation16/18; majority9/18. This unfavorable neural comparison is retained and interpreted. No graph-family superiority, reliable human-behavior inference or new-graph claim. All source IDs/label roles, counts, actual history epochs and relevant weights/probabilities retained.

Exact path: matrix versus explicit receiver/sender loop error8.88e−16; permutation property passes; symmetric eigenvalues−1/6,1/2,1;40-step raw limit2.12842564/2.60677839/2.12842564 with degree-divided constant1.50502420. Bias and scalar-update figures match independent hand formulas. Actual fitted-model relabeling maximum discrepancy2.8611e−6. Removed-propagation scores hold graph-derived features fixed, disclosed in every relevant evidence/spec view. Small collision/sampling/changed-practice calculations are directly derivable and use no learned-data claim.

Author content/visual-spec reread: local edge/task meaning before matrices; synchronized state before layers; unequal-degree normalization and a real update; first-pass/deeper routes explicit; complete code and all real inputs; honest weak results and data dependence; specific graph/matrix/arrow/state/map/spectral/tree representations at first-use locations; actual input-bound predictions and nulls; closed solutions; retained original breadth with corrected statements; next-topic continuity. Root author is not an independent reviewer. Implementation/browser rendering, independent correctness/learning review and final integration are deferred.

Status: implementation not started; author small computation complete; browser/visual review deferred; independent review deferred; user review pending. Next: explicit finish request must pass --work finish and consume all eight packet files, implement the specified topic-specific figures and bounded investigations, execute final displayed code, complete independent reviews and applicable fixes/build/browser/integration. Preserve the packet and current content checkpoint; it is not scratch.
