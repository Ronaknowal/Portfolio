# Message Passing & Graph Convolutions (GCN, GAT, GraphSAGE)

**Explore as you read.** Change directed edges/node features, aggregation/normalization, attention scores, masks and supported fitted-graph inputs. Update adjacency, degree factors, synchronized round states, reachability, information boundaries and fitted outputs together. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose aggregation and sampling from information flow, normalization and expressiveness while avoiding evaluation leakage.


A paper’s words help identify its subject. Its citations may help too: a short ambiguous paper could cite several unmistakable robotics papers. A model that reads each paper independently misses those relationships. A graph neural network gives each paper a representation that can be updated using information from connected papers.

The central operation is simple: **send information along permitted edges, combine the incoming messages, then update each node.** The same learned rule is reused across nodes. Nodes can have different numbers of neighbors, and renaming them should not change the underlying predictions.

This lesson turns that idea into three concrete layers: GCN uses degree-based weights, GraphSAGE separates a node from a summary of its neighbors, and GAT learns attention weights over permitted neighbors. We will calculate a small example, train actual models on a small observed network, and inspect where these methods lose information or fail to generalize.

**First pass:** §§1–6 and core practice 1–5. Return to §§7–9 for spectral limits, expressiveness, sampling and applications, then the deeper practice. You need vectors, a weighted sum, a neural layer and a supervised loss. You do not need to know spectral graph theory to begin. The earlier [cross-attention lesson](/learn/path/full-curriculum/interleaved-cross-attention-architectures?module=deep-learning-fundamentals) provides a useful connection: a node can read a neighborhood as a small memory. Here we also update that memory’s node states, in synchronized rounds.

Your finish line is practical: explain an edge’s meaning, calculate a node update, preserve the result under relabeling, distinguish known context from held-out labels, run a complete graph-learning experiment, and diagnose when more layers or learned attention do not solve the problem. The core reading is about an hour; allow a separate session for the program and exercises.

## 1. Decide what the graph represents

A graph contains **nodes** and **edges**. Features describe a node, an edge, or sometimes the whole graph. A molecular graph might use atoms as nodes and bonds as edges. A road network might use intersections as nodes and roads as edges. A citation graph uses papers and citation relationships. These choices determine which messages are meaningful.

An edge does not automatically mean “these labels should match.” A bond connects different atom types; a purchase connects a user and a product; a citation can disagree with the cited work. **Homophily** means connected nodes tend to share the relevant property. It is a possible pattern in a dataset, not the definition of an edge.

There are three common prediction units:

| Task | Input context | Output unit | Example |
| --- | --- | --- | --- |
| Node prediction | Node features and available relationships | One output per requested node | Paper subject |
| Edge prediction | Endpoint representations and permitted context | One output per candidate pair | Whether a future relationship will occur |
| Graph prediction | All nodes/edges in one example graph | One output per graph | A molecular property |

A node model needs outputs to move with the nodes when they are relabeled. A graph-level property should stay unchanged under that relabeling. These are **permutation equivariance** and **permutation invariance**, respectively. “The result is unchanged” is too vague unless we specify which kind of output we mean.

### Keep direction and indexing explicit

We use a receiver-row convention: \(A_{vu}>0\) means node v can receive from node u. For an undirected graph both directions are present. In an edge list stored as `(source, target)`, the corresponding matrix entry is therefore `A[target, source]`. Some datasets and libraries use another convention; inspect it before multiplying.

For now use an undirected path with three nodes:

\[
0\;—\;1\;—\;2,\qquad x_0=1,\quad x_1=2,\quad x_2=4.
\]

The scalar features are constructed numbers, not measured properties. They let us inspect every contribution. The adjacency without self-loops is

\[
A=\begin{bmatrix}0&1&0\\1&0&1\\0&1&0\end{bmatrix}.
\]

**Visual: graph and matrix together.** Clicking the edge 1→0 highlights row 0/column 1 and the corresponding message arrow. Reversing a directed edge should visibly change its receiver. A node table displays the current feature, incoming neighbors and next state. The graph supplies a spatial explanation; the table makes the same calculation accessible without dragging or color.

## 2. One synchronized message-passing round

A general layer computes messages and updates:

\[
m_v^{(l+1)}=\operatorname{AGG}_{u\in\mathcal N (v)}
M_l (h_v^{(l)},h_u^{(l)},e_{vu}),\qquad
h_v^{(l+1)}=U_l (h_v^{(l)},m_v^{(l+1)}).
\]

Here h is a node state, e is an edge feature, M constructs a message, AGG combines messages and U updates the receiver. The superscript l means layer or round, not exponentiation. The [MPNN paper](https://arxiv.org/pdf/1704.01212) uses this framework to connect several graph models, with a separate readout when predicting a property of the entire graph.

All messages in round l+1 use the states from round l. If a loop overwrites node 0 and then lets node 1 read that new value immediately, the result depends on processing order. That is a different, asynchronous algorithm. Store the old states and write new states separately.

Neighbors form a multiset: order is irrelevant, but repeated feature values may matter. Sum, mean and componentwise maximum all ignore ordering while preserving different information. For scalar neighbors (1,3,5), sum is 9, mean is 3 and maximum is 5. Adding another neighbor with value 5 gives 14,3.5 and 5. A maximum cannot reveal how often its largest value occurred; a mean can hide repeated copies of an entire neighborhood.

A layer need not average. Edge-dependent matrices, nonlinear messages, sums and gated updates all fit the framework. Calling every GNN “average your neighbors” would conceal useful models and their limitations.

### How far can a node read?

With one round of strictly local messages, node 0 can use initial states from nodes 0 and 1 if the update preserves self-information. After two rounds it can also use node 2 through node 1. An L-layer local model depends on nodes **at most L hops away**, under these local-update assumptions. It may fail to use some of them because of zero weights, masks, sampling, nonlinearities or cancellation. Global attention, graph-wide normalization and global features can create additional paths.

The reachable set and its numerical influence are different views. An architecture diagram shows a possible dependency; a derivative or controlled edit shows an actual dependency at a specified input and parameter setting.

**Investigation: move a value through two rounds.** Start on the three-node path with synchronized neighbor means. Inspect which outputs change when x₂ changes from 4 to 8. Reveal round 1 and round 2 separately. Then use an intentionally asynchronous update and reorder processing to expose the difference. Reset returns to synchronized semantics. The learner should explain a path, not just press Next through an animation.

## 3. GCN: a normalized weighted sum

The [Kipf–Welling GCN](https://arxiv.org/pdf/1609.02907) adds self-loops and symmetrically normalizes the resulting adjacency before applying a shared feature transformation. Define

\[
\hat A=A+I,\qquad \hat d_v=\sum_u\hat A_{vu},\qquad
S=\hat D^{-1/2}\hat A\hat D^{-1/2}.
\]

Each permitted contribution from u to v is weighted by \(1/\sqrt{\hat d_v\hat d_u}\) for an unweighted edge. Use the degrees **after** adding loops. In the path, they are (2,3,2), giving

\[
S=\begin{bmatrix}
1/2&1/\sqrt 6&0\\
1/\sqrt 6&1/3&1/\sqrt 6\\
0&1/\sqrt 6&1/2
\end{bmatrix}.
\]

The first propagation, with no learned transformation yet, is

\[
Sx\approx (1.316497,\;2.707908,\;2.816497)^T.
\]

For node 0, calculate \(1/2\cdot1+1/\sqrt6\cdot2\). Node 2’s value 4 cannot contribute directly in one round. The second propagation is approximately (1.763747,2.589923,2.513747): node 2 can now influence node 0 through node 1.

**This is not ordinary row averaging.** The row sums are approximately (.908248,1.149830,.908248). The symmetric operator treats both endpoint degrees explicitly. The random-walk operator \(P=\hat D^{-1}\hat A\) instead has rows summing to one. On a regular graph they coincide; an unequal-degree example is necessary to see the difference.

A neural GCN layer adds learned W, optional bias b and nonlinearity:

\[
H'=\sigma (SHW+\mathbf 1b^T).
\]

H has shape N×dᵢₙ; W is dᵢₙ×dₒᵤₜ; H′ has one dₒᵤₜ-vector per node. The matrix S mixes nodes while W mixes feature channels. The same W applies at every node, so parameter count need not grow with graph size.

Associativity allows \(S(HW)=(SH)W\), but a bias needs care. \(S(HW+\mathbf1b^T)\) usually differs from \(SHW+\mathbf1b^T\), because S1 need not equal 1. For our scalar path with bias 1, applying bias before propagation gives (2.224745,3.857738,3.724745); after propagation gives (2.316497,3.707908,3.816497). The program adds bias after the aggregation.

### One parameter actually learns

Use only node 0 as a supervised example, one shared scalar weight w=.5, target 1 and loss ½(ŷ−1)². Its aggregated feature is a=1.31649658, so ŷ=aw=.65824829. The weight derivative is

\[
\frac{dL}{dw}=(aw-1)a\approx-.44991496.
\]

A gradient step of size.1 gives w′=.54499150 and prediction≈.71747944. The loss decreases. This is ordinary backpropagation through a graph aggregation; edges route features and gradients, while W is fitted to the specified target. The exact program checks the scalar derivative and a node-by-node implementation against matrix multiplication.

**Visual: degree weights and the shared update.** Each incoming arrow shows its endpoint-degree factor and weighted contribution. A separate channel-mixing panel displays w and the supervised error, then traces that one parameter update. Editing a remote value beyond the current hop bound must give a null effect; editing a permitted feature must update the arrow, table and loss consistently.

For a nonnegative undirected graph, this normalized propagation has spectral radius at most one. That fact alone does not bound an entire learned network: W can amplify values, bias can accumulate, and nonlinear/residual paths change the system. We will isolate the fixed linear propagation in §7 before discussing depth.

## 4. GraphSAGE and GAT change the aggregation decision

### GraphSAGE: distinguish self from neighbors

A mean-neighbor GraphSAGE layer can use

\[
\bar h_v=\frac 1{|\mathcal N (v)|}\sum_{u\in\mathcal N (v)}h_u,
\qquad h'_v=\sigma (W_s h_v+W_n\bar h_v+b).
\]

Separate matrices make the distinction between “my state” and “my neighborhood” explicit. Concatenating the two vectors and using one larger matrix is equivalent. Our program uses a zero neighbor summary for an isolated node and retains the self branch. The original [GraphSAGE paper](https://arxiv.org/pdf/1706.02216) also considers pooling and randomly ordered LSTM aggregation, neighborhood sampling, and L2 normalization of intermediate embeddings. These are choices to specify, not interchangeable defaults.

For a node with own value 4 and neighbors (1,3,5), let Wₛ=2 and Wₙ=1 with no bias or nonlinearity. The output is 2·4+3=11. Increasing every neighbor value by 1 increases the output by 1. Increasing the node’s own value by 1 increases it by 2. The model can learn those roles separately.

An LSTM over neighbors is order-sensitive in one run. Randomly permuting its inputs does not make each individual prediction exactly invariant. A distribution over uniformly sampled orders can be invariant in expectation, but it introduces another source of variability. Use a set aggregator when exact order invariance is part of the contract.

Sampling limits how many neighbors are used. It is a computation strategy, not the definition of inductive learning. A learned GCN, GraphSAGE or GAT layer with transferable input features can be applied to a new graph. A model whose feature columns are a lookup table of training-node IDs cannot acquire useful features for arbitrary new IDs merely by changing the layer name.

### GAT: score permitted messages

An original [GAT](https://arxiv.org/pdf/1710.10903) head projects node features to z and scores a sender u for receiver v:

\[
e_{vu}=\operatorname{LeakyReLU}(a_r^Tz_v+a_s^Tz_u),\qquad
\alpha_{vu}=\frac{\exp e_{vu}}{\sum_{j\in\mathcal N[v]}\exp e_{vj}},
\qquad h'_v=\sigma\left (\sum_{u\in\mathcal N[v]}\alpha_{vu}z_u\right).
\]

The softmax is over the receiver’s allowed senders, including its self-loop here. Non-edges receive no message. Intermediate multi-head layers can concatenate outputs; an output layer can average heads. These operations have different widths. The complete small study uses one head so every attention coefficient remains easy to inspect.

A subtle limit: in the original additive scoring form, the receiver contributes the same scalar to every candidate sender’s score. Because LeakyReLU is monotone, it cannot reverse their ranking within a common candidate set. Weights can vary, but the strongest sender among the same candidates is not freely question-dependent. [GATv2](https://arxiv.org/abs/2105.14491) studies this distinction between static and dynamic attention. Learned attention is therefore not a blanket guarantee that every node can select a completely different neighbor ranking.

For example, take sender scores 1 and 2. Receiver score 0 gives preactivations (1,2); receiver score−3 gives (−2,−1), or (−.4,−.2) after a slope.2 LeakyReLU. The second sender ranks first in both cases, although the normalized weights are less unequal in the second. Changing the permitted neighbor set is another way weights can differ.

**Visual: compare actual aggregation rules.** Keep one graph and one feature table fixed. GCN arrows display degree factors; GraphSAGE displays a separate self rail and neighbor mean; GAT displays scores, the local softmax denominator and weighted messages. The learner constructs a feature change and inspects which rule’s weights change. A zero attention vector gives uniform permitted weights; a forbidden sender stays absent even if its score would be large. High attention is a mixing coefficient, not proof that an edge caused a correct decision.

## 5. The data split is part of the graph model

In **transductive node prediction**, the task may explicitly provide the full graph and all node features during fitting while revealing labels for only some nodes. Unlabeled nodes can participate in message passing. Their labels must not be read by the loss, feature construction or model selection. This is a permitted information boundary when it matches the task.

In **inductive evaluation**, some nodes or entire graphs are unavailable during fitting. Construct the fitting graph accordingly. For a future-time task, edges or features created after the prediction time must not enter earlier representations. Recomputing normalizers on a new graph may be necessary; cached adjacency from a different graph is not valid just because the weights remain the same.

For **link prediction**, the held-out relationship itself generally must be removed from the message graph before constructing graph-derived features. Otherwise a model asked to predict whether an edge exists can already use that edge’s presence. Define negative pairs and time boundaries consistently; a sampled absent edge is not necessarily a true permanent negative.

The label mask and the message graph answer different questions. “Do not score this node during fitting” does not mean “this node cannot supply known features,” and it does not automatically authorize using its future relationships. State the intended setting in ordinary language before choosing a split API.

**Investigation: place information behind the boundary.** A timeline has an observed graph, fitting labels and later edges/labels. The learner chooses which items may enter training for transductive, unseen-graph or future-edge tasks. Reveal one actual leakage path at a time. This activity evaluates the learner’s protocol choices, rather than pretending a generic train/test checkbox settles graph leakage.

## 6. A complete experiment on an observed network

The offline [karate-club.json](karate-club.json) records Zachary’s34-node,78-edge karate-club network as distributed by NetworkX 3.6.1. The historical observations include relationships and eventual club affiliation. The program uses binary adjacency but retains original interaction-context weights in the source file. Node IDs are NetworkX’s zero-based labels; paper IDs are one larger. Attribution, the BSD-licensed NetworkX representation and the transformation are documented in [data provenance](data-provenance.md).

We ask a modest question: with this whole historical network known and ten affiliation labels available for supervision, how well can a small model classify the other labeled nodes? This is a classroom transductive experiment on one small dependent network. It is not a forecast of people’s behavior, an unseen-community evaluation or an estimate of production reliability.

The input features are a constant 1, degree divided by 33 and the **clustering coefficient**: the fraction of possible neighbor-to-neighbor edges that actually exist. For degree d≥2, the denominator is d (d−1)/2; define it as zero for smaller degrees. These structural features come from the available graph. They contain no affiliation labels and no trainable node-ID table. They are also limited: two structurally similar nodes may have different affiliations.

Split labels once with seed 133: five fitting, three development and nine assessment nodes per class. Keep all graph edges/features available, as the stated transductive task permits. Fit a two-layer MLP on the same structural features, a GCN, a full-neighbor mean GraphSAGE variant and a one-head GAT. Use hidden width 16, ReLU, two output logits,300 fixed epochs, AdamW.02, weight decay.01 and seeds 11/29/47. No epoch or hyperparameter is selected from the assessment results; all twelve fits are retained.

There is another useful baseline: start a two-column label-score matrix with one-hot values only on the fitting nodes, repeatedly average scores over closed neighborhoods, and restore those fitting labels after each step. This **label propagation** baseline explicitly carries the known training labels across edges. The neural models instead learn shared parameters from those labels and receive only the three structural features as inputs. They use graph information differently; label propagation can be an excellent choice for this particular known-graph task.

**Compare the actual results.** Inspect learned attention beside label propagation, then remove message edges and follow the effect. Explain why neither change has a guaranteed improvement.

### Complete offline program

Save the program next to its JSON and run it with Python, NumPy and CPU PyTorch. NetworkX is not required to reproduce the training: the real graph has already been serialized. The source file also provides the exact tiny path and the scalar update used earlier.

~~~python
"""Complete CPU study and exact fixtures; reads the adjacent real graph offline."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def graph_operators(adjacency):
    closed = adjacency + torch.eye(len(adjacency), dtype=adjacency.dtype)
    degree = closed.sum(-1)
    symmetric = degree.rsqrt()[:, None] * closed * degree.rsqrt()[None, :]
    mean = adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1)
    return symmetric, mean, closed.bool()


class GraphLayer(nn.Module):
    def __init__(self, incoming, outgoing, kind):
        super().__init__()
        self.kind = kind
        self.linear = nn.Linear(incoming, outgoing, bias=False)
        self.bias = nn.Parameter(torch.zeros(outgoing))
        if kind == "sage":
            self.self_linear = nn.Linear(incoming, outgoing, bias=False)
        if kind == "gat":
            self.receiver_score = nn.Parameter(torch.randn(outgoing) * .1)
            self.sender_score = nn.Parameter(torch.randn(outgoing) * .1)

    def forward(self, features, operators):
        symmetric, mean, allowed = operators
        transformed = self.linear(features)
        weights = None
        if self.kind == "gcn":
            output = symmetric @ transformed
        elif self.kind == "sage":
            output = mean @ transformed + self.self_linear(features)
        elif self.kind == "gat":
            scores = transformed @ self.receiver_score
            scores = scores[:, None] + (transformed @ self.sender_score)[None, :]
            scores = nn.functional.leaky_relu(scores, .2).masked_fill(~allowed, -torch.inf)
            weights = scores.softmax(-1)
            output = weights @ transformed
        else:
            output = transformed
        return output + self.bias, weights


class NodeClassifier(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.first = GraphLayer(3, 16, kind)
        self.last = GraphLayer(16, 2, kind)

    def forward(self, features, operators):
        hidden, weights = self.first(features, operators)
        hidden = hidden.relu()
        logits, _ = self.last(hidden, operators)
        return logits, hidden, weights


def exact_cases():
    # A three-node unequal-degree path: 0--1--2, with self-loops only in GCN.
    adjacency = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]], dtype=torch.float64)
    symmetric, mean, allowed = graph_operators(adjacency)
    features = torch.tensor([[1.], [2.], [4.]], dtype=torch.float64)
    direct = torch.zeros_like(features)
    degrees = (adjacency + torch.eye(3)).sum(-1)
    for receiver in range(3):
        for sender in range(3):
            if allowed[receiver, sender]:
                direct[receiver] += features[sender] / torch.sqrt(degrees[receiver] * degrees[sender])
    assert torch.allclose(direct, symmetric @ features, atol=1e-12)
    p = torch.tensor([2, 0, 1])
    permuted, _, _ = graph_operators(adjacency[p][:, p])
    assert torch.allclose(permuted @ features[p], (symmetric @ features)[p])
    direction = degrees.sqrt(); direction /= direction.norm()
    limit = direction[:, None] * (direction @ features)[None, :]
    propagated = features.clone(); steps = []
    for step in range(41):
        if step in (0, 1, 2, 4, 10, 40):
            steps.append({"step": step, "raw": propagated.flatten().tolist(), "degree_divided": (propagated[:, 0] / degrees.sqrt()).tolist()})
        propagated = symmetric @ propagated
    assert torch.allclose(propagated, limit, atol=1e-10)
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    # Bias before propagation is not interchangeable with bias after it.
    before = symmetric @ (features + 1)
    after = symmetric @ features + 1
    # One scalar shared weight and one observed target on node 0.
    weight = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    prediction = (symmetric @ features)[0, 0] * weight
    loss = .5 * (prediction - 1).square()
    loss.backward()
    updated = .5 - .1 * weight.grad.item()
    return {"adjacency": adjacency.tolist(), "features": features.tolist(), "symmetric": symmetric.tolist(), "row_sums": symmetric.sum(-1).tolist(), "first_step": direct.flatten().tolist(), "second_step": (symmetric @ direct).flatten().tolist(), "eigenvalues": eigenvalues.tolist(), "propagation": steps, "limit": limit.flatten().tolist(), "bias_before": before.flatten().tolist(), "bias_after": after.flatten().tolist(), "single_update": {"weight": .5, "prediction": prediction.item(), "loss": loss.item(), "gradient": weight.grad.item(), "new_weight": updated, "new_prediction": (symmetric @ features)[0, 0].item() * updated}, "max_loop_matrix_error": (direct - symmetric @ features).abs().max().item()}


def main():
    data = json.loads((HERE / "karate-club.json").read_text(encoding="utf-8"))
    count = len(data["nodes"])
    adjacency = torch.zeros(count, count)
    for edge in data["edges"]:
        adjacency[edge[0], edge[1]] = adjacency[edge[1], edge[0]] = 1
    features = torch.tensor([[1., node["degree"] / 33., node["clustering"]] for node in data["nodes"]])
    targets = torch.tensor([node["club"] for node in data["nodes"]])
    operators = graph_operators(adjacency)
    rng = np.random.default_rng(133)
    roles = {"fit": [], "development": [], "assessment": []}
    for label in (0, 1):
        order = rng.permutation(np.flatnonzero(targets.numpy() == label)).tolist()
        roles["fit"].extend(order[:5]); roles["development"].extend(order[5:8]); roles["assessment"].extend(order[8:])
    roles = {role: sorted(ids) for role, ids in roles.items()}
    # A non-neural graph baseline: repeatedly average known-label scores and clamp fit labels.
    label_scores = torch.zeros(count, 2)
    fit = roles["fit"]
    label_scores[fit] = nn.functional.one_hot(targets[fit], 2).float()
    transition = (adjacency + torch.eye(count)) / (adjacency.sum(-1, keepdim=True) + 1)
    for _ in range(200):
        label_scores = transition @ label_scores
        label_scores[fit] = nn.functional.one_hot(targets[fit], 2).float()
    baseline = {role: int((label_scores[ids].argmax(-1) == targets[ids]).sum()) for role, ids in roles.items()}
    measured = []
    for kind in ("mlp", "gcn", "sage", "gat"):
        for seed in (11, 29, 47):
            torch.manual_seed(seed)
            model = NodeClassifier(kind)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=.01)
            history = []
            for epoch in range(300):
                logits, _, _ = model(features, operators)
                loss = nn.functional.cross_entropy(logits[fit], targets[fit])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                if epoch in (0, 9, 49, 99, 299):
                    with torch.no_grad():
                        output, _, _ = model(features, operators)
                        history.append({"epoch": epoch + 1, "fit_loss": nn.functional.cross_entropy(output[fit], targets[fit]).item(), "development_correct": int((output[roles["development"]].argmax(-1) == targets[roles["development"]]).sum())})
            with torch.no_grad():
                logits, hidden, weights = model(features, operators)
                record = {"model": kind, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()), "history": history, "correct": {role: int((logits[ids].argmax(-1) == targets[ids]).sum()) for role, ids in roles.items()}, "probabilities": logits.softmax(-1).tolist()}
                removed, _, _ = model(features, graph_operators(torch.zeros_like(adjacency)))
                record["propagation_removed_assessment_correct"] = int((removed[roles["assessment"]].argmax(-1) == targets[roles["assessment"]]).sum())
                permutation = torch.tensor(np.random.default_rng(91).permutation(count))
                permuted_output, _, _ = model(features[permutation], graph_operators(adjacency[permutation][:, permutation]))
                record["permutation_max_error"] = (permuted_output - logits[permutation]).abs().max().item()
                assert record["permutation_max_error"] < 1e-4
                if seed == 11:
                    record["hidden"] = hidden.tolist()
                    record["state_dict"] = {key: tensor.tolist() for key, tensor in model.state_dict().items()}
                    if weights is not None:
                        record["attention"] = weights.tolist()
                measured.append(record)
    result = {"versions": {"torch": torch.__version__, "numpy": np.__version__}, "roles": roles, "graph": {"nodes": count, "undirected_edges": len(data["edges"]), "same_label_edges": sum(int(targets[a] == targets[b]) for a, b, _ in data["edges"])}, "label_propagation_correct": baseline, "measurements": measured, "exact": exact_cases()}
    (HERE / "calculated-inputs.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"roles": {k: len(v) for k, v in roles.items()}, "baseline": baseline, "measurements": [{"model": r["model"], "seed": r["seed"], "correct": r["correct"], "removed": r["propagation_removed_assessment_correct"]} for r in measured], "exact": result["exact"]}, indent=2))


if __name__ == "__main__":
    main()
~~~

### What happened, and what follows from it?

Actual CPU results, with all seeds and raw denominators:

| Model | Seed | Fit / 10 | Development / 6 | Assessment / 18 | Propagation removed / 18 |
| --- | --- | --- | --- | --- | --- |
| mlp | 11 | 9 | 1 | 10 | 10 |
| mlp | 29 | 9 | 1 | 10 | 10 |
| mlp | 47 | 10 | 1 | 10 | 10 |
| gcn | 11 | 10 | 3 | 8 | 11 |
| gcn | 29 | 10 | 3 | 8 | 9 |
| gcn | 47 | 10 | 2 | 8 | 9 |
| sage | 11 | 10 | 5 | 10 | 7 |
| sage | 29 | 10 | 5 | 11 | 11 |
| sage | 47 | 10 | 2 | 9 | 11 |
| gat | 11 | 10 | 2 | 9 | 8 |
| gat | 29 | 10 | 2 | 9 | 9 |
| gat | 47 | 10 | 2 | 9 | 9 |
| Label propagation | fixed | 10 | 6 | 16 | Not this diagnostic |

Always guessing one class scores 9/18 on this balanced assessment. The measured neural models have mlp: 98 parameters; gcn: 98 parameters; sage: 178 parameters; gat: 134 parameters. These are small distinct models, not a parameter-matched family comparison.

These weak neural results are useful. Most models fit the ten supervised nodes perfectly yet do poorly on the eighteen assessment nodes. Adding a learned neighborhood rule cannot compensate automatically for a tiny supervision set, weak input features and a fixed architecture/training budget. The non-neural label-propagation baseline uses the particular known-graph task effectively. The practical conclusion for this example is to retain that baseline, not to keep tuning on the displayed assessment until a neural model wins.

The “propagation removed” diagnostic sets adjacency to zero **while holding the three input features fixed**. It isolates the fitted model’s message paths. It does not remove all graph information, since degree and clustering were computed from the original graph. Some scores improve, some fall and some are unchanged. A perturbation need not hurt for every model; report the result instead of interpreting every architecture as helpful by definition.

The program also relabels nodes and permutes feature rows and both adjacency axes together. Predictions should permute in the same way, within floating-point tolerance. It retains all probabilities and, for seed 11, actual hidden states, parameters and GAT weights. The graph and matrix view can therefore inspect a real incorrect prediction, rather than a hand-painted “learned” heatmap.

**Investigation: inspect a fitted decision.** Select an assessment node to inspect its probability, true label, available feature vector and neighbors immediately. Observe whether a node relabeling changes its semantic prediction; run the stored-model permutation check. Compare the same node across model types without changing the dataset silently. For a feature or edge edit, use the retained seed 11 model and recompute all specified layers; label graph-derived features as fixed or recomputed, rather than mixing those two interventions. This is a bounded model on 34 nodes, not browser training on a hidden large graph.

### Move the same layer from a matrix to an edge list

The dense matrices above are useful because every permitted contribution is visible. A real graph often has far fewer edges than N². Keep the same equation while storing only the edges: project each node once, gather the sending vectors, multiply by an edge coefficient, and add them into the receiving rows. This is the mechanism behind the complete [sparse implementation and PyG bridge](graph_library_bridge.py), not a second independently fitted experiment.

`SparseGraphLayer` implements all three operators with tensor primitives. `index_add_` performs the sum over incoming messages. GCN builds the two endpoint-degree factors after inserting one loop per node. GraphSAGE divides each incoming contribution by the receiver's neighbor count and adds a separate self transform. GAT computes one score per edge, subtracts the receiver's maximum before exponentiating, divides by that receiver's sum, and then aggregates. Detaching the maximum in this stabilization is valid: a common additive shift cancels from the softmax, including its derivative. It does not detach the scores or learned attention parameters.

The input contract is a simple unweighted graph: `edge_index[0]` names sources, `edge_index[1]` names targets, with no duplicate pairs or pre-existing loops. For an undirected edge supply both directions. A dataset with repeated edges requires an explicit multigraph/weight policy; silently leaving duplicates in an edge list while assigning a dense entry to one changes the operation. GCN/GAT insert their own loops; an isolated node therefore reads itself. A GraphSAGE isolate has zero neighbor contribution and retains its self branch. The directed example checks the stated receiving-degree convention; the symmetric spectral theorem in §7 still requires an undirected graph.

| Scratch parameter | PyG parameter | Semantic choice fixed here |
| --- | --- | --- |
| GCN `linear.weight`, `bias` | `GCNConv.lin.weight`, `bias` | Receiving degrees, add one loop, `improved=False`, `cached=False` |
| SAGE neighbor map, self map, bias | `SAGEConv.lin_l.weight`, `lin_r.weight`, `lin_l.bias` | Mean neighbors, no projection activation, no final L2 normalization |
| GAT projection, sender score, receiver score | `GATConv.lin.weight`, `att_src`, `att_dst` | One head, slope .2, zero dropout, no extra residual |

The file supplies the fixture, all imports and both implementations. In an environment with PyTorch, `python graph_library_bridge.py --scratch-only` compares the sparse mechanism with its dense equation. The author ran that bounded check with Torch 2.14.0 CPU: all nine graph/operator cases passed, including an isolate, a changed directed graph and an empty edge list; the largest displayed output discrepancy was about 5.56e-17. These are arithmetic fixtures, not a new accuracy benchmark.

For the ordinary package route, install `torch-geometric==2.9.0` into a compatible PyTorch environment, then run `python graph_library_bridge.py`. The supplied mapping follows the [GCNConv](https://pytorch-geometric.readthedocs.io/en/2.9.0/generated/torch_geometric.nn.conv.GCNConv.html), [SAGEConv](https://pytorch-geometric.readthedocs.io/en/2.9.0/generated/torch_geometric.nn.conv.SAGEConv.html) and [GATConv](https://pytorch-geometric.readthedocs.io/en/2.9.0/generated/torch_geometric.nn.conv.GATConv.html) contracts. **This package route is written but has not been executed for this content revision.** Its assertions compare outputs, feature gradients, every mapped parameter gradient and an equal .03 SGD step. Expected results are agreement within the stated float64 tolerances, not invented saved output. This same layer can replace each corresponding layer in `NodeClassifier`; preserve the label mask, two nonlinear stages and training protocol when doing so.

The sparse mechanism uses O(N d_in d_out + E d_out) arithmetic and O(N d_out + E d_out) working storage with the explicit gathered edge messages shown here, plus graph indices and parameters. This avoids a dense N×N attention/propagation matrix; it is not a universal speed guarantee. For very large graphs, a fused scatter/message kernel or sampled computation can reduce temporary storage. Keep `cached=False` when editing edges: cached normalizers describe the old graph.

**Take control.** Remove both directions of edge 1—2, then add node 3→1 only. Run all three comparisons and inspect node 3's own output before and after the addition. Add a second GAT head by giving each head its own projection/score vectors, then decide whether to concatenate or average.

<details><summary>Hint</summary>Source and target determine who changes directly. Concatenating two width-d heads produces width 2d; averaging retains d.</details>
<details><summary>Solution and success criteria</summary>The new directed edge permits node 1 to read node 3. GraphSAGE node 3 still has no incoming neighbors and keeps its self branch. GCN also changes degree factors on affected endpoints, so message direction alone is not sufficient to enumerate every changed coefficient. Match `heads=2` and `concat` in PyG, copy each head separately, and compare each receiver's coefficients, input/parameter gradients and one update. Repeating the same head twice is a useful equality fixture but does not demonstrate independently learned heads.</details>

## 7. Why repeated propagation can blur distinctions

This deeper branch concerns a **fixed linear operator**. Remove learned weights, biases and nonlinearities and repeatedly compute \(x^{(l+1)}=Sx^{(l)}\). It makes one source of information loss mathematically visible.

For a finite connected nonnegative undirected graph with positive self-loops, let

\[
u=\frac{\sqrt{\hat d}}{\sqrt{\sum_v\hat d_v}}.
\]

Then Su=u. Other modes have eigenvalues of magnitude less than one under these assumptions, and

\[
S^l x\longrightarrow uu^T x.
\]

The surviving node-coordinate pattern is proportional to **square root of degree**, not usually a constant vector. For our path, repeated propagation tends to approximately (2.128426,2.606778,2.128426). Dividing each coordinate by \(\sqrt{\hat d_v}\) gives the same limiting value≈1.505024. The middle node’s larger raw coordinate does not contradict smoothing in the normalized geometry.

The random-walk operator P and symmetric operator S are related by a change of coordinates:

\[
P=\hat D^{-1/2}S\hat D^{1/2}.
\]

They therefore have the same eigenvalues for this undirected positive-degree setting, even though P is not generally symmetric. “Not symmetric” does not imply “has complex eigenvalues.” Their coordinates and norm interpretations differ.

Define the normalized Laplacian \(L=I-S\). A mode with Laplacian eigenvalue λ is multiplied by 1−λ per propagation. The magnitude |1−λ| is not monotonically decreasing over[0,2]: it decreases to zero at 1 and grows again toward 2. Thus a universal “increasingly aggressive low-pass filter” picture is incomplete. For the self-loop path, S has eigenvalues (−1/6,1/2,1): the first mode alternates sign while decaying, the second decays without sign reversal and the last survives.

A polynomial graph filter connects the spectral view to locality. If p (S)=a₀I+a₁S+⋯+aᵣSʳ, the highest power is r, so the output can depend on nodes at most r hops away. Sʲ sums contributions along walks of length j; some coefficients or walk contributions can vanish. A filter described as K terms from power 0 through power K−1 therefore has an at-most-(K−1)-hop bound, not automatically K hops. Chebyshev polynomial filters can compute such polynomials by a recurrence, avoiding an explicit eigenvector matrix; their polynomial degree still determines this support bound. See the earlier [Spectral Graph Theory](/learn/path/full-curriculum/spectral-graph-theory?module=math-foundations) lesson for the fuller frequency interpretation.

Without self-loops a two-node graph simply swaps its two values each round. Starting (1,0) alternates forever. The self-loop/aperiodicity assumptions matter. Disconnected graphs have a surviving component for each connected piece, and directed or signed graphs require a different analysis.

**Visual: three coordinates, three modes.** Plot actual iterates for the unequal-degree path beside degree-divided iterates. A mode panel shows signed gain and magnitude separately. Show whether the raw middle value matches the endpoints as the learner edits the graph, then compare with the two-node no-loop cycle. Do not replace this calculation with a fabricated “all GNNs collapse after four layers” chart.

In a trained network, **over-smoothing** refers broadly to representations losing useful node distinctions through mixing. **Over-squashing** is different: many distant influences must pass through limited-size states or narrow connectivity. A tree can collect exponentially many distant inputs into one fixed-width vector even when those inputs are not averaged to the same value. **Optimization failure** and **overfitting** are additional possibilities. Diagnose them separately using training error, held-out behavior, representation variation and a task-specific long-range intervention.

Residual or jumping-knowledge connections can preserve earlier states; rewiring or global routes can shorten bottlenecks. Each changes the model’s information paths and cost. A high cosine similarity threshold alone does not prove a particular failure, and no universal useful-depth limit follows from the fixed linear example.

## 8. What aggregation can distinguish

Suppose two neighborhoods contain (1,3) and (2,2). Their raw sums and means agree. A deterministic MLP applied **after that identical sum** cannot recover which neighborhood produced it. It receives the same input in both cases.

A nonlinear transformation before aggregation can preserve a distinction: map each scalar x to (x,x²). The two sums become (4,10) and (4,8). This concrete example explains why the representation being summed matters, not only the choice of the sum operator.

The [GIN analysis](https://arxiv.org/pdf/1810.00826) establishes conditions under which learned multiset aggregation and readout can match the one-dimensional Weisfeiler–Lehman test. The result relies on sufficiently discriminative/injective functions and its domain assumptions; it does not say an arbitrary raw sum followed by any trained MLP distinguishes every graph. The familiar update is

\[
h'_v=\operatorname{MLP}\left ((1+\epsilon)h_v+\sum_{u\in\mathcal N (v)}h_u\right).
\]

The states supplied to that sum must themselves preserve the distinctions needed by later layers. The WL test repeatedly refines a node’s label using its old label and multiset of neighboring labels. A simple limitation: a six-cycle and two disjoint triangles both give every node degree 2. With identical initial node features, ordinary local message passing gives every node the same state at every round; a sum readout over six nodes also agrees. It does not discover connectivity merely by stacking more of the same indistinguishable updates.

Mean fails to count repeated copies of a multiset: (a,b) and (a,a,b,b) have the same mean. Maximum loses multiplicity even more directly. This does not make mean or maximum poor choices for every prediction task. Sometimes the intended property is an average or a presence signal, and ignoring size is helpful.

**Investigation: construct a collision.** Enter two multisets and choose sum, mean or max. Watch the two outputs, then add a feature map $(x,x^2)$ and inspect which information becomes distinguishable. A second view compares a six-cycle and two triangles with identical inputs. A component indicator or valid positional encoding supplies additional information; plain GIN does not acquire it automatically.

## 9. Sampling, batching and useful applications

### Count a sampled computation tree honestly

If a target samples s₁ first-hop neighbors and each samples s₂ additional neighbors, a simple occurrence bound is 1+s₁+s₁s₂. For fanouts (25,10), that is 276 occurrences including the root, not 250 unique nodes. Shared neighbors, repeated samples, self branches and deduplication affect the actual distinct nodes and edge work. More layers can grow the sampled computation tree multiplicatively.

A uniform neighbor-sample mean is an unbiased estimate of the full mean under the stated sampling scheme. Passing that estimate through a nonlinear update generally destroys equality of expected outputs. For neighbors (1,3,5,9), all six size-two sample means are (2,3,5,4,6,7), averaging 4.5, equal to the full mean. Their squared values average 139/6≈23.1667, while the squared full mean is 20.25. Sampling can change the distribution of the model’s computation, not just its runtime.

At scale, sparse propagation costs roughly O (E·d) for aggregation at width d, plus feature projections O (N·dᵢₙ·dₒᵤₜ). A dense N×N matrix is appropriate for our 34-node explanation, not for a million-node sparse graph. There is no universal node-count threshold beyond which every full-graph method is impossible: edges, widths, saved activations, hardware and implementation determine memory. Attention over edges can avoid constructing all N² scores.

For several independent graphs, concatenate node arrays and offset edge indices so the adjacency is block diagonal. Keep a node-to-graph ID for readout. Accidentally connecting the last node of one graph to the first of another changes the data. Padding can also work with correct masks; it is a representation choice with overhead, not inherently invalid.

**Visual: sampled ancestry and graph batching.** Distinguish sampled occurrences from unique nodes, show where two paths share a node, and count work under explicit fanouts. The learner repairs an off-by-offset edge in a two-graph batch and verifies that changing one graph cannot alter another’s output. A small sample-mean distribution shows the unbiased mean/nonlinear discrepancy above without claiming every sampler has unbiased gradients.

### Three applications clarify different design choices

**Molecules:** let atom states send bond-conditioned messages, such as \(M_{vu}=W_{\text{bond type}}h_u\). A single shared matrix ignoring bond type cannot distinguish otherwise identical neighborhoods connected by different bond categories in that layer. For an extensive graph property that scales with system size, a sum readout may be a useful prior; for an average property, a mean may be more appropriate. Units and the actual target determine the choice. Three-dimensional positions and rotations introduce further requirements in the next lesson.

**Program analysis:** an abstract syntax tree and a data-flow graph connect different relationships. “Parent syntax node” and “value used by this operation” should not be collapsed into an unexplained generic edge. A relation-specific message can carry information from a variable definition to its uses even when the source tokens are far apart. Whether an edge is available before executing a program and what the label measures still need explicit definition.

**Traffic prediction:** neighboring road sensors exchange time-dependent measurements, but a directed road connection is not the same as symmetric friendship. A practical model may combine temporal processing with directional edge messages. Using future readings or a graph estimated from the entire future series can leak information. The useful lesson is to choose direction, features and time boundary together, rather than importing an undirected GCN formula unchanged.

For implementation beyond this small study, inspect current graph-library conventions for edge direction, duplicate/self-loop handling, normalization, sampling and batching. The current [PyG GCNConv documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html) explicitly describes source-to-target weights, normalization and cached graph state. Keep a tiny dense or loop reference as a correctness oracle for the intended operation. A library name does not certify that your data conventions match its defaults, and an old comment claiming numerical equivalence is not an executed check.

## 10. Practice with changed examples

### 1. Compute a new GCN read — core

Use the three-node path and the same self-loop normalization, but features (2,0,3). Compute the first output at all three nodes. Can changing x₂ affect node 0 in this one round?

<details><summary>Hint</summary>
The endpoint diagonal weight is 1/2 and each path-edge weight is 1/√6.
</details>
<details><summary>Solution</summary>
The outputs are (1,5/√6,1.5)≈(1,2.041241,1.5). Node 0 has no direct message from node 2, so changing x₂ does not change node 0’s one-round output. A second round can transmit that change through node 1.
</details>

### 2. Find an order-dependent bug — core

Two connected nodes start at (1,3). The rule is to average self and neighbor. Compute one synchronized round, then an in-place loop that updates node 0 before node 1. Explain the discrepancy and repair the implementation.

<details><summary>Hint</summary>
In the faulty loop, node 1 reads a value that has already changed.
</details>
<details><summary>Solution</summary>
Synchronized output is (2,2). The in-place loop first sets node 0=2, then node 1=(3+2)/2=2.5. Reversing loop order changes the result again. Read from an unchanged old-state array and write all next states into another array.
</details>

### 3. Separate own and neighbor information — core

A mean GraphSAGE node has own value 2, neighbors (0,4,8), Wₛ=3 and Wₙ=−1, with no bias/nonlinearity. Find its output. What changes if you duplicate every neighbor? What if you replace mean by sum?

<details><summary>Hint</summary>
Duplicating an entire multiset preserves its mean and doubles its sum.
</details>
<details><summary>Solution</summary>
Mean output 6−4=2, unchanged after duplicating all neighbors. Sum output 6−12=−6 before duplication and 6−24=−18 after it. Neither behavior is universally correct; decide whether neighbor multiplicity should matter for the target.
</details>

### 4. Describe a valid split — core

You are predicting tomorrow’s citations. A colleague builds degree features and message edges using the graph after tomorrow, then masks tomorrow’s labels out of the loss. Identify the leakage and give a corrected information boundary.

<details><summary>Hint</summary>
Graph-derived features can reveal an edge even if its label is not scored.
</details>
<details><summary>Solution</summary>
Tomorrow’s graph exposes relationships not available at prediction time, through both edges and degree features. Build graph/features only from information available at the stated time; hold future target edges out of message construction and feature fitting. Define negative candidates and temporal evaluation consistently. A loss mask alone is insufficient.
</details>

### 5. Interpret the real study — core

A GCN fits 10/10 known labels but scores 8/18 on assessment, while label propagation scores 16/18. Does this show graphs are useless, GCN is always worse, or the training code must be broken? State a supported conclusion and a disciplined next experiment.

<details><summary>Hint</summary>
Separate this dataset/protocol/model from a universal claim.
</details>
<details><summary>Solution</summary>
It shows this fitted GCN with these structural features and fixed small protocol generalizes poorly to the displayed labels. Label propagation demonstrates a useful graph-based alternative here. Inspect errors/features and compare on new appropriate evaluation data before selecting a revised model. The current assessment has been inspected; repeatedly tuning against it would destroy its role as independent evidence.
</details>

### 6. Fix a misleading sum claim — deeper

Neighborhoods (0,4) and (1,3) have the same sum. Can an MLP that sees only that raw sum distinguish them? Supply a two-coordinate per-element transformation whose sums differ.

<details><summary>Hint</summary>
Create the distinction before the information has collapsed.
</details>
<details><summary>Solution</summary>
No deterministic function of the identical sum 4 can distinguish them. Mapping x→(x,x²) yields sums (4,16) and (4,10). This demonstrates one useful distinction, not injectivity for every possible multiset.
</details>

### 7. Which coordinates become constant? — deeper

The self-loop path has degrees (2,3,2). A report says its repeated symmetric propagation converges to equal raw values. Correct the claim and give a two-node counterexample when the necessary loop/aperiodicity condition is removed.

<details><summary>Hint</summary>
Use S√d=√d and consider an operator that swaps two entries.
</details>
<details><summary>Solution</summary>
The connected fixed linear limit is proportional to (√2,√3,√2), with scale determined by the input’s projection. Dividing coordinates by√d gives a constant. Without self-loops, a two-node edge swaps (1,0) to (0,1) and back, so convergence fails. Learned weights/nonlinearities require further analysis.
</details>

### 8. Count sampling and batching — deeper

A target samples 4 first-hop occurrences and each samples 3 second-hop occurrences. Give the no-sharing occurrence bound including the target. Then describe a two-graph batch test that detects accidental cross-graph edges.

<details><summary>Hint</summary>
The second-hop product does not count the root or first hop.
</details>
<details><summary>Solution</summary>
The bound is 1+4+4·3=17 occurrences; distinct nodes may be fewer. Compare each graph’s separate output with its slice in a block-diagonal batch. Changing features in graphB should not change graphA’s output for a local per-graph model without cross-graph normalization. A failed test points to offsets, masks, graph IDs or another shared computation.
</details>

## 11. References and another way to learn

- [Hamilton, Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf): a legitimately free author-posted draft. Chapters 5–6 connect message functions, aggregation, updates, graph readout, tasks and sampling. Chapter 7 is the optional theory path after the core example.
- [GCN](https://arxiv.org/pdf/1609.02907), [GraphSAGE](https://arxiv.org/pdf/1706.02216), and [GAT](https://arxiv.org/pdf/1710.10903): read their method sections while annotating the same receiver/message/update diagram. Note which training setting and aggregator each actually studies.
- [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212):§2 is the concise common framework; use the chemistry sections to see why edge features and graph readout matter.
- [How Powerful are Graph Neural Networks?](https://arxiv.org/pdf/1810.00826): read the multiset conditions and failure examples before turning “sum is expressive” into a claim about arbitrary numerical features.
- [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491): the static-ranking limitation and the motivation for GATv 2; useful after reproducing the two-sender example.
- [Stanford CS224W, Message Passing and Node Classification](https://www.youtube.com/watch?v=6g9vtxUmfwM): a spoken/visual route into the mechanism; [the instructor’s teaching page](https://ai.stanford.edu/~jure/teaching.html) links the course series. The lecture identity/topic and course association were checked; no claim is made that its entire video was watched for this packet.
- [NetworkX’s karate-club documentation](https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html): data history, label meaning and indexing for reproducing the actual graph example.

Continue to [Graph Transformers & Geometric Deep Learning](/learn/path/full-curriculum/graph-transformers-geometric-deep-learning?module=deep-learning-fundamentals). It asks what changes when every node can read distant nodes, when graph structure must be encoded explicitly, and when coordinates should rotate while a physical prediction remains consistent. The local message and relabeling contracts here are the foundation for those decisions.
