# Graph Transformers & Geometric Deep Learning

A molecular diagram tells you which atoms are bonded. A set of three-dimensional coordinates tells you where the atoms are. These are different kinds of information. Two conformations can have the same bonds and different shapes; rotating the entire molecule changes its coordinates without changing those internal distances.

This lesson asks two connected questions: **How can a node consult distant parts of a graph while retaining information about its structure? How should predictions change when we relabel nodes or transform physical coordinates?** Graph transformers address the first. Geometric deep learning provides a framework for the second, including architectures that satisfy the required transformation rule by construction.

In the previous [message-passing lesson](/learn/path/full-curriculum/message-passing-graph-convolutions-gcn-gat-graphsage?module=deep-learning-fundamentals), information traveled along local edges. Keep that mechanism: global attention is another information route, and geometric constraints can apply to either route. Neither attention nor symmetry alone guarantees that a model makes accurate predictions.

**First pass:** §§1–4 explain graph structure and global reads; §§5–7 build the symmetry contract from a square to a point cloud; §8 gives a complete real-data experiment. Attempt practice 1–5. Sections 9–10 and practice 6–8 extend the lesson into tensor features, physical outputs, applications and resource decisions. Allow separate reading and experiment sessions. You need weighted sums, graph adjacency and a learned neural layer. The earlier [groups and symmetry actions lesson](/learn/path/full-curriculum/abstract-algebra-groups-symmetry-actions?module=math-foundations) supplies additional algebra, but each transformation used below is defined locally.

## 1. Three things a graph drawing can conceal

Separate **node identity**, **connectivity** and **physical position**. A graph-layout program might move nodes apart to make edges readable. That motion does not change the graph. An atom-coordinate measurement has a different meaning: moving one atom while holding the others fixed changes distances and potentially the predicted property.

For N nodes, write H as an N×d table of node features and A as an N×N adjacency matrix. We use receiver rows: Aᵢⱼ indicates that i receives from j. A geometric graph additionally has X, an N×3 coordinate table. Features such as atom type can be scalars under spatial rotations even when their numerical encoding is a vector of several channels. “Vector of features” does not automatically mean “physical vector.”

**Inline illustration: the same network, three edits.** Relabel its nodes, drag its layout, then move an actual coordinate. Keep the node table and edge set visible. A layout drag changes neither A nor H. Relabeling permutes both adjacency axes and feature rows. A physical edit changes X and any geometric features derived from X. This distinction prevents many later mistakes.

Let P be a permutation matrix: multiplying PH reorders node rows. For node outputs, the desired contract is

\[
F(PH,PAP^T)=P F(H,A).
\]

The output follows the node it describes. For a graph-level scalar, the output should instead be unchanged. A sum or mean of scalar node outputs can provide that invariance, although those choices preserve different information about graph size.

### A transformer cannot use information it never receives

Self-attention on H alone compares every node with every other node. If A is never used in features, masks, biases or another branch, changing only A cannot change the result. Such a model processes a set of node features. It can be useful, but it does not become aware of bonds or citations merely because its inputs are called nodes.

If the features already contain degree, triangle counts or message-passing states, some graph structure is present even without an explicit attention bias. An ablation called “no graph information” must remove those paths too. In our real experiment, every model retains degree and clustering features; the comparison isolates additional routes into graph structure.

## 2. Give an attention read structural information

For receiver i and sender j, begin with a usual attention score, then add a structural bias:

\[
s_{ij}=q_i^Tk_j/\sqrt{d_k}+b_{ij},\qquad
\alpha_{ij}=\frac{e^{s_{ij}}}{\sum_{u\in\mathcal A(i)}e^{s_{iu}}},\qquad
y_i=\sum_{j\in\mathcal A(i)}\alpha_{ij}v_j.
\]

Here q asks for relevant information, k determines a match, v supplies content and the allowed set A(i) determines which senders can participate. A **bias** changes relative preference; a **mask** removes access. A finite negative bias is not the same as forbidding a message.

Use the path 0—1—2 with values (1,2,4). Give receiver 0 zero content scores and a bias of −distance×ln2. Its three exponentiated scores are (1,1/2,1/4). Dividing by their sum gives weights (4/7,2/7,1/7), and the output is

\[
\frac47\cdot1+\frac27\cdot2+\frac17\cdot4=\frac{12}{7}.
\]

Node 2 contributes immediately through global attention even though it is two graph hops away. If we mask attention to the closed one-hop neighborhood, the weights become (2/3,1/3,0) and the output becomes 4/3. If instead we retain global attention and change node 2's value from 4 to 11, the output rises by exactly 1. The mask would make this edit have no effect at receiver 0 in this layer.

**Investigation: build the allowed read.** Edit the edge table, sender values and permitted set. Record a prediction about a selected output before revealing scores, normalization and contributions. A distance preference and a hard mask use different marks. Changing the graph recomputes shortest paths; simply moving nodes in the layout leaves the computation unchanged. The default numbers are constructed arithmetic, not trained attention.

### Graphormer and GPS make different design choices

[Graphormer](https://arxiv.org/pdf/2106.05234) uses degree embeddings on nodes, shortest-path-dependent attention bias and an edge-feature bias along a selected shortest path. A learnable distance table can favor or discourage particular distances; it need not decrease monotonically. Disconnected pairs need a separate category, rather than being confused with a long finite path. Its special graph-summary node also has a distinct role from physical edges.

Shortest-path length is invariant to node names. Selecting one of several equally short paths can be more delicate: if their edge features differ, a tie-break based on node index can affect the resulting bias. A claim of exact relabeling consistency must include this preprocessing, not only the attention formula. Our teaching model uses distances and no path-edge aggregation.

[GPS](https://arxiv.org/pdf/2205.12454) combines structural or positional inputs, a local message-passing branch and a global attention branch. Its layer combines those routes with residual/normalization operations and an MLP. This is useful when local edges supply meaningful inductive structure and distant interactions also matter. The global branch can use different attention operators; their mathematical and computational tradeoffs remain relevant.

**Inline diagram: three ways structure reaches a prediction.** Show a node feature route, pair-bias route and local-message route, each ending in the same node representation. A separate diagram shows local and global GPS branches rather than pretending GPS is simply a renamed Graphormer. Encodings can also strengthen ordinary message-passing networks; their benefit is not exclusive to transformers.

## 3. Structural encodings are measurements of a graph

### Random-walk return probabilities

On an unweighted undirected graph with positive degrees, define T=D⁻¹A. Row i is a probability distribution for one step from i to a uniformly selected neighbor. The entry (Tᵏ)ᵢᵢ is the probability of returning to i after k steps. A node can receive several such probabilities as additional features.

The diagonal of Aᵏ instead counts walks on an unweighted graph. Counts and probabilities are different: each step of a random walk includes a degree-dependent denominator. For an isolated node, choose and document a policy. Our code uses a zero transition row, yielding zero positive-step return values; an absorbing self-loop would define a different walk.

Consider a six-cycle and two disjoint triangles. Both have six nodes, all degree two. With identical initial features, plain local message passing cannot distinguish their nodes by repeated neighbor multisets. But a three-step return probability is zero on the six-cycle and 1/4 on a triangle. In a triangle there are two returning three-step routes, each with probability (1/2)³. This feature distinguishes the examples immediately.

It does not follow that a trained model must use the feature successfully. Nor does it follow that a transformer is necessary: a small model reading that return probability can already separate this pair. A random output difference on two graphs is evidence of different representations at those weights, not an accuracy benchmark.

**Inline walk diagram.** Draw the two returning three-step triangle routes and their probabilities beside the impossible odd return on the six-cycle. Let a learner add an edge and predict a changed return probability, then enumerate or multiply the tiny transition matrix. Keep the original nodes and exact step count visible.

### Laplacian modes

The combinatorial Laplacian L=D−A is a symmetric matrix for an undirected graph. Its eigenvectors are patterns on nodes: Lu=λu means the operator scales that pattern by λ. Low eigenvalues correspond to patterns that vary relatively little across edges under the associated quadratic form. The number of zero eigenvalues equals the number of connected components.

For a connected graph, the zero mode is constant. For a disconnected graph there is an entire space of componentwise constant patterns. The symmetric normalized Laplacian uses degree-scaled coordinates instead; do not silently transfer the constant-vector convention. Isolated-node definitions require additional care. These distinctions were developed in [Spectral Graph Theory](/learn/path/full-curriculum/spectral-graph-theory?module=math-foundations).

Rows of selected eigenvectors can become positional features. Yet the numerical eigenvectors are not unique. A unit eigenvector and its negative represent the same mode. If an eigenvalue repeats, any orthonormal basis of its eigenspace is valid. This happens in connected graphs too: a four-cycle has eigenvalues (0,2,2,4), so its eigenvalue-2 space has a freely rotatable two-vector basis.

Making the first nonzero entry positive does not solve the full problem. “First” depends on node order, and sign choices do not remove rotations inside a repeated eigenspace. Selecting only part of a repeated eigenspace creates another ambiguity. Sorting connected components by arbitrary IDs is not a general relabeling-safe repair.

An instructive invariant is the **projector** UUᵀ for a complete orthonormal eigenspace basis U. Replacing U by UQ for an orthogonal Q gives UQQᵀUᵀ=UUᵀ. Relabeling nodes still permutes its two node axes. [SignNet and BasisNet](https://arxiv.org/html/2202.13013v4) explicitly address sign and basis symmetries; the projector calculation shows why the distinction matters. It is a conceptual route, not a claim that dense projectors are the cheapest encoding for every graph.

**Investigation: change the basis without changing the graph.** Rotate the two columns of the four-cycle eigenspace through an editable angle. Display both changing columns and the unchanged projector. Predict which changes represent a new graph. A sign flip is a special case; a genuine edge edit is a separate intervention. The correct conclusion is “same subspace,” not “all encodings contain identical information.”

## 4. More global access has limits

Full attention reduces the number of layers needed for a distant node to be reachable. It does not guarantee that its information survives the weighted sum, that all structures become distinguishable, or that the model can be trained effectively with the available labels.

For example, if every value vector equals v, then every normalized attention row returns v, regardless of its distance biases. Different weight matrices alone do not create distinct outputs in this case. A downstream bias or nonlinear layer receiving identical node states preserves that equality when its rule is shared. Node features, degree information, edge values or other routes may remove the collision.

Graph spectra are also not complete graph fingerprints: different graphs can be cospectral. An encoding's conditions determine its expressiveness; the label “graph transformer” does not settle them. Separately measure access, distinguishability, fitting and held-out prediction.

**Practice pause.** For the three-node path, replace all three values by 7. Predict the global and locally masked outputs, despite their different weights.

<details><summary>Hint</summary>
A weighted average of equal values has a simple result.
</details>
<details><summary>Solution</summary>
Both outputs equal 7 because each allowed row is normalized. The attention distributions differ; the supplied messages do not expose that difference. This is a useful null case for an implementation and an interpretation warning.
</details>

## 5. State the symmetry contract before choosing a network

A transformation g can act differently on inputs and outputs. Write those actions as P_g and Q_g. The architectural requirement is

\[
F(P_g x)=Q_g F(x).
\]

Invariance is the special case where Q_g leaves the output unchanged. It is not a universal ranking in which one kind of prediction is “weaker” than another: a scalar energy and a vector force require different output actions. The [geometric deep learning text](https://geometricdeeplearning.com/book/algebraicpriors.html) develops this representation-based view.

### Start with four sensors on a square

Number four sensors around a square. A quarter-turn moves their scalar readings cyclically: R(a,b,c,d)=(d,a,b,c). A reflection fixes positions 0 and 2 and swaps 1 and 3: S(a,b,c,d)=(a,d,c,b). A filter W=R copies the preceding sensor into each output.

This filter commutes with quarter-turns because powers of R commute. It does not commute with reflection. On readings (1,2,4,8), reflecting after filtering gives (8,4,2,1), while filtering after reflecting gives (2,1,8,4). A constant input such as (1,1,1,1) hides the failure: both routes agree. One successful input test cannot prove a whole-map property.

For a linear layer, WP_g=Q_gW proves the contract for every input. Averaging the eight square-symmetry conjugates of W produces a map commuting with the entire finite square group. Here it becomes (R+R⁻¹)/2, averaging the two adjacent readings. Its output on our example is (5,2.5,5,2.5). Eight discrete square symmetries do not cover all continuous planar rotations.

**Investigation: two routes through the square.** The learner edits all four readings, chooses rotation or reflection, and predicts whether “filter then transform” equals “transform then filter.” Reveal both value routes and the matrix commutator. Offer the tied filter as a repair; keep the constant-input false reassurance as an explicit challenge. A calibration that genuinely depends on fixed sensor position may require different weights, so verify that the intended task actually has this symmetry.

## 6. Move from node permutations to physical rotations

A spatial rotation is an orthogonal matrix R with determinant +1. It preserves lengths and angles. Adding a translation t changes a point x to Rx+t. SE(3) includes these proper rotations and translations in three dimensions. E(3) also allows reflections, whose orthogonal matrices have determinant −1.

| Quantity | Under a common rotation/reflection Q and translation t |
| --- | --- |
| Position | Qx+t |
| Displacement, velocity or ordinary force | Qv; translation adds nothing |
| Scalar energy of an isolated system with the assumed symmetries | Same scalar |
| Axial vector such as a cross product of two polar vectors | det(Q)Qv |
| Node label/order | Follows the corresponding node permutation |

Forces and electric dipole vectors are ordinary polar vectors, not generally pseudovectors. Angular momentum is an axial vector. They transform the same way under proper rotations but differently under reflections. A charged system's dipole also depends on origin: translating all charges adds total-charge×translation. Specify the physical setting rather than treating the table as permission to ignore external fields or boundaries.

### Scalar channels and vector channels need different nonlinearities

Applying the same scalar ReLU to each channel commutes with a permutation of scalar channels. It generally fails for the Cartesian components of a rotating vector. With v=(1,−2) and a 90° rotation, ReLU(Rv)=(2,1), while R ReLU(v)=(0,1). The discrepancy is 2 in the first component.

Multiplying a vector by a scalar function of its norm is rotation-equivariant: the norm is unchanged, so the same multiplier accompanies the rotated vector. Summing or averaging vectors that share the same rotation action is also equivariant. Averaging force vectors does not magically turn them into a rotation-invariant scalar; it yields another rotating vector.

**Inline illustration: rotate an arrow and its components.** Show the actual arrow, its Cartesian decomposition and the two ReLU routes. Then substitute norm-based gating. Labels distinguish “scalar feature channel” from “component of a geometric vector.” A coordinate table supplies an exact alternative to an animation.

A finite image grid adds another distinction. A 90° rotation of a square array can permute pixels exactly; a 17° rotation usually needs interpolation and boundary handling. Repeated interpolation need not equal one resampling by the composed angle. A proof about continuous functions or exact point coordinates does not automatically prove exact equivariance for that raster pipeline. State the sampled domain, action, padding/cropping and error tolerance when applying the idea to images.

## 7. Build an equivariant coordinate update

Start with scalar node states hᵢ, positions xᵢ and invariant edge attributes aᵢⱼ. Construct messages from quantities that do not change under the intended rigid transformation:

\[
m_{ij}=\phi_e(h_i,h_j,\|x_i-x_j\|^2,a_{ij}),\qquad
s_{ij}=\phi_x(m_{ij}).
\]

Each sᵢⱼ is a scalar. An EGNN-style coordinate update has the form

\[
x'_i=x_i+c\sum_{j\in\mathcal N(i)}(x_i-x_j)s_{ij},\qquad
h'_i=\phi_h\left(h_i,\sum_jm_{ij}\right).
\]

The [EGNN paper](https://proceedings.mlr.press/v139/satorras21a/satorras21a.pdf) develops this construction and related velocity/edge variants. State the normalization c and neighbor rule for an actual implementation; the paper's all-pairs coordinate update uses c=1/(N−1). Our next hand calculation uses a separately declared step coefficient and a fixed scalar function so we can inspect every contribution.

### The proof is a sequence of simple checks

Under xᵢ→Qxᵢ+t, relative positions become Q(xᵢ−xⱼ): translation cancels. Their squared lengths stay unchanged because QᵀQ=I. Consequently messages and scalar multipliers stay unchanged. Each directional contribution rotates or reflects by Q, and their sum does too. Adding the transformed starting point gives Qx′ᵢ+t. Scalar hidden-state updates remain unchanged. Relabeling nodes simply relabels the same sums.

This proof requires invariant initial scalar features, compatible edges and correctly transforming attributes. Building neighbors with an axis-specific rule such as “larger x-coordinate” can violate it. Radius neighbors are compatible in exact arithmetic, but cutoffs, ties and floating-point tolerances still need a defined implementation policy.

### Calculate the arrows

Use three constructed points (0,0,0), (1,0,0), (0,2,0). Connect every pair and set

\[
x'_i=x_i+0.1\sum_{j\ne i}\frac{x_i-x_j}{1+\|x_i-x_j\|^2}.
\]

For point 0, the two relative vectors are (−1,0,0) and (0,−2,0), with multipliers 1/2 and 1/5. The displacement is (−.05,−.04,0). The three updated points are

\[
(-.05,-.04,0),\quad(1.066667,-.033333,0),\quad(-.016667,2.073333,0).
\]

Now rotate 90° around z and translate by (3,−2,1). Updating after the transform gives (3.04,−2.05,1), (3.033333,−.933333,1) and (.926667,−2.016667,1). These are the transformed original outputs. The retained double-precision calculation agrees to about 10⁻¹⁶ for this case.

**Investigation: move the actual points.** Edit one point, predict its update, and inspect relative vectors, squared distances, scalar weights and the final displacement. Then apply the same rigid transformation to all points and compare both computation routes. A global rotation is a no-change test for distances, but moving just one point generally is not. An intentionally labeled Cartesian ReLU or axis-based neighbor rule supplies a failing contrast. Numeric tests check an implementation; the algebra explains the all-input guarantee.

This coordinate update is not automatically a stable physical simulator. Equivariance permits large coefficients, explosive repeated updates and inaccurate predictions. Replacing a relative vector by a norm-normalized direction preserves rigid-motion equivariance if done consistently, but does not bound unbounded learned scalars or guarantee stability. A degree normalization, residual scale or bounded gate changes numerical behavior and must be evaluated as part of the model. Applying arbitrary coordinatewise normalization can destroy the geometric contract.

The complete [geometry-calculations.py](geometry-calculations.py) reproduces the attention arithmetic, return walks, eigenspace projector, square action, coordinate update and changed-output checks. These are constructed mathematical examples. They are separate from the observed graph experiment below.

## 8. Run a complete experiment with real graph data

We reuse the observed 34-node, 78-edge karate-club graph from the preceding lesson. This creates a useful controlled continuation: same source IDs, binary edges, affiliation labels and supervision split. The small offline [graph data](karate-club.json) and [provenance](data-provenance.md) retain the licensed NetworkX representation and explain the original interaction weights, which this experiment does not use.

Each node receives three structural features: constant 1, degree/33 and clustering coefficient. A full known-graph transductive task permits those features and edges at all nodes. Only ten affiliation labels enter the fitting loss; six development and eighteen assessment labels are separate. The split is fixed before fitting. This is one historically observed dependent network, not a population study or a test on unseen graphs.

Compare four deliberately small models:

1. A two-layer GCN with 16 hidden units.
2. A two-block set-attention model on the same structural features, with two heads of width 8.
3. The same attention model with learned shortest-path-distance biases.
4. Attention receiving four random-walk return features in addition to the original three.

The GCN has far fewer parameters. The comparison is a small learning investigation, not a parameter-matched benchmark or a reproduction of Graphormer/GPS. All fits use 300 epochs, AdamW learning rate .003, weight decay .01 and seeds 11,29,47. We keep every run, without choosing an epoch or tuning parameters from the assessment scores. This GCN's learning rate differs from the preceding lesson's .02; compare the variants within this table, not two lessons as if they were one matched experiment.

**Before running:** predict whether giving the model more structural information necessarily raises its assessment score. Then follow one node from input features to q/k/v, two residual blocks, logits, supervised loss and a parameter update.

### Complete offline program

Save this program beside karate-club.json. It requires NumPy and CPU PyTorch; it downloads nothing and does not need a graph library. It computes shortest paths with an explicit unreachable category, return probabilities, normalized GCN propagation, attention blocks, fitting, assessment and node-relabeling checks.

~~~python
"""Offline graph-structure study. This is a small teaching model, not Graphormer/GPS."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def structural_inputs(adjacency):
    n = len(adjacency)
    degree = adjacency.sum(-1)
    transition = adjacency / degree[:, None].clamp_min(1)
    power = torch.eye(n)
    returns = []
    for _ in range(4):
        power = power @ transition
        returns.append(power.diagonal())
    distance = torch.full((n, n), float('inf'))
    distance.fill_diagonal_(0)
    distance[adjacency > 0] = 1
    for middle in range(n):
        distance = torch.minimum(distance, distance[:, middle:middle+1] + distance[middle:middle+1, :])
    distance = torch.where(distance.isfinite(), distance, n).long()
    closed = adjacency + torch.eye(n)
    inv = closed.sum(-1).rsqrt()
    symmetric = inv[:, None] * closed * inv[None, :]
    return torch.stack(returns, -1), distance, symmetric


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(16), nn.LayerNorm(16)
        self.qkv = nn.Linear(16, 48)
        self.out = nn.Linear(16, 16)
        self.ff = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))

    def forward(self, hidden, bias):
        n = len(hidden)
        q, k, value = self.qkv(self.norm1(hidden)).reshape(n, 3, 2, 8).unbind(1)
        scores = torch.einsum('ihd,jhd->hij', q, k) / np.sqrt(8)
        weights = (scores + bias).softmax(-1)
        mixed = torch.einsum('hij,jhd->ihd', weights, value).reshape(n, 16)
        hidden = hidden + self.out(mixed)
        return hidden + self.ff(self.norm2(hidden)), weights


class Model(nn.Module):
    def __init__(self, kind, n):
        super().__init__()
        self.kind = kind
        self.embed = nn.Linear(7 if kind == 'walk' else 3, 16)
        self.head = nn.Linear(16, 2)
        if kind != 'gcn':
            self.blocks = nn.ModuleList([AttentionBlock(), AttentionBlock()])
        if kind == 'distance':
            self.bias = nn.Embedding(n + 1, 2)
            nn.init.zeros_(self.bias.weight)

    def forward(self, features, structure):
        returns, distance, symmetric = structure
        x = torch.cat([features, returns], -1) if self.kind == 'walk' else features
        if self.kind == 'gcn':
            hidden = (symmetric @ (x @ self.embed.weight.T) + self.embed.bias).relu()
            return symmetric @ (hidden @ self.head.weight.T) + self.head.bias, None
        hidden = self.embed(x)
        bias = self.bias(distance).permute(2, 0, 1) if self.kind == 'distance' else 0
        for block in self.blocks:
            hidden, weights = block(hidden, bias)
        return self.head(hidden), weights


def run():
    raw = json.loads((HERE / 'karate-club.json').read_text())
    n = len(raw['nodes'])
    adjacency = torch.zeros(n, n)
    for left, right, _ in raw['edges']:
        adjacency[left, right] = adjacency[right, left] = 1
    features = torch.tensor([[1., row['degree'] / 33., row['clustering']] for row in raw['nodes']])
    labels = torch.tensor([row['club'] for row in raw['nodes']])
    rng = np.random.default_rng(133)
    roles = {key: [] for key in ['fit', 'development', 'assessment']}
    for label in range(2):
        ids = rng.permutation(np.flatnonzero(labels.numpy() == label))
        for key, part in zip(roles, [ids[:5], ids[5:8], ids[8:]]):
            roles[key].extend(int(i) for i in part)
    roles = {key: sorted(value) for key, value in roles.items()}
    assert len(set(sum(roles.values(), []))) == n
    structure = structural_inputs(adjacency)
    permutation = torch.tensor(np.random.default_rng(8).permutation(n))
    permuted_structure = structural_inputs(adjacency[permutation][:, permutation])
    edited = adjacency.clone(); edited[0, 1] = edited[1, 0] = 0
    edited_structure = structural_inputs(edited)
    records = []
    for kind in ['gcn', 'set', 'distance', 'walk']:
        for seed in [11, 29, 47]:
            torch.manual_seed(seed)
            model = Model(kind, n)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
            history = []
            for epoch in range(1, 301):
                optimizer.zero_grad()
                logits, _ = model(features, structure)
                loss = nn.functional.cross_entropy(logits[roles['fit']], labels[roles['fit']])
                loss.backward(); optimizer.step()
                if epoch in [1, 10, 50, 100, 300]:
                    history.append({'epoch': epoch, 'pre_update_fit_loss': float(loss.detach())})
            with torch.no_grad():
                logits, weights = model(features, structure)
                permuted, _ = model(features[permutation], permuted_structure)
                error = float((permuted - logits[permutation]).abs().max())
                assert error < 1e-4
                changed, _ = model(features, edited_structure)
                probability, altered = logits.softmax(-1), changed.softmax(-1)
                record = {'kind': kind, 'seed': seed, 'parameters': sum(p.numel() for p in model.parameters()),
                          'correct': {key: int((logits[ids].argmax(-1) == labels[ids]).sum()) for key, ids in roles.items()},
                          'probabilities': probability.tolist(), 'edge_removed_probabilities': altered.tolist(),
                          'permutation_logit_error': error, 'history': history}
                if seed == 11:
                    record['state_dict'] = {key: value.tolist() for key, value in model.state_dict().items()}
                    record['last_attention'] = None if weights is None else weights.tolist()
                records.append(record)
    result = {'protocol': {'epochs': 300, 'learning_rate': .003, 'weight_decay': .01,
                          'seeds': [11, 29, 47], 'edge_edit': [0, 1], 'structural_features_fixed_on_edit': True,
                          'torch': torch.__version__, 'numpy': np.__version__}, 'roles': roles,
              'features': features.tolist(), 'walk_returns': structure[0].tolist(), 'distances': structure[1].tolist(),
              'records': records}
    (HERE / 'calculated-inputs.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([{key: row[key] for key in ['kind', 'seed', 'parameters', 'correct', 'permutation_logit_error']} for row in records], indent=2))


if __name__ == '__main__':
    run()
~~~

### Read the actual result

| Model | Seed | Parameters | Fit /10 | Development /6 | Assessment /18 |
| --- | --- | ---: | ---: | ---: | ---: |
| gcn | 11 | 98 | 9 | 2 | 9 |
| gcn | 29 | 98 | 9 | 2 | 9 |
| gcn | 47 | 98 | 8 | 2 | 10 |
| set | 11 | 4546 | 10 | 1 | 10 |
| set | 29 | 4546 | 10 | 1 | 10 |
| set | 47 | 4546 | 10 | 1 | 10 |
| distance | 11 | 4616 | 10 | 1 | 9 |
| distance | 29 | 4616 | 10 | 1 | 9 |
| distance | 47 | 4616 | 10 | 1 | 9 |
| walk | 11 | 4610 | 10 | 2 | 10 |
| walk | 29 | 4610 | 10 | 2 | 9 |
| walk | 47 | 4610 | 10 | 2 | 10 |

Always predicting one affiliation scores 9/18 on this balanced assessment. Most models fit their ten supervised labels perfectly yet score only 9 or 10 on assessment. More parameters, global access and extra structural features do not establish improved generalization here. The preceding lesson's fixed label-propagation baseline scored 16/18 under the same graph/label split; it remains a useful practical comparison, not a reason to tune these networks against exposed assessment answers.

The code also removes edge (0,1) after fitting, holds the original degree/clustering inputs fixed and recomputes each model's explicit graph operators. This isolates propagation, distances or return-feature paths without claiming to erase every trace of topology. The set-attention model is exactly unchanged because it receives the same feature table and no other graph input. The other changes can be small: for the seed-11 distance model, the largest assessment probability change is about .0001265. Sensitivity is not guaranteed to be large merely because a pathway exists.

**Real investigation: compare an input path with an actual decision.** Inspect source IDs, label roles, two attention-head matrices and class probabilities for a selected assessment node. Predict the result of semantic relabeling, then run the retained seed-11 model with matching row/column permutations. For an edge edit, choose whether original structural features stay fixed or are recomputed, and explain which intervention the result answers. Show tiny numerical changes honestly; the earlier exact path offers a large visible contrast without inventing one for a trained model.

## 9. Deeper: choose outputs that match geometry and physics

### Chirality and reflection

Distances alone cannot distinguish a configuration from its mirror image. For four ordered, distinguishable points define

\[
\chi=\det[x_1-x_0,\;x_2-x_0,\;x_3-x_0].
\]

Translation cancels; a proper rotation preserves χ; a reflection reverses its sign. The unit-axis tetrahedron gives +1 and its x-reflection gives −1. This is six times oriented volume, with cubed-length units. Swapping two of the ordered points also reverses the sign. A chirality feature therefore needs chemically meaningful roles or a suitable permutation-consistent construction, not arbitrary index ordering.

A reflection-sensitive target may need information that an E(3)-invariant scalar model discards. But not every molecular property should distinguish enantiomers in every environment. A chiral environment must be represented when its interaction matters. Requiring a physically incorrect symmetry can make the target impossible to learn; omitting a useful symmetry can increase the burden on data. Evaluate the specific target and inputs.

### An energy-derived force adds another contract

If a differentiable scalar energy E(X) is invariant under rigid motions, forces defined by Fᵢ=−∇ₓᵢE transform as polar vectors. The chain rule establishes their rotational behavior. Translation invariance additionally gives zero total internal force when all relevant positions are included. Rotational invariance similarly constrains total torque under the smooth isolated-system assumptions.

Use a constructed spring-like energy E=½Σᵢ<ⱼ‖xᵢ−xⱼ‖², in declared arbitrary units. For our triangle E=5, and the forces are (1,2,0), (−2,2,0), (1,−4,0). They sum to zero. The program checks them against central finite differences, with maximum error about 3.8×10⁻¹¹ at its stated step. This is a mathematical force example, not a fitted molecular potential.

A directly predicted equivariant vector field need not be the gradient of a scalar energy and need not conserve energy during simulation. Even exact energy gradients require an appropriate numerical integrator; equivariance is not a substitute for conservation, stability or empirical validation.

### Beyond scalar and vector states

A three-component vector rotates through a 3×3 representation matrix. More general features can transform in irreducible blocks: for SO(3), a degree-ℓ block has 2ℓ+1 components. ℓ=0 is a scalar, ℓ=1 is vector-like and ℓ=2 can encode a symmetric traceless rank-two tensor with five independent components. A general 3×3 matrix has nine components and is not just one ℓ=2 block.

[SE(3)-Transformers](https://arxiv.org/abs/2006.10503) combine invariant attention weights with appropriately transforming features. Spherical harmonics encode directions; tensor-product rules combine representation types without treating arbitrary Cartesian components as unrelated scalars. [e3nn's representation documentation](https://docs.e3nn.org/en/stable/api/o3/o3_irreps.html) makes rotation degree and reflection parity explicit. These are additional representational tools, not a universal speed or accuracy ranking over EGNN.

## 10. Applications and engineering decisions

**Molecular property prediction:** graph connectivity describes bonds; geometric inputs describe conformations. Hold out appropriate molecules or chemical families when the intended claim concerns new chemistry. Near-duplicate conformers across splits can exaggerate generalization. Record target units, whether coordinates are available at prediction time and whether the property is extensive or intensive. A model requiring optimized coordinates cannot silently be compared with one receiving only a bond graph.

**Protein frames:** a residue can carry a local coordinate frame. Invariant point attention compares points after placing them in a common global frame; applying one common rigid motion preserves their pairwise distances. Pulling a global weighted point back into a residue's local frame supplies a frame-consistent local quantity. This helps explain the structure module discussed in the [geometric-graphs chapter's AlphaFold 2 case study](https://geometricdeeplearning.com/book/geometricgraphs.html). It is a design principle, not a claim that this small lesson implements protein folding or that all folding architectures are interchangeable.

**Sensor and robotic geometry:** an output displacement should rotate when the entire sensor frame rotates. Yet gravity, walls, camera calibration and fixed world directions can be meaningful inputs. If those are held fixed while the scene rotates, the task may no longer have the same symmetry. Transform the whole relevant input representation or state exactly what remains anchored.

### Count the resource, not a universal node cutoff

Materializing H attention heads for one N-node graph in b-byte precision requires HN²b bytes for that tensor. With eight heads and float32, N=1,000 gives 32,000,000 bytes; N=10,000 gives 3,200,000,000 bytes, before other activations or gradients. These are decimal MB/GB, not MiB/GiB. Multiple layers and a batch add more tensors, but exact peak memory depends on saved intermediates and execution strategy.

Exact tiled attention can avoid retaining the entire score matrix while still performing dense pair interactions. Structural bias tensors and shortest-path preprocessing can remain quadratic even when the attention kernel is memory-efficient. Sparse masks change the allowed computation; kernel-feature methods change or approximate the attention operator; pooling changes the representations. Attention on a coarsened graph is not generally identical to attention on the original graph.

For local geometric layers, costs depend on edge count, MLP widths, radial features and representation channels. An edge MLP can cost O(d²), so “O(E·d)” is not a complete universal layer estimate. There is no fixed node threshold or fixed EGNN-to-tensor-network speed ratio that settles all choices.

The current [PyG GPSConv documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GPSConv.html) places positional/structural encodings in preprocessing and exposes local/global components. An API does not infer the intended encodings or certify the whole action contract for you. Check mask, graph batching, normalization, edge feature and representation conventions with small known examples before scaling.

## 11. Practice with new inputs

### 1. A changed remote message — core

Use the path attention weights (4/7,2/7,1/7), but values (2,0,7). Compute the global output and the one-hop masked output for receiver 0. Explain the difference.

<details><summary>Hint</summary>
The local read renormalizes the first two weights; it does not simply discard a contribution without renormalizing.
</details>
<details><summary>Solution</summary>
Global output is 15/7. The local weights are (2/3,1/3,0), giving 4/3. The remote value enters globally in one layer; local access forbids that route.
</details>

### 2. Count walks and probabilities — core

At the center of a three-node path, what are (A²)₁₁ and (T²)₁₁ without self-loops? Explain why they differ.

<details><summary>Hint</summary>
There are two choices on the first step, but each endpoint has only one return neighbor.
</details>
<details><summary>Solution</summary>
There are two length-two returning walks, so the count is 2. Each has probability 1/2×1, so total return probability is 1. A probability feature requires degree normalization.
</details>

### 3. Find a concealed symmetry failure — core

Use square readings (1,0,0,0) and W=R from §5. Compare SRx and RSx. Why would the all-ones input have missed the bug?

<details><summary>Hint</summary>
Track the single nonzero reading through the two operations.
</details>
<details><summary>Solution</summary>
SRx=(0,0,0,1), while RSx=(0,1,0,0). A constant reading vector is fixed by both permutations, so it supplies no distinguishing input. A matrix commutation identity, rather than that one passing input, establishes an all-input linear contract.
</details>

### 4. Choose the output action — core

Rotate a point cloud by 90° around z and translate it by (3,4,0). How should a predicted point (1,0,0), a displacement (1,0,0) and an invariant scalar 5 transform?

<details><summary>Hint</summary>
Translations act on positions, not on free displacement vectors.
</details>
<details><summary>Solution</summary>
The point becomes (3,5,0), the displacement becomes (0,1,0), and the scalar stays 5. Applying the same numerical transformation to every output type would be wrong.
</details>

### 5. Interpret the real comparison — core

A distance-biased model fits 10/10 labels and scores 9/18 on assessment. The set model scores 10/18. Does this establish that graph distances hurt learning? Propose one useful follow-up without tuning repeatedly on those assessment answers.

<details><summary>Hint</summary>
Separate this small fixed protocol from a general architectural claim.
</details>
<details><summary>Solution</summary>
It establishes the displayed outcomes for these models, features, labels and training budget. Investigate fitting/development behavior, input features and errors, then evaluate a justified new comparison on fresh appropriate held-out data. Preserve a strong simple baseline. Neither one-point difference nor perfect fitting supports a universal ranking.
</details>

### 6. Repair an eigenspace claim — deeper

A connected graph has a repeated nonzero Laplacian eigenvalue. A colleague flips each eigenvector so its first nonzero entry is positive and declares all ambiguity removed. Give the missing transformation and a basis-invariant representation of the entire eigenspace.

<details><summary>Hint</summary>
An orthogonal rotation can mix the columns while preserving the same subspace.
</details>
<details><summary>Solution</summary>
U and UQ are both valid orthonormal bases for any orthogonal Q within the repeated eigenspace. Sign fixing does not remove that freedom and an index-based sign rule can itself conflict with relabeling. UUᵀ is unchanged by U→UQ; under node permutation it becomes PUUᵀPᵀ. A truncated portion of a repeated eigenspace is not the same complete object.
</details>

### 7. Verify a force and its units — deeper

For two points (0,0,0) and (2,0,0), use E=½‖x₀−x₁‖² with unit spring coefficient. Compute E and both forces. What happens under a common translation? What extra claim does a conservative energy-derived force make beyond equivariance?

<details><summary>Hint</summary>
Differentiate with respect to each point, with the force defined as the negative gradient.
</details>
<details><summary>Solution</summary>
E=2; forces are (2,0,0) and (−2,0,0). Common translation changes neither differences, energy nor forces. A force derived from a smooth scalar potential has the corresponding conservative structure; a general equivariant vector predictor need not. For physical units, the spring coefficient carries energy/length² and force has energy/length units. Numerical time stepping still needs its own accuracy/stability analysis.
</details>

### 8. Build a complete action test — deeper

Design a test for a geometric graph model with scalar node states, coordinates and vector outputs. Include a genuine null, a failing control and one independent prediction-quality test.

<details><summary>Hint</summary>
Transform input and output representations correctly, and do not confuse algebraic agreement with a useful prediction.
</details>
<details><summary>Solution</summary>
Permute nodes and both adjacency axes, and apply a valid orthogonal transformation plus translation to coordinates. Scalars follow nodes; vector outputs rotate without translation. Compare both routes with a scale-aware tolerance and check the transformation matrix itself. Identity is a null; a coordinatewise ReLU or axis-dependent edge rule supplies a controlled failing case on a non-symmetric input. Separately score appropriate held-out targets with declared units and split boundaries. A zero or constant predictor can pass a symmetry check while predicting poorly.
</details>

## References and another way to learn

- [Geometric Deep Learning book and free draft chapters](https://geometricdeeplearning.com/book/): Chapters 3,5 and 8 connect actions, graphs and geometric graphs. Use the chapter route matching your current hurdle; the whole book is not a prerequisite for this lesson.
- [Graphormer](https://arxiv.org/pdf/2106.05234), §3: compare the three structural inputs with our deliberately smaller distance-bias model. [GPS](https://arxiv.org/pdf/2205.12454), §3/Figure 1: trace local and global branches and note how its encoding choices also help message passing.
- [SignNet and BasisNet](https://arxiv.org/html/2202.13013v4), §2: follow the sign-versus-basis distinction after reproducing the four-cycle projector.
- [EGNN](https://proceedings.mlr.press/v139/satorras21a/satorras21a.pdf), §3 and Appendix A: read the invariant messages and coordinate proof together, then identify the assumptions in your own code.
- [SE(3)-Transformers](https://arxiv.org/abs/2006.10503) and [e3nn representations](https://docs.e3nn.org/en/stable/api/o3/o3_irreps.html): a deeper route into feature types and reflection parity.
- [The authors' GDL lecture recordings and exercises](https://geometricdeeplearning.com/lectures/): Geometric Priors I/II and Graphs & Sets I/II offer a spoken/visual alternative; the site links each YouTube recording and accompanying material. The author course page and lesson associations were inspected; the full recordings were not watched for this packet.

Next in this module is [Boltzmann Machines & Restricted Boltzmann Machines](/learn/path/full-curriculum/boltzmann-machines-restricted-boltzmann-machines-rbm?module=deep-learning-fundamentals). That changes the modeling question: instead of arranging information routes to predict a target, we assign probabilities to configurations through an energy function. Keep the distinction between an architectural property, an optimization procedure and empirical evidence—it will matter there too.
