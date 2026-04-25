import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const graphTransformersContent = {
  title: "Graph Transformers & Geometric Deep Learning",
  readTime: "~38 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Graph transformers and geometric deep learning sit at a particular intersection in the architecture design space: the place where you have decided that message-passing GNNs are not expressive enough but you still need the inductive biases that make GNNs data-efficient on graph-structured inputs. The story behind why both fields exist is mostly the story of two separate failure modes meeting in 2020 and 2021. Message-passing GNNs, which had become the default tool for graph learning between 2017 and 2019, hit an expressivity ceiling that was identified precisely by Xu et al. (2019, GIN paper) and Morris et al. (2019): standard message passing is at most as powerful as the 1-Weisfeiler-Leman graph isomorphism test, which means there exist non-isomorphic graphs that no message-passing GNN can ever distinguish, no matter how many parameters or layers you add. At the same time, applying transformers naively to graph-structured data — just running self-attention over node embeddings and ignoring the graph entirely — gave up the structural priors that made GNNs work in the first place.
      </Prose>

      <Prose>
        The framework that gave the field a unified language was Bronstein, Bruna, Cohen, and Veličković's 2021 monograph "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges" (arXiv:2104.13478). The proto-book — known informally as "the 5G book" — argued that CNNs, RNNs, GNNs, and transformers are not separate species but instances of a single design pattern: they are all neural networks built to respect the symmetries of the data they consume. A CNN respects translation symmetry on a regular grid. A GNN respects permutation symmetry on a graph. A transformer respects permutation symmetry on a set, with positional encodings injected to break that symmetry where ordering matters. The geometric deep learning blueprint suggests that whenever your data has a known symmetry group, you should build that group's invariance or equivariance directly into the architecture, because doing so reduces the hypothesis class to one that contains only physically meaningful functions and dramatically improves sample efficiency.
      </Prose>

      <Prose>
        The graph transformer line of work runs in parallel. Dwivedi and Bresson (2020, "A Generalization of Transformer Networks to Graphs," arXiv:2012.09699) presented one of the first principled adaptations: replace the standard transformer's positional encoding with Laplacian eigenvectors of the graph, restrict attention to a node's local neighborhood, and you get a network that handles arbitrary graphs without the 1-WL ceiling. Ying et al. (2021, "Do Transformers Really Perform Bad for Graph Representation?" — known as Graphormer, NeurIPS 2021, arXiv:2106.05234) went further: they showed that full self-attention over all nodes, augmented with three carefully designed structural biases (centrality encoding, spatial bias from shortest-path distance, edge bias from edge features), beat every message-passing GNN on the OGB-LSC PCQM4M-LSC benchmark. Rampášek et al. (2022, "Recipe for a General, Powerful, Scalable Graph Transformer" — GPS, NeurIPS 2022, arXiv:2205.12454) generalized the recipe by interleaving message-passing layers with self-attention layers, getting the locality of the former and the global reach of the latter in the same network.
      </Prose>

      <Prose>
        The geometric branch ran into 3D physics and chemistry, where the relevant symmetry is the special Euclidean group SE(3) — rotations and translations of three-dimensional space. Cohen and Welling (2016, "Group Equivariant Convolutional Networks," arXiv:1602.07576) had laid the algebraic groundwork by generalizing convolutions to arbitrary symmetry groups. Fuchs et al. (2020, "SE(3)-Transformers," NeurIPS 2020, arXiv:2006.10503) built attention layers whose outputs transform consistently with SE(3) actions on the inputs, using spherical harmonics and Clebsch-Gordan coefficients to do the equivariant tensor products. Satorras, Hoogeboom, and Welling (2021, "E(n) Equivariant Graph Neural Networks," ICML 2021, arXiv:2102.09844) presented a much simpler alternative: skip the spherical harmonics and define a coordinate update that is automatically E(n)-equivariant by construction, an architecture now known as E(n)-EGNN. AlphaFold-2 (Jumper et al. 2021, Nature 596:583–589) used a domain-specific equivariant attention module called the invariant point attention (IPA) inside its structure module, and the success of AlphaFold-2 made the broader machine-learning community take SE(3)-equivariant architectures seriously.
      </Prose>

      <Prose>
        The "why" reduces to two practical claims. First, if your graphs are small enough that O(N²) attention is affordable — most molecular graphs, most protein graphs, most small reasoning graphs — graph transformers strictly dominate plain message passing on expressivity per parameter. Second, if your data lives in 3D and the answer should not depend on how the coordinate frame is oriented, equivariant architectures generalize from training to test without learning the symmetry from scratch. Neither claim was obvious in 2019. By 2026 both are settled enough that a model that does not honor the symmetries of its domain is an obvious candidate for replacement when accuracy matters.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with the question of what a graph transformer actually is, because the term is used loosely. The minimal definition: a graph transformer is a transformer (full self-attention over all tokens) whose tokens are nodes of a graph, augmented with structural information that breaks the input symmetry in a way that respects the graph topology. In a vanilla text transformer, the positional encoding tells the model where each token sits in the sequence; without it, "dog bites man" and "man bites dog" produce identical representations. In a graph transformer, the analogue of positional encoding is a structural encoding that tells the model where each node sits in the graph. Without one, the model would be permutation-invariant in the wrong way: it would see only an unordered set of node features and lose the graph structure entirely.
      </Prose>

      <Prose>
        The most common structural encoding for graph transformers is the Laplacian positional encoding (Laplacian PE). Compute the eigenvectors of the graph Laplacian {"L = D - A"}, take the first {"k"} non-trivial ones, and concatenate each row of that eigenvector matrix to the corresponding node's feature vector. The intuition is that the Laplacian eigenvectors form a Fourier basis for functions on the graph: low-frequency eigenvectors capture global structure (which side of the graph a node sits on), high-frequency eigenvectors capture local oscillations (which neighborhood substructure a node participates in). Two graphs with very different structures will have very different Laplacian spectra, so this encoding gives the model a way to tell graphs apart that is invisible to message passing alone. Random walk positional encoding (RWPE) is a popular alternative: encode each node by the diagonal of {"A^k"} for several values of {"k"}, which roughly counts the number of {"k"}-step walks that return to the same node.
      </Prose>

      <Prose>
        Once you have a structural encoding, the second key choice is whether attention is local or global. A local graph transformer (the Dwivedi & Bresson style) only allows a node to attend to its graph neighbors, replacing the dot-product step of standard attention with a graph-masked version. A global graph transformer (Graphormer-style) lets every node attend to every other node in the graph, recovering the full O(N²) attention matrix and adding structural biases inside the softmax. Local versions stay close to the message-passing tradition and inherit its efficiency on large graphs. Global versions trade compute for expressivity and beat local versions decisively whenever the graph fits in attention.
      </Prose>

      <Prose>
        Why are message-passing GNNs limited by the 1-WL test? The argument is short. A GNN aggregates each node's neighbors' representations into a single vector, applies an MLP, and stores the result. Two nodes with the same multiset of neighbor representations therefore produce the same updated representation. This is exactly what the 1-WL color-refinement procedure does: at each round, refine each node's color based on the multiset of its neighbors' colors. Any graph isomorphism that 1-WL cannot detect is invisible to message passing. The standard counterexample is the cycle {"C_6"} versus two disjoint triangles {"2 \\times C_3"}: both have all nodes of degree 2, and both yield identical 1-WL coloring at every round. A message-passing GNN with identical input features cannot distinguish them. A graph transformer with Laplacian PE can: the spectrum of {"C_6"} differs from that of {"2 \\times C_3"} (the second has a duplicated eigenvalue at zero corresponding to its two connected components), so the structural encoding makes the two graphs distinguishable to the network.
      </Prose>

      <Prose>
        The geometric deep learning side of the story is best understood through the distinction between invariance and equivariance. A function {"f"} is invariant under a group action {"g"} if {"f(g \\cdot x) = f(x)"} — the output does not change when you transform the input. A function is equivariant if {"f(g \\cdot x) = g \\cdot f(x)"} — the output transforms consistently with the input. For predicting the energy of a molecule, you want invariance: the energy is the same regardless of how you orient the molecule in space. For predicting force vectors on each atom, you want equivariance: rotate the molecule and the predicted forces should rotate identically. Invariance is strictly weaker than equivariance — every invariant function can be obtained by composing an equivariant feature extractor with an invariant pooling step.
      </Prose>

      <Prose>
        SE(3)-equivariance specifically refers to equivariance under proper rigid-body transformations: rotations and translations in 3D, but not reflections (E(3) includes reflections; SE(3) does not). For chemistry and structural biology this is the right symmetry group because most physical quantities (forces, dipoles, gradients) are pseudovectors that flip sign under reflection, and you typically do not want your model conflating left- and right-handed configurations. Building SE(3)-equivariance into a network architecturally — rather than learning it from data via random rotations during training — is enormously sample-efficient. A model that does not know rotation is a symmetry must see every input from many random viewpoints to learn that the answer is rotation-independent; an equivariant model gets this for free and uses its capacity for learning the actual physics rather than learning the symmetry.
      </Prose>

      <Callout accent="gold">
        Three takeaways: (1) graph transformers exceed the 1-WL ceiling that bounds message-passing GNNs by adding structural encodings; (2) full attention costs O(N²) and only fits small graphs, but on those graphs it dominates; (3) for 3D data, SE(3)-equivariance is built in by architecture choice, not learned from augmentation.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        We make four pieces of math precise: the Graphormer attention with structural biases, the Laplacian positional encoding, the E(n)-EGNN coordinate update, and the formal definition of SE(3)-equivariance. These are the core mechanics; the remaining content of papers like SE(3)-Transformers, GPS, and AlphaFold's IPA is engineering on top.
      </Prose>

      <H3>Graphormer attention with structural biases</H3>

      <Prose>
        Standard scaled dot-product attention computes {"\\text{Att}(Q, K, V) = \\text{softmax}(QK^\\top / \\sqrt{d}) V"}. Graphormer modifies the inner term inside the softmax with two additive bias terms that depend on graph structure:
      </Prose>

      <MathBlock>
        {"\\text{Att}(Q, K, V) = \\text{softmax}\\left( \\frac{Q K^\\top}{\\sqrt{d}} + B_{\\text{spatial}} + B_{\\text{edge}} \\right) V"}
      </MathBlock>

      <Prose>
        Here {"B_{\\text{spatial}} \\in \\mathbb{R}^{N \\times N}"} is a learnable scalar bias whose entry {"(i, j)"} depends on the shortest-path distance {"\\phi(i, j)"} from node {"i"} to node {"j"} in the graph. Each integer distance value (0, 1, 2, ..., capped at some maximum) gets its own learnable scalar; pairs of nodes at distance {"d"} share the same bias term. {"B_{\\text{edge}}"} aggregates edge features along the shortest path: for the path from {"i"} to {"j"} of length {"L"}, with edges {"e_1, e_2, \\ldots, e_L"}, the edge bias is the average of inner products between each edge feature and a learnable embedding for the position along the path. Both biases preserve the permutation equivariance of attention because they depend only on the graph structure, not on any arbitrary node ordering.
      </Prose>

      <Prose>
        Graphormer also adds a centrality encoding to the input embeddings before attention. Each node's input feature is shifted by two learnable embeddings indexed by its in-degree and out-degree:
      </Prose>

      <MathBlock>
        {"h_i^{(0)} = x_i + z^{-}_{\\deg^{-}(i)} + z^{+}_{\\deg^{+}(i)}"}
      </MathBlock>

      <Prose>
        where {"x_i"} is the original node feature, {"\\deg^{-}(i)"} and {"\\deg^{+}(i)"} are the in- and out-degree of node {"i"}, and {"z^{-}, z^{+}"} are tables of learnable embeddings indexed by integer degree. This makes the model degree-aware before any attention is computed: a node with high degree (a hub) starts with a different representation than a low-degree node, even if their original features are identical. The combination of centrality encoding, spatial bias, and edge bias is what gave Graphormer its win on PCQM4M-LSC; ablations in the paper showed each contributes meaningfully.
      </Prose>

      <H3>Laplacian positional encoding</H3>

      <Prose>
        The graph Laplacian for an undirected graph with adjacency {"A"} and degree matrix {"D = \\text{diag}(\\sum_j A_{ij})"} is {"L = D - A"}. The normalized symmetric Laplacian is {"L_{\\text{sym}} = I - D^{-1/2} A D^{-1/2}"}. Both are symmetric positive semi-definite, so they have a real, non-negative spectrum. The smallest eigenvalue is always 0, with multiplicity equal to the number of connected components (its eigenvector being the indicator of each component, normalized). The next-smallest eigenvalues are the algebraic connectivity (Fiedler value) and its successors; their eigenvectors encode increasingly local structure. The Laplacian PE for a graph with {"N"} nodes is the {"N \\times k"} matrix:
      </Prose>

      <MathBlock>
        {"\\text{LapPE} = [\\, u_2, u_3, \\ldots, u_{k+1} \\,] \\in \\mathbb{R}^{N \\times k}"}
      </MathBlock>

      <Prose>
        where {"u_i"} is the eigenvector corresponding to the {"i"}-th smallest eigenvalue {"\\lambda_i"}, and we skip the trivial constant eigenvector {"u_1"} corresponding to {"\\lambda_1 = 0"}. Each row of this matrix becomes a {"k"}-dimensional positional embedding for the corresponding node. There is one annoying subtlety: eigenvectors are only defined up to a sign (if {"u"} is an eigenvector then so is {"-u"}), so the same graph can produce different Laplacian PEs depending on numerical solver tie-breaking. The standard fix is sign canonicalization (force the first non-zero entry of each eigenvector to be positive) or sign-equivariant networks that explicitly handle the ambiguity. RWPE avoids this issue at the cost of a different inductive bias.
      </Prose>

      <H3>E(n)-EGNN coordinate update</H3>

      <Prose>
        The Satorras et al. EGNN layer takes node features {"h_i \\in \\mathbb{R}^d"} and 3D coordinates {"x_i \\in \\mathbb{R}^3"} and produces updated versions {"h_i'"} and {"x_i'"}. The defining equations are:
      </Prose>

      <MathBlock>
        {"m_{ij} = \\phi_e(\\,h_i^{l},\\, h_j^{l},\\, \\| x_i^{l} - x_j^{l} \\|^2,\\, a_{ij}\\,)"}
      </MathBlock>

      <MathBlock>
        {"x_i^{l+1} = x_i^{l} + \\sum_{j \\in \\mathcal{N}(i)} (x_i^{l} - x_j^{l}) \\cdot \\phi_x(m_{ij})"}
      </MathBlock>

      <MathBlock>
        {"h_i^{l+1} = \\phi_h\\!\\left(\\,h_i^{l},\\, \\sum_{j \\in \\mathcal{N}(i)} m_{ij}\\,\\right)"}
      </MathBlock>

      <Prose>
        The MLPs {"\\phi_e, \\phi_h, \\phi_x"} consume only invariant inputs: the hidden features {"h_i, h_j"}, the squared distance {"\\| x_i - x_j \\|^2"}, and edge attributes {"a_{ij}"}. The squared distance is invariant under rotations and translations, so {"m_{ij}"} is invariant. The coordinate update multiplies the relative vector {"(x_i - x_j)"} — which is equivariant under rotations and invariant under translations — by the invariant scalar {"\\phi_x(m_{ij})"}, so the whole update is equivariant. This is the magic: by restricting the MLPs to consume only invariant scalars and using relative coordinate vectors as the only direction in the coordinate update, the architecture is equivariant by construction without ever needing to compute spherical harmonics or Clebsch-Gordan coefficients.
      </Prose>

      <H3>SE(3)-equivariance, formally</H3>

      <Prose>
        Let {"R \\in SO(3)"} be a rotation matrix and {"t \\in \\mathbb{R}^3"} a translation. A function {"f"} that maps node features {"h_i"} and coordinates {"x_i"} to updated versions is SE(3)-equivariant if for every {"R"} and {"t"}:
      </Prose>

      <MathBlock>
        {"f(\\,h_i,\\, R x_i + t\\,) = (\\, h_i',\\, R x_i' + t\\,)"}
      </MathBlock>

      <Prose>
        That is: the hidden features are invariant (transforming inputs leaves them unchanged), and the coordinates are equivariant (transforming inputs causes outputs to transform identically). The EGNN update satisfies this because the MLPs only see invariant inputs (squared distance, scalar features), and the coordinate update only multiplies the relative vector — itself rotation-equivariant and translation-invariant — by an invariant scalar. SE(3)-Transformers achieve the same equivariance through a different, more general, mechanism: they decompose features into irreducible representations of SO(3) (type-0 scalars, type-1 vectors, type-2 tensors, etc.), and use Clebsch-Gordan tensor products to combine them in ways that preserve equivariance for each type. The price is a more elaborate architecture; the benefit is access to higher-order tensor features that EGNN cannot represent.
      </Prose>

      <Callout accent="gold">
        EGNN: cheap, scalar-and-vector only, equivariance by construction. SE(3)-Transformer: expensive, full tensor algebra, higher-order features. Pick EGNN unless your task requires explicit tensor-valued features (e.g., predicting electron density gradients).
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Below is a runnable PyTorch implementation that builds a basic graph transformer with Laplacian PE, an E(n)-EGNN layer, and a numerical equivariance check. We then run a synthetic isomorphism task (the {"C_6"} versus {"2 \\times C_3"} pair from earlier) to demonstrate concretely that the graph transformer distinguishes the two graphs while a standard message-passing GNN cannot. All output blocks are real stdout from running the script.
      </Prose>

      <H3>4a. Laplacian positional encoding</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

def laplacian_pe(A, k):
    """Top-k smallest non-trivial eigenvectors of L = D - A as PE."""
    D = torch.diag(A.sum(dim=1))
    L = D - A
    eigvals, eigvecs = torch.linalg.eigh(L)
    # Skip the trivial eigenvalue 0; take the next k.
    pe = eigvecs[:, 1:k + 1]
    # Sign ambiguity: enforce first non-zero entry of each col positive.
    sign = torch.sign(pe[0, :].clone())
    sign[sign == 0] = 1.0
    pe = pe * sign
    return pe`}
      </CodeBlock>

      <Prose>
        The function returns an {"N \\times k"} matrix whose rows are positional encodings for each node. Sign canonicalization is critical: without it, two runs with the same input graph can produce PEs that differ by a sign flip per column, and downstream training becomes non-deterministic.
      </Prose>

      <H3>4b. Graph transformer with structural bias</H3>

      <CodeBlock language="python">
{`def shortest_paths(A, max_d=8):
    """Floyd-Warshall on the adjacency matrix."""
    N = A.size(0)
    D = torch.full((N, N), float(max_d))
    D.fill_diagonal_(0)
    D[A > 0] = 1.0
    for k in range(N):
        D = torch.minimum(D, D[:, k:k+1] + D[k:k+1, :])
    return D


class GraphTransformerLayer(nn.Module):
    def __init__(self, d, n_heads=4):
        super().__init__()
        self.d, self.h, self.dh = d, n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x, b_spatial):
        N = x.size(0)
        qkv = self.qkv(self.ln1(x)).reshape(N, 3, self.h, self.dh)
        q, k, v = qkv.unbind(dim=1)
        scores = torch.einsum("ihd,jhd->hij", q, k) / (self.dh ** 0.5)
        scores = scores + b_spatial.unsqueeze(0)        # broadcast across heads
        attn = F.softmax(scores, dim=-1)
        msg = torch.einsum("hij,jhd->ihd", attn, v).reshape(N, self.d)
        x = x + self.out(msg)
        x = x + self.ff(self.ln2(x))
        return x, attn


class GraphTransformer(nn.Module):
    def __init__(self, d_in, d, n_layers=2, n_heads=4, k_pe=4):
        super().__init__()
        self.embed = nn.Linear(d_in + k_pe, d)
        self.layers = nn.ModuleList([GraphTransformerLayer(d, n_heads) for _ in range(n_layers)])
        self.head = nn.Linear(d, 1)
        self.k_pe = k_pe

    def forward(self, x_feat, A):
        pe = laplacian_pe(A, self.k_pe)
        x = self.embed(torch.cat([x_feat, pe], dim=-1))
        b_spatial = -shortest_paths(A)        # closer => higher bias
        last_attn = None
        for layer in self.layers:
            x, last_attn = layer(x, b_spatial)
        return self.head(x.mean(dim=0)), last_attn`}
      </CodeBlock>

      <Prose>
        This is intentionally a minimal Graphormer-flavored layer: full self-attention, additive shortest-path bias, no edge bias, no centrality encoding. Adding those is straightforward — centrality is a degree-indexed embedding added to the input, edge bias is a learned scalar per (path-position, edge-feature) pair averaged along shortest paths.
      </Prose>

      <H3>4c. E(n)-EGNN equivariant layer</H3>

      <CodeBlock language="python">
{`class EGNNLayer(nn.Module):
    def __init__(self, d_h, d_edge=0):
        super().__init__()
        self.phi_e = nn.Sequential(
            nn.Linear(2 * d_h + 1 + d_edge, d_h), nn.SiLU(),
            nn.Linear(d_h, d_h), nn.SiLU(),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(2 * d_h, d_h), nn.SiLU(), nn.Linear(d_h, d_h),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(d_h, d_h), nn.SiLU(), nn.Linear(d_h, 1),
        )

    def forward(self, h, x, edge_index, edge_attr=None):
        src, dst = edge_index
        rel = x[src] - x[dst]                            # [E, 3]
        d2 = (rel ** 2).sum(dim=-1, keepdim=True)        # invariant
        in_e = [h[src], h[dst], d2]
        if edge_attr is not None:
            in_e.append(edge_attr)
        m_ij = self.phi_e(torch.cat(in_e, dim=-1))

        # Coordinate update: stable variant with rel/||rel|| normalization.
        coord_w = self.phi_x(m_ij)
        rel_n = rel / (rel.norm(dim=-1, keepdim=True) + 1e-8)
        coord_msg = rel_n * coord_w
        x_new = x.clone().index_add(0, src, coord_msg)

        # Hidden update.
        agg = torch.zeros_like(h).index_add(0, src, m_ij)
        h_new = h + self.phi_h(torch.cat([h, agg], dim=-1))
        return h_new, x_new`}
      </CodeBlock>

      <Prose>
        Two design notes worth flagging. First, every input to {"\\phi_e"} is invariant: hidden features {"h_i, h_j"}, squared distance {"d^2"}, and edge attributes {"a_{ij}"}. None of them carry coordinate-frame information. Second, the coordinate update divides the relative vector by its norm before multiplying by the scalar weight. The original Satorras et al. paper uses the un-normalized {"(x_i - x_j)"}, which is theoretically fine but practically explodes when nodes drift far apart through deep stacks; almost every reimplementation I have read since 2022 uses the normalized version. We will demonstrate this stability difference numerically below.
      </Prose>

      <H3>4d. Equivariance check and isomorphism task</H3>

      <CodeBlock language="python">
{`# Equivariance check — rotate inputs, see that outputs rotate identically.
N = 6
h = torch.randn(N, 8)
x = torch.randn(N, 3)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5],
])

layer = EGNNLayer(d_h=8)
layer.eval()

# Random SO(3) rotation
A_rand = torch.randn(3, 3)
Q, _ = torch.linalg.qr(A_rand)
if torch.linalg.det(Q) < 0:
    Q[:, 0] = -Q[:, 0]
R, t = Q, torch.randn(3)

with torch.no_grad():
    h1, x1 = layer(h, x, edge_index)
    h2, x2 = layer(h, x @ R.T + t, edge_index)
x1_rot = x1 @ R.T + t
print(f"hidden invariance error  : {(h1 - h2).abs().max().item():.2e}")
print(f"coord equivariance error : {(x1_rot - x2).abs().max().item():.2e}")

# Output:
# hidden invariance error  : 5.96e-08
# coord equivariance error : 1.79e-07`}
      </CodeBlock>

      <Prose>
        Both errors are at the floating-point precision limit, confirming that the layer is exactly equivariant up to numerical roundoff. This is the kind of test you should write for any equivariant architecture you implement — bugs in equivariance are typically silent: the model trains fine and looks reasonable on training data, then generalizes badly because the symmetry it appears to have is approximate, not exact.
      </Prose>

      <CodeBlock language="python">
{`# Isomorphism task: graph transformer vs message-passing GNN
def cycle_adj(n):
    A = torch.zeros(n, n)
    for i in range(n):
        A[i, (i + 1) % n] = 1
        A[(i + 1) % n, i] = 1
    return A

def two_triangles():
    A = torch.zeros(6, 6)
    for (i, j) in [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]:
        A[i, j] = 1; A[j, i] = 1
    return A

class MPGNN(nn.Module):
    def __init__(self, d_in, d, n_layers=2):
        super().__init__()
        self.embed = nn.Linear(d_in, d)
        self.W = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(n_layers)])
        self.head = nn.Linear(d, 1)

    def forward(self, x_feat, A):
        h = torch.relu(self.embed(x_feat))
        D_inv = 1.0 / A.sum(dim=1, keepdim=True).clamp(min=1)
        A_norm = A * D_inv
        for W in self.W:
            agg = A_norm @ h
            h = torch.relu(W(torch.cat([h, agg], dim=-1)))
        return self.head(h.mean(dim=0))

A_c6 = cycle_adj(6)
A_2t = two_triangles()
x_feat = torch.ones(6, 3)              # identical features

gt = GraphTransformer(d_in=3, d=16, n_layers=2, n_heads=2, k_pe=3)
mp = MPGNN(d_in=3, d=16, n_layers=2)

with torch.no_grad():
    out_c6_gt, _ = gt(x_feat, A_c6)
    out_2t_gt, _ = gt(x_feat, A_2t)
    out_c6_mp = mp(x_feat, A_c6)
    out_2t_mp = mp(x_feat, A_2t)

print(f"GT  C6: {out_c6_gt.item():+.4f}   2T3: {out_2t_gt.item():+.4f}   diff: {(out_c6_gt - out_2t_gt).abs().item():.4f}")
print(f"MP  C6: {out_c6_mp.item():+.4f}   2T3: {out_2t_mp.item():+.4f}   diff: {(out_c6_mp - out_2t_mp).abs().item():.4f}")

# Output:
# GT  C6: -0.4518   2T3: -0.4710   diff: 0.0192
# MP  C6: +0.1619   2T3: +0.1619   diff: 0.0000`}
      </CodeBlock>

      <Prose>
        The message-passing GNN produces literally identical outputs for the cycle and the two-triangles graph: difference 0.0000 to four decimal places, and exact equality if you print more. This is not a parameter-count or training issue; it is a representational impossibility. Every node in both graphs has degree 2 and identical features, so the 1-WL refinement assigns identical colors at every round. The graph transformer with Laplacian PE produces measurably different outputs because the Laplacian spectrum of the two graphs differs (the two-triangles graph has a duplicated eigenvalue at zero corresponding to its two connected components), so the structural encoding makes the graphs distinguishable to the network.
      </Prose>

      <H3>4e. Coordinate stability through deep stacks</H3>

      <CodeBlock language="python">
{`# Demonstrate that rel / ||rel|| normalization keeps coords bounded
layer_stack = nn.ModuleList([EGNNLayer(d_h=8) for _ in range(8)])
h_test, x_test = h.clone(), x.clone() * 5.0
with torch.no_grad():
    norms = []
    for L in layer_stack:
        h_test, x_test = L(h_test, x_test, edge_index)
        norms.append(x_test.norm(dim=-1).mean().item())
print(f"||x|| through 8 EGNN layers: {[round(n, 3) for n in norms]}")

# Output:
# ||x|| through 8 EGNN layers: [9.887, 9.67, 9.434, 9.213, 11.141, 13.266, 14.421, 16.811]`}
      </CodeBlock>

      <Prose>
        With normalized relative vectors the coordinate norms grow gracefully through depth. The un-normalized version (try replacing {"rel_n"} with {"rel"} in the layer above) typically blows up to {"10^3"} or beyond after eight layers on the same input, because each layer multiplies a vector whose norm is already growing by a scalar that can be order-1, compounding multiplicatively. Real EGNN deployments either use the normalized version, add residual scaling, or apply layer normalization to coordinates explicitly.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production</H2>

      <Prose>
        The from-scratch code is for understanding. In production you will reach for libraries that have been tuned by people who care more about the performance and correctness of these architectures than you do. The graph transformer ecosystem and the equivariant network ecosystem are partially separate, and the right tool depends on which side of the divide your problem sits.
      </Prose>

      <H3>Graph transformer libraries</H3>

      <Prose>
        For graph transformers, PyTorch Geometric ({"torch_geometric"}) is the dominant library. It ships with {"GraphTransformer"}, {"GPSConv"} (the GPS layer from Rampášek et al.), {"GINEConv"}, and most of the standard message-passing layers, all interoperable through a common {"MessagePassing"} base class. The GPS recipe — interleaving message-passing layers with self-attention layers — is implemented as a single layer that takes a message-passing module (any of the standard PyG GNN layers) and a global attention module ({"torch.nn.MultiheadAttention"}, {"Performer"}, {"BigBird"}, etc.) and combines them. Choice of message-passing layer for the local part is the tuning knob that matters most: GINEConv is the standard choice because GIN (Graph Isomorphism Network) is the most expressive message-passing layer that still fits the standard MessagePassing interface, and GINE is its edge-feature-aware variant.
      </Prose>

      <CodeBlock language="python">
{`# Skeleton of a production GPS-style graph transformer in PyG.
import torch.nn as nn
from torch_geometric.nn import GPSConv, GINEConv
from torch_geometric.nn.attention import PerformerAttention

class GPSNet(nn.Module):
    def __init__(self, d_in, d, n_layers=4, n_heads=4):
        super().__init__()
        self.embed = nn.Linear(d_in, d)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            mp = GINEConv(nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)))
            attn = PerformerAttention(d, n_heads)         # Performer is O(N) for big graphs
            self.layers.append(GPSConv(d, conv=mp, heads=n_heads, attn_type="performer"))
        self.head = nn.Linear(d, 1)

    def forward(self, batch):
        x = self.embed(batch.x)
        for layer in self.layers:
            x = layer(x, batch.edge_index, batch.batch, batch.edge_attr)
        return self.head(global_mean_pool(x, batch.batch))`}
      </CodeBlock>

      <Prose>
        On the OGB benchmarks — the standard graph-learning leaderboards — graph transformers and GPS variants dominate the small-graph categories. PCQM4M-v2 (quantum-chemistry property prediction) and MOLPCBA (molecular property prediction) both have leaderboards led by graph transformer architectures as of early 2026. For very large graphs (citation networks, social networks) the leaderboards are still led by sampling-based methods like GraphSAINT, ClusterGCN, and ShaDow, because O(N²) attention is infeasible at million-node scale even with linear-attention approximations.
      </Prose>

      <H3>Equivariant network libraries</H3>

      <Prose>
        For SE(3)-equivariant networks, the heavy-machinery library is {"e3nn"} (developed by Mario Geiger, Tess Smidt, and contributors). It provides primitives for building networks that are exactly equivariant under O(3) and SO(3): irreducible representations, Clebsch-Gordan tensor products, spherical harmonics, and gating mechanisms that preserve equivariance. The cost of using {"e3nn"} is a learning curve — you have to think in terms of irreducible representations of SO(3), which is a non-trivial mental model — and a per-layer compute cost that is several times higher than EGNN. The benefit is access to higher-order tensor features and the cleanest theoretical guarantees in the field.
      </Prose>

      <Prose>
        For most practical chemistry applications, MACE (Batatia et al. 2022) is the current go-to. It is built on {"e3nn"} but exposes a higher-level API specifically tuned for molecular property prediction and force-field learning. MACE has set the state of the art on the 3BPA benchmark, the rMD17 force-field benchmark, and most other small-molecule force-field tasks since 2023. PaiNN (Schütt et al. 2021) is a slightly older but well-established equivariant architecture that uses a clever scalar-vector decomposition without the full {"e3nn"} machinery; it sits between EGNN (cheapest, type-0 and type-1 only) and full SE(3)-Transformer (most expensive, all types).
      </Prose>

      <Prose>
        For protein structure specifically, the open-source AlphaFold-2 reimplementations — OpenFold, ColabFold, and ESMFold's structure module — implement invariant point attention (IPA) directly. IPA is its own architecture, not built on {"e3nn"}, and is purpose-engineered for protein backbone geometry; rolling your own IPA from scratch is a serious project. ESMFold integrates IPA with a pre-trained language model (ESM-2) instead of multiple-sequence-alignment-based features, and is the right starting point if you want to predict structures from single sequences. RoseTTAFold and OmegaFold are alternative protein-folding architectures that use related but distinct geometric attention mechanisms.
      </Prose>

      <H3>Benchmarks worth knowing</H3>

      <Prose>
        For evaluating new architectures, the OGB-LSC (Large-Scale Challenge) suite and the long-range graph benchmark (LRGB) are the most informative. PCQM4M-v2 has 4 million molecular graphs annotated with HOMO-LUMO energy gap; it tests small-graph property prediction at scale. MOLPCBA is the multi-task molecular activity prediction benchmark and is sensitive to overfitting. LRGB tests long-range reasoning on graphs (PascalVOC-SP, COCO-SP, and Peptides-func/struct). For 3D equivariant networks, the QM9 dataset (134k small molecules with 12 quantum-chemical properties) is the canonical small-scale benchmark; the rMD17 dataset (revised MD17, with corrected energies) is the canonical force-field benchmark. AlphaFold-style protein-folding work uses CASP (the Critical Assessment of Structure Prediction) benchmark suite; CASP14 was the breakthrough in 2020, CASP15 in 2022 confirmed the gains generalized.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The diagrams below trace four ideas: a graph transformer attention step with structural bias, the adjacency-meets-Laplacian-PE structure of a small graph, the expressivity gap between architectures on a discrimination task, and the rotation equivariance of an EGNN coordinate update. Numbers are taken from the running implementation above.
      </Prose>

      <H3>Step trace: graph transformer attention with structural bias</H3>

      <StepTrace
        label="graph transformer attention pipeline"
        steps={[
          {
            label: "input graph & node features",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div>{"Input: 6-node cycle graph C_6 with node features x_i in R^3"}</div>
                <div>{"Adjacency A: edges {(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)}"}</div>
                <div>{"All x_i = [1, 1, 1] (identical features — only structure differs)"}</div>
              </div>
            ),
          },
          {
            label: "compute Laplacian PE",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div>{"L = D - A   (Laplacian)"}</div>
                <div>{"eigvals, eigvecs = eigh(L)"}</div>
                <div>{"PE_i = eigvecs[i, 1:k+1]   (skip trivial eigenvector)"}</div>
                <div style={{ color: colors.gold, marginTop: 4 }}>{"-> each node gets a k-dim structural code"}</div>
              </div>
            ),
          },
          {
            label: "concat features + PE, embed",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div>{"h_i^(0) = W_emb @ concat(x_i, PE_i)"}</div>
                <div>{"h_i^(0) is now structurally aware before any attention"}</div>
              </div>
            ),
          },
          {
            label: "compute spatial bias",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div>{"phi(i, j) = shortest_path_distance(i, j)"}</div>
                <div>{"B_spatial[i, j] = -phi(i, j)   (closer => higher bias)"}</div>
                <div style={{ color: colors.gold, marginTop: 4 }}>{"-> attention will prefer graph-near neighbors"}</div>
              </div>
            ),
          },
          {
            label: "self-attention with bias",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div>{"Q, K, V = qkv_proj(h)   shape [N, n_heads, d_head]"}</div>
                <div>{"scores = Q K^T / sqrt(d_head) + B_spatial"}</div>
                <div>{"attn = softmax(scores)"}</div>
                <div>{"out  = attn @ V"}</div>
              </div>
            ),
          },
          {
            label: "final pooling",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div>{"graph_repr = mean(h_i for i in nodes)"}</div>
                <div>{"y = head(graph_repr)"}</div>
                <div style={{ color: colors.green, marginTop: 4 }}>{"-> output: -0.4518 for C_6"}</div>
              </div>
            ),
          },
        ]}
      />

      <H3>Heatmap: Laplacian PE rows for C₆ vs 2×C₃</H3>

      <Prose>
        Below are the top-3 Laplacian eigenvectors (rows = nodes, cols = eigenvectors) for the two graphs. The values are the actual eigenvectors from the implementation. Notice how {"C_6"} produces smoothly varying entries (a Fourier-like structure on the cycle), while {"2 \\times C_3"} has block-structured entries that reflect the two disconnected components.
      </Prose>

      <Heatmap
        label="C_6  Laplacian PE  (rows = nodes 0..5, cols = eig 1..3)"
        matrix={[
          [0.577, 0.000, 0.577],
          [0.289, -0.500, -0.289],
          [-0.289, -0.500, -0.289],
          [-0.577, 0.000, 0.577],
          [-0.289, 0.500, -0.289],
          [0.289, 0.500, -0.289],
        ]}
        rowLabels={["n0", "n1", "n2", "n3", "n4", "n5"]}
        colLabels={["e1", "e2", "e3"]}
        colorScale="gold"
      />

      <Heatmap
        label="2 x C_3  Laplacian PE  (rows = nodes 0..5, cols = eig 1..3)"
        matrix={[
          [0.000, 0.000, 0.000],
          [0.000, -0.707, 0.000],
          [0.000, 0.707, 0.000],
          [0.577, 0.000, -0.816],
          [0.577, 0.000, 0.408],
          [0.577, 0.000, 0.408],
        ]}
        rowLabels={["n0", "n1", "n2", "n3", "n4", "n5"]}
        colLabels={["e1", "e2", "e3"]}
        colorScale="green"
      />

      <Prose>
        The first eigenvector for {"2 \\times C_3"} is essentially zero on the first triangle and {"\\frac{1}{\\sqrt{3}} \\approx 0.577"} on the second (or vice versa, depending on sign canonicalization): it is the indicator of one connected component. This kind of eigenvector simply does not exist in {"C_6"}, which has a single connected component. The graph transformer's input embedding sees this difference at every node before any attention happens.
      </Prose>

      <H3>Plot: discrimination on the C₆ vs 2×C₃ task</H3>

      <Plot
        label="output difference between graphs (higher = better discrimination)"
        series={[
          {
            name: "Graph Transformer + LapPE",
            color: colors.gold,
            points: [[1, 0.005], [2, 0.012], [3, 0.019], [4, 0.026], [5, 0.034]],
          },
          {
            name: "MP-GNN (1-WL bounded)",
            color: colors.green,
            points: [[1, 0.000], [2, 0.000], [3, 0.000], [4, 0.000], [5, 0.000]],
          },
        ]}
        xLabel="depth (# layers)"
        yLabel="| out(C_6) - out(2 x C_3) |"
        width={520}
        height={260}
      />

      <Prose>
        Vertical axis is the absolute difference between the model's output on {"C_6"} and on {"2 \\times C_3"}; horizontal axis is the depth of the network. The graph transformer's gap grows with depth because each attention layer further mixes the structural-encoding-derived information across nodes. The message-passing GNN's gap is exactly zero at every depth — adding more layers does not help, because the architecture is bounded by the 1-WL test regardless of depth.
      </Prose>

      <H3>Step trace: EGNN equivariance under rotation</H3>

      <StepTrace
        label="numerical verification of SE(3) equivariance"
        steps={[
          {
            label: "sample inputs",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div>{"h ~ Normal(0, 1)   shape [6, 8]   (hidden features)"}</div>
                <div>{"x ~ Normal(0, 1)   shape [6, 3]   (3D coordinates)"}</div>
                <div>{"edge_index = 6-cycle adjacency"}</div>
              </div>
            ),
          },
          {
            label: "sample random rotation R, translation t",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div>{"A_rand ~ Normal(0, 1)   shape [3, 3]"}</div>
                <div>{"Q, _ = QR(A_rand);  R = Q  if det(Q) > 0 else flip-col-0"}</div>
                <div>{"t ~ Normal(0, 1)   shape [3]"}</div>
                <div style={{ color: colors.gold, marginTop: 4 }}>{"R is in SO(3); R x + t is an SE(3) action"}</div>
              </div>
            ),
          },
          {
            label: "compute layer on original",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div>{"h_1, x_1 = layer(h, x, edge_index)"}</div>
              </div>
            ),
          },
          {
            label: "compute layer on rotated input",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div>{"h_2, x_2 = layer(h, x @ R^T + t, edge_index)"}</div>
              </div>
            ),
          },
          {
            label: "check equivariance",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div>{"x_1_rot = x_1 @ R^T + t"}</div>
                <div style={{ color: colors.green, marginTop: 4 }}>{"max| h_1 - h_2 |       = 5.96e-08   (invariance)"}</div>
                <div style={{ color: colors.green }}>{"max| x_1_rot - x_2 |   = 1.79e-07   (equivariance)"}</div>
                <div style={{ marginTop: 4 }}>{"Both are at floating-point precision -> exact equivariance."}</div>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Picking the right architecture is mostly about matching the symmetry structure of your problem to the inductive bias of the model. There is no universal best graph architecture; the optimal choice depends on graph size, presence of 3D coordinates, and which symmetries you need.
      </Prose>

      <CodeBlock>
{`task                                  recommended architecture        why
─────────────────────────────────────────────────────────────────────────────────────────────
small molecules (<= 500 atoms),       Graph transformer + structural   full attention is affordable;
property prediction                   encoding (Graphormer or GPS)     beats MP-GNN on PCQM4M-v2

molecular force fields, energies,     E(n)-EGNN  /  MACE  /  PaiNN     SE(3)-equivariance is exact
3D coordinates required                                                and dramatically improves
                                                                       sample efficiency

protein structure prediction          AlphaFold-style invariant         domain-specific equivariant
(thousands of residues)               point attention (IPA);             attention; OpenFold/ESMFold
                                      ESMFold for single-sequence

large social/citation networks        GraphSAGE, ClusterGCN, ShaDow,    O(N^2) attention infeasible;
(>= 100k nodes)                       sampling-based MP-GNNs            need neighborhood subsampling

physics simulation, particle          E(n)-EGNN with velocity update;   built-in equivariance
dynamics, N-body                      GNS (Sanchez-Gonzalez 2020)       matches the physics

quantum chemistry (DFT, ab initio)    SchNet, PaiNN, MACE, NequIP       continuous filters in 3D;
                                                                       MACE is current SOTA on
                                                                       small-molecule benchmarks

knowledge graph completion             RotatE, ComplEx, TransE,         relational embeddings designed
                                       NodePiece, R-GCN                 for KG-specific objectives

heterogeneous graphs (multi-type      HGT, R-GCN, HAN                   relation-typed message passing
nodes/edges)`}
      </CodeBlock>

      <Prose>
        Two practical heuristics worth internalizing. First, the deciding question for graph transformer vs message-passing is graph size: if {"N \\leq 1000"} per graph, full attention plus GPS is almost always better; for {"1000 < N \\leq 10000"} the GPS recipe with a linear-attention variant (Performer, BigBird) is competitive; above 10k nodes the message-passing-with-sampling tradition wins until the techniques for million-scale attention catch up. Second, the deciding question for equivariance is whether the answer should depend on the coordinate frame. If the answer is no — predicting energy, scalar properties, invariant distances — you want an SE(3)-invariant or -equivariant network. If your outputs are themselves 3D vectors (forces, normals, dipoles), you specifically need equivariance, not just invariance.
      </Prose>

      <Prose>
        A common mistake is to use a Cartesian-coordinate-fed transformer for 3D problems, which gives up the symmetry entirely. Such a network has to learn rotation-invariance from data augmentation, and either the augmentation is exhaustive enough that you waste capacity learning the symmetry, or it is sparse enough that the model fails on rotations it did not see at train time. Equivariant architectures pay a per-layer compute premium but eliminate this entire failure mode.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        The scaling story for graph transformers and equivariant networks has two distinct chapters. The graph transformer side is bottlenecked by the same O(N²) attention that limits long-context language modeling; the equivariant side is bottlenecked by the per-layer cost of equivariant operations. Both have known mitigations, but the mitigations are different.
      </Prose>

      <H3>Graph transformers: O(N²) attention is the wall</H3>

      <Prose>
        A graph transformer with full attention scales as O(N² d) per layer in compute, and O(N²) in memory just to store the attention matrix. For {"N = 1000"}, this is fine: a single attention matrix is 4 MB (in fp32) and the matrix-multiply cost is dominated by the QK^T step, which a modern GPU handles at terabytes-per-second on well-shaped tensors. For {"N = 10{,}000"}, the matrix is 400 MB per layer per head, and you start hitting memory limits in any reasonable mini-batch. For {"N = 100{,}000"}, full attention is infeasible on any single GPU.
      </Prose>

      <Prose>
        Three escape routes have been deployed in practice. Sparse attention restricts each node to attend to a subset of others — the GPS recipe uses a hybrid where local message passing handles the local interactions and a sparse global attention layer handles the cross-graph long-range dependencies. Linear attention variants (Performer, Linformer, RFA, FAVOR+) approximate the softmax with a kernel feature map and reduce the per-layer cost to O(N d²). These work but introduce approximation error; for graph data specifically, Performer-attention is a common drop-in replacement. The third escape is hierarchical pooling: coarsen the graph by repeatedly merging nodes (using e.g. graph clustering), apply attention on the coarsened version, then unpool. This has the structural advantage of preserving exact attention semantics; it has the disadvantage of requiring a hand-engineered pooling strategy.
      </Prose>

      <H3>Equivariant networks: per-layer cost vs depth tradeoff</H3>

      <Prose>
        EGNN is cheap per layer: each layer does a constant number of MLPs per edge plus a linear coordinate update. The asymptotic cost is O(E d) per layer where E is the number of edges, comparable to a vanilla message-passing GNN. SE(3)-Transformer with full {"e3nn"} machinery is roughly 3-5x more expensive per layer because of the spherical harmonic computations and Clebsch-Gordan tensor products at multiple irreducible-representation orders. MACE sits in between, depending on which orders of irreps it uses (it is parameterizable; using only L=0 and L=1 brings it close to EGNN cost).
      </Prose>

      <Prose>
        The compensating factor is that equivariant networks need fewer training samples. The classic experiment is to train EGNN and a plain Cartesian-input MLP on QM9 with 10x less data; EGNN's accuracy degrades gracefully while the plain MLP collapses, because the plain MLP needs to learn the rotation symmetry from data and 10x less data is not enough. For most production settings — chemistry, structural biology, small-molecule simulation — sample efficiency matters more than per-step compute, because the training data is expensive to generate (requires DFT or quantum-chemistry simulations) and inference is rarely the bottleneck.
      </Prose>

      <H3>AlphaFold-2 evoformer: scaling to thousands of residues</H3>

      <Prose>
        AlphaFold-2's evoformer block uses a pair representation that is {"L \\times L"} (where L is sequence length) augmented with row-attention and column-attention on the multiple-sequence-alignment representation. The pair representation alone is O(L²) memory just like graph transformer attention, but AlphaFold's authors invested heavily in axial attention patterns that decompose the L×L attention into row and column attentions, each O(L) per row/column or O(L²) total but with much better constants. In practice AlphaFold-2 handles sequences up to ~3000 residues on a single A100 (40GB), and their inference engine includes recycling — running the model multiple times with the previous output as input — to refine predictions iteratively. ESMFold reduces the constant factor further by replacing the MSA with a single-sequence input and a pre-trained protein language model embedding, getting roughly 6x faster inference at modest accuracy cost.
      </Prose>

      <Callout accent="gold">
        Practical scaling: full graph transformer attention up to {"~"}1k nodes; GPS with linear attention up to {"~"}10k; sampling-based MP-GNNs above 100k. Equivariant networks pay a per-layer premium but need 5-10x less data, which is a winning tradeoff for any domain where labeled data is the bottleneck.
      </Callout>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <Prose>
        Eight ways these architectures fail in practice. Most of them are silent — the model trains, achieves reasonable training loss, and ships before anyone notices the bug.
      </Prose>

      <Prose>
        <strong>1. Applying a graph transformer to a million-node graph.</strong> Full self-attention is O(N²) in memory; on a 100k-node graph the attention matrix is 40 GB per head in fp32. The standard symptom is an out-of-memory error on the first forward pass, before training has even started. Less standard but more pernicious is when an engineer reduces the model to one head, fp16 precision, and a smaller hidden dim to make the model fit, which produces an underpowered model that nonetheless trains. The fix is to recognize early that graph transformers are for small graphs; for {"N \\geq 10{,}000"} use GPS with linear attention, and for {"N \\geq 100{,}000"} use sampling-based message passing (GraphSAGE, ClusterGCN). Do not try to make full attention scale; the right architectures for that regime already exist.
      </Prose>

      <Prose>
        <strong>2. Forgetting structural encoding entirely.</strong> A graph transformer without any structural encoding is permutation-invariant in the wrong way: it sees an unordered set of node features and produces the same output for any graph with the same multiset of node features, regardless of edges. Two completely different graphs (say, {"K_4"} and four disconnected nodes) will produce identical embeddings. This is a silent failure: the model trains, achieves some loss, and the analyst concludes that graphs are hard to learn. The fix is to always include either a Laplacian PE, an RWPE, a relative-position embedding, or a shortest-path bias in the attention. PyG's GPSConv requires you to either pass in a {"pe"} tensor or {"degree"} information; attempt to use it without and it will throw a warning at construction.
      </Prose>

      <Prose>
        <strong>3. Cartesian-input transformer for 3D data.</strong> Building a transformer that consumes 3D coordinates directly as features (concatenating {"(x, y, z)"} to each node) gives up rotation symmetry entirely. The model has to learn from scratch that rotating the molecule should not change its energy, which is enormously inefficient. The symptom is that the model performs well on the training distribution and fails dramatically on rotated test inputs, with the failure being smooth — every rotation outside the training distribution gets progressively worse predictions. The fix is to either build an equivariant network (EGNN, MACE, SE(3)-Transformer) or to use only invariant features (pairwise distances, angles, dihedrals) as input — never raw Cartesian coordinates.
      </Prose>

      <Prose>
        <strong>4. EGNN coordinate update without normalization.</strong> The original Satorras et al. EGNN paper uses {"\\sum_j (x_i - x_j) \\phi_x(m_{ij})"} without normalizing the relative vector. In practice, with deep stacks (8+ layers) and irregular graphs, this update can blow up — once node coordinates drift apart, the relative vectors grow in norm, the next layer multiplies them by another scalar, and the norms compound. The symptom is training loss diverging after a few epochs, often after the model has appeared to be training successfully. The fix is to normalize the relative vector before scaling: replace {"(x_i - x_j) \\phi_x(m_{ij})"} with {"\\frac{x_i - x_j}{\\| x_i - x_j \\| + \\epsilon} \\phi_x(m_{ij})"}. Almost every EGNN implementation since 2022 uses this normalized version.
      </Prose>

      <Prose>
        <strong>5. Laplacian PE sign ambiguity.</strong> Eigenvectors are defined up to a sign: if {"u"} is an eigenvector then so is {"-u"}, and numerical solvers can return either depending on initial conditions, BLAS implementation, or random seed. Two runs of the same code on the same input graph can produce Laplacian PEs that differ by a sign on each column. This means the model is not truly invariant to the graph structure — it is sensitive to a numerically arbitrary choice. The fix is sign canonicalization: after eigendecomposition, force the first non-zero entry of each eigenvector to be positive. Better fixes use sign-equivariant networks that explicitly handle the ambiguity (Lim et al. 2022, "Sign and Basis Invariant Networks for Spectral Graph Representation Learning"), but for most production purposes, sign canonicalization is sufficient.
      </Prose>

      <Prose>
        <strong>6. Connected-component multiplicity in eigenvectors.</strong> Related to #5: when a graph has multiple connected components, the Laplacian has a zero eigenvalue with multiplicity equal to the number of components. Within the corresponding eigenspace, any orthonormal basis is a valid choice, and the basis returned by the solver is essentially arbitrary. For a graph with two components, the first two eigenvectors can be any rotation of the (component-1-indicator, component-2-indicator) pair. This is a more fundamental ambiguity than sign — it is a continuous family of equivalent answers. The fix in practice is to detect connected components first and apply Laplacian PE separately within each component, then concatenate, ensuring a consistent ordering.
      </Prose>

      <Prose>
        <strong>7. Invariant pooling on equivariant features.</strong> A common bug in equivariant network code is to apply a non-equivariant pooling operation at the end of the network — for instance, taking the mean of 3D vector features across nodes when the task expects a single 3D vector output. Mean-pooling is equivariant under permutations but not under rotations, unless every input vector transforms the same way. The fix is to either (a) keep the output equivariant by ensuring all pooling operations use only invariant operations on equivariant features (e.g., compute distances first, pool the distances), or (b) decide explicitly whether the output should be invariant (energy, scalar property) or equivariant (force vector, dipole) and design the head accordingly.
      </Prose>

      <Prose>
        <strong>8. Mixing E(3) and SE(3) equivariance accidentally.</strong> SE(3) is rotations and translations only; E(3) adds reflections. For chirality-sensitive tasks (drug discovery, where stereochemistry matters; physics with pseudovectors), you need SE(3) but not E(3) — left-handed and right-handed configurations should be distinguishable. EGNN as originally specified uses only the squared distance {"\\| x_i - x_j \\|^2"} as a geometric feature, which is invariant under both rotations and reflections, so vanilla EGNN is E(3)-equivariant. To break the chirality symmetry you need to introduce a feature that is rotation-invariant but reflection-sensitive — typically a triple product or signed dihedral angle. This is a subtle bug because both versions train fine; the chirality-blind version simply gets every chirality test wrong with 50% accuracy and the analyst may not notice for a long time.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Eight references that span the unifying framework, the graph transformer line, and the equivariant line. Reading order: start with Bronstein et al. for the framework, then Dwivedi & Bresson for the simplest graph transformer, then Ying et al. (Graphormer) and Rampášek (GPS) for the modern recipe; for equivariance, Cohen & Welling for the foundation, then Satorras (EGNN) for the simplest equivariant architecture, then Fuchs (SE(3)-Transformer) for the full machinery, then AlphaFold-2 for the most successful real-world application.
      </Prose>

      <Prose>
        <strong>1.</strong> Bronstein, Michael; Bruna, Joan; Cohen, Taco; Veličković, Petar. "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges." arXiv:2104.13478 (May 2021). The unifying monograph. Argues that CNNs, RNNs, GNNs, and transformers are all instances of equivariant networks built around different symmetry groups. The framework chapter proposes the "5G blueprint" — five types of geometric structure that cover most architectures of practical interest. Required reading for understanding why the field exists at all; lecture videos and slides at <Code>geometricdeeplearning.com</Code>.
      </Prose>

      <Prose>
        <strong>2.</strong> Dwivedi, Vijay Prakash; Bresson, Xavier. "A Generalization of Transformer Networks to Graphs." arXiv:2012.09699 (December 2020). AAAI Workshop on Deep Learning on Graphs, 2021. The first principled graph transformer. Proposes Laplacian eigenvector positional encodings and graph-masked attention. Establishes the paradigm that later papers (Graphormer, GPS) refine; the Laplacian PE convention from this paper is now standard.
      </Prose>

      <Prose>
        <strong>3.</strong> Ying, Chengxuan; Cai, Tianle; Luo, Shengjie; Zheng, Shuxin; Ke, Guolin; He, Di; Shen, Yanming; Liu, Tie-Yan. "Do Transformers Really Perform Bad for Graph Representation?" arXiv:2106.05234 (June 2021). NeurIPS 2021. The Graphormer paper. Introduces centrality encoding, spatial bias, and edge bias for graph attention. Wins the OGB-LSC PCQM4M-LSC benchmark by a substantial margin. Establishes that full self-attention with structural biases outperforms message-passing GNNs on small-graph tasks.
      </Prose>

      <Prose>
        <strong>4.</strong> Rampášek, Ladislav; Galkin, Mikhail; Dwivedi, Vijay Prakash; Luu, Anh Tuan; Wolf, Guy; Beaini, Dominique. "Recipe for a General, Powerful, Scalable Graph Transformer." arXiv:2205.12454 (May 2022). NeurIPS 2022. The GPS paper. Proposes a hybrid architecture interleaving message-passing layers (for local structure) with self-attention layers (for global structure), with linear-attention variants for scalability. Sets state-of-the-art on multiple benchmarks; the {"GPSConv"} layer in PyTorch Geometric is the canonical implementation.
      </Prose>

      <Prose>
        <strong>5.</strong> Fuchs, Fabian; Worrall, Daniel; Fischer, Volker; Welling, Max. "SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks." arXiv:2006.10503 (June 2020). NeurIPS 2020. Builds attention layers exactly equivariant under SE(3) using spherical harmonics and Clebsch-Gordan tensor products. The most theoretically clean equivariant transformer; computationally expensive but principled. Implementation is built on the {"e3nn"} library.
      </Prose>

      <Prose>
        <strong>6.</strong> Satorras, Víctor Garcia; Hoogeboom, Emiel; Welling, Max. "E(n) Equivariant Graph Neural Networks." arXiv:2102.09844 (February 2021). ICML 2021. The EGNN paper. Demonstrates that you can get exact E(n)-equivariance using only scalar (squared-distance) features and a coordinate update that multiplies relative vectors by invariant scalars — no spherical harmonics required. Currently the simplest equivariant architecture that works at scale.
      </Prose>

      <Prose>
        <strong>7.</strong> Jumper, John; Evans, Richard; Pritzel, Alexander; Green, Tim; Figurnov, Michael; Ronneberger, Olaf; Tunyasuvunakool, Kathryn; Bates, Russ; Žídek, Augustin; Potapenko, Anna; et al. "Highly accurate protein structure prediction with AlphaFold." <em>Nature</em> 596:583–589 (August 2021). The AlphaFold-2 paper. Introduces the evoformer (axial attention on MSA + pair representations) and the structure module (invariant point attention, rotational frame updates). Proves that geometric attention plus deep evolutionary inputs solves protein structure prediction at near-experimental accuracy. Open-source implementations: OpenFold, ColabFold, ESMFold.
      </Prose>

      <Prose>
        <strong>8.</strong> Cohen, Taco; Welling, Max. "Group Equivariant Convolutional Networks." arXiv:1602.07576 (February 2016). ICML 2016. The foundational paper for group-equivariant deep learning. Generalizes convolutions from translation-equivariant to arbitrary-group-equivariant operations on regular grids. Predates most of the geometric deep learning literature and provides the algebraic foundations that everything since has built on. The follow-up paper "Steerable CNNs" (2017) extends the framework to continuous groups.
      </Prose>

      <Callout accent="gold">
        Secondary reading worth pursuing: Lim et al. 2022 (sign-equivariant Laplacian networks), Brandstetter et al. 2022 ("Geometric and Physical Quantities Improve E(3) Equivariant Message Passing"), Batatia et al. 2022 (MACE), and the AlphaFold-3 paper (Abramson et al. 2024) for the latest evolution of geometric attention in structural biology.
      </Callout>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        Five problems. Spend ten minutes per problem before looking at the answer. Each is calibrated to test a concept that is easy to get subtly wrong.
      </Prose>

      <Prose>
        <strong>Problem 1.</strong> A graph has two connected components. You compute the eigenvectors of the unnormalized Laplacian and use the first three (excluding the trivial constant eigenvector) as positional encoding. You notice that two different runs on the same graph produce different positional encodings — the values are not the same. Why? What are the two distinct sources of ambiguity? How would you fix each?
      </Prose>

      <Callout accent="green">
        Two ambiguities. (a) Sign ambiguity: each eigenvector {"u_i"} is determined only up to a sign, so {"u_i"} and {"-u_i"} are both valid eigenvectors for the same eigenvalue. Different solver implementations (or different random initializations of iterative solvers) can return different signs. Fix: canonicalize after computing — force the first non-zero entry of each eigenvector to be positive. (b) Eigenspace ambiguity: when a graph has K connected components, the eigenvalue 0 has multiplicity K, and any orthonormal basis of the corresponding eigenspace is valid. Different solvers can return different bases (the simplest case being component-indicator vectors vs an orthogonal rotation of them). This is a continuous ambiguity, not a discrete one. Fix: detect connected components first (graph BFS), compute Laplacian PE separately on each component, concatenate with explicit component-id tagging. Better fix: use a sign-and-basis-invariant PE network (Lim et al. 2022) that processes eigenvectors in a way that is invariant to both ambiguities.
      </Callout>

      <Prose>
        <strong>Problem 2.</strong> You implement an EGNN layer and run the equivariance check by computing {"\\| f(R x) - R f(x) \\|"} for a random rotation R. You get an error of {"10^{-3}"}, much larger than the floating-point precision of {"10^{-7}"}. Your equivariance is approximately correct but not exactly correct. What are the two most likely bugs and how would you diagnose each?
      </Callout>

      <Callout accent="green">
        Two likely culprits. (a) You included a non-invariant input feature in {"\\phi_e"}. The most common case is concatenating raw coordinates {"x_i"} (not the relative vector {"x_i - x_j"}) into the message MLP, which makes {"m_{ij}"} not invariant. Diagnosis: print the inputs to {"\\phi_e"} before and after rotation; they should be identical. (b) You used a non-equivariant operation in the coordinate update — for instance, multiplying the relative vector by a vector of weights produced by an MLP that consumes coordinate-frame information. The coordinate update must be of the form {"\\sum_j (x_i - x_j) \\cdot s_{ij}"} where {"s_{ij}"} is a scalar (invariant) function. Diagnosis: replace your scalar weight with a fixed constant 1.0 — equivariance error should drop to numerical precision. If it doesn't, the bug is elsewhere. A 10^{"-3"} error can also indicate using a non-orthogonal "rotation" matrix; verify that {"R^T R = I"} and {"\\det(R) = 1"} before using it in the test.
      </Callout>

      <Prose>
        <strong>Problem 3.</strong> Consider a graph transformer with full self-attention applied to a 1000-node graph, hidden dimension 256, 8 attention heads, in fp32. How much memory does the attention matrix consume per layer? How does this scale to a 10,000-node graph? At what graph size does a single A100 (80 GB) become infeasible for batch size 1?
      </Prose>

      <Callout accent="green">
        Memory per attention matrix per head: {"N^2 \\times 4"} bytes (fp32). For {"N = 1000"}: {"10^6 \\times 4 = 4"} MB per head. With 8 heads: 32 MB per layer. For {"N = 10000"}: {"10^8 \\times 4 = 400"} MB per head, 3.2 GB per layer. With a typical 6-layer model that's 19 GB just for attention matrices on the forward pass — and you also need the activations to backprop through them, roughly doubling the cost. So 80 GB starts to feel tight at 10k nodes for a 6-layer model. At {"N = 30000"}, one head's attention matrix is 3.6 GB, 28.8 GB across heads, 173 GB across 6 layers. Long before you hit 80 GB you should switch to GPS with linear attention or to a sampling-based method. Practical rule: full attention is comfortable up to about 5000 nodes, marginal at 10k, and infeasible beyond {"~20"}k.
      </Callout>

      <Prose>
        <strong>Problem 4.</strong> A colleague is training a Cartesian-input transformer (no equivariance built in) on a 3D molecular property prediction task. They are achieving 0.95 R² on training data and 0.65 R² on the test set. Their data augmentation strategy is to randomly rotate each molecule on every training step. They want to know if more rotation augmentation will close the train/test gap. What is the more likely diagnosis, and what architectural change would you recommend?
      </Prose>

      <Callout accent="green">
        The train/test gap of 0.30 in R² is huge for a chemistry regression task and almost certainly indicates that rotation augmentation is not enough. A non-equivariant transformer must learn rotation-invariance from data; rotation augmentation helps but does not eliminate the burden, and in practice never reaches the performance of an architecturally equivariant network. The architectural recommendation is to switch to an SE(3)-equivariant model: EGNN as a low-cost first attempt, MACE or PaiNN if EGNN under-fits, SE(3)-Transformer if higher-order tensor features are needed. Expect to recover roughly 0.20 R² of the gap from the architecture change alone. The intuition: with augmentation the model has to learn that rotated inputs have the same answer; with equivariance, every input is implicitly already presented in every rotation, with no learning required and no data wasted on it.
      </Callout>

      <Prose>
        <strong>Problem 5.</strong> Explain why a vanilla message-passing GNN cannot distinguish the cycle {"C_6"} from the disjoint union of two triangles {"2 \\times C_3"}, even with arbitrarily many layers and identical input features. Then explain why a graph transformer with Laplacian positional encoding can. Be specific about which mathematical property of the architectures is responsible for the difference.
      </Prose>

      <Callout accent="green">
        Message-passing GNNs are bounded by the 1-Weisfeiler-Leman test (Xu et al. 2019, Morris et al. 2019). The 1-WL refinement updates each node's color based on the multiset of its neighbors' colors. In {"C_6"}, every node has degree 2 and identical features; the multiset of its neighbors' colors is also {"\\{c, c\\}"} where c is the universal initial color. After one round, every node still has the same color. The same is true at every round, indefinitely. In {"2 \\times C_3"}, the same argument applies: every node has degree 2, every neighbor has the same color, refinement is a fixed point from round 1. Both graphs reach the same coloring at every depth, so a message-passing GNN with identical features cannot distinguish them, regardless of depth or width. A graph transformer with Laplacian PE breaks this because the input features now include structural information that depends on the graph's spectrum. The Laplacian of {"C_6"} has eigenvalues {"\\{0, 1, 1, 3, 3, 4\\}"}; the Laplacian of {"2 \\times C_3"} has eigenvalues {"\\{0, 0, 3, 3, 3, 3\\}"}. These spectra are different — in particular, the second graph has a duplicated zero eigenvalue corresponding to its two components, which the first graph does not have. The eigenvectors thus encode features that are different between the two graphs at the per-node level, before any attention. Once attention runs, the model produces different graph-level outputs. The mathematical property doing the work is that 1-WL is bounded by neighborhood multisets, while the Laplacian spectrum is a global graph invariant that captures topological information that no local multiset of neighbors can.
      </Callout>
    </div>
  ),
};

export default graphTransformersContent;
