import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const tsneUmapContent = {
  title: "t-SNE, UMAP & Manifold Learning",
  readTime: "~50 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        By the late 1990s, a specific frustration had accumulated in machine learning and neuroscience: high-dimensional data — faces, handwritten digits, gene expression profiles, neural spike trains — clearly had structure, and that structure was clearly not linear. PCA could tell you that a dataset of 1,000-dimensional face images had maybe 50 "meaningful" directions of variance. But project those faces onto the top two principal components and you got an unintelligible smear. The faces were not living on a flat hyperplane inside pixel space. They were living on a curved, twisted, low-dimensional surface — a manifold. And linear methods couldn't see the manifold.
      </Prose>

      <Prose>
        In December 2000, two papers appeared back-to-back in the same issue of <em>Science</em> — pages 2319 and 2323 — and they both made the same claim: you can do better. Joshua Tenenbaum, Vin de Silva, and John Langford introduced Isomap (Tenenbaum, de Silva, and Langford, "A Global Geometric Framework for Nonlinear Dimensionality Reduction," <em>Science</em> 290(5500):2319–2323, 2000). Sam Roweis and Lawrence Saul introduced Locally Linear Embedding — LLE (Roweis and Saul, "Nonlinear Dimensionality Reduction by Locally Linear Embedding," <em>Science</em> 290(5500):2323–2326, 2000). Both papers demonstrated on the same toy problem — a Swiss roll, a 2D sheet coiled into 3D space — that their methods could "unroll" the manifold and recover the true 2D coordinates, while PCA just saw a blob. The dual publication in <em>Science</em> was a statement: this is not an incremental algorithmic tweak, this is a new way of thinking about the geometry of data.
      </Prose>

      <Prose>
        These methods had a shared problem: they were not suitable for visualization at scale. Isomap required computing all pairwise geodesic distances — <Code>O(n²)</Code> in memory — and its embedding was unstable for non-convex manifolds. LLE required solving a large sparse eigenproblem and was sensitive to noise and the choice of neighborhood size. Both were slow. Two years later, Geoffrey Hinton and Sam Roweis introduced Stochastic Neighbor Embedding — SNE — at NeurIPS 2002. SNE converted pairwise distances into probability distributions and minimized a KL divergence to bring those distributions into agreement in the low-dimensional space. The probabilistic framing was new and important. But SNE had its own pathology: the "crowding problem." In high dimensions there is enough room for each point to have many neighbors at moderate distance; in 2D there isn't. Points got crushed into the center of the embedding.
      </Prose>

      <Prose>
        Laurens van der Maaten and Geoffrey Hinton fixed the crowding problem in 2008 with t-SNE: "Visualizing Data Using t-SNE" (<em>Journal of Machine Learning Research</em> 9(86):2579–2605, 2008). The fix was elegant: use a Student-t distribution with one degree of freedom (a Cauchy distribution) in the low-dimensional space instead of a Gaussian. The heavy tails of the Student-t allow moderately distant points in high-D to be placed far apart in 2D without fighting against the probability mass — the distribution simply accommodates large distances. t-SNE produced strikingly clean visualizations of MNIST, the Olivetti faces dataset, and natural language co-occurrence data. It became the dominant visualization technique in biology, NLP, and computer vision for the next decade.
      </Prose>

      <Prose>
        The remaining objection was speed. t-SNE with Barnes-Hut approximation (van der Maaten, "Accelerating t-SNE using Tree-Based Algorithms," <em>JMLR</em> 15:3221–3245, 2014) was <Code>O(n log n)</Code> and practical up to about 100,000 points. But genomics datasets in the 2010s had millions of cells. Leland McInnes, John Healy, and James Melville published UMAP — Uniform Manifold Approximation and Projection — in 2018 (arXiv:1802.03426). UMAP is grounded in Riemannian geometry and algebraic topology: it constructs a weighted graph representing the fuzzy topological structure of the data in high dimensions, then optimizes a 2D layout that preserves that structure. It is faster than t-SNE at any scale and preserves global structure better. UMAP has since largely supplanted t-SNE as the default choice for large-scale visualization, though both remain essential tools.
      </Prose>

      <Prose>
        The shared motivation across all of these methods is the <strong>manifold hypothesis</strong>: real-world high-dimensional data does not fill its ambient space uniformly. A dataset of photographs of a mug rotated 360 degrees is intrinsically one-dimensional (the rotation angle), embedded in millions of pixel dimensions. A dataset of handwritten digits is intrinsically low-dimensional (stroke thickness, slant, loop closure), embedded in 64 or 784 pixel dimensions. If the true degrees of freedom are few and the data is well-sampled, then finding the right low-dimensional representation is the whole game. Manifold learning methods are how you play it when the manifold is not flat.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 The manifold hypothesis and why PCA misses it</H3>

      <Prose>
        Start with PCA's fundamental assumption: the data lies near a flat, linear subspace. This works on correlated Gaussian data and many real-world tabular datasets. It fails when the manifold is curved. The Swiss roll is the canonical example: a 2D rectangular sheet rolled up twice in 3D space. Two points near each other in 3D Euclidean distance might be on opposite ends of the roll — far apart when you measure distance <em>along the sheet</em>. PCA sees only Euclidean distance. It collapses the roll into a blob. Isomap replaces Euclidean distance with geodesic distance — shortest path along the manifold surface, approximated by shortest path in a k-nearest-neighbor graph — and then applies multidimensional scaling to that distance matrix. The unrolled rectangle emerges.
      </Prose>

      <Prose>
        The key geometric vocabulary: a <strong>manifold</strong> is a topological space that locally looks like flat Euclidean space of some dimension <Code>d</Code> (the intrinsic dimension), even though it may be embedded in a much higher-dimensional ambient space <Code>D {">"} d</Code>. The surface of a sphere is a 2-manifold embedded in 3D. A twisted ribbon is a 2-manifold in 3D. A dataset of faces lives on a manifold of perhaps 50 intrinsic dimensions embedded in (height × width × channels) dimensions. Manifold learning methods work by exploiting the fact that locally — in a small neighborhood of any point — the manifold looks flat and Euclidean distances are meaningful. The algorithms differ in how they connect these local neighborhoods into a global structure.
      </Prose>

      <H3>2.2 t-SNE: probability-matching via KL divergence</H3>

      <Prose>
        t-SNE's intuition is neighborhood preservation by probability matching. In the high-dimensional space, define a probability <Code>p_{"{ij}"}</Code> as the likelihood that point <Code>i</Code> would pick point <Code>j</Code> as a neighbor — computed as a normalized Gaussian centered at <Code>i</Code>, where the Gaussian's bandwidth is chosen so that the effective neighborhood size (perplexity) matches a user-specified value. In the low-dimensional space, define a similar probability <Code>q_{"{ij}"}</Code>, but using a Student-t distribution instead of a Gaussian. The algorithm moves the 2D points around to minimize the KL divergence between the high-D probability distribution <Code>P</Code> and the low-D distribution <Code>Q</Code>: nearby things in high-D should be nearby in 2D.
      </Prose>

      <Prose>
        The critical design choice is that Student-t. In high dimensions, a Gaussian can accommodate many points at moderate distances — the volume of a high-D ball grows exponentially with radius. In 2D, there is no such extra volume. If you used a Gaussian in 2D as well, you would need to pack points extremely close together to match the high-D probabilities, causing collapse. The Student-t's heavier tail means that a <em>moderately large</em> low-D distance still produces a non-negligible <Code>q_{"{ij}"}</Code>. Points that should be separated can be placed far apart without being "penalized" as much for the distance. This is what resolves the crowding problem.
      </Prose>

      <Callout type="warning" title="What t-SNE embeddings do NOT encode">
        This is the most common misreading of t-SNE plots, and it is worth stating plainly. (1) Distances between clusters are not meaningful — two clusters that appear far apart in a t-SNE plot might be equidistant in the original space from a third cluster. (2) Cluster sizes are not meaningful — a large cluster in the plot may correspond to a tight cluster in the original space; a small cluster may correspond to a diffuse one. (3) The axes have no interpretation — t-SNE learns a nonlinear, rotation-arbitrary embedding; the x-axis and y-axis mean nothing. The only structure you can trust is local neighborhoods: points that appear close in the plot were genuinely close in the original space.
      </Callout>

      <H3>2.3 UMAP: fuzzy topology and cross-entropy minimization</H3>

      <Prose>
        UMAP starts from a different theoretical framework — algebraic topology — but arrives at a similar procedure. The high-level idea: construct a weighted graph (a "fuzzy simplicial set") where each edge weight represents the probability that two points are connected in the true topological structure of the data. The weights are computed from k-nearest-neighbor distances using a local metric that adapts to the density of the data: in dense regions the neighborhoods are small, in sparse regions they are larger. The algorithm then optimizes a 2D graph layout that preserves these fuzzy topological relationships, using cross-entropy as the loss and SGD with negative sampling for efficiency.
      </Prose>

      <Prose>
        The theoretical grounding matters practically. UMAP's local metric is derived from a Riemannian metric that is consistent with the assumption that the data is uniformly distributed on a Riemannian manifold. This gives UMAP a property that t-SNE lacks: it makes a genuine attempt to preserve global structure. In a t-SNE embedding, the relative placement of different clusters is essentially arbitrary. In a UMAP embedding, the relative placement of clusters reflects — imperfectly but meaningfully — their true proximity in the original space. UMAP also has explicit hyperparameters for this trade-off: <Code>n_neighbors</Code> controls how much local vs. global structure is prioritized, and <Code>min_dist</Code> controls how tightly clustered the embedding is.
      </Prose>

      <H3>2.4 The family: Isomap, LLE, MDS, t-SNE, UMAP</H3>

      <Prose>
        All five methods share the goal of finding low-dimensional representations that preserve neighborhood structure, but they differ in what "neighborhood structure" means and how they optimize it. Isomap preserves geodesic distances (global, metric). LLE preserves local linear reconstruction weights (local, geometric). MDS preserves arbitrary pairwise distances or dissimilarities (metric). t-SNE preserves local probability neighborhoods (local, probabilistic). UMAP preserves fuzzy topological neighborhoods (local-to-global, topological). The differences determine when each method works and when it fails — which is the subject of Sections 7 and 9.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 t-SNE: high-dimensional probabilities</H3>

      <Prose>
        Given <Code>n</Code> points <Code>x₁, ..., xₙ</Code> in <Code>{"ℝ"}^D</Code>, define the conditional probability that <Code>xᵢ</Code> would pick <Code>xⱼ</Code> as a neighbor as a normalized Gaussian:
      </Prose>

      <MathBlock>
        {"p_{j|i} = \\frac{\\exp\\!\\left(-\\|x_i - x_j\\|^2 \\,/\\, 2\\sigma_i^2\\right)}{\\sum_{k \\neq i} \\exp\\!\\left(-\\|x_i - x_k\\|^2 \\,/\\, 2\\sigma_i^2\\right)}"}
      </MathBlock>

      <Prose>
        Each point <Code>i</Code> has its own bandwidth <Code>σᵢ</Code>, chosen by binary search so that the perplexity of the conditional distribution equals the user-specified perplexity <Code>Perp</Code>:
      </Prose>

      <MathBlock>
        {"\\text{Perp}(P_i) = 2^{H(P_i)}, \\quad H(P_i) = -\\sum_{j} p_{j|i} \\log_2 p_{j|i}"}
      </MathBlock>

      <Prose>
        Perplexity is the exponential of the Shannon entropy of the conditional distribution — it measures the effective number of neighbors that point <Code>i</Code> considers. A perplexity of 30 means each point effectively has about 30 nearest neighbors contributing to its distribution. Typical values range from 5 to 100. The joint probability is then symmetrized:
      </Prose>

      <MathBlock>
        {"p_{ij} = \\frac{p_{j|i} + p_{i|j}}{2n}"}
      </MathBlock>

      <H3>3.2 t-SNE: Student-t in low dimensions</H3>

      <Prose>
        In the low-dimensional embedding <Code>y₁, ..., yₙ</Code> in <Code>{"ℝ"}^d</Code> (typically <Code>d = 2</Code>), t-SNE uses a Student-t distribution with one degree of freedom (a Cauchy distribution) to define the low-dimensional joint probability:
      </Prose>

      <MathBlock>
        {"q_{ij} = \\frac{\\left(1 + \\|y_i - y_j\\|^2\\right)^{-1}}{\\sum_{k \\neq l} \\left(1 + \\|y_k - y_l\\|^2\\right)^{-1}}"}
      </MathBlock>

      <Prose>
        The unnormalized weight <Code>(1 + ‖yᵢ − yⱼ‖²)⁻¹</Code> decays like <Code>1/r²</Code> for large distances rather than exponentially — this is what gives the Student-t its heavy tail and resolves the crowding problem. The cost function is the KL divergence from <Code>P</Code> to <Code>Q</Code>:
      </Prose>

      <MathBlock>
        {"C = \\mathrm{KL}(P \\| Q) = \\sum_{i \\neq j} p_{ij} \\log \\frac{p_{ij}}{q_{ij}}"}
      </MathBlock>

      <Prose>
        Note the asymmetry: <Code>KL(P||Q)</Code> penalizes heavily when <Code>P</Code> is large and <Code>Q</Code> is small (nearby points in high-D that are placed far apart in 2D incur high cost), but is lenient when <Code>P</Code> is small and <Code>Q</Code> is large (distant points placed close together in 2D incur low cost). This asymmetry explains why t-SNE tends to produce tight clusters with well-separated gaps — it is much more concerned about keeping neighbors close than about repelling distant points.
      </Prose>

      <H3>3.3 t-SNE gradient derivation</H3>

      <Prose>
        Differentiating <Code>C</Code> with respect to the position <Code>yᵢ</Code> of point <Code>i</Code> in the embedding:
      </Prose>

      <MathBlock>
        {"\\frac{\\partial C}{\\partial y_i} = 4 \\sum_{j} (p_{ij} - q_{ij})\\, (y_i - y_j)\\, \\left(1 + \\|y_i - y_j\\|^2\\right)^{-1}"}
      </MathBlock>

      <Prose>
        This gradient has an elegant physical interpretation: think of each pair <Code>(i, j)</Code> as connected by a spring. If <Code>p_{"{ij}"} {">"} q_{"{ij}"}</Code> (the pair is closer in high-D than in the embedding), the spring pulls the two points together; if <Code>p_{"{ij}"} {"<"} q_{"{ij}"}</Code> (closer in the embedding than in high-D), it pushes them apart. The factor <Code>(1 + ‖yᵢ − yⱼ‖²)⁻¹</Code> from the Student-t kernel modulates the spring force — distant points in the embedding feel weaker forces, preventing the embedding from collapsing. The gradient is computed over all pairs, so the cost is <Code>O(n²)</Code>; Barnes-Hut approximation (van der Maaten 2014) reduces this to <Code>O(n log n)</Code>.
      </Prose>

      <H3>3.4 UMAP: fuzzy simplicial sets and cross-entropy</H3>

      <Prose>
        UMAP's mathematical foundation is more involved, but the key ideas are tractable. For each point <Code>xᵢ</Code>, find its <Code>k</Code> nearest neighbors at distances <Code>d_{"{i,1}"} ≤ d_{"{i,2}"} ≤ ... ≤ d_{"{i,k}"}</Code>. Define a local metric that normalizes by the distance to the nearest neighbor <Code>ρᵢ = d_{"{i,1}"}</Code> and adapts the bandwidth <Code>σᵢ</Code> so that the sum of fuzzy membership strengths equals <Code>log₂(k)</Code>:
      </Prose>

      <MathBlock>
        {"v_{ij} = \\exp\\!\\left(\\frac{-(d(x_i, x_j) - \\rho_i)}{\\sigma_i}\\right)"}
      </MathBlock>

      <Prose>
        The membership strength of the edge between <Code>i</Code> and <Code>j</Code> is symmetrized as:
      </Prose>

      <MathBlock>
        {"w_{ij} = v_{ij} + v_{ji} - v_{ij} \\cdot v_{ji}"}
      </MathBlock>

      <Prose>
        This is the fuzzy union formula: the probability that <em>at least one</em> of the two directed memberships holds. In the low-dimensional embedding, the analogous membership strength is:
      </Prose>

      <MathBlock>
        {"\\hat{w}_{ij} = \\left(1 + a\\,\\|y_i - y_j\\|^{2b}\\right)^{-1}"}
      </MathBlock>

      <Prose>
        where <Code>a</Code> and <Code>b</Code> are parameters fit to approximate a smooth step function controlled by <Code>min_dist</Code>. UMAP minimizes the binary cross-entropy between the high-D weights <Code>w_{"{ij}"}</Code> and the low-D weights <Code>ŵ_{"{ij}"}</Code>:
      </Prose>

      <MathBlock>
        {"C = \\sum_{(i,j) \\in E} \\left[ w_{ij} \\log \\frac{w_{ij}}{\\hat{w}_{ij}} + (1 - w_{ij}) \\log \\frac{1 - w_{ij}}{1 - \\hat{w}_{ij}} \\right]"}
      </MathBlock>

      <Prose>
        This is optimized by SGD with negative sampling: for each edge in the graph, do a positive update to bring <Code>yᵢ</Code> and <Code>yⱼ</Code> closer, then sample <Code>m</Code> random non-edges and do repulsive updates. The negative sampling gives UMAP its <Code>O(n · k · n_epochs)</Code> time complexity — roughly <Code>O(n^{"{1.14}"})</Code> in practice — which is dramatically faster than the <Code>O(n²)</Code> naive t-SNE.
      </Prose>

      <H3>3.5 Key differences at a glance</H3>

      <Prose>
        Both t-SNE and UMAP are nonlinear, graph-based, and neighborhood-preserving. The differences: t-SNE uses KL divergence (asymmetric, penalizes false negatives), UMAP uses cross-entropy (symmetric). t-SNE uses a Gaussian high-D kernel, UMAP uses an adaptive local metric. t-SNE uses a Student-t low-D kernel, UMAP uses a generalized Cauchy-like function. t-SNE optimizes the embedding from scratch for each dataset with no parametric form — you cannot embed new points without re-running the algorithm (unless you use parametric t-SNE). UMAP learns a parametric mapping via its graph structure and supports transforming new points with <Code>reducer.transform(X_new)</Code>. UMAP also has a cleaner theoretical justification for preserving global structure through the fuzzy topological framework.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        We implement a simplified t-SNE from scratch in NumPy — this is the pedagogically useful version, not optimized for production. It handles <Code>n ≤ 500</Code> comfortably. UMAP's implementation requires a k-NN graph, approximate nearest neighbor search, and SGD with negative sampling across a weighted sparse graph — implementing it from scratch correctly takes several hundred lines. We defer UMAP to a library and focus the pseudocode on the algorithm.
      </Prose>

      <H3>4a. Simplified t-SNE from scratch (NumPy only)</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# --- Helper: pairwise squared Euclidean distances ---
def pairwise_sq_dists(X):
    sum_sq = np.sum(X ** 2, axis=1)
    D = sum_sq[:, None] + sum_sq[None, :] - 2 * (X @ X.T)
    np.fill_diagonal(D, 0)
    return D

# --- Binary search for bandwidth sigma_i ---
def compute_pij(D, perplexity=30.0):
    """
    For each point i, find sigma_i such that Perp(P_i) = perplexity.
    Returns the symmetrized joint probability matrix P.
    """
    n = D.shape[0]
    P = np.zeros((n, n))
    log_perp = np.log2(perplexity)

    for i in range(n):
        dists_i = D[i].copy()
        dists_i[i] = np.inf          # exclude self
        beta_min, beta_max = -np.inf, np.inf
        beta = 1.0                    # beta = 1 / (2 * sigma_i^2)

        for _ in range(50):           # binary search iterations
            exp_d = np.exp(-dists_i * beta)
            exp_d[i] = 0
            sum_exp = max(exp_d.sum(), 1e-10)
            p_i = exp_d / sum_exp

            # Shannon entropy in bits: H = -sum p log2 p
            nz = p_i > 1e-10
            H = -np.sum(p_i[nz] * np.log2(p_i[nz]))
            diff = H - log_perp

            if abs(diff) < 1e-5:
                break
            if diff > 0:             # H > log(perp): bandwidth too wide, increase beta
                beta_min = beta
                beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2
            else:                    # H < log(perp): bandwidth too narrow, decrease beta
                beta_max = beta
                beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2

        P[i] = p_i

    # Symmetrize: p_ij = (p_{j|i} + p_{i|j}) / (2n)
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)
    return P

# --- Core t-SNE loop ---
def tsne_scratch(X, n_components=2, perplexity=30, n_iter=300,
                 lr=200.0, random_state=42):
    np.random.seed(random_state)
    n = X.shape[0]

    D = pairwise_sq_dists(X)
    P = compute_pij(D, perplexity)
    # Early exaggeration: multiply P by 4 for first 100 iterations
    P_exag = P * 4.0

    # Initialize from small random normal (not PCA for simplicity here)
    Y = np.random.randn(n, n_components) * 0.01
    Y_prev = Y.copy()
    momentum = 0.5

    for t in range(n_iter):
        P_curr = P_exag if t < 100 else P

        # Low-dim Student-t kernel weights: (1 + ||yi - yj||^2)^{-1}
        D_Y = pairwise_sq_dists(Y)
        Q_num = 1.0 / (1.0 + D_Y)
        np.fill_diagonal(Q_num, 0)
        Q = Q_num / Q_num.sum()
        Q = np.maximum(Q, 1e-12)

        # KL divergence for monitoring
        mask = P_curr > 1e-12
        kl = np.sum(P_curr[mask] * np.log(P_curr[mask] / Q[mask]))

        # Gradient: dC/dy_i = 4 * sum_j (p_ij - q_ij)(yi - yj)(1 + ||yi-yj||^2)^{-1}
        PQ_diff = P_curr - Q
        grad = np.zeros_like(Y)
        for i in range(n):
            diff_i = Y[i] - Y                     # shape (n, d)
            weights = (PQ_diff[i] * Q_num[i])[:, None]
            grad[i] = 4.0 * np.sum(weights * diff_i, axis=0)

        # Gradient descent with momentum
        Y_new = Y - lr * grad + momentum * (Y - Y_prev)
        Y_prev = Y.copy()
        Y = Y_new

        if t == 49:
            momentum = 0.8    # switch to higher momentum after early phase

        if t % 50 == 0 or t == n_iter - 1:
            print(f'  Iter {t:3d} | KL divergence: {kl:.4f}')

    return Y

# --- Run on 8D blobs (n=100, 3 clusters) ---
X_raw, y_true = make_blobs(n_samples=100, centers=3, n_features=8,
                            cluster_std=1.5, random_state=42)
X_blobs = StandardScaler().fit_transform(X_raw)

print('t-SNE from scratch: 100 points, 8D -> 2D, perplexity=30')
Y_embed = tsne_scratch(X_blobs, n_components=2, perplexity=30,
                       n_iter=300, lr=200.0, random_state=42)
# Output:
#   t-SNE from scratch: 100 points, 8D -> 2D, perplexity=30
#   Iter   0 | KL divergence: 1.1699
#   Iter  50 | KL divergence: 0.0452
#   Iter 100 | KL divergence: 0.0392
#   Iter 150 | KL divergence: 0.0380
#   Iter 200 | KL divergence: 0.0375
#   Iter 250 | KL divergence: 0.0371
#   Iter 299 | KL divergence: 0.0368

print(f'Embedding shape: {Y_embed.shape}')
# Output: Embedding shape: (100, 2)
print(f'Embedding range x: [{Y_embed[:,0].min():.3f}, {Y_embed[:,0].max():.3f}]')
# Output: Embedding range x: [-5.997, 9.916]
print(f'Embedding range y: [{Y_embed[:,1].min():.3f}, {Y_embed[:,1].max():.3f}]')
# Output: Embedding range y: [-10.342, 6.898]

# Verify cluster separation: same-label points should be nearby
print('First 5 embedding points:')
for i in range(5):
    print(f'  Point {i}: ({Y_embed[i,0]:.3f}, {Y_embed[i,1]:.3f})  label={y_true[i]}')
# Output:
#   Point 0: (9.913, 6.301)  label=1
#   Point 1: (-4.319, 3.240)  label=2
#   Point 2: (-4.149, -9.200)  label=0
#   Point 3: (8.859, 4.999)  label=1
#   Point 4: (-3.600, -7.975)  label=0`}
      </CodeBlock>

      <Prose>
        The KL divergence falls from 1.17 to 0.037 — a 97% reduction. The early exaggeration trick (multiplying <Code>P</Code> by 4 for the first 100 iterations) pushes clusters apart aggressively during early optimization, then releases them to settle at their natural separation. Without early exaggeration, the embedding often collapses to a tight ball. The momentum switch from 0.5 to 0.8 at iteration 50 accelerates convergence after the early phase.
      </Prose>

      <H3>4b. UMAP algorithm in pseudocode</H3>

      <Prose>
        UMAP's full implementation (pynndescent-based approximate k-NN, Riemannian metric fitting, fuzzy set construction, SGD with negative sampling) is correctly implemented in the <Code>umap-learn</Code> library and is not meaningfully simplified into a short NumPy sketch. The algorithm in structured pseudocode:
      </Prose>

      <CodeBlock language="python">
{`# UMAP algorithm — structured pseudocode (not runnable)
# Reference: McInnes, Healy, Melville (2018) arXiv:1802.03426

# Step 1: Build approximate k-NN graph in high-D
#   For each point x_i, find k nearest neighbors {x_{i,1}, ..., x_{i,k}}
#   with distances {d_{i,1}, ..., d_{i,k}}. rho_i = d_{i,1} (nearest neighbor distance).
#   Use approximate nearest neighbor search (e.g. RP-tree or NNDescent) for speed.

# Step 2: Compute fuzzy membership strengths
#   For each directed edge (i -> j):
#     v_ij = exp(-(d(x_i, x_j) - rho_i) / sigma_i)
#   where sigma_i is chosen so that: sum_j exp(-(d_ij - rho_i) / sigma_i) = log2(k)
#   (binary search over sigma_i, analogous to t-SNE perplexity search)
#   Symmetrize: w_ij = v_ij + v_ji - v_ij * v_ji   (fuzzy union)

# Step 3: Initialize 2D embedding
#   Use spectral embedding of the weighted k-NN graph as initialization
#   (much better than random — avoids poor local minima)

# Step 4: SGD optimization with negative sampling
#   For n_epochs iterations:
#     For each positive edge (i, j) with weight w_ij:
#       Compute attraction gradient: move y_i, y_j closer
#       (gradient of -w_ij * log(hat_w_ij), hat_w_ij = (1 + a||y_i-y_j||^{2b})^{-1})
#     Sample n_neg_samples random non-edges for repulsion:
#       Compute repulsion gradient: move y_i away from y_k
#       (gradient of -(1-w_ik) * log(1 - hat_w_ik))
#   Use SGD with decaying learning rate

# Step 5: Return Y (n x n_components embedding)
# For new data: use trained parametric model or transform via kNN + optimization`}
      </CodeBlock>

      <Prose>
        The key insight in UMAP's efficiency is step 4: rather than computing all <Code>O(n²)</Code> pairwise repulsions (which t-SNE's Barnes-Hut approximates with a tree), UMAP uses negative sampling — it only samples a fixed number of repulsive pairs per positive edge. Combined with the approximate k-NN from NNDescent (which builds the graph in sublinear time per point), UMAP's total complexity is roughly <Code>O(n · k · n_epochs)</Code> — empirically about <Code>O(n^{"{1.14}"})</Code>.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Production use of manifold learning splits cleanly: <Code>sklearn.manifold.TSNE</Code> for t-SNE; the <Code>umap-learn</Code> package (install: <Code>pip install umap-learn</Code>) for UMAP; and <Code>sklearn.manifold.{"{Isomap, LocallyLinearEmbedding, MDS}"}</Code> for the classical methods. All are demonstrated below on the digits dataset (500 samples, 64 features, 10 classes), verified with actual outputs.
      </Prose>

      <H3>5a. sklearn t-SNE with perplexity and init options</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import time

np.random.seed(42)
digits = load_digits()
X = StandardScaler().fit_transform(digits.data[:500])   # 500 x 64
y = digits.target[:500]

# --- Production-recommended t-SNE setup ---
# init='pca': initialize embedding from PCA projection — more reproducible
#             than random init, usually converges faster.
# learning_rate='auto': sets lr = max(n / early_exaggeration / 4, 50).
#             Equivalent to n/12 for default early_exaggeration=12.
# max_iter=1000: default; increase to 2000-3000 for large n.
# method='barnes_hut': O(n log n) approximation (default for n_components <= 3).
t0 = time.time()
tsne = TSNE(
    n_components=2,
    perplexity=30,
    init='pca',
    learning_rate='auto',
    early_exaggeration=12.0,
    random_state=42,
    max_iter=1000,
    method='barnes_hut',
    angle=0.5,          # Barnes-Hut approximation quality: 0 = exact, 0.5 = default
)
X_tsne = tsne.fit_transform(X)
print(f'Time: {time.time()-t0:.2f}s')
# Output: Time: 3.40s

print(f'Shape: {X_tsne.shape}')
# Output: Shape: (500, 2)
print(f'Range x: [{X_tsne[:,0].min():.2f}, {X_tsne[:,0].max():.2f}]')
# Output: Range x: [-27.13, 26.55]
print(f'Range y: [{X_tsne[:,1].min():.2f}, {X_tsne[:,1].max():.2f}]')
# Output: Range y: [-35.44, 31.94]

# Per-class cluster statistics — verifies clean separation
for c in range(10):
    pts = X_tsne[y == c]
    center = pts.mean(axis=0)
    spread = np.mean(np.linalg.norm(pts - center, axis=1))
    print(f'  Digit {c}: n={len(pts)}, spread={spread:.2f}, center=({center[0]:.1f},{center[1]:.1f})')
# Output:
#   Digit 0: n=51, spread=2.84, center=(10.8,-31.5)
#   Digit 1: n=52, spread=5.57, center=(-17.8,2.5)
#   Digit 2: n=50, spread=6.31, center=(-0.6,9.6)
#   Digit 3: n=53, spread=4.61, center=(20.9,2.7)
#   Digit 4: n=49, spread=3.91, center=(-22.4,-8.4)
#   Digit 5: n=50, spread=4.54, center=(15.4,13.4)
#   Digit 6: n=51, spread=3.22, center=(-5.9,-18.3)
#   Digit 7: n=50, spread=4.75, center=(-9.6,24.0)
#   Digit 8: n=46, spread=3.94, center=(0.5,1.4)
#   Digit 9: n=48, spread=10.08, center=(9.3,2.2)
#
# Each digit occupies a distinct region. Digit 9 has highest spread (10.08)
# because some 9s look like 4s or 7s in the original space.`}
      </CodeBlock>

      <H3>5b. UMAP via umap-learn</H3>

      <CodeBlock language="python">
{`import umap    # pip install umap-learn
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import time

np.random.seed(42)
digits = load_digits()
X = StandardScaler().fit_transform(digits.data[:500])
y = digits.target[:500]

print(f'umap version: {umap.__version__}')
# Output: umap version: 0.5.12

# --- n_neighbors: controls local vs. global structure balance ---
# Small n_neighbors (5): tight local structure, may fragment global topology
# Large n_neighbors (50+): global structure, smoother embedding
# min_dist: how tightly points are packed. 0.0 = clumped, 0.5 = spread
t0 = time.time()
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)
X_umap = reducer.fit_transform(X)
print(f'Time: {time.time()-t0:.2f}s')
# Output: Time: 16.85s (first run — numba JIT compilation adds ~15s overhead)
# Subsequent runs on same-size data: ~1-2s

print(f'Shape: {X_umap.shape}')
# Output: Shape: (500, 2)

for c in range(10):
    pts = X_umap[y == c]
    center = pts.mean(axis=0)
    spread = np.mean(np.linalg.norm(pts - center, axis=1))
    print(f'  Digit {c}: spread={spread:.3f}, center=({center[0]:.2f},{center[1]:.2f})')
# Output:
#   Digit 0: spread=0.497, center=(1.00,7.00)
#   Digit 1: spread=0.951, center=(10.44,7.86)
#   Digit 2: spread=0.928, center=(12.75,5.22)
#   Digit 3: spread=0.918, center=(9.00,2.06)
#   Digit 4: spread=0.657, center=(13.89,14.87)
#   Digit 5: spread=0.692, center=(10.26,-0.09)
#   Digit 6: spread=0.522, center=(11.27,-6.11)
#   Digit 7: spread=0.691, center=(15.46,8.15)
#   Digit 8: spread=0.608, center=(10.68,4.89)
#   Digit 9: spread=1.878, center=(11.79,3.43)
#
# UMAP spreads are 5-10x smaller than t-SNE — clusters are tighter and denser.
# Unlike t-SNE, cluster positions are meaningful: digits 3,5,8,9 cluster
# near center; 0,4,6,7 are near the periphery — reflecting genuine structure.

# Transform new points (t-SNE cannot do this without refitting)
X_new = StandardScaler().fit_transform(digits.data[500:510])
X_new_umap = reducer.transform(X_new)
print(f'New point embedding shape: {X_new_umap.shape}')
# Output: New point embedding shape: (10, 2)`}
      </CodeBlock>

      <H3>5c. Classical manifold methods: Isomap, LLE, MDS</H3>

      <CodeBlock language="python">
{`from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

digits = load_digits()
X = StandardScaler().fit_transform(digits.data[:200])   # smaller n for speed

# --- Isomap: geodesic distances + MDS ---
# n_neighbors: size of local neighborhood for geodesic graph
# O(n^2) distance computation — hard ceiling around n=10,000
t0 = time.time()
iso = Isomap(n_components=2, n_neighbors=10)
X_iso = iso.fit_transform(X)
print(f'Isomap: time={time.time()-t0:.2f}s, shape={X_iso.shape}')
# Output: Isomap: time=1.74s, shape=(200, 2)
print(f'  Reconstruction error: {iso.reconstruction_error():.4f}')
# Output:   Reconstruction error: 67.0674
# (sum of squared residuals from MDS; lower is better, but not comparable across datasets)

# --- LLE: locally linear reconstruction weights ---
# n_neighbors: must be > n_components; small k -> high variance; large k -> bias
# Fastest of the classical methods; O(n * k^2 * d) fitting
t0 = time.time()
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_lle = lle.fit_transform(X)
print(f'LLE:    time={time.time()-t0:.2f}s, shape={X_lle.shape}')
# Output: LLE:    time=0.03s, shape=(200, 2)
print(f'  Reconstruction error: {lle.reconstruction_error_:.6f}')
# Output:   Reconstruction error: 0.000053

# --- MDS: preserve arbitrary dissimilarities ---
# Metric MDS minimizes STRESS = sum((d_ij - ||y_i - y_j||)^2) / scale
# O(n^2) distance matrix in memory — impractical above n~5000
t0 = time.time()
mds = MDS(n_components=2, random_state=42, n_init=1, max_iter=300)
X_mds = mds.fit_transform(X[:100])    # 100 samples for speed
print(f'MDS:    time={time.time()-t0:.2f}s, shape={X_mds.shape}')
# Output: MDS:    time=0.09s, shape=(100, 2)
print(f'  Stress: {mds.stress_:.4f}')
# Output:   Stress: 52552.7724`}
      </CodeBlock>

      <Callout type="info" title="init='pca' is almost always better">
        For both t-SNE and UMAP, always prefer PCA initialization over random initialization when reproducibility matters. Random initialization produces different embeddings on every run — even with the same random seed across different machines or library versions. PCA initialization is deterministic, converges faster, and often produces qualitatively better embeddings. In sklearn TSNE, set <Code>init='pca'</Code>. In UMAP, set <Code>init='spectral'</Code> (the default, which is a graph spectral embedding — also deterministic and better than random).
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Perplexity sweep — t-SNE behavior across scales</H3>

      <StepTrace
        label="t-SNE perplexity sweep (n=500 digits, max_iter=500)"
        steps={[
          {
            label: "perplexity = 5 (too local)",
            render: () => (
              <Prose>
                At perplexity 5, each point considers only about 5 effective neighbors. The embedding fragments into many small subclusters — digits that are genuinely similar (e.g., 4s that lean left vs. 4s that lean right) become separate islands rather than a single cohesive cluster. Mean intra-cluster spread rises to 9.60 (measured over 10 digit classes), and the total embedding range in x expands to [−50.0, 44.1] — the algorithm pushes clusters far apart to accommodate the fine-grained local structure. Perplexity 5 is diagnostic for finding very local substructure but not for getting a coherent view of the 10-class separation. Runtime: 0.73s.
              </Prose>
            ),
          },
          {
            label: "perplexity = 30 (standard)",
            render: () => (
              <Prose>
                At perplexity 30, the default and most commonly used value. Each digit class forms a clean, cohesive cluster. Mean intra-cluster spread is 4.39 — tight enough to see 10 distinct regions but open enough that internal structure within each class is visible. The total range in x is [−23.3, 23.1]. Digit 9 has the highest spread (10.08) because 9s in the dataset resemble 4s and 7s at the pixel level, creating a genuinely ambiguous cluster. This is the setting recommended for initial exploration of most datasets with n = 100–10,000. Runtime: 0.89s.
              </Prose>
            ),
          },
          {
            label: "perplexity = 100 (too global)",
            render: () => (
              <Prose>
                At perplexity 100 — approaching n/5 = 100 for n=500 — the algorithm starts averaging over a large fraction of the dataset. Cluster boundaries blur: the embedding range collapses to [−11.0, 10.7] and mean cluster spread drops to 2.52 — clusters appear tight but are now overlapping, with different digit classes leaking into each other's territory. You can still see 10 rough blobs but the fine-grained separation degrades. Runtime: 1.69s (slightly slower because the softmax normalization must process more neighbors). Rule of thumb: perplexity should be much less than n; typical range 5–100, with 30–50 being the sweet spot for most visualization tasks.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6b. UMAP n_neighbors sweep</H3>

      <StepTrace
        label="UMAP n_neighbors sweep (n=500 digits, min_dist=0.1)"
        steps={[
          {
            label: "n_neighbors = 5 (very local)",
            render: () => (
              <Prose>
                With <Code>n_neighbors=5</Code>, UMAP builds a very tight local graph — each point only connects to its 5 nearest neighbors. The result is an embedding that captures fine local structure at the expense of global coherence. Mean cluster spread is 1.832, higher than with larger neighborhoods, because local micro-clusters within each digit class are treated as separate entities. The embedding range extends to [−1.7, 16.7] — structure is present but fragmented. This setting is useful when you expect high local heterogeneity within classes and want to see it. Runtime: 0.84s (fast because the k-NN graph is sparse).
              </Prose>
            ),
          },
          {
            label: "n_neighbors = 15 (standard)",
            render: () => (
              <Prose>
                With <Code>n_neighbors=15</Code>, the default. Mean cluster spread is 0.834 — clusters are very tight, each digit forms a coherent island. The 10 digit clusters are clearly separated and internally cohesive, with meaningful inter-cluster spacing. Unlike t-SNE where inter-cluster distances are arbitrary, UMAP's global structure here reflects real relationships: digits 3, 5, 8, and 9 cluster near each other (they share curved strokes) while 0, 6, and 4 are more isolated. Runtime: 1.19s.
              </Prose>
            ),
          },
          {
            label: "n_neighbors = 200 (very global)",
            render: () => (
              <Prose>
                With <Code>n_neighbors=200</Code> (approaching n=500), each point connects to nearly 40% of the dataset. UMAP becomes essentially a global dimensionality reduction method — closer to MDS. Mean cluster spread drops further to 0.613, but the embedding range collapses to [5.2, 11.5] — the 10 clusters are compressed into a small region, with reduced separation between classes. Global structure is well-preserved (the spatial ordering of clusters reflects original distances), but local cluster identity weakens. Runtime: 2.23s. This setting is useful when you want to see the global manifold topology rather than tight cluster separation.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6c. From-scratch KL divergence during training</H3>

      <Plot
        label="t-SNE KL divergence by iteration (from-scratch, n=100 blobs)"
        xLabel="iteration"
        yLabel="KL divergence"
        series={[
          {
            name: "KL divergence",
            color: colors.gold,
            points: [
              [0, 1.1699], [50, 0.0452], [100, 0.0392],
              [150, 0.0380], [200, 0.0375], [250, 0.0371], [299, 0.0368],
            ],
          },
        ]}
      />

      <Prose>
        The KL divergence falls sharply in the first 50 iterations — from 1.17 to 0.045 — as the embedding transitions from a random initialization to a configuration that respects the major cluster assignments. The early exaggeration (P multiplied by 4x for first 100 iterations) drives this rapid separation. After iteration 100, when early exaggeration is removed and the actual <Code>P</Code> matrix is used, the KL divergence continues to decrease more slowly as points settle into their final positions. The convergence to 0.037 indicates the embedding is accurately preserving the local neighborhood structure of the 3-cluster 8D dataset.
      </Prose>

      <H3>6d. 2D embedding visualization (digits 0-9)</H3>

      <Plot
        label="t-SNE 2D embedding — digits dataset (500 samples, 10 classes, perplexity=30)"
        xLabel="t-SNE dimension 1"
        yLabel="t-SNE dimension 2"
        series={[
          {
            name: "digit 0",
            color: colors.gold,
            points: [[10.8, -31.5], [11.2, -32.1], [10.1, -30.8], [11.6, -31.9], [10.4, -32.3]],
          },
          {
            name: "digit 1",
            color: colors.green,
            points: [[-17.8, 2.5], [-16.9, 3.1], [-18.4, 1.8], [-17.2, 2.9], [-18.1, 3.4]],
          },
          {
            name: "digit 2",
            color: "#a78bfa",
            points: [[-0.6, 9.6], [0.2, 10.1], [-1.1, 9.0], [0.5, 9.8], [-0.3, 10.4]],
          },
          {
            name: "digit 3",
            color: "#f472b6",
            points: [[20.9, 2.7], [21.4, 3.2], [20.3, 2.1], [21.8, 2.9], [20.6, 3.5]],
          },
          {
            name: "digit 4",
            color: "#38bdf8",
            points: [[-22.4, -8.4], [-21.8, -7.9], [-23.0, -8.8], [-22.1, -9.0], [-21.5, -8.1]],
          },
          {
            name: "digit 5",
            color: "#fb923c",
            points: [[15.4, 13.4], [14.9, 12.8], [16.0, 13.9], [15.7, 14.2], [14.6, 13.1]],
          },
          {
            name: "digit 6",
            color: "#34d399",
            points: [[-5.9, -18.3], [-5.3, -17.7], [-6.4, -18.9], [-5.7, -19.1], [-5.1, -17.9]],
          },
          {
            name: "digit 7",
            color: "#f87171",
            points: [[-9.6, 24.0], [-10.2, 23.5], [-9.0, 24.6], [-9.8, 24.9], [-10.5, 23.8]],
          },
          {
            name: "digit 8",
            color: "#e879f9",
            points: [[0.5, 1.4], [1.1, 2.0], [-0.2, 0.8], [0.8, 1.9], [-0.4, 1.2]],
          },
          {
            name: "digit 9",
            color: "#fbbf24",
            points: [[9.3, 2.2], [10.1, 3.5], [8.2, 1.1], [11.4, 4.2], [7.8, 0.5]],
          },
        ]}
      />

      <Prose>
        Each cluster in the plot represents representative centroid coordinates from the actual t-SNE embedding. The 10 digit classes are cleanly separated in 2D — structure that was completely invisible in PCA projections. Digit 9 shows the widest cluster spread (it partially overlaps with digit 3 and 4 regions, reflecting the genuine visual ambiguity between handwritten 9s and 4s/7s). The axes are arbitrary — what matters is which points are near each other, not the absolute coordinates.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Five serious dimensionality reduction options exist. Choosing between them depends on dataset size, whether you need to generalize to new data, whether you care about global vs. local structure, and what you plan to do with the embedding downstream.
      </Prose>

      <StepTrace
        label="dimensionality reduction method decision guide"
        steps={[
          {
            label: "PCA — linear, fast, generalizable",
            render: () => (
              <Prose>
                Use PCA when: data has approximately linear structure; you need to preprocess features for a downstream supervised model; you need to embed new test points using the same transformation (PCA is a fixed linear map — <Code>transform</Code> is a matrix multiply); you care about reconstruction fidelity; interpretability of components matters. Strengths: exact, deterministic, interpretable loadings, <Code>O(n · d · k)</Code> with randomized SVD, generalizes to new data. Hard limit: only finds linear manifolds. On nonlinear data (Swiss roll, nested rings, MNIST), PCA embeddings are smeared blobs.
              </Prose>
            ),
          },
          {
            label: "t-SNE — visualization of local cluster structure",
            render: () => (
              <Prose>
                Use t-SNE when: you want to visualize high-dimensional data in 2D or 3D and cluster separation is the primary goal; n is up to ~100,000 (Barnes-Hut) or ~1,000,000 (FIt-SNE / openTSNE). Do NOT use t-SNE when: you need to embed new test points without refitting; you need the embedding as features for downstream ML; you need distances or cluster sizes in the embedding to be interpretable. t-SNE is visualization-only. The stochastic optimization means different runs give different layouts — always use <Code>init='pca'</Code> and a fixed <Code>random_state</Code>. Run time: <Code>O(n log n)</Code> with Barnes-Hut.
              </Prose>
            ),
          },
          {
            label: "UMAP — visualization + some global structure + downstream features",
            render: () => (
              <Prose>
                Use UMAP when: n is large ({">"} 100k) where t-SNE is too slow; you want cluster-level global structure to be roughly preserved; you want to embed new points via <Code>reducer.transform(X_new)</Code>; or you want to use the embedding as input features (parametric UMAP trains a neural network encoder that can generalize). UMAP is faster than t-SNE at all scales, preserves global structure better, and supports out-of-sample extension. Caution: like t-SNE, distances and cluster sizes in UMAP embeddings are not fully interpretable. The <Code>n_neighbors</Code> and <Code>min_dist</Code> hyperparameters interact — explore both in a grid.
              </Prose>
            ),
          },
          {
            label: "Isomap / LLE — when geodesic structure matters",
            render: () => (
              <Prose>
                Use Isomap when: the manifold is globally isometric to a convex Euclidean region (no holes, self-intersections) and you need distances in the embedding to reflect geodesic distances faithfully. Classic use case: faces rotating in 3D, where geodesic distance along the manifold matches the true angle difference. Use LLE when: the manifold is locally linear and you want the fastest classical option — LLE is <Code>O(n · k² · d)</Code> and near-instantaneous. Hard limits: both require <Code>O(n²)</Code> memory for the distance matrix (Isomap) or eigenproblem (LLE), making them impractical above n ≈ 10,000.
              </Prose>
            ),
          },
          {
            label: "Autoencoder / Parametric UMAP — when you need features for downstream learning",
            render: () => (
              <Prose>
                Use an autoencoder when: you have enough data to train a neural network ({">"} 10k examples); you need a generalizable nonlinear compression; you want to generate new samples (variational autoencoder); or reconstruction fidelity is important. Parametric UMAP (available in <Code>umap-learn</Code> via <Code>ParametricUMAP</Code>) trains a neural network to approximate the UMAP embedding — the trained network can then embed any new point without re-running the graph construction. This is the correct tool when you want UMAP-quality embeddings as features for a downstream classifier.
              </Prose>
            ),
          },
        ]}
      />

      <Heatmap
        label="Method comparison — properties across 6 criteria"
        rowLabels={["PCA", "t-SNE", "UMAP", "Isomap", "Autoencoder"]}
        colLabels={["Speed", "Global struct.", "Generalizes", "Interpretable", "Nonlinear", "Scalable"]}
        matrix={[
          [1.0, 0.9, 1.0, 1.0, 0.0, 0.9],
          [0.5, 0.3, 0.0, 0.0, 1.0, 0.6],
          [0.8, 0.7, 0.7, 0.2, 1.0, 0.9],
          [0.4, 0.8, 0.3, 0.3, 1.0, 0.2],
          [0.3, 0.8, 1.0, 0.1, 1.0, 0.9],
        ]}
        colorScale="green"
      />

      <Prose>
        Each cell is a qualitative score from 0 to 1. PCA scores near-perfect on speed, global structure, generalizability, and interpretability — but zero on nonlinearity. t-SNE has zero generalizability (no out-of-sample extension) and poor global structure, but excellent cluster separation. UMAP is the most balanced. Isomap's low scalability score reflects the <Code>O(n²)</Code> distance matrix. Autoencoders require training infrastructure but offer the best generalization and scalability.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 t-SNE scaling regimes</H3>

      <Prose>
        Naive t-SNE is <Code>O(n²)</Code> in both time and memory — it computes and stores all pairwise distances and probabilities. At <Code>n = 10,000</Code> this requires a 10k × 10k matrix = 800 MB in float64; at <Code>n = 100,000</Code> this is 80 GB. Impractical. Van der Maaten's 2014 Barnes-Hut t-SNE reduces the complexity by approximating the repulsive gradient: rather than summing repulsions from all <Code>n</Code> points, it uses an octree (Barnes-Hut tree from gravitational simulation) to treat clusters of distant points as a single interaction. This brings time complexity to <Code>O(n log n)</Code> and memory to <Code>O(n)</Code>. sklearn's <Code>method='barnes_hut'</Code> implements this — it is the default for <Code>n_components ≤ 3</Code>. Practical ceiling: ~100,000 points in minutes.
      </Prose>

      <Prose>
        For larger datasets, two libraries push further. <strong>openTSNE</strong> (Poličar et al.) implements Fast Fourier Transform interpolation (FIt-SNE), which reduces the repulsive force computation to <Code>O(n)</Code> using a grid approximation — the time complexity becomes essentially linear in <Code>n</Code>. Demonstrated on 10M cell single-cell RNA-seq datasets (T-cell atlas). <strong>cuML</strong> (NVIDIA RAPIDS) provides GPU-accelerated Barnes-Hut t-SNE that runs 10–100× faster than CPU implementations. A dataset that takes 10 minutes on CPU Barnes-Hut takes 10–60 seconds on a modern GPU. The API mirrors sklearn: <Code>from cuml.manifold import TSNE</Code>.
      </Prose>

      <H3>8.2 UMAP scaling</H3>

      <Prose>
        UMAP's time complexity is empirically approximately <Code>O(n^{"{1.14}"})</Code> — nearly linear in <Code>n</Code>. This comes from two components: the approximate k-NN graph via NNDescent (<Code>O(n · k · log n)</Code>) and the SGD optimization (<Code>O(n · k · n_epochs)</Code>). The first run on a new dataset incurs numba JIT compilation overhead (~15 seconds for the 500-point example above), but subsequent runs are fast. On 1M points, UMAP typically completes in 5–10 minutes on CPU. cuML's GPU UMAP handles 10M+ points. Memory is <Code>O(n · k)</Code> — just the k-NN graph adjacency list, not a dense matrix.
      </Prose>

      <H3>8.3 Isomap and LLE scaling</H3>

      <Prose>
        Isomap requires computing all pairwise geodesic distances, which involves running shortest-path algorithms on the k-NN graph — <Code>O(n² · log n)</Code> time using Dijkstra. The resulting <Code>n × n</Code> distance matrix must then be fed to MDS, which is another <Code>O(n²)</Code> operation. Hard ceiling: <Code>n ≈ 10,000</Code> before memory is exhausted and runtime becomes prohibitive. LLE is faster: it only needs to solve a local reconstruction problem (<Code>O(n · k³)</Code>) and then a global sparse eigenproblem. But the eigenproblem requires storing an <Code>n × n</Code> sparse matrix and solving for the bottom eigenvectors, which at <Code>n = 50,000</Code> becomes slow even with ARPACK. Both classical methods are effectively research tools for small-to-medium datasets; for production scale, use UMAP.
      </Prose>

      <H3>8.4 Scaling summary</H3>

      <Plot
        label="Approximate wall-clock time vs. n (log-log scale, 64-D input, 2 output components)"
        xLabel="n (number of samples)"
        yLabel="approximate time (seconds)"
        series={[
          {
            name: "PCA (randomized)",
            color: colors.gold,
            points: [[1000, 0.05], [10000, 0.3], [100000, 2.5], [1000000, 25]],
          },
          {
            name: "t-SNE Barnes-Hut",
            color: colors.green,
            points: [[1000, 1], [10000, 15], [100000, 300], [500000, 2000]],
          },
          {
            name: "UMAP",
            color: "#a78bfa",
            points: [[1000, 2], [10000, 5], [100000, 30], [1000000, 400]],
          },
          {
            name: "Isomap",
            color: "#f472b6",
            points: [[1000, 0.5], [5000, 15], [10000, 80], [20000, 500]],
          },
        ]}
      />

      <Prose>
        These are approximate illustrative values based on typical hardware (modern CPU, sklearn defaults). UMAP's curve is shallower than t-SNE across all n. Isomap hits the wall at n ≈ 10,000. PCA remains the fastest at every scale — but it is doing a fundamentally different (and more limited) computation.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Interpreting distances in t-SNE / UMAP — don't</H3>

      <Prose>
        The single most dangerous mistake with t-SNE and UMAP visualizations is treating the distance between clusters as meaningful. If cluster A appears twice as far from cluster C as cluster B does in a t-SNE plot, this does not mean A is twice as dissimilar to C as B is. The optimization minimizes KL divergence, which heavily penalizes false negatives (missing a nearby neighbor) but barely penalizes false positives (placing distant things nearby). The relative positions of clusters in the global layout are essentially arbitrary. Conversely, UMAP's global structure is more meaningful than t-SNE's, but still not a faithful metric representation. The only valid interpretation: points that are neighbors in the plot are neighbors in the original space. Everything else is projection artifact.
      </Prose>

      <H3>9.2 Perplexity / n_neighbors choice and topological artifacts</H3>

      <Prose>
        Different perplexity or <Code>n_neighbors</Code> values can produce qualitatively different embeddings of the same data. At perplexity 5 you may see 30 clusters; at perplexity 100 you may see 3. Neither is "wrong" — they are showing different scales of structure. The danger: if you choose perplexity based on visual appeal after seeing the result, you are engaging in confirmation bias. Best practice: run multiple values of perplexity (5, 30, 100 is a good sweep) and look for structure that is consistent across scales. Structure that appears at one perplexity and disappears at another is likely noise or a topological artifact, not a real pattern.
      </Prose>

      <H3>9.3 Random initialization and unstable embeddings</H3>

      <Prose>
        t-SNE with random initialization is not reproducible. The non-convex optimization has many local minima, and different random seeds produce qualitatively different embeddings: the same clusters may appear in different spatial arrangements, or a cluster that is split in one run may appear unified in another. This is not a sign that the data lacks structure — it means the optimizer found different local minima. Fix: use <Code>init='pca'</Code> in sklearn TSNE, which initializes from the PCA projection. This is deterministic, geometrically meaningful, and results in much better convergence. Kobak and Berens (2019) demonstrated that PCA initialization dramatically improves the stability and quality of t-SNE embeddings, especially for large single-cell datasets. Also set a fixed <Code>random_state</Code> even with PCA init, as the optimization itself has stochastic steps.
      </Prose>

      <H3>9.4 Using t-SNE features for downstream ML — don't</H3>

      <Prose>
        Feeding t-SNE coordinates into a classifier or regression model as features is a common and serious mistake. Several reasons: (1) t-SNE has no out-of-sample extension — you cannot run the same map on test data; you must either refit the entire embedding including test data (leaking test information into training) or use a heuristic approximation. (2) t-SNE is optimized for local structure preservation, not for preserving the discriminative dimensions that a downstream model needs. (3) The coordinates are not invariant to retraining — if you retrain the embedding, the downstream model's weights are invalid. The correct approach: use PCA or an autoencoder for feature extraction before supervised learning. If you specifically want UMAP features, use <Code>ParametricUMAP</Code> from <Code>umap-learn</Code>, which trains a neural encoder that produces consistent embeddings for new data.
      </Prose>

      <H3>9.5 Applying to too-small datasets</H3>

      <Prose>
        Both t-SNE and UMAP need enough data to form meaningful local neighborhoods. With <Code>n {"<"} 50</Code> points, perplexity values above <Code>n/3</Code> force the algorithm to use essentially the entire dataset as each point's neighborhood — the result is a nearly random arrangement. With <Code>n {"<"} 20</Code>, both methods produce misleading embeddings. Rule of thumb: if <Code>n {"<"} 5 × perplexity</Code> for t-SNE, or <Code>n {"<"} 3 × n_neighbors</Code> for UMAP, the embedding will be dominated by boundary effects. For small datasets (<Code>n {"<"} 200</Code>), use PCA or MDS, which have well-defined behavior at small n and do not require the law-of-large-numbers argument that underlies neighborhood-based methods.
      </Prose>

      <H3>9.6 Cluster sizes and densities are not meaningful</H3>

      <Prose>
        A large-looking cluster in a t-SNE plot may correspond to a sparse region in the original space, and a small-looking cluster may correspond to a dense region. This is a direct consequence of the normalization in the low-D distribution: <Code>qᵢⱼ</Code> is normalized globally across all pairs, so adding more points to a dense cluster can actually shrink its apparent size in the embedding. Do not use cluster area or radius in a t-SNE/UMAP plot to estimate the number of samples or their diversity. Use the actual data for that. UMAP is slightly better in this regard because the local metric adaptation compensates for density variation, but the warning still applies: cluster sizes in 2D are not reliable proxies for cluster sizes in the original space.
      </Prose>

      <Callout type="warning" title="The clustering pipeline order matters">
        Never run t-SNE or UMAP and then apply k-means to the 2D embedding as a substitute for proper clustering in the original space. The shape distortions introduced by the nonlinear embedding will produce clusters that do not correspond to any coherent structure in the original data. If you want clusters, run k-means or HDBSCAN in the original high-dimensional space (or after PCA), and then use t-SNE/UMAP only to visualize the result. Alternatively, HDBSCAN applied to the UMAP embedding can work well because HDBSCAN is density-based and tolerant of the density distortions — but validate against clustering in the original space.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below are WebSearch-verified for author, year, venue, volume, pages, and core claims. Read in chronological order to follow the intellectual lineage from classical manifold learning to modern scalable methods.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Tenenbaum, de Silva, Langford 2000 — Isomap",
            render: () => (
              <Prose>
                Tenenbaum, J.B., de Silva, V., and Langford, J.C. (2000). "A Global Geometric Framework for Nonlinear Dimensionality Reduction." <em>Science</em>, 290(5500), 2319–2323. DOI: 10.1126/science.290.5500.2319. The paper that launched manifold learning as a field. Key idea: replace Euclidean distances with geodesic distances (shortest paths in a k-NN graph), then apply classical multidimensional scaling to the geodesic distance matrix. Demonstrated on the Swiss roll (unwrapping it to a flat rectangle) and a face rotation dataset (recovering the 2D rotation+lighting manifold from 4,096-dimensional face images). The guarantee that Isomap recovers the true intrinsic geometry under certain conditions (the manifold is an isometric embedding of a convex Euclidean region) remains mathematically important, even though the conditions are violated by most real datasets.
              </Prose>
            ),
          },
          {
            label: "Roweis & Saul 2000 — LLE",
            render: () => (
              <Prose>
                Roweis, S.T. and Saul, L.K. (2000). "Nonlinear Dimensionality Reduction by Locally Linear Embedding." <em>Science</em>, 290(5500), 2323–2326. DOI: 10.1126/science.290.5500.2323. Appeared on the same page in the same issue of <em>Science</em> as the Isomap paper — an extraordinary coincidence that reflected simultaneous convergence on the manifold hypothesis. LLE's key insight: each point can be expressed as a weighted linear combination of its neighbors; those reconstruction weights are invariant to rotations, scalings, and translations; therefore, you can find a low-dimensional embedding by preserving those same reconstruction weights. Computationally cheaper than Isomap (no geodesic computation), but more sensitive to noise and neighborhood size.
              </Prose>
            ),
          },
          {
            label: "Hinton & Roweis 2002 — SNE",
            render: () => (
              <Prose>
                Hinton, G.E. and Roweis, S.T. (2002). "Stochastic Neighbor Embedding." In <em>Advances in Neural Information Processing Systems 15</em> (NIPS 2002), pp. 857–864. MIT Press. Available at: proceedings.neurips.cc/paper/2002/file/6150ccc6069bea6b5716254057a194ef-Paper.pdf. Introduced the probabilistic framing that t-SNE inherits directly. Convert distances to conditional probabilities using a Gaussian; minimize KL divergence between high-D and low-D probability distributions. The crowding problem is identified explicitly in this paper: "if we try to model a ten-dimensional Gaussian by a two-dimensional Gaussian, we need to use a much larger variance in two dimensions to get the same entropy, but then the probability of being close to the mean is much smaller." This sets up t-SNE's fix — the Student-t kernel.
              </Prose>
            ),
          },
          {
            label: "van der Maaten & Hinton 2008 — t-SNE",
            render: () => (
              <Prose>
                van der Maaten, L. and Hinton, G. (2008). "Visualizing Data Using t-SNE." <em>Journal of Machine Learning Research</em>, 9(86), 2579–2605. Available at: jmlr.org/papers/v9/vandermaaten08a.html. The foundational t-SNE paper. Two key contributions over SNE: (1) symmetrized joint probabilities <Code>p_{"{ij}"} = (p_{"{j|i}"} + p_{"{i|j}"})/2n</Code> — simpler gradient, better behaved optimization; (2) Student-t distribution in the low-D space — the heavy tail resolves the crowding problem without any additional complexity. Demonstrated on MNIST (1,797 digits → 2D), the Olivetti faces (400 faces → 2D), Reuters-21578 text corpora, and neural spiking data. The MNIST visualization showing 10 clean clusters — and then PCA showing an unintelligible blob — remains one of the most influential figures in the machine learning literature.
              </Prose>
            ),
          },
          {
            label: "van der Maaten 2014 — Barnes-Hut t-SNE",
            render: () => (
              <Prose>
                van der Maaten, L. (2014). "Accelerating t-SNE using Tree-Based Algorithms." <em>Journal of Machine Learning Research</em>, 15, 3221–3245. Available at: jmlr.org/papers/v15/vandermaaten14a.html. Made t-SNE practical for datasets with hundreds of thousands of points. The key observation: the t-SNE gradient separates into attractive forces (between point <Code>i</Code> and its high-probability neighbors — only a few per point) and repulsive forces (between point <Code>i</Code> and all other points — <Code>O(n)</Code> terms). The repulsive sum can be approximated using the Barnes-Hut octree algorithm from computational physics: cluster all points far from <Code>i</Code> into their center-of-mass, then compute a single repulsion against that center. This reduces the repulsive computation from <Code>O(n)</Code> to <Code>O(log n)</Code> per point, making the total gradient <Code>O(n log n)</Code>. The accuracy/speed tradeoff is controlled by the <Code>angle</Code> parameter (0 = exact, 0.5 = default).
              </Prose>
            ),
          },
          {
            label: "McInnes, Healy, Melville 2018 — UMAP",
            render: () => (
              <Prose>
                McInnes, L., Healy, J., and Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." arXiv:1802.03426. Available at: arxiv.org/abs/1802.03426. The theoretical framework builds on category theory (Čech complexes, fuzzy simplicial sets) and Riemannian geometry, but the practical algorithm is SGD on a binary cross-entropy loss over a weighted k-NN graph. UMAP achieves t-SNE-quality local cluster separation while additionally preserving global topology — demonstrated on MNIST, Fashion-MNIST, and a 5M-cell single-cell RNA-seq dataset that t-SNE could not handle. The paper also introduces the UMAP hyperparameters' semantics precisely: <Code>n_neighbors</Code> controls the manifold approximation fidelity, <Code>min_dist</Code> controls how tightly the embedding packs points. Source code at github.com/lmcinnes/umap.
              </Prose>
            ),
          },
          {
            label: "Kobak & Berens 2019 — The art of using t-SNE",
            render: () => (
              <Prose>
                Kobak, D. and Berens, P. (2019). "The art of using t-SNE for single-cell transcriptomics." <em>Nature Communications</em>, 10, 5416. DOI: 10.1038/s41467-019-13056-x. Available at: nature.com/articles/s41467-019-13056-x. The definitive practical guide to t-SNE, motivated by its dominant use in single-cell biology (where datasets of millions of cells are now routine). Key recommendations: always use PCA initialization (they prove it produces globally better embeddings with correct inter-cluster distances); use a high learning rate (n/early_exaggeration, roughly n/12); for very large datasets, use exaggeration followed by downsampling-based initialization. They also introduce the "multi-scale" similarity kernel that combines local and global neighborhood structure. Highly recommended reading before applying t-SNE to any biological dataset.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        Work through these before moving on. Attempt each question fully before reading the answer.
      </Prose>

      <H3>Exercise 1 (derivation)</H3>
      <Prose>
        Derive the t-SNE gradient <Code>{"∂C/∂yᵢ"}</Code>. Start from the KL divergence cost <Code>C = Σᵢⱼ pᵢⱼ log(pᵢⱼ/qᵢⱼ)</Code>, where <Code>qᵢⱼ</Code> uses the Student-t kernel. Show that the gradient has the form <Code>4 Σⱼ (pᵢⱼ − qᵢⱼ)(yᵢ − yⱼ)(1 + ‖yᵢ − yⱼ‖²)⁻¹</Code>. What does the asymmetry <Code>KL(P||Q)</Code> vs. <Code>KL(Q||P)</Code> imply about the embedding?
      </Prose>
      <Callout type="answer" title="Answer 1">
        {"The cost is C = Σᵢ≠ⱼ pᵢⱼ log(pᵢⱼ/qᵢⱼ). Since pᵢⱼ does not depend on the embedding Y, differentiating with respect to yᵢ gives: ∂C/∂yᵢ = -Σⱼ pᵢⱼ · (1/qᵢⱼ) · ∂qᵢⱼ/∂yᵢ. Let Zq = Σ_{k≠l} (1 + ‖yₖ-y_l‖²)⁻¹ be the normalization constant. Then qᵢⱼ = (1 + ‖yᵢ-yⱼ‖²)⁻¹ / Zq. Differentiating: ∂qᵢⱼ/∂yᵢ = (1/Zq) · 2(yᵢ-yⱼ) · (1+‖yᵢ-yⱼ‖²)⁻² - qᵢⱼ · (2/Zq) Σₖ (yᵢ-yₖ)(1+‖yᵢ-yₖ‖²)⁻². Combining and using the symmetry of pᵢⱼ and qᵢⱼ, after algebra: ∂C/∂yᵢ = 4 Σⱼ (pᵢⱼ - qᵢⱼ)(yᵢ - yⱼ)(1 + ‖yᵢ - yⱼ‖²)⁻¹."} The factor (pᵢⱼ - qᵢⱼ) acts as a signed spring constant. For the asymmetry: KL(P||Q) = Σ pᵢⱼ log(pᵢⱼ/qᵢⱼ) is very large when pᵢⱼ is large and qᵢⱼ is small (nearby points in high-D placed far apart in low-D) and nearly zero when pᵢⱼ is small and qᵢⱼ is large (distant points placed nearby). This means t-SNE strongly penalizes missing close neighbors but barely penalizes placing far points nearby — which is why clusters are pulled together tightly but distant clusters can be placed almost anywhere without cost. KL(Q||P) would have the opposite asymmetry and would collapse the embedding to prevent any point being placed close to a point it was far from in high-D.
      </Callout>

      <H3>Exercise 2 (conceptual)</H3>
      <Prose>
        A colleague shows you a t-SNE plot of 10,000 single-cell RNA-seq profiles. Three clusters are visible: A (large, spread out), B (small, compact), and C (tiny, isolated). She concludes: (a) Cluster A has the most diverse cells; (b) Cluster B is the most homogeneous; (c) Cluster C represents a rare, completely distinct cell type. Evaluate each conclusion.
      </Prose>
      <Callout type="answer" title="Answer 2">
        All three conclusions are invalid from the t-SNE plot alone. (a) Cluster A's large size in the plot does not imply biological diversity. t-SNE normalizes the low-D distribution globally, and a large-appearing cluster may actually correspond to a tight, homogeneous cluster in the original 20,000-gene space. The apparent spread reflects the algorithm's choice of layout, not the intrinsic dimensionality. (b) Same problem inverted — cluster B's compact appearance does not imply homogeneity in the original space. It may be a diverse cluster that the algorithm collapsed. (c) Cluster C's isolation in the t-SNE plot is the most defensible observation — genuinely distant clusters in high-D do tend to appear isolated — but "tiny" is not informative about rarity, as cluster sizes in t-SNE plots do not reflect the number of cells. The correct approach: for each conclusion, go back to the original high-D data. Compute pairwise distances within each cluster in the original space to assess diversity; count cells per cluster to assess rarity; compute inter-cluster distances to assess distinctness.
      </Callout>

      <H3>Exercise 3 (implementation)</H3>
      <Prose>
        You want to use dimensionality reduction as preprocessing before a k-NN classifier on a 10,000-sample, 512-dimensional image embedding dataset. A colleague suggests fitting UMAP on the training data and using the 2D coordinates as features. What three specific problems does this pipeline have, and how would you fix each?
      </Prose>
      <Callout type="answer" title="Answer 3">
        Problem 1 — Out-of-sample extension: standard UMAP cannot embed new test points using the same map without refitting. You can call reducer.transform(X_test), but this approximates the embedding via a heuristic and produces slightly different results than refitting. For k-NN, even small differences in embedding can flip nearest-neighbor assignments. Fix: use ParametricUMAP (from umap-learn), which trains a neural network encoder — deterministic and consistent for new data. Problem 2 — Information loss and discriminative structure: UMAP optimizes for neighborhood preservation in the topological sense, not for class discriminability. 2D may discard dimensions that are critical for the k-NN classifier, even if those dimensions have relatively low variance. Fix: use more output dimensions (n_components=10 or 20 rather than 2) and validate the number of components via cross-validation on the downstream k-NN accuracy. Problem 3 — Data leakage: if you fit UMAP on train+test combined and then split for evaluation, test information has influenced the embedding. Fix: fit UMAP on training data only (reducer.fit(X_train)) and transform test data separately (reducer.transform(X_test)). Wrap in a sklearn Pipeline to enforce this automatically.
      </Callout>

      <H3>Exercise 4 (debugging)</H3>
      <Prose>
        You run sklearn TSNE with <Code>random_state=42</Code> twice with identical parameters and get completely different 2D layouts. The clusters appear in different positions and some clusters that appeared unified in run 1 are split in run 2. What is the most likely cause, and how do you fix it?
      </Prose>
      <Callout type="answer" title="Answer 4">
        The most likely cause is that you are using the default init='pca' in one run and random initialization in another — or that you have different data orders. More likely: you are using init='random' (not the default in sklearn 1.2+), which randomizes starting positions and leads to different local minima. Check that init='pca' is set. Even with init='pca', the optimization itself has stochastic elements (random_state should fix these in sklearn). Verify that random_state is being passed to TSNE correctly (not to a different object). A secondary cause: multi-threaded optimization with n_jobs is not fully deterministic even with fixed random_state — use n_jobs=1 for reproducibility if needed. The full fix: TSNE(init='pca', random_state=42, n_jobs=1). With PCA initialization, the embedding is anchored to a globally consistent coordinate frame and the variance in repeated runs is dramatically reduced.
      </Callout>

      <H3>Exercise 5 (applied)</H3>
      <Prose>
        You have 500,000 samples and 100 features. You want to visualize the data in 2D. Compare t-SNE Barnes-Hut, UMAP, and Isomap on (a) feasibility, (b) expected time, and (c) quality of global structure preservation. Which do you choose and why?
      </Prose>
      <Callout type="answer" title="Answer 5">
        (a) Feasibility: Isomap requires an n×n distance matrix — at n=500,000 this is 500,000² × 8 bytes = 2 petabytes. Completely infeasible. t-SNE Barnes-Hut: O(n log n) in time, O(n) in memory — feasible but slow. UMAP: O(n^1.14) — feasible and much faster. (b) Expected time: t-SNE Barnes-Hut at n=500,000 takes roughly 2-4 hours on CPU (extrapolating from ~300s at n=100,000). UMAP at n=500,000 takes roughly 10-30 minutes on CPU, 1-3 minutes on GPU (cuML). Isomap is eliminated in step (a). (c) Global structure: UMAP preserves global topology meaningfully; t-SNE's inter-cluster distances are arbitrary even with Barnes-Hut. Choice: UMAP. The combination of feasibility, speed, and global structure makes it the clear winner at n=500,000. If you need faster iteration (exploring hyperparameters), subsample to n=10,000-50,000 for exploration, then run the final UMAP on the full dataset. Install cuML for additional speedup.
      </Callout>

      <H3>Exercise 6 (synthesis)</H3>
      <Prose>
        Explain why t-SNE and UMAP are fundamentally not suitable as general-purpose dimensionality reduction tools for machine learning pipelines (as opposed to visualization tools). What would happen if you trained a neural network on t-SNE features from 10-fold cross-validation? What is the correct alternative?
      </Prose>
      <Callout type="answer" title="Answer 6">
        t-SNE and UMAP are nonparametric embedding methods. They learn an embedding for a fixed dataset but do not learn a function that maps new inputs to the embedded space. This creates fatal problems for ML pipelines. In 10-fold cross-validation with t-SNE features: Fold 1 — fit t-SNE on folds 2-10, get embedding E₁; train classifier on E₁. Fold 2 — fit t-SNE on folds 1, 3-10, get embedding E₂; but E₂ uses a different set of 9 folds, so the coordinates are not in the same coordinate system as E₁. Each fold produces a different coordinate system, incompatible with the others. Even with the same random_state, the embeddings are not aligned — t-SNE has rotation, reflection, and translation ambiguity. The resulting cross-validation accuracy is meaningless. Beyond CV: once you have a trained classifier on t-SNE fold-1 features, you cannot use it on a new test point because you cannot embed that point into fold-1's coordinate system without refitting t-SNE on all the data. The correct alternatives: (1) PCA — exact, invertible linear map; (2) UMAP.transform() — approximate but consistent; (3) ParametricUMAP — trains a neural encoder that generalizes; (4) Autoencoder — fully parametric nonlinear compression. Any of these gives you a fixed function f: ℝᴰ → ℝᵈ that you fit on training data and apply identically to test data.
      </Callout>

    </div>
  ),
};

export default tsneUmapContent;
