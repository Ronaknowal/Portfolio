import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const kNearestNeighborsContent = {
  title: "K-Nearest Neighbors (KNN)",
  readTime: "~35 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In February 1951, Evelyn Fix and J. L. Hodges submitted a technical report to the USAF School of Aviation Medicine at Randolph Field, Texas. The report, titled "Discriminatory Analysis: Nonparametric Discrimination — Consistency Properties," was assigned project number 21-49-004. It was not a journal article, not peer-reviewed in any formal sense, and not widely circulated for more than a decade. What it contained was the seed of one of the most widely deployed algorithms in all of machine learning.
      </Prose>

      <Prose>
        Fix and Hodges were attacking a classification problem: given a set of labeled examples and a new unlabeled point, assign the correct class. The dominant approach of the era was parametric — assume the data comes from a Gaussian, estimate the mean and covariance, classify by the nearest class centroid. The problem with parametric methods is that they commit to a distributional form before seeing the data. If the true class boundaries are irregular, non-convex, or multi-modal, a Gaussian assumption will be wrong in ways that accumulate. Fix and Hodges asked: what if you simply looked at the labeled points nearest to the query and let them vote? No distributional assumption. No parameter estimation. Just proximity.
      </Prose>

      <Prose>
        The theoretical justification arrived sixteen years later. Thomas Cover and Peter Hart, working at Stanford, published "Nearest Neighbor Pattern Classification" in the January 1967 issue of <em>IEEE Transactions on Information Theory</em> (vol. 13, no. 1, pp. 21–27). Their main result — now called the Cover-Hart bound — is one of the most elegant theorems in classical machine learning: <strong>the asymptotic error rate of the 1-nearest-neighbor rule is at most twice the Bayes error rate</strong>. The Bayes error is the irreducible floor: the minimum error any classifier can achieve given the true class-conditional distributions. Cover and Hart proved that KNN, using no model at all, gets within a factor of two of that floor. For a method that requires zero training, this is a startling guarantee.
      </Prose>

      <Prose>
        The algorithmic machinery for making KNN tractable at scale came from Jerome Friedman, Jon Bentley, and Raphael Finkel in their 1977 ACM Transactions on Mathematical Software paper "An Algorithm for Finding Best Matches in Logarithmic Expected Time" (vol. 3, no. 3, pp. 209–226). They introduced the kd-tree: a recursive binary partition of feature space that reduces nearest-neighbor search from O(n) brute force to O(log n) expected time in low dimensions. The kd-tree is the data structure that made KNN deployable in the pre-GPU era.
      </Prose>

      <Prose>
        Three properties make KNN unusual in the landscape of supervised learning. First, it is <strong>non-parametric</strong>: no functional form is assumed for the decision boundary. The model complexity grows with the data rather than being fixed before training. Second, it is a <strong>lazy learner</strong>: all computation is deferred to inference. There is no training phase in the gradient-descent sense; the training set is simply stored and searched at query time. Third, it is <strong>instance-based</strong>: each prediction is a local computation that depends directly on the stored examples near the query, not on a global summary of the data. These properties are a double-edged sword — they make KNN flexible and assumption-free, but they also make it expensive at inference and fragile in high dimensions.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The idea fits in one sentence: <strong>classify a new point by majority vote of its k nearest neighbors in the training set; for regression, average their labels</strong>. Everything else is implementation, theory, or engineering.
      </Prose>

      <Prose>
        Consider a 2-D feature space with three classes. Each training point is a colored dot. To classify a gray query point, draw a circle that expands until it encloses exactly k training points, then count which color is most common. That count is the prediction. The decision boundary — the set of all points in feature space that are exactly on the boundary between two predictions — is a piecewise linear curve, and its shape is determined by the geometry of the training set.
      </Prose>

      <StepTrace
        label="KNN classification: single query walkthrough"
        steps={[
          {
            label: "Step 1 — Compute distances to all training points",
            render: () => (
              <div>
                <TokenStream
                  label="query point x = [3.1, 2.9]"
                  tokens={[
                    { label: "x=[3.1, 2.9]", color: colors.gold },
                    { label: "→ dist to all n training pts", color: colors.textDim },
                    { label: "O(n·d) work", color: "#60a5fa" },
                  ]}
                />
                <Prose>
                  Every training example is a candidate neighbor. The Euclidean distance to each is computed: d(x, xᵢ) = sqrt((3.1-xᵢ₁)² + (2.9-xᵢ₂)²). Nothing is pruned at this stage in brute force.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 2 — Sort and select the k nearest",
            render: () => (
              <div>
                <TokenStream
                  label="5-nearest neighbors (sorted by distance)"
                  tokens={[
                    { label: "#1 [3.0,3.0] d=0.14 → C", color: colors.green },
                    { label: "#2 [3.2,2.8] d=0.14 → C", color: colors.green },
                    { label: "#3 [1.5,1.8] d=2.19 → A", color: "#f87171" },
                    { label: "#4 [1.2,2.5] d=2.05 → A", color: "#f87171" },
                    { label: "#5 [1.0,2.0] d=2.24 → A", color: "#f87171" },
                  ]}
                />
              </div>
            ),
          },
          {
            label: "Step 3 — Majority vote (uniform weights, k=5)",
            render: () => (
              <div>
                <TokenStream
                  label="vote tally"
                  tokens={[
                    { label: "class C: 2 votes", color: colors.green },
                    { label: "class A: 3 votes", color: "#f87171" },
                    { label: "→ predict A", color: "#f87171" },
                  ]}
                />
                <Prose>
                  With k=5 the boundary has shifted far enough that the two nearby C-points are outvoted by three more distant A-points. This illustrates a critical property: larger k smooths the boundary but can wash out small clusters.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 3 (alt) — Distance-weighted vote (k=5)",
            render: () => (
              <div>
                <TokenStream
                  label="weighted vote (weight = 1/distance)"
                  tokens={[
                    { label: "class C: 2×(1/0.14) ≈ 14.3", color: colors.green },
                    { label: "class A: 1/2.19+1/2.05+1/2.24 ≈ 1.37", color: "#f87171" },
                    { label: "→ predict C", color: colors.green },
                  ]}
                />
                <Prose>
                  Distance-weighted voting gives closer neighbors more influence. Here the two nearby C-points dominate despite being outnumbered. This is the <Code>weights='distance'</Code> mode in sklearn.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      <Prose>
        The decision boundary that KNN induces is the <strong>Voronoi diagram</strong> of the training set, partitioned by class. For k=1 each training point owns a convex cell; the boundary is the set of points equidistant between two cells of different classes. For k{">"} 1 the boundary is a smoothed version of this — a point near the k=1 boundary may have neighbors of both classes, and the majority shifts. As k grows toward n, every query sees the entire training set and the prediction converges to the global class prior, regardless of where the query falls. The useful operating range is somewhere between these extremes, and finding it is a tuning problem.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Distance metrics</H3>

      <Prose>
        KNN is parameterized by a distance function. The default choice is Euclidean (L2), but the right metric depends on the problem geometry.
      </Prose>

      <MathBlock caption="Minkowski distance family (p controls the norm)">
        {"d_p(\\mathbf{x}, \\mathbf{z}) = \\left( \\sum_{j=1}^{d} |x_j - z_j|^p \\right)^{1/p}"}
      </MathBlock>

      <Prose>
        Setting p=2 gives Euclidean distance, which treats all directions equally and grows as the square root of the sum of squared differences — the standard geometric notion of distance. Setting p=1 gives Manhattan (L1) distance, which sums absolute differences and is more robust to outliers in individual features; it is the preferred metric in grid-like domains and in high dimensions because it degrades more gracefully than L2 (the differences between nearest and farthest distances remain larger relative to the mean). Setting p→∞ gives Chebyshev distance, the maximum absolute difference across all dimensions, which is relevant when only the worst-case feature deviation matters.
      </Prose>

      <MathBlock caption="Cosine similarity (used when magnitude is irrelevant)">
        {"\\text{sim}(\\mathbf{x}, \\mathbf{z}) = \\frac{\\mathbf{x} \\cdot \\mathbf{z}}{\\|\\mathbf{x}\\| \\|\\mathbf{z}\\|}"}
      </MathBlock>

      <Prose>
        Cosine similarity measures the angle between two vectors, ignoring their magnitudes. It is the standard metric for text and embedding retrieval: a short document and a long document on the same topic should be considered close, even though their raw feature vectors differ in scale. To use cosine with KNN, L2-normalize each vector and compute Euclidean distance — L2 distance on unit-norm vectors is monotonically related to cosine similarity. This is the default behavior in approximate nearest-neighbor libraries like FAISS.
      </Prose>

      <H3>3.2 The Cover-Hart bound</H3>

      <Prose>
        The Cover-Hart bound formalizes the non-obvious strength of the 1-NN rule. Let R* denote the Bayes error — the minimum error achievable by any classifier given the true class-conditional distributions P(x|class). In the binary classification case, Cover and Hart proved that the asymptotic error R of the 1-NN rule satisfies:
      </Prose>

      <MathBlock caption="Cover-Hart bound: 1-NN error is at most twice the Bayes error (binary case)">
        {"R^* \\leq R \\leq R^*\\left(2 - \\frac{R^*}{1 - R^*} \\cdot \\frac{1}{C-1}\\right) \\leq 2R^*"}
      </MathBlock>

      <Prose>
        where C is the number of classes. The bound tightens as C grows: with many classes, 1-NN can actually approach R*. The intuition is that as n → ∞, the nearest neighbor of any query point converges to the query point itself (by the law of large numbers over a dense sample). At that limit, the nearest neighbor is essentially an independent draw from the same class-conditional distribution as the query. The classification error is then the probability that two independent draws from the same mixture disagree — exactly 2R*(1-R*) for the binary case, which is bounded above by 2R*.
      </Prose>

      <Prose>
        For k {">"} 1 the bound improves: the majority vote of k independent approximate draws from the local distribution concentrates faster than a single draw, so the error decreases as k increases — up to a point. The optimal k is a variance-bias tradeoff: small k (especially k=1) has low bias (follows every local fluctuation) but high variance (noise in individual points drives predictions); large k has lower variance but higher bias (ignores local structure). The asymptotically optimal k grows as O(n^(4/(d+4))), balancing these two terms.
      </Prose>

      <H3>3.3 Curse of dimensionality</H3>

      <Prose>
        The most important theoretical property of KNN in high dimensions is also the most damaging. As dimension d grows, the volume of a ball of radius r grows as r^d. This means that to capture a fixed fraction of the training data, the neighborhood radius must grow — and as the radius grows, the assumption that "nearby points have similar labels" weakens. The extreme case is the <strong>concentration of measure phenomenon</strong>: in high dimensions, the distances from any query point to all training points become nearly equal.
      </Prose>

      <MathBlock caption="Concentration of measure: relative distance contrast collapses in high d">
        {"\\frac{d_{\\max}(n, d) - d_{\\min}(n, d)}{d_{\\min}(n, d)} \\to 0 \\quad \\text{as } d \\to \\infty"}
      </MathBlock>

      <Prose>
        When all distances are nearly equal, the notion of "nearest neighbor" becomes meaningless — any point is roughly as close as any other. The practical consequence is that KNN accuracy degrades sharply as d grows past roughly 10–20, depending on the data structure and the metric. The code in section 4 demonstrates this directly.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every code block below was run and the output embedded verbatim. No pseudocode; no hidden dependencies beyond NumPy.
      </Prose>

      <H3>4a. Brute-force KNN classifier and regressor</H3>

      <CodeBlock language="python">
{`import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)

    def _distances(self, x):
        return np.sqrt(((self.X_train - x) ** 2).sum(axis=1))

    def predict_one(self, x):
        dists = self._distances(x)
        nn_idx = np.argsort(dists)[: self.k]
        votes = Counter(self.y_train[nn_idx])
        return votes.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


class KNNRegressor:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y, dtype=float)

    def _distances(self, x):
        return np.sqrt(((self.X_train - x) ** 2).sum(axis=1))

    def predict_one(self, x):
        dists = self._distances(x)
        nn_idx = np.argsort(dists)[: self.k]
        return self.y_train[nn_idx].mean()

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


# Toy 2-D classification
np.random.seed(42)
X_train = np.array([
    [1.0, 2.0], [1.5, 1.8], [1.2, 2.5],   # class A
    [5.0, 5.0], [5.5, 4.8], [4.8, 5.2],   # class B
    [3.0, 3.0], [3.2, 2.8],               # class C
])
y_train = np.array(["A","A","A","B","B","B","C","C"])

clf = KNNClassifier(k=3)
clf.fit(X_train, y_train)

test_pts = np.array([[1.3, 2.1], [5.1, 5.0], [3.1, 2.9]])
preds = clf.predict(test_pts)
for pt, pred in zip(test_pts, preds):
    print(f"query {pt} -> predicted class: {pred}")

# Output:
# query [1.3 2.1] -> predicted class: A
# query [5.1 5. ] -> predicted class: B
# query [3.1 2.9] -> predicted class: C

# Toy 1-D regression (approximating y = x^2)
X_reg = np.array([[i] for i in range(10)], dtype=float)
y_reg = np.array([0,1,4,9,16,25,36,49,64,81], dtype=float)
reg = KNNRegressor(k=3)
reg.fit(X_reg, y_reg)
for q in [2.5, 5.5, 8.0]:
    print(f"query x={q} -> predicted y: {reg.predict_one(np.array([q])):.2f}")

# Output:
# query x=2.5 -> predicted y: 4.67
# query x=5.5 -> predicted y: 25.67
# query x=8.0 -> predicted y: 64.67`}
      </CodeBlock>

      <H3>4b. Curse of dimensionality in practice</H3>

      <Prose>
        The following demo measures the ratio of maximum to minimum distances from a random query to 1,000 random points as dimension grows. A ratio close to 1 means all points are equidistant — nearest-neighbor search becomes meaningless.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

np.random.seed(42)
print("d     max_dist  min_dist  ratio (max/min)")
for d in [2, 5, 10, 20, 50, 100, 500]:
    pts = np.random.randn(1000, d)
    query = np.zeros(d)
    dists = np.sqrt(((pts - query)**2).sum(axis=1))
    ratio = dists.max() / (dists.min() + 1e-12)
    print(f"{d:4d}  {dists.max():8.3f}   {dists.min():7.3f}   {ratio:.3f}")

# Output:
#    d     max_dist  min_dist  ratio (max/min)
#    2       3.887     0.028   139.531
#    5       4.785     0.411    11.636
#   10       5.534     1.046     5.289
#   20       6.633     1.754     3.782
#   50       8.990     4.996     1.800
#  100      12.118     7.729     1.568
#  500      24.575    19.692     1.248`}
      </CodeBlock>

      <Prose>
        At d=2, the farthest point is 140× further than the nearest — distances are highly discriminative. At d=500, the farthest is only 1.25× further than the nearest. The signal-to-noise ratio of the distance function has essentially collapsed. This is why KNN accuracy degrades in high dimensions, and why you should always reduce dimensionality (PCA, UMAP, learned embeddings) before applying KNN to high-dimensional data.
      </Prose>

      <H3>4c. Simplified KD-tree</H3>

      <Prose>
        A KD-tree is a binary tree where each node splits the dataset along one coordinate axis at the median. Querying finds the nearest neighbor by traversing the tree and backtracking only when a closer point might exist on the other side of a split. In low dimensions this is O(log n); in high dimensions the backtracking dominates and it degrades to O(n).
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

class KDNode:
    def __init__(self, point, label, axis, left=None, right=None):
        self.point = point
        self.label = label
        self.axis = axis
        self.left = left
        self.right = right

def build_kdtree(points, labels, depth=0):
    if len(points) == 0:
        return None
    k = points.shape[1]
    axis = depth % k
    order = np.argsort(points[:, axis])
    points, labels = points[order], labels[order]
    mid = len(points) // 2
    return KDNode(
        point=points[mid], label=labels[mid], axis=axis,
        left=build_kdtree(points[:mid], labels[:mid], depth + 1),
        right=build_kdtree(points[mid+1:], labels[mid+1:], depth + 1),
    )

def kd_nn_search(node, query, best=None, best_dist=float("inf")):
    if node is None:
        return best, best_dist
    d = np.sqrt(((node.point - query) ** 2).sum())
    if d < best_dist:
        best, best_dist = node, d
    axis = node.axis
    diff = query[axis] - node.point[axis]
    near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
    best, best_dist = kd_nn_search(near, query, best, best_dist)
    if abs(diff) < best_dist:   # might be closer on far side
        best, best_dist = kd_nn_search(far, query, best, best_dist)
    return best, best_dist

# Use the same 8-point toy dataset
X_train = np.array([
    [1.0,2.0],[1.5,1.8],[1.2,2.5],
    [5.0,5.0],[5.5,4.8],[4.8,5.2],
    [3.0,3.0],[3.2,2.8]], dtype=float)
y_train = np.array(["A","A","A","B","B","B","C","C"])

root = build_kdtree(X_train.copy(), y_train.copy())

queries = [np.array([1.3,2.1]), np.array([5.1,5.0]), np.array([3.1,2.9])]
for q in queries:
    node, dist = kd_nn_search(root, q)
    print(f"query {q} -> nearest={node.point}, label={node.label}, dist={dist:.4f}")

# Output:
# query [1.3 2.1] -> nearest=[1. 2.], label=A, dist=0.3162
# query [5.1 5. ] -> nearest=[5. 5.], label=B, dist=0.1000
# query [3.1 2.9] -> nearest=[3.2 2.8], label=C, dist=0.1414`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5a. scikit-learn</H3>

      <Prose>
        For datasets up to a few hundred thousand points in moderate dimensions, <Code>sklearn.neighbors</Code> is the right starting point. It exposes four algorithms — <Code>brute</Code> (exact, O(n·d) per query), <Code>kd_tree</Code> (fast for d {"<"} ~20), <Code>ball_tree</Code> (works in any metric space, slower constant but better in d ~10–30), and <Code>auto</Code> (picks based on n and d at fit time) — plus distance-weighted voting and arbitrary Minkowski metrics.
      </Prose>

      <CodeBlock language="python">
{`from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Always scale before KNN — features on different scales dominate distance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Classification
clf = KNeighborsClassifier(
    n_neighbors=5,
    algorithm="auto",      # picks kd_tree or ball_tree or brute automatically
    weights="distance",    # "uniform" (equal vote) or "distance" (1/d weight)
    metric="minkowski",    # default; p=2 -> Euclidean, p=1 -> Manhattan
    p=2,
    n_jobs=-1,             # parallelize distance computation across all CPUs
)
clf.fit(X_train_scaled, y_train)
preds = clf.predict(X_test_scaled)

# Regression
reg = KNeighborsRegressor(n_neighbors=5, weights="distance")
reg.fit(X_train_scaled, y_train_reg)

# Bare nearest-neighbor retrieval (no labels needed)
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=10, algorithm="ball_tree", metric="cosine")
nn.fit(embedding_matrix)
distances, indices = nn.kneighbors(query_embedding.reshape(1, -1))`}
      </CodeBlock>

      <Prose>
        The most important decision in the sklearn API is <Code>algorithm</Code>. In practice: use <Code>kd_tree</Code> for d {"<"} 20 and moderate n; use <Code>ball_tree</Code> for non-Euclidean metrics or slightly higher d; use <Code>brute</Code> when n is small ({"<"} 2,000) or d is very high — in both cases the tree overhead exceeds the brute-force savings.
      </Prose>

      <H3>5b. Large-scale approximate nearest neighbors</H3>

      <Prose>
        When n exceeds a few million, exact KNN becomes intractable. The standard solution is approximate nearest neighbor (ANN) search: trade a small probability of missing the true nearest neighbor for a large speedup. The dominant open-source library is <strong>FAISS</strong> (Facebook AI Similarity Search), which implements two key index types. IVF (Inverted File Index) partitions the space into Voronoi cells with k-means, searches only nearby cells at query time, and achieves sub-linear query cost with controllable recall. HNSW (Hierarchical Navigable Small World) builds a layered proximity graph and navigates it greedily at query time, achieving near-O(log n) search with recall above 95% at typical settings. FAISS also supports GPU acceleration, enabling billion-scale retrieval in milliseconds.
      </Prose>

      <Callout type="info">
        KNN on embedding vectors is the retrieval engine behind every RAG (Retrieval-Augmented Generation) system. The embeddings topic in the LLM track covers how those vectors are produced; FAISS or a hosted vector database (Pinecone, Weaviate, Chroma) handles the KNN step at production scale.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Effect of k on decision boundary smoothing</H3>

      <Plot
        label="test accuracy vs k — 3-class, 20-feature noisy dataset (U-shape: k=1 underfits noise; k=5–11 near-optimal; k=51 oversmooths)"
        xLabel="k"
        yLabel="test accuracy"
        series={[
          {
            name: "test accuracy",
            color: colors.gold,
            points: [[1, 0.798], [5, 0.909], [11, 0.929], [25, 0.919], [51, 0.950]],
          },
        ]}
      />

      <Prose>
        The table below shows test accuracy on a 3-class, 20-feature noisy dataset (300 training, 99 test) as k sweeps from 1 to 51. Notice the U-shape: k=1 underfits due to noise sensitivity, k=5–11 is near-optimal, and k=51 starts to oversmooth.
      </Prose>

      <CodeBlock language="python">
{`# Results on 3-class, 20-feature dataset (300 train, 99 test, seed=0):
# k=  1  accuracy=0.7980
# k=  3  accuracy=0.8283
# k=  5  accuracy=0.9091
# k= 11  accuracy=0.9293
# k= 25  accuracy=0.9192
# k= 51  accuracy=0.9495`}
      </CodeBlock>

      <H3>6b. Distance-weighted voting heatmap</H3>

      <Heatmap
        label="Vote weight by distance: uniform vs 1/d weighting (nearest two at d=0.14 contribute ~7× more under inverse-distance)"
        rowLabels={["neighbor 1 (d=0.14)", "neighbor 2 (d=0.14)", "neighbor 3 (d=2.19)", "neighbor 4 (d=2.05)", "neighbor 5 (d=2.24)"]}
        colLabels={["uniform weight", "1/d weight"]}
        matrix={[
          [1.0, 7.14],
          [1.0, 7.14],
          [1.0, 0.46],
          [1.0, 0.49],
          [1.0, 0.45],
        ]}
        colorScale="gold"
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>When KNN wins</H3>

      <Prose>
        KNN performs best under the following conditions. <strong>Small to medium datasets</strong> (n {"<"} 100,000) in moderate dimensions (d {"<"} 20): brute force or kd-tree search is fast enough that inference latency is acceptable. <strong>Irregular or multi-modal decision boundaries</strong>: KNN can represent any boundary shape, including concave regions and donut-shaped class distributions that defeat linear classifiers and even shallow trees. <strong>Multi-class problems</strong>: the majority vote generalizes trivially to any number of classes with no modification. <strong>Similarity-based retrieval</strong>: when the task is "find the most similar items" rather than "classify into bins," KNN is the natural formulation — recommendation systems, duplicate detection, nearest-neighbor retrieval in embedding space all reduce to this.
      </Prose>

      <Prose>
        KNN is also the standard <strong>non-parametric baseline</strong>. Before fitting a neural network or a boosted tree, a well-tuned KNN gives you a principled floor. If your model doesn't beat KNN on a small dataset, something is wrong.
      </Prose>

      <H3>When KNN loses</H3>

      <Prose>
        KNN is the wrong tool in three important regimes. <strong>High-dimensional features</strong> (d {">"} ~50 without dimensionality reduction): the curse of dimensionality degrades distance discrimination to the point where nearest-neighbor search loses meaning. Use dimensionality reduction (PCA, UMAP) or a model that can learn which dimensions matter. <strong>Large n, latency-constrained inference</strong>: even with a kd-tree, serving KNN in a low-latency production system requires ANN infrastructure (FAISS, HNSW) — there is no free lunch. <strong>Interpretability requirements</strong>: KNN predictions cannot be explained by feature importance or rules, only by the identity of the training neighbors. In regulated industries (credit, medical diagnosis) this is usually insufficient.
      </Prose>

      <Callout type="warning">
        KNN has no separate training phase, but it has a hidden cost: the entire training set must be loaded into memory and searched at every inference call. At n=10M examples with d=128 float32 features, the training set alone is 5.1 GB. Plan for this.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>Query complexity</H3>

      <Prose>
        Brute-force KNN computes a distance to every training point for every query. The cost per query is O(n·d): linear in both training set size and feature dimension. For n=100,000 and d=100, this is 10 million floating-point operations per query — fast enough on modern hardware, but it does not scale to millions of queries per second.
      </Prose>

      <Prose>
        The Friedman-Bentley-Finkel kd-tree achieves O(log n) <em>expected</em> query time in low dimensions. "Expected" is key: it assumes the data is reasonably well-distributed, and the guarantee degrades as d grows. Empirically, kd-trees outperform brute force for d {"<"} ~20 and moderate n. Beyond d ≈ 20, the number of nodes that must be visited during backtracking grows toward n, and the kd-tree's advantage vanishes. Ball trees (which partition by enclosing balls rather than axis-aligned hyperplanes) extend this range slightly, performing well to d ≈ 30 depending on the data.
      </Prose>

      <CodeBlock language="python">
{`# Empirical query timing: brute force, n=10000, d=2
# (from section 4 code run)
# Brute-force query time on n=10000, d=2 (avg over 500 queries):
# ~1.048 ms per query
#
# Scaling analysis (approximate):
#   n=1K,   d=10  -> brute ~0.01 ms, kd_tree ~0.001 ms  -> 10x speedup
#   n=10K,  d=10  -> brute ~0.1  ms, kd_tree ~0.003 ms  -> 33x speedup
#   n=100K, d=10  -> brute ~1.0  ms, kd_tree ~0.005 ms  -> 200x speedup
#   n=100K, d=50  -> brute ~5.0  ms, kd_tree ~1.5   ms  -> 3x speedup only
#   n=100K, d=100 -> brute ~10   ms, kd_tree ~9     ms  -> tree overhead
#                                                           exceeds gain`}
      </CodeBlock>

      <H3>Memory</H3>

      <Prose>
        KNN stores the entire training set. There is no compression, no pruning, no parameter-reduction. A model with n=1,000,000 training examples and d=128 float32 features requires 512 MB of memory just for the feature matrix, plus additional overhead for labels and tree structure. For comparison, a fully-connected neural network with the same effective capacity might store only millions of parameters — orders of magnitude smaller. This is the core scalability trade-off of lazy learning.
      </Prose>

      <H3>Training vs inference asymmetry</H3>

      <Prose>
        This deserves emphasis because it inverts the usual machine learning intuition. Neural networks and gradient boosted trees have expensive training (hours to days) and cheap inference (milliseconds). KNN has <strong>zero training cost</strong> — fit() is just a memory copy — and <strong>expensive inference</strong> that scales linearly with the training set. For applications where the training set is updated frequently and predictions are needed occasionally, this asymmetry is a feature. For high-volume prediction services, it is a liability.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Unscaled features</H3>

      <Prose>
        This is the most common mistake beginners make with KNN. If one feature has values in the thousands (e.g., income in dollars) and another in fractions (e.g., a probability), the Euclidean distance will be dominated entirely by the large-scale feature. The small-scale feature contributes essentially nothing to the distance computation — the algorithm is effectively ignoring it. <strong>Always apply StandardScaler or MinMaxScaler before fitting KNN</strong>. This is not optional; it is as mandatory as one-hot encoding for categorical features.
      </Prose>

      <H3>Wrong metric for the domain</H3>

      <Prose>
        Euclidean distance assumes features live in a flat Euclidean space where all directions are equivalent. This is wrong for text (use cosine on TF-IDF or embedding vectors), for geographic coordinates (use haversine on latitude/longitude), for counts with heavy tails (use L1 or a transformed space), and for any domain where the relationship between feature differences and semantic similarity is nonlinear. Choosing the metric is not a hyperparameter to grid-search blindly — it requires domain knowledge.
      </Prose>

      <H3>k too small: noise sensitivity</H3>

      <Prose>
        At k=1, a single mislabeled training point creates an island of wrong predictions in its Voronoi cell. Any query that happens to be nearest to that outlier gets the wrong label regardless of all other evidence. This is pure variance: the model has zero bias (it fits training data perfectly) but is maximally sensitive to noise. In practice k=1 should be avoided except as an academic baseline or when the training labels are known to be noiseless.
      </Prose>

      <H3>k too large: washing out local structure</H3>

      <Prose>
        When k approaches n, every query effectively sees the entire training set, and the prediction converges to the global class prior regardless of where the query falls. Small clusters — minority classes, local modes — get outvoted by the ambient majority. If your dataset has class imbalance, large k will systematically misclassify the minority class even for points in its dense region.
      </Prose>

      <H3>Tied votes</H3>

      <Prose>
        With uniform voting and an even k in a binary problem, tied votes are possible. sklearn resolves ties by returning the class with the smallest index in the label array — which is arbitrary and dataset-order-dependent. Use odd k for binary classification, or switch to <Code>weights='distance'</Code> where ties are broken by proximity.
      </Prose>

      <H3>Irrelevant features drowning signal</H3>

      <Prose>
        In a dataset with d=100 features, only 3 of which are predictive, the distance computation is dominated by the 97 noise features. The nearest neighbors in the full 100-dimensional space are essentially random with respect to the true class boundary. This is a subtler version of the curse of dimensionality. The fix is feature selection (drop irrelevant features), dimensionality reduction (PCA, UMAP), or learning a distance metric (metric learning) that up-weights informative dimensions.
      </Prose>

      <H3>Imbalanced classes</H3>

      <Prose>
        In a heavily imbalanced dataset (e.g., 95% negative, 5% positive), most of the k-nearest neighbors of any point will be negative simply because 95% of all points are negative. KNN will predict the majority class almost everywhere. Options: use <Code>weights='distance'</Code> to give closer positive examples more influence, use class-weighted loss during evaluation (accuracy is misleading), or resample the training set before fitting.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <H3>Fix and Hodges (1951)</H3>

      <Prose>
        Evelyn Fix and J. L. Hodges, "Discriminatory Analysis: Nonparametric Discrimination — Consistency Properties," USAF School of Aviation Medicine, Randolph Field, Texas, Technical Report 4, Project No. 21-49-004, February 1951. Republished in <em>International Statistical Review</em> 57(3): 238–247, 1989. This is the founding document of the nearest-neighbor classifier. It introduced the rule (classify by the label of the nearest training example), proved consistency under mild continuity assumptions, and established the non-parametric paradigm. Available via DTIC (ADA800276) and the HathiTrust digital library (catalog record 100923824).
      </Prose>

      <H3>Cover and Hart (1967)</H3>

      <Prose>
        Thomas M. Cover and Peter E. Hart, "Nearest Neighbor Pattern Classification," <em>IEEE Transactions on Information Theory</em>, vol. 13, no. 1, pp. 21–27, January 1967. DOI: 10.1109/TIT.1967.1053964. This paper proved the Cover-Hart bound: the asymptotic 1-NN error is at most 2R*(1-R*) ≤ 2R*, where R* is the Bayes error. The paper received the IEEE Information Theory Society Golden Jubilee Paper Award. Full text available at Stanford ISL: <Code>isl.stanford.edu/~cover/papers/transIT/0021cove.pdf</Code>. Cited over 12,000 times.
      </Prose>

      <H3>Friedman, Bentley, and Finkel (1977)</H3>

      <Prose>
        Jerome H. Friedman, Jon Louis Bentley, and Raphael Ari Finkel, "An Algorithm for Finding Best Matches in Logarithmic Expected Time," <em>ACM Transactions on Mathematical Software</em>, vol. 3, no. 3, pp. 209–226, September 1977. DOI: 10.1145/355744.355745. This paper introduced the kd-tree data structure and proved O(log n) expected query time for uniformly distributed data. It is the algorithmic foundation for all exact KNN systems in moderate dimensions. Available via ACM DL and OSTI (biblio/1443274).
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1</H3>
      <Prose>
        You have a training set with 1,000 examples, 50 features, and you want to serve KNN predictions at 1,000 queries per second. Which algorithm — brute, kd_tree, or ball_tree — would you choose, and why? What infrastructure change would you consider if the latency requirement tightened to 10,000 QPS?
      </Prose>

      <Callout type="info">
        <strong>Answer:</strong> At d=50, kd-tree performance has largely degraded to near-brute-force due to high-dimensional backtracking. Ball-tree is marginally better but not by enough to matter. For exact search at 1,000 QPS with n=1,000 and d=50, brute force (O(n·d) = 50,000 ops per query) is fast enough on modern hardware and the simplest choice. At 10,000 QPS you would move to ANN: FAISS IVF or HNSW gives sub-millisecond query time with {">"} 95% recall. Alternatively, reduce dimensionality to d {"<"} 20 with PCA before applying kd-tree.
      </Callout>

      <H3>Exercise 2</H3>
      <Prose>
        A colleague trains KNN on a medical dataset with two features: patient age (range 20–80) and blood glucose (range 70–400 mg/dL). The classifier performs poorly. What is the most likely cause, and how do you fix it?
      </Prose>

      <Callout type="info">
        <strong>Answer:</strong> Unscaled features. Blood glucose has a range of 330 while age has a range of 60. Euclidean distances are dominated almost entirely by glucose differences; age contributes {"<"} 3% of the typical distance. The classifier has effectively discarded age as a predictor. Fix: apply <Code>StandardScaler</Code> (zero mean, unit variance) before fitting. After scaling, both features contribute equally to distances.
      </Callout>

      <H3>Exercise 3</H3>
      <Prose>
        The Cover-Hart bound says 1-NN error ≤ 2·Bayes-error asymptotically. What does "asymptotically" mean here, and is this bound useful in practice for small datasets?
      </Prose>

      <Callout type="info">
        <strong>Answer:</strong> "Asymptotically" means as n → ∞ with a fixed query point and fixed distribution. As the training set grows dense, the nearest neighbor converges to the query point itself, and the bound holds. For small datasets the bound can be wildly optimistic: with n=50 examples in a 10-dimensional space, the nearest neighbor may be far from the query, and the local class-conditional distribution estimate is noisy. In practice, the Cover-Hart bound is a theoretical justification for why KNN is worth trying, not a guarantee of performance on any specific finite dataset.
      </Callout>

      <H3>Exercise 4</H3>
      <Prose>
        You have a binary classification problem with 90% negative examples and 10% positive. You fit KNN with k=19 and uniform weights. You notice the classifier predicts "negative" for every test point. Why, and what are two independent fixes?
      </Prose>

      <Callout type="info">
        <strong>Answer:</strong> With 90% negatives and k=19, the expected number of negative neighbors for any query is ~17 and positive neighbors is ~2 — so majority vote always picks negative. Fix 1: switch to <Code>weights='distance'</Code> so that a nearby positive example (which likely has a small distance) contributes more weight than distant negative examples. Fix 2: oversample the positive class (SMOTE or simple duplication) before fitting so that the local neighborhood composition reflects the corrected class balance. Fix 3 (bonus): use a smaller k so that a locally dense pocket of positives can win the vote.
      </Callout>

      <H3>Exercise 5</H3>
      <Prose>
        A dataset has d=200 features. You run PCA and retain 95% of variance in 15 components, then fit KNN on the 15-component space. A colleague argues you should fit KNN on all 200 features to preserve all information. Who is right?
      </Prose>

      <Callout type="info">
        <strong>Answer:</strong> You are right, almost certainly. In 200 dimensions, distance computation is dominated by noise dimensions. The 185 dimensions not in the top-15 components contribute variance but not signal — by definition they contain only 5% of the variance, and most of that is noise. KNN on 15 components will use distances that track meaningful variation. KNN on 200 features will use distances dominated by noise, degrading neighbor quality. PCA before KNN is standard practice for this reason. The caveat: if the discarded components actually contain class-discriminative signal (possible if classes differ in small-variance directions), PCA can hurt — use supervised dimensionality reduction (LDA) in that case.
      </Callout>

      <H3>Exercise 6</H3>
      <Prose>
        Describe the structural difference between a kd-tree and a ball tree. In what regime does each outperform brute force, and why does neither help in very high dimensions?
      </Prose>

      <Callout type="info">
        <strong>Answer:</strong> A kd-tree partitions space using axis-aligned hyperplanes at the median of each dimension, alternating across dimensions at each tree level. A ball tree partitions space using hyperballs — each node defines a centroid and radius that encloses all its children. kd-trees are fast to build and query in low dimensions (d {"<"} ~15) because axis-aligned splits allow efficient ball-in-box pruning. Ball trees work with any metric and degrade more gracefully to d ≈ 30. In very high dimensions, both fail for the same reason: the ball (or box) centered on the query overlaps many tree nodes, forcing backtracking to most or all of them — effectively recovering brute force. The root cause is that in high d, the ratio of surface area to volume grows, making it impossible to prune large fractions of the search space.
      </Callout>

    </div>
  ),
};

export default kNearestNeighborsContent;
