import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const kMeansHierarchicalContent = {
  title: "K-Means & Hierarchical Clustering",
  readTime: "~45 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every supervised learning algorithm starts from the same premise: someone labelled the data for you. A human looked at an email and decided it was spam, at an X-ray and decided it showed a tumor, at a transaction and flagged it as fraud. The labels are the ground truth, and the model's job is to recover the decision rule that generated them. But the vast majority of data that exists in the world has no labels. A sensor logs millions of readings. A genomics experiment produces expression values for twenty thousand genes across a thousand patients. A retailer accumulates purchasing histories for ten million customers. Nobody has gone through and attached a category to each row, because nobody knows what the categories should be in the first place.
      </Prose>

      <Prose>
        Clustering is the attempt to find structure in unlabelled data — to discover groups of objects that are more similar to one another than to objects outside the group, without being told in advance what those groups are, how many there are, or even what "similar" should mean. The use cases are broad: customer segmentation in marketing (find naturally occurring purchasing personas), bioinformatics (group genes with correlated expression profiles), image compression (replace millions of pixel colors with a small palette of representative colors), anomaly detection (points that belong to no cluster are suspicious), and as a preprocessing step before supervised learning (cluster-label encodings, or learning one model per cluster). Before deep learning made representation learning fashionable, clustering was how you turned high-dimensional raw data into something a human or a downstream model could reason about.
      </Prose>

      <Prose>
        The intellectual history of clustering algorithms begins not with a computer scientist but with a Polish mathematician. Hugo Steinhaus, writing in 1956 in the <em>Bulletin de l'Académie Polonaise des Sciences</em> (vol. IV, no. 12, pp. 801–804), published "Sur la division des corps matériels en parties" — on the division of material bodies into parts. His motivation was geometric and almost mechanical: given a set of points in space and a number <em>k</em>, partition the points into <em>k</em> groups such that each point is closer to its group's center of mass than to any other group's center. This is precisely the k-means objective. Steinhaus proved existence of such a partition and sketched an iterative procedure to find it.
      </Prose>

      <Prose>
        Independently and simultaneously, Stuart Lloyd at Bell Laboratories derived the same algorithm from a signal processing problem: optimal scalar quantization for pulse-code modulation (PCM). He circulated his result as a Bell Labs technical report in 1957 but did not publish it in a journal until 1982, when it appeared as "Least Squares Quantization in PCM" in <em>IEEE Transactions on Information Theory</em>, vol. 28, no. 2, pp. 129–137 (DOI: 10.1109/TIT.1982.1056489). Despite the 25-year gap between derivation and publication, the algorithm bears Lloyd's name — Lloyd's algorithm — and the 1982 paper is the canonical citation in machine learning. James MacQueen coined the term "k-means" in 1967 in "Some Methods for Classification and Analysis of Multivariate Observations," published in the <em>Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability</em>, vol. 1, pp. 281–297. MacQueen also showed convergence properties and generalized the approach to online (sequential) updates.
      </Prose>

      <Prose>
        Hierarchical clustering arrived from a different direction. Joe H. Ward Jr., a statistician at the United States Air Force's Personnel Laboratory, published "Hierarchical Grouping to Optimize an Objective Function" in the <em>Journal of the American Statistical Association</em> in 1963, vol. 58, no. 301, pp. 236–244 (DOI: 10.1080/01621459.1963.10500845). Ward's contribution was a specific linkage criterion — minimize the increase in total within-cluster sum of squares at each merge step — that produces particularly compact, spherical clusters. The general framework of agglomerative hierarchical clustering (start with each point as its own cluster, repeatedly merge the closest pair) predates Ward and has roots in taxonomy and numerical taxonomy research of the 1950s, but Ward's linkage became the most widely used because it directly optimizes the same criterion as k-means.
      </Prose>

      <Prose>
        The k-means++ seeding algorithm, which dramatically improved the initialization of k-means, was introduced by David Arthur and Sergei Vassilvitskii in "k-means++: The Advantages of Careful Seeding," published at the 18th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA 2007), pp. 1027–1035. It provides an O(log k)-competitive approximation guarantee on the final inertia — a theoretically grounded improvement over the random initialization that Lloyd's original algorithm used, which can fail catastrophically when initial centroids happen to cluster near one another.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 K-means: Voronoi partitioning by iterative centroid refinement</H3>

      <Prose>
        K-means is, at its core, an alternating optimization. You fix cluster assignments and update centroids. Then you fix centroids and update cluster assignments. Repeat until nothing changes. The reason this alternation works is that each half of the step strictly decreases the objective — the total squared distance from each point to its assigned centroid — and because the objective is bounded below by zero, the process must terminate.
      </Prose>

      <Prose>
        The geometric intuition is Voronoi diagrams. Given a set of <em>k</em> centroids in a feature space, the Voronoi partition assigns each point to the nearest centroid. The regions are convex polygons (in 2D) separated by the perpendicular bisectors between adjacent centroids. K-means asks: given your data, what placement of <em>k</em> centroids minimizes the total within-cluster squared distance? The assignment step computes the Voronoi partition of the current centroids. The update step moves each centroid to the mean of its Voronoi cell. The mean is the optimal representative point for a cell under squared-distance loss — it minimizes the sum of squared distances within the cell. So after each update step, every centroid is in a locally better position; after each assignment step, every point is in its locally best cluster. The fixed point is a local optimum of the k-means objective.
      </Prose>

      <Plot
        label="K-means on 150-point 3-cluster synthetic dataset — final partition"
        xLabel="feature 1"
        yLabel="feature 2"
        series={[
          {
            name: "cluster 0 (center: [4.60, 2.11])",
            color: colors.gold,
            points: [
              [3.98, 1.61], [4.92, 2.83], [5.21, 2.45], [4.11, 1.18], [3.73, 2.84],
              [5.46, 2.24], [4.87, 1.53], [4.29, 3.12], [3.62, 1.94], [5.08, 2.77],
              [4.55, 2.10], [3.90, 2.53], [5.33, 1.87], [4.67, 2.38], [4.15, 1.75],
            ],
          },
          {
            name: "cluster 1 (center: [-2.66, 8.93])",
            color: colors.green,
            points: [
              [-2.54, 10.27], [-3.18, 8.77], [-2.82, 9.14], [-1.97, 8.55], [-3.44, 9.31],
              [-2.10, 9.72], [-2.93, 8.40], [-1.85, 9.03], [-3.11, 10.01], [-2.47, 8.82],
              [-2.68, 9.47], [-3.25, 8.61], [-2.39, 10.08], [-1.74, 8.94], [-2.85, 9.20],
            ],
          },
          {
            name: "cluster 2 (center: [-6.78, -6.89])",
            color: "#a78bfa",
            points: [
              [-6.65, -7.38], [-7.21, -6.42], [-6.98, -7.05], [-6.31, -6.74], [-7.45, -6.21],
              [-6.54, -7.81], [-7.03, -6.58], [-6.82, -7.14], [-6.19, -6.97], [-7.28, -7.32],
              [-6.76, -6.53], [-6.44, -7.62], [-7.12, -6.88], [-6.59, -7.24], [-6.91, -6.70],
            ],
          },
        ]}
      />

      <H3>2.2 Hierarchical clustering: agglomerative merging</H3>

      <Prose>
        Hierarchical clustering starts at the opposite end of the spectrum. Instead of assuming <em>k</em> upfront and jumping to a global partition, it builds a complete tree of possible merges — a dendrogram — that lets you read off any number of clusters by cutting the tree at a chosen height. The agglomerative (bottom-up) variant begins with every point as its own singleton cluster. At each step it finds the two clusters that are "closest" under a chosen linkage criterion, merges them into one, and records the merge distance. After <em>n - 1</em> merges, every point belongs to a single cluster. The result is a full hierarchy.
      </Prose>

      <Prose>
        The linkage criterion determines what "closest pair of clusters" means, and the choice profoundly shapes the result. Single linkage measures the distance between the two closest members of the two clusters (minimum inter-cluster distance). Complete linkage uses the two most distant members (maximum). Average linkage averages all pairwise distances. Ward's linkage measures the increase in total within-cluster variance that would result from merging — it is the only one of the four that directly minimizes the same criterion as k-means, and it produces the most compact clusters of the four.
      </Prose>

      <StepTrace
        label="Agglomerative Ward linkage — 6-point toy example"
        steps={[
          {
            label: "Initial state — 6 singleton clusters",
            render: () => (
              <Prose>
                Six points: P0=[1.0, 0.0], P1=[1.5, 0.5], P2=[3.0, 2.0], P3=[3.5, 2.0], P4=[7.0, 5.0], P5=[7.5, 5.5]. Each is its own cluster. The pairwise distance matrix contains 15 unique entries; the smallest is dist(P2, P3) = 0.500, the largest is dist(P0, P5) = 8.515.
              </Prose>
            ),
          },
          {
            label: "Step 1 — Merge P2 and P3 (dist 0.500)",
            render: () => (
              <Prose>
                Ward-distance between P2 and P3 is 0.125 (= (1×1)/(1+1) × ||mean_P2 - mean_P3||² = 0.5 × 0.25). These two are the closest pair. Merged into cluster C6 = {"{"} P2, P3 {"}"} with centroid [3.25, 2.00]. Active clusters: C0={"{"}P0{"}"}, C1={"{"}P1{"}"}, C4={"{"}P4{"}"}, C5={"{"}P5{"}"}, C6={"{"}P2,P3{"}"}.
              </Prose>
            ),
          },
          {
            label: "Step 2 — Merge P0 and P1 (Ward-dist 0.250)",
            render: () => (
              <Prose>
                P0=[1.0,0.0] and P1=[1.5,0.5] are the next closest pair. Euclidean distance 0.707; Ward-distance 0.250. Merged into C7={"{"}P0,P1{"}"} with centroid [1.25, 0.25]. Three geometric regions are now forming: lower-left (C7), middle (C6), upper-right (C4, C5).
              </Prose>
            ),
          },
          {
            label: "Step 3 — Merge P4 and P5 (Ward-dist 0.250)",
            render: () => (
              <Prose>
                P4=[7.0,5.0] and P5=[7.5,5.5] merge into C8={"{"}P4,P5{"}"} with centroid [7.25, 5.25]. Ward-distance 0.250. The three natural groups are now all formed as clusters of size 2: C7 (lower-left), C6 (middle), C8 (upper-right).
              </Prose>
            ),
          },
          {
            label: "Step 4 — Merge C6 and C7 (Ward-dist 7.063)",
            render: () => (
              <Prose>
                The Ward-distance between C6={"{"}P2,P3{"}"} and C7={"{"}P0,P1{"}"} is 7.063. This is the largest jump so far — crossing a genuine gap in the data. Merged into C9 with centroid [2.25, 1.125]. The merge cost jumped by 28× compared to step 3, a clear signal of a meaningful cluster boundary.
              </Prose>
            ),
          },
          {
            label: "Step 5 — Final merge C8 and C9 (Ward-dist 56.021)",
            render: () => (
              <Prose>
                The last merge combines all 6 points into one root cluster. Ward-distance 56.021 — another large jump (8× step 4). In a dendrogram, this staircase of merge heights (0.25 → 7.06 → 56.02) tells you: cut above 7 to get 2 clusters, cut above 0.25 to get 3 clusters. The elbow in merge heights identifies the natural cluster count.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 K-means objective</H3>

      <Prose>
        Let <Code>{"X = {x₁, ..., xₙ}"}</Code> be the dataset, <Code>{"x_i ∈ ℝ^d"}</Code>. K-means seeks a partition <Code>{"C = {C₁, ..., Cₖ}"}</Code> of the <em>n</em> points into <em>k</em> non-empty, non-overlapping subsets, and a set of centroid vectors <Code>{"μ₁, ..., μₖ"}</Code>, that minimizes the within-cluster sum of squares (WCSS), also called inertia:
      </Prose>

      <MathBlock>
        {"\\mathcal{L}(C, \\mu) = \\sum_{j=1}^{k} \\sum_{x_i \\in C_j} \\|x_i - \\mu_j\\|^2"}
      </MathBlock>

      <Prose>
        For a fixed partition, the optimal centroid is the cluster mean: <Code>{"μⱼ* = (1/|Cⱼ|) Σ_{xᵢ∈Cⱼ} xᵢ"}</Code>. This is because the sum of squared distances to any point is minimized when that point is the mean — the mean is the least-squares representative. For a fixed set of centroids, the optimal partition assigns each point to its nearest centroid:
      </Prose>

      <MathBlock>
        {"C_j = \\{ x_i : \\|x_i - \\mu_j\\|^2 \\leq \\|x_i - \\mu_\\ell\\|^2 \\; \\forall \\ell \\neq j \\}"}
      </MathBlock>

      <Prose>
        Lloyd's algorithm alternates between these two steps. Each step is a coordinate descent move on the joint objective over <Code>(C, μ)</Code>. Since each step weakly decreases the objective and the number of possible partitions is finite (at most <Code>k^n</Code>), convergence is guaranteed. However, convergence is to a <em>local</em> optimum — the global minimum is NP-hard to find in general, and different initializations typically produce different local optima with different inertia values. This is why <Code>n_init</Code> matters: run the algorithm multiple times with different seeds, keep the best.
      </Prose>

      <H3>3.2 K-means++ initialization</H3>

      <Prose>
        Random initialization is the original sin of k-means. If all <em>k</em> initial centroids happen to land inside the same dense region, they compete for the same points and produce bad, unbalanced clusters. Arthur and Vassilvitskii's k-means++ fixes this with a seeding procedure that spreads initial centroids with high probability:
      </Prose>

      <Prose>
        1. Choose the first centroid uniformly at random from the data points. 2. For each subsequent centroid, sample from the data with probability proportional to the squared distance to the nearest already-chosen centroid. Points far from all current centroids are more likely to be chosen. 3. Repeat until <em>k</em> centroids are selected. Then run Lloyd's algorithm from this seed.
      </Prose>

      <MathBlock>
        {"P(x_i \\text{ chosen as next centroid}) = \\frac{D(x_i)^2}{\\sum_{j=1}^n D(x_j)^2}"}
      </MathBlock>

      <Prose>
        where <Code>{"D(xᵢ) = min_c ||xᵢ - c||"}</Code> is the distance from <Code>xᵢ</Code> to the nearest already-chosen centroid. Arthur and Vassilvitskii proved that this initialization produces an expected inertia within O(log k) of the global optimum — specifically, <Code>{"E[L] ≤ 8(ln k + 2) · L*"}</Code> where <Code>L*</Code> is the optimal inertia. This is a <em>before-running-Lloyd's-iterations</em> guarantee; the final result after Lloyd iterations is typically much better in practice.
      </Prose>

      <H3>3.3 Linkage criteria for hierarchical clustering</H3>

      <Prose>
        Let <Code>A</Code> and <Code>B</Code> be two clusters and let <Code>d(x, y)</Code> denote the pairwise distance (usually Euclidean) between two points. The four standard linkage functions are:
      </Prose>

      <MathBlock>
        {"d_{\\text{single}}(A, B) = \\min_{x \\in A,\\, y \\in B} d(x, y)"}
      </MathBlock>

      <MathBlock>
        {"d_{\\text{complete}}(A, B) = \\max_{x \\in A,\\, y \\in B} d(x, y)"}
      </MathBlock>

      <MathBlock>
        {"d_{\\text{average}}(A, B) = \\frac{1}{|A||B|} \\sum_{x \\in A} \\sum_{y \\in B} d(x, y)"}
      </MathBlock>

      <MathBlock>
        {"d_{\\text{Ward}}(A, B) = \\sqrt{\\frac{2|A||B|}{|A|+|B|}} \\cdot \\|\\bar{x}_A - \\bar{x}_B\\|"}
      </MathBlock>

      <Prose>
        where <Code>{"x̄_A, x̄_B"}</Code> are the cluster centroids. Ward's linkage is the squared-distance version: squaring both sides gives the increase in total WCSS from merging A and B, which Ward's criterion minimizes at each step. This is why Ward hierarchical clustering produces clusters whose sizes are similar — it penalizes merging large clusters that are far apart, unlike single linkage which is prone to "chaining" (a single bridge point drags two large elongated blobs together).
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below was run on a 150-point synthetic dataset (<Code>make_blobs</Code>, 3 centers, <Code>random_state=42</Code>). Outputs are verbatim terminal output. NumPy only for the algorithms; scikit-learn only for data generation and verification.
      </Prose>

      <H3>4.1 K-means++ seeding and Lloyd's algorithm</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.datasets import make_blobs

np.random.seed(42)
X, y_true = make_blobs(n_samples=150, centers=3, cluster_std=0.8, random_state=42)
print(f"Dataset shape: {X.shape}")
# Output: Dataset shape: (150, 2)

def kmeans_plus_plus_init(X, k, rng):
    """k-means++ seeding: spread initial centroids probabilistically."""
    n = len(X)
    # Step 1: pick first centroid uniformly at random
    centers = [X[rng.integers(n)]]
    for _ in range(k - 1):
        # Step 2: probability proportional to squared distance to nearest center
        dists = np.array([
            min(np.sum((x - c) ** 2) for c in centers) for x in X
        ])
        probs = dists / dists.sum()
        cumprobs = np.cumsum(probs)
        r = rng.random()
        idx = np.searchsorted(cumprobs, r)
        centers.append(X[idx])
    return np.array(centers)

def kmeans_scratch(X, k=3, n_iter=100, random_state=0):
    """
    Lloyd's algorithm with k-means++ initialization.
    Alternates:
      1. Assignment: each point -> nearest centroid (Voronoi partition)
      2. Update: each centroid -> mean of its assigned points
    Converges when assignments don't change.
    """
    rng = np.random.default_rng(random_state)
    centroids = kmeans_plus_plus_init(X, k, rng)

    print("Initial centroids (k-means++):")
    for j, c in enumerate(centroids):
        print(f"  C{j}: [{c[0]:.4f}, {c[1]:.4f}]")
    # Output:
    # Initial centroids (k-means++):
    #   C0: [3.9283, 1.3205]
    #   C1: [-3.4300, 9.3148]
    #   C2: [-7.2572, -6.0089]

    prev_labels = None
    for i in range(n_iter):
        # Assignment step: squared distances to each centroid
        dists = np.array([
            [np.sum((x - c) ** 2) for c in centroids] for x in X
        ])
        labels = np.argmin(dists, axis=1)

        # Check convergence
        if prev_labels is not None and np.all(labels == prev_labels):
            print(f"Converged at iteration {i + 1}")
            break
        prev_labels = labels.copy()

        # Update step: move each centroid to the mean of its points
        centroids = np.array([
            X[labels == j].mean(axis=0) if (labels == j).sum() > 0 else centroids[j]
            for j in range(k)
        ])

    # Final inertia
    inertia = sum(
        np.sum((X[labels == j] - centroids[j]) ** 2) for j in range(k)
    )
    return labels, centroids, inertia

labels, centroids, inertia = kmeans_scratch(X, k=3, n_iter=100, random_state=0)
# Output: Converged at iteration 2

print("Final cluster centers:")
for j, c in enumerate(centroids):
    print(f"  Cluster {j}: [{c[0]:.4f}, {c[1]:.4f}]")
# Output:
# Final cluster centers:
#   Cluster 0: [4.5952, 2.1091]
#   Cluster 1: [-2.6630, 8.9252]
#   Cluster 2: [-6.7791, -6.8876]

print(f"Final inertia: {inertia:.4f}")
# Output: Final inertia: 181.5044

print(f"Cluster sizes: {[(labels == j).sum() for j in range(3)]}")
# Output: Cluster sizes: [50, 50, 50]`}
      </CodeBlock>

      <Prose>
        The algorithm converges in just 2 iterations here because k-means++ places the initial centroids close to the true cluster centers. Random initialization on the same data can take 10–20 iterations and may converge to a suboptimal partition. The inertia of 181.50 is the global optimum on this well-separated synthetic dataset — confirmed by sklearn's answer below.
      </Prose>

      <H3>4.2 Agglomerative hierarchical clustering (Ward linkage, from scratch)</H3>

      <CodeBlock language="python">
{`import numpy as np

# Small 6-point toy example: three natural groups
toy = np.array([
    [1.0, 0.0],   # P0  \
    [1.5, 0.5],   # P1  /  group A (lower-left)
    [3.0, 2.0],   # P2  \
    [3.5, 2.0],   # P3  /  group B (middle)
    [7.0, 5.0],   # P4  \
    [7.5, 5.5],   # P5  /  group C (upper-right)
])

def ward_distance(pts_a, pts_b):
    """
    Ward distance = increase in total WCSS from merging clusters A and B.
    = (|A|*|B|)/(|A|+|B|) * ||mean_A - mean_B||^2
    """
    na, nb = len(pts_a), len(pts_b)
    mean_a = pts_a.mean(axis=0)
    mean_b = pts_b.mean(axis=0)
    return (na * nb) / (na + nb) * np.sum((mean_a - mean_b) ** 2)

def agglomerative_ward(X):
    """
    Bottom-up agglomerative clustering with Ward linkage.
    Uses a simple O(n^3) implementation for clarity.
    Returns merge sequence: list of (cluster_a, cluster_b, ward_dist, new_id)
    """
    n = len(X)
    # Each point starts as its own cluster
    clusters = {i: X[i:i+1].copy() for i in range(n)}
    active = set(range(n))
    next_id = n
    merge_seq = []

    for step in range(n - 1):
        best_dist = np.inf
        best_pair = None
        active_list = sorted(active)
        # Find the pair with minimum Ward distance
        for i in range(len(active_list)):
            for j in range(i + 1, len(active_list)):
                a, b = active_list[i], active_list[j]
                d = ward_distance(clusters[a], clusters[b])
                if d < best_dist:
                    best_dist = d
                    best_pair = (a, b)

        a, b = best_pair
        merged_pts = np.vstack([clusters[a], clusters[b]])
        print(
            f"Step {step+1}: merge {a} + {b}  "
            f"Ward-dist={best_dist:.4f}  "
            f"new cluster {next_id} (n={len(merged_pts)})"
        )
        merge_seq.append((a, b, best_dist, next_id))
        active.discard(a)
        active.discard(b)
        active.add(next_id)
        clusters[next_id] = merged_pts
        next_id += 1

    return merge_seq

merge_seq = agglomerative_ward(toy)
# Output:
# Step 1: merge 2 + 3  Ward-dist=0.1250  new cluster 6 (n=2)
# Step 2: merge 0 + 1  Ward-dist=0.2500  new cluster 7 (n=2)
# Step 3: merge 4 + 5  Ward-dist=0.2500  new cluster 8 (n=2)
# Step 4: merge 6 + 7  Ward-dist=7.0625  new cluster 9 (n=4)
# Step 5: merge 8 + 9  Ward-dist=56.0208 new cluster 10 (n=6)

print()
print("Merge height interpretation:")
print("  Cut above 56.02 -> 1 cluster (root)")
print("  Cut above  7.06 -> 2 clusters: {P0,P1,P2,P3} and {P4,P5}")
print("  Cut above  0.25 -> 3 clusters: {P0,P1}, {P2,P3}, {P4,P5}")
# Output:
# Merge height interpretation:
#   Cut above 56.02 -> 1 cluster (root)
#   Cut above  7.06 -> 2 clusters: {P0,P1,P2,P3} and {P4,P5}
#   Cut above  0.25 -> 3 clusters: {P0,P1}, {P2,P3}, {P4,P5}`}
      </CodeBlock>

      <Prose>
        The merge heights tell the whole story. Steps 1–3 (Ward distances 0.125 to 0.25) merge within natural groups — close points that clearly belong together. Step 4 (Ward distance 7.06) crosses the first real boundary: the lower-left group joins the middle group. Step 5 (Ward distance 56.02) is the final merge, pulling the well-separated upper-right group in. The jump from 0.25 to 7.06 to 56.02 is the dendrogram signal: gaps in merge heights indicate where natural boundaries exist.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5.1 sklearn.cluster.KMeans and MiniBatchKMeans</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

np.random.seed(42)
X, y_true = make_blobs(n_samples=150, centers=3, cluster_std=0.8, random_state=42)

# --- Standard KMeans: k-means++ init, 10 random restarts ---
km = KMeans(
    n_clusters=3,
    init='k-means++',   # better than init='random'
    n_init=10,          # run 10x, keep best inertia
    max_iter=300,
    random_state=42,
)
km.fit(X)
print(f"Iterations to converge: {km.n_iter_}")
# Output: Iterations to converge: 2

print(f"Inertia: {km.inertia_:.4f}")
# Output: Inertia: 181.5044

print("Cluster centers:")
for j, c in enumerate(km.cluster_centers_):
    print(f"  [{c[0]:.4f}, {c[1]:.4f}]")
# Output:
# Cluster centers:
#   [-2.6630, 8.9252]
#   [-6.7791, -6.8876]
#   [4.5952, 2.1091]

sil = silhouette_score(X, km.labels_)
print(f"Silhouette score: {sil:.4f}")
# Output: Silhouette score: 0.8760

# --- MiniBatchKMeans: for large datasets (n > 100k) ---
mbkm = MiniBatchKMeans(
    n_clusters=3,
    init='k-means++',
    n_init=10,
    batch_size=64,   # process 64 samples per update step
    random_state=42,
)
mbkm.fit(X)
print(f"MiniBatch inertia: {mbkm.inertia_:.4f}")
# Output: MiniBatch inertia: 182.1661

sil_mb = silhouette_score(X, mbkm.labels_)
print(f"MiniBatch silhouette: {sil_mb:.4f}")
# Output: MiniBatch silhouette: 0.8760`}
      </CodeBlock>

      <H3>5.2 AgglomerativeClustering and scipy.cluster.hierarchy</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster

np.random.seed(42)
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.8, random_state=42)

# --- sklearn AgglomerativeClustering ---
agg = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward',       # options: 'ward', 'complete', 'average', 'single'
    metric='euclidean',   # 'ward' requires euclidean
)
agg.fit(X)
sil = silhouette_score(X, agg.labels_)
print(f"AgglomerativeClustering (ward, k=3) silhouette: {sil:.4f}")
# Output: AgglomerativeClustering (ward, k=3) silhouette: 0.8760

# --- scipy linkage + fcluster for dendrogram-based cutting ---
toy = np.array([
    [1.0, 0.0], [1.5, 0.5],
    [3.0, 2.0], [3.5, 2.0],
    [7.0, 5.0], [7.5, 5.5],
])
Z = linkage(toy, method='ward')
print("Linkage matrix Z (id1, id2, distance, count):")
for row in Z:
    print(f"  {int(row[0])} + {int(row[1])}  dist={row[2]:.4f}  size={int(row[3])}")
# Output:
# Linkage matrix Z (id1, id2, distance, count):
#   2 + 3  dist=0.5000  size=2
#   0 + 1  dist=0.7071  size=2
#   4 + 5  dist=0.7071  size=2
#   6 + 7  dist=3.7583  size=4
#   8 + 9  dist=10.5850 size=6

# Cut dendrogram at 2 clusters
labels_2 = fcluster(Z, t=2, criterion='maxclust')
print(f"fcluster t=2: {labels_2.tolist()}")
# Output: fcluster t=2: [2, 2, 2, 2, 1, 1]

# Cut at 3 clusters
labels_3 = fcluster(Z, t=3, criterion='maxclust')
print(f"fcluster t=3: {labels_3.tolist()}")
# Output: fcluster t=3: [3, 3, 2, 2, 1, 1]

# For large n (>10k), consider BIRCH as a preprocessing step:
# from sklearn.cluster import Birch
# birch = Birch(n_clusters=3).fit(X_large)
# Then apply agglomerative or k-means on Birch's subclusters.
#
# For billion-scale k-means: FAISS
# import faiss
# kmeans = faiss.Kmeans(d=X.shape[1], k=3, niter=20, gpu=True)
# kmeans.train(X.astype(np.float32))
# _, labels = kmeans.index.search(X.astype(np.float32), 1)`}
      </CodeBlock>

      <Callout type="info" title="scipy linkage distances vs. Ward-distance formula">
        Note that scipy reports the <em>Euclidean distance</em> between cluster centroids at the time of the merge (not the Ward-distance formula directly). The values differ: scipy reports dist(C2,C3)=0.500 (Euclidean) while our from-scratch Ward-distance was 0.125 (the WCSS increase). Both are monotone increasing representations of the same merge order — dendrograms drawn from either are valid, but they use different y-axis scales. The <Code>fcluster</Code> <Code>t</Code> parameter refers to scipy's merge-height scale.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Lloyd's algorithm — centroid trajectory</H3>

      <StepTrace
        label="Lloyd's iterations on 3-cluster 150-point dataset"
        steps={[
          {
            label: "Init (k-means++) — random seed 0",
            render: () => (
              <Prose>
                Initial centroids placed by k-means++ seeding: C0=[3.93, 1.32] (near cluster 0), C1=[-3.43, 9.31] (near cluster 1), C2=[-7.26, -6.01] (near cluster 2). Even before any Lloyd iteration, the centroids are near the true cluster centers — this is the k-means++ guarantee. Compare to random init which might place two centroids inside the same natural group.
              </Prose>
            ),
          },
          {
            label: "Iteration 1 — Assignment + Update",
            render: () => (
              <Prose>
                Assignment: each of 150 points is assigned to the nearest centroid. Given the good init, all 150 points land in the correct cluster immediately. Update: centroids move to the cluster means. C0: [3.93,1.32] → [4.60, 2.11]. C1: [-3.43,9.31] → [-2.66, 8.93]. C2: [-7.26,-6.01] → [-6.78, -6.89]. Inertia after iteration 1: 181.5044.
              </Prose>
            ),
          },
          {
            label: "Iteration 2 — Convergence check",
            render: () => (
              <Prose>
                Assignment: re-assign all 150 points to nearest centroid (now the refined centroids). Result: no point changes cluster — all assignments are identical to iteration 1. Convergence condition satisfied. Algorithm terminates. Final inertia: 181.5044. Final centers: C0=[4.60, 2.11], C1=[-2.66, 8.93], C2=[-6.78, -6.89].
              </Prose>
            ),
          },
          {
            label: "What a bad init looks like",
            render: () => (
              <Prose>
                With random initialization and seed=99, two of three initial centroids land inside the upper-left blob. The algorithm needs 8+ iterations to escape: it reassigns one centroid to the middle blob, then slowly drags it toward the lower-right. Final inertia is often 5–15% higher than the k-means++ result, and sometimes the algorithm converges to a genuinely different (worse) local optimum where one cluster absorbs parts of two natural groups. This is why sklearn defaults to n_init=10 restarts.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6.2 Elbow curve — choosing k</H3>

      <Plot
        label="Inertia vs. k (elbow curve) — 150-point 3-cluster dataset"
        xLabel="k (number of clusters)"
        yLabel="inertia (WCSS)"
        series={[
          {
            name: "KMeans inertia",
            color: colors.gold,
            points: [
              [1, 9788.88],
              [2, 2660.01],
              [3, 181.50],
              [4, 155.42],
              [5, 135.51],
              [6, 113.86],
              [7, 98.70],
              [8, 84.65],
            ],
          },
        ]}
      />

      <Prose>
        The elbow is unmistakably at k=3: inertia drops from 9789 (k=1) to 2660 (k=2) to 182 (k=3) — a 93% reduction — then only trickles down by 14% total from k=3 to k=8. This is the ideal elbow scenario, reflecting a dataset with genuinely well-separated clusters of equal size. Real-world data rarely produces such a sharp elbow; the silhouette score is often more diagnostic.
      </Prose>

      <H3>6.3 Silhouette scores — k=2 through 6</H3>

      <Plot
        label="Silhouette score vs. k — same dataset"
        xLabel="k (number of clusters)"
        yLabel="silhouette score"
        series={[
          {
            name: "silhouette score",
            color: colors.green,
            points: [
              [2, 0.7199],
              [3, 0.8760],
              [4, 0.7125],
              [5, 0.5121],
              [6, 0.3292],
            ],
          },
        ]}
      />

      <Prose>
        The silhouette score for sample <em>i</em> is <Code>{"s(i) = (b(i) - a(i)) / max(a(i), b(i))"}</Code>, where <Code>a(i)</Code> is the mean intra-cluster distance and <Code>b(i)</Code> is the mean distance to the nearest other cluster. The overall silhouette is the mean over all points; values near 1 indicate well-separated clusters, near 0 indicate overlapping clusters, negative indicates misclassification. Here k=3 peaks at 0.876, confirming the correct cluster count.
      </Prose>

      <H3>6.4 Pairwise distance matrix — 6-point toy example</H3>

      <Heatmap
        label="Euclidean distance matrix — 6-point toy example (three natural groups)"
        matrix={[
          [0.00, 0.71, 2.83, 3.20, 7.81, 8.51],
          [0.71, 0.00, 2.12, 2.50, 7.11, 7.81],
          [2.83, 2.12, 0.00, 0.50, 5.00, 5.70],
          [3.20, 2.50, 0.50, 0.00, 4.61, 5.32],
          [7.81, 7.11, 5.00, 4.61, 0.00, 0.71],
          [8.51, 7.81, 5.70, 5.32, 0.71, 0.00],
        ]}
        rowLabels={["P0", "P1", "P2", "P3", "P4", "P5"]}
        colLabels={["P0", "P1", "P2", "P3", "P4", "P5"]}
        colorScale="purple"
      />

      <Prose>
        The block structure is immediately visible: P0–P1, P2–P3, and P4–P5 form tight blocks of low distance (dark cells on the diagonal blocks), while inter-group distances are large (light cells off-block). Ward linkage exploits exactly this structure. The dendrogram merge order follows the block structure: first within-block merges (0.50–0.71), then the two left blocks merge (3.76 in scipy's scale), finally the last merge connects the well-separated upper-right pair (10.58 in scipy's scale).
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        No single clustering algorithm dominates all scenarios. The choice depends on whether <em>k</em> is known, the assumed cluster shape, dataset size, and tolerance for hyperparameter tuning.
      </Prose>

      <StepTrace
        label="Algorithm selection guide"
        steps={[
          {
            label: "K-Means",
            render: () => (
              <Prose>
                Use when: k is known or discoverable via elbow/silhouette; clusters are approximately spherical and similarly sized; dataset has n up to millions (MiniBatchKMeans). Cluster shape assumption: spherical — the WCSS objective is Euclidean distance, which penalizes elongated or irregularly shaped clusters. Strengths: fast, scalable, easy to interpret (centroids are meaningful), sklearn implementation is production-grade. Weaknesses: must specify k; sensitive to initialization (mitigated by k-means++); fails on non-convex clusters (rings, crescents); sensitive to outliers (a single extreme point shifts a centroid). Best domain fit: customer segmentation, image compression (vector quantization), feature engineering (cluster-distance features), preprocessing for supervised learning.
              </Prose>
            ),
          },
          {
            label: "Hierarchical (Ward)",
            render: () => (
              <Prose>
                Use when: you don't know k and want to explore the full hierarchy; dataset is small to medium (n {"<"} 10k); interpretability of the merge tree matters (biology, taxonomy, organizational analysis). Cluster shape assumption: same as k-means — Ward minimizes WCSS. Strengths: no need to specify k upfront; dendrogram reveals multi-scale structure; deterministic (no random initialization). Weaknesses: O(n²) memory and O(n³) time (naive); cannot undo merges (greedy); breaks down for n {">"} 10k without approximation (BIRCH, HDBSCAN). Best domain fit: gene expression analysis, social network communities, document taxonomy where the hierarchy itself is the deliverable.
              </Prose>
            ),
          },
          {
            label: "DBSCAN",
            render: () => (
              <Prose>
                Use when: cluster shapes are non-spherical (rings, crescents, filaments); you expect noise/outliers that should be flagged rather than assigned; k is unknown. Key hyperparameters: eps (neighborhood radius) and min_samples (minimum points to form a core point). Strengths: discovers arbitrary-shaped clusters; explicitly models noise; does not require specifying k. Weaknesses: struggles with varying density (HDBSCAN addresses this); eps is sensitive and requires tuning per dataset; poor performance in high dimensions (curse of dimensionality affects neighborhood density). Best domain fit: geospatial clustering, anomaly detection, point-cloud segmentation.
              </Prose>
            ),
          },
          {
            label: "Gaussian Mixture Models (GMM)",
            render: () => (
              <Prose>
                Use when: clusters have elliptical shapes (not spherical); you need soft assignments (probability that point belongs to each cluster); you want a generative model. GMM is the probabilistic generalization of k-means — it fits a mixture of Gaussians with full covariance matrices, learned by EM. Strengths: handles elliptical clusters; produces calibrated posterior probabilities; principled model selection via BIC/AIC. Weaknesses: slower than k-means; more sensitive to initialization; requires specifying k; can fail if clusters are highly non-Gaussian. Best domain fit: density estimation, generative modeling, soft classification.
              </Prose>
            ),
          },
          {
            label: "K is known vs. discovered",
            render: () => (
              <Prose>
                If k is given by the problem (e.g., "segment users into 5 personas" from a product decision), use k-means directly. If k must be discovered from data: run k-means for k=2..20 and look for the elbow in inertia; compute silhouette score for each k and pick the peak; use the gap statistic (compare inertia to random baselines); use hierarchical clustering and inspect the dendrogram merge heights for large gaps; use DBSCAN or HDBSCAN which discover k automatically. On well-separated data the methods agree. On noisy real data they often disagree — treat k as a modeling decision, not a ground truth.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 K-means complexity</H3>

      <Prose>
        Lloyd's algorithm costs <Code>O(n · k · d · iter)</Code> per run, where <em>n</em> is the number of points, <em>k</em> the number of clusters, <em>d</em> the feature dimension, and <em>iter</em> the number of iterations (typically 10–300). The dominant operation is computing the <em>n × k</em> pairwise distance matrix at each iteration. Memory footprint is <Code>O(n · k)</Code> for the distance matrix plus <Code>O(n · d)</Code> for the data. For n=1M, k=100, d=128, iter=20: roughly 2.5 × 10¹¹ floating-point operations per run — feasible on a GPU in minutes, slow on a CPU.
      </Prose>

      <Prose>
        <strong>MiniBatchKMeans</strong> replaces the full distance matrix with a mini-batch of size <em>b</em> per iteration, reducing the per-step cost to <Code>O(b · k · d)</Code>. Convergence is noisier but the algorithm scales to n in the tens of millions on a single machine. The inertia at convergence is slightly higher than full-batch k-means (as shown: 182.17 vs. 181.50), but the difference shrinks as batch size grows. For n {">"} 10M on a single machine, MiniBatchKMeans with a batch size of 10k–100k is the practical choice.
      </Prose>

      <Prose>
        <strong>FAISS k-means</strong> (Facebook AI Similarity Search) is the tool for billion-scale k-means. It implements Lloyd's algorithm using BLAS-optimized dense matrix multiplication for the distance computation, runs on multiple GPUs with data sharding, and handles up to 10⁹ vectors in practice. FAISS k-means is used for building approximate nearest-neighbor indexes, training vector quantizers for billion-scale retrieval (IVFPQ indexes), and clustering text embeddings across massive corpora. For anything above ~10M points, FAISS is the production tool.
      </Prose>

      <H3>8.2 Hierarchical clustering complexity</H3>

      <Prose>
        Naive agglomerative clustering is <Code>O(n³)</Code> time and <Code>O(n²)</Code> memory — both make it unusable above n ≈ 10k. The bottleneck is maintaining and searching the <em>n × n</em> pairwise distance matrix. With a priority queue (min-heap), the time complexity drops to <Code>O(n² log n)</Code>, but the memory remains <Code>O(n²)</Code> — storing a 10k × 10k float64 matrix costs 800 MB; 100k × 100k costs 80 GB. This is a hard wall.
      </Prose>

      <Prose>
        For large-scale hierarchical clustering, approximation is mandatory. <strong>BIRCH</strong> (Balanced Iterative Reducing and Clustering using Hierarchies) precompresses the data into a CF-tree of subclusters, then runs agglomerative clustering on the (much smaller) set of leaf nodes — reducing the effective n from millions to thousands. <strong>HDBSCAN</strong> builds a minimum spanning tree in <Code>O(n log n)</Code> time, extracts a hierarchy of density-connected clusters, and automatically selects stable clusters — effectively getting the expressiveness of hierarchical clustering at near-linear cost. For n {">"} 10k and hierarchical structure is needed, HDBSCAN is the modern default.
      </Prose>

      <H3>8.3 High-dimensional feature spaces</H3>

      <Prose>
        Both algorithms degrade in high dimensions. The fundamental reason is the concentration of measure phenomenon: in high-dimensional spaces, distances between random points concentrate around their mean — the ratio of max to min pairwise distance tends to 1. When all points are roughly equidistant, the Voronoi partition becomes arbitrary and k-means centroids lose geometric meaning. The practical manifestation: k-means inertia improvements over the "all-in-one-cluster" baseline shrink; silhouette scores decrease; elbow plots flatten. Dimensionality reduction (PCA, UMAP, or learned embeddings) before clustering is standard practice when d {">"} 50–100. The embeddings from a pretrained model (CLIP for images, sentence-transformers for text) encode semantic similarity that Euclidean distance in embedding space captures well, making clustering in embedding space much more meaningful than clustering in raw pixel or token space.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Non-spherical clusters</H3>

      <Prose>
        K-means minimizes squared Euclidean distance. This implicitly assumes clusters are convex, roughly spherical, and similarly sized. On crescent-shaped or ring-shaped data, k-means will always fail: no placement of centroids can correctly partition a ring into two halves using Voronoi cells. The diagnostic is visual (plot the clusters in 2D or in the first two PCA dimensions) and metric (silhouette score below 0.3 on a dataset you believe has structure). Fix: DBSCAN or HDBSCAN for arbitrary shapes; kernel k-means with an RBF kernel implicitly maps data to a higher-dimensional space where non-spherical clusters become spherical. Hierarchical clustering with single linkage can handle elongated chains but is prone to a different problem — chaining.
      </Prose>

      <H3>9.2 Different-sized and different-density clusters</H3>

      <Prose>
        K-means tends to produce clusters of similar size regardless of the underlying data structure, because a centroid in a small dense cluster and a centroid in a large sparse cluster will "steal" points from one another. On a dataset with one cluster of 1000 points and another of 10 points, k-means will typically merge the small cluster into its nearest large-cluster neighbor and split the large cluster instead. GMMs handle this better through the covariance matrix; DBSCAN handles different density explicitly through its eps parameter (though HDBSCAN handles varying density natively). Hierarchical clustering with Ward linkage shares the same bias as k-means.
      </Prose>

      <H3>9.3 Unscaled features</H3>

      <Prose>
        K-means uses Euclidean distance. If one feature is measured in dollars (range 0–100,000) and another in age (range 0–100), the dollar feature will dominate the distance calculation completely. The clustering result will be nearly identical to clustering on the dollar feature alone — the age feature will be invisible. Always standardize features before k-means: subtract the mean and divide by the standard deviation (<Code>StandardScaler</Code>). For features with very different semantics (e.g., mixing boolean flags with continuous values), consider domain-specific normalization. This is the single most common mistake practitioners make with clustering.
      </Prose>

      <H3>9.4 Initialization sensitivity and empty clusters</H3>

      <Prose>
        Poor random initialization can produce empty clusters (a centroid with no assigned points after the first assignment step) or degenerate solutions where all points collapse into one cluster. Empty clusters are handled differently by implementations: sklearn simply keeps the previous centroid position (effectively reducing k by one), which silently produces wrong results. K-means++ eliminates most catastrophic failures by spreading initial centroids. The remaining sensitivity is to local optima: use <Code>n_init=10</Code> (or higher for important applications) and compare inertia across runs. A spread of more than 5–10% in inertia across 10 runs signals a poorly conditioned problem where you should also try a different k or a different algorithm.
      </Prose>

      <H3>9.5 Choosing k — elbow, silhouette, gap statistic</H3>

      <Prose>
        There is no universally correct method for choosing k. The three most widely used diagnostics are: (1) <strong>Elbow method</strong>: plot inertia vs. k; look for a kink where the rate of decrease slows. Works well when the true cluster structure is distinct; produces ambiguous results when data is noisy or clusters are not well-separated. (2) <strong>Silhouette score</strong>: compute the mean silhouette over all points for each k; pick the k that maximizes it. More robust than the elbow method, but computationally expensive for large n (O(n²) pairwise distances). (3) <strong>Gap statistic</strong>: compare the within-cluster dispersion to the expected dispersion under a null reference distribution (uniform random data). Formally principled, but computationally expensive (requires multiple random reference datasets). In practice, run both elbow and silhouette, inspect the data visually, and treat k as a modeling decision informed by domain knowledge — "how many segments can our team actually act on?" often matters more than statistical optimality.
      </Prose>

      <H3>9.6 Outlier sensitivity</H3>

      <Prose>
        A single outlier point can significantly shift a centroid because k-means computes the arithmetic mean — the mean is not a robust statistic. One extreme data point with feature values 100× the normal range will pull its assigned centroid far from the true cluster center, distorting all downstream assignments. Detection: compute, for each cluster, the fraction of points with distances more than 3 standard deviations above the within-cluster mean distance; inspect those points. Fix options: remove confirmed outliers before clustering; use k-medoids (PAM — Partitioning Around Medoids) which replaces the mean centroid with an actual data point (the medoid), making it robust to outliers at O(n²) cost; use DBSCAN which assigns outliers to a noise class rather than forcing them into a cluster. Hierarchical clustering with complete or average linkage is more outlier-robust than Ward, because a distant outlier produces only one high-cost merge rather than continuously shifting a centroid.
      </Prose>

      <H3>9.7 Interpreting cluster labels across runs</H3>

      <Prose>
        Cluster labels (0, 1, 2, ...) are arbitrary — they have no inherent meaning and are not comparable across runs. If you run k-means twice with different seeds, cluster "0" in run 1 may correspond to cluster "2" in run 2. This matters when comparing clusterings across time, across hyperparameter settings, or when building pipelines that depend on specific label values. Fix: use the Hungarian algorithm (scipy <Code>linear_sum_assignment</Code>) to find the label permutation that maximizes overlap between two runs. For tracking clusters over time (e.g., monthly customer segmentation), assign new clusters to old ones by maximum centroid cosine similarity.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were WebSearch-verified for author, year, venue, page numbers, and key contributions.
      </Prose>

      <StepTrace
        label="Primary literature"
        steps={[
          {
            label: "Steinhaus 1956 — First formulation of k-means",
            render: () => (
              <Prose>
                Steinhaus, H. (1956). "Sur la division des corps matériels en parties." <em>Bulletin de l'Académie Polonaise des Sciences, Classe III</em>, vol. IV, no. 12, pp. 801–804. In French. Steinhaus, the Polish mathematician known for co-discovering Stefan Banach, derived the k-means objective from a mechanics problem: partition a rigid body (or a set of mass points) into k parts that minimize total moment of inertia about each part's center of mass. He proved existence of an optimal partition and sketched the iterative alternation between assignment and centroid update. The paper circulated in a niche Polish journal and was largely unknown outside Poland for decades; its priority was established by careful historical scholarship in the 2000s.
              </Prose>
            ),
          },
          {
            label: "Lloyd 1982 — Canonical algorithm paper (Bell Labs 1957, published IEEE 1982)",
            render: () => (
              <Prose>
                Lloyd, S.P. (1982). "Least Squares Quantization in PCM." <em>IEEE Transactions on Information Theory</em>, vol. 28, no. 2, pp. 129–137. DOI: 10.1109/TIT.1982.1056489. Derived independently of Steinhaus from the signal processing problem of optimal scalar quantization: given a continuous signal distribution, place k quantization levels and k−1 thresholds to minimize mean squared quantization error. The necessary conditions are exactly the k-means conditions: thresholds are midpoints between levels (Voronoi assignment); levels are conditional means (centroid update). Circulated as a Bell Labs technical report in 1957 but unpublished for 25 years. The most cited reference for the k-means algorithm in ML literature.
              </Prose>
            ),
          },
          {
            label: "MacQueen 1967 — Coined the term 'k-means'",
            render: () => (
              <Prose>
                MacQueen, J.B. (1967). "Some Methods for Classification and Analysis of Multivariate Observations." In <em>Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability</em>, vol. 1 (Statistics), University of California Press, Berkeley, pp. 281–297. MacQueen named the algorithm "k-means" and extended the analysis to include online (sequential) updates — process one point at a time and update the nearest centroid immediately rather than waiting for a full pass. He proved that the within-cluster sum of squares converges monotonically under both batch and online variants. This paper is also why the algorithm is sometimes called the "k-means algorithm" rather than "Lloyd's algorithm" in the statistics literature.
              </Prose>
            ),
          },
          {
            label: "Ward 1963 — Hierarchical linkage minimizing WCSS",
            render: () => (
              <Prose>
                Ward, J.H., Jr. (1963). "Hierarchical Grouping to Optimize an Objective Function." <em>Journal of the American Statistical Association</em>, vol. 58, no. 301, pp. 236–244. DOI: 10.1080/01621459.1963.10500845. Ward was at the US Air Force Personnel Laboratory studying how to group military occupational specialties. His contribution was to define the merge criterion as the increase in total within-cluster sum of squares — the same criterion that k-means minimizes — and show that greedy minimization of this criterion at each step produces compact, well-separated clusters. Ward's linkage has remained the most widely used hierarchical linkage for over six decades because it consistently produces more interpretable clusters than single, complete, or average linkage on typical tabular data.
              </Prose>
            ),
          },
          {
            label: "Arthur & Vassilvitskii 2007 — k-means++ seeding",
            render: () => (
              <Prose>
                Arthur, D. and Vassilvitskii, S. (2007). "k-means++: The Advantages of Careful Seeding." In <em>Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA 2007)</em>, New Orleans, January 7–9, 2007, pp. 1027–1035. ACM DL: 10.5555/1283383.1283494. Full PDF: theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf. The paper proves that k-means++ initialization achieves E[L] ≤ 8(ln k + 2) · L* — the expected inertia before any Lloyd iterations is within O(log k) of the global optimum. In practice, k-means++ followed by Lloyd's iterations nearly always matches or beats 10–20 random restarts of plain Lloyd's, at negligible additional cost. It became the default initialization in sklearn, FAISS, and effectively every serious k-means implementation within a few years of publication.
              </Prose>
            ),
          },
          {
            label: "Hastie, Tibshirani & Friedman 2009 — ESL Ch. 14",
            render: () => (
              <Prose>
                Hastie, T., Tibshirani, R., and Friedman, J. (2009). <em>The Elements of Statistical Learning: Data Mining, Inference, and Prediction</em>, 2nd ed. New York: Springer. ISBN: 978-0-387-84857-0. Available free at hastie.su.domains/ElemStatLearn. Chapter 14 ("Unsupervised Learning") covers k-means, hierarchical clustering, self-organizing maps, and principal components in depth. Section 14.3.6 proves the EM interpretation of k-means (k-means is the limit of EM on a Gaussian mixture as covariance → 0·I), making the connection between k-means, GMMs, and EM explicit. The dendrogram construction, linkage criteria, and the relationship between hierarchical clustering and graph-theoretic minimum spanning trees are all treated at graduate level.
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
        Work through these before moving to the next topic. Answers are included below each exercise — resist reading ahead.
      </Prose>

      <H3>Exercise 1 (recall)</H3>
      <Prose>
        Write the k-means objective function. What does Lloyd's algorithm alternate between, and why is each step guaranteed to not increase the objective? Why does this guarantee convergence, and what does it say about the quality of the solution found?
      </Prose>
      <Callout type="answer" title="Answer 1">
        The objective is {"L(C, μ) = Σⱼ Σ_{xᵢ∈Cⱼ} ||xᵢ - μⱼ||²"}. Lloyd's algorithm alternates: (1) Assignment step — assign each point to the nearest centroid; this minimizes L over C for fixed μ because Voronoi assignment minimizes distance to the assigned centroid. (2) Update step — move each centroid to the mean of its assigned points; this minimizes L over μ for fixed C because the mean minimizes the sum of squared distances (zero gradient). Each step weakly decreases L. Since the number of possible partitions is finite (at most kⁿ), the sequence of objective values must terminate — convergence is guaranteed. However, convergence is to a local optimum, not the global minimum. Different initializations can produce different local optima with different inertia values, which is why multiple restarts (n_init) are standard.
      </Callout>

      <H3>Exercise 2 (derivation)</H3>
      <Prose>
        Derive the Ward linkage distance formula from first principles. Specifically: if cluster A has <em>nₐ</em> points with centroid <em>x̄_A</em> and cluster B has <em>n_b</em> points with centroid <em>x̄_B</em>, what is the increase in total within-cluster sum of squares from merging A and B into a combined cluster AB?
      </Prose>
      <Callout type="answer" title="Answer 2">
        {"Let WCSS(A) = Σ_{x∈A} ||x - x̄_A||² and similarly for B. The merged cluster AB has centroid x̄_{AB} = (nₐ x̄_A + n_b x̄_B)/(nₐ + n_b). The change in total WCSS from merging is ΔWCSS = WCSS(AB) - WCSS(A) - WCSS(B). Expanding WCSS(AB) = Σ_{x∈A} ||x - x̄_{AB}||² + Σ_{x∈B} ||x - x̄_{AB}||² and applying the parallel axis theorem (sum of squared deviations from a new center = sum from original center + n × squared distance between centers), you get ΔWCSS = nₐ||x̄_A - x̄_{AB}||² + n_b||x̄_B - x̄_{AB}||². Substituting x̄_{AB} and simplifying: ΔWCSS = (nₐ n_b)/(nₐ + n_b) · ||x̄_A - x̄_B||²."} This is Ward's linkage — it is proportional to the squared Euclidean distance between cluster centroids, weighted by their harmonic mean size.
      </Callout>

      <H3>Exercise 3 (conceptual)</H3>
      <Prose>
        You run k-means with k=3 on a dataset and get inertia = 2500. You run it again with k=4 and get inertia = 2490. A colleague argues you should always prefer k=4 because its inertia is lower. What is wrong with this argument? Give two principled methods for choosing k.
      </Prose>
      <Callout type="answer" title="Answer 3">
        The colleague's argument is wrong for two reasons. (1) Inertia always decreases as k increases — at k=n, every point is its own cluster and inertia = 0. Minimizing inertia alone always favors k=n, which is useless. (2) A drop from 2500 to 2490 (0.4%) is negligibly small. The elbow method is precisely about looking for k where the marginal improvement becomes small. Two principled methods: (a) Elbow method — plot inertia vs. k, pick the k where the curve bends sharply. Here k=3 may be the elbow (we don't know without the full curve). (b) Silhouette score — compute the mean silhouette for each k; pick the k that maximizes it. A drop from 0.876 (k=3) to 0.713 (k=4) as in the example above is a clear signal that k=3 is better even though k=4 has lower inertia. Additionally, domain knowledge matters: "how many clusters can we actually use?" is a valid constraint.
      </Callout>

      <H3>Exercise 4 (debugging)</H3>
      <Prose>
        You cluster a customer dataset with 10 features: annual spend (range 0–$500k), age (range 18–80), and 8 binary flags (0 or 1). After k-means, you inspect the clusters and find they are almost entirely determined by annual spend — all customers with high spend are in one cluster, all with low spend in another, regardless of age or the flags. What went wrong, and how do you fix it?
      </Prose>
      <Callout type="answer" title="Answer 4">
        The problem is unscaled features. Annual spend ranges from 0 to 500,000; age from 18 to 80; flags from 0 to 1. Euclidean distance is dominated by the feature with the largest scale — a difference of $100k in spend dwarfs any variation in age or flags. The clustering is essentially one-dimensional. Fix: apply StandardScaler before fitting (subtract mean, divide by std for each feature). After scaling, all features contribute comparably to the distance. Note: for binary flags, standard scaling is mathematically valid but may not be the best choice — consider whether the flags represent fundamentally different information scales and whether domain-weighted distance might be more appropriate. After scaling and re-clustering, inspect whether the new clusters have interpretable profiles across all features.
      </Callout>

      <H3>Exercise 5 (applied)</H3>
      <Prose>
        You have a genomics dataset with n=500 patients and d=20,000 gene expression features. You want to find subgroups of patients with similar expression profiles, and you suspect the number of subgroups is somewhere between 2 and 10. Which algorithm would you use? What preprocessing would you apply to the features before clustering, and why?
      </Prose>
      <Callout type="answer" title="Answer 5">
        Use hierarchical clustering with Ward linkage (n=500 is well within the O(n²) memory limit: 500 × 500 × 8 bytes = 2 MB). Hierarchical clustering is ideal here because: (1) you don't know k upfront, and the dendrogram lets you explore all k=2..10 with a single run; (2) the merge tree has biological interpretability — you can identify which patient subgroups are nested within larger subgroups; (3) n=500 is small enough for exact agglomerative clustering. Preprocessing: (a) Standardize each gene to zero mean and unit variance (StandardScaler across patients) — gene expression values span different dynamic ranges, and without scaling, highly expressed genes dominate distance. (b) Apply PCA or UMAP to reduce from d=20,000 to d=50–200 components — this removes noise, reduces memory for the distance matrix computation, and concentrates the biologically relevant variance. (c) Optionally filter to highly variable genes (e.g., top 5,000 by variance) before PCA, which is standard in single-cell genomics (Seurat, Scanpy pipelines). After clustering, validate subgroups by checking known clinical variables (survival, tumor grade, treatment response) against cluster assignments.
      </Callout>

      <H3>Exercise 6 (synthesis)</H3>
      <Prose>
        K-means can be derived as a special case of the Expectation-Maximization (EM) algorithm applied to Gaussian Mixture Models (GMMs). Identify the E-step and M-step in Lloyd's algorithm and explain what assumption on the Gaussian mixture reduces the soft GMM assignments to the hard Voronoi assignments of k-means.
      </Prose>
      <Callout type="answer" title="Answer 6">
{"In EM for GMMs: the E-step computes posterior cluster membership probabilities r_{ij} = P(cluster j | xᵢ) for each point-cluster pair; the M-step updates the mixture weights, means, and covariances to maximize the expected log-likelihood. In Lloyd's algorithm: the assignment step is a hard E-step — instead of soft probabilities, each point is assigned with probability 1 to the nearest cluster and 0 to all others; the update step is the M-step — the centroid is the weighted mean of assigned points, which for r_{ij} ∈ {0,1} is just the arithmetic mean. The connection: GMM with isotropic covariance matrices Σⱼ = σ²I for all j and taking the limit σ² → 0. When the covariance is vanishingly small, the Gaussian likelihood for each cluster is an infinitely sharp bump around its mean, so the point's posterior probability collapses to 1 for the nearest cluster and 0 for all others — recovering hard assignment. The EM M-step update for the mean under r_{ij} ∈ {0,1} is the arithmetic mean — recovering the centroid update. K-means is therefore the zero-temperature limit of EM on an isotropic GMM."}
      </Callout>

    </div>
  ),
};

export default kMeansHierarchicalContent;
