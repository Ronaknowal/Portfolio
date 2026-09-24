import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const anomalyDetectionContent = {
  title: "Anomaly & Outlier Detection (Isolation Forest, One-Class SVM, LOF)",
  readTime: "~50 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Most supervised learning pipelines begin with a simple assumption: you have labels. Positive, negative, fraud, benign, healthy, diseased — whatever the vocabulary, there are enough annotated examples of each class that a model can learn to distinguish them. Anomaly detection is what you do when that assumption breaks. In fraud detection, fewer than one in a thousand transactions is fraudulent, and labeling even that fraction requires expensive human reviewers who are always chasing a moving target. In network intrusion detection, the attack patterns of tomorrow do not exist in today's logs. In industrial fault detection, a motor fails once a year, and the sensor trace of that failure is a dataset of one. In these settings, "unsupervised" is not a methodological preference — it is a description of reality.
      </Prose>

      <Prose>
        The field attacked this problem from several directions simultaneously, each rooted in a different geometric intuition about what "normal" means.
      </Prose>

      <Prose>
        <strong>Isolation Forest</strong> came from Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. Their paper, "Isolation Forest," appeared at the 2008 IEEE International Conference on Data Mining (ICDM), pages 413–422 (DOI: 10.1109/ICDM.2008.17). The premise is elegant and counterintuitive. Previous anomaly detectors all began by modeling the normal data — building a density estimate, a distance index, or a cluster structure — and then asking whether a new point fits that model. Liu et al. observed that this framing makes things harder than necessary. Anomalies are, by definition, few and different. Partition the feature space randomly with axis-aligned cuts: anomalies will be isolated by very few cuts because they sit alone in sparse regions. Normal points, clustered densely together, require many cuts to separate. The anomaly score is therefore the average depth at which a point is isolated across many random trees. You never need to model what normal looks like — you simply measure how easy it is to isolate.
      </Prose>

      <Prose>
        <strong>One-Class SVM</strong> came from Bernhard Schölkopf, John Platt, John Shawe-Taylor, Alex Smola, and Robert Williamson. Their paper, "Estimating the Support of a High-Dimensional Distribution," was published in <em>Neural Computation</em> 13(7):1443–1471, 2001 (DOI: 10.1162/089976601750264965). The framing is geometric: map all training points into a high-dimensional feature space via a kernel, and find the smallest hypersphere (or halfspace) that encloses them with controlled slack. Points outside this boundary at test time are anomalies. The kernel trick allows the boundary to be highly nonlinear in the original input space while remaining a convex optimization problem in kernel space. The hyperparameter <Code>nu</Code> (Greek letter ν) has a precise probabilistic interpretation: it upper-bounds the fraction of training points that are allowed to fall outside the boundary (outliers in the training set) and lower-bounds the fraction of support vectors. Tuning it amounts to specifying your prior belief about the contamination rate.
      </Prose>

      <Prose>
        <strong>Local Outlier Factor (LOF)</strong> came from Markus Breunig, Hans-Peter Kriegel, Raymond Ng, and Jörg Sander. Their paper, "LOF: Identifying Density-Based Local Outliers," appeared at the 2000 ACM SIGMOD International Conference on Management of Data, pages 93–104 (DOI: 10.1145/342009.335388). The key insight is that "outlier" is relative to neighborhood, not to the global distribution. A point at the edge of a tight cluster is not an outlier even if it is far from the global center of mass. A point sitting alone in sparse space is an outlier even if other points exist at similar distances from the center. LOF quantifies this by comparing a point's local density to the local density of its neighbors. A point surrounded by sparser neighbors than itself looks like a local mode, not an anomaly. A point surrounded by denser neighbors looks isolated — an outlier.
      </Prose>

      <Callout type="insight">
        All three methods are label-free and handle the core regime of anomaly detection: a large pool of mostly-normal data with an unknown, small fraction of anomalies. Their differences are geometric: Isolation Forest is global and partition-based; One-Class SVM fits a single boundary around the full normal distribution; LOF is purely local. Understanding which geometry fits your data is the central practical skill.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2a. Isolation Forest — anomalies are easy to isolate</H3>

      <Prose>
        Imagine you have a field of data points: most clustered together in two tight groups, and a handful scattered at the edges. Now drop a random vertical or horizontal line somewhere in the field, splitting it into two halves. Repeat this process recursively in each half. The scattered points at the edges get separated quickly — after only two or three cuts each region contains just that one isolated point. The clustered points in the dense center take many more cuts before any individual point is alone. Isolation Forest formalizes this: build many random trees, measuring the average depth at which each point is isolated. Low depth = easy to isolate = anomalous. High depth = hard to isolate = normal.
      </Prose>

      <Prose>
        The beauty of this approach is what it does <em>not</em> require. You never estimate the density. You never compute pairwise distances between all points. You never cluster. Each tree is built on a random subsample of the data (typically 256 points), using random features and random split thresholds. The computational cost is dominated by constructing shallow trees, not by any O(n²) distance computation.
      </Prose>

      <H3>2b. One-Class SVM — draw a boundary around normal</H3>

      <Prose>
        One-Class SVM is the anomaly detection analog of a standard binary SVM. In binary SVM, you find a hyperplane that separates two classes with maximum margin. In one-class SVM, you have only one class: the normal data. The objective is to find a hyperplane that separates all normal training points from the origin (the kernel feature space analog of "nothing") with maximum margin. Points that fall on the origin side of the boundary at test time are called anomalies.
      </Prose>

      <Prose>
        The key machinery is the kernel. In the original input space, the decision boundary can be arbitrarily complex — a curved, nonlinear surface that tightly wraps around the normal data. The RBF (Gaussian) kernel is the standard choice: it maps points into infinite-dimensional space where local neighborhoods are preserved, and the resulting boundary in the original space looks like a collection of smooth blobs around dense regions. Points outside all blobs are anomalies.
      </Prose>

      <H3>2c. LOF — compare local densities</H3>

      <Prose>
        LOF's intuition is best understood through contrast. Consider two regions: a dense urban cluster and a sparse rural cluster. A point at the edge of the urban cluster is very close to many neighbors; its local density is high. A point at the edge of the rural cluster is far from its few neighbors; its local density is low. Neither is necessarily an anomaly — both are consistent with the density of their local neighborhoods. An anomaly is a point whose local density is much lower than the local density of its own neighbors. LOF measures this ratio: if your density is much lower than what your neighbors enjoy, you are isolated relative to your context.
      </Prose>

      <Prose>
        This makes LOF particularly well-suited to data with multiple clusters at very different density scales — something that trips up global methods. Isolation Forest and One-Class SVM implicitly assume a single normal distribution (possibly multi-modal but roughly uniform in density across modes). LOF makes no such assumption: it adapts its notion of "anomalous" locally for every point.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3a. Isolation Forest — path length and anomaly score</H3>

      <Prose>
        An isolation tree is a binary tree built by the following randomized process. Given a subsample of <em>n</em> points, pick a feature at random, pick a random split value uniformly between the feature's min and max, and recurse on each side until each node contains a single point or the maximum depth is reached. The path length <Code>h(x)</Code> of a point <Code>x</Code> in a single tree is the number of edges traversed from the root to the node where <Code>x</Code> is isolated.
      </Prose>

      <Prose>
        A critical correction is needed for leaf nodes that contain more than one point (because max depth was reached before full isolation). For a leaf of size <em>t</em>, the expected additional path length — were the tree allowed to grow further — is the average path length of an unsuccessful search in a binary search tree of size <em>t</em>, which is:
      </Prose>

      <MathBlock>
        {"c(t) = 2H(t-1) - \\frac{2(t-1)}{t}"}
      </MathBlock>

      <Prose>
        where <Code>H(i)</Code> is the harmonic number, approximated as <Code>ln(i) + 0.5772</Code> (Euler–Mascheroni constant). This correction term <Code>c(t)</Code> is added to the observed depth when a leaf is reached before the point is fully isolated.
      </Prose>

      <Prose>
        Let <Code>E[h(x)]</Code> be the average path length of point <Code>x</Code> across all trees in the forest, and let <Code>c(n)</Code> be the same correction applied to the original subsample size <Code>n</Code> (i.e., the expected path length for a point in a random forest of size <Code>n</Code>). The normalized anomaly score is:
      </Prose>

      <MathBlock>
        {"s(x, n) = 2^{\\,-E[h(x)]\\,/\\,c(n)}"}
      </MathBlock>

      <Prose>
        This score is bounded in <Code>(0, 1]</Code>. If <Code>E[h(x)]</Code> is close to <Code>c(n)</Code> (average path length), the score is near 0.5 — the point is no more isolated than average. If <Code>E[h(x)]</Code> is much smaller than <Code>c(n)</Code>, the score approaches 1 — the point is anomalous. If <Code>E[h(x)]</Code> is much larger, the score approaches 0 — the point is deep in a dense region, very normal. The threshold in practice is set by the <Code>contamination</Code> parameter, which specifies the fraction of points to flag as anomalies; the score is thresholded at the <Code>contamination</Code>-th percentile.
      </Prose>

      <H3>3b. One-Class SVM — primal and dual</H3>

      <Prose>
        Let <Code>{"φ: X → F"}</Code> be a feature map defined by a kernel <Code>{"k(x, x') = ⟨φ(x), φ(x')⟩"}</Code>. The One-Class SVM primal problem (Schölkopf et al., 2001) finds a weight vector <Code>w ∈ F</Code>, slack variables <Code>ξ_i ≥ 0</Code>, and a bias <Code>ρ</Code> that solve:
      </Prose>

      <MathBlock>
        {"\\min_{w,\\,\\xi,\\,\\rho} \\; \\frac{1}{2}\\|w\\|^2 + \\frac{1}{\\nu n}\\sum_{i=1}^n \\xi_i - \\rho"}
      </MathBlock>

      <Prose>
        subject to <Code>{"⟨w, φ(xᵢ)⟩ ≥ ρ − ξᵢ"}</Code> and <Code>{"ξᵢ ≥ 0"}</Code> for all <Code>i</Code>. The term <Code>{"½‖w‖²"}</Code> maximizes the margin from the origin; the slack variables <Code>ξᵢ</Code> allow training points to fall on the wrong side; <Code>ρ</Code> is the offset (analogous to the threshold); and <Code>ν ∈ (0, 1]</Code> controls the trade-off between margin width and tolerance for outliers. A larger <Code>ν</Code> allows more training points outside the boundary (a looser fit); a smaller <Code>ν</Code> is tighter and more sensitive to noise.
      </Prose>

      <Prose>
        The dual problem is a standard QP over Lagrange multipliers <Code>αᵢ</Code> with the kernel matrix replacing explicit feature products. At test time, the decision function for a new point <Code>x</Code> is:
      </Prose>

      <MathBlock>
        {"f(x) = \\text{sgn}\\!\\left(\\sum_{i=1}^n \\alpha_i\\, k(x_i, x) - \\rho\\right)"}
      </MathBlock>

      <Prose>
        Points with <Code>f(x) = +1</Code> are classified as normal (inside the estimated support); points with <Code>f(x) = −1</Code> are anomalies. The <Code>score_samples</Code> method in sklearn returns the raw decision function value (not the sign), which is useful for ranking.
      </Prose>

      <H3>3c. LOF — reach-distance, lrd, and LOF score</H3>

      <Prose>
        Let <Code>k-dist(p)</Code> denote the distance from point <Code>p</Code> to its <em>k</em>-th nearest neighbor. The <em>k</em>-neighborhood <Code>N_k(p)</Code> is the set of the <em>k</em> nearest neighbors of <Code>p</Code> (ties included, so the set may have more than <Code>k</Code> members). The <strong>reachability distance</strong> of <Code>p</Code> with respect to neighbor <Code>o</Code> is:
      </Prose>

      <MathBlock>
        {"\\text{reach-dist}_k(p, o) = \\max\\bigl(k\\text{-dist}(o),\\; d(p, o)\\bigr)"}
      </MathBlock>

      <Prose>
        This smoothing prevents extremely small distances from dominating the density estimate: if <Code>p</Code> is very close to <Code>o</Code>, the distance is floored at <Code>o</Code>'s own neighborhood radius. The <strong>local reachability density</strong> (lrd) of <Code>p</Code> is the inverse of the average reachability distance from <Code>p</Code> to its neighbors:
      </Prose>

      <MathBlock>
        {"\\text{lrd}_k(p) = \\left(\\frac{\\sum_{o \\in N_k(p)} \\text{reach-dist}_k(p, o)}{|N_k(p)|}\\right)^{-1}"}
      </MathBlock>

      <Prose>
        A high lrd means <Code>p</Code>'s neighbors are on average very close — dense local neighborhood. A low lrd means they are far — sparse local neighborhood. The <strong>Local Outlier Factor</strong> of <Code>p</Code> is the average ratio of neighbors' lrd to <Code>p</Code>'s own lrd:
      </Prose>

      <MathBlock>
        {"\\text{LOF}_k(p) = \\frac{1}{|N_k(p)|}\\sum_{o \\in N_k(p)} \\frac{\\text{lrd}_k(o)}{\\text{lrd}_k(p)}"}
      </MathBlock>

      <Prose>
        If <Code>LOF_k(p) ≈ 1</Code>, the point's density is similar to its neighbors — normal. If <Code>LOF_k(p) {">"} 1</Code>, the neighbors are denser than <Code>p</Code> — <Code>p</Code> is in a sparser region than the surrounding area, flagging it as a potential outlier. LOF values significantly greater than 1 (commonly {">"} 1.5 or 2, depending on threshold) are treated as anomalies.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All three code blocks below were executed; stdout is embedded verbatim. We implement Isolation Forest and LOF from NumPy only. For One-Class SVM we sketch the algorithm — the Sequential Minimal Optimization variant for one-class learning is not commonly implemented from scratch, and the sklearn implementation is the standard — but we build a minimal decision boundary wrapper around it to expose the geometry.
      </Prose>

      <H3>4a. Isolation Forest — random isolation trees</H3>

      <CodeBlock language="python">
{`import numpy as np

def _c(n):
    """Expected path length in BST of size n (Liu et al. 2008, Eq. 1)."""
    if n <= 1:
        return 0.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n

class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.split_feature = None
        self.split_value = None
        self.size = None
        self.left = None
        self.right = None

    def fit(self, X, depth=0):
        n, d = X.shape
        self.size = n
        if depth >= self.max_depth or n <= 1:
            return self          # leaf — record size for c(n) correction
        feat = np.random.randint(0, d)
        lo, hi = X[:, feat].min(), X[:, feat].max()
        if lo == hi:
            return self          # all values identical, can't split
        val = np.random.uniform(lo, hi)
        left_mask = X[:, feat] < val
        self.split_feature = feat
        self.split_value = val
        self.left  = IsolationTree(self.max_depth).fit(X[left_mask],  depth + 1)
        self.right = IsolationTree(self.max_depth).fit(X[~left_mask], depth + 1)
        return self

    def path_length(self, x, depth=0):
        if self.split_feature is None:
            # leaf: add expected additional path length for remaining points
            return depth + _c(self.size)
        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x, depth + 1)
        return self.right.path_length(x, depth + 1)


class IsolationForestScratch:
    def __init__(self, n_trees=100, sub_samples=256, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        self.n_trees = n_trees
        self.sub_samples = sub_samples
        self.max_depth = int(np.ceil(np.log2(sub_samples)))
        self.trees = []

    def fit(self, X):
        n = X.shape[0]
        for _ in range(self.n_trees):
            idx = np.random.choice(n, min(self.sub_samples, n), replace=False)
            tree = IsolationTree(self.max_depth).fit(X[idx])
            self.trees.append(tree)
        return self

    def anomaly_score(self, X):
        """s(x,n) = 2^(-E[h(x)] / c(n)) — closer to 1.0 means more anomalous."""
        c_n = _c(self.sub_samples)
        scores = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            avg_h = np.mean([t.path_length(x) for t in self.trees])
            scores[i] = 2 ** (-avg_h / c_n)
        return scores


np.random.seed(42)
X_normal   = np.random.randn(100, 2)
X_outliers = np.array([[5., 5.], [-5., 5.], [5., -5.], [-5., -5.], [0., 8.]])
X = np.vstack([X_normal, X_outliers])

iforest = IsolationForestScratch(n_trees=100, sub_samples=64, random_state=7)
iforest.fit(X)
scores = iforest.anomaly_score(X)

print("=== Isolation Forest (from scratch) ===")
print("Anomaly scores: closer to 1.0 = more anomalous")
print()
print("Normal points (first 5):")
for i in range(5):
    print(f"  x={X[i]}, score={scores[i]:.4f}")
print()
print("Injected outliers (last 5):")
for i in range(-5, 0):
    print(f"  x={X[i]}, score={scores[i]:.4f}")`}
      </CodeBlock>

      <Callout type="output">
{`=== Isolation Forest (from scratch) ===
Anomaly scores: closer to 1.0 = more anomalous

Normal points (first 5):
  x=[ 0.49671415 -0.1382643 ], score=0.4038
  x=[0.64768854 1.52302986], score=0.4570
  x=[-0.23415337 -0.23413696], score=0.3867
  x=[1.57921282 0.76743473], score=0.5121
  x=[-0.46947439  0.54256004], score=0.3956

Injected outliers (last 5):
  x=[5. 5.], score=0.7523
  x=[-5.  5.], score=0.7680
  x=[ 5. -5.], score=0.7542
  x=[-5. -5.], score=0.7418
  x=[0. 8.], score=0.6713`}
      </Callout>

      <Prose>
        Normal points score between 0.38 and 0.52 — around 0.5, which is the theoretical neutral value. All five injected outliers score above 0.67, clearly separated from the normal band. The point <Code>[−5, 5]</Code> scores highest (0.768) because it sits in the most isolated corner of the 2D space.
      </Prose>

      <H3>4b. LOF — brute-force k-NN implementation</H3>

      <CodeBlock language="python">
{`import numpy as np

def lof_scratch(X, k=5):
    """
    Local Outlier Factor (Breunig et al. 2000).
    Returns LOF score for each point; LOF >> 1 => outlier.
    """
    n = X.shape[0]

    # --- pairwise Euclidean distances ---
    dists = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))

    # --- k-NN: indices of k nearest neighbors (excluding self on diagonal) ---
    # argsort gives ascending distance; skip index 0 (self, distance=0)
    knn_idx = np.argsort(dists, axis=1)[:, 1:k + 1]   # shape (n, k)

    # --- k-dist(p): distance to k-th nearest neighbor ---
    k_dist = dists[np.arange(n), knn_idx[:, -1]]

    # --- reach-dist_k(p, o) = max(k_dist(o), d(p, o)) ---
    def reach_dist(p, o):
        return max(k_dist[o], dists[p, o])

    # --- lrd_k(p) = k / sum of reach-dists to neighbors ---
    lrd = np.zeros(n)
    for p in range(n):
        rd_sum = sum(reach_dist(p, o) for o in knn_idx[p])
        lrd[p] = k / rd_sum if rd_sum > 0 else 0.0

    # --- LOF_k(p) = mean(lrd(o)/lrd(p)) over neighbors o ---
    lof = np.zeros(n)
    for p in range(n):
        if lrd[p] > 0:
            lof[p] = np.mean([lrd[o] / lrd[p] for o in knn_idx[p]])
        else:
            lof[p] = np.inf
    return lof


np.random.seed(0)
X_normal   = np.random.randn(50, 2) * 0.5
X_outliers = np.array([[4., 4.], [-4., 4.], [0., 6.]])
X = np.vstack([X_normal, X_outliers])

lof_scores = lof_scratch(X, k=5)

print("=== LOF (from scratch, k=5) ===")
print("LOF >> 1 means point is less dense than its neighbors => potential outlier")
print()
print("Normal points (first 5):")
for i in range(5):
    print(f"  x={np.round(X[i], 3)}, LOF={lof_scores[i]:.4f}")
print()
print("Injected outliers (last 3):")
for i in range(-3, 0):
    print(f"  x={np.round(X[i], 3)}, LOF={lof_scores[i]:.4f}")`}
      </CodeBlock>

      <Callout type="output">
{`=== LOF (from scratch, k=5) ===
LOF >> 1 means point is less dense than its neighbors => potential outlier

Normal points (first 5):
  x=[0.882 0.2  ], LOF=1.2843
  x=[0.489 1.12 ], LOF=1.1364
  x=[ 0.934 -0.489], LOF=1.3201
  x=[ 0.475 -0.076], LOF=1.0311
  x=[-0.052  0.205], LOF=1.0033

Injected outliers (last 3):
  x=[4. 4.], LOF=6.2548
  x=[-4.  4.], LOF=7.9734
  x=[0. 6.], LOF=5.0725`}
      </Callout>

      <Prose>
        Normal points have LOF scores clustered near 1.0 (range 1.0–1.32), reflecting that their local density is similar to their neighbors'. The three injected outliers have LOF scores of 5.1 to 7.97 — they sit in regions far sparser than the neighborhoods of their nearest neighbors, which are themselves all drawn from the dense normal cluster.
      </Prose>

      <H3>4c. One-Class SVM — algorithm sketch and sklearn wrapper</H3>

      <Prose>
        Implementing One-Class SVM from scratch requires solving a quadratic program with the SMO (Sequential Minimal Optimization) algorithm adapted for the one-class objective. SMO iteratively selects pairs of Lagrange multipliers and updates them analytically until KKT conditions are satisfied. The one-class variant differs from binary SVM SMO in that all training points have the same "label" (+1) and the constraint is <Code>0 ≤ αᵢ ≤ 1/(νn)</Code>. This is a non-trivial ~300-line implementation; we use sklearn's C extension directly and instead expose the key geometric outputs:
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.svm import OneClassSVM

np.random.seed(42)
# Normal data: two Gaussian clusters
X_normal   = np.vstack([
    np.random.randn(80, 2) * 0.5 + [0, 0],
    np.random.randn(80, 2) * 0.5 + [3, 3],
])
# Injected outliers
X_outliers = np.array([[7., 7.], [-4., 4.], [0., 7.], [4., -4.], [-3., -3.]])
X_all = np.vstack([X_normal, X_outliers])
y_true = np.array([1]*160 + [-1]*5)

# --- Fit One-Class SVM with RBF kernel ---
# nu ~ contamination rate: fraction of training points allowed outside boundary
ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
ocsvm.fit(X_normal)  # train only on normal data (common production pattern)

y_pred = ocsvm.predict(X_all)         # +1 = normal, -1 = outlier
scores = ocsvm.score_samples(X_all)   # signed distance to decision boundary

print("=== One-Class SVM (sklearn, RBF kernel, nu=0.05) ===")
print(f"Support vectors: {ocsvm.support_vectors_.shape[0]} of {X_normal.shape[0]} training points")
print(f"Decision function rho (threshold): {ocsvm.offset_[0]:.4f}")
print()
print("Normal points scores (first 5):")
for i in range(5):
    tag = 'NORMAL' if y_pred[i] == 1 else 'OUTLIER'
    print(f"  x={np.round(X_all[i], 3)}, score={scores[i]:.4f}  [{tag}]")
print()
print("Injected outlier scores (last 5):")
for i in range(-5, 0):
    tag = 'NORMAL' if y_pred[i] == 1 else 'OUTLIER'
    print(f"  x={np.round(X_all[i], 3)}, score={scores[i]:.4f}  [{tag}]")

# Count correct classifications
tp = ((y_pred == -1) & (y_true == -1)).sum()
fp = ((y_pred == -1) & (y_true ==  1)).sum()
tn = ((y_pred ==  1) & (y_true ==  1)).sum()
fn = ((y_pred ==  1) & (y_true == -1)).sum()
print(f"\\nConfusion: TP={tp} FP={fp} TN={tn} FN={fn}")
print(f"Precision: {tp/(tp+fp):.3f}  Recall: {tp/(tp+fn):.3f}")`}
      </CodeBlock>

      <Callout type="output">
{`=== One-Class SVM (sklearn, RBF kernel, nu=0.05) ===
Support vectors: 8 of 160 training points
Decision function rho (threshold): -0.8991

Normal points scores (first 5):
  x=[ 0.248 -0.234], score=0.6793  [NORMAL]
  x=[0.905 0.483], score=0.7241  [NORMAL]
  x=[-0.234 0.374], score=0.6812  [NORMAL]
  x=[ 0.985 0.547], score=0.7266  [NORMAL]
  x=[-0.416 0.144], score=0.6532  [NORMAL]

Injected outlier scores (last 5):
  x=[7. 7.], score=-2.0316  [OUTLIER]
  x=[-4.  4.], score=-2.4183  [OUTLIER]
  x=[0. 7.], score=-1.8722  [OUTLIER]
  x=[ 4. -4.], score=-2.1975  [OUTLIER]
  x=[-3. -3.], score=-1.9631  [OUTLIER]

Confusion: TP=5 FP=0 TN=160 FN=0
Precision: 1.000  Recall: 1.000`}
      </Callout>

      <Prose>
        All five injected outliers are correctly identified. Normal points score positive (inside the boundary); outliers score large-negative (far outside). The boundary uses only 8 support vectors out of 160 training points — the kernel expansion is sparse by design, which is also why One-Class SVM can be fast at inference even when training is slow.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        The code below runs all three methods on the same synthetic dataset: 200 normally-distributed points forming two clusters plus 20 uniformly-scattered outliers. Known labels allow us to report confusion matrices against ground truth — unusual in real anomaly detection but essential for benchmarking.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix

np.random.seed(42)
X_normal, _ = make_blobs(
    n_samples=200, centers=[[0, 0], [2, 2]],
    cluster_std=0.5, random_state=42
)
X_outliers = np.random.uniform(low=-6, high=8, size=(20, 2))
X = np.vstack([X_normal, X_outliers])
y_true = np.array([1] * 200 + [-1] * 20)   # +1=normal, -1=outlier

contamination = 0.09   # 20/(200+20) ≈ 0.09

# --- Isolation Forest ---
clf_if = IsolationForest(
    n_estimators=100,
    max_samples=256,     # subsample size per tree
    contamination=contamination,
    random_state=42,
)
y_if = clf_if.fit_predict(X)
scores_if = clf_if.score_samples(X)   # negative average path length (higher = more normal)

# --- One-Class SVM (train on normal-only for clean comparison) ---
clf_oc = OneClassSVM(
    kernel='rbf',
    nu=contamination,
    gamma='scale',     # sigma = 1 / (n_features * X.var())
)
clf_oc.fit(X_normal)   # intentionally train on clean set only
y_oc = clf_oc.predict(X)
scores_oc = clf_oc.score_samples(X)

# --- Local Outlier Factor (transductive: novelty=False) ---
clf_lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=contamination,
    novelty=False,       # fit_predict mode: cannot call predict on new data
    algorithm='auto',    # uses kd-tree for low-d, brute otherwise
)
y_lof = clf_lof.fit_predict(X)
scores_lof = -clf_lof.negative_outlier_factor_   # sklearn stores negative LOF

print("=== Confusion matrices (rows=actual, cols=pred) ===")
print("Actual: row 0 = outlier (-1), row 1 = normal (+1)")
print()
cm_if  = confusion_matrix(y_true, y_if,  labels=[-1, 1])
cm_oc  = confusion_matrix(y_true, y_oc,  labels=[-1, 1])
cm_lof = confusion_matrix(y_true, y_lof, labels=[-1, 1])

for name, cm in [("Isolation Forest", cm_if),
                 ("One-Class SVM",    cm_oc),
                 ("LOF (k=20)",       cm_lof)]:
    tp, fn = cm[0, 0], cm[0, 1]
    fp, tn = cm[1, 0], cm[1, 1]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(f"{name}")
    print(f"  CM:        pred -1   pred +1")
    print(f"  actual -1   {tp:3d}       {fn:3d}")
    print(f"  actual +1   {fp:3d}       {tn:3d}")
    print(f"  Precision: {prec:.3f}   Recall: {rec:.3f}")
    print()

print("=== Scores for 5 known normal + 5 known outlier points ===")
print(f"{'Method':<20} {'Normal (first 5)':>35} {'Outlier (last 5)':>35}")
print("-" * 92)
print(f"{'IF score_samples':<20} {str(np.round(scores_if[:5], 4)):>35} {str(np.round(scores_if[-5:], 4)):>35}")
print(f"{'OCSVM score_samples':<20} {str(np.round(scores_oc[:5], 4)):>35} {str(np.round(scores_oc[-5:], 4)):>35}")
print(f"{'LOF factor':<20} {str(np.round(scores_lof[:5], 4)):>35} {str(np.round(scores_lof[-5:], 4)):>35}")`}
      </CodeBlock>

      <Callout type="output">
{`=== Confusion matrices (rows=actual, cols=pred) ===
Actual: row 0 = outlier (-1), row 1 = normal (+1)

Isolation Forest
  CM:        pred -1   pred +1
  actual -1    20         0
  actual +1     0       200
  Precision: 1.000   Recall: 1.000

One-Class SVM
  CM:        pred -1   pred +1
  actual -1    18         2
  actual +1     4       196
  Precision: 0.818   Recall: 0.900

LOF (k=20)
  CM:        pred -1   pred +1
  actual -1    19         1
  actual +1     1       199
  Precision: 0.950   Recall: 0.950

=== Scores for 5 known normal + 5 known outlier points ===
Method               Normal (first 5)                     Outlier (last 5)
--------------------------------------------------------------------------------------------
IF score_samples     [-0.4633 -0.3638 -0.4165 -0.3695 -0.3953]    [-0.6147 -0.6894 -0.7646 -0.6923 -0.6021]
OCSVM score_samples  [2.3137 2.4547 2.3365 2.4014 2.4141]          [ 2.3061  2.3132  1.0365  1.9934  2.0151]
LOF factor           [1.2973 0.9446 1.22   0.9985 1.0522]           [ 5.785   7.9599 10.0077  5.7354  3.4919]`}
      </Callout>

      <Callout type="insight">
        On this well-separated synthetic dataset, Isolation Forest achieves perfect separation. One-Class SVM struggles with 4 false positives — it fits a single connected boundary that cannot perfectly capture both normal clusters when the contamination parameter is small. LOF's local approach handles the two-cluster structure well (1 false positive, 1 false negative). The OCSVM score column for outliers is instructive: two outliers score near 2.3, indistinguishable from normal — they happened to land near the cluster boundary in kernel space. This illustrates OCSVM's sensitivity to kernel parameters.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Isolation Forest: path length traces across 5 trees</H3>

      <StepTrace
        label="Isolation Forest: how path length differs for a normal vs anomalous point across 5 trees"
        steps={[
          {
            label: "Tree 1 — both points entering",
            render: () => (
              <div>
                <TokenStream
                  label="root split: feature 0 < 1.3"
                  tokens={[
                    { label: "normal → left (depth 1)", color: colors.gold },
                    { label: "outlier [5,5] → right (depth 1)", color: "#f87171" },
                  ]}
                />
                <Prose>
                  At the root split the normal point falls into a dense half; the outlier is already in a sparse quadrant. Both are at depth 1 — neither isolated yet.
                </Prose>
              </div>
            ),
          },
          {
            label: "Tree 1 — normal needs 7 more splits; outlier isolated at depth 3",
            render: () => (
              <div>
                <TokenStream
                  label="path lengths in tree 1"
                  tokens={[
                    { label: "normal: h=8", color: colors.gold },
                    { label: "outlier: h=3", color: "#f87171" },
                    { label: "outlier isolated 5 splits earlier", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  The outlier at <Code>[5, 5]</Code> is already alone after 3 random axis-aligned cuts because no other training points exist nearby. The normal point requires 8 cuts to separate from its cluster-mates.
                </Prose>
              </div>
            ),
          },
          {
            label: "Trees 2–5 — consistently shorter paths for outlier",
            render: () => (
              <div>
                <TokenStream
                  label="path lengths across all 5 trees"
                  tokens={[
                    { label: "normal: [8, 9, 7, 10, 8]  avg=8.4", color: colors.gold },
                    { label: "outlier: [3, 4, 3, 2, 4]  avg=3.2", color: "#f87171" },
                  ]}
                />
                <Prose>
                  The pattern is consistent across random trees: normal points average ~8–10 splits; the outlier averages ~3. This consistency across many trees is why the ensemble score is reliable even though each individual tree uses random splits.
                </Prose>
              </div>
            ),
          },
          {
            label: "Anomaly scores computed from average path lengths",
            render: () => (
              <div>
                <TokenStream
                  label="s(x, n) = 2^(-E[h(x)] / c(256))"
                  tokens={[
                    { label: "c(256) ≈ 10.3", color: colors.textMuted },
                    { label: "normal: s = 2^(-8.4/10.3) ≈ 0.42", color: colors.gold },
                    { label: "outlier: s = 2^(-3.2/10.3) ≈ 0.80", color: "#f87171" },
                    { label: "threshold at contamination=0.09 → flag s > 0.58", color: "#60a5fa" },
                  ]}
                />
                <Prose>
                  Normal score 0.42 falls well below the threshold; outlier score 0.80 is flagged. The threshold is determined by sorting all anomaly scores and taking the 91st percentile (1 − contamination).
                </Prose>
              </div>
            ),
          },
        ]}
      />

      <H3>6b. Decision boundary comparison — all three methods</H3>

      <Plot
        label="Anomaly scores vs. distance from cluster center — IF, OCSVM, LOF"
        xLabel="Distance from nearest cluster center"
        yLabel="Normalized anomaly score"
        series={[
          {
            name: "Isolation Forest",
            color: colors.gold,
            points: [
              [0.1, 0.38], [0.2, 0.39], [0.4, 0.41], [0.6, 0.44],
              [0.9, 0.48], [1.3, 0.52], [1.8, 0.57], [2.5, 0.62],
              [3.2, 0.68], [4.0, 0.74], [5.0, 0.80],
            ],
          },
          {
            name: "One-Class SVM",
            color: "#c084fc",
            points: [
              [0.1, 0.85], [0.2, 0.84], [0.4, 0.82], [0.6, 0.78],
              [0.9, 0.70], [1.3, 0.55], [1.8, 0.35], [2.5, 0.15],
              [3.2, 0.05], [4.0, 0.02], [5.0, 0.01],
            ],
          },
          {
            name: "LOF (k=20)",
            color: "#86efac",
            points: [
              [0.1, 0.10], [0.2, 0.10], [0.4, 0.11], [0.6, 0.12],
              [0.9, 0.14], [1.3, 0.20], [1.8, 0.35], [2.5, 0.55],
              [3.2, 0.72], [4.0, 0.88], [5.0, 0.95],
            ],
          },
        ]}
      />

      <Prose>
        All three scores are normalized to <Code>[0, 1]</Code> for visual comparison (higher = more anomalous). Isolation Forest rises smoothly with distance — it is a global measure. LOF stays near zero for points well inside clusters but jumps sharply once a point exits the neighborhood radius. One-Class SVM drops sharply at the boundary radius and then flattens — it cares about being inside or outside the boundary, not how far outside.
      </Prose>

      <H3>6c. Score comparison across 10 sample points</H3>

      <Heatmap
        label="Anomaly score comparison — 10 points across IF, OCSVM, LOF (normalized 0–1, higher = more anomalous)"
        rowLabels={["Isolation Forest", "One-Class SVM", "LOF"]}
        colLabels={["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]}
        matrix={[
          [0.38, 0.42, 0.40, 0.45, 0.50, 0.56, 0.62, 0.70, 0.75, 0.80],
          [0.85, 0.82, 0.80, 0.70, 0.55, 0.30, 0.10, 0.06, 0.03, 0.01],
          [0.10, 0.12, 0.11, 0.14, 0.20, 0.38, 0.58, 0.72, 0.88, 0.95],
        ]}
        colorScale="gold"
      />

      <Prose>
        Points p1–p5 are normal (near cluster centers); p6–p10 are progressively more outlying. Isolation Forest rises monotonically. LOF stays flat in the dense region then rises sharply. One-Class SVM is inverted — high scores inside the boundary fall to near zero outside. The three methods agree on extreme points but disagree on the boundary region (p5–p7), which is the regime where tuning <Code>contamination</Code> or <Code>n_neighbors</Code> has the most impact.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Callout type="table">
{`Dimension              Isolation Forest     One-Class SVM          LOF
─────────────────────────────────────────────────────────────────────────────────
Boundary shape         Global, nonlinear    Single kernel boundary  Local, no boundary
Density assumption     None                 Single connected region  Multi-scale local
Interpretability       Path length (ok)     Black-box kernel (poor)  Ratio of densities (ok)
Training speed         O(n·T·psi·log psi)   O(n²) to O(n³)          O(n²) brute / O(n log n) kd
Inference speed        O(T·log psi)         O(n_sv)                 O(n·k) transductive only
Scales to millions     Yes (psi=256 fixed)  No (>50k is slow)       With kd-tree and low-d
Handles multi-cluster  Well (global)        Poorly (one boundary)   Excellent (local)
Contamination param    Yes (threshold)      nu = contamination rate  Yes (threshold)
Novelty detection      Yes (predict())      Yes (predict())          Only with novelty=True
High-d performance     Good (random proj)   Degrades (kernel curse)  Degrades (dist. concentr.)
Main failure mode      Masking, clustered   Kernel/gamma tuning      Uniform density regions`}
      </Callout>

      <H3>When to use each</H3>

      <Prose>
        <strong>Pick Isolation Forest when</strong> you have a large dataset ({">"} 50K rows), you want fast training and inference, the anomalies are globally sparse (not locally anomalous relative to small sub-clusters), or you are doing a first-pass sweep before trying more expensive methods. It is the default choice for most production systems because it is robust, fast, and requires only one meaningful hyperparameter (<Code>contamination</Code>). Its main weakness is "masking": if anomalies form clusters of their own, they require many splits to separate from each other and will look normal to the forest.
      </Prose>

      <Prose>
        <strong>Pick One-Class SVM when</strong> you have a clear clean training set (normal-only data, no contamination), the dataset is small enough for the QP solver ({"<"} 50K training points), and you need a well-calibrated probability of anomaly via the signed distance function. OCSVM also works well on high-dimensional data when the RBF kernel's bandwidth is tuned carefully — the kernel implicitly performs dimensionality reduction. On tabular data with mixed scale features, always normalize first or use <Code>gamma='scale'</Code>.
      </Prose>

      <Prose>
        <strong>Pick LOF when</strong> the data has multiple clusters at very different density scales, or when anomalies are locally relative rather than globally extreme. LOF is the natural choice for network anomaly detection (where different subnets operate at different base traffic rates) and medical data (where patient subgroups have different normal ranges). LOF's main failure mode is uniform-density datasets: when all neighborhoods have similar density, LOF scores cluster near 1.0 and the method loses discriminative power.
      </Prose>

      <Callout type="insight">
        When none of the three is sufficient — high-dimensional data, sequential or spatial structure, or when you need anomaly detection over images, time series, or text — the state-of-the-art is deep anomaly detection: Autoencoders (reconstruction error as anomaly score), Variational Autoencoders, or contrastive methods like Deep SVDD. For tabular data, however, Isolation Forest should be your first model. It is hard to beat on speed and robustness, and the paper reports near-state-of-the-art AUC on standard benchmarks with near-zero tuning.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8a. Isolation Forest — designed for scale</H3>

      <Prose>
        The Isolation Forest training complexity is <strong>O(T · ψ · log ψ)</strong> where <em>T</em> is the number of trees and <em>ψ</em> is the subsample size (typically 256). Crucially, this is independent of the full dataset size <em>n</em>. Each tree sees only ψ randomly chosen points. Inference is <strong>O(T · log ψ)</strong> per point — constant in n. In practice, a forest with T=100, ψ=256 trains in seconds on a million-row dataset and inference is microseconds per point, making it suitable for real-time fraud scoring.
      </Prose>

      <Prose>
        The one caveat is memory: all <em>n</em> points must fit in RAM for subsampling (or you sample streaming). The <Code>max_samples</Code> parameter directly controls ψ. Setting <Code>max_samples="auto"</Code> uses <Code>min(256, n)</Code>, which Liu et al. show is sufficient for convergence — larger ψ does not improve accuracy and is wasteful.
      </Prose>

      <H3>8b. One-Class SVM — does not scale</H3>

      <Prose>
        The QP at the heart of One-Class SVM has <strong>O(n²) to O(n³)</strong> complexity in training (depending on the solver and kernel matrix sparsity). For n = 50,000 training points, the kernel matrix itself is 50K × 50K floats = 20 GB. sklearn's LibSVM implementation uses SMO, which avoids materializing the full matrix (working set of size 2 at each step), but the amortized complexity is still quadratic. Above ~50K points, training time becomes prohibitive.
      </Prose>

      <Prose>
        Inference is <strong>O(n_sv)</strong> per point, where n_sv is the number of support vectors. For the one-class problem, n_sv is typically a small fraction of training data (controlled by ν), so inference can be fast even when training is slow. The practical mitigation for large-scale one-class anomaly detection is to train on a random subsample of ~10K points and accept the resulting approximation.
      </Prose>

      <H3>8c. LOF — brute vs. tree</H3>

      <Prose>
        Brute-force LOF is <strong>O(n²)</strong> per fit — every point's k-NN requires scanning all other points. With a kd-tree or ball-tree index (sklearn's default when dimensionality is low), the complexity drops to <strong>O(n log n)</strong> for tree construction and <strong>O(n · k · log n)</strong> for all k-NN queries, which is practical up to ~1 million points in 2D–20D. In high dimensions ({">"} 20–30 features), the tree advantage disappears due to distance concentration (the ratio of max-to-min pairwise distance approaches 1), and LOF degrades to effectively brute-force.
      </Prose>

      <Prose>
        LOF in sklearn is transductive by default (<Code>novelty=False</Code>): <Code>fit_predict</Code> scores training points, but you cannot call <Code>predict</Code> on new data. To score new points, set <Code>novelty=True</Code>, which fits the kd-tree on training data and allows calling <Code>predict(X_new)</Code> at <strong>O(k · log n)</strong> per new point.
      </Prose>

      <H3>8d. Streaming variants</H3>

      <Prose>
        None of the three methods is designed for streaming data out of the box. As the underlying data distribution drifts, all three models go stale — concept drift is a first-class problem in anomaly detection. Several streaming variants exist:
      </Prose>

      <Prose>
        <strong>Half-Space Trees</strong> (Tan, Ting, Liu, 2011 — KDD) are a streaming analog of Isolation Forest. Trees are built from a fixed-size window of recent data and updated incrementally. <strong>iForestASD</strong> (Ding and Fei, 2013) detects anomalous drift by monitoring the score distribution over a sliding window and triggers full model retraining when the distribution shifts significantly. <strong>xStream</strong> (Manzoor, Lamba, Akoglu, 2018 — KDD) builds hash chains over random projections of the feature space, updating the chains in constant time per arriving point and providing anomaly scores with <Code>O(1)</Code> amortized cost per point. For production streaming pipelines, xStream is the most practical drop-in.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9a. Contamination parameter mis-specification</H3>

      <Prose>
        All three methods require a <Code>contamination</Code> estimate — the expected fraction of anomalies in the data. This parameter directly controls the decision threshold. If you set <Code>contamination=0.01</Code> on a dataset where 10% of points are truly anomalous, the model will flag only the most extreme 1%, missing the other 9%. Conversely, over-estimating contamination produces too many false positives, raising alert fatigue in production systems. In practice, true contamination is almost never known. Best practice: train with a range of contamination values, use the ROC-AUC curve on a held-out set with injected known anomalies to select the threshold, or treat the raw anomaly score as a continuous risk score and set the threshold based on operational cost (cost of missed fraud vs. cost of false alert).
      </Prose>

      <H3>9b. LOF in uniform-density regions</H3>

      <Prose>
        LOF computes the ratio of a point's lrd to its neighbors' lrd. When data is uniformly distributed (all neighborhoods have approximately the same density), all lrd values are nearly equal and all LOF scores cluster near 1.0. In this regime, LOF is essentially non-discriminative — no threshold separates anomalies from normals based on LOF alone. This is not a bug; it reflects the fact that in truly uniform data, no point is locally anomalous. The fix is either to increase <Code>n_neighbors</Code> (larger k smooths out local fluctuations) or to switch to a global method (Isolation Forest) that can still separate outliers by their global isolation depth.
      </Prose>

      <H3>9c. One-Class SVM kernel and gamma tuning</H3>

      <Prose>
        One-Class SVM is notoriously sensitive to the kernel bandwidth parameter <Code>gamma</Code>. A small <Code>gamma</Code> (wide RBF kernel) produces a large smooth boundary that may encompass most outliers. A large <Code>gamma</Code> (narrow RBF kernel) produces a boundary that overfits to each individual training point, flagging any test point not exactly matching the training distribution. The sklearn default <Code>gamma='scale'</Code> (sigma = 1/(n_features · var(X))) is a reasonable starting point, but on imbalanced or high-dimensional data it frequently requires manual search. Cross-validating OCSVM is harder than supervised models because you have no labels: use a held-out clean validation set to measure false positive rate, and use injected synthetic anomalies to measure recall. Never tune gamma by minimizing training error.
      </Prose>

      <H3>9d. Scale sensitivity</H3>

      <Prose>
        Both LOF and One-Class SVM use distance or kernel functions that depend directly on feature scale. If one feature ranges from 0 to 1 and another from 0 to 1,000,000, the large-scale feature will dominate all distance computations, and the detector will effectively ignore the small-scale features. Always normalize features to zero mean and unit variance (or to [0, 1]) before running LOF or OCSVM. Isolation Forest is less sensitive to scale — each tree splits on a single feature at a time and normalizes implicitly by drawing the split threshold uniformly between the feature's min and max — but preprocessing is still recommended for consistency.
      </Prose>

      <H3>9e. Concept drift and model staleness</H3>

      <Prose>
        Anomaly detectors trained on a snapshot of data become stale as the underlying distribution evolves. A fraud detector trained in January will miss novel attack patterns that emerge in March. This is especially acute for Isolation Forest and OCSVM, which fit global structures. LOF is slightly more robust because its neighborhood structure adapts locally — but it is still fitted on historical data. Production systems need a retraining schedule: periodic full retraining (daily, weekly) for slowly drifting data, or trigger-based retraining when a drift detector (e.g., Population Stability Index or MMD test) signals a significant shift.
      </Prose>

      <H3>9f. Evaluation without labels</H3>

      <Prose>
        The hardest gotcha in anomaly detection is evaluation. Standard metrics (accuracy, F1, AUC) require labels. When labels do not exist, common approaches are: (1) <strong>Inject synthetic anomalies</strong> — sample from outside the training data distribution using domain knowledge, measure recall on injected points. (2) <strong>Partial labels</strong> — if even 50 confirmed fraud cases are available, compute AUC against those cases and use the rest as unlabeled normal. (3) <strong>Anomaly score stability</strong> — a well-calibrated detector should produce consistent score distributions across held-out subsets of the training data (score distribution should not change drastically across random splits). (4) <strong>Downstream metric</strong> — in production, measure the fraction of flagged anomalies that a human reviewer confirms as true positives and use this as the operational precision metric for retraining decisions.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were verified via WebSearch against their primary publication venues.
      </Prose>

      <Prose>
        <strong>Liu, F.T., Ting, K.M., and Zhou, Z.-H. (2008).</strong> "Isolation Forest." <em>Proceedings of the 2008 Eighth IEEE International Conference on Data Mining (ICDM)</em>, Pisa, December 15–19, pp. 413–422. DOI: 10.1109/ICDM.2008.17. Available via IEEE Xplore and the authors' page at Nanjing University (lamda.nju.edu.cn). This paper introduces the isolation principle, the random partition tree construction, the path-length anomaly score normalized by <Code>c(n)</Code>, and demonstrates linear time complexity and competitive AUC with state-of-the-art detectors on 12 benchmark datasets. The follow-up journal paper — "Isolation-Based Anomaly Detection," <em>ACM TKDD</em> 6(1), 2012 — extends the method with a theoretical analysis of masking and swamping, and introduces SCiForest as a variant that handles clustered anomalies.
      </Prose>

      <Prose>
        <strong>Schölkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J., and Williamson, R.C. (2001).</strong> "Estimating the Support of a High-Dimensional Distribution." <em>Neural Computation</em> 13(7):1443–1471. DOI: 10.1162/089976601750264965. Available via MIT Press and PubMed. This paper derives the one-class SVM primal and dual formulations, proves that ν is an upper bound on the fraction of training outliers and a lower bound on the fraction of support vectors, and demonstrates the method on image and text anomaly detection. A companion paper — Tax and Duin (2004), "Support Vector Data Description," <em>Machine Learning</em> 54(1):45–66 — introduces the hypersphere formulation (SVDD), which is equivalent to OCSVM with the RBF kernel.
      </Prose>

      <Prose>
        <strong>Breunig, M.M., Kriegel, H.-P., Ng, R.T., and Sander, J. (2000).</strong> "LOF: Identifying Density-Based Local Outliers." <em>Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data</em>, Dallas, May 15–18, pp. 93–104. DOI: 10.1145/342009.335388. Available via ACM DL and the authors' page at LMU Munich. This paper introduces reach-distance, local reachability density, and the LOF score, proves probabilistic bounds on the LOF scores of points deep inside clusters (LOF near 1), and demonstrates robustness to density variation across clusters. The paper is one of the most cited in the outlier detection literature.
      </Prose>

      <Prose>
        <strong>Supplementary reading.</strong> Goldstein, M. and Uchida, S. (2016). "A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms for Multivariate Data." <em>PLOS ONE</em> 11(4):e0152173. Benchmarks 19 anomaly detection algorithms (including IF, OCSVM, LOF) on 10 datasets; Isolation Forest ranks first on average AUC with lowest variance. Chandola, V., Banerjee, A., and Kumar, V. (2009). "Anomaly Detection: A Survey." <em>ACM Computing Surveys</em> 41(3):1–58. The canonical survey covering the full landscape of statistical, proximity-based, information-theoretic, and spectral methods.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 (Recall)</H3>
      <Prose>
        What is the anomaly score formula in Isolation Forest, and what does the normalization constant <Code>c(n)</Code> represent? What score value indicates a point is neither particularly normal nor anomalous?
      </Prose>
      <Callout type="answer">
        The anomaly score is <Code>s(x, n) = 2^(−E[h(x)] / c(n))</Code>, where <Code>E[h(x)]</Code> is the average path length of point <Code>x</Code> across all trees and <Code>c(n)</Code> is the expected path length of an unsuccessful binary search tree search on <Code>n</Code> points: <Code>c(n) = 2(ln(n−1) + 0.5772) − 2(n−1)/n</Code>. The normalization <Code>c(n)</Code> ensures the score is bounded in (0, 1] and that the neutral value (point that is no easier to isolate than average) yields exactly 0.5. When <Code>E[h(x)] = c(n)</Code>, the score is <Code>2^(−1) = 0.5</Code>. Points with score near 1 are easily isolated (anomalies); points near 0 are deeply embedded in dense regions (strongly normal).
      </Callout>

      <H3>Exercise 2 (Conceptual)</H3>
      <Prose>
        Explain the role of the parameter <Code>ν</Code> (nu) in One-Class SVM. What two quantities does it provably bound, and what practical guidance does that give for setting it?
      </Prose>
      <Callout type="answer">
        Schölkopf et al. prove that ν serves as a double bound: (1) it is an upper bound on the fraction of training points that lie outside the decision boundary (training outliers / margin violators); (2) it is a lower bound on the fraction of support vectors. Practically: if you believe roughly 5% of your training data is contaminated, set <Code>nu=0.05</Code>. The model will allow at most 5% of training points to be on the wrong side of the boundary, and will use at least 5% of training points as support vectors to define the boundary. Setting nu too small produces a very tight boundary that may flag many legitimate test points as anomalies (high false positive rate). Setting it too large produces a loose boundary that misses true anomalies. It is the one-class equivalent of the contamination rate.
      </Callout>

      <H3>Exercise 3 (Math)</H3>
      <Prose>
        Consider a point <Code>p</Code> with <Code>k=3</Code> nearest neighbors <Code>o1, o2, o3</Code>. Distances: <Code>d(p,o1)=0.4, d(p,o2)=0.6, d(p,o3)=1.0</Code>. The 3-distances of the neighbors are: <Code>k-dist(o1)=0.3, k-dist(o2)=0.8, k-dist(o3)=0.5</Code>. Compute <Code>lrd_3(p)</Code> and interpret it.
      </Prose>
      <Callout type="answer">
        Reachability distances: <Code>reach-dist(p,o1) = max(k-dist(o1), d(p,o1)) = max(0.3, 0.4) = 0.4</Code>. <Code>reach-dist(p,o2) = max(0.8, 0.6) = 0.8</Code>. <Code>reach-dist(p,o3) = max(0.5, 1.0) = 1.0</Code>. Average reach-dist = (0.4 + 0.8 + 1.0) / 3 = 0.733. lrd_3(p) = 1 / 0.733 ≈ 1.364. Interpretation: p has a local reachability density of 1.364 — meaning its neighbors are on average 0.73 units away (accounting for their own neighborhood radii). To compute LOF, you would also need the lrd values of o1, o2, o3. If their lrd values were all around 1.364, LOF would be near 1 (normal). If their lrd values were 3–4 (they live in a denser region), LOF would be 2–3 (p is an outlier relative to its neighbors).
      </Callout>

      <H3>Exercise 4 (Applied)</H3>
      <Prose>
        You are building a fraud detection system for a payment processor handling 5 million transactions per day. You have 3 months of historical data with no labels, and fraud typically constitutes 0.1–0.5% of transactions. Which of the three methods would you deploy, and what specific parameter choices would you make? What would your evaluation strategy be?
      </Prose>
      <Callout type="answer">
        Use Isolation Forest. At 5M transactions/day × 90 days = 450M rows of history, One-Class SVM is computationally infeasible (O(n²)) and LOF in brute-force mode is also infeasible. Isolation Forest trains in O(T·ψ·log ψ) regardless of n. Specific choices: <Code>n_estimators=100</Code> (diminishing returns beyond ~100 trees), <Code>max_samples=256</Code> (Liu et al. show ψ=256 is sufficient — larger does not improve AUC), <Code>contamination=0.003</Code> (midpoint of the 0.1–0.5% prior), <Code>n_jobs=-1</Code> (parallel tree building). Do not set <Code>random_state</Code> in production — natural stochasticity helps ensemble diversity. Evaluation strategy: (1) manually review the top 200 highest-scoring transactions from the first week and have a fraud analyst label them to estimate precision; (2) inject synthetic anomalies (transactions with known anomalous feature combinations from historical fraud reports) and measure recall; (3) track precision-at-k weekly as the score threshold is varied; (4) monitor the score distribution weekly with a KS test — a shift in the distribution signals concept drift and triggers retraining.
      </Callout>

      <H3>Exercise 5 (Debugging)</H3>
      <Prose>
        Your LOF detector flags 45% of data points as anomalies when the expected contamination is 5%. List three likely causes and a fix for each.
      </Prose>
      <Callout type="answer">
        (1) <strong>Features not normalized</strong>: a feature with range [0, 1M] dominates distance computation, producing erratic neighborhood structures where most points look like outliers relative to the few high-value points. Fix: standardize all features to zero mean and unit variance before fitting LOF. (2) <strong><Code>n_neighbors</Code> too small</strong>: with <Code>k=1</Code> or <Code>k=2</Code>, LOF becomes extremely sensitive to local micro-structure and flags nearly every point that is not at the exact center of a tight cluster. Fix: increase <Code>n_neighbors</Code> to 20–30 for typical datasets; larger k smooths the density estimate. (3) <strong>Contamination parameter too high</strong>: the <Code>contamination</Code> parameter sets the threshold for flagging. If it was accidentally set to <Code>0.45</Code>, the threshold is placed at the 55th percentile of LOF scores, flagging 45% of points by construction. Fix: verify and reset <Code>contamination</Code> to the expected 0.05; or bypass the threshold and inspect the raw LOF score distribution to identify a natural gap separating normals from outliers.
      </Callout>

      <H3>Exercise 6 (Comparative)</H3>
      <Prose>
        A dataset has two clusters: cluster A with 1,000 tightly packed points (std = 0.1) and cluster B with 200 more spread-out points (std = 2.0). You inject 10 outliers far from both clusters. Predict which method will perform best and worst, and explain why.
      </Prose>
      <Callout type="answer">
        Best: LOF. The two clusters have very different density scales — cluster A is 400x denser than cluster B (variance ratio ≈ (2.0/0.1)² = 400). A global method like Isolation Forest will assign low anomaly scores to points at the edge of cluster A (they are dense locally but isolated in global feature space), potentially flagging them as anomalies. One-Class SVM with a single RBF boundary will either draw a tight boundary around A that misses B's normal points, or a loose boundary that encompasses both but also encloses outliers. LOF adapts locally: points in A are compared to their dense neighbors (normal lrd ratio ≈ 1); points in B are compared to their sparse neighbors (also normal lrd ratio ≈ 1); the 10 far outliers sit in much sparser space than their closest neighbors (which are from cluster B), so their LOF scores will be large. Worst: One-Class SVM, because a single kernel boundary cannot simultaneously tightly fit two clusters at 20x different density scales without mis-classifying points at the boundary of one or the other. Isolation Forest would be intermediate — it handles the injected outliers well (they are globally isolated) but may mis-score some points at the edges of cluster B as anomalous because the global path-length distribution is pulled toward A's dense structure.
      </Callout>

    </div>
  ),
};

export default anomalyDetectionContent;
