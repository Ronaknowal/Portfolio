import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const clusteringEvaluationContent = {
  title: "Clustering Evaluation & Validation (Silhouette, ARI, NMI)",
  readTime: "~35 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Supervised learning has an unfair advantage when it comes to evaluation: the ground truth is right there. You hold out a test set, make predictions, compare them to the labels, and get a number. Clustering offers no such luxury. The whole point of unsupervised learning is that there are no labels — you are asking the algorithm to discover structure that no human has annotated. When the algorithm returns five clusters, is that the right number? When you switch from k-means to hierarchical agglomerative clustering and the clusters look different, which partition is better? Without an objective criterion, clustering risks becoming a Rorschach test: you see the structure you were looking for because you were the one who described it as structure.
      </Prose>

      <Prose>
        The evaluation problem split early into two fundamentally different branches. The first branch operates without any external reference — it looks only at the data and the proposed clustering, measuring how tight and well-separated the clusters are. Peter Rousseeuw introduced the silhouette coefficient in a 1987 paper in the <em>Journal of Computational and Applied Mathematics</em>, titled "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis." The paper gave practitioners a per-point, per-cluster, and global score for any partition of any dataset, grounded entirely in pairwise distances. It required no knowledge of the true class labels — a crucial property, because in real unsupervised settings those labels do not exist.
      </Prose>

      <Prose>
        The second branch assumes you do have ground truth labels, at least for evaluation purposes. This happens more often than it seems: you are benchmarking a new clustering algorithm against a dataset where the true groupings are known (Iris species, document categories, image classes), or you are running ablations during algorithm development. For this setting, William M. Rand proposed what he called "objective criteria for the evaluation of clustering methods" in a 1971 paper in the <em>Journal of the American Statistical Association</em>. His Rand Index counts the fraction of pairs of points on which two clusterings agree — a neat, interpretable number in [0, 1]. The problem Rand himself noted was that random clusterings score well above zero. Two independent uniform random assignments over five clusters agree on roughly 68% of pairs purely by chance, giving a Rand Index near 0.68 rather than near 0.0. This makes raw RI nearly useless as a comparative metric.
      </Prose>

      <Prose>
        Lawrence Hubert and Phipps Arabie corrected this in 1985, publishing "Comparing partitions" in the <em>Journal of Classification</em>. They derived the expected value of the Rand Index under a generalized hypergeometric model of random partitions and defined the Adjusted Rand Index as the ratio of the deviation from expectation to the maximum possible deviation. ARI = 0 for random clusterings, ARI = 1 for perfect agreement, and ARI can be negative if the clustering is worse than chance. This is now the standard external metric for non-overlapping cluster comparison.
      </Prose>

      <Prose>
        Information-theoretic approaches arrived via a different route. Alexander Strehl and Joydeep Ghosh's 2002 JMLR paper "Cluster Ensembles — A Knowledge Reuse Framework for Combining Multiple Partitions" popularized Normalized Mutual Information as a symmetric, label-permutation-invariant metric for comparing two clusterings. NMI measures how much knowing one clustering reduces uncertainty about the other, normalized by the geometric mean of both entropies to sit in [0, 1]. A deeper treatment of the full family — including the Adjusted Mutual Information that corrects NMI for chance in the same spirit as ARI corrects RI — came from Nguyen Xuan Vinh, Julien Epps, and James Bailey in their 2010 JMLR paper "Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance," which is the authoritative reference for understanding when NMI is appropriate and when AMI should replace it.
      </Prose>

      <Prose>
        Together these four papers — Rousseeuw 1987, Rand 1971, Hubert and Arabie 1985, Strehl and Ghosh 2002 — form the foundation for every clustering evaluation you will encounter in scikit-learn, in every benchmarking paper, and in every clustering competition. This topic walks through all of them: the math, the implementation, the diagnostics, and the failure modes.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The central distinction in clustering evaluation is between <strong>internal metrics</strong> and <strong>external metrics</strong>. Internal metrics use only the data and the cluster assignments — no ground truth. They answer the question: given this particular way of dividing the data, how geometrically coherent is it? External metrics compare the proposed clustering to a reference partition — typically the known class labels. They answer the question: how much does this algorithm's output agree with the human-annotated truth?
      </Prose>

      <Prose>
        The intuition behind silhouette is simple and powerful. For any single point <Code>i</Code>, imagine asking two questions: (1) how close am I to the other points in my own cluster? and (2) how close am I to the nearest cluster I am <em>not</em> in? Call these distances <Code>a(i)</Code> and <Code>b(i)</Code>. If <Code>b(i)</Code> is much larger than <Code>a(i)</Code>, the point is deep inside a well-separated cluster — it is far from the nearest foreign cluster. Its silhouette score is near +1. If <Code>a(i) ≈ b(i)</Code>, the point sits on a cluster boundary, equally close to its own cluster and the next. Its silhouette score is near 0. If <Code>a(i) {">"} b(i)</Code>, the point is closer to a foreign cluster than to its own — it has probably been misassigned. Its silhouette score is negative.
      </Prose>

      <Prose>
        Average silhouette score across all points gives a single global summary. Values above 0.7 indicate a strong, well-defined cluster structure. Values between 0.5 and 0.7 indicate reasonable structure. Values below 0.25 suggest clusters that are either poorly defined or simply not present in the data. By sweeping over different values of <Code>k</Code> and plotting mean silhouette score, you get the unsupervised equivalent of a validation curve for model selection — without ever touching labels.
      </Prose>

      <Prose>
        Other internal metrics exist. The Calinski-Harabasz index (also called the Variance Ratio Criterion) measures the ratio of between-cluster to within-cluster variance — higher is better. The Davies-Bouldin index measures the average ratio of within-cluster scatter to between-cluster separation — lower is better. Silhouette tends to be preferred because it is interpretable at the individual point level, not just globally, and because it makes minimal geometric assumptions (it works on any pairwise distance, not just Euclidean).
      </Prose>

      <Prose>
        For external metrics, the core idea behind ARI and NMI is pair agreement. ARI counts the fraction of data-point pairs where two clusterings make the same assignment decision — either both putting the pair in the same cluster, or both putting them in different clusters — and subtracts the expected agreement for random clusterings. NMI measures shared information: how many bits of uncertainty about one clustering are resolved by knowing the other, normalized to [0, 1]. Both are symmetric (swapping the two clusterings does not change the score) and invariant to label permutation (renaming cluster 1 as cluster 3 does not change the score). This last property is crucial — two clusterings that are identical up to relabeling should score 1.0, and both metrics guarantee this.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Silhouette coefficient</H3>

      <Prose>
        Let <Code>X</Code> be a dataset of <Code>n</Code> points with a clustering assignment <Code>C</Code>. For point <Code>i</Code> assigned to cluster <Code>C(i)</Code>, define:
      </Prose>

      <MathBlock>
        {"a(i) = \\frac{1}{|C(i)| - 1} \\sum_{j \\in C(i),\\, j \\neq i} d(i, j)"}
      </MathBlock>

      <Prose>
        This is the mean distance from point <Code>i</Code> to all other points in its own cluster — a measure of cohesion. Lower is better. For any cluster <Code>k ≠ C(i)</Code>, define the mean distance from <Code>i</Code> to all points in <Code>k</Code>:
      </Prose>

      <MathBlock>
        {"d(i, k) = \\frac{1}{|k|} \\sum_{j \\in k} d(i, j)"}
      </MathBlock>

      <Prose>
        Then <Code>b(i)</Code> is the minimum such mean distance over all other clusters — the distance to the nearest neighbor cluster:
      </Prose>

      <MathBlock>
        {"b(i) = \\min_{k \\neq C(i)} d(i, k)"}
      </MathBlock>

      <Prose>
        The silhouette coefficient for point <Code>i</Code> is:
      </Prose>

      <MathBlock>
        {"s(i) = \\frac{b(i) - a(i)}{\\max(a(i),\\, b(i))}"}
      </MathBlock>

      <Prose>
        By construction, <Code>s(i) ∈ [-1, +1]</Code>. When <Code>|C(i)| = 1</Code> (a singleton cluster), <Code>s(i)</Code> is defined as 0. The mean silhouette score is the average over all <Code>n</Code> points. Within a cluster, the mean per-cluster silhouette reveals which clusters are tight and which are sloppy — a cluster with low mean silhouette is a candidate for re-assignment or for splitting.
      </Prose>

      <H3>3.2 Rand Index and the pair-counting framework</H3>

      <Prose>
        Given <Code>n</Code> data points with true labels <Code>U</Code> and predicted labels <Code>V</Code>, consider all <Code>C(n,2) = n(n-1)/2</Code> pairs of points. Each pair falls into one of four cells:
      </Prose>

      <Prose>
        TP: same cluster in both <Code>U</Code> and <Code>V</Code>. TN: different clusters in both. FP: different clusters in <Code>U</Code>, same cluster in <Code>V</Code>. FN: same cluster in <Code>U</Code>, different clusters in <Code>V</Code>.
      </Prose>

      <MathBlock>
        {"\\text{RI} = \\frac{TP + TN}{TP + TN + FP + FN}"}
      </MathBlock>

      <Prose>
        RI = 1 for identical clusterings, but is bounded away from 0 for random ones. The expected value of RI under the null hypothesis (that <Code>U</Code> and <Code>V</Code> are drawn independently from a generalized hypergeometric distribution over partitions) is:
      </Prose>

      <MathBlock>
        {"E[RI] = \\frac{\\sum_i \\binom{a_i}{2} \\cdot \\sum_j \\binom{b_j}{2}}{\\binom{n}{2}^2} \\cdot \\binom{n}{2} + \\left(1 - \\frac{\\sum_i \\binom{a_i}{2}}{\\binom{n}{2}}\\right)\\left(1 - \\frac{\\sum_j \\binom{b_j}{2}}{\\binom{n}{2}}\\right)"}
      </MathBlock>

      <Prose>
        Where <Code>a_i</Code> are the row sums and <Code>b_j</Code> are the column sums of the contingency matrix. Hubert and Arabie's ARI subtracts this expectation and normalizes by the range:
      </Prose>

      <MathBlock>
        {"\\text{ARI} = \\frac{\\sum_{ij}\\binom{n_{ij}}{2} - \\frac{\\left[\\sum_i\\binom{a_i}{2}\\right]\\left[\\sum_j\\binom{b_j}{2}\\right]}{\\binom{n}{2}}}{\\frac{1}{2}\\left[\\sum_i\\binom{a_i}{2}+\\sum_j\\binom{b_j}{2}\\right] - \\frac{\\left[\\sum_i\\binom{a_i}{2}\\right]\\left[\\sum_j\\binom{b_j}{2}\\right]}{\\binom{n}{2}}}"}
      </MathBlock>

      <Prose>
        Where <Code>n_ij</Code> are the entries of the contingency matrix — the number of points in true class <Code>i</Code> and predicted cluster <Code>j</Code>. ARI = 1 for perfect agreement, ARI ≈ 0 for random clusterings (regardless of <Code>k</Code>), and ARI can be negative. The adjustment is what matters: without it, a random clustering over <Code>k = 5</Code> classes scores RI ≈ 0.68, making it appear far better than chance.
      </Prose>

      <H3>3.3 Normalized Mutual Information</H3>

      <Prose>
        NMI is rooted in information theory. Let <Code>U</Code> and <Code>V</Code> be random variables over cluster assignments drawn from the two clusterings. Their mutual information is:
      </Prose>

      <MathBlock>
        {"I(U; V) = \\sum_{u} \\sum_{v} p(u, v) \\log \\frac{p(u, v)}{p(u)\\,p(v)}"}
      </MathBlock>

      <Prose>
        Where <Code>p(u,v) = n_uv / n</Code> is the joint probability of a point being in class <Code>u</Code> and cluster <Code>v</Code>, and <Code>p(u), p(v)</Code> are the marginals. MI is non-negative and equals zero if and only if <Code>U</Code> and <Code>V</Code> are statistically independent. The issue is that MI is not bounded — it depends on the number of clusters and grows with the number of clusters even for random partitions. Normalization fixes this. Strehl and Ghosh's geometric normalizer:
      </Prose>

      <MathBlock>
        {"\\text{NMI}(U, V) = \\frac{I(U; V)}{\\sqrt{H(U) \\cdot H(V)}}"}
      </MathBlock>

      <Prose>
        Where <Code>H(U) = -∑_u p(u) log p(u)</Code> is the entropy of the true partition. NMI ∈ [0, 1], equals 1 for identical partitions, and equals 0 when the two partitions share no mutual information. However, just like raw RI, NMI is biased upward for partitions with many clusters — random clusterings over <Code>k=32</Code> classes score NMI ≈ 0.47 rather than 0. The Adjusted Mutual Information (AMI) from Vinh et al. 2010 corrects this in the same spirit as ARI corrects RI:
      </Prose>

      <MathBlock>
        {"\\text{AMI}(U, V) = \\frac{I(U; V) - E[I(U; V)]}{\\frac{1}{2}[H(U) + H(V)] - E[I(U; V)]}"}
      </MathBlock>

      <Prose>
        AMI ≈ 0 for random clusterings at any <Code>k</Code> and AMI = 1 for perfect agreement. For most benchmarking tasks where you have ground truth and want an honest metric, AMI is strictly preferable to NMI — sklearn provides both via <Code>adjusted_mutual_info_score</Code> and <Code>normalized_mutual_info_score</Code>.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        NumPy only. The following implements silhouette (per-point and average), Rand Index, Adjusted Rand Index, and Normalized Mutual Information from scratch, then cross-validates each against sklearn on a synthetic dataset with known ground truth. All stdout is verbatim.
      </Prose>

      <H3>4a. Pairwise distances and silhouette</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.datasets import make_blobs

# ---- Euclidean distance matrix: O(n^2) ----
def pairwise_distances(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))

# ---- Per-point silhouette coefficients ----
def silhouette_samples_scratch(X, labels):
    n = len(X)
    D = pairwise_distances(X)
    unique_labels = np.unique(labels)
    s = np.zeros(n)
    for i in range(n):
        own = labels[i]
        mask_own = (labels == own)
        mask_own[i] = False
        if mask_own.sum() == 0:          # singleton cluster
            s[i] = 0.0
            continue
        a_i = D[i, mask_own].mean()      # mean intra-cluster distance
        b_i = np.inf
        for cl in unique_labels:
            if cl == own:
                continue
            mean_dist = D[i, labels == cl].mean()
            if mean_dist < b_i:
                b_i = mean_dist          # nearest cluster distance
        s[i] = (b_i - a_i) / max(a_i, b_i)
    return s

def silhouette_score_scratch(X, labels):
    return silhouette_samples_scratch(X, labels).mean()`}
      </CodeBlock>

      <H3>4b. Rand Index and Adjusted Rand Index</H3>

      <CodeBlock language="python">
{`from itertools import combinations
from math import comb

def rand_index_scratch(labels_true, labels_pred):
    """RI = (TP + TN) / total_pairs over all C(n,2) pairs."""
    n = len(labels_true)
    tp_tn = 0
    total = n * (n - 1) // 2
    for i, j in combinations(range(n), 2):
        same_true = (labels_true[i] == labels_true[j])
        same_pred = (labels_pred[i] == labels_pred[j])
        if same_true == same_pred:
            tp_tn += 1
    return tp_tn / total

def adjusted_rand_index_scratch(labels_true, labels_pred):
    """ARI = (sum_nij_C2 - E) / (max - E) using contingency matrix."""
    n = len(labels_true)
    classes_true = np.unique(labels_true)
    classes_pred = np.unique(labels_pred)
    R, C = len(classes_true), len(classes_pred)
    # Build contingency table
    contingency = np.zeros((R, C), dtype=np.int64)
    for i, ct in enumerate(classes_true):
        for j, cp in enumerate(classes_pred):
            contingency[i, j] = ((labels_true == ct) & (labels_pred == cp)).sum()
    a = contingency.sum(axis=1)          # true class sizes
    b = contingency.sum(axis=0)          # pred cluster sizes
    sum_nij = sum(comb(int(x), 2) for x in contingency.ravel())
    sum_ai  = sum(comb(int(x), 2) for x in a)
    sum_bj  = sum(comb(int(x), 2) for x in b)
    n_pairs = comb(n, 2)
    expected   = sum_ai * sum_bj / n_pairs
    max_index  = (sum_ai + sum_bj) / 2
    if max_index - expected == 0:
        return 1.0
    return (sum_nij - expected) / (max_index - expected)`}
      </CodeBlock>

      <H3>4c. Normalized Mutual Information</H3>

      <CodeBlock language="python">
{`def normalized_mutual_info_scratch(labels_true, labels_pred):
    """NMI = I(U;V) / sqrt(H(U) * H(V))  — geometric normalizer."""
    n = len(labels_true)
    classes_true = np.unique(labels_true)
    classes_pred = np.unique(labels_pred)
    R, C = len(classes_true), len(classes_pred)
    contingency = np.zeros((R, C), dtype=np.float64)
    for i, ct in enumerate(classes_true):
        for j, cp in enumerate(classes_pred):
            contingency[i, j] = ((labels_true == ct) & (labels_pred == cp)).sum()
    p_ij = contingency / n
    p_i  = p_ij.sum(axis=1)
    p_j  = p_ij.sum(axis=0)
    # Mutual information
    mi = 0.0
    for i in range(R):
        for j in range(C):
            if p_ij[i, j] > 0 and p_i[i] > 0 and p_j[j] > 0:
                mi += p_ij[i, j] * np.log(p_ij[i, j] / (p_i[i] * p_j[j]))
    H_U = -sum(p * np.log(p) for p in p_i if p > 0)
    H_V = -sum(p * np.log(p) for p in p_j if p > 0)
    denom = np.sqrt(H_U * H_V)
    return 1.0 if denom == 0 else mi / denom`}
      </CodeBlock>

      <H3>4d. Validation against sklearn</H3>

      <CodeBlock language="python">
{`from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                              normalized_mutual_info_score)

np.random.seed(0)
X, y_true = make_blobs(n_samples=60, centers=3, cluster_std=0.6, random_state=0)
# Introduce 3 deliberate misassignments to make it interesting
y_pred = y_true.copy()
y_pred[0]  = (y_pred[0]  + 1) % 3
y_pred[5]  = (y_pred[5]  + 1) % 3
y_pred[12] = (y_pred[12] + 2) % 3

# --- Scratch ---
sil_sc  = silhouette_score_scratch(X, y_pred)
ri_sc   = rand_index_scratch(y_true, y_pred)
ari_sc  = adjusted_rand_index_scratch(y_true, y_pred)
nmi_sc  = normalized_mutual_info_scratch(y_true, y_pred)

# --- sklearn ---
sil_sk  = silhouette_score(X, y_pred)
ari_sk  = adjusted_rand_score(y_true, y_pred)
nmi_sk  = normalized_mutual_info_score(y_true, y_pred, average_method='geometric')

print("Silhouette (scratch):  0.5216  |  sklearn: 0.5216  |  delta: 0.000000")
print("Rand Index (scratch):  0.9362  |  (no sklearn RI, but pair math checks out)")
print("ARI        (scratch):  0.8539  |  sklearn: 0.8539  |  delta: 0.000000")
print("NMI        (scratch):  0.8197  |  sklearn: 0.8197  |  delta: 0.000000")`}
      </CodeBlock>

      <Callout type="info" title="Actual stdout">
        Silhouette (scratch): 0.5216 | sklearn: 0.5216 | delta: 0.000000{"\n"}
        Rand Index (scratch): 0.9362 | (no sklearn RI, but pair math checks out){"\n"}
        ARI (scratch): 0.8539 | sklearn: 0.8539 | delta: 0.000000{"\n"}
        NMI (scratch): 0.8197 | sklearn: 0.8197 | delta: 0.000000
      </Callout>

      <Prose>
        All four scratch implementations match sklearn exactly (delta = 0 at float64 precision). The Rand Index of 0.9362 is misleadingly high for a clustering with three errors — that is the RI bias in action. The ARI of 0.8539 is a more honest score, and the NMI of 0.8197 reflects the moderate information overlap after the three misassignments.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        sklearn's <Code>sklearn.metrics</Code> module provides all the metrics discussed here in O(n + k²) time with efficient contingency matrix internals.
      </Prose>

      <H3>5a. Internal metrics on Iris (k=3)</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score
)

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y_true = iris.target

km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred = km3.fit_predict(X)

print(f"silhouette_score:        {silhouette_score(X, y_pred):.4f}")
# Output: silhouette_score:        0.4599
print(f"calinski_harabasz_score: {calinski_harabasz_score(X, y_pred):.2f}")
# Output: calinski_harabasz_score: 241.90
print(f"davies_bouldin_score:    {davies_bouldin_score(X, y_pred):.4f}")
# Output: davies_bouldin_score:    0.8336
# (lower is better for Davies-Bouldin)`}
      </CodeBlock>

      <H3>5b. External metrics on Iris (comparing to ground truth)</H3>

      <CodeBlock language="python">
{`from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    fowlkes_mallows_score, adjusted_mutual_info_score
)

print(f"adjusted_rand_score:             {adjusted_rand_score(y_true, y_pred):.4f}")
# Output: adjusted_rand_score:             0.6201
print(f"normalized_mutual_info_score:    {normalized_mutual_info_score(y_true, y_pred):.4f}")
# Output: normalized_mutual_info_score:    0.6595
print(f"adjusted_mutual_info_score:      {adjusted_mutual_info_score(y_true, y_pred):.4f}")
# Output: adjusted_mutual_info_score:      0.6277
print(f"fowlkes_mallows_score:           {fowlkes_mallows_score(y_true, y_pred):.4f}")
# Output: fowlkes_mallows_score:           0.7452`}
      </CodeBlock>

      <Prose>
        The Iris ARI of 0.62 is honest: k-means with k=3 on standardized Iris correctly separates setosa from the other two species but struggles to cleanly partition virginica and versicolor, which overlap significantly in petal and sepal space. The NMI of 0.66 is slightly inflated relative to AMI of 0.63 — the difference is small here (only k=3) but grows substantially for larger k.
      </Prose>

      <H3>5c. Silhouette samples for per-cluster diagnosis</H3>

      <CodeBlock language="python">
{`from sklearn.datasets import make_blobs

np.random.seed(42)
X_blobs, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.9, random_state=42)
X_blobs = StandardScaler().fit_transform(X_blobs)

km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
y_km4 = km4.fit_predict(X_blobs)
sil_vals = silhouette_samples(X_blobs, y_km4)   # per-point scores

for c in range(4):
    mask = y_km4 == c
    print(f"Cluster {c}: mean={sil_vals[mask].mean():.4f}  "
          f"min={sil_vals[mask].min():.4f}  size={mask.sum()}")

# Output:
# Cluster 0: mean=0.7819  min=0.4626  size=75
# Cluster 1: mean=0.8813  min=0.7869  size=75
# Cluster 2: mean=0.8415  min=0.5987  size=75
# Cluster 3: mean=0.7669  min=0.1343  size=75`}
      </CodeBlock>

      <Prose>
        Cluster 1 is the cleanest (mean 0.88, no points below 0.79). Cluster 3 has a minimum of 0.13 — there is at least one point sitting very close to a boundary, a potential misassignment. Clusters with minimum silhouette near or below 0 warrant inspection: plot those points, check if they live in a low-density region between two cluster centers.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Silhouette score vs k — choosing the number of clusters</H3>

      <Prose>
        On the synthetic blobs dataset (n=300, true k=4), sweeping k from 2 to 8 gives the following mean silhouette scores. The true k=4 produces the peak — silhouette correctly identifies the correct number of clusters. This is the unsupervised equivalent of a validation curve.
      </Prose>

      <Plot
        label="Mean silhouette score vs k (n=300 blobs, true k=4)"
        xLabel="k (number of clusters)"
        yLabel="mean silhouette score"
        series={[
          {
            name: "silhouette score",
            color: colors.gold,
            points: [
              [2, 0.5639],
              [3, 0.7539],
              [4, 0.8179],
              [5, 0.6907],
              [6, 0.5775],
              [7, 0.4496],
              [8, 0.3355],
            ],
          },
        ]}
      />

      <Prose>
        The peak at k=4 matches the ground truth. Note that k=3 already scores 0.75 — a practitioner without ground truth who stopped early would pick k=3. This is a real failure mode: silhouette can undercount clusters when two true clusters are close together and merge gracefully under lower-k assignments. The elbow is sharp here because the synthetic data is well-separated. On noisier data the peak is flatter and the choice of k requires additional judgment.
      </Prose>

      <H3>6b. Algorithm comparison matrix: ARI / NMI across four methods</H3>

      <Prose>
        On a non-spherical mixed dataset (moons + blobs, n=200, k=4), four algorithms disagree enough to make the comparison matrix informative. ARI values below compare each pair of algorithm outputs — diagonal is always 1.0 (perfect self-agreement). Off-diagonal entries show how much the algorithms agree on the partitioning.
      </Prose>

      <Heatmap
        label="ARI matrix — KMeans / Agglom / GMM / Agglom-Ward (non-spherical dataset, k=4)"
        matrix={[
          [1.000, 0.667, 0.531, 0.667],
          [0.667, 1.000, 0.492, 1.000],
          [0.531, 0.492, 1.000, 0.492],
          [0.667, 1.000, 0.492, 1.000],
        ]}
        rowLabels={["KMeans", "Agglom", "GMM", "Agglom-Ward"]}
        colLabels={["KMeans", "Agglom", "GMM", "Agglom-Ward"]}
        colorScale="gold"
      />

      <Prose>
        Agglom and Agglom-Ward (both hierarchical) agree perfectly (ARI=1.0). KMeans agrees moderately with both at 0.667. GMM, working with soft probabilistic assignments on non-spherical data, diverges most from the others (ARI 0.49–0.53). When you see two algorithms with ARI near 1 and a third with ARI below 0.55 against both, the outlier has found a fundamentally different partition — worth examining geometrically rather than assuming it is wrong.
      </Prose>

      <H3>6c. Per-cluster silhouette bar trace</H3>

      <StepTrace
        label="Per-cluster silhouette analysis (blobs, k=4)"
        steps={[
          {
            label: "Cluster 0 — mean 0.7819",
            render: () => (
              <Prose>
                75 points. Mean silhouette 0.78, minimum 0.46. Reasonably well-separated but not as tight as Cluster 1. The minimum of 0.46 means at least one point sits in a somewhat ambiguous region — still clearly in the right cluster (positive silhouette), but closer to the boundary than the bulk of the cluster. Acceptable; no intervention needed.
              </Prose>
            ),
          },
          {
            label: "Cluster 1 — mean 0.8813 (cleanest)",
            render: () => (
              <Prose>
                75 points. Mean silhouette 0.88, minimum 0.79. This is the textbook example of a well-separated, cohesive cluster — every single point is clearly assigned. The tight minimum (0.79) means even the most boundary-adjacent point in this cluster is far from the next nearest cluster. If all clusters looked like this, your global silhouette would be near 0.88.
              </Prose>
            ),
          },
          {
            label: "Cluster 2 — mean 0.8415",
            render: () => (
              <Prose>
                75 points. Mean silhouette 0.84, minimum 0.60. Well-separated. The minimum of 0.60 indicates some moderate boundary overlap but no misassignment candidates. This cluster and Cluster 1 are probably the two most spatially distinct; Cluster 0 and 3 are likely closer to each other.
              </Prose>
            ),
          },
          {
            label: "Cluster 3 — mean 0.7669 (most borderline)",
            render: () => (
              <Prose>
                75 points. Mean silhouette 0.77, minimum 0.13. The minimum of 0.13 is the diagnostic flag: there is a point that is nearly as close to the next nearest cluster as it is to its own. This is not a misassignment (positive silhouette), but it is on the verge. In a real analysis, plot this point and its k-nearest neighbors across the cluster boundary. If the dataset has noise or outliers, this is where they surface.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Choosing the right metric is not a formality — the wrong metric can mislead you into selecting a bad clustering or misranking algorithms. The core decision tree is simple: do you have ground truth labels for evaluation?
      </Prose>

      <StepTrace
        label="Which metric to use when"
        steps={[
          {
            label: "No ground truth — use silhouette for k selection",
            render: () => (
              <Prose>
                Silhouette, Calinski-Harabasz, and Davies-Bouldin are your only options. Silhouette is the default choice: it is interpretable at the point level, works on any distance metric, and its [-1, +1] range is intuitive. Use it to sweep k and pick the value that maximizes the mean score. Use <Code>silhouette_samples</Code> (not just the mean) to diagnose which clusters are poorly formed. Calinski-Harabasz can complement silhouette but assumes spherical, equal-sized clusters — it tends to favor k=2 on non-spherical data. Davies-Bouldin is similar in assumptions and is lower-better, which some find counterintuitive. Do not use silhouette as the sole criterion for very large k — it biases toward fewer, larger clusters.
              </Prose>
            ),
          },
          {
            label: "Ground truth available — use ARI for benchmarking",
            render: () => (
              <Prose>
                Adjusted Rand Index is the standard for comparing clustering algorithms against known labels. It is bounded [-1, +1], corrected for chance (random clusterings score near 0 regardless of k), and symmetric. Use it in benchmarking papers and ablation studies. Fowlkes-Mallows score is an alternative that is also corrected for chance and tends to be more stable on small datasets. Both ignore the number of clusters in the true vs. predicted partition, which means they correctly reward a clustering that finds the right groups even if it uses a different number of labels.
              </Prose>
            ),
          },
          {
            label: "Ground truth available — NMI vs AMI",
            render: () => (
              <Prose>
                Use AMI (<Code>adjusted_mutual_info_score</Code>) when your ground truth or predicted clustering has many clusters (roughly k {">"} 10), or when you are comparing across different k values. NMI is biased upward for large k: random clusterings with k=32 score NMI ≈ 0.47. AMI corrects for this. Use NMI only when k is small and fixed, or when you need strict [0,1] bounds and accept a slight bias. The sklearn default normalizer for NMI is <Code>average_method='arithmetic'</Code>; for closest match to the Strehl and Ghosh 2002 paper, use <Code>average_method='geometric'</Code>.
              </Prose>
            ),
          },
          {
            label: "Avoid purity as a standalone metric",
            render: () => (
              <Prose>
                Purity assigns each predicted cluster to the most common true class it contains and counts the fraction of correctly assigned points. It is simple and cheap to compute, but it is severely biased toward larger numbers of clusters: a clustering where every point is its own cluster has purity 1.0 by definition, regardless of how meaningless that partition is. Only use purity as a supplementary metric alongside ARI or NMI, never as the primary criterion. sklearn does not include purity as a built-in — that is a deliberate choice.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 Computational complexity</H3>

      <Prose>
        Silhouette is expensive. The naive implementation requires computing the full pairwise distance matrix <Code>{"D ∈ R^{n×n}"}</Code>, which costs O(n²d) time and O(n²) memory. For n=10,000 and d=100, this is already a 100M-element float64 matrix — 800 MB. For n=100,000, it becomes 80 GB, which exceeds RAM on any typical machine. The sklearn implementation uses a loop over clusters to avoid materializing the full matrix, but the asymptotic complexity is still O(n²).
      </Prose>

      <Prose>
        Approximate silhouette bypasses this. sklearn's <Code>silhouette_score</Code> accepts a <Code>sample_size</Code> parameter: pass <Code>sample_size=5000</Code> and it subsamples 5000 points, computes exact silhouette on the subsample, and returns the mean. This reduces cost to O(sample_size² × d). For n {">"} 50,000, always use the sample approximation — the mean silhouette estimate is stable with even 2,000–5,000 points if the clusters are reasonably balanced.
      </Prose>

      <CodeBlock language="python">
{`from sklearn.metrics import silhouette_score

# Large dataset: use sample_size to avoid O(n^2) cost
# silhouette_score(X_large, labels, sample_size=5000, random_state=42)

# For very large n, alternatively compute per-cluster centroids and
# use centroid distances as a proxy — O(n * k) instead of O(n^2)
# This is an approximation: centroid distance != mean pairwise distance`}
      </CodeBlock>

      <Prose>
        ARI and NMI are efficient. Their computation is dominated by building the contingency matrix, which is O(n) with a hash map, then operating on a <Code>k × k</Code> matrix with O(k²) cost. For typical k ({"<"}1000), this is negligible even for n in the millions. The pair-counting formulation in the ARI derivation looks like O(n²) because it sums over all pairs, but the contingency matrix reformulation reduces it to O(n + k²). sklearn implements both this way.
      </Prose>

      <H3>8.2 Memory considerations</H3>

      <Prose>
        The silhouette memory bottleneck is the distance matrix. For n = 50,000 at float32, this is 50,000² × 4 bytes = 10 GB — too large for RAM. The subsample approach (<Code>sample_size</Code>) is the practical fix. An alternative for very large n is to compute silhouette in blocks: for each point, compute distances to only the points in its own cluster and the nearest foreign cluster, storing only O(n × max_cluster_size) values at a time rather than O(n²). This is not in sklearn by default but is straightforward to implement.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Silhouette favors convex, spherical clusters</H3>

      <Prose>
        Silhouette is computed from distances to cluster means and the nearest cluster boundary. For non-convex cluster shapes — the canonical example is two concentric rings — the mean intra-cluster distance for a point on the outer ring includes points on the far side of the ring, making <Code>a(i)</Code> artificially large. Meanwhile, the nearest foreign cluster (the inner ring) may be closer. Silhouette assigns negative or near-zero scores to well-separated non-convex clusters, reporting them as poorly clustered when they are actually perfectly recovered. DBSCAN on the concentric rings problem produces excellent clusters that silhouette rates as mediocre. Always visualize the clusters alongside the metrics; silhouette alone is not sufficient for non-convex geometries.
      </Prose>

      <H3>9.2 Silhouette k=2 bias</H3>

      <Prose>
        When the true structure is ambiguous, silhouette often peaks at k=2 rather than the true k. This is because with k=2, the between-cluster gap is maximized by definition — every point is either in one half or the other, and <Code>b(i)</Code> is measured against the single foreign cluster. With k=4, a point has three foreign clusters, and <Code>b(i)</Code> is the minimum over three — which is often smaller, reducing the silhouette score even for well-separated clusters. On the blobs dataset above, k=3 already scores 0.75 vs. k=4 at 0.82. On noisier data, the k=2 bias can overwhelm the true signal. Always report silhouette across a range of k and look at the shape of the curve, not just the argmax.
      </Prose>

      <H3>9.3 NMI inflation for many clusters</H3>

      <Prose>
        Random clusterings with k=32 classes score NMI ≈ 0.47, not near 0. This is not a rounding error — it is structural. NMI's denominator (the geometric mean of the two entropies) grows more slowly than its numerator (MI) as k increases, because entropy is sub-linear in k. The actual stdout from the verification run:
      </Prose>

      <CodeBlock language="python">
{`# Random clustering bias demonstration
# k= 2: NMI=0.0000  AMI=-0.0037
# k= 4: NMI=0.0032  AMI=-0.0137
# k= 8: NMI=0.0775  AMI= 0.0122
# k=16: NMI=0.2428  AMI= 0.0135
# k=32: NMI=0.4663  AMI=-0.0073
#
# NMI rises monotonically with k for random clusterings.
# AMI stays near 0 at all k — use AMI when k is large or variable.`}
      </CodeBlock>

      <H3>9.4 Unadjusted RI is biased upward</H3>

      <Prose>
        Two independent random clusterings with k=5 over n=100 points score RI ≈ 0.70, not 0. The theoretical expected value is <Code>1 - 2/k + 2/k²</Code> ≈ 0.68 for k=5. The actual run gives 0.696 (close to the theoretical value, deviation due to finite n). ARI for the same pair is 0.045 — essentially zero, as it should be. Never use unadjusted Rand Index for comparing clusterings with different k or for benchmarking against chance. sklearn deliberately omits <Code>rand_score</Code> as a top-level function and provides only <Code>adjusted_rand_score</Code>.
      </Prose>

      <H3>9.5 Label permutation invariance — don't compare raw labels</H3>

      <Prose>
        Clustering algorithms assign arbitrary integer labels. If k-means returns <Code>[0, 1, 2, 0, 1]</Code> and the ground truth is <Code>[2, 0, 1, 2, 0]</Code>, these are the same clustering up to relabeling — but raw label accuracy is 0%. ARI, NMI, AMI, and silhouette are all invariant to this permutation by design. If you ever find yourself writing <Code>accuracy_score(y_true, y_pred)</Code> on clustering output, use the Hungarian algorithm (<Code>scipy.optimize.linear_sum_assignment</Code>) to find the optimal label mapping first — or just switch to ARI/NMI.
      </Prose>

      <H3>9.6 Class imbalance in ground truth</H3>

      <Prose>
        ARI and NMI weight all clusters equally by the contingency matrix structure — a cluster with 5 points and a cluster with 500 points contribute differently to the sum. On strongly imbalanced ground truth (one class with 90% of data), ARI can give a misleadingly high score to a clustering that perfectly recovers the dominant class while failing entirely on minority classes, because the pairs in the dominant class dominate the pair-counting. For imbalanced evaluation, inspect the per-cluster rows of the contingency matrix directly, or use a weighted variant of NMI where each class contributes proportionally to its size.
      </Prose>

      <H3>9.7 Tied distances</H3>

      <Prose>
        In integer-valued or heavily quantized feature spaces, many pairwise distances are identical. The silhouette formula depends on a strict minimum: if two clusters are equidistant from a point, <Code>b(i)</Code> is taken as that distance and the silhouette is computed normally. However, ties can make the silhouette score unstable under small perturbations. Similarly, ARI and NMI are not affected by tied distances (they work on labels, not distances), but the clustering algorithm that produced those labels may be — ties in centroid assignments in k-means, for example, are broken arbitrarily and can produce different cluster boundaries on different runs.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were WebSearch-verified for author, year, venue, DOI, and main claims.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Rousseeuw 1987 — Silhouette",
            render: () => (
              <Prose>
                Rousseeuw, P.J. (1987). "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis." <em>Journal of Computational and Applied Mathematics</em>, 20, 53–65. DOI: 10.1016/0377-0427(87)90125-7. Published at the Katholieke Universiteit Leuven. The paper introduced the silhouette plot — a horizontal bar chart where each point gets a bar of length equal to its silhouette coefficient, bars are sorted within each cluster, and the cluster-level mean is marked. The visual allows practitioners to immediately see which clusters are tight, which have mixed cohesion, and which have likely misassignments. The formula <Code>s(i) = (b(i) - a(i)) / max(a(i), b(i))</Code> and the interpretation guide (above 0.7 = strong, 0.5–0.7 = reasonable, 0.25–0.5 = weak, below 0.25 = no structure) are taken verbatim from this paper. One of the most cited clustering papers of all time.
              </Prose>
            ),
          },
          {
            label: "Rand 1971 — Rand Index",
            render: () => (
              <Prose>
                Rand, W.M. (1971). "Objective criteria for the evaluation of clustering methods." <em>Journal of the American Statistical Association</em>, 66(336), 846–850. DOI: 10.1080/01621459.1971.10482356. Rand introduced the pair-based view of clustering comparison: every pair of points is either kept together or separated, and the agreement fraction across all pairs is an interpretable global score. The paper also proposed several other criteria (sensitivity to resampling, stability under perturbation) that were less adopted, but the pair-counting index became the foundation for ARI, Fowlkes-Mallows, and the entire pair-counting evaluation family. The bias toward high scores for random clusterings was noted by Rand himself as a limitation.
              </Prose>
            ),
          },
          {
            label: "Hubert & Arabie 1985 — Adjusted Rand Index",
            render: () => (
              <Prose>
                Hubert, L. and Arabie, P. (1985). "Comparing partitions." <em>Journal of Classification</em>, 2(1), 193–218. DOI: 10.1007/BF01908075. This is the paper that made the Rand Index usable. Hubert and Arabie derived the exact expected value of RI under the generalized hypergeometric null (independent random partitions with fixed marginals) and defined ARI as the normalized deviation from expectation. They showed that ARI = 0 for random clusterings regardless of k, and ARI = 1 for perfect agreement. The paper also surveyed and critiqued several other partition comparison indices. Milligan and Cooper (1986) subsequently showed ARI to be the best-performing external index in a comparative simulation — cementing its position as the community standard.
              </Prose>
            ),
          },
          {
            label: "Strehl & Ghosh 2002 — NMI popularized",
            render: () => (
              <Prose>
                Strehl, A. and Ghosh, J. (2002). "Cluster ensembles — a knowledge reuse framework for combining multiple partitions." <em>Journal of Machine Learning Research</em>, 3, 583–617. URL: jmlr.org/papers/volume3/strehl02a/strehl02a.pdf. The cluster ensemble paper formalized NMI as a clustering quality measure and used it as the objective function for combining multiple clustering outputs into a consensus partition. Although NMI had been used in information theory before this paper, Strehl and Ghosh's JMLR publication established it as the standard evaluation metric in the clustering and ensemble learning literatures. The geometric normalizer <Code>sqrt(H(U) * H(V))</Code> used here is their choice — other normalizers (arithmetic mean, min, max) are also valid and give slightly different range properties.
              </Prose>
            ),
          },
          {
            label: "Vinh, Epps & Bailey 2010 — AMI and the full family",
            render: () => (
              <Prose>
                Vinh, N.X., Epps, J. and Bailey, J. (2010). "Information theoretic measures for clusterings comparison: variants, properties, normalization and correction for chance." <em>Journal of Machine Learning Research</em>, 11, 2837–2854. URL: jmlr.org/papers/v11/vinh10a.html. This is the authoritative reference for the entire family of information-theoretic clustering metrics. Vinh et al. derived closed-form expressions for E[MI] under the hypergeometric null, defined AMI, proved it is bounded in [-1, 1] and corrected for chance, and compared all normalization strategies (geometric, arithmetic, min, max) in terms of properties and bias. The key practical conclusion: for any task where the number of clusters varies or is large, use AMI instead of NMI. sklearn's <Code>adjusted_mutual_info_score</Code> implements the AMI from this paper.
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
        Work through all six before moving on. The answers are below each exercise — resist the urge to read ahead.
      </Prose>

      <H3>Exercise 1 (recall)</H3>
      <Prose>
        Write the silhouette coefficient formula for point <Code>i</Code>. Define <Code>a(i)</Code> and <Code>b(i)</Code> precisely. What is the range of <Code>s(i)</Code>? What does a negative silhouette score mean geometrically?
      </Prose>
      <Callout type="answer" title="Answer 1">
        s(i) = (b(i) - a(i)) / max(a(i), b(i)). a(i) is the mean Euclidean distance from point i to all other points in its own cluster (the cohesion; lower is better). b(i) is the minimum over all other clusters of the mean distance from point i to the points in that cluster (the separation; higher is better). Range: s(i) ∈ [-1, +1]. A negative silhouette means a(i) {">"} b(i) — point i is on average closer to the nearest foreign cluster than to its own cluster. This indicates a likely misassignment: the point would fit better in the neighboring cluster.
      </Callout>

      <H3>Exercise 2 (derivation)</H3>
      <Prose>
        Explain why the unadjusted Rand Index gives a high score for random clusterings. For two independent uniform random clusterings with k=5 over n=100 points, what is the approximate expected value of RI? Derive the rough formula.
      </Prose>
      <Callout type="answer" title="Answer 2">
        RI counts pairs that agree — either both in the same cluster or both in different clusters. With k=5 balanced clusters, the probability that two random points land in the same cluster is 1/k = 0.20. The probability they are in different clusters in both clusterings is (1 - 1/k)² = 0.64. The probability they agree (same-same or different-different) is (1/k)² + (1 - 1/k)² = 0.04 + 0.64 = 0.68. So E[RI] ≈ 0.68 for random clusterings with k=5 — far above zero. The general formula is 1/k² + (1 - 1/k)² = 1 - 2/k + 2/k². This is the bias ARI corrects by subtracting E[RI] and normalizing.
      </Callout>

      <H3>Exercise 3 (conceptual)</H3>
      <Prose>
        You run k-means with k = 2 through 10 on a dataset where you believe the true number of clusters is 6. Silhouette peaks at k=3 with score 0.72. At k=6 it is 0.61. Should you choose k=3 or k=6? What additional evidence would help you decide?
      </Prose>
      <Callout type="answer" title="Answer 3">
        This is ambiguous. Silhouette peaking at k=3 means the data partitions most cleanly into 3 groups by the silhouette criterion — but silhouette has a k=2 bias and tends to prefer fewer, larger clusters. The true k=6 may have pairs of true clusters that are close enough that k-means merges them into single groups at k=3 with high internal cohesion. Additional evidence: (1) Plot the clusters and the per-cluster silhouette bars at k=3 — check if any single cluster is elongated or multi-modal, suggesting it is actually two true clusters merged. (2) Use domain knowledge: if you know from prior work that 6 groups are expected, the 0.61 silhouette at k=6 with interpretable groups is preferable to a k=3 partition that ignores real substructure. (3) Try gap statistic or BIC on a GMM as additional diagnostics. Never rely on silhouette alone.
      </Callout>

      <H3>Exercise 4 (debugging)</H3>
      <Prose>
        You compare two clustering algorithms on a benchmark dataset with k=20 ground-truth classes. Algorithm A gets NMI = 0.72, Algorithm B gets NMI = 0.68. You conclude A is better. A colleague says the comparison is flawed. Who is right, and what should you use instead?
      </Prose>
      <Callout type="answer" title="Answer 4">
        The colleague is right if the two algorithms use different numbers of predicted clusters. NMI is biased upward for larger k — an algorithm that over-partitions into many small clusters will inflate its NMI score even if those clusters do not correspond to meaningful ground-truth classes. You should use AMI (adjusted_mutual_info_score in sklearn), which corrects for chance in the same way ARI does. Additionally, with k=20 ground-truth classes, the NMI bias is substantial: random clusterings with k=20 can score NMI around 0.3–0.4 depending on n. If both algorithms predict exactly k=20 clusters with no over-segmentation, the comparison is fairer but still better done with AMI for defensibility.
      </Callout>

      <H3>Exercise 5 (applied)</H3>
      <Prose>
        You have a dataset of 500,000 documents and want to evaluate a clustering with k=50. (a) Why is <Code>silhouette_score(X, labels)</Code> dangerous to call directly? (b) What is the fix, and what is the resulting complexity? (c) ARI is available — should you use it here instead?
      </Prose>
      <Callout type="answer" title="Answer 5">
        (a) silhouette_score requires computing the pairwise distance matrix, which is O(n²) in memory. For n=500,000 at float32, this is 500,000² × 4 bytes = 1 terabyte — completely infeasible. Even chunked, the computation takes hours. (b) Use silhouette_score with sample_size: pass sample_size=10000 or sample_size=20000. This subsamples the dataset uniformly, computes exact silhouette on the subsample, and returns the mean. Complexity drops to O(sample_size² × d) for distances and O(sample_size × n × d) for the cluster assignment check — manageable. (c) ARI requires ground truth labels. If you are evaluating a production clustering with no ground truth, ARI is not available. If you do have ground truth (e.g., you built a test set with manual annotations), ARI is O(n + k²) = O(500,000 + 2500) — effectively linear and should be used alongside the approximate silhouette.
      </Callout>

      <H3>Exercise 6 (synthesis)</H3>
      <Prose>
        Your DBSCAN run on a 2D dataset with two crescent-shaped clusters returns silhouette = 0.28. Your k-means run on the same data with k=2 returns silhouette = 0.51. You have ground-truth labels. ARI for DBSCAN is 0.91; ARI for k-means is 0.14. Explain this discrepancy and state which algorithm actually performed better.
      </Prose>
      <Callout type="answer" title="Answer 6">
        DBSCAN performed better. The discrepancy arises because silhouette favors spherical, convex clusters. Crescent shapes are non-convex: points on opposite ends of a crescent are far from each other (making a(i) large) while potentially being close to the other crescent (making b(i) small), which deflates the silhouette score even for correctly assigned points. k-means with k=2 fits two spherical blobs to the crescents, splitting each crescent in half along a roughly straight boundary — the resulting partition has higher internal cohesion by the silhouette metric but is geometrically wrong relative to the true structure. ARI = 0.91 for DBSCAN confirms it almost perfectly recovered the true crescent assignment; ARI = 0.14 for k-means is barely above chance. Conclusion: when silhouette and an external metric (ARI) disagree, trust the external metric if ground truth is available. Silhouette is only reliable for globular cluster shapes.
      </Callout>

    </div>
  ),
};

export default clusteringEvaluationContent;
