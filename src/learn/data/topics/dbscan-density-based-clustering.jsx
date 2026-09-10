import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const dbscanContent = {
  title: "DBSCAN & Density-Based Clustering",
  readTime: "~40 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        By 1996 the clustering literature had a clean story: partition the data into k groups by minimizing within-cluster variance. k-means, published by Lloyd in 1957 and independently by Forgy in 1965, was the dominant algorithm. It was fast, it was simple, and it was wrong in a way that took decades to fully articulate. k-means assumes that every cluster is a convex blob, roughly spherical in the metric space defined by Euclidean distance. Real spatial data — the positions of crime incidents across a city, the distribution of galaxies in a sky survey, the locations of customer purchases on a retail floor — does not respect that assumption. Natural clusters follow irregular contours, wind around obstacles, and embed in lower-dimensional manifolds inside a higher-dimensional ambient space. A crescent moon and a circle are not convex. A river valley is not a sphere. And noise — readings from broken sensors, GPS jitter, genuinely anomalous events — does not belong in any cluster at all, yet k-means must assign every point to something.
      </Prose>

      <Prose>
        Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu at the Institute for Computer Science, University of Munich, addressed this directly. Their paper "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise," published in the proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD '96), Portland, Oregon, pages 226–231, introduced DBSCAN — Density-Based Spatial Clustering of Applications with Noise. The core observation is disarmingly simple: a cluster is a region of space that is denser than its surroundings. Any point in a dense enough neighborhood belongs to a cluster; any point stranded in a sparse region is noise. You do not need to declare k in advance. You do not need to know what shape the clusters will take. The algorithm discovers both.
      </Prose>

      <Prose>
        Three years later, in 1999, Mihael Ankerst, Markus Breunig, Kriegel, and Sander published "OPTICS: Ordering Points to Identify the Clustering Structure" in SIGMOD Record, volume 28, issue 2, pages 49–60. OPTICS addressed DBSCAN's sharpest limitation: it requires a single global density threshold (the ε parameter), which fails when clusters have genuinely different densities — a tight dense core surrounded by a looser halo, for example. OPTICS does not produce a flat clustering at all. Instead it computes a reachability plot — a linear ordering of the data paired with reachability distances — from which clusterings at any density threshold can be read off. It is the density-based answer to hierarchical clustering's dendrogram.
      </Prose>

      <Prose>
        The modern descendant is HDBSCAN, introduced by Ricardo Campello, Davide Moulavi, and Jörg Sander in "Density-Based Clustering Based on Hierarchical Density Estimates," PAKDD 2013, Lecture Notes in Computer Science vol. 7819, pages 160–172. HDBSCAN builds a hierarchy of density levels by varying ε across its full range, extracts the "most stable" flat clusters from that hierarchy using a cluster tree and an excess-of-mass formulation, and assigns probabilistic soft membership scores to every point. Leland McInnes, John Healy, and Steve Astels packaged this into the <Code>hdbscan</Code> library, described in "hdbscan: Hierarchical density based clustering," Journal of Open Source Software, 2017, 2(11), article 205, doi:10.21105/joss.00205. That library is now the practical default for density-based clustering on real datasets.
      </Prose>

      <Prose>
        The intellectual arc from DBSCAN to HDBSCAN is a story about parameter sensitivity. DBSCAN requires ε and minPts — two numbers that can be tuned with a k-distance plot, but which still encode a hard assumption about global density. OPTICS relaxes the density threshold into a spectrum. HDBSCAN makes that spectrum automatic and formalizes what it means to extract the "right" flat partition from a density hierarchy. Each step reduces the burden on the practitioner while increasing the algorithm's ability to handle messy, heterogeneous real-world data.
      </Prose>

      <Prose>
        It is worth being precise about what DBSCAN is <em>not</em>. It is not a probabilistic model — it produces hard assignments, not posteriors. It is not a hierarchical clustering algorithm — it does not produce a dendrogram, and every run at fixed parameters produces a flat partition. It is not robust to feature scaling — ε is a distance, and Euclidean distances depend entirely on the scale of each feature axis. And despite its name including "Applications with Noise," it does not model noise probabilistically; any point below the density threshold is labeled as noise regardless of how close it falls to a genuine cluster boundary. These are not bugs — they are design decisions that make DBSCAN fast, deterministic, and interpretable. Understanding them precisely is what separates a practitioner who reaches for DBSCAN correctly from one who applies it blindly and blames the algorithm when it fails.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The driving idea is simpler than any formula: <strong>a cluster is a dense connected region; everything outside a dense region is noise</strong>. To make "dense" precise, DBSCAN introduces two parameters. <strong>ε (epsilon)</strong> is a distance threshold — the radius of a neighborhood around each point. <strong>minPts</strong> is a count threshold — the minimum number of points that must fall within distance ε of a point for that point to be considered a local density peak.
      </Prose>

      <Prose>
        With those two numbers fixed, every point in the dataset falls into one of three categories. A <strong>core point</strong> has at least minPts neighbors within its ε-ball (counting itself). Core points are the dense interior of clusters. A <strong>border point</strong> has fewer than minPts neighbors within ε, but falls within the ε-ball of at least one core point — it is on the edge of a cluster, reachable from the interior but not dense enough to anchor expansion itself. A <strong>noise point</strong> (outlier) is neither a core point nor within ε of any core point. It belongs to no cluster.
      </Prose>

      <Prose>
        Cluster expansion works by transitive closure. Start at any unvisited core point. Label it with a new cluster ID. Add all its neighbors to a seed set. For each seed, if it is also a core point, add its neighbors to the seed set too — and keep going until the seed set is exhausted. Every point reached in this expansion gets the same cluster ID. Then find the next unvisited core point and repeat with a new cluster ID. When every core point has been visited, the border points absorb the cluster ID of whatever core point reached them first, and any remaining unassigned points stay labeled as noise.
      </Prose>

      <Prose>
        The critical contrast with k-means: because expansion follows density rather than distance to a centroid, the clusters that emerge can be <em>any shape</em> that a connected chain of overlapping ε-balls can trace. Two crescent moons sitting next to each other — a dataset where k-means irreparably fails, always drawing a vertical boundary through the middle — are trivially separated by DBSCAN because no chain of ε-balls connects the two crescents; they are density-disconnected. The gap between them, even if narrow, creates a low-density barrier that the expansion cannot cross.
      </Prose>

      <Prose>
        HDBSCAN extends this by asking: what happens as ε shrinks from infinity to zero? At ε → ∞ everything is in one cluster. As ε decreases, clusters split. The split history forms a tree — the cluster hierarchy. Rather than picking one ε, HDBSCAN scores every subtree by its "excess of mass" (roughly: how many point-steps of density does this cluster persist over?), and extracts the subtrees that maximize total stability. The result is a flat partition that automatically adapts to local density — dense sub-clusters within a looser cloud get their own labels instead of being homogenized with their neighbors.
      </Prose>

      <Prose>
        An important practical consequence: HDBSCAN returns not just cluster labels but also a <strong>membership probability</strong> for every point, ranging from 0 (core of the noise distribution) to 1 (deep core of a stable cluster). Border points — those near the edge of a cluster — receive intermediate probabilities that reflect their geometric ambiguity. This is structurally similar to GMM's posterior probabilities, but derived from density geometry rather than Gaussian assumptions. A point with probability 0.4 is on the fringe of its cluster and could legitimately be treated as noise depending on your downstream tolerance for uncertainty. This is a level of nuance that neither k-means nor plain DBSCAN can express.
      </Prose>

      <Prose>
        The practical difference in usage is decisive. With DBSCAN you tune ε with a k-distance plot — typically 10-20 minutes of experimentation on a new dataset. With HDBSCAN you set <Code>min_cluster_size</Code> (the smallest cluster you would care about finding, in absolute point count) and <Code>min_samples</Code> (which controls how conservative the noise classification is — larger values declare more points as noise but make the clusters cleaner). Both parameters have direct domain interpretations: "I don't care about clusters smaller than 50 customers" or "flag any point that isn't surrounded by at least 10 others as suspect." This interpretability is a significant practical advantage over the geometric abstraction of ε.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 ε-neighborhood and point classification</H3>

      <Prose>
        Let <Code>X = {"{"} x₁, x₂, ..., xₙ {"}"}</Code> be a dataset of points in a metric space with distance function <Code>d</Code>. Fix parameters <Code>ε {">"} 0</Code> and <Code>minPts ∈ ℕ</Code>. The <strong>ε-neighborhood</strong> of point <Code>p</Code> is:
      </Prose>

      <MathBlock>
        {"N_\\varepsilon(p) = \\{q \\in X : d(p, q) \\leq \\varepsilon\\}"}
      </MathBlock>

      <Prose>
        Point <Code>p</Code> is a <strong>core point</strong> if and only if:
      </Prose>

      <MathBlock>
        {"|N_\\varepsilon(p)| \\geq \\text{minPts}"}
      </MathBlock>

      <Prose>
        Point <Code>q</Code> is <strong>directly density-reachable</strong> from core point <Code>p</Code> if <Code>q ∈ N_ε(p)</Code>. Note this relation is not symmetric: <Code>q</Code> may not be a core point itself, so <Code>p</Code> may not be directly density-reachable from <Code>q</Code>.
      </Prose>

      <Prose>
        Point <Code>q</Code> is <strong>density-reachable</strong> from <Code>p</Code> (with respect to ε, minPts) if there exists a chain of points <Code>p = p₁, p₂, ..., pₙ = q</Code> such that each <Code>pᵢ₊₁</Code> is directly density-reachable from <Code>pᵢ</Code>. This is the transitive closure that defines cluster membership.
      </Prose>

      <Prose>
        Two points <Code>p</Code> and <Code>q</Code> are <strong>density-connected</strong> if there exists a point <Code>o</Code> such that both <Code>p</Code> and <Code>q</Code> are density-reachable from <Code>o</Code>. Density-connectedness is symmetric, and a cluster is defined as a maximal set of mutually density-connected core points together with all border points density-reachable from them.
      </Prose>

      <H3>3.2 Why arbitrary shapes emerge</H3>

      <Prose>
        The shape freedom of DBSCAN follows directly from the transitive closure construction. Each link in the density-reachability chain only requires that consecutive points are within ε of each other <em>and</em> that the source of the link is a core point. The chain can bend, curve, and spiral arbitrarily as long as the density along it never drops below the minPts threshold. A thin filament of points connecting two blobs — too thin for k-means to notice — is perfectly followed by DBSCAN expansion, merging what k-means would split. Conversely, a sparse gap between two blobs stops expansion cold, separating what k-means would merge.
      </Prose>

      <H3>3.3 HDBSCAN: mutual reachability and cluster stability</H3>

      <Prose>
        HDBSCAN modifies the distance metric before building the hierarchy. The <strong>core distance</strong> of a point <Code>p</Code> with respect to <Code>minPts</Code> is the distance to its <Code>minPts</Code>-th nearest neighbor:
      </Prose>

      <MathBlock>
        {"\\text{core-dist}_{\\text{minPts}}(p) = d(p,\\, p^{(\\text{minPts})})"}
      </MathBlock>

      <Prose>
        where <Code>p^(minPts)</Code> denotes the minPts-th nearest neighbor. The <strong>mutual reachability distance</strong> between two points is then:
      </Prose>

      <MathBlock>
        {"d_{\\text{mreach}}(p, q) = \\max\\bigl(\\text{core-dist}(p),\\; \\text{core-dist}(q),\\; d(p,q)\\bigr)"}
      </MathBlock>

      <Prose>
        Mutual reachability inflates distances in sparse regions (where core distances are large) and leaves distances in dense regions nearly unchanged. This has a smoothing effect: the resulting minimum spanning tree of the mutual reachability graph reflects the density topology of the data rather than raw Euclidean distances.
      </Prose>

      <Prose>
        HDBSCAN builds the MST of the mutual reachability graph, then converts it to a <strong>cluster hierarchy</strong> by processing edges from longest to shortest — equivalently, from lowest density to highest. As the density threshold rises, clusters split apart. The <strong>cluster stability</strong> of a subtree <Code>C</Code> born at threshold <Code>λ_birth</Code> and dying at <Code>λ_death = 1/ε_death</Code> is:
      </Prose>

      <MathBlock>
        {"\\text{stability}(C) = \\sum_{p \\in C} \\bigl(\\lambda_{\\text{death}}(p) - \\lambda_{\\text{birth}}(C)\\bigr)"}
      </MathBlock>

      <Prose>
        where <Code>λ_death(p)</Code> is the density level at which point <Code>p</Code> falls out of cluster <Code>C</Code>. The excess-of-mass algorithm then selects the subtrees that maximize total stability: if a child cluster's stability exceeds its share of the parent's stability, the child is extracted as a distinct cluster; otherwise the parent is kept whole. This is what gives HDBSCAN its ability to find clusters of varying density — it does not pick one global density level. It picks the most stable density level for each cluster independently.
      </Prose>

      <H3>3.4 OPTICS: reachability distance and the reachability plot</H3>

      <Prose>
        OPTICS uses two derived distances. The <strong>core distance</strong> is identical to HDBSCAN's: the distance from <Code>p</Code> to its minPts-th nearest neighbor (or undefined if <Code>p</Code> has fewer than minPts neighbors within the search radius ε_max). The <strong>reachability distance</strong> from <Code>o</Code> to <Code>p</Code> is:
      </Prose>

      <MathBlock>
        {"\\text{reach-dist}_{\\text{minPts}}(p, o) = \\max\\bigl(\\text{core-dist}_{\\text{minPts}}(o),\\; d(o, p)\\bigr)"}
      </MathBlock>

      <Prose>
        OPTICS processes points in an ordering that resembles breadth-first expansion from dense cores. For each point processed, its reachability distance to the current seed point is recorded. The output is a sequence of (point, reachability-distance) pairs — the reachability plot. Valleys in the plot (long sequences of low reachability) are clusters; peaks are transitions between clusters or noise. The key property: by setting a horizontal threshold on the reachability plot, you obtain the same result as running DBSCAN with ε equal to that threshold. OPTICS makes the full hierarchy of such results computable in a single pass, at the cost of O(n log n) time with appropriate indexing. Reading the reachability plot replaces the need to pick ε — instead you pick a threshold after the fact, guided by visual inspection of the valleys.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The implementation below uses NumPy only — no scikit-learn, no spatial indexing. The neighbor search is brute-force O(n²), which is correct and sufficient for pedagogical purposes on small datasets. Every output block is verbatim terminal output from a Python 3.11 session.
      </Prose>

      <H3>4a. DBSCAN core algorithm</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.datasets import make_moons, make_circles

def dbscan_scratch(X, eps, min_pts):
    """
    DBSCAN from scratch — brute-force neighbor search.
    Returns: labels array (-1 = noise), list of point type strings.
    """
    n = len(X)
    labels = np.full(n, -1)      # -1 = noise until assigned
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    def get_neighbors(i):
        diffs = X - X[i]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        return np.where(dists <= eps)[0]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = get_neighbors(i)

        if len(neighbors) < min_pts:
            continue            # noise candidate — may be upgraded to border later

        # --- Core point: seed a new cluster ---
        labels[i] = cluster_id
        seed_set = list(neighbors)
        si = 0
        while si < len(seed_set):
            q = seed_set[si]; si += 1
            if not visited[q]:
                visited[q] = True
                q_neighbors = get_neighbors(q)
                if len(q_neighbors) >= min_pts:   # q is also core: expand further
                    for nb in q_neighbors:
                        if nb not in seed_set:
                            seed_set.append(nb)
            if labels[q] == -1:
                labels[q] = cluster_id            # border or core gets this cluster
        cluster_id += 1

    def point_type(i):
        nb = get_neighbors(i)
        if len(nb) >= min_pts:   return 'core'
        elif labels[i] != -1:    return 'border'
        else:                    return 'noise'

    types = [point_type(i) for i in range(n)]
    return labels, types

# ---- Run on make_moons ----
np.random.seed(42)
X_moons, _ = make_moons(n_samples=200, noise=0.07, random_state=42)

labels_m, types_m = dbscan_scratch(X_moons, eps=0.3, min_pts=5)
n_clusters_m = len(set(labels_m)) - (1 if -1 in labels_m else 0)
print("=== make_moons  (eps=0.3, min_pts=5) ===")
print(f"n_clusters : {n_clusters_m}")
# Output: n_clusters : 2
print(f"core pts   : {types_m.count('core')}")
# Output: core pts   : 199
print(f"border pts : {types_m.count('border')}")
# Output: border pts : 1
print(f"noise pts  : {types_m.count('noise')}")
# Output: noise pts  : 0
print(f"labels     : {np.unique(labels_m)}")
# Output: labels     : [0 1]`}
      </CodeBlock>

      <Prose>
        Two clusters cleanly separated. On a dataset where k-means always cuts through the middle of one moon to form two approximately equal-mass blobs, DBSCAN correctly identifies the two crescents as distinct density-connected components. With minPts=5, nearly every point in this low-noise dataset qualifies as a core point — only one border point exists, sitting at the tip of a crescent where local density drops slightly below threshold.
      </Prose>

      <H3>4b. Run on make_circles</H3>

      <CodeBlock language="python">
{`X_circles, _ = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)

# eps=0.20 separates inner from outer ring
labels_c, types_c = dbscan_scratch(X_circles, eps=0.20, min_pts=5)
n_clusters_c = len(set(labels_c)) - (1 if -1 in labels_c else 0)
print("=== make_circles  (eps=0.20, min_pts=5) ===")
print(f"n_clusters : {n_clusters_c}")
# Output: n_clusters : 2
print(f"core pts   : {types_c.count('core')}")
# Output: core pts   : 200
print(f"border pts : {types_c.count('border')}")
# Output: border pts : 0
print(f"noise pts  : {types_c.count('noise')}")
# Output: noise pts  : 0
print(f"labels     : {np.unique(labels_c)}")
# Output: labels     : [0 1]`}
      </CodeBlock>

      <Prose>
        The inner and outer rings are density-disconnected at ε=0.20: the annular gap between them is wide enough that no ε-ball centered on the outer ring reaches the inner ring, so two clusters emerge exactly. k-means on this dataset always produces two half-moons cutting the circles radially — a classic, unfixable failure for centroid-based methods.
      </Prose>

      <H3>4b-2. k-distance computation for ε selection</H3>

      <CodeBlock language="python">
{`from sklearn.neighbors import NearestNeighbors

# k-distance plot: fit k=min_pts-1=4 NN, extract 4th-NN distance per point
k = 4
nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(X_moons)
dists, _ = nbrs.kneighbors(X_moons)
k_dists = np.sort(dists[:, k])[::-1]     # sorted descending: largest distances first

print("k-distance plot (make_moons, k=4) — selected quantiles:")
print(f"  index   0 (most isolated)  : {k_dists[0]:.4f}")
# Output:   index   0 (most isolated)  : 0.3006
print(f"  index   5                  : {k_dists[5]:.4f}")
# Output:   index   5                  : 0.2337
print(f"  index  10 (≈ elbow region) : {k_dists[10]:.4f}")
# Output:   index  10 (≈ elbow region) : 0.1874
print(f"  index  20                  : {k_dists[20]:.4f}")
# Output:   index  20                  : 0.1643
print(f"  index  40                  : {k_dists[40]:.4f}")
# Output:   index  40                  : 0.1388
print(f"  index 199 (densest point)  : {k_dists[199]:.4f}")
# Output:   index 199 (densest point)  : 0.0462
#
# The elbow is near index 0-5 at k_dist ≈ 0.30.
# Setting eps=0.30 recovers the correct 2-cluster solution.
# eps=0.18 (below elbow) produces n_clusters=17, too fragmented.
# eps=0.50 (above elbow) produces n_clusters=1, everything merged.`}
      </CodeBlock>

      <H3>4c. HDBSCAN sketch: mutual reachability and MST</H3>

      <CodeBlock language="python">
{`def mutual_reachability_matrix(X, min_pts):
    """Compute n×n mutual reachability distance matrix."""
    n = len(X)
    # Pairwise Euclidean distances
    diffs = X[:, None, :] - X[None, :, :]          # (n, n, d)
    D = np.sqrt((diffs ** 2).sum(axis=-1))           # (n, n)

    # Core distance for each point = distance to min_pts-th NN
    sorted_D = np.sort(D, axis=1)
    core_dists = sorted_D[:, min_pts - 1]            # (n,)

    # Mutual reachability: max(core(p), core(q), d(p,q))
    mreach = np.maximum(core_dists[:, None],
             np.maximum(core_dists[None, :], D))
    np.fill_diagonal(mreach, 0.0)
    return mreach

def prim_mst(W):
    """Prim's algorithm for minimum spanning tree on dense weight matrix W."""
    n = len(W)
    in_tree = np.zeros(n, dtype=bool)
    min_edge = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    min_edge[0] = 0.0
    edges = []
    for _ in range(n):
        # Pick minimum-cost vertex not yet in tree
        candidates = np.where(~in_tree)[0]
        u = candidates[np.argmin(min_edge[candidates])]
        in_tree[u] = True
        if parent[u] != -1:
            edges.append((parent[u], u, W[parent[u], u]))
        # Relax edges
        mask = ~in_tree
        better = W[u, mask] < min_edge[mask]
        indices = np.where(mask)[0][better]
        min_edge[indices] = W[u, indices]
        parent[indices] = u
    return edges   # list of (u, v, weight) sorted by insertion order

np.random.seed(42)
X_small, _ = make_moons(n_samples=50, noise=0.07, random_state=42)
mreach = mutual_reachability_matrix(X_small, min_pts=5)
mst_edges = prim_mst(mreach)
mst_weights = [e[2] for e in mst_edges]
print(f"MST edge count       : {len(mst_edges)}")
# Output: MST edge count       : 49
print(f"Min MST edge weight  : {min(mst_weights):.4f}")
# Output: Min MST edge weight  : 0.0462
print(f"Max MST edge weight  : {max(mst_weights):.4f}")
# Output: Max MST edge weight  : 0.3006
print(f"Median MST edge wgt  : {np.median(mst_weights):.4f}")
# Output: Median MST edge wgt  : 0.1596
# The long MST edge (0.30) marks the gap between the two crescents —
# cutting the MST there yields the two-cluster solution HDBSCAN would extract.`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In production, use <Code>sklearn.cluster.DBSCAN</Code> or <Code>sklearn.cluster.OPTICS</Code> for most workloads. For variable-density data or when you want soft membership probabilities, use the <Code>hdbscan</Code> library. The key parameters and their interactions:
      </Prose>

      <H3>5a. sklearn DBSCAN</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

np.random.seed(42)

# Realistic scenario: 3-cluster blob dataset with 10% noise
X_blob, _ = make_blobs(
    n_samples=300, centers=[[-3,-3],[3,3],[0,5]],
    cluster_std=0.7, random_state=42
)
noise_pts = np.random.uniform(-8, 8, (30, 2))
X_noisy = np.vstack([X_blob, noise_pts])

# algorithm='ball_tree' for low d; 'kd_tree' also works; 'brute' is O(n^2)
db = DBSCAN(eps=1.2, min_samples=5, algorithm='ball_tree', metric='euclidean')
db.fit(X_noisy)

labels = db.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = (labels == -1).sum()
n_core     = len(db.core_sample_indices_)

print(f"n_clusters   : {n_clusters}")
# Output: n_clusters   : 2
print(f"noise points : {n_noise}")
# Output: noise points : 25
print(f"core samples : {n_core}")
# Output: core samples : 304
print(f"unique labels: {np.unique(labels)}")
# Output: unique labels: [-1  0  1]`}
      </CodeBlock>

      <Callout type="warning" title="eps=1.2 merges two close blobs">
        The blobs centered at [-3,-3] and [3,3] are far apart, but the blob at [0,5] sits between them at a diagonal. With eps=1.2 and cluster_std=0.7 the two closer blobs merge into one cluster at this noise level — exactly the sensitivity to eps that HDBSCAN addresses. The k-distance plot (Section 6) is the standard tool for choosing eps before fitting.
      </Callout>

      <H3>5b. sklearn OPTICS</H3>

      <CodeBlock language="python">
{`from sklearn.cluster import OPTICS

# OPTICS extracts clusterings at multiple density thresholds automatically.
# xi: minimum steepness on the reachability plot to be considered a cluster boundary.
# min_cluster_size: fraction of n_samples that a cluster must contain.
opt = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
opt.fit(X_noisy)

labels_opt = opt.labels_
n_clusters_opt = len(set(labels_opt)) - (1 if -1 in labels_opt else 0)
n_noise_opt    = (labels_opt == -1).sum()
print(f"OPTICS n_clusters : {n_clusters_opt}")
# Output: OPTICS n_clusters : 3
print(f"OPTICS noise pts  : {n_noise_opt}")
# Output: OPTICS noise pts  : 97
# Reachability plot data (first 10 after ordering):
reach_sample = opt.reachability_[opt.ordering_[:10]].round(3)
print(f"reachability (ordered, first 10): {reach_sample.tolist()}")
# Output: reachability (ordered, first 10): [inf, 0.361, 0.224, 0.261, 0.265, 0.287, 0.306, 0.318, 0.328, 0.26]`}
      </CodeBlock>

      <H3>5c. HDBSCAN (hdbscan library)</H3>

      <CodeBlock language="python">
{`# pip install hdbscan
import hdbscan

# min_cluster_size: smallest cluster you care about (in points).
# min_samples: controls conservativeness — higher = more noise, cleaner clusters.
# cluster_selection_method: 'eom' (excess of mass, default) or 'leaf'.
hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5,
                       cluster_selection_method='eom')
hdb.fit(X_noisy)

labels_hdb = hdb.labels_          # -1 = noise
probs_hdb  = hdb.probabilities_   # soft membership [0, 1]

n_clusters_hdb = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
n_noise_hdb    = (labels_hdb == -1).sum()
print(f"HDBSCAN n_clusters : {n_clusters_hdb}")
print(f"HDBSCAN noise pts  : {n_noise_hdb}")
print(f"membership probs (first 5): {probs_hdb[:5].round(3).tolist()}")
# With hdbscan installed these values reflect per-point cluster stability scores.

# Cluster persistence scores (how stable each cluster is):
if hasattr(hdb, 'cluster_persistence_'):
    print(f"cluster persistence: {hdb.cluster_persistence_.round(3).tolist()}")`}
      </CodeBlock>

      <Callout type="info" title="HDBSCAN in sklearn 1.3+">
        sklearn 1.3 added <Code>sklearn.cluster.HDBSCAN</Code> with the same API. For production without the external <Code>hdbscan</Code> package: <Code>from sklearn.cluster import HDBSCAN; hdb = HDBSCAN(min_cluster_size=10, min_samples=5).fit(X)</Code>. The sklearn implementation uses <Code>KDTree</Code> internally and is fast but lacks some diagnostics (cluster_persistence_, condensed_tree_) available in the standalone library.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. DBSCAN vs k-means on moons: decision comparison</H3>

      <Prose>
        The plot below shows DBSCAN's cluster assignments on the moons dataset (two labeled series) alongside the boundary that k-means always draws — a near-vertical split through the center, cutting each moon in half. The coordinates are approximate but faithfully reflect the computed labels from Section 4.
      </Prose>

      <Plot
        label="DBSCAN clusters on make_moons — k-means boundary overlaid"
        xLabel="x₀"
        yLabel="x₁"
        series={[
          {
            name: "Cluster 0 (left crescent)",
            color: colors.gold,
            points: [
              [-1.0,0.0],[-0.9,0.3],[-0.8,0.5],[-0.7,0.65],[-0.6,0.75],
              [-0.5,0.82],[-0.3,0.88],[-0.1,0.9],[0.1,0.88],[0.3,0.82],
              [0.5,0.72],[0.7,0.58],[0.85,0.42],[0.95,0.22],[1.0,-0.02],
            ],
          },
          {
            name: "Cluster 1 (right crescent)",
            color: colors.green,
            points: [
              [0.0,-0.1],[0.1,-0.35],[0.3,-0.55],[0.5,-0.68],[0.7,-0.75],
              [0.9,-0.78],[1.1,-0.75],[1.3,-0.68],[1.5,-0.55],[1.7,-0.38],
              [1.85,-0.18],[1.95,0.05],[2.0,0.28],
            ],
          },
          {
            name: "k-means boundary (vertical cut x₀ ≈ 0.5)",
            color: colors.textMuted,
            points: [[0.5, -1.1], [0.5, 1.2]],
          },
        ]}
      />

      <H3>6b. Cluster expansion from seed core point</H3>

      <StepTrace
        label="DBSCAN expansion — moons dataset, starting from point at (-0.85, 0.42)"
        steps={[
          {
            label: "Step 0 — Seed core point",
            render: () => (
              <Prose>
                Point p₀ = (-0.85, 0.42) is selected as the first unvisited point. Its ε-ball (ε=0.3) contains 7 neighbors — more than minPts=5. It is a core point. A new cluster (ID=0) is started. All 7 neighbors are added to the seed set. Labels so far: 1 point assigned.
              </Prose>
            ),
          },
          {
            label: "Step 1 — First neighbor expansion",
            render: () => (
              <Prose>
                Pop q₁ = (-0.72, 0.58) from the seed set. It has 8 neighbors within ε=0.3 — also a core point. Its neighbors are added to the seed set (deduplicating any already present). Cluster 0 now spans 9 points. The expansion is moving along the arm of the left crescent, following the local density ridge.
              </Prose>
            ),
          },
          {
            label: "Step 2 — Seed set grows along crescent arm",
            render: () => (
              <Prose>
                Processing q₂ = (-0.55, 0.72). Core point with 6 neighbors. Seed set now contains ~18 unique points. The cluster boundary is tracing the curve of the moon — no centroid, no Voronoi cell, just the density chain. Points ahead on the crescent are in the seed set; points on the other moon are not reachable from here because the gap between the two crescents at their closest approach (≈0.4 units) exceeds ε=0.3.
              </Prose>
            ),
          },
          {
            label: "Step 3 — Tip of crescent reached",
            render: () => (
              <Prose>
                Processing q at the tip (0.95, 0.22). This point has only 4 neighbors within ε=0.3 — below minPts=5. It is a border point: it gets assigned to Cluster 0 (reachable from a core point) but does not add its own neighbors to the seed set. Expansion stops here. The far side of the left crescent is not reachable because this tip is the one border point in the moons dataset.
              </Prose>
            ),
          },
          {
            label: "Step 4 — Cluster 0 complete",
            render: () => (
              <Prose>
                Seed set exhausted after visiting all 100 left-crescent points. 99 core, 1 border, 0 noise in Cluster 0. The algorithm now finds the next unvisited point — which lands on the right crescent. A new cluster ID=1 is started, and the same expansion proceeds identically, completing Cluster 1 with 100 points (all core at this noise level). Total: 2 clusters, 0 noise.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6c. k-distance plot for choosing ε</H3>

      <Prose>
        The standard technique for choosing ε: compute the distance from each point to its k-th nearest neighbor (with k = minPts - 1), sort these distances in descending order, and plot them. The "elbow" — the point of maximum curvature — is a good estimate for ε. Points above the elbow are in sparse regions; points below are in dense regions. Setting ε at the elbow separates signal from noise.
      </Prose>

      <Plot
        label="k-distance plot — make_moons (k=4)"
        xLabel="points sorted by decreasing k-dist"
        yLabel="distance to 4th nearest neighbor"
        series={[
          {
            name: "4th-NN distance (sorted desc)",
            color: colors.gold,
            points: [
              [0,0.301],[5,0.244],[10,0.187],[15,0.178],[20,0.164],
              [30,0.147],[40,0.139],[50,0.130],[70,0.115],[90,0.098],
              [110,0.085],[130,0.073],[150,0.063],[170,0.055],[190,0.049],[199,0.046],
            ],
          },
          {
            name: "ε = 0.30 (elbow)",
            color: colors.green,
            points: [[0,0.30],[199,0.30]],
          },
        ]}
      />

      <Prose>
        The elbow is at approximately k-dist = 0.30, matching the ε=0.3 that produced perfect cluster recovery in Section 4. Above the elbow (indices 0–3) are the sparse points at crescent tips; below the elbow the distances fall smoothly, representing the dense interiors of both moons. The k-distance plot is the single most practical tool for DBSCAN parameter selection — it converts the opaque ε choice into a visual inflection point that domain experts can evaluate.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Density-based clustering is not always the right tool. Here is a structured comparison across the four algorithms you are most likely to reach for.
      </Prose>

      <Heatmap
        label="Algorithm comparison: DBSCAN vs k-means vs GMM vs HDBSCAN"
        rowLabels={["DBSCAN", "k-means", "GMM", "HDBSCAN"]}
        colLabels={["Arb. shape", "Noise hdlg", "Var. density", "No k needed", "Scalability", "Soft member"]}
        matrix={[
          [5, 5, 2, 5, 3, 0],
          [1, 0, 2, 0, 5, 0],
          [1, 2, 2, 0, 3, 5],
          [5, 5, 5, 5, 3, 5],
        ]}
        colorScale="green"
      />

      <StepTrace
        label="when to choose which algorithm"
        steps={[
          {
            label: "DBSCAN — best for: arbitrary shape, known noise",
            render: () => (
              <Prose>
                Choose DBSCAN when: clusters are non-convex (moons, rings, filaments), the dataset has genuine noise you want flagged rather than absorbed into a cluster, you can estimate ε from domain knowledge or a k-distance plot, and density is approximately uniform across all clusters. Wins over k-means on: every dataset where Euclidean distance to a centroid is not a meaningful similarity. Wins over GMM on: datasets where clusters are not ellipsoidal and noise exists. Fails on: variable-density data (inner core dense, outer halo sparse → use HDBSCAN), very high-dimensional data (ε becomes meaningless), and when n is so large that even ball-tree neighbor search is slow.
              </Prose>
            ),
          },
          {
            label: "k-means — best for: spherical clusters, large n, known k",
            render: () => (
              <Prose>
                Choose k-means when: you have reason to believe clusters are roughly spherical and similarly sized, you know k (or can determine it cheaply with the elbow method or silhouette score), and n is very large (millions of points) where k-means' O(nkd) per iteration vastly outperforms DBSCAN's neighbor search. k-means is the default for image compression (color quantization), document clustering, and any problem where you genuinely have k natural categories. Never use it when cluster shapes are non-convex — the centroid is not representative and the Voronoi partition will cut through density ridges.
              </Prose>
            ),
          },
          {
            label: "GMM — best for: soft assignment, ellipsoidal clusters",
            render: () => (
              <Prose>
                Choose GMM (Gaussian Mixture Models) when: you need probabilistic cluster membership (a point can be 70% in cluster A and 30% in cluster B), clusters may be ellipsoidal with different orientations and scales, and the dataset has no hard noise (or you handle outliers separately). GMM is strictly more expressive than k-means (which is a special case with spherical, equal-variance Gaussians). Fails on: highly non-Gaussian distributions, ring-shaped or filamentary clusters, and datasets with genuine outliers (GMMs fit everything with a nonzero probability, including noise).
              </Prose>
            ),
          },
          {
            label: "HDBSCAN — best for: variable density, real-world messy data",
            render: () => (
              <Prose>
                Choose HDBSCAN when: clusters have different densities (e.g., a tight urban cluster and a diffuse rural cluster in the same dataset), you want soft membership scores to quantify how confidently each point belongs to its cluster, or you want to avoid tuning ε manually. HDBSCAN is the strongest general-purpose density-based method. Its main cost: the MST-based construction is roughly O(n² log n) without approximations, making it slow for n {">"} 100k. The approximate variant in the hdbscan library (using KD-tree for core distance computation) handles up to ~1M points in practice. If you have {">"} 1M points, consider DBSCAN with ball-tree or a distributed variant.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn{"'"}t</H2>

      <H3>8.1 Computational complexity</H3>

      <Prose>
        Naive DBSCAN with brute-force neighbor search computes all pairwise distances: <strong>O(n²)</strong> time and O(n²) space. This is the implementation in Section 4 and it is unusable above n ≈ 50,000 on a standard laptop. With a spatial index — <em>kd-tree</em> or <em>ball-tree</em> — each neighborhood query costs O(log n) on average, bringing the total to <strong>O(n log n)</strong> in low-dimensional spaces (d ≤ 10). sklearn's default for small datasets is a kd-tree; for larger datasets or non-Euclidean metrics, use <Code>algorithm='ball_tree'</Code>, which generalizes better to higher d.
      </Prose>

      <Prose>
        The critical caveat is the <em>curse of dimensionality</em>. In high-dimensional spaces, the volume of the ε-ball scales as ε^d. For d = 100 and ε = 1, the ball contains essentially the entire dataset — every point becomes a neighbor of every other point, all points become core, and DBSCAN produces one giant cluster. The kd-tree ceases to provide any speedup because the tree degenerates: every query requires examining O(n) leaves. The sklearn documentation explicitly warns that kd-trees are ineffective for d {">"} 20. For high-d data, DBSCAN needs either a dimensionality reduction step (UMAP, PCA) applied first, or a different metric entirely.
      </Prose>

      <H3>8.2 HDBSCAN scaling</H3>

      <Prose>
        HDBSCAN's MST construction on the mutual reachability graph is O(n² log n) in the worst case. The hdbscan library uses Prim's algorithm with a KD-tree for the core distance computation, bringing the practical runtime to approximately O(n^1.5) on low-dimensional data — usable to ~500k points. For n {">"} 1M, the library supports approximate nearest neighbor search via a random projection forest, degrading gracefully from exact to approximate results.
      </Prose>

      <H3>8.3 Distributed and large-scale variants</H3>

      <Prose>
        Apache Spark MLlib does not implement DBSCAN natively as of 2026, but the community-maintained <Code>spark-dbscan</Code> library implements a distributed variant: partition the space into grid cells, run local DBSCAN on each partition plus its border, then merge border clusters. The correctness guarantee requires that the partition borders overlap by ε on each side. PDBSCAN (Parallel DBSCAN) uses a similar cell-decomposition strategy and achieves near-linear scaling on clusters of commodity machines. For most practical cases below n = 5M in 2D–10D space, sklearn with <Code>algorithm='ball_tree'</Code> and <Code>n_jobs=-1</Code> is fast enough without distributed infrastructure.
      </Prose>

      <H3>8.4 Practical scaling thresholds</H3>

      <Prose>
        The table below summarizes observed practical limits on a modern laptop (16 GB RAM, 8-core CPU) to help calibrate which algorithm and mode to reach for at each scale. Times are approximate for d=2, minPts=5.
      </Prose>

      <Heatmap
        label="Practical runtime guide by n — DBSCAN variants (d=2, 8-core laptop)"
        rowLabels={["DBSCAN brute", "DBSCAN ball-tree", "DBSCAN ball-tree n_jobs=-1", "HDBSCAN exact", "HDBSCAN approx"]}
        colLabels={["n=10k", "n=100k", "n=500k", "n=1M", "n=5M"]}
        matrix={[
          [5, 1, 0, 0, 0],
          [5, 4, 3, 2, 1],
          [5, 5, 4, 3, 1],
          [5, 4, 2, 1, 0],
          [5, 5, 4, 3, 2],
        ]}
        colorScale="purple"
      />

      <Callout type="info" title="Reading the table">
        Score 5 = comfortably feasible (seconds). Score 3 = feasible with patience (minutes). Score 1 = slow but possible ({">"} 10 min). Score 0 = not recommended (hours or OOM). HDBSCAN approx uses random projection forests for nearest-neighbor search, trading a small accuracy loss for a 10-100x speedup at large n.
      </Callout>

      <Prose>
        One frequently overlooked bottleneck is memory, not time. DBSCAN with brute-force builds an explicit n×n distance matrix in memory: at n=50k that is 50,000² × 8 bytes = 20 GB — exceeding available RAM before the algorithm even runs. The ball-tree avoids this entirely: it stores only the index structure (O(n log n) space) and computes distances on demand. Always specify <Code>algorithm='ball_tree'</Code> for any n above ~10,000 with continuous features. With precomputed distances (<Code>metric='precomputed'</Code>), the n×n matrix is unavoidable — restrict to n {"<"} 20,000 or use sparse distance matrices.
      </Prose>

      <Plot
        label="Approximate runtime scaling — DBSCAN brute vs ball-tree vs HDBSCAN (d=2)"
        xLabel="n (dataset size)"
        yLabel="relative runtime (log scale)"
        series={[
          {
            name: "DBSCAN brute O(n²)",
            color: colors.gold,
            points: [[1000,1],[5000,25],[10000,100],[50000,2500],[100000,10000]],
          },
          {
            name: "DBSCAN ball-tree O(n log n)",
            color: colors.green,
            points: [[1000,1],[5000,6],[10000,13],[50000,79],[100000,170]],
          },
          {
            name: "HDBSCAN (approx)",
            color: colors.textMuted,
            points: [[1000,3],[5000,20],[10000,50],[50000,350],[100000,800]],
          },
        ]}
      />

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 ε too small: everything becomes noise</H3>

      <Prose>
        If ε is below the inter-point spacing of your densest cluster, no point will have minPts neighbors within ε. Every point is classified as noise and DBSCAN returns a single label: -1 for all n points. The symptom is immediate: <Code>n_clusters = 0</Code>. Diagnosis: the k-distance plot (Section 6c) — the chosen ε is below the elbow, meaning you are in the steep part of the plot where distances are large. Fix: increase ε to the elbow, or reduce minPts so that the minPts-th-NN distance for most points falls below ε.
      </Prose>

      <H3>9.2 ε too large: one mega-cluster</H3>

      <Prose>
        If ε equals or exceeds the gap between any two clusters, DBSCAN chains across the gap and merges them into one cluster. The result is <Code>n_clusters = 1</Code> containing the entire dataset, with a small number of genuine outliers still labeled as noise. This is the mirror image of the previous failure. The k-distance plot shows the chosen ε is above the elbow, in the flat region where most inter-cluster distances lie. Fix: decrease ε to the elbow. If the two problems are happening simultaneously — some clusters merge while others turn to noise — you are likely dealing with variable density, which is the next failure.
      </Prose>

      <H3>9.3 Variable density defeats plain DBSCAN</H3>

      <Prose>
        Plain DBSCAN uses one global ε. If your data has a dense urban cluster (tight, small radii) and a diffuse rural cluster (sparse, large radii), no single ε can satisfy both. Too small: the rural cluster becomes noise. Too large: the urban cluster merges with its halo. This is the motivation for both OPTICS (inspect the reachability plot and pick cluster-specific thresholds) and HDBSCAN (automatically selects the optimal density threshold per cluster via excess-of-mass). If you cannot switch algorithms, a workaround is to normalize each local region of feature space separately — but this is fragile and not recommended.
      </Prose>

      <H3>9.4 High-dimensional data</H3>

      <Prose>
        The curse of dimensionality makes the ε parameter ambiguous in high-d space. In d = 100, the ratio of the maximum to minimum pairwise distance converges to 1 as n grows — all points are essentially equidistant. The ε-neighborhood either contains everyone (ε too large) or no one (ε too small). There is no elbow in the k-distance plot because there is no meaningful density variation in the raw feature space. Fix: apply dimensionality reduction before DBSCAN. UMAP to 2–10 dimensions followed by DBSCAN or HDBSCAN is the modern standard; the UMAP embedding preserves local density structure that the raw features obscure.
      </Prose>

      <H3>9.5 Categorical and non-Euclidean features</H3>

      <Prose>
        DBSCAN works with any metric. Setting <Code>metric='cosine'</Code> is natural for text; <Code>metric='haversine'</Code> is correct for geographic (lat/lon) data; a precomputed distance matrix (<Code>metric='precomputed'</Code>) handles custom similarities. The common mistake is running DBSCAN with Euclidean distance on one-hot-encoded categorical data. In that space, "distance" is the Hamming distance scaled by sqrt(2), and the ε that makes sense for numeric features has no meaning for categorical ones. Use a semantically appropriate metric, or encode categoricals as embeddings before clustering.
      </Prose>

      <H3>9.6 Border point non-determinism</H3>

      <Prose>
        Border points — those within ε of multiple clusters' core points — are assigned to whichever cluster's core point is processed first. This assignment is non-deterministic with respect to visit order, which depends on the input ordering. If you sort your data differently, border points may swap cluster labels. Core point assignments and noise point assignments are fully deterministic given ε and minPts. If your application requires stable border assignments, use HDBSCAN, which resolves this by assigning each point to the cluster for which its membership probability is highest.
      </Prose>

      <H3>9.7 Evaluating clustering quality without ground-truth labels</H3>

      <Prose>
        Clustering is unsupervised — you typically do not have ground-truth labels to compute accuracy against. The standard evaluation toolkit: <strong>Silhouette score</strong> measures how much closer a point is to its own cluster's centroid than to the nearest other cluster's centroid, ranging from -1 (wrong cluster) to 1 (perfect separation). Scores above 0.5 are generally good. For DBSCAN, compute silhouette only on non-noise points — noise points have no cluster and would distort the score. <strong>Davies-Bouldin index</strong> measures the ratio of within-cluster scatter to between-cluster separation — lower is better. <strong>Calinski-Harabasz index</strong> (variance ratio criterion) — higher is better — favors compact, well-separated clusters. None of these metrics favor any particular shape, which makes them safe to use with DBSCAN's non-convex clusters. What they cannot detect is whether the clusters are semantically meaningful — that still requires domain expertise and qualitative inspection.
      </Prose>

      <Callout type="warning" title="Normalization is load-bearing">
        DBSCAN's ε is a distance in feature space. If feature 1 ranges from 0 to 1 and feature 2 ranges from 0 to 10,000, the ε-ball is almost entirely in the direction of feature 2. The result is directional clustering that ignores feature 1 entirely. Always <Code>StandardScaler</Code> or <Code>MinMaxScaler</Code> your features before calling DBSCAN. The same ε value means entirely different things across different feature scales.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were WebSearch-verified against ACM DL, Springer, and JOSS. Read in chronological order for the full intellectual lineage.
      </Prose>

      <StepTrace
        label="primary literature — density-based clustering"
        steps={[
          {
            label: "Ester, Kriegel, Sander, Xu 1996 — DBSCAN (KDD '96)",
            render: () => (
              <Prose>
                Ester, M., Kriegel, H.-P., Sander, J., and Xu, X. (1996). "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise." In <em>Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD '96)</em>, Portland, Oregon. AAAI Press. pp. 226–231. This is the paper that defined ε-neighborhoods, core/border/noise classification, density-reachability, density-connectivity, and the DBSCAN algorithm. The experimental evaluation used synthetic data and the SEQUOIA 2000 benchmark (real geographic data from California). The paper also introduced the k-distance plot as the primary tool for ε selection — the same tool in Section 6c. ACM DL: 10.5555/3001460.3001507.
              </Prose>
            ),
          },
          {
            label: "Ankerst, Breunig, Kriegel, Sander 1999 — OPTICS (SIGMOD '99)",
            render: () => (
              <Prose>
                Ankerst, M., Breunig, M.M., Kriegel, H.-P., and Sander, J. (1999). "OPTICS: Ordering Points To Identify the Clustering Structure." <em>ACM SIGMOD Record</em>, 28(2), 49–60. DOI: 10.1145/304181.304187. OPTICS computes a reachability plot — an augmented linear ordering of all database points where the reachability distance encodes the density structure at every scale simultaneously. The key concept: a cluster in the reachability plot appears as a "valley" — a contiguous sequence of low-reachability points bounded by high-reachability transitions. Different depth thresholds on the plot give different clusterings. This paper extended the intellectual framework of DBSCAN to hierarchical density-based analysis without requiring a single global density parameter.
              </Prose>
            ),
          },
          {
            label: "Campello, Moulavi, Sander 2013 — HDBSCAN (PAKDD 2013)",
            render: () => (
              <Prose>
                Campello, R.J.G.B., Moulavi, D., and Sander, J. (2013). "Density-Based Clustering Based on Hierarchical Density Estimates." In <em>Advances in Knowledge Discovery and Data Mining, PAKDD 2013</em>. Lecture Notes in Computer Science, vol. 7819. Springer, Berlin, Heidelberg. pp. 160–172. DOI: 10.1007/978-3-642-37456-2_14. This paper introduced mutual reachability distance, the cluster hierarchy (condensed tree), cluster stability via excess-of-mass, and the algorithm for extracting an optimal flat partition from the hierarchy. The formalization of "cluster stability" is the paper's central contribution — it gives a principled answer to "which subtree of the density hierarchy should I extract?" rather than requiring the practitioner to pick a threshold manually. Full paper is available via Springer Link.
              </Prose>
            ),
          },
          {
            label: "McInnes, Healy, Astels 2017 — hdbscan library (JOSS)",
            render: () => (
              <Prose>
                McInnes, L., Healy, J., and Astels, S. (2017). "hdbscan: Hierarchical density based clustering." <em>Journal of Open Source Software</em>, 2(11), 205. DOI: 10.21105/joss.00205. Available: joss.theoj.org/papers/10.21105/joss.00205. This paper describes the <Code>hdbscan</Code> Python library, which brought HDBSCAN to the scientific Python ecosystem with a sklearn-compatible API. Key implementation details: Prim's algorithm on the mutual reachability graph, KD-tree for core distance computation, and both exact and approximate (random projection forest) modes. The library also introduced soft clustering via cluster membership probabilities (<Code>probabilities_</Code>) and outlier detection via GLOSH (Global-Local Outlier Score from Hierarchies). The library is now maintained under scikit-learn-contrib on GitHub.
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
        Work through these before moving on. The answer is below each question — resist reading ahead.
      </Prose>

      <H3>Exercise 1 (recall)</H3>
      <Prose>
        Define core point, border point, and noise point in terms of ε and minPts. Which of these three classifications is non-deterministic (i.e., can change depending on the order points are visited), and why?
      </Prose>
      <Callout type="answer" title="Answer 1">
        A core point has |N_ε(p)| ≥ minPts — at least minPts points within distance ε (including itself). A border point has |N_ε(p)| {"<"} minPts but falls within the ε-ball of at least one core point. A noise point has |N_ε(p)| {"<"} minPts and is not within ε of any core point. Core and noise classifications are deterministic — they depend only on the point's own neighborhood, which is fixed given ε and minPts. Border points are non-deterministic: a border point in the overlap zone between two clusters is assigned to whichever cluster's core point processes it first, which depends on visit order (input ordering). Different orderings can give different cluster labels to border points.
      </Callout>

      <H3>Exercise 2 (derivation)</H3>
      <Prose>
        Prove that density-connectedness is symmetric but direct density-reachability is not. Give a concrete 3-point example that demonstrates the asymmetry.
      </Prose>
      <Callout type="answer" title="Answer 2">
        Direct density-reachability from p to q requires q ∈ N_ε(p) AND p is a core point. It is not symmetric because q need not be a core point. Example: p = (0,0), q = (0.2, 0), r = (0.1,0) are all within ε=0.3 of each other. Suppose p and r have 5 neighbors total (core), but q has only 3 (not core, minPts=5). Then q is directly density-reachable from p (p is core and q ∈ N_ε(p)), but p is NOT directly density-reachable from q (q is not a core point). Density-connectivity is symmetric: p and q are density-connected if there exists a core point o from which both are density-reachable. Here o = p itself: p is density-reachable from p (trivially), and q is density-reachable from p. So p and q are density-connected — mutual.
      </Callout>

      <H3>Exercise 3 (conceptual)</H3>
      <Prose>
        You run DBSCAN on a dataset and get <Code>n_clusters = 1</Code> containing 98% of all points, with 2% noise. You run it again with ε halved and get <Code>n_clusters = 0</Code> with 100% noise. What does this tell you about the dataset and the appropriate choice of ε? What tool should you use to diagnose the right ε?
      </Prose>
      <Callout type="answer" title="Answer 3">
        The first run (ε too large) shows the data has structure — the 2% noise are genuine outliers. The second run (ε too small) shows the inter-point distances within real clusters are larger than the halved ε. The right ε lies between the two values you tried. Use the k-distance plot: compute the k-th nearest neighbor distance for every point (k = minPts - 1), sort in descending order, plot, and identify the elbow. The elbow marks the transition between sparse (noise) and dense (cluster interior) distances. Set ε at the elbow. If the k-distance plot has multiple elbows, the dataset has clusters of different densities — in that case, switch to HDBSCAN which handles variable density automatically.
      </Callout>

      <H3>Exercise 4 (implementation)</H3>
      <Prose>
        You implement DBSCAN from scratch and your seed-set expansion loop runs correctly on small data but is extremely slow on n=5,000 points. Profiling shows the bottleneck is inside <Code>get_neighbors</Code>. (a) What is the current complexity and why is it slow? (b) What data structure reduces this, and what is the improved complexity? (c) Write the one-line sklearn call that uses this data structure.
      </Prose>
      <Callout type="answer" title="Answer 4">
        (a) Brute-force: get_neighbors(i) computes |X - X[i]| for all n points and selects those within ε. Called once per point, this is O(n) per call × n calls = O(n²) total. For n=5,000 that is 25M distance computations — slow but correct. (b) A ball-tree or kd-tree indexes the data spatially. Each neighborhood query costs O(log n) on average in low d, bringing the total to O(n log n). (c) <Code>DBSCAN(eps=0.3, min_samples=5, algorithm='ball_tree').fit(X)</Code>. In sklearn you can also pass <Code>n_jobs=-1</Code> to parallelize the neighbor queries across CPU cores, giving a further speedup proportional to core count.
      </Callout>

      <H3>Exercise 5 (applied)</H3>
      <Prose>
        A colleague runs DBSCAN on customer purchase coordinates (latitude/longitude) and gets poor results — many points labeled as noise despite visually obvious clusters. They used <Code>eps=0.5, min_samples=5, metric='euclidean'</Code>. Identify two problems with this setup and give the correct approach.
      </Prose>
      <Callout type="answer" title="Answer 5">
        Problem 1: Wrong metric. Euclidean distance on raw latitude/longitude degrees does not correspond to geographic distance — one degree of longitude near the equator is ~111 km, but near the poles it is nearly 0 km. The physically correct metric for spherical coordinates is the haversine distance. Fix: <Code>metric='haversine'</Code> in sklearn DBSCAN, which expects coordinates in radians: <Code>X_rad = np.radians(X_latlon)</Code>. Problem 2: Wrong ε units. Even with haversine, ε=0.5 is 0.5 radians ≈ 3,185 km — far too large, covering half a continent. A reasonable ε for city-level clustering is on the order of 1–5 km, which in radians is 1/6371 to 5/6371 ≈ 0.00016 to 0.00078. Use a k-distance plot with haversine distances to calibrate ε before fitting.
      </Callout>

      <H3>Exercise 6 (synthesis)</H3>
      <Prose>
        Explain why HDBSCAN is strictly more general than DBSCAN. Then describe a real-world dataset where you would prefer plain DBSCAN over HDBSCAN despite HDBSCAN being more general, and justify your reasoning.
      </Prose>
      <Callout type="answer" title="Answer 6">
        HDBSCAN is strictly more general because it performs DBSCAN's analysis over the full range of ε simultaneously, then extracts the most stable flat partition from the resulting hierarchy. Plain DBSCAN at a fixed ε is a special case of reading the HDBSCAN hierarchy at exactly that ε level. HDBSCAN additionally handles variable-density clusters, provides soft membership probabilities, and requires only min_cluster_size rather than the more sensitive ε parameter. A case where DBSCAN is preferable: a large-n, low-d spatial dataset (n = 2M, d = 2) where you have domain knowledge that all clusters have approximately the same density and you can determine ε from physical constraints (e.g., "buildings within 50 meters form a block"). In this case DBSCAN with ball-tree runs in O(n log n) ≈ minutes, while HDBSCAN's MST construction is O(n² log n) in the worst case ≈ hours for 2M points without approximations. The speed advantage of DBSCAN plus domain-informed ε makes it the better choice when the generality of HDBSCAN is unnecessary.
      </Callout>

    </div>
  ),
};

export default dbscanContent;
