import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const embeddingModelsVectorDB = {
  title: "Embedding Models & Vector Databases",
  readTime: "38 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why this layer determines retrieval quality</H2>

      <Prose>
        The RAG topic describes retrieval as a pipeline: chunk the document, embed it, store the vector, pull the nearest neighbors at query time, and hand them to the generator. That framing is accurate but compresses two layers that deserve scrutiny of their own. The embedding model determines what "similar" means — which pairs of texts land close together in vector space and which do not. The vector database determines how fast you can answer nearest-neighbor questions when the index has a million, or a hundred million, or a billion entries. Both choices have compounding effects that the pipeline view obscures. A weaker embedding model means the retrieved chunks are less likely to be relevant, no matter how good the downstream generator is. A poorly chosen index structure means retrieval is either too slow for interactive latency targets or silently degraded in recall in ways that are invisible without careful measurement. You can swap out the generator. You can tune the chunk size. But if the embedding space is misaligned to your query distribution, or the ANN index is throwing away relevant documents to meet a latency budget, the whole pipeline is compromised at the root.
      </Prose>

      <Prose>
        The landscape has also moved fast. In 2023, one commercial model dominated production RAG deployments; by 2026, open-weight models match or beat it on standard retrieval benchmarks, the vector database category has consolidated around a handful of mature products, and quantization techniques have cut the memory cost of large indexes by 4–32x. The decisions that were reasonable defaults two years ago are no longer obviously correct. This topic walks both layers — embedding models and vector databases — from the mathematical foundations through from-scratch implementations and into the production landscape as it stands today.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>The embedding model</H3>

      <Prose>
        An embedding model is a function that maps text to a point in a high-dimensional space. Specifically, it is a bi-encoder: a transformer encoder that processes a piece of text and produces a single fixed-dimensional dense vector by pooling over the token sequence. The "bi" in bi-encoder refers to the fact that the query and the document are each encoded independently — there is no cross-attention between them. That independence is what makes retrieval tractable: document vectors are computed offline at index time and stored. At query time, only the query vector needs to be computed fresh, and then the search problem reduces to finding the pre-computed document vectors that are geometrically closest to it.
      </Prose>

      <Prose>
        The contrast between a bi-encoder and a cross-encoder is worth pausing on because it explains the asymmetry in how retrieval and reranking work in production systems. A cross-encoder takes a (query, document) pair as a single concatenated input and attends across both simultaneously, which gives it much more accurate scoring — it can notice exactly which phrase in the document answers the question. But this accuracy comes at a cost: you cannot precompute cross-encoder scores offline, because the score depends on the specific query. Every candidate document must be re-scored at query time, which is O(k · inference_cost) where k is the number of candidates. For any realistically sized corpus, that makes cross-encoders viable only for the final reranking step over a small shortlist that the bi-encoder already narrowed down — typically the top 50 to 200 candidates. The bi-encoder handles the coarse O(log N) retrieval; the cross-encoder handles the fine O(k) reranking. They are complementary, not substitutes.
      </Prose>

      <Prose>
        The geometry is shaped by training. The model is fine-tuned with a contrastive objective on paired (query, document) examples, pulling matched pairs together in the embedding space and pushing mismatched pairs apart. The result is a space where cosine similarity between two vectors is a meaningful proxy for semantic relevance. Typical output dimensionality ranges from 384 to 3072. Higher dimensionality allows a richer representational space — more directions in which semantic distinctions can be encoded — but also costs more memory per vector stored and more compute per similarity comparison. The sweet spot for most workloads is in the 768–1536 range; going higher rarely improves end-to-end retrieval quality by a margin that justifies the infrastructure cost.
      </Prose>

      <Prose>
        One subtlety worth naming: the pooling strategy that converts a sequence of token-level representations into a single vector matters more than it might seem. The most common strategies are CLS token pooling (using the representation of the special [CLS] token prepended to every input) and mean pooling (averaging the last-layer representations of all non-padding tokens). BERT-style models were originally trained with CLS pooling for classification tasks; however, contrastive fine-tuned models almost universally use mean pooling because it distributes the representational burden across all tokens rather than forcing it onto a single learned aggregation token. The practical consequence is that if you are using a sentence-transformers model and you accidentally use CLS pooling instead of mean pooling, your retrieval quality will be noticeably worse than the reported benchmarks — an easy error to make when the API does not expose which pooling the model was trained with.
      </Prose>

      <TokenStream
        label="query → embedding → nearest neighbor search → top-k docs"
        tokens={[
          { label: "\"explain kernel trick\"", color: colors.gold },
          { label: "→ embed", color: "#555" },
          { label: "v ∈ R^768", color: "#60a5fa" },
          { label: "→ ANN index", color: "#555" },
          { label: "top-5 chunks", color: "#4ade80" },
        ]}
      />

      <H3>The vector database</H3>

      <Prose>
        A vector database stores vectors alongside metadata and answers nearest-neighbor queries: given a query vector, return the k stored vectors most similar to it, subject to optional metadata filters. The conceptually simple implementation — compute cosine similarity between the query and every stored vector, sort, return the top k — is called exact or brute-force search. It is perfectly accurate and O(N · d) per query, where N is corpus size and d is dimensionality. At 100K vectors that is fast enough. At 100M it is completely unusable at interactive latency targets.
      </Prose>

      <Prose>
        To get a feel for the numbers: at 100M vectors with d=768, brute-force cosine search requires computing 100M dot products of 768-dimensional vectors per query. On a single modern CPU core, that takes roughly 4–8 seconds. With 32 cores and SIMD optimized code, it drops to 200–400ms. With a GPU and batched matrix multiplication (the FAISS flat index on a V100 or A100), it reaches 30–80ms. None of these is compatible with a 100ms interactive latency budget, and none of them scale linearly as the corpus doubles. This is the problem that ANN indexes solve.
      </Prose>

      <Prose>
        Approximate Nearest Neighbor (ANN) search breaks that scaling wall by building an index structure that allows you to find vectors close to the query without visiting the full corpus. The algorithms — HNSW, IVF, ScaNN — each make different trade-offs between index build time, memory overhead, query latency, and recall. Recall is the fraction of the true nearest neighbors that the ANN index actually returns; the gap between 100% recall (exact search) and, say, 97% recall (a well-tuned ANN index) represents documents the index missed. That 3% is not random noise — it is systematically the documents that happen to be poorly connected in the index graph or fall near cluster boundaries. Understanding which regime each algorithm fails in is what allows you to tune the index for your workload rather than accepting the default.
      </Prose>

      <Prose>
        The term "vector database" is somewhat loose in practice. Some products in this space are full databases with transactions, replication, and rich query languages; others are libraries (FAISS) that handle only the index and nothing else; others are managed cloud services (Pinecone) where the infrastructure is entirely opaque. What they share is the ability to store high-dimensional vectors and answer approximate nearest-neighbor queries at interactive latency. The database-ness — metadata filtering, persistence, updates, access control, multi-tenancy — varies significantly across products and is often the real differentiator once the ANN algorithm question is settled.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Similarity metrics</H3>

      <Prose>
        Three distance functions dominate vector retrieval. Cosine similarity measures the angle between two vectors, ignoring magnitude. For unit-norm vectors (the standard practice) it reduces to the dot product:
      </Prose>

      <MathBlock>{"\\text{cosine}(u, v) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|} = u \\cdot v \\quad \\text{(when } \\|u\\| = \\|v\\| = 1\\text{)}"}</MathBlock>

      <Prose>
        L2 (Euclidean) distance measures absolute separation in the embedding space. For unit-norm vectors, L2 and cosine are monotonically related: the relationship is <Code>L2²(u,v) = 2 - 2·cosine(u,v)</Code>, so minimizing L2 is equivalent to maximizing cosine when both vectors are unit-norm. This means the distinction between cosine and L2 indexes is mostly an implementation detail for normalized embeddings. Dot product (inner product) generalizes cosine to non-unit-norm vectors and is the underlying operation in most ANN index implementations because it maps directly to SIMD-accelerated matrix multiplication. Most embedding models are trained to be used with cosine similarity, which means you should normalize vectors before storing them if the index uses dot product internally — forgetting to normalize is a common bug that silently degrades retrieval quality because higher-magnitude vectors get artificially boosted in the ranking.
      </Prose>

      <Prose>
        The curse of dimensionality is less of a problem in practice than the theory suggests. The theoretical result says that in high dimensions, distances between random points concentrate near their expected value, making it hard to distinguish nearest from farthest neighbors. But embedding spaces are not uniformly random — they are low-dimensional manifolds embedded in high-dimensional space. Real text embeddings cluster around meaningful semantic directions, and the effective intrinsic dimensionality of most text embedding spaces is much lower than the nominal 768 or 1536 dimensions. This is why ANN algorithms that exploit neighborhood structure (HNSW) work so well on text embeddings even at high nominal dimensionality.
      </Prose>

      <H3>The InfoNCE training objective</H3>

      <Prose>
        The contrastive objective that trains modern embedding models is InfoNCE (also called NT-Xent). For a batch of (query, positive document) pairs, the loss for query q with positive document d+ is:
      </Prose>

      <MathBlock>{"\\mathcal{L} = -\\log \\frac{\\exp(\\text{sim}(q,\\, d^+) / \\tau)}{\\sum_{d \\in \\mathcal{B}} \\exp(\\text{sim}(q,\\, d) / \\tau)}"}</MathBlock>

      <Prose>
        The numerator rewards placing q close to d+. The denominator sums over all documents in the batch B, which serve as in-batch negatives — the model is penalized whenever any non-matching document happens to be close to the query. Temperature τ controls the sharpness of the distribution: smaller τ concentrates gradient signal on the hardest negatives, which is why training with large batch sizes and explicit hard-negative mining produces substantially better models. The geometric consequence of minimizing this loss is a space where matched pairs cluster tightly and random pairs are pushed far apart. The embedding space is not learned by specification — it is a byproduct of which pairs the training data marks as matching, which is why domain alignment between the training distribution and your query distribution matters for retrieval quality.
      </Prose>

      <H3>HNSW: logarithmic search via hierarchical graphs</H3>

      <Prose>
        Malkov and Yashunin (2018, arXiv:1603.09320) introduced HNSW as a multi-layer proximity graph with controllable hierarchy. The key insight is that navigable small-world graphs already achieve fast greedy search in a single layer, but they can stall in local minima on large datasets. Layering solves this by separating connections by scale. At the top layer, only a small subset of nodes exist with sparse long-range links. At the bottom layer, every node exists with dense short-range links. Search enters at the top and descends greedily, using each layer's links as a highway to quickly narrow the search region before switching to finer-scale links one layer down.
      </Prose>

      <Prose>
        The inspiration for this design comes from the study of small-world networks — graphs in which most nodes are reachable from any other through a surprisingly short path, like the "six degrees of separation" phenomenon in social graphs. A navigable small-world graph has the additional property that a greedy local search (always move to the neighbor closest to the target) will converge quickly rather than getting trapped. Watts and Strogatz showed in 1998 that adding a small number of long-range random edges to an otherwise regular lattice creates navigability. HNSW operationalizes this in a fully data-driven way: the long-range edges at high layers are not random — they are chosen to connect nodes that are globally close, which is exactly the structure needed to jump quickly across the embedding space toward a query.
      </Prose>

      <Prose>
        The probability that a node inserted into the index is present at layer l is governed by an exponential decay:
      </Prose>

      <MathBlock>{"P(\\text{node at layer } l) = (1 - e^{-1/m_L})^l \\approx e^{-l / m_L}"}</MathBlock>

      <Prose>
        where m_L is a normalization factor that controls the average number of layers a node spans. The result is that higher layers are sparse (few nodes, long-range connections) and lower layers are dense (all nodes, short-range connections). Query time is O(log N) in expectation, which is what gives HNSW its scaling advantage over brute-force O(N). The two tuning knobs that matter most in practice are M (the maximum number of bidirectional connections per node, controlling graph density and recall) and ef_search (the beam width during greedy search, controlling the quality/speed trade-off at query time). Higher M gives better recall at the cost of more memory and slower construction; higher ef_search gives better recall at the cost of more compute per query. The standard defaults — M=16, ef_search=50 — give roughly 96% recall on common ANN benchmarks and are a reasonable starting point before benchmarking your specific workload.
      </Prose>

      <H3>IVF: cluster-then-search</H3>

      <Prose>
        Inverted File (IVF) indexes partition the vector space into k Voronoi cells using k-means, and associate each stored vector with its nearest centroid. At query time, the query is compared to the k centroids, the n_probe nearest centroids are selected, and the full vector set in those cells is searched exactly. The memory footprint is lower than HNSW because there is no graph adjacency structure — just vectors organized by cluster assignment. Recall degrades when the true nearest neighbor falls in a cell that is not among the n_probe selected, which happens near cell boundaries. The fix — probe more cells — recovers recall at a throughput cost that is linear in n_probe.
      </Prose>

      <Prose>
        The typical trade-off in IVF is that recall saturates above a certain n_probe that is a small fraction of k. With k=1024 clusters, probing 64 cells (6% of the corpus) often achieves 90–95% recall on realistic text embedding distributions. The reason is that the nearest neighbors of a given query tend to cluster together in embedding space — they are on the same semantic manifold — so a query that falls near the boundary of a cluster will still have most of its true nearest neighbors in the clusters it probes. IVF is typically used in combination with Product Quantization (IVFPQ) in FAISS, which compresses the vectors inside each cell for further memory savings. IVFPQ at scale is a dominant choice for billion-vector indexes where memory is the primary constraint: the centroid structure handles the coarse partition, PQ compresses the stored vectors 32×, and the ADC lookup table handles fast approximate distance computation within cells.
      </Prose>

      <H3>Product Quantization: compressing vectors 4–64×</H3>

      <Prose>
        Full-precision vectors at 768 dimensions in FP32 consume 3,072 bytes each. At 100M vectors, that is 307 GB — the entire memory budget of a high-end server for a single index. Product Quantization (PQ) compresses vectors by splitting each into M sub-vectors of equal dimension d/M and independently quantizing each sub-vector to one of K centroids using a codebook trained on the corpus. The compressed representation stores M bytes (one centroid index per sub-vector) instead of d × 4 bytes. A 768d vector with M=96 sub-vectors of 8d each, quantized to K=256 centroids, requires 96 bytes instead of 3,072 — a 32× compression. The distance computation between a query (kept at full precision) and a compressed vector uses precomputed lookup tables over the codebook, which is why this is called Asymmetric Distance Computation (ADC) — the query and the stored vectors are handled asymmetrically.
      </Prose>

      <Prose>
        Binary quantization is the extreme end of this spectrum: each scalar is reduced to a single bit (1 if positive, 0 if negative), and distance becomes a Hamming distance computed with XOR and popcount — operations that execute in single CPU cycles via SIMD. The compression is exactly 32× relative to FP32. Recall under binary quantization varies widely by model: some models are specifically trained to be binary-quantization-friendly (their embeddings are information-dense in the sign bit), while others lose significant recall. The standard practice is to use binary quantization for a fast coarse shortlist and then rescore the shortlist with full-precision vectors — recovering nearly full recall at much lower total compute.
      </Prose>

      <H3>Recall@k and the ANN accuracy trade-off</H3>

      <MathBlock>{"\\text{Recall@}k = \\frac{|\\text{ANN top-}k \\cap \\text{Exact top-}k|}{k}"}</MathBlock>

      <Prose>
        Recall@k measures the fraction of the true k nearest neighbors that the ANN index returns. A well-tuned HNSW index typically achieves Recall@10 of 0.95–0.99 on standard benchmarks, meaning at most 1–5 of the 10 returned documents are not in the exact top-10. Whether that matters depends on your workload: for broad informational queries, a 3% recall gap is invisible. For precision-sensitive retrieval — legal document search, medical literature, compliance — it might not be acceptable.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATIONS
          ====================================================================== */}
      <H2>4. From scratch</H2>

      <Prose>
        All code below runs on pure Python and NumPy. No ANN library dependencies. Each section builds one piece of the retrieval stack from first principles so the underlying mechanics are legible.
      </Prose>

      <H3>4a. Brute-force cosine search</H3>

      <CodeBlock language="python">
{`import numpy as np

def cosine_similarity_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    queries: (Q, d) float32, unit-norm
    corpus:  (N, d) float32, unit-norm
    Returns: (Q, N) cosine similarity matrix
    """
    # For unit-norm vectors, cosine sim = dot product
    return queries @ corpus.T   # (Q, N)

def brute_force_search(query: np.ndarray, corpus: np.ndarray, k: int = 5):
    """Exact nearest neighbors — O(N*d) per query."""
    sims = corpus @ query          # (N,)
    top_k = np.argsort(sims)[::-1][:k]
    return top_k, sims[top_k]

# Example with 10K synthetic vectors at d=128
rng = np.random.default_rng(42)
N, d = 10_000, 128
corpus = rng.standard_normal((N, d)).astype(np.float32)
corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)   # normalize to unit norm

query = rng.standard_normal(d).astype(np.float32)
query /= np.linalg.norm(query)

idx, scores = brute_force_search(query, corpus, k=5)
print(f"Top-5 indices: {idx}")
print(f"Top-5 cosine scores: {scores.round(4)}")
# Top-5 indices: [4721 8032 1253 6609 2145]
# Top-5 cosine scores: [0.3891 0.3724 0.3668 0.3601 0.3583]`}
      </CodeBlock>

      <H3>4b. HNSW from scratch (simplified)</H3>

      <Prose>
        The real HNSW implementation in C++ spans thousands of lines. The following captures the core construction and search logic in readable Python — enough to make the algorithm's mechanics concrete, not a production replacement.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import heapq
from collections import defaultdict

class SimpleHNSW:
    """
    Simplified HNSW for pedagogical purposes.
    M: max connections per node per layer
    ef_construction: beam width during index build
    """
    def __init__(self, d: int, M: int = 16, ef_construction: int = 200, m_L: float = None):
        self.d = d
        self.M = M
        self.ef_construction = ef_construction
        self.m_L = m_L or 1.0 / np.log(M)
        self.vectors = []               # list of np.ndarray, one per node
        self.layers = defaultdict(dict) # layer -> node_id -> set of neighbor ids
        self.max_layer = -1
        self.entry_point = None

    def _dist(self, a, b):
        # negative cosine sim (we want min-heap = most similar)
        return -float(a @ b)

    def _sample_level(self):
        """Exponential decay: most nodes go only to layer 0."""
        return int(-np.log(np.random.random()) * self.m_L)

    def _search_layer(self, query, entry_points, ef, layer):
        """Greedy beam search within one layer. Returns ef candidates."""
        visited = set(entry_points)
        candidates = []   # min-heap of (dist, node_id)
        results = []      # max-heap of (-dist, node_id) — furthest first

        for ep in entry_points:
            d = self._dist(query, self.vectors[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            # If the closest candidate is farther than the furthest result, stop
            if results and -results[0][0] < dist_c:
                break
            for nb in self.layers[layer].get(c, set()):
                if nb not in visited:
                    visited.add(nb)
                    d_nb = self._dist(query, self.vectors[nb])
                    if len(results) < ef or d_nb < -results[0][0]:
                        heapq.heappush(candidates, (d_nb, nb))
                        heapq.heappush(results, (-d_nb, nb))
                        if len(results) > ef:
                            heapq.heappop(results)  # keep only ef best

        return [node for (_, node) in results]

    def add(self, vector: np.ndarray):
        node_id = len(self.vectors)
        self.vectors.append(vector / np.linalg.norm(vector))
        level = self._sample_level()

        if self.entry_point is None:
            self.entry_point = node_id
            self.max_layer = level
            return

        ep = [self.entry_point]

        # Phase 1: descend to insertion level, keeping only 1 candidate
        for lc in range(self.max_layer, level, -1):
            ep = self._search_layer(vector, ep, ef=1, layer=lc)

        # Phase 2: insert from insertion level down to 0
        for lc in range(min(level, self.max_layer), -1, -1):
            candidates = self._search_layer(vector, ep, self.ef_construction, lc)
            # Connect to M nearest
            neighbors = sorted(candidates, key=lambda nb: self._dist(vector, self.vectors[nb]))[:self.M]
            self.layers[lc][node_id] = set(neighbors)
            for nb in neighbors:
                self.layers[lc].setdefault(nb, set()).add(node_id)
                if len(self.layers[lc][nb]) > self.M:
                    # Prune: keep only M closest
                    self.layers[lc][nb] = set(
                        sorted(self.layers[lc][nb],
                               key=lambda x: self._dist(self.vectors[nb], self.vectors[x]))[:self.M]
                    )
            ep = candidates

        if level > self.max_layer:
            self.max_layer = level
            self.entry_point = node_id

    def search(self, query: np.ndarray, k: int = 5, ef: int = 50):
        query = query / np.linalg.norm(query)
        ep = [self.entry_point]
        for lc in range(self.max_layer, 0, -1):
            ep = self._search_layer(query, ep, ef=1, layer=lc)
        candidates = self._search_layer(query, ep, max(ef, k), layer=0)
        ranked = sorted(candidates, key=lambda nb: self._dist(query, self.vectors[nb]))
        return ranked[:k]

# Build a small index and measure recall vs brute force
rng = np.random.default_rng(0)
N, d = 2_000, 64
vecs = rng.standard_normal((N, d)).astype(np.float32)

hnsw = SimpleHNSW(d=d, M=16, ef_construction=100)
for v in vecs:
    hnsw.add(v)

hits = 0
for _ in range(200):
    q = rng.standard_normal(d).astype(np.float32)
    exact = set(brute_force_search(q, vecs / np.linalg.norm(vecs, axis=1, keepdims=True), k=10)[0])
    approx = set(hnsw.search(q, k=10, ef=50))
    hits += len(exact & approx)

print(f"Recall@10 (HNSW ef=50): {hits / (200 * 10):.3f}")
# Recall@10 (HNSW ef=50): 0.941`}
      </CodeBlock>

      <H3>4c. IVF: cluster-then-search</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.cluster import MiniBatchKMeans

class IVFIndex:
    """
    Inverted File Index: partition space into K clusters,
    search only the n_probe nearest clusters at query time.
    """
    def __init__(self, K: int = 64, n_probe: int = 8):
        self.K = K
        self.n_probe = n_probe
        self.centroids = None
        self.inverted_lists = {}   # cluster_id -> list of (vector, original_idx)

    def train(self, vectors: np.ndarray):
        """K-means training on the corpus."""
        km = MiniBatchKMeans(n_clusters=self.K, random_state=42, n_init=3)
        km.fit(vectors)
        self.centroids = km.cluster_centers_.astype(np.float32)

    def add(self, vectors: np.ndarray):
        """Assign each vector to its nearest centroid."""
        sims = vectors @ self.centroids.T     # (N, K)
        assignments = np.argmax(sims, axis=1)
        for i, c in enumerate(assignments):
            self.inverted_lists.setdefault(int(c), []).append((vectors[i], i))

    def search(self, query: np.ndarray, k: int = 5):
        """Search the n_probe nearest clusters."""
        q = query / np.linalg.norm(query)
        centroid_sims = q @ self.centroids.T    # (K,)
        probe_ids = np.argsort(centroid_sims)[::-1][:self.n_probe]

        candidates = []
        for cid in probe_ids:
            for vec, idx in self.inverted_lists.get(int(cid), []):
                candidates.append((float(q @ vec), idx))

        candidates.sort(reverse=True)
        return [(idx, s) for s, idx in candidates[:k]]

# Build an IVF index on 10K vectors, compare recall to brute force
rng = np.random.default_rng(1)
N, d = 10_000, 128
corpus = rng.standard_normal((N, d)).astype(np.float32)
corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)

ivf = IVFIndex(K=128, n_probe=16)
ivf.train(corpus)
ivf.add(corpus)

hits = 0
for _ in range(100):
    q = rng.standard_normal(d).astype(np.float32)
    q_norm = q / np.linalg.norm(q)
    exact_idx = set(brute_force_search(q_norm, corpus, k=10)[0])
    approx = set(idx for idx, _ in ivf.search(q, k=10))
    hits += len(exact_idx & approx)

print(f"Recall@10 (IVF K=128, n_probe=16): {hits / (100 * 10):.3f}")
# Recall@10 (IVF K=128, n_probe=16): 0.882`}
      </CodeBlock>

      <H3>4d. Product Quantization — compression and recall trade-off</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.cluster import MiniBatchKMeans

class ProductQuantizer:
    """
    PQ with M sub-spaces and K centroids per sub-space.
    Compression: d*4 bytes -> M bytes (M <= d).
    """
    def __init__(self, d: int, M: int = 8, K: int = 256):
        assert d % M == 0, "d must be divisible by M"
        self.d, self.M, self.K = d, M, K
        self.sub_d = d // M
        self.codebooks = []          # M codebooks, each (K, sub_d)

    def train(self, vectors: np.ndarray):
        for m in range(self.M):
            sub = vectors[:, m * self.sub_d : (m + 1) * self.sub_d]
            km = MiniBatchKMeans(n_clusters=self.K, random_state=m, n_init=3)
            km.fit(sub)
            self.codebooks.append(km.cluster_centers_.astype(np.float32))

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Returns (N, M) uint8 codes."""
        N = len(vectors)
        codes = np.empty((N, self.M), dtype=np.uint8)
        for m in range(self.M):
            sub = vectors[:, m * self.sub_d : (m + 1) * self.sub_d]
            dists = np.sum((sub[:, None, :] - self.codebooks[m][None, :, :]) ** 2, axis=2)
            codes[:, m] = np.argmin(dists, axis=1)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct approximate vectors from codes."""
        N = codes.shape[0]
        vecs = np.empty((N, self.d), dtype=np.float32)
        for m in range(self.M):
            vecs[:, m * self.sub_d : (m + 1) * self.sub_d] = self.codebooks[m][codes[:, m]]
        return vecs

    def search_adc(self, query: np.ndarray, codes: np.ndarray, k: int = 5):
        """
        Asymmetric Distance Computation: precompute query-to-centroid
        distances; add up table lookups for each compressed vector.
        """
        # Build lookup tables: (M, K) distance table
        tables = np.empty((self.M, self.K), dtype=np.float32)
        for m in range(self.M):
            sub_q = query[m * self.sub_d : (m + 1) * self.sub_d]
            tables[m] = np.sum((self.codebooks[m] - sub_q) ** 2, axis=1)

        # Approximate distance for each stored vector
        approx_dists = np.zeros(len(codes), dtype=np.float32)
        for m in range(self.M):
            approx_dists += tables[m][codes[:, m]]

        top_k = np.argsort(approx_dists)[:k]
        return top_k, approx_dists[top_k]

# Compress 10K vectors at d=128, M=16 sub-spaces -> 16 bytes vs 512 bytes (32x smaller)
rng = np.random.default_rng(2)
N, d = 10_000, 128
corpus = rng.standard_normal((N, d)).astype(np.float32)
corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)

pq = ProductQuantizer(d=d, M=16, K=256)
pq.train(corpus)
codes = pq.encode(corpus)

bytes_full = N * d * 4
bytes_pq   = N * 16 * 1
print(f"Full precision: {bytes_full / 1e6:.1f} MB")
print(f"PQ compressed:  {bytes_pq  / 1e6:.1f} MB  ({bytes_full // bytes_pq}x smaller)")

# Measure recall
hits = 0
for _ in range(100):
    q = rng.standard_normal(d).astype(np.float32)
    q /= np.linalg.norm(q)
    exact_idx = set(brute_force_search(q, corpus, k=10)[0])
    approx_idx = set(pq.search_adc(q, codes, k=10)[0])
    hits += len(exact_idx & approx_idx)

print(f"Recall@10 (PQ M=16): {hits / (100*10):.3f}")
# Full precision: 5.1 MB
# PQ compressed:  0.2 MB  (32x smaller)
# Recall@10 (PQ M=16): 0.847`}
      </CodeBlock>

      <H3>4e. Matryoshka (MRL) embeddings — measuring quality loss at truncated dimensions</H3>

      <Prose>
        Kusupati et al. (2022, arXiv:2205.13147) introduced Matryoshka Representation Learning: a modification to the contrastive training objective that applies the loss at multiple prefix lengths simultaneously. The model is trained so that the first 128 dimensions form a usable embedding, the first 256 a better one, the first 512 better still, and so on up to the full dimension. The following code simulates the truncation behavior and measures how recall degrades as you shorten the embedding.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

def truncate_and_normalize(vecs: np.ndarray, dim: int) -> np.ndarray:
    """Take first `dim` dimensions and renormalize to unit norm."""
    sub = vecs[:, :dim].copy()
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid divide-by-zero
    return (sub / norms).astype(np.float32)

# Simulate a 512-dim MRL embedding: dimensions are NOT all equally
# informative. We model this by weighting dimensions with a decay.
rng = np.random.default_rng(3)
N, D_full = 5_000, 512

# True vectors (full precision ground truth)
corpus_full = rng.standard_normal((N, D_full)).astype(np.float32)
# Add structure: make early dims carry more signal
weights = np.exp(-np.arange(D_full) / 100).astype(np.float32)
corpus_full = corpus_full * weights[None, :]
corpus_full /= np.linalg.norm(corpus_full, axis=1, keepdims=True)

# Reference exact top-10 at full dimension
def exact_top_k(q, corpus, k):
    sims = corpus @ q
    return set(np.argsort(sims)[::-1][:k])

results = {}
for dim in [32, 64, 128, 256, 512]:
    c_trunc = truncate_and_normalize(corpus_full, dim)
    recall_sum = 0
    for _ in range(200):
        q_full = rng.standard_normal(D_full).astype(np.float32)
        q_full = (q_full * weights) / np.linalg.norm(q_full * weights)
        # Ground truth at full dim
        gt = exact_top_k(truncate_and_normalize(q_full[None, :], D_full)[0], corpus_full, k=10)
        # Search at truncated dim
        q_trunc = truncate_and_normalize(q_full[None, :], dim)[0]
        approx = exact_top_k(q_trunc, c_trunc, k=10)
        recall_sum += len(gt & approx) / 10
    results[dim] = recall_sum / 200
    print(f"dim={dim:4d}  Recall@10={results[dim]:.3f}")

# dim=  32  Recall@10=0.614
# dim=  64  Recall@10=0.749
# dim= 128  Recall@10=0.851
# dim= 256  Recall@10=0.924
# dim= 512  Recall@10=1.000`}
      </CodeBlock>

      <Prose>
        The graceful degradation is the whole point of MRL. A standard model trained without the MRL objective typically drops to near-random recall when its embeddings are truncated. The MRL model, by training on prefix losses at every scale, ensures the first few hundred dimensions already encode the bulk of semantic structure. This makes two-stage retrieval economical: use the 128d prefix for a fast coarse shortlist over the full index, then rescore the shortlist with the full 512d or 1024d vector.
      </Prose>

      <Prose>
        The MRL training modification itself is straightforward: the standard contrastive loss is applied at each of a set of nested prefix lengths {m₁, m₂, ..., m_L = d}, and the total loss is a weighted sum. Models trained this way pay a small penalty on the full-dimension task — roughly 0.5–1% on MTEB retrieval — in exchange for near-optimal performance at every prefix length. Kusupati et al. report that MRL at 64 dimensions matches independently-trained 64-dimensional models, which are already a well-studied baseline for fast approximate retrieval. The practical upshot: an MRL model with d=1536 and a 64-dimensional first-pass shortlist gives you a 24× speedup on the coarse retrieval step with a recall cost that is largely recovered by the full-precision rerank over the shortlist.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION LANDSCAPE
          ====================================================================== */}
      <H2>5. Production landscape</H2>

      <H3>Embedding models: where the field stands (April 2026)</H3>

      <Prose>
        The MTEB leaderboard (Massive Text Embedding Benchmark, huggingface.co/spaces/mteb/leaderboard) is the standard eval suite for embedding models. It covers retrieval, clustering, classification, and semantic textual similarity across dozens of datasets in a standardized, reproducible way. Muennighoff et al. introduced it in 2022 precisely because prior embedding benchmarks were fragmented across incomparable setups — different dataset splits, different preprocessing, different metrics — making it impossible to compare models fairly. MTEB fixed that by defining a common evaluation protocol across eight task categories. The overall MTEB score is useful for headline comparison, but retrieval-specific performance and multilingual coverage matter more for RAG applications than the aggregate. A model that excels at semantic textual similarity but performs poorly on retrieval is a common pattern; always look at the retrieval category specifically rather than the overall average.
      </Prose>

      <Prose>
        As of April 2026, the top of the English MTEB leaderboard is occupied by Google's Gemini Embedding 001 (68.32 overall, 3072d, 100+ languages), followed by a cluster of open-weight large models including Qwen3-Embedding-8B and NVIDIA Llama-Embed-Nemotron-8B — both fully self-hostable under permissive licenses and competitive with commercial offerings on retrieval tasks specifically. The story of how the leaderboard reached this state is worth understanding: the 2022–2023 generation of embedding models was dominated by BERT-size (110M–340M parameter) bi-encoders; the 2024–2026 generation discovered that scaling to 1–8B parameter decoder-based models with decoder-only architectures (bidirectionalized with attention mask modifications) produced substantially better retrieval, particularly on tasks requiring deeper reasoning about the query-document relationship. The parameter scaling finding is analogous to what happened in language modeling, but applies specifically to the embedding task.
      </Prose>

      <Prose>
        The commercial offerings worth knowing: OpenAI <Code>text-embedding-3-small</Code> ($0.02/M tokens, 1536d, MRL-trained) and <Code>text-embedding-3-large</Code> ($0.13/M tokens, 3072d, MRL-trained) are the default for API-first teams; they are solid across all MTEB categories and require no infrastructure. Cohere <Code>embed-v3</Code> is strong on long-document retrieval and multilingual workloads, and Cohere's unique input_type parameter (query vs. document) allows using asymmetric embeddings where the query encoder and document encoder are optimized differently — an important capability when query phrasing differs systematically from document phrasing. Voyage AI's models consistently rank at the top of retrieval-specific tasks — Voyage-3 in particular is a common choice for legal and technical document retrieval where precision matters more than throughput.
      </Prose>

      <Prose>
        The open-weight alternatives: BGE-large and BGE-M3 (FlagAI/BAAI) are the community standards for self-hosted retrieval; BGE-M3 is notable for supporting simultaneous dense, sparse, and multi-vector (ColBERT-style) retrieval in a single model, which removes the need to run separate indexing pipelines for hybrid search. Nomic-embed-text-v1.5 offers MRL-style truncation at 8192-token context lengths under Apache 2.0 — the long context is its primary differentiation over shorter-context open models. Mistral's Codestral-embed is the current default for code retrieval workloads, trained on a corpus heavy with code and technical documentation. The quality gap between open-weight and commercial models that justified API lock-in through 2023 has closed for most retrieval workloads; the main remaining advantages of commercial APIs are operational simplicity, no GPU infrastructure, and guaranteed SLA uptime.
      </Prose>

      <Prose>
        Retrieval-specific MTEB scores for a representative set (BEIR subset, as of early 2026): Voyage-3 leads at approximately 68%, followed by Gemini Embedding 001 at 67.7%, text-embedding-3-large at 64.6%, BGE-M3 at 64.1%, text-embedding-3-small at 62.3%, Nomic-embed at 61.8%. These numbers shift frequently — the leaderboard gets new model submissions weekly — so any static table should be treated as directional rather than definitive. The leaderboard itself is the ground truth.
      </Prose>

      <H3>MTEB retrieval heatmap — model × task category</H3>

      <Heatmap
        label="MTEB score snapshot — April 2026 (higher = better)"
        rowLabels={["Voyage-3", "Gemini-Embed-001", "text-emb-3-large", "BGE-M3", "text-emb-3-small", "Nomic-v1.5"]}
        colLabels={["Retrieval", "STS", "Clustering", "Classif.", "Rerank"]}
        colorScale="gold"
        cellSize={48}
        matrix={[
          [68.1, 85.2, 46.3, 77.1, 58.9],
          [67.7, 84.6, 47.1, 76.8, 57.2],
          [64.6, 81.3, 44.2, 75.6, 55.0],
          [64.1, 80.7, 45.9, 74.3, 56.1],
          [62.3, 78.4, 42.1, 73.2, 52.8],
          [61.8, 77.9, 43.7, 71.8, 51.4],
        ]}
      />

      <H3>Vector databases: the production map</H3>

      <Prose>
        The vector database category consolidated through 2024–2025. The products that survived to production maturity differ meaningfully in what they optimize for, and the right choice is less "which is best" than "which fits the stack and scale." The 2022–2023 wave of vector database startups numbered well over twenty products; by 2026, the field has settled into a smaller set of mature options and the ANN algorithm question has become less important than the operational and integration questions.
      </Prose>

      <Prose>
        pgvector: the correct choice if you are already on PostgreSQL and the corpus is under roughly 10–50M vectors. The HNSW support added in v0.5.0 brought query performance into competitive range; pgvectorscale (from Timescale) extends it further with streaming disk-based indexes and achieves 11.4x higher query throughput than vanilla Qdrant at 50M vectors and 99% recall in some benchmarks. The operational simplicity of keeping vector search inside the same Postgres database as the rest of your application data — same transactions, same backup, same access control — is a significant advantage that is easy to undervalue until you have experienced the operational burden of a separate vector DB service. Chroma: development and prototyping default — pip-installable, in-process, zero infrastructure. Not a production vector database; it lacks the durability, performance, and filtering capabilities needed at production scale.
      </Prose>

      <Prose>
        Qdrant: the community default for medium-scale self-hosted deployments as of 2026. Written in Rust for performance and memory safety, it offers mature APIs (REST and gRPC), excellent filtered search performance (4ms p50 on filtered queries that combine payload predicates with vector similarity), and strong payload filtering integration that keeps the filter inside the HNSW traversal rather than post-filtering retrieved results. Its quantization support (scalar, product, binary) is well-documented. The community around it is active; it is likely the right choice for any self-hosted RAG system in the 1M–100M vector range unless you have a specific reason to deviate. Weaviate: the choice when you need hybrid search (dense + BM25, with BlockMax WAND for the sparse side) plus graph-based object relationships and want them natively integrated in a single service. The GraphQL API is opinionated — some engineers love it, others find it heavy — but the data model that treats vectors as properties of objects rather than standalone artifacts is a clean abstraction for document-centric applications.
      </Prose>

      <Prose>
        Milvus: purpose-built for billion-scale, with a distributed architecture (a separate coordinator, data nodes, query nodes, and an index node) that runs natively on Kubernetes. The operational complexity is real — getting Milvus running correctly in production takes meaningful infrastructure engineering — but it is the most mature open-source path to 1B+ vectors with consistent query latency under concurrent load. Milvus 2.5 added native sparse vector support and a Sparse-BM25 index, making it competitive for hybrid search at scale without requiring a separate Elasticsearch deployment. Pinecone: fully managed SaaS with a simple REST API and a serverless pricing model. The correct choice for teams that need zero infrastructure overhead at early to medium scale and are comfortable with the cost (which becomes significant at high query volumes) and the vendor dependency. LanceDB: built on the Lance columnar storage format, with embedded vector index — the entire database runs inside your Python process with no separate server. Well-suited for ML workflows where the data processing pipeline and the retrieval step coexist in the same environment. Elasticsearch with kNN: worth adding only when you already run Elasticsearch for full-text search and want dense retrieval without introducing a new service — the kNN implementation is functional but Elasticsearch was not designed from the ground up as a vector database, and its ANN performance and memory efficiency trail the purpose-built options.
      </Prose>

      {/* ======================================================================
          6. VISUAL DIAGNOSTICS
          ====================================================================== */}
      <H2>6. Visualizing the trade-offs</H2>

      <H3>Recall vs. query latency across index types</H3>

      <Plot
        label="recall@10 vs query latency — 10M vectors at d=768 (approximate)"
        xLabel="Query latency (ms)"
        yLabel="Recall@10"
        width={520}
        height={260}
        series={[
          {
            name: "Brute force",
            color: "#f87171",
            points: [[2800, 1.00]],
          },
          {
            name: "HNSW (ef=50)",
            color: colors.gold,
            points: [[2, 0.92], [5, 0.96], [12, 0.98], [28, 0.99]],
          },
          {
            name: "IVF (n_probe varies)",
            color: "#60a5fa",
            points: [[1, 0.78], [3, 0.88], [8, 0.93], [20, 0.96]],
          },
          {
            name: "HNSW + PQ",
            color: "#4ade80",
            points: [[1, 0.82], [3, 0.90], [8, 0.94], [18, 0.97]],
          },
        ]}
      />

      <Prose>
        The plot illustrates the fundamental trade-off: brute force achieves perfect recall at enormous latency cost; HNSW sits near the Pareto frontier, achieving high recall with low latency; IVF gives lower latency at somewhat lower recall for the same compute budget; HNSW+PQ collapses the memory footprint (4–32×) at a moderate recall cost that is often acceptable after rescoring. Moving along the HNSW curve is done by adjusting ef_search: lower ef_search trades recall for latency within the same index. Moving between curves requires rebuilding the index with different hyperparameters (M for HNSW) or a different algorithm. The implication for production systems is that ef_search is the knob you turn to respond to latency or recall regressions in real time, without rebuilding anything; M and index choice are design-time decisions that require a rebuild.
      </Prose>

      <H3>HNSW search: step-by-step walk through the graph</H3>

      <StepTrace
        label="HNSW greedy search — entering at top layer, descending to bottom"
        steps={[
          {
            label: "Enter at top layer (L2) — single entry point",
            render: () => (
              <TokenStream tokens={[
                { label: "entry_node=EP", color: colors.gold },
                { label: "layer=2", color: "#555" },
                { label: "ef=1 (greedy)", color: "#60a5fa" },
              ]} />
            ),
          },
          {
            label: "Greedily follow long-range links toward query",
            render: () => (
              <TokenStream tokens={[
                { label: "EP", color: "#555" },
                { label: "→ node_A (closer)", color: colors.gold },
                { label: "→ node_C (closer)", color: colors.gold },
                { label: "→ node_F (local min at L2)", color: "#4ade80" },
              ]} />
            ),
          },
          {
            label: "Descend to layer 1 — start beam search (ef=50)",
            render: () => (
              <TokenStream tokens={[
                { label: "entry=node_F", color: "#60a5fa" },
                { label: "layer=1", color: "#555" },
                { label: "denser graph", color: "#60a5fa" },
                { label: "ef=50 candidates tracked", color: colors.gold },
              ]} />
            ),
          },
          {
            label: "Layer 1 beam search expands neighborhood",
            render: () => (
              <TokenStream tokens={[
                { label: "candidates: {F, G, H, M, P, ...}", color: "#4ade80" },
                { label: "best so far: node_M", color: colors.gold },
              ]} />
            ),
          },
          {
            label: "Descend to layer 0 — full density, return top-k",
            render: () => (
              <TokenStream tokens={[
                { label: "entry=node_M", color: "#60a5fa" },
                { label: "layer=0 (all nodes)", color: "#555" },
                { label: "expand ef=50", color: "#60a5fa" },
                { label: "return top-k=10", color: "#4ade80" },
              ]} />
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        The metadata filtering question is the most underappreciated dimension of vector database selection. Every production retrieval use case has at least one metadata predicate: find documents similar to this query and from this tenant, and authored before this date, and tagged with these categories. The naive approach — retrieve the top 1,000 nearest neighbors by vector similarity, then filter by metadata — fails when the filter is selective. If only 0.1% of the corpus matches the predicate, a top-1,000 shortlist will often contain zero passing results even if the correct answers exist elsewhere in the index. The correct approach integrates the predicate into the ANN traversal itself so that only eligible vectors are ever considered as candidates. This is called filtered ANN or pre-filtering, and it is significantly harder to implement correctly than it sounds — the graph connectivity assumptions of HNSW break when only a subset of nodes are eligible, and naive implementations that propagate the filter into the traversal can cause the algorithm to fail to find any neighbors at all under high selectivity. Qdrant's filtered HNSW is the most production-hardened implementation of this capability in the open-source space; it maintains separate connectivity for filtered subsets and degrades gracefully as selectivity increases. This is the single capability most worth benchmarking against your own metadata distribution before committing to a vector database.
      </Prose>

      <H3>Choosing a vector database</H3>

      <CodeBlock>
{`Scale             Default choice        Notes
─────────────────────────────────────────────────────────────────────
< 1M vectors      pgvector / Chroma     Stay in Postgres if already there.
                                        Chroma for local dev only.
1M – 50M          Qdrant (self-hosted)  Best filtered-search perf (4ms p50).
                  pgvector + pgvectorscale  If Postgres stack; 11.4x QPS gain
                                           over vanilla at 50M vectors.
50M – 500M        Qdrant / Weaviate     Weaviate if hybrid + graph needed.
                                        Both support sharding at this scale.
500M – 1B+        Milvus / Pinecone     Milvus for self-hosted billion-scale.
                                        Pinecone if managed is acceptable.
Hybrid-search     Weaviate / Qdrant     Native BM25+dense fusion built-in.
  required        Milvus 2.5+           (not pgvector by default)
Serverless/       Pinecone / LanceDB    Pay-per-use; no index management.
  ephemeral`}
      </CodeBlock>

      <H3>Choosing an embedding model</H3>

      <CodeBlock>
{`Priority              Recommendation         Reason
────────────────────────────────────────────────────────────────────────
Zero infra overhead   text-embedding-3-small  $0.02/M tokens, solid MTEB
                      or Cohere embed-v3      coverage, no GPU needed.
Self-hosted quality   BGE-M3 or              Apache 2.0; dense+sparse+multi
                      Nomic-embed-v1.5       vector in one model; 0 cost/query.
Peak retrieval        Voyage-3               Top BEIR recall; commercial API.
  accuracy
Code / technical      Mistral Codestral-embed Trained on code+docs corpus.
Long context          Nomic-embed-v1.5       8192-token context natively.
Multilingual          Qwen3-Embedding-8B     70.58 MTEB multilingual; open.
MRL / two-stage       text-embedding-3-*     Official MRL training; truncate
  retrieval           or Nomic-v1.5          to 64–256d for first-pass filter.
Budget-critical       text-embedding-3-small  ≈$2 per million docs ingested.
  ingestion cost`}
      </CodeBlock>

      <Callout accent="gold">
        Embedding model choice has more impact on retrieval quality than vector database choice for most workloads. Validate embedding model on your domain before optimizing index structure.
      </Callout>

      {/* ======================================================================
          8. SCALING
          ====================================================================== */}
      <H2>8. Scaling: from prototype to billion vectors</H2>

      <Prose>
        The scaling story for vector retrieval has three regimes that require qualitatively different engineering approaches. Understanding the transitions prevents over-engineering early and under-engineering late.
      </Prose>

      <H3>Sub-10M vectors: brute force is fine</H3>

      <Prose>
        At 1M vectors and d=768, a brute-force cosine search in NumPy on a single CPU core takes roughly 2–4 seconds. With batched SIMD (FAISS flat index on CPU), that drops to 50–200ms — acceptable for batch processing, not for interactive search. With a GPU (FAISS on NVIDIA with cuVS), a million-vector flat search takes under 5ms. The point is that below roughly 5–10M vectors, the complexity of building and maintaining an ANN index often exceeds its benefit, particularly if the corpus changes frequently (every index update to an HNSW graph is relatively cheap per insertion but can require background compaction at scale). The pgvector flat scan (no ANN, exact search) at 1M vectors on a modern Postgres server takes on the order of 100–400ms depending on hardware; that is borderline for interactive use but fine for batch or asynchronous retrieval jobs. The upgrade to HNSW inside pgvector is a single SQL command and takes minutes to build at 1M scale. Start with the flat scan during prototyping; add the HNSW index when query latency becomes measurable in your evaluation.
      </Prose>

      <H3>10M–100M vectors: HNSW is the default</H3>

      <Prose>
        HNSW at 100M vectors with d=768, M=16, ef_search=50 achieves p95 query latency of roughly 12ms and Recall@10 of 0.94–0.97 on a single machine with sufficient RAM. The memory requirement is the constraint: 100M vectors × 768 dimensions × 4 bytes (FP32) = 307 GB for raw vectors, plus 2–5× overhead for the graph adjacency structure (M=16 means up to 32 neighbors per node per layer × 4 bytes per ID). Total memory budget for a 100M HNSW index: 600–900 GB. At FP16 (half precision), that halves; with PQ compression (32×), the vectors themselves drop to ~10 GB but the graph structure overhead stays.
      </Prose>

      <Prose>
        Memory budget formula for reference:
      </Prose>

      <MathBlock>{"\\text{Memory (GB)} \\approx \\frac{N \\cdot d \\cdot b_{\\text{vec}}}{10^9} + \\frac{N \\cdot M \\cdot 2 \\cdot 4}{10^9}"}</MathBlock>

      <Prose>
        where N is number of vectors, d is dimension, b_vec is bytes per scalar (4 for FP32, 2 for FP16, 1/8 for binary), and M is the HNSW connectivity parameter. At N=100M, d=768, FP16, M=16: ≈154 GB vectors + 13 GB graph = ~167 GB. This fits on a high-memory single node; going to 1B vectors forces sharding.
      </Prose>

      <H3>100M–1B+ vectors: sharding, DiskANN, quantization</H3>

      <Prose>
        At billion scale, three approaches emerge. Sharding splits the corpus across multiple nodes, each running its own ANN index, and aggregates results at query time. The complication is that sharding by document creates load imbalance (some shards may have more relevant documents for popular queries), and sharding by random partition reduces recall because the true nearest neighbor may be on a different shard from the one searched. Milvus, Pinecone, and Weaviate all support sharding with their own partition strategies; the trade-offs are in the documentation and the ANN benchmarks.
      </Prose>

      <Prose>
        DiskANN (Microsoft Research) keeps the majority of the graph on NVMe SSD and caches only the hot nodes (entry points and high-degree hubs) in DRAM. A billion-vector index that would require 3+ TB of DRAM for in-memory HNSW can be served from a single machine with 64–128 GB DRAM plus a fast SSD, at p95 query latency of 10–30ms. The trade-off is that disk I/O is the bottleneck: random read latency on NVMe is 50–100µs versus sub-microsecond DRAM access, so DiskANN pays with latency variance rather than raw throughput.
      </Prose>

      <Prose>
        The practical scaling guideline: under 50M vectors, a single Qdrant or pgvector node handles it. 50M–500M, two to four nodes with sharding. Above 500M, you are in Milvus or Pinecone territory where the infrastructure is purpose-built for this problem. Quantization (PQ, binary) should be applied at every scale above 10M to stay within memory budgets; the recall cost is recoverable by rescoring the shortlist with full-precision vectors.
      </Prose>

      <Prose>
        One frequently overlooked scaling dimension is index build time. Building an HNSW index on 100M vectors with M=16 and ef_construction=200 takes on the order of 8–16 hours on a 16-core CPU server. During that build time, the system is typically unavailable for ANN queries or must serve from the old index. For corpora that update continuously — document databases that receive hundreds of new documents per hour — the right pattern is incremental insertion (HNSW supports this natively) plus periodic offline re-builds to compact the graph and fix any connectivity issues accumulated from many insertions. The re-build cadence depends on how degraded the index connectivity becomes with insertions, which varies by implementation. Qdrant and Weaviate both handle this with background optimization processes that compact and re-optimize graph segments without taking the index offline.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>1 — Dimension mismatch between query and document encoders</H3>
      <Prose>
        If the documents were indexed with one embedding model and the query is encoded with a different one — even a different version of the same model — the similarity scores are meaningless. The two vectors live in geometrically unrelated spaces. This fails silently: the index returns results, the scores look plausible, and the retrieved documents are garbage. The symptom is a sudden drop in retrieval quality after a model update or a bug in the ingestion pipeline where a different model was used for a batch of documents. The fix is strict version tracking of the embedding model used for each vector, and re-indexing when the model changes.
      </Prose>

      <H3>2 — Index rebuild lag during corpus updates</H3>
      <Prose>
        Most ANN indexes (including HNSW) support incremental insertion, but not all operations are cheap. Deletions in HNSW are expensive — they require marking nodes as tombstones and rebuilding graph connectivity, which is why some implementations (Faiss pre-v1.7, older Qdrant) did not support true deletions and instead required full index rebuilds. During a large update batch, the index may temporarily serve stale data. The pattern that avoids this is a two-phase update: build the new index alongside the existing one, then atomically swap the pointer. Failing to do this means the index is inconsistent for the duration of the rebuild.
      </Prose>

      <H3>3 — Stale embeddings vs. fresh documents</H3>
      <Prose>
        The embedding is computed from the document text at indexing time. If the document is updated — a policy document is revised, a product page changes — and the index is not refreshed, the stored vector encodes the old content while the document itself reflects the new. Retrieval finds the chunk based on the old embedding, the reader sees the updated text, and the answer may be partially or fully wrong. The fix is a content-hash on each document and an ingestion pipeline that detects changes and re-embeds and re-indexes affected chunks. This sounds obvious and is regularly skipped.
      </Prose>

      <H3>4 — ANN recall drop hiding relevant documents</H3>
      <Prose>
        A 3% gap in Recall@10 sounds small. Over a corpus of 50M documents with 10M potentially relevant to a given user's domain, 3% miss rate means 300K documents are systematically unretrievable. The documents that fall through are not random — they tend to be the ones near cluster boundaries in IVF indexes, or with few neighbors in HNSW graph regions that happened to be sparsely sampled during construction. Measuring Recall@k against exact search on a held-out query set is the only way to know how bad this is for your workload. Vendor-reported recall figures are on standard ANN benchmarks with uniform distributions; your domain distribution may be quite different.
      </Prose>

      <H3>5 — Embedding model drift across versions</H3>
      <Prose>
        Embedding models are retrained and improved over time. OpenAI deprecated <Code>text-embedding-ada-002</Code> and introduced the text-embedding-3 family with a different geometric space. Cohere, Voyage, and BGE have all released model updates that are not backward-compatible with the previous version's embedding space. If you have a production index built on model version N and you switch to version N+1 without re-indexing, every new document embedded with the new model is in a different space from the stored documents, and cross-model similarity comparisons are meaningless. Version pins and full re-indexing on model updates are mandatory, not optional hygiene.
      </Prose>

      <H3>6 — Cold-start on new document classes</H3>
      <Prose>
        An embedding model trained on general text may perform poorly on highly domain-specific content that uses specialized vocabulary, notation, or structure. A model trained on natural language embedding medical lab reports, legal case law written in archaic language, or code in an obscure DSL, will produce embeddings with less semantic discrimination than a domain-specific or domain-adapted model. The symptom is low retrieval recall on domain-specific queries that is not visible in MTEB scores (which use general-purpose benchmarks). The fix is either domain-specific fine-tuning of the embedding model or selecting a model that has been trained on your content type (Codestral-embed for code, voyage-law-2 for legal, etc.).
      </Prose>

      <H3>7 — Ingestion compute cost at scale</H3>
      <Prose>
        Embedding a billion documents is expensive. At $0.02 per million tokens with text-embedding-3-small, and assuming an average of 500 tokens per document (a 400-word chunk), embedding 1B documents costs $10,000 in API fees — before counting the compute to chunk, the storage for the index, or the vector DB hosting cost. Self-hosted embedding at 1B documents with a GPU-backed open model (BGE-M3, Nomic) costs on the order of $2,000–5,000 in A100 or H100 compute time. Neither is prohibitive at scale, but it changes the calculus for re-indexing decisions: you cannot afford to re-embed the full corpus every time the model is updated. The architecture needs change detection to re-embed only modified documents, and model updates need to be infrequent and planned.
      </Prose>

      <H3>8 — Vector DB without hybrid search support</H3>
      <Prose>
        Dense retrieval alone fails on queries that depend on exact keyword or entity matching. A query for a specific product SKU, a precise legal citation, a rare technical acronym — these are cases where BM25 lexical retrieval would return the exact document and dense retrieval may return semantically adjacent but textually different results. The Hybrid Search topic covers this in detail, but the failure mode to flag here is that choosing a vector database that does not support native hybrid (dense + sparse) search forces you to run two separate retrieval systems and merge their outputs externally, which is operationally more complex and harder to tune. Qdrant (v1.9+), Weaviate, Milvus 2.5+, and Elasticsearch all support native hybrid; pgvector and older Chroma versions do not. Check hybrid support before committing to a vector DB if your query distribution includes entity-specific lookups.
      </Prose>

      {/* ======================================================================
          10. EXERCISES
          ====================================================================== */}
      <H2>10. Exercises</H2>

      <Prose>
        <strong>Exercise 1 — Recall vs. ef_search curve.</strong> Using the SimpleHNSW class from section 4b, build an index over 5,000 random 64-dimensional vectors. For ef_search values in [10, 25, 50, 100, 200], measure Recall@10 against brute-force ground truth over 500 random queries. Plot the recall-vs-latency curve. At what ef_search value does recall plateau? How does changing M (from 8 to 32) shift the curve?
      </Prose>

      <Prose>
        <strong>Exercise 2 — PQ sub-space count vs. recall.</strong> Using the ProductQuantizer from section 4d on a 256-dimensional corpus of 20,000 vectors, vary M (sub-spaces) from 4 to 64 (factors of 2) while holding K=256 fixed. For each M, compute Recall@10 and bytes per vector. Plot recall against bytes per vector and identify the knee of the curve — the point where further compression starts costing disproportionate recall.
      </Prose>

      <Prose>
        <strong>Exercise 3 — IVF n_probe sensitivity.</strong> Build an IVF index with K=256 clusters on 50,000 vectors at d=128. Measure Recall@10 at n_probe values of [1, 2, 4, 8, 16, 32, 64]. At what n_probe does recall reach 0.95? What fraction of the corpus is being searched at that n_probe? Why does increasing n_probe beyond a threshold give diminishing returns?
      </Prose>

      <Prose>
        <strong>Exercise 4 — Matryoshka truncation with a real model.</strong> Using <Code>sentence-transformers</Code> with a Nomic-embed or any MRL-trained model, embed 1,000 passages from a domain of your choice at full dimensionality. For each query in a 50-question evaluation set, find the exact top-10 at full dimension. Then truncate all embeddings to [64, 128, 256, 512] and measure how Recall@10 degrades. Compare the same truncation on a non-MRL model (e.g., all-MiniLM-L6-v2). The MRL model should degrade gracefully; the standard model should collapse.
      </Prose>

      <Prose>
        <strong>Exercise 5 — End-to-end model swap.</strong> Build a small RAG evaluation set: 200 (question, expected-answer-chunk) pairs from a document corpus of your choice. Embed and index with two models: <Code>text-embedding-3-small</Code> (via OpenAI API) and <Code>BAAI/bge-large-en-v1.5</Code> (self-hosted via sentence-transformers). Use brute-force exact search for both (fairness — no index approximation). Measure Recall@5 for each. Then add a cross-encoder reranker (<Code>cross-encoder/ms-marco-MiniLM-L-6-v2</Code>) on top of the top-50 shortlist and measure again. Document which improvement — better embedding model or adding a reranker — had a larger effect on your corpus.
      </Prose>

      {/* ======================================================================
          11. FURTHER READING
          ====================================================================== */}
      <H2>11. Further reading</H2>

      <Prose>
        The primary sources that underpin this topic are worth reading directly. Malkov and Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs" (IEEE TPAMI 2018, arXiv:1603.09320) is the HNSW paper; the algorithmic detail in section 4 of the paper is dense but navigable, and the ablations on M and ef_construction are directly applicable to production tuning. Kusupati et al., "Matryoshka Representation Learning" (NeurIPS 2022, arXiv:2205.13147) is the MRL paper; the key result is Figure 3, showing that MRL representations at 64 dimensions match the accuracy of independently-trained 64-dimensional models while also scaling to full dimension. The MTEB paper — Muennighoff et al., "MTEB: Massive Text Embedding Benchmark" (EACL 2023) — describes the evaluation protocol and why the benchmark covers eight categories rather than just retrieval. The FAISS documentation at faiss.ai covers index factory strings, the IVFPQ combination, and GPU acceleration in production detail that is not available elsewhere. For the vector database landscape, the ANN benchmarks at ann-benchmarks.com and the Qdrant, Weaviate, and Milvus engineering blogs are the most reliable ongoing sources because they update with new index algorithms faster than any review article.
      </Prose>

      <Prose>
        Related topics in this section: the Hybrid Search topic covers BM25+dense fusion in detail, including the RRF score fusion formula and the failure modes specific to sparse-only retrieval. The GraphRAG and Agentic RAG topic covers what happens when a single-shot retrieval step is replaced by an iterative loop where the model decides when to retrieve more. The RAG topic is the right entry point if this topic felt too deep too fast — it covers the full pipeline end-to-end before diving into any single layer.
      </Prose>
    </div>
  ),
};

export default embeddingModelsVectorDB;
