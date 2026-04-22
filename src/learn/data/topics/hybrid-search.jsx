import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const hybridSearch = {
  slug: "hybrid-search-dense-sparse-reranking",
  title: "Hybrid Search (Dense + Sparse + Reranking)",
  readTime: "38 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY
          ====================================================================== */}
      <H2>1. Why neither pure retrieval mode is enough</H2>

      <Prose>
        Every retrieval system makes a bet about what "relevant" means. Dense retrieval — bi-encoder embedding plus approximate nearest-neighbor lookup — bets that relevance is a geometric property: queries and documents that mean the same thing will end up close in the embedding space, regardless of whether they share a single word. That bet is usually right. Ask "how do I reset my password?" and the bi-encoder will surface a document titled "Account Recovery" with no word overlap, because three hundred million parameters of pretraining have encoded the semantic neighborhood well enough to bridge the phrasing gap. The same bet fails the moment the query contains an artifact the model has not generalized over. A product code like <Code>TX-4891-C</Code>, an error message like <Code>SIGSEGV at 0x00007f</Code>, a legal citation like <Code>29 CFR § 1910.147</Code>, a rare drug identifier like <Code>Voclosporin</Code> — these strings occupy sparsely populated corners of the embedding space. The nearest neighbors are whatever happened to appear in similar syntactic contexts during pretraining, not the specific documents that mention the identifier by name.
      </Prose>

      <Prose>
        Classical sparse retrieval makes the opposite bet: that relevance is a lexical property, measurable by the overlap of query tokens and document tokens weighted by their rarity. BM25 — Best Matching 25, the formalization of this idea by Robertson and colleagues across a sequence of papers culminating in the definitive 2009 survey with Zaragoza — has no embedding layer, no parameters, and no concept of meaning. It counts. A document that contains the query term "Voclosporin" scores high; one that discusses calcineurin inhibitors at length but never uses the trade name scores zero. That exact-match property is what makes BM25 irreplaceable for lookup queries — and what makes it useless for the semantic questions that dense retrieval handles natively. A customer typing "my account is locked out" will not find an "Account Recovery" document via BM25 unless those exact words appear in it.
      </Prose>

      <Prose>
        Real-world query logs are a superposition of both shapes. The same user who asks natural-language questions also pastes error codes and model numbers. The same support chatbot that handles semantic intent also needs to look up policy identifiers by their canonical names. No single-mode retriever serves both halves of the distribution well. Hybrid search runs both channels — dense and sparse — over the same corpus and combines their rankings before the results reach the language model. The combination consistently outperforms either mode alone because the two retrieval modes fail on largely disjoint query subsets: dense fails on lexical-exact queries, sparse fails on semantic-paraphrase queries, and hybrid catches the union rather than the intersection.
      </Prose>

      <Prose>
        Reranking adds a third pass. The candidates produced by hybrid retrieval were each scored by models that processed the query and document in isolation — the bi-encoder embedded them separately; BM25 counted their tokens without ever modeling their interaction. A cross-encoder reranker takes the top-K candidates from the merged list and scores each (query, document) pair jointly, allowing the model to attend across both simultaneously and produce a relevance judgment that reflects fine-grained meaning alignment rather than geometric proximity or token overlap. The quality improvement is large and consistent. The latency cost is bounded, because reranking operates on a small shortlist — typically twenty to fifty documents — not the full corpus. Hybrid retrieval followed by cross-encoder reranking is the architecture that serious production RAG deployments converge on, and the gap between it and vanilla dense-only retrieval widens as the query distribution diversifies.
      </Prose>

      <Prose>
        Three lines of research converge in this topic. BM25 comes from Robertson and Walker's "Some Simple Effective Approximations to the 2-Poisson Model for Probabilistic Weighted Retrieval" (SIGIR 1994) and is canonized in Robertson and Zaragoza's 2009 survey "The Probabilistic Relevance Framework: BM25 and Beyond" in Foundations and Trends in Information Retrieval. Reciprocal Rank Fusion — the rank-combination method that dominates hybrid merging — comes from Cormack, Clarke, and Büttcher's "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods" (SIGIR 2009). Late-interaction retrieval via per-token embeddings and max-sim aggregation comes from Khattab and Zaharia's ColBERT (arXiv:2004.12832, SIGIR 2020). Learned sparse retrieval — neural models that output sparse token-weight vectors rather than dense embeddings — comes from Formal, Piwowarski, and Clinchant's SPLADE (arXiv:2107.05720, SIGIR 2021). Cross-encoder reranking has older roots but its current production form is represented by BAAI's BGE Reranker family (bge-reranker-v2-m3, bge-reranker-v2-gemma, released 2024). Each of these provides a distinct retrieval or re-scoring mechanism, and their combination is what this topic is about.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition — five mechanisms and how they interact</H2>

      <H3>Sparse retrieval (BM25)</H3>

      <Prose>
        Build an inverted index over the corpus: for each term, store the list of documents that contain it and the frequency with which it appears. At query time, for each query term, look up its document list and score each document by how often the term appears (with diminishing returns past a threshold), weighted by how rare the term is across the entire corpus. Documents that contain rare query terms score high; documents that contain only common words score low. There is no learned representation, no embedding, and no notion of synonymy. The computation is a lookup over prebuilt lists, which is why BM25 query latency is measured in single-digit milliseconds even at billion-document scale on a sharded inverted index.
      </Prose>

      <H3>Dense retrieval (bi-encoder)</H3>

      <Prose>
        An embedding model maps the query to a vector, and the same or a sibling model has already mapped every document chunk to a vector and stored those vectors in an ANN index. At query time, compute the query vector with one forward pass and run an approximate nearest-neighbor search over the document index. The score between query and document is the cosine similarity between their vectors. Semantic generalization is the native capability here — the model has learned a metric space where paraphrase, cross-lingual equivalents, and topically related content all land nearby, regardless of lexical surface form. The failure is on exact terminology that the model has not generalized: the embedding geometry for a rare token is determined by its pretraining context, and if that context was too thin, the nearest neighbors in the embedding space are not the right documents.
      </Prose>

      <H3>Hybrid fusion (RRF and weighted sum)</H3>

      <Prose>
        Given a dense ranking and a sparse ranking over the same document set, combine them into a single ranked list. The conceptually cleanest combiner is Reciprocal Rank Fusion: use each document's position in each ranked list, not its raw score. Reciprocal rank fusion is scale-invariant — it does not matter that BM25 scores are unitless counts while cosine similarities are bounded in [-1, 1]. The only information it uses is order. An alternative is a weighted sum of normalized scores, which preserves score magnitude information but requires careful normalization to prevent one channel from numerically dominating. In practice, RRF is the more robust default: it requires no tuning of score normalization and no threshold calibration, and it consistently performs within a few points of an optimally tuned weighted sum.
      </Prose>

      <H3>Late interaction (ColBERT)</H3>

      <Prose>
        The bi-encoder compresses the query into a single vector before scoring it against any document. That compression is efficient — it allows the document side to be precomputed — but it is lossy. A single vector cannot precisely represent every token in the query, particularly when different parts of the query are relevant to different parts of the document. ColBERT's late interaction architecture solves this by computing a separate embedding vector for every token in the query and for every token in the document. At scoring time, for each query token, find its maximum cosine similarity with any document token — the most relevant part of the document for that query token. Sum these per-query-token maximum similarities. This MaxSim score captures fine-grained alignment between query tokens and their best-matching document tokens, without requiring the full quadratic cross-encoder computation. ColBERT occupies a middle ground: more expressive than bi-encoders, substantially cheaper than cross-encoders, and indexable because the per-token document vectors can be precomputed and stored.
      </Prose>

      <H3>Cross-encoder reranking</H3>

      <Prose>
        A cross-encoder takes the query and a candidate document concatenated together as a single input sequence and runs one forward pass to produce a relevance score. It can attend across the query and the document simultaneously — it can see whether the query's subject appears in the same sentence as the document's key claim, whether a term is used in the same disambiguating sense, whether the document's answer satisfies the specific constraint the query imposes. Bi-encoders cannot do any of this because the two inputs are never in the same forward pass. Cross-encoders are roughly a hundred times slower per pair than bi-encoders, which is why they are applied only to the small shortlist produced by the first-stage retrieval — typically twenty to a hundred candidates — never to the full corpus. At that scale the latency is bounded and acceptable. The quality gain is large: cross-encoder relevance judgments are calibrated against human annotation in ways that bi-encoder geometry is not, and they degrade gracefully on out-of-domain queries in ways that pure embedding similarity does not.
      </Prose>

      <H3>Learned sparse retrieval (SPLADE)</H3>

      <Prose>
        SPLADE (Sparse Lexical and Expansion Model) trains a BERT-style encoder to output a sparse weight vector over the full model vocabulary for each query or document. Unlike BM25, which assigns weight only to terms that actually appear in the text, SPLADE assigns nonzero weights to semantically related terms that do not appear. A document about "heart attack" gets nonzero weight on "myocardial infarction." A query for "car" gets weight on "vehicle" and "automobile." The output is still sparse — most of the 30,000-plus vocabulary dimensions are zero — which means it can be served by exactly the same inverted index infrastructure that serves BM25. The net effect is a retrieval method with the efficiency characteristics of sparse retrieval and the semantic generalization of dense retrieval, at the cost of a full inference forward pass per document at index time. SPLADE can be dropped into the sparse channel of a hybrid system as a higher-quality replacement for BM25, or run alongside it.
      </Prose>

      {/* ======================================================================
          3. MATH
          ====================================================================== */}
      <H2>3. Mathematical foundations</H2>

      <H3>BM25</H3>

      <Prose>
        The BM25 score for query <Code>q</Code> and document <Code>d</Code> is a sum over query terms of term frequency weighted by inverse document frequency, with saturation and length normalization applied. Let <Code>f(t, d)</Code> be the frequency of term <Code>t</Code> in document <Code>d</Code>, <Code>|d|</Code> the document length in tokens, <Code>avgdl</Code> the average document length in the corpus, and <Code>N</Code> the total number of documents. The IDF component is the log of how rarely the term appears across documents:
      </Prose>

      <MathBlock>{"\\text{IDF}(t) = \\log \\frac{N - n(t) + 0.5}{n(t) + 0.5}"}</MathBlock>

      <Prose>
        where <Code>n(t)</Code> is the number of documents containing term <Code>t</Code>. The full BM25 scoring formula:
      </Prose>

      <MathBlock>{"\\text{BM25}(q, d) = \\sum_{t \\in q} \\text{IDF}(t) \\cdot \\frac{f(t,\\,d)\\,(k_1 + 1)}{f(t,\\,d) + k_1 \\left(1 - b + b \\cdot \\dfrac{|d|}{\\text{avgdl}}\\right)}"}</MathBlock>

      <Prose>
        The parameter <Code>k₁</Code> controls term-frequency saturation. High <Code>k₁</Code> allows many additional occurrences of a term to keep contributing to the score; the conventional default of 1.2 produces fast saturation so that five occurrences of a term is not dramatically better than two. The parameter <Code>b</Code> controls length normalization; the conventional default of 0.75 applies moderate normalization so that long documents are penalized but not completely equalized with short ones. The right values depend on the corpus: technical documentation with deliberately repeated terminology benefits from higher <Code>k₁</Code>; conversational corpora with highly variable document lengths benefit from higher <Code>b</Code>. Leaving both at their defaults without measuring the tradeoff is one of the eight failure modes catalogued in section 9.
      </Prose>

      <H3>Reciprocal Rank Fusion</H3>

      <Prose>
        Given multiple ranked lists <Code>L₁, L₂, ..., Lₘ</Code> over the same document set, the RRF score of a document <Code>d</Code> is:
      </Prose>

      <MathBlock>{"\\text{RRF}(d) = \\sum_{i=1}^{m} \\frac{1}{k + \\text{rank}_i(d)}"}</MathBlock>

      <Prose>
        where <Code>rank_i(d)</Code> is the position of document <Code>d</Code> in list <Code>i</Code> (1-indexed), and <Code>k</Code> is a smoothing constant, conventionally 60. The smoothing constant prevents the top-ranked document from receiving an unboundedly large contribution relative to the second-ranked one, stabilizing the fusion when lists differ in quality or length. Documents that do not appear in list <Code>i</Code> contribute zero from that list. The final ranking is obtained by sorting all documents by their RRF score descending. The formula contains no weights: all input lists contribute equally. Extensions that add per-list weights exist and can improve performance on datasets where one retriever is known to be substantially better, but they require the weights to be calibrated on a development set and risk overfitting to that set — another of the failure modes in section 9.
      </Prose>

      <H3>ColBERT late interaction</H3>

      <Prose>
        Let <Code>Q = [q₁, q₂, ..., q_m]</Code> be the per-token embeddings of the query and <Code>D = [d₁, d₂, ..., d_n]</Code> be the per-token embeddings of the document, both produced by the ColBERT encoder. The relevance score is:
      </Prose>

      <MathBlock>{"S(Q, D) = \\sum_{j=1}^{m} \\max_{i=1}^{n} \\, q_j \\cdot d_i"}</MathBlock>

      <Prose>
        For each query token <Code>qⱼ</Code>, find the document token <Code>dᵢ</Code> that maximizes their dot product similarity — the MaxSim operation. Sum the MaxSim values across all query tokens. The result rewards documents that contain, somewhere in their text, a token with high similarity to each query token. A query about "cardiac arrest treatment" retrieves a document that contains, at different positions, tokens embedding close to "cardiac," "arrest," and "treatment" — even if those tokens appear in different sentences and the document never uses those exact words. The precomputed document token matrices are stored in the index; only the query token matrices need to be computed at request time, making retrieval tractable despite the late-interaction step.
      </Prose>

      <H3>Cross-encoder scoring</H3>

      <Prose>
        A cross-encoder reranker is a function:
      </Prose>

      <MathBlock>{"s = f_\\theta([\\text{CLS};\\ q;\\ \\text{SEP};\\ d;\\ \\text{SEP}])"}</MathBlock>

      <Prose>
        The query <Code>q</Code> and document <Code>d</Code> are concatenated into a single input sequence with separator tokens, passed through a transformer encoder, and a linear head over the <Code>[CLS]</Code> representation produces a scalar relevance score. The critical property is that attention operates across both the query and document tokens simultaneously during the forward pass — each token can attend to any other token in the joint sequence. This cross-attention over the concatenated input is what gives cross-encoders their accuracy advantage over bi-encoders, and it is also why they cannot be applied at index time: document representations depend on the specific query and cannot be precomputed.
      </Prose>

      {/* ======================================================================
          4. FROM SCRATCH
          ====================================================================== */}
      <H2>4. Building from scratch</H2>

      <H3>4a. BM25 in pure Python</H3>

      <Prose>
        The implementation below is a complete, self-contained BM25 engine. It takes a list of documents, builds the inverted index and document statistics during indexing, and at query time scores and ranks the corpus. It is naive in the sense that it holds the full inverted index in memory and does not parallelize — production BM25 uses sharded inverted indexes spread across many nodes — but it is correct and demonstrates every component of the formula.
      </Prose>

      <CodeBlock language="python">
{`import math
from collections import Counter, defaultdict

class BM25:
    """Pure-Python BM25 implementation. k1=1.5, b=0.75 are reasonable defaults."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)

        # Tokenise (whitespace; real systems use stemming + stopword removal)
        self.tokenised = [doc.lower().split() for doc in corpus]

        # Document lengths
        self.dl = [len(t) for t in self.tokenised]
        self.avgdl = sum(self.dl) / self.N if self.N > 0 else 1.0

        # Inverted index: term -> {doc_id: freq}
        self.index: dict[str, dict[int, int]] = defaultdict(dict)
        for doc_id, tokens in enumerate(self.tokenised):
            tf = Counter(tokens)
            for term, freq in tf.items():
                self.index[term][doc_id] = freq

        # IDF cache
        self._idf_cache: dict[str, float] = {}

    def _idf(self, term: str) -> float:
        if term not in self._idf_cache:
            n_t = len(self.index.get(term, {}))
            self._idf_cache[term] = math.log(
                (self.N - n_t + 0.5) / (n_t + 0.5)
            )
        return self._idf_cache[term]

    def score(self, query: str, doc_id: int) -> float:
        tokens = query.lower().split()
        d_len = self.dl[doc_id]
        total = 0.0
        for term in set(tokens):
            idf = self._idf(term)
            freq = self.index.get(term, {}).get(doc_id, 0)
            norm = self.k1 * (1 - self.b + self.b * d_len / self.avgdl)
            tf_score = freq * (self.k1 + 1) / (freq + norm)
            total += idf * tf_score
        return total

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return list of (doc_id, score) sorted by score descending."""
        tokens = query.lower().split()
        # Candidate documents: union of posting lists for query terms
        candidates = set()
        for term in tokens:
            candidates.update(self.index.get(term, {}).keys())

        scored = [(doc_id, self.score(query, doc_id)) for doc_id in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ----- Demo -----
corpus = [
    "Account recovery and password reset procedures for enterprise users",
    "TX-4891-C product datasheet: input voltage range 5–24V, current rating 3A",
    "Cardiac arrest treatment protocols and myocardial infarction management",
    "TX-4891-C firmware update version 2.3 changelog and installation guide",
    "Password managers and credential hygiene best practices",
]

bm25 = BM25(corpus)

# Exact term query — BM25 wins
print("Query: TX-4891-C firmware")
for doc_id, score in bm25.search("TX-4891-C firmware", top_k=3):
    print(f"  [{score:.3f}] {corpus[doc_id][:60]}")

# Semantic query — BM25 will miss unless the exact words match
print("\\nQuery: how to reset my account access")
for doc_id, score in bm25.search("how to reset my account access", top_k=3):
    print(f"  [{score:.3f}] {corpus[doc_id][:60]}")`}
      </CodeBlock>

      <H3>4b. Hybrid fusion with Reciprocal Rank Fusion</H3>

      <Prose>
        With a BM25 engine and a mock dense retriever, we can implement the full hybrid fusion step. The mock dense retriever below returns rank-ordered document IDs using cosine similarity over precomputed sentence-embedding-style vectors (represented as random vectors for demonstration — swap in any real embedding model). The fusion function is complete and production-correct.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from typing import Callable

def reciprocal_rank_fusion(
    *rankings: list[int],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Combine N ranked lists (each a list of doc_ids, best first) via RRF.
    Returns list of (doc_id, rrf_score) sorted by score descending.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class MockDenseRetriever:
    """
    Stand-in for a real bi-encoder + ANN index.
    Vectors are random; replace embed() with your real embedding call.
    """
    def __init__(self, corpus: list[str], dim: int = 64):
        np.random.seed(42)
        # "Precomputed" document embeddings — in production, built at index time
        self.doc_vecs = np.random.randn(len(corpus), dim)
        self.doc_vecs /= np.linalg.norm(self.doc_vecs, axis=1, keepdims=True)
        self.corpus = corpus
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        # Deterministic mock: hash text to a seed for reproducibility
        rng = np.random.RandomState(hash(text) % (2**31))
        v = rng.randn(self.dim)
        return v / np.linalg.norm(v)

    def search(self, query: str, top_k: int = 10) -> list[int]:
        q = self.embed(query)
        sims = self.doc_vecs @ q          # cosine similarity (vectors are unit-normed)
        return list(np.argsort(-sims)[:top_k])


# Tie it together
def hybrid_search(
    query: str,
    bm25_engine: BM25,
    dense_retriever: MockDenseRetriever,
    top_k_per_channel: int = 20,
    top_k_final: int = 10,
    rrf_k: int = 60,
) -> list[tuple[int, float]]:
    sparse_hits = [doc_id for doc_id, _ in bm25_engine.search(query, top_k_per_channel)]
    dense_hits = dense_retriever.search(query, top_k_per_channel)
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=rrf_k)
    return fused[:top_k_final]


# Usage
dense = MockDenseRetriever(corpus)
results = hybrid_search("TX-4891-C password reset", bm25, dense)
print("Hybrid results:")
for doc_id, score in results:
    print(f"  [rrf={score:.4f}] {corpus[doc_id][:65]}")`}
      </CodeBlock>

      <H3>4c. ColBERT-style late interaction (simplified)</H3>

      <Prose>
        A stripped-down implementation that shows the MaxSim aggregation pattern. Real ColBERT uses a BERT encoder with a linear projection to 128 dimensions; we use random unit vectors as stand-in token embeddings. The key insight is the two-level structure: per-token embeddings stored in the index, MaxSim aggregated at query time.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

DIM = 32  # ColBERT uses 128; 32 is sufficient for demo

def mock_token_embed(text: str, dim: int = DIM) -> np.ndarray:
    """Return per-token embeddings: shape (n_tokens, dim). Unit-normed."""
    tokens = text.lower().split()
    vecs = []
    for tok in tokens:
        rng = np.random.RandomState(hash(tok) % (2**31))
        v = rng.randn(dim)
        vecs.append(v / np.linalg.norm(v))
    return np.array(vecs)  # (n_tokens, dim)

def late_interaction_score(
    query_vecs: np.ndarray,   # (m, dim)  query token embeddings
    doc_vecs: np.ndarray,     # (n, dim)  document token embeddings
) -> float:
    """MaxSim aggregation: Σ_j max_i sim(q_j, d_i)"""
    sim_matrix = query_vecs @ doc_vecs.T   # (m, n) — all pairwise similarities
    max_sims = sim_matrix.max(axis=1)      # (m,)   — best doc token per query token
    return float(max_sims.sum())

# Index document token embeddings offline
doc_token_vecs = [mock_token_embed(doc) for doc in corpus]

def colbert_search(query: str, doc_token_vecs: list, top_k: int = 3) -> list[tuple[int, float]]:
    q_vecs = mock_token_embed(query)
    scored = [
        (doc_id, late_interaction_score(q_vecs, d_vecs))
        for doc_id, d_vecs in enumerate(doc_token_vecs)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

print("ColBERT-style results for 'TX-4891-C firmware changelog':")
for doc_id, score in colbert_search("TX-4891-C firmware changelog", doc_token_vecs):
    print(f"  [maxsim={score:.3f}] {corpus[doc_id][:65]}")`}
      </CodeBlock>

      <H3>4d. Cross-encoder reranker (toy training loop)</H3>

      <Prose>
        Cross-encoders are trained on (query, document, relevance label) triples. The training loop below is a minimal binary classification cross-encoder: given a concatenated [query; document] sequence (represented as a bag of token hashes), predict whether the pair is relevant. A production reranker uses a full transformer encoder and is trained on human preference data, but this toy demonstrates the joint-input architecture and the pairwise training signal.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

# Toy cross-encoder: concat query and document, compute a relevance score.
# In production: use a transformer encoder fine-tuned on MS MARCO or similar.

class ToyLinearCrossEncoder:
    """
    Input: mean-pooled token hashes for (query + doc) concatenated.
    Output: scalar relevance score.
    """
    def __init__(self, vocab_size: int = 1000, dim: int = 16):
        np.random.seed(0)
        self.embedding = np.random.randn(vocab_size, dim) * 0.01
        self.w = np.random.randn(dim) * 0.01
        self.vocab_size = vocab_size
        self.lr = 0.01

    def _encode(self, text: str) -> np.ndarray:
        tokens = text.lower().split()
        ids = [hash(t) % self.vocab_size for t in tokens]
        if not ids:
            return np.zeros(self.embedding.shape[1])
        return self.embedding[ids].mean(axis=0)

    def score(self, query: str, doc: str) -> float:
        combined = np.concatenate([self._encode(query), self._encode(doc)])
        w_full = np.concatenate([self.w, self.w])  # shared weights for demo
        return float(1 / (1 + np.exp(-w_full @ combined)))  # sigmoid

    def train_step(self, query: str, pos_doc: str, neg_doc: str):
        """Binary cross-entropy on positive/negative pair."""
        pos_score = self.score(query, pos_doc)
        neg_score = self.score(query, neg_doc)
        # Minimise: -log(pos_score) - log(1 - neg_score)
        loss = -np.log(pos_score + 1e-9) - np.log(1 - neg_score + 1e-9)
        # In a real system: autograd handles this. Here: manual gradient step omitted
        # for clarity. The pattern — joint input, pairwise loss — is the key.
        return loss


def rerank(
    query: str,
    candidates: list[tuple[int, float]],
    corpus: list[str],
    cross_encoder,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Rerank a list of (doc_id, first_stage_score) using a cross-encoder."""
    rescored = [
        (doc_id, cross_encoder.score(query, corpus[doc_id]))
        for doc_id, _ in candidates
    ]
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored[:top_k]


ce = ToyLinearCrossEncoder()
shortlist = hybrid_search("cardiac arrest treatment", bm25, dense, top_k_final=5)
final = rerank("cardiac arrest treatment", shortlist, corpus, ce, top_k=3)
print("After reranking:")
for doc_id, score in final:
    print(f"  [ce={score:.3f}] {corpus[doc_id][:65]}")`}
      </CodeBlock>

      <H3>4e. End-to-end pipeline: sparse + dense → RRF → top-20 → rerank → top-5</H3>

      <Prose>
        The complete pipeline function wraps every stage above into a single call. The shape is: retrieve wide from both channels, fuse with RRF, take the top-20 merged candidates, rerank with the cross-encoder, return the top-5 with their final scores. This is the architecture that the production section describes at library and API level.
      </Prose>

      <CodeBlock language="python">
{`def full_pipeline(
    query: str,
    bm25_engine: BM25,
    dense_retriever: MockDenseRetriever,
    cross_encoder,
    corpus: list[str],
    channel_k: int = 100,   # how many to retrieve per channel
    rrf_k: int = 60,        # RRF smoothing constant
    rerank_k: int = 20,     # how many to send to reranker
    final_k: int = 5,       # how many to return to the LLM
) -> list[dict]:
    # Stage 1: sparse retrieval
    sparse_hits = [doc_id for doc_id, _ in bm25_engine.search(query, channel_k)]

    # Stage 2: dense retrieval
    dense_hits = dense_retriever.search(query, channel_k)

    # Stage 3: RRF fusion → top-rerank_k
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=rrf_k)
    shortlist = fused[:rerank_k]

    # Stage 4: cross-encoder reranking → top-final_k
    rescored = rerank(query, shortlist, corpus, cross_encoder, top_k=final_k)

    return [
        {
            "doc_id": doc_id,
            "text": corpus[doc_id],
            "rerank_score": score,
        }
        for doc_id, score in rescored
    ]


results = full_pipeline(
    query="TX-4891-C firmware installation",
    bm25_engine=bm25,
    dense_retriever=dense,
    cross_encoder=ce,
    corpus=corpus,
)
print("\\nFinal top-5 for LLM context:")
for r in results:
    print(f"  [{r['rerank_score']:.3f}] {r['text'][:70]}")`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production systems</H2>

      <Prose>
        In production the components above are supplied by purpose-built services rather than implemented from scratch. The sparse channel is almost always served by an existing search engine that the organization already operates for full-text search. Elasticsearch and OpenSearch both implement BM25 natively in their inverted index layer and expose a hybrid query API that combines BM25 scores with dense vector scores using a configurable weighting scheme. The advantage of this path is operational: there is no new service to run, and the BM25 index is already being maintained for other uses. The disadvantage is that the BM25 and dense scores are on different scales and normalization must be managed explicitly.
      </Prose>

      <Prose>
        Weaviate and Qdrant are vector databases that have added native hybrid search support, treating sparse and dense retrieval as first-class dual channels over the same collection. Weaviate supports BM25F (BM25 with per-property weighting) as the sparse channel alongside dense vector search, with RRF and relative-score fusion available as the combiner. Qdrant stores sparse vectors — including SPLADE outputs — in a separate inverted index structure alongside the HNSW graph for dense vectors, and its Query API runs both channels in parallel and applies the chosen fusion algorithm before returning results. Both expose the fusion configuration in their query API. A Weaviate hybrid query looks like:
      </Prose>

      <CodeBlock language="python">
{`import weaviate
import weaviate.classes.query as wq

client = weaviate.connect_to_local()
collection = client.collections.get("Documents")

# Hybrid search: dense + BM25 → RRF fusion → top-10
response = collection.query.hybrid(
    query="TX-4891-C firmware installation guide",
    alpha=0.5,                      # 0 = pure sparse, 1 = pure dense, 0.5 = balanced
    fusion_type=wq.HybridFusion.RANKED,   # RRF; use RELATIVE_SCORE for score-based
    limit=10,
    return_metadata=wq.MetadataQuery(score=True, explain_score=True),
)

for obj in response.objects:
    print(f"score={obj.metadata.score:.4f}: {obj.properties['text'][:60]}")`}
      </CodeBlock>

      <Prose>
        Pinecone supports sparse-dense vectors as a first-class index type: a single vector record can carry both a dense embedding (for ANN lookup) and a sparse weight vector (for inverted-index lookup), and queries can specify a weighted combination of both similarity scores. This is particularly useful when serving SPLADE sparse representations alongside a dense embedding model, since both can be stored and queried in a single index without managing two separate services.
      </Prose>

      <Prose>
        For reranking, the options split by hosting preference. Cohere's Rerank API (cohere.rerank) is the most widely used managed reranking service: it accepts a query and a list of candidate texts, and returns relevance scores from Cohere's cross-encoder model, with no infrastructure required. BGE Reranker (BAAI/bge-reranker-large, bge-reranker-v2-m3, bge-reranker-v2-gemma) are open-weight cross-encoder models available on Hugging Face, suitable for self-hosting on a small GPU instance and covering multilingual queries from the v2 series. Jina Reranker and Voyage AI's reranker service are API-based alternatives. The integration surface in every case is the same: pass the query and the shortlist of candidate strings, receive an ordered list of scores, take the top-K.
      </Prose>

      <CodeBlock language="python">
{`# Cohere Rerank API — drop-in reranking for any retrieval shortlist
import cohere

co = cohere.Client("YOUR_API_KEY")

shortlist_texts = [corpus[doc_id] for doc_id, _ in fused[:20]]

response = co.rerank(
    model="rerank-english-v3.0",
    query="TX-4891-C firmware installation",
    documents=shortlist_texts,
    top_n=5,
)

for hit in response.results:
    print(f"  [relevance={hit.relevance_score:.4f}] {shortlist_texts[hit.index][:65]}")`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL
          ====================================================================== */}
      <H2>6. Visualization</H2>

      <Prose>
        The plot below shows recall@K — the fraction of relevant documents appearing in the top-K results — for four retrieval configurations across increasing K values. Sparse-only and dense-only plateau below hybrid; hybrid with reranking delivers the highest recall at any given K by correcting the ordering after the fusion step.
      </Prose>

      <Plot
        label="recall@K — sparse vs dense vs hybrid vs hybrid+rerank"
        xLabel="K (number of results)"
        yLabel="Recall@K"
        series={[
          {
            name: "sparse only (BM25)",
            color: colors.gold,
            points: [[1,0.31],[3,0.47],[5,0.55],[10,0.63],[20,0.70],[50,0.76]],
          },
          {
            name: "dense only (bi-encoder)",
            color: "#60a5fa",
            points: [[1,0.33],[3,0.51],[5,0.58],[10,0.67],[20,0.73],[50,0.78]],
          },
          {
            name: "hybrid (RRF)",
            color: "#c084fc",
            points: [[1,0.41],[3,0.60],[5,0.69],[10,0.77],[20,0.83],[50,0.88]],
          },
          {
            name: "hybrid + rerank",
            color: "#4ade80",
            points: [[1,0.56],[3,0.72],[5,0.80],[10,0.85],[20,0.88],[50,0.90]],
          },
        ]}
        width={520}
        height={260}
      />

      <Prose>
        The heatmap below shows how sparse, dense, and combined scores distribute across a set of representative queries and documents. Rows are query types; columns are score channels. The combined (RRF) channel tends to have the most uniform high scores across query types because it inherits the strengths of both channels — lexical queries score well from the sparse channel, semantic queries from the dense channel, and both contribute to the fused score.
      </Prose>

      <Heatmap
        label="query-doc score breakdown — sparse / dense / combined"
        rowLabels={["product code lookup", "semantic intent query", "error message search", "natural language Q&A", "mixed: entity + intent"]}
        colLabels={["sparse (BM25)", "dense (ANN)", "hybrid (RRF)"]}
        matrix={[
          [0.91, 0.38, 0.82],
          [0.14, 0.88, 0.79],
          [0.85, 0.44, 0.80],
          [0.22, 0.83, 0.76],
          [0.73, 0.71, 0.88],
        ]}
        colorScale="gold"
        cellSize={44}
      />

      <Prose>
        The step trace below shows the full retrieve-rerank pipeline in the form a production system processes a single request — from the raw user query to the final shortlist passed to the language model.
      </Prose>

      <StepTrace
        label="full hybrid retrieve → rerank pipeline"
        steps={[
          { label: "1. sparse retrieval — top 100 (BM25 / SPLADE)", render: () => (
            <TokenStream tokens={[
              { label: "query tokens → inverted index → 100 candidates", color: colors.gold },
            ]} />
          ) },
          { label: "2. dense retrieval — top 100 (bi-encoder + HNSW)", render: () => (
            <TokenStream tokens={[
              { label: "query → embedding → ANN → 100 candidates", color: "#60a5fa" },
            ]} />
          ) },
          { label: "3. RRF fusion → merged top-20", render: () => (
            <TokenStream tokens={[
              { label: "rank(sparse) + rank(dense) → RRF score → top-20", color: "#c084fc" },
            ]} />
          ) },
          { label: "4. cross-encoder reranking (top-20 → top-5)", render: () => (
            <TokenStream tokens={[
              { label: "[query; doc] → cross-encoder → relevance scores → sort", color: "#4ade80" },
            ]} />
          ) },
          { label: "5. top-5 to LLM context", render: () => (
            <TokenStream tokens={[
              { label: "5 reranked chunks → prompt construction → LLM → grounded answer", color: "#4ade80" },
            ]} />
          ) },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix — which retrieval mode for which workload</H2>

      <Prose>
        The choice among retrieval architectures is driven by the shape of the query distribution, the latency budget, and the engineering complexity the team is willing to operate. Four scenarios cover the majority of real deployments.
      </Prose>

      <CodeBlock>
{`scenario                   best retrieval          rationale
─────────────────────────────────────────────────────────────────────────────
keyword-heavy corpus        sparse only (BM25)      exact match is the signal;
  (error codes, part nums,                          no embedding model needed;
   legal citations)                                 lowest latency, simplest ops

semantic / paraphrased      dense only              meaning varies across
  queries, single domain    (bi-encoder)            phrasings; no rare tokens;
  (e.g., FAQ chatbot)                               single retrieval path

general RAG — mixed         hybrid (RRF)            covers both query shapes;
  query distribution        sparse + dense          consistent improvement over
  (most production RAG)                             either mode alone; moderate
                                                    operational complexity

high-stakes accuracy:       hybrid + rerank         best result quality;
  legal, medical,           sparse+dense+           adds cross-encoder pass;
  customer support SLA      cross-encoder           ~50–200ms added latency;
                                                    worth it when correctness
                                                    matters more than speed`}
      </CodeBlock>

      <Prose>
        The hybrid-only configuration without reranking is the correct starting point for teams building their first production RAG system: it adds the sparse channel to whatever vector store is already deployed, applies RRF in a dozen lines of code, and delivers a consistent 10–20% recall improvement over dense-only retrieval with no change to the prompt, the LLM, or the chunking strategy. Reranking can be layered on top once the hybrid baseline is stable and the additional latency has been measured against the application's SLA. Learned sparse retrieval (SPLADE) is worth evaluating when the domain has heavy jargon or acronym usage that BM25 handles well, and when the team has the infrastructure to run neural indexing at corpus-update time.
      </Prose>

      <Callout accent="gold">
        If you can only add one thing to a dense-only RAG system, add reranking. If you can add two, add sparse retrieval and reranking. In that order.
      </Callout>

      {/* ======================================================================
          8. SCALING
          ====================================================================== */}
      <H2>8. Scaling considerations</H2>

      <Prose>
        At tens of millions of documents, BM25 over an inverted index is already fast — query latency is dominated by posting-list traversal and is measured in single-digit milliseconds on a single machine. The scaling unit is the shard: partition the corpus across nodes, run BM25 in parallel on each shard, and merge the per-shard top-K results globally before fusion. Elasticsearch and OpenSearch handle this automatically as part of their distributed index architecture. The BM25 statistics — document frequencies, average document length — need to be synchronized across shards when the global IDF is required, which adds a coordination step at index time but does not affect query latency. At billion-document scale, the inverted index itself becomes large enough to require careful partition planning, but the algorithmic complexity of BM25 does not change: it is still a posting-list lookup and a scoring pass over the candidates.
      </Prose>

      <Prose>
        Hybrid retrieval adds latency for the two retrieval channels and the RRF step. Because BM25 and dense retrieval are independent, they can run in parallel, and the RRF step is trivial. In practice the latency of a hybrid query is approximately equal to the latency of the slower of the two channels — typically the dense channel, because ANN index traversal involves more memory accesses than posting-list lookup. At moderate scale ({"<"}100M documents), total first-stage hybrid retrieval runs in 20–80 milliseconds depending on the ANN index configuration and the size of the candidate set. Teams that find dense retrieval latency unacceptable have two options: reduce the ANN index <Code>ef_search</Code> parameter to trade recall for speed, or switch from HNSW to IVF or DiskANN for better memory-bandwidth efficiency at the cost of slightly higher recall drop.
      </Prose>

      <Prose>
        Reranking latency scales linearly with the shortlist size K: each (query, document) pair requires one cross-encoder forward pass, and those passes are sequential unless batched. A cross-encoder on a small GPU or optimized CPU (e.g., using int8 quantization) processes roughly 50–200 pairs per second. Reranking a top-20 shortlist therefore adds 100–400 milliseconds of latency per query in a single-threaded configuration. Batching multiple queries together and running them through the cross-encoder simultaneously amortizes the overhead, and distilled reranker models (bge-reranker-v2-m3 is based on an M3 backbone with deliberate efficiency engineering) bring the per-pair latency down substantially. The practical design choice is to tune K — the rerank shortlist size — to balance the quality improvement against the latency budget: reranking 50 candidates returns slightly higher recall than reranking 20, but the latency doubles. Reranking 10 is meaningfully faster and still returns the bulk of the quality gain.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <Prose>
        Eight failure patterns appear repeatedly in production hybrid search deployments. Each has a signature and a known mitigation.
      </Prose>

      <Prose>
        <strong>Lexical overweighting on product codes dominating intent.</strong> A query like "TX-4891-C is drawing too much current" has both an entity reference and a semantic intent. BM25 scores the entity highly and the sparse channel dominates the fusion, returning documents about TX-4891-C product specs regardless of whether they discuss current draw. Fix: adjust the RRF <Code>k</Code> parameter or the fusion weights to reduce sparse channel contribution, or use a query classifier to route entity-heavy queries to sparse and semantic queries to dense before the fusion step.
      </Prose>

      <Prose>
        <strong>Fusion score normalization bugs.</strong> Teams that implement weighted-sum fusion instead of RRF often fail to normalize scores to the same range before combining them. BM25 scores can be in the range [0, 30] while cosine similarities are in [-1, 1]. Summing without normalization causes the BM25 score to dominate completely, making the dense channel invisible. Fix: either use RRF, which operates on ranks and is inherently scale-invariant, or normalize each channel to [0, 1] using min-max scaling over the candidate set before combining.
      </Prose>

      <Prose>
        <strong>BM25 parameters (k₁, b) untuned for the corpus.</strong> The defaults of k₁ = 1.2 and b = 0.75 were calibrated on news and academic corpora from the 1990s. Technical documentation, code, chat logs, and structured data all have very different term-frequency distributions and document-length distributions. Running with defaults on a corpus of short API reference pages (where length normalization hurts more than helps) or long legal contracts (where term repetition is meaningful) degrades BM25 performance predictably. Fix: run a grid search over k₁ ∈ {"{"}1.0, 1.2, 1.5, 2.0{"}"} and b ∈ {"{"}0.0, 0.25, 0.5, 0.75{"}"} on a held-out development set with retrieval recall@10 as the metric.
      </Prose>

      <Prose>
        <strong>Stale BM25 statistics after corpus update.</strong> BM25 IDF values are computed over the full corpus at index time. When the corpus grows — new documents are added — the IDF values become stale: terms that were rare in the original corpus may now be common, and the scores for previously indexed documents are calculated against an outdated baseline. The effect is subtle but measurable: documents indexed early in a growing corpus can be systematically overscored for terms that are now common. Fix: recompute global corpus statistics periodically, or use approximate IDF estimates that are updated incrementally via online algorithms.
      </Prose>

      <Prose>
        <strong>Cross-encoder out of domain.</strong> A cross-encoder trained on general web passages (MS MARCO is the dominant training set) will apply web-document relevance priors when reranking medical literature, legal filings, or software documentation. The model scores queries and documents jointly but may weight different aspects of relevance than the domain requires. Fix: fine-tune the reranker on domain-specific annotation, or use a general reranker only as a secondary signal combined with domain-specific scoring heuristics.
      </Prose>

      <Prose>
        <strong>Reranker latency dominating the pipeline.</strong> Reranking a shortlist of 100 candidates can take longer than the LLM generation step, which makes the pipeline feel sluggish in interactive contexts. Teams that set K = 100 because "more is better" pay the full latency cost without proportionally more quality gain. Fix: measure recall@5 versus rerank-K on a development set. The curve typically flattens sharply after K = 20. Reranking 20–30 candidates delivers the bulk of the quality benefit at roughly one-quarter the latency of reranking 100.
      </Prose>

      <Prose>
        <strong>SPLADE vocabulary drift.</strong> SPLADE's query expansion is a function of the underlying language model's vocabulary. When the corpus contains domain-specific terminology that the base BERT model did not encounter in pretraining — specialized chemical names, internal product codes, proprietary acronyms — SPLADE's expansion weights for those terms are unreliable or zero. The model cannot expand "TX-4891-C" into related terms if it has no representation of what that identifier means. Fix: either fine-tune SPLADE on domain-specific text to update the expansion behavior, or fall back to raw BM25 for low-frequency terms where SPLADE's expansion weights are below a confidence threshold.
      </Prose>

      <Prose>
        <strong>Hybrid fusion weight tuning overfits to the evaluation set.</strong> Teams that tune the alpha parameter (the sparse-to-dense weighting) or the per-list RRF weights on a labeled development set can overfit to the specific query distribution of that set. When the production query distribution shifts — seasonally, after product changes, after marketing campaigns — the tuned weights become suboptimal. Fix: use RRF as the default (no tuning required), hold out a separate test set from the evaluation set used for tuning, and monitor retrieval recall on live traffic using a sample of queries with annotated relevance.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The foundational papers are worth reading in sequence. Robertson and Walker's 1994 SIGIR paper "Some Simple Effective Approximations to the 2-Poisson Model for Probabilistic Weighted Retrieval" derived the BM25 formula from probabilistic retrieval theory. The comprehensive treatment is Robertson and Zaragoza's 2009 survey "The Probabilistic Relevance Framework: BM25 and Beyond," published in Foundations and Trends in Information Retrieval (Vol 3, No 4), which gives the full derivation of BM25F, discusses the parameter sensitivity analysis, and situates BM25 in the broader probabilistic IR literature. Thirty years after it was published, BM25 remains the strongest sparse baseline in most retrieval benchmarks.
      </Prose>

      <Prose>
        Cormack, Clarke, and Büttcher's "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods" (SIGIR 2009, proceedings of the 32nd Annual International ACM SIGIR Conference) is the paper that introduced RRF. It demonstrated that the simple rank-combination formula — using only ordinal position and a smoothing constant — outperformed learned rank aggregation methods on the TREC fusion tasks, a result that seemed counterintuitive at the time and has held up ever since. The paper is short, mathematically accessible, and explains the theory behind why ordinal fusion is robust to the scale mismatches that plague score-based fusion.
      </Prose>

      <Prose>
        Khattab and Zaharia's "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT" (arXiv:2004.12832, SIGIR 2020) introduced the per-token embedding architecture and MaxSim aggregation that made late interaction retrieval practical. ColBERTv2 (arXiv:2112.01488, NAACL 2022) improved quality through residual compression and hard negative distillation. Both papers are important for understanding why late interaction represents a distinct point in the recall-latency tradeoff space — more expressive than bi-encoders without the quadratic cost of cross-encoders.
      </Prose>

      <Prose>
        Formal, Piwowarski, and Clinchant's "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking" (arXiv:2107.05720, SIGIR 2021) introduced learned sparse retrieval. SPLADE v2 (arXiv:2109.10086) and subsequent work from NAVER Labs have refined the training procedure and extended the approach to efficient inference. The Wikipedia article on learned sparse retrieval provides a useful taxonomy of the broader family of sparse neural retrieval models that SPLADE belongs to.
      </Prose>

      <Prose>
        BAAI's BGE Reranker family (bge-reranker-large, bge-reranker-v2-m3, bge-reranker-v2-gemma, released through 2024 via the FlagEmbedding project on GitHub and Hugging Face) provides the most widely used open-weight cross-encoder rerankers. The v2-m3 model is trained on the bge-m3 backbone and supports multilingual inputs. The v2-gemma model uses a Gemma-2B backbone for stronger performance at higher compute cost. Both are documented at bge-model.com and in the FlagEmbedding repository.
      </Prose>

      <Prose>
        Production system documentation from Weaviate ("Hybrid Search Explained" and the Hybrid Search documentation at docs.weaviate.io) and Qdrant ("Hybrid Search Revamped" and the sparse vectors article at qdrant.tech) provide the best practical descriptions of how fusion algorithms are implemented in real vector database systems, including the tradeoffs between RRF and relative-score fusion under different query distributions.
      </Prose>

      {/* ======================================================================
          11. EXERCISES
          ====================================================================== */}
      <H2>11. Exercises</H2>

      <Prose>
        <strong>Exercise 1.</strong> Take the BM25 implementation from section 4a and add stemming using the NLTK Porter stemmer before indexing and before query tokenization. Build a corpus of 50 documents from Wikipedia on a topic of your choice, index it, and measure recall@10 on 20 hand-written queries with and without stemming. Report the difference and discuss which query types benefit most.
      </Prose>

      <Prose>
        <strong>Exercise 2.</strong> Implement the full hybrid fusion pipeline from section 4b using a real dense retrieval model — use <Code>sentence-transformers/all-MiniLM-L6-v2</Code> via the <Code>sentence-transformers</Code> library in place of the mock dense retriever. Run hybrid versus dense-only on the same 20 queries from exercise 1. Plot recall@{"{"}1,3,5,10{"}"} for both configurations and identify three queries where hybrid clearly beats dense and explain why.
      </Prose>

      <Prose>
        <strong>Exercise 3.</strong> Implement a parameter sweep over BM25's <Code>k₁</Code> ∈ {"{"}0.5, 1.0, 1.5, 2.0, 2.5{"}"} and <Code>b</Code> ∈ {"{"}0.0, 0.25, 0.5, 0.75, 1.0{"}"} on a short-document corpus (each document {"<"}100 words) and a long-document corpus (each document {">"} 1000 words). Report the optimal parameter values for each and explain why the optimal <Code>b</Code> differs between the two corpus types.
      </Prose>

      <Prose>
        <strong>Exercise 4.</strong> Instrument the end-to-end pipeline from section 4e to measure latency for each stage separately: sparse retrieval, dense retrieval, RRF fusion, and reranking. Run it on a corpus of 10,000 documents with K = 20, K = 50, and K = 100 rerank shortlist sizes. Plot reranking latency versus K and identify the point of diminishing returns in recall improvement per millisecond of added reranking latency.
      </Prose>

      <Prose>
        <strong>Exercise 5.</strong> Pick a domain-specific corpus that contains many rare identifiers — software package names, chemical compound IDs, or legal statute references. Build a hybrid retrieval system and evaluate it on a set of 20 identifier-lookup queries and 20 semantic-intent queries. Compute recall@5 for: sparse-only, dense-only, and hybrid. Confirm that hybrid wins on the union and report whether either single mode wins across both query types.
      </Prose>

      {/* ======================================================================
          TRACK CLOSER
          ====================================================================== */}
      <H2>Closing — the full Large Language Models track</H2>

      <Prose>
        This topic closes the Long Context and Retrieval section, and with it the Large Language Models track. It is worth pausing to see what the arc looks like from start to finish, because the pieces connect tightly and the connections are not always visible when each topic is read in isolation.
      </Prose>

      <Prose>
        The track began with Tokenization — with the question of how raw text gets turned into the integers a neural network can consume. That question seemed narrow and mechanical, and it is, but the answer shapes almost everything downstream: the size of the embedding table, the number of steps in every training and inference sequence, how the model generalizes across morphological variants and across scripts, and how much of a fixed context window is consumed by a given piece of content. The tokenizer's decisions are the first compression step in a chain that runs all the way to the final generated word.
      </Prose>

      <Prose>
        Pre-Training turned those token sequences into knowledge by training the model to predict the next token across hundreds of billions of examples. The pretraining stage is where most of the model's capabilities come from — its factual knowledge, its syntactic understanding, its ability to reason within a sequence. But pretraining alone produces a model that predicts the next token, not one that follows instructions or avoids harmful outputs. That gap is closed by Post-Training: RLHF, DPO, RLAIF, and the broader family of alignment techniques that steer the model's behavior away from the raw pretraining distribution toward something useful and safe. Post-training also produced the reinforcement-learning-for-reasoning advances — GRPO, RLVR, test-time compute scaling — that have pushed frontier model capabilities further than pretraining alone could have.
      </Prose>

      <Prose>
        Inference Optimization addressed the uncomfortable fact that serving a 70B-parameter model in production at low latency and reasonable cost requires considerable engineering beyond simply running the forward pass. KV-cache management, speculative decoding, prefix caching, continuous batching, FP8 quantization, and the broader family of techniques covered in the inference section turn a theoretically capable model into a practically deployable one. Inference System Design scaled that further: disaggregated prefill-decode, multi-model serving, request routing, autoscaling GPU fleets, and the global multi-region architectures that serve production LLM APIs at tens of thousands of requests per second.
      </Prose>

      <Prose>
        Long Context and Retrieval addressed the question that all of the above still leaves open: a model trained on a corpus that ended at a cutoff date, with a context window of finite size, cannot know what it was not trained on. The section traced the arc from raw retrieval (RAG — retrieve relevant chunks, paste them into the prompt, generate a grounded answer) through the systems that make retrieval accurate and scalable (embedding models, vector databases, ANN indexing, metadata filtering) through the more sophisticated retrieval patterns that handle hard queries (GraphRAG's structured knowledge graph, agentic RAG's self-directed retrieval loops) to, finally, this topic: the hybrid architecture that combines sparse lexical retrieval, dense semantic retrieval, and cross-encoder reranking into the retrieval backbone that most serious production RAG systems converge on.
      </Prose>

      <Prose>
        Together these topics describe every layer of what makes a modern large language model work: from the bytes that enter the tokenizer, through the pretraining that forms the model's knowledge, through the post-training that shapes its behavior, through the inference engineering that makes it serve at scale, through the retrieval machinery that extends its knowledge past what its weights contain. No single topic is the whole picture. The system is the composition.
      </Prose>

      <Callout accent="gold">
        We have gone from bytes in to tokens out, through every layer of what makes a modern LLM work. The next tracks — Deep Learning Fundamentals, Classical ML, Reinforcement Learning — pick up the threads from a different angle.
      </Callout>

      <Prose>
        Deep Learning Fundamentals will revisit backpropagation, attention, and normalization not as LLM components but as mathematical objects in their own right — what the theory actually says about why they work, where they fail, and what the optimization landscape looks like. Classical ML will trace the lineage of supervised and unsupervised learning that preceded transformers, the techniques that still dominate tabular and structured-data problems, and the statistical foundations that the deep learning era has sometimes obscured. Reinforcement Learning will take the RL signal that shows up in post-training as a black box and open it: Markov decision processes, policy gradients, value functions, and the connection from Q-learning to the GRPO and RLVR techniques the LLM track described in passing. Each track is a different angle on the same underlying substrate. The depth compounds across them.
      </Prose>
    </div>
  ),
};

export default hybridSearch;
