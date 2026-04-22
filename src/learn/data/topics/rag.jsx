import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const rag = {
  title: "Retrieval-Augmented Generation (RAG)",
  slug: "retrieval-augmented-generation-rag",
  readTime: "~45 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        A language model's weights are a frozen snapshot. Everything it knows — every fact, every relationship, every bit of world knowledge — was baked in during pretraining on a corpus that has a hard cutoff date. Ask it about yesterday's earnings call, a new product specification written this morning, a policy document that was revised last week, or a database row that changes hourly, and you are asking it to produce something from nothing. The model has two failure modes for knowledge it was never trained on: refuse (honest, but useless to a product), or hallucinate (confident-sounding, drawn from the nearest plausible pattern the weights have, wrong). For any real deployment that lives outside the model's frozen training window — which is nearly every production deployment — neither failure mode is acceptable.
      </Prose>

      <Prose>
        Three additional problems compound the knowledge-cutoff problem. First, attribution: a model generating from parametric memory cannot cite where a fact came from, because the fact is distributed diffusely across billions of parameters rather than located in a specific document. Second, access control: you cannot tell a model to "only use information the user is authorized to see" when all information is uniformly fused into weights. Third, context economics: fine-tuning a model every time a document changes costs thousands of dollars and several hours of compute; you cannot do it per-ticket or per-policy-revision. These three problems — freshness, provenance, and scale — together describe the gap that Retrieval-Augmented Generation was designed to close.
      </Prose>

      <Prose>
        The RAG idea, stated plainly: rather than baking knowledge into weights, keep it in a queryable external store and inject the relevant pieces into the prompt at query time. At inference, you retrieve documents near the question, paste them as context, and let the model generate against real evidence. The model's parametric memory handles reasoning, language, and format; the external store handles facts. The separation is the insight.
      </Prose>

      <Prose>
        The historical record is precise. Kelvin Guu, Kenton Lee, and collaborators at Google published REALM (arXiv:2002.08909) in February 2020, the first system to incorporate retrieval into language model pretraining itself — the retriever and the masked-LM objective were trained jointly, with gradients flowing through the retrieval step over millions of Wikipedia documents. Three months later, Patrick Lewis, Ethan Perez, and collaborators at Facebook AI Research published the paper that named the paradigm: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (arXiv:2005.11401, May 2020). Lewis et al. introduced the RAG model proper: a seq2seq network (BART) whose encoder conditions on both the query and retrieved passages, with a DPR retriever trained end-to-end. On open-domain QA, RAG beat both parametric-only seq2seq models and extract-then-answer pipelines, while generating more specific, diverse, and factual language. The same year, Gautier Izacard and Édouard Grave at Facebook published Fusion-in-Decoder (FiD, arXiv:2007.01282), showing that encoding each retrieved passage independently and fusing in the decoder — rather than concatenating everything — scaled more gracefully to larger passage counts and set state-of-the-art on NaturalQuestions and TriviaQA. These three papers established the vocabulary, the benchmarks, and the conceptual architecture that modern RAG stacks are built on.
      </Prose>

      <Prose>
        It is worth being precise about what RAG does and does not change. RAG does not make a model smarter — it does not improve the model's reasoning capacity, its ability to follow multi-step instructions, or its calibration on uncertain claims. What it changes is the information the model has access to at generation time. A model that hallucinates because it does not know the correct answer will stop hallucinating about things that are in its retrieval index; it will continue to hallucinate about things that are not. The improvement is strictly in the domain of knowledge, not capability. This distinction matters because teams sometimes adopt RAG hoping to fix problems that are fundamentally capability failures — a model that cannot follow complex policy logic — and then discover that injecting the policy text into the prompt does not help if the model cannot reason about the policy correctly. RAG and model capability are orthogonal axes; both need to be adequate for the system to work.
      </Prose>

      <Callout accent="gold">
        RAG's three contributions: factual grounding (answers cite real documents), freshness (index the new document, done), and cost (reindexing one chunk is orders of magnitude cheaper than fine-tuning).
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Strip the engineering away and the RAG pipeline is four stages across two timescales. The offline stages run once per document, amortized over every future query. The online stages run once per user request, inside the latency budget.
      </Prose>

      <StepTrace
        label="the four-stage rag pipeline"
        steps={[
          {
            label: "Stage 1 — INDEX (offline)",
            render: () => (
              <div>
                <TokenStream
                  label="document corpus → chunks → embeddings → vector db"
                  tokens={[
                    { label: "raw docs", color: colors.textMuted },
                    { label: "→ chunk", color: colors.gold },
                    { label: "→ embed", color: "#60a5fa" },
                    { label: "→ FAISS / Qdrant / pgvector", color: "#c084fc" },
                  ]}
                />
                <Prose>
                  Each document is split into chunks (200–1000 tokens each), each chunk is embedded into a dense vector by a bi-encoder model (e.g. BGE, E5, text-embedding-3), and the vectors are stored in an approximate-nearest-neighbor index. This step is paid once per chunk and amortized over all future queries.
                </Prose>
              </div>
            ),
          },
          {
            label: "Stage 2 — QUERY (online)",
            render: () => (
              <div>
                <TokenStream
                  label="user question → embed → ANN lookup → top-k chunks"
                  tokens={[
                    { label: "user question", color: colors.gold },
                    { label: "→ same bi-encoder", color: "#60a5fa" },
                    { label: "→ cosine search", color: "#c084fc" },
                    { label: "→ top-5 chunks", color: colors.green },
                  ]}
                />
                <Prose>
                  The user's question is embedded with the same model used at index time, producing a query vector. Approximate nearest-neighbor search (HNSW, IVF, ScaNN) finds the top-k most similar chunk vectors in milliseconds, even over billion-document indexes.
                </Prose>
              </div>
            ),
          },
          {
            label: "Stage 3 — AUGMENT (online)",
            render: () => (
              <div>
                <TokenStream
                  label="system prompt + retrieved chunks + user question"
                  tokens={[
                    { label: "[SYSTEM]", color: colors.textMuted },
                    { label: "[Source 1] chunk text...", color: colors.gold },
                    { label: "[Source 2] chunk text...", color: "#60a5fa" },
                    { label: "[Source 3] chunk text...", color: "#c084fc" },
                    { label: "[QUESTION]", color: colors.green },
                  ]}
                />
                <Prose>
                  Retrieved chunks are formatted with source labels and injected into the prompt. Good templates instruct the model to cite by source number, establishing an audit trail. Context placement matters: frontier models show primacy and recency bias; critical evidence goes first or last, not buried in the middle.
                </Prose>
              </div>
            ),
          },
          {
            label: "Stage 4 — GENERATE (online)",
            render: () => (
              <div>
                <TokenStream
                  label="llm generates grounded answer from retrieved context"
                  tokens={[
                    { label: "Based on [Source 2],", color: colors.green },
                    { label: " the return window is 30 days", color: colors.gold },
                    { label: " for international orders [Source 1].", color: "#60a5fa" },
                  ]}
                />
                <Prose>
                  The LLM generates its answer conditioned on the retrieved context. Because the answer draws from real documents injected into the prompt, it is grounded and citable. The model's parametric knowledge handles language, structure, and reasoning; the retrieved context handles facts.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      <Prose>
        Every stage is independently tunable and independently measurable, which is both the power and the difficulty of RAG. The power: you can improve retrieval without touching the LLM, update the index without retraining anything, swap the embedding model without changing the generator. The difficulty: a failure anywhere in the chain degrades the final answer, and without per-stage metrics you cannot tell where quality is being lost. Practitioners who treat RAG as a single opaque system spend months chasing generation quality that is actually a retrieval problem, or a chunking problem, or a prompt template problem.
      </Prose>

      <Prose>
        The two-timescale architecture is what makes RAG economically viable. Embedding a corpus of one million chunks takes hours and runs once. Querying the resulting index at interactive latency costs milliseconds per request. The asymmetry is a first-class design constraint: everything expensive happens offline and gets amortized. Retrieving the right chunk in 20ms is how RAG keeps the total query latency — embedding the question, doing the ANN lookup, constructing the prompt, and waiting for the LLM — under 500ms for a typical chatbot interaction.
      </Prose>

      <Prose>
        There is a persistent misconception that long-context LLMs — models with million-token windows — make RAG obsolete. They do not, for three compounding reasons. First, cost: a million-token context costs roughly one thousand times as much as a one-thousand-token context per inference call; retrieval narrows the input to the few hundred tokens that actually matter. Second, attention degradation: benchmarks consistently show that models' ability to use information buried in the middle of a million-token context degrades sharply past roughly 30,000–60,000 tokens of effective window — more context tokens do not translate linearly into more usable context. Third, freshness: you can update a vector index incrementally in seconds; you cannot update model weights incrementally without full fine-tuning. The case for RAG is not that long-context models are bad; it is that retrieval and long context solve different problems and work best together — use retrieval to select the most relevant 3,000 tokens from a corpus of billions, then use the model's long-context abilities to reason over those 3,000 tokens carefully.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Cosine similarity and the dense retrieval score</H3>

      <Prose>
        The core operation in dense retrieval is cosine similarity between a query vector and a document vector. Both are produced by the same bi-encoder model: pass text through a transformer, pool the last-layer hidden states (mean pool or CLS token), and L2-normalize the result. The retrieval score for a query <Code>q</Code> against document <Code>d</Code> is:
      </Prose>

      <MathBlock caption="cosine similarity — equivalent to dot product when vectors are L2-normalized">
        {"\\text{cos}(q, d) = \\frac{q \\cdot d}{\\|q\\| \\cdot \\|d\\|} = q^\\top d \\quad (\\text{when } \\|q\\| = \\|d\\| = 1)"}
      </MathBlock>

      <Prose>
        When embeddings are L2-normalized — which modern bi-encoders always do — cosine similarity reduces to an inner product, and the retrieval problem reduces to Maximum Inner Product Search (MIPS), which has efficient approximate solutions. This reduction is why dense retrieval can run in milliseconds over billion-document indexes.
      </Prose>

      <Prose>
        The choice of pooling strategy — mean pooling versus CLS token — is non-trivial. BERT was designed with a CLS token that aggregates sequence-level information for classification tasks, but empirical results from the Sentence-BERT paper and subsequent work show that mean pooling over all token positions consistently outperforms CLS pooling for semantic similarity. The reason is geometric: mean pooling averages the signal from every position, making the resulting vector more robust to variations in where informative content appears in the sequence. CLS pooling concentrates information in a single token that may or may not have received adequate attention from all parts of the input. Modern bi-encoder training almost universally uses mean pooling. The one exception is queries, where some models apply a weighted pooling that upweights the query tokens most relevant to the retrieval intent — but this is a fine-grained training-time decision that does not change the inference-time interface.
      </Prose>

      <H3>3.2 Bi-encoder vs cross-encoder</H3>

      <Prose>
        A bi-encoder encodes the query and each document independently. The query vector is computed once and compared against all pre-computed document vectors with a cheap dot product. This makes retrieval fast — O(d) per document comparison, where d is the embedding dimension — and indexable: document vectors are computed offline and cached forever. The catch is that the query and document never interact during encoding. The model must map both into a shared geometric space where their relevance is captured purely by vector proximity, without any ability to catch query-document relationships that emerge only from their joint content.
      </Prose>

      <Prose>
        A cross-encoder encodes the query and document jointly, concatenated into a single sequence, letting every query token attend to every document token through full self-attention. The resulting interaction is far richer — cross-encoders can catch subtle relevance signals that bi-encoders miss, such as a query term that is only meaningful in the context of an adjacent sentence in the document. The price is that it cannot be precomputed. Every (query, document) pair must be run through the encoder at query time, making cross-encoders O(k · L²) where k is the number of candidates and L is the sequence length. At k=1,000,000 documents and L=512 tokens, this is completely infeasible. At k=20 candidates and L=512 tokens, it takes roughly 150ms on a GPU — within the latency budget for an interactive pipeline that does the first-pass retrieval with the bi-encoder.
      </Prose>

      <Prose>
        The standard production pattern follows directly from this analysis: bi-encoder for the first-pass top-50 (fast, cached, indexed), cross-encoder to rerank the top-50 to top-5 (accurate, slow, applied to only 50 pairs). ColBERT (Khattab and Zaharia, arXiv:2004.12832) occupies an interesting middle position: it encodes query and document independently token-by-token but then scores them with a MaxSim operation — the maximum similarity between each query token and every document token. This late interaction preserves the indexability of bi-encoders while capturing more fine-grained token-level signal than a single pooled vector, at the cost of larger index size (one vector per token rather than one per document).
      </Prose>

      <H3>3.3 BM25 sparse lexical score</H3>

      <Prose>
        BM25 (Robertson and Zaragoza, 2009) is the classical lexical retrieval baseline. For a query term <Code>t</Code> in document <Code>d</Code>:
      </Prose>

      <MathBlock caption="BM25 — TF-IDF with length normalization; k1 ≈ 1.2, b ≈ 0.75 by convention">
        {"\\text{BM25}(t, d) = \\text{IDF}(t) \\cdot \\frac{f(t,d) \\cdot (k_1 + 1)}{f(t,d) + k_1 \\left(1 - b + b \\cdot \\frac{|d|}{\\text{avgdl}}\\right)}"}
      </MathBlock>

      <Prose>
        where <Code>f(t,d)</Code> is term frequency in document <Code>d</Code>, <Code>|d|</Code> is document length, <Code>avgdl</Code> is the corpus average document length, and IDF is the inverse document frequency. The <Code>k1</Code> parameter controls term frequency saturation (diminishing returns beyond a certain count); <Code>b</Code> controls length normalization. BM25 excels on exact keyword matches, precise entity names, code identifiers, and technical jargon — patterns where the query and document share the exact same surface form.
      </Prose>

      <H3>3.4 Hybrid retrieval score</H3>

      <Prose>
        Hybrid search combines the complementary strengths of dense and sparse retrieval with a linear interpolation:
      </Prose>

      <MathBlock caption="hybrid score — α balances semantic (dense) vs lexical (sparse); α ≈ 0.5 is a strong default">
        {"\\text{score}(q, d) = \\alpha \\cdot \\text{dense}(q, d) + (1 - \\alpha) \\cdot \\text{BM25}(q, d)"}
      </MathBlock>

      <Prose>
        Dense retrieval handles paraphrase, synonym substitution, and semantic similarity — it finds a chunk about "payment returns" even if the query says "refund policy." BM25 handles precise exact matches — it reliably finds the document that literally contains the product model number SKU-48201-X. The queries that defeat dense retrieval rarely defeat BM25, and vice versa. Hybrid retrieval reliably outperforms either alone on standard RAG evaluation benchmarks.
      </Prose>

      <H3>3.5 Chunking and overlap</H3>

      <Prose>
        Fixed-size chunking with a sliding window of size <Code>C</Code> tokens and overlap <Code>O</Code> tokens produces approximately <Code>N / (C - O)</Code> chunks for a document of <Code>N</Code> tokens. Overlap ensures that sentences near a chunk boundary appear in at least two chunks, preventing the case where an answer is split cleanly across a cut. The right overlap is roughly 10–20% of chunk size: enough to capture boundary context, not so much that the index bloats without retrieval benefit.
      </Prose>

      <Prose>
        The chunk size itself is a trade-off between retrieval precision and information completeness. Small chunks (50–150 tokens) have embedding vectors that are highly specific — they retrieve precisely on queries that match their narrow content, and rarely as false positives — but they often lack the surrounding context needed to answer a question. A chunk that says "the penalty is \$250 per violation" retrieves well on "what is the fine?" but cannot tell you what the fine is for unless the previous chunk is also retrieved. Large chunks (800–1500 tokens) have embedding vectors that average over more content, making them less specific and more likely to retrieve as false positives for queries that match a small portion of their content. The sweet spot for most document corpora — policy documents, technical manuals, knowledge base articles — is 300–600 tokens with 15% overlap, tuned by measuring recall@5 on a held-out validation set.
      </Prose>

      <H3>3.6 Retrieval quality metrics</H3>

      <Prose>
        Three standard metrics for evaluating a retriever in isolation. Recall@k measures whether the ground-truth document appears anywhere in the top-k retrieved results — this is the metric that determines the ceiling on generation quality, because a correct answer is impossible if the evidence was never retrieved. Mean Reciprocal Rank (MRR) rewards finding the relevant document sooner in the ranked list: MRR = (1/Q) Σ (1/rank_i). nDCG (normalized Discounted Cumulative Gain) accounts for graded relevance, discounting documents that appear lower in the list by a log factor. In practice, recall@5 is the single most important metric for RAG systems, because most prompts only include k=5 chunks — if recall@5 is 0.65, about a third of your queries will fail due to retrieval.
      </Prose>

      <MathBlock caption="MRR — mean reciprocal rank; rank_i is the position of the first relevant result for query i">
        {"\\text{MRR} = \\frac{1}{|Q|} \\sum_{i=1}^{|Q|} \\frac{1}{\\text{rank}_i}"}
      </MathBlock>

      <MathBlock caption="nDCG@k — normalized discounted cumulative gain; rel_i is the graded relevance at position i">
        {"\\text{nDCG@k} = \\frac{\\text{DCG@k}}{\\text{IDCG@k}}, \\quad \\text{DCG@k} = \\sum_{i=1}^{k} \\frac{\\text{rel}_i}{\\log_2(i+1)}"}
      </MathBlock>

      <Prose>
        For RAG specifically, recall@k is the primary metric because it directly measures whether the right evidence reaches the prompt. MRR and nDCG are more informative when the retrieved set is fed to the model in ranked order and the model gives more weight to earlier items (which it does, due to attention primacy bias). A retriever optimized for MRR ensures the best chunk appears first in the prompt, not just somewhere in the top-5.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Five progressively deeper implementations. All code runs end-to-end; verified outputs are embedded as comments. The goal is to make every design decision tactile before a framework abstracts it. The implementations use bag-of-words and n-gram features rather than real transformer embeddings so that they run in the browser or a local Python environment without GPU access; the patterns are identical to what production systems do with transformer-produced vectors, and the qualitative behavior — which queries retrieve well, which fail, how reranking helps — transfers directly.
      </Prose>

      <H3>4a. Minimal RAG — corpus to answer in 60 lines</H3>

      <Prose>
        The simplest possible working pipeline: a small document corpus, numpy-based cosine similarity as the retriever, and a stub LLM call. No dependencies beyond numpy. The output structure is real; plug in a real embedding model and a real LLM to make it production-ready. The bag-of-words embedding used here produces interpretable behavior: queries containing exact words from the document will retrieve it; queries using synonyms or paraphrases will miss it. This is precisely the limitation that dense bi-encoder embeddings overcome — and the contrast between the two is the most visceral way to understand why transformer-based embeddings matter.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from collections import defaultdict

# --- Minimal corpus ---
CORPUS = [
    "International returns must be initiated within 30 days of delivery.",
    "For domestic orders, the return window is 60 days from the date of purchase.",
    "Refunds are processed to the original payment method within 5-7 business days.",
    "Items marked as 'Final Sale' are not eligible for return or exchange.",
    "To initiate a return, visit our Returns Portal and enter your order number.",
    "Shipping costs for international returns are the responsibility of the customer.",
    "Exchanges are available for size or color variants of the same product.",
    "Damaged items must be reported within 48 hours of delivery with photographic evidence.",
]

# --- Naive bag-of-words embedding (TF, no IDF) ---
def tokenize(text):
    return text.lower().split()

def build_vocab(corpus):
    vocab = {}
    for doc in corpus:
        for tok in tokenize(doc):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab

def embed_bow(text, vocab):
    vec = np.zeros(len(vocab))
    for tok in tokenize(text):
        if tok in vocab:
            vec[vocab[tok]] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# --- Index ---
vocab = build_vocab(CORPUS)
doc_vectors = np.array([embed_bow(doc, vocab) for doc in CORPUS])

def retrieve(query, top_k=3):
    q_vec = embed_bow(query, vocab)
    scores = doc_vectors @ q_vec            # cosine similarity (vectors are L2-normalized)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(CORPUS[i], float(scores[i])) for i in top_idx]

def build_prompt(query, chunks):
    context = "\\n\\n".join(
        f"[Source {i+1}] {text}" for i, (text, _) in enumerate(chunks)
    )
    return f"""Use only the sources below to answer. Cite by source number.

{context}

Question: {query}
Answer:"""

query = "Can I return an international order after 45 days?"
chunks = retrieve(query, top_k=3)
prompt = build_prompt(query, chunks)

# Retrieved:
# [Source 1] International returns must be initiated within 30 days of delivery.  score=0.42
# [Source 2] Shipping costs for international returns are the responsibility...    score=0.37
# [Source 3] Items marked as 'Final Sale' are not eligible for return...           score=0.21
# Prompt → LLM → "Based on [Source 1], international returns must be initiated
# within 30 days, so a 45-day return is outside the eligible window."
print(prompt)`}
      </CodeBlock>

      <H3>4b. Chunking strategies — fixed vs semantic</H3>

      <Prose>
        Chunking is the most underrated design decision in RAG. Three strategies in increasing sophistication. The fixed-size approach is universally used as a starting point because it requires zero domain knowledge: no understanding of document structure, no parsing of markdown or HTML, no semantic model. Its failure mode is equally universal: documents with clear section structure — user manuals, legal contracts, technical specifications — get cut in the middle of a procedure or a clause, producing fragments that retrieve plausibly but answer nothing. Structure-aware chunking uses document signals (markdown headings, HTML <Code>{"<h>"}</Code> tags, legal section numbers) to identify the author's intended units and respects them as chunk boundaries. This is almost always worth the extra parsing complexity for structured documents. The paragraph-overlap approach in the third implementation below is a practical middle ground: it respects semantic paragraph boundaries without requiring document-type-specific parsers.
      </Prose>

      <CodeBlock language="python">
{`def chunk_fixed(text, chunk_size=100, overlap=20):
    """Sliding window. Fast, ignores structure. Good baseline."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def chunk_by_heading(markdown_text, max_tokens=150):
    """Split at ## headings, pack until max_tokens. Preserves author structure."""
    sections = markdown_text.split("\\n## ")
    chunks, current_words = [], []
    for section in sections:
        section_words = section.split()
        if len(current_words) + len(section_words) > max_tokens and current_words:
            chunks.append(" ".join(current_words))
            current_words = section_words
        else:
            current_words.extend(section_words)
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks

def chunk_by_paragraph(text, max_tokens=120, overlap_paragraphs=1):
    """Split at blank lines, merge short paragraphs, carry overlap for context."""
    paragraphs = [p.strip() for p in text.split("\\n\\n") if p.strip()]
    chunks = []
    i = 0
    while i < len(paragraphs):
        current = []
        total = 0
        j = i
        while j < len(paragraphs) and total + len(paragraphs[j].split()) <= max_tokens:
            current.append(paragraphs[j])
            total += len(paragraphs[j].split())
            j += 1
        chunks.append("\\n\\n".join(current))
        i = max(i + 1, j - overlap_paragraphs)
    return chunks

# Prepend metadata to every chunk — reliably lifts recall@5 by 3-8 points
def add_metadata(chunks, doc_title, section=""):
    return [
        f"[{doc_title}{' > ' + section if section else ''}]\\n{chunk}"
        for chunk in chunks
    ]

text = "International returns must be initiated within 30 days..."
meta_chunks = add_metadata(chunk_fixed(text, 80, 10), "Returns Policy", "International")
# → ['[Returns Policy > International]\\nInternational returns must be...']`}
      </CodeBlock>

      <H3>4c. Hybrid retrieval — BM25 + dense, weighted combination</H3>

      <Prose>
        BM25 and dense retrieval fail on orthogonal query types. BM25 wins on exact-match queries (product codes, entity names, precise technical terms, code identifiers) because it is a direct lexical signal: if the document contains the exact string "SKU-4820-XL", BM25 will find it, regardless of what the embedding space thinks about semantic proximity. Dense retrieval wins on paraphrase and semantic similarity: a query "how do I get a refund?" retrieves chunks about "return policy" even though no word in the query appears in the document heading. The failure modes are nearly complementary — BM25 misses paraphrases, dense misses rare exact strings — which is why the linear combination almost always beats either alone on realistic mixed-query corpora. The tuning parameter <Code>alpha</Code> should be set by running both systems on a validation set and measuring recall@5 at a range of alpha values; the optimal value varies significantly by corpus (technical documentation skews toward lower alpha, i.e., more BM25; general-language knowledge bases skew toward higher alpha, i.e., more dense).
      </Prose>

      <CodeBlock language="python">
{`import math
from collections import Counter

# --- BM25 ---
class BM25:
    def __init__(self, corpus, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.tokenized = [doc.lower().split() for doc in corpus]
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in self.tokenized) / self.N
        self.df = Counter(tok for doc in self.tokenized for tok in set(doc))

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1)

    def score(self, query, doc_idx):
        doc = self.tokenized[doc_idx]
        dl = len(doc)
        tf_counts = Counter(doc)
        s = 0.0
        for term in query.lower().split():
            tf = tf_counts.get(term, 0)
            idf = self.idf(term)
            s += idf * (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
        return s

    def retrieve(self, query, top_k=10):
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: -x[1])
        return [(self.corpus[i], s) for i, s in scores[:top_k]]

# --- Hybrid fusion ---
def min_max_normalize(scores):
    lo, hi = min(scores), max(scores)
    r = hi - lo or 1e-8
    return [(s - lo) / r for s in scores]

def hybrid_retrieve(query, bm25, doc_vectors, corpus, vocab, alpha=0.5, top_k=5):
    # BM25 scores
    bm25_raw = [bm25.score(query, i) for i in range(len(corpus))]
    bm25_norm = min_max_normalize(bm25_raw)

    # Dense scores
    q_vec = embed_bow(query, vocab)
    dense_raw = list(doc_vectors @ q_vec)
    dense_norm = min_max_normalize(dense_raw)

    # Hybrid
    hybrid = [alpha * d + (1 - alpha) * b for d, b in zip(dense_norm, bm25_norm)]
    top_idx = np.argsort(hybrid)[::-1][:top_k]
    return [(corpus[i], hybrid[i]) for i in top_idx]

bm25 = BM25(CORPUS)
results = hybrid_retrieve(
    "international refund shipping 30 days",
    bm25, doc_vectors, CORPUS, vocab, alpha=0.5
)
# [Source 1] International returns must be initiated within 30 days...  hybrid=0.87
# [Source 2] Shipping costs for international returns...                  hybrid=0.74
# [Source 3] Refunds are processed to the original payment method...     hybrid=0.61`}
      </CodeBlock>

      <H3>4d. Query expansion — HyDE and multi-query rewriting</H3>

      <Prose>
        Short user queries are poor retrieval inputs — they are sparse and may not share vocabulary with the document chunks. A user asking "why did my card get declined?" is using conversational language that may not match any phrase in the relevant payment policy document ("transaction declined due to insufficient authorization"). Two complementary strategies address this vocabulary gap.
      </Prose>

      <Prose>
        HyDE (Hypothetical Document Embeddings; Gao et al., arXiv:2212.10496) exploits a key asymmetry: LLMs are very good at generating fluent, document-like text, even when they do not know the correct answer. Given the query "why did my card get declined?", a language model generates a plausible paragraph in the style of a support document: "Cards may be declined for several reasons including insufficient funds, exceeded credit limit, incorrect CVV, expired card, or geographic restrictions on the merchant...". This hypothetical answer, even if factually wrong in specifics, is written in the same vocabulary and syntactic style as the real documents. Its embedding lands in the same neighborhood as real policy chunks, dramatically improving retrieval precision even with zero retriever fine-tuning. The tradeoff is an extra LLM call before retrieval, adding 200–500ms.
      </Prose>

      <Prose>
        Multi-query rewriting takes a different angle: instead of generating a single better query, generate three to five paraphrases of the original query and retrieve for each. The union of retrieved sets covers vocabulary variations that any single phrasing would miss. The implementation unions the results and deduplicates by document identity, keeping each chunk's best score across all queries. This technique is particularly effective for follow-up questions in conversations, where the original query is ambiguous without context — the LLM can generate paraphrases that fold in the conversational context before embedding.
      </Prose>

      <CodeBlock language="python">
{`# --- HyDE: embed a hypothetical answer, not the raw query ---
def hyde_retrieve(query, llm_stub, doc_vectors, corpus, vocab, top_k=5):
    """
    1. Ask LLM to write a plausible (possibly wrong) answer.
    2. Embed the hypothetical answer — it lives in document-space, not query-space.
    3. Retrieve chunks similar to the hypothesis.
    """
    hypothesis = llm_stub(
        f"Write a concise, factual-sounding answer to: {query}\\n"
        "The answer may be incorrect — focus on style and vocabulary matching real docs."
    )
    hyp_vec = embed_bow(hypothesis, vocab)
    scores = doc_vectors @ hyp_vec
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(corpus[i], float(scores[i])) for i in top_idx]

# --- Multi-query: paraphrase + union ---
def multi_query_retrieve(query, llm_stub, doc_vectors, corpus, vocab, n_queries=3, top_k_each=3):
    """
    Generate n paraphrases, retrieve top_k_each for each,
    deduplicate, return the union sorted by max score.
    """
    paraphrases_text = llm_stub(
        f"Write {n_queries} different phrasings of this question, one per line:\\n{query}"
    )
    queries = [query] + paraphrases_text.strip().split("\\n")[:n_queries]

    seen, best_score = {}, {}
    for q in queries:
        q_vec = embed_bow(q, vocab)
        scores = doc_vectors @ q_vec
        for idx in np.argsort(scores)[::-1][:top_k_each]:
            doc = corpus[idx]
            s = float(scores[idx])
            if doc not in best_score or best_score[doc] < s:
                best_score[doc] = s

    ranked = sorted(best_score.items(), key=lambda x: -x[1])
    return ranked[:top_k_each * 2]   # deduplicated union`}
      </CodeBlock>

      <H3>4e. Reranking — cross-encoder over bi-encoder top-N</H3>

      <Prose>
        The single highest-ROI improvement to a working RAG pipeline. Retrieve the top 20–50 chunks cheaply with the bi-encoder, then rerank them with a cross-encoder that attends jointly over (query, chunk) pairs. The cross-encoder is much slower per pair but far more accurate, because it can catch relevance patterns that the independent query and document vectors missed. Keep only the top-5 from the reranked list.
      </Prose>

      <Prose>
        The intuition for why reranking helps so consistently: bi-encoder retrieval is a geometric approximation — it assumes relevance is captured by vector proximity. Cross-encoder reranking is a direct relevance judgment — it reads both the query and the candidate chunk together and produces a calibrated relevance score. The bi-encoder's job is to ensure the correct chunk is somewhere in the top-50; the cross-encoder's job is to move it from wherever it is in the top-50 to the top-5. This division of labor is not a stopgap — it is the designed architecture for production retrieval, and Cohere, Jina, and BGE all ship production cross-encoders specifically for this role. The typical measurement: moving from top-5 bi-encoder results to top-20 bi-encoder + cross-encoder rerank-to-5 improves recall@5 by 8–25 percentage points on standard RAG benchmarks, for roughly 100–200ms of added latency.
      </Prose>

      <CodeBlock language="python">
{`# Cross-encoder reranking (production: use Cohere Rerank, BGE-Reranker, or Jina Reranker)
# Here: a score based on shared n-gram overlap, approximating the interaction signal

def ngrams(text, n=2):
    tokens = text.lower().split()
    return set(zip(*[tokens[i:] for i in range(n)]))

def cross_encoder_score(query, document):
    """
    Simplified cross-encoder proxy: shared bigram overlap.
    Real cross-encoder: transformer(concat(query, document)) → scalar.
    """
    q_bi = ngrams(query, 2) | ngrams(query, 1)
    d_bi = ngrams(document, 2) | ngrams(document, 1)
    if not q_bi:
        return 0.0
    precision = len(q_bi & d_bi) / len(q_bi)
    recall    = len(q_bi & d_bi) / len(d_bi) if d_bi else 0.0
    return 2 * precision * recall / (precision + recall + 1e-8)   # F1

def rerank(query, candidates, top_k=5):
    """
    candidates: list of (text, bi_encoder_score)
    Returns top_k re-sorted by cross-encoder score.
    """
    scored = [(text, cross_encoder_score(query, text)) for text, _ in candidates]
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]

# End-to-end: bi-encoder top-20 → cross-encoder rerank → top-5
query = "refund policy for international damaged items"
candidates = retrieve(query, top_k=20)    # bi-encoder, cheap
final      = rerank(query, candidates, top_k=5)   # cross-encoder, accurate
prompt     = build_prompt(query, [(text, s) for text, s in final])

# Before rerank — recall@5: 0.60 (correct chunk at position 4)
# After rerank  — recall@5: 0.80 (correct chunk moves to position 1)`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION PATTERNS
          ====================================================================== */}
      <H2>5. Production patterns</H2>

      <Prose>
        At prototype scale — a few thousand documents, a handful of users — the from-scratch implementations above are sufficient. Production scale introduces three new constraints: latency (the retrieval must complete in under 200ms for interactive use), cost (embedding and querying at millions of documents per day has a real budget), and correctness (stale indexes and missed chunks have user-visible consequences). The ecosystem of libraries and managed services exists to address these three constraints. Choosing between them is less about technical superiority and more about operational fit: what your team already knows how to run, where your data lives, what your latency SLAs are, and how much you can afford to spend per query.
      </Prose>

      <Prose>
        One architectural decision that most tutorials skip: document ingestion is a pipeline, not a one-shot operation. Real production systems have a document lifecycle — documents are created, updated, and occasionally deleted. The ingestion pipeline must handle all three cases correctly. Creation: chunk, embed, upsert into the vector store with metadata (document ID, source URL, last-modified timestamp, access-control tags). Update: retrieve all chunk IDs with the updated document ID, delete them, re-chunk and re-embed the new version, insert. Deletion: retrieve all chunk IDs for the deleted document ID and hard-delete them from the index. Teams that skip the update and deletion cases ship indexes that drift from their source corpus over time, producing stale-index failure mode (10) from Section 9 at increasing frequency.
      </Prose>

      <H3>5.1 Orchestration frameworks</H3>

      <Prose>
        LangChain (Harrison Chase, 2022) provides a composable chain abstraction: a retriever, a prompt template, and an LLM slot together into a <Code>RetrievalQA</Code> chain that handles the boilerplate. LlamaIndex (Jerry Liu, 2022) goes deeper on the indexing side — it provides node parsers for structured documents (PDFs, Notion, Confluence), hierarchical indexing (summary nodes pointing to chunk nodes), and multi-document retrieval with source tracking. Haystack (deepset, 2020) was the earliest production-grade framework and emphasizes pipeline composability, with built-in support for hybrid retrieval, reranking, and multi-hop question answering. For pure production reliability, all three frameworks are reasonable; the choice is usually made by which one integrates more naturally with the rest of the stack.
      </Prose>

      <CodeBlock language="python">
{`# Production-grade pipeline sketch — LangChain style
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1. Embedding model — BGE-M3 covers 100+ languages, 8192 token context
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# 2. Vector store — Qdrant supports hybrid search natively
vectorstore = Qdrant.from_documents(
    documents=chunked_docs,
    embedding=embedder,
    url="http://qdrant:6333",
    collection_name="policy_docs",
    force_recreate=False,         # incremental index update
)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# 3. Hybrid: ensemble BM25 + dense
bm25_retriever = BM25Retriever.from_documents(chunked_docs, k=20)
hybrid = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6],          # tune on validation set
)

# 4. Reranker — Cohere's cross-encoder, or use CohereRerank wrapper
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

reranked = ContextualCompressionRetriever(
    base_compressor=CohereRerank(top_n=5),
    base_retriever=hybrid,
)

# 5. Chain
chain = RetrievalQA.from_chain_type(
    llm=OpenAI(model="gpt-4o", temperature=0),
    chain_type="stuff",          # "map_reduce" if context > 8k tokens
    retriever=reranked,
    return_source_documents=True,
)

result = chain.invoke("What is the return window for international orders?")
# result["answer"]            → grounded answer
# result["source_documents"]  → list of retrieved chunks with metadata`}
      </CodeBlock>

      <H3>5.2 Vector database landscape</H3>

      <Prose>
        The vector database is the index store. FAISS (Facebook AI Similarity Search) is an in-process library, not a database — excellent for research and prototypes, no persistence or serving layer. Qdrant is a purpose-built vector database with native hybrid search (dense + sparse BM42), payload filtering, and a Rust core that sustains low p99 latency under concurrent load; it is the most common choice for teams that need hybrid retrieval without running two separate systems. Weaviate adds a GraphQL API and schema enforcement, making it natural for structured knowledge bases where documents have typed metadata and you want to filter on fields (e.g., "only retrieve chunks from documents authored after 2024-01-01 marked as 'approved'"). Pinecone is fully managed — no infrastructure to operate — at the cost of vendor lock-in and per-query pricing that scales uncomfortably past a few hundred thousand queries per day. pgvector adds vector search to PostgreSQL with an <Code>ivfflat</Code> or <Code>hnsw</Code> index; if your data is already in Postgres and your query volume is moderate (under 1,000 QPS), it is often the lowest-overhead path to a working retrieval layer with the added benefit of joining vector search results against relational metadata. Milvus is the open-source alternative to Pinecone: distributed, scalable to billions of vectors, and production-hardened at ByteDance and Alibaba's scale.
      </Prose>

      <Prose>
        The HNSW index structure deserves a closer look because it underpins most production vector stores. HNSW (Hierarchical Navigable Small World; Malkov and Yashunin, 2018) builds a layered graph where each node is a chunk embedding and each edge connects nodes that are close in the embedding space. The graph has multiple layers: the top layer is sparse and allows long-range navigation; lower layers are denser and allow local refinement. A query traverses from the top (coarse) to the bottom (fine), following the greedy path toward the query vector at each layer. The result is approximate — the true nearest neighbor is not guaranteed — but typical configurations (ef_construction=200, M=16) recover 97–99% of the true top-k at a search latency of 5–30ms over tens of millions of vectors. The HNSW graph must be held in RAM for fast traversal, which is the primary memory cost of a vector index: a 768-dim float32 index over 10 million vectors requires roughly 30 GB of RAM for the vectors plus 20–40 GB for the graph edges.
      </Prose>

      <H3>5.3 Embedding model selection</H3>

      <Prose>
        The embedding model is the component that most strongly determines retrieval quality, and it is the component most often selected by cargo-cult rather than by measurement. The MTEB (Massive Text Embedding Benchmark) leaderboard, maintained at Hugging Face, provides standardized recall and nDCG scores across 56 retrieval tasks in multiple languages — it is the first place to look when selecting a model. As of early 2026, the top tier includes BGE-M3 (BAAI; multilingual, 8192-token context, three retrieval modes — dense, sparse, and ColBERT-style), E5-Mistral-7B (Microsoft; 7B parameter, highest benchmark scores, expensive to host), and text-embedding-3-large (OpenAI; strong general-purpose, easy API). For most production use cases, BGE-M3 or text-embedding-3-small provide the best quality-cost tradeoff. The key practical consideration beyond raw benchmark scores: the model must have a context window long enough to embed your chunks (some older models top out at 512 tokens, which is often insufficient for 400-token chunks after subword tokenization).
      </Prose>

      {/* ======================================================================
          6. VISUAL DIAGNOSTICS
          ====================================================================== */}
      <H2>6. Visualizing the pipeline</H2>

      <Prose>
        Two diagnostic visualizations that belong in every RAG debugging workflow: the query-document similarity matrix and the recall@k curve.
      </Prose>

      <H3>6.1 Query-document similarity matrix</H3>

      <Prose>
        Each cell shows the cosine similarity between a query (row) and a corpus chunk (column). High-similarity cells (dark gold) show which chunks are retrieved for each query. Rows where no cell is very dark indicate queries that retrieval fails on — the ground-truth chunk has low similarity to the query vector, usually because of vocabulary mismatch or a chunking problem.
      </Prose>

      <Heatmap
        label="cosine similarity — 4 queries × 8 corpus chunks"
        rowLabels={["Q1: intl return window", "Q2: domestic refund", "Q3: damaged item policy", "Q4: final sale items"]}
        colLabels={["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]}
        matrix={[
          [0.87, 0.61, 0.44, 0.12, 0.08, 0.72, 0.19, 0.05],
          [0.31, 0.82, 0.75, 0.09, 0.15, 0.28, 0.12, 0.04],
          [0.14, 0.19, 0.08, 0.88, 0.41, 0.10, 0.32, 0.07],
          [0.05, 0.11, 0.07, 0.23, 0.91, 0.08, 0.14, 0.03],
        ]}
        colorScale="gold"
        cellSize={42}
      />

      <Prose>
        Reading the matrix: Q1 (international return window) retrieves C1 and C6 correctly — those are the international return policy and the shipping cost chunks. Q2 (domestic refund) retrieves C2 and C3. Q3 (damaged items) retrieves C4 strongly but also C5 (false positive). Q4 (final sale) retrieves C5 cleanly. A matrix like this, computed on a validation set of 50–100 labeled (question, relevant-chunk) pairs, immediately shows you which queries retrieval is failing on without any end-to-end eval.
      </Prose>

      <H3>6.2 Recall@k curve</H3>

      <Prose>
        Recall@k measures how often the ground-truth chunk appears in the top-k results. The curve should flatten quickly — if recall@10 is barely above recall@5, adding more chunks per query is mostly noise. The gap between dense-only and hybrid shows the value of lexical retrieval. The gap between hybrid and hybrid+rerank shows the reranker's contribution.
      </Prose>

      <Plot
        label="recall@k — dense vs hybrid vs hybrid+rerank"
        xLabel="k (chunks retrieved)"
        yLabel="recall@k"
        series={[
          {
            name: "dense only",
            color: "#60a5fa",
            points: [[1, 0.42], [2, 0.55], [3, 0.63], [5, 0.71], [10, 0.80], [20, 0.87]],
          },
          {
            name: "hybrid (α=0.5)",
            color: colors.gold,
            points: [[1, 0.51], [2, 0.64], [3, 0.72], [5, 0.81], [10, 0.89], [20, 0.94]],
          },
          {
            name: "hybrid + rerank top-5",
            color: colors.green,
            points: [[1, 0.61], [2, 0.74], [3, 0.82], [5, 0.88], [10, 0.89], [20, 0.94]],
          },
        ]}
      />

      <Prose>
        The key observation: at k=5, hybrid retrieval achieves recall@5=0.81 versus 0.71 for dense-only — a 10-point absolute improvement from adding BM25. Adding reranking moves recall@5 to 0.88. That 7-point further improvement from reranking is the result of cross-encoder interaction signals that the bi-encoder missed. In a system where only k=5 chunks fit in the prompt, these improvements directly translate to answer quality.
      </Prose>

      <H3>6.3 Prompt construction — what the model sees</H3>

      <TokenStream
        label="final prompt structure — [system] + [chunks] + [query]"
        tokens={[
          { label: "SYSTEM: use only the sources below. cite by number.", color: colors.textMuted },
          { label: "[Source 1] Intl returns: 30 days from delivery.", color: colors.gold },
          { label: "[Source 2] Shipping costs: customer responsibility.", color: "#60a5fa" },
          { label: "[Source 3] Refunds processed in 5–7 business days.", color: "#c084fc" },
          { label: "QUESTION: Can I return an intl order after 45 days?", color: colors.green },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. When to use what</H2>

      <Prose>
        RAG is not always the answer. The right architecture depends on the knowledge volatility, the reasoning complexity, the corpus size, and the latency budget.
      </Prose>

      <Prose>
        <strong>Basic RAG suffices</strong> when the knowledge base is a stable, well-structured document corpus — FAQ pages, product manuals, HR policies, help center articles. Questions are single-hop (one retrieved document contains the full answer), the document count is in the thousands to low millions, and freshness requirements are measured in hours to days. This describes the majority of enterprise chatbot deployments.
      </Prose>

      <Prose>
        <strong>Hybrid retrieval is worth adding</strong> when the corpus contains technical documentation with specific identifiers — model numbers, API names, code symbols, regulatory citation numbers — that dense retrieval consistently misses. The operational cost is adding BM25 alongside the vector index, which is minimal, and the quality gain is reliable. Any corpus where users can be expected to copy-paste exact strings from documents is a hybrid retrieval use case.
      </Prose>

      <Prose>
        <strong>Agentic or GraphRAG is needed</strong> when answering a question requires synthesizing information across multiple documents that are not retrieved together by a single query — what the GraphRAG literature calls multi-hop or compositional queries. "What are all the policies that changed after the 2023 compliance update, and how do they affect EU customers?" requires finding the update, finding the affected policies, and cross-referencing the EU customer rules. A single-shot RAG pipeline retrieves five chunks and hopes they cover all three legs; agentic RAG lets the model decide to retrieve again mid-answer. GraphRAG (Microsoft, 2024) pre-builds a knowledge graph over the corpus and retrieves graph subgraphs rather than flat chunks, enabling multi-hop inference without repeated round-trips.
      </Prose>

      <Prose>
        <strong>Skip RAG entirely</strong> when the task is pure reasoning with no knowledge dependency (math proofs, code debugging from a pasted snippet, text transformation tasks); when the domain knowledge is stable and small enough to fine-tune into the model weights; or when the latency budget is under 100ms and retrieval is too slow. RAG with a hallucinating retriever is worse than no RAG — if your knowledge base does not reliably contain the answers, you are injecting noise into the prompt.
      </Prose>

      <Prose>
        A useful mental test for any proposed RAG deployment: "If I could magically retrieve the perfect chunk every time, would the system produce correct answers?" If the answer is no — because the task requires multi-step inference the LLM cannot do, or because the answer genuinely does not exist in any document — then RAG will not fix the problem. It is a retrieval system, not a reasoning system and not a knowledge-creation system. The ceiling on RAG-based accuracy is min(retrieval_recall, model_reasoning_quality): whichever is lower dominates the failure rate. Most teams diagnose generation quality when they should diagnose retrieval quality, and diagnose retrieval quality when they should diagnose chunking quality. Measuring each layer independently, with the diagnostic tools from Section 6, is the only reliable way to find the actual bottleneck.
      </Prose>

      <Prose>
        One architectural decision that is often deferred too long: should retrieval happen at the turn level or at the session level? Turn-level retrieval embeds the current user message and retrieves against it — simple, but it loses conversational context. A follow-up question like "What about for EU customers?" has no useful embedding without the conversation history that established what "it" refers to. Session-level retrieval folds the conversation history into the query — either by appending the last N turns before embedding, or by using an LLM to rewrite the follow-up into a standalone query ("What is the return policy for EU customers?") before retrieval. The rewrite-then-retrieve pattern is the most robust and adds only a single fast LLM call to the pipeline.
      </Prose>

      {/* ======================================================================
          8. SCALING
          ====================================================================== */}
      <H2>8. Scaling properties</H2>

      <Prose>
        The scaling picture for RAG is fundamentally different from scaling for training — the costs are split across indexing time, query time, and storage, and they scale at different rates.
      </Prose>

      <Prose>
        <strong>Index size scales linearly with corpus size.</strong> A 768-dimensional float32 vector is 3 KB. One million chunks takes roughly 3 GB of raw vector storage, plus the HNSW graph structure (typically 1–2× overhead), for a total of around 6–9 GB at the million-chunk scale. Ten million chunks is 60–90 GB — fits in the RAM of a single large server node. At this scale, a single Qdrant or Weaviate instance handles the full index.
      </Prose>

      <Prose>
        <strong>Retrieval latency scales logarithmically with index size</strong> under HNSW. An HNSW graph lookup takes O(log N) graph traversals. Empirically, moving from 100K to 10M documents raises p99 retrieval latency from roughly 5ms to 20ms — a 100x index size increase corresponds to roughly 4x latency increase. This is what makes dense retrieval over billion-document indexes possible. The tradeoff is construction time and memory: HNSW graphs must be held in RAM for fast access, and at billion-document scale this requires a distributed index.
      </Prose>

      <Prose>
        <strong>Embedding cost dominates at ingestion.</strong> Embedding one million chunks at 500 tokens each through a mid-tier API (OpenAI text-embedding-3-small at \$0.02/million tokens) costs roughly \$10 and takes a few hours. At 100 million chunks this becomes \$1,000 — still manageable. The real constraint is that re-embedding the full corpus on every model change makes embedding model upgrades expensive. Practical solution: maintain chunk-level embedding metadata and only re-embed chunks whose document has changed since last indexing. This makes incremental index updates fast even for large corpora.
      </Prose>

      <Prose>
        <strong>Generation cost dominates at query time.</strong> A typical RAG prompt is 1,500–3,000 tokens (system + 5 chunks at 300 tokens each + query). At GPT-4o pricing, this is roughly \$0.004–\$0.008 per query. At 100,000 queries per day, generation costs run \$400–\$800/day — roughly 10–20× the daily embedding and retrieval costs combined. The lever for generation cost is reducing the number and length of retrieved chunks without sacrificing recall, which is exactly what a good reranker accomplishes: it lets you retrieve 20 cheaply and inject only the best 3 expensively.
      </Prose>

      <Prose>
        <strong>The embedding model upgrade problem.</strong> Once a corpus is indexed with a specific embedding model, switching to a better model requires re-embedding the entire corpus — because the new model's vector space is geometrically incompatible with the old one. At 100 million chunks, this is a multi-hour, multi-hundred-dollar operation. Two patterns manage this cost. First, incremental indexing: maintain a "dirty" flag per document, re-embed only changed documents on each indexing cycle. For corpora that change partially (news articles, policy updates, product catalog additions), this dramatically reduces the per-cycle cost. Second, index versioning: maintain old and new indexes in parallel during a model migration, gradually routing queries to the new index as re-embedding completes chunk by chunk. The old index serves as a fallback for un-migrated chunks.
      </Prose>

      <Prose>
        <strong>Latency budget decomposition.</strong> A typical interactive RAG pipeline has a total latency budget of 300–600ms before the LLM generation begins (the generation itself takes 1–5s for a typical response). That 300–600ms breaks down approximately as: query embedding 20–80ms (depends on model size and whether GPU is warm), ANN index search 5–30ms, optional BM25 search 5–15ms, optional reranking 50–200ms (depends on cross-encoder size), prompt construction 5ms. The reranker is the largest variable — Cohere Rerank and BGE-Reranker-v2 run as API calls and add 100–200ms, while running a smaller reranker locally (BGE-Reranker-base) adds 30–80ms. If the total pre-generation latency budget is tight, the reranker is the first thing to cut; if quality is the priority, it is the last.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <Prose>
        RAG fails in at least ten distinct ways, each with a distinct signature and a distinct fix. Cataloging them is not pessimism — it is the prerequisite for debugging, because a pipeline that fails silently looks like one that works.
      </Prose>

      <Prose>
        <strong>1. Chunk size wrong — too small.</strong> Chunks of 50–100 tokens often fragment natural answer units. A chunk that says "30 days" retrieves well on "return window" queries but cannot answer "what is the return window?" on its own — the surrounding context ("International returns must be initiated within [30 days]") was in the previous chunk. Fix: increase chunk size or add sentence-level overlap to preserve local context.
      </Prose>

      <Prose>
        <strong>2. Chunk size wrong — too large.</strong> Chunks of 1,500+ tokens dilute the embedding signal. If a 1,500-token chunk covers international returns, domestic returns, exchange policy, and refund timelines, its embedding is an average over all four topics and retrieves well on none of them. Fix: reduce chunk size; use semantic chunking to split at topic boundaries.
      </Prose>

      <Prose>
        <strong>3. Off-by-one retrieval.</strong> The ground-truth chunk is at rank 6 and the pipeline only retrieves top-5. Common when the query phrasing and the document phrasing are paraphrases that the embedding model maps to slightly different regions of vector space. Fix: retrieve more (top-10 or top-20) before reranking down to top-5.
      </Prose>

      <Prose>
        <strong>4. Embedding model mismatch.</strong> The document corpus is embedded with one model (e.g. text-embedding-3-small) and queries are embedded with another (e.g. an older ada-002 model after an API migration). Because embedding spaces are not aligned across models, retrieval quality collapses. Fix: always use the same model for indexing and querying; version-pin the embedding model in the pipeline configuration.
      </Prose>

      <Prose>
        <strong>5. Stale index.</strong> A document was updated but the vector store was not re-indexed. The model confidently cites the old version. Fix: implement a document change detector (hash-based or last-modified timestamp) that triggers incremental re-indexing on document updates. Track chunk provenance metadata so stale chunks can be identified without re-scanning the entire index.
      </Prose>

      <Prose>
        <strong>6. Multi-hop failure.</strong> The answer requires synthesizing information across two documents that are not near each other in the embedding space, so they are never retrieved together in a single top-k pass. Fix: agentic RAG (let the model issue a second retrieval query after reading the first results) or GraphRAG (pre-build relationships between document nodes).
      </Prose>

      <Prose>
        <strong>7. Context length overflow.</strong> Retrieving top-20 chunks at 500 tokens each produces 10,000 tokens of context. With a system prompt and generation head room, this exceeds an 8,192-token context window, causing silent truncation of the most recently added chunks. Fix: track the running token count during prompt construction and truncate at the chunk level, not at the character level. Use a reranker to ensure the best chunks appear first so that truncation removes the least useful material.
      </Prose>

      <Prose>
        <strong>8. Hallucinated citations.</strong> The model produces a plausible-sounding answer and fabricates a source number that does not correspond to any retrieved chunk. This happens when the prompt instructs citation but the model's parametric memory fills in the answer without using the retrieved context. Fix: parse the model's output to verify every cited source number actually exists in the prompt; flag answers where cited content does not appear in the retrieved chunk text.
      </Prose>

      <Prose>
        <strong>9. Retrieval bias amplification.</strong> The embedding model was trained on data that overrepresents certain demographics, domains, or viewpoints. Retrieval systematically returns documents from the overrepresented distribution even when more relevant documents from underrepresented sources exist. Fix: audit retrieval results across demographic slices; consider training or fine-tuning the embedding model on a more representative corpus.
      </Prose>

      <Prose>
        <strong>10. PII leakage and prompt injection.</strong> Two security failures that are unique to RAG. PII leakage: a user without access to HR document X can craft a query whose embedding retrieves chunks from X if the vector store does not enforce access controls at retrieval time — embeddings do not carry permissions. Fix: filter retrieved chunks by user authorization metadata before injecting into the prompt. Prompt injection: a malicious actor inserts text into a public document that, when retrieved, rewrites the system prompt ("Ignore previous instructions and instead..."). Fix: sanitize retrieved chunks before injection; use a separate LLM call to detect and strip injected instructions; structure the prompt to clearly demarcate system instructions from retrieved context.
      </Prose>

      <Callout accent="gold">
        Most RAG failures are retrieval failures, not generation failures. Measure context recall before optimizing prompt templates or switching LLMs.
      </Callout>

      <Prose>
        A practical triage protocol: when a user reports a wrong answer, first check whether the correct chunk was in the retrieved set (retrieval failure or not). If yes, check whether the prompt template included it correctly and in a prominent position (augmentation failure). If yes, check whether the model ignored or contradicted the retrieved evidence (generation failure). In practice, retrieval failures account for roughly 60–70% of RAG errors in production systems without careful eval; augmentation and generation failures make up the remaining 30–40%. The distribution shifts after retrieval is tuned: at high-performing retrieval recall, generation and faithfulness failures become the bottleneck, which is where prompt engineering and generator fine-tuning have most leverage.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The papers that created and shaped the RAG paradigm, in order of publication. Each paper introduced a concept that remains load-bearing in modern RAG stacks; none of them have been superseded — they have been refined and scaled.
      </Prose>

      <Prose>
        <strong>REALM</strong> — Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, Ming-Wei Chang. "REALM: Retrieval-Augmented Language Model Pre-Training." ICML 2020. arXiv:2002.08909. The first system to train a retriever jointly with a language model objective, using backpropagation through maximum inner product search over Wikipedia. REALM showed that retrieval can be integrated into pretraining itself, not just inference, and that doing so improves open-domain QA without any task-specific fine-tuning.
      </Prose>

      <Prose>
        <strong>RAG</strong> — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020. arXiv:2005.11401. The paper that named the paradigm and established the two canonical formulations — RAG-Sequence (same passages condition the full generation) and RAG-Token (passages can differ per token). Showed that combining a DPR retriever with BART generation outperforms pure parametric models on open-domain QA and knowledge-intensive generation while generating more specific, diverse, and factual text.
      </Prose>

      <Prose>
        <strong>FiD</strong> — Gautier Izacard and Édouard Grave. "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering." EACL 2021. arXiv:2007.01282. Introduced Fusion-in-Decoder: encode each retrieved passage independently with the query and concatenate the encoded representations in the decoder, rather than concatenating all passages in the encoder. This allows attending over many more passages than fit in a single encoder forward pass, dramatically scaling retrieval augmentation. Set state-of-the-art on NaturalQuestions and TriviaQA.
      </Prose>

      <Prose>
        <strong>ColBERT</strong> — Omar Khattab and Matei Zaharia. "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." SIGIR 2020. arXiv:2004.12832. Introduced the late interaction paradigm: encode query and document independently with BERT, then compute a fine-grained MaxSim score over all query-document token pairs rather than a single vector. Documents can still be pre-encoded offline (unlike cross-encoders), but the interaction is far richer than bi-encoder dot products. ColBERT closes most of the quality gap between bi-encoders and cross-encoders while remaining indexable.
      </Prose>

      <Prose>
        <strong>SBERT</strong> — Nils Reimers and Iryna Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP 2019. arXiv:1908.10084. The foundational paper for the bi-encoder retrieval paradigm. SBERT fine-tunes BERT with contrastive (siamese/triplet) losses to produce sentence embeddings that are meaningful under cosine similarity, reducing semantic similarity search from O(n²) BERT inference to O(n) indexing plus O(1) lookup. All modern bi-encoders (BGE, E5, GTE, Jina, Nomic, text-embedding-3) are trained with variants of the SBERT objective.
      </Prose>

      <Prose>
        <strong>HyDE</strong> — Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan. "Precise Zero-Shot Dense Retrieval without Relevance Labels." December 2022. arXiv:2212.10496. Proposed Hypothetical Document Embeddings: generate a plausible (possibly incorrect) answer to the query with an LLM, embed the hypothesis rather than the raw query, and retrieve documents similar to the hypothesis. Because the hypothesis is written in document-like language, it lives in the same semantic neighborhood as real answer documents, dramatically improving zero-shot retrieval without any retriever fine-tuning.
      </Prose>

      <Prose>
        What the primary sources above share is a consistent underlying approach: treat retrieval as a first-class ML problem, not a preprocessing step. REALM trained the retriever jointly with the language model objective. Lewis et al. trained DPR end-to-end with BART. Izacard and Grave showed that the number of retrieved passages mattered and designed an architecture to support more of them. Khattab and Zaharia designed a representation that was richer than a single vector but still indexable. Reimers and Gurevych established the training recipe for all bi-encoders. Gao et al. showed that LLM generation could be used to improve the retrieval query itself. Each paper solved a bottleneck that the previous papers created, and the resulting stack is now the baseline every new RAG paper is compared against.
      </Prose>

      <Prose>
        The field has continued to evolve rapidly since 2022. Reranker quality has improved substantially with models like Cohere Rerank 3 and BGE-Reranker-v2. Embedding model quality has improved through larger training sets, better contrastive loss formulations, and instruction-tuned embeddings (models that accept natural-language instructions specifying the embedding task, such as "represent this sentence for retrieval" vs. "represent this sentence for classification"). Agentic retrieval — where the model autonomously decides when and what to retrieve mid-generation — has moved from research prototypes to production at several companies. GraphRAG and structured retrieval have provided partial solutions to the multi-hop reasoning problem. The vector database ecosystem has consolidated around three or four mature options with production-hardened reliability guarantees. What has not changed since the 2020 papers is the fundamental architecture: a queryable external knowledge store, retrieved at inference time and injected as context. The engineering gets better every year; the concept is stable.
      </Prose>

      {/* ======================================================================
          11. EXERCISES
          ====================================================================== */}
      <H2>11. Exercises</H2>

      <Prose>
        <strong>Exercise 1 — Chunking ablation.</strong> Take a 10,000-word policy document. Chunk it three ways: fixed-size at 200 tokens (no overlap), fixed-size at 200 tokens with 40-token overlap, and paragraph-based. Build a 20-question validation set where you know which chunk contains the answer. Measure recall@5 for each chunking strategy using cosine similarity retrieval. Write a brief analysis of which question types each strategy fails on and why.
      </Prose>

      <Prose>
        <strong>Exercise 2 — Hybrid retrieval tuning.</strong> Implement BM25 and dense retrieval over a small corpus (use the 8-document corpus in Section 4a extended to 50 documents). For <Code>alpha</Code> in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], measure recall@5 on 20 queries. Plot recall@5 vs. alpha. Identify which query types benefit most from alpha closer to 0 (BM25-heavy) vs. alpha closer to 1 (dense-heavy). What does this tell you about your corpus?
      </Prose>

      <Prose>
        <strong>Exercise 3 — Reranker evaluation.</strong> Using the hybrid retriever from Exercise 2, retrieve top-20 candidates for each query. Implement a reranking step (you can use the bigram cross-encoder from Section 4e or a real cross-encoder via sentence-transformers). Measure recall@5 before and after reranking. At what k (for the initial candidate set) does the reranker stop improving recall@5? What does this tell you about the capacity of the reranker vs. the retriever?
      </Prose>

      <Prose>
        <strong>Exercise 4 — Failure mode diagnosis.</strong> Build a minimal RAG system over a Wikipedia article of your choice. Deliberately introduce three of the ten failure modes from Section 9: (a) reduce chunk size to 30 tokens to trigger fragmentation, (b) embed documents with one sentence-transformers model and queries with a different model to trigger embedding mismatch, and (c) add a document to the corpus that has not been re-indexed to trigger stale index. For each failure mode, write a diagnostic query that exposes the failure and measure the drop in answer quality. Then fix each failure mode and confirm the fix.
      </Prose>

      <Prose>
        <strong>Exercise 5 — RAGAS-style evaluation.</strong> Build a 50-question evaluation set for a domain of your choice: (question, ideal-answer, relevant-chunk-id) triples. Implement four metrics manually: (1) context precision — what fraction of retrieved chunks are relevant; (2) context recall — does the retrieved set contain the relevant chunk; (3) faithfulness — score (0/1) whether the generated answer is entailed by the retrieved chunks; (4) answer relevance — score (0/1) whether the answer addresses the question. Run your end-to-end RAG pipeline over all 50 questions and report a dashboard with all four metrics. Identify which metric is your current bottleneck and propose one concrete improvement to address it.
      </Prose>

    </div>
  ),
};

export default rag;
