import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";

const embeddingModelsVectorDB = {
  title: "Embedding Models & Vector Databases",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        The RAG topic treats embedding and retrieval as black boxes: chunk the document, embed it, store it, retrieve the nearest neighbors at query time, and hand them to the generator. That framing is useful for building intuition but it papers over two layers that have genuine depth and real production footguns. The embedding model determines what "similar" means — which text pairs end up close in vector space, and which do not. The vector database determines how fast you can answer nearest-neighbor questions when the index has ten million, or a hundred million, or a billion entries. Both have moved fast in the last two years, and the choices available today are meaningfully different from the choices available in 2023.
      </Prose>

      <H2>What an embedding model is</H2>

      <Prose>
        An embedding model is a function that maps a piece of text to a fixed-dimensional vector such that semantically similar texts produce vectors with high cosine similarity. The output is not a classification or a next-token prediction — it is a point in a high-dimensional space whose geometry encodes meaning. Modern embedding models are bi-encoders: BERT-style transformer encoders fine-tuned with a contrastive objective. The query and the document are each encoded independently (hence bi-encoder, as opposed to a cross-encoder, which concatenates them). Independent encoding is what makes retrieval tractable: document vectors can be computed offline and stored; only the query vector needs to be computed at request time.
      </Prose>

      <Prose>
        The training signal that shapes the geometry is contrastive. The model learns to pull queries close to their matching documents and push them away from random ones. The specific objective used across almost all modern embedding models is InfoNCE, the same loss used in CLIP and in contrastive self-supervised learning:
      </Prose>

      <MathBlock>{"\\mathcal{L} = -\\log \\frac{\\exp(\\text{sim}(q, d^+) / \\tau)}{\\sum_{d \\in \\text{batch}} \\exp(\\text{sim}(q, d) / \\tau)}"}</MathBlock>

      <Prose>
        The numerator rewards placing the query <Code>q</Code> close to its matching document <Code>d+</Code>. The denominator sums over all documents in the batch, penalizing the model whenever any other document is nearly as close. The temperature <Code>τ</Code> controls sharpness: smaller values concentrate the gradient signal on the hardest negatives in the batch, which is why training with large batch sizes and hard negative mining tends to produce substantially better models than training with small batches and random negatives. The geometry that results encodes not just rough topic similarity but fine-grained semantic relationship — at least within the domain the model was trained on.
      </Prose>

      <H2>Open vs closed models — where the field stands</H2>

      <Prose>
        In 2023, OpenAI's <Code>text-embedding-ada-002</Code> was the default choice for most production retrieval systems, and it was a reasonable one: it was good, it was available via API, and the open-weight alternatives were meaningfully behind it on the MTEB benchmark (Massive Text Embedding Benchmark, the standard eval suite covering retrieval, clustering, classification, and reranking across dozens of datasets). That gap has closed.
      </Prose>

      <Prose>
        By 2025, a cluster of open-weight models — BGE (Beijing Academy of AI), E5 (Microsoft), GTE (Alibaba DAMO), Jina, Nomic, Mistral's codestral-embed — match or exceed OpenAI's embeddings on MTEB overall, and often exceed them on specific retrieval tasks. A model in the 500M–1B parameter range, self-hosted, gives production-quality retrieval at effectively zero marginal cost per query. Commercial offerings from OpenAI (<Code>text-embedding-3-small</Code> and <Code>text-embedding-3-large</Code>), Cohere (<Code>embed-v3</Code>), and Voyage still win on specific benchmarks and offer genuine convenience — single API, no infrastructure — but the quality gap that justified vendor lock-in through 2023 no longer exists for most applications. Two empirical findings are worth internalizing: embedding quality scales with parameters up to roughly the 1–7B range, then plateaus. And output dimensionality matters less than the marketing suggests — 768d is often indistinguishable from 1536d in end-to-end retrieval quality for typical workloads. If you are paying a per-token fee for embeddings, the dimension cost may not be buying what you think it is.
      </Prose>

      <H3>Matryoshka embeddings</H3>

      <Prose>
        A useful training technique introduced by Kusupati et al. in 2022 and now supported by several production models. Standard contrastive training optimizes the full embedding vector and says nothing about the structure of its dimensions. Matryoshka Representation Learning (MRL) adds a constraint: the model is trained so that the prefix of the embedding — the first 128, 256, 512, or 1024 dimensions — is itself a usable embedding, with quality that degrades gracefully as you shorten the prefix rather than collapsing entirely. OpenAI's <Code>text-embedding-3</Code> family explicitly supports this; several open-weight models are trained with it as well.
      </Prose>

      <Prose>
        The practical payoff is a single model that can serve two workloads simultaneously. Use the full-dimensional vector for the final precision pass; use a truncated prefix for the coarse first-stage retrieval that narrows a large index down to a candidate set.
      </Prose>

      <CodeBlock language="python">
{`# A matryoshka embedding truncated to lower dimensions is still a usable embedding.
full = embed("some query")            # shape: (3072,)
coarse = full[:256] / np.linalg.norm(full[:256])   # shape: (256,)
medium = full[:1024] / np.linalg.norm(full[:1024]) # shape: (1024,)

# Retrieval strategy: coarse shortlist, medium rerank, full for the final few.`}
      </CodeBlock>

      <Prose>
        The normalization step is not optional: a truncated prefix is not unit-norm even if the full vector was. Renormalize before computing cosine similarities or distances.
      </Prose>

      <H2>Vector databases — what they actually do</H2>

      <Prose>
        A vector database stores vectors alongside metadata and answers nearest-neighbor queries: given a query vector, return the <Code>k</Code> stored vectors most similar to it. The conceptually simple version — compute cosine similarity between the query and every stored vector, sort, return the top <Code>k</Code> — is called exact or brute-force search. It is perfectly accurate. It is also <Code>O(n · d)</Code> per query, where <Code>n</Code> is the number of stored vectors and <Code>d</Code> is their dimensionality. At 100K vectors and 768 dimensions that is tolerable. At 10M it is painful. At 100M it is unusable at any reasonable latency target.
      </Prose>

      <Prose>
        Approximate nearest neighbor (ANN) search trades a small, controlled amount of recall for one to three orders of magnitude of speedup. A well-tuned ANN index at 100M vectors will find the true nearest neighbors with ~97–99% recall@10 — meaning that out of the ten results it returns, nine or ten would also appear in the exact brute-force top ten — while answering in single-digit milliseconds rather than seconds. The algorithms differ in how they build the index structure and how they navigate it at query time.
      </Prose>

      <H3>HNSW — the industry default</H3>

      <Prose>
        Hierarchical Navigable Small World (HNSW) graphs are the most widely deployed ANN algorithm in production vector databases. The construction procedure builds a multi-layer graph. At the bottom layer, every vector connects to its approximate nearest neighbors, forming a dense proximity graph. Each successive higher layer contains a randomly sampled subset of the vectors, with sparser long-range connections that serve as highway edges for navigating across the graph quickly. At query time, search begins at the top layer, greedily moves toward the query vector through the sparse long-range links, descends to the next layer, greedy-searches within that denser subgraph, and repeats until reaching the bottom, where the final candidate set is evaluated exactly.
      </Prose>

      <Prose>
        The result is fast to query (typically sub-millisecond for medium-scale indexes), fast to insert into (no full index rebuild required), and achieves high recall at modest <Code>ef_search</Code> parameters. pgvector, Qdrant, Weaviate, Milvus, and Redis Stack all use HNSW variants as their primary index type. The main cost is memory: the graph structure stores adjacency lists for every node, which can consume two to five times the raw vector storage depending on the connectivity parameter <Code>M</Code>. For a hundred-million-vector index at 768d, that overhead becomes significant.
      </Prose>

      <H3>IVF, ScaNN, DiskANN — when HNSW doesn't fit</H3>

      <Prose>
        For billion-scale indexes or memory-constrained deployments, other approaches become attractive. Inverted File (IVF) indexes cluster the vector space using k-means and build a list of vectors per cluster. At query time, the query is compared to cluster centroids and only the vectors in the nearest clusters are searched. Memory footprint is lower than HNSW for the same recall, but recall degrades faster when the query falls near a cluster boundary. The solution — probe more clusters — recovers recall at a throughput cost.
      </Prose>

      <Prose>
        Google's ScaNN adds asymmetric quantization within clusters: the indexed vectors are compressed to lower bit-width representations, while the query is kept at full precision. The asymmetry matters because quantization error on the stored vectors averages out over many candidates, while quantization error on a single query would be additive. ScaNN achieves very tight recall-per-FLOP trade-offs and is one of the dominant choices for high-throughput retrieval at Google scale.
      </Prose>

      <Prose>
        DiskANN, from Microsoft Research, approaches the billion-scale problem differently: it keeps most of the HNSW-style graph on disk and aggressively caches the hot nodes (entry points and high-degree hub nodes) in memory. With NVMe SSDs and careful prefetching, it serves billion-vector indexes from a single machine with DRAM overhead orders of magnitude smaller than an in-memory HNSW. The trade-off is query latency measured in low tens of milliseconds rather than sub-millisecond — acceptable for many batch or asynchronous workloads, less so for interactive search.
      </Prose>

      <H2>Where to actually run it</H2>

      <Prose>
        The vector database landscape has consolidated somewhat since the 2022–2023 explosion of new offerings, but there are still eight or nine products worth knowing. The right choice depends on the existing stack, the expected scale, and the operational budget.
      </Prose>

      <CodeBlock>
{`product          scale          pros                              cons
pgvector         <10M docs      inside your postgres; simple ops  ANN is secondary feature
Qdrant           <100M docs     mature, fast, open-source         separate service
Weaviate         <100M docs     hybrid search + graph built-in    slightly heavier ops
Milvus           <1B+ docs      designed for scale                operational complexity
Pinecone         any            fully managed SaaS                cost, vendor lock-in
Vespa            any            search + vector, battle-tested    steeper learning curve
turbopuffer      any            serverless, pay-per-use           newer, smaller community
Elasticsearch    <100M docs     if you already use ES             late to dense retrieval`}
      </CodeBlock>

      <Prose>
        A few practical notes behind the table. pgvector is the correct choice if you are already on Postgres and your corpus is under roughly five to ten million documents — the operational simplicity of staying in a single database is worth quite a lot, and pgvector's HNSW support added in v0.5.0 (2023) brought its retrieval performance into acceptable range. Qdrant has become the community default for medium-scale self-hosted deployments; it is well-documented, actively maintained, and its filtering performance is good. Milvus handles serious scale but the operational surface area is significantly larger. Pinecone's managed offering removes infrastructure burden entirely, which is worth paying for at early stages of a product; the cost at high query volume is real. Elasticsearch is worth choosing only when you already have it deployed for full-text search and want to add dense retrieval without introducing another service — the hybrid search integration is reasonable but it was not designed as a vector database and it shows at the edges.
      </Prose>

      <H3>Metadata filtering — the operational wildcard</H3>

      <Prose>
        Pure nearest-neighbor search is the easy part. Nearly every production retrieval use case adds at least one metadata predicate: find documents similar to this query <em>and</em> authored before this date, <em>and</em> belonging to this tenant, <em>and</em> tagged with this category. The naive approach — retrieve the top 1,000 nearest neighbors by vector similarity, then filter by metadata — fails when the filter is selective. If only 0.1% of the corpus matches the metadata predicate, a top-1,000 vector shortlist will often return zero passing results, or will return results from the tail of the similarity distribution after the relevant documents were excluded.
      </Prose>

      <Prose>
        The correct approach integrates filtering into the ANN search itself. Pre-filtering computes the eligible set by metadata predicate first and restricts the vector search to that subset. Filtered HNSW, implemented in Qdrant and Weaviate among others, propagates the filter constraint into the graph traversal so that the greedy walk only visits nodes that pass the predicate. The implementation complexity is real, and the performance characteristics under different filter selectivities vary substantially across products. A system that benchmarks well on pure vector recall can degrade badly under a 1% selectivity filter, or under a filter that combines multiple metadata dimensions. This is the single dimension most worth benchmarking against your specific workload rather than relying on published numbers from vendors.
      </Prose>

      <Callout accent="gold">
        Pure vector search is the easy part. Vector search with metadata filters at production scale is where vector database products meaningfully differ.
      </Callout>

      <Prose>
        The embedding-plus-index layer is an area where the open-source stack has largely caught up with commercial offerings. The choice of embedding model has more impact on retrieval quality than the choice of vector database for most workloads, and the open-weight model options available today give production results at zero marginal cost. The next topics in this section push retrieval further: graph-structured retrieval, agentic retrieval loops, and hybrid search that combines dense vector signals with sparse term-matching to handle the distribution of queries that neither approach handles well alone.
      </Prose>
    </div>
  ),
};

export default embeddingModelsVectorDB;
