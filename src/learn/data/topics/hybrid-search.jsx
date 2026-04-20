import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { StepTrace, TokenStream } from "../../components/viz";

const hybridSearch = {
  title: "Hybrid Search (Dense + Sparse + Reranking)",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        Dense vector search is good at finding meaning across phrasing. Ask "how do I reset my
        password" and a well-trained bi-encoder will retrieve a document titled "Account
        Recovery" even though the word "reset" appears nowhere in it — because the two phrases
        occupy nearby regions of the embedding space. That generalization is the source of its
        value, and the source of its failure. Ask for a specific product ID, a function name, a
        legal citation, or a rare error code, and the dense retriever will return whatever happens
        to be semantically adjacent, which is rarely the exact document you need. Exact
        terminology sits poorly in a space built for approximate meaning.
      </Prose>

      <Prose>
        Classical sparse search inverts that tradeoff exactly. BM25 and its ancestors operate on
        term overlap: a query matches documents that share its tokens, weighted by how rare those
        tokens are across the corpus. Exact match is the native operation. Paraphrase is
        invisible — a document about "credential recovery" will score zero against a query for
        "password reset" unless those words appear in both. Most real query logs contain both
        types: natural-language questions where meaning matters and lookup queries where spelling
        is everything. Hybrid search runs both retrieval modes, combines their rankings, and
        consistently outperforms either alone. Every serious production RAG system uses some
        variant of this pattern.
      </Prose>

      <H2>What BM25 actually is</H2>

      <Prose>
        BM25 — Best Matching 25 — is a refinement of TF-IDF formalized by Robertson and
        colleagues in 1994, and it remained the dominant sparse retrieval method for the three
        decades of information retrieval research that preceded the embedding era. The formula
        is a weighted sum over each query term: how often the term appears in the document
        (term frequency, but dampened so that repetition has diminishing returns), how rare the
        term is across the entire corpus (inverse document frequency), and a length normalization
        that prevents long documents from winning simply by virtue of containing more words.
      </Prose>

      <MathBlock>{"\\text{BM25}(q, d) = \\sum_{t \\in q} \\text{IDF}(t) \\cdot \\frac{f(t, d) \\cdot (k_1 + 1)}{f(t, d) + k_1 \\cdot (1 - b + b \\cdot |d|/\\text{avgdl})}"}</MathBlock>

      <Prose>
        Two tunable parameters control the behavior. <Code>k₁</Code> governs term-frequency
        saturation — how quickly additional occurrences of a term stop contributing to the
        score; typical values are 1.2 to 2.0. <Code>b</Code> governs length normalization;
        0.75 is the conventional default, with 0 disabling length normalization entirely and 1
        applying it fully. There is no neural network. BM25 is counting, two hyperparameters,
        and an inverted index. On many benchmarks, it still beats naive dense retrieval for
        exact-term queries — not because it is more sophisticated, but because its native
        operation is exactly what those queries require.
      </Prose>

      <H2>When each wins</H2>

      <Prose>
        The failure modes are complementary, which is the whole motivation for combining them.
        Dense retrieval wins where language varies but meaning is stable: semantic questions
        ("how do I reset my password" finds "account recovery"), cross-lingual retrieval where
        the query and document are in different languages, and paraphrased queries where the
        user's phrasing has no word-level overlap with the relevant document. Sparse retrieval
        wins where exact terminology is the signal: product IDs and model numbers, error codes
        and stack trace signatures, specific function names and API endpoints, rare proper nouns
        the embedding model encountered too infrequently to place accurately in the embedding
        space, domain jargon that did not appear often enough in the model's pretraining corpus
        to carry a reliable vector. Medical abbreviations, legal citations, chemical identifiers,
        part numbers — these are sparse retrieval territory.
      </Prose>

      <Prose>
        Real query logs are a mix of both shapes. A customer asking "my order for item
        TX-4891 hasn't shipped" is simultaneously a natural-language question (dense wins on the
        intent) and a lookup query (sparse wins on the specific item ID). Any single-mode
        retriever will fail one half or the other. The hybrid approach does not require knowing
        in advance which mode a query needs — it retrieves candidates from both channels and lets
        the fusion step resolve it.
      </Prose>

      <H2>Reciprocal Rank Fusion — the simple combiner</H2>

      <Prose>
        Given a dense ranking and a sparse ranking, both over the same document corpus, the
        simplest way to merge them is Reciprocal Rank Fusion. For each document, compute its
        reciprocal rank in each input ranking — <Code>1 / rank</Code> — and sum across all
        rankings. Documents that rank near the top of both lists accumulate the highest scores;
        documents that appear only in one list are included at a lower score but not discarded.
        The result is a single combined ranking with a clean property: it is robust to the two
        input lists operating on different score scales, because only rank positions are used,
        never raw scores.
      </Prose>

      <CodeBlock language="python">
{`def reciprocal_rank_fusion(*rankings, k=60):
    """Combine multiple rankings into one. k is a smoothing constant."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=scores.get, reverse=True)

# Usage — combine dense and sparse top-100 lists:
combined = reciprocal_rank_fusion(
    dense_top100,   # from vector store
    sparse_top100,  # from BM25
)`}
      </CodeBlock>

      <Prose>
        The smoothing constant <Code>k</Code>, conventionally set to 60, makes the fusion stable
        when one ranking assigns very different score magnitudes than the other. Without it, a
        dense retriever that returns confidence scores between 0.98 and 0.99 would have its ranks
        collapse toward each other; RRF sidesteps that entirely by throwing away the scores and
        operating only on position. The implementation above is complete and production-ready for
        the merging step; actual retrieval of <Code>dense_top100</Code> and
        <Code>sparse_top100</Code> is handled by whatever vector store and BM25 engine the
        system already has deployed.
      </Prose>

      <H3>Learned sparse — SPLADE and friends</H3>

      <Prose>
        Between the fully dense bi-encoder and the purely term-counting BM25 sits a class of
        models that produce sparse vectors over the vocabulary dimension rather than dense
        vectors over a learned embedding space. SPLADE (Formal et al., 2021) takes a BERT-style
        encoder and trains it to output a weighted bag of vocabulary terms for each query or
        document — not just the terms that appear in the raw text, but terms that are
        semantically relevant, including synonyms and related concepts the original text does not
        contain. A document about "heart attack" gets nonzero weights on "myocardial infarction";
        a query for "car" gets weight on "vehicle" and "automobile". The output is still sparse
        — most vocabulary dimensions are zero — which means it can be indexed and queried with
        the same inverted-index infrastructure that serves BM25, at roughly similar latency.
        Quality on standard benchmarks approaches dense retrieval, with better behavior on
        out-of-domain queries where the embedding model's geometry has not been well-calibrated.
        SPLADE can be run in the sparse channel of a hybrid system in place of or alongside
        BM25, giving term-level infrastructure access with representation-learning-level
        generalization.
      </Prose>

      <H2>Reranking — the third stage</H2>

      <Prose>
        The dense and sparse channels together produce a merged shortlist — typically 50 to 100
        candidates after RRF. That shortlist is still scored by models that processed the query
        and document independently: the bi-encoder embedded each into a vector without attending
        to the other; BM25 counted terms without modeling any interaction. Reranking runs a
        cross-encoder over the shortlist — a model that takes both the query and a candidate
        document as a single joint input and produces a relevance score by attending across both
        simultaneously. The cross-encoder can see whether the query's subject appears in the
        same sentence as the document's key claim, whether a term is used in the same sense in
        both texts, whether the document's answer actually addresses the specific constraint the
        query imposes. Bi-encoders cannot do any of this by construction; their vectors are
        computed in isolation.
      </Prose>

      <Prose>
        The practical consequence is large. Cross-encoders are typically 10 to 30 percentage
        points more accurate at identifying the genuinely relevant result in a shortlist, on
        standard retrieval benchmarks. The cost is proportional to shortlist size, and shortlists
        are small — reranking 50 documents adds a few hundred milliseconds in the worst case,
        which is negligible next to generation latency. Production rerankers available today
        include BGE-Rerank and its successors (open weight, self-hostable), Cohere Rerank
        (commercial API), Jina Reranker, and mixedbread-ai's reranker family. The integration
        is a single call wrapping the shortlist; nothing else in the pipeline changes. The
        standard pattern is to retrieve wide — 100 to 200 candidates across both channels —
        rerank down to a tight shortlist of 3 to 10, and pass that to the language model.
      </Prose>

      <H3>Putting it all together</H3>

      <StepTrace
        label="a production hybrid search pipeline"
        steps={[
          { label: "1. dense retrieval — top 100", render: () => (
            <TokenStream tokens={[
              { label: "query → embedding → HNSW → 100 candidates", color: "#60a5fa" },
            ]} />
          ) },
          { label: "2. sparse retrieval — top 100", render: () => (
            <TokenStream tokens={[
              { label: "query tokens → BM25 → 100 candidates", color: "#c084fc" },
            ]} />
          ) },
          { label: "3. merge via rrf", render: () => (
            <TokenStream tokens={[
              { label: "dense + sparse → RRF → top 50", color: "#e2b55a" },
            ]} />
          ) },
          { label: "4. rerank the top 50", render: () => (
            <TokenStream tokens={[
              { label: "cross-encoder scores all 50", color: "#4ade80" },
            ]} />
          ) },
          { label: "5. top 5 to llm", render: () => (
            <TokenStream tokens={[
              { label: "best 5 → prompt → llm → answer", color: "#4ade80" },
            ]} />
          ) },
        ]}
      />

      <H3>Practical guidance</H3>

      <Prose>
        If you can only deploy one thing beyond vanilla dense retrieval, deploy reranking. The
        implementation surface is a single API call or a small model serving endpoint; the
        quality return is large and consistent across query types. It does not require changes
        to your vector store, your chunking, or your embedding model. It slots in between
        retrieval and prompt construction and requires nothing else to change. If you can deploy
        two things, add BM25 via whatever text-search infrastructure is already in your stack —
        Elasticsearch, OpenSearch, Postgres full-text search, SQLite FTS5. The sparse channel
        is already indexing your documents in most production architectures; surfacing its top-k
        for fusion requires configuring an additional query path, not building new
        infrastructure. Hybrid plus reranker is the mature-stack default. It is uncommon to
        find a production RAG deployment that has been running long enough to have been
        optimized and that does not use both.
      </Prose>

      <Callout accent="gold">
        Hybrid search with reranking is close to a free lunch for RAG quality. The cost is
        engineering complexity; the gain is consistent, measurable improvement across most
        query shapes.
      </Callout>

      <Prose>
        The Large Language Models track traced the arc from raw text to production system:
        tokenization turned text into tokens, pre-training turned tokens into knowledge,
        post-training turned knowledge into aligned behavior, inference optimization turned
        behavior into served products, system design turned services into global infrastructure,
        and retrieval turned frozen knowledge into fresh-context reasoning. At every layer the
        underlying question recurs: how do you get the most useful behavior out of a finite
        budget of parameters, tokens, GPUs, and careful human labor?
      </Prose>

      <Prose>
        The answers keep shifting — the field moves fast enough that the "state of the art"
        entry in this track will be outdated in months. But the shape of the pipeline is stable
        now. Something that looks like tokenization → pre-training → post-training → inference
        will remain the frame for how we build these systems. What changes, and will keep
        changing, is what each of those layers actually does. That's the invitation of the
        remaining tracks in this hub — to follow the same technical substrate into new domains.
      </Prose>
    </div>
  ),
};

export default hybridSearch;
