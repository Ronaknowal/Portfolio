import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { StepTrace, TokenStream } from "../../components/viz";

const rag = {
  title: "Retrieval-Augmented Generation (RAG)",
  readTime: "16 min",
  content: () => (
    <div>
      <Prose>
        A language model knows exactly one thing: the text it saw during pretraining. Ask it about a document written yesterday, a company-internal policy, a customer ticket opened an hour ago, a news event that postdates the cutoff — and the model has two bad options. It can refuse, pleading ignorance, which is honest but useless. Or it can hallucinate a plausible-sounding answer constructed from whatever vaguely related patterns the weights happen to encode, which is useful-looking but wrong. Neither failure is acceptable for a product that serves a domain the model was not trained on, which is nearly every product.
      </Prose>

      <Prose>
        Retrieval-Augmented Generation is the workaround, and it fits in a sentence: at query time, fetch relevant text from an external store, paste it into the prompt, and let the model generate with that context in hand. The description is short. The operational gap between a RAG pipeline that serves correct, grounded answers and one that quietly returns confident nonsense spans an enormous amount of systems work — chunking, embedding, indexing, retrieval, reranking, prompt construction, evaluation, and the choice of which layer to fix when quality drops. This topic walks the whole pipeline, the failure modes that distinguish working RAG from theatre, and what moves one into the other.
      </Prose>

      <H2>The canonical RAG pipeline</H2>

      <Prose>
        Five stages, split across two timescales. Three run at query time: embed the user's question into a vector, retrieve the closest matching chunks from a vector store, and generate the answer with those chunks injected into the prompt. Two run offline: chunk the documents into retrievable pieces, embed each chunk, and index the embeddings. The offline work is done once per document; the online work is paid on every request. The separation exists because embedding a million chunks costs real money and several hours of compute; doing it at query time would make every request unusable. Precomputing once and reading the index back is what makes retrieval fast enough to sit inside a chat loop.
      </Prose>

      <StepTrace
        label="a RAG query — end to end"
        steps={[
          { label: "1. user asks a question", render: () => (
            <TokenStream tokens={[
              { label: "What's our return policy for international orders?", color: "#e2b55a" },
            ]} />
          ) },
          { label: "2. embed the query", render: () => (
            <TokenStream tokens={[
              { label: "query → embedding model → vector ∈ R^d", color: "#60a5fa" },
            ]} />
          ) },
          { label: "3. retrieve top-k similar chunks", render: () => (
            <TokenStream tokens={[
              { label: "vector db", color: "#c084fc" },
              { label: " →", color: "#555" },
              { label: " 5 chunks about returns policy", color: "#4ade80" },
            ]} />
          ) },
          { label: "4. construct prompt", render: () => (
            <TokenStream tokens={[
              { label: "system + context + question", color: "#e2b55a" },
            ]} />
          ) },
          { label: "5. llm generates answer grounded in the chunks", render: () => (
            <TokenStream tokens={[
              { label: "Based on our policy document...", color: "#4ade80" },
            ]} />
          ) },
        ]}
      />

      <Prose>
        Every stage is a place where quality can silently evaporate. If chunking cuts the answer in half, the right piece is never whole enough to retrieve. If the embedding model maps the query close to lexically similar but semantically unrelated text, the retrieved chunks are plausible-looking distractors. If the prompt template buries the context below a wall of boilerplate, the model reads the first paragraph and improvises the rest. Each layer is simple; the compounding is where production RAG lives or dies, and the only way to know which layer is failing is to measure each independently — which most teams do not.
      </Prose>

      <H2>Why not just stuff everything into the context?</H2>

      <Prose>
        The obvious question now that frontier models advertise one-million-token context windows. If the model can read a million tokens, why not paste the entire knowledge base into the prompt and skip retrieval altogether? The answer is three separate forces working against that plan, and none of them is going away even as context windows keep growing.
      </Prose>

      <Prose>
        The first is cost. A million-token context costs roughly a thousand times as much as a one-thousand-token context, because the per-token forward pass scales linearly with input length and the KV cache footprint scales on top of that. Retrieval narrows the input to the few kilobytes that actually matter, which is how per-query cost stays in cents rather than dollars. The second is quality. Long-context benchmarks show consistent degradation past roughly thirty to sixty thousand tokens of effective attention — the model technically reads the input, but its ability to use a specific fact buried in the middle falls off steeply, a phenomenon documented in the Context Window Extension topic. More context is not free for the model either; it is hidden cost paid in accuracy. The third is freshness. A context window is pasted in per request; a vector store is updated once and every subsequent query benefits. Change a policy document, reindex the chunk, and the model's next answer reflects the new policy within seconds.
      </Prose>

      <H2>Chunking — the unsung critical step</H2>

      <Prose>
        Before anything can be retrieved, raw documents have to be chopped into retrievable pieces. This stage gets the least attention in RAG papers and causes the most failures in RAG products, because it is the one step where the shape of the data meets the shape of the retrieval layer and neither was designed with the other in mind. A chunk that is too small lacks the context to answer questions about itself; a chunk about "Section 4.2" without its preamble is a fragment that retrieves on keyword match but cannot stand alone. A chunk that is too large dilutes the retrieval signal, because the one relevant sentence is embedded alongside five paragraphs of unrelated material and the average vector drifts away from the query. The right chunk size is workload-dependent, almost always between 200 and 1000 tokens, and picking it by default rather than by measurement is the most common first mistake.
      </Prose>

      <Prose>
        Three approaches, in rough order of sophistication. Fixed-size chunks with overlap are the classic starting point — slide a window of N words across the document, advance by N minus a small overlap, emit each window as a chunk. It works for homogeneous text and nothing else, because it ignores the structure the document already has. Structure-aware chunking splits at paragraph, section, or heading boundaries and packs until hitting a size cap, which preserves the semantic units the author originally intended. Semantic chunking goes further: run a small model over the text to detect natural breakpoints — topic shifts, argumentative pivots, changes in subject — and cut there. The sophistication costs inference compute at index time but pays back at query time in cleaner retrieval.
      </Prose>

      <CodeBlock language="python">
{`def chunk_document(text, chunk_size=500, overlap=50):
    """Fixed-size sliding window. Simple; loses structural info."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def chunk_by_structure(markdown_doc, max_chunk_size=800):
    """Split at markdown section boundaries, pack until hitting max size."""
    sections = markdown_doc.split("\\n## ")
    chunks, current = [], ""
    for section in sections:
        if len(current) + len(section) > max_chunk_size and current:
            chunks.append(current)
            current = section
        else:
            current += ("\\n## " if current else "") + section
    if current: chunks.append(current)
    return chunks`}
      </CodeBlock>

      <Prose>
        One practical move that costs nothing and helps almost everywhere: prepend document-level metadata — title, section path, source URL — to every chunk before embedding. A chunk that starts with <Code>[Returns Policy &gt; International Orders]</Code> embeds closer to queries about international returns than the same chunk without the header, because the embedding model is now seeing hierarchical context the table of contents previously supplied. It is a one-line change and reliably moves retrieval recall up by several points.
      </Prose>

      <H2>Embedding — how "similarity" gets computed</H2>

      <Prose>
        An embedding model maps text to a fixed-dimensional vector such that semantically-similar text produces vectors with a small angle between them. Cosine similarity on those vectors is what retrieval uses to score candidate chunks against the query. The modern standard is a bi-encoder — a BERT-style transformer fine-tuned with contrastive loss on paired queries and documents, so that matched pairs pull together in the embedding space and mismatched ones push apart. Open-weight models in this shape — BGE, E5, GTE, Jina, Nomic — now match or exceed commercial offerings like OpenAI's text-embedding-3 and Cohere's embed-v3 on the public MTEB leaderboard, and they can be self-hosted for a fraction of the cost at production volumes.
      </Prose>

      <Prose>
        Typical dimensions are 384, 768, 1024, or 1536; a new generation of Matryoshka models nests several dimensions inside one — the first 128 dimensions are usable as a standalone low-dimensional embedding, the first 768 as a mid-dimensional one, the full 1536 as the highest-quality variant. One model serves fast low-dimensional retrieval for the first-pass top-100 and high-quality high-dimensional rerank for the final top-5, which saves both memory and latency without running two separate embedding models. The embedding is query-document symmetric in most modern models: the user's question is embedded with the same encoder as the document chunks, which is what makes retrieval cheap — it is a nearest-neighbor lookup over an index that has already been built, not a pairwise scoring over the full corpus.
      </Prose>

      <H3>The retrieval step</H3>

      <Prose>
        At million-document scale, exhaustive nearest-neighbor search is too slow for an interactive request. Production stacks use approximate nearest neighbor (ANN) indexes — HNSW, IVF, ScaNN — that trade a small quality drop for a massive speed gain. HNSW, the most common, builds a layered graph where each node links to close neighbors at multiple resolutions; queries traverse from coarse to fine in log-time rather than linear. Typical configurations recover 95 to 99 percent of the true top-k at a hundred-fold speedup over exact search. The vector database behind the index — FAISS, pgvector, Qdrant, Weaviate, Pinecone, Milvus — is increasingly a deployment preference rather than a quality decision; the algorithms are well-understood and converge on similar tradeoffs.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

def retrieve(query, embedding_model, vector_store, top_k=5):
    """Minimal retrieval — no reranking, no hybrid, no metadata filter."""
    query_vec = embedding_model.embed(query)
    results = vector_store.search(query_vec, top_k=top_k)
    return [
        {"text": hit.document, "score": hit.score, "source": hit.metadata["source"]}
        for hit in results
    ]

def build_rag_prompt(query, chunks):
    context = "\\n\\n".join(f"[Source {i+1}] {c['text']}" for i, c in enumerate(chunks))
    return f"""Use the following sources to answer the question. Cite by source number.

{context}

Question: {query}
Answer:"""`}
      </CodeBlock>

      <Prose>
        The prompt template in the second function looks incidental and is not. Numbering the sources and asking the model to cite by number does two things at once: it gives the model an explicit grounding signal, and it gives you an audit trail at evaluation time — you can check whether the cited source actually contained the cited claim. Templates that skip citation lose both benefits and tend to produce answers that drift from the retrieved context. A prompt template is not decoration; it is a training signal applied at inference time, and small changes move answer quality in measurable ways.
      </Prose>

      <H2>Where naive RAG goes wrong</H2>

      <Prose>
        The failure modes are distinctive and catalogable, which is the good news — each one has a signature that is visible in the right eval, and each has a known fix. The bad news is that a vanilla pipeline hits most of them simultaneously, and telling them apart without instrumentation is nearly impossible. Semantic mismatch is the first: the query embeds near irrelevant chunks because of lexical overlap or query ambiguity, and the retrieved set never contained the answer in the first place. Chunk fragmentation is the second: the answer spans two or three chunks that happen to be separated in the index, and retrieval pulls one piece without its context. Stale data is the third and most embarrassing: the source document changed and the vector store did not, so the model confidently quotes outdated information. Retrieval overconfidence is the fourth, and the one teams underestimate most: the top-5 chunks do not contain the answer, but the model answers from them anyway, fabricating the gap because the prompt template asked it to ground in the sources and the sources did not ground anything. And the "who cares" failure: the question did not actually need retrieval — the model's pretrained knowledge was already better than what the chunk store returned — but the retrieval ran anyway and polluted the context.
      </Prose>

      <Prose>
        Detecting these requires evaluation that goes further than "did the answer sound good." It means asking four questions per eval example: did the retrieved chunks contain the answer (context recall), were they mostly relevant rather than noise (context precision), did the generated answer actually use the chunks rather than hallucinate around them (faithfulness), and did the answer address what was asked rather than dodge it (answer relevance). All four decompose cleanly into things you can measure automatically with a judge model, and none of them can be traded off against the others without regressing somewhere visible. A RAG system where only one metric is tracked will get optimized until the other three silently collapse.
      </Prose>

      <H3>Reranking — the under-used fix</H3>

      <Prose>
        The single cheapest meaningful improvement to a working-but-mediocre RAG pipeline. Retrieve the top fifty chunks with the bi-encoder — which is cheap, because the index lookup is the whole cost — then rerun those fifty through a more expensive cross-encoder model that scores each (query, chunk) pair jointly rather than independently. Keep the top five from that reranked list, drop the rest. Cross-encoders are roughly a hundred times slower per pair than bi-encoders, but they are also vastly more accurate, because they can attend across the query and the document simultaneously rather than mapping each to a vector independently and hoping the geometry aligns. Cohere Rerank, BGE-Rerank, Jina-Reranker, and several commercial alternatives are all production-ready. The typical improvement on retrieval recall@5 is 10 to 30 percentage points on standard RAG benchmarks, for a few milliseconds of added latency and a small CPU bill. It is the most favorable ratio of effort to quality in the whole pipeline, and it is routinely skipped.
      </Prose>

      <H2>Evaluation — the thing most teams skip</H2>

      <Prose>
        RAG evaluation decomposes cleanly into retrieval quality and generation quality, and the frameworks that have emerged — RAGAS, Trulens, DeepEval, Phoenix — all converge on a similar set of metrics operationalized through a judge LLM. Context precision asks whether the retrieved chunks were relevant to the question. Context recall asks whether the chunks contained enough information to answer it in the first place. Faithfulness asks whether the generated answer stayed grounded in the provided chunks rather than drifting into invented content. Answer relevance asks whether the answer addressed the question rather than a related but different one. The four metrics pull against each other in practice — optimizing only faithfulness can make the model refuse to answer anything where the chunks are imperfect, optimizing only answer relevance can let the model improvise past missing context, and so on — which is why the standard dashboard tracks all four and regressions in any one of them block deploys.
      </Prose>

      <Callout accent="gold">
        A RAG system's ceiling is set by retrieval quality. You cannot out-prompt bad retrieval. Fix retrieval evaluation first; optimize generation second.
      </Callout>

      <Prose>
        The cheapest version of "real" RAG evaluation is roughly a hundred hand-written (question, ideal-answer, source-chunks) triples curated by someone who understands the domain. Run the pipeline over the hundred questions, have a judge model score each of the four metrics, aggregate, track the trend across deploys. That is it. The elaborate eval frameworks are scaffolding around this core loop; the core loop works with a spreadsheet and a small script. Teams that skip it ship pipelines that score well on vibes and fail silently on real traffic, because vibes scale with how pleasant the top-of-funnel queries feel and not with how reliably the product answers the long tail.
      </Prose>

      <H2>Advanced patterns — a brief survey</H2>

      <Prose>
        Several techniques layer on top of the vanilla pipeline and show up in production stacks often enough to mention, though each has its own topic later in this section. Query rewriting uses the LLM to expand, decompose, or reformulate the user's question before embedding — turning "what about international orders?" into a self-contained query with the conversation context folded in. Multi-query RAG embeds several paraphrases of the question in parallel and unions the retrieved sets, which catches cases where a single phrasing happened to miss the chunk with a slightly different vocabulary. Hybrid search blends dense retrieval with a classical lexical score like BM25, which is covered in its own topic; the combination reliably beats either alone because dense and sparse retrievers fail on different queries. Iterative and agentic RAG lets the model decide whether it needs to retrieve again mid-generation — covered in the GraphRAG and Agentic RAG topic — which turns the single-shot pipeline above into a loop where the model can ask for more evidence before committing to an answer. Each of these is a layer on top of the same base pipeline, and none of them substitutes for the base being correct first.
      </Prose>

      <H3>What RAG doesn't fix</H3>

      <Prose>
        Worth naming explicitly, because the marketing tends to blur it. RAG does not teach the model new skills. If the model cannot follow a complex multi-clause policy, retrieving the policy and pasting it into the prompt does not suddenly make it able to; the gap is in the reasoning, not the information. RAG does not fix reasoning failures. A model that cannot chain a three-step inference will not start chaining one just because the premises are present in context. RAG does not fix attention limits. Stuffing a hundred retrieved chunks into an eight-thousand-token context window means the model reads the first thirty and ignores the rest, which looks like the retrieval worked but actually means half the evidence was silently dropped. And RAG does not fix hallucination in any categorical sense — it only reduces the rate on questions where the retrieval succeeded; on questions where it failed, the model still invents, often more confidently because the prompt instructed it to ground in sources and the sources looked relevant enough to hide behind. The line between "RAG fixed the problem" and "RAG hid the problem" is narrow and only visible from the eval dashboard.
      </Prose>

      <H2>Closing</H2>

      <Prose>
        RAG has become table stakes for any LLM product with a domain-specific knowledge base, which is most of them. The shift from 2022, when retrieval was an advanced pattern, to now, where it is the default architecture for any chatbot that has to know something the base model does not, happened fast enough that a lot of the tooling is still maturing. The pipeline this topic describes is the scaffolding the rest of the section will decorate: embedding models and their training regimes, vector stores and their index structures, hybrid and lexical retrieval, graph-structured and agentic variants, and the evaluation machinery that separates RAG systems that work from ones that look like they do. The picture as a whole is less a new technique than a renegotiation of where the knowledge in an LLM product lives — not in the weights, where it is expensive to update and easy to outdate, but in an external store the model consults at the moment it answers.
      </Prose>
    </div>
  ),
};

export default rag;
