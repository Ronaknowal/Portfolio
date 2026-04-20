import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { StepTrace, TokenStream } from "../../components/viz";

const graphragAgentic = {
  title: "GraphRAG & Agentic RAG",
  readTime: "12 min",
  content: () => (
    <div>
      <Prose>
        The previous topic covered vanilla RAG: embed query, retrieve top-k chunks by cosine similarity, hand those chunks to the model as context. That works well when the answer lives inside a single chunk. It breaks down in two common situations. First, when the answer requires connecting facts spread across many documents — no individual chunk contains the full picture, so no single retrieval step surfaces it. Second, when the question is itself vague or multi-step, so a single retrieval query fails to gather the right evidence. Two lines of work address these gaps. GraphRAG treats the corpus as a knowledge graph, built offline so that structured relationships are queryable at inference time. Agentic RAG lets the model actively steer the retrieval loop — issuing queries, examining results, deciding whether to retrieve more. Both represent a layer above basic retrieval: more capable, more expensive, worth it on the right class of questions.
      </Prose>

      <H2>What GraphRAG actually does</H2>

      <Prose>
        Microsoft Research's GraphRAG (Edge et al., 2024) separates retrieval into two phases: an expensive offline index build and a fast online query. During indexing, an LLM reads every chunk in the corpus and extracts structured (entity, relationship, entity) triples — the kind of extraction a traditional information-extraction pipeline would do, but via a general-purpose language model rather than a task-specific NER system. Those triples form a graph. A community-detection algorithm — typically Louvain — then partitions the graph into clusters, and the LLM writes a short summary for each cluster. At query time, the query is embedded and used to select the most relevant community summaries, which become the context handed to the final generation step.
      </Prose>

      <Prose>
        The preprocessing is expensive in the most literal sense: one LLM call per chunk at index time, plus summary generation per community. For a 10M-word corpus that can reach $1k–10k in indexing compute depending on model and chunk granularity. But the cost is amortized: you pay it once, then query the resulting index indefinitely. Runtime query cost is roughly comparable to vanilla RAG — an embedding call plus a final generation call.
      </Prose>

      <StepTrace
        label="graphrag — offline indexing + online retrieval"
        steps={[
          { label: "1. extract entities + relations per chunk", render: () => (
            <TokenStream tokens={[
              { label: "chunk", color: "#888" },
              { label: " → LLM extract →", color: "#c084fc" },
              { label: "(Alice, works_at, Foo Inc)", color: "#4ade80" },
            ]} />
          ) },
          { label: "2. build graph + community clusters", render: () => (
            <TokenStream tokens={[
              { label: "nodes/edges → louvain clustering → summary per cluster", color: "#60a5fa" },
            ]} />
          ) },
          { label: "3. at query time, pick relevant communities", render: () => (
            <TokenStream tokens={[
              { label: "query → embed → top-k community summaries", color: "#e2b55a" },
            ]} />
          ) },
          { label: "4. generate grounded in those summaries", render: () => (
            <TokenStream tokens={[
              { label: "community summaries + query → llm → answer", color: "#4ade80" },
            ]} />
          ) },
        ]}
      />

      <H2>What GraphRAG is actually good at</H2>

      <Prose>
        GraphRAG addresses three specific failure modes of vanilla retrieval. Multi-hop questions are the canonical case: "Which companies are connected to Alice through shared board members?" No single chunk contains the full traversal; the graph does, because the entity-extraction step explicitly built the board-member edges. Community summaries make global questions tractable: "What are the main themes of this corpus?" Chunk-level retrieval returns a random sample of the corpus, biased toward whatever happens to be close in embedding space to the query phrasing. Community summaries give structured, corpus-wide coverage that chunk retrieval structurally cannot. Finally, relationship-heavy corpora — organizational hierarchies, legal chains of custody, scientific citation networks — are cases where the relationships between entities are as important as the entities themselves, and where building the graph pays dividends on almost every query.
      </Prose>

      <Prose>
        The cases where GraphRAG is not worth it are equally clear. Simple fact lookup ("What is the boiling point of ethanol?") costs the same to answer with vanilla RAG and requires no graph traversal; GraphRAG adds latency and complexity for no benefit. Frequently-updating corpora are expensive to maintain because entity extraction and community summarization must be rerun whenever the corpus changes significantly — the index is not incremental by default. Heterogeneous or noisy data is a harder problem: LLM-based entity extraction degrades on informal text, domain-specific jargon without context, or documents with low information density, and a noisy graph with spurious edges produces misleading community summaries that can hurt generation quality rather than help it.
      </Prose>

      <H3>The cost</H3>

      <Prose>
        The indexing bill dominates any GraphRAG deployment decision. Preprocessing a corpus into a GraphRAG index uses LLM inference at index time — typically $10–100 per 100k words for moderate-quality extraction, depending on model choice and chunk size. For a 10M-word corpus that is $1k–10k of indexing compute before any graph construction or community summarization is counted. A 100M-word corpus scales linearly. The bill is not a one-time surprise; it recurs whenever the corpus is updated substantially. Runtime query cost is comparable to vanilla RAG — one embedding call and one generation call against the retrieved summaries rather than retrieved chunks. The break-even is entirely determined by whether the reasoning-quality gains justify the indexing bill, a calculation that strongly favors relationship-heavy, stable corpora and disfavors document sets that change weekly.
      </Prose>

      <H2>Agentic RAG — the orthogonal direction</H2>

      <Prose>
        GraphRAG changes what is indexed. Agentic RAG changes who controls the retrieval loop. Rather than a single retrieve-then-generate pass, the model itself issues retrieval queries mid-generation, inspects the results, and decides whether to retrieve more, reformulate the query, or produce a final answer. The model is an active participant in the information-gathering process rather than a passive recipient of whatever the retrieval pipeline returns.
      </Prose>

      <Prose>
        The simplest form is self-RAG (Asai et al., 2023): a fine-tuned model learns to emit special retrieval tokens at appropriate points in its generation — a signal that the generation requires external evidence before continuing. The model produces the tokens, a retrieval step runs, the results are injected into the context, and generation resumes. More sophisticated implementations use general tool-use: the model calls a <Code>search</Code> tool with a query string, receives chunk results as a structured observation, appends that to the conversation, and continues reasoning. This is the ReAct pattern (Reason + Act) applied to retrieval — the model interleaves reasoning steps with external actions.
      </Prose>

      <H3>The self-RAG / ReAct pattern</H3>

      <CodeBlock language="python">
{`# Simplified agentic RAG loop — the model controls retrieval

async def agentic_rag(query, model, vector_store, max_steps=5):
    transcript = [{"role": "user", "content": query}]
    for step in range(max_steps):
        response = await model.generate(transcript, tools=["search", "answer"])

        if response.tool_call == "answer":
            return response.final_answer

        if response.tool_call == "search":
            results = vector_store.search(response.search_query, top_k=5)
            transcript.append({"role": "assistant", "tool": "search", "query": response.search_query})
            transcript.append({"role": "tool", "content": format_chunks(results)})

    return "Could not find a confident answer after search."`}
      </CodeBlock>

      <H2>What agentic RAG wins and loses</H2>

      <Prose>
        The wins are structural. A single-shot retrieval must guess the right query on the first try. An agentic loop can follow a reasoning chain: first retrieve background on entity X, then use what it learned about X to formulate a more precise query for the relationship between X and Y. Ambiguous queries get a second chance through reformulation — if the first search returns noise, the model can rephrase or decompose the question before trying again. The loop also handles the common failure mode where the top-k results are close but not quite right, by issuing a follow-up query rather than silently hallucinating around the gap. The net effect is that hard multi-hop questions that vanilla RAG cannot answer in one pass become tractable.
      </Prose>

      <Callout accent="gold">
        Agentic RAG trades single-shot simplicity for reasoning. It's worth the tradeoff when the questions genuinely require multi-hop retrieval; it's waste when they don't.
      </Callout>

      <Prose>
        The losses are also structural. Each retrieval-reasoning cycle is a sequential LLM call — you cannot parallelize a chain where step two depends on the output of step one. A five-step agentic loop costs five times the latency and five times the token budget of a single-shot retrieve-generate pass. Failure modes shift from "bad retrieval" to "the model loops or gets stuck" — the loop needs a termination condition, and choosing it correctly is non-trivial. The approach works best on reasoning-tuned models (o-series, Claude, DeepSeek-R1) where the model reliably follows multi-step tool-use instructions; weaker models tend to issue poorly-formed queries, fail to stop at the right moment, or ignore retrieved evidence in favor of memorized answers.
      </Prose>

      <H3>Combining both</H3>

      <Prose>
        GraphRAG and agentic RAG are not mutually exclusive. A mature stack can use GraphRAG as the index layer and agentic retrieval as the query layer: the model walks the knowledge graph via tool calls, retrieving community summaries for one subgraph, then issuing a follow-up retrieval into an adjacent subgraph, accumulating context across several steps. This is roughly the architecture that production research-grade agents implement — what vendors market as "Deep Research" assistants. The model navigates the index rather than querying it once. The engineering surface is substantial: you need reliable entity extraction, a maintained graph index, a tool-use-capable model, and careful loop management. The capability gain on genuinely hard multi-hop, multi-document questions is real and not achievable by simpler means.
      </Prose>

      <Prose>
        GraphRAG and agentic RAG represent the two directions retrieval is evolving: richer index structures that encode relationships the document text only implies, and more active retrieval loops that let the model gather exactly the evidence it needs rather than accepting a fixed set of chunks. Neither dominates vanilla RAG on simple, single-chunk questions; both meaningfully extend what retrieval can answer when the questions are hard. The section's remaining topics — model merging and hybrid search — cover complementary techniques for getting more capability out of existing models and indexes without the full engineering weight of either approach.
      </Prose>
    </div>
  ),
};

export default graphragAgentic;
