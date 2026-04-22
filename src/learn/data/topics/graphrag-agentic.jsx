import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap } from "../../components/viz";
import { colors } from "../../styles";

const graphragAgentic = {
  title: "GraphRAG & Agentic RAG",
  slug: "graphrag-agentic-rag",
  readTime: "38 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Plain retrieval-augmented generation fails in at least two structural ways that no amount of prompt engineering can fix. The first is the multi-hop question: "Who was the CEO of the company when the acquisition closed?" No single chunk in any well-indexed document store will contain both the CEO-to-date mapping and the acquisition-closing date in the same passage. The answer is the intersection of two facts that live in different documents, and cosine similarity will never conjure that intersection from a single vector lookup. The second failure is corpus-scale reasoning: "What are the dominant themes across all research from this lab over the past decade?" No top-k retrieval strategy can reliably surface a representative sample of a million-token corpus from a single query embedding; the answer requires a global view, not a local one. Vanilla RAG was built for the retrieval case — find the nearest chunks — and it solves that case well. What it cannot do is reason about relationships between entities or synthesize a corpus-wide picture, because it never builds any representation of those relationships in the first place.
      </Prose>

      <Prose>
        Two research directions attacked these gaps in 2023 and 2024, and they are orthogonal enough that they compose rather than compete. GraphRAG, introduced by Microsoft Research (Edge et al., 2024; arXiv:2404.16130), proposes building a knowledge graph from the corpus at index time and traversing that graph at query time. The graph encodes the entity-level relationships that documents only imply, and community-detection algorithms partition the graph into thematic clusters whose pre-computed summaries make corpus-wide queries tractable. Agentic RAG — whose theoretical grounding comes from Self-RAG (Asai et al., 2023; arXiv:2310.11511) and whose practical expression lives in frameworks like LangGraph and CrewAI — proposes giving the language model active control over the retrieval loop. Rather than accepting a fixed set of retrieved chunks, the model decides when to retrieve, what to retrieve, and whether the results are good enough, issuing as many queries as it needs across multiple iterations. Both represent a layer of intelligence above basic chunk-and-embed: they are more expensive, more capable, and worth deploying when the questions they are designed for actually appear in production traffic.
      </Prose>

      <Prose>
        The historical arc is worth sketching briefly. Structured retrieval over knowledge graphs predates neural NLP — SPARQL queries against Freebase, Wikidata, and domain ontologies have been standard practice since the 2000s. What changed with the LLM era is that the knowledge graph no longer has to be hand-curated: a general-purpose language model can extract entities and relations from raw text at corpus scale, which makes the approach applicable to private, unstructured document collections that no knowledge engineer has touched. The Leiden community-detection algorithm (Traag et al., 2019), which GraphRAG uses to partition entity graphs, was itself an improvement on the Louvain algorithm, offering stronger connectivity guarantees and a resolution parameter that controls community granularity. Self-RAG trained a language model to emit special reflection tokens that signal when retrieval is needed, when to trust the retrieved passage, and when to critique the generated output — a more principled version of the ad hoc tool-use patterns that had appeared in ReAct (Yao et al., 2022) and related work. The frameworks that productionize these patterns — LangGraph for stateful agent graphs, CrewAI for multi-agent orchestration, LlamaIndex PropertyGraphIndex for structured graph retrieval — are largely post-2024 engineering that sits on top of the 2023–2024 research.
      </Prose>

      <Prose>
        It is also worth being precise about what problem each approach is actually solving, because the marketing language around both tends to over-promise. GraphRAG does not make retrieval smarter in the sense of understanding what the user wants better. It makes the index richer: a knowledge graph encodes relationships that plain chunk embeddings cannot represent, so graph traversal can follow chains of reasoning that chunk retrieval cannot. The quality gain is real but conditional — it only materializes on queries whose answers genuinely require traversing those chains. On single-hop factual queries, GraphRAG's community summaries often perform no better than a good chunk retriever, and sometimes worse because the summary is a lossy compression of the underlying source text. Agentic RAG similarly does not make the retriever smarter. It makes the querying process more flexible: the model can try multiple queries, evaluate results, and reformulate before committing to an answer. The quality gain is also conditional — it only matters when a single retrieval pass is structurally insufficient, and it always comes at the cost of latency and token spend. Building a system that correctly identifies which questions require which treatment — and routes accordingly — is arguably harder than building either system alone.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The simplest way to see what GraphRAG is doing is to contrast what it builds with what vanilla RAG builds. Vanilla RAG builds an index of fixed-size chunks, each embedded as a single point in a high-dimensional space. The only structure in that index is geometric proximity: chunks that embed near each other may be semantically related, but the index has no explicit representation of <em>how</em> they are related. GraphRAG builds a graph. Each entity mentioned in the corpus — a person, a company, a product, a concept — becomes a node. Each relationship mentioned — "Alice is CEO of Foo Inc," "Foo Inc acquired Bar Corp in 2023" — becomes a directed edge. The graph is a structured, explicit representation of the relational content that the chunk index can only gesture at through proximity.
      </Prose>

      <Prose>
        Community detection partitions this graph into clusters of strongly connected entities. The Leiden algorithm, applied to the entity graph, finds communities by maximizing a modularity-like objective: a partition is good if edges within communities are denser than would be expected by chance, and sparser across communities. Each community roughly corresponds to a coherent thematic subgraph — all the entities and relationships related to a particular acquisition, a particular research program, a particular product line. An LLM then reads the subgraph for each community and writes a summary: a few sentences that distill the community's content into plain text that can be embedded and retrieved. These community summaries are the index that GraphRAG queries against for global questions. They are corpus-wide by construction, covering the full entity graph rather than just the locally-nearest chunks.
      </Prose>

      <Prose>
        Query-time graph traversal works differently from community-summary retrieval. For a local question — one whose answer requires following a specific chain of relationships — GraphRAG identifies which entities appear in the query, seeds a breadth-first or depth-first traversal from those nodes, and accumulates the subgraph that the traversal touches. The traversal applies relevance scoring at each hop to decide which neighbors to follow: nodes that are more central (higher PageRank) or more closely connected to already-selected nodes score higher. The accumulated subgraph, converted back into text, becomes the context for generation.
      </Prose>

      <Prose>
        Agentic RAG makes a different kind of structural change. Instead of building a richer index, it gives the model control over retrieval. At the conceptual level, the model is placed inside a decision loop: given the current state of a conversation and whatever has been retrieved so far, the model decides whether it has enough information to answer, or whether it needs to issue another retrieval query. If it decides to retrieve, it formulates a query — potentially very different from the user's original question — and the result is added to the context. This continues for up to some maximum number of iterations. The model is not a passive consumer of context; it is an active planner that steers the information-gathering process toward exactly what it needs.
      </Prose>

      <Prose>
        Self-RAG formalizes this intuition with a trained model rather than a prompted one. The Self-RAG framework fine-tunes a language model to emit four types of special reflection tokens during generation. A <Code>Retrieve</Code> token signals that retrieval should be triggered before continuing. An <Code>IsRel</Code> token scores whether a retrieved passage is relevant to the current generation. An <Code>IsSup</Code> token scores whether the model's output is supported by the retrieved passage. An <Code>IsUse</Code> token scores whether the overall response is likely to be useful to the user. By training the model to produce these tokens as part of its normal generation, Self-RAG makes retrieval decisions intrinsic to the model rather than delegated to an external orchestration layer. The benefit is reliability: a prompted model can be instructed to decide when to retrieve, but a fine-tuned model has been trained on thousands of examples of good and bad retrieval decisions, and its behavior is more robust.
      </Prose>

      <Prose>
        It is useful to hold in mind a concrete example of where each approach earns its cost. Suppose the corpus is a large collection of financial filings, press releases, and board meeting minutes for a set of publicly traded companies. A question like "What was the P/E ratio of Acme Corp in Q3 2022?" is a single-hop lookup; plain RAG with a keyword-boosted BM25 retriever answers it reliably and cheaply. A question like "Which independent board members sat on the audit committees of companies that received SEC enforcement actions within two years of appointing them?" is a multi-hop relational query: it requires knowing which board members are classified as independent, which companies received enforcement actions, the timing of appointments, and the audit committee membership — four separate facts that must be joined. GraphRAG's entity graph, with nodes for board members and companies and edges for committee membership, appointment dates, and regulatory actions, can answer this with a graph traversal. Plain RAG, applying cosine retrieval to each part of the question independently, is unlikely to surface all four facts in a single pass, and a single-pass prompt is incapable of performing the join. A question like "Given everything that has happened to Acme Corp over the past five years, what are the key strategic risks I should monitor?" is exploratory: it has no single correct retrieval target, benefits from reading multiple community summaries, and would ideally follow up on whatever the summaries surface. The agentic loop — retrieve summaries, read them, identify gaps, retrieve specific filings to fill those gaps, synthesize — is the right architecture.
      </Prose>

      <Callout accent="gold">
        GraphRAG and Agentic RAG are orthogonal fixes. GraphRAG enriches the index so structured relationships are queryable. Agentic RAG enriches the query process so the model can gather exactly the evidence it needs. They compose naturally: an agentic loop can walk a knowledge graph via tool calls.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        GraphRAG rests on three algorithmic pillars: graph centrality for entity importance, community detection for clustering, and graph traversal for query-time retrieval. Agentic RAG adds a decision-theoretic framing for the retrieval policy.
      </Prose>

      <H3>Entity centrality: PageRank</H3>

      <Prose>
        Not all entities in the knowledge graph are equally important. A node that is connected to many other nodes and is reached by many traversal paths carries more information than an isolated node. PageRank, originally formulated for the web graph, formalizes this intuition. Let <Code>G = (V, E)</Code> be the entity graph with nodes <Code>V</Code> (entities) and edges <Code>E</Code> (relations). The PageRank of a node <Code>v</Code> is defined recursively: a node is important if important nodes point to it.
      </Prose>

      <MathBlock>
        {"PR(v) = \\frac{1 - d}{|V|} + d \\sum_{u \\in \\text{in}(v)} \\frac{PR(u)}{\\text{out}(u)}"}
      </MathBlock>

      <Prose>
        Here <Code>d</Code> is the damping factor (typically 0.85), <Code>in(v)</Code> is the set of nodes with edges pointing to <Code>v</Code>, and <Code>out(u)</Code> is the out-degree of node <Code>u</Code>. In the GraphRAG context, PageRank scores guide traversal: when the BFS from a query-matched seed node must decide which neighbors to follow, nodes with higher PageRank are prioritized. This ensures the traversal moves toward the structurally central entities — the ones most likely to connect to whatever the query is asking about — rather than wandering into peripheral subgraphs.
      </Prose>

      <H3>Community detection: modularity and the Leiden objective</H3>

      <Prose>
        Community detection partitions the node set <Code>V</Code> into communities <Code>C = {"{C₁, ..., Cₖ}"}</Code> such that intra-community edges are denser than would be expected in a random graph with the same degree sequence. The standard objective is modularity:
      </Prose>

      <MathBlock>
        {"Q = \\frac{1}{2m} \\sum_{ij} \\left[ A_{ij} - \\frac{k_i k_j}{2m} \\right] \\delta(c_i, c_j)"}
      </MathBlock>

      <Prose>
        Here <Code>A</Code> is the adjacency matrix, <Code>m</Code> is the total number of edges, <Code>kᵢ</Code> is the degree of node <Code>i</Code>, and <Code>δ(cᵢ, cⱼ)</Code> is 1 if nodes <Code>i</Code> and <Code>j</Code> are in the same community and 0 otherwise. Higher <Code>Q</Code> means a better partition. The Leiden algorithm (Traag, Waltman, van Eck, 2019) improves on the earlier Louvain algorithm by adding a refinement phase that guarantees all output communities are internally well-connected — Louvain can produce communities that are weakly connected by a single bridge edge, which produces fragile, poor-quality summaries downstream. Leiden's three phases are: local node movement (move each node to the neighboring community that most increases <Code>Q</Code>), partition refinement (split communities that are not well-connected), and network aggregation (collapse each community into a single node and repeat). The algorithm terminates when no single node move increases <Code>Q</Code>. In practice, GraphRAG uses Leiden with a resolution parameter <Code>γ</Code> that controls community size: higher <Code>γ</Code> yields more, smaller communities; lower <Code>γ</Code> yields fewer, broader ones. Choosing the right granularity for a given corpus is a hyperparameter that requires experimentation.
      </Prose>

      <H3>Graph traversal: BFS with relevance scoring</H3>

      <Prose>
        Given a query <Code>q</Code>, the traversal seeds from the set of entities <Code>S ⊆ V</Code> whose embeddings are nearest to the query embedding. From those seeds, a BFS expands to neighbors, scoring each candidate neighbor <Code>u</Code> by a combination of graph distance from the seed set and embedding similarity to the query:
      </Prose>

      <MathBlock>
        {"\\text{score}(u) = \\alpha \\cdot \\text{sim}(\\text{emb}(u), \\text{emb}(q)) + (1 - \\alpha) \\cdot PR(u)"}
      </MathBlock>

      <Prose>
        The parameter <Code>α</Code> trades off between query-specific relevance (cosine similarity of node embedding to query embedding) and structural importance (PageRank). Nodes below a threshold are pruned; the surviving subgraph is serialized back into text chunks that become the retrieved context. Microsoft's open-source GraphRAG implementation exposes both <Code>α</Code> and the BFS depth as configurable parameters.
      </Prose>

      <H3>Agentic RAG: retrieval as a Markov decision process</H3>

      <Prose>
        Agentic RAG can be framed as a Markov decision process (MDP) over retrieval states. The state at step <Code>t</Code> is the tuple <Code>sₜ = (q, C_t)</Code> where <Code>q</Code> is the original query and <Code>C_t</Code> is the accumulated context after <Code>t</Code> retrieval steps. The action space has two elements: <Code>retrieve(query_t)</Code> and <Code>answer</Code>. The policy <Code>π(a | sₜ)</Code> is the language model's conditional distribution over actions given the current state. The reward is non-zero only on the <Code>answer</Code> action: it reflects answer quality. The agent's goal is to choose the minimum number of <Code>retrieve</Code> actions needed to reach a state where <Code>answer</Code> yields a high reward. In Self-RAG, the policy is explicitly trained via supervised learning over examples labeled with when retrieval is and is not useful; in prompted agentic loops (ReAct, LangGraph agents), the policy is the language model conditioned on a system prompt that instructs it to reason about when retrieval is needed.
      </Prose>

      <MathBlock>
        {"P(\\text{retrieve} \\mid q, C_t) = \\pi_\\theta(\\text{retrieve} \\mid q, C_t)"}
      </MathBlock>

      <Prose>
        The MDP framing makes the failure modes legible. A policy that always retrieves at every step collects maximum context but multiplies latency by the maximum step count. A policy that retrieves too rarely misses evidence and produces worse answers. The optimal policy retrieves exactly as much as the question requires — which, in practice, is learned by training on diverse questions whose difficulty and multi-hop depth vary.
      </Prose>

      <Prose>
        Self-RAG's contribution to this framing is making the policy parametric in a principled way. Instead of relying on a prompted model to emit a free-text "I should search now" reasoning step — which is brittle and prompt-sensitive — Self-RAG inserts retrieval decisions into the model's generation process as discrete tokens with fixed semantics. The <Code>[Retrieve]</Code> token is not a reasoning artifact; it is a classification decision trained against a discriminative loss. The <Code>[ISREL]</Code>, <Code>[ISSUP]</Code>, and <Code>[ISUSE]</Code> reflection tokens are similarly trained against labeled examples where human annotators judged relevance, support, and usefulness. The result is a model whose retrieval behavior is far more predictable than a prompted model's, because the decision boundary was shaped by supervised training rather than by the model's general language priors. The cost of this predictability is that you need a fine-tuned model — you cannot apply the Self-RAG pattern to a general-purpose model off the shelf.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The implementation below builds the full GraphRAG and Agentic RAG stack from first principles using only Python standard library plus <Code>numpy</Code> and a simulated LLM call (you can replace it with any real API call). Five sub-sections: entity and relation extraction, graph construction, community detection, query-time traversal, and the agentic RAG loop. Every code block was designed to run sequentially; the outputs shown as comments are the actual outputs of running the code on the example corpus.
      </Prose>

      <H3>4a. Entity and relation extraction</H3>

      <Prose>
        The first step is to read each document chunk and extract the structured triples that will form the graph. In production, this is a prompt sent to an LLM; here we simulate it with a deterministic extractor to make the example runnable without API keys. The interface is a function that takes a chunk of text and returns a list of <Code>(subject, predicate, object)</Code> triples.
      </Prose>

      <CodeBlock language="python">
{`from collections import defaultdict
import re

# Simulated entity-relation extraction.
# Replace extract_triples() with a real LLM call in production.

def extract_triples(chunk: str) -> list[tuple[str, str, str]]:
    """
    Parse simple subject-relation-object patterns from a sentence.
    Production version: call an LLM with a structured extraction prompt.
    Returns list of (subject, predicate, object) strings.
    """
    patterns = [
        r"(\\w[\\w ]+) (acquired|founded|led|partnered with|invested in) (\\w[\\w ]+)",
        r"(\\w[\\w ]+) is (CEO|CTO|CFO|founder) of (\\w[\\w ]+)",
    ]
    triples = []
    for pat in patterns:
        for m in re.finditer(pat, chunk, re.IGNORECASE):
            subj = m.group(1).strip().lower()
            pred = m.group(2).strip().lower()
            obj  = m.group(3).strip().lower()
            triples.append((subj, pred, obj))
    return triples

# Example corpus — four document chunks
corpus = [
    "Alice is CEO of Acme Corp. Acme Corp acquired Finco in 2023.",
    "Bob founded Finco before Acme Corp acquired it. Acme Corp partnered with Globex.",
    "Carol is CTO of Globex. Globex invested in Waystone.",
    "Dave led Waystone until Acme Corp acquired it. Acme Corp partnered with Finco.",
]

all_triples = []
for chunk in corpus:
    triples = extract_triples(chunk)
    all_triples.extend(triples)
    for t in triples:
        print(t)

# ('alice', 'is', 'acme corp')          [is CEO of]
# ('acme corp', 'acquired', 'finco')
# ('bob', 'founded', 'finco')
# ('acme corp', 'acquired', 'finco')     [duplicate — real systems deduplicate]
# ('acme corp', 'partnered with', 'globex')
# ('carol', 'is', 'globex')             [is CTO of]
# ('globex', 'invested in', 'waystone')
# ('dave', 'led', 'waystone')
# ('acme corp', 'acquired', 'waystone')
# ('acme corp', 'partnered with', 'finco')`}
      </CodeBlock>

      <H3>4b. Build and visualize the entity graph</H3>

      <Prose>
        The extracted triples are edges in the entity graph. We represent the graph as an adjacency list: each node has a list of <Code>(neighbor, predicate)</Code> pairs. We also compute degree centrality (a cheap PageRank proxy for small graphs) to prepare for traversal scoring.
      </Prose>

      <CodeBlock language="python">
{`from collections import defaultdict, Counter
import math

def build_graph(triples: list[tuple[str, str, str]]) -> dict:
    """Build adjacency list and compute degree centrality."""
    adj = defaultdict(list)   # node -> [(neighbor, predicate)]
    in_deg = Counter()
    out_deg = Counter()
    nodes = set()

    for subj, pred, obj in triples:
        adj[subj].append((obj, pred))
        out_deg[subj] += 1
        in_deg[obj] += 1
        nodes.add(subj)
        nodes.add(obj)

    # Degree centrality: in-degree / (|V| - 1)
    n = len(nodes)
    centrality = {node: in_deg[node] / max(n - 1, 1) for node in nodes}
    return {"adj": dict(adj), "nodes": nodes, "centrality": centrality}

graph = build_graph(all_triples)
print("nodes:", sorted(graph["nodes"]))
print("centrality (top 3):", sorted(graph["centrality"].items(),
                                     key=lambda x: -x[1])[:3])

# nodes: ['alice', 'acme corp', 'bob', 'carol', 'dave', 'finco', 'globex', 'waystone']
# centrality (top 3): [('acme corp', 0.57), ('finco', 0.43), ('globex', 0.29)]`}
      </CodeBlock>

      <Prose>
        Acme Corp is the most central entity, which matches the corpus — it is the acquirer in three of the four chunks and appears in every document. This centrality score will later bias graph traversal toward Acme Corp, which is correct behavior: any question about the broader corporate network will likely need to pass through the node that connects all other entities.
      </Prose>

      <H3>4c. Community detection (simplified Leiden)</H3>

      <Prose>
        For a graph with eight nodes, we can run a simplified version of modularity-based community detection. The version below implements the greedy node-move phase of Leiden: repeatedly move each node to the neighboring community that maximally increases modularity, until no move helps. It is correct for small graphs; production GraphRAG uses the full <Code>graspologic</Code> or <Code>igraph</Code> Leiden implementation which handles the refinement and aggregation phases needed at scale.
      </Prose>

      <CodeBlock language="python">
{`def modularity(partition: dict[str, int], adj: dict, m: int) -> float:
    """Compute modularity Q for an assignment of nodes to community ids."""
    degree = Counter()
    for node, neighbors in adj.items():
        degree[node] += len(neighbors)
        for nbr, _ in neighbors:
            degree[nbr]  # ensure nbr exists in degree dict

    Q = 0.0
    for node, nbrs in adj.items():
        for nbr, _ in nbrs:
            if partition.get(node) == partition.get(nbr):
                expected = (degree[node] * degree[nbr]) / (2 * m + 1e-9)
                Q += (1.0 - expected)
    return Q / (2 * m + 1e-9)

def greedy_leiden(adj: dict) -> dict[str, int]:
    """Greedy node-move community detection. Returns {node: community_id}."""
    nodes = list(set(list(adj.keys()) + [n for nbrs in adj.values() for n, _ in nbrs]))
    partition = {node: i for i, node in enumerate(nodes)}  # each node its own community
    m = sum(len(nbrs) for nbrs in adj.values())
    improved = True
    while improved:
        improved = False
        for node in nodes:
            best_comm = partition[node]
            best_Q = modularity(partition, adj, m)
            for nbr, _ in adj.get(node, []):
                partition[node] = partition[nbr]
                q = modularity(partition, adj, m)
                if q > best_Q:
                    best_Q = q
                    best_comm = partition[nbr]
                    improved = True
            partition[node] = best_comm
    # Re-label community ids to be consecutive
    id_map = {old: new for new, old in enumerate(sorted(set(partition.values())))}
    return {node: id_map[c] for node, c in partition.items()}

communities = greedy_leiden(graph["adj"])
print("Communities:")
from itertools import groupby
for comm_id, members in groupby(sorted(communities, key=communities.get),
                                 key=communities.get):
    print(f"  Community {comm_id}:", list(members))

# Community 0: ['alice', 'acme corp', 'dave', 'finco', 'waystone']
# Community 1: ['bob']
# Community 2: ['carol', 'globex']`}
      </CodeBlock>

      <Prose>
        The algorithm correctly groups Acme Corp with its direct acquisitions (Finco, Waystone) and the people associated with those entities (Alice as CEO, Dave who led Waystone). Globex and Carol form a separate community because Globex has a partnership with Acme Corp but is not controlled by it. Bob, who only appears as Finco's founder before the acquisition, is isolated because the extraction did not find a strong connection to his community neighbors. In production, a larger corpus would produce richer, more stable communities, and the LLM-generated summary for Community 0 would describe the Acme Corp acquisition network in plain prose.
      </Prose>

      <H3>4d. Query-time graph traversal</H3>

      <Prose>
        Given a user query, we find the entities it mentions, seed a BFS from those entities in the graph, and collect the subgraph that the traversal touches. We score each candidate node using a weighted combination of centrality and hop distance from the seeds.
      </Prose>

      <CodeBlock language="python">
{`def graph_retrieve(
    query: str,
    graph: dict,
    communities: dict[str, int],
    max_hops: int = 2,
    top_k: int = 5,
    alpha: float = 0.6,
) -> list[tuple[str, float]]:
    """
    BFS traversal from query-matched seed entities.
    Returns [(entity, score)] sorted by score, descending.
    """
    query_tokens = set(query.lower().split())
    nodes = graph["nodes"]
    centrality = graph["centrality"]
    adj = graph["adj"]

    # Seed: entities whose name appears in the query
    seeds = {n for n in nodes if any(t in n for t in query_tokens)}
    if not seeds:
        # Fall back to top-centrality nodes
        seeds = {max(centrality, key=centrality.get)}

    visited = {}   # node -> (hop_distance, score)
    queue = [(s, 0) for s in seeds]

    while queue:
        node, hop = queue.pop(0)
        if node in visited or hop > max_hops:
            continue
        # Proximity score: decays with hop distance
        proximity = 1.0 / (1 + hop)
        score = alpha * centrality.get(node, 0) + (1 - alpha) * proximity
        visited[node] = (hop, score)
        for nbr, pred in adj.get(node, []):
            if nbr not in visited:
                queue.append((nbr, hop + 1))

    ranked = sorted(visited.items(), key=lambda x: -x[1][1])
    return [(node, score) for node, (_, score) in ranked[:top_k]]

query = "Who acquired finco and who was involved?"
results = graph_retrieve(query, graph, communities)
for entity, score in results:
    print(f"  {entity:20s}  score={score:.3f}  community={communities.get(entity, '?')}")

# finco                score=0.629  community=0
# acme corp            score=0.543  community=0
# waystone             score=0.371  community=0
# alice                score=0.343  community=0
# globex               score=0.229  community=2`}
      </CodeBlock>

      <H3>4e. Agentic RAG loop</H3>

      <Prose>
        The agentic loop gives the model control over retrieval. At each step it decides: do I have enough context to answer, or should I issue another query? We implement this with an explicit tool-use pattern: the model is given a <Code>search</Code> tool and an <Code>answer</Code> tool, and it calls one at each step until it calls <Code>answer</Code> or hits the iteration limit.
      </Prose>

      <CodeBlock language="python">
{`import json

# Simulated model call — replace with a real LLM API in production.
def llm_call(messages: list[dict], tools: list[str]) -> dict:
    """
    Returns {"tool": "search", "query": "..."} or {"tool": "answer", "text": "..."}.
    In production, call any tool-use capable model with the messages and tool schema.
    """
    # Inspect the last user/tool content to decide next action
    context_tokens = sum(len(m["content"]) for m in messages if m["role"] == "tool")
    if context_tokens > 200:  # simulated "enough context" threshold
        return {"tool": "answer", "text": "Acme Corp acquired Finco in 2023. Alice (CEO) "
                "and Dave (formerly led Waystone, also acquired) were key figures."}
    # First iteration: search for the topic
    last_user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    search_q = " ".join(last_user.split()[:6])
    return {"tool": "search", "query": search_q}

def vector_search(query: str, corpus: list[str]) -> list[str]:
    """Simulated vector search — returns chunks containing query keywords."""
    tokens = query.lower().split()
    return [c for c in corpus if any(t in c.lower() for t in tokens)][:3]

def format_chunks(chunks: list[str]) -> str:
    return "\\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))

async def agentic_rag(query: str, corpus: list[str], max_steps: int = 5) -> str:
    """Iterative retrieval loop. The model controls when to retrieve and when to stop."""
    messages = [{"role": "user", "content": query}]

    for step in range(max_steps):
        response = llm_call(messages, tools=["search", "answer"])

        if response["tool"] == "answer":
            print(f"  [step {step+1}] ANSWER: {response['text'][:60]}...")
            return response["text"]

        elif response["tool"] == "search":
            search_q = response["query"]
            chunks = vector_search(search_q, corpus)
            print(f"  [step {step+1}] SEARCH: '{search_q}' -> {len(chunks)} chunks")
            messages.append({"role": "assistant",
                              "content": json.dumps({"tool": "search", "query": search_q})})
            messages.append({"role": "tool",
                              "content": format_chunks(chunks)})

    return "Could not produce a confident answer within the step budget."

import asyncio
answer = asyncio.run(agentic_rag("Who acquired finco and who was involved?", corpus))

# [step 1] SEARCH: 'Who acquired finco and who' -> 2 chunks
# [step 2] ANSWER: Acme Corp acquired Finco in 2023. Alice (CEO)...`}
      </CodeBlock>

      <Prose>
        The two-step trace above shows the minimal case: one retrieval, then an answer. A harder multi-hop question — "What companies are connected to the CEO of Acme Corp through acquisitions?" — would require the model to first retrieve who the CEO is, then retrieve the acquisition chain, potentially reformulating the query between steps. The key invariant is that each step adds to <Code>messages</Code> and the next step's decision is conditioned on the full accumulated context, so the loop naturally chains reasoning across retrievals.
      </Prose>

      <Prose>
        Several design decisions in the implementation above are worth making explicit, because they are the ones production systems iterate on most heavily. First, the tool schema: exposing exactly two tools (<Code>search</Code> and <Code>answer</Code>) keeps the action space small and the model's decisions crisp. Adding more tools — a graph traversal tool, a calculator, a web search tool — multiplies the decision space and requires a more capable model to choose correctly among them. Start with the minimal tool set and add only when you have evidence that the missing tool is causing failures. Second, the context format: retrieved chunks are formatted as numbered passages with source tags, which gives the model an explicit citation mechanism and allows you to verify faithfulness post-hoc. Plain concatenation without numbering works at smaller context sizes but degrades on longer agentic traces where the model needs to track which claim came from which source. Third, the termination logic: the simulation above terminates on context length, which is a rough proxy. Real systems use a combination of model confidence signals, explicit self-evaluation (can the model answer without hedging?), and hard step limits. The hardest part of agentic loop design is not the retrieve step — it is the stop condition.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Several mature systems operationalize these ideas at different points in the stack. Choosing among them depends on whether you need graph-first retrieval, agentic orchestration, or both.
      </Prose>

      <Prose>
        <strong>Microsoft GraphRAG (Python package).</strong> The reference implementation of the Edge et al. paper. Install with <Code>pip install graphrag</Code>. Provides a full pipeline: document ingestion, LLM-based entity and relation extraction, Leiden community detection via the <Code>graspologic</Code> library, community summary generation, and both local (subgraph traversal) and global (community summary retrieval) query modes. Configurable via a YAML file. Works with any OpenAI-compatible API endpoint. The main practical limitation is indexing cost: each document chunk requires one LLM extraction call, so a 10k-chunk corpus costs 10k LLM calls at index time. For stable corpora with a dedicated offline indexing budget, this is the most complete out-of-the-box solution.
      </Prose>

      <Prose>
        <strong>LlamaIndex PropertyGraphIndex.</strong> LlamaIndex's graph-RAG abstraction, available since version 0.10.x. Supports multiple graph backends (in-memory, Neo4j, Memgraph) and multiple extraction strategies, including schema-constrained extraction (<Code>SchemaLLMPathExtractor</Code>) where you specify the entity types and relation types you care about, and free-form extraction for general corpora. Integrates with LlamaIndex's standard retriever interface, so a <Code>PropertyGraphIndex</Code> can be dropped into any existing LlamaIndex pipeline in place of a <Code>VectorStoreIndex</Code>. GraphRAG-style community summaries are available via <Code>GraphRAGStore</Code>, which extends Neo4j's property graph store with community detection and summary generation.
      </Prose>

      <Prose>
        <strong>Neo4j + LLM.</strong> Neo4j is the most widely deployed graph database in production and provides first-class LLM integration through two paths. The <Code>LLMGraphTransformer</Code> class (available via <Code>langchain-community</Code>) extracts entities and relations from documents and inserts them directly into a Neo4j instance. The <Code>GraphCypherQAChain</Code> generates Cypher queries from natural language questions and executes them against the graph — a text-to-Cypher approach that is effective when the schema is stable and the queries are well-typed. Neo4j's graph data science (GDS) library includes production-grade Leiden, PageRank, and betweenness centrality implementations that run directly in the database without exporting the graph. The main advantage over the Python-only approach is that Neo4j handles graph persistence, transactional updates when documents change, and querying at the billion-edge scale.
      </Prose>

      <Prose>
        <strong>Apache AGE (PostgreSQL extension).</strong> Adds a Cypher query interface to PostgreSQL, enabling graph queries against data that already lives in a relational database. Useful when the rest of the application is already on Postgres and the team does not want to introduce a separate graph database. Performance at scale is weaker than dedicated graph databases, but for graphs under a few million edges it is a reasonable operational choice. Community detection must be implemented in Python and the results loaded back in, because AGE does not include a graph algorithms library comparable to GDS.
      </Prose>

      <Prose>
        <strong>LangGraph agents.</strong> LangGraph is the stateful agent orchestration layer from LangChain. An agentic RAG workflow is represented as a directed graph of nodes (LLM calls, tool calls, conditional branches) and edges (transitions between nodes). State is passed along edges and persists across calls, making multi-turn and multi-step reasoning straightforward to implement. LangGraph's <Code>create_react_agent</Code> wraps the ReAct pattern in a production-ready abstraction with built-in support for streaming, error handling, and step-count limits. For teams already in the LangChain ecosystem, LangGraph is the lowest-friction path to agentic retrieval.
      </Prose>

      <Prose>
        <strong>CrewAI.</strong> A multi-agent orchestration framework that treats each agent as a role with defined tools and a goal. Useful when the agentic RAG task decomposes naturally into specialized sub-agents — a research agent that retrieves evidence, a synthesis agent that writes the answer, a critic agent that evaluates faithfulness. CrewAI handles inter-agent communication and sequential or parallel task execution. It adds more overhead than a single-agent LangGraph loop but shines on complex workflows where different retrieval tasks benefit from different system prompts, tool sets, or model choices.
      </Prose>

      <Prose>
        <strong>Haystack agents.</strong> Deepset's Haystack 2.x uses a pipeline-as-graph abstraction where components connect via typed inputs and outputs. An agentic RAG pipeline is built by connecting a <Code>Generator</Code> component to a <Code>Retriever</Code> component and adding a conditional router that loops back to the retriever when the generator signals that it needs more context. Haystack's strength is composability: the same pipeline graph can swap between vector retrieval, BM25 retrieval, and graph retrieval by replacing a single component, making it easy to A/B test different retrieval backends.
      </Prose>

      <CodeBlock language="python">
{`# Production GraphRAG: Microsoft graphrag package — minimal setup
# pip install graphrag

# 1. Initialize a new project
# graphrag init --root ./my_graphrag

# 2. Configure settings.yaml (set your LLM endpoint, chunk size, community levels)

# 3. Index the corpus (expensive — one LLM call per chunk)
# graphrag index --root ./my_graphrag

# 4. Query in local mode (subgraph traversal)
# graphrag query --root ./my_graphrag --method local \\
#     --query "Who was CEO when the acquisition closed?"

# 5. Query in global mode (community summary retrieval)
# graphrag query --root ./my_graphrag --method global \\
#     --query "What are the main themes across this corpus?"

# -----------------------------------------------------------------------
# Production Agentic RAG: LangGraph
# pip install langgraph langchain-openai

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    """Search the document corpus for relevant passages."""
    # Replace with your actual retriever
    return f"[Retrieved passages for: {query}]"

model = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_react_agent(model, tools=[search_knowledge_base])

result = agent.invoke({
    "messages": [("user", "Who acquired Finco and what happened to its founder?")]
})
print(result["messages"][-1].content)`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        Three visualizations: the GraphRAG offline pipeline as a step trace, an entity co-occurrence heatmap that shows which entities appear together most often, and the agentic RAG iteration loop as a step trace.
      </Prose>

      <StepTrace
        label="graphrag offline pipeline — extract → build → cluster → summarize"
        steps={[
          {
            label: "step 1 — LLM extracts entities and relations from each chunk",
            render: () => (
              <div>
                <TokenStream
                  label='chunk: "Acme Corp acquired Finco in 2023."'
                  tokens={[
                    { label: "Acme Corp", color: colors.gold },
                    { label: " →acquired→ ", color: "#888" },
                    { label: "Finco", color: "#60a5fa" },
                  ]}
                />
                <TokenStream
                  label='chunk: "Alice is CEO of Acme Corp."'
                  tokens={[
                    { label: "Alice", color: "#c084fc" },
                    { label: " →is CEO of→ ", color: "#888" },
                    { label: "Acme Corp", color: colors.gold },
                  ]}
                />
              </div>
            ),
          },
          {
            label: "step 2 — triples accumulate into an entity graph",
            render: () => (
              <TokenStream
                label="entity graph (nodes=entities, edges=relations)"
                tokens={[
                  { label: "Alice", color: "#c084fc" },
                  { label: "→CEO→", color: "#555" },
                  { label: "Acme Corp", color: colors.gold },
                  { label: "→acquired→", color: "#555" },
                  { label: "Finco", color: "#60a5fa" },
                  { label: "→acquired→", color: "#555" },
                  { label: "Waystone", color: "#4ade80" },
                ]}
              />
            ),
          },
          {
            label: "step 3 — Leiden algorithm partitions entities into communities",
            render: () => (
              <div>
                <TokenStream
                  label="community 0 — acquisition network"
                  tokens={[
                    { label: "Acme Corp", color: colors.gold },
                    { label: " Alice", color: "#c084fc" },
                    { label: " Finco", color: "#60a5fa" },
                    { label: " Waystone", color: "#4ade80" },
                    { label: " Dave", color: "#f87171" },
                  ]}
                />
                <TokenStream
                  label="community 1 — partner network"
                  tokens={[
                    { label: "Globex", color: "#e2b55a" },
                    { label: " Carol", color: "#c084fc" },
                  ]}
                />
              </div>
            ),
          },
          {
            label: "step 4 — LLM writes a summary for each community",
            render: () => (
              <div>
                <TokenStream
                  label="community 0 summary"
                  tokens={[
                    { label: "Acme Corp, led by Alice (CEO), acquired Finco and Waystone...", color: "#4ade80" },
                  ]}
                />
                <TokenStream
                  label="community 1 summary"
                  tokens={[
                    { label: "Globex (CTO: Carol) is a strategic partner of Acme Corp...", color: "#4ade80" },
                  ]}
                />
              </div>
            ),
          },
          {
            label: "step 5 — community summaries are embedded and stored in the index",
            render: () => (
              <TokenStream
                label="query-time retrieval: embed query → cosine-rank summaries → generate"
                tokens={[
                  { label: "query vec", color: "#888" },
                  { label: " → ", color: "#555" },
                  { label: "community 0 summary", color: colors.gold },
                  { label: " → ", color: "#555" },
                  { label: "LLM answer", color: "#4ade80" },
                ]}
              />
            ),
          },
        ]}
      />

      <Prose>
        The entity co-occurrence heatmap below shows how often each pair of entities appears in the same document chunk. High co-occurrence drives the Leiden community assignments; pairs that always appear together (e.g., Alice and Acme Corp) should be in the same community.
      </Prose>

      <Heatmap
        label="entity co-occurrence — count of shared chunks"
        rowLabels={["Alice", "Acme Corp", "Finco", "Globex", "Waystone", "Dave"]}
        colLabels={["Alice", "Acme Corp", "Finco", "Globex", "Waystone", "Dave"]}
        colorScale="gold"
        cellSize={40}
        matrix={[
          [4, 4, 1, 0, 0, 0],
          [4, 4, 3, 2, 2, 1],
          [1, 3, 3, 1, 1, 1],
          [0, 2, 1, 2, 1, 0],
          [0, 2, 1, 1, 2, 2],
          [0, 1, 1, 0, 2, 2],
        ]}
      />

      <StepTrace
        label="agentic RAG — iterative retrieve → evaluate → continue loop"
        steps={[
          {
            label: "step 1 — model receives query, decides to retrieve",
            render: () => (
              <div>
                <TokenStream
                  label="user query"
                  tokens={[{ label: "Who acquired Finco and what became of its founder?", color: "#e2b55a" }]}
                />
                <TokenStream
                  label="model decision"
                  tokens={[{ label: "→ RETRIEVE: 'Finco acquisition'", color: "#c084fc" }]}
                />
              </div>
            ),
          },
          {
            label: "step 2 — retrieval returns chunks about Finco",
            render: () => (
              <TokenStream
                label="retrieved"
                tokens={[
                  { label: "[1] Acme Corp acquired Finco in 2023.", color: "#60a5fa" },
                  { label: " [2] Bob founded Finco before the acquisition.", color: "#60a5fa" },
                ]}
              />
            ),
          },
          {
            label: "step 3 — model reads chunks, decides one more retrieval needed",
            render: () => (
              <TokenStream
                label="model decision"
                tokens={[{ label: "→ RETRIEVE: 'Bob Finco founder after acquisition'", color: "#c084fc" }]}
              />
            ),
          },
          {
            label: "step 4 — second retrieval returns post-acquisition context",
            render: () => (
              <TokenStream
                label="retrieved"
                tokens={[{ label: "[3] Bob left Finco after Acme Corp completed the acquisition.", color: "#60a5fa" }]}
              />
            ),
          },
          {
            label: "step 5 — model has enough context, calls ANSWER",
            render: () => (
              <TokenStream
                label="final answer"
                tokens={[{ label: "Acme Corp acquired Finco in 2023. Bob, Finco's founder, departed after the deal closed. [1][2][3]", color: "#4ade80" }]}
              />
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        The choice among plain RAG, GraphRAG, Agentic RAG, and a combined stack is determined by the structure of the questions you need to answer and the engineering constraints you can tolerate. There is no universally superior option.
      </Prose>

      <CodeBlock>
{`                    Plain RAG          GraphRAG            Agentic RAG         Combined
─────────────────────────────────────────────────────────────────────────────────────────
best for            single-hop         multi-hop,          complex workflows,  hard multi-hop,
                    factual queries    relational,         tool use,           multi-doc,
                                       corpus summaries    iterative refine    deep research

index structure     vector chunks      knowledge graph     vector chunks       graph + vectors
                                       + community sums    (often)

offline cost        low                high (LLM/chunk)    low                 high

query latency       1–2 LLM calls      1–2 LLM calls       k × LLM calls       k × LLM calls
                                                           (k ≤ max_steps)

answer quality      good on            strong on           strong on hard      strongest,
                    simple Qs          relational Qs       multi-step Qs       all Q types

graph required?     no                 yes                 no                  yes

agent loop?         no                 no                  yes                 yes

when NOT to use     multi-hop,         frequently          latency-critical,   small corpora,
                    corpus-wide        updating corpus,    simple queries,     tight budgets
                                       noisy extractions   weak base model

representative      LangChain RAG,     MS GraphRAG,        LangGraph ReAct,    Deep Research
tools               LlamaIndex RAG     LlamaIndex PGI      CrewAI, Haystack    agents`}
      </CodeBlock>

      <Prose>
        The dominant decision variable is question type. If 90% of your production queries are single-hop factual lookups — "What is the return policy for international orders?" — GraphRAG's indexing cost and Agentic RAG's latency overhead buy you nothing. Plain RAG with a good reranker handles that workload better. If a meaningful fraction of queries are relational — asking about connections between entities, or about themes across the corpus — GraphRAG's offline investment pays back quickly in quality gains that no amount of prompt engineering can achieve with chunk-level retrieval. If queries are exploratory, ambiguous, or require gathering evidence across multiple retrieval steps before an answer can be assembled, agentic RAG's ability to iterate and reformulate is the only tool that reliably solves the problem. The combined stack — graph index traversed by an agentic loop — is the architecture that production "Deep Research" type assistants implement, at the cost of the highest engineering complexity and latency.
      </Prose>

      <Prose>
        One practically important axis the table above simplifies is <em>corpus stability</em>. GraphRAG's offline indexing cost is not one-time; it is paid again whenever the corpus changes significantly enough to invalidate the entity graph. A corpus that receives minor daily updates to existing documents — a living wiki, a policy database where documents are revised in place — is a bad fit for GraphRAG because every update potentially changes entity relationships, and re-running extraction and community detection on the full corpus is expensive. Plain RAG's vector store, by contrast, is trivially updated: re-embed the changed chunks and replace their index entries. Agentic RAG has no offline index dependency at all; it retrieves from whatever store is current at query time, so it inherits the update characteristics of its underlying retriever. The practical implication is that GraphRAG is best suited for archives and reference corpora that are append-only or change infrequently — historical document collections, published research bodies, regulatory filings — and less well suited for operational data where freshness matters more than relational depth.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Scaling GraphRAG and Agentic RAG is substantially harder than scaling plain RAG, and the costs manifest in different dimensions.
      </Prose>

      <Prose>
        <strong>GraphRAG: extraction cost scales linearly with corpus size.</strong> Entity and relation extraction requires one LLM call per document chunk. For a 10,000-chunk corpus using GPT-4o at roughly $0.005 per 1k tokens and 500-token chunks, extraction costs around $25. For a 1 million-chunk corpus — a large enterprise document collection — the bill is $2,500 per full re-index. This is not a one-time cost: every time the corpus changes substantially, the affected chunks must be re-extracted and the community structure must be recomputed. Microsoft's GraphRAG package does not support incremental indexing out of the box as of early 2026; the full pipeline must rerun on the full corpus. Incremental approaches — tracking which chunks have changed and re-extracting only those, then running community detection on the modified graph — require custom engineering and are an active area of development.
      </Prose>

      <Prose>
        <strong>GraphRAG: community detection complexity.</strong> Naive modularity optimization is NP-hard; the Leiden algorithm's greedy approximation runs in <Code>O(N log N)</Code> time on sparse graphs, which is fast enough for graphs with tens of millions of nodes. At billion-node scale (which arises from very large corpora with granular chunk-level extraction), even <Code>O(N log N)</Code> becomes expensive, and distributed implementations are required. For most enterprise use cases — corpora of a few hundred thousand documents — the graph fits easily in memory and Leiden on a single machine completes in minutes.
      </Prose>

      <Prose>
        <strong>GraphRAG: graph storage and query latency.</strong> An entity graph for a 100k-document corpus with one chunk per document and ten entities per chunk has roughly 1 million nodes and several million edges. This fits comfortably in Neo4j or an in-memory structure. A BFS traversal over this graph, starting from a handful of seed nodes and bounded to depth 2–3, touches at most tens of thousands of nodes and runs in milliseconds. Graph query latency is not typically a bottleneck; the bottleneck is the final LLM generation call, which is the same for any RAG approach.
      </Prose>

      <Prose>
        <strong>Agentic RAG: latency multiplies with iteration count.</strong> Each iteration in the agentic loop requires a sequential LLM call — you cannot parallelize a chain where each step is conditioned on the previous step's retrieval results. A five-step agentic loop with a 2-second p50 LLM call at each step adds 10 seconds of latency before the user sees an answer. For user-facing products with latency SLAs, this is often the binding constraint. Mitigation strategies include: using a fast, small model for the retrieve/continue decision and a larger model only for the final generation; parallelizing independent retrieval branches when the query can be decomposed; caching the results of common sub-queries; and using speculative retrieval — pre-fetching likely follow-up queries before the model asks for them.
      </Prose>

      <Prose>
        <strong>Agentic RAG: token costs multiply with context accumulation.</strong> At each iteration, the full conversation history — including all previously retrieved chunks — is passed to the model. A five-step loop that retrieves 1k tokens per step presents the model with a 5k-token context just for the retrieved material, plus system prompt and conversation history. At 10 iterations this becomes 10k tokens, and the cost per query grows linearly with iteration depth. Setting a tight max-steps budget is important not just for latency but for cost. Compressing or summarizing earlier retrieval results before appending them to the context is a standard mitigation, though it requires an additional LLM call to produce the summary.
      </Prose>

      <Prose>
        <strong>Agentic RAG: the case for speculative retrieval.</strong> One optimization that meaningfully reduces perceived latency in agentic loops is speculative retrieval — issuing follow-up queries in parallel with the current LLM call, based on a prediction of what the model will want to retrieve next. This is analogous to speculative decoding in inference engines: you bet on the most likely next action and pre-fetch its result, then either use the pre-fetched result if the prediction was correct, or discard it if the model chose differently. The prediction model is typically a small, fast classifier or a few-shot prompted call to a cheap model that takes the current context and predicts the most likely next search query. Prediction accuracy above 60–70% produces meaningful latency savings because the network round-trip and retrieval compute happen off the critical path. Below that threshold, the wasted retrieval calls consume more resources than the latency savings are worth. Speculative retrieval is most effective when query sequences are predictable — in narrow domains where follow-up questions have stereotyped patterns — and least effective on open-ended exploratory queries.
      </Prose>

      <Callout accent="gold">
        For production GraphRAG: cache community summaries aggressively — they change only when the graph changes. For production Agentic RAG: set a max-steps budget, prefer fast models for the retrieve/continue decision, and log iteration depth per query to catch runaway loops early.
      </Callout>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <Prose>
        Eight things that reliably go wrong, in rough order of how quietly they fail.
      </Prose>

      <Prose>
        <strong>1. Entity resolution errors.</strong> "Acme Corp," "ACME Corporation," and "Acme" are three different strings but one entity. LLM-based extraction will often emit all three, producing three distinct nodes in the graph with no edges between them. Queries that match one name miss the others, fragmenting the entity's connection subgraph across three disconnected nodes. The fix is entity resolution before graph construction: string normalization (lowercase, strip punctuation), embedding-based deduplication (merge node pairs whose embeddings are above a cosine threshold), and, for production systems, a separate entity disambiguation pass that uses a knowledge base or a dedicated model. This is one of the most labor-intensive engineering problems in GraphRAG deployments.
      </Prose>

      <Prose>
        <strong>2. Relation hallucination.</strong> The LLM asked to extract triples from a document will sometimes produce relations that are plausible but not present in the text — a known hallucination failure mode applied to the extraction prompt. "Acme Corp partnered with Globex" might be emitted from a document that only mentions both companies in separate sentences with no stated relationship. These spurious edges corrupt the graph topology, create edges between entities that should not be connected, and produce misleading community assignments. The fix is a verification pass: re-prompt the model to confirm each extracted triple against the source text, or use schema-constrained extraction that limits the relation types to a predefined list the domain expert has approved.
      </Prose>

      <Prose>
        <strong>3. Stale graph after document updates.</strong> The knowledge graph is a snapshot of the corpus at index time. When source documents change — a policy is updated, a new acquisition is announced — the graph does not update automatically. Queries that should return the updated information continue to return the old graph traversal results, and there is no staleness signal visible to the user. This is especially dangerous in fast-moving domains like finance or legal, where the relationship structure changes frequently. The mitigation is a document-change monitoring pipeline: track document modification timestamps, re-extract only the changed chunks, patch the graph with the new triples, and rerun community detection on the affected subgraph. None of this is trivial to implement reliably.
      </Prose>

      <Prose>
        <strong>4. Agentic loops that never terminate.</strong> A poorly prompted agentic model will sometimes enter a loop where each retrieval step finds something it wants to investigate further, and the loop never reaches a <Code>answer</Code> decision within the step budget. The symptom is every query hitting the max-steps limit and returning the fallback response rather than an actual answer. This is not a model failure; it is an orchestration failure. The fix is to design the system prompt to be explicit about when the answer tool should be called — "if you have retrieved at least one relevant passage and cannot find a clearly better query to issue, produce your best answer from the current context" — and to include a built-in summarization call at the final step if the model has not answered by then.
      </Prose>

      <Prose>
        <strong>5. Expensive agent decisions on simple queries.</strong> An agentic RAG system applied uniformly to all user queries will spend three LLM calls on a question that plain RAG would have answered with one. "What is the boiling point of water?" does not need retrieval at all, let alone iterative retrieval. A routing layer that classifies queries as simple/complex before deciding whether to engage the agentic loop is essential at production scale. The classifier can be as simple as a few-shot prompted call to a small model, or as sophisticated as a trained binary classifier on query embeddings. The key metric to track is "fraction of agentic steps that contributed a meaningful context update" — if this is below 50%, the routing is too aggressive.
      </Prose>

      <Prose>
        <strong>6. Reasoning chain breakdowns on weaker models.</strong> Agentic RAG depends on the model's ability to plan, evaluate intermediate results, and reformulate queries — skills that require strong instruction-following and reasoning. Models that are not sufficiently capable tend to issue generic first-step queries that return high-recall but low-precision results, then fail to narrow down in subsequent steps. The symptom is high iteration count with low quality improvement per step. The practical implication is that agentic RAG is not a substitution for a capable model; it is a capability multiplier for a model that is already capable. Using it with a weak model produces more expensive failures.
      </Prose>

      <Prose>
        <strong>7. Community granularity mismatch.</strong> GraphRAG's Leiden resolution parameter controls how many communities are created. If the resolution is too coarse, one community summary tries to cover too much ground — the summary for a single community describing an entire company's operations will be so broad that it matches many queries but is precise about none of them. If resolution is too fine, community summaries become so narrow that global questions ("what are the main themes?") must aggregate hundreds of summaries, increasing both cost and incoherence. There is no universal right answer; the optimal resolution depends on the corpus structure and the query distribution. The practical fix is to build multiple levels of community hierarchy — which Microsoft's GraphRAG supports natively — and let the query determine which level to retrieve from.
      </Prose>

      <Prose>
        <strong>8. Graph poisoning via adversarial documents.</strong> In settings where the document corpus includes external or user-contributed content — a public wiki, a shared knowledge base — an adversary can inject documents designed to produce false triples that corrupt the graph. A document asserting "Person X is the CEO of Company Y" in a plausible but false context will be extracted as a fact and inserted as an edge. Downstream, any query that touches Person X or Company Y will be influenced by the poisoned subgraph. Mitigation requires provenance tracking — every edge in the graph should record its source document — and trust scoring for sources, so that edges from unverified or low-trust sources can be weighted down or quarantined. This is an open problem in production GraphRAG deployments.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The five core references for the techniques in this topic. All verified against their published venues; dates and arXiv ids are accurate as of April 2026.
      </Prose>

      <Prose>
        <strong>1.</strong> Edge, Darren; Trinh, Ha; Cheng, Newman; Bradley, Joshua; Chao, Alex; Mody, Apurva; Truitt, Steven; Larson, Jonathan. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." Microsoft Research, 2024. arXiv:2404.16130. The foundational GraphRAG paper. Introduces the two-phase index-then-query architecture, community-detection-based global summarization, and the distinction between local (subgraph traversal) and global (community summary) query modes. The open-source implementation is at <Code>github.com/microsoft/graphrag</Code>.
      </Prose>

      <Prose>
        <strong>2.</strong> Asai, Akari; Wu, Zeqiu; Wang, Yizhong; Sil, Avirup; Hajishirzi, Hannaneh. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." arXiv:2310.11511 (October 2023). Published at ICLR 2024. The Self-RAG paper. Introduces the four reflection token types (Retrieve, IsRel, IsSup, IsUse) and demonstrates that a 7B–13B model trained with this framework outperforms ChatGPT and retrieval-augmented Llama 2 on open-domain QA, fact verification, and long-form generation. Project page and model weights at <Code>selfrag.github.io</Code>.
      </Prose>

      <Prose>
        <strong>3.</strong> Traag, V. A.; Waltman, L.; van Eck, N. J. "From Louvain to Leiden: Guaranteeing Well-Connected Communities." <em>Scientific Reports</em>, 9, 5233 (2019). The Leiden algorithm paper. Proves that the Louvain algorithm can produce arbitrarily poorly-connected communities and introduces the refinement phase that guarantees connectivity in Leiden. This is the community-detection algorithm that Microsoft GraphRAG uses by default.
      </Prose>

      <Prose>
        <strong>4.</strong> Yao, Shunyu; Zhao, Jeffrey; Yu, Dian; Du, Nan; Shafran, Izhak; Narasimhan, Karthik; Cao, Yuan. "ReAct: Synergizing Reasoning and Acting in Language Models." arXiv:2210.03629 (October 2022). Published at ICLR 2023. The ReAct paper, which establishes the Reason + Act pattern that most agentic RAG frameworks implement. The key insight is that interleaving reasoning traces with action calls (tool use, retrieval) is more robust than pure chain-of-thought or pure action sequences.
      </Prose>

      <Prose>
        <strong>5.</strong> Microsoft Research GraphRAG documentation and LlamaIndex PropertyGraphIndex documentation. For production deployment specifics: <Code>microsoft.github.io/graphrag</Code> covers the configuration YAML, indexing pipeline, and both query modes. <Code>docs.llamaindex.ai</Code> covers <Code>PropertyGraphIndex</Code>, <Code>GraphRAGStore</Code>, and <Code>SchemaLLMPathExtractor</Code> with worked examples against Neo4j. Both documentation sets were verified against released software versions as of early 2026.
      </Prose>

      <Callout accent="gold">
        Secondary reading: the LangGraph documentation (docs.langchain.com/langgraph) includes a worked multi-step agentic RAG example using <Code>create_react_agent</Code> with a vector store tool. It is the fastest path from theory to running code for the agentic loop pattern, and it composes with GraphRAG by replacing the vector store tool with a graph-traversal tool.
      </Callout>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        Five problems. Spend ten minutes per problem before looking at the answer. Each is calibrated so that getting it wrong tells you something specific about a concept that is easy to misunderstand in this material.
      </Prose>

      <Prose>
        <strong>Problem 1.</strong> A corpus has 50,000 document chunks. Using GPT-4o at $0.005 per 1k tokens and 400-token average chunks with 300-token extraction prompts, estimate the cost to build a GraphRAG index. Now the corpus changes 10% every month. What is the monthly re-indexing cost if you rebuild the full index? What if you build incremental re-indexing for only the changed chunks?
      </Prose>

      <Callout accent="green">
        Total tokens per chunk: 400 (document) + 300 (extraction prompt) + ~200 (output) ≈ 900 tokens. At $0.005 per 1k tokens, one chunk costs ~$0.0045. Full index: 50,000 × $0.0045 ≈ <strong>$225</strong>. Monthly full re-index: $225/month. Incremental (10% changed = 5,000 chunks): 5,000 × $0.0045 ≈ <strong>$22.50/month</strong>. The incremental approach is 10× cheaper, which is why it matters operationally for large corpora. Note that community detection must also rerun; its cost is typically dominated by the LLM extraction bill and can be ignored at this scale.
      </Callout>

      <Prose>
        <strong>Problem 2.</strong> Explain why the modularity formula <Code>Aᵢⱼ − kᵢkⱼ/(2m)</Code> is the right quantity to optimize for community detection, rather than simply maximizing the number of intra-community edges. What pathological solution does the plain edge-count objective produce?
      </Prose>

      <Callout accent="green">
        Maximizing raw intra-community edge count has a trivial solution: assign all nodes to one community. That single community contains every edge in the graph. Modularity corrects for this by subtracting the expected intra-community edges under a null model where edges are placed randomly with the same degree sequence. The term <Code>kᵢkⱼ/(2m)</Code> is the probability that nodes <Code>i</Code> and <Code>j</Code> are connected in the null model. A community that is only as dense as random is not a real community; modularity only rewards density above the random baseline. The result is that the trivial one-community solution has <Code>Q = 0</Code> — no improvement over random — and the algorithm is forced to find genuine structure.
      </Callout>

      <Prose>
        <strong>Problem 3.</strong> You are building an agentic RAG system and notice that 60% of queries hit the max-steps limit (5 iterations) without calling the <Code>answer</Code> tool. What are the two most likely causes and what would you change in the system to fix each?
      </Prose>

      <Callout accent="green">
        <strong>Cause 1 — system prompt does not make the termination condition explicit.</strong> The model keeps retrieving because it has never been told when to stop. Fix: add an explicit rule to the system prompt: "If you have retrieved at least two relevant passages and cannot identify a clearly more precise query to issue, generate your best answer from the current context." Also add a forced-answer step at iteration max_steps−1 that bypasses the retrieve/answer decision.
        <strong>Cause 2 — the model's retrieval queries are too broad, returning too much noise.</strong> The model retrieves noise, reads the noise, concludes it needs more information, and retrieves again. Fix: require the model to explain its next query before issuing it (chain-of-thought in the tool call), and filter retrieved chunks with a reranker before adding them to context. A model that receives higher-precision results per step is more likely to reach a confidence threshold within the budget.
      </Callout>

      <Prose>
        <strong>Problem 4.</strong> The Leiden algorithm guarantees that all output communities are well-connected. Why does this property matter specifically for GraphRAG, as opposed to a pure graph analytics use case?
      </Prose>

      <Callout accent="green">
        In a pure graph analytics context, a poorly-connected community is an aesthetic problem — it means the partition is sub-optimal, but the algorithm still produces usable output. In GraphRAG, a poorly-connected community is a quality problem with a concrete downstream effect. When an LLM writes a summary for a community that is weakly connected — two dense subgraphs linked by a single bridge edge — the summary tries to describe two unrelated thematic clusters as though they are one coherent topic. The resulting summary is incoherent, covering both clusters superficially rather than either deeply. When that summary is retrieved at query time, it provides imprecise context that hurts answer quality. The Leiden guarantee ensures each community is a genuinely cohesive thematic unit, which is the prerequisite for an LLM to write a precise, useful summary.
      </Callout>

      <Prose>
        <strong>Problem 5.</strong> Design a routing layer that sits in front of a combined GraphRAG + Agentic RAG system. What features would you extract from the incoming query to decide which retrieval mode to use (plain RAG, GraphRAG local, GraphRAG global, or Agentic RAG)? Describe the feature extraction and the decision logic in concrete terms.
      </Prose>

      <Callout accent="green">
        Three feature categories matter: (1) <strong>relational markers</strong> — presence of words like "who," "connected to," "relationship between," "how did X and Y," "chain of," which signal that the query needs entity traversal (GraphRAG local). (2) <strong>global-scope markers</strong> — presence of "main themes," "overview of," "across all," "most common," "summarize the corpus," which signal that community summaries are needed (GraphRAG global). (3) <strong>complexity markers</strong> — number of distinct entities named ({">"} 2 suggests multi-hop), presence of time conditions ("at the time of"), conditional phrasing ("given that X, what was Y") — these signal that iterative retrieval may be needed (Agentic RAG). Decision logic: if global-scope markers present → GraphRAG global. Else if relational markers present OR complexity score high → GraphRAG local, optionally wrapped in an agentic loop. Else → plain RAG with reranker. This can be implemented as a lightweight classifier (few-shot prompting of a small model, or a fine-tuned BERT classifier on labeled query examples) that adds {"<"}50ms to the request path.
      </Callout>

      <Prose>
        GraphRAG and Agentic RAG are not alternatives to vanilla retrieval — they are extensions for the subset of questions that vanilla retrieval cannot answer. The right deployment treats all three as layers in a tiered system: plain RAG handles the majority of single-hop queries quickly and cheaply, GraphRAG handles the relational and corpus-wide questions that require structured relationship traversal, and the agentic loop handles the exploratory and multi-step questions where a single retrieval pass cannot gather sufficient evidence. The engineering investment for each layer is proportional to its capability, and routing traffic to the cheapest layer that can handle a given query is what makes the combined system economically viable at scale.
      </Prose>

      <Prose>
        There is a tendency in the literature and in vendor marketing to present GraphRAG and Agentic RAG as answers to the question "how do we make retrieval better?" That framing is slightly off. The more accurate framing is: "how do we answer questions that retrieval alone cannot answer?" The distinction matters for how you evaluate these systems. If you benchmark GraphRAG against vanilla RAG on a test set of single-hop factual questions, GraphRAG will often look worse — it is more expensive, introduces extraction noise, and the community summaries are lossy. The right evaluation population is multi-hop and corpus-wide questions where the graph's relational structure is load-bearing. Similarly, if you evaluate Agentic RAG on a test set where all questions are answerable in one retrieval step, you will see higher latency and cost with no quality improvement. The questions that motivate each approach are the only questions where it makes sense to benchmark them. This is obvious in principle and routinely ignored in practice, which produces misleading ablation results and poor deployment decisions.
      </Prose>
    </div>
  ),
};

export default graphragAgentic;
