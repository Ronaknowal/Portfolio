import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { StepTrace, TokenStream } from "../../components/viz";

const prefixCaching = {
  title: "Prefix Caching & Prompt Caching",
  readTime: "10 min",
  content: () => (
    <div>
      <Prose>
        Most production LLM traffic is repetitive in a way that is invisible from the outside. Every chat turn re-sends the same system prompt and the full conversation history. Every agentic loop re-attaches hundreds or thousands of tokens of tool definitions before appending the latest observation. Every few-shot classification pipeline prepends the same eight examples to each new input. These shared prefixes can amount to sixty, seventy, eighty percent of the total token budget on a request — and without any caching in place, the GPU prefills them from scratch on every single call. A deployment that handles ten thousand requests a day may be recomputing the same ten-thousand-token system prompt ten thousand times. The compute is not cheap, the latency is not free, and none of that work produces a result that differs from what was produced the first time.
      </Prose>

      <Prose>
        Prefix caching eliminates that redundancy. Instead of recomputing the KV states for a shared prefix, the serving stack identifies that the prefix has already been processed, retrieves the cached KV blocks from memory, and proceeds directly to the tokens that are actually new. The shared work happens once. Every subsequent request that shares the same prefix pays only for its unique tail. At scale, this transforms the economics of agentic and multi-turn workloads — latency drops because the prefill phase is shorter, cost drops because fewer tokens are billed as fresh, and throughput rises because GPU cycles are freed for genuinely new work.
      </Prose>

      <H2>The two-layer distinction</H2>

      <Prose>
        Two related mechanisms share the "prefix caching" label, and conflating them causes confusion. They rely on the same underlying trick — reuse computed KV cache pages — but they live at different layers of the stack and are activated differently.
      </Prose>

      <Prose>
        <strong>Prefix caching</strong> is a runtime feature of the serving stack. It operates automatically: when two concurrent or sequential requests share a common prefix, the scheduler detects the overlap and points both requests at the same underlying KV blocks. No API change is required on the client side. The system does the detection, the deduplication, and the page-table wiring invisibly. Every request that happens to share a prefix benefits — the client does not need to know the feature exists.
      </Prose>

      <Prose>
        <strong>Prompt caching</strong> is an API-level feature that lets the client explicitly declare which segments of its prompt are stable across requests. The client marks those segments; the server pins their KV pages in memory — or at least elevates their eviction priority — and bills cache hits at a fraction of the base input-token rate. Anthropic introduced <Code>cache_control</Code> markers. OpenAI added both implicit caching (automatic, based on prefix detection) and explicit cache keys. Google Cloud has context caching as a top-level API object with configurable TTLs. The surface varies, but the semantics are consistent: you tell the platform what is worth keeping warm, and the platform rewards you with lower latency and a lower bill.
      </Prose>

      <Prose>
        Both mechanisms depend on PagedAttention being in place. Paging is what makes it physically possible to share KV blocks across requests without copying them — each request gets its own page table, and multiple page tables can point at the same physical block. Without paging, the contiguous per-sequence layout of older serving stacks makes block sharing intractable. Paging is the enabler; prefix caching and prompt caching are the two ways to exploit it.
      </Prose>

      <H2>How prefix caching works mechanically</H2>

      <Prose>
        The core operation is a hash lookup. When a new request arrives, the serving stack walks its token sequence in fixed-size block increments — sixteen tokens is a typical block size. For each block, it computes a hash that covers both the block's own token ids and every token that preceded it. The prefix dependency in the hash is essential: the KV state of a block is a deterministic function of the full token sequence up to that point, not just the block in isolation. Two requests that share the first 256 tokens and then diverge will share the same hash for each of their first sixteen blocks, but produce different hashes the moment their content differs.
      </Prose>

      <Prose>
        When the lookup finds a hash match, the block already lives in GPU memory — its KV rows are already computed and stored in a physical page. The new request's page table is updated to point at that existing page, and a reference count is incremented. No computation happens; the prefill is skipped for that block entirely. When the lookup misses, the serving stack allocates a fresh physical page, runs the prefill kernel to populate it, stores the resulting hash, and sets the reference count to one. Eviction uses a least-recently-used policy on blocks with reference count zero — when memory fills and a new allocation is needed, the oldest unreferenced block is reclaimed.
      </Prose>

      <StepTrace
        label="prefix caching — two requests sharing a system prompt"
        steps={[
          { label: "1. request A arrives with system prompt + user query", render: () => (
            <TokenStream tokens={[
              { label: "[system 0-255]", color: "#c084fc" },
              { label: "[user query A]", color: "#e2b55a" },
            ]} />
          ) },
          { label: "2. A's prefix blocks are hashed, stored", render: () => (
            <TokenStream tokens={[
              { label: "block 0 hash → pg 42", color: "#4ade80" },
              { label: "block 1 hash → pg 43", color: "#4ade80" },
              { label: "...", color: "#555" },
              { label: "(16 system blocks)", color: "#4ade80" },
            ]} />
          ) },
          { label: "3. request B arrives with same system prompt", render: () => (
            <TokenStream tokens={[
              { label: "[system 0-255 ✓ cache hit]", color: "#4ade80" },
              { label: "[user query B]", color: "#e2b55a" },
            ]} />
          ) },
          { label: "4. B reuses pages 42-57; only prefills B's unique tail", render: () => (
            <TokenStream tokens={[
              { label: "B page table: [42, 43, ..., 57, 89]", color: "#4ade80" },
              { label: "— 90% of prefill skipped", color: "#4ade80" },
            ]} />
          ) },
        ]}
      />

      <Prose>
        The simplified implementation below shows the shape of the logic. Real serving stacks wrap this in thread-safe allocation pools, eviction queues, and GPU-aware memory management — but the hash-lookup-then-allocate skeleton is present in all of them.
      </Prose>

      <CodeBlock language="python">
{`import hashlib

class PrefixCache:
    """Simplified: hash-based prefix cache mapping token block hashes to physical pages."""
    def __init__(self):
        self.hash_to_block = {}  # { hash: block_id }
        self.ref_counts = {}     # { block_id: int }

    def _hash_block(self, token_ids_prefix, token_ids_block):
        """Hash depends on the whole prefix, not just this block — identity must match fully."""
        return hashlib.sha1(str(token_ids_prefix + token_ids_block).encode()).hexdigest()

    def lookup_or_allocate(self, token_ids_prefix, new_block_tokens, pool):
        h = self._hash_block(token_ids_prefix, new_block_tokens)
        if h in self.hash_to_block:
            block_id = self.hash_to_block[h]
            self.ref_counts[block_id] += 1
            return block_id, True  # hit
        block_id = pool.allocate()
        # ... write KV for new_block_tokens into block_id ...
        self.hash_to_block[h] = block_id
        self.ref_counts[block_id] = 1
        return block_id, False  # miss`}
      </CodeBlock>

      <H3>What the serving stack actually caches</H3>

      <Prose>
        Different inference engines implement prefix caching with different data structures, and the choice affects the longest prefix they can match. vLLM's automatic prefix caching uses a flat hash map over fixed-size blocks — simple, fast, and effective for cases where the shared prefix is always a prefix of the request (as in system-prompt sharing). SGLang's RadixAttention is more general: it maintains an explicit radix tree where each path from root to leaf represents a sequence of cached token blocks. A new request's prefix is matched against the tree with a longest-prefix-match traversal, finding the deepest ancestor in logarithmic time. This matters for tree-structured workloads — agent frameworks where many branches share long common ancestors, or batch jobs that share prefixes at multiple levels. TensorRT-LLM implements KV cache reuse as a scheduler-level policy, marking blocks as pinnable when they belong to high-frequency prefixes identified ahead of time.
      </Prose>

      <Prose>
        In practice, vLLM's automatic prefix caching handles the majority of production workloads adequately; SGLang's RadixAttention pays off when the sharing structure is non-linear or when requests arrive in an order that a flat map cannot exploit. The key insight is the same in both cases: the block hash is a compact, verifiable identity for a segment of computed KV state, and matching hashes means the computation is safely reusable.
      </Prose>

      <H2>Prompt caching as an API feature</H2>

      <Prose>
        The client-facing version makes the economics explicit. You mark a segment of your prompt as cacheable; the API provider pins that segment's KV pages and bills any subsequent cache hit at a fraction of the base rate. Anthropic's <Code>cache_control</Code> annotation uses an ephemeral TTL (five minutes by default, one hour on extended tier). OpenAI's implicit caching fires automatically when the prefix matches, at no annotation cost, billed at fifty percent of the standard input rate for hits. Google Cloud's context caching is a first-class API resource you create and reference by ID, with a configurable TTL and a per-hour storage fee.
      </Prose>

      <Prose>
        The billing math matters at scale. Claude 3.5 Sonnet lists standard input tokens at roughly three dollars per million. Cache-hit tokens are billed at around thirty cents per million — roughly ten percent. A pipeline that sends a ten-thousand-token system prompt with every request and handles ten thousand requests per day is consuming one hundred million prompt tokens per day in that prefix alone. At base rate that is three hundred dollars per day from the prefix alone. With prompt caching and a high hit rate, it drops to thirty. The annotation is a one-line change; the savings are immediate.
      </Prose>

      <CodeBlock language="python">
{`# Anthropic-style cache_control — explicit, minute-scale TTL.
messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": LONG_SYSTEM_PROMPT},
            {"type": "text", "text": TOOL_DEFINITIONS, "cache_control": {"type": "ephemeral"}},
        ],
    },
    {"role": "user", "content": "Analyze the attached document."},
]

# First request: full prefill cost (~$3/MTok input on Claude 3.5 Sonnet)
# Second request with same system+tools: cache-hit rate on the marked segment,
# billed at ~10% of base rate. 3x price reduction on repetitive traffic.`}
      </CodeBlock>

      <Prose>
        One operational detail: the marked segment must be byte-for-byte identical across requests for the cache to hit. Even a single character difference — a trailing space, a changed timestamp in the prompt, a dynamic field that was not moved outside the cached region — breaks the match and incurs a full prefill for that segment. Designing prompts for cacheability means identifying the stable core and isolating all dynamic content to the non-cached tail. System prompts, tool schemas, and few-shot examples are natural candidates. Per-user personalization fields, current timestamps, and request-specific context are not.
      </Prose>

      <H2>When prefix caching is a big deal vs. not</H2>

      <Prose>
        The wins are large and well-documented in workloads where shared-prefix tokens constitute a significant fraction of total request volume. Agent frameworks are the clearest case: a framework that sends fifty tool definitions before every action step may be paying for five thousand tokens of shared prefix on every request in a multi-step trajectory. RAG pipelines where the same retrieved context is used across a multi-turn conversation share not just the system prompt but the retrieval results. Batch classification jobs with fixed few-shot examples share everything except the example being classified. Document analysis with a stable system prompt and varying document content shares the system prompt across every document in the batch.
      </Prose>

      <Prose>
        The wins are small when requests are genuinely independent and varied. A consumer chatbot where each user's conversation history is unique and the system prompt is short will see low hit rates — the prefix is only a few hundred tokens and the conversational tail dominates. One-shot API calls with fully dynamic content have nothing to share. Diversity of content is the enemy of cache efficiency; sameness is its ally. A rough heuristic: if your P50 request shares more than thirty percent of its token budget with another request processed in the last minute, prefix caching is worth enabling and measuring. For many production agentic pipelines, the shared-prefix fraction sits between sixty and ninety percent, and the throughput and cost improvements are proportional.
      </Prose>

      <H3>TTL and eviction</H3>

      <Prose>
        Cached KV blocks cannot live in GPU memory indefinitely. Memory is finite, new requests constantly need fresh capacity, and a block that has not been referenced in a long time is unlikely to be referenced again soon. Serving stacks evict using LRU on blocks whose reference count has dropped to zero — a block actively used by an in-flight request cannot be evicted, but the moment the last request referencing it completes, it becomes an eviction candidate. When memory pressure rises, the LRU eviction policy walks from the coldest end of the queue until enough pages are reclaimed for the incoming request.
      </Prose>

      <Prose>
        At the API layer, TTLs are explicit. Anthropic's ephemeral tier gives cached prompts a five-minute lifetime after the last access; the extended tier offers one hour. OpenAI's implicit caching has a TTL of a few minutes, undisclosed but observable in latency patterns. Google's context caching TTL is user-configurable, with a default of one hour and a maximum of several days, billed by the hour for storage. These TTLs mean that cache hit rates depend not just on how many requests share a prefix, but on how quickly those requests arrive. A batch job that serializes a thousand requests with the same system prompt will get nearly perfect hit rates; a deployment with sporadic traffic that drops to zero for thirty minutes will see the cache evict and pay full prefill cost on the first request after the lull.
      </Prose>

      <H3>The security consideration</H3>

      <Prose>
        Prefix caching introduces a side channel that is worth understanding even when it is not a realistic threat in your deployment. If the KV cache is shared across tenants — across different users or API keys — then a cache hit leaks information. An attacker who can observe response latency or inspect cache-hit metadata in API responses can infer that a particular prompt segment has been processed recently by someone. If that prompt segment contains confidential content — a proprietary document, a list of passwords, personally identifiable information — the attacker learns that this content exists in the system and was submitted within the cache TTL window, without being able to read the content itself. The attack is a timing side channel rather than a data exfiltration, but it is a real information leak.
      </Prose>

      <Prose>
        Most commercial providers address this by isolating caches per API key or per organization — your cache is only shared with your own subsequent requests, not with other customers. If you operate your own serving stack with vLLM or SGLang, the default configuration typically scopes prefix caching to the request batch, which means concurrent requests from different users on the same server can share blocks. Whether that is acceptable depends on your threat model. For a multi-tenant service handling sensitive data, the right answer is to partition the prefix cache at the same boundary as your other access controls: one cache namespace per tenant, never shared across tenants.
      </Prose>

      <Callout accent="gold">
        Prefix caching is a performance feature that can leak information across tenants if not isolated. Put prefix-cache boundaries where you put your other security boundaries.
      </Callout>

      <Prose>
        Prefix caching is among the highest-leverage deployment-time optimizations available: no model changes, no architectural modifications, and in many cases no code changes beyond enabling a flag or adding a cache annotation. Modern serving stacks activate it automatically; commercial APIs offer it as a billing-level feature. For agentic workloads with large stable context, the throughput improvement is typically two to five times, and the cost reduction on commercial APIs is commensurate. The next topic builds on this foundation to examine how those cost reductions actually compute across the full economics of inference — token rates, batching, hardware amortization, and where the real levers are when you are trying to make a production deployment pencil out.
      </Prose>
    </div>
  ),
};

export default prefixCaching;
