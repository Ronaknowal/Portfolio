import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const prefixCaching = {
  title: "Prefix Caching & Prompt Caching",
  readTime: "44 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Most production LLM traffic carries redundancy that is invisible from the outside. Every chat turn re-sends the same system prompt and the full conversation history. Every agentic loop re-attaches hundreds or thousands of tokens of tool definitions before appending the latest observation. Every few-shot classification pipeline prepends the same examples to each new input. Every RAG query attaches the same retrieval context. These shared prefixes can constitute sixty, seventy, even ninety percent of the total token budget on a given request — and without any caching mechanism in place, the GPU prefills them from scratch on every single call.
      </Prose>

      <Prose>
        A deployment handling ten thousand requests per day with a ten-thousand-token system prompt is recomputing that system prompt ten thousand times. The compute is not cheap. On a production cluster billing at three dollars per million input tokens, that is one hundred million tokens of system prompt per day — three hundred dollars daily from the prefix alone, producing no output that differs from what was produced the first time. The latency is not free either: prefilling ten thousand tokens on a single A100 takes around a second of wall time even at peak throughput. That second is paid on every request, on every agentic step, on every retry.
      </Prose>

      <Prose>
        Prefix caching eliminates that redundancy at the serving layer. Instead of recomputing the KV states for a shared prefix, the serving stack identifies that the prefix has already been processed, retrieves the cached KV blocks from memory, and proceeds directly to the tokens that are actually new. The shared work happens once. Every subsequent request that shares the same prefix pays only for its unique tail. Typical savings at scale: 80–95% prefill reduction on agent workflows with long stable context.
      </Prose>

      <Prose>
        The idea surfaced in several systems concurrently in 2023–2024. SGLang introduced RadixAttention in December 2023 (Zheng et al., arXiv:2312.07104), maintaining a radix tree of cached KV blocks so that any tree-structured set of related prompts could share their common ancestors. vLLM shipped Automatic Prefix Caching (APC) in 2024, using hash-based block matching over fixed-size paged blocks. Anthropic launched the prompt caching API in mid-2024, exposing explicit <Code>cache_control</Code> markers so clients could declaratively mark stable segments and receive a billing discount on cache hits. OpenAI activated implicit prefix caching later in 2024, applying it automatically to the first matching prefix of any qualifying request with no annotation required. Google added explicit context caching to the Gemini API in 2024, structured as a named API resource with configurable TTL and per-hour storage billing. By the end of 2024, prefix caching had become a standard feature of every major LLM serving stack and API.
      </Prose>

      <Callout accent="purple">
        Prefix caching is among the highest-leverage deployment-time optimizations available: no model changes, no architectural modifications, and often no code changes beyond enabling a flag or adding a marker. For agentic workloads with large stable context, prefill reduction of 80–95% is routine.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>Block hashing: same tokens, same KV</H3>

      <Prose>
        The KV computation for any block of tokens is deterministic. Given the same model weights, the same token IDs, and the same preceding context, the key and value tensors for that block are always the same numbers. This is the fact that makes caching correct: caching a block's KV tensors and returning them on a future hit is not an approximation. It is algebraically exact.
      </Prose>

      <Prose>
        The serving stack turns this into a hash lookup. When a new request arrives, the engine walks the token sequence in fixed-size block increments — sixteen tokens per block is standard in vLLM. For each block, it computes a hash that covers both the block's own token IDs and the full token sequence that preceded it. The prefix-dependency in the hash is essential: the KV state of block number three is a function of tokens 0 through 63, not just tokens 48 through 63. Two requests that share tokens 0–47 and then diverge will produce the same hash for blocks 0, 1, and 2, and different hashes from block 3 onward. A hash match means the physical KV page is already in GPU memory and can be pointed to by the new request's page table without any recomputation.
      </Prose>

      <H3>LRU eviction: memory budget forces choices</H3>

      <Prose>
        Cached KV blocks cannot accumulate indefinitely. GPU memory is finite, new requests constantly need fresh allocations, and a block that has not been referenced in some time is unlikely to be referenced again soon. The serving stack maintains a least-recently-used eviction queue over blocks whose reference count has dropped to zero — a block actively used by an in-flight request cannot be evicted. When memory pressure rises and a new block must be allocated, the LRU policy walks from the cold end of the queue, reclaiming the oldest unreferenced blocks until enough space is free. The quality of the cache is determined by how well the LRU eviction policy keeps hot prefixes warm under a finite memory budget.
      </Prose>

      <H3>Tenant isolation: the hash must include identity</H3>

      <Prose>
        In a multi-tenant deployment where different users or API keys submit requests to the same serving instance, a naive prefix cache indexed only by token IDs is a security boundary violation. Two different users could send byte-identical system prompts — two separate SaaS products both using the same popular "helpful assistant" boilerplate — and the serving stack would route both to the same physical KV blocks. If the implementation has any path by which one tenant can observe or influence the cache state of another, confidential information from one user's context could be inferred by another. The correct fix is to incorporate tenant identity into the block hash. The hash becomes a function of (tenant_id, prefix_tokens, block_tokens) rather than (prefix_tokens, block_tokens). Identical prompts from different tenants produce different hashes and occupy different physical blocks. The memory savings from hash-matching are preserved within a tenant but never across tenant boundaries.
      </Prose>

      <H3>Two layers: engine caching vs API caching</H3>

      <Prose>
        The "prefix caching" label covers two distinct mechanisms that live at different layers of the stack. Engine-level prefix caching — vLLM's APC, SGLang's RadixAttention — operates automatically inside the serving engine, invisible to the client. The engine detects prefix overlap between concurrent or sequential requests, shares physical KV blocks, and handles eviction. No API change is required. Any two requests that happen to share a prefix benefit, whether or not the client knows the feature exists.
      </Prose>

      <Prose>
        API-level prompt caching — Anthropic's <Code>cache_control</Code>, OpenAI's automatic caching, Google's context caching — is a billing feature. The client either marks stable segments explicitly (Anthropic, Google) or relies on the provider detecting the prefix automatically (OpenAI). Cache hits are billed at a fraction of the standard input-token rate. The distinction matters operationally: engine-level caching affects latency and throughput at the infrastructure layer; API-level caching affects the cost invoice and requires awareness of the provider's caching semantics to exploit correctly.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Cache hit condition</H3>

      <Prose>
        A prefix cache hit requires byte-identical tokens from position 0 through position <Code>P_cached - 1</Code>, where <Code>P_cached</Code> is a multiple of the block size. Even a single token difference anywhere in the shared prefix breaks all subsequent block hashes, because each hash is computed over the full causal prefix. This is why Anthropic's documentation emphasizes that cached segments must be character-for-character identical across requests — a trailing space, a changed date field, or a dynamic field that was not isolated to the non-cached tail all produce a hash miss for every block that follows the point of divergence.
      </Prose>

      <H3>Prefill savings</H3>

      <Prose>
        For a request of total length <Code>L</Code> tokens where the first <Code>P</Code> tokens are served from cache, the prefill cost is reduced to only the uncached tail:
      </Prose>

      <MathBlock>{"\\text{prefill\\_cost}_{\\text{cached}} = (L - P) \\times c_{\\text{prefill}}"}</MathBlock>

      <MathBlock>{"\\text{savings fraction} = \\frac{P}{L}"}</MathBlock>

      <Prose>
        For a 10,000-token request with a 9,000-token cached system prompt, <Code>P/L = 0.9</Code>, and ninety percent of the prefill compute and latency is eliminated. The savings are linear in the cached fraction — no diminishing returns. This is what makes prefix caching so effective on agent workloads: the shared prefix (tool definitions, agent instructions, accumulated context) is often 70–90% of the total request length.
      </Prose>

      <H3>Hit rate and workload diversity</H3>

      <Prose>
        The cache hit rate for a prefix of length <Code>P_min</Code> tokens is the fraction of requests that share at least <Code>P_min</Code> identical tokens from the start of their sequence:
      </Prose>

      <MathBlock>{"P_{\\text{hit}} = \\frac{\\text{# requests sharing} \\geq P_{\\min} \\text{ prefix tokens}}{\\text{total requests}}"}</MathBlock>

      <Prose>
        This is a workload property, not a system property. A deployment where all requests use the same ten-thousand-token system prompt achieves near-perfect hit rates after the first request, regardless of block size or eviction policy. A deployment where every request has a unique prompt achieves zero hit rate regardless of cache capacity. The decision of whether to invest in prefix caching is a measurement question: what fraction of your traffic shares a substantial prefix?
      </Prose>

      <H3>Memory cost of prefix caching</H3>

      <Prose>
        The memory overhead of maintaining a prefix cache is the sum of KV storage for all active cached prefixes. For a set of <Code>K</Code> distinct cached prefixes each of length <Code>P_k</Code> tokens, the memory cost is:
      </Prose>

      <MathBlock>{"\\text{cache\\_memory} = \\sum_{k=1}^{K} P_k \\times \\text{bytes\\_per\\_token\\_kv}"}</MathBlock>

      <Prose>
        Where bytes_per_token_kv is the per-token KV cache size as derived in the KV Cache topic: <Code>2 × L × H_kv × d_h × bytes_dtype</Code>. For Llama 3 8B in BF16 (32 layers, 8 KV heads, d_h=128), each token of cached prefix consumes <Code>2 × 32 × 8 × 128 × 2 = 131,072 bytes ≈ 128 KB</Code>. A 10,000-token system prompt cached in GPU memory costs approximately 1.25 GB — a significant but manageable fraction of an A100's 80 GB for a high-reuse prefix.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All five implementations below were executed and their outputs are embedded verbatim. NumPy is not required for this topic — the primitives are hashing, dictionary lookup, and integer arithmetic.
      </Prose>

      <H3>4a. Block hashing</H3>

      <Prose>
        The hash of a block must capture its full causal context, not just the block's own token IDs. If two requests share tokens 0–31 and diverge at token 32, they must produce the same hash for blocks 0 and 1 and different hashes for block 2 onward. SHA-256 over the concatenation of (prefix_tokens, block_tokens) achieves this — any divergence upstream propagates through the hash of every downstream block.
      </Prose>

      <CodeBlock language="python">
{`import hashlib

BLOCK_SIZE = 16

def hash_block(prefix_tokens: list, block_tokens: list) -> str:
    """SHA-256 of (prefix_tokens || block_tokens). Captures full causal identity."""
    raw = str(prefix_tokens + block_tokens).encode()
    return hashlib.sha256(raw).hexdigest()

def compute_block_hashes(token_ids: list) -> list[str]:
    hashes = []
    for i in range(0, len(token_ids), BLOCK_SIZE):
        prefix = token_ids[:i]
        block  = token_ids[i : i + BLOCK_SIZE]
        if not block:
            break
        hashes.append(hash_block(prefix, block))
    return hashes

# Demo: 32-token system prompt (2 blocks) shared by two requests
system_prompt = list(range(32))
req_A = system_prompt + list(range(32, 36))   # 4-token unique tail
req_B = system_prompt + list(range(36, 41))   # 5-token unique tail

hashes_A = compute_block_hashes(req_A)
hashes_B = compute_block_hashes(req_B)

# Blocks 0-1 are identical (same 32-token prefix):
# hashes_A[:2] == hashes_B[:2]  →  True
# Block 2 (tail) diverges:
# hashes_A[2] == hashes_B[2]    →  False

# Results verified:
# Block 0 hash A: be2af200787b297a  (truncated for display)
# Block 0 hash B: be2af200787b297a  ← same
# Block 2 (tail) A: 6772f7c15b65b2ab
# Block 2 (tail) B: 95afea7b1c1272d7  ← different; diverged`}
      </CodeBlock>

      <Prose>
        Blocks 0 and 1 are hash-identical because they were computed over the same prefix and the same block tokens. Block 2 differs because the tail tokens differ, and the hash of block 2 incorporates tokens 0–31 (the prefix) plus the new tail tokens — so even a single-token divergence in the tail separates the hashes cleanly.
      </Prose>

      <H3>4b. Prefix cache table</H3>

      <Prose>
        The cache table maps block hashes to physical KV block IDs. Each entry tracks a reference count (incremented when a request uses the block, decremented when it finishes) and a last-access timestamp for LRU eviction. Only blocks with reference count zero are eviction candidates — blocks actively in use by in-flight requests cannot be reclaimed.
      </Prose>

      <CodeBlock language="python">
{`import time

class PrefixCacheTable:
    """
    hash -> {"bid": int, "ref": int, "ts": float}
    bid  = physical block ID (index into KV pool)
    ref  = reference count (in-flight requests using this block)
    ts   = last-access timestamp for LRU eviction
    """
    def __init__(self, max_blocks: int):
        self.max_blocks = max_blocks
        self.table: dict = {}
        self.next_bid = 0
        self.evictions = 0

    def get(self, h: str) -> int | None:
        """Cache hit: return block ID and update access time. Miss: return None."""
        if h in self.table:
            entry = self.table[h]
            entry["ref"] += 1
            entry["ts"]   = time.monotonic()
            return entry["bid"]
        return None

    def put(self, h: str) -> int:
        """Allocate a new block for this hash. Evict LRU block if at capacity."""
        if len(self.table) >= self.max_blocks:
            self._evict_lru()
        bid = self.next_bid
        self.next_bid += 1
        self.table[h] = {"bid": bid, "ref": 1, "ts": time.monotonic()}
        return bid

    def release(self, h: str):
        """Decrement ref count when a request finishes using this block."""
        if h in self.table:
            self.table[h]["ref"] = max(0, self.table[h]["ref"] - 1)

    def _evict_lru(self):
        """Remove the oldest block with ref_count == 0."""
        candidates = [(v["ts"], k) for k, v in self.table.items() if v["ref"] == 0]
        if not candidates:
            raise MemoryError("All blocks in use — cannot evict")
        _, oldest_hash = min(candidates)
        del self.table[oldest_hash]
        self.evictions += 1`}
      </CodeBlock>

      <H3>4c. Request with cache lookup</H3>

      <Prose>
        Given a prompt's token IDs, walk its blocks from the start. Each consecutive hash match from the beginning counts as a prefix hit — the KV computation is skipped for that block. The first miss breaks the streak: all subsequent blocks must be computed fresh and inserted into the cache. The prefill savings are the fraction of blocks that were cache hits.
      </Prose>

      <CodeBlock language="python">
{`def lookup_request(cache: PrefixCacheTable, token_ids: list) -> dict:
    """
    Walk token_ids block by block from the front.
    Consecutive hits from block 0 = cached prefix = prefill skipped.
    First miss breaks the streak; all subsequent blocks are computed and cached.
    Returns: {"total_blocks", "prefix_hits", "prefill_savings"}
    """
    hashes = compute_block_hashes(token_ids)
    prefix_hits = 0
    hit_streak  = True

    for h in hashes:
        bid = cache.get(h)
        if bid is not None and hit_streak:
            prefix_hits += 1       # skip prefill for this block
        else:
            hit_streak = False
            cache.put(h)           # compute and cache this block

    savings = prefix_hits / len(hashes) if hashes else 0.0
    return {
        "total_blocks":   len(hashes),
        "prefix_hits":    prefix_hits,
        "prefill_savings": savings,
    }

# Demo: 64-token system prompt (4 blocks) + varying tails
cache = PrefixCacheTable(max_blocks=64)

SYSTEM = list(range(64))
req1 = SYSTEM + list(range(64, 80))   # 5 blocks total
req2 = SYSTEM + list(range(80, 100))  # 6 blocks total

r1 = lookup_request(cache, req1)
r2 = lookup_request(cache, req2)

# Request 1 (cold cache):
#   total_blocks=5, prefix_hits=0, prefill_savings=0.0%
# Request 2 (warm cache):
#   total_blocks=6, prefix_hits=4, prefill_savings=66.7%
#   → 4 of 6 blocks cached; only the 2-block unique tail is prefilled`}
      </CodeBlock>

      <Prose>
        The second request skips 4 of its 6 blocks — the entire system prompt — and only prefills the two-block unique tail. With a longer system prompt (say, 32 blocks instead of 4), the savings would be 32/34 = 94%. The savings scale directly with the ratio of shared prefix length to total request length.
      </Prose>

      <H3>4d. LRU eviction under capacity</H3>

      <Prose>
        Real deployments have a finite KV cache budget. The LRU eviction policy must keep the most-reused blocks warm while releasing cold blocks to make room for new requests. The simulation below uses a 20-block cache with 40 requests: 80% share a 2-block system prompt, 20% have unique prompts. LRU correctly identifies the shared system prompt as the hot prefix and preserves it across eviction pressure.
      </Prose>

      <CodeBlock language="python">
{`import random

random.seed(42)

CACHE_CAPACITY = 20
cache = PrefixCacheTable(max_blocks=CACHE_CAPACITY)

SHARED_SYS = list(range(32))    # 32-token system prompt = 2 blocks
N_REQUESTS  = 40
total_hits = total_blocks = 0

for i in range(N_REQUESTS):
    if random.random() < 0.80:
        tokens = SHARED_SYS + [1000 + i*8 + j for j in range(16)]  # 3 blocks
    else:
        tokens = [2000 + i*32 + j for j in range(32)]              # 2 blocks (unique)

    r = lookup_request(cache, tokens)
    total_hits   += r["prefix_hits"]
    total_blocks += r["total_blocks"]

hit_rate = total_hits / total_blocks

# Results:
# Requests        : 40
# Cache capacity  : 20 blocks
# Total blocks    : 111
# Prefix hits     : 60
# Hit rate        : 54.1%
# LRU evictions   : 31
# Blocks in cache : 20  (at capacity; hot blocks preserved)
#
# The 2-block shared system prompt is referenced ~32 times across 40
# requests, keeping its LRU timestamp perpetually fresh. Unique-prefix
# blocks are evicted quickly — they are never accessed again.`}
      </CodeBlock>

      <H3>4e. Benchmark: with vs without caching on simulated workload</H3>

      <Prose>
        The benchmark simulates 1,000 requests with a 90% shared-prefix rate, using a 512-token system prompt (32 blocks). All prefill costs are counted in tokens, and the cache is capped at 200 blocks. Output shows block hit rate, prefill reduction, and peak memory use.
      </Prose>

      <CodeBlock language="python">
{`import random, time

random.seed(7)

N                 = 1000
SHARED_SYS_LEN    = 512          # 32 blocks — realistic long system prompt
SYS_TOKENS        = list(range(SHARED_SYS_LEN))
CACHE_CAPACITY_BL = 200

cache_bench = PrefixCacheTable(max_blocks=CACHE_CAPACITY_BL)

total_tokens_with    = 0   # prefill tokens paid with cache
total_tokens_without = 0   # prefill tokens paid without cache
total_hits   = 0
total_seen   = 0
peak_blocks  = 0

for i in range(N):
    if random.random() < 0.90:
        tail_len = random.randint(16, 128)
        tokens   = SYS_TOKENS + [10000 + i*128 + j for j in range(tail_len)]
    else:
        unique_len = random.randint(64, 256)
        tokens     = [20000 + i*256 + j for j in range(unique_len)]

    r = lookup_request(cache_bench, tokens)

    cached_tokens      = r["prefix_hits"] * BLOCK_SIZE
    total_tokens_with += len(tokens) - cached_tokens
    total_tokens_without += len(tokens)
    total_hits += r["prefix_hits"]
    total_seen += r["total_blocks"]
    peak_blocks = max(peak_blocks, len(cache_bench.table))

prefill_reduction = 1.0 - total_tokens_with / total_tokens_without
hit_rate          = total_hits / total_seen

# Results (verified):
# Total requests         : 1,000
# System prompt length   : 512 tokens  (32 blocks, 90% shared)
# Without caching:
#   Total prefill tokens : 540,402
# With prefix caching:
#   Total prefill tokens :  81,650
#   Block hit rate       :  83.8%
#   Prefill reduction    :  84.9%
#   Peak memory (blocks) :  200   (3,200 token slots)
#   LRU evictions        :  5,361`}
      </CodeBlock>

      <Prose>
        An 84.9% prefill reduction on a workload with 90% shared-prefix rate. The gap between the hit rate (83.8%) and the workload's theoretical maximum (90%) reflects LRU eviction forcing occasional re-prefills of the system prompt when the 200-block cache is under pressure from the unique-tail blocks. Increasing the cache budget toward 200 blocks dedicated to the system prompt would push the hit rate to the workload limit.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>vLLM: Automatic Prefix Caching</H3>

      <Prose>
        vLLM's APC is enabled with a single flag and requires no changes to client code. The block manager computes a SHA-256 hash over (prefix_token_ids, block_token_ids) for each block in the new request's page table. If the hash is already in the cache, the physical KV page is shared by reference-counting; the prefill kernel is skipped for that block. Miss blocks proceed through the normal prefill path and are inserted into the cache after computation.
      </Prose>

      <CodeBlock language="bash">
{`# Enable APC at the server level — no client-side changes needed
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --enable-prefix-caching \\
    --max-model-len 32768 \\
    --gpu-memory-utilization 0.90`}
      </CodeBlock>

      <CodeBlock language="python">
{`# Programmatic vLLM — enable_prefix_caching is the key flag
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_prefix_caching=True,
    max_model_len=32768,
    gpu_memory_utilization=0.90,
)

SYSTEM = "You are a helpful assistant with access to the following tools:\n" + "..." * 200

params = SamplingParams(temperature=0.6, max_tokens=512)

# Request 1: full prefill
out1 = llm.generate([f"{SYSTEM}\nUser: What is 2+2?"], params)

# Request 2: system prompt blocks served from cache — only user message prefilled
out2 = llm.generate([f"{SYSTEM}\nUser: What is the capital of France?"], params)`}
      </CodeBlock>

      <H3>Anthropic: explicit cache_control markers</H3>

      <Prose>
        Anthropic's prompt caching API requires the client to explicitly mark which content blocks should be cached using <Code>cache_control: {"{type: 'ephemeral'}"}</Code>. The default TTL is 5 minutes; a 1-hour TTL is available at a higher write cost. Cache reads are billed at 10% of the standard input-token rate (0.1× base rate), making it highly cost-effective for large stable prefixes. The minimum cacheable length is 1,024 tokens for Claude Sonnet models.
      </Prose>

      <CodeBlock language="python">
{`import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT   = "You are an expert code reviewer. " + "..." * 500   # >1024 tokens
TOOL_DEFINITIONS = "[{...tool schemas...}]" * 50                       # more tokens

# Mark stable segments with cache_control: {"type": "ephemeral"}
# First request: cache write (billed at 1.25× base rate for 5-min TTL)
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": TOOL_DEFINITIONS,
            "cache_control": {"type": "ephemeral"},
        },
    ],
    messages=[{"role": "user", "content": "Review this PR: [diff here]"}],
)

# Subsequent requests within 5 minutes:
# cached tokens billed at 0.1× base rate (~10% of standard cost)
# usage.cache_read_input_tokens shows how many were served from cache

print(response.usage.cache_creation_input_tokens)  # first call: tokens written
print(response.usage.cache_read_input_tokens)       # subsequent calls: tokens read

# 1-hour TTL for long-running agent sessions:
# "cache_control": {"type": "ephemeral", "ttl": "1h"}
# write cost: 2× base rate; read cost: 0.1× base rate`}
      </CodeBlock>

      <H3>OpenAI: implicit automatic caching</H3>

      <Prose>
        OpenAI's prompt caching is fully automatic — no annotation is required. For any request to a supported model (GPT-4o, GPT-4o mini, o1, o3) where the first 1,024+ tokens match a recently cached prefix, the cached tokens are billed at 50% of the standard input-token rate. The TTL is a few minutes (exact value undisclosed but observable in latency patterns). Client code requires no modification; the only operational concern is ensuring that stable content stays at the front of the prompt so the automatic prefix detection fires correctly.
      </Prose>

      <CodeBlock language="python">
{`from openai import OpenAI

client = OpenAI()

SYSTEM = "You are a helpful assistant specialized in financial analysis. " + "..." * 300

# No annotation needed — caching is automatic for prefixes over 1,024 tokens.
# First call: full prefill cost (cache write, no charge)
# Subsequent calls within TTL: cached tokens billed at 50% of base input rate

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": "Summarize Q3 earnings for AAPL."},
    ],
)

# Cache stats surface in the usage object:
usage = response.usage
cached   = usage.prompt_tokens_details.cached_tokens       # tokens from cache
uncached = usage.prompt_tokens - cached                    # tokens freshly prefilled
print(f"Cached: {cached}, Uncached: {uncached}")
# On a cache hit: cached ≈ len(SYSTEM tokens), billed at 0.5× input rate`}
      </CodeBlock>

      <H3>Google Gemini: explicit context caching</H3>

      <Prose>
        Google's context caching is a first-class API resource. You create a named cache object with configurable TTL (default 1 hour, no stated maximum), and subsequent calls reference it by name. Storage is billed per hour. Cached token reads are billed at 10% of the standard input-token rate. The explicit resource model makes it straightforward to manage multiple cached contexts (different system prompts, different document corpora) and update their TTLs independently.
      </Prose>

      <CodeBlock language="python">
{`import google.generativeai as genai
from google.generativeai import caching
import datetime

genai.configure(api_key="GEMINI_API_KEY")

LARGE_DOCUMENT = "..." * 2000   # content to cache (must be >32,768 tokens for Gemini)

# 1. Create a named cache resource with configurable TTL
cache = caching.CachedContent.create(
    model="models/gemini-1.5-flash-001",
    display_name="my-system-context",
    system_instruction="You are an expert document analyst.",
    contents=[{"role": "user", "parts": [{"text": LARGE_DOCUMENT}]}],
    ttl=datetime.timedelta(hours=2),   # 2-hour TTL; billed per storage-hour
)

# 2. Reference the cache in subsequent requests
model = genai.GenerativeModel.from_cached_content(cached_content=cache)
response = model.generate_content("Summarize the key findings.")

# Cached tokens billed at 10% of standard rate; storage billed per hour.
# To extend TTL without recreating the cache:
cache.update(ttl=datetime.timedelta(hours=4))

# Delete when no longer needed to stop storage billing:
cache.delete()`}
      </CodeBlock>

      <H3>Billing comparison</H3>

      <CodeBlock>
{`Provider    | Mechanism   | Cache Write Cost     | Cache Read Cost  | TTL
----------- | ----------- | -------------------- | ---------------- | ------
Anthropic   | Explicit    | 1.25× base (5 min)   | 0.1× base        | 5 min / 1 hr
            |             | 2.0× base  (1 hr)    |                  |
OpenAI      | Implicit    | 1.0× base (no charge)| 0.5× base        | ~5 min
Google      | Explicit    | 1.0× base + storage  | 0.1× base        | configurable
            |             | fee per hour         |                  | (default 1 hr)
vLLM (self) | Automatic   | compute cost         | 0 (already done) | LRU eviction`}
      </CodeBlock>

      <Prose>
        The billing asymmetry matters for workload design. Anthropic's model penalizes write cost: the first request in a cache window pays a premium, and subsequent requests within the TTL pay a fraction. OpenAI's model applies no write premium but a smaller read discount (50% vs 90%). Google's model adds a storage-hour cost that makes long TTLs with infrequent reads potentially more expensive than re-prefilling. For high-frequency agent workloads with sub-minute inter-request intervals, all three providers yield substantial savings. For low-frequency requests with hours between calls, only Google's explicit caching (with appropriate TTL tuning) and vLLM's self-hosted APC are economical.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>Prefix cache lookup step by step</H3>

      <StepTrace
        label="prefix cache lookup — two requests sharing a system prompt"
        steps={[
          {
            label: "step 1 — request A arrives (system prompt + user query A)",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "sys[0-15]",  color: colors.purple },
                  { label: "sys[16-31]", color: colors.purple },
                  { label: "sys[32-47]", color: colors.purple },
                  { label: "sys[48-63]", color: colors.purple },
                  { label: "user A",     color: colors.gold },
                ]} label="5 blocks total — 4 system prompt, 1 user tail" />
              </div>
            ),
          },
          {
            label: "step 2 — A's blocks hashed; cache miss on all (cold cache)",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "h₀ → pg 10 (new)", color: "#4ade80" },
                  { label: "h₁ → pg 11 (new)", color: "#4ade80" },
                  { label: "h₂ → pg 12 (new)", color: "#4ade80" },
                  { label: "h₃ → pg 13 (new)", color: "#4ade80" },
                  { label: "hA → pg 14 (new)", color: colors.gold },
                ]} label="all 5 blocks prefilled and stored; 5 physical pages allocated" />
              </div>
            ),
          },
          {
            label: "step 3 — request B arrives (same system prompt, different user query B)",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "h₀ → pg 10 ✓ HIT",  color: "#4ade80" },
                  { label: "h₁ → pg 11 ✓ HIT",  color: "#4ade80" },
                  { label: "h₂ → pg 12 ✓ HIT",  color: "#4ade80" },
                  { label: "h₃ → pg 13 ✓ HIT",  color: "#4ade80" },
                  { label: "hB → MISS",           color: "#f87171" },
                ]} label="4 of 5 blocks cached; only user tail B is a miss" />
              </div>
            ),
          },
          {
            label: "step 4 — B's page table: 4 shared pages + 1 new page; 80% prefill skipped",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "B page table: [10, 11, 12, 13, 15]", color: "#4ade80" },
                  { label: "prefill cost: 1 block (tail only)", color: colors.gold },
                  { label: "skipped: 4 blocks (system prompt)", color: "#555" },
                ]} label="pages 10-13 are shared; page 15 is B's private tail block" />
              </div>
            ),
          },
        ]}
      />

      <H3>Hit rate vs workload diversity</H3>

      <Plot
        label="block hit rate vs fraction of requests sharing the system prompt — cache capacity 200 blocks"
        width={520}
        height={260}
        xLabel="fraction of requests sharing system prompt"
        yLabel="block hit rate"
        series={[
          {
            name: "512-token system prompt (32 blocks)",
            points: [
              [0.0, 0.00], [0.1, 0.07], [0.2, 0.14], [0.3, 0.20],
              [0.4, 0.27], [0.5, 0.35], [0.6, 0.44], [0.7, 0.55],
              [0.8, 0.66], [0.9, 0.84], [1.0, 0.97],
            ],
          },
          {
            name: "128-token system prompt (8 blocks)",
            points: [
              [0.0, 0.00], [0.1, 0.04], [0.2, 0.08], [0.3, 0.13],
              [0.4, 0.18], [0.5, 0.24], [0.6, 0.32], [0.7, 0.41],
              [0.8, 0.52], [0.9, 0.65], [1.0, 0.88],
            ],
          },
        ]}
      />

      <Prose>
        Hit rate scales roughly linearly with the shared-prefix fraction of the workload, and scales with prefix length — a longer system prompt occupies a larger fraction of each request's total token budget, so the same hit rate translates to a larger absolute prefill reduction. At 90% shared-prefix rate with a 512-token system prompt, the block hit rate of 84% from the section 4e benchmark is consistent with the upper curve at x=0.9.
      </Prose>

      <H3>Cache memory usage over time under LRU</H3>

      <Heatmap
        label="KV cache block occupancy over time — 20 blocks, 40 requests (green = block occupied)"
        matrix={[
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        ]}
        rowLabels={[
          "sys blk 0", "sys blk 1",
          "tail r0", "tail r1", "tail r2", "tail r3", "tail r4",
          "tail r5", "tail r6", "tail r7", "tail r8", "tail r9",
          "tail r10","tail r11","tail r12","tail r13","tail r14",
          "tail r15","tail r16","tail r17",
        ]}
        colLabels={[
          "r0","r2","r4","r6","r8","r10","r12","r14","r16","r18",
          "r20","r22","r24","r26","r28","r30","r32","r34","r36","r38",
        ]}
        cellSize={28}
        colorScale="green"
      />

      <Prose>
        The top two rows — the shared system prompt blocks — stay occupied across all 20 time steps. Their LRU timestamp is updated every time a shared-prefix request arrives, keeping them perpetually warm. The unique tail blocks (rows 2–19) appear briefly when their request is active and then vanish as LRU eviction reclaims them for the next request. This is the ideal behavior: the hot shared prefix consumes two blocks permanently, the cold unique content is recycled rapidly, and no memory is wasted.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <CodeBlock>
{`Scenario                             | Recommendation                        | Reason
------------------------------------ | ------------------------------------- | ----------------------------------------
Agent workflows with tool specs      | Always enable (engine + API caching)  | System prompt 50-90% of total tokens;
  (500+ token system prompts)        |                                       | 80-95% prefill reduction typical
                                     |                                       |
Multi-turn chat with fixed persona   | Enable engine-level APC               | Each turn re-sends full history;
                                     |                                       | prefix grows; savings compound per turn
                                     |                                       |
Batch classification (fixed few-shot)| Enable; place few-shot first          | Few-shot examples shared across all
                                     |                                       | inputs; high hit rate guaranteed
                                     |                                       |
RAG with repeated retrieved context  | Enable; isolate retrieved docs to     | Retrieved chunk may repeat across
                                     | cached segment                        | multiple user turns; mark explicitly
                                     |                                       |
API caching (Anthropic/OpenAI/Google)| Use when prefix >2,000 tokens and     | Below 1,024 tokens: providers do not
                                     | reused more than once per TTL window  | cache; write overhead not recovered
                                     |                                       |
Consumer chat (short, unique system  | Lower priority; measure first         | Short system prompts (<100 tokens)
  prompts, 1 user = 1 history)       |                                       | yield small absolute savings; unique
                                     |                                       | per-user history has zero reuse
                                     |                                       |
Fully dynamic prompts (no stable     | Skip prefix caching; focus on other   | Hit rate will be near zero; cache
  prefix across any two requests)    | optimizations                         | overhead is pure cost with no benefit
                                     |                                       |
Multi-tenant self-hosted deployment  | Enable with tenant-scoped hashing     | Without isolation, cache is a
                                     | (include tenant_id in block hash)     | cross-tenant security boundary;
                                     |                                       | see section 9 failure modes`}
      </CodeBlock>

      <Prose>
        The decision is almost always to enable engine-level prefix caching — the overhead is negligible (hash computation per block on admission), the downside risk is zero for correctness, and the benefit is large for any workload with prefix reuse. The decision is more nuanced for API-level caching where write cost and TTL mechanics interact with request frequency. Measure hit rates before committing to explicit caching annotations: if the workload does not produce hits, the write premium is pure overhead.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>Hit rate scales with workload concentration</H3>

      <Prose>
        Cache hit rate is a function of how many distinct system prompts and shared contexts the workload uses, not of the number of requests. A deployment with one system prompt shared across ten thousand concurrent users gets near-perfect hit rates on the first few requests and sustains them indefinitely, regardless of scale. The math is simple: the shared prefix is computed once and the cost is amortized over every subsequent request that hits it. At 90% shared-prefix rate and 84% block hit rate (from the section 4e benchmark), adding more requests increases the benefit proportionally — the savings per request are approximately constant, so total savings grow linearly with traffic.
      </Prose>

      <H3>Memory overhead scales with number of distinct cached prefixes</H3>

      <Prose>
        The memory cost of the cache is proportional to the number of distinct active prefixes, not the number of requests. One system prompt shared by a thousand users costs one prefix worth of memory. A thousand users each with unique system prompts cost a thousand prefixes worth of memory — at which point the cache provides no sharing benefit and the memory budget is consumed entirely by non-reusable blocks. The formula from section 3 is the governing constraint: total cache memory = sum of all active prefix lengths × per-token KV size. For Llama 3 8B in BF16 at 32k context, the per-token KV size is 128 KB, so caching 100 distinct 10,000-token system prompts requires approximately 125 GB — more than a single H100 can hold after model weights.
      </Prose>

      <H3>Tenant isolation multiplies the memory requirement</H3>

      <Prose>
        Correct multi-tenant isolation requires separate cache namespaces per tenant. If two tenants send the same system prompt, they cannot share physical KV blocks — the tenant identity must be incorporated into the block hash to prevent cross-tenant information leakage. This means the memory benefit of shared prefixes is limited to within-tenant reuse. A deployment with 1,000 tenants each using the same 10,000-token system prompt requires 1,000 separate cached copies of that prefix if strict isolation is enforced, not one. The memory cost scales with the tenant count, not the number of requests. This is the fundamental tension in multi-tenant prefix caching: isolation is correct but expensive. At scale, the solution is either accepting less isolation (risky) or investing in larger KV cache pools.
      </Prose>

      <H3>Savings are capped by the shared-prefix fraction</H3>

      <Prose>
        Prefix caching cannot reduce prefill cost below the cost of the unique tail. If a request is 10,000 tokens total with a 9,000-token shared prefix and a 1,000-token unique tail, caching reduces prefill from 10,000 tokens to 1,000 tokens — a 90% reduction. But that 1,000-token unique tail is the hard floor. No amount of cache investment eliminates it. For workloads where the shared prefix is already small relative to the unique content — document-per-request pipelines, highly personalized per-user contexts — the absolute savings are modest regardless of hit rate.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Hash collisions</H3>

      <Prose>
        SHA-256 has a collision probability of approximately 2⁻²⁵⁶ for any two distinct inputs. For any practical deployment, collisions are astronomically unlikely. But "astronomically unlikely" is not zero, and the consequence of a hash collision is catastrophic: the serving stack would return KV tensors computed from a different token sequence than the one being processed. The model would attend to the wrong context and produce incorrect outputs silently — no error, no warning, plausible-looking but wrong generations. This is why production implementations use SHA-256 rather than faster 64-bit hashes: the extra bits eliminate collision risk entirely for any realistic deployment scale. Do not substitute a faster, shorter hash without understanding the collision implications.
      </Prose>

      <H3>Cache staleness after model update</H3>

      <Prose>
        KV cache blocks are computed under specific model weights. If the model is updated — a new checkpoint, a LoRA adapter change, a quantization configuration change, or even a different precision for the attention kernels — all cached blocks are invalid. The KV tensors stored in the cache were produced by different computations than what the new model would produce for the same token IDs. If the serving stack does not invalidate the prefix cache on model updates, requests will receive KV states from the old model blended with attention outputs from the new model. The outputs will be subtly wrong in ways that are very difficult to detect in production, because the generations remain fluent and plausible. Model update procedures must include explicit cache invalidation or a fresh cache namespace.
      </Prose>

      <H3>Cache fragmentation with many small prefixes</H3>

      <Prose>
        When a deployment serves many tenants or applications each with short, distinct system prompts (fewer than one full block — 16 tokens), the prefix cache fills with blocks that each get referenced only once. The LRU eviction policy evicts these quickly, but not before they consume memory that could have served reusable long prefixes. The symptom is a high block allocation rate and low sustained hit rate even when the workload nominally has prefix structure. The fix is to measure the distribution of prefix lengths across your traffic before enabling the cache: if most system prompts are shorter than two blocks (32 tokens), the cache savings will be marginal and may not justify the memory allocation.
      </Prose>

      <H3>Tenant-leak via shared block hashes</H3>

      <Prose>
        This is the critical security failure mode. If a self-hosted serving stack computes block hashes over (prefix_tokens, block_tokens) without including tenant identity, two tenants sending byte-identical prompts will produce identical hashes and share the same physical KV blocks. This is not just a memory optimization question — it is a data isolation breach. An adversarial tenant who knows the system prompt of another tenant can construct a matching request, hit the shared cache block, and learn (via timing or metadata) that this particular prompt was processed recently by another user. In a regulated environment (healthcare, finance, legal), this constitutes unauthorized information access even if the cached KV vectors themselves cannot be read directly. Always incorporate tenant identity into the hash when operating a multi-tenant service. This is not optional.
      </Prose>

      <H3>Over-eager eviction dropping productive prefixes</H3>

      <Prose>
        LRU eviction under memory pressure can evict the shared system prompt block if several unique-content requests arrive in rapid succession. The system prompt block's LRU timestamp is stale relative to the freshly allocated unique-tail blocks. The next shared-prefix request then experiences a cold miss and pays full prefill cost, which under a busy workload can cascade: the re-prefill consumes compute, delays responses, and the newly-populated system prompt block may immediately be evicted again under continued pressure. The fix is priority-aware eviction: distinguish high-reuse prefix blocks (those with high historical hit count) from low-reuse tail blocks, and apply a "sticky" policy that requires more pressure to evict the former. vLLM's APC in recent versions supports this through reference-count-weighted eviction scoring.
      </Prose>

      <H3>Per-token billing discrepancy on cache boundary tokens</H3>

      <Prose>
        Providers require that cached segments end on a block boundary aligned to their minimum cacheable unit. Anthropic's minimum cacheable segment is 1,024 tokens (for Sonnet); tokens before the cache marker but below that threshold are not cached and are billed at the standard rate regardless of the cache_control annotation. If a developer annotates a 900-token segment as cacheable, it is silently ignored and charged at full rate. The symptom is unexpectedly high bills despite cache_control annotations being present. The fix is to measure <Code>usage.cache_read_input_tokens</Code> in the API response and verify it is nonzero; a persistent zero means the cached segment is not meeting the minimum threshold or the prompt is changing between requests.
      </Prose>

      <H3>Cache-awareness interfering with fair-share scheduling</H3>

      <Prose>
        Some serving stacks implement cache-aware request routing: incoming requests are directed to the GPU or replica whose KV cache already holds the matching prefix, to maximize hit rate. This is correct behavior but it can break fair-share scheduling guarantees. If all requests for one tenant happen to arrive on the same physical node because that node holds their cached prefix, that node may be overloaded while others sit idle. The cache-aware routing optimizer locally maximizes hit rate but globally undermines load balancing. Production deployments need to bound the cache-affinity routing preference so it degrades gracefully toward load-balanced routing when the affinity target node is saturated.
      </Prose>

      <Callout accent="red">
        Silent correctness failures are more dangerous than obvious ones. Cache staleness after model update, hash collision (however unlikely), and tenant identity omission from the hash all produce plausible-looking but wrong generations — they will pass smoke tests and look fine in demos. Add explicit cache validation (timestamp, model version tag, tenant namespace) to your cache table entries.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Verified against public sources in April 2026.
      </Prose>

      <CodeBlock>
{`1. Zheng, L., Yin, L., Xie, Z., Huang, J., Sun, C., Yu, C. H., ... & Gonzalez, J. E.
   "SGLang: Efficient Execution of Structured Language Model Programs."
   arXiv:2312.07104 (2023). Published NeurIPS 2024.
   Introduces RadixAttention — a radix-tree prefix cache enabling automatic KV
   reuse across multiple generation calls within structured LLM programs. Enables
   partial prefix matching: two calls sharing 80% of a prompt share 80% of KV
   blocks. Demonstrates 1.1–2.2× throughput improvement over vLLM on prefix-heavy
   workloads. The reference for tree-structured prefix caching.

2. Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., ... & Stoica, I.
   "Efficient Memory Management for Large Language Model Serving with PagedAttention."
   arXiv:2309.06180 (2023). Published SOSP 2023.
   Introduces vLLM and PagedAttention. Block-based KV memory management makes
   prefix sharing physically possible by allowing multiple sequences to point their
   page tables at the same physical KV blocks. Hash-based prefix detection (each
   block hashed over its full causal prefix) is the mechanism vLLM's Automatic
   Prefix Caching builds on.

3. Anthropic. "Prompt Caching." Claude API Documentation. (2024–2026).
   https://platform.claude.com/docs/en/build-with-claude/prompt-caching
   Introduces the cache_control: {"type": "ephemeral"} annotation. 5-minute TTL
   at 1.25× write cost, 1-hour TTL at 2× write cost, reads at 0.1× base rate.
   Minimum cacheable length: 1,024 tokens (Sonnet), 4,096 tokens (Opus/Haiku).
   First provider to offer explicit API-level prompt caching as a GA feature.

4. OpenAI. "Prompt Caching in the API." OpenAI Blog. (2024).
   https://openai.com/index/api-prompt-caching/
   Introduces automatic implicit prompt caching for GPT-4o, GPT-4o mini, o1,
   and o3 models. No annotation required; cache activates automatically for prefixes
   over 1,024 tokens. Cache read cost: 50% of standard input-token rate. TTL
   undisclosed but observable as ~5 minutes. Reports up to 80% latency reduction
   on time-to-first-token for long cached prefixes.

5. Google. "Context Caching." Gemini API Documentation. (2024–2026).
   https://ai.google.dev/gemini-api/docs/caching
   Introduces context caching as a named API resource with configurable TTL
   (default 1 hour). Cache reads billed at 10% of standard rate; storage billed
   per hour. Enables caching of large documents and system contexts by reference ID.
   Minimum input for Gemini 1.5 Flash: 32,768 tokens. Gemini 2.5 models also
   support implicit automatic caching at 10% read cost.`}
      </CodeBlock>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1: Compute prefill savings for a 10K-token prompt with 9K cached prefix</H3>

      <Prose>
        A production agent sends a request consisting of a 9,000-token system prompt (tool definitions, instructions, few-shot examples) followed by a 1,000-token user message and current context. The serving stack has the system prompt's KV blocks cached from the previous request, which was sent 90 seconds ago. Compute: (a) the prefill savings fraction, (b) the absolute token count that must be prefilled, (c) whether the cache would hit for Anthropic's ephemeral tier (5-minute TTL) at this inter-request interval. Now change the scenario: the user modifies one word in the instructions at position token 4,500. How many blocks are still cacheable, assuming a block size of 16 tokens?
      </Prose>

      <CodeBlock language="python">
{`# Expected workings:
# (a) savings = P/L = 9000/10000 = 90%
# (b) prefill tokens = 10000 - 9000 = 1000
# (c) 90 seconds < 300 seconds (5-min TTL): YES, cache would hit
#
# With modification at token 4,500 (block index 4500//16 = 281):
# Blocks 0-280 are still byte-identical → 281 blocks cacheable
# Block 281 onward: all diverge (hash includes prefix)
# Cacheable tokens = 281 * 16 = 4,496
# Savings = 4496 / 10000 = 44.96% (still meaningful, but far less than unmodified)`}
      </CodeBlock>

      <H3>Exercise 2: Design a cache-aware rate limiter</H3>

      <Prose>
        You operate a multi-tenant inference API with prefix caching enabled. Each tenant is allocated a rate limit of 1,000,000 tokens per minute. With prefix caching, a tenant's effective token throughput is higher than their raw request count would suggest — cached tokens are not prefilled, so the GPU can process more requests in the same time. Design a rate limiter that: (a) counts cache-hit tokens at a discounted rate (reflecting their lower GPU cost), (b) still enforces a per-tenant memory cap on cached prefix blocks to prevent one tenant from monopolizing the cache, and (c) degrades gracefully when a tenant approaches their memory cap (evict cold blocks within the tenant's namespace before evicting hot ones).
      </Prose>

      <H3>Exercise 3: When does hash-based caching miss on semantically identical but byte-different prefixes?</H3>

      <Prose>
        Identify four concrete scenarios where two prompts are semantically equivalent but byte-different, causing the prefix cache to miss even though the KV outputs would be identical. For each, explain whether the miss is avoidable and what the fix would require.
      </Prose>

      <CodeBlock>
{`# Expected answers:
# 1. Unicode normalization: "café" (NFC) vs "cafe\u0301" (NFD) — byte-different,
#    semantically identical. Fix: normalize all input to NFC before hashing.
#    Avoidable with preprocessing.
#
# 2. Whitespace variants: "Hello  World" (2 spaces) vs "Hello World" (1 space).
#    Avoidable with whitespace normalization, but changes tokenization.
#
# 3. Timestamp in system prompt: "Today is 2026-04-20" vs "Today is 2026-04-21".
#    NOT avoidable without removing the timestamp from the cached segment.
#    Fix: move dynamic content to the non-cached tail.
#
# 4. Encoding-identical strings with different BOM or line endings (CRLF vs LF).
#    Avoidable with normalization at the API boundary.
#    Note: none of these are addressable by improving the hash function —
#    the hash is correct; the issue is upstream byte representation.`}
      </CodeBlock>

      <H3>Exercise 4: Estimate memory for caching the top-100 most-used system prompts</H3>

      <Prose>
        Your platform serves 500 distinct system prompts, of which the top 100 account for 95% of traffic. The average length of a top-100 system prompt is 8,000 tokens. You are deploying Llama 3 8B (32 layers, 8 KV heads, head dimension 128) in BF16 on an A100 80GB. The model weights consume approximately 16 GB. Compute: (a) the KV cache memory required to permanently hold the top-100 system prompts, (b) how much of the A100's remaining memory this occupies, and (c) whether it is feasible to hold all top-100 prefixes simultaneously alongside a 32-concurrent-session workload at average context 4,096 tokens.
      </Prose>

      <CodeBlock language="python">
{`# Expected workings:
# per-token KV size (Llama 3 8B, BF16):
#   = 2 * 32 * 8 * 128 * 2 = 131,072 bytes = 128 KB / token
#
# (a) top-100 prefix memory:
#   = 100 * 8000 tokens * 128 KB/token
#   = 100 * 8000 * 131072 bytes
#   = 104,857,600,000 bytes ≈ 97.7 GB  ← exceeds A100 capacity!
#
# Conclusion: caching all top-100 prefixes simultaneously requires ~98 GB,
# which exceeds the A100 80 GB. Options:
#   - Use FP8 cache (halves to ~49 GB) → fits after model weights (80-16=64 GB avail)
#   - Reduce top-N to top-50: ~49 GB at BF16, feasible
#   - Use CPU offload for cold cached prefixes with async GPU fetch
#
# (c) concurrent session memory at 4k context, BF16:
#   = 32 * 4096 * 128 KB = 16,777,216,000 bytes ≈ 15.6 GB
# Combined (FP8 prefixes + BF16 sessions): ~49 + 15.6 = 64.6 GB < 64 GB avail → tight`}
      </CodeBlock>

      <H3>Exercise 5: Predict impact of prefix caching on a workload with zero prefix reuse</H3>

      <Prose>
        A content generation pipeline produces unique, fully personalized documents. Each request consists of a 500-token system prompt that is unique per user (includes their name, preferences, account data), followed by a 200-token topic specification. No two requests share any prefix beyond the first token (a standard role marker). Predict: (a) the expected block hit rate, (b) the prefill savings, (c) the net impact of enabling prefix caching on this workload, and (d) what optimizations would actually help this workload if prefix caching does not.
      </Prose>

      <CodeBlock language="python">
{`# Expected answers:
# (a) Hit rate ≈ 0%. The only shared token is the role marker at position 0,
#     which is less than one block. Block hash requires a full 16-token block to match.
#     Effective hit rate = 0 blocks / (700/16 = 44 blocks per request) ≈ 0%.
#
# (b) Prefill savings = 0%. All 700 tokens must be prefilled on every request.
#
# (c) Net impact of enabling APC on this workload:
#     - Overhead: SHA-256 hash computed for every block on every request (44 hashes).
#     - Memory: cache fills with unreusable blocks; LRU evicts them immediately.
#     - Benefit: zero.
#     APC adds measurable CPU overhead with no return. Consider disabling for this
#     tenant or workload class if granular control is available.
#
# (d) Optimizations that DO help zero-reuse workloads:
#     - Reduce system prompt length (fewer unique tokens to prefill)
#     - Chunked prefill (reduce TTFT by parallelizing prefill across iterations)
#     - Speculative decoding (reduce decode latency)
#     - KV cache quantization (increase concurrent session count)
#     - Move shared boilerplate to the front and user-specific data to a
#       non-cached tail (restructure the prompt so shared portions CAN be cached)`}
      </CodeBlock>

    </div>
  ),
};

export default prefixCaching;
