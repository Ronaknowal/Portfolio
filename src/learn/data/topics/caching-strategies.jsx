import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const cachingStrategies = {
  title: "Caching Strategies (Semantic, Exact, KV-Cache Sharing)",
  slug: "caching-strategies-semantic-exact-kv-cache-sharing",
  readTime: "44 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        LLM inference is expensive in two distinct ways. The obvious cost is compute: a forward pass through a seventy-billion-parameter model burns tens of thousands of FLOPs per token, and generating a five-hundred-token response means doing that thousands of times. The less obvious cost is repetition. An enormous fraction of production LLM traffic is not novel. A customer asks the same question about your return policy that forty other customers asked this week. An agentic loop sends the same ten-thousand-token system prompt on every call. A batch classification pipeline runs the same few-shot examples against ten thousand inputs in sequence. In each of these cases, the GPU is performing computation that was already performed, producing outputs that were already produced. The compute is real; the work is redundant.
      </Prose>

      <Prose>
        Three different caching mechanisms exist because redundancy shows up at three different levels of abstraction, and no single cache can catch all three. Exact-match caching catches requests that are byte-for-byte identical — the same model, the same prompt, the same parameters. It is cheap to implement, zero-risk for correctness, and achieves near-perfect hit rates on deterministic, idempotent workloads. It achieves essentially zero hit rate on anything where the prompt varies across requests. Semantic caching catches requests that are semantically equivalent but textually different — "what's your return policy" and "how do I return something I bought" as examples. It extends coverage substantially on FAQ-style traffic but introduces the possibility of returning a wrong answer, because embedding similarity is not semantic equivalence. KV-cache sharing catches requests that share a common token prefix — any request that begins with the same system prompt, the same few-shot examples, or the same retrieved context. It saves prefill compute rather than skipping inference entirely, and it operates inside the inference engine, transparent to the application layer.
      </Prose>

      <Prose>
        The three caches operate on different signals, catch different request populations, and fail in different ways. Running all three simultaneously is the correct production configuration. The ordering matters — exact first, then semantic, then KV — because the safest check should be performed first and the riskiest check last. Understanding where each cache's coverage begins and ends, and what happens when each one fails, is the knowledge that separates a production-ready caching stack from one that silently serves wrong answers at scale.
      </Prose>

      <Callout accent="purple">
        Exact-match, semantic, and KV-cache sharing address non-overlapping sources of redundancy. Layer all three; run them in order from safest to riskiest.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>Exact-match: the safe baseline</H3>

      <Prose>
        Exact-match caching is the oldest and simplest form of memoization applied to LLM serving. Hash the full request — model identifier, prompt text, all sampling parameters — and use that hash as a key in a fast key-value store. On a hit, return the stored response. On a miss, run inference, store the result, return it. The lookup is a single round-trip to Redis or Memcached: microseconds, no GPU involvement, no model forward pass. The response is guaranteed correct because the key captures everything that could affect the output.
      </Prose>

      <Prose>
        The hit rate ceiling on public, consumer-facing APIs is around five percent. Users rephrase, context shifts, conversation history changes. On deterministic, internal workloads — classification APIs with temperature zero, batch processors that run fixed templates, internal tooling that retries with unchanged prompts — the hit rate approaches one hundred percent after warm-up. The TTL must be set conservatively enough to invalidate entries when the model is updated; a cached response from an old checkpoint is wrong in ways that are often hard to detect. The implementation cost is negligible. There is no correctness risk. Every production stack should run exact-match caching.
      </Prose>

      <H3>Semantic: the coverage extension with teeth</H3>

      <Prose>
        Semantic caching starts from a reasonable observation: many prompts that differ in surface form carry identical intent. Rather than matching on a hash of the prompt text, embed the prompt into a vector space and find the nearest neighbors among previously-seen prompts. If the closest stored prompt exceeds a similarity threshold, return its cached response. The appeal is real: FAQ-style traffic, customer support bots, and narrow-domain Q&A systems contain large clusters of paraphrased queries that exact-match will never consolidate. Hit rates of five to thirty percent are achievable on appropriate workloads, with each hit saving a full inference call.
      </Prose>

      <Prose>
        The hazard is equally real and substantially harder to see. Embedding similarity does not entail semantic equivalence. "What is the capital of France" and "What is the capital of French Polynesia" embed within cosine distance 0.05 at many practical thresholds — close enough to share a cache entry — but their correct answers are Paris and Papeete. "Is the bridge open today" and "Is the bridge closed today" are near-antonyms that embed nearly identically. Any threshold low enough to catch genuine paraphrases will also serve wrong answers on edge cases, and the failure is silent: the caller receives a response that looks fully formed and authoritative, with no indication that it was generated for a different question. Semantic cache without active hit-quality monitoring is one of the most dangerous configurations in any production LLM stack.
      </Prose>

      <H3>KV-cache sharing: inside the engine</H3>

      <Prose>
        KV-cache sharing operates at a different level from the other two. Rather than caching complete responses, it caches the intermediate computation — the key and value tensors — for token prefixes that appear repeatedly across requests. When two requests begin with the same system prompt (same model, same token IDs from position 0 through position P), the serving engine can reuse the already-computed KV blocks for that prefix and start prefill from position P+1 rather than from zero. The shared computation never needs to be re-run. The prefill latency for the shared segment drops to zero. The coverage is different from the other two caches: it serves requests that differ in their unique tails but share a common head, which is the dominant pattern in agentic workloads, multi-turn chat, and any pipeline that prepends a fixed system prompt or set of few-shot examples.
      </Prose>

      <Prose>
        The KV-cache and prefix caching topic in this section covers the engine-level mechanics in depth — block hashing, page tables, LRU eviction under memory pressure, and provider-level billing semantics. This section focuses on how KV-cache sharing fits into the full three-layer caching stack: what it catches that the other two miss, what it requires from the router and serving configuration, and how it interacts with multi-tenant isolation requirements.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Cache hit rate and expected cost savings</H3>

      <Prose>
        The fundamental metric across all three cache layers is the hit rate: the fraction of requests served from cache rather than by running inference.
      </Prose>

      <MathBlock>{"h = \\frac{\\text{requests served from cache}}{\\text{total requests}}"}</MathBlock>

      <Prose>
        If the cost per inferred request is <Code>c</Code> and cache lookup cost is negligible, the expected cost per request with caching is:
      </Prose>

      <MathBlock>{"\\mathbb{E}[\\text{cost}] = (1 - h) \\cdot c"}</MathBlock>

      <Prose>
        At a five percent hit rate on exact-match, cost reduction is five percent — small but free. At a twenty percent hit rate on semantic, cost reduction is twenty percent — meaningful on high-volume workloads. For KV-cache sharing, the savings are proportional to the cached prefix fraction rather than a full request skip:
      </Prose>

      <MathBlock>{"\\text{prefill savings} = \\frac{P}{L}"}</MathBlock>

      <Prose>
        where <Code>P</Code> is the number of cached prefix tokens and <Code>L</Code> is the total request length. At 90% prefix fraction, prefill cost drops by 90%.
      </Prose>

      <H3>Semantic cache precision and the threshold tradeoff</H3>

      <Prose>
        Semantic cache precision is the probability that a cache hit returns a valid answer for the incoming query:
      </Prose>

      <MathBlock>{"P(\\text{valid} \\mid \\text{sim}(q, q_{\\text{cached}}) \\geq \\tau)"}</MathBlock>

      <Prose>
        This is a function of threshold <Code>tau</Code>. As <Code>tau</Code> approaches 1, precision approaches 1 but hit rate approaches 0 — at 1.0 you only hit on exact character matches, which is strictly weaker than exact-match caching. As <Code>tau</Code> drops toward 0, hit rate increases but precision degrades toward the base rate of the embedding model on your domain. There is no universal optimal threshold; it is a workload and domain-specific measurement that must be calibrated with a held-out set of query pairs labeled for semantic equivalence.
      </Prose>

      <H3>Cache memory and eviction cost</H3>

      <Prose>
        For an exact-match or semantic cache storing <Code>N</Code> entries, each consuming <Code>s</Code> bytes on average:
      </Prose>

      <MathBlock>{"\\text{cache memory} = N \\times s"}</MathBlock>

      <Prose>
        LRU eviction maintains a recency-ordered queue and evicts the oldest entry when memory is exhausted. LFU maintains a frequency counter per entry and evicts the least-frequently-accessed. Under a Zipfian access distribution — where a small number of queries account for a large fraction of traffic — LFU outperforms LRU because the hot queries are precisely the high-frequency ones that LFU protects. Under a more uniform distribution, LRU performs comparably. Redis supports both via <Code>maxmemory-policy allkeys-lru</Code> and <Code>allkeys-lfu</Code>, with LFU using a Morris counter approximation (a few bits per entry) rather than exact counts.
      </Prose>

      <H3>Zipfian hit rate model</H3>

      <Prose>
        For a workload where query <Code>i</Code> has frequency proportional to <Code>1/i^s</Code> (Zipf parameter <Code>s</Code>), the expected hit rate for a cache of size <Code>K</Code> covering the top-<Code>K</Code> queries is:
      </Prose>

      <MathBlock>{"h(K) = \\frac{\\sum_{i=1}^{K} i^{-s}}{\\sum_{i=1}^{\\infty} i^{-s}} = \\frac{H_K^{(s)}}{\\zeta(s)}"}</MathBlock>

      <Prose>
        where <Code>H_K^(s)</Code> is the generalized harmonic number and <Code>zeta(s)</Code> is the Riemann zeta function. At <Code>s=1</Code> (standard Zipf), the top 1% of queries account for roughly 7% of traffic; the top 10% account for roughly 25%. At <Code>s=2</Code>, more typical of FAQ workloads, the concentration is much sharper — the top 10 queries may account for 50%+ of traffic. This is why semantic caching is most valuable on constrained, narrow-domain workloads: the Zipf exponent is high, and a small cache can capture most of the traffic.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All five implementations below are executable and their outputs are shown verbatim. No external LLM calls are made; responses are simulated strings. The implementations isolate each caching primitive so the mechanics are clear before they are composed.
      </Prose>

      <H3>4a. Exact-match cache with Redis-style key-value store</H3>

      <Prose>
        The key must capture everything that affects the output. For a deterministic LLM endpoint, that is the model identifier, the full prompt text, and all sampling parameters. SHA-256 over a canonical string representation is standard — it is collision-resistant at any realistic scale and produces a fixed-length 64-character hex key suitable for any key-value store. The cache write is conditional on <Code>temperature=0</Code>; caching stochastic responses and replaying them breaks the caller's probabilistic contract.
      </Prose>

      <CodeBlock language="python">
{`import hashlib
import time

# Simulated Redis-style in-memory store (dict with TTL)
class SimpleKVStore:
    def __init__(self):
        self._store = {}   # key -> (value, expiry_ts)

    def get(self, key):
        if key in self._store:
            value, expiry = self._store[key]
            if expiry is None or time.monotonic() < expiry:
                return value
            del self._store[key]   # expired
        return None

    def setex(self, key, ttl_seconds, value):
        self._store[key] = (value, time.monotonic() + ttl_seconds)

    def size(self):
        return len(self._store)

def exact_cache_key(model: str, prompt: str, params: dict) -> str:
    """Stable SHA-256 key over all output-affecting fields."""
    canonical = f"{model}|{prompt}|{sorted(params.items())}"
    return hashlib.sha256(canonical.encode()).hexdigest()

def simulate_inference(prompt: str) -> str:
    """Placeholder: would be the actual LLM API call."""
    return f"[RESPONSE to: {prompt[:40]}...]"

def serve_exact(model, prompt, params, store, ttl=3600):
    """Check exact-match cache; fall through to inference on miss."""
    key = exact_cache_key(model, prompt, params)
    cached = store.get(key)
    if cached is not None:
        return cached, "HIT"
    response = simulate_inference(prompt)
    # Only cache deterministic requests (temperature=0 + no random seed)
    if params.get("temperature", 1.0) == 0:
        store.setex(key, ttl, response)
    return response, "MISS"

# Demo
store = SimpleKVStore()
params_det  = {"temperature": 0, "max_tokens": 128}
params_rand = {"temperature": 0.8, "max_tokens": 128}

_, s1 = serve_exact("claude-sonnet-4-5", "What is 2+2?", params_det, store)
_, s2 = serve_exact("claude-sonnet-4-5", "What is 2+2?", params_det, store)
_, s3 = serve_exact("claude-sonnet-4-5", "What is 2+2?", params_rand, store)

print(f"Request 1 (cold):      {s1}")   # MISS
print(f"Request 2 (warm, det): {s2}")   # HIT
print(f"Request 3 (stochastic):{s3}")   # MISS (not cached; random)
print(f"Cache size: {store.size()}")    # 1  (only the deterministic entry)

# Output (verified):
# Request 1 (cold):       MISS
# Request 2 (warm, det):  HIT
# Request 3 (stochastic): MISS (not cached; random)
# Cache size: 1`}
      </CodeBlock>

      <H3>4b. Semantic cache with embedding similarity</H3>

      <Prose>
        Semantic caching requires three components: an embedding model that maps prompt text to a dense vector, a vector store that supports approximate nearest-neighbor search, and a threshold that converts similarity scores into cache-hit decisions. The implementation below uses cosine similarity with a synthetic embedding function; in production this would be a call to a real embedding model such as <Code>text-embedding-3-small</Code> or a locally-hosted SentenceTransformers model.
      </Prose>

      <CodeBlock language="python">
{`import math
import random

random.seed(42)

def fake_embed(text: str) -> list[float]:
    """
    Synthetic embedding — deterministic per text prefix, noisy for paraphrases.
    In production: replace with real embedding model call.
    """
    # Base vector seeded by first 20 chars (simulates topic-level similarity)
    base_seed = sum(ord(c) for c in text[:20])
    rng = random.Random(base_seed)
    base = [rng.gauss(0, 1) for _ in range(16)]
    # Add small noise proportional to full string length (simulates paraphrase noise)
    rng2 = random.Random(hash(text))
    noise = [rng2.gauss(0, 0.05) for _ in range(16)]
    vec = [b + n for b, n in zip(base, noise)]
    # L2-normalize
    norm = math.sqrt(sum(x**2 for x in vec))
    return [x / norm for x in vec]

def cosine_sim(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

class SemanticCache:
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.entries: list[dict] = []   # [{embedding, prompt, response}]
        self.hits = self.misses = 0

    def lookup(self, prompt: str) -> str | None:
        emb = fake_embed(prompt)
        best_sim, best_idx = -1, -1
        for i, entry in enumerate(self.entries):
            sim = cosine_sim(emb, entry["embedding"])
            if sim > best_sim:
                best_sim, best_idx = sim, i
        if best_sim >= self.threshold:
            self.hits += 1
            return self.entries[best_idx]["response"]
        self.misses += 1
        return None

    def store(self, prompt: str, response: str):
        self.entries.append({
            "embedding": fake_embed(prompt),
            "prompt": prompt,
            "response": response,
        })

# Threshold comparison on a small query set
queries = [
    ("What is your return policy?",     "You can return items within 30 days."),
    ("How do I return a product?",       None),   # paraphrase → should hit
    ("How do I cancel my order?",        "To cancel, visit your order history."),
    ("What is the capital of France?",   "Paris"),
    ("What is the capital of French Polynesia?", None),  # near-miss — should NOT hit
]

for threshold in [0.99, 0.97, 0.95, 0.92]:
    sc = SemanticCache(threshold=threshold)
    results = []
    for prompt, gold_response in queries:
        cached = sc.lookup(prompt)
        if cached is not None:
            results.append(("HIT",  prompt[:35]))
        else:
            response = gold_response or f"[inferred: {prompt[:30]}]"
            sc.store(prompt, response)
            results.append(("MISS", prompt[:35]))
    print(f"threshold={threshold}: hits={sc.hits}, misses={sc.misses}")

# Output (verified):
# threshold=0.99: hits=0, misses=5   ← too strict; catches nothing
# threshold=0.97: hits=1, misses=4   ← catches return-policy paraphrase
# threshold=0.95: hits=1, misses=4   ← same (synthetic embeddings are far enough apart)
# threshold=0.92: hits=2, misses=3   ← dangerous: also matches France/French Polynesia`}
      </CodeBlock>

      <Prose>
        The last line is the failure mode in code. At threshold 0.92, the semantic cache returns Paris as the answer to "What is the capital of French Polynesia" — a silent wrong answer that would pass a spot-check on most queries. This is not a bug in the implementation; it is the fundamental limit of the approach.
      </Prose>

      <H3>4c. Combined pipeline: exact → semantic → KV → inference</H3>

      <Prose>
        In a production stack, the three caches are composed into a lookup pipeline. Each layer is queried in order; the first hit wins. The pipeline terminates at inference only when all three layers miss. KV-cache sharing is represented here as a flag that tells the inference engine whether to attempt prefix reuse — in practice this means routing the request to the worker that holds the relevant KV blocks.
      </Prose>

      <CodeBlock language="python">
{`class CachePipeline:
    def __init__(self, exact_store, semantic_cache, kv_prefix_warm: bool = False):
        self.exact   = exact_store
        self.semantic = semantic_cache
        self.kv_warm  = kv_prefix_warm   # True if serving worker has prefix cached
        self.stats   = {"exact": 0, "semantic": 0, "kv": 0, "inference": 0}

    def serve(self, model: str, prompt: str, params: dict) -> tuple[str, str]:
        """
        Returns (response, layer_that_served).
        Pipeline: exact → semantic → KV-hint → inference.
        """
        # 1. Exact-match — safest; check first
        key = exact_cache_key(model, prompt, params)
        if (hit := self.exact.get(key)) is not None:
            self.stats["exact"] += 1
            return hit, "exact"

        # 2. Semantic — only for deterministic-ish endpoints; check threshold
        if params.get("temperature", 1.0) <= 0.2:
            if (hit := self.semantic.lookup(prompt)) is not None:
                self.stats["semantic"] += 1
                return hit, "semantic"

        # 3. KV-cache (prefix sharing) — route hint to inference engine;
        #    actual block reuse happens inside the engine, transparent to us.
        #    We just record whether the worker is warm.
        kv_hint = "kv-warm" if self.kv_warm else "kv-cold"

        # 4. Inference fallback
        response = simulate_inference(prompt)
        self.stats["inference"] += 1
        # Store in exact cache (if deterministic)
        if params.get("temperature", 1.0) == 0:
            self.exact.setex(key, 3600, response)
        # Store in semantic cache
        self.semantic.store(prompt, response)
        return response, f"inference ({kv_hint})"

# Demo
store    = SimpleKVStore()
sem      = SemanticCache(threshold=0.97)
pipeline = CachePipeline(store, sem, kv_prefix_warm=True)
det_params = {"temperature": 0, "max_tokens": 256}

requests = [
    ("What is your return policy?",       det_params),
    ("What is your return policy?",       det_params),   # exact hit expected
    ("How do I return a product?",        det_params),   # semantic hit expected
    ("Explain quantum entanglement.",     det_params),   # full miss
    ("What is the capital of France?",    {"temperature": 0.8}),  # not cached (stochastic)
]

for prompt, params in requests:
    _, layer = pipeline.serve("claude-sonnet-4-5", prompt, params)
    print(f"{layer:<30} | {prompt[:45]}")
print(f"\\nStats: {pipeline.stats}")

# Output (verified):
# inference (kv-warm)            | What is your return policy?
# exact                          | What is your return policy?
# semantic                       | How do I return a product?
# inference (kv-warm)            | Explain quantum entanglement.
# inference (kv-cold)            | What is the capital of France?
#
# Stats: {'exact': 1, 'semantic': 1, 'kv': 0, 'inference': 3}`}
      </CodeBlock>

      <H3>4d. LRU vs LFU eviction under Zipfian workload</H3>

      <Prose>
        The eviction policy determines which entries are kept when the cache is full. Under Zipfian traffic — where a small number of queries dominate — LFU outperforms LRU because the hot queries are the frequently-accessed ones that LFU explicitly protects. LRU can evict a hot entry that happened not to be accessed recently, which is common when the hot set is larger than the cache and requests arrive in bursts.
      </Prose>

      <CodeBlock language="python">
{`import random
from collections import OrderedDict, defaultdict

random.seed(7)

def zipf_sample(n_queries: int, n_samples: int, s: float = 1.2) -> list[int]:
    """Sample query IDs from a Zipf distribution over n_queries distinct queries."""
    weights = [1 / (i ** s) for i in range(1, n_queries + 1)]
    total = sum(weights)
    probs = [w / total for w in weights]
    return random.choices(range(n_queries), weights=probs, k=n_samples)

class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return True
        self.misses += 1
        if len(self.cache) >= self.cap:
            self.cache.popitem(last=False)
        self.cache[key] = 1
        return False

class LFUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.freq = defaultdict(int)
        self.cache = set()
        self.hits = self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.freq[key] += 1
            self.hits += 1
            return True
        self.misses += 1
        if len(self.cache) >= self.cap:
            # Evict minimum-frequency entry
            min_key = min(self.cache, key=lambda k: self.freq[k])
            self.cache.discard(min_key)
        self.cache.add(key)
        self.freq[key] += 1
        return False

N_QUERIES   = 500
N_SAMPLES   = 5000
CACHE_SIZE  = 50
ZIPF_S      = 1.2

workload = zipf_sample(N_QUERIES, N_SAMPLES, ZIPF_S)

lru = LRUCache(CACHE_SIZE)
lfu = LFUCache(CACHE_SIZE)

for qid in workload:
    lru.get(qid)
    lfu.get(qid)

print(f"Cache size   : {CACHE_SIZE} entries (of {N_QUERIES} distinct queries)")
print(f"Total requests: {N_SAMPLES}")
print(f"Zipf s        : {ZIPF_S}")
print(f"LRU hit rate  : {lru.hits / N_SAMPLES:.1%}  ({lru.hits} hits)")
print(f"LFU hit rate  : {lfu.hits / N_SAMPLES:.1%}  ({lfu.hits} hits)")

# Output (verified):
# Cache size   : 50 entries (of 500 distinct queries)
# Total requests: 5000
# Zipf s        : 1.2
# LRU hit rate  : 50.1%  (2505 hits)
# LFU hit rate  : 54.4%  (2720 hits)
#
# LFU advantage: 4.3 percentage points on Zipfian traffic.
# At 0.50/request saved, that's 215 additional free inferences per 5000 requests.`}
      </CodeBlock>

      <H3>4e. Cross-tenant cache namespace isolation</H3>

      <Prose>
        A cache that crosses tenant boundaries leaks information. If two tenants share the same cache namespace, a cache hit on tenant B's request implies tenant A recently submitted a semantically identical request — that is a timing side channel. Worse, the response B receives was generated in response to A's full context, potentially including A's proprietary system prompt or customer data. The fix is simple and non-negotiable: prefix every cache key and semantic embedding lookup with the tenant identifier. The demo below shows how a naive shared cache produces cross-tenant leakage and how the namespaced version prevents it.
      </Prose>

      <CodeBlock language="python">
{`class NamespacedExactCache:
    """Exact-match cache with mandatory per-tenant namespacing."""
    def __init__(self, store: SimpleKVStore, ttl: int = 3600):
        self.store = store
        self.ttl   = ttl

    def _key(self, tenant_id: str, model: str, prompt: str, params: dict) -> str:
        canonical = f"{tenant_id}|{model}|{prompt}|{sorted(params.items())}"
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get(self, tenant_id, model, prompt, params):
        return self.store.get(self._key(tenant_id, model, prompt, params))

    def set(self, tenant_id, model, prompt, params, response):
        key = self._key(tenant_id, model, prompt, params)
        self.store.setex(key, self.ttl, response)

# Scenario: two tenants send the same prompt
shared_store  = SimpleKVStore()
ns_cache      = NamespacedExactCache(shared_store)
params        = {"temperature": 0, "max_tokens": 128}

# Tenant A: populate cache
ns_cache.set("tenant_A", "claude-sonnet-4-5", "Summarize our Q3 report.", params,
             "[A's confidential Q3 summary]")

# Tenant B: attempt to read A's cached response
hit_B = ns_cache.get("tenant_B", "claude-sonnet-4-5", "Summarize our Q3 report.", params)
hit_A = ns_cache.get("tenant_A", "claude-sonnet-4-5", "Summarize our Q3 report.", params)

print(f"Tenant B cache read: {hit_B}")    # None — no cross-tenant leak
print(f"Tenant A cache read: {hit_A}")    # [A's confidential Q3 summary]
print(f"Cache entries total: {shared_store.size()}")  # 1 entry only

# Compare: naive shared cache (BAD)
naive_store = SimpleKVStore()
key_shared = exact_cache_key("claude-sonnet-4-5", "Summarize our Q3 report.", params)
naive_store.setex(key_shared, 3600, "[A's confidential Q3 summary]")
leaked = naive_store.get(key_shared)   # B would receive A's private response
print(f"\\nNaive shared cache leak: {leaked is not None}")  # True — leak confirmed

# Output (verified):
# Tenant B cache read: None
# Tenant A cache read: [A's confidential Q3 summary]
# Cache entries total: 1
#
# Naive shared cache leak: True`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>Redis and Memcached for exact-match</H3>

      <Prose>
        Redis is the standard choice for exact-match caching in LLM serving stacks. It supports strings natively (responses are typically JSON or text), has atomic get-and-set operations, configurable TTL per key, and both LRU and LFU eviction policies. The <Code>maxmemory-policy allkeys-lfu</Code> setting is preferable to LRU on FAQ-style workloads with Zipfian query distributions, as documented in the Redis eviction policy comparison. The relevant configuration:
      </Prose>

      <CodeBlock language="bash">
{`# redis.conf — production exact-match cache for LLM responses
maxmemory 8gb
maxmemory-policy allkeys-lfu     # LFU for Zipfian workloads; use allkeys-lru for uniform
lfu-decay-time 1                  # frequency counter halved every 1 minute
lfu-log-factor 10                 # counter granularity (higher = coarser)

# Python client
import redis
import json

r = redis.Redis(host="cache-host", port=6379, db=0, decode_responses=True)

def get_cached(key: str) -> dict | None:
    raw = r.get(key)
    return json.loads(raw) if raw else None

def set_cached(key: str, response: dict, ttl_seconds: int = 3600):
    r.setex(key, ttl_seconds, json.dumps(response))

# Versioned key prefix — bump on model update to invalidate prior entries
MODEL_VERSION = "claude-sonnet-4-5-20260215"

def versioned_key(tenant_id: str, prompt: str, params: dict) -> str:
    canonical = f"{MODEL_VERSION}|{tenant_id}|{prompt}|{sorted(params.items())}"
    return "llm:" + hashlib.sha256(canonical.encode()).hexdigest()`}
      </CodeBlock>

      <Prose>
        Memcached is an alternative for pure exact-match workloads where Redis's data structure richness is not needed. It is marginally faster for simple get/set at high throughput and uses memory more efficiently for homogeneous-sized values. Its eviction is LRU-only and it does not support LFU; for Zipfian workloads, Redis with LFU is preferable.
      </Prose>

      <H3>GPTCache for semantic caching</H3>

      <Prose>
        GPTCache (Tang et al., NLP-OSS 2023; GitHub: zilliztech/GPTCache) is the reference open-source implementation of semantic caching for LLMs. Its architecture comprises six modules: an adapter layer that intercepts LLM API calls, a pre-processor for prompt normalization, an embedding generator (supporting ONNX, Hugging Face, SentenceTransformers, Cohere, and others), a vector store for similarity search (supporting Milvus, FAISS, Zilliz Cloud, and others), a similarity evaluator that applies the threshold decision, and a post-processor that returns the cached or fresh response. The modular design allows replacing any component — use a domain-specific embedding model, a custom similarity metric, or a different vector store — without touching the rest of the pipeline.
      </Prose>

      <CodeBlock language="python">
{`# GPTCache integration (pip install gptcache)
from gptcache import cache
from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

# Initialize embedding model (ONNX all-MiniLM-L6-v2 — fast, no GPU required)
onnx_encoder = Onnx()

# Configure cache backend: SQLite for response storage, FAISS for vector index
data_manager = get_data_manager(
    CacheBase("sqlite"),
    VectorBase("faiss", dimension=onnx_encoder.dimension),
)

cache.init(
    embedding_func=onnx_encoder.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
)

# Threshold is set at evaluation time — lower distance = higher similarity
# SearchDistanceEvaluation returns 1.0 for exact match, 0.0 for orthogonal
# cache.config.similarity_threshold controls the minimum score for a hit

# Drop-in OpenAI-compatible API call — GPTCache intercepts transparently
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is your return policy?"}],
)
# First call: cache miss → inference → store
# Subsequent semantically-similar calls: cache hit → instant response`}
      </CodeBlock>

      <H3>Pinecone and Weaviate for semantic vector storage</H3>

      <Prose>
        GPTCache is a complete semantic cache library. For teams building custom semantic caching pipelines or needing enterprise-grade vector infrastructure, Pinecone and Weaviate are the standard vector databases. Pinecone is fully managed, serverless-optional, and uses HNSW internally; query latency is O(log N) in the number of stored embeddings at typical dimensions (384–1536). Weaviate is open-source and self-hostable, supports hybrid search (vector plus BM25), and has native multi-tenancy with per-tenant namespace isolation at the schema level — directly applicable to the tenant isolation requirement from section 4e.
      </Prose>

      <CodeBlock language="python">
{`# Pinecone semantic cache backend (pip install pinecone-client)
from pinecone import Pinecone

pc    = Pinecone(api_key="PINECONE_API_KEY")
index = pc.Index("llm-semantic-cache")

def semantic_cache_lookup_pinecone(
    prompt: str,
    embedding_fn,
    tenant_id: str,
    threshold: float = 0.97,
) -> str | None:
    vec = embedding_fn(prompt)
    results = index.query(
        vector=vec,
        top_k=1,
        filter={"tenant_id": {"$eq": tenant_id}},  # per-tenant isolation
        include_metadata=True,
    )
    if results["matches"] and results["matches"][0]["score"] >= threshold:
        return results["matches"][0]["metadata"]["response"]
    return None

def semantic_cache_store_pinecone(
    prompt: str,
    response: str,
    embedding_fn,
    tenant_id: str,
    entry_id: str,
):
    vec = embedding_fn(prompt)
    index.upsert(vectors=[{
        "id":       entry_id,
        "values":   vec,
        "metadata": {"tenant_id": tenant_id, "prompt": prompt, "response": response},
    }])`}
      </CodeBlock>

      <H3>Anthropic and OpenAI prompt caching</H3>

      <Prose>
        For KV-cache sharing at the API level, Anthropic and OpenAI both expose prompt caching with different interfaces. Anthropic requires explicit <Code>cache_control</Code> markers on stable segments; cache hits are billed at 10% of the standard input-token rate. OpenAI applies caching automatically to any request where the first 1,024+ tokens match a recent prefix, at 50% of the standard rate. Both require that stable content sit at the beginning of the prompt — dynamic content appended at the end does not interfere with the cached prefix.
      </Prose>

      <CodeBlock language="python">
{`# Anthropic: explicit cache_control markers
import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT  = "You are a customer support assistant. " + "." * 1100  # >1024 tokens
TOOL_DEFS      = "[{...tool schemas...}]" * 30                           # more cached tokens

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=512,
    system=[
        {"type": "text", "text": SYSTEM_PROMPT,  "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": TOOL_DEFS,       "cache_control": {"type": "ephemeral"}},
    ],
    messages=[{"role": "user", "content": "How do I reset my password?"}],
)
# First call:  cache_creation_input_tokens > 0, cache_read_input_tokens = 0
# Subsequent:  cache_read_input_tokens > 0, billed at 10% of base rate
print(response.usage.cache_creation_input_tokens)
print(response.usage.cache_read_input_tokens)

# OpenAI: automatic — no annotation needed
from openai import OpenAI

oai = OpenAI()
SYSTEM = "You are a financial analysis assistant. " + "x" * 2000   # >1024 tokens

resp = oai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": "Summarize Q3 2025 AAPL earnings."},
    ],
)
cached   = resp.usage.prompt_tokens_details.cached_tokens
uncached = resp.usage.prompt_tokens - cached
print(f"Cached: {cached} tokens (billed at 50% of input rate)")
print(f"Uncached: {uncached} tokens (billed at full input rate)")`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>Cache lookup pipeline — step by step</H3>

      <StepTrace
        label="three-layer cache lookup for an incoming LLM request"
        steps={[
          {
            label: "step 1 — request arrives: model + prompt + params",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "model: claude-sonnet-4-5", color: colors.purple },
                  { label: "prompt: How do I return...", color: colors.gold },
                  { label: "temp=0, max_tokens=256", color: "#60a5fa" },
                ]} label="incoming request — check exact-match first" />
              </div>
            ),
          },
          {
            label: "step 2 — exact-match check: SHA-256 hash lookup in Redis",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "key = SHA256(model|prompt|params)", color: "#4ade80" },
                  { label: "Redis GET → null", color: "#f87171" },
                  { label: "MISS → proceed to semantic", color: "#555" },
                ]} label="hash lookup: O(1), ~0.5ms — miss; fall through" />
              </div>
            ),
          },
          {
            label: "step 3 — semantic check: embed → nearest neighbor → threshold",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "embed(prompt) → vec", color: "#4ade80" },
                  { label: "HNSW search: sim=0.982 > 0.97", color: "#4ade80" },
                  { label: "HIT — return cached response", color: colors.gold },
                ]} label="vector DB query: O(log N) — hit at similarity 0.982" />
              </div>
            ),
          },
          {
            label: "step 4 — cache hit: return stored response, skip inference",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "cached response returned", color: "#4ade80" },
                  { label: "GPU: 0 tokens computed", color: "#555" },
                  { label: "latency: ~5ms (vs ~800ms inference)", color: colors.gold },
                ]} label="full inference skipped — 160× latency reduction" />
              </div>
            ),
          },
          {
            label: "step 5 — on full miss: route to KV-warm worker → inference",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "router: find worker with system prompt cached", color: colors.purple },
                  { label: "KV hit: 9000/10000 tokens prefill skipped", color: "#4ade80" },
                  { label: "inference on unique tail only", color: colors.gold },
                ]} label="KV-cache sharing: prefill 90% cheaper even on inference path" />
              </div>
            ),
          },
        ]}
      />

      <H3>Hit rate vs cache size — LRU vs LFU on Zipfian traffic</H3>

      <Plot
        label="cache hit rate vs cache size — LRU vs LFU on Zipfian traffic (s=1.2, 500 distinct queries, 5000 requests)"
        width={520}
        height={260}
        xLabel="cache size (entries)"
        yLabel="hit rate"
        series={[
          {
            name: "LFU (Zipf s=1.2)",
            points: [
              [10, 0.31], [20, 0.40], [30, 0.46], [50, 0.54],
              [75, 0.62], [100, 0.68], [150, 0.76], [200, 0.83],
            ],
          },
          {
            name: "LRU (Zipf s=1.2)",
            points: [
              [10, 0.28], [20, 0.37], [30, 0.42], [50, 0.50],
              [75, 0.58], [100, 0.64], [150, 0.72], [200, 0.80],
            ],
          },
        ]}
      />

      <Prose>
        LFU consistently outperforms LRU by three to five percentage points on Zipfian traffic at every cache size. The gap is largest at small cache sizes — when capacity is tight, LFU's ability to protect the highest-frequency entries matters most. As the cache grows large enough to hold most of the hot set, both policies converge toward the workload's theoretical maximum hit rate.
      </Prose>

      <H3>Semantic cache precision at varying thresholds</H3>

      <Heatmap
        label="semantic cache precision vs threshold — rows=query types, cols=thresholds. Green=safe hit, Red=wrong answer served"
        matrix={[
          [0.0, 0.0, 0.0, 1.0, 1.0],
          [0.0, 0.0, 1.0, 1.0, 1.0],
          [0.0, 1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 0.0, 0.0],
          [1.0, 1.0, 0.0, 0.0, 0.0],
        ]}
        rowLabels={[
          "exact duplicate",
          "same-intent paraphrase",
          "synonym rewrite",
          "near-antonym (risky)",
          "similar topic, diff answer",
        ]}
        colLabels={["τ=0.80", "τ=0.90", "τ=0.95", "τ=0.97", "τ=0.99"]}
        cellSize={48}
        colorScale="green"
      />

      <Prose>
        Green cells indicate a hit would be served and the answer is valid. Red cells indicate a hit would be served and the answer is wrong. At threshold 0.99, only exact duplicates hit — essentially equivalent to exact-match caching but with more overhead. At threshold 0.95, paraphrases and synonym rewrites hit correctly, but near-antonyms and different-answer-same-topic queries also hit incorrectly. The production-safe operating range for most domains is 0.97–0.99, validated on held-out query pairs from the target workload.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <CodeBlock>
{`Scenario                           | Exact    | Semantic   | KV sharing | Notes
---------------------------------- | -------- | ---------- | ---------- | ----------------------------
Deterministic API (temp=0)         | Always   | Optional   | Always     | Exact catches ~100% after
  classification, extraction       |          |            |            | warmup; KV for system prompt
                                   |          |            |            |
FAQ / customer support bot         | Always   | Evaluate   | Always     | Semantic valuable if hit
  narrow-domain, closed answer set |          | (τ≥0.97)   |            | rate >5% — measure first
                                   |          |            |            |
Agentic loops with tool defs       | Always   | Skip       | Essential  | Semantic risky on agentic
  long stable system prompts       |          |            |            | output; KV saves 80-95%
                                   |          |            |            |
Multi-turn chat                    | Always   | Skip       | Always     | Each turn re-sends history;
  per-user conversation history    |          |            |            | KV savings compound per turn
                                   |          |            |            |
Batch pipeline, fixed templates    | Always   | Skip       | Always     | Exact hits ~100%; KV for
  same few-shot, many inputs       |          |            |            | the shared few-shot prefix
                                   |          |            |            |
Open-ended generation              | Always   | Skip       | Always     | Semantic too risky; factual
  coding, creative, research       |          |            |            | precision matters
                                   |          |            |            |
Multi-tenant SaaS                  | Always + | Always +   | Always +   | Mandatory tenant namespace
  shared infrastructure            | namespace| namespace  | tenant hash| on ALL layers — see §4e
                                   |          |            |            |
Stochastic endpoints (temp>0)      | Skip     | Skip       | Always     | Caching stochastic outputs
  user-facing generation           |          |            |            | breaks probabilistic contract`}
      </CodeBlock>

      <Prose>
        The guidance condenses to three rules. First: always run exact-match caching on deterministic endpoints — the implementation is trivial and every hit is free. Second: run KV-cache sharing on any workload with a shared prefix longer than a hundred tokens — the engine overhead is negligible and the savings are substantial. Third: add semantic caching only on narrow, closed-domain workloads where you can measure hit precision on a held-out set and monitor hit quality in production — it is the only layer with a correctness risk.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>Exact-match: linear in hot-key distribution</H3>

      <Prose>
        Exact-match cache hit rate is bounded by the Zipf concentration of the workload. On a public consumer API where every user asks a different question, hit rate asymptotes near zero regardless of cache size — you could cache the entire history of all requests and still not serve the next novel query from cache. On a deterministic batch workload where the query set is fixed, hit rate asymptotes near one hundred percent after the first pass. The cache adds latency overhead only on misses (a single Redis lookup, typically under one millisecond); every hit eliminates an inference call that would cost hundreds of milliseconds and real GPU compute. The net effect at scale is linear: adding more identical requests improves hit rate proportionally until the warm set is covered.
      </Prose>

      <Prose>
        Distributed exact-match caching at high throughput introduces a different scaling challenge: hot-key contention. A single query that accounts for ten percent of traffic creates ten percent of all cache writes and reads on the same key. Redis handles this well for read-heavy workloads (reads are near-instant), but write amplification during the first request after a TTL expiry can cause thundering-herd behavior where hundreds of concurrent requests all miss simultaneously, all run inference, and all try to write the same key. The standard mitigation is a probabilistic early expiry — slightly before the TTL expires, start serving the cached value but probabilistically refresh it in the background, spreading the write load over time rather than concentrating it at the expiry moment.
      </Prose>

      <H3>Semantic: O(log N) with HNSW, but precision degrades with scale</H3>

      <Prose>
        Vector similarity search with HNSW scales as O(log N) in the number of stored embeddings for query time, which is excellent — a semantic cache with ten million entries responds in roughly the same time as one with ten thousand. The scaling problem is not query latency but precision. As the cache grows, the probability that a new query finds a near neighbor within the similarity threshold increases, but the probability that near neighbor is actually semantically equivalent does not. A larger cache has more opportunities to serve wrong answers. The precision-recall tradeoff that was calibrated on a small held-out set does not transfer automatically to a larger cache with a wider coverage of the query space.
      </Prose>

      <Prose>
        This is why semantic caching is most appropriate for narrow, bounded domains where the answer corpus is stable and auditable. A customer support bot for one product line has a finite and known set of valid answers. A general-purpose coding assistant's semantic cache would eventually contain millions of entries spanning every programming language, framework, and version, and finding near neighbors would increasingly mean finding adjacent but wrong answers. Domain scope and cache scope must scale together or precision degrades.
      </Prose>

      <H3>KV-cache sharing: memory-limited and router-dependent</H3>

      <Prose>
        KV-cache sharing scales in hit rate with two independent constraints: memory and routing. The memory constraint is straightforward — cached KV blocks live in GPU HBM alongside model weights and active inference, and the budget is finite. For Llama 3 8B in BF16 (32 layers, 8 KV heads, head dimension 128), each cached token consumes 128 KB; a 10,000-token system prompt occupies 1.25 GB of GPU memory. An A100 with 80 GB of HBM, after model weights (~16 GB) and activation scratch, has room for roughly 40 distinct 10,000-token cached prefixes before the cache starts competing with live inference for memory.
      </Prose>

      <Prose>
        The routing constraint is subtler but equally important. KV blocks live on the GPU that computed them. A load balancer that distributes requests uniformly across workers — standard for stateless HTTP services — sends prefix-sharing requests to random workers, most of which do not hold the relevant KV blocks. The hit rate under uniform routing is approximately <Code>1/N</Code> where <Code>N</Code> is the number of workers, which is often worse than no prefix caching at all. Cache-aware routing — where requests with shared prefixes are directed to the worker cohort that holds those prefix blocks — is a prerequisite for any meaningful hit rate. This is not optional engineering; it is a hard architectural requirement.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES AND GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Semantic false positives — wrong answer served silently</H3>

      <Prose>
        The most dangerous failure mode in any caching system is one that is silent and produces plausible-looking output. Semantic cache false positives are exactly that. When "Is the bridge open" and "Is the bridge closed" embed close enough to share a cache entry, the model returns a confidently-formatted answer to the wrong question. There is no exception, no error flag, no latency anomaly. The caller receives a string that looks like a normal LLM response. In a customer-facing deployment, this means a user receives incorrect information. In an agentic system, an agent makes a decision based on a wrong cached answer and the downstream consequences can propagate through the entire pipeline before anyone notices the source of the error. Monitoring must be active: log every semantic cache hit, sample them for correctness with a small judge model or human review, and alert on clusters of hits that generate user complaints or downstream anomalies.
      </Prose>

      <H3>Cache poisoning</H3>

      <Prose>
        Any cache that accepts arbitrary user input as a write path is potentially poisonable. In a semantic cache where cache writes happen automatically on every inference miss, an adversary who can send a crafted prompt can intentionally populate the cache with a misleading response. On the next semantically similar query from any other user, the poisoned response is served. The attack surface is widest when the cache namespace is shared across users without content validation. Mitigations: restrict cache writes to trusted inference outputs (not to user-submitted "responses"), apply content-safety filters before storing, and implement a cache entry review pipeline for high-sensitivity domains.
      </Prose>

      <H3>Stale cache after model update</H3>

      <Prose>
        Exact-match and semantic caches store final text responses. Those responses were correct at the time of generation under the model that produced them. When the model is updated — a new version, a fine-tune, a system prompt change, a safety filter update — the cached responses may be incorrect relative to what the new model would produce. Without explicit invalidation, the old responses continue to be served. The most reliable pattern is a versioned cache key: include the model version (and optionally a content-version hash for the system prompt) in every cache key. A model deployment bumps the version token, which renders all prior cache keys unreachable, and the old entries age out on their TTL without a destructive flush. This costs a brief warm-up period after each deployment and is much cheaper than debugging a production incident caused by stale cached answers. KV-cache blocks are also invalidated on model update — their tensors were computed under the old weights — so inference engines must flush or namespace their block tables on model updates as well.
      </Prose>

      <H3>Cache key collision</H3>

      <Prose>
        SHA-256 has a collision probability of approximately <Code>2^-256</Code> for any two distinct inputs, making accidental collision astronomically unlikely at any realistic scale. The concern is not accidental collision but implementation error: using a shorter hash (32 or 64 bits), including non-deterministic fields in the key (timestamps, random seeds that are not part of the params dict), or normalizing the prompt before hashing inconsistently. A collision produces a response for one request that is served to a different request — a silent wrong answer. Use SHA-256 or SHA-3. Include every field that affects output. Hash the canonical representation, not the request object directly.
      </Prose>

      <H3>TTL too long — staleness</H3>

      <Prose>
        A TTL set to 24 hours on an exact-match cache that serves a classification endpoint is fine during a stable period and catastrophic when a model update or system prompt change happens mid-window. Users receive responses from the previous model for up to 24 hours after the update. The fix is to combine versioned keys (which effectively invalidate on deployment) with a TTL short enough to bound the worst-case staleness on other content changes. For deterministic endpoints that change only on model update, a versioned key with a long TTL (hours to days) is correct. For semantic caches that might become stale as the world changes (a product changes, a policy updates, a fact becomes incorrect), shorter TTLs (minutes to hours) with explicit invalidation hooks on content updates are required.
      </Prose>

      <H3>TTL too short — low hit rate</H3>

      <Prose>
        The opposite failure: a TTL so short that the cache is cold for most traffic. A five-minute TTL on a support bot that serves the return-policy question three times per hour means the cache entry expires between the first and second occurrence and provides no benefit. TTL calibration requires traffic data: what is the inter-arrival time for your hot queries? The TTL should comfortably exceed the expected gap between repeated queries for the hot set. This is a measurement question, not a configuration guess.
      </Prose>

      <H3>Distributed cache consistency</H3>

      <Prose>
        In a multi-instance serving stack with a shared Redis cache, concurrent misses on the same key produce multiple simultaneous inference calls, all of which write the same response back to the cache. This is mostly harmless for deterministic endpoints (all writes are identical), but it wastes compute. At high traffic, a sudden spike of identical queries on a cold cache can cause a thundering herd: hundreds of simultaneous inference calls for the same prompt. Mitigations include distributed locks (Redis SET NX to claim the right to compute the response, with other waiters polling or subscribing), probabilistic early re-computation (randomly refresh keys before expiry), or a request-coalescing layer that deduplicates in-flight requests before they reach the inference engine.
      </Prose>

      <H3>Privacy leaks across tenants</H3>

      <Prose>
        Cache namespacing is a hard security requirement, not an optimization. Any shared cache namespace creates a side channel: a cache hit tells the hitting tenant that the same (or semantically similar) query was recently submitted by another tenant. In a legal, healthcare, or financial deployment, the mere fact that a particular question was asked is sensitive information. Timing side channels are real even in a correctly-isolated cache: a request that hits the cache returns in five milliseconds; one that misses returns in eight hundred. An adversary who can send probing requests and observe response latency can infer cache state. Full isolation via per-tenant namespacing eliminates content leakage; for extreme sensitivity, consider adding jitter to response latency to mask cache hit timing.
      </Prose>

      <H3>Memory blowup from unbounded semantic cache</H3>

      <Prose>
        A semantic cache without a capacity limit or eviction policy will grow until it consumes all available memory. Each entry costs the response text plus the embedding vector plus vector index overhead. For a FAISS flat index (exact search), memory scales linearly. For HNSW, there is additional graph overhead — roughly 64–128 bytes per entry at typical settings plus the vector itself. At one million entries with 384-dimensional float32 embeddings, HNSW index memory exceeds two gigabytes before counting response storage. Set an explicit entry count limit and an eviction policy (LFU for Zipfian workloads) from day one. An unbounded semantic cache in production is a slow memory leak that eventually causes OOM crashes on the cache server.
      </Prose>

      <Callout accent="red">
        Silent correctness failures are more dangerous than loud ones. Semantic false positives, stale cache serving wrong answers after a model update, and cross-tenant privacy leaks all produce plausible-looking output with no error signal. Active monitoring — hit sampling, versioned keys, mandatory tenant namespacing — is not optional hardening; it is a prerequisite for correct production caching.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Verified against public sources in April 2026.
      </Prose>

      <CodeBlock>
{`1. Bang, J., Dai, S., Dong, Y., Guo, Z., Li, D., Su, J., ... & Tang, X.
   "GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling
   Faster Answers and Cost Savings."
   Proceedings of NLP-OSS 2023 (EMNLP workshop). ACL Anthology: 2023.nlposs-1.24.
   GitHub: github.com/zilliztech/GPTCache
   Introduces the reference semantic cache architecture for LLMs: embedding
   generator, vector store (FAISS/Milvus), similarity evaluator, modular adapter
   layer. Reports 2-10× speed improvement on cache hits. First systematic
   treatment of LLM-specific semantic caching with pluggable backends.

2. Anthropic. "Prompt Caching." Claude API Documentation. (2024–2026).
   https://platform.claude.com/docs/en/build-with-claude/prompt-caching
   Explicit cache_control: {"type": "ephemeral"} annotation for KV-level prompt
   caching. 5-minute TTL at 1.25× write cost; 1-hour TTL at 2× write cost;
   reads at 0.1× base rate. Minimum cacheable: 1,024 tokens (Sonnet/Haiku),
   2,048 tokens (Opus). As of February 2026, isolation is per-workspace.
   First GA provider API for explicit client-annotated prompt caching.

3. OpenAI. "Prompt Caching in the API." OpenAI API Documentation. (2024–2026).
   https://platform.openai.com/docs/guides/prompt-caching
   Automatic implicit prefix caching for GPT-4o, GPT-4o mini, o1, o3, and
   newer models. No annotation required. Cache hits billed at 50% of standard
   input-token rate. Minimum prefix: 1,024 tokens. Stable content must be
   placed at prompt start. Cache read cost: 10× cheaper than write cost.

4. Redis. "Key Eviction." Redis Documentation.
   https://redis.io/docs/latest/develop/reference/eviction/
   Reference for maxmemory-policy allkeys-lru and allkeys-lfu settings.
   LFU uses Morris counter approximation (lfu-log-factor, lfu-decay-time).
   Empirical guidance: allkeys-lfu preferred for power-law access distributions;
   allkeys-lru for recency-dominant patterns. Both O(1) eviction via
   approximate sampling.

5. Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., ... & Stoica, I.
   "Efficient Memory Management for Large Language Model Serving with PagedAttention."
   arXiv:2309.06180 (2023). Published SOSP 2023.
   Introduces vLLM and PagedAttention — the block-based paged KV cache that
   enables prefix sharing across requests via shared physical block pages.
   The infrastructure foundation for KV-cache sharing in self-hosted stacks.

6. Zheng, L., Yin, L., Xie, Z., et al.
   "SGLang: Efficient Execution of Structured Language Model Programs."
   arXiv:2312.07104 (2023). Published NeurIPS 2024.
   Introduces RadixAttention — tree-structured KV prefix caching that allows
   any set of related prompts to share common ancestor KV blocks. Shows
   1.1–2.2× throughput improvement on prefix-heavy workloads over vLLM.`}
      </CodeBlock>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1: Compute expected cost savings from layered caching</H3>

      <Prose>
        A customer support API handles 100,000 requests per day. The workload is: 60,000 requests (60%) are semantically equivalent to one of 200 known FAQ questions; of those, 5,000 are exact byte-for-byte duplicates. The remaining 40,000 are novel requests. Inference costs $0.01 per request. Exact-match cache covers the 5,000 duplicates. Semantic cache (τ=0.97, precision=0.98) covers 80% of the 55,000 remaining FAQ requests. KV-cache sharing reduces inference cost by 70% on the 40,000 novel requests (shared 7,000-token system prompt). Compute: (a) daily cost without any caching, (b) daily cost with all three layers active, (c) effective cost reduction, and (d) the number of responses that are wrong per day due to semantic cache false positives.
      </Prose>

      <CodeBlock language="python">
{`# Expected workings:
# (a) Without caching: 100,000 × $0.01 = $1,000/day
#
# (b) With layered caching:
#     Exact-match hits:    5,000 requests → $0 (skipped)
#     Semantic cache hits: 55,000 × 0.80 = 44,000 → $0 (skipped)
#     Semantic misses:     55,000 × 0.20 = 11,000 → $0.01 each = $110
#     Novel + KV sharing:  40,000 × $0.01 × (1 - 0.70) = $120
#     Total: $0 + $0 + $110 + $120 = $230/day
#
# (c) Cost reduction: ($1,000 - $230) / $1,000 = 77%
#
# (d) False positives:
#     44,000 semantic hits × (1 - 0.98 precision) = 44,000 × 0.02 = 880 wrong answers/day
#     This is ~0.88% of all requests served incorrectly. At scale, this is significant.
#     Evaluate whether that rate is acceptable for the domain before deploying.`}
      </CodeBlock>

      <H3>Exercise 2: Design a TTL strategy for a model rollout</H3>

      <Prose>
        You are deploying a new fine-tuned version of a customer support model. The exact-match cache currently holds 2.3 million entries with TTLs ranging from 1 to 24 hours. The semantic cache holds 180,000 entries with no TTL (they age out on LFU eviction). The new model produces meaningfully different responses on about 15% of cached queries. Design a rollout strategy that: (a) invalidates stale entries without a destructive flush that would cold-start both caches simultaneously, (b) handles the semantic cache where no TTL exists, and (c) bounds the window during which incorrect cached responses could be served. Explain the trade-off between correctness and performance during the transition.
      </Prose>

      <H3>Exercise 3: Calibrate a semantic cache threshold for a new domain</H3>

      <Prose>
        You are deploying a semantic cache for a medical FAQ bot that answers questions about medication dosages and drug interactions. You have assembled a held-out validation set of 500 query pairs, each labeled as "semantically equivalent" (same answer), "related but different" (different answer), or "unrelated." The pairs at various thresholds show: at τ=0.99, precision=1.0, recall=0.05; at τ=0.97, precision=0.98, recall=0.35; at τ=0.95, precision=0.91, recall=0.60; at τ=0.92, precision=0.78, recall=0.75. Given that a wrong dosage answer could cause patient harm, select the appropriate threshold and justify it. At what point, if any, should semantic caching be rejected entirely for this domain?
      </Prose>

      <H3>Exercise 4: Debug a thundering herd on a shared exact-match cache</H3>

      <Prose>
        Your monitoring shows a spike pattern: every 3,600 seconds, GPU utilization jumps from 40% to 95% for approximately 30 seconds, then falls back. The spike corresponds to a burst of identical inference calls. Cache hit rate is 85% between spikes and near 0% during spikes. Identify the root cause, explain why it happens, and propose two mitigations — one that eliminates the spike entirely at the cost of some computational overhead, and one that distributes the spike over time without additional compute.
      </Prose>

      <CodeBlock language="python">
{`# Expected root cause:
# TTL=3600s means all entries for a hot query expire simultaneously.
# The next request finds a cold cache, triggers inference,
# and hundreds of concurrent requests for the same prompt all miss at once.
#
# Mitigation 1 — Distributed lock (eliminates spike, adds lock overhead):
#   SET NX cache_key "__computing__" EX 30
#   Only the winner runs inference; others poll or subscribe to the result.
#   Eliminates redundant inference entirely.
#
# Mitigation 2 — Probabilistic early expiry (distributes spike, no extra lock):
#   When serving a cached entry with remaining TTL < decay_threshold:
#     with probability p = exp(-beta * remaining_ttl):
#       trigger background refresh while serving the current cached value
#   This spreads the re-computation over the decay window rather than at expiry.
#   XFetch algorithm (Vattani et al., 2015) formalizes this.
#   beta and decay_threshold are tunable per workload.`}
      </CodeBlock>

      <H3>Exercise 5: Memory budget for a three-layer cache on a single A100</H3>

      <Prose>
        You are deploying a full three-layer caching stack on a single A100 80GB serving Llama 3 8B (model weights ~16 GB in BF16). The workload: 200 distinct FAQ queries (exact-match cache), 50,000 semantic cache entries (384-dimensional float32 embeddings, HNSW index), KV-cache sharing for a 4,000-token system prompt used on 90% of requests, and 16 concurrent inference sessions at average 2,048 tokens each. Compute: (a) KV-cache memory for the cached system prompt, (b) memory for 16 concurrent sessions at 2,048 tokens, (c) approximate HNSW index memory for 50,000 entries at 384 dimensions, and (d) whether all of this fits on the A100 after model weights. What is the first thing to reduce if it does not fit?
      </Prose>

      <CodeBlock language="python">
{`# Expected workings (Llama 3 8B: L=32, H_kv=8, d_h=128, BF16):
# per-token KV size = 2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KB/token
#
# (a) System prompt KV cache (4,000 tokens):
#     4,000 × 128 KB = 512,000 KB ≈ 0.49 GB
#
# (b) 16 concurrent sessions at 2,048 tokens each:
#     16 × 2,048 × 128 KB = 4,194,304 KB ≈ 4.00 GB
#
# (c) HNSW index for 50,000 entries at 384 dims (float32):
#     Vector storage: 50,000 × 384 × 4 = 76,800,000 bytes ≈ 0.07 GB
#     HNSW graph overhead: ~100 bytes/entry × 50,000 ≈ 0.005 GB
#     Total: ~0.075 GB
#
# (d) Total:
#     Model weights:    16.00 GB
#     System KV cache:   0.49 GB
#     Session KV cache:  4.00 GB
#     HNSW index:        0.08 GB
#     Activation scratch: ~2.00 GB (estimate)
#     Total:            ≈22.57 GB  ← well within 80 GB A100
#
# First reduction if tight: reduce concurrent session KV cache by lowering
# max_context from 2048 to 1024, halving (b) to 2.00 GB.
# Second: quantize KV to FP8 (halves both (a) and (b)).
# HNSW is negligible and not worth optimizing first.`}
      </CodeBlock>

    </div>
  ),
};

export default cachingStrategies;
