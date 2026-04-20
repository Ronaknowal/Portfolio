import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";

const cachingStrategies = {
  title: "Caching Strategies (Semantic, Exact, KV-Cache Sharing)",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        Production LLM serving runs three different caches side by side. Each catches a different kind of repeated work, and none of them alone is sufficient. Exact-match caches eliminate redundant inference on identical requests. KV-cache sharing eliminates redundant prefill on shared token prefixes. Semantic caches eliminate redundant inference on paraphrased requests — a different and more treacherous problem. Understanding which cache catches which traffic, where each one fails, and where the boundaries interact is one of the more useful frames for thinking about LLM infrastructure at scale.
      </Prose>

      <H2>The three caches</H2>

      <Prose>
        The table below captures the core tradeoffs. Hit rate and hazard are not independent: the caches that catch the most traffic tend to carry the most risk, and the one with zero risk catches the least traffic. They serve non-overlapping request populations and fail in non-overlapping ways. A production stack that wants to capture all three traffic types runs all three, layered, with explicit rules about which layer gets checked first and how a miss falls through.
      </Prose>

      <CodeBlock>
{`cache          key                       hit rate   hazard                      latency savings
exact match    hash(model, prompt, args)  <5%        none                        100% (skip inference)
prefix / KV    hash of token prefix       30-60%     tenant isolation needed     up to 90% of prefill
semantic       embedding similarity       5-30%      can silently mismatch       100% (but on wrong answer)`}
      </CodeBlock>

      <Prose>
        The ordering matters. Exact-match should be checked first — it is the cheapest lookup (a single hash) and the safest hit (the response is byte-for-byte correct). Prefix/KV is handled inside the inference engine and fires automatically; it requires no separate check but does require cache-aware routing so requests with shared prefixes land on the same worker. Semantic is checked last and with the most skepticism, because its hits are probabilistic rather than exact.
      </Prose>

      <H2>Exact-match cache</H2>

      <Prose>
        The simplest possible cache: hash the full request — model, prompt, sampling parameters — and look up the stored response. On a hit, return the stored bytes and skip inference entirely. There is no token computation, no model forward pass, no network call to a GPU worker. The cost per hit is effectively the cost of a Redis or Memcached lookup. The downside is narrow coverage. Literal repetition is rare in consumer-facing workloads; users rephrase, conversation history changes, and even identical questions arrive in different contexts. The hit rate is typically below five percent of total traffic.
      </Prose>

      <Prose>
        Where exact-match earns its keep is on deterministic, idempotent endpoints: classification APIs with <Code>temperature=0</Code>, batch pipelines that run the same prompt schema against many inputs, internal tooling that re-calls an LLM with fixed prompts across retries or retriggers. On those workloads, the hit rate approaches one hundred percent after warm-up, and the cost per request drops to near zero. The TTL should be tight enough to invalidate on model updates — more on that below.
      </Prose>

      <CodeBlock language="python">
{`import hashlib
import redis

def exact_match_key(model, prompt, params):
    # Stable hash of everything that affects output
    key_data = f"{model}|{prompt}|{sorted(params.items())}"
    return hashlib.sha256(key_data.encode()).hexdigest()

async def serve(request, redis_client, fallback):
    key = exact_match_key(request.model, request.prompt, request.params)
    cached = await redis_client.get(key)
    if cached:
        return cached  # full-response hit
    response = await fallback(request)
    # Only cache if deterministic (temperature=0 and no seed randomness)
    if request.params.get("temperature") == 0:
        await redis_client.setex(key, ttl=3600, value=response)
    return response`}
      </CodeBlock>

      <Prose>
        The guard on <Code>temperature=0</Code> is important. Caching a stochastic response and replaying it breaks the probabilistic contract the caller expects — every "new" call returns the same sample, which is incorrect for workloads that rely on generation diversity. Exact-match cache is only correct for fully deterministic inference. If your endpoint supports non-zero temperature, skip the write.
      </Prose>

      <H2>Prefix / KV cache</H2>

      <Prose>
        The KV-cache and prefix-caching topic in this section covers the mechanics in depth — how block hashing works, how page tables enable block sharing across requests, and how eviction under memory pressure behaves. For system design purposes, treat prefix caching as a given feature of the inference engine: when two requests share a common token prefix, the engine reuses the computed KV blocks for that prefix rather than recomputing them. The prefill cost for the shared portion drops to zero.
      </Prose>

      <Prose>
        From an infrastructure perspective, prefix caching has three requirements that are architectural rather than mechanical. First, cache-aware routing: requests with shared prefixes must land on the same worker, because KV blocks live in GPU memory local to that worker. A load balancer that distributes requests uniformly across workers destroys prefix-cache hit rates by splitting shared-prefix traffic across machines. Second, memory budget decisions at the worker level: the KV cache competes for GPU memory with model weights and activations, and the serving stack must decide how much of that budget to dedicate to cached blocks versus live inference. Third, per-tenant isolation at the router boundary, which is a security requirement covered in the tenant isolation section below. None of these is handled automatically; all three require explicit choices in how the serving stack is deployed.
      </Prose>

      <H2>Semantic cache — the controversial one</H2>

      <Prose>
        Semantic caching extends the exact-match idea to near-matches. Embed the incoming prompt; find the nearest stored prompt by cosine similarity in a vector database; if the similarity exceeds a threshold, return the cached response without running inference. The appeal is real: "what's your return policy" and "how do I return something I bought" carry the same intent and should share an answer. FAQ-style traffic, customer support bots, and narrow-domain Q&A systems often have large clusters of semantically equivalent queries that exact-match and prefix caching will never consolidate.
      </Prose>

      <Prose>
        The hazard is equally real and harder to see. Embedding similarity does not guarantee semantic equivalence. "What's the capital of France" and "What's the capital of French Polynesia" embed close enough to share a cache entry at many practical thresholds, but their correct answers are Paris and Papeete. "Is the bridge open today" and "Is the bridge closed today" are antonyms that embed nearly identically. Any threshold low enough to catch useful paraphrases will also serve wrong answers for edge cases — and the failure mode is silent. The caller receives a response that looks fully formed and authoritative. There is no error, no flag, no indication that the answer came from a different question. Debugging silent mismatch at scale is substantially harder than debugging a cache miss.
      </Prose>

      <CodeBlock language="python">
{`def semantic_cache_lookup(prompt, embedding_model, vector_store, threshold=0.95):
    """Return cached response if a sufficiently similar prompt exists."""
    query_embedding = embedding_model.embed(prompt)
    result = vector_store.similarity_search(query_embedding, k=1)
    if result and result.similarity > threshold:
        return result.response
    return None

# Threshold tuning: 0.99 → near-duplicate only (safe but low hit rate)
#                   0.90 → catches paraphrases (risky; can serve wrong answers)`}
      </CodeBlock>

      <Prose>
        The threshold is a dial between correctness and hit rate with no good universal setting. At 0.99, you are catching near-duplicate strings — minor whitespace or punctuation differences — and the semantic cache adds little over exact-match. At 0.90, you catch real paraphrases but you also catch adversarial near-misses. Monitoring must be active: log every cache hit, sample them for correctness, and watch for clusters of hits that receive complaints or low feedback scores. Semantic cache without monitoring is the most dangerous configuration in the stack.
      </Prose>

      <H2>When each cache is a good bet</H2>

      <Prose>
        Rough practitioner guidance, not universal rules:
      </Prose>

      <Prose>
        <strong>Exact-match</strong> is always worth enabling on deterministic endpoints. The implementation is ten lines and a Redis instance. The hit rate may be low, but every hit is free inference, and the cost of a miss is just the lookup latency — a millisecond or two. Add a TTL short enough to invalidate on model updates. There is no meaningful downside.
      </Prose>

      <Prose>
        <strong>Prefix / KV caching</strong> is essential for any agentic or multi-turn chat workload. If your average request includes a system prompt longer than a few hundred tokens, or your agent framework prepends tool definitions to every call, prefix caching pays for itself immediately. The prerequisite is a router that is aware of which worker holds which prefix's KV blocks — a uniform round-robin router defeats it entirely. Make sure the routing layer sends prefix-sharing requests to the same worker cohort.
      </Prose>

      <Prose>
        <strong>Semantic caching</strong> is only appropriate for narrow, closed-domain FAQ-style traffic where you can control the answer corpus, audit cache entries manually, and monitor hit quality in production. A customer support bot with a fixed product FAQ is a reasonable candidate. An open-ended generative assistant, a coding helper, or any system where factual precision matters is not. When in doubt, skip it — the latency savings do not justify the correctness risk in most production systems.
      </Prose>

      <H3>Cache coherence — the forgotten problem</H3>

      <Prose>
        Cached responses age. Your model changes — fine-tunes, version upgrades, safety-filter updates — and cached responses from the old model are suddenly wrong. Exact-match cache entries for a classification endpoint that was calibrated on one model will return the old model's outputs when served by the new one. Semantic cache entries are wrong about product details the moment the product changes. Prefix KV caches on the inference engine are less affected — they store intermediate activations, not final answers — but exact-match and semantic caches hold final text responses that can become stale.
      </Prose>

      <Prose>
        Cache invalidation must be an explicit part of your release pipeline. The most reliable pattern is a versioned cache key: every cache read and write includes the model version (and optionally a data-version tag for semantic caches). A model deployment bumps the version, which changes the key prefix, which effectively invalidates the entire prior cache without a destructive flush operation. Old entries age out on their TTL; new entries accumulate under the new key. The transition period where old and new entries coexist is short and benign. This is cheaper to implement than it sounds and much cheaper than debugging a production incident caused by stale cached answers serving at scale.
      </Prose>

      <Callout accent="gold">
        Every cache layer is a trade of storage for latency. Semantic caches also trade correctness for latency. Always know which trade you're making.
      </Callout>

      <H3>Tenant isolation</H3>

      <Prose>
        Caches that cross tenant boundaries leak information. The mechanism is subtle but real. If two tenants share the same exact-match or semantic cache namespace, a cache hit on tenant B's request reveals that tenant A issued a semantically identical request recently — and the response tenant B receives was generated in response to A's context, not B's. In some cases the leak is benign; in others it is a serious data exposure. Tenant A's proprietary system prompt, internal document, or customer data becomes implicitly visible through the responses that cache under its associated keys.
      </Prose>

      <Prose>
        Even without response leakage, timing side channels exist. A request that hits the cache returns faster than one that misses. An adversary with the ability to send probing requests and observe response latency can infer whether a given prompt was recently submitted by another tenant. This is a weaker attack than direct content exposure, but it is a real information leak in high-sensitivity deployments.
      </Prose>

      <Prose>
        The fix is straightforward and non-negotiable: partition every cache layer per tenant, where tenant is defined at the same boundary as your other access controls — per API key, per organization ID, or per user depending on your isolation model. Prefix it into the cache key for exact-match and semantic layers; use separate cache namespaces for KV prefix caching at the inference engine level. Most commercial providers do this by default. If you operate your own serving stack, it is a design requirement from day one, not an optimization to add later.
      </Prose>

      <Prose>
        Caching is where LLM serving meets classic distributed systems: the same problems — coherence, invalidation, multi-tenancy — with LLM-specific hazards layered on top. The exact-match cache is a solved problem from web serving; semantic caching is genuinely new and genuinely risky; KV-cache sharing at the inference engine level is the most impactful optimization in modern serving stacks. Running all three correctly means knowing which cache catches which traffic, instrumenting each layer, and treating correctness as a first-class constraint alongside latency. The next topic looks at a related capacity-management problem — serving multiple models behind one endpoint.
      </Prose>
    </div>
  ),
};

export default cachingStrategies;
