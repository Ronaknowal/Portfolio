import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";

const requestRoutingLB = {
  title: "Request Routing & Load Balancing",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        Classic load balancing distributes requests evenly across a pool of identical workers. Every instance is equivalent, so the only meaningful optimization is utilization. Round-robin, least-connections, weighted random — the algorithms differ, but they share an assumption: a request handled by instance A has the same cost as the same request handled by instance B. That assumption holds for HTTP servers and database replicas. It does not hold for LLM instances.
      </Prose>

      <Prose>
        LLM load balancing inverts the problem. The KV cache on each instance makes instances non-identical. A session's prompt prefix has already been computed and stored on the instance that served the previous turn; sending the next turn to a different instance means paying for a full prefill that the first instance could have answered from cache in milliseconds. Routing becomes a multi-objective problem: balance load AND maximize cache hits AND respect tail latency SLOs. The algorithms that work well differ substantially from what webservers need, and the tradeoffs are much less forgiving.
      </Prose>

      <H2>Round-robin is wrong here</H2>

      <Prose>
        The classic stateless approach assigns each incoming request to the next instance in a rotating sequence. Works perfectly when every request is identical work — static file serving, read-only database queries, stateless API calls. For LLMs, the cost structure is different in a way that makes round-robin actively harmful.
      </Prose>

      <Prose>
        Suppose a user sends turn one of a conversation to instance A. Instance A runs prefill, stores the KV cache, generates the response. The user replies — turn two. Round-robin hands it to instance B. Instance B has never seen this conversation; it must run a full prefill over the entire conversation history to reconstruct the context that instance A already holds in cache. That prefill takes roughly two seconds for a typical chat context. Instance A could have answered in milliseconds with a cache hit. The routing decision just spent two seconds of GPU time for no reason.
      </Prose>

      <Prose>
        In agentic workloads — multi-step tool calls, long-running tasks with repeated context — this waste compounds. An agent loop that sends ten turns to ten different instances pays full prefill ten times. Measured against a cache-aware router, naive round-robin wastes 30–60% of all prefill compute in agent-heavy traffic. That is not a small optimization opportunity; it is a category error in the routing architecture.
      </Prose>

      <H2>Session affinity</H2>

      <Prose>
        The simplest fix is session affinity: all turns from the same conversation go to the same instance. Hash the session ID or user ID, map it deterministically to one instance, and the KV cache on that instance stays warm for the entire conversation. No cross-instance prefill waste. Implementation is a few lines.
      </Prose>

      <CodeBlock language="python">
{`def session_affinity_route(request, instances):
    """All turns of the same conversation pin to one instance."""
    session_id = request.session_id
    # Consistent-hash to an instance; survives adding/removing instances.
    return consistent_hash(session_id, instances)`}
      </CodeBlock>

      <Prose>
        Consistent hashing is important here. Naive modulo hashing — <Code>instance_id = hash(session_id) % len(instances)</Code> — reshuffles most sessions whenever the instance pool changes. Consistent hashing arranges instances on a hash ring; adding or removing one instance only displaces the sessions it was directly responsible for, leaving the rest undisturbed. A deployment event that would invalidate fifty percent of cache affinity with modulo hashing invalidates one or two percent with consistent hashing.
      </Prose>

      <Prose>
        Session affinity solves the chat-session case cleanly. Its limitation is equally clean: it only captures cache reuse within a single session. Two different users sending the same system prompt to two different instances both pay full prefill, because the session IDs differ. For a product where every API request carries a 2,000-token system prompt and ten different users are active simultaneously, session affinity leaves most of the cache opportunity on the table.
      </Prose>

      <H2>Prefix-aware routing</H2>

      <Prose>
        The generalization hashes on the prompt prefix rather than the session identity. Two requests that share a long common prefix — a system prompt repeated across every call to a tenant, a RAG context block shared across a batch, a few-shot example block reused across a test suite — will land on the same instance and both get a prefix cache hit. Session boundary no longer matters; prefix identity does.
      </Prose>

      <CodeBlock language="python">
{`import hashlib

def prefix_route(request, instances, prefix_tokens=256):
    """Route by the hash of the first N prompt tokens."""
    prefix = tuple(request.prompt_tokens[:prefix_tokens])
    h = int(hashlib.sha1(str(prefix).encode()).hexdigest(), 16)
    return instances[h % len(instances)]

# Refinement: consistent hashing so instance membership changes don't
# shuffle everything. Each instance owns a range of the hash ring.`}
      </CodeBlock>

      <Prose>
        The choice of prefix length matters. Too short — say, 32 tokens — and different prompts that happen to start identically collide without sharing enough context to make a cache hit useful. Too long — the full prompt — and routing misses requests that share most of their prefix but differ at the end. In practice, 128 to 512 tokens covers most shared-prefix patterns: system prompts, RAG headers, few-shot blocks. The prefix length is a tunable hyperparameter of the router, not the model.
      </Prose>

      <Prose>
        Prefix-aware routing also needs consistent hashing. A system prompt shared across a thousand concurrent requests is routing all thousand to one instance — which is exactly what you want for cache hits, but means that instance restarts scatter all that traffic indiscriminately without consistent hashing to guide the fallback.
      </Prose>

      <H2>The overload problem</H2>

      <Prose>
        Prefix-aware routing has a structural failure mode: it concentrates load. A popular prefix — a shared system prompt used by every request to a high-traffic tenant — routes all of that traffic to a single instance. That instance's KV cache fills, its request queue grows, and tail latency spikes. At some point the instance becomes the bottleneck for all traffic in that prefix group. The cache hit that made it attractive is now the reason it is overloaded.
      </Prose>

      <Prose>
        The standard response is load shedding with replica fallback. When the primary instance for a prefix group exceeds a queue-depth or latency threshold, the router stops sending it new requests from that group and redirects to a replica. The replica pays full prefill cost for those requests — it has no cached prefix — but keeps latency acceptable. The primary drains, recovers, and takes traffic again. This is a deliberate degradation: you trade cache efficiency for stability, and you need explicit logic to make that trade at the right time.
      </Prose>

      <Prose>
        Choosing the threshold is non-trivial. Too conservative — shed at the first sign of queue depth — and you never build the cache warm enough to matter. Too aggressive — hold on until latency is already spiking — and you expose users to the tail before the fallback activates. Production systems typically monitor both queue depth and p95 latency per instance and shed when either crosses a threshold, with the thresholds tuned to the latency SLO of the deployment.
      </Prose>

      <H3>Power-of-two choices</H3>

      <Prose>
        A classical load-balancing trick adapts well here. Instead of sending every request to the single best instance for a given prefix, the router identifies a small set of candidate instances with acceptable cache affinity and picks the least-loaded among them. Two candidates is the classic version; for prefix-aware routing, three to five gives enough room to avoid the overloaded primary without fully abandoning affinity.
      </Prose>

      <CodeBlock language="python">
{`def power_of_two_with_affinity(request, hash_ring, k=3):
    """Consider k cache-affinity candidates, pick the least-loaded."""
    candidates = hash_ring.top_k_instances(request.prefix_hash, k)
    # Each instance publishes its current queue_depth via heartbeats.
    return min(candidates, key=lambda i: i.queue_depth)`}
      </CodeBlock>

      <Prose>
        The key insight is that "best affinity" and "best load" are not the same instance under pressure, and choosing only on affinity produces a hot-spot. Choosing among the top-k affinity candidates by load means you almost always get a good cache hit — the primary is usually lightly loaded — and gracefully degrade when it is not, without needing a separate overload-detection system. Power-of-two choices is not as optimal as a full backpressure-aware routing system, but it is dramatically simpler and captures most of the win.
      </Prose>

      <H3>Radix routing — SGLang's contribution</H3>

      <Prose>
        SGLang's router takes prefix-aware routing further by maintaining an explicit radix tree of token prefixes across all instances in the pool. Every block of tokens cached on every instance is registered in the tree. When a new request arrives, the router walks the tree to find the instance with the longest matching prefix — not a hash approximation of prefix similarity, but an exact longest-prefix-match over the actual cached token sequences.
      </Prose>

      <Prose>
        The practical difference is significant. Hash-based prefix routing bins requests into buckets — requests whose first 256 tokens hash to the same value land together, but requests sharing 512 tokens might hash differently and miss each other. The radix tree finds the actual longest match regardless of where the divergence happens. Agentic workloads, where successive requests share progressively growing prefixes as the agent accumulates context, benefit disproportionately: each new turn matches more of the prefix than the last, and the tree finds that longer match where a fixed-length hash would not. SGLang reports effective cache hit rates of 60–80% on agentic workloads, compared to 30–40% for fixed-length hashed prefix routing on the same traffic.
      </Prose>

      <Prose>
        The cost is state. The radix tree needs to track the actual cached contents of every instance — which means the routing layer has a distributed consistency problem as instances add and evict cache blocks. When an instance evicts a prefix block to make room for new traffic, the tree must be updated or routing decisions based on stale state will send requests to instances that no longer have the prefix cached. Production implementations handle this with periodic heartbeats carrying cache summaries, accepting small windows of staleness in exchange for manageable coordination overhead.
      </Prose>

      <H2>Global queue vs. per-instance queue</H2>

      <Prose>
        A deeper architectural choice underlies all of the above. When a request arrives at the router, it can be committed to a specific instance's queue immediately — Option A, per-instance queues — or it can sit in a global queue and be pulled by whichever instance becomes available — Option B, global queue. The routing algorithms described so far are all Option A variants: the router decides which instance handles the request at arrival time.
      </Prose>

      <Prose>
        Per-instance queues give the router maximum control over cache affinity. The decision is made once, immediately, with full information about prefix similarity. Cache-warm instances get the requests they are best positioned to serve. The downside is head-of-line blocking: if the chosen instance is temporarily slow — a long prefill on a large request, a GC pause, a memory pressure event — requests behind it in the queue wait even if other instances are idle. The queue depth the router observes via heartbeats is always slightly stale, so it is possible to route into a queue that has grown since the last heartbeat.
      </Prose>

      <Prose>
        A global queue gives perfect load balance by construction. Instances pull when they are ready; no instance sits idle while another is overloaded. Backpressure is natural — when all instances are busy, the global queue grows and new requests wait, with no possibility of a single instance becoming a hot spot. The downside is lost affinity: an instance pulls the next request from the queue without any preference for requests that match its cached prefixes. Cache hits become incidental rather than engineered.
      </Prose>

      <Callout accent="gold">
        Every LLM routing design is a tradeoff between cache affinity and load balance. You cannot maximize both simultaneously; the right point on the curve depends on your workload.
      </Callout>

      <Prose>
        Most production systems blend the two. The primary path is affinity-first with per-instance queues — prefix-aware routing, consistent hashing, power-of-two choices. When per-instance queues exceed a depth threshold, overflow requests enter a global pool and are claimed by the first instance with capacity. The global pool is the escape valve, not the primary routing path. Affinity-first with global-queue overflow captures most of the cache efficiency of pure Option A and most of the load-balance stability of pure Option B.
      </Prose>

      <H3>Anycast and geo-routing</H3>

      <Prose>
        Layered above instance-level routing is geographic routing: directing requests to the nearest datacenter before per-instance routing happens at all. For interactive chat, the datacenter choice dominates first-byte latency — a transatlantic round trip adds 80–150 ms to every response even before a token is generated. DNS-based geo-routing is the standard approach at the edge, where a request's source IP determines which datacenter's IP the DNS resolver returns. Requests reach the nearest point-of-presence, then enter that region's instance-level routing.
      </Prose>

      <Prose>
        For long-running inference — large batch jobs, multi-minute agent tasks — datacenter proximity matters much less. When the inference itself takes thirty seconds, the 100 ms routing overhead is noise. What matters more for those workloads is datacenter capacity and instance availability, not latency. Some systems run a hybrid: interactive traffic uses strict geo-routing for latency; batch traffic uses a global scheduler that places jobs wherever GPU capacity is available, sometimes crossing region boundaries to avoid idle hardware.
      </Prose>

      <Prose>
        Within each region, geo-routing hands off to the cache-aware routing stack described above. The two layers are largely independent: geo-routing is stateless and DNS-based; instance routing is stateful and prefix-aware. The only interaction point is cache warming after failover — if geo-routing redirects traffic from one datacenter to another during an incident, the receiving region's instances are cold for all the prefixes the failed region was serving, and the first wave of redirected traffic pays full prefill regardless of how good the instance routing is.
      </Prose>

      <H2>Putting it together</H2>

      <Prose>
        Routing and load balancing are the least glamorous, highest-leverage infrastructure work in an LLM serving stack. The model weights are fixed; the training is done; the hardware is purchased. What remains is the question of how efficiently that hardware converts incoming requests into completed responses. A 2× improvement in cache hit rate via better routing often beats kernel-level inference optimizations in practical cost impact, because it reduces prefill compute across the entire fleet rather than shaving milliseconds off individual decode steps.
      </Prose>

      <Prose>
        The progression from round-robin to session affinity to prefix-aware routing to radix-tree routing is a progression from ignoring cache structure entirely to exploiting it as precisely as the routing layer can afford. Each step trades routing complexity for compute savings, and each step has a regime where it is the right choice: round-robin for truly stateless workloads, session affinity for simple chat products, prefix hashing for multi-tenant APIs with shared system prompts, radix routing for agentic pipelines where long-context reuse is the primary cost driver.
      </Prose>

      <Prose>
        The next topic covers what happens when routing is not enough — when cache efficiency is already high but request volume exceeds what the current instance pool can handle, and you need to scale the pool itself. Autoscaling an LLM fleet has its own set of constraints that differ from conventional services, starting with the fact that adding a new instance to the pool means the routing layer needs to warm its cache before it can serve traffic efficiently.
      </Prose>
    </div>
  ),
};

export default requestRoutingLB;
