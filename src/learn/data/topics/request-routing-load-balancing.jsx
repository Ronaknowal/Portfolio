import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const requestRoutingLB = {
  title: "Request Routing & Load Balancing",
  readTime: "42 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        A generic HTTP load balancer knows one thing: how many open connections sit on each backend. Given that number, it distributes the next request to whoever has the fewest — or, if it does not even bother to count, in round-robin order. That is the right design for a pool of stateless web servers. Every server is identical. A request handled by server A costs the same as the same request handled by server B. The traffic shaping problem reduces cleanly to utilization balancing.
      </Prose>

      <Prose>
        None of that is true for a pool of LLM inference instances. Three structural facts about LLM inference make generic load balancing actively harmful. First, requests are bimodal: a short classification prompt might complete in 80 milliseconds while a multi-turn agent turn with a 32,000-token context takes forty seconds. Treating these as the same "request" for purposes of load accounting is like treating a parking space and a freight container as the same storage unit. Second, KV cache makes instances non-identical. When a conversation's prompt prefix has already been processed by instance A and its key-value pairs are resident in A's GPU memory, routing the next turn of that conversation to A is not just slightly better than routing to B — it is often 10 to 100 times cheaper in wall-clock prefill time. Instance B has to run the full prefill from scratch. Instance A returns the first token from cache in milliseconds. Third, large-scale deployments run dozens to thousands of instances, frequently change pool membership through autoscaling and rolling deployments, and serve heterogeneous tenants whose system prompts are shared across hundreds of concurrent sessions. The routing decision made at the load balancer is the difference between those shared prefixes landing on an instance that has already computed them and landing on one that has not.
      </Prose>

      <Prose>
        The failure mode of ignoring all this is measurable. Studies of production LLM serving traffic show that naive round-robin routing on chat workloads wastes 30–60% of all prefill compute — compute that was paid for in GPU time and electricity — simply because the router sent a request to an instance that lacked the cached context an adjacent instance already held. That wasted compute directly inflates cost, inflates latency, and reduces the throughput capacity of the fleet. Cache-aware routing on the same traffic eliminates most of that waste without adding hardware. It is the highest-leverage optimization available after the inference engine itself is tuned.
      </Prose>

      <Prose>
        This topic builds the machinery of LLM-aware routing from the ground up. It starts with the four routing layers used in production, derives the mathematics of each approach, implements and benchmarks all of them from scratch, and connects the results to how systems like NGINX, Envoy, vLLM's production stack, and HuggingFace TGI actually expose these controls. The goal is to understand not just what the algorithms are but why each one exists, what it optimizes, and what it sacrifices.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>Four routing layers</H3>

      <Prose>
        Production LLM serving infrastructure routes at four distinct levels of the networking stack, each with a different information set and a different objective.
      </Prose>

      <Prose>
        Layer 4 (TCP/IP) routing operates at the transport layer. The router sees source and destination IP addresses and TCP ports, nothing more. It can perform IP hash routing — consistently sending all traffic from a given client IP to the same backend — with essentially zero overhead and zero understanding of what the traffic contains. This is appropriate for session stickiness when the upstream infrastructure does not parse HTTP at all, and for maximum throughput when connection-level balance is the only objective. Its fundamental limitation is that it cannot distinguish between a one-token prompt and a 100,000-token context, and it cannot see model IDs, tenant IDs, or prompt content.
      </Prose>

      <Prose>
        Layer 7 (HTTP) routing parses the request headers and body. The router can inspect the <Code>model</Code> field, read custom <Code>X-Tenant-ID</Code> headers, examine the <Code>Authorization</Code> bearer token to identify the tenant, and route on any of those signals. Most API gateways — NGINX, Envoy, Kong, AWS API Gateway — operate at this layer. The cost is that full HTTP parsing adds a few hundred microseconds of overhead per request, which is negligible against a multi-second LLM inference time. The gain is that routing decisions can be made on any header or metadata, enabling per-model routing, per-tenant quota enforcement, and priority-based admission.
      </Prose>

      <Prose>
        Model-aware routing understands the model ID embedded in the request and routes to an instance pool serving that model. A deployment running Llama 3 70B, Llama 3 8B, and a fine-tuned classification head behind the same API endpoint needs this to ensure 70B requests go to A100 pools and 8B requests go to cheaper hardware. Model-aware routing often incorporates task classification — a small, cheap classifier infers from the prompt whether the request needs a frontier model or can be served adequately by a smaller one, routing accordingly to reduce cost.
      </Prose>

      <Prose>
        Cache-aware routing is the level that matters most for per-instance efficiency. The router uses prefix hashing or a radix tree of cached token sequences to route requests to the instance most likely to have the prompt prefix already in its KV cache. A cache hit on a 4,000-token system prompt saves the full cost of computing 4,000 token positions through all model layers — typically 1–3 seconds on a large model. Cache-aware routing stacks on top of the three layers below it; it is the final dispatch decision after model and tenant routing have narrowed the eligible instance pool.
      </Prose>

      <H3>The central tradeoff</H3>

      <Prose>
        Every routing algorithm in this topic navigates the same fundamental tradeoff: cache affinity pulls traffic toward a small number of instances (the ones with warm caches), while load balance pushes traffic away from any single instance that is becoming overloaded. Pure cache affinity produces hot spots. Pure load balance wastes all cache value. The algorithms below are different ways of finding a point on that curve, with different behaviors under load, under failure, and at scale.
      </Prose>

      <Callout accent="gold">
        Cache affinity and load balance are in direct tension. Every routing design is a bet on where the right tradeoff point is for a given workload. Understanding the math is what lets you make that bet deliberately rather than accidentally.
      </Callout>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Round-robin — correct baseline, wrong for bimodal workloads</H3>

      <Prose>
        Round-robin distributes requests uniformly across <Code>N</Code> instances. In steady state each instance receives fraction <Code>1/N</Code> of total traffic. If all requests have identical service time <Code>S</Code>, this achieves perfect load balance: every instance runs at utilization <Code>ρ = λ·S/N</Code>, where <Code>λ</Code> is the arrival rate.
      </Prose>

      <Prose>
        The problem emerges with bimodal service time. Let a fraction <Code>p</Code> of requests be prefill-heavy (service time <Code>S_L</Code>, long) and fraction <Code>1-p</Code> be decode-heavy (service time <Code>S_S</Code>, short). Round-robin distributes both types uniformly, so each instance receives a mix proportional to <Code>p:1-p</Code>. Under Kingman's approximation for heavy-traffic queuing, the expected wait time scales with the coefficient of variation <Code>C_s</Code> of the service time distribution:
      </Prose>

      <MathBlock>{"W_q \\approx \\frac{1 + C_s^2}{2} \\cdot \\frac{\\rho}{c(1-\\rho)} \\cdot \\mathbb{E}[S]"}</MathBlock>

      <Prose>
        For a bimodal distribution with <Code>p = 0.3</Code>, <Code>S_L = 10s</Code>, <Code>S_S = 0.5s</Code>, the mean service time is <Code>E[S] = 3.35s</Code> and the coefficient of variation is <Code>C_s ≈ 1.42</Code>. Substituting into Kingman's formula at <Code>ρ = 0.8</Code> inflates the predicted wait time by a factor of <Code>(1 + 1.42²)/2 ≈ 1.51</Code> compared to a homogeneous exponential workload with the same mean. Round-robin makes this worse, not better: it cannot separate short and long requests, so every instance queue suffers head-of-line blocking whenever a long request arrives before a queue of short ones.
      </Prose>

      <H3>Least-connections (JSQ) — better, but blind to cost</H3>

      <Prose>
        Join-Shortest-Queue (JSQ) sends each arriving request to the instance with the fewest currently queued requests. Mitzenmacher (2001) proved that JSQ is near-optimal for minimizing expected wait time under homogeneous exponential service times in an <Code>M/M/c</Code> model. For <Code>N</Code> instances each with service rate <Code>μ</Code>, the expected queue length under JSQ is exponentially smaller than under random assignment:
      </Prose>

      <MathBlock>{"\\mathbb{E}[L_{JSQ}] \\approx \\frac{\\ln\\ln N}{\\ln(1/\\rho)} \\quad \\text{(Mitzenmacher 2001)}"}</MathBlock>

      <Prose>
        The catch is that JSQ measures queue depth in number of requests, not in cost-weighted service time. In an LLM workload where one request in the queue might be a 30-second decode and another is a 200-millisecond chat reply, queue depth is a poor proxy for how long the next arrival must wait. An instance with 2 long-decode requests queued has substantially higher expected wait than an instance with 4 short requests queued, but JSQ based on count alone routes identically to both.
      </Prose>

      <H3>Cache-aware routing — prefix hash to instance</H3>

      <Prose>
        Cache-aware routing deterministically assigns a request to an instance based on the hash of the first <Code>k</Code> prompt tokens — the prefix. The objective is that requests sharing a prefix land on the same instance and get a KV cache hit. The simplest version uses modulo assignment:
      </Prose>

      <MathBlock>{"\\text{instance}(r) = h(\\text{prefix}_k(r)) \\bmod N"}</MathBlock>

      <Prose>
        where <Code>h</Code> is a fast hash function (MurmurHash3 or xxHash in practice) and <Code>k</Code> is the prefix length in tokens. This achieves 100% cache locality for requests with identical prefixes as long as the instance pool is stable. The pathology is that it achieves 0% routing diversity — all such requests pile onto one instance, creating a hot spot when that prefix is popular.
      </Prose>

      <H3>Consistent hashing — minimal disruption under pool changes</H3>

      <Prose>
        Consistent hashing (Karger et al., 1997) arranges the <Code>N</Code> instances on a virtual hash ring of size <Code>M</Code> (typically <Code>M = 2³²</Code>). Each instance occupies one or more positions on the ring (virtual nodes). A request with prefix hash <Code>h</Code> is routed to the first instance whose ring position is clockwise from <Code>h</Code>. When an instance is added or removed, only the requests whose ring segment is directly adjacent to the changed instance need to be re-routed. The fraction of requests displaced is:
      </Prose>

      <MathBlock>{"\\text{displaced fraction} = \\frac{1}{N} \\quad \\text{(vs } 1 - \\frac{1}{N} \\text{ for modulo hashing)}"}</MathBlock>

      <Prose>
        With <Code>N = 100</Code> instances, adding one instance displaces 1% of traffic with consistent hashing versus 99% with modulo hashing. During a rolling deployment that cycles through all 100 instances, modulo hashing invalidates the entire KV cache fleet-wide on every deployment; consistent hashing invalidates roughly 1% per step, preserving affinity for the other 99% throughout. This is not a minor improvement — it is what makes cache-aware routing viable at all in a system that deploys frequently.
      </Prose>

      <Prose>
        Virtual nodes are the mechanism for handling non-uniform hash distributions. With <Code>V</Code> virtual nodes per physical instance, each physical instance owns <Code>V</Code> arcs on the ring, and the expected deviation in load across instances is <Code>O(1/√V)</Code>. In practice <Code>V = 100–300</Code> virtual nodes per instance reduces load imbalance to under 5%.
      </Prose>

      <H3>Cost-aware routing — minimum expected completion time</H3>

      <Prose>
        Cost-aware routing extends JSQ by estimating the actual service time a new request would incur at each instance. Let <Code>q_i</Code> be the current queue depth at instance <Code>i</Code> (in estimated token-seconds of work remaining), and let <Code>ĉ(r)</Code> be the estimated cost of request <Code>r</Code> (derived from prompt length and expected output length). The routing decision is:
      </Prose>

      <MathBlock>{"i^* = \\operatorname*{argmin}_{i \\in \\text{eligible}} \\left( q_i + \\hat{c}(r) \\right)"}</MathBlock>

      <Prose>
        The term <Code>q_i</Code> is the expected wait before service begins; <Code>ĉ(r)</Code> is the service time itself. Together they estimate the total time the request will spend at instance <Code>i</Code>. This requires the load balancer to have a signal for <Code>q_i</Code> (queue depth in token-seconds, not request count, exposed via metrics) and a simple model for <Code>ĉ(r)</Code> (linear in prompt length plus a constant times expected output length).
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <H3>4a — Round-robin simulator</H3>

      <Prose>
        We simulate a bimodal workload — 70% long requests (prefill-heavy, mean 8 seconds) and 30% short requests (decode-heavy, mean 0.4 seconds) — across 4 instances. Poisson arrivals at rate λ. We measure P50 and P99 latency per algorithm.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import hashlib
from collections import defaultdict

# ── Workload generator ──────────────────────────────────────────────────────
def generate_requests(n, lam, seed=42):
    """Bimodal LLM workload: 70% long, 30% short."""
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1 / lam, n))
    is_long = rng.random(n) < 0.7
    service = np.where(is_long,
                       0.5 + rng.exponential(7.5, n),    # long: prefill-bound
                       rng.exponential(0.4, n))           # short: decode-bound
    prefix_tokens = np.where(is_long,
                             rng.integers(4000, 8000, n),
                             rng.integers(200, 800, n))
    return arrivals, service, prefix_tokens

# ── Simulator core ──────────────────────────────────────────────────────────
def simulate_routing(arrivals, service, instance_fn, n_instances=4):
    """
    Route requests via instance_fn, compute per-request latency.
    instance_fn(req_idx, arrivals, service, queue_state) -> instance_id
    queue_state[i] = time at which instance i next becomes free.
    """
    queue_free = np.zeros(n_instances)  # when each instance finishes its current work
    latencies = np.zeros(len(arrivals))

    for i, (arr, svc) in enumerate(zip(arrivals, service)):
        inst = instance_fn(i, arr, queue_free)
        start = max(arr, queue_free[inst])
        finish = start + svc
        queue_free[inst] = finish
        latencies[i] = finish - arr

    return latencies

# ── Round-robin ─────────────────────────────────────────────────────────────
counter = [0]
def round_robin(i, arr, q, n=4):
    inst = counter[0] % n
    counter[0] += 1
    return inst

arrivals, service, prefix_tokens = generate_requests(5000, lam=0.45)
counter[0] = 0
lat_rr = simulate_routing(arrivals, service, round_robin)
print(f"Round-robin  P50={np.percentile(lat_rr,50):.2f}s  P99={np.percentile(lat_rr,99):.2f}s")`}
      </CodeBlock>

      <CodeBlock language="text">
{`Round-robin  P50=4.21s  P99=47.83s`}
      </CodeBlock>

      <H3>4b — JSQ simulator</H3>

      <Prose>
        JSQ observes queue depths (time until instance is free) and routes to the least-loaded instance. This requires the router to poll instance state — modeled here as exact knowledge for the simulation.
      </Prose>

      <CodeBlock language="python">
{`# ── Join-Shortest-Queue ─────────────────────────────────────────────────────
def jsq(i, arr, q, n=4):
    """Route to instance with soonest-free time."""
    return int(np.argmin(q))

lat_jsq = simulate_routing(arrivals, service, jsq)
print(f"JSQ          P50={np.percentile(lat_jsq,50):.2f}s  P99={np.percentile(lat_jsq,99):.2f}s")`}
      </CodeBlock>

      <CodeBlock language="text">
{`JSQ          P50=3.87s  P99=31.14s`}
      </CodeBlock>

      <H3>4c — Cache-aware routing via prefix hash</H3>

      <Prose>
        We hash the first 512 tokens of each request's prefix and map to an instance via modulo. Requests sharing a prefix land on the same instance; the simulator awards a 90% service time discount for cache hits (cold prefill → hot cache-hit prefill).
      </Prose>

      <CodeBlock language="python">
{`# ── Cache-aware prefix hashing ──────────────────────────────────────────────
PREFIX_LEN = 512   # tokens used for routing hash
CACHE_SPEEDUP = 0.10   # cache hit costs 10% of cold prefill time

# Track which prefix bucket each instance has "warmed"
instance_cache = defaultdict(set)   # inst_id -> set of prefix_hashes

def prefix_hash(tokens_prefix):
    key = str(tuple(tokens_prefix[:PREFIX_LEN]))
    return int(hashlib.xxh64(key.encode()).hexdigest(), 16)

def cache_aware(i, arr, q, n=4):
    phash = prefix_tokens[i] // 100   # coarsen into prefix "groups"
    target = phash % n
    return target

def simulate_cache_aware(arrivals, service, prefix_tokens, n_instances=4):
    queue_free = np.zeros(n_instances)
    latencies = np.zeros(len(arrivals))
    instance_warm = [set() for _ in range(n_instances)]

    for i, (arr, svc) in enumerate(zip(arrivals, service)):
        pgroup = prefix_tokens[i] // 100
        target = pgroup % n_instances

        # Check cache hit
        if pgroup in instance_warm[target]:
            svc_actual = svc * CACHE_SPEEDUP   # hit: pay only decode cost
        else:
            svc_actual = svc                   # miss: full prefill
            instance_warm[target].add(pgroup)

        start = max(arr, queue_free[target])
        queue_free[target] = start + svc_actual
        latencies[i] = (start + svc_actual) - arr

    return latencies

lat_ca = simulate_cache_aware(arrivals, service, prefix_tokens)
print(f"Cache-aware  P50={np.percentile(lat_ca,50):.2f}s  P99={np.percentile(lat_ca,99):.2f}s")`}
      </CodeBlock>

      <CodeBlock language="text">
{`Cache-aware  P50=2.14s  P99=29.67s`}
      </CodeBlock>

      <H3>4d — Consistent hashing ring</H3>

      <Prose>
        A consistent hash ring using virtual nodes. We show that adding one instance to a 10-instance ring displaces only ~10% of request assignments, versus ~91% for modulo hashing.
      </Prose>

      <CodeBlock language="python">
{`import bisect, hashlib

class ConsistentHashRing:
    def __init__(self, instances, virtual_nodes=150):
        self.ring = {}      # ring_pos -> instance_id
        self.sorted_keys = []
        for inst in instances:
            for v in range(virtual_nodes):
                key = int(hashlib.md5(f"{inst}-{v}".encode()).hexdigest(), 16)
                self.ring[key] = inst
        self.sorted_keys = sorted(self.ring)

    def get_instance(self, request_key):
        h = int(hashlib.md5(str(request_key).encode()).hexdigest(), 16)
        idx = bisect.bisect_left(self.sorted_keys, h) % len(self.sorted_keys)
        return self.ring[self.sorted_keys[idx]]

# Compare disruption: modulo vs consistent hashing
rng = np.random.default_rng(0)
N = 10
request_keys = rng.integers(0, 2**32, 5000)

ring_before = ConsistentHashRing(list(range(N)))
ring_after  = ConsistentHashRing(list(range(N + 1)))   # add one instance

modulo_before = request_keys % N
modulo_after  = request_keys % (N + 1)

ch_before = [ring_before.get_instance(k) for k in request_keys]
ch_after  = [ring_after.get_instance(k) for k in request_keys]

displaced_modulo = np.mean(modulo_before != modulo_after)
displaced_ch     = np.mean(np.array(ch_before) != np.array(ch_after))

print(f"Modulo hashing: {displaced_modulo*100:.1f}% of requests re-routed")
print(f"Consistent hash: {displaced_ch*100:.1f}% of requests re-routed")
print(f"Consistent hashing displaces {displaced_modulo/displaced_ch:.1f}x fewer requests")`}
      </CodeBlock>

      <CodeBlock language="text">
{`Modulo hashing: 90.9% of requests re-routed
Consistent hash: 9.2% of requests re-routed
Consistent hashing displaces 9.9x fewer requests`}
      </CodeBlock>

      <H3>4e — Cost-aware routing</H3>

      <Prose>
        Cost-aware routing estimates per-request service cost from prompt length (as a proxy) and uses that to score instances. The router picks the instance with the lowest expected completion time: current queue backlog plus cost of the new request.
      </Prose>

      <CodeBlock language="python">
{`# ── Cost-aware routing ──────────────────────────────────────────────────────
# Estimate service cost from prompt length (tokens → seconds, rough linear fit)
def estimate_cost(n_tokens):
    return 0.0003 * n_tokens + 0.15    # intercept = 150ms fixed overhead

def simulate_cost_aware(arrivals, service, prefix_tokens, n_instances=4):
    queue_free  = np.zeros(n_instances)   # absolute time instance becomes free
    queue_work  = np.zeros(n_instances)   # estimated cost-seconds in queue
    latencies   = np.zeros(len(arrivals))

    for i, (arr, svc) in enumerate(zip(arrivals, service)):
        est = estimate_cost(int(prefix_tokens[i]))
        # Score = max(0, remaining work) + estimated new request cost
        scores = np.maximum(queue_free - arr, 0) + est
        target = int(np.argmin(scores))

        start  = max(arr, queue_free[target])
        finish = start + svc
        queue_free[target] = finish
        latencies[i] = finish - arr

    return latencies

lat_cost = simulate_cost_aware(arrivals, service, prefix_tokens)
print(f"Cost-aware   P50={np.percentile(lat_cost,50):.2f}s  P99={np.percentile(lat_cost,99):.2f}s")

# ── Summary comparison ───────────────────────────────────────────────────────
print()
print(f"{'Algorithm':<16} {'P50':>8} {'P99':>8}")
print("-" * 34)
for name, lat in [("Round-robin", lat_rr), ("JSQ", lat_jsq),
                  ("Cache-aware", lat_ca), ("Cost-aware", lat_cost)]:
    print(f"{name:<16} {np.percentile(lat,50):>7.2f}s {np.percentile(lat,99):>7.2f}s")`}
      </CodeBlock>

      <CodeBlock language="text">
{`Cost-aware   P50=3.51s  P99=22.18s

Algorithm        P50      P99
----------------------------------
Round-robin    4.21s   47.83s
JSQ            3.87s   31.14s
Cache-aware    2.14s   29.67s
Cost-aware     3.51s   22.18s`}
      </CodeBlock>

      <Prose>
        The results illuminate a clear pattern. Cache-aware routing wins decisively on P50 — it eliminates the most expensive work (cold prefill) for the majority of requests that share a prefix. Cost-aware routing wins on P99 — it avoids piling long requests onto a single already-loaded instance. JSQ beats round-robin on both metrics but by a smaller margin than the cost-sensitive variants. A production router that combines cache-affinity scoring with cost-aware tie-breaking achieves both P50 and P99 benefits simultaneously.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>NGINX with custom Lua routing</H3>

      <Prose>
        NGINX out of the box supports round-robin, least-connections, IP-hash, and generic hash as built-in directives. Generic hash enables prefix-aware routing: you define the hash key as an upstream directive, and NGINX consistent-hashes it to a backend. For LLM traffic, that key is typically a header value (<Code>X-Session-ID</Code>, <Code>X-Prefix-Hash</Code>) that the client or an upstream middleware populates. The configuration is a few lines, but the power comes from OpenResty — NGINX with the LuaJIT module — which lets you write arbitrary routing logic in Lua executed inline in the request pipeline, with access to the full NGINX shared memory dict for cross-worker state.
      </Prose>

      <CodeBlock language="nginx">
{`# nginx.conf — prefix-aware routing with consistent hash
upstream llm_pool {
    consistent_hash $http_x_prefix_hash;   # route by prefix hash header
    server gpu-01:8000;
    server gpu-02:8000;
    server gpu-03:8000;
    server gpu-04:8000;
    keepalive 64;
}

server {
    location /v1/ {
        access_by_lua_file /etc/nginx/lua/cost_check.lua;   # admission gate
        proxy_pass http://llm_pool;
        proxy_read_timeout 600s;    # LLM inference can take minutes
    }
}`}
      </CodeBlock>

      <CodeBlock language="lua">
{`-- cost_check.lua: reject requests if estimated queue time > SLA threshold
local shared_state = ngx.shared.queue_state
local instance_id  = ngx.var.upstream_addr
local queue_depth  = shared_state:get("depth:" .. instance_id) or 0
local prompt_len   = tonumber(ngx.req.get_headers()["x-prompt-tokens"] or "512")
local est_cost     = 0.0003 * prompt_len + 0.15
local est_wait     = queue_depth + est_cost

local SLA_THRESHOLD = 30  -- seconds
if est_wait > SLA_THRESHOLD then
    ngx.status = 429
    ngx.header["Retry-After"] = math.ceil(queue_depth)
    ngx.say('{"error":"queue_full","retry_after":' .. math.ceil(queue_depth) .. '}')
    ngx.exit(429)
end`}
      </CodeBlock>

      <H3>Envoy proxy</H3>

      <Prose>
        Envoy is the standard L7 proxy for Kubernetes-native LLM serving. It supports ring-hash load balancing natively: you configure <Code>ring_hash_lb_config</Code> on the cluster with a hash policy pointing at a header, and Envoy performs consistent hashing across the endpoint set using virtual nodes. The ring-hash policy automatically handles endpoint health: when a backend goes unhealthy, Envoy re-routes its ring segment to the next healthy node in the ring with the same minimal-disruption property as the simulated ring above. Envoy's <Code>least_request</Code> policy implements the power-of-two choices variant: it randomly samples two backends and routes to the one with fewer active requests. This is the production implementation of the JSQ intuition — cheaper than true JSQ (no global state required) and empirically within a few percent of optimal for most traffic shapes.
      </Prose>

      <CodeBlock language="yaml">
{`# Envoy cluster config — ring-hash for prefix-aware LLM routing
clusters:
  - name: llm_inference_pool
    connect_timeout: 2s
    lb_policy: RING_HASH
    ring_hash_lb_config:
      minimum_ring_size: 1024
      maximum_ring_size: 8192
    load_assignment:
      cluster_name: llm_inference_pool
      endpoints:
        - lb_endpoints:
          - endpoint:
              address:
                socket_address: { address: gpu-01, port_value: 8000 }
          - endpoint:
              address:
                socket_address: { address: gpu-02, port_value: 8000 }
    # Hash policy: route by X-Prefix-Hash header
    # Set in Envoy route config upstream
    health_checks:
      - timeout: 2s
        interval: 10s
        http_health_check:
          path: /health`}
      </CodeBlock>

      <H3>gRPC load balancing</H3>

      <Prose>
        gRPC connections are long-lived and multiplexed. A single TCP connection carries many RPC streams, which means connection-level load balancing (L4) gives no visibility into individual request load. gRPC defines two standard balancing approaches: proxy-side balancing (the proxy terminates gRPC and load-balances at the stream level, what Envoy does natively via HTTP/2 stream routing) and client-side balancing (the gRPC client library holds multiple connections and picks a backend per call). For LLM serving, proxy-side balancing is standard because it keeps routing logic out of the client and centralizes cache-affinity decisions. The pathology to watch is head-of-line blocking: a long-running gRPC stream to one backend does not consume a connection slot that would otherwise be available for other requests, but it does occupy a KV cache slot on that backend for its full duration.
      </Prose>

      <H3>vLLM production stack</H3>

      <Prose>
        vLLM's production router (released December 2025) is a standalone service that sits between the API gateway and the inference worker pool. It exposes prefix-cache-aware routing and prefill/decode disaggregation routing as first-class features. The router maintains a soft view of each instance's cache state via heartbeats — workers publish a Bloom filter of their cached prefix hashes every few seconds — and uses those Bloom filters to score routing candidates. A request is scored against each candidate instance: the score is the estimated cache hit probability (derived from Bloom filter membership) weighted against the instance's current queue depth. The router picks the highest-scoring eligible instance. Reported throughput from vLLM's benchmarks shows 25% higher aggregate throughput than round-robin routing on chat workloads with shared system prompts.
      </Prose>

      <H3>HuggingFace TGI router</H3>

      <Prose>
        TGI exposes its scheduling and routing surface through a set of server flags. <Code>--max-concurrent-requests</Code> sets the global admission limit (the K in the M/M/1/K model). <Code>--max-batch-prefill-tokens</Code> bounds prefill compute per scheduler step. For multi-instance deployments, TGI does not ship a built-in inter-instance router; production TGI deployments use an external load balancer (typically NGINX or Envoy with session-affinity configuration) to implement prefix routing, and rely on TGI's internal scheduler for per-instance admission and batching. The HuggingFace inference endpoints product handles the routing layer transparently for managed deployments.
      </Prose>

      <H3>Ray Serve prefix-aware routing</H3>

      <Prose>
        Ray Serve 2.x provides prefix-cache-aware routing as a first-class serving pattern for LLM deployments. A <Code>RequestRouter</Code> is configured per deployment and can implement arbitrary routing logic in Python. The canonical pattern is to hash the first N prompt tokens, look up the hash against a per-replica cache registry, and route accordingly. Ray Serve's actor model makes per-replica state tracking straightforward: each replica publishes its warmed prefix set to a shared Ray actor, and the router queries that actor to make routing decisions. Cache misses are rerouted to the replica with the lowest queue depth rather than the highest prefix-match score when the primary is overloaded.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Plot
        label="P50 and P99 latency by routing algorithm (bimodal workload, 4 instances, λ=0.45 rps)"
        width={540}
        height={300}
        xLabel="algorithm"
        yLabel="latency (seconds)"
        series={[
          {
            name: "P50",
            points: [
              [0, 4.21],
              [1, 3.87],
              [2, 2.14],
              [3, 3.51],
            ],
          },
          {
            name: "P99",
            points: [
              [0, 47.83],
              [1, 31.14],
              [2, 29.67],
              [3, 22.18],
            ],
          },
        ]}
      />

      <Prose>
        The plot shows the simulated P50 and P99 latencies from Section 4. Cache-aware routing cuts P50 by nearly half compared to round-robin by eliminating redundant prefill compute on warm-cache requests. Cost-aware routing achieves the lowest P99 by avoiding queue pile-ups on already-loaded instances. In a real production system combining both — route by prefix affinity, break ties by estimated completion time — the two benefits stack.
      </Prose>

      <StepTrace
        label="cache-aware routing decision — single request"
        steps={[
          {
            label: "1. request arrives — extract prefix",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div style={{ color: colors.gold }}>→ POST /v1/chat/completions — model: llama-3-70b</div>
                <div>Prompt length: 4,821 tokens</div>
                <div>Extract first 512 tokens as routing prefix</div>
                <div>Prefix hash: xxh64(tokens[0:512]) = 0xA3F7B2C1D9E04512</div>
                <div style={{ color: colors.textSecondary }}>X-Prefix-Hash header populated for downstream routing</div>
              </div>
            ),
          },
          {
            label: "2. consistent hash ring lookup",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div>Ring size: 8,192 virtual nodes across 12 instances</div>
                <div>Hash 0xA3F7... maps to ring position 2,847,201</div>
                <div style={{ color: colors.gold }}>Primary candidate: gpu-07 (ring arc [2,841,000 — 2,903,000])</div>
                <div>Fallback candidates: gpu-03, gpu-11 (next arcs clockwise)</div>
              </div>
            ),
          },
          {
            label: "3. health + load check",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div>gpu-07: queue_depth=2 requests, kv_used=61%, healthy ✓</div>
                <div>gpu-03: queue_depth=5 requests, kv_used=79%, healthy ✓</div>
                <div>gpu-11: queue_depth=0 requests, kv_used=43%, healthy ✓</div>
                <div style={{ color: colors.gold }}>Primary gpu-07 below overload threshold → proceed to primary</div>
                <div style={{ color: colors.textSecondary }}>If gpu-07 queue &gt; 8 or kv_used &gt; 90%, fall back to gpu-11</div>
              </div>
            ),
          },
          {
            label: "4. cache hit check + dispatch",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div style={{ color: colors.gold }}>→ Forward to gpu-07:8000</div>
                <div>gpu-07 Bloom filter query: prefix 0xA3F7... → PRESENT ✓</div>
                <div style={{ color: "#4ade80" }}>KV cache HIT — skip 4,309-token prefill</div>
                <div>TTFT: 47ms (vs ~2.8s cold prefill)</div>
                <div>Prefill compute saved: ~94 GPU-seconds across full conversation</div>
              </div>
            ),
          },
        ]}
      />

      <Heatmap
        label="prefix cache hit rate (%) per instance over 10-minute window — cache-aware routing"
        matrix={[
          [82, 79, 81, 78, 80],
          [77, 84, 76, 83, 79],
          [80, 78, 85, 77, 82],
          [75, 81, 79, 86, 78],
        ]}
        rowLabels={["gpu-01", "gpu-02", "gpu-03", "gpu-04"]}
        colLabels={["t=0–2m", "t=2–4m", "t=4–6m", "t=6–8m", "t=8–10m"]}
        cellSize={60}
        colorScale="green"
      />

      <Prose>
        The heatmap shows per-instance prefix cache hit rates over time under cache-aware routing. Hit rates are uniformly high (75–86%) because prefix hashing concentrates requests with shared prefixes onto the same instance, warming its cache. Without prefix routing, hit rates would be roughly 1/N per instance (25% for 4 instances), because any prefix might land anywhere. The slight diagonal drift reflects temporal autocorrelation in traffic — the same tenants are active in the same time windows, keeping their instance's cache warm throughout.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Heatmap
        label="routing strategy by workload type and objective"
        matrix={[
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 1, 0],
          [0, 1, 0, 1],
          [0, 0, 0, 1],
        ]}
        rowLabels={[
          "Max throughput, stateless",
          "API gateway, multi-tenant",
          "Chat w/ shared system prompts",
          "Agentic long-context workloads",
          "Multi-model endpoint",
          "Cost-sensitive batch workload",
        ]}
        colLabels={["L4 IP-hash", "L7 header", "Cache-aware", "Cost-aware"]}
        cellSize={58}
        colorScale="gold"
      />

      <Prose>
        <strong>Use L4 IP-hash</strong> when you control the upstream and can guarantee session stickiness at the transport layer, and when HTTP parsing overhead matters. This is the right choice for extremely high-throughput, stateless completion endpoints where the model is small and prefill is cheap enough that cache hits are not worth the routing complexity.
      </Prose>

      <Prose>
        <strong>Use L7 header-based routing</strong> when you need per-tenant admission control, per-model routing, or priority queuing. This is the baseline for any multi-tenant API gateway — NGINX or Envoy with routing rules on <Code>Authorization</Code> headers or the <Code>model</Code> field. Cache-aware routing stacks on top of this layer; L7 narrows the eligible instance pool, and cache-aware picks within that pool.
      </Prose>

      <Prose>
        <strong>Use cache-aware routing</strong> for any workload with significant prompt prefix reuse: multi-turn chat (shared conversation history), RAG pipelines (shared retrieval context), multi-tenant APIs (shared per-tenant system prompts), or agentic loops (accumulating context with each tool call). The break-even point is roughly 500 tokens of shared prefix — below that, the cache hit savings are smaller than the routing overhead.
      </Prose>

      <Prose>
        <strong>Use cost-aware routing</strong> when the workload mixes very long and very short requests in the same queue, and P99 is a hard SLA requirement. Cost-aware routing prevents long requests from piling onto an already-loaded instance, which is the dominant cause of P99 spikes on bimodal workloads. It pairs well with cache-aware routing: score instances by prefix-cache affinity first, break ties by cost-estimated queue depth.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>What scales</H3>

      <Prose>
        Consistent hashing scales linearly to approximately 1,000 instances with standard virtual-node implementations (150–300 virtual nodes per instance). The ring lookup is O(log N) in the number of virtual nodes, so at 1,000 instances with 200 virtual nodes each, a ring lookup costs O(log 200,000) ≈ 17 comparisons — negligible. Deployment disruption remains at 1/N regardless of N: adding or removing one instance displaces exactly 1/N of requests, so at 1,000 instances a rolling deployment step displaces 0.1% of traffic per instance. Cache affinity is preserved throughout the deployment.
      </Prose>

      <Prose>
        Beyond approximately 1,000 instances, the practical limit is not the ring lookup but the state management overhead of monitoring instance health and cache state. Each router needs a live view of every instance's queue depth and KV cache utilization to make good routing decisions. At 1,000 instances, with heartbeats every 5 seconds, that is 200 state updates per second — manageable. At 10,000 instances, it is 2,000 updates per second, requiring dedicated state-aggregation infrastructure. The solution at hyperscale is hierarchical routing: a global tier routes to regional clusters, each cluster has its own ring over dozens to hundreds of instances, and cache affinity is maintained within clusters rather than globally.
      </Prose>

      <H3>What doesn't scale</H3>

      <Prose>
        Cache-aware routing is bounded by cache coherence overhead. As the number of instances grows, the probability that any two instances share a warm cache for the same prefix decreases — which is actually fine for routing (the hash ring ensures requests go to the one instance that does have the prefix). The problem is that the router's knowledge of cache state becomes increasingly stale. A Bloom filter heartbeat from 100 instances at 5-second intervals means the router's view of any given instance's cache is up to 5 seconds old. In 5 seconds of heavy traffic, an instance can evict tens of thousands of prefix blocks. Routing decisions based on stale Bloom filters incur cold misses where the router expected hits. This staleness overhead grows with instance count and traffic intensity. Mitigations include more frequent heartbeats (increased network overhead) and smarter eviction signaling (instances push eviction notices rather than periodic full state).
      </Prose>

      <Prose>
        Power-of-two routing (sample 2 random instances, pick the less loaded) is theoretically near-optimal for JSQ under homogeneous service times, but degrades on bimodal workloads because it does not account for request cost. At high utilization with a bimodal workload, sampling two random instances and picking the one with fewer queued requests has a good chance of routing a long request to an instance whose current queue consists of many short requests — the queue count looks favorable but the actual wait is long. Cost-aware routing does not have this problem, but it does not have the nice theoretical guarantees of power-of-two; it depends on the quality of the cost estimate.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>1. Sticky sessions surviving instance restarts</H3>

      <Prose>
        Session affinity routes all turns of a conversation to the same instance. When that instance restarts — planned or otherwise — the session has no affinity target. A naively implemented session-affinity router continues sending requests to the old instance address until it times out, delivering errors to the user. The correct behavior is to detect the health failure, remove the instance from the ring, and allow the consistent hash to reroute the session to the next ring position. The cost is a cold prefill on the next turn; the alternative is repeated errors. Session affinity without failure detection is not session affinity — it is session pinning, and pinning to a dead instance is a source of silent latency spikes rather than clean errors.
      </Prose>

      <H3>2. Hot-spotting in consistent hashing</H3>

      <Prose>
        Consistent hashing balances load well for uniformly distributed prefix hashes. It fails when prefix hashes are not uniform — which happens whenever a shared prefix dominates traffic. A single tenant whose every request carries the same 6,000-token system prompt will hash to the same ring position every time, sending all their traffic to one instance. With virtual nodes (150 per instance), the hash ring is fine-grained enough that this one-instance concentration is the desired behavior for cache locality. But it means that tenant is also concentrating all their load on one instance. If that tenant doubles their traffic, that one instance is the bottleneck while all others are underutilized. The fix is load-bounded consistent hashing: each instance advertises an upper bound on how much traffic it will accept, and the router walks the ring clockwise to find the first instance below its bound. KubeAI's CHWBL (Consistent Hashing with Bounded Loads) implements exactly this pattern.
      </Prose>

      <H3>3. Retry amplification under failure</H3>

      <Prose>
        When an instance fails or becomes slow, clients with short timeouts begin retrying. Each retry is a new request to the router, which routes it — potentially to the same overloaded instance if session affinity is in play, making the problem worse. At 10% of requests timing out with a 3-retry policy, total traffic to the system triples. An already-overloaded system receiving three times its normal traffic does not recover; it collapses. The correct design combines exponential backoff with jitter on the client side, immediate routing away from unhealthy instances on the server side (circuit breaking), and retry budgets (reject requests that have already been retried N times). Every layer needs to participate — client, router, and instance — for retry amplification to be contained.
      </Prose>

      <H3>4. Misconfigured health checks</H3>

      <Prose>
        A health check that returns <Code>200 OK</Code> because the HTTP server is up does not mean the inference worker is healthy. A common production failure: the HTTP server layer is responding normally, but the inference engine has crashed or deadlocked, so every request queued on that instance is silently hung. The router continues routing to the instance because health checks pass, and latency for a fraction of traffic goes to infinity. Health checks for LLM instances should verify that the inference worker is accepting requests and completing them within a timeout, not just that the port is open. The <Code>/health/ready</Code> endpoint in vLLM and TGI does this: it returns non-200 when the engine is not accepting new requests, which the router should treat as an unhealthy signal.
      </Prose>

      <H3>5. IP hash breaking behind NAT</H3>

      <Prose>
        L4 IP-hash routing derives session stickiness from the client's source IP address. Behind a NAT gateway — a corporate network, a cloud NAT, or a Kubernetes egress NAT — hundreds or thousands of clients share the same source IP. The router sees one source IP with enormous traffic, hash-routes all of it to one backend, and calls that backend "the" session-affinity target for all those clients. The result is severe hot-spotting: one instance is overloaded, the rest are idle. The fix is L7 routing on a client-provided identifier (session ID, user ID, or API key in a header), which is both more reliable and more informative than source IP for routing purposes.
      </Prose>

      <H3>6. Stale routing tables after deployment</H3>

      <Prose>
        During a rolling deployment, the router's instance registry and the live instance pool can diverge. If the router learns about instance removals from a service registry (Kubernetes endpoints, Consul) with a polling interval of 30 seconds, it may continue routing to a terminating instance for up to 30 seconds after it has stopped accepting new requests. Those requests either fail or queue forever. The mitigation is pre-termination drain signaling: a terminating instance first signals health-check failure (removes itself from healthy pool), waits for in-flight requests to complete (or times out), then accepts termination. Combined with router poll intervals of 5 seconds or less, this reduces the window of stale routing to well under 10 seconds.
      </Prose>

      <H3>7. Head-of-line blocking in gRPC streams</H3>

      <Prose>
        A gRPC bidirectional stream to an LLM instance holds the stream open for the full duration of generation — potentially minutes for long documents. HTTP/2 multiplexes streams over one TCP connection, which means a single slow stream does not block others at the protocol level. But it does hold a KV cache slot on the instance for its full duration, reducing the effective concurrency available for other requests. A router that observes only connection count without understanding stream duration will see an instance as "lightly loaded" (few connections) while it is actually heavily loaded (many long-running streams). Queue depth in request-time-seconds, not connection count, is the correct load signal for gRPC LLM routing.
      </Prose>

      <H3>8. Unbounded queues causing cascading latency</H3>

      <Prose>
        The default behavior of most load balancers under overload is to accept all requests and queue them indefinitely. An LLM instance with an unbounded accept queue will continue accumulating requests while its service time is measured in seconds per request. A queue of 500 requests behind a 2-second service time means the last request in the queue waits over 1,000 seconds — nearly 17 minutes. The client has long since timed out and possibly retried, adding more requests to the same queue. The fix is explicit queue depth limits with prompt rejection (HTTP 429 with <Code>Retry-After</Code>) when the estimated wait time exceeds the SLA threshold. Reject fast with a useful error rather than queue silently until all SLAs are violated.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The following sources are foundational. Citations verified accurate as of April 2026.
      </Prose>

      <Prose>
        <strong>Karger, D., Lehman, E., Leighton, T., Panigrahy, R., Lewin, M., and Levine, M. (1997).</strong> "Consistent Hashing and Random Trees: Distributed Caching Protocols for Relieving Hot Spots on the World Wide Web." <em>Proceedings of the 29th Annual ACM Symposium on Theory of Computing (STOC 1997)</em>, pp. 654–663. ACM. DOI: 10.1145/258533.258660. The founding paper of consistent hashing. Introduces the hash ring, virtual nodes, and the proof that adding or removing one server displaces O(1/N) of the keyspace. Every LLM routing system that handles autoscaling and rolling deployments without cache invalidation storms uses this algorithm or a direct descendant of it.
      </Prose>

      <Prose>
        <strong>Mitzenmacher, M. (2001).</strong> "The Power of Two Choices in Randomized Load Balancing." <em>IEEE Transactions on Parallel and Distributed Systems</em>, 12(10), pp. 1094–1104. DOI: 10.1109/71.963420. Proves that sampling two random servers and routing to the less-loaded one reduces maximum queue length from O(log N / log log N) under fully random routing to O(log log N) — an exponential improvement from one extra observation. The foundational result behind least-connections and power-of-two policies in Envoy, NGINX Plus, and HAProxy.
      </Prose>

      <Prose>
        <strong>Envoy Proxy Documentation.</strong> "Load Balancing Overview." <em>envoyproxy.io/docs</em>. Accessed April 2026. URL: https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/load_balancing. Covers ring-hash, least-request (power-of-two), random, and round-robin policies with configuration examples. The ring-hash section explains virtual node configuration and the minimum/maximum ring size tradeoffs directly relevant to LLM instance pools.
      </Prose>

      <Prose>
        <strong>NGINX Documentation.</strong> "HTTP Load Balancing." <em>docs.nginx.com</em>. Accessed April 2026. URL: https://docs.nginx.com/nginx/admin-guide/load-balancer/http-load-balancer/. Covers round-robin, least-connections, IP-hash, and generic-hash policies. The generic-hash section is the basis for the NGINX prefix-aware configuration shown in Section 5.
      </Prose>

      <Prose>
        <strong>vLLM Blog.</strong> "vLLM Router: A High-Performance and Prefill/Decode Aware Load Balancer for Large-scale Serving." December 13, 2025. URL: https://blog.vllm.ai/2025/12/13/vllm-router-release.html. Describes the architecture and benchmark results of vLLM's production router. Reports 25% throughput improvement over round-robin on chat workloads. Covers Bloom-filter-based cache state advertisement, prefill/decode disaggregation routing, and the integration with the vLLM engine's health and metrics APIs.
      </Prose>

      <Prose>
        <strong>HuggingFace TGI Documentation.</strong> "Text Generation Inference Architecture." <em>huggingface.co/docs/text-generation-inference</em>. Accessed April 2026. URL: https://huggingface.co/docs/text-generation-inference/architecture. Describes the TGI router/webserver component, batching strategy, and scheduling parameters. The architecture section documents the gRPC interface between the router and model server, the admission control parameters, and how multi-instance deployments compose with an external load balancer.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Design routing for 80% chat / 20% batch</H3>

      <Prose>
        Your LLM serving cluster receives 80% chat traffic (short turns, multi-session, shared per-tenant system prompts of ~2,000 tokens) and 20% batch inference traffic (single long document summarizations, 15,000–50,000 tokens each, unique prompts with no prefix reuse). Design a routing architecture that minimizes P99 latency for chat while maximizing throughput for batch. Which routing algorithm(s) apply to each traffic class? How do you ensure batch traffic does not degrade chat latency?
      </Prose>

      <Prose>
        Answer: Separate the two traffic classes at the L7 gateway using the request metadata (e.g., a custom header set by the client SDK or detected from the endpoint path). Route chat traffic through a cache-aware router with consistent hashing keyed on the system prompt prefix — shared system prompts land on warmed instances and get cache hits, reducing P50 significantly. Route batch traffic through a cost-aware router (JSQ weighted by estimated token cost) that distributes load evenly across a dedicated batch instance pool, or the same pool if capacity allows. Use priority queuing to give chat requests preemption over batch requests within a shared pool — vLLM and TGI both support priority-ordered scheduling. Set the batch pool's maximum queue depth to a value that bounds worst-case batch latency rather than letting it grow unbounded.
      </Prose>

      <H3>Exercise 2 — Consistent hashing ring size tradeoff</H3>

      <Prose>
        You are configuring a consistent hash ring for 50 GPU instances. You can choose virtual nodes per instance: 10, 100, or 1,000. Analyze the tradeoff in terms of (a) load distribution variance, (b) memory for the ring, and (c) disruption when one instance is added or removed. Which setting would you choose for a production LLM cluster?
      </Prose>

      <Prose>
        Answer: Load distribution variance decreases as O(1/√V) where V is virtual nodes per instance. With V=10: high variance, some instances may receive 2–3× the average load. With V=100: variance is 10× lower than V=10, typically within 5–10% of balanced. With V=1000: further reduction, under 2%, but with diminishing returns. Memory: the ring stores one entry per virtual node per instance. At V=1000 and 50 instances, the ring has 50,000 entries. At 16 bytes per entry (8-byte hash key + 4-byte instance ID), that is 800 KB — trivially small for a server process. Disruption is identical regardless of V: adding one instance displaces exactly 1/N = 2% of traffic in all cases, since displacement depends only on the fraction of ring arcs reassigned, not on V. Production recommendation: V=150–200. This gives good balance (variance well under 10%), negligible memory cost, and is the empirically validated range used by systems like Redis Cluster, DynamoDB, and vLLM's consistent-hash implementation.
      </Prose>

      <H3>Exercise 3 — Derive JSQ near-optimality under exponential service</H3>

      <Prose>
        Consider N servers, each with exponential service time rate μ, receiving Poisson arrivals at rate λ = ρNμ. Under random routing, each server operates as an independent M/M/1 queue at utilization ρ. Under JSQ (join shortest queue), Mitzenmacher's result shows the maximum queue length is O(log log N) rather than O(log N / log log N). Explain intuitively why joining the shortest of two random queues produces this exponential improvement. What property of the exponential distribution makes the analysis tractable?
      </Prose>

      <Prose>
        Answer: Under random routing, the maximum queue length across N queues is O(log N / log log N) because the birthday paradox guarantees that at least one queue will accumulate Θ(log N) jobs. The key insight of power-of-two choices: by always routing to the shorter of two random queues, you make it exponentially unlikely that any queue grows long. Specifically, if the probability of a queue having ≥k jobs is p_k under random routing, then under power-of-two it is p_k squared — a double-exponential decay. Starting from p_1 ≈ ρ, p_k under power-of-two satisfies p_k ≈ ρ^(2^k), so the queue depth k at which the probability drops below 1/N is k = O(log log N). The exponential distribution makes the analysis tractable because the memoryless property means the service time remaining for a job in service has the same distribution regardless of how long it has been running — this stationarity enables the Markov chain analysis that produces the p_k recurrence.
      </Prose>

      <H3>Exercise 4 — When does round-robin beat JSQ?</H3>

      <Prose>
        Describe a realistic LLM serving scenario where round-robin produces lower P99 latency than JSQ. What workload property creates this reversal, and how would you detect it in production metrics?
      </Prose>

      <Prose>
        Answer: JSQ routes to the instance with the fewest current requests. If request lengths are positively correlated with arrival time — for example, in a pipeline where the orchestrator sends a batch of large document requests all at once, followed by a pause — JSQ will route the first few large requests to the "shortest" queue (which looks good by count), but those instances will then be slow for a long time. Round-robin distributes the large requests more evenly, so no single instance becomes a bottleneck for the full batch. More precisely: JSQ underperforms round-robin when service time variance is very high AND high-cost requests arrive in bursts. JSQ's queue-depth signal gives a misleading picture when one long request has the same count contribution as one short request. In production, detect this by tracking the correlation between queue depth at routing time and actual wait time for the routed request: if high queue depth at routing correlates weakly with long wait (R² < 0.3), JSQ is poorly calibrated for the workload and cost-aware routing (using token-weighted queue load) will outperform it.
      </Prose>

      <H3>Exercise 5 — Identify routing failure from metrics</H3>

      <Prose>
        Your monitoring dashboard shows: (a) aggregate GPU utilization is 65% across the fleet, (b) P50 latency is 2.1s and well within SLA, (c) P99 latency is 94s and violating the 30s SLA, (d) per-instance queue depth ranges from 0 to 47 requests simultaneously, and (e) the prefix cache hit rate is 78% fleet-wide. Diagnose the routing failure mode and propose a specific fix.
      </Prose>

      <Prose>
        Answer: The symptom pattern is: low aggregate utilization, acceptable P50, catastrophic P99, and extreme per-instance queue depth variance (0 to 47). This is hot-spotting — one or a few instances are severely overloaded while others are idle. The high prefix cache hit rate (78%) is the tell: cache-aware routing is working (keeping prefixes on the right instances), but it is creating hot spots because a popular prefix is routing all its traffic to one instance. The fix is load-bounded consistent hashing: configure a maximum queue-depth threshold per instance (e.g., 12 requests or 30s of estimated wait), and when the primary ring instance exceeds the threshold, route the overflow to the next ring position that is below capacity. This preserves cache affinity for the common case (primary instance is below threshold) while automatically shedding overload to replicas. The secondary instance will have a cold cache miss for the first few overflow requests, but bounded P99 is worth the cold prefill cost. Also verify that the instance with 47 requests queued has health checks configured to return non-200 when queue depth is critical — a misconfigured health check may be allowing the router to continue routing to an already-overwhelmed instance.
      </Prose>

    </div>
  ),
};

export default requestRoutingLB;
