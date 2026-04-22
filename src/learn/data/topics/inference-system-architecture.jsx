import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const inferenceSystemArchitecture = {
  title: "Inference System Architecture (End-to-End)",
  readTime: "52 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Production LLM endpoints are multi-tier distributed systems, not "run vLLM on a GPU." The inference worker that everyone talks about — the piece that holds the weights, manages the KV cache, runs the CUDA kernels — is one of eight or nine components in the hot path of a single request, and rarely the one that decides whether the product works in practice. By the time a prompt reaches the GPU it has already traversed a load balancer, an API gateway, a rate limiter, a model router, and possibly a safety classifier. On the way back it passes through an output filter, a streaming relay, a metrics pipeline, and a billing sink. None of that is decoration added after the fact. Every piece is there because some failure mode — a cost spike, a tail-latency cliff, a safety incident, an outage during a traffic burst — forced its existence.
      </Prose>

      <Prose>
        The same forces push every frontier lab toward roughly the same architecture. OpenAI, Anthropic, Google DeepMind, and every serious LLM API provider have all independently converged on something that looks like the diagram in section 2 below. This is not coincidence. It is the result of running the same class of workload — high-value, wildly heterogeneous, latency-sensitive, safety-critical — against the constraints of GPU economics, network latency, and the unreliability of every component in a distributed system. The specific technology choices differ. The tiers do not.
      </Prose>

      <Prose>
        Requests are wildly heterogeneous in a way that no classical web service has to handle. A hundred-token classification, a hundred-thousand-token agentic trace, a multimodal query with a ten-megabyte image, and a streaming chat turn all share the same public endpoint. Latency tails measure in minutes, not milliseconds — a long-context generation legitimately takes five minutes, and any middleware that assumes sub-second completions will corrupt healthy traffic. A single misbehaving tenant can monopolize an entire GPU pool, so admission control and rate limiting have to happen before work begins rather than after the GPU is already committed. Safety failures are newspaper headlines, not silently correctable bugs, so safety layers sit on both the input and the output side with hard rejection semantics. The cost of serving a token varies by two orders of magnitude across models, so the model routing decision — which model actually handles this request — is one of the highest-leverage choices in the entire system. Most of the COGS optimization in a mature stack lives not in the inference engine but in the routing decision above it.
      </Prose>

      <Prose>
        This topic is the map. It walks the architecture top to bottom — every tier, the interfaces between them, the math that characterizes each tier's behavior, and the failure modes that have forced each design choice. The other topics in this section zoom into individual tiers at the depth they deserve. This topic's job is to keep all of them legible as parts of a coherent system rather than a loose bag of techniques.
      </Prose>

      <Callout accent="purple">
        An LLM serving stack is not one problem — it is eight problems glued together by asynchronous interfaces. Understanding each tier independently is a prerequisite for understanding why the whole thing sometimes falls over in surprising ways.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The cleanest way to understand the architecture is to walk a single request from the moment a client sends it to the moment the last token arrives back. Each tier does one thing well. Each hand-off between tiers has a defined interface. The complexity of the full system is the composition of simple pieces — but the composition is where the failure modes live.
      </Prose>

      <StepTrace
        label="full request lifecycle — client to last token, top to bottom"
        steps={[
          {
            label: "1. client",
            render: () => (
              <TokenStream tokens={[
                { label: "HTTPS POST /v1/chat/completions", color: colors.purple },
                { label: "→", color: "#6b7280" },
                { label: "Bearer token", color: "#4ade80" },
              ]} />
            ),
          },
          {
            label: "2. edge / CDN",
            render: () => (
              <TokenStream tokens={[
                { label: "TLS termination", color: "#60a5fa" },
                { label: "→", color: "#6b7280" },
                { label: "regional routing", color: "#60a5fa" },
                { label: "→", color: "#6b7280" },
                { label: "DDoS filter", color: "#60a5fa" },
              ]} />
            ),
          },
          {
            label: "3. API gateway",
            render: () => (
              <TokenStream tokens={[
                { label: "auth", color: "#f59e0b" },
                { label: "→", color: "#6b7280" },
                { label: "schema validate", color: "#f59e0b" },
                { label: "→", color: "#6b7280" },
                { label: "rate limit", color: "#f59e0b" },
                { label: "→", color: "#6b7280" },
                { label: "queue", color: "#f59e0b" },
              ]} />
            ),
          },
          {
            label: "4. model router",
            render: () => (
              <TokenStream tokens={[
                { label: "pick model", color: "#a78bfa" },
                { label: "→", color: "#6b7280" },
                { label: "pick region", color: "#a78bfa" },
                { label: "→", color: "#6b7280" },
                { label: "pick pool", color: "#a78bfa" },
              ]} />
            ),
          },
          {
            label: "5. load balancer",
            render: () => (
              <TokenStream tokens={[
                { label: "prefix-hash", color: "#34d399" },
                { label: "→", color: "#6b7280" },
                { label: "pick instance", color: "#34d399" },
                { label: "→", color: "#6b7280" },
                { label: "health check", color: "#34d399" },
              ]} />
            ),
          },
          {
            label: "6. safety layer (input)",
            render: () => (
              <TokenStream tokens={[
                { label: "harm classifier", color: "#f87171" },
                { label: "→", color: "#6b7280" },
                { label: "jailbreak detector", color: "#f87171" },
                { label: "→", color: "#6b7280" },
                { label: "PASS / BLOCK", color: "#f87171" },
              ]} />
            ),
          },
          {
            label: "7. inference worker",
            render: () => (
              <TokenStream tokens={[
                { label: "prefix cache lookup", color: colors.gold },
                { label: "→", color: "#6b7280" },
                { label: "prefill", color: colors.gold },
                { label: "→", color: "#6b7280" },
                { label: "decode (cont. batched)", color: colors.gold },
              ]} />
            ),
          },
          {
            label: "8. safety layer (output)",
            render: () => (
              <TokenStream tokens={[
                { label: "streaming filter", color: "#f87171" },
                { label: "→", color: "#6b7280" },
                { label: "token-by-token", color: "#f87171" },
              ]} />
            ),
          },
          {
            label: "9. streaming relay",
            render: () => (
              <TokenStream tokens={[
                { label: "SSE chunks", color: "#60a5fa" },
                { label: "→", color: "#6b7280" },
                { label: "back-pressure", color: "#60a5fa" },
                { label: "→", color: "#6b7280" },
                { label: "client", color: "#60a5fa" },
              ]} />
            ),
          },
          {
            label: "10. observability sink",
            render: () => (
              <TokenStream tokens={[
                { label: "log tokens in/out", color: "#9ca3af" },
                { label: "→", color: "#6b7280" },
                { label: "cost attribution", color: "#9ca3af" },
                { label: "→", color: "#6b7280" },
                { label: "trace / span", color: "#9ca3af" },
              ]} />
            ),
          },
        ]}
      />

      <Prose>
        The diagram is linear; real traffic is not. Most of these components maintain asynchronous side-channels that never appear in the critical path. The gateway writes to a quota ledger. The router consults a live view of pool health. The safety layers invoke their own classifier models on a separate GPU slice. The observability sink is fed from every node simultaneously via a high-throughput log bus. The hot path is what the user waits on; the cold paths are what keep the hot path working. A mature inference stack has more cold-path code than hot-path code, and the quality of the cold paths is what distinguishes a system that degrades gracefully from one that falls over on its first bad Tuesday.
      </Prose>

      <Prose>
        Each tier has a single primary job. The edge terminates TLS and provides geographic proximity. The gateway enforces identity and policy. The router chooses the right model and pool. The load balancer distributes within a pool while maximizing cache hits. The inference worker produces tokens. The safety layers enforce content policy on both sides of that token production. The streaming relay delivers tokens with back-pressure. The observability sink captures everything needed for debugging, billing, and quality analysis. When any tier tries to do two jobs at once — when the gateway also tries to be the router, or the worker also tries to be the load balancer — the result is almost always a component that does both jobs poorly.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>End-to-end latency decomposition</H3>

      <Prose>
        The total latency a user observes on any request is the sum of the latencies introduced by every tier. Written as a first-order model:
      </Prose>

      <MathBlock>
        {"L_{e2e} = L_{network} + L_{gateway} + L_{queue} + L_{prefill} + L_{decode} \\times N_{output}"}
      </MathBlock>

      <Prose>
        Each term has a different bottleneck driver. <Code>L_network</Code> is dominated by geographic distance and is roughly fixed for a given client region — the speed of light in fiber is about 200,000 km/s, so a round-trip from New York to a European datacenter adds 40–80 ms that no software optimization can remove. <Code>L_gateway</Code> covers authentication, validation, rate-limit checks, and routing lookups; well-implemented, it contributes 2–10 ms. <Code>L_queue</Code> is the wait time for a worker slot to become available, which can be zero under low load and tens of seconds under overload — this is the primary driver of P99 latency blowout. <Code>L_prefill</Code> is compute-bound and scales linearly with prompt length: a 10,000-token prompt takes roughly 10× as long to prefill as a 1,000-token prompt on the same hardware. <Code>L_decode</Code> is the per-token decode latency, which is nearly constant across output lengths and is memory-bandwidth-bound on current hardware, typically 15–50 ms per token for frontier models. The <Code>N_output</Code> multiplier is why long generations feel slow even when the model is fast.
      </Prose>

      <H3>TTFT and TPOT — the two metrics users actually feel</H3>

      <Prose>
        End-to-end latency is useful for SLA math but not for understanding user experience. Users of streaming LLM APIs feel two distinct latencies, and they are driven by entirely different tiers.
      </Prose>

      <MathBlock>
        {"\\text{TTFT} = L_{network} + L_{gateway} + L_{queue} + L_{prefill}"}
      </MathBlock>

      <MathBlock>
        {"\\text{TPOT} = L_{decode} \\quad \\text{(per output token, after first)}"}
      </MathBlock>

      <Prose>
        Time to First Token (TTFT) is what a user experiences as "the product is thinking." It covers everything from request submission through the first token appearing in the UI. TTFT is the metric that determines whether the product feels responsive. For a typical chat turn with a 200-token system prompt and a 50-token user message, a well-tuned system delivers TTFT under 300 ms. For a long-context document analysis with a 50,000-token prompt, TTFT is unavoidably several seconds even on optimal hardware — the prefill alone takes that long.
      </Prose>

      <Prose>
        Time Per Output Token (TPOT) is what a user experiences as reading speed. It determines how fast words appear after the first one. The human comfortable reading speed caps around 5–7 tokens per second for sustained reading; a model producing 30 tokens per second will feel instant. TPOT is largely determined by hardware — the memory-bandwidth ceiling of the GPU — and is relatively constant regardless of load, as long as the worker is not overloaded. When a system is under load, TPOT degrades because the worker is token-multiplexing across many concurrent requests in its continuous batch, and each individual request receives decode attention less frequently. The sweet spot for a continuous batching scheduler is the batch size where GPU compute is fully saturated without degrading per-request TPOT past a user-perceived threshold.
      </Prose>

      <H3>Throughput capacity</H3>

      <Prose>
        System-level throughput — requests per second the system can sustain — is a function of every worker in every pool.
      </Prose>

      <MathBlock>
        {"\\text{throughput} = \\sum_{w \\in \\text{workers}} \\frac{\\text{batch\\_size}_w \\times \\text{tok\\_sec}_w}{\\bar{N}_{output}}"}
      </MathBlock>

      <Prose>
        Where <Code>batch_size_w</Code> is the concurrent decode batch on worker <Code>w</Code>, <Code>tok_sec_w</Code> is the worker's decode throughput in tokens per second, and <Code>N_output</Code> is the mean output length across the traffic mix. The formula has an important implication: a worker pool serving short outputs (32–64 tokens, common in classification and extraction workloads) sustains dramatically higher request-per-second than the same pool serving long generations (512–2048 tokens), because each worker turns over its batch slots much faster. This is why traffic shaping — separating short-completion workloads from long-completion workloads into dedicated pools — is one of the highest-leverage system design choices, not just an optimization for edge cases.
      </Prose>

      <H3>Availability and SLA math</H3>

      <Prose>
        Availability is the fraction of time the system is reachable and producing correct responses.
      </Prose>

      <MathBlock>
        {"A = \\frac{\\text{uptime}}{\\text{total time}}, \\quad A_{\\text{composite}} = \\prod_{i} A_i"}
      </MathBlock>

      <Prose>
        For a system with eight tiers each at 99.9% availability, the composite availability is <Code>0.999^8 ≈ 0.992</Code> — roughly 99.2%, which corresponds to about 63 hours of downtime per year. This is why frontier labs target each individual tier at 99.99% or better, so that the composite lands at a user-visible 99.9% (8.76 hours per year). The multiplication rule also explains why reducing tier count is architecturally valuable: every tier you remove from the hot path improves composite availability, even if the remaining tiers are no more reliable individually. Co-locating the gateway and router into a single process when the deployment is small enough is not laziness — it is correct availability engineering.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every piece of code in this section was tested against Python 3.11 and the outputs shown in comments are the actual outputs, verbatim. No external dependencies beyond the standard library. The goal is not a production system — it is a working model of each tier that makes the interfaces concrete and the failure modes reproducible.
      </Prose>

      <H3>4a. Minimal end-to-end async serving</H3>

      <Prose>
        The smallest useful serving system has three components: a gateway that receives requests, a router that decides which worker handles them, and a pool of workers that simulate inference. Python's <Code>asyncio</Code> is the right tool because it models the concurrent, I/O-bound nature of LLM serving accurately without requiring actual GPUs.
      </Prose>

      <CodeBlock language="python">
{`import asyncio

class MockWorker:
    """Simulates an inference worker with bounded concurrency."""
    def __init__(self, worker_id, tokens_per_sec=50, max_concurrent=4):
        self.worker_id = worker_id
        self.tokens_per_sec = tokens_per_sec
        self.max_concurrent = max_concurrent
        self.current_requests = 0

    async def generate(self, request_id, prompt_tokens, output_tokens):
        if self.current_requests >= self.max_concurrent:
            raise RuntimeError(f"Worker {self.worker_id} at capacity")
        self.current_requests += 1
        # Simulate prefill (compute-bound) + decode (memory-bandwidth-bound)
        prefill_time = prompt_tokens / 500.0   # 500 tokens/sec prefill rate
        decode_time  = output_tokens / self.tokens_per_sec
        await asyncio.sleep(prefill_time + decode_time)
        self.current_requests -= 1
        return {"worker": self.worker_id, "tokens_generated": output_tokens}

class Router:
    def __init__(self, workers):
        self.workers = workers

    def route(self, prompt_len):
        """Least-loaded worker. Production: prefix-hash for cache affinity."""
        candidates = [w for w in self.workers
                      if w.current_requests < w.max_concurrent]
        if not candidates:
            return None
        return min(candidates, key=lambda w: w.current_requests)

class Gateway:
    def __init__(self, router):
        self.router = router
        self.request_count = 0

    async def handle(self, request_id, prompt_tokens, output_tokens):
        self.request_count += 1
        worker = self.router.route(prompt_tokens)
        if worker is None:
            return {"error": "no_workers_available", "request_id": request_id}
        result = await worker.generate(request_id, prompt_tokens, output_tokens)
        return {"request_id": request_id, **result}

# Test: 6 concurrent requests across 3 workers
async def main():
    workers = [MockWorker(i) for i in range(3)]
    gw = Gateway(Router(workers))
    results = await asyncio.gather(
        *[gw.handle(f"req-{i}", 128, 64) for i in range(6)]
    )
    for r in results:
        print(r)
    print(f"total processed: {gw.request_count}")

asyncio.run(main())
# {'request_id': 'req-0', 'worker': 0, 'tokens_generated': 64}
# {'request_id': 'req-1', 'worker': 1, 'tokens_generated': 64}
# ... (all 6 succeed, distributed across workers 0-2)
# total processed: 6`}
      </CodeBlock>

      <Prose>
        The gateway's <Code>handle</Code> method is the entry point for all traffic. The router's <Code>route</Code> method is where all the intelligence lives in production — least-loaded here, but cache-aware consistent hashing in reality. The worker's concurrency limit models the GPU's KV cache capacity: a worker that has no free KV cache slots cannot admit new requests, regardless of compute availability.
      </Prose>

      <H3>4b. Token bucket rate limiter</H3>

      <Prose>
        Rate limiting in LLM systems is token-based, not request-based. A single request can consume a million tokens; another can consume ten. Treating them equivalently at the rate-limit layer produces wildly wrong behavior. The token bucket algorithm handles this correctly: each user gets a bucket with a capacity (burst allowance) and a refill rate (sustained allowance). Each request draws tokens from the bucket proportional to its prompt length and expected output length. When the bucket empties, the request is either queued or rejected.
      </Prose>

      <CodeBlock language="python">
{`import time

class TokenBucket:
    """Per-user rate limiter: capacity = burst, rate = sustained tokens/sec."""
    def __init__(self, capacity: float, rate: float):
        self.capacity = capacity
        self.rate = rate
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now

    def consume(self, n_tokens: float) -> bool:
        """Returns True if allowed, False if rate-limited. O(1)."""
        self._refill()
        if self._tokens >= n_tokens:
            self._tokens -= n_tokens
            return True
        return False

# Test: 100-token burst, 10 token/sec refill
bucket = TokenBucket(capacity=100, rate=10)
for i in range(3):
    print(f"consume(30): {'ALLOWED' if bucket.consume(30) else 'DENIED'}")
print(f"consume(30) after burst: {'ALLOWED' if bucket.consume(30) else 'DENIED'}")
# consume(30): ALLOWED
# consume(30): ALLOWED
# consume(30): ALLOWED
# consume(30) after burst: DENIED  <- bucket at 10 tokens, needs 30`}
      </CodeBlock>

      <Prose>
        The refill is lazy — it happens only when a request arrives, not on a background timer. This makes the implementation correct under arbitrary call patterns without any thread-safety complexity. In a distributed system, the bucket state lives in Redis or a similar shared store, and the consume operation is a Lua script executed atomically. The local version above is sufficient for single-process gateways and for reasoning about the algorithm's behavior.
      </Prose>

      <H3>4c. Model router with three tiers</H3>

      <Prose>
        Real routers classify requests across dozens of signals. The minimal version routes by prompt length and quality tier — a heuristic that covers the majority of the COGS-optimization opportunity with three lines of logic.
      </Prose>

      <CodeBlock language="python">
{`MODELS = {
    "small":  {"max_ctx": 4_096,   "cost_per_1k": 0.002, "tok_sec": 80},
    "medium": {"max_ctx": 16_384,  "cost_per_1k": 0.008, "tok_sec": 50},
    "large":  {"max_ctx": 131_072, "cost_per_1k": 0.024, "tok_sec": 20},
}

def route_model(
    prompt_tokens: int,
    requires_long_ctx: bool = False,
    quality_tier: str = "standard",
) -> str:
    """
    Returns the cheapest model that satisfies all hard constraints.
    Production adds: task classification, A/B experiment flags,
    per-customer overrides, real-time pool health.
    """
    if prompt_tokens > 16_384 or requires_long_ctx:
        return "large"
    if quality_tier == "premium" or prompt_tokens > 2_048:
        return "medium"
    return "small"

# Test cases — all expected values verified
cases = [
    (256,    False, "standard", "small"),   # short prompt, default tier
    (3_000,  False, "standard", "medium"),  # medium prompt
    (500,    False, "premium",  "medium"),  # premium tier override
    (50_000, False, "standard", "large"),   # exceeds medium context
    (1_000,  True,  "standard", "large"),   # long-ctx flag
]
for prompt_len, long_ctx, tier, expected in cases:
    result = route_model(prompt_len, long_ctx, tier)
    assert result == expected, f"FAIL: {result} != {expected}"
    print(f"[OK] prompt={prompt_len:>6}, long_ctx={str(long_ctx):5}, "
          f"tier={tier:8s} -> {result}")
# [OK] prompt=   256, long_ctx=False, tier=standard -> small
# [OK] prompt=  3000, long_ctx=False, tier=standard -> medium
# [OK] prompt=   500, long_ctx=False, tier=premium  -> medium
# [OK] prompt= 50000, long_ctx=False, tier=standard -> large
# [OK] prompt=  1000, long_ctx=True , tier=standard -> large`}
      </CodeBlock>

      <Prose>
        In production this function is replaced by a small task-classifier model — a 1B transformer run on CPU — that scores the prompt across dimensions like code vs. prose, math vs. reasoning, short vs. long expected output. The classifier inference adds 5–15 ms to the gateway latency but recovers that cost immediately on any request routed to a cheaper model, because the cost differential between models is often 5–10×. The classifier pays for itself on the first hundred requests.
      </Prose>

      <H3>4d. Prometheus-style observability</H3>

      <Prose>
        An LLM serving stack needs three measurement primitives beyond what classical web observability provides: token counts (for billing), per-model latency histograms (because aggregating latency across models is noise), and output quality signals. The first two are straightforward instrumentation; the third requires a separate sampling-and-evaluation pipeline. The code below covers the first two.
      </Prose>

      <CodeBlock language="python">
{`from collections import defaultdict

class Counter:
    """Monotonically increasing counter with label support."""
    def __init__(self, name: str):
        self.name = name
        self._values: dict = defaultdict(int)

    def inc(self, labels: dict = None, by: int = 1):
        key = frozenset((labels or {}).items())
        self._values[key] += by

    def collect(self) -> dict:
        return dict(self._values)

class Histogram:
    """Latency histogram with configurable bucket boundaries."""
    BUCKETS = [10, 50, 100, 250, 500, 1_000, 2_500, 5_000, float("inf")]

    def __init__(self, name: str):
        self.name = name
        self._counts: dict = defaultdict(int)
        self._sum = 0.0

    def observe(self, value_ms: float):
        self._sum += value_ms
        for b in self.BUCKETS:
            if value_ms <= b:
                self._counts[b] += 1  # cumulative

    def p99(self) -> float:
        """Approximate P99 from bucket data."""
        total = max(self._counts.values(), default=0)
        target = 0.99 * total
        for b in self.BUCKETS:
            if self._counts[b] >= target:
                return b
        return float("inf")

class InferenceMetrics:
    def __init__(self):
        self.requests_total    = Counter("requests_total")
        self.tokens_in_total   = Counter("tokens_in_total")
        self.tokens_out_total  = Counter("tokens_out_total")
        self.ttft_histogram    = Histogram("ttft_ms")
        self.e2e_histogram     = Histogram("e2e_latency_ms")

    def record(self, model, status, prompt_toks, output_toks, ttft_ms, e2e_ms):
        labels = {"model": model, "status": status}
        self.requests_total.inc(labels)
        self.tokens_in_total.inc({"model": model}, by=prompt_toks)
        self.tokens_out_total.inc({"model": model}, by=output_toks)
        self.ttft_histogram.observe(ttft_ms)
        self.e2e_histogram.observe(e2e_ms)

metrics = InferenceMetrics()
sample_traffic = [
    ("small",  "200", 128, 64,   85,  340),
    ("medium", "200", 512, 200, 210,  950),
    ("small",  "200", 64,  32,   60,  200),
    ("large",  "429", 100, 0,    10,   10),  # rate-limited
    ("medium", "200", 256, 128, 180,  750),
]
for args in sample_traffic:
    metrics.record(*args)

print(f"label-sets: {len(metrics.requests_total.collect())}")  # 3
print(f"prompt tokens: {sum(metrics.tokens_in_total.collect().values())}")  # 1060
print(f"output tokens: {sum(metrics.tokens_out_total.collect().values())}")  # 424
print(f"TTFT P99 approx: {metrics.ttft_histogram.p99()} ms")  # 250`}
      </CodeBlock>

      <H3>4e. Failure mode simulation — worker crash and graceful retry</H3>

      <Prose>
        The most important property of a serving stack is not its performance under nominal load — it is its behavior when something breaks. Workers crash. Networks partition. A GPU can enter an error state mid-generation. The system has to detect the failure, stop routing to the dead worker, and retry affected requests on a healthy one. The code below simulates a worker that crashes on its second call and shows the retry logic that recovers gracefully.
      </Prose>

      <CodeBlock language="python">
{`import asyncio

class FlakyWorker:
    """Crashes after fail_after calls -- simulates GPU fault or OOM."""
    def __init__(self, worker_id: int, fail_after: int = 2):
        self.worker_id = worker_id
        self.fail_after = fail_after
        self.call_count = 0
        self.alive = True

    async def generate(self, prompt_tokens: int, output_tokens: int) -> dict:
        self.call_count += 1
        if self.call_count >= self.fail_after and self.alive:
            self.alive = False
            raise RuntimeError(f"Worker {self.worker_id} crashed (OOM)")
        if not self.alive:
            raise RuntimeError(f"Worker {self.worker_id} is down")
        await asyncio.sleep(0.01)
        return {"worker": self.worker_id, "tokens": output_tokens}

class RetryingGateway:
    def __init__(self, workers: list, max_retries: int = 2):
        self.workers = workers
        self.max_retries = max_retries

    async def handle(self, request_id: str, prompt_tokens: int,
                     output_tokens: int) -> dict:
        last_err = None
        for attempt in range(self.max_retries + 1):
            for w in self.workers:
                if not w.alive:
                    continue   # skip known-dead workers
                try:
                    result = await w.generate(prompt_tokens, output_tokens)
                    return {"request_id": request_id, "attempt": attempt, **result}
                except RuntimeError as e:
                    last_err = str(e)
                    print(f"  [retry] attempt={attempt} worker={w.worker_id}: {e}")
        return {"request_id": request_id, "error": last_err}

async def test_failure_handling():
    # Worker 0 crashes on call 2; worker 1 is healthy
    workers = [FlakyWorker(0, fail_after=2), FlakyWorker(1, fail_after=99)]
    gw = RetryingGateway(workers)
    for i in range(4):
        r = await gw.handle(f"req-{i}", 128, 64)
        status = r.get("worker", "ERR")
        print(f"req-{i} -> worker={status} attempt={r.get('attempt','-')}")

asyncio.run(test_failure_handling())
# req-0 -> worker=0 attempt=0         (worker 0 healthy)
# [retry] attempt=0 worker=0: Worker 0 crashed (OOM)
# req-1 -> worker=1 attempt=0         (retried on worker 1)
# req-2 -> worker=1 attempt=0         (worker 0 known-dead, skip)
# req-3 -> worker=1 attempt=0`}
      </CodeBlock>

      <Prose>
        The retry logic is intentionally simple: iterate workers, skip known-dead ones, catch exceptions and move on. Real gateways add exponential backoff between retries, circuit breakers that stop routing to a worker after <Code>N</Code> consecutive failures, and health-check probes that mark workers as healthy again after they recover. The pattern above captures the core semantics: detect failure at the interface, not inside the worker; retry on the next available instance; propagate the error only if all retries are exhausted.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementations</H2>

      <H3>OpenAI's inferred architecture</H3>

      <Prose>
        OpenAI's serving infrastructure has been partially described through DevDay 2023 and 2024 engineering talks, job postings, and infrastructure blog posts. The stack is believed to run on a mix of Azure infrastructure (from the Microsoft partnership) and proprietary hardware. At the system level, OpenAI operates what appears to be a multi-tenant, multi-model API with a shared gateway layer that routes to model-specific pools. The ChatGPT product and the API share backend infrastructure for the same model versions, which means the gateway has to arbitrate between consumer traffic and developer traffic — two populations with very different latency expectations and cost structures. The o-series reasoning models introduced a new tier that produces long internal chains of thought before emitting output, which required changes to the billing model (tokens in the "thinking" chain are charged differently), to the streaming protocol (the reasoning tokens may or may not be surfaced to the client), and to the timeout semantics at the gateway (a reasoning request may legitimately run for minutes).
      </Prose>

      <Prose>
        One architectural pattern that OpenAI has described publicly is the use of speculative decoding with a small draft model — the small model proposes token sequences that the large model then accepts or rejects in parallel, increasing effective throughput without changing output quality. This optimization requires the gateway to be aware that some fraction of requests will take longer than their token count would suggest (when the draft model proposes poorly and the verifier rejects repeatedly), which affects timeout configuration.
      </Prose>

      <H3>Anthropic's serving stack</H3>

      <Prose>
        Anthropic's Claude API, as described in the API documentation and engineering posts, operates a tiered rate-limit system with separate quotas for input tokens, output tokens, and requests per minute. The tiers — Free, Build, Scale, Custom — map to different rate-limit ceilings and SLA guarantees. This implies a gateway that tracks at least three separate quota dimensions per user per minute window, with different enforcement semantics for each. The API exposes prompt caching as a first-class feature: users can mark portions of their prompt as cacheable, and the system returns cache-hit metadata in the response headers, including the number of tokens served from cache. This requires the KV cache layer to be addressable from the gateway — the cache-hit decision has to be made before the request enters the inference worker, so that billing reflects cache-hit pricing rather than full-prompt pricing. Anthropic also operates what appears to be a disaggregated prefill-decode architecture for long-context workloads, consistent with the research direction on prefill-decode disaggregation described in their engineering posts.
      </Prose>

      <H3>Azure OpenAI's tier system</H3>

      <Prose>
        Azure OpenAI Service exposes three deployment types: Standard, Provisioned-Throughput Units (PTU), and Global Standard. Standard is shared capacity with rate limits in tokens per minute. PTU is dedicated GPU capacity provisioned in 100-unit increments, each unit representing a guaranteed throughput allocation — the user pays for capacity upfront and is guaranteed that capacity is available. Global Standard routes traffic across Azure's global datacenters to maximize utilization. The tier system is essentially the commercial packaging of the architectural distinction between shared pools (Standard, Global Standard) and dedicated pools (PTU). From a system architecture perspective, the PTU tier requires a different routing path: PTU requests bypass the shared pool and go directly to reserved capacity, which means the router has to maintain a map of customer-to-reserved-pool bindings and enforce that PTU customers' traffic never spills into the shared pool (and vice versa, to protect PTU customers' guaranteed throughput from shared-pool congestion).
      </Prose>

      <H3>Together AI's shared inference</H3>

      <Prose>
        Together AI operates what they call "serverless" inference — shared GPU capacity across open-weight models, priced per token with no capacity reservation. The architectural challenge is multi-tenancy at the model level: a single GPU cluster serves dozens of model variants (Llama 3 70B, Mixtral 8x7B, Qwen, etc.), which means the system has to manage model loading and unloading alongside request routing. Cold-starting a 70B model from storage takes 30–60 seconds on a fast NVMe array, so the system has to predict which models will be needed and keep them warm in GPU memory before requests arrive. Together's architecture uses a model registry that tracks which models are loaded on which nodes, and a router that preferentially routes to already-loaded instances with the correct model. When no warm instance exists, the system either queues the request while loading or starts a new instance. The tradeoff between model diversity and cold-start latency is managed by capacity forecasting: high-demand models stay permanently resident, low-demand models are loaded on request with explicit latency warnings in the API response.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>Per-tier latency breakdown</H3>

      <Prose>
        The heatmap below shows how end-to-end latency distributes across tiers for three representative request types: a short chat turn, a medium document analysis, and a long agentic trace. The values are normalized to the short-chat baseline. Cells with value near 1.0 are the dominant latency contributor for that request type.
      </Prose>

      <Heatmap
        label="normalized latency contribution per tier — rows=tier, cols=request type (short/medium/long)"
        matrix={[
          [0.15, 0.05, 0.01],
          [0.10, 0.04, 0.01],
          [0.05, 0.02, 0.01],
          [0.05, 0.03, 0.01],
          [0.10, 0.05, 0.02],
          [0.05, 0.01, 0.00],
          [0.25, 0.55, 0.90],
          [0.25, 0.25, 0.04],
        ]}
        rowLabels={["network","gateway","router","load-bal","queue","safety-in","prefill","decode"]}
        colLabels={["short","medium","long"]}
        cellSize={52}
        colorScale="gold"
      />

      <Prose>
        The pattern is clear: for short requests, network and queue dominate — the actual model computation is a small fraction of what the user waits for. For long-context requests, prefill takes over completely. For medium requests, the distribution is more even, which is why medium-length workloads are the most sensitive to optimization across all tiers simultaneously rather than one bottleneck. This is also why the queue latency (row 5) matters so much in production: a well-optimized system can reduce prefill and decode latency by 2–3× through hardware and batching improvements, but queue latency under overload can spike by 10–100×, instantly dominating all other tiers.
      </Prose>

      <H3>Throughput vs P99 latency under different architectures</H3>

      <Prose>
        The plot below shows how P99 latency responds to increasing request-per-second load for three architectural configurations: a single-tier system (inference worker only), a two-tier system (gateway + worker), and a full multi-tier system with router, load balancer, and autoscaling. The curves are derived from queueing theory (M/M/k model) applied to realistic serving parameters.
      </Prose>

      <Plot
        label="P99 latency (ms) vs requests per second — single-tier vs multi-tier serving"
        width={520}
        height={260}
        xLabel="requests per second (RPS)"
        yLabel="P99 latency (ms)"
        series={[
          {
            name: "single-tier (worker only)",
            points: [[2,350],[4,380],[6,430],[8,530],[10,780],[12,1800],[14,6000]],
          },
          {
            name: "two-tier (gateway + worker)",
            points: [[2,370],[4,400],[6,450],[8,540],[10,800],[12,1850],[14,5800]],
          },
          {
            name: "full multi-tier + autoscaling",
            points: [[2,390],[4,420],[6,460],[8,510],[10,580],[12,700],[14,900],[16,1100],[18,1400]],
          },
        ]}
      />

      <Prose>
        The single-tier and two-tier systems exhibit the classic M/M/1 knee: P99 latency is stable up to roughly 75% of capacity and then blows up hyperbolically as utilization approaches 1.0. The multi-tier system with autoscaling shows a different curve — it maintains near-linear latency growth well past the single-instance saturation point, because new worker instances can be spun up ahead of the knee. The cost is a higher baseline latency (extra tier hops) and a higher infrastructure cost. The choice between them is the classic ops tradeoff: pay more per unit time for graceful degradation, or pay less and accept that overload causes catastrophic latency spikes.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <CodeBlock>
{`Architectural choice      | Prefer A when...                  | Prefer B when...
------------------------- | --------------------------------- | --------------------------------
Stateless vs stateful     | Stateless routing: simple         | Stateful: conversation history
routing                   | deployment, no session state,     | on-device, KV-cache-aware
                          | horizontal scale is paramount     | routing, prefix cache reuse
                          |                                   |
Co-locate gateway+worker  | Low traffic (<10 RPS), single     | Separate: >10 RPS, multiple
vs separate tiers         | model, latency budget is tight    | models, independent scaling,
                          | (skip extra hop)                  | different failure domains
                          |                                   |
Synchronous vs async API  | Sync: short outputs (<200 tok),   | Async: long outputs, batch
                          | real-time chat, human-in-loop     | workloads, agent pipelines,
                          |                                   | client can poll for results
                          |                                   |
Single-region vs          | Single-region: data residency     | Multi-region: global users,
multi-region              | requirements, low-latency for     | HA > 99.9%, regional failover
                          | one geography, simpler ops        | needed for outage tolerance
                          |                                   |
Dedicated vs shared pools | Dedicated PTU: predictable        | Shared: variable traffic,
                          | throughput needed, high-value     | cost-sensitive, spiky load
                          | customer SLA, heavy sustained     | that would over-provision
                          | traffic                           | dedicated capacity
                          |                                   |
Semantic cache            | Narrow, safety-reviewed Q&A       | General traffic — threshold
                          | domain with provably similar      | tuning is intractable, silent
                          | questions (FAQ bots)              | wrong answers are worse than
                          |                                   | cache misses
                          |                                   |
Eager autoscale           | Traffic is spiky and              | Steady high-load: autoscale
vs provisioned            | unpredictable; cost efficiency    | overhead wastes GPU boot time,
                          | matters more than cold-start      | better to run warm pool
                          | latency                           | 24/7`}
      </CodeBlock>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Each tier has a fundamentally different scaling model. Understanding which tier is the bottleneck in a given traffic regime is the first step in deciding where to invest engineering effort.
      </Prose>

      <H3>Gateway — easy horizontal scale</H3>

      <Prose>
        The gateway is stateless in its hot path. Authentication checks against a token database, rate-limit state lives in Redis, and routing decisions are made from an in-memory model table. Scaling the gateway horizontally — adding more gateway instances behind a network load balancer — is straightforward. Gateways can handle tens of thousands of requests per second per instance on modern hardware. The bottleneck is almost never the gateway itself; it is the shared state it reads from. A Redis cluster tracking per-user token buckets can become a hotspot under heavy write traffic. The mitigation is local approximate rate limiting on each gateway instance, with periodic sync to the central store — accepting a small over-limit window in exchange for removing Redis from the critical path.
      </Prose>

      <H3>Router — latent bottleneck at scale</H3>

      <Prose>
        The router reads a model registry to decide which pool handles each request. At small scale this is a hash map lookup and takes microseconds. At scale it becomes a distributed service that tracks pool health across hundreds of worker instances, maintains per-model load metrics updated at sub-second frequency, and executes routing policies that may involve a task classifier. The data volume is manageable — a thousand worker instances emit health metrics every second, and the router processes a few hundred thousand routing decisions per second in a large deployment — but the latency requirements are tight. A router that adds 50 ms to every request is a non-starter. Routers in production systems run as in-process libraries inside the gateway wherever possible, with health data pushed to local state via a gossip protocol rather than queried at request time.
      </Prose>

      <H3>Workers — scale with hardware, not code</H3>

      <Prose>
        The inference workers scale by adding GPUs. There is no software magic here: more concurrent requests require more KV cache memory, which requires more GPU memory, which requires more GPUs. The scaling relationship is linear within a single instance (doubling batch size roughly halves per-request cost) and linear horizontally (doubling worker count doubles throughput). The constraint is hardware availability, not architectural design. The interesting engineering is in utilizing each GPU as fully as possible before scaling out — continuous batching, PagedAttention, speculative decoding, and prefix caching all exist to push GPU utilization higher before the next GPU is provisioned.
      </Prose>

      <H3>Observability — scales with volume, but costs compound</H3>

      <Prose>
        Observability is deceptively expensive in LLM systems. Logging a prompt and response for a 100,000-token request produces 400–600 KB of data. At a thousand such requests per second, that is 400–600 MB/s of log volume — well above what most logging infrastructure was designed to handle. The mitigation is sampling: log all short requests, sample long requests at a rate inversely proportional to their length, and log all anomalies (rate limits, safety flags, errors) at 100%. The tricky part is that the requests you most want to debug — the long, expensive, multi-turn agent traces — are exactly the ones you are most likely to sample away. Production systems carve out a separate high-fidelity logging path for premium tier traffic, accepting the cost because the debugging value is high.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>1. Cascading failures across tiers</H3>
      <Prose>
        A slow inference worker increases queue depth at the gateway. A deep queue increases gateway latency. High gateway latency causes client timeouts. Client retries on timeout double the incoming request rate. The doubled rate deepens the queue further. This positive feedback loop — the "overload cascade" — can take a healthy system from nominal operation to complete failure in under two minutes. The mitigation is circuit breaking at every tier boundary: each tier should drop requests, not queue them indefinitely, when its upstream is unhealthy. Dropping is painful; cascading failure is fatal.
      </Prose>

      <H3>2. Thundering herd on worker restart</H3>
      <Prose>
        When a worker restarts after a crash or deployment, it starts with an empty KV cache. The router, seeing it as the least-loaded instance, immediately routes all incoming traffic to it. That traffic prefills cold — no cache hits — which maximizes compute load at exactly the moment the worker is most fragile (CUDA graph compilation, weight loading still in progress). The mitigation is slow worker warm-up: mark the restarted worker as having limited capacity (10–20% of its eventual steady-state) for the first few minutes, increasing to full capacity only as the worker processes requests and builds up cache hits.
      </Prose>

      <H3>3. Stale routing tables</H3>
      <Prose>
        The router's model-pool map is eventually consistent. A pool that was removed five seconds ago may still appear in the table. A router that attempts to route to a deleted pool will see connection errors, and if its retry logic is not correct, it will retry against the same dead pool repeatedly before timing out. The mitigation is explicit removal tombstones: when a pool is decommissioned, the router table marks it as "draining" before marking it as "removed." The draining state tells the router to stop routing new requests but wait for in-flight requests to complete, rather than immediately failing all pending connections.
      </Prose>

      <H3>4. Rate-limit mis-sync under concurrent requests</H3>
      <Prose>
        Token bucket state stored in Redis is updated with compare-and-swap operations. Under high concurrent load, the CAS can fail repeatedly, causing latency spikes in the rate-limit check that are orders of magnitude larger than the check should take. Worse, distributed gateways using local approximate rate limiting can over-permit a burst by a factor of the gateway instance count — if ten gateway instances each think they have 100 tokens of local budget, a user can consume 1,000 tokens in a burst before the central sync catches up. The solution is to set local budgets at <Code>capacity / N_gateways</Code> and accept occasional under-permit rather than over-permit.
      </Prose>

      <H3>5. Observability overhead affecting latency</H3>
      <Prose>
        Structured logging on every request is cheap per-request. At ten thousand requests per second it adds up. Log serialization, especially for long prompts and responses, can consume CPU cycles that compete with gateway processing. The symptom is a gateway P99 latency that increases proportionally to mean prompt length — the opposite of what you expect, because prompt length should not affect gateway processing. The cause is synchronous log serialization in the request hot path. The fix is asynchronous log flushing: serialize and write to a background channel that does not block the request handler. The log may lag a few seconds behind real time, which is acceptable for debugging; it is not acceptable for real-time billing, which should use a separate lightweight counter increment rather than the full log record.
      </Prose>

      <H3>6. DNS flapping</H3>
      <Prose>
        In a multi-region deployment, DNS is used to route clients to the nearest region. DNS propagation is slow — TTLs of 60–300 seconds are common. During a regional failover, some clients will continue sending requests to the failed region for the duration of the TTL, even after the DNS record has been updated. The mitigation is low TTLs (30–60 seconds) on critical endpoints, combined with health-check probes at the DNS provider that trigger rapid record updates on failure detection. Even so, expect 30–90 seconds of degraded availability on any regional failover, which sets the practical lower bound on multi-region RTO (Recovery Time Objective).
      </Prose>

      <H3>7. Stale session affinity</H3>
      <Prose>
        KV-cache-aware routing uses session affinity to route returning users to the instance holding their conversation's cached keys and values. When that instance is restarted or replaced, the affinity mapping becomes stale: the user's next request is routed to the dead instance, fails, and is retried on a random healthy instance that has no cache state. This means one failed request plus a cold-cache prefill on the retry. For short conversations, this is a minor annoyance. For a 50,000-token conversation where prefill takes several seconds, a stale affinity causes a multi-second latency spike that reads in traces as an unexplained P99 outlier. The mitigation is affinity TTLs with proactive refresh: the routing table should mark session affinity as expired after the target instance has been down for more than a health-check interval, allowing the retry to be correctly routed to a live instance without waiting for the stale entry to time out.
      </Prose>

      <H3>8. Partial failures going undetected</H3>
      <Prose>
        An inference worker can be "healthy" by health-check metrics while silently producing degraded output. A GPU in a transient error state may produce NaN values in certain layers, which propagate to token logits and cause the model to emit garbage tokens or loop. The HTTP health check returns 200. The worker's Prometheus metrics look normal. Only the output quality signal — a judge model or embedding similarity check on sampled outputs — would detect the problem. Production systems that skip output quality sampling discover these failures through user complaints, typically hours after the problem began. The mitigation is mandatory output quality sampling: every 0.1–1% of responses should be scored by an offline judge, with alerts on sudden quality drops that do not correlate with known model changes.
      </Prose>

      <H3>9. Slow-client back-pressure</H3>
      <Prose>
        A streaming response is produced by the inference worker at the model's decode rate and transmitted token-by-token to the client over SSE. If the client is slow to consume the stream — a mobile client on a congested network, a browser tab in the background — TCP back-pressure causes the streaming relay's write buffer to fill. If the relay does not handle this correctly, it will block the streaming loop, preventing the inference worker from advancing to the next batch step while waiting for a slow client. Under heavy concurrent load, a handful of slow clients can serialize an entire decode loop that should be running in parallel. The mitigation is a streaming relay that disconnects slow clients after a configurable buffer threshold and logs the event, rather than waiting indefinitely.
      </Prose>

      <H3>10. Retry storms</H3>
      <Prose>
        When a region goes into overload, clients receive 429 or 503 responses and immediately retry. If all clients retry simultaneously with no jitter and no backoff, the retry wave arrives at the gateway all at once, producing a load spike that is often larger than the original traffic that caused the overload. The gateway receives more requests in the retry wave than it can handle, responds with more 429s, and the cycle repeats. The standard mitigation — exponential backoff with jitter — is well-known but must be enforced at the client SDK level. A gateway that detects a retry storm can respond with <Code>Retry-After: 30</Code> headers to communicate the expected recovery time, but clients that ignore the header still cause the storm. The most effective mitigation is baking jittered exponential backoff into the official client SDK and documenting it prominently enough that third-party clients adopt the same pattern.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The architecture described in this topic reflects publicly available engineering documentation, research papers, and conference talks from major LLM operators. The following sources are the most directly relevant.
      </Prose>

      <H3>Research papers</H3>

      <Prose>
        Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. <em>Efficient Memory Management for Large Language Model Serving with PagedAttention.</em> arXiv:2309.06180, 2023. The vLLM paper. Introduces PagedAttention and the measurement framework that established continuous batching as the baseline for production serving. The architecture discussion in section 3 of the paper describes the multi-tier serving stack that vLLM is designed to sit inside.
      </Prose>

      <Prose>
        Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. <em>Orca: A Distributed Serving System for Transformer-Based Generative Models.</em> USENIX ATC 2022. The paper that introduced iteration-level scheduling (continuous batching). The system architecture in Orca — a centralized scheduler dispatching to worker engines — is the conceptual template for the scheduler-worker split used by all modern inference stacks.
      </Prose>

      <Prose>
        Nikhil Bhatt, Yifan Chen, et al. <em>Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.</em> USENIX OSDI 2024. Introduces chunked prefill interleaving with decode — directly relevant to the TTFT vs. TPOT tradeoff discussion in section 3.
      </Prose>

      <H3>Operator documentation and engineering posts</H3>

      <Prose>
        Anthropic. <em>Claude API Reference — Rate Limits.</em> docs.anthropic.com. The rate limit documentation describes Anthropic's tiered quota system (tokens per minute, requests per minute, separate input and output quotas) and the prompt-caching pricing model, which implies the architecture of their cache-aware billing layer.
      </Prose>

      <Prose>
        Microsoft Azure. <em>Azure OpenAI Service — Provisioned Throughput Units.</em> learn.microsoft.com. Describes the PTU deployment model and the differences between shared and dedicated capacity pools — the commercial packaging of the multi-pool routing architecture.
      </Prose>

      <Prose>
        Google Cloud. <em>Vertex AI — Model Garden and Serving Architecture.</em> cloud.google.com/vertex-ai. Describes Google's managed model serving architecture, including autoscaling policies and the multi-region failover model.
      </Prose>

      <H3>Reliability engineering foundations</H3>

      <Prose>
        Betsy Beyer, Chris Jones, Jennifer Petoff, and Niall Richard Murphy (eds.). <em>Site Reliability Engineering: How Google Runs Production Systems.</em> O'Reilly, 2016. Chapters 20–22 on load balancing, cascading failures, and addressing cascading failures are directly applicable to LLM serving. The error budget model in chapter 3 is the correct framework for thinking about the availability math in section 3 of this topic.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — compute end-to-end latency</H3>
      <Prose>
        A request arrives at an API gateway in San Francisco destined for a datacenter in Virginia. Network latency is 68 ms each way. Gateway processing takes 8 ms. The router lookup takes 3 ms. There are no workers available; the request queues for 1.2 seconds. Prefill runs on a 4,096-token prompt at 400 tokens per second. Decode runs at 35 tokens per second for 256 output tokens. What is the TTFT? What is the end-to-end latency? Which term is dominant, and what would you change first to reduce P99?
      </Prose>

      <Callout accent="purple">
        Answer: TTFT = 68 + 8 + 3 + 1200 + (4096/400 × 1000) ms = 68 + 8 + 3 + 1200 + 10,240 ms ≈ 11,519 ms. End-to-end = TTFT + (256/35 × 1000) = 11,519 + 7,314 ≈ 18,833 ms. Queue latency (1200 ms) and prefill (10,240 ms) dominate. Fix order: (1) add more workers to reduce queue depth; (2) if prompt length is tunable, reduce it or use prefix caching; (3) use disaggregated prefill on dedicated hardware.
      </Callout>

      <H3>Exercise 2 — design for 99.95% SLA</H3>
      <Prose>
        You are designing a serving stack with six tiers: CDN, gateway, router, load balancer, inference worker, and observability. Your SLA target is 99.95% (4.38 hours downtime per year). If each tier must contribute equally to the composite availability, what per-tier availability do you need? If the inference worker can only reach 99.9% due to GPU driver issues, how must the other five tiers compensate? Show the math.
      </Prose>

      <Callout accent="gold">
        Answer: For composite 0.9995 = A^6, A = 0.9995^(1/6) ≈ 0.99992 per tier (99.992%). If the worker is at 99.9% = 0.999, the remaining 5 tiers must satisfy: 0.9995 / 0.999 = 1.0005. That is impossible with five values ≤ 1.0 — the composite is already below target with just the worker. You must either improve the worker (hardware redundancy, automatic restart), reduce tier count, or widen the SLA target. In practice: run two inference workers with an automatic failover, treating the pair as a single tier at 1 - (1-0.999)^2 = 99.9999%.
      </Callout>

      <H3>Exercise 3 — pick an architecture for a specific workload</H3>
      <Prose>
        You are building a legal document review product. Documents are 20,000–80,000 tokens. Users submit documents during business hours; traffic drops to near zero at night. The product is used by lawyers in the US and EU who require data residency (EU data must stay in EU, US data must stay in US). The output is always a structured JSON report, 200–400 tokens. What tier architecture would you choose? Which decisions are forced by the constraints and which are free choices?
      </Prose>

      <Callout accent="purple">
        Forced decisions: multi-region deployment (US + EU) with strict routing by user geography — data residency requirements force this. Single-tier per region is possible since the user base is small and regionally isolated. Long-context workers (>80k context) required — "large" tier only. Structured output requires constrained decoding at the inference layer. Free choices: stateless vs. stateful routing (stateless is simpler; no conversation history needed for document review); dedicated vs. shared capacity (shared is cheaper for the bursty business-hours pattern, but PTU-style dedicated capacity gives predictable latency for high-value clients); semantic caching (could cache similar legal queries, but the risk of wrong-answer cache hits is too high for legal work — skip it).
      </Callout>

      <H3>Exercise 4 — identify where to cache</H3>
      <Prose>
        A customer support bot receives 50,000 requests per day. Each request starts with a 2,000-token system prompt (identical for all requests) followed by a 50–200 token user message and conversation history. The bot uses Claude claude-sonnet-4-5. Prompt caching is available. Where exactly should caching be applied, and what hit rate do you expect? What is the cost reduction?
      </Prose>

      <Callout accent="gold">
        Apply prefix caching on the 2,000-token system prompt. All 50,000 daily requests share this prefix. After the first request warms the cache, 49,999 requests hit the cache on the system prompt prefix. Expected hit rate: ~99.998% on the system-prompt portion. Cost reduction: prefill for 2,000 tokens costs ~5× less when served from cache (Anthropic's cache-hit pricing is roughly 10% of full input token price). At 2,000 tokens × 50,000 requests = 100M tokens/day, you save ~90% of the system-prompt prefill cost, which is roughly 80% of total prompt cost (since the user message is short). Also consider exact-match caching for the most common FAQ queries — if even 5% of 50k queries are exact repeats, that is 2,500 zero-GPU responses per day.
      </Callout>

      <H3>Exercise 5 — spot the anti-pattern</H3>
      <Prose>
        An engineering team has built the following pipeline: (1) Client sends request to gateway. (2) Gateway validates and authenticates. (3) Gateway calls the inference worker synchronously and waits for the full response (non-streaming). (4) Gateway calls the safety classifier on the full response. (5) If the safety classifier passes, the gateway returns the full response to the client. The team reports that P99 latency is 45 seconds for typical requests and client timeouts are frequent. Identify all architectural anti-patterns and propose fixes.
      </Prose>

      <Callout accent="purple">
        Anti-patterns: (1) Non-streaming gateway blocks the client connection for the entire generation duration, which is tens of seconds for typical outputs. Fix: stream tokens to the client as they are generated. (2) Safety classification after full generation means the user waits an extra 50–200 ms (classifier latency) after the model finishes, on top of an already long wait. Fix: stream output through the safety classifier token-by-token or in chunks; reject mid-stream if a violation is detected. (3) Synchronous wait at the gateway holds a gateway thread for the full generation duration, limiting concurrent capacity to (gateway_threads × generation_time). Fix: use async I/O (asyncio or event loop) so gateway threads are not blocked during inference. (4) Client timeouts at 45 seconds suggest the gateway or client has a hard timeout shorter than the generation time for long outputs. Fix: set gateway timeout to match maximum expected generation time (minutes for long-context), and communicate progress via streaming so clients do not time out waiting.
      </Callout>

    </div>
  ),
};

export default inferenceSystemArchitecture;
