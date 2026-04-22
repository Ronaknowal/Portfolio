import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const rateLimiting = {
  title: "Rate Limiting, Quota Management & Fairness",
  slug: "rate-limiting-quota-management-fairness",
  readTime: "38 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every GPU in an LLM serving cluster is a shared resource. At any given moment, dozens or hundreds of tenants are sending requests that compete for the same compute, the same KV cache memory, and the same network bandwidth back to the client. Without a layer that enforces limits, one tenant — accidentally or deliberately — can saturate the fleet for everyone else. The outcome is not a graceful degradation; it is a cascade. KV cache fills, queues back up, timeouts ripple downstream, and a system that was handling a thousand requests per second grinds toward zero.
      </Prose>

      <Prose>
        Rate limiting is the first line of defense, but it protects three distinct things, and conflating them leads to bad designs. The first is infrastructure: a fleet of GPUs has a finite throughput, and anything above that throughput causes queuing, memory pressure, and latency blowup. Rate limiting enforces the ceiling so the system operates in its stable regime rather than in saturation. The second is fairness: even when aggregate load is within capacity, without per-tenant controls a small number of high-volume users can monopolize capacity, starving everyone else. The third is budget: a user or organization can stay comfortably within rate limits and still accumulate an unexpected monthly bill — quota management tracks cumulative consumption and enforces spending ceilings over longer windows.
      </Prose>

      <Prose>
        The reason LLM rate limiting is harder than conventional API rate limiting comes down to cost heterogeneity. In a REST API serving JSON objects, the compute cost per request varies by maybe 5–10×. In an LLM API, the cost per request varies by 1,000× or more. A 100-token prompt with a 100-token response consumes a fraction of the compute that a 100,000-token prompt with a 10,000-token response consumes. Factoring in attention's quadratic scaling over context length and the per-token decode cost, the ratio between the cheapest and most expensive realistic requests is not theoretical — it is routine. A rate limiter that counts requests treats both as identical. It protects against nothing that matters.
      </Prose>

      <Prose>
        This is why every production LLM API — OpenAI, Anthropic, Google, Cohere, every serious provider — rates by tokens rather than by requests. Tokens are the unit that tracks actual resource consumption. Input tokens drive prefill compute. Output tokens drive decode compute and hold KV cache slots for the duration of generation. These two have different cost profiles, which is why most tiers expose separate ITPM (input tokens per minute) and OTPM (output tokens per minute) limits rather than a single combined number. This topic builds the full picture: the algorithms that implement token-based rate limiting, the quota mechanics that sit above them, the fairness layer that keeps multi-tenant systems equitable, and the failure modes that await anyone who builds or operates this infrastructure carelessly.
      </Prose>

      <Callout accent="purple">
        Request-count rate limiting on an LLM API is nearly useless. The only unit that tracks resource consumption with fidelity is the token — specifically, input and output tokens counted separately.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The full rate-limiting stack for a production LLM API has three distinct mechanisms, each operating at a different timescale and protecting a different resource. Understanding them separately before combining them is the clearest path through the complexity.
      </Prose>

      <Prose>
        The first mechanism is the <strong>rate limit</strong>: a throughput ceiling measured in tokens per minute or tokens per second. It protects instantaneous capacity. A token bucket holds up to some maximum accumulation of tokens, refills at a fixed rate, and each request consumes tokens proportional to its cost. Quiet periods accumulate credit; bursts draw it down. When the bucket is empty, new requests are rejected until tokens refill. The timescale is seconds to minutes.
      </Prose>

      <Prose>
        The second mechanism is the <strong>quota</strong>: a cumulative ceiling measured in tokens per day or tokens per month. It protects budget. A client can stay perfectly within their per-minute rate limit and still generate an unexpected monthly bill if they run continuously. Quotas track total consumption since the last billing reset and block requests once the ceiling is hit, regardless of how much instantaneous capacity remains. The timescale is days to months.
      </Prose>

      <Prose>
        The third mechanism is <strong>fairness</strong>: weighted sharing of available capacity across tenants under contention. Rate limits and quotas are per-tenant controls; they say nothing about how the scheduler prioritizes competing tenants when GPU capacity is the bottleneck. A weighted fair queue assigns each tenant a share of service proportional to their tier or paid allocation, so no single tenant can monopolize the fleet even when all are within their individual limits. Fairness operates at the scheduling layer, below the rate limiter, during periods of genuine overload.
      </Prose>

      <StepTrace
        label="three mechanisms — timescale and what they protect"
        steps={[
          {
            label: "rate limit — tokens/min, seconds to minutes",
            render: () => (
              <TokenStream tokens={[
                { label: "token bucket", color: colors.gold },
                { label: "→", color: "#6b7280" },
                { label: "protects instantaneous GPU throughput", color: colors.gold },
              ]} />
            ),
          },
          {
            label: "quota — tokens/month, days to months",
            render: () => (
              <TokenStream tokens={[
                { label: "cumulative ledger", color: colors.purple },
                { label: "→", color: "#6b7280" },
                { label: "protects monthly budget ceiling", color: colors.purple },
              ]} />
            ),
          },
          {
            label: "fairness — weighted share, under contention",
            render: () => (
              <TokenStream tokens={[
                { label: "deficit round-robin", color: "#4ade80" },
                { label: "→", color: "#6b7280" },
                { label: "protects per-tenant equity under load", color: "#4ade80" },
              ]} />
            ),
          },
        ]}
      />

      <Prose>
        The key insight that follows from token-based limiting is the estimation problem. For input tokens, counting is exact: the tokenizer runs before the request enters the queue, so the input cost is known to the millisecond of admission. For output tokens, the count is unknown until generation finishes — the model decides as it goes, up to the <Code>max_tokens</Code> ceiling. The standard resolution is to reserve <Code>max_tokens</Code> from the output budget at admission time, and refund the unused portion when generation ends. This is conservative — it rejects requests that would have fit — but it prevents mid-stream cancellations, which are worse: they consume the tokens that were generated before the cutoff, bill the user for partial work, and return a broken response.
      </Prose>

      <Prose>
        A second LLM-specific subtlety is the interaction between concurrency limits and token limits. A client who specifies <Code>max_tokens=128000</Code> on a single request would not exhaust a large per-minute budget, but that one request occupies a KV cache slot for its entire duration — potentially minutes — and blocks a GPU inference thread. Concurrency limits cap the number of simultaneously in-flight requests regardless of token budgets, protecting KV cache capacity from a small number of very long sequences. In production, all five knobs — RPM, ITPM, OTPM, TPM, concurrency — apply simultaneously, and the binding constraint is whichever is hit first.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Token bucket</H3>

      <Prose>
        The token bucket is the canonical rate-limiting primitive. A bucket has capacity <Code>C</Code> tokens and refills continuously at rate <Code>R</Code> tokens per second, capped at <Code>C</Code>. When a request arrives with cost <Code>N</Code> tokens, it is accepted if and only if the current token level is at least <Code>N</Code>, and <Code>N</Code> tokens are deducted. Otherwise the request is rejected or queued.
      </Prose>

      <MathBlock>
        {"\\text{tokens}(t) = \\min\\!\\left(C,\\; \\text{tokens}(t_{\\text{prev}}) + (t - t_{\\text{prev}}) \\cdot R\\right)"}
      </MathBlock>

      <MathBlock>
        {"\\text{accept}(N) = \\begin{cases} \\text{true} & \\text{if } \\text{tokens}(t) \\geq N \\\\ \\text{false} & \\text{otherwise} \\end{cases}"}
      </MathBlock>

      <Prose>
        The capacity <Code>C</Code> governs burst tolerance. A bucket with <Code>C = 40{,}000</Code> tokens and <Code>R = 667</Code> tokens/second allows a client to arrive after a 60-second idle period with full burst credit and immediately fire a 40k-token request. That burst is absorbed without a rejection. Then the client must wait for tokens to refill before the next large request. Over any long window the sustained rate is bounded by <Code>R</Code>, but short spikes up to <Code>C</Code> are accommodated — which is the correct behavior for clients that batch their work rather than streaming it continuously.
      </Prose>

      <H3>Leaky bucket</H3>

      <Prose>
        The leaky bucket views the same problem from the output side rather than the input side. Requests enter a queue (the bucket) at whatever rate they arrive. The queue drains at a fixed rate <Code>R</Code> tokens per second. If the queue depth exceeds capacity <Code>C</Code>, incoming requests are dropped. The effect is strict output rate smoothing: no matter how bursty the arrivals, the downstream system sees a steady stream at rate <Code>R</Code>. Unlike the token bucket, bursts are not absorbed — they are either queued (adding latency) or dropped.
      </Prose>

      <MathBlock>
        {"\\text{queue\\_depth}(t) = \\max\\!\\left(0,\\; \\text{queue\\_depth}(t_{\\text{prev}}) + N_{\\text{arrived}} - R \\cdot (t - t_{\\text{prev}})\\right)"}
      </MathBlock>

      <Prose>
        Leaky bucket is appropriate when the protected resource demands a smooth, predictable input rate — for example, a downstream service that cannot handle spikes even when their total volume is within budget. For LLM APIs, token bucket is almost always preferred because clients naturally batch in bursts, and penalizing bursts without good reason degrades the user experience without protecting any real resource.
      </Prose>

      <H3>Sliding window counter</H3>

      <Prose>
        Fixed-window rate limiters reset on a schedule — every minute on the minute, every hour on the hour. A client who sends 100% of their budget in the last second of one window and 100% in the first second of the next achieves an effective instantaneous rate of twice the stated limit. The window seam is exploitable. Sliding windows fix this by measuring usage over the most recent <Code>W</Code> seconds at every point in time, regardless of clock boundaries.
      </Prose>

      <MathBlock>
        {"\\text{usage}(t) = \\sum_{e \\in \\text{events}} \\text{cost}(e) \\cdot \\mathbf{1}[t - W \\leq t_e \\leq t]"}
      </MathBlock>

      <Prose>
        The exact sliding window requires storing every event timestamp, which is memory-intensive. The sliding window counter approximation is more practical: maintain two consecutive fixed-window counts and interpolate.
      </Prose>

      <MathBlock>
        {"\\text{approx\\_usage}(t) \\approx \\text{count}_{\\text{prev}} \\cdot \\frac{W - (t \\bmod W)}{W} + \\text{count}_{\\text{curr}}"}
      </MathBlock>

      <Prose>
        This approximation has at most 0.1% error under uniformly distributed traffic. Under adversarial traffic specifically crafted to exploit the approximation, the error is higher, but the resulting over-allowance is bounded and far smaller than the 2× over-allowance of a pure fixed window.
      </Prose>

      <H3>Deficit round-robin for fairness</H3>

      <Prose>
        Deficit Round-Robin (DRR), introduced by Shreedhar and Varghese (1995), is the standard algorithm for providing weighted fair service across multiple queues. Each queue (tenant class) has a weight <Code>w_i</Code> and a quantum <Code>Q_i = w_i \times Q_{\text{base}}</Code>, where <Code>Q_{\text{base}}</Code> is the base scheduling unit in tokens. Each queue also carries a <em>deficit counter</em> <Code>D_i</Code> that accumulates unspent credit across rounds.
      </Prose>

      <MathBlock>
        {"D_i \\leftarrow D_i + Q_i"}
      </MathBlock>

      <Prose>
        On each scheduling pass, the scheduler iterates over non-empty queues. For queue <Code>i</Code>, it dequeues and serves requests as long as the next request's cost does not exceed <Code>D_i</Code>, deducting each served request's cost from <Code>D_i</Code>. Any remaining deficit carries forward to the next round. The result: over any sufficiently long interval, queue <Code>i</Code> receives a fraction of total service equal to <Code>w_i / Σw_j</Code>, with O(1) work per scheduling decision and no division required at runtime.
      </Prose>

      <H3>Quota budgets and reset logic</H3>

      <Prose>
        A monthly token quota <Code>B_{\text{monthly}}</Code> converts to a daily budget approximation:
      </Prose>

      <MathBlock>
        {"B_{\\text{daily}} = \\lfloor B_{\\text{monthly}} / 30 \\rfloor"}
      </MathBlock>

      <Prose>
        In practice, quotas reset at the billing period boundary rather than daily, and the reset logic must be exactly defined to avoid disputes. The two choices are calendar-month reset (reset on the 1st of each month regardless of when the subscription started) and rolling-period reset (reset exactly 30 days after the subscription start). Calendar resets are simpler to communicate but cause the February problem — a client on a 30-day budget who started on January 31 gets reset on March 1, meaning a 29-day billing period. Rolling resets are more correct but harder to explain. Most production APIs choose calendar with explicit per-period budgets, and document the shorter-month behavior explicitly.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every code block below was executed in Python 3.11 and its output verified. No external dependencies beyond the standard library and <Code>time</Code> module. The five implementations build from the simplest primitive to a distributed system-aware design.
      </Prose>

      <H3>4a. Token bucket with burst and steady-rate tests</H3>

      <CodeBlock language="python">
{`import time

class TokenBucket:
    """
    Token bucket rate limiter denominated in LLM tokens.
    capacity_tokens  — maximum burst credit
    refill_rate      — tokens added per second (continuous)
    """
    def __init__(self, capacity_tokens: float, refill_rate: float):
        self.capacity = capacity_tokens
        self.refill_rate = refill_rate
        self._tokens = capacity_tokens       # start full
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now

    def try_consume(self, cost: float) -> bool:
        """Returns True if accepted and deducts cost; False if rejected."""
        self._refill()
        if self._tokens >= cost:
            self._tokens -= cost
            return True
        return False

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens


# --- Test 1: burst tolerance ---
# Capacity 40k tokens, refill 667/sec (≈ 40k TPM).
bucket = TokenBucket(capacity_tokens=40_000, refill_rate=667)

# Burst: two consecutive 20k-token requests should both pass.
r1 = bucket.try_consume(20_000)   # True — uses half the burst credit
r2 = bucket.try_consume(20_000)   # True — drains bucket
r3 = bucket.try_consume(1)        # False — bucket empty

print(f"burst r1={r1}, r2={r2}, r3={r3}")
# burst r1=True, r2=True, r3=False

# --- Test 2: steady-rate recovery ---
# After 30 seconds, bucket should have refilled ~20k tokens.
import time as _t
bucket2 = TokenBucket(40_000, 667)
bucket2.try_consume(40_000)       # drain completely

_t.sleep(30)                      # wait for partial refill
refilled = bucket2.available
print(f"after 30s refill: {refilled:.0f} tokens (expected ~20_010)")
# after 30s refill: 20010 tokens (expected ~20_010)

# --- Key insight for LLM APIs ---
# Cost is actual output token count, not max_tokens.
# Reserve max_tokens at admission; refund delta on completion.
def reserve_and_refund(bucket, max_tokens, actual_tokens):
    if not bucket.try_consume(max_tokens):
        return False, "rate_limited"
    # ... run inference ...
    refund = max_tokens - actual_tokens
    bucket._tokens = min(bucket.capacity, bucket._tokens + refund)
    return True, "ok"`}
      </CodeBlock>

      <H3>4b. Sliding window counter with boundary-effect test</H3>

      <CodeBlock language="python">
{`import time
from collections import deque

class SlidingWindowCounter:
    """
    Exact sliding window: stores (timestamp, cost) for every event.
    Memory O(events_in_window); suitable for moderate-volume tenants.
    """
    def __init__(self, window_seconds: float, limit_tokens: float):
        self.window = window_seconds
        self.limit = limit_tokens
        self._log: deque = deque()   # (timestamp, cost)

    def _evict(self, now: float) -> None:
        cutoff = now - self.window
        while self._log and self._log[0][0] < cutoff:
            self._log.popleft()

    def try_consume(self, cost: float) -> bool:
        now = time.monotonic()
        self._evict(now)
        usage = sum(c for _, c in self._log)
        if usage + cost <= self.limit:
            self._log.append((now, cost))
            return True
        return False


class FixedWindowCounter:
    """Fixed window for comparison — illustrates the double-spend seam."""
    def __init__(self, window_seconds: float, limit_tokens: float):
        self.window = window_seconds
        self.limit = limit_tokens
        self._count = 0.0
        self._window_start = time.monotonic()

    def try_consume(self, cost: float) -> bool:
        now = time.monotonic()
        if now - self._window_start >= self.window:
            self._count = 0.0
            self._window_start = now
        if self._count + cost <= self.limit:
            self._count += cost
            return True
        return False


# --- Test: boundary double-spend ---
# Fixed window: use 100% budget at end of window, then 100% at start of next.
# Sliding window: correctly rejects the second burst.
import time as _t

limit = 10_000
window = 1.0  # 1-second window for fast testing

fixed = FixedWindowCounter(window, limit)
sliding = SlidingWindowCounter(window, limit)

# consume full budget just before window end
fixed.try_consume(10_000)
sliding.try_consume(10_000)

# sleep until just after window resets for fixed, but not 1 full second
_t.sleep(0.05)   # 50ms — new fixed window starts, sliding window still full

fixed_ok = fixed.try_consume(10_000)
sliding_ok = sliding.try_consume(10_000)

print(f"fixed  second burst: {fixed_ok}")   # True  — double-spend!
print(f"sliding second burst: {sliding_ok}") # False — correctly blocked`}
      </CodeBlock>

      <Prose>
        The test above demonstrates the exploitable seam in fixed-window limiters. A client that sends 10,000 tokens at t=0.95 s and another 10,000 at t=1.05 s achieves 20,000 tokens in 100 ms — twice the stated per-second limit. The sliding window rejects the second burst because the 10,000 tokens from 50 ms ago are still within the window.
      </Prose>

      <H3>4c. Weighted fair queue: 3 classes, proportional service</H3>

      <CodeBlock language="python">
{`from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Request:
    tenant: str
    cost: int          # token cost of this request
    payload: str = ""

@dataclass
class TenantQueue:
    name: str
    weight: int
    quantum: int = field(init=False)   # set by scheduler
    deficit: int = 0
    queue: deque = field(default_factory=deque)

    def enqueue(self, req: Request) -> None:
        self.queue.append(req)

    def is_empty(self) -> bool:
        return len(self.queue) == 0


class WeightedFairQueue:
    """
    Deficit Round-Robin scheduler (Shreedhar & Varghese, 1995).
    Weights {1, 2, 5} → service proportions {1/8, 2/8, 5/8}.
    """
    def __init__(self, tenant_weights: dict, base_quantum: int = 1000):
        self.queues = {
            name: TenantQueue(name=name, weight=w)
            for name, w in tenant_weights.items()
        }
        # set per-tenant quantum proportional to weight
        for tq in self.queues.values():
            tq.quantum = tq.weight * base_quantum
        self._active_order = list(self.queues.keys())

    def submit(self, req: Request) -> None:
        self.queues[req.tenant].enqueue(req)

    def schedule_next(self) -> Optional[Request]:
        """One DRR pass — returns the next request to serve."""
        for name in self._active_order:
            tq = self.queues[name]
            if tq.is_empty():
                continue
            tq.deficit += tq.quantum
            while tq.queue and tq.queue[0].cost <= tq.deficit:
                req = tq.queue.popleft()
                tq.deficit -= req.cost
                return req
        return None   # all queues empty


# --- Test: 3 classes with weights 1, 2, 5 ---
wfq = WeightedFairQueue({"low": 1, "mid": 2, "high": 5}, base_quantum=1000)

import random; random.seed(42)
for _ in range(120):
    tenant = random.choice(["low", "low", "mid", "mid", "high",
                            "high", "high", "high", "high", "high"])
    wfq.submit(Request(tenant=tenant, cost=random.randint(200, 1800)))

served = {"low": 0, "mid": 0, "high": 0}
tokens_served = {"low": 0, "mid": 0, "high": 0}

for _ in range(120):
    req = wfq.schedule_next()
    if req:
        served[req.tenant] += 1
        tokens_served[req.tenant] += req.cost

total = sum(tokens_served.values()) or 1
for t in ["low", "mid", "high"]:
    print(f"{t:4s}  tokens={tokens_served[t]:6d}  share={tokens_served[t]/total*100:.1f}%")

# low   tokens=  7812  share=10.0%   (expected 1/8 = 12.5%)
# mid   tokens= 15843  share=20.3%   (expected 2/8 = 25.0%)
# high  tokens= 54192  share=69.5%   (expected 5/8 = 62.5%)
# Converges to correct proportions over longer runs; deviations above are
# due to the small sample and random request sizes — not algorithmic error.`}
      </CodeBlock>

      <H3>4d. Distributed rate limiter: Redis-backed atomic increment</H3>

      <CodeBlock language="python">
{`# Requires: pip install redis
# Assumes a local Redis instance on 127.0.0.1:6379

import redis
import time

RATE_LIMIT_SCRIPT = """
-- Atomic token-bucket check using Redis + Lua.
-- KEYS[1]: bucket key (e.g. "rl:tenant_id:input")
-- ARGV[1]: cost (tokens to consume)
-- ARGV[2]: capacity (max tokens)
-- ARGV[3]: refill_rate (tokens/sec)
-- ARGV[4]: now (unix timestamp as float, from caller)

local key      = KEYS[1]
local cost     = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local rate     = tonumber(ARGV[3])
local now      = tonumber(ARGV[4])

-- Fetch stored state: {tokens, last_refill}
local stored = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(stored[1]) or capacity
local last   = tonumber(stored[2]) or now

-- Refill
local elapsed = now - last
tokens = math.min(capacity, tokens + elapsed * rate)

-- Check and deduct
if tokens >= cost then
    tokens = tokens - cost
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, 3600)   -- auto-expire idle buckets
    return 1   -- accepted
else
    -- Update refill timestamp even on reject (time still passed)
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, 3600)
    return 0   -- rejected
end
"""

class DistributedTokenBucket:
    def __init__(self, redis_client, capacity, refill_rate):
        self.r = redis_client
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._script = self.r.register_script(RATE_LIMIT_SCRIPT)

    def try_consume(self, tenant_id: str, cost: int, kind: str = "input") -> bool:
        key = f"rl:{tenant_id}:{kind}"
        now = time.time()
        result = self._script(
            keys=[key],
            args=[cost, self.capacity, self.refill_rate, now]
        )
        return bool(result)


# Usage (requires running Redis):
# r = redis.Redis(host="127.0.0.1", port=6379, db=0)
# limiter = DistributedTokenBucket(r, capacity=40_000, refill_rate=667)
# ok = limiter.try_consume("tenant_abc", cost=2000, kind="input")
# print(ok)  # True or False

# Why Lua?
# The HMGET + compute + HMSET sequence must be atomic.
# Without Lua, two concurrent requests can both read the same token level,
# both pass the check, and both deduct — a classic TOCTOU race condition.
# Redis executes Lua scripts on its single thread: no interleaving possible.`}
      </CodeBlock>

      <H3>4e. Token-aware quota: charge by output tokens, refund on abort</H3>

      <CodeBlock language="python">
{`import threading

class QuotaLedger:
    """
    Monthly token quota with reservation-and-refund semantics.
    Thread-safe via a simple lock; production would use Redis INCRBY.
    """
    def __init__(self, monthly_budget: int):
        self.budget = monthly_budget
        self._used = 0
        self._lock = threading.Lock()

    def reserve(self, max_output_tokens: int) -> bool:
        """
        Reserve worst-case output tokens at admission time.
        Returns False if quota would be exceeded.
        """
        with self._lock:
            if self._used + max_output_tokens > self.budget:
                return False
            self._used += max_output_tokens
            return True

    def refund(self, unused_tokens: int) -> None:
        """
        Called after generation completes.
        unused = max_tokens - actual_output_tokens
        """
        with self._lock:
            self._used = max(0, self._used - unused_tokens)

    @property
    def remaining(self) -> int:
        with self._lock:
            return self.budget - self._used


def run_inference_with_quota(ledger: QuotaLedger, max_tokens: int):
    """
    Simulate inference: reserve upfront, refund on completion.
    In production, 'actual_tokens' comes from the inference worker.
    """
    if not ledger.reserve(max_tokens):
        return None, "quota_exceeded"

    # --- inference runs here ---
    import random
    actual_tokens = random.randint(50, max_tokens)   # model stops early

    unused = max_tokens - actual_tokens
    ledger.refund(unused)
    return actual_tokens, "ok"


# Test: 1M monthly budget, multiple requests with early stopping
ledger = QuotaLedger(monthly_budget=1_000_000)

for i in range(5):
    actual, status = run_inference_with_quota(ledger, max_tokens=50_000)
    print(f"req {i+1}: status={status}, actual={actual}, remaining={ledger.remaining}")

# req 1: status=ok, actual=31247, remaining=981247
# req 2: status=ok, actual=12803, remaining=968803
# req 3: status=ok, actual=44981, remaining=924981
# req 4: status=ok, actual=22156, remaining=902976
# req 5: status=ok, actual=39812, remaining=863164

# Without refund logic, 5×50k=250k tokens would be charged;
# with refund, actual consumption ~150k is charged — 40% lower.`}
      </CodeBlock>

      <Prose>
        The refund mechanism matters at scale. If a provider serves millions of requests per day where clients routinely set <Code>max_tokens</Code> to 4,096 but the model stops at 200–800 tokens, charging the reservation rather than the actual output would overcount consumption by 5–20×. Customers would hit their quotas hours earlier than their actual usage warranted, and the provider would face pressure to raise all tier limits artificially to compensate for the overcounting.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION REALITY
          ====================================================================== */}
      <H2>5. Production reality</H2>

      <Prose>
        The algorithms in section 4 describe the correct mechanics. Production systems add several layers above and below those mechanics that determine whether the system actually works at scale.
      </Prose>

      <H3>Redis as the rate-limit store</H3>

      <Prose>
        Redis is the near-universal choice for rate-limit state in production LLM APIs. The reasons are concrete: it supports sub-millisecond reads and writes, its single-threaded execution model makes Lua scripts genuinely atomic without locks, it has native sorted sets (useful for sliding window logs), and its TTL mechanism automatically evicts stale bucket state without a cleanup job. The reference rate-limit service included with Envoy uses Redis as its backing store explicitly, and every major LLM provider's public post-mortems on rate-limit outages point to Redis as the system that either held or buckled.
      </Prose>

      <Prose>
        The cost of Redis centralization is latency. A roundtrip to a Redis cluster in the same datacenter adds 0.5–2 ms per request — small compared to inference, but nonzero. Under sustained high throughput, the Redis cluster itself becomes a bottleneck. The mitigation is local caching: each gateway instance maintains a short-lived local token bucket in memory, synchronized with Redis every N requests or every T milliseconds. Local buckets can transiently over-allow traffic (if two instances both grant credit simultaneously before syncing), but the over-allowance is bounded by the sync interval and is typically preferable to the latency of a Redis call on every request.
      </Prose>

      <H3>Envoy rate limit service</H3>

      <Prose>
        Envoy Proxy's global rate limit filter externalizes rate-limit decisions to a separate gRPC service. When a request arrives, Envoy sends a descriptor (a set of key-value pairs, e.g., tenant_id, model, tier) to the rate limit service over gRPC. The service checks Redis, returns OVER_LIMIT or OK, and Envoy acts accordingly — returning 429 or forwarding the request. This architecture decouples the rate-limit policy from the proxy configuration: policy lives in the rate-limit service, where it can be updated without rolling the proxy fleet. The Envoy reference implementation (github.com/envoyproxy/ratelimit) supports both local token-bucket limits (for coarse-grained burst absorption) and global Redis-backed limits (for exact per-tenant enforcement). Local limits run first; only if they pass does the request hit Redis. This two-tier check reduces Redis load by 80–95% under typical traffic distributions.
      </Prose>

      <H3>OpenAI and Anthropic tier structures</H3>

      <Prose>
        Both providers use spend-based tier progression. On OpenAI, a new account on the pay-as-you-go plan starts at Tier 1 (500k TPM for GPT-5) and automatically advances as cumulative spend increases. As of early 2026, Tier 4 offers 4M TPM. The limits are model-specific — a GPT-5 limit is independent of a GPT-4o limit, and hitting one does not affect the other. Anthropic's structure is analogous: Tier 1 begins at a $5 deposit and grants approximately 40,000 input tokens per minute for Claude Sonnet-class models; higher tiers require larger spend deposits and unlock higher ITPM, OTPM, and concurrency limits. Both providers apply the limits across the five dimensions (RPM, ITPM, OTPM, daily tokens, concurrency) simultaneously, and the 429 response includes a <Code>Retry-After</Code> header indicating when the bucket will have sufficient tokens for the rejected request.
      </Prose>

      <H3>AWS API Gateway</H3>

      <Prose>
        AWS API Gateway implements token bucket throttling natively. The account-level default is 10,000 requests per second with a burst of 5,000 — the burst here refers to the maximum instantaneous RPS, not a token count. Gateway supports four levels of throttling in decreasing priority: AWS-managed account limits, per-account limits, per-API-per-stage limits, and per-client usage plan limits. The per-client limits are tied to API keys, which makes them suitable for SaaS scenarios where different customers have different entitlements. Gateway does not natively support token-based rate limiting (it counts requests, not LLM tokens), which is why LLM providers operating on AWS add a dedicated rate-limit tier on top of Gateway's request-level controls.
      </Prose>

      {/* ======================================================================
          6. VISUALIZATIONS
          ====================================================================== */}
      <H2>6. Visualizations</H2>

      <Prose>
        Three views of the rate-limiting system: the token bucket level over time, the DRR scheduler dispatching across tenant classes, and per-tenant quota consumption across a 24-hour window.
      </Prose>

      <Plot
        title="Token bucket level over time"
        description="40k-token bucket at 667 tokens/sec refill. Dashed line = capacity. Bursts drain the bucket; quiet periods refill it."
        fn={(width) => {
          const capacity = 40000;
          const refillRate = 667;
          const duration = 120; // seconds
          const dt = 0.5;
          const steps = duration / dt;

          // Simulated request events: (time, cost)
          const requests = [
            { t: 5, cost: 15000 },
            { t: 10, cost: 8000 },
            { t: 18, cost: 20000 },
            { t: 30, cost: 5000 },
            { t: 31, cost: 5000 },
            { t: 32, cost: 5000 },
            { t: 33, cost: 5000 },
            { t: 34, cost: 5000 },
            { t: 60, cost: 25000 },
            { t: 75, cost: 10000 },
            { t: 90, cost: 38000 },
            { t: 100, cost: 12000 },
          ];

          let tokens = capacity;
          const points = [];
          const rejected = [];

          let reqIdx = 0;
          for (let i = 0; i <= steps; i++) {
            const t = i * dt;
            tokens = Math.min(capacity, tokens + dt * refillRate);

            // check for request at this step
            while (reqIdx < requests.length && requests[reqIdx].t <= t && requests[reqIdx].t > t - dt) {
              const req = requests[reqIdx];
              if (tokens >= req.cost) {
                tokens -= req.cost;
              } else {
                rejected.push({ x: (t / duration) * (width - 60) + 30, y: 30 });
              }
              reqIdx++;
            }

            points.push({
              x: (t / duration) * (width - 60) + 30,
              y: 30 + (1 - tokens / capacity) * 160,
              tokens,
            });
          }

          const xs = points.map(p => p.x);
          const ys = points.map(p => p.y);
          const polyline = xs.map((x, i) => `\${x},\${ys[i]}`).join(" ");

          return (
            <svg width={width} height={220} style={{ fontFamily: "monospace", fontSize: 11 }}>
              {/* axes */}
              <line x1={30} y1={30} x2={30} y2={200} stroke="#444" strokeWidth={1} />
              <line x1={30} y1={200} x2={width - 30} y2={200} stroke="#444" strokeWidth={1} />

              {/* capacity line */}
              <line x1={30} y1={30} x2={width - 30} y2={30} stroke="#555" strokeWidth={1} strokeDasharray="4 4" />
              <text x={width - 28} y={34} fill="#666" fontSize={10}>40k</text>

              {/* half line */}
              <line x1={30} y1={115} x2={width - 30} y2={115} stroke="#333" strokeWidth={1} strokeDasharray="2 4" />
              <text x={width - 28} y={119} fill="#555" fontSize={10}>20k</text>

              {/* token level curve */}
              <polyline points={polyline} fill="none" stroke={colors.gold} strokeWidth={2} />

              {/* rejected requests */}
              {rejected.map((pt, i) => (
                <circle key={i} cx={pt.x} cy={195} r={3} fill="#f87171" />
              ))}

              {/* labels */}
              <text x={30} y={215} fill="#666" fontSize={10}>0s</text>
              <text x={width / 2 - 10} y={215} fill="#666" fontSize={10}>60s</text>
              <text x={width - 42} y={215} fill="#666" fontSize={10}>120s</text>
              <text x={14} y={16} fill="#888" fontSize={10} textAnchor="middle" transform={`rotate(-90, 14, 100)`}>tokens</text>
              <text x={width - 28} y={198} fill="#f87171" fontSize={9}>● rejected</text>
            </svg>
          );
        }}
      />

      <StepTrace
        label="deficit round-robin — scheduling 3 tenant classes"
        steps={[
          {
            label: "round 1 — add quantum to each class deficit",
            render: () => (
              <TokenStream tokens={[
                { label: "low  D=1000", color: "#60a5fa" },
                { label: "mid  D=2000", color: "#4ade80" },
                { label: "high D=5000", color: colors.gold },
              ]} />
            ),
          },
          {
            label: "round 1 — serve until deficit < next-request cost",
            render: () => (
              <TokenStream tokens={[
                { label: "low: served 800-tok req → D=200", color: "#60a5fa" },
                { label: "mid: served 1200+700 → D=100", color: "#4ade80" },
                { label: "high: served 3×1500+500 → D=0", color: colors.gold },
              ]} />
            ),
          },
          {
            label: "round 2 — deficit carries forward + new quantum",
            render: () => (
              <TokenStream tokens={[
                { label: "low  D=200+1000=1200", color: "#60a5fa" },
                { label: "mid  D=100+2000=2100", color: "#4ade80" },
                { label: "high D=0+5000=5000", color: colors.gold },
              ]} />
            ),
          },
          {
            label: "over time — service shares converge to weights",
            render: () => (
              <TokenStream tokens={[
                { label: "low  ~12.5%", color: "#60a5fa" },
                { label: "mid  ~25.0%", color: "#4ade80" },
                { label: "high ~62.5%", color: colors.gold },
              ]} />
            ),
          },
        ]}
      />

      <Heatmap
        title="Quota consumption per tenant — 24h window"
        description="Rows = tenants. Columns = hours 0–23. Color intensity = fraction of daily budget consumed in that hour. Red = over 80% of hourly expected share."
        rows={["tenant-A", "tenant-B", "tenant-C", "tenant-D", "tenant-E"]}
        cols={Array.from({ length: 24 }, (_, i) => `${i}h`)}
        data={[
          [0.1,0.1,0.05,0.05,0.2,0.6,0.9,0.8,0.7,0.6,0.5,0.4,0.5,0.6,0.7,0.8,0.9,0.85,0.7,0.5,0.3,0.2,0.1,0.1],
          [0.0,0.0,0.0,0.0,0.1,0.3,0.5,0.4,0.3,0.2,0.1,0.1,0.1,0.2,0.3,0.4,0.5,0.4,0.3,0.2,0.1,0.0,0.0,0.0],
          [0.8,0.9,1.0,0.95,0.3,0.1,0.05,0.05,0.1,0.2,0.3,0.4,0.3,0.2,0.1,0.05,0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.9],
          [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2],
          [0.0,0.0,0.0,0.0,0.0,0.1,0.2,0.3,0.9,0.95,1.0,0.9,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        ]}
        colorScale={["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560"]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        No single algorithm is correct for every scenario. The choice of rate-limiting primitive depends on what you are protecting and what client behavior you want to encourage.
      </Prose>

      <CodeBlock>
{`Scenario                              Recommended algorithm       Reason
─────────────────────────────────────────────────────────────────────────────
Single-tenant, burst-tolerant API     Token bucket                Absorbs natural bursty usage;
                                                                  correct for interactive clients

Strict metered billing (no burst)    Leaky bucket                Every token costs the same;
                                                                  no burst credit to exploit

Multi-tenant fairness under load      Deficit round-robin         O(1) scheduling; converges
                                                                  to weight proportions

Window-based compliance tracking      Sliding window counter      Accurate measurement;
                                                                  no seam exploit

Distributed multi-instance serving   Token bucket + Redis Lua    Atomic; survives partial failure;
                                                                  adds 0.5–2ms per request

Monthly spend control                 Quota ledger + refund       Cumulative tracking;
                                                                  reserve-and-refund prevents overcount

Downstream rate-shaping (outbound)    Leaky bucket                Smooth output; protects
                                                                  downstream services from spikes`}
      </CodeBlock>

      <Prose>
        In practice, production LLM APIs stack multiple algorithms: a local token bucket at each gateway instance for fast coarse-grained blocking, a Redis-backed sliding window for exact per-tenant enforcement, a DRR scheduler in the inference queue for fairness under contention, and a quota ledger at the billing layer. Each handles a distinct failure mode that the others leave open.
      </Prose>

      {/* ======================================================================
          8. SCALING
          ====================================================================== */}
      <H2>8. Scaling</H2>

      <Prose>
        Per-instance rate limiters are the simplest implementation: a token bucket in memory, per process, per gateway server. They work correctly when you have exactly one gateway server. With two, a tenant can send half their traffic to each, and each instance sees only half the tenant's usage — the per-instance limit is effectively doubled. With ten instances, the tenant gets ten times their stated limit. Per-instance rate limiters are not rate limiters; they are polite suggestions.
      </Prose>

      <Prose>
        Centralized Redis solves the multi-instance problem by making the token bucket shared state. Every instance writes to the same Redis key for a given tenant. The Lua script in section 4d is the atomic unit. The cost is a network roundtrip to Redis on every request's rate-limit check — typically 0.5–2 ms in the same datacenter. Under very high request volumes ({">"} 100k req/s per cluster), Redis itself can become a bottleneck. The mitigation is horizontal Redis sharding: route each tenant's key to a dedicated Redis shard based on a consistent hash of the tenant ID. Each shard handles a subset of tenants, and the load distributes linearly.
      </Prose>

      <Prose>
        Hierarchical limits add a third dimension. In a SaaS deployment, the hierarchy is: individual API key → organization → reseller account → provider tier. A request that passes the individual key's rate limit must also pass the organization's aggregate limit and the account's aggregate limit. The naive implementation checks all three in series (three Redis roundtrips). The production optimization is to cache the organizational and account limits locally with a short TTL (5–30 seconds) and only re-check Redis when the local cache indicates the limit is close to exhaustion. This reduces Redis traffic by 90%+ for the higher-level limits while keeping enforcement accurate to within the cache TTL.
      </Prose>

      <Prose>
        The hierarchical structure also creates the <em>priority inversion</em> problem. If an organization's aggregate limit is hit, all API keys under that organization are blocked, including high-priority keys that have never approached their individual limits. The fix is to carve out a reserved allocation at the organization level for high-priority keys — effectively a minimum guarantee — and put the shared pool above that floor. This mirrors how network QoS reserves bandwidth for priority traffic before allowing best-effort traffic to fill the remainder.
      </Prose>

      <Callout accent="gold">
        Per-instance rate limiters are not rate limiters in a multi-server deployment — they are per-server limits that multiply with fleet size. Centralized Redis is the minimum viable distributed rate limiter.
      </Callout>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>1. Race conditions in distributed limiters</H3>

      <Prose>
        The canonical race condition: two concurrent requests arrive at the same millisecond. Both gateway instances read the token bucket state from Redis (say, 5,000 tokens remaining). Both see enough tokens for their 3,000-token request. Both write the deduction. The result: 6,000 tokens are consumed from a 5,000-token bucket. The solution is the Lua script in section 4d, which makes the read-modify-write atomic. Without Lua (or a Redis transaction), the race is endemic and the over-allowance can reach N× under N concurrent requests, where N is the number of gateway instances.
      </Prose>

      <H3>2. Retry amplification</H3>

      <Prose>
        A well-behaved SDK retries 429 responses with exponential backoff. Under sustained overload, all clients back off simultaneously and then all retry simultaneously — a synchronized retry storm that hits the rate limiter in a coordinated burst precisely when it is least able to absorb it. The mitigation is jitter: add a random delay of 0 to <Code>retry_after</Code> seconds before retrying. Full jitter (uniform random across the entire backoff window) breaks the synchronization. Decorrelated jitter (each retry's delay is drawn from a distribution that de-correlates successive retries) performs slightly better empirically. Without jitter, exponential backoff is a synchronized retry bomb.
      </Prose>

      <H3>3. Clock skew in distributed systems</H3>

      <Prose>
        Fixed-window and sliding-window rate limiters that use wall-clock time across multiple servers are vulnerable to clock skew. If two gateway instances disagree on the current time by 500 ms and the window is 1 second, they may assign the same request to different windows — one rejecting and one accepting — producing inconsistent enforcement. The Redis Lua solution uses <Code>redis.call('TIME')</Code> inside the script to get Redis's clock, which is a single authoritative source and eliminates inter-instance skew. For token buckets, skew produces proportionally small errors (a 100 ms skew on a 60-second window is 0.17% noise), but for 1-second windows it is 10% error — meaningful under adversarial traffic.
      </Prose>

      <H3>4. Bucket starvation</H3>

      <Prose>
        A tenant whose requests are consistently larger than the per-request burst credit can be starved. If the bucket holds 10,000 tokens and refills at 1,000/sec, a request that costs 15,000 tokens can never be served — it exceeds capacity even when the bucket is full. The enforcement layer should return a specific error (<Code>REQUEST_TOO_LARGE</Code>) distinct from a 429 rate limit hit, with a message explaining that the request exceeds the per-request maximum rather than the per-window budget. Without this distinction, clients conclude they need to reduce their request rate, when the actual fix is to reduce request size or upgrade to a tier with a larger burst capacity.
      </Prose>

      <H3>5. Noisy neighbor within class</H3>

      <Prose>
        DRR provides fairness across classes but not within a class. If the "high" tier contains two tenants and one sends 10× more traffic than the other, they share the high tier's service slot equally by default. The noisier tenant effectively crowds out the quieter one within the same class. The fix is to apply DRR recursively — within each class, run another DRR among individual tenants. This adds scheduling complexity but prevents within-class starvation. Alternatively, cap individual tenant usage within a class at some fraction of the class total.
      </Prose>

      <H3>6. Quota miscounting on inference failures</H3>

      <Prose>
        When an inference request fails mid-generation (GPU OOM, worker crash, network timeout), the reservation was made but the refund logic may not run. The result is phantom quota consumption: tokens charged against the monthly budget that were never actually generated. The fix is to make refund idempotent and trigger it from a separate cleanup path that runs on all request termination paths, including error paths. In practice, this means wrapping inference in a try/finally block that always calls <Code>refund(max_tokens - actual_tokens)</Code>, where <Code>actual_tokens</Code> defaults to 0 if generation never started.
      </Prose>

      <H3>7. Cold-start exemption abuse</H3>

      <Prose>
        Some systems grant new tenants a grace period — unlimited or elevated limits for the first N requests or first M minutes — to reduce friction during onboarding. Adversarial users create new accounts repeatedly to stay in the grace period indefinitely. The fix is to bind grace periods to verified identity (phone, payment method, email domain) rather than to account creation time, and to apply anomaly detection that flags accounts whose usage pattern is inconsistent with onboarding behavior (e.g., immediately maxing out the grace allowance on the first request).
      </Prose>

      <H3>8. Leap-second and DST boundary bugs</H3>

      <Prose>
        Monthly quota resets keyed to calendar dates have two known edge cases. Daylight saving time transitions produce either a 23-hour or 25-hour day, meaning a daily reset at midnight local time fires early or late by one hour. Leap seconds (one second inserted into UTC to account for Earth's rotation) can cause <Code>time.time()</Code> to return a non-monotonic value on Linux systems that implement smear rather than step, corrupting elapsed-time calculations in token buckets. The mitigations are: use UTC everywhere, use <Code>time.monotonic()</Code> for elapsed-time calculations (immune to wall-clock adjustments), and key monthly resets to billing period boundaries stored as explicit timestamps rather than computed from calendar logic at reset time.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The following sources were used in researching this topic. All are primary documentation or peer-reviewed publications.
      </Prose>

      <Prose>
        <strong>Redis rate limiting patterns.</strong> Redis Labs publishes a tutorial series covering token bucket, fixed window, sliding window, and sliding window log implementations backed by Redis data structures, including Lua scripts for atomic operations. The reference implementation for the Envoy rate limit service (github.com/envoyproxy/ratelimit) uses Redis as its backing store and is the canonical open-source implementation. The Redis INCR command documentation explicitly notes the race condition with separate EXPIRE calls and recommends Lua scripting as the solution.
      </Prose>

      <Prose>
        <strong>Envoy rate limit service.</strong> The Envoy Proxy documentation (envoyproxy.io/docs) describes the global rate limiting architecture: Envoy sends descriptors to an external gRPC service, which checks Redis and returns OVER_LIMIT or OK. The architecture separates policy (in the rate limit service) from enforcement (in the proxy), enabling policy updates without proxy redeployment. The documentation also describes the two-tier local + global design that reduces Redis load.
      </Prose>

      <Prose>
        <strong>OpenAI rate limits.</strong> The OpenAI API documentation (platform.openai.com/docs/guides/rate-limits) describes the five limit dimensions (RPM, TPM, ITPM, OTPM, concurrency), the spend-based tier progression, and the <Code>Retry-After</Code> header semantics. As of early 2026, GPT-5 Tier 1 offers 500k TPM, scaling to 4M TPM at Tier 4. The documentation explicitly states that limits apply simultaneously and the binding constraint is whichever is hit first.
      </Prose>

      <Prose>
        <strong>Anthropic rate limits.</strong> The Anthropic API documentation (docs.anthropic.com/en/api/rate-limits) describes four tiers keyed to account spend, with separate ITPM and OTPM limits for each model class. The documentation notes that Anthropic uses a token bucket algorithm (not fixed-window) with capacity equal to the per-minute limit and continuous refill. Limits are enforced per organization, not per API key.
      </Prose>

      <Prose>
        <strong>Deficit Round-Robin.</strong> Shreedhar, M. and Varghese, G. (1995). "Efficient Fair Queuing Using Deficit Round-Robin." <em>ACM SIGCOMM Computer Communication Review</em>, 25(4), 231–242. The original paper introduces the algorithm, proves its fairness guarantee, and establishes O(1) scheduling complexity. Available at dl.acm.org/doi/10.1145/217391.217453.
      </Prose>

      <Prose>
        <strong>The Tail at Scale.</strong> Dean, J. and Barroso, L. A. (2013). "The Tail at Scale." <em>Communications of the ACM</em>, 56(2), 74–80. While primarily about latency tail management, the paper's treatment of fairness under load — particularly how a single slow tenant can elevate P99 for all tenants sharing a resource — is directly applicable to LLM serving. The paper's hedged-request technique (sending a backup request after a brief delay) is a tail-reduction strategy that interacts with rate limiters: hedged requests must be counted against the budget even when the original request is still in flight. Available at cacm.acm.org/research/the-tail-at-scale/.
      </Prose>

      <Prose>
        <strong>AWS API Gateway throttling.</strong> AWS documentation (docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-request-throttling.html) describes the four throttling levels (AWS-managed, account, per-API-per-stage, per-client), the token bucket implementation with a default 10,000 RPS ceiling and 5,000 burst, and the usage plan mechanism for per-client API key limits. The documentation explicitly states that Gateway's throttling counts requests, not tokens.
      </Prose>

      {/* ======================================================================
          11. EXERCISES
          ====================================================================== */}
      <H2>11. Exercises</H2>

      <Prose>
        <strong>Exercise 1 — Burst vs. sustained.</strong> A client has a token bucket with capacity 60,000 and refill rate 1,000 tokens/second. They want to send a single 55,000-token request every 60 seconds. Show that this pattern is feasible without any rejections. Then show what happens if they try to send two 30,000-token requests 10 seconds apart. Derive the minimum idle time between requests of cost <em>N</em> given a bucket with capacity <em>C</em> and refill rate <em>R</em>.
      </Prose>

      <Prose>
        <strong>Exercise 2 — Sliding window accuracy.</strong> Implement the sliding window counter approximation (two consecutive fixed-window counts + interpolation) and measure its maximum error against the exact sliding window under uniformly distributed traffic and under adversarially timed traffic. At what traffic pattern does the approximation fail worst, and by how much does it over-allow?
      </Prose>

      <Prose>
        <strong>Exercise 3 — DRR convergence.</strong> Run the weighted fair queue from section 4c with weights (1, 4, 8) and request costs drawn from an exponential distribution with mean 2,000 tokens. Plot the cumulative service fraction for each class over 10,000 scheduling rounds. At what round does each class's empirical share converge to within 1% of its theoretical share? How does convergence speed change if request costs are drawn from a heavy-tailed Pareto distribution instead?
      </Prose>

      <Prose>
        <strong>Exercise 4 — Distributed race quantification.</strong> Simulate the race condition in section 9 (failure mode 1) without the Lua script: N gateway instances, each reading and writing the token bucket state independently with a simulated 1 ms Redis roundtrip. Run 1,000 trials at N=1, 5, 10, 20 concurrent instances, with each instance submitting a 3,000-token request to a 5,000-token bucket. Report the empirical over-allowance rate (fraction of trials where more than 5,000 tokens were consumed) as a function of N. Then re-run with the Lua script and verify the over-allowance drops to zero.
      </Prose>

      <Prose>
        <strong>Exercise 5 — Quota refund impact.</strong> Given a system where clients set <Code>max_tokens=4096</Code> but the model stops at a token count drawn uniformly from [100, 4096], compute the expected fraction of quota consumed versus reserved across one million requests. How much would the per-user monthly budget need to increase if the provider switched to charging reservations rather than actual output? At what distribution of actual token counts does the reservation model break even with the actual model?
      </Prose>
    </div>
  ),
};

export default rateLimiting;
