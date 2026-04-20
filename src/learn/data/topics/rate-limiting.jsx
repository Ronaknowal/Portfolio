import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { StepTrace, TokenStream } from "../../components/viz";

const rateLimiting = {
  title: "Rate Limiting, Quota Management & Fairness",
  readTime: "10 min",
  content: () => (
    <div>
      <Prose>
        Rate limiting on LLM APIs is harder than on conventional APIs because the "amount of work" per request is not fixed. A 100-token prompt with a 100-token response consumes a small fraction of the compute that a 100k-token prompt with a 10k-token response consumes. The ratio between those two cases is not 10× or 100×; it is closer to 1,000× when you factor in the quadratic attention cost of the longer context. Limiting by request count treats both requests identically. It misses almost everything that matters. Production LLM APIs converge on limiting by tokens — input tokens, output tokens, or both — because that is the only unit that tracks the actual resource consumption with reasonable fidelity.
      </Prose>

      <Prose>
        The consequence is that the rate-limiting layer of an LLM API has to be aware of token counts at the point of enforcement, which means it runs after tokenization but before inference is queued. That placement imposes latency on every request and requires the infrastructure to maintain per-tenant counters with high write throughput. Neither of those is free, but both are cheaper than the alternative: letting expensive requests through unchecked and discovering the problem on the billing statement or, worse, in a cascade failure when the GPU fleet runs out of KV cache memory.
      </Prose>

      <H2>What to limit</H2>

      <Prose>
        Every production LLM API exposes some combination of five knobs. <Code>RPM</Code> (requests per minute) is the classic throttle, useful primarily against pure abuse — a client hammering the endpoint with rapid-fire small requests. <Code>ITPM</Code> and <Code>OTPM</Code> (input tokens per minute, output tokens per minute) are the resource-aware throttles; they are separate because input and output tokens have different compute profiles. Input tokens run a single forward pass through the prefill phase; output tokens run one forward pass each through the decode phase and also hold a KV cache slot for the duration of the sequence. Output tokens are more expensive per token and compete for a scarcer resource, which is why output budgets are often set at roughly 20–25% of the corresponding input budget. <Code>TPM</Code> (total tokens per minute) collapses the distinction into one number. <Code>Concurrency</Code> (simultaneous in-flight requests) enforces a hard cap on KV cache occupancy regardless of token counts — it is the backstop when the other limits would otherwise allow a small number of very long sequences to monopolize the fleet.
      </Prose>

      <CodeBlock>
{`tier           rpm     itpm        otpm        tpm         concurrency
free           20      40,000      8,000       —           5
pay-as-you-go  3,500   2,000,000   400,000     —           40
tier-1         5,000   4,000,000   800,000     —           100
tier-2         10,000  8,000,000   1,600,000   —           200
enterprise     custom  custom      custom      custom      custom`}
      </CodeBlock>

      <Prose>
        Multiple limits apply simultaneously. The binding constraint is whichever limit is hit first. A client running one massive 800k-token request will hit the concurrency limit before the token budget moves at all. A client running a thousand 2k-token requests per minute will hit the RPM limit before the token budget is a concern. The limits are designed to be independently tunable so that the enforcement layer can respond to different attack surfaces without widening every window at once.
      </Prose>

      <H2>Token bucket — the classic algorithm</H2>

      <Prose>
        The token bucket algorithm is the standard implementation underneath most rate limiters, LLM or otherwise. A bucket has capacity <Code>N</Code> and refills at rate <Code>R</Code> tokens per second, continuously, up to the capacity ceiling. Each request consumes tokens proportional to its cost. If the bucket has enough tokens, the request proceeds and the tokens are deducted. If not, the request is rejected or queued. The bucket accumulates credit during quiet periods, which allows short bursts above the sustained rate up to the capacity limit. That burst tolerance is intentional — a client making one large request every few minutes should not be penalized for not spreading the load across many tiny requests.
      </Prose>

      <Prose>
        In an LLM context the "tokens" in the token bucket are LLM tokens, not the HTTP authorization tokens the algorithm was named after. The capacity is denominated in LLM tokens, the refill rate is in LLM tokens per second, and each request's cost is its actual token consumption. A free-tier bucket might hold 40,000 input tokens and refill at roughly 667 per second, which works out to the 40,000 ITPM limit in the table above.
      </Prose>

      <CodeBlock language="python">
{`import time

class TokenBucket:
    """Rate limit by LLM token budget — capacity is in LLM tokens, not HTTP tokens."""
    def __init__(self, capacity_tokens, refill_tokens_per_sec):
        self.capacity = capacity_tokens
        self.refill_rate = refill_tokens_per_sec
        self.tokens = capacity_tokens
        self.last_refill = time.time()

    def try_consume(self, cost_tokens):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        if self.tokens >= cost_tokens:
            self.tokens -= cost_tokens
            return True
        return False

# Estimate cost BEFORE running inference — reject or queue up front rather than
# after a partial generation, which would bill the user for wasted work.`}
      </CodeBlock>

      <H2>The estimation problem</H2>

      <Prose>
        There is a subtlety specific to LLM rate limiting that has no real equivalent in conventional API throttling: output token count is unknown before generation begins. The model decides how many tokens to produce as it generates them, up to the <Code>max_tokens</Code> ceiling the client supplies. For input tokens this is not a problem — you can count them exactly during tokenization, before the request enters the queue. For output tokens you are enforcing a budget against a number you do not yet have.
      </Prose>

      <Prose>
        The practical resolution is to reserve <Code>max_tokens</Code> from the output budget at the point of admission, treat that reservation as consumed, and refund the unused portion when generation finishes. If a client specifies <Code>max_tokens=4096</Code> and the model stops at 300 tokens, the refund is 3,796 tokens. From the rate limiter's perspective this is conservative: it rejects requests that would, in practice, fit within the remaining budget, because the reservation is the worst-case bound rather than the expected outcome. Clients with small actual output counts but large <Code>max_tokens</Code> values see rejections that feel spurious. This is the correct tradeoff — a mid-stream cancellation after partial generation is more expensive, more confusing, and still consumes the tokens that were generated before the cutoff. Front-loading the enforcement prevents all of that at the cost of some unnecessary rejections.
      </Prose>

      <H3>Sliding window vs fixed window</H3>

      <Prose>
        The token bucket gives you burst tolerance, but the enforcement window — the time interval over which usage is measured — matters independently. Fixed windows reset on a schedule: every minute on the minute, every hour on the hour. A client can use 100% of the per-minute budget in the last second of one window and 100% in the first second of the next, achieving an effective instantaneous rate that is double the stated limit. This is not a theoretical concern; it is a well-known attack against any fixed-window rate limiter, and it requires only a single burst at the right moment.
      </Prose>

      <Prose>
        Sliding windows fix this by tracking usage over the most recent <Code>N</Code> seconds rather than since the last reset boundary. At any instant, the enforced limit is usage in the preceding 60 seconds, regardless of where the clock falls relative to a minute boundary. There is no exploitable seam. The cost is higher storage and compute — instead of a single counter per tenant that resets periodically, you need to store and query a time-series of usage events, or maintain an approximation using techniques like sliding window logs or the leaky bucket variant. Modern APIs treat sliding windows as table stakes. A fixed-window rate limiter is a known vulnerability, not a simplification.
      </Prose>

      <H2>Priority and fairness</H2>

      <Prose>
        Per-tenant rate limits prevent any single tenant from consuming more than their stated budget. They do not, by themselves, ensure that every tenant gets their fair share of available capacity. The distinction matters when the fleet is under load. If tenant A is consistently filling their budget with requests that take a long time to complete — long sequences, high concurrency — their requests may occupy GPU capacity that prevents tenant B's lower-cost requests from being scheduled promptly, even though both are operating within their individual limits. Tenant B is not exceeding any limit; they are being starved by a scheduling policy that processes queued requests in order of arrival or priority class without regard for per-tenant fairness.
      </Prose>

      <Prose>
        The solution is fair queueing at the scheduler level. In weighted fair queueing, each tenant is assigned a share of forward progress proportional to their tier or paid allocation. The scheduler selects the next request to process by picking the tenant whose cumulative service has fallen furthest behind their expected share, then choosing from that tenant's queue. No tenant can consume more than their proportional share of GPU capacity over any significant time window, regardless of the mix of request sizes. Burst capacity is still available when other tenants are idle — fair queueing only enforces proportionality under contention.
      </Prose>

      <StepTrace
        label="weighted fair queueing across tenants"
        steps={[
          { label: "incoming — 3 tenants with different shares", render: () => (
            <TokenStream tokens={[
              { label: "tenant A (50%)", color: "#e2b55a" },
              { label: "tenant B (30%)", color: "#4ade80" },
              { label: "tenant C (20%)", color: "#c084fc" },
            ]} />
          ) },
          { label: "scheduler picks proportional to share", render: () => (
            <TokenStream tokens={[
              { label: "A A B A B C A B A C", color: "#888" },
            ]} />
          ) },
          { label: "result — each tenant gets guaranteed slice", render: () => (
            <TokenStream tokens={[
              { label: "A served 5/10 steps", color: "#e2b55a" },
              { label: "B served 3/10", color: "#4ade80" },
              { label: "C served 2/10", color: "#c084fc" },
            ]} />
          ) },
        ]}
      />

      <H2>What happens when a limit is hit</H2>

      <Prose>
        When a request arrives and the relevant bucket or window is exhausted, the API has three options, each with distinct tradeoffs. The first is a <Code>429 Too Many Requests</Code> response — the HTTP-standard signal that the client should back off and retry. This is the simplest option and the most visible to the user. Most SDKs implement exponential backoff with jitter on 429, so well-behaved clients recover automatically without intervention. The downside is that the client experiences a hard failure and must either retry or propagate the error. For latency-sensitive applications, even a brief backoff window is disruptive.
      </Prose>

      <Prose>
        The second option is to queue the request and hold it until capacity is available, up to a configured maximum wait time. From the client's perspective the request eventually succeeds, with a longer-than-usual time-to-first-token. The 429 is hidden. This is useful for absorbing short traffic bursts that exceed the sustained rate limit but would fit within the burst capacity if the API could delay them slightly. The risk is that the queue can grow without bound if the sustained arrival rate genuinely exceeds capacity, at which point the hidden rejections re-emerge as timeouts rather than 429s — which is harder for clients to distinguish from infrastructure failures.
      </Prose>

      <Prose>
        The third option is to fall back to a smaller or less capable model. The model router — covered in the previous topic — is the usual home for this logic, but the rate limiter can trigger it: if the request would exhaust the budget for the target model but would fit within the budget for a smaller model, route to the smaller model and annotate the response accordingly. This turns a hard failure into a graceful degradation. Latency may actually improve, since smaller models generate tokens faster. The trade-off is that the client asked for one model and got another, which is only acceptable when the caller has signaled willingness to accept degraded quality in exchange for availability.
      </Prose>

      <H3>Quota vs rate limit</H3>

      <Prose>
        Rate limits and quotas are related controls that protect against different failure modes. A rate limit is a throughput constraint: at most <Code>N</Code> tokens per minute. It protects the infrastructure from instantaneous overload — a client cannot consume the fleet's capacity in a single burst. A quota is a cumulative constraint: at most <Code>M</Code> tokens per month. It protects the budget from runaway usage that is individually slow enough to stay within rate limits but adds up to unexpected expenditure over time. A script that runs one inference per minute, every minute, for a month will never trigger a rate limit. At a large enough model it will trigger a quota.
      </Prose>

      <Prose>
        Quotas are denominated in money, not just tokens, because the cost per token varies by model. A $50/month tier and a $500/month tier give the client different total token budgets depending on which models they use. Enforcement tracks dollar-equivalent consumption against the monthly ceiling, resets at the billing period boundary, and sends a warning notification before the ceiling is hit so the client can adjust before the hard stop. The combination of rate limit and quota closes the failure modes that each one alone leaves open.
      </Prose>

      <Callout accent="gold">
        Rate limits protect the infrastructure. Quotas protect the customer from themselves. Both matter; a system with only one has failure modes the other would catch.
      </Callout>

      <Prose>
        Rate limiting is the customer-facing edge of capacity management. The next topic turns to a different kind of protection: guardrails and safety layers that filter what goes into and comes out of the model itself.
      </Prose>
    </div>
  ),
};

export default rateLimiting;
