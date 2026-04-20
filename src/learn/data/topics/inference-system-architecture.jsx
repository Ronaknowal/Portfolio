import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { StepTrace, TokenStream } from "../../components/viz";

const inferenceSystemArchitecture = {
  title: "Inference System Architecture (End-to-End)",
  readTime: "16 min",
  content: () => (
    <div>
      <Prose>
        A production LLM endpoint looks nothing like "run vLLM on a GPU." The inference worker that everyone talks about — the piece that holds the weights, manages the KV cache, runs the kernels — is one of eight or nine components in the hot path of a request, and rarely the one that decides whether the product works. By the time a prompt reaches the GPU it has traversed a load balancer, an API gateway, a rate limiter, a model router, and a safety classifier; on the way back it passes through an output filter, a streaming relay, a metrics pipeline, and a billing sink. None of that is decoration. Every piece is there because some failure mode — a cost spike, a tail-latency cliff, a safety incident, an outage during a traffic burst — forced it to be.
      </Prose>

      <Prose>
        Every frontier lab and every serious LLM product has converged on roughly the same architecture, because the same forces push them there. Requests are wildly heterogeneous — a hundred-token classification and a hundred-thousand-token agent trace share an endpoint. Latency tails measure in minutes, not milliseconds. A single misbehaving tenant can monopolize a GPU pool. Safety failures are newspaper headlines. The cost of serving a token varies by two orders of magnitude across models, and most of the COGS optimization in a mature stack lives not in the inference engine but in the decision of which engine to route to. This topic walks the architecture top to bottom — the pieces, the interfaces between them, and what each one does.
      </Prose>

      <H2>The shape of a production stack</H2>

      <Prose>
        A request's journey through a mature LLM system passes about eight components before it reaches a GPU, and nearly as many on the way back. Most of these layers exist in ordinary web services too — gateways, load balancers, caches, observability — but the pressures of LLM traffic reshape each of them enough that off-the-shelf components rarely survive contact with production without modification.
      </Prose>

      <StepTrace
        label="a request's path — client to token, top to bottom"
        steps={[
          { label: "1. client / api gateway", render: () => (
            <TokenStream tokens={["HTTPS", " →", " auth check", " →", " rate limit", " →", " queue"]} />
          ) },
          { label: "2. router", render: () => (
            <TokenStream tokens={["pick model", " →", " pick region", " →", " pick instance pool"]} />
          ) },
          { label: "3. load balancer", render: () => (
            <TokenStream tokens={["pick instance (KV-cache-aware)", " →", " forward"]} />
          ) },
          { label: "4. safety layer (input)", render: () => (
            <TokenStream tokens={["classifier filter", " →", " jailbreak detector", " →", " pass"]} />
          ) },
          { label: "5. inference engine", render: () => (
            <TokenStream tokens={["prefix cache lookup", " →", " prefill", " →", " decode (continuous batched)"]} />
          ) },
          { label: "6. safety layer (output)", render: () => (
            <TokenStream tokens={["content filter on streaming output"]} />
          ) },
          { label: "7. streaming response", render: () => (
            <TokenStream tokens={["SSE back to client", " →", " metrics"]} />
          ) },
          { label: "8. observability sink", render: () => (
            <TokenStream tokens={["log tokens", " →", " cost attribution", " →", " trace"]} />
          ) },
        ]}
      />

      <Prose>
        The diagram is linear; real traffic is not. Most of these components maintain asynchronous side-channels: the gateway writes to a quota ledger, the router consults a live view of pool health, the safety layers invoke their own classifier models, the observability sink is fed from every node. The hot path is what the user waits on; the cold paths are what keep the hot path working. A mature inference stack has more cold-path code than hot-path code, and the quality of the cold paths is what distinguishes a system that degrades gracefully from one that falls over on its first bad Tuesday.
      </Prose>

      <H2>The API gateway — the front door</H2>

      <Prose>
        The API gateway owns a familiar list of responsibilities: authentication, request validation, schema enforcement, quota tracking, per-user and per-team rate limits, admission control. Nothing on that list is LLM-specific, and for a while it was tempting to reuse whatever gateway the rest of the company already had. That mostly does not work. LLM traffic violates three assumptions those gateways were built under. Tail latency is minutes, not milliseconds — a long-context generation legitimately takes five minutes, and a gateway that times out at thirty seconds will kill healthy work. Response bodies stream for the entire duration, so connection-per-request accounting and buffered logging both break. And requests are expensive enough that admission control — the decision to queue, reject, or accept — has to happen at the gateway, not deep inside the inference worker, because by the time an overloaded worker realizes it cannot handle the request, the cost of the round trip has already been paid.
      </Prose>

      <Prose>
        The gateway that ships in mature stacks is a thin custom service, often built on envoy or a similar proxy, with LLM-aware middleware bolted on: token-based rate limiting rather than request-based, queue-depth-aware admission control, long-lived streaming support, and per-tenant quotas that track input tokens, output tokens, and spent dollars separately. It is a boring component by frontier-AI standards, and the first thing to fail when anything goes wrong.
      </Prose>

      <H2>The router — one of the highest-leverage pieces</H2>

      <Prose>
        For any given request the router answers three questions: which model should serve it, which datacenter should it go to, and which pool inside that datacenter should take it. A modern deployment might have Sonnet, Haiku, a reasoning model, a math model, a vision model, and a code model all sitting behind the same public endpoint. Even when the user specified a model by name, the router often has latitude to route to a cheaper near-equivalent when load is high or when a task classifier on the prompt suggests the smaller model is sufficient. Within a chosen model there are usually multiple pools — different hardware generations, different regions, dedicated enterprise capacity.
      </Prose>

      <Prose>
        The features the router uses are a mix of request metadata and cheap inference on the prompt. Prompt length is the first-order signal — a two-million-token prefill cannot go to a pool that does not support long context. User tier, feature flags, and geographic origin determine the shortlist of eligible pools. A small task classifier, often a distilled 1B transformer run on CPU, produces a vector — is this code, math, a short chat turn, agentic — that the routing logic consumes as one more input. Done well, thirty to sixty percent of traffic ends up on smaller or cheaper models with no user-visible quality drop, because most of what a frontier lab serves did not need a frontier model in the first place. This is where most of the COGS optimization in a mature stack lives, and it is almost invisible because the outputs look identical to what the named model would have produced.
      </Prose>

      <H2>Load balancing — KV-cache-aware</H2>

      <Prose>
        Inside a model pool, classic round-robin is wrong in a way that is easy to miss until you measure it. An instance that already has a user's conversation prefix in its KV cache should almost always receive their next turn, because a prefix-cache hit skips prefill entirely — and prefill is where most of the wall-clock time on short-to-medium turns gets spent. Round-robin sends the next turn to whichever instance happens to be next, which nearly always means a cold cache and a full prefill over a prompt that some other instance could have served in a fraction of the time.
      </Prose>

      <Prose>
        The minimum correct behavior is sticky routing by session: the same conversation goes to the same instance for its entire lifetime. Better stacks consistent-hash on the leading tokens of the prompt itself, so that any request sharing a prefix — not just the same user's turns, but any request that happens to reuse the same system prompt or tool preamble — lands on the instance most likely to have it cached. The ring also has to account for instance health: if the primary is overloaded, the router should fall back to a replica, accepting a cold cache rather than piling into a hot-spotted node.
      </Prose>

      <CodeBlock language="python">
{`def select_instance(request, pool, hash_ring):
    """Route request to the instance most likely to have its prefix cached."""
    # Hash the first N tokens of the prompt — stable across same-prefix requests.
    prefix_hash = hash(tuple(request.prompt_tokens[:256]))

    # Consistent-hash to a primary instance
    primary = hash_ring.get_instance(prefix_hash)

    # If primary is overloaded (latency > threshold), fall back to replicas
    if primary.p50_latency_ms > 2000:
        replicas = hash_ring.replicas_for(prefix_hash, k=3)
        return min(replicas, key=lambda i: i.queue_depth)
    return primary`}
      </CodeBlock>

      <Prose>
        The choice of prefix length — 256 tokens in the snippet — is a tuning parameter. Too short and unrelated requests collide. Too long and the hash is too specific to reuse across sibling requests. A well-tuned cache-aware load balancer can double the effective QPS of a pool serving chat traffic with long system prompts, purely by keeping hot caches hot.
      </Prose>

      <H2>The inference worker</H2>

      <Prose>
        The previous section covered this layer in depth — continuous batching, PagedAttention, speculative decoding, quantization. At the system level the worker is a black box with a small interface: <Code>/v1/generate</Code> streams tokens, <Code>/metrics</Code> exposes Prometheus counters, <Code>/health</Code> reports readiness. The architectural question is not how the worker works internally but how many of them there should be — which GPU generation, serving which models, in which regions. Workers are deliberately fungible within a pool: any worker in the Sonnet pool can serve any Sonnet request, because the routing decisions above have already narrowed the set. That fungibility is what makes cache-aware routing an optimization rather than a hard constraint — fallback is always possible, just more expensive.
      </Prose>

      <H2>Caching — three layers, not one</H2>

      <Prose>
        "The cache" is not a single component; it is three distinct caches operating at different layers, with different hit rates, correctness properties, and failure modes. Confusing them is a common architectural mistake.
      </Prose>

      <Prose>
        The <em>exact-match cache</em> sits in front of the inference layer, keyed by a hash of <Code>(model, prompt, parameters)</Code>. A hit returns the cached response without touching a GPU. It works only for idempotent requests, which sounds narrow but covers a surprising fraction of enterprise traffic: classification endpoints, batch extraction, cron summarization, evaluation harnesses. Hit rates are usually a few percent; the cost to serve a hit is effectively zero. The <em>prefix or KV cache</em> is the one the previous section covered — per-layer keys and values reused across requests that share a leading token sequence. On agentic workloads with long shared system prompts, thirty to sixty percent hit rates are normal, saving prefill compute. The <em>semantic cache</em> is the controversial one. It embeds incoming prompts, does a nearest-neighbor lookup, and returns a cached answer if similarity exceeds a threshold. It can serve paraphrased questions without a model call, but can also silently return the wrong answer when two prompts are close in embedding space but semantically different. The threshold is nearly impossible to tune for general traffic; most labs restrict semantic caching to narrow, safety-reviewed contexts or avoid it entirely.
      </Prose>

      <Prose>
        The three caches compose rather than substitute. A mature pipeline checks exact-match first, falls through to the inference worker which consults the prefix cache, and optionally consults a semantic cache at the gateway for specific routed use cases. The dedicated caching topic goes into all three in depth.
      </Prose>

      <H3>Safety layers — input and output</H3>

      <Prose>
        On both sides of the inference call sit safety layers: classifiers that block clearly malicious input, and streaming output filters that monitor generated tokens as they are emitted. The input side is easier. Small dedicated classifier models — Llama Guard, Aegis, OpenAI's moderation endpoint — score the prompt across a taxonomy of harm categories and disallow patterns, with a distinct category of prompt-injection detectors looking for attempts to override system instructions. A well-tuned input filter rejects a fraction of a percent of traffic in tens of milliseconds, cheap enough to sit in the hot path.
      </Prose>

      <Prose>
        Output filtering is harder. The model emits tokens one at a time; the filter has to decide mid-stream whether a partial response is heading somewhere disallowed. The three available actions are all bad in different ways. Killing a response mid-stream leaves the user staring at a truncated sentence and a vague refusal. Continuing with a pre-written fallback is more graceful but requires a second answer that may not fit the prompt. Letting the response through and flagging it for offline review catches the issue statistically but does nothing for the user who received it. Most deployments run a mix: hard kill on unambiguous violations, silent flag-and-review on borderline cases, canned fallbacks for specific prompt shapes. The tuning never finishes.
      </Prose>

      <H2>Observability — what you can't see you can't debug</H2>

      <Prose>
        LLM observability is harder than classical web observability, and teams who treat it as "just add Datadog" discover the gap six months in when they cannot explain why their quality numbers are drifting. Requests are wildly heterogeneous — the same endpoint serves a hundred-token classification and a hundred-thousand-token agent trace, and any metric aggregated across that spread without careful bucketing is noise. Logging is expensive in three ways at once: prompts and responses are large, they often contain PII, and the generated-text volume across a production fleet can exceed the web-traffic logs it is supposed to live alongside. And the failure modes that matter most — quality regressions, alignment drift, a new jailbreak that slipped past the classifier — are invisible to HTTP metrics. The 200 status code tells you nothing about whether the answer was right.
      </Prose>

      <Prose>
        Mature stacks instrument along axes classical frameworks do not have first-class support for: tokens in and out per request (for cost attribution), per-model tail latencies bucketed by prompt length, per-user and per-team spend, prefix-cache hit rate by pool, prompt-injection detection rate, fraction of responses that tripped the output filter, and — the important one — quality eval scores computed on a sampled subset of real traffic against a held-out rubric or judge model. The dedicated observability topic goes into the specific metrics, sampling strategies, and privacy-preserving logging techniques; for now the point is that the gap between "serving is green" and "serving is actually working" is where almost all the interesting production incidents live.
      </Prose>

      <H3>Streaming — the default transport</H3>

      <Prose>
        Almost every real LLM endpoint streams. The standard wire format is Server-Sent Events; some stacks use WebSockets or gRPC bidirectional streams, but SSE dominates the public-facing API surface. Streaming is not a performance optimization in the usual sense — it does not change the total compute — but it transforms what the user feels. Time-to-first-token becomes the metric that determines whether the product feels fast; end-to-end completion time fades into the background unless it crosses an absolute threshold. A response that finishes in twenty seconds but shows its first token in two hundred milliseconds feels vastly faster than one that takes eight and blocks until done.
      </Prose>

      <Prose>
        The operational cost of streaming lands on the layers above the inference worker. Long-lived connections break the assumptions of most load balancers, CDNs, and logging middleware. Off-the-shelf gateways configured for request-response traffic will buffer streaming output, kill connections on idle timers, or rotate pools underneath active streams in ways that corrupt the output. Every layer has to be explicitly audited for streaming compatibility, and retrofitting streaming into an existing gateway stack is more work than most teams budget for.
      </Prose>

      <H2>Autoscaling — the unique problem</H2>

      <Prose>
        Autoscaling an LLM fleet is not a variation on classical autoscaling; it is a different problem that happens to share vocabulary. CPU utilization is meaningless on a machine whose work happens on a GPU. GPU compute utilization is worse than useless — an inference worker can be fully loaded at forty percent compute because it is memory-bandwidth-bound, and the monitor registers headroom that does not exist. GPU memory is closer, but a worker at ninety percent memory is already rejecting new requests because the KV cache has no room to admit them. The right signal fired ten minutes ago.
      </Prose>

      <Prose>
        The signals that work are KV cache pressure (what fraction of the paged pool is allocated), queue depth (how many requests are waiting to enter prefill), and TTFT trend (how fast time-to-first-token is growing as load climbs). All three become useful well before the system is in distress, which is necessary because the response to a spike is slow: spinning up a new GPU worker for a 70B model — acquiring the hardware, loading the weights, warming CUDA graphs, registering with the router — rarely finishes in under a few minutes. By the time any signal is unambiguous, the system has been failing longer than it takes to react. The only workable posture is forecasting under constraints, not reactive scaling. Production autoscalers look like small forecasting systems — historical traffic patterns, current trend, per-customer scheduled load, safety margins tuned against past incidents. The dedicated autoscaling topic goes into the specific predictors.
      </Prose>

      <Callout accent="gold">
        LLM autoscaling is forecasting under constraints, not reactive scaling. By the time CPU graphs spike, you're already dropping requests.
      </Callout>

      <H3>The hyperscaler vs in-house split</H3>

      <Prose>
        The architecture is not specific to any one deployment model. Most LLM products today are some blend of three postures: calls to commercial APIs, self-hosted open-weight models on managed GPU platforms (Together, Fireworks, Modal, SageMaker), and, at enterprises with compliance constraints, fully in-house deployments. What changes across the three is only where the trust boundaries sit. A pure API consumer does not own the inference workers but still needs its own gateway, router across providers, caching layer, and observability — the provider's dashboard is not a substitute for understanding how requests from one's own product behave. The in-house operator owns more of the stack but still builds the same routing, caching, and autoscaling machinery; the inference-engine layer just happens to be code they wrote. The architecture generalizes because the pressures generalize.
      </Prose>

      <H2>Where this section goes next</H2>

      <Prose>
        Each of the following topics zooms into one component of this diagram at the depth it deserves. Multi-model routing and the economics of task classification. Autoscaling under forecasting pressure. Disaggregated serving, where prefill and decode live on separate pools because their bottlenecks differ. The caching trio. Rate limiting in a world where one request can cost as much as a thousand. Safety layers on both sides of the call. Observability for heterogeneous traffic. Streaming transports. Cost optimization across models and regions. Edge deployment. Multi-region failover. The goal of this flagship is to keep those topics legible as parts of a coherent system rather than a loose bag of techniques.
      </Prose>
    </div>
  ),
};

export default inferenceSystemArchitecture;
