import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";

const observabilityLLM = {
  title: "Observability & LLM Monitoring",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        Classical application observability rests on three assumptions that hold well enough for typical web services. Requests are roughly similar in cost: a GET to a REST endpoint takes a few milliseconds and a predictable amount of CPU. Errors are obvious: a 500 status code or an exception in the logs tells you something broke. And the three pillars — logs, metrics, traces — cover the failure modes that actually matter in production. A service either responds or it doesn't; if it responds, the response is structurally correct or it isn't; if something goes wrong, the stack trace tells you where.
      </Prose>
      <Prose>
        LLM observability breaks all three assumptions at once. A single production endpoint might serve requests ranging from a 50-token classifier call to a 10,000-token document rewrite — a 200× variance in cost from the same API surface. Latency swings from 200ms to 45 seconds on the same model, same user, same nominal request class. And the model can fail silently: it produces fluent, grammatically correct, confidently stated output that is subtly wrong, quietly evasive, or semantically drifted from what it produced last week. No 5xx. No exception. No stack trace. Just an answer that is a little worse than it used to be, distributed across millions of requests, invisible to standard monitoring. "Traces" of a 10,000-token generation don't fit the distributed-tracing shape either — the interesting span is a single contiguous decode that runs for seconds inside one process, not a tree of microservice calls. This topic is about the instrumentation that a mature LLM product actually needs.
      </Prose>

      <H2>Metrics that matter</H2>

      <Prose>
        Standard infrastructure metrics — CPU utilization, memory, request count, error rate — are necessary but not sufficient. An LLM system needs a layer of model-specific metrics that expose what is happening inside the generation process. The table below is a working baseline for any production inference deployment.
      </Prose>

      <CodeBlock>
{`metric                           unit       granularity       why it matters
requests_per_second              rps        per-model         throughput & capacity planning
ttft (time to first token)       ms         p50 / p95 / p99   user-felt latency
itl (inter-token latency)        ms         p50 / p95         smoothness of streaming output
tokens_per_second_decode         tok/s      per-instance      GPU utilization & health
input_tokens / output_tokens     count      per-request       cost attribution
kv_cache_utilization             %          per-instance      admission-control signal
cache_hit_rate_prefix            %          per-router        routing quality
rejected_input                   rate       per-policy        safety layer activity
filtered_output                  rate       per-category      output filter activity
cost_per_request                 $          per-request       billing & product economics`}
      </CodeBlock>

      <Prose>
        Several of these are not obvious at first glance. <Code>kv_cache_utilization</Code> is an admission-control signal: when the KV cache fills, the serving engine must evict earlier context, which degrades quality silently or forces request rejection — monitoring it lets you shed load before quality degrades rather than after. <Code>tokens_per_second_decode</Code> is a direct proxy for GPU health; a sudden drop in decode throughput, with no change in request volume, usually means a thermal throttle, a driver issue, or a memory bandwidth problem on the instance. <Code>cache_hit_rate_prefix</Code> measures how effectively your router is co-locating requests with shared prefixes onto the same instances, which is the primary knob for reducing TTFT and GPU cost on chatbot workloads with long system prompts.
      </Prose>

      <H2>TTFT and ITL — user-felt latency</H2>

      <Prose>
        End-to-end latency is the wrong metric to watch for streaming LLM systems. It is the right metric for batch jobs, for evaluations, for billing — but not for measuring what users actually experience. What users feel is two distinct things: how quickly did the first token appear, and how smoothly did subsequent tokens stream. These are <Code>TTFT</Code> (time to first token) and <Code>ITL</Code> (inter-token latency), and they have different causes, different remediation paths, and different SLO thresholds.
      </Prose>
      <Prose>
        TTFT is dominated by prefill time — the forward pass over the full input context before any generation begins. A 10,000-token prompt takes roughly 20× longer to prefill than a 500-token prompt on the same hardware, which is why long-context requests feel sluggish even before they start generating. The most effective levers for TTFT are prefix caching (if the system prompt is shared across requests, cache its KV state), chunked prefill (interleave prefill and decode so the GPU doesn't block generation entirely during a long prefill), and request prioritization (queue shorter prefills ahead of longer ones for latency-sensitive traffic). ITL, by contrast, is dominated by decode throughput — tokens per second per instance. It degrades under batch contention: when the serving engine batches too many concurrent decodes together, each individual request sees higher ITL even though total throughput improves. The tradeoff between throughput and per-request ITL is a product decision, not just an infra one.
      </Prose>
      <Prose>
        In practice: a 10-second response that starts flowing in 400ms feels fast to most users. A 3-second response that waits 2 seconds before the first token appears feels slow, even though its end-to-end latency is lower. Production dashboards should track TTFT p99 as the primary latency SLO. End-to-end latency is a secondary metric useful for cost and capacity modeling, but paging on end-to-end latency alone will cause you to miss the user experience degradations that actually generate complaints.
      </Prose>

      <H2>Cost attribution</H2>

      <Prose>
        Token costs are not uniform, and they are not predictable from request metadata alone — a "summarize this document" call might spend 2,000 input tokens or 20,000, and the cost difference is 10×. Every request's actual cost must be measured and attributed: to a user, a team, a feature flag, an API key, or whatever granularity your billing and product analytics need. Cost attribution done at the application layer, after the fact, is fragile — models get upgraded, pricing changes, caching changes the effective cost, and the attribution logic drifts from reality. The right pattern is to emit a cost metric at request completion, computed from the actual token counts returned by the model API.
      </Prose>

      <CodeBlock language="python">
{`async def emit_cost_metrics(request, response, timing):
    price = get_price(request.model)
    cost = (
        response.input_tokens * price.input_per_mtok / 1e6 +
        response.output_tokens * price.output_per_mtok / 1e6
    )
    await metrics.emit(
        "llm.request.cost",
        value=cost,
        tags={
            "model": request.model,
            "user_id": request.user_id,
            "team_id": request.team_id,
            "feature": request.feature_flag,
            "cache_hit": str(response.cache_hit),
        },
    )`}
      </CodeBlock>

      <Prose>
        The <Code>cache_hit</Code> tag matters more than it looks. Prompt caching (where supported) can reduce effective input token cost by 60–90% for requests with long shared prefixes. If your attribution logic counts input tokens at full price regardless of cache status, your per-feature cost estimates will be wrong — sometimes by an order of magnitude for heavy chatbot workloads with long system prompts. Billing, per-customer rate management, runaway detection, and product analytics all depend on this being right. An LLM product that cannot answer "which feature is responsible for 40% of our inference bill this month" cannot make rational product prioritization decisions.
      </Prose>

      <H2>Quality drift — the thing that can't be graphed easily</H2>

      <Prose>
        HTTP metrics go red when something breaks. Quality drift goes nowhere. Your model is slightly worse at code generation today than it was last week. Your refusal rate shifted by 3 percentage points. Your answers got longer and vaguer. Your citation accuracy dropped. Infrastructure metrics look fine — latency is normal, error rates are flat, GPU utilization is healthy. The model's outputs changed, but nothing in your observability stack noticed. This is the failure mode that distinguishes toy LLM deployments from production-grade ones: the absence of any signal for the thing that matters most.
      </Prose>
      <Prose>
        Quality drift has multiple causes, and most of them are invisible in infrastructure metrics. A model upgrade — even a minor version bump — can shift tone, verbosity, and task-specific accuracy. A prompt-template change that seemed neutral in A/B testing might interact badly with a segment of real-world queries. A new safety classifier in the output filter might be suppressing responses it shouldn't. A shift in the input distribution — users asking different kinds of questions — changes what "good" means even if the model is identical. None of these trigger alerts on standard dashboards. Production mitigations operate at three levels: sampled eval, where a small fraction of production traffic is run through a quality scorer (human rater, LLM judge, or benchmark replay) and scores are tracked over time; canary deploys, where a small fraction of traffic is routed to the new model or configuration and scores are compared before any broader rollout; and shadow traffic, where old and new configurations run in parallel on the same requests and divergence is surfaced directly. All three are standard practice at mature LLM product teams. None of them are built into off-the-shelf monitoring stacks.
      </Prose>

      <H3>Distributed tracing adapted</H3>

      <Prose>
        Standard distributed traces, in the OpenTelemetry model, show a request's path through a tree of services — API gateway, auth service, database, cache, downstream APIs. For LLM systems, this shape is still useful but needs extension. A useful LLM trace includes spans for: input classification (routing tier, context-length estimation, safety pre-screen), router decision (which instance or model variant handles this request), prefill (the forward pass over the input — often the largest single span for long contexts), decode (the token-by-token generation phase, sometimes multiple spans for multi-turn agent loops), output filtering (safety classifier, PII redaction, format validation), and final response assembly. Each span carries token counts and latency alongside the standard timing data. This gives you the ability to answer questions that infrastructure-only traces cannot: why did this specific request take 8 seconds? (3.2s prefill, 4.1s decode, 0.7s output filter) — and was the slow output filter an anomaly or a pattern?
      </Prose>
      <Prose>
        OpenTelemetry's generative-AI semantic conventions, proposed in 2024 and stabilizing through 2025, standardize the span attributes for exactly this use case. The key attributes are <Code>gen_ai.request.model</Code>, <Code>gen_ai.usage.input_tokens</Code>, <Code>gen_ai.usage.output_tokens</Code>, and <Code>gen_ai.response.finish_reason</Code>. Instrumenting to these conventions means your traces are compatible with observability backends that understand LLM workloads without custom dashboards — Grafana, Honeycomb, Jaeger, and several purpose-built LLM observability platforms all have support for the gen-AI semantic conventions either shipped or in preview.
      </Prose>

      <H3>Logging prompts — the privacy problem</H3>

      <Prose>
        Logs are the easiest LLM observability tool to misuse. Full-prompt logging — capturing every input and output to a persistent log store — is valuable for debugging and eval dataset construction, but it is expensive in storage, slow to query at scale, and dangerous for PII. Users put sensitive information in LLM inputs: names, addresses, health conditions, financial data, API keys, passwords, internal business strategy. A prompt log is a concentrated store of exactly the data attackers want and regulators scrutinize most carefully. Most mature production stacks apply a tiered logging policy: request metadata is logged always (model, token counts, latency, cost, feature flags), prompt content is logged at a configured sample rate (commonly 1–5% of traffic), and full prompts are only retained with explicit user or enterprise opt-in. Redaction pipelines — running regex or a small classifier over logged prompts before storage to strip PII patterns — run before persistence, not after. Logging then redacting is strictly worse than redacting then logging, because the window between ingestion and redaction is an exposure window.
      </Prose>

      <Callout accent="gold">
        The cheapest way to fix a privacy incident is to not have the data in the first place. Sample, redact, and expire your prompt logs aggressively.
      </Callout>

      <H2>Alerting — what to page on</H2>

      <Prose>
        Alert fatigue on LLM systems is easy to induce. The metric surface is large, many metrics are naturally noisy, and the instinct to alert on everything produces an on-call rotation that ignores pages. The discipline is to alert only on conditions where a human intervention in the next 15 minutes would change the outcome. That filters the candidate list aggressively. Alert on: error rate spikes from inference workers (5xx responses indicate the serving layer is actively broken); TTFT p99 exceeding the committed SLO (users are experiencing latency that violates the product contract); queue depth sustained at non-zero for more than a few minutes (the system cannot clear requests as fast as they arrive, which means capacity shortage); safety-filter activation rate spiking above baseline (a sudden increase in blocked inputs or outputs is a signature of a jailbreak campaign or a classifier regression — both require human review); and daily cost exceeding a budget forecast threshold (runaway spend from a buggy caller or an unexpected traffic surge). Do not alert on individual slow requests — single-request latency outliers are noise. Do not alert on individual refusals — the safety layer is supposed to fire. Do not alert on raw GPU utilization dips — utilization varies normally with request mix. Each of these is a metric worth having on a dashboard; none of them justify waking someone up.
      </Prose>

      <H3>Eval in production</H3>

      <Prose>
        Beyond real-time metrics, continuous eval is the mechanism for catching quality drift before users notice it. A fixed set of eval prompts — covering the task distributions the product handles — runs against the production endpoint on a scheduled cadence, typically daily. Scores are tracked over time, and regression on eval scores is a release blocker. This is distinct from the sampled quality monitoring described earlier: sampled evals measure the live distribution of production queries (including the long tail); scheduled evals measure a fixed benchmark with known answers, which makes regression detection reliable even when the query distribution shifts. The two complement each other. Most large labs have separate eval infrastructure from production observability — eval runs are batched, offline, and not latency-sensitive, while production observability is real-time. Integrated stacks that consolidate both — LangSmith, Braintrust, Weave, Arize Phoenix — have been gaining adoption since 2024, particularly among teams that don't have the engineering capacity to maintain two separate systems. The tradeoff is standardization versus flexibility: integrated platforms cover the common cases well and the unusual ones less so.
      </Prose>

      <Prose>
        Good observability is what separates an LLM product that can be debugged, improved, and run reliably at scale from one that can only be hoped at. The metrics, tracing, logging, alerting, and eval discipline described here are not premature optimization — they are the minimum viable instrumentation for a system whose failure modes are genuinely novel. The next topic — streaming and SSE — covers the transport layer that most of this observability has to instrument: a persistent connection that emits tokens incrementally, and where every assumption from the HTTP request-response model needs to be revisited.
      </Prose>
    </div>
  ),
};

export default observabilityLLM;
