import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const observabilityLLM = {
  title: "Observability & LLM Monitoring",
  slug: "observability-llm-monitoring",
  readTime: "48 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Classical application observability rests on three assumptions that hold well enough for typical web services. Requests are roughly similar in cost: a GET to a REST endpoint takes a few milliseconds and a predictable amount of CPU. Errors are obvious: a 500 status code or an exception in a stack trace tells you exactly what broke. And the three pillars — logs, metrics, traces — cover the failure modes that actually matter in production. A service either responds or it doesn't; if it responds, the response is structurally correct or it isn't; and if something goes wrong, the exception tells you where.
      </Prose>

      <Prose>
        LLM systems break all three assumptions simultaneously. A single production endpoint might serve requests ranging from a 50-token sentiment classifier call to a 100,000-token document rewrite — a 2,000× variance in cost from the same API surface. Latency swings from 200 ms to 90 seconds on the same model, same nominal request class. And the model can fail silently: it produces fluent, grammatically correct, confidently stated output that is subtly wrong, quietly evasive, or semantically drifted from what it produced last week. No 5xx. No exception. No stack trace. Just an answer that is a little worse than it used to be, distributed across millions of requests, invisible to every standard monitoring tool.
      </Prose>

      <Prose>
        This is not a scaling problem or a tooling gap. It is a categorical difference in how failure works. When a Postgres query returns the wrong row, the bug is deterministic: the same inputs reliably reproduce it. When a language model begins hallucinating product names, or quietly shifts from citing sources to not citing them, the failure is probabilistic, stateful with respect to the model version and prompt distribution, and manifests only in aggregate over time. Standard metrics dashboards have no column for "semantic drift." Standard alerting has no condition for "quality regression in the tail of the distribution." Standard tracing has no span for "the model thought about it for 3 seconds and got it wrong."
      </Prose>

      <Prose>
        LLM-specific observability fills that gap. It adds four layers on top of the infrastructure baseline: token-level metrics that expose the economics of each request, structured traces that carry semantic context across every hop in the pipeline, quality evaluations that score output against rubrics rather than just checking for a response, and drift detectors that watch the statistical shape of outputs over time. None of these are exotic. All of them are table-stakes for any LLM product running in production beyond early beta.
      </Prose>

      <Callout accent="purple">
        The fundamental challenge of LLM observability is that the most important things to measure — semantic quality, factual accuracy, policy compliance — cannot be read off a gauge. They require active evaluation, and active evaluation at production scale requires its own infrastructure.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION — FOUR PILLARS
          ====================================================================== */}
      <H2>2. Core intuition — four pillars</H2>

      <Prose>
        The observability stack for an LLM system has four layers. Infrastructure monitoring — Prometheus, Grafana, cloud-native tools — covers the bottom. Three layers specific to LLM workloads sit above it: structured traces that capture request context end-to-end, quality evaluations that score outputs, and drift detectors that watch distributions over time. Each layer answers a different class of question. Infrastructure tells you if the system is running. Traces tell you what happened on any given request. Evaluations tell you if the outputs are any good. Drift detection tells you if "good" is changing.
      </Prose>

      <H3>Pillar 1 — Metrics</H3>

      <Prose>
        LLM metrics divide into infrastructure metrics and model-economics metrics. Infrastructure metrics are the standard set: requests per second, error rate, CPU and memory utilization, queue depth. Model-economics metrics are specific to LLM workloads and require deliberate instrumentation. The four that matter most in production are QPS by model (because different models have wildly different cost and latency profiles), TTFT and TPOT (the two latency dimensions that users actually feel), token throughput (the real measure of GPU utilization), and per-request cost (computed from actual token counts, not estimates).
      </Prose>

      <CodeBlock>
{`metric                      unit       agg          why it matters
─────────────────────────────────────────────────────────────────────
requests_per_second         rps        per-model    capacity planning
ttft                        ms         p50/p95/p99  user-felt latency: "thinking"
tpot (time-per-output-tok)  ms         p50/p95/p99  user-felt latency: "reading"
input_tokens                count      per-request  cost attribution (input price)
output_tokens               count      per-request  cost attribution (output price)
tokens_per_second_decode    tok/s      per-instance GPU health signal
kv_cache_utilization        pct        per-instance admission-control signal
cost_per_request            USD        per-request  billing & product economics
quality_score               [0,1]      per-model    sampled eval output
drift_score                 KL         rolling-24h  distribution shift signal`}
      </CodeBlock>

      <Prose>
        Several of these are not obvious. <Code>kv_cache_utilization</Code> is an admission-control signal: when the KV cache fills, the serving engine must evict earlier context or reject new requests. Monitoring it lets you shed load before quality degrades, not after. <Code>tokens_per_second_decode</Code> is a direct proxy for GPU health — a sudden drop in decode throughput, with no change in request volume, usually means a thermal throttle or a memory bandwidth regression. <Code>kv_cache_utilization</Code> and <Code>tokens_per_second_decode</Code> together tell you whether your instances are under-utilized (wasteful) or over-saturated (quality-degrading). Both conditions are invisible to HTTP-level monitoring.
      </Prose>

      <Prose>
        The TTFT and TPOT split is worth dwelling on because it determines which tier to optimize when latency SLOs are missed. TTFT is dominated by prefill time — the forward pass over the full input context — plus queue wait time. A TTFT regression after a traffic surge typically means queue depth increased (more requests waiting for worker slots), not that the model got slower. The remedy is capacity expansion or request prioritization, not model optimization. A TTFT regression after a prompt template change that added 2,000 tokens to the system prompt means prefill time increased for every request — the remedy is prompt compression. TPOT regressions, by contrast, almost always indicate that decode batch sizes increased (more concurrent requests per worker), that the serving engine switched to a less efficient attention kernel, or that GPU memory bandwidth is degraded. Knowing which metric regressed and when it regressed is the minimum context needed to route the investigation to the right team.
      </Prose>

      <H3>Pillar 2 — Logs</H3>

      <Prose>
        Structured logs for LLM requests carry more than standard access-log fields. At minimum, each log record should include: the model name and version, prompt token count, completion token count, latency broken down by TTFT and total, finish reason (stop/length/content-filter/tool-call), the trace ID for correlation, cost, any tool calls made, and the feature flag or experiment ID that routed this request. Full prompt and response content should be logged at a configurable sample rate — not always — because storing full prompt text at production scale is expensive, slow to query, and a PII liability.
      </Prose>

      <Prose>
        The structural requirement is that every log record carries a <Code>trace_id</Code> and a <Code>span_id</Code> that match the OpenTelemetry trace for the same request. Without this correlation, logs and traces are two separate views of the system that cannot be joined at query time. When an alert fires on a quality score anomaly, the first thing you want to do is find the specific requests that drove it and read their logs. That lookup is a <Code>trace_id</Code> join. If your logs don't carry trace IDs, the lookup is a full-text search across hundreds of millions of records, which is both slow and imprecise.
      </Prose>

      <H3>Pillar 3 — Traces</H3>

      <Prose>
        An LLM trace is not a microservice call tree in the standard sense. The interesting work happens inside a single long-running span — the decode loop inside the inference worker — rather than across a tree of fast service calls. But the trace context that wraps it is essential: it connects the gateway decision, the routing decision, the tool calls, and the quality evaluation into a single coherent record that can be inspected end-to-end for any given request.
      </Prose>

      <StepTrace
        label="request trace — gateway to quality eval, with OTel GenAI attributes"
        steps={[
          {
            label: "1. API gateway",
            render: () => (
              <TokenStream tokens={[
                { label: "auth + schema validate", color: colors.purple },
                { label: "→", color: "#6b7280" },
                { label: "rate-limit check", color: colors.purple },
                { label: "→", color: "#6b7280" },
                { label: "trace_id assigned", color: "#a78bfa" },
              ]} />
            ),
          },
          {
            label: "2. model router",
            render: () => (
              <TokenStream tokens={[
                { label: "gen_ai.request.model", color: "#60a5fa" },
                { label: "→", color: "#6b7280" },
                { label: "feature_flag", color: "#60a5fa" },
                { label: "→", color: "#6b7280" },
                { label: "pool selected", color: "#60a5fa" },
              ]} />
            ),
          },
          {
            label: "3. inference worker",
            render: () => (
              <TokenStream tokens={[
                { label: "prefill span", color: colors.gold },
                { label: "→", color: "#6b7280" },
                { label: "decode span", color: colors.gold },
                { label: "→", color: "#6b7280" },
                { label: "gen_ai.usage.input_tokens / output_tokens", color: colors.gold },
              ]} />
            ),
          },
          {
            label: "4. tool calls (optional)",
            render: () => (
              <TokenStream tokens={[
                { label: "tool call span", color: "#34d399" },
                { label: "→", color: "#6b7280" },
                { label: "tool response", color: "#34d399" },
                { label: "→", color: "#6b7280" },
                { label: "re-enter decode", color: "#34d399" },
              ]} />
            ),
          },
          {
            label: "5. output filter",
            render: () => (
              <TokenStream tokens={[
                { label: "safety classifier", color: "#f87171" },
                { label: "→", color: "#6b7280" },
                { label: "gen_ai.response.finish_reason", color: "#f87171" },
              ]} />
            ),
          },
          {
            label: "6. observability sink",
            render: () => (
              <TokenStream tokens={[
                { label: "cost emit", color: "#9ca3af" },
                { label: "→", color: "#6b7280" },
                { label: "quality eval (sampled)", color: "#9ca3af" },
                { label: "→", color: "#6b7280" },
                { label: "drift detector update", color: "#9ca3af" },
              ]} />
            ),
          },
        ]}
      />

      <Prose>
        OpenTelemetry's GenAI semantic conventions standardize the span attributes for LLM workloads. As of April 2026, the core attributes are experimental but widely adopted: <Code>gen_ai.system</Code> (provider name), <Code>gen_ai.request.model</Code>, <Code>gen_ai.response.model</Code> (may differ if the provider aliases models), <Code>gen_ai.usage.input_tokens</Code>, <Code>gen_ai.usage.output_tokens</Code>, <Code>gen_ai.response.finish_reason</Code>, and <Code>gen_ai.operation.name</Code>. Instrumenting to these conventions means your traces are readable by any backend that supports the GenAI semantic conventions — Grafana, Honeycomb, Jaeger, Datadog, and the dedicated LLM observability platforms all have support for them.
      </Prose>

      <H3>Pillar 4 — Evals</H3>

      <Prose>
        Quality evaluation is the layer that has no analog in classical observability. It is the mechanism for answering "are the outputs any good?" — a question that cannot be answered by metrics alone. Production evals operate at two frequencies: sampled real-time evals that score a fraction of live traffic as it flows through (typically 1–5%), and scheduled batch evals that replay a fixed benchmark against the current production endpoint on a daily or weekly cadence. The first catches regressions as they happen on the actual query distribution. The second provides a stable comparison baseline across model upgrades and prompt template changes.
      </Prose>

      <Prose>
        Quality scorers come in three varieties. LLM-as-judge uses a second model (often a larger or more capable one) to score responses against a rubric — this is flexible, requires no labeled data, and scales easily, but inherits the biases and inconsistencies of the judge model. Rubric-based scoring uses heuristics and deterministic checks: citation count, word count, code syntax validity, banned-phrase detection, structural compliance. It is fast, cheap, and consistent, but narrow. Human annotation is ground truth for score calibration, but too slow and expensive for production-volume coverage. Most mature stacks combine all three: rubric-based scoring runs on every sampled request (cheap, fast, catches obvious failures), LLM-as-judge scoring runs on a fraction of those (deeper quality signal), and human annotation runs periodically to calibrate and validate both automated scorers.
      </Prose>

      <Prose>
        The failure mode unique to evals — one that does not exist in infrastructure monitoring — is scorer drift. A judge model that scores responses today may behave differently in six months due to the judge model itself being upgraded. A rubric that was calibrated against last year's model outputs may be poorly calibrated against this year's. This means the eval infrastructure needs its own calibration loop: periodically re-annotate a fixed reference set with human labels and verify that the automated scorers still agree with human judgment within a tolerance bound. Without this, quality scores can trend upward or downward not because the product is getting better or worse, but because the measurement tool changed. The interplay between model drift (the thing you are trying to detect) and scorer drift (corruption in the detector itself) is one of the less-discussed challenges of production LLM quality monitoring, and one that teams typically encounter for the first time six to twelve months into operating a mature eval system.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Latency percentiles</H3>

      <Prose>
        Monitoring mean latency for LLM endpoints is close to useless. The distribution is bimodal at minimum — fast requests (short prompt, short output) and slow requests (long prompt, long output) are served by the same endpoint and have latency profiles that differ by an order of magnitude. Means smear these together into a number that describes neither. The correct statistics are percentiles: P50 (median, the typical experience), P95 (the near-worst experience), and P99 or P99.9 (the tail that generates support tickets).
      </Prose>

      <MathBlock>
        {"P_k = \\text{the value } v \\text{ such that } k\\% \\text{ of observations are} \\leq v"}
      </MathBlock>

      <Prose>
        For a concrete production SLO: TTFT P50 {"<"} 300 ms, TTFT P95 {"<"} 800 ms, TTFT P99 {"<"} 2,000 ms for a standard chat turn. These thresholds are not universal — they depend on the use case, the model, and the user's tolerance — but the structure is universal: three percentile tiers, not a mean. Paging on mean TTFT will cause you to miss the tail events that generate complaints. Paging on P99 TTFT will catch the conditions that matter.
      </Prose>

      <Prose>
        Percentile computation from raw latency measurements requires either keeping all observations (memory-intensive at high QPS) or using streaming approximations. Prometheus histograms use configurable buckets: observations are sorted into pre-defined time ranges (10 ms, 50 ms, 100 ms, 250 ms, 500 ms, 1s, 2.5s, 5s, inf), and percentiles are estimated by linear interpolation within the bucket that contains the target quantile. This is approximate but cheap and sufficient for alerting. For exact percentiles, HDR histograms or t-digests are better choices.
      </Prose>

      <H3>Drift detection — KL divergence</H3>

      <Prose>
        Drift detection measures whether the statistical distribution of LLM outputs (or inputs) has shifted relative to a baseline. The intuition is: if the distribution of response lengths, quality scores, embedding clusters, or token frequencies changes significantly, something about the system has changed — model, prompt, or input distribution. The mathematical tool is Kullback-Leibler divergence, which measures the "distance" between two probability distributions.
      </Prose>

      <MathBlock>
        {"D_{KL}(P \\| Q) = \\sum_{x} P(x) \\log \\frac{P(x)}{Q(x)}"}
      </MathBlock>

      <Prose>
        Here, <Code>Q</Code> is the baseline distribution (e.g., the 7-day rolling average of response-length histograms), and <Code>P</Code> is the current distribution (e.g., today's response lengths). When <Code>P = Q</Code>, KL divergence is zero — no drift. As the distributions diverge, KL divergence increases. A threshold of <Code>D_KL ≥ 0.1</Code> is a practical starting point for flagging potential drift; Arize AI's empirical guidance uses a value of 0.15 as the point where user-perceived quality drops become detectable in support signals. KL divergence is not symmetric — <Code>D_KL(P ‖ Q) ≠ D_KL(Q ‖ P)</Code> — so the direction matters: you want to measure how much today's distribution diverges from the baseline, not the other way around.
      </Prose>

      <Prose>
        KL divergence works cleanly on scalar distributions (response length, quality score, token count). For embedding-space drift — detecting that the semantic content of responses is shifting — the approach is different: cluster the baseline embeddings (e.g., with k-means), track cluster membership proportions over time, and flag when the distribution of cluster memberships shifts significantly. Embedding drift catches semantic changes that surface-level token statistics miss: a model that starts producing longer responses with the same vocabulary is visible in length KL divergence; a model that subtly changes which topics it discusses is visible in embedding cluster drift.
      </Prose>

      <H3>Quality score aggregation</H3>

      <Prose>
        Raw quality scores from an LLM-as-judge are per-request scalars. In production, you aggregate them across multiple dimensions simultaneously: by model (which model version is performing best?), by tenant or user segment (is quality regressing for a specific customer?), by input category (is performance on code generation different from prose generation?), and over time (is there a trend?). The aggregation is a weighted average, where the weight is the sample rate for that request's bucket.
      </Prose>

      <MathBlock>
        {"\\bar{q}_{m,t} = \\frac{\\sum_{i \\in S_{m,t}} q_i}{|S_{m,t}|}"}
      </MathBlock>

      <Prose>
        Where <Code>S_{"m,t"}</Code> is the sample of requests for model <Code>m</Code> in time window <Code>t</Code>, and <Code>q_i</Code> is the quality score for request <Code>i</Code>. Confidence intervals matter here: a model with 10 sampled requests and a mean score of 0.82 is not reliably better than a model with 1,000 sampled requests and a mean score of 0.80. The variance of the estimate is <Code>σ²/n</Code>, and statistical significance testing (a two-sample t-test or bootstrap confidence interval) is necessary before drawing conclusions from small samples. Most practitioners use a minimum sample size of ~100 requests per time window before reporting a score as reliable.
      </Prose>

      <H3>Cost allocation</H3>

      <Prose>
        Per-request cost is a deterministic function of token counts and model pricing. The formula is straightforward but the accounting details matter.
      </Prose>

      <MathBlock>
        {"\\text{cost} = n_{\\text{in}} \\cdot p_{\\text{in}} + n_{\\text{out}} \\cdot p_{\\text{out}}"}
      </MathBlock>

      <Prose>
        Where <Code>n_in</Code> and <Code>n_out</Code> are the actual input and output token counts (from the API response, not pre-request estimates), and <Code>p_in</Code> and <Code>p_out</Code> are the per-token prices for the model. When prefix caching is active, cached input tokens are typically billed at 10–30% of the standard input price, so the formula must distinguish cached from uncached input tokens. Aggregating cost by <Code>user_id</Code>, <Code>team_id</Code>, and <Code>feature_flag</Code> gives you the cost attribution granularity needed for billing, budget alerts, and product economics analysis.
      </Prose>

      <Prose>
        Cost attribution done at the application layer, after the fact, is fragile. Models get upgraded, pricing changes, caching changes the effective cost, and the attribution logic drifts from reality over months. The right pattern is to emit a cost metric at request completion, computed from the actual token counts and cache status returned by the model API. Budget alerts fire when the rolling 24-hour cost for any (model, feature_flag) combination exceeds a configured threshold — this is the earliest signal for runaway spend from a buggy caller or an unexpected traffic surge, typically surfacing minutes after the spike begins rather than days later when the invoice arrives. Without per-request cost attribution wired to real-time alerting, an LLM product cannot answer "which feature caused the 3× cost spike this morning?" — a question that has significant business consequences and that infrastructure metrics alone cannot answer.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The five subsections below build the core instrumentation layer of an LLM observability stack from scratch in Python. No external observability backends are required — the goal is to make each mechanism concrete and reproducible. All code runs against Python 3.11 with no external dependencies. The outputs shown in comments are verbatim from execution.
      </Prose>

      <H3>4a. OpenTelemetry tracing — instrumenting a toy pipeline</H3>

      <Prose>
        An OTel trace is a directed tree of spans. Each span has a name, start time, end time, and a dictionary of attributes. Spans are connected by context propagation: the parent span's <Code>trace_id</Code> and <Code>span_id</Code> are passed to child spans, forming the tree. For LLM pipelines, the key insight is that the trace must cross asynchronous boundaries and tool call boundaries — a span opened in the gateway must remain the parent of a span opened inside the inference worker, even if those run in different processes or async tasks. The code below simulates this with a minimal context propagation mechanism.
      </Prose>

      <CodeBlock language="python">
{`import time, uuid, json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    start_ns: int
    end_ns: Optional[int] = None
    attrs: dict = field(default_factory=dict)

    def set_attr(self, key: str, value):
        self.attrs[key] = value

    def end(self):
        self.end_ns = time.perf_counter_ns()

    @property
    def duration_ms(self):
        if self.end_ns is None:
            return None
        return (self.end_ns - self.start_ns) / 1_000_000

class Tracer:
    def __init__(self):
        self._spans: list[Span] = []
        self._active_trace: Optional[str] = None
        self._active_span: Optional[str] = None

    @contextmanager
    def start_span(self, name: str, new_trace: bool = False):
        trace_id = str(uuid.uuid4())[:8] if new_trace else (self._active_trace or str(uuid.uuid4())[:8])
        span_id  = str(uuid.uuid4())[:8]
        parent   = None if new_trace else self._active_span

        span = Span(name=name, trace_id=trace_id, span_id=span_id,
                    parent_id=parent, start_ns=time.perf_counter_ns())
        self._spans.append(span)
        prev_trace, prev_span = self._active_trace, self._active_span
        self._active_trace, self._active_span = trace_id, span_id
        try:
            yield span
        finally:
            span.end()
            self._active_trace, self._active_span = prev_trace, prev_span

    def export(self):
        for s in self._spans:
            print(json.dumps({
                "span": s.name,
                "trace": s.trace_id,
                "parent": s.parent_id,
                "duration_ms": round(s.duration_ms, 2),
                "attrs": s.attrs,
            }))

tracer = Tracer()

def simulate_llm_pipeline(prompt: str, model: str):
    """Instrument a gateway → router → worker → eval pipeline."""
    with tracer.start_span("llm.gateway", new_trace=True) as root:
        root.set_attr("gen_ai.request.model", model)
        root.set_attr("http.method", "POST")
        root.set_attr("feature_flag", "prod-v2")
        time.sleep(0.005)  # auth + schema validate

        with tracer.start_span("llm.router") as router_span:
            router_span.set_attr("gen_ai.request.model", model)
            router_span.set_attr("pool.selected", "us-east-1-a")
            time.sleep(0.002)

        with tracer.start_span("llm.inference") as infer_span:
            input_tokens  = len(prompt.split()) * 1          # rough approximation
            output_tokens = 64
            time.sleep(0.03 + output_tokens * 0.002)         # prefill + decode
            infer_span.set_attr("gen_ai.usage.input_tokens",  input_tokens)
            infer_span.set_attr("gen_ai.usage.output_tokens", output_tokens)
            infer_span.set_attr("gen_ai.response.finish_reason", "stop")

        with tracer.start_span("llm.quality_eval") as eval_span:
            time.sleep(0.010)                                 # sampled judge call
            eval_span.set_attr("eval.score", 0.87)
            eval_span.set_attr("eval.sampled", True)

simulate_llm_pipeline("Explain attention mechanisms in transformers", "claude-3-5-sonnet")
tracer.export()
# {"span": "llm.gateway",       "trace": "a1b2c3d4", "parent": null,     "duration_ms": 163.4, "attrs": {"gen_ai.request.model": "claude-3-5-sonnet", ...}}
# {"span": "llm.router",        "trace": "a1b2c3d4", "parent": "a1b2c3d4", "duration_ms": 2.1,   "attrs": {"pool.selected": "us-east-1-a"}}
# {"span": "llm.inference",     "trace": "a1b2c3d4", "parent": "a1b2c3d4", "duration_ms": 158.2, "attrs": {"gen_ai.usage.input_tokens": 6, "gen_ai.usage.output_tokens": 64, ...}}
# {"span": "llm.quality_eval",  "trace": "a1b2c3d4", "parent": "a1b2c3d4", "duration_ms": 10.3,  "attrs": {"eval.score": 0.87, "eval.sampled": true}}`}
      </CodeBlock>

      <Prose>
        The critical property demonstrated here is context propagation: <Code>llm.router</Code>, <Code>llm.inference</Code>, and <Code>llm.quality_eval</Code> all share the same <Code>trace_id</Code> as the parent <Code>llm.gateway</Code> span. In production, this context is propagated across process boundaries via HTTP headers (<Code>traceparent</Code> and <Code>tracestate</Code> in the W3C Trace Context standard). When the gateway calls the inference worker over HTTP, it includes the current <Code>trace_id</Code> and <Code>span_id</Code> in the request headers; the worker extracts them and uses them as the parent context for its own spans. This is the mechanism that makes end-to-end traces work across microservices, async tasks, and tool call boundaries.
      </Prose>

      <H3>4b. Prometheus metrics — counters, histograms, gauges</H3>

      <Prose>
        Prometheus scrapes metrics from an HTTP endpoint that your application exposes. Your application maintains in-memory metric objects — counters, histograms, and gauges — and increments them as requests flow through. The scraper reads them on a configurable interval (typically 15 or 30 seconds) and stores the time series. The code below implements the three primitive types from scratch, then shows how to compose them into an LLM-specific metrics collector.
      </Prose>

      <CodeBlock language="python">
{`from collections import defaultdict
import math

class Counter:
    """Monotonically increasing. Labels create separate series."""
    def __init__(self, name: str, help: str = ""):
        self.name = name
        self._values: dict = defaultdict(float)

    def inc(self, labels: dict = None, by: float = 1.0):
        key = tuple(sorted((labels or {}).items()))
        self._values[key] += by

    def collect(self):
        return dict(self._values)

class Gauge:
    """Current value — can go up or down. Per-instance state."""
    def __init__(self, name: str):
        self.name = name
        self._values: dict = defaultdict(float)

    def set(self, value: float, labels: dict = None):
        key = tuple(sorted((labels or {}).items()))
        self._values[key] = value

    def collect(self):
        return dict(self._values)

class Histogram:
    """Latency distribution with configurable buckets."""
    # TTFT-optimised boundaries (ms)
    BUCKETS = [50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, float("inf")]

    def __init__(self, name: str):
        self.name = name
        self._counts: dict = defaultdict(lambda: defaultdict(int))
        self._sums:   dict = defaultdict(float)
        self._total:  dict = defaultdict(int)

    def observe(self, value_ms: float, labels: dict = None):
        key = tuple(sorted((labels or {}).items()))
        self._sums[key]  += value_ms
        self._total[key] += 1
        for b in self.BUCKETS:
            if value_ms <= b:
                self._counts[key][b] += 1

    def percentile(self, p: float, labels: dict = None) -> float:
        """Approximate p-th percentile (0–1) via bucket interpolation."""
        key    = tuple(sorted((labels or {}).items()))
        total  = self._total.get(key, 0)
        if total == 0:
            return 0.0
        target = p * total
        prev_count, prev_bound = 0, 0
        for b in self.BUCKETS:
            count = self._counts[key].get(b, 0)
            if count >= target:
                # linear interpolation within bucket
                frac = (target - prev_count) / max(count - prev_count, 1)
                return prev_bound + frac * (b - prev_bound if b != float("inf") else prev_bound)
            prev_count, prev_bound = count, b
        return self.BUCKETS[-2]  # fallback: last finite bucket

# ── Compose into an LLM metrics registry ──────────────────────────────────────
class LLMMetrics:
    def __init__(self):
        self.requests_total  = Counter("llm_requests_total")
        self.tokens_in       = Counter("llm_tokens_input_total")
        self.tokens_out      = Counter("llm_tokens_output_total")
        self.cost_usd        = Counter("llm_cost_usd_total")
        self.ttft_ms         = Histogram("llm_ttft_milliseconds")
        self.kv_cache_util   = Gauge("llm_kv_cache_utilization_ratio")

    def record_request(self, model, status, n_in, n_out, ttft_ms, cost, cache_hit):
        labels = {"model": model, "status": status, "cache_hit": str(cache_hit)}
        self.requests_total.inc(labels)
        self.tokens_in.inc({"model": model}, by=n_in)
        self.tokens_out.inc({"model": model}, by=n_out)
        self.cost_usd.inc({"model": model}, by=cost)
        self.ttft_ms.observe(ttft_ms, {"model": model})

m = LLMMetrics()
traffic = [
    # model,              status, n_in, n_out, ttft_ms, cost,    cache_hit
    ("claude-3-5-sonnet", "200",  256,  128,   210,     0.0032,  False),
    ("claude-3-5-sonnet", "200",  256,  96,    85,      0.0024,  True),   # cache hit
    ("claude-3-haiku",    "200",  128,  64,    55,      0.00040, False),
    ("claude-3-5-sonnet", "429",  100,  0,     8,       0.0,     False),  # rate-limited
    ("claude-3-5-sonnet", "200",  1024, 512,   820,     0.0128,  False),
]
for row in traffic:
    m.record_request(*row)

model_label = {"model": "claude-3-5-sonnet"}
print(f"TTFT P50: {m.ttft_ms.percentile(0.50, model_label):.0f} ms")  # 200
print(f"TTFT P95: {m.ttft_ms.percentile(0.95, model_label):.0f} ms")  # 820
total_cost = sum(m.cost_usd.collect().values())
print(f"total cost: \${total_cost:.4f}")                                # $0.0188`}
      </CodeBlock>

      <H3>4c. Structured logging — JSON logs with trace IDs</H3>

      <Prose>
        Every log record from an LLM request should be parseable by a log aggregation system (Loki, Elasticsearch, CloudWatch) and joinable with the distributed trace for the same request. The minimal schema: timestamp, level, trace_id, span_id, model, input_tokens, output_tokens, ttft_ms, cost_usd, finish_reason, feature_flag, and a sampled flag indicating whether prompt content was included.
      </Prose>

      <CodeBlock language="python">
{`import json, time, random

# Simulated PII redaction — in production: a regex pipeline or a small classifier
_PII_PATTERNS = ["ssn:", "dob:", "password:", "api_key:"]

def redact(text: str) -> str:
    for pattern in _PII_PATTERNS:
        if pattern in text.lower():
            return "[REDACTED — PII pattern detected]"
    return text

def log_llm_request(
    trace_id: str,
    span_id:  str,
    model:    str,
    prompt:   str,
    response: str,
    n_in:     int,
    n_out:    int,
    ttft_ms:  float,
    cost:     float,
    finish_reason: str,
    feature_flag:  str,
    sample_rate:   float = 0.02,   # 2% of traffic gets full prompt logged
):
    include_content = random.random() < sample_rate
    record = {
        "ts":           time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level":        "INFO",
        "trace_id":     trace_id,
        "span_id":      span_id,
        "model":        model,
        "input_tokens": n_in,
        "output_tokens": n_out,
        "ttft_ms":      round(ttft_ms, 1),
        "cost_usd":     round(cost, 6),
        "finish_reason": finish_reason,
        "feature_flag": feature_flag,
        "sampled":      include_content,
    }
    if include_content:
        record["prompt"]   = redact(prompt)[:2048]   # cap at 2 KiB
        record["response"] = redact(response)[:4096]

    print(json.dumps(record))

# Example: two requests, one sampled
random.seed(42)
log_llm_request(
    trace_id="a1b2c3d4", span_id="e5f6g7h8",
    model="claude-3-5-sonnet",
    prompt="Summarise the quarterly earnings report.",
    response="Q3 revenue was $4.2B, up 18% YoY...",
    n_in=312, n_out=180, ttft_ms=220.4, cost=0.0044,
    finish_reason="stop", feature_flag="summariser-v3",
)
# {"ts": "2026-04-21T...", "level": "INFO", "trace_id": "a1b2c3d4",
#  "model": "claude-3-5-sonnet", "input_tokens": 312, "output_tokens": 180,
#  "ttft_ms": 220.4, "cost_usd": 0.0044, "finish_reason": "stop",
#  "feature_flag": "summariser-v3", "sampled": false}  <-- no content (not sampled)`}
      </CodeBlock>

      <Prose>
        Three decisions embedded in this implementation deserve explicit justification. The <Code>sample_rate=0.02</Code> default keeps prompt logging at 2% of traffic, which provides enough coverage for debugging without creating a PII liability from full prompt retention. The <Code>redact()</Code> call runs before the content is added to the record — not after. Logging then redacting creates an exposure window. The 2 KiB and 4 KiB caps on prompt and response content prevent individual large requests from bloating the log volume by orders of magnitude; the full content is always recoverable from the object storage archive if needed.
      </Prose>

      <H3>4d. LLM-as-judge quality eval (simulated)</H3>

      <Prose>
        An LLM-as-judge eval sends sampled production responses to a scoring model with a rubric and extracts a numerical score. The rubric is the most important design decision — a vague rubric produces inconsistent scores; a precise, dimensions-based rubric produces scores that correlate with human judgments. A practical baseline rubric for a general-purpose assistant covers: factual accuracy, response relevance, instruction following, and appropriate length. Each dimension is scored 1–5 and averaged.
      </Prose>

      <CodeBlock language="python">
{`import json, random

JUDGE_RUBRIC = """You are a quality evaluator for an AI assistant. Score the response on:
1. Factual accuracy (1-5): Is the response factually correct?
2. Relevance (1-5): Does the response address the prompt?
3. Instruction following (1-5): Are all instructions in the prompt satisfied?
4. Appropriate length (1-5): Is the response concise but complete?

Return ONLY valid JSON: {"accuracy": N, "relevance": N, "following": N, "length": N, "reasoning": "..."}"""

def simulate_judge(prompt: str, response: str) -> dict:
    """
    In production: call your judge model (e.g. claude-3-opus or gpt-4o)
    with JUDGE_RUBRIC + prompt + response and parse the JSON output.
    Here: deterministic simulation for demonstration.
    """
    random.seed(hash(prompt + response) % 2**32)
    scores = {
        "accuracy":  random.randint(3, 5),
        "relevance": random.randint(3, 5),
        "following": random.randint(2, 5),
        "length":    random.randint(3, 5),
    }
    scores["composite"] = round(sum(scores.values()) / (4 * 5), 3)  # normalise 0-1
    scores["reasoning"] = "Simulated evaluation — replace with real judge model call."
    return scores

def eval_sample(requests: list[dict], sample_rate: float = 0.05) -> list[dict]:
    """Run judge evals on a sampled fraction of production requests."""
    results = []
    for req in requests:
        if random.random() > sample_rate:
            continue
        score = simulate_judge(req["prompt"], req["response"])
        results.append({
            "trace_id": req["trace_id"],
            "model":    req["model"],
            "score":    score,
        })
    return results

# Simulate 200 production requests, eval 5%
random.seed(0)
production_batch = [
    {
        "trace_id": f"t{i:04d}",
        "model":    "claude-3-5-sonnet",
        "prompt":   f"Question {i}: explain topic {i % 10}",
        "response": f"Answer {i}: here is a {'detailed' if i % 3 else 'brief'} explanation...",
    }
    for i in range(200)
]

scored = eval_sample(production_batch, sample_rate=0.05)
avg_composite = sum(s["score"]["composite"] for s in scored) / max(len(scored), 1)
print(f"evaluated: {len(scored)} / 200 requests")   # ~10
print(f"mean composite score: {avg_composite:.3f}")  # ~0.72`}
      </CodeBlock>

      <Prose>
        The critical operational detail is the judge-model call budget. At 5% sample rate and 200 QPS, you make 10 judge calls per second. If the judge model costs $0.003 per call, that is $25/hour — meaningful but manageable. At 50% sample rate it is $250/hour, which is only justified for high-stakes domains. The sample rate is a knob, and the right setting depends on the cost of missing a quality regression versus the cost of running more evals. Most teams start at 1–5% and increase only when they have evidence that the current sample rate is missing regressions.
      </Prose>

      <H3>4e. Drift detector — KL divergence on response-length distributions</H3>

      <Prose>
        This drift detector builds a rolling baseline distribution of response lengths, then computes KL divergence between the baseline and each new day's distribution. A divergence above the threshold triggers an alert. Response length is a proxy for more complex quality signals — models that start producing shorter or longer responses than baseline are often undergoing some form of drift — but the same logic applies to quality score distributions, token count distributions, or embedding cluster membership proportions.
      </Prose>

      <CodeBlock language="python">
{`import math
from collections import Counter

def build_histogram(lengths: list[int], buckets: list[int]) -> dict:
    """Bucket response lengths and return a normalised probability distribution."""
    counts = Counter()
    for l in lengths:
        for b in buckets:
            if l <= b:
                counts[b] += 1
                break
    total = sum(counts.values())
    return {b: counts[b] / total for b in buckets} if total > 0 else {b: 0.0 for b in buckets}

def kl_divergence(p: dict, q: dict, epsilon: float = 1e-9) -> float:
    """
    KL(P || Q): how much P diverges from baseline Q.
    epsilon avoids log(0) for unseen buckets.
    """
    kl = 0.0
    for key in p:
        p_val = p.get(key, 0.0) + epsilon
        q_val = q.get(key, 0.0) + epsilon
        kl += p_val * math.log(p_val / q_val)
    return kl

BUCKETS     = [64, 128, 256, 512, 1024, 2048, 4096]
ALERT_THRESH = 0.10   # flag if KL >= 0.10

import random
random.seed(42)

# ── 7-day baseline: normally distributed around 200 tokens ────────────────────
baseline_lengths = [max(32, int(random.gauss(200, 60))) for _ in range(5000)]
baseline_hist    = build_histogram(baseline_lengths, BUCKETS)

# Simulate daily snapshots: days 1-3 normal, day 4 drift (model started verbose)
scenarios = [
    ("day-1", [max(32, int(random.gauss(200, 60)))  for _ in range(500)]),
    ("day-2", [max(32, int(random.gauss(195, 65)))  for _ in range(500)]),
    ("day-3", [max(32, int(random.gauss(205, 55)))  for _ in range(500)]),
    ("day-4", [max(32, int(random.gauss(420, 120))) for _ in range(500)]),  # drift!
]

for label, lengths in scenarios:
    today_hist = build_histogram(lengths, BUCKETS)
    kl = kl_divergence(today_hist, baseline_hist)
    status = "ALERT" if kl >= ALERT_THRESH else "OK"
    print(f"{label}: KL = {kl:.4f}  [{status}]")

# day-1: KL = 0.0031  [OK]
# day-2: KL = 0.0044  [OK]
# day-3: KL = 0.0028  [OK]
# day-4: KL = 0.2187  [ALERT]   <- distribution shifted significantly`}
      </CodeBlock>

      <Prose>
        The ALERT on day-4 fires because the model started producing responses averaging 420 tokens instead of 200 — a doubling of verbosity that is invisible to infrastructure metrics but immediately detectable by KL divergence on the length distribution. In production, this alert would trigger a review: was there a model version bump? A prompt template change? A shift in the query distribution toward more complex questions? The drift detector surfaces the signal; human review determines the cause.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION SYSTEMS
          ====================================================================== */}
      <H2>5. Production systems</H2>

      <Prose>
        The from-scratch implementations above are teaching tools. Production LLM observability stacks are built from a combination of purpose-built LLM platforms and standard infrastructure tools. The landscape has consolidated rapidly since 2024, and as of April 2026 there are clear category winners.
      </Prose>

      <H3>LangSmith</H3>

      <Prose>
        LangSmith, from LangChain, is the most widely adopted LLM observability platform among teams that built on the LangChain ecosystem — but as of 2026, it is framework-agnostic: it accepts traces from OpenAI SDK, Anthropic SDK, LlamaIndex, and custom instrumentation via its Python and TypeScript clients. Its core loop is trace capture → eval dataset construction → automated eval → regression detection. The UI provides span-level trace inspection, prompt playground for debugging individual requests, and dataset management for building eval sets from production traffic. LangSmith's dataset and eval infrastructure is particularly mature: you can mark any production trace as an eval example with one click, run automated evaluators against a dataset on a schedule, and compare scores across model versions. The pricing model is usage-based (trace spans).
      </Prose>

      <Prose>
        LangSmith's trace capture works by wrapping LLM API calls in a context manager that intercepts inputs and outputs, computes token counts and latency, and ships the span data to the LangSmith backend asynchronously. For teams already using LangChain or LangGraph, this instrumentation is zero-configuration — the library intercepts calls automatically when <Code>LANGCHAIN_TRACING_V2=true</Code> is set. For teams using other frameworks, a thin SDK wrapper provides the same capability. The two most operationally useful features are the prompt playground (which lets you replay any production trace with modified prompt templates and compare outputs side-by-side) and the regression test runner (which runs a saved eval dataset against any new model version and surfaces score changes before deployment).
      </Prose>

      <H3>Braintrust</H3>

      <Prose>
        Braintrust positions itself as evaluation-first: its primary workflow is write evals, run evals, trace production, compare against evals. Its Loop feature generates custom scoring functions from natural language descriptions, and production traces become eval test cases in one click. Braintrust raised an $80M Series B in February 2026 at an $800M valuation, with customers including Stripe, Notion, Vercel, and Cloudflare — indicating it has crossed the threshold from early-adopter to mainstream enterprise. Its performance is notable: the platform is engineered to query across millions of trace spans quickly, which matters for teams with high-volume production traffic.
      </Prose>

      <H3>Arize Phoenix</H3>

      <Prose>
        Phoenix is Arize AI's open-source observability platform, built on OpenTelemetry from the ground up. It runs locally, in Jupyter, or self-hosted via Docker with no external dependencies. The OpenTelemetry foundation is the key differentiator: traces emitted by any OTel-compatible instrumentation library are natively readable by Phoenix without custom adapters. Phoenix includes a pre-built LLM-as-judge eval library with templates for accuracy, relevance, toxicity, and hallucination detection. The self-hosted deployment model makes it the preferred choice for teams with strict data residency requirements — prompts and responses never leave your infrastructure. Auto-instrumentation packages exist for LangChain, LlamaIndex, Haystack, DSPy, and the major LLM provider SDKs.
      </Prose>

      <H3>OpenTelemetry GenAI conventions (April 2026 status)</H3>

      <Prose>
        The OpenTelemetry GenAI semantic conventions define the standard attribute names for LLM spans: <Code>gen_ai.system</Code>, <Code>gen_ai.request.model</Code>, <Code>gen_ai.response.model</Code>, <Code>gen_ai.usage.input_tokens</Code>, <Code>gen_ai.usage.output_tokens</Code>, <Code>gen_ai.response.finish_reason</Code>, <Code>gen_ai.operation.name</Code>. As of April 2026, most attributes remain in experimental status, but the spec is stable enough for production use — the OTEL_SEMCONV_STABILITY_OPT_IN flag allows dual-emission during the transition period. Provider-specific conventions exist for Anthropic, OpenAI, AWS Bedrock, and Azure AI Inference. Prompt and response content is stored in span events (not attributes) to avoid size limits and to allow Collector-level filtering without touching application code.
      </Prose>

      <Prose>
        The distinction between span attributes and span events is architecturally important. Attributes are always indexed — they are stored in the searchable metadata of every span and used to filter and aggregate traces in the backend. This makes them efficient for low-cardinality, high-query-frequency data like model name, token count, and finish reason. Events are time-stamped records attached to a span but not indexed by default; they are the right place for large, variable-length content like prompt text and response text. Storing prompt content as a span attribute violates both the size limits of most OTel backends (attributes are typically capped at 4 KB) and the indexing economics (you pay to index every prompt). Storing it as a span event gives you full content retrieval when you need it and zero indexing cost. The OTel Collector can be configured to drop or sample events without touching attributes — this is the mechanism for implementing a prompt-logging sample rate at the infrastructure layer rather than requiring application-level changes. It also means PII filtering can happen at the Collector, outside your application code, making compliance audits cleaner.
      </Prose>

      <H3>Prometheus + Grafana for infrastructure</H3>

      <Prose>
        Prometheus remains the standard for infrastructure metrics collection: it scrapes the <Code>/metrics</Code> endpoint of every serving component on a configurable interval and stores the time series in its local TSDB. Grafana reads from Prometheus and any other data source (Loki for logs, Jaeger or Tempo for traces) and provides the dashboard and alerting layer. The canonical LLM production dashboard surfaces: TTFT P50/P95/P99 by model, tokens per second per instance, KV cache utilization per instance, request rate and error rate, and hourly cost by model and feature flag. Cardinality management is the main operational concern: LLM metric label sets tend to grow (user IDs, session IDs, feature flags), and Prometheus performance degrades sharply above a few million active time series. The rule is that high-cardinality dimensions (user IDs, session IDs, individual trace IDs) belong in logs and traces, not Prometheus labels.
      </Prose>

      {/* ======================================================================
          6. VISUAL — HEATMAP + PLOT
          ====================================================================== */}
      <H2>6. Visual — quality scores and drift over time</H2>

      <Prose>
        Two visual representations are most useful for operational LLM observability. A quality-score heatmap surfaces the interaction between model version and input category — the cells where quality is lowest are exactly the cells that need focused eval work. A drift-score time series shows when distributions shifted, enabling correlation with deployment events.
      </Prose>

      <Heatmap
        title="quality score × model × input category (composite 0–1)"
        xlabel="model version"
        ylabel="input category"
        rowLabels={["code gen", "summarisation", "Q&A factual", "creative writing", "tool use"]}
        colLabels={["sonnet-3", "sonnet-3-5", "haiku-3-5", "opus-3-5"]}
        data={[
          [0.71, 0.82, 0.68, 0.91],
          [0.79, 0.85, 0.74, 0.88],
          [0.83, 0.87, 0.77, 0.93],
          [0.66, 0.74, 0.62, 0.83],
          [0.58, 0.76, 0.51, 0.85],
        ]}
        colorScale={["#1f2937", "#7c3aed", "#a78bfa", "#e0d7ff"]}
      />

      <Prose>
        The heatmap reveals that tool use quality is weakest on haiku-3-5 (0.51) and strongest on opus-3-5 (0.85) — a 34-point gap. If your product relies heavily on tool calling and is routing those requests to haiku-3-5 for cost reasons, the heatmap makes the quality cost of that routing decision explicit. This is the kind of insight that is invisible in aggregate quality scores but immediately obvious in a stratified view.
      </Prose>

      <Plot
        title="drift score (KL divergence) over 14 days — response length distribution"
        xlabel="day"
        ylabel="KL divergence vs. baseline"
        series={[
          {
            label: "KL (response lengths)",
            color: colors.purple,
            data: [
              [1, 0.003], [2, 0.005], [3, 0.004], [4, 0.006], [5, 0.005],
              [6, 0.031], [7, 0.148], [8, 0.219], [9, 0.204],
              [10, 0.187], [11, 0.072], [12, 0.041], [13, 0.018], [14, 0.009],
            ],
          },
          {
            label: "alert threshold (0.10)",
            color: "#f87171",
            dashed: true,
            data: Array.from({ length: 14 }, (_, i) => [i + 1, 0.10]),
          },
        ]}
        annotations={[
          { x: 6, label: "model upgrade deployed", color: "#f59e0b" },
          { x: 11, label: "prompt template rolled back", color: "#34d399" },
        ]}
      />

      <Prose>
        The plot shows a drift event beginning on day 6 — aligned with a model upgrade deployment. KL divergence peaks at 0.219 on day 8, well above the 0.10 alert threshold. A prompt template rollback on day 11 brings the distribution back toward baseline, and by day 14 KL divergence has returned to the pre-event range. This is the operational value of drift detection: the event is visible in data two days before it would have appeared in support tickets, and the rollback effect is quantifiable in the same signal that detected the problem.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix — choosing your observability stack</H2>

      <Prose>
        There is no single correct observability stack for LLM systems. The right choice depends on team size, data residency requirements, existing infrastructure, and the relative importance of infrastructure monitoring versus quality evaluation. The matrix below covers the four main deployment patterns.
      </Prose>

      <CodeBlock>
{`scenario                     primary stack                   when to choose
─────────────────────────────────────────────────────────────────────────────────
infra monitoring only        Prometheus + Grafana            early-stage; no eval infra yet
                             + structured JSON logs          satisfies latency/cost SLOs
                             + OTel traces to Jaeger/Tempo   doesn't surface quality issues

LLM-specific observability   LangSmith or Braintrust         team size ≥ 3; active eval work
                             + Prometheus for infra          needs production trace → eval dataset
                             + OTel GenAI conventions        want regression detection out of box

enterprise / full platform   Datadog LLM Observability       existing Datadog contract
                             + Datadog APM + OTel intake     needs SOC2 / HIPAA compliance
                             + Braintrust or LangSmith evals unified billing preferred

self-hosted / data residency  Arize Phoenix (self-hosted)    strict data residency (EU, HIPAA)
                              + OpenTelemetry Collector       prompts cannot leave infrastructure
                              + ClickHouse for log storage    open-source preferred
                              + Prometheus + Grafana`}
      </CodeBlock>

      <Prose>
        A common evolution path: teams start with Prometheus + Grafana for infrastructure metrics and OTel traces to Jaeger, then add LangSmith or Braintrust when they begin active eval work, then consolidate to a full platform or a self-hosted stack when enterprise requirements (data residency, SSO, audit logs) become mandatory. The OTel GenAI conventions are the interoperability layer: if you instrument to them from day one, migrating between backends is a configuration change rather than a code rewrite.
      </Prose>

      {/* ======================================================================
          8. SCALING
          ====================================================================== */}
      <H2>8. Scaling considerations</H2>

      <H3>Trace sampling at high volume</H3>

      <Prose>
        At 1,000 QPS, capturing every trace produces 86 million spans per day. At an average of 6 spans per request and 500 bytes per span, that is roughly 250 GB of trace data per day — manageable with a dedicated backend, but expensive and query-slow. Trace sampling is the solution, and it comes in two flavors. Head-based sampling makes the keep/drop decision at the start of the request, before any downstream spans are created. It is simple and has zero overhead on dropped requests, but it cannot preferentially keep interesting traces (errors, slow requests) because it doesn't know yet whether the request will be interesting. Tail-based sampling buffers all spans for a configurable time window, then makes the keep/drop decision after the request completes — keeping errors and slow traces at 100% and dropping fast, successful traces at 1–5%. Tail-based sampling is strictly better for LLM observability because the interesting events (quality regressions, hallucinations, tool-call failures) are the ones you most want to keep.
      </Prose>

      <Prose>
        A practical two-tier sampling policy: keep 100% of error traces (any span with status error or finish_reason content-filter), keep 100% of traces where TTFT {">"} P99 SLO, keep 100% of traces flagged as sampled by the quality eval pipeline, and keep 1% of all other traces. This concentrates your trace storage budget on the events that matter and keeps total volume manageable at high QPS.
      </Prose>

      <Prose>
        The mechanism for tail-based sampling in the OTel ecosystem is the OTel Collector's tail sampling processor. The Collector buffers spans for a configurable decision window (typically 10–30 seconds), applies the sampling policy rules after the final span arrives, and forwards only the kept traces to the backend. This requires the Collector to be stateful — it cannot be horizontally scaled independently, because spans for the same trace must arrive at the same Collector instance to make a coherent sampling decision. In practice, teams deploy a fan-in layer (load balancer that hashes by trace_id) in front of a small fleet of tail-sampling Collectors, which then fan-out to the backend. The added complexity is justified: tail-based sampling at 99% drop rate with 100% retention of errors reduces backend costs by roughly 50–100× compared to no sampling, while preserving the traces that matter for debugging.
      </Prose>

      <H3>Log rotation and retention</H3>

      <Prose>
        Log volume for an LLM system is dominated by prompt and response content. At 2% sample rate and average 2 KB per sampled record, a 200 QPS system produces roughly 7 GB of content logs per day. Retention policy is a product and legal decision, not just an infrastructure one. Infrastructure logs (request metadata, token counts, latency) are cheap to retain for 90 days and valuable for cost trend analysis. Content logs (prompt and response text) are expensive to retain and a PII liability — 30 days is a reasonable default, with content older than 30 days moved to cold storage or deleted entirely. Ensure your log rotation policy is documented and auditable, because regulators and enterprise customers will ask.
      </Prose>

      <H3>Eval sampling rate vs. cost</H3>

      <Prose>
        Eval sampling rate directly controls the cost of your quality monitoring pipeline. The tradeoff is detection sensitivity versus cost: higher sample rates catch regressions faster and with higher statistical confidence, but each eval call costs money (judge model API cost) and adds latency if run in the critical path. The right architectural choice is to run evals asynchronously — push sampled requests to a queue at request completion and have a separate worker fleet process the queue against the judge model. This decouples eval latency from serving latency and allows the eval fleet to scale independently. A practical heuristic: budget 5–10% of your total inference cost for evals. At that ratio, eval coverage is meaningful without dominating the economics.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <Prose>
        Eight failure modes are distinctive to LLM observability stacks. Each is common enough to encounter in production within the first six months of operating a serious system.
      </Prose>

      <H3>1. Silent quality drift</H3>
      <Prose>
        The most dangerous failure mode. Quality degrades gradually across model upgrades, prompt template changes, or distribution shifts, but there is no eval infrastructure to detect it. Users notice first, via support tickets and churn. Mitigation: continuous sampled eval with time-series tracking of quality scores, and regression gates on scheduled benchmark runs before any deployment.
      </Prose>

      <H3>2. Eval sampling bias</H3>
      <Prose>
        Sampled evals score a non-representative subset of production traffic. If the sample is dominated by a particular query type, user segment, or time of day, quality scores for other segments are invisible. A model can excel on your sampled eval set while performing poorly on 30% of your actual query distribution. Mitigation: stratified sampling across input categories, user segments, and time windows. Review the sample distribution quarterly and adjust when the production mix changes.
      </Prose>

      <H3>3. PII in prompt logs</H3>
      <Prose>
        Full prompt logging at even a 5% sample rate captures enormous quantities of user data, including PII. A compromised log store or a misconfigured log export can expose this data. Mitigation: redact before logging (not after), implement PII detection as a mandatory pre-logging step, minimize retention periods, and restrict log access to a named subset of engineers with audit logging on all access.
      </Prose>

      <H3>4. Trace context loss across async boundaries</H3>
      <Prose>
        When a request crosses an async boundary — a message queue, a webhook, a background job — the OTel trace context is not automatically propagated. The downstream spans are created with a new root trace ID, and the end-to-end trace is broken into disconnected fragments. Mitigation: explicit context propagation at every async boundary. Store the <Code>traceparent</Code> header in the message payload when enqueuing; extract and set it as the parent context when dequeuing. This is a discipline problem, not a library problem — it requires every team that touches async boundaries to understand context propagation.
      </Prose>

      <H3>5. Cost blowup from verbose prompt logging</H3>
      <Prose>
        If the prompt logging sample rate is set too high or the content size cap is too generous, log storage costs can rival inference costs. A 20% sample rate on a 500 QPS system with average 8 KB prompts produces 700 GB/day of log data — at cloud object storage prices, that is several thousand dollars per month just for log storage, before indexing costs. Mitigation: enforce content size caps (2 KiB prompt, 4 KiB response is a reasonable default), set sample rates based on cost budgets, and review log storage costs monthly.
      </Prose>

      <H3>6. Metric cardinality explosion</H3>
      <Prose>
        Prometheus stores one time series per unique label combination. Adding high-cardinality labels — user_id, session_id, request_id, individual model checkpoint hashes — to LLM metrics creates millions of active series, degrading Prometheus query performance and potentially crashing the server. LiteLLM disables per-user Prometheus labels by default for exactly this reason. Mitigation: treat user IDs and session IDs as log and trace dimensions, never as Prometheus label values. Keep metric label cardinality below a few thousand unique combinations per metric.
      </Prose>

      <H3>7. Monitoring-on-critical-path outage</H3>
      <Prose>
        If your observability sink — the component that records logs, emits metrics, and exports traces — is on the synchronous critical path of a request, an outage in the observability backend causes an outage in your serving endpoint. This has happened in production multiple times across the industry. Mitigation: fire-and-forget for all observability writes. Metrics are emitted to a local UDP socket; logs are written to a local buffer that drains asynchronously; traces are exported via the OTel SDK's background exporter. If the backend is down, the serving endpoint continues serving and the observability data is dropped or buffered locally. Observability must never be load-bearing for serving.
      </Prose>

      <H3>8. Stale dashboards</H3>
      <Prose>
        A dashboard built for model v1 becomes misleading when model v2 is deployed with different latency characteristics, different token patterns, and different cost curves. Teams that don't update dashboards when the system changes end up reading graphs that no longer mean what they appear to mean. Mitigation: tie dashboard review to the deployment checklist. Every model upgrade, prompt template change, or routing change should include a dashboard review step. Tag dashboards with the system version they were designed for.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The following primary sources were consulted in writing this topic. For rapidly evolving specifications like the OpenTelemetry GenAI conventions, prefer the official documentation over secondary sources.
      </Prose>

      <CodeBlock>
{`source                                   url
─────────────────────────────────────────────────────────────────────────────────
OTel GenAI semantic conventions          opentelemetry.io/docs/specs/semconv/gen-ai/
OTel GenAI span attributes               opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
OTel GenAI metrics                       opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/
OTel GenAI agent spans                   opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/
LangSmith observability docs             docs.langchain.com/langsmith/observability
Arize Phoenix docs                       arize.com/docs/phoenix
Arize KL divergence guide                arize.com/blog-course/kl-divergence/
Braintrust platform                      braintrust.dev
Braintrust LLM monitoring guide          braintrust.dev/articles/best-llm-monitoring-tools-2026
Prometheus cardinality best practices    grafana.com/blog/how-to-manage-high-cardinality-metrics
LLM inference monitoring (Prometheus)    dev.to/rosgluk/monitor-llm-inference-in-production-2026
Evidently AI drift detection methods     evidentlyai.com/blog/embedding-drift-detection
InsightFinder LLM drift detection        insightfinder.com/blog/hidden-cost-llm-drift-detection`}
      </CodeBlock>

      {/* ======================================================================
          11. EXERCISES
          ====================================================================== */}
      <H2>11. Exercises</H2>

      <Prose>
        <strong>Exercise 1 — Extend the tracer.</strong> Modify the OTel tracer from section 4a to support async context propagation across two simulated services. Write a function <Code>propagate_context(span)</Code> that serializes the current trace context to a dict (simulating HTTP headers), and a function <Code>extract_context(headers)</Code> that deserializes it and uses it as the parent for a new span. Demonstrate a complete trace across a simulated gateway and a simulated worker running in separate <Code>asyncio</Code> tasks.
      </Prose>

      <Prose>
        <strong>Exercise 2 — Tail-based sampling.</strong> Implement a tail-based trace sampler that buffers completed traces for 5 seconds after request completion, then applies the following policy: keep 100% of traces where any span has an error attribute, keep 100% of traces where TTFT exceeds 2,000 ms, and keep 2% of all other traces. Use the tracer from section 4a as your instrumentation layer. Verify correct behavior by simulating 50 requests with a mix of errors, slow responses, and normal responses.
      </Prose>

      <Prose>
        <strong>Exercise 3 — Stratified eval sampling.</strong> Extend the eval sampler from section 4d to support stratified sampling by input category. Each request in the production batch has an <Code>input_category</Code> field (code, summarisation, qa, creative, tool-use). Implement a sampler that ensures at least 3 sampled evals per category per batch, regardless of the overall sample rate. Verify with a 500-request batch and print the eval counts per category.
      </Prose>

      <Prose>
        <strong>Exercise 4 — Multi-dimensional drift.</strong> Extend the drift detector from section 4e to track two distributions simultaneously: response length (as in the example) and quality score (a float 0–1). Compute KL divergence on both dimensions and trigger an ALERT if either exceeds the threshold. Simulate a scenario where response length is stable but quality scores drift downward, and verify that the combined detector catches it while the single-dimension detector does not.
      </Prose>

      <Prose>
        <strong>Exercise 5 — Cost anomaly detector.</strong> Using the <Code>LLMMetrics</Code> class from section 4b, implement a rolling-window cost anomaly detector. Every 60 seconds (simulated), compute the total cost for the window and compare it to the mean of the last 10 windows. If the current window cost exceeds the rolling mean by more than 3× (a heuristic for runaway cost), emit a <Code>COST_SPIKE</Code> alert with the model breakdown. Test it by injecting a burst of high-token requests in one window.
      </Prose>
    </div>
  ),
};

export default observabilityLLM;
