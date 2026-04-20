import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { Plot } from "../../components/viz";

const inferenceCostEconomics = {
  title: "Inference Cost Economics & Compute Scaling",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        Training an LLM costs tens of millions of dollars. Serving one costs more — over the model's lifetime. For any product that sees real traffic, the cumulative inference bill eclipses training cost within months. The gap between a well-tuned inference stack and a naive one is regularly 10× in per-token cost, which means it is also 10× in whether a product is economically viable or not. This topic is a practitioner-level view of where inference dollars go and how to spend them better.
      </Prose>

      <Prose>
        Unlike training costs, which are incurred once and then amortized, inference costs compound with every user, every request, every product feature that calls the model. A company that ships a model-backed feature at $0.05 per interaction will pay $5,000 per day at 100K daily active users — and $50,000 per day if the feature becomes popular. Getting that number right before scaling is the difference between a business and a burn rate.
      </Prose>

      <H2>The fundamental unit — $ per million tokens</H2>

      <Prose>
        Every inference bill reduces to dollars per million tokens (MTok), tracked separately for input and output. That single quantity is the common denominator across providers, hardware generations, and serving frameworks. Everything else — GPU choice, batch size, model size — is upstream of this number.
      </Prose>

      <MathBlock>{"\\text{\\$ per MTok} = \\frac{\\text{GPU \\$/hr} \\cdot 10^6}{\\text{tokens/sec} \\cdot 3600}"}</MathBlock>

      <Prose>
        Concrete numbers give the formula texture. An H100 runs at roughly $3/hr on a hyperscaler spot market. A well-tuned Llama 3 70B on a single H100, at moderate concurrency with continuous batching, achieves around 2,000 output tokens per second. Plugging in: <Code>3 / (2000 × 3600) × 1e6 ≈ $0.42</Code> per MTok output. The same H100 running Llama 3 8B hits closer to 12,000 tokens per second, yielding roughly $0.06/MTok. At the frontier, GPT-4o's published output rate is around $10/MTok. That 100× spread — from a small self-hosted model to a frontier commercial model — spans most of the market and most of the engineering decisions in this space. Every optimization discussed in this track is an attack on one side of that fraction.
      </Prose>

      <H2>Prefill vs decode economics</H2>

      <Prose>
        The two phases of inference have radically different cost profiles. Prefill — processing the input prompt — is compute-bound. The entire prompt is processed in parallel via matrix multiplications, so it scales roughly linearly with prompt length in wall-clock time, and that cost is amortized across whatever batch is running. A 10,000-token prompt costs roughly 10× a 1,000-token prompt in compute, but many requests can share the same GPU cycles if they arrive together.
      </Prose>

      <Prose>
        Decode — generating each output token one at a time — is memory-bandwidth-bound. Each decode step must load the model weights from HBM for every token generated, and that bandwidth is fixed regardless of batch size up to the point where the KV cache fills. The result: decode throughput is nearly constant per token but total serving capacity depends on how many parallel sequences you can fit in KV cache. Extend the sequence, reduce concurrency. This is why output tokens cost more than input tokens across every commercial API — it is not arbitrary pricing; it is the compute reality. The ratio is typically 3–5× in commercial APIs.
      </Prose>

      <Plot
        label="cost per mtok — input vs output across commercial models (approx 2025 $)"
        width={520}
        height={240}
        xLabel="model tier"
        yLabel="$ per mtok"
        series={[
          { name: "input", points: [[1, 0.15], [2, 1.0], [3, 3.0], [4, 10.0]] },
          { name: "output", points: [[1, 0.60], [2, 3.0], [3, 15.0], [4, 60.0]] },
        ]}
      />

      <Prose>
        The plot above uses approximate 2025-era public pricing for four tiers: small open-weight (e.g., 8B-class), medium (e.g., Sonnet-class), large (e.g., 70B-class or GPT-4o-mini), and frontier (GPT-4o, Claude 3 Opus). The output-to-input cost ratio is remarkably consistent across tiers — roughly 3–6× everywhere — because the physics of decode versus prefill do not change with model size.
      </Prose>

      <H2>What drives cost per token</H2>

      <Prose>
        Three factors drive per-token cost, in rough order of impact.
      </Prose>

      <Prose>
        <strong>Model size (active parameters).</strong> This is the dominant lever. A 70B model requires roughly 10× the memory bandwidth per decode step as a 7B model, because each token generation must load proportionally more weights. At the same throughput target, the 70B model costs close to 10× the 7B in per-token compute. For mixture-of-experts architectures, what matters is active parameters per forward pass, not total parameters — DeepSeek-V3 has 671B total parameters but only 37B active per token, which is why its inference cost is closer to a 37B dense model than a 671B one.
      </Prose>

      <Prose>
        <strong>Context length.</strong> Attention is quadratic in sequence length in theory, though FlashAttention and paged attention implementations reduce the practical scaling to closer to linear in the KV cache read. The dominant cost at long context is actually memory: storing and retrieving the KV cache for a 128K-token context requires roughly 128× the memory of a 1K-token context, which directly limits how many concurrent sequences you can serve. Long-context calls therefore cost disproportionately more than their token count suggests, because they crowd out other requests from the GPU.
      </Prose>

      <Prose>
        <strong>Utilization.</strong> A GPU billing you $3/hr is billing you $3/hr whether it is at 20% compute utilization or 90%. Naive serving — one request at a time, waiting for decode to finish before starting the next prefill — routinely achieves 20–30% GPU utilization. vLLM with continuous batching and prefix caching on a well-matched workload reaches 70–85%. That gap is pure margin. The tokens-per-dollar difference between a well-tuned and a naive serving stack is often 3–4×, with no change to the model.
      </Prose>

      <H2>Caching makes some traffic nearly free</H2>

      <Prose>
        Prefix caching and prompt caching let you skip the prefill cost for any portion of a prompt that is identical to a prior request and still resident in the KV cache. For many production workloads — agent frameworks re-sending large system prompts and tool definitions per step, RAG pipelines with a fixed context preamble, customer-service bots with identical policy documents — a large fraction of every prompt is shared across requests. Caching turns that shared prefix from a repeated cost into a one-time cost.
      </Prose>

      <Prose>
        Commercial APIs reflect this in pricing. Anthropic's cache-hit input tokens are billed at roughly 10% of the fresh input rate. OpenAI's automatic prompt caching applies a 50% discount on cache hits. Google's context caching charges a storage fee plus a fractional use rate. The effective reduction in input cost depends on your hit rate: a workload with 80% cache hits and Anthropic's pricing sees its input cost drop to roughly <Code>0.8 × 0.10 + 0.2 × 1.0 = 0.28×</Code> of baseline — a 3.5× reduction without any change to output cost. For agent workflows generating dozens of tool-call steps per session, this is often the single highest-leverage optimization available.
      </Prose>

      <H3>The cost-scaling flip — smaller models on more hardware</H3>

      <Prose>
        One counterintuitive pattern emerges repeatedly in production: a smaller model running on more concurrent GPU slots is often cheaper than a larger model on fewer slots, at equivalent quality. A Llama 3 8B instance serving 500 concurrent sequences costs less per token than a Llama 3 70B instance serving 50 concurrent sequences, even if the total hardware cost per hour is similar — because the 8B finishes requests faster, recycles the KV cache slots faster, and achieves higher throughput per dollar.
      </Prose>

      <Prose>
        The catch is "at equivalent quality." For many production tasks — classification, extraction, short-form Q&A, code completion in constrained domains — the quality gap between 8B and 70B is small enough that the 8B wins on economics outright. For tasks where the 70B's extra capacity is genuinely load-bearing, you pay for it. This is the economic foundation of model routing: send easy traffic to the cheap model, hard traffic to the capable model, and calibrate the routing threshold to match your quality-cost tradeoff. The economics are compelling enough that routing has become a standard part of inference system design for any high-traffic deployment.
      </Prose>

      <H2>What the inference bill actually decomposes into</H2>

      <Prose>
        A rough cost model is more useful than an intuition. The function below takes a traffic shape — requests per day, average token counts, caching behavior — and returns a monthly cost breakdown. It is the kind of model you run before committing to a model tier or an API provider.
      </Prose>

      <CodeBlock language="python">
{`def inference_bill_month(
    requests_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    input_price_per_mtok: float,
    output_price_per_mtok: float,
    cache_hit_rate_input: float = 0.0,
    cache_hit_price_multiplier: float = 0.10,
):
    """Monthly inference bill for a given traffic shape."""
    monthly_requests = requests_per_day * 30
    fresh_input = avg_input_tokens * (1 - cache_hit_rate_input)
    cached_input = avg_input_tokens * cache_hit_rate_input

    fresh_input_cost = monthly_requests * fresh_input * input_price_per_mtok / 1e6
    cached_input_cost = monthly_requests * cached_input * input_price_per_mtok * cache_hit_price_multiplier / 1e6
    output_cost = monthly_requests * avg_output_tokens * output_price_per_mtok / 1e6

    return {
        "fresh_input": round(fresh_input_cost, 2),
        "cached_input": round(cached_input_cost, 2),
        "output": round(output_cost, 2),
        "total": round(fresh_input_cost + cached_input_cost + output_cost, 2),
    }`}
      </CodeBlock>

      <Prose>
        Running concrete numbers makes the cost structure concrete. A product with 50K requests/day, 2,000 average input tokens, 500 average output tokens, on a mid-tier model at $1/MTok input and $3/MTok output, with no caching: fresh input runs $3,000/month, output runs $2,250/month, total $5,250/month. Introduce a 70% cache hit rate at 10% cache pricing and the input cost drops to $990/month, total $3,240/month — a 38% reduction from one change. Swap the model tier to a small model at $0.15/MTok input and $0.60/MTok output and the total falls to under $700/month. Each of these decisions is independently available; most production systems have room for all three.
      </Prose>

      <H3>Self-hosting vs API</H3>

      <Prose>
        The self-hosting vs. managed API question is really a break-even calculation. Managed APIs have zero upfront cost, no MLOps overhead, immediate access to the latest models, and pricing that reflects commercial margins on top of hardware costs. Self-hosting on reserved or spot GPU capacity eliminates those margins but adds engineering cost, operational complexity, and a minimum viable traffic threshold below which the reserved capacity goes underutilized.
      </Prose>

      <Prose>
        Self-hosting wins when: traffic is high enough and steady enough to keep GPUs above 60% utilization; the workload is on an open-weight model that performs well enough for the task; the team has MLOps capability to manage serving infrastructure and model updates. API wins when: traffic is bursty or unpredictable; you need a model that is not available open-weight; or engineering time is the binding constraint. The rough break-even for a mid-size model on reserved hardware is around 50,000 requests per day at steady, predictable load — below that, API pricing is usually competitive once engineering costs are included.
      </Prose>

      <H3>Cost of serving reasoning models</H3>

      <Prose>
        Reasoning models — o1, o3, R1, and their successors — differ economically from classical chat models in one fundamental way: they generate far more output tokens per query. A classical chat response to a complex question might be 300–500 tokens. The same question routed through a reasoning model generates a chain-of-thought sequence that runs 3,000 to 30,000 tokens before producing the visible answer. The per-token price is often similar to or lower than frontier chat models; the per-query cost is proportionally higher.
      </Prose>

      <Callout accent="gold">
        Reasoning models aren't inherently expensive per token; they're expensive because a single reasoning answer is 10,000 tokens instead of 100. Quality scales with tokens; so does the bill.
      </Callout>

      <Prose>
        This creates a different optimization calculus. For reasoning models, the useful levers are: routing only genuinely hard queries to the reasoning model and easy ones to a cheaper model; capping max output tokens where a shorter chain-of-thought is acceptable; and batching reasoning workloads offline where latency is not a constraint, which enables larger batch sizes and better hardware utilization. The per-token economics still apply — they are just applied to a much larger token budget per query.
      </Prose>

      <Prose>
        It is also worth noting that reasoning model output is largely invisible to the user — the extended thinking traces are internal. That means you are paying for tokens the user never sees. Whether those tokens buy enough quality improvement for the use case is a question that has to be answered empirically, per task, with actual evals, not assumed from model benchmarks alone.
      </Prose>

      <H2>Closing</H2>

      <Prose>
        The economics of inference are the economics of tokens generated. Every technique examined in this track — KV cache management, paged attention, speculative decoding, prefix caching, continuous batching, model routing — is ultimately an attack on one side of the dollars-per-million-tokens equation: either reducing the cost per token through better hardware utilization, or reducing the number of tokens required through smarter serving. The gains stack. A team that applies all of them to a well-chosen model can close most of that 100× gap between naive and optimal inference cost.
      </Prose>

      <Prose>
        The next topic takes a closer look at one of the largest levers on the other side of the equation: test-time compute scaling, where you deliberately generate more tokens — much more — in exchange for meaningfully better answers. Understanding when that tradeoff is worth making is its own discipline, and it starts with understanding what those extra tokens are actually buying.
      </Prose>
    </div>
  ),
};

export default inferenceCostEconomics;
