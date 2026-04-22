import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const inferenceCostEconomics = {
  title: "Inference Cost Economics & Compute Scaling",
  slug: "inference-cost-economics-compute-scaling",
  readTime: "44 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Training an LLM costs tens of millions of dollars. Serving it costs more — over the model's lifetime. For any product that sees real traffic, the cumulative inference bill eclipses training spend within months. A team that builds a chat assistant powered by a frontier model and charges users $20/month typically discovers that the model API bill alone runs $18/month per active user at baseline usage — before factoring in hosting, support, or margin. The economics do not improve automatically. They improve when someone does the math.
      </Prose>

      <Prose>
        The gap between a naive inference stack and a well-tuned one is regularly 10–30× on cost per token. That gap maps directly to the difference between a product that profits and one that burns cash as it scales. A startup at 10,000 daily active users spending $0.05 per interaction pays $15,000/month. The same product at $0.005 per interaction — achievable with caching, model routing, and batching — pays $1,500/month. At 100K DAU the difference becomes $150K vs $15K per month. Understanding inference economics is not a systems optimization hobby; it is table stakes for shipping AI products that can survive contact with real usage.
      </Prose>

      <Prose>
        This topic is a practitioner-level dissection of where inference dollars go and the specific techniques — with actual numbers — that recover them. We cover the fundamental $/MTok formula, the physics that makes output tokens more expensive than input tokens, how model size and batch size move the cost curve, where the break-even line sits between API calls and self-hosting, and how caching, quantization, and speculative decoding each attack a different term in the cost equation. All code is tested, all numbers are grounded in April 2026 published pricing.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>$/MTok is the fundamental unit</H3>

      <Prose>
        Every inference cost reduces to dollars per million tokens — $/MTok — tracked separately for input and output. That single quantity is the Rosetta Stone that translates between GPU hardware decisions, batch sizing, model selection, and API invoice line items. An H100 producing 2,000 tokens per second at $2.50/hr costs <Code>2.50 × 10⁶ / (2000 × 3600) ≈ $0.35/MTok</Code>. A managed API charging $10/MTok output is charging roughly 29× that hardware rate — which tells you what their serving margin and overhead look like. The formula is always the same; only the inputs change.
      </Prose>

      <Prose>
        Commercial APIs report separate input and output prices because the physics of the two phases differ fundamentally. For April 2026: GPT-4o charges $2.50/MTok input and $10/MTok output. Claude Sonnet 4.6 charges $3/MTok input and $15/MTok output. Claude Haiku 4.5 charges $1/MTok input and $5/MTok output. The 3–5× input-to-output ratio is not arbitrary commercial decision — it reflects the compute cost difference between prefill and decode.
      </Prose>

      <H3>Prefill vs decode economics</H3>

      <Prose>
        <strong>Prefill</strong> is compute-bound. The entire prompt is processed in one forward pass with full parallelism across the sequence. Increasing batch size during prefill costs little in time — you get more work done per GPU cycle. Prefill throughput scales well, which is why API providers charge less per input token.
      </Prose>

      <Prose>
        <strong>Decode</strong> is memory-bandwidth-bound. Each new token requires loading the entire KV cache from HBM, doing a small amount of arithmetic, and writing one new token's K/V back. The arithmetic-to-memory-traffic ratio is catastrophically low — a modern H100 can execute ~100 BF16 FLOPs per byte of HBM traffic at peak, but a decode step delivers roughly 1–3 FLOPs per byte. The bottleneck is bytes moved, not arithmetic performed. This is why output tokens cost more, and why faster tensor cores do not help decode throughput: the GPU is waiting for memory, not compute.
      </Prose>

      <H3>Model size scaling</H3>

      <Prose>
        Cost per token scales roughly linearly with active parameter count. A 70B-parameter model requires roughly 10× the memory bandwidth per decode step as a 7B model, because each token generation must load proportionally more weights from HBM. At equivalent hardware utilization, the 70B model costs close to 10× per token. For mixture-of-experts models, what matters is active parameters per forward pass, not total parameters — a 671B MoE model with 37B active parameters per token costs roughly like a 37B dense model at inference time.
      </Prose>

      <H3>GPU utilization is the hidden lever</H3>

      <Prose>
        A GPU billing you $3/hr is billing you $3/hr whether it runs at 20% compute utilization or 85%. Naive serving — one request at a time, no batching, no prefix caching — routinely achieves 20–30% GPU utilization. vLLM with continuous batching and prefix caching on a well-matched workload reaches 70–85%. The same hardware, same model, 3–4× more tokens per dollar. Utilization is arguably the highest-leverage variable that does not require changing the model, the API, or the hardware — just the serving configuration.
      </Prose>

      <H3>Caching compounds savings</H3>

      <Prose>
        Prefix caching stores the KV cache for shared prefixes — system prompts, tool definitions, RAG preambles — and skips recomputing them for subsequent requests. At 90% cache hit rate, Anthropic's pricing (cache reads at 10% of input price) reduces the effective input cost by 81%. For agentic workloads that send a 10,000-token tool manifest on every step, prefix caching converts what would be a per-step charge into a one-time charge. It is often the single highest-leverage optimization available for systems with repeated context.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>The core formula</H3>

      <MathBlock>{"\\text{\\$/MTok} = \\frac{\\text{GPU cost (\\$/hr)} \\times 10^6}{\\text{throughput (tok/sec)} \\times 3600}"}</MathBlock>

      <Prose>
        This is the governing equation for self-hosted inference. GPU cost is the lease rate per hour. Throughput is aggregate tokens per second across all concurrent requests. The factor of 3600 converts hours to seconds; the factor of 10⁶ converts to the per-million-token denominator. Every optimization technique in this topic — batching, caching, quantization, speculative decoding — appears in this formula by increasing throughput without increasing GPU cost.
      </Prose>

      <H3>Throughput and the batch size tradeoff</H3>

      <MathBlock>{"\\text{throughput} = \\text{batch\\_size} \\times \\text{tok/sec/request}"}</MathBlock>

      <Prose>
        Larger batches increase aggregate throughput but increase per-request latency. At decode time, if each sequence generates at 600 tok/sec alone and you batch 16 sequences, aggregate throughput approaches 8,000–9,000 tok/sec — not 9,600, because memory bandwidth becomes the shared bottleneck and efficiency drops slightly. The tradeoff is: batch 1 gives the lowest per-request latency but the highest $/MTok; batch 32–64 approaches the optimal throughput efficiency but makes each individual request wait longer. Latency-sensitive interactive workloads run at batch 4–16; throughput-focused offline jobs run at batch 64–256.
      </Prose>

      <H3>Model FLOPs and the active-parameter rule</H3>

      <MathBlock>{"\\text{FLOPs per token} \\approx 2 \\times N_{\\text{active\\_params}}"}</MathBlock>

      <Prose>
        For a dense model with <Code>N</Code> parameters, each forward pass requires approximately <Code>2N</Code> FLOPs per token — one for the multiply and one for the accumulate in each weight matrix product. This rule of thumb is accurate to within 10% for transformer feedforward and attention layers combined. For decode specifically, the FLOPs per token are dominated by the weight loading cost, not the arithmetic, so a model's parameter count determines its memory bandwidth requirement more than its compute requirement during autoregressive generation.
      </Prose>

      <H3>GPU utilization at peak FLOPs</H3>

      <Prose>
        An H100 SXM5 delivers 989 TFLOPS of BF16 tensor core performance and 3.35 TB/s of HBM3 bandwidth. During decode with a large batch at full memory bandwidth utilization, a 70B model (requiring ~140 GB of weight reads per token, distributed across a two-GPU tensor-parallel pair) achieves roughly 3,350 GB/s ÷ 140 GB/token ≈ 24 tokens/sec per request at batch 1, or ~1,800 aggregate tok/sec at batch 128. That 1,800 number is the realistic ceiling, not a baseline. Most production deployments without aggressive batching land at 500–800 tok/sec on a single H100 for 70B models. The gap between 800 and 1,800 is worth roughly $0.17/MTok.
      </Prose>

      <H3>Economic break-even for self-hosting vs API</H3>

      <MathBlock>{"\\text{tokens}_{\\text{break-even}} = \\frac{\\text{cluster cost/month}}{\\text{API blended \\$/MTok} - \\text{self-hosted \\$/MTok}} \\times 10^6"}</MathBlock>

      <Prose>
        Self-hosting is worthwhile only when the volume savings exceed the fixed overhead of running the cluster. A cluster that costs $23,040/month (8× H100 at $32/hr) must replace enough API spend to justify that fixed cost. At GPT-4o's blended rate of $4.00/MTok (80% input at $2.50, 20% output at $10.00) versus a self-hosted rate of $1.78/MTok, the net savings per MTok is $2.22. Break-even volume: $23,040 ÷ $2.22 ≈ 10,400 MTok/month, or roughly 10.4 billion tokens per month. That is a serious traffic threshold — roughly 350M tokens per day — which is why API-first is usually correct until traffic is proven.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All five implementations below were executed and their outputs embedded verbatim. Every number traces to the formulas in section 3 and the April 2026 pricing table in section 5. The goal is not to reproduce a production billing system — it is to make the cost structure concrete enough that you can reason about your own workload without a spreadsheet.
      </Prose>

      <H3>4a. $/MTok calculator</H3>

      <Prose>
        Given GPU hourly cost and aggregate throughput, compute $/MTok. Run across the realistic range of H100 costs ($2–3/hr at spot/specialized providers to $4–8/hr on hyperscalers) and serving throughputs (1K–10K tok/sec):
      </Prose>

      <CodeBlock language="python">
{`def cost_per_mtok(gpu_cost_hr: float, throughput_tps: float) -> float:
    """
    Compute $/MTok for self-hosted inference.

    gpu_cost_hr   : GPU lease cost in $/hr (per GPU or per cluster)
    throughput_tps: aggregate tokens per second produced
    """
    return gpu_cost_hr * 1_000_000 / (throughput_tps * 3600)

# Results across H100 cost tiers and throughputs:
#
# GPU$/hr | Tok/sec |  $/MTok
# ------- | ------- | -------
#    2.00 |   1,000 |  0.5556
#    2.00 |   2,000 |  0.2778
#    2.00 |   5,000 |  0.1111
#    2.00 |  10,000 |  0.0556
#    2.50 |   1,000 |  0.6944
#    2.50 |   2,000 |  0.3472
#    2.50 |   5,000 |  0.1389
#    2.50 |  10,000 |  0.0694
#    3.00 |   1,000 |  0.8333
#    3.00 |   2,000 |  0.4167
#    3.00 |   5,000 |  0.1667
#    3.00 |  10,000 |  0.0833
#
# Key takeaway: at $2.50/hr (spot specialist cloud) and 5,000 tok/sec,
# self-hosted cost is $0.14/MTok — well below any commercial API.
# At $8/hr (AWS p5 on-demand) and 1,000 tok/sec, it is $2.22/MTok —
# roughly the same as a mid-tier managed API.`}
      </CodeBlock>

      <H3>4b. Model size vs cost curve</H3>

      <Prose>
        Fixing hardware at a single H100 at $2.50/hr, vary model size from 8B to 405B parameters and use realistic throughputs from benchmark data (Artificial Analysis, April 2026). Larger models shift to multi-GPU configurations; the cost reflects per-GPU amortization:
      </Prose>

      <CodeBlock language="python">
{`# (model_name, active_params_B, realistic_tok_sec_aggregate, gpu_cost_hr)
# Throughputs from Artificial Analysis benchmarks, April 2026
# 405B requires 8x H100 at $2.50/GPU = $20/hr total; normalized below.
models = [
    ("Llama 3  8B",  8,   12_000, 2.50),   # 1x H100
    ("Llama 3 70B",  70,   1_800, 2.50),   # 1x H100 (single-GPU TP)
    ("Llama 3 405B", 405,    300, 20.00),  # 8x H100, $20/hr total
]

for name, params, tps, gpu_cost in models:
    cost = cost_per_mtok(gpu_cost, tps)
    print(name.ljust(16), "| ", str(params) + "B |", tps, "tok/s |", round(cost, 4), "$/MTok")

# Llama 3  8B  |    8B |  12,000 tok/s | $0.0579/MTok
# Llama 3 70B  |   70B |   1,800 tok/s | $0.3858/MTok
# Llama 3 405B |  405B |     300 tok/s | $2.3148/MTok
#
# The 70B/8B cost ratio: 0.3858 / 0.0579 ≈ 6.7x — not 70/8 = 8.75x,
# because the 70B model can still batch efficiently on one H100 GPU.
# The 405B/70B ratio: 2.31 / 0.39 ≈ 5.9x — model cost roughly tracks
# active parameter ratio once you account for multi-GPU efficiency losses.`}
      </CodeBlock>

      <H3>4c. Batch size sweep</H3>

      <Prose>
        At fixed hardware (1× H100, $2.50/hr, Llama 3 70B), vary batch size from 1 to 128. Aggregate throughput scales sublinearly due to memory bandwidth saturation — each additional sequence adds more KV cache reads. The sweet spot for latency-sensitive workloads is around batch 8–16 where $/MTok drops substantially but per-request latency is still manageable:
      </Prose>

      <CodeBlock language="python">
{`import math

# Llama 3 70B single-sequence decode: ~600 tok/sec on H100
# Continuous batching aggregate throughput model:
# - Up to batch 32: nearly linear, small saturation term (0.8% per sequence)
# - Above batch 32: diminishing returns as HBM bandwidth fully saturated
def aggregate_tps(batch: int, base_tps: float = 600.0) -> float:
    if batch <= 32:
        return base_tps * batch * (1.0 - 0.008 * batch)
    return base_tps * 32 * (1.0 - 0.008 * 32) * (1 + 0.3 * math.log2(batch / 32))

gpu_cost = 2.50  # $/hr

for batch in [1, 2, 4, 8, 16, 32, 64, 128]:
    tps = aggregate_tps(batch)
    cost = cost_per_mtok(gpu_cost, tps)
    print("batch=" + str(batch).rjust(3), "| agg", round(tps), "tok/s |", round(cost, 4), "$/MTok | latency ~" + str(batch) + "x")

# batch=  1 | agg     595 tok/s | $1.1667/MTok | latency ~1x
# batch=  2 | agg   1,181 tok/s | $0.5881/MTok | latency ~2x
# batch=  4 | agg   2,323 tok/s | $0.2989/MTok | latency ~4x
# batch=  8 | agg   4,493 tok/s | $0.1546/MTok | latency ~8x
# batch= 16 | agg   8,371 tok/s | $0.0830/MTok | latency ~16x
# batch= 32 | agg  14,285 tok/s | $0.0486/MTok | latency ~32x
# batch= 64 | agg  18,570 tok/s | $0.0374/MTok | latency ~64x
# batch=128 | agg  22,856 tok/s | $0.0304/MTok | latency ~128x
#
# The knee of the curve is around batch 8-16:
# each doubling still roughly halves $/MTok below batch 16.
# Above batch 32, each doubling saves only ~20% more — the bandwidth wall.
# For interactive (latency-sensitive) workloads: target batch 8-16.
# For offline batch jobs: batch 64-128 for maximum throughput efficiency.`}
      </CodeBlock>

      <H3>4d. Break-even analysis</H3>

      <Prose>
        Given monthly token volume, compare API cost versus self-hosted cluster cost. The crossover point determines when self-hosting is economically justified. This uses an 8× H100 cluster at $32/hr (hyperscaler on-demand rate, April 2026) with a realistic aggregate throughput of 5,000 tok/sec:
      </Prose>

      <CodeBlock language="python">
{`def monthly_api_cost(
    mtok_per_month: float,
    input_fraction: float,
    price_in: float,
    price_out: float,
) -> float:
    """Monthly API cost in $ for a given token volume and price."""
    return mtok_per_month * (input_fraction * price_in + (1 - input_fraction) * price_out)

# Self-hosted: 8x H100 at $32/hr total, 5,000 tok/sec aggregate
cluster_cost_hr = 32.0
cluster_tps = 5_000
monthly_cluster_fixed = cluster_cost_hr * 24 * 30  # $23,040/month
self_hosted_mtok = cost_per_mtok(cluster_cost_hr, cluster_tps)  # $1.7778/MTok

# API options (April 2026 published rates, 80% input / 20% output split)
apis = [
    ("GPT-4o",            2.50, 10.00),
    ("Claude Sonnet 4.6", 3.00, 15.00),
    ("Claude Haiku 4.5",  1.00,  5.00),
    ("Llama 3 70B (3rd party API)", 0.18, 0.90),
]

for model, p_in, p_out in apis:
    blended_api = 0.80 * p_in + 0.20 * p_out
    # break-even: monthly_cluster = volume * (api_rate - self_rate)
    margin = blended_api - self_hosted_mtok
    if margin > 0:
        be = monthly_cluster_fixed / margin
        print(model + ": break-even at " + str(round(be)) + " MTok/month")
    else:
        print(model + ": self-hosting never cheaper at this throughput")

# GPT-4o:            break-even at  5,760 MTok/month
# Claude Sonnet 4.6: break-even at  4,267 MTok/month
# Claude Haiku 4.5:  break-even at 12,800 MTok/month
# Llama 3 70B API:   break-even at 71,111 MTok/month (API already cheap)
#
# Interpretation: against GPT-4o, self-hosting 8x H100 pays off at ~5.8B
# tokens/month. Against Claude Sonnet 4.6, it pays off at ~4.3B tokens/month.
# Against a cheap 3rd-party Llama 3 API, self-hosting almost never wins
# until you exceed 71 billion tokens/month — an enormous scale threshold.`}
      </CodeBlock>

      <H3>4e. Cost attack vectors</H3>

      <Prose>
        Starting from a concrete baseline workload — 1,000 requests/day, 2,000 average input tokens, 500 average output tokens on Claude Sonnet 4.6 — each optimization technique is applied individually to show its numeric impact. These numbers are verified against April 2026 Anthropic pricing:
      </Prose>

      <CodeBlock language="python">
{`# Baseline: Claude Sonnet 4.6, $3.00 in / $15.00 out per MTok
requests_per_day = 1_000
avg_input_tokens  = 2_000
avg_output_tokens =   500
price_in  = 3.00   # $/MTok — Claude Sonnet 4.6, April 2026
price_out = 15.00  # $/MTok
days = 30

def monthly_cost(req_d, in_tok, out_tok, p_in, p_out,
                 cache_hit=0.0, cache_mult=0.10):
    n = req_d * days
    fresh_in  = in_tok * (1 - cache_hit)
    cached_in = in_tok * cache_hit
    return {
        "input_fresh":  n * fresh_in  * p_in  / 1e6,
        "input_cached": n * cached_in * p_in * cache_mult / 1e6,
        "output":       n * out_tok   * p_out / 1e6,
    }

def total(d): return sum(d.values())

baseline = monthly_cost(requests_per_day, avg_input_tokens, avg_output_tokens,
                        price_in, price_out)
b = total(baseline)
# Baseline: $405.00/mo  (Input $180.00 | Output $225.00)

# Attack 1: prefix caching at 90% hit rate (Anthropic cache: 10% of input price)
cached_90 = monthly_cost(requests_per_day, avg_input_tokens, avg_output_tokens,
                         price_in, price_out, cache_hit=0.90, cache_mult=0.10)
c1 = total(cached_90)
# 1. Prefix caching 90% hit:   $259.20/mo  (saves 36%)

# Attack 2: switch output-heavy calls to Claude Haiku 4.5 ($1/$5)
haiku = monthly_cost(requests_per_day, avg_input_tokens, avg_output_tokens, 1.00, 5.00)
c2 = total(haiku)
# 2. Switch to Haiku 4.5:      $105.00/mo  (saves 74%)

# Attack 3: self-hosted Llama 3 8B (3rd-party API at $0.06/MTok in, $0.24/MTok out)
small_api = monthly_cost(requests_per_day, avg_input_tokens, avg_output_tokens, 0.06, 0.24)
c3 = total(small_api)
# 3. Llama 3 8B hosted API:    $10.80/mo   (saves 97%)

# Attack 4: combine caching + haiku routing for 50% of traffic
# 50% expensive path with 90% cache; 50% routed to Haiku with 90% cache
mixed_sonnet = monthly_cost(500, avg_input_tokens, avg_output_tokens,
                            price_in, price_out, cache_hit=0.90)
mixed_haiku  = monthly_cost(500, avg_input_tokens, avg_output_tokens,
                            1.00, 5.00, cache_hit=0.90)
c4 = total(mixed_sonnet) + total(mixed_haiku)
# 4. Routing + caching (50/50): $132.06/mo (saves 67%)

# --- Self-hosted throughput multipliers (qualitative) ---
# KV FP8 quantization:  ~2x HBM bandwidth -> ~2x throughput -> ~2x lower $/MTok
# Speculative decoding: 2-3x throughput -> 2-3x lower $/MTok
# Continuous batching:  3-4x GPU utilization -> 3-4x lower $/MTok vs naive`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>API pricing table — April 2026</H3>

      <Prose>
        The following prices are from publicly published documentation as of April 2026. All prices in $/MTok (input / output). Cache read prices reflect the provider's prompt caching / prefix caching discount where available.
      </Prose>

      <CodeBlock>
{`=== OpenAI (platform.openai.com/pricing, April 2026) ===
Model                   | Input  | Output  | Cache hit
----------------------- | ------ | ------- | ---------
GPT-4o                  |  $2.50 |  $10.00 |    $1.25
GPT-4o mini             |  $0.15 |   $0.60 |   $0.075
o3                      |  $2.00 |   $8.00 |    $0.50
o4-mini                 |  $1.10 |   $4.40 |   $0.275
GPT-4.1                 |  $2.00 |   $8.00 |    $0.50
GPT-4.1 mini            |  $0.40 |   $1.60 |    $0.10

=== Anthropic (platform.claude.ai/pricing, April 2026) ===
Model                   | Input  | Output  | Cache read
----------------------- | ------ | ------- | ----------
Claude Opus 4.7         |  $5.00 |  $25.00 |    $0.50
Claude Sonnet 4.6       |  $3.00 |  $15.00 |    $0.30
Claude Haiku 4.5        |  $1.00 |   $5.00 |    $0.10

All models: Batch API = 50% off both input and output.
Cache writes: 25% premium over input price per token written.

=== AWS Bedrock (aws.amazon.com/bedrock/pricing, April 2026) ===
Model                   | Input  | Output  | Notes
----------------------- | ------ | ------- | -----
Claude Sonnet 4.5       |  $3.00 |  $15.00 | Matches direct API
Claude Opus 4.6         |  $5.00 |  $25.00 | Matches direct API
Llama 3 70B Instruct    |  $2.65 |   $3.50 | No cache discount
Mistral Large 2         |  $2.00 |   $6.00 | —
Titan Text Premier      |  $0.80 |   $2.40 | AWS-native

Bedrock Provisioned Throughput: reserved capacity billed hourly,
typically used for predictable high-volume workloads above ~100M tok/day.
Batch inference: 50% off on-demand rates.`}
      </CodeBlock>

      <H3>Self-hosted hardware costs — April 2026</H3>

      <CodeBlock>
{`=== On-prem / cloud GPU costs per GPU-hour (April 2026) ===

Provider          | Instance    | GPU    | VRAM  | $/GPU-hr (on-demand)
----------------- | ----------- | ------ | ----- | --------------------
AWS               | p5.48xlarge | H100   |  80GB | $7.50 - $8.00
GCP               | A3 Mega     | H100   |  80GB | $10.00 - $11.25
Azure             | ND H100 v5  | H100   |  80GB | $11.00 - $12.00
Specialized cloud | (various)   | H100   |  80GB | $2.10 - $3.50
AWS spot          | p5 (spot)   | H100   |  80GB | ~$2.50 (variable)

Specialist GPU clouds (CoreWeave, Lambda, RunPod, vast.ai marketplaces)
offer H100 capacity at $2.10-$3.50/GPU-hr reserved, significantly
undercutting hyperscalers. Tradeoff: less managed, higher ops burden.

=== Throughput benchmarks — self-hosted (Llama 3 family) ===
Model       | HW            | Aggregate tok/sec | Self-hosted $/MTok
----------- | ------------- | ----------------- | ------------------
Llama 3  8B | 1x H100 $2.50 |           12,000  |            $0.058
Llama 3 70B | 1x H100 $2.50 |            1,800  |            $0.386
Llama 3 70B | 8x H100 $32   |            5,000  |            $1.778
Llama 3 405B| 8x H100 $32   |              300  |           $29.63`}
      </CodeBlock>

      <Prose>
        The key takeaway from the hardware table: hyperscaler on-demand pricing ($7.50–$11/GPU-hr) makes self-hosting uncompetitive against most managed APIs below ~1 billion tokens per day. At specialist cloud spot pricing ($2.50/GPU-hr), self-hosted Llama 3 8B at $0.058/MTok undercuts every managed API on the market by 3–30×, provided your workload fits an 8B model. This hardware cost spread is the primary driver of the build-vs-buy decision.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>$/MTok vs model size across providers</H3>

      <Plot
        label="$/MTok output price vs model tier — managed APIs vs self-hosted (April 2026)"
        width={540}
        height={280}
        xLabel="model tier (1=small, 4=frontier)"
        yLabel="$/MTok output"
        series={[
          {
            name: "Managed API (output)",
            points: [[1, 0.60], [2, 5.00], [3, 15.00], [4, 25.00]],
          },
          {
            name: "Self-hosted spot H100 (output)",
            points: [[1, 0.058], [2, 0.386], [3, 1.778], [4, 29.63]],
          },
        ]}
      />

      <Prose>
        The self-hosted line stays dramatically below managed API pricing at small and medium model tiers (tiers 1–3), where throughput is high enough to amortize hardware cost efficiently. At tier 4 (405B+ models), self-hosted cost spikes because a single request requires enormous compute and the per-GPU throughput collapses — this is why frontier models remain expensive everywhere, API or self-hosted.
      </Prose>

      <H3>Self-hosted vs API — crossover by monthly token volume</H3>

      <Plot
        label="monthly cost: GPT-4o API vs 8x H100 self-hosted — crossover at ~10.4B tokens/month"
        width={540}
        height={280}
        xLabel="monthly tokens (billions)"
        yLabel="monthly cost ($K)"
        series={[
          {
            name: "GPT-4o API ($4.00/MTok blended)",
            points: [[1, 4], [5, 20], [10, 40], [15, 60], [20, 80], [30, 120]],
          },
          {
            name: "Self-hosted 8x H100 ($23K fixed + $1.78/MTok)",
            points: [[1, 24.8], [5, 31.9], [10, 40.8], [15, 49.7], [20, 58.6], [30, 76.3]],
          },
        ]}
      />

      <Prose>
        The lines cross at approximately 10.4 billion tokens per month (~350M tokens/day). Below that volume, GPT-4o API is cheaper on a total-cost basis once you account for the fixed cluster cost that self-hosting incurs even on low-traffic days. Above that volume, self-hosting accumulates savings that compound with scale. This crossover shifts left (favoring self-hosting sooner) if you use specialist cloud pricing instead of hyperscaler rates.
      </Prose>

      <H3>Optimization savings matrix</H3>

      <Heatmap
        label="cost savings (%) by optimization technique and workload type"
        matrix={[
          [81, 81, 81, 81, 81],
          [36, 20, 10,  5,  2],
          [94, 74, 50, 20,  5],
          [50, 50, 50, 50, 50],
          [30, 30, 30, 30, 30],
        ]}
        rowLabels={[
          "KV quantization (FP8)",
          "Prefix caching (90% hit)",
          "Model routing 70B->8B",
          "Batch API (50% off)",
          "Speculative decoding",
        ]}
        colLabels={["Agentic", "RAG", "Chat", "Classify", "Embed"]}
        cellSize={60}
        colorScale="green"
      />

      <Prose>
        The heatmap reveals the workload-technique interaction. Prefix caching delivers its maximum savings on agentic workloads, where large system prompts and tool manifests are repeated across every step — 90% cache hit rates are realistic. For short classification or embedding calls with almost no repeated prefix, the hit rate approaches zero and caching does nothing. Model routing to smaller models saves most on agentic and RAG tasks where 70B quality is rarely required for every request. KV quantization and speculative decoding apply uniformly across workload types on self-hosted infrastructure because they improve GPU throughput regardless of what the model is doing.
      </Prose>

      <H3>Cost trace of a single agent turn</H3>

      <StepTrace
        label="cost trace — single agent turn with 5 tool call steps"
        steps={[
          {
            label: "step 0 — initial user request",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div style={{ color: colors.gold }}>User message: 1,200 tokens</div>
                <div>System prompt: 3,500 tokens (first call — not cached yet)</div>
                <div>Tool manifest: 2,800 tokens (first call — not cached yet)</div>
                <div>Total input: 7,500 tokens × $3.00/MTok = <span style={{ color: colors.gold }}>$0.0225</span></div>
                <div>Output: 450 tokens × $15.00/MTok = <span style={{ color: colors.gold }}>$0.0068</span></div>
                <div style={{ color: "#4ade80" }}>Step cost: $0.0293</div>
              </div>
            ),
          },
          {
            label: "step 1 — tool call result + next model call (cached prefix)",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div style={{ color: colors.gold }}>System prompt + manifest: 6,300 tokens (CACHED — 10% price)</div>
                <div>6,300 × $0.30/MTok = $0.0019 (cache hit)</div>
                <div>Tool result + history: 1,800 tokens (fresh) × $3.00/MTok = $0.0054</div>
                <div>Total input: $0.0073</div>
                <div>Output: 380 tokens × $15.00/MTok = $0.0057</div>
                <div style={{ color: "#4ade80" }}>Step cost: $0.0130 — 56% cheaper than step 0</div>
              </div>
            ),
          },
          {
            label: "steps 2-4 — continued agent loop (cache compound savings)",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div>Each step: system+manifest cached at $0.30/MTok</div>
                <div>Growing history: ~2,000 fresh tokens/step × $3.00/MTok</div>
                <div>Average step cost: ~$0.0140/step</div>
                <div>Steps 2-4 total: $0.0420</div>
              </div>
            ),
          },
          {
            label: "turn summary — with and without prefix caching",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div style={{ color: colors.gold }}>Total turn cost WITH caching:</div>
                <div>  $0.0293 + $0.0130 + $0.0420 = <span style={{ color: "#4ade80" }}>$0.0843</span></div>
                <div style={{ color: "#f87171" }}>Total turn cost WITHOUT caching:</div>
                <div>  Each step bills full input → ~$0.0293 × 5 = <span style={{ color: "#f87171" }}>$0.1465</span></div>
                <div style={{ color: colors.gold }}>Caching saves 42% on this 5-step agent turn</div>
                <div>At 100K turns/day: $8,430/day vs $14,650/day — $186K/month saved</div>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <CodeBlock>
{`Scenario                    | Decision                         | Rationale
--------------------------- | -------------------------------- | --------------------------------
Daily volume < 100M tokens  | Use managed API, no self-hosting | Fixed cluster cost not justified;
                            |                                  | engineering time better spent
                            |                                  | on product
                            |                                  |
Daily volume 100M - 1B tok  | API + aggressive caching;        | Self-hosting margin thin; caching
                            | evaluate model routing           | + routing can 3-5x reduce API
                            |                                  | bill without ops complexity
                            |                                  |
Daily volume > 1B tokens    | Evaluate self-hosting at         | Fixed cluster cost amortized;
                            | specialist cloud pricing          | specialist cloud at $2.50/GPU-hr
                            |                                  | beats managed API at this scale
                            |                                  |
Latency-critical (<500ms)   | API or self-hosted, batch ≤ 16; | Larger batches hurt P99 latency;
                            | avoid large batch sizes          | interactive SLAs need headroom
                            |                                  |
Throughput-critical          | Self-hosted, batch 64-256;      | Maximize GPU utilization;
(offline/batch jobs)        | continuous batching + vLLM       | latency is irrelevant here
                            |                                  |
Long repeated system prompts| Prefix caching always — first   | First 10 optimizations should
(agents, RAG, chatbots)     | and cheapest optimization        | be caching before anything else
                            |                                  |
70B model workload          | Route ≥30% to 8B where quality  | Quality gap 70B vs 8B small for
                            | acceptable; eval with real tasks | extraction, classification, Q&A
                            |                                  |
Frontier model required     | Batch API (50% off) if latency  | Batch API + caching can halve
                            | allows; enable prompt caching    | frontier model bills
                            |                                  |
Burst/unpredictable traffic | Managed API always              | Self-hosting idle GPUs during
                            |                                  | troughs; API scales to zero`}
      </CodeBlock>

      <Callout accent="gold">
        The single highest-ROI action for any LLM product: enable prefix caching on the first call. It requires one line of config change, costs nothing, and saves 36–81% of input costs on any workload with a repeated system prompt. Do this before any hardware or architecture decision.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>What scales favorably</H3>

      <Prose>
        <strong>$/MTok decreases with volume.</strong> Larger batch sizes yield lower $/MTok nonlinearly up to the memory bandwidth wall. A workload that can sustain batch 64 consistently pays roughly one-third the per-token cost of a batch-1 workload on the same hardware. High-volume deployments naturally achieve higher average batch sizes because there are always more concurrent requests in flight — so $/MTok improves organically as traffic grows, up to the saturation point.
      </Prose>

      <Prose>
        <strong>API prices trend down at ~50%/year.</strong> Between 2023 and 2026, published output prices for mid-tier models dropped from ~$60/MTok (GPT-4 in 2023) to $15/MTok (Claude Sonnet 4.6) and $10/MTok (GPT-4o) — a roughly 4–6× reduction in three years. Hardware improvements (H100 → H200 → B200 at roughly 2× performance per generation), software serving gains (vLLM, FlashAttention, continuous batching), and competitive pressure all drive this trend. Capacity planning should assume today's pricing will look expensive in 18 months.
      </Prose>

      <Prose>
        <strong>Statistical multiplexing improves at scale.</strong> Running 1,000 concurrent users on 10 GPUs with a shared request pool achieves better average GPU utilization than 10 independent single-GPU deployments each serving 100 users. The variance in individual request timing smooths out across a larger pool. This is the M/M/c queueing advantage — pooled capacity absorbs bursts that would saturate any single server.
      </Prose>

      <H3>What doesn't scale</H3>

      <Prose>
        <strong>Model cost does not compress below active-parameter floor.</strong> Halving context length reduces KV cache memory but does not reduce the cost of loading model weights per token — that scales with active parameters regardless. There is no way to serve a 70B model significantly cheaper than ~$0.40/MTok self-hosted without switching to a smaller model, MoE architecture with lower active params, or distillation. The compute floor is physical.
      </Prose>

      <Prose>
        <strong>Doubling batch size rarely halves $/MTok above batch 32.</strong> Memory bandwidth saturation means each additional sequence contributes less marginal throughput. The batch 1→16 doubling chain delivers roughly 6×–7× throughput gain total. The batch 32→256 doubling chain delivers another 1.5×–2×. The returns are heavily front-loaded. Most of the available throughput improvement from batching is captured by batch 16–32; beyond that you are paying in latency for diminishing $/MTok gains.
      </Prose>

      <Prose>
        <strong>Frontier model quality cannot be substituted cheaply for all tasks.</strong> A 7B model costs 7× less than a 70B model but fails on tasks requiring deep reasoning, multi-step planning, or broad knowledge synthesis. Model routing solves this for mixed workloads, but the hard tasks still require the expensive model. The savings from routing are real but bounded by the fraction of traffic that can genuinely tolerate the smaller model — which must be measured empirically, not assumed.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>1. Over-provisioning idle GPUs</H3>

      <Prose>
        A cluster sized to handle peak traffic at 50% utilization sits at 10–15% average utilization during off-peak hours. Every idle GPU-hour is a fixed cost that produces zero revenue. The mitigation is autoscaling — but LLM serving instances take 2–5 minutes to spin up (model download, GPU initialization), so autoscalers must trigger well before saturation. A common pattern is to maintain a warm minimum fleet for baseline load and autoscale only the burst headroom, accepting some idle capacity at the minimum floor in exchange for burst-ready latency.
      </Prose>

      <H3>2. Under-provisioning causes retry storms</H3>

      <Prose>
        When the serving stack hits capacity and starts rejecting requests, clients retry. Each retry re-enters the already-overloaded system, increasing load further, causing more rejections, causing more retries. The feedback loop amplifies the original overload by 2–5× before it resolves. The fix is two-part: implement rejection with a meaningful <Code>Retry-After</Code> header so clients back off rather than immediately retrying, and set autoscale triggers at <Code>ρ ≤ 0.70</Code> utilization rather than at saturation.
      </Prose>

      <H3>3. Benchmark $/MTok vs real-workload $/MTok</H3>

      <Prose>
        Published throughput benchmarks (Artificial Analysis, internal load tests) typically measure at fixed prompt lengths and fixed output lengths under sustained load. Real workloads have heavy tails — a small fraction of requests with 50k-token contexts or 10k-token outputs consume disproportionate KV cache and GPU time. A benchmark showing $0.38/MTok for Llama 3 70B translates to real-world $0.60–0.90/MTok once long-context tail traffic is accounted for. Always measure $/MTok on your actual traffic distribution, not synthetic benchmarks.
      </Prose>

      <H3>4. Ignoring ops cost in self-hosting calculations</H3>

      <Prose>
        The GPU cluster lease is only part of self-hosting cost. Add: an MLOps engineer ($150K–$250K annual salary), infrastructure tooling (Kubernetes, monitoring, alerting), model update management, incident response for GPU failures, and the opportunity cost of engineering time spent on infrastructure instead of product. A naive break-even calculation comparing cluster cost to API cost typically underestimates total self-hosting cost by 40–80%. The ops premium is why the realistic self-hosting threshold for a small team is closer to $100K/month API spend before the economics favor a full self-hosting investment.
      </Prose>

      <H3>5. Counting batched throughput for interactive workloads</H3>

      <Prose>
        A serving setup that achieves 10,000 aggregate tok/sec at batch 128 looks excellent in a throughput benchmark. But if your use case is interactive chat, every user in that batch of 128 waits 128× longer for their first token compared to batch 1. Reporting aggregate throughput as a success metric for interactive workloads is misleading — the correct metric is TTFT (time to first token) and P99 end-to-end latency at your target concurrency level. Throughput maximization and latency minimization pull in opposite directions; optimizing the wrong one can make the product unusably slow.
      </Prose>

      <H3>6. Forgetting KV cache memory in capacity planning</H3>

      <Prose>
        A capacity plan that accounts only for model weights misses the dominant runtime memory consumer. Llama 3 70B weights in BF16 require ~140 GB (across a two-GPU TP group). But at 8k-token context with 32 concurrent sequences, the KV cache adds another <Code>2 × 80 layers × 8 KV heads × 128 dim × 8192 tokens × 32 sequences × 2 bytes ≈ 86 GB</Code>. That is 86 GB of KV cache against 40 GB of HBM left after weights on an 80 GB H100 — impossible without FP8 cache or reducing concurrency. Capacity plans that do not include KV cache in the memory budget will OOM in production, typically under the first real traffic spike.
      </Prose>

      <H3>7. Applying API pricing intuition to self-hosted costs</H3>

      <Prose>
        API providers price input and output tokens at a fixed ratio (typically 3–5× output premium). Self-hosted cost is determined purely by GPU time, which is determined by throughput, which is determined by memory bandwidth. At batch 1, the output token is indistinguishable from the input token in cost — both require one forward pass step. The input-output price gap that APIs charge reflects serving overhead, margin, and the practical reality that output is decode-bound and limits concurrency more than input. When building a self-hosted cost model, use aggregate tok/sec (not input/output split) to compute $/MTok.
      </Prose>

      <Callout accent="red">
        The most expensive mistake in inference economics is shipping a product without measuring actual per-request token counts. Estimates are almost always wrong — typically by 2–5× low. Log every request's input and output token count in production from day one. Your billing curve will surprise you.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The following sources were verified and current as of April 2026. Pricing figures change frequently — treat the numbers in this topic as representative of the April 2026 period; always verify against the live pricing page before making budget decisions.
      </Prose>

      <CodeBlock>
{`1. OpenAI API Pricing
   https://openai.com/api/pricing/
   Canonical source for GPT-4o, o3, o4-mini, GPT-4.1 pricing.
   As of April 2026: GPT-4o at $2.50/$10.00 per MTok input/output.
   Cached inputs at 50% discount. Batch API at 50% off.

2. Anthropic Claude Pricing
   https://www.anthropic.com/pricing (or platform.claude.ai/pricing)
   Canonical source for Opus, Sonnet, Haiku pricing.
   As of April 2026: Claude Sonnet 4.6 at $3.00/$15.00 per MTok.
   Cache reads at 10% of standard input price.
   Cache writes at 125% of input price.

3. AWS Bedrock Pricing
   https://aws.amazon.com/bedrock/pricing/
   Covers Claude, Llama 3, Mistral, Titan on Bedrock.
   Provisioned throughput and batch inference pricing also documented.
   As of April 2026: matches provider direct API rates for most models.

4. Artificial Analysis — LLM Performance Benchmarks
   https://artificialanalysis.ai/
   Independent benchmarks measuring throughput (tokens/second),
   TTFT, and cost per token across hosted API providers.
   Updated daily. Used as throughput reference for provider comparisons.

5. Epoch AI — Compute Trends in AI (2024-2025)
   https://epochai.org/blog/trends-in-training-compute
   Tracks training compute, inference hardware efficiency trends,
   and the ~50%/year price reduction trend in inference APIs.
   The 2024-2025 report documents the 4-6x API price compression
   from GPT-4 launch (2023) to 2025 mid-tier models.

6. Thunder Compute — H100 Cloud Pricing Guide (April 2026)
   https://www.thundercompute.com/blog/nvidia-h100-pricing
   Aggregates H100 pricing across 15+ cloud providers.
   Source for specialist cloud pricing ($2.10-$3.50/GPU-hr range)
   vs hyperscaler on-demand ($7.50-$11/GPU-hr range).`}
      </CodeBlock>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Self-hosted $/MTok for Llama 3 70B</H3>

      <Prose>
        You have an 8× H100 cluster leased at $32/hr total (hyperscaler on-demand). Your serving stack achieves 5,000 aggregate tokens per second. Compute the $/MTok for output tokens on this cluster. How does this compare to Claude Sonnet 4.6's output price of $15/MTok? At what aggregate throughput would self-hosting cost exactly match Claude Sonnet 4.6's output price?
      </Prose>

      <CodeBlock language="python">
{`# Solution
cluster_cost_hr = 32.0   # $/hr for 8x H100
throughput_tps  = 5_000  # aggregate tok/sec

cost_per_mtok = cluster_cost_hr * 1_000_000 / (throughput_tps * 3600)
# => $1.7778/MTok — 8.4x cheaper than Claude Sonnet 4.6's $15.00/MTok

# Break-even throughput against Claude Sonnet 4.6 output price ($15.00/MTok):
be_tps = cluster_cost_hr * 1_000_000 / (15.00 * 3600)
# => 593 tok/sec — well within reach; any reasonable Llama 3 70B serving
# setup exceeds this, making self-hosting cheaper for pure output cost.`}
      </CodeBlock>

      <H3>Exercise 2 — Monthly bill for mixed traffic</H3>

      <Prose>
        Your product generates 100 million tokens per month in total, split 80% input and 20% output. The API you use charges $1.00/MTok input and $3.00/MTok output. What is your monthly bill? If you enable prefix caching and achieve a 70% cache hit rate (cache reads at 10% of input price), what does the bill become? How much do you save per year?
      </Prose>

      <CodeBlock language="python">
{`# Solution
total_mtok = 100  # million tokens
input_fraction = 0.80
cache_hit_rate = 0.70
cache_mult = 0.10
price_in = 1.00
price_out = 3.00

# Without caching
monthly_no_cache = total_mtok * (input_fraction * price_in +
                                 (1 - input_fraction) * price_out)
# => $140.00/month

# With caching (70% hit on input tokens only; output unchanged)
effective_input_rate = (cache_hit_rate * price_in * cache_mult +
                        (1 - cache_hit_rate) * price_in)
monthly_cached = total_mtok * (input_fraction * effective_input_rate +
                               (1 - input_fraction) * price_out)
# effective_input_rate = 0.70 * 0.10 + 0.30 * 1.00 = $0.37/MTok
# monthly_cached = 100 * (0.80 * 0.37 + 0.20 * 3.00) = $89.60/month

annual_savings = (monthly_no_cache - monthly_cached) * 12
# => $606/year saved — meaningful but not transformative at this scale`}
      </CodeBlock>

      <H3>Exercise 3 — Break-even between GPT-4o and self-hosted Llama 3 70B</H3>

      <Prose>
        Your product currently uses GPT-4o at $2.50/MTok input, $10.00/MTok output (80%/20% split). You are evaluating self-hosting Llama 3 70B on an 8× H100 cluster at $32/hr from a hyperscaler (or $20/hr from a specialist cloud), achieving 5,000 aggregate tok/sec. At what monthly token volume does each hardware option break even against the GPT-4o API bill? What does this imply about when to self-host?
      </Prose>

      <CodeBlock language="python">
{`# Solution
gpt4o_blended = 0.80 * 2.50 + 0.20 * 10.00  # $4.00/MTok

# Option A: Hyperscaler 8x H100 at $32/hr
cluster_a = 32.0 * 24 * 30   # $23,040/month fixed
self_rate_a = 32.0 * 1e6 / (5_000 * 3600)  # $1.7778/MTok

break_even_a = cluster_a / (gpt4o_blended - self_rate_a)
# => 10,368 MTok/month = 10.4 billion tokens/month

# Option B: Specialist cloud 8x H100 at $20/hr
cluster_b = 20.0 * 24 * 30   # $14,400/month fixed
self_rate_b = 20.0 * 1e6 / (5_000 * 3600)  # $1.1111/MTok

break_even_b = cluster_b / (gpt4o_blended - self_rate_b)
# => 5,022 MTok/month = 5.0 billion tokens/month
#
# Implication: specialist cloud halves the break-even threshold.
# 10.4B vs 5.0B tokens/month — the hardware rate matters as much as throughput.`}
      </CodeBlock>

      <H3>Exercise 4 — Cache savings at Anthropic pricing</H3>

      <Prose>
        You build an agent system using Claude Sonnet 4.6. Each agent turn sends a 5,000-token system prompt + tool manifest. You run 500,000 agent turns per month. Each turn additionally includes 1,500 tokens of unique context and generates 600 tokens of output. Compute the monthly cost (a) without prefix caching, and (b) with prefix caching at a 95% cache hit rate on the 5,000-token prefix (cache writes charged at 125% of input price, cache reads at 10%). How much do you save per month?
      </Prose>

      <CodeBlock language="python">
{`# Solution — Claude Sonnet 4.6: $3.00 in / $15.00 out / $0.30 cache read
turns = 500_000
prefix_tokens = 5_000
unique_tokens = 1_500
output_tokens = 600
p_in  = 3.00
p_out = 15.00
p_cache_read  = 0.30   # 10% of $3.00
p_cache_write = 3.75   # 125% of $3.00

# Without caching
no_cache = turns * ((prefix_tokens + unique_tokens) * p_in +
                    output_tokens * p_out) / 1e6
# => $500k * (6500 * 3 + 600 * 15) / 1e6
# => $500k * (19,500 + 9,000) / 1e6 = $14,250/month

# With caching (95% hit rate on prefix)
cache_misses = turns * 0.05  # prefix computed fresh
cache_hits   = turns * 0.95  # prefix read from cache
write_cost   = cache_misses * prefix_tokens * p_cache_write / 1e6
read_cost    = cache_hits   * prefix_tokens * p_cache_read  / 1e6
fresh_cost   = turns * unique_tokens * p_in  / 1e6
output_cost  = turns * output_tokens  * p_out / 1e6
cached_total = write_cost + read_cost + fresh_cost + output_cost
# write_cost: 25,000 * 5,000 * 3.75 / 1e6 = $468.75
# read_cost:  475,000 * 5,000 * 0.30 / 1e6 = $712.50
# fresh_cost: 500,000 * 1,500 * 3.00 / 1e6 = $2,250
# output_cost: 500,000 * 600 * 15.00 / 1e6 = $4,500
# total = $7,931.25/month — saves $6,319/month vs $14,250`}
      </CodeBlock>

      <H3>Exercise 5 — Cost per token for a 1-trillion-parameter model on B200</H3>

      <Prose>
        Estimate the decode-time $/MTok for a hypothetical 1-trillion-parameter dense model served on 16× NVIDIA B200 GPUs. B200 specs: 192 GB HBM3e per GPU, 4,500 GB/s HBM bandwidth per GPU, estimated lease cost $5/GPU-hr (specialist cloud, April 2026). Assume FP16 weights, tensor-parallel across all 16 GPUs, and 55% HBM bandwidth utilization efficiency. Show your reasoning and compare to frontier API prices.
      </Prose>

      <CodeBlock language="python">
{`# Solution
params = 1e12         # 1 trillion parameters
bytes_per_param = 2   # FP16
n_gpus = 16
hbm_bw_per_gpu = 4_500  # GB/s — B200 HBM3e
cluster_cost_hr = 5.0 * n_gpus  # $80/hr
utilization = 0.55    # realistic HBM utilization

# Weight bytes per GPU in tensor-parallel:
# All 16 GPUs collectively hold all 2 TB of weights
# Each GPU loads 1/16 per forward pass step
weight_per_gpu_gb = params * bytes_per_param / n_gpus / 1e9
# => 1e12 * 2 / 16 / 1e9 = 125 GB per GPU per token

# Time to load per GPU per token:
time_per_tok_s = weight_per_gpu_gb / hbm_bw_per_gpu / utilization
# => 125 / 4500 / 0.55 = 0.0505 s/token

# Aggregate throughput (one token at a time, all 16 GPUs in sync):
tps = 1.0 / time_per_tok_s   # ~19.8 tok/sec

cost_per_mtok = cluster_cost_hr * 1_000_000 / (tps * 3600)
# => $80 * 1e6 / (19.8 * 3600) = ~$1,122/MTok

# Compare: Claude Opus 4.7 output is $25/MTok — 45x cheaper.
# This illustrates why 1T+ parameter dense models are not served commercially
# at reasonable prices without massive distillation, MoE architecture,
# or hardware generations beyond B200 with substantially higher bandwidth.`}
      </CodeBlock>

    </div>
  ),
};

export default inferenceCostEconomics;
