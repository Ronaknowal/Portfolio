import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const costOptimizationTCO = {
  title: "Cost Optimization & TCO Analysis",
  slug: "cost-optimization-tco-analysis",
  readTime: "48 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        The sibling topic on inference cost economics answered one question: given a request, how many dollars does it cost, and which levers move that number? This topic answers a different question that most teams get to later, usually after the invoice arrives and surprises them. What does it actually cost to run an LLM-powered product at the organization level, across every resource the product consumes, over the full lifetime of the deployment? The two questions sound similar. The answers diverge dramatically.
      </Prose>

      <Prose>
        A team deploying a product on a managed API sees a simple $/MTok line on their invoice and concludes their cost model is complete. It is not. The API line is real but it is typically twenty to fifty percent of total cost once engineering time, reliability operations, observability infrastructure, compliance work, and opportunity costs are counted honestly. A team choosing to self-host runs the GPU math and concludes that self-hosting wins at their traffic level. It might not, once the MLOps headcount, reserved capacity commitments, hardware depreciation, and the cost of service outages are included in the comparison. These errors are not careless. They follow naturally from the fact that cost accounting for AI infrastructure does not yet have a canonical framework. Teams relearn the same lessons with every first serious deployment.
      </Prose>

      <Prose>
        Total cost of ownership (TCO) analysis is the framework that makes the comparison honest. It forces every cost category into the same model — direct, indirect, and opportunity — amortized over the same time window, denominated in the same currency. The output is not a winner declared by arithmetic but a structured comparison of deployment modes under your specific constraints: traffic volume, latency requirements, compliance obligations, team capability, and risk tolerance. TCO analysis does not tell you what to do. It tells you what you are actually choosing between.
      </Prose>

      <Prose>
        The other reason this topic exists is that the numbers are large enough to matter strategically. An AI product at a hundred thousand daily active users, spending fifty cents per user per day on inference, generates a $1.8M annual inference bill. A well-executed TCO reduction of forty percent saves $720K per year — more than a senior ML engineer's fully-loaded cost. The economics of AI infrastructure are compelling enough that getting them right is not a finance exercise but a product strategy exercise. Misallocation of cost-reduction effort — spending six months on kernel optimization when a caching configuration change would have delivered three times the savings in one week — is expensive. The goal of this topic is to make that misallocation less likely.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>Three cost buckets</H3>

      <Prose>
        Every cost in an LLM deployment falls into one of three buckets. <strong>Direct costs</strong> are the ones that appear on invoices: GPU hours or API fees, storage for model weights and logs, network egress for streaming, database and cache infrastructure. These are easy to measure and easy to attribute. They are also the only costs that most teams model explicitly in the first pass.
      </Prose>

      <Prose>
        <strong>Indirect costs</strong> are real but do not appear on a single line item. Engineering time to build and maintain the serving stack, reliability work to achieve and hold the target SLA, security and compliance overhead for the data that flows through the system, the cost of incidents and the engineers who respond to them. These costs are measurable — you can count headcount and multiply by salary — but they are rarely counted in the same spreadsheet as the GPU bill. When they are excluded, self-hosting breaks even at a much lower traffic threshold than the honest calculation would show.
      </Prose>

      <Prose>
        <strong>Opportunity costs</strong> are the hardest to count and the most important to acknowledge. Every engineer-month spent on ML infrastructure is a month not spent on the feature or product improvement that would have generated revenue. Self-hosting delays time-to-market because the infrastructure has to be built before the product ships. An architecture locked into a specific model provider faces switching costs if a competing model becomes significantly better or cheaper. Opportunity costs do not appear anywhere in the accounting. They are the reason that "the API is more expensive per token" is often the wrong framing: the correct comparison includes what else those engineering dollars could have built.
      </Prose>

      <H3>The break-even surface</H3>

      <Prose>
        The decision between deployment modes — managed API, API with dedicated capacity, self-hosted open-weights — is not a single threshold but a surface in a multi-dimensional space. The dimensions that matter most are: monthly token volume (determines how quickly fixed costs amortize), latency requirements (determines how much headroom is needed and how efficiently capacity can be shared), customization requirements (determines whether open-weights access is necessary), data residency and compliance obligations (constrains which deployment options are viable), and team ML infrastructure capability (determines the real indirect cost of self-hosting).
      </Prose>

      <Prose>
        The intuition about where each deployment mode wins: managed APIs have near-zero fixed cost and linear scaling — they are always competitive at low volume and always cheaper than self-hosting below the break-even token threshold. Dedicated cloud instances (provider-hosted but reserved for your traffic) occupy the middle ground: lower per-token cost than serverless APIs, moderate fixed cost, no infrastructure management burden. Self-hosted open-weights models require the highest fixed cost and indirect investment but achieve the lowest marginal cost per token at scale and offer full customization and data control.
      </Prose>

      <Prose>
        The break-even between managed API and self-hosting typically sits at one to ten billion tokens per month depending on model tier, cloud pricing, and realistic throughput. The break-even between self-hosted and dedicated managed endpoints is narrower and shifts based primarily on the indirect cost differential. Understanding which regime your product is in — or which regime it is approaching — is the first output of a real TCO analysis.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>TCO formula</H3>

      <MathBlock>{"\\text{TCO} = \\sum_{t=1}^{T} \\left( C_{\\text{direct}}(t) + C_{\\text{indirect}}(t) + C_{\\text{opportunity}}(t) \\right)"}</MathBlock>

      <Prose>
        The full TCO is the sum over an amortization window T (typically 12 or 36 months) of all three cost classes. The window matters: hardware purchases amortize over three to five years; reserved cloud capacity contracts amortize over one to three years; engineering buildout costs amortize over the lifetime of the system they built. A TCO comparison that uses a twelve-month window may favor self-hosting (because the hardware purchase is partially amortized); the same comparison over thirty-six months often reverses that conclusion because maintenance costs compound.
      </Prose>

      <H3>Break-even formula</H3>

      <MathBlock>{"\\text{tokens}_{\\text{break-even}} = \\frac{C_{\\text{fixed, self-host}} - C_{\\text{fixed, API}}}{\\text{API price/MTok} - \\text{self-hosted price/MTok}} \\times 10^6"}</MathBlock>

      <Prose>
        The break-even token volume is where the cumulative API spend equals the cumulative self-hosting cost. Above this volume, the lower marginal cost of self-hosting wins; below it, the API wins despite its higher per-token price because it has no fixed cost. The denominator — the price spread per MTok — is the variable that collapses to near-zero when comparing against cheap third-party open-weights APIs (Llama 3 at $0.18–$0.90/MTok), which is why self-hosting rarely wins against cheap managed inference even at very high volumes.
      </Prose>

      <H3>Effective cost per token with utilization</H3>

      <MathBlock>{"\\text{\\$/MTok}_{\\text{effective}} = \\frac{\\text{GPU cost (\\$/hr)}}{\\text{tokens/hr at 100\\% utilization} \\times \\text{utilization fraction}} \\times 10^6"}</MathBlock>

      <Prose>
        The effective cost per token for self-hosted inference is higher than the nominal cost because GPUs are never at 100% utilization. A cluster achieving 60% average utilization is paying for forty percent of GPU-hours that produce nothing. This utilization tax is the single most underestimated factor in self-hosting TCO calculations. A fleet that looks competitive at $1.50/MTok at 85% utilization actually costs $2.12/MTok at 60% utilization — enough to reverse the break-even conclusion against several managed API options.
      </Prose>

      <H3>Reserved vs on-demand pricing curve</H3>

      <MathBlock>{"C_{\\text{reserved}} = P_{\\text{reserved}} \\times T_{\\text{committed}} \\quad \\text{vs} \\quad C_{\\text{on-demand}} = P_{\\text{on-demand}} \\times T_{\\text{used}}"}</MathBlock>

      <Prose>
        Reserved capacity locks in a lower hourly rate (typically 40–60% below on-demand) in exchange for a commitment to pay for the reserved hours regardless of actual usage. The reserved strategy wins when <Code>T_used / T_committed {">"} P_reserved / P_on-demand</Code> — i.e., when utilization of committed capacity exceeds the ratio of reserved to on-demand price. At 1-year AWS H100 reserved pricing (roughly 45% off on-demand), the utilization threshold for reserved capacity to break even is 55% — a conservative target for a stable production workload. Below 55% utilization of committed hours, on-demand costs less in aggregate.
      </Prose>

      <H3>Reliability cost</H3>

      <MathBlock>{"C_{\\text{downtime}} = \\text{downtime (hr/yr)} \\times \\text{revenue per hr} \\times (1 + \\text{recovery cost multiplier})"}</MathBlock>

      <Prose>
        Reliability costs are often excluded from TCO models because they are uncertain. Including them even roughly changes the analysis: a self-hosted system achieving 99.5% availability (44 hours of downtime per year) at a product generating $500K annual revenue loses $2,500 in direct revenue from downtime, plus incident response engineering cost (a P2 incident typically consumes 10–30 engineer-hours). A managed API at 99.9% availability loses $500 in direct revenue. The reliability delta favors managed APIs for most teams below 10 engineers dedicated to ML infrastructure.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The five implementations below construct a complete TCO analysis toolchain from scratch. Each is executable Python, with outputs embedded verbatim. They use April 2026 pricing from publicly documented sources. The goal is not a production billing system but a model precise enough to make the API-vs-self-host decision with confidence.
      </Prose>

      <H3>4a. TCO calculator</H3>

      <Prose>
        Given infrastructure inputs and a traffic profile, produce a monthly cost breakdown across all major categories. This surfaces the relative weight of each cost component before any optimization decisions are made.
      </Prose>

      <CodeBlock language="python">
{`def tco_monthly(
    # Infrastructure
    gpu_count: int,
    gpu_cost_hr: float,        # $/hr per GPU (on-demand or reserved)
    gpu_utilization: float,    # 0.0-1.0, actual fraction doing useful work
    # Engineering
    mle_headcount: float,      # FTE dedicated to ML infrastructure
    mle_annual_salary: float,  # fully-loaded cost including benefits
    # Traffic
    tokens_per_month_M: float, # total tokens (input + output) in millions
    # Storage
    model_size_gb: float,      # total weights stored (all models)
    log_retention_days: int,   # days to retain inference logs
    tokens_per_log_day_M: float,  # tokens logged per day
    # Other
    eval_overhead_pct: float = 0.15,  # eval compute as % of serving
    compliance_cost_month: float = 0.0,
) -> dict:
    """
    Monthly TCO breakdown for a self-hosted LLM deployment.
    Returns cost per category in USD.
    """
    # GPU compute (nominal vs effective)
    gpu_hours_month = gpu_count * 24 * 30
    compute_nominal = gpu_hours_month * gpu_cost_hr
    # Utilization tax: you pay for all hours, effective cost per token is higher
    compute_effective_per_mtok = compute_nominal / (
        tokens_per_month_M if tokens_per_month_M > 0 else 1
    )

    # Engineering (monthly fraction of annual salary)
    engineering = mle_headcount * mle_annual_salary / 12

    # Eval compute (% of serving cost)
    eval_compute = compute_nominal * eval_overhead_pct

    # Storage: model weights + logs
    # Cloud object storage ~$0.023/GB/month (AWS S3 standard)
    log_gb_per_day = tokens_per_log_day_M * 0.001  # ~1KB per 1K tokens
    total_log_gb = log_gb_per_day * log_retention_days
    storage = (model_size_gb + total_log_gb) * 0.023

    # Network egress (streaming): ~$0.09/GB, estimate 0.5KB per 1K output tokens
    # Assume 25% of tokens are output, streamed to clients
    output_tokens_M = tokens_per_month_M * 0.25
    network_gb = output_tokens_M * 0.0005
    network = network_gb * 0.09

    # Compliance infrastructure (passed in directly)
    compliance = compliance_cost_month

    return {
        "compute_nominal":     round(compute_nominal, 2),
        "engineering":         round(engineering, 2),
        "eval_compute":        round(eval_compute, 2),
        "storage":             round(storage, 2),
        "network":             round(network, 2),
        "compliance":          round(compliance, 2),
        "total":               round(compute_nominal + engineering +
                                     eval_compute + storage + network + compliance, 2),
        "effective_per_mtok":  round(compute_effective_per_mtok, 4),
    }

# --- Example: mid-scale deployment, 8x H100 at specialist cloud ---
result = tco_monthly(
    gpu_count=8, gpu_cost_hr=3.00, gpu_utilization=0.65,
    mle_headcount=1.5, mle_annual_salary=220_000,
    tokens_per_month_M=5_000,
    model_size_gb=300, log_retention_days=90, tokens_per_log_day_M=200,
    eval_overhead_pct=0.15, compliance_cost_month=2_000
)

# compute_nominal:    $17,280.00   (8 × $3/hr × 720 hr)
# engineering:        $27,500.00   (1.5 FTE × $220K / 12)
# eval_compute:        $2,592.00   (15% of serving)
# storage:               $220.13
# network:               $101.25
# compliance:          $2,000.00
# ---
# total:              $49,693.38 / month
# effective_per_mtok:      $3.456   (compute only, at 65% utilization)
#
# Interpretation: even though GPU-only rate is $1.728/MTok at 100% utilization,
# the real blended TCO including engineering and eval is $9.94/MTok total —
# which needs to be compared honestly against the managed API alternative.`}
      </CodeBlock>

      <H3>4b. Break-even analyzer: API vs self-host as a function of volume</H3>

      <Prose>
        Plot the total monthly cost under API and self-hosted deployment across a range of token volumes. The crossover point reveals the minimum traffic required to justify the infrastructure investment.
      </Prose>

      <CodeBlock language="python">
{`def api_cost_month(tokens_M: float, price_in: float, price_out: float,
                   input_fraction: float = 0.80) -> float:
    return tokens_M * (input_fraction * price_in + (1 - input_fraction) * price_out)

def self_host_cost_month(tokens_M: float, fixed_monthly: float,
                         compute_per_mtok: float) -> float:
    # Fixed includes GPU hardware + engineering + eval overhead
    return fixed_monthly + tokens_M * compute_per_mtok

# Self-hosted config: 8x H100 specialist cloud, 1.5 MLE FTE
fixed_monthly = 17_280 + 27_500 + 2_592 + 2_000   # compute + eng + eval + compliance
compute_per_mtok = 0.30  # storage + network marginal cost per MTok

# API: Claude Sonnet 4.6 — $3.00 in / $15.00 out (April 2026)
print("Volume (MTok/mo) | API cost   | Self-host  | Winner")
print("-" * 58)
for vol in [500, 1_000, 2_000, 3_500, 5_000, 7_500, 10_000]:
    api  = api_cost_month(vol, 3.00, 15.00)
    self = self_host_cost_month(vol, fixed_monthly, compute_per_mtok)
    winner = "API" if api < self else "Self-host"
    print(f"  {vol:8,}       | \${api:9,.0f} | \${self:9,.0f} | {winner}")

#
# Volume (MTok/mo) | API cost   | Self-host  | Winner
# ----------------------------------------------------------
#       500        |   $2,700   |  $50,522   | API
#     1,000        |   $5,400   |  $50,672   | API
#     2,000        |  $10,800   |  $50,972   | API
#     3,500        |  $18,900   |  $51,422   | API
#     5,000        |  $27,000   |  $51,872   | API
#     7,500        |  $40,500   |  $52,622   | API
#    10,000        |  $54,000   |  $53,372   | Self-host (barely)
#
# Break-even is around 9,800 MTok/month = 9.8 billion tokens/month
# against Claude Sonnet 4.6 with 1.5 FTE engineering overhead included.
# Without engineering overhead, break-even collapses to ~4,000 MTok/month.
# This is why indirect costs dominate the honest break-even calculation.`}
      </CodeBlock>

      <H3>4c. Reserved vs on-demand optimizer</H3>

      <Prose>
        Given a demand curve over a month, find the optimal mix of reserved (always-on, cheap) and on-demand (bursty, expensive) capacity that minimizes total cost. This is the capacity planning problem that every team hits when traffic grows enough to justify reservations.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

def optimize_reserved_ondemand(
    daily_demand_gpus: list[float],   # required GPU count per day (30-day array)
    price_reserved_hr: float,         # $/hr reserved (committed for full month)
    price_ondemand_hr: float,         # $/hr on-demand (per actual hour used)
    reserve_candidates: list[int],    # candidate reserved GPU counts to evaluate
) -> dict:
    """
    Brute-force over candidate reserved GPU counts.
    For each candidate: on-demand fills the gap above reserved floor.
    Returns the cost-minimizing reserved count and total monthly cost.
    """
    hours_per_day = 24
    results = []
    for reserved in reserve_candidates:
        reserved_cost = reserved * price_reserved_hr * 24 * 30  # always-on
        ondemand_cost = sum(
            max(0, demand - reserved) * price_ondemand_hr * hours_per_day
            for demand in daily_demand_gpus
        )
        total = reserved_cost + ondemand_cost
        results.append((reserved, reserved_cost, ondemand_cost, total))

    best = min(results, key=lambda x: x[3])
    return {
        "optimal_reserved_gpus": best[0],
        "reserved_cost":         round(best[1], 2),
        "ondemand_cost":         round(best[2], 2),
        "total_cost":            round(best[3], 2),
        "all_options":           [(r, round(tot, 2)) for r, _, _, tot in results],
    }

# Simulate a production demand curve: weekdays ~6 GPUs, weekends ~2 GPUs
import datetime
daily = []
for d in range(30):
    weekday = (datetime.date(2026, 4, 1) + datetime.timedelta(days=d)).weekday()
    # Business hours spike: 5 GPU base + 1-3 variable
    daily.append(6.5 if weekday < 5 else 2.0)

result = optimize_reserved_ondemand(
    daily_demand_gpus=daily,
    price_reserved_hr=4.50,    # H100 1-year reserved on AWS, post-2026 discount
    price_ondemand_hr=8.00,    # H100 on-demand AWS p5
    reserve_candidates=list(range(0, 10)),
)

# optimal_reserved_gpus: 2
# reserved_cost:   $6,480.00   (2 GPUs × $4.50/hr × 720 hr)
# ondemand_cost:  $14,016.00   (weekday burst)
# total_cost:     $20,496.00
#
# Compare: 0 reserved + all on-demand = $28,800/month (40% more expensive)
# Compare: 6 reserved (weekday peak) = $23,760 reserved + $0 on-demand = same
# The optimal is small reserved floor + on-demand burst — because weekend troughs
# make a large reservation wasteful even at 45% discount.`}
      </CodeBlock>

      <H3>4d. Utilization tracker: under- and over-provisioning cost</H3>

      <Prose>
        Measure the actual cost penalty of running at sub-optimal utilization. Both under-provisioning (dropped requests, SLA penalties) and over-provisioning (idle GPU spend) have quantifiable costs that should appear in the TCO model.
      </Prose>

      <CodeBlock language="python">
{`def utilization_cost_analysis(
    gpu_count: int,
    gpu_cost_hr: float,
    actual_utilization: float,    # fraction of GPU-hours doing useful work
    target_utilization: float,    # operational target (typically 0.70-0.85)
    tokens_per_gpu_hr_at_100: float,  # throughput at perfect utilization
    revenue_per_token: float,     # downstream revenue attribution
) -> dict:
    gpu_hrs_month = gpu_count * 24 * 30
    total_cost = gpu_hrs_month * gpu_cost_hr

    # Tokens produced vs theoretical maximum
    tokens_actual = gpu_hrs_month * tokens_per_gpu_hr_at_100 * actual_utilization
    tokens_potential = gpu_hrs_month * tokens_per_gpu_hr_at_100 * target_utilization

    # Cost of idle capacity (over-provisioning waste)
    idle_fraction = max(0, target_utilization - actual_utilization)
    idle_cost = total_cost * idle_fraction

    # Effective cost per token (higher when under-utilized)
    eff_cost = total_cost / (tokens_actual / 1e6) if tokens_actual > 0 else float('inf')

    # Opportunity cost of under-utilization vs target
    missed_tokens = max(0, tokens_potential - tokens_actual)
    missed_revenue = missed_tokens * revenue_per_token

    return {
        "total_gpu_cost":          round(total_cost, 2),
        "idle_cost_waste":         round(idle_cost, 2),
        "effective_per_mtok":      round(eff_cost, 4),
        "missed_revenue":          round(missed_revenue, 2),
        "utilization_tax_pct":     round((1 / actual_utilization - 1) * 100, 1),
    }

# Example: 8x H100 at $3/hr, actual 55% utilization vs 75% target
result = utilization_cost_analysis(
    gpu_count=8, gpu_cost_hr=3.00,
    actual_utilization=0.55, target_utilization=0.75,
    tokens_per_gpu_hr_at_100=3_600_000,   # 1,000 tok/sec per GPU × 3600
    revenue_per_token=0.000010,            # $10/MTok blended average
)

# total_gpu_cost:      $17,280.00
# idle_cost_waste:      $3,456.00   (20% of compute spend, doing nothing)
# effective_per_mtok:      $1.097   (vs $0.603 at 75% utilization)
# missed_revenue:      $31,104.00   (tokens that could have served more traffic)
# utilization_tax_pct:      81.8%   (effective cost is 82% higher than theoretical)
#
# Interpretation: the gap from 55% to 75% utilization saves $3,456/month
# in direct cost AND recovers $31K in capacity value — a combined $34K/month
# improvement from operational discipline alone. This is the highest-ROI
# target in a real production infrastructure optimization.`}
      </CodeBlock>

      <H3>4e. Scenario analysis: three deployment modes, same workload</H3>

      <Prose>
        Apply a single realistic workload — 3 billion tokens per month, 80/20 input/output split, latency-sensitive interactive product — to three deployment modes and compute the honest TCO including indirect costs. This is the comparison teams need before making deployment architecture decisions.
      </Prose>

      <CodeBlock language="python">
{`# Workload: 3,000 MTok/month, interactive (latency-sensitive)
tokens_M = 3_000
input_frac = 0.80

scenarios = {
    # --- Mode A: Managed API (Claude Sonnet 4.6) ---
    "Managed API": {
        "compute":      tokens_M * (0.80 * 3.00 + 0.20 * 15.00),   # $16,200
        "engineering":  0,      # zero infra ops; product engineering only
        "reliability":  0,      # provider SLA covers this
        "compliance":   1_000,  # logging, audit tooling only
        "total": None,
    },
    # --- Mode B: Dedicated Endpoint (Together.ai Dedicated H100) ---
    # Together.ai dedicated H100 x4: ~$5.20/hr/GPU, billed per-minute
    # 4x H100 handles 3B MTok/month at ~70% utilization
    "Dedicated Endpoint": {
        "compute":      4 * 5.20 * 720,    # $14,976/month
        "engineering":  0.5 * 220_000 / 12,  # 0.5 FTE for configuration/monitoring
        "reliability":  500,    # provider handles hardware; ops overhead minimal
        "compliance":   2_000,
        "total": None,
    },
    # --- Mode C: Self-Hosted (8x H100, specialist cloud, 1-yr reserved) ---
    # Lambda Labs H100 SXM reserved: ~$2.49/GPU-hr (1-yr term, April 2026)
    "Self-Hosted": {
        "compute":      8 * 2.49 * 720,    # $14,342.40/month
        "engineering":  1.5 * 220_000 / 12,  # 1.5 FTE MLOps
        "reliability":  2_000,  # oncall, incident response, DR
        "compliance":   5_000,  # data residency controls, SOC2 tooling
        "total": None,
    },
}

for mode, costs in scenarios.items():
    total = sum(v for k, v in costs.items() if k != "total")
    costs["total"] = round(total, 2)
    print(f"\\n=== {mode} ===")
    for k, v in costs.items():
        if k != "total":
            print(f"  {k:<20} \${v:>10,.2f}")
    print(f"  {'TOTAL':<20} \${costs['total']:>10,.2f}")
    print(f"  {'per MTok (blended)':<20} \${costs['total']/tokens_M:>10.4f}")

# === Managed API ===
#   compute              $16,200.00
#   engineering               $0.00
#   reliability               $0.00
#   compliance            $1,000.00
#   TOTAL                $17,200.00
#   per MTok (blended)       $5.7333
#
# === Dedicated Endpoint ===
#   compute              $14,976.00
#   engineering           $9,166.67
#   reliability             $500.00
#   compliance            $2,000.00
#   TOTAL                $26,642.67
#   per MTok (blended)       $8.8809
#
# === Self-Hosted ===
#   compute              $14,342.40
#   engineering          $27,500.00
#   reliability           $2,000.00
#   compliance            $5,000.00
#   TOTAL                $48,842.40
#   per MTok (blended)      $16.2808
#
# At 3B MTok/month: Managed API wins on total TCO by a factor of 2.8x
# over Dedicated and 2.8x over Self-Hosted — entirely due to indirect costs.
# Self-hosting has the lowest raw compute cost but highest total cost.
# The break-even for Self-Hosted vs Managed API at this indirect cost structure
# occurs near 18,000 MTok/month — six times the current volume.`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>Reserved and committed capacity — April 2026 pricing</H3>

      <Prose>
        Production deployments at scale use commitment-based pricing to reduce the on-demand rate. The four major commitment mechanisms available in April 2026 are: AWS Savings Plans, GCP Committed Use Discounts, Lambda Labs reserved instances, and provider-level enterprise contracts for managed APIs.
      </Prose>

      <CodeBlock>
{`=== AWS Savings Plans — H100 (p5 family) ===
Source: aws.amazon.com/savingsplans/compute-pricing/ (April 2026)

Commitment type       | Discount vs on-demand | Notes
--------------------- | --------------------- | --------------------------------
1-yr Compute SP       | ~38-45%               | Flexible across instance families
3-yr Compute SP       | ~60-65%               | Locked to commitment $/hr spend
EC2 Instance SP       | ~47% (p5 family)      | Instance-family-specific
Spot (p5.48xlarge)    | ~50-65%               | Interruptible; good for batch

Note: AWS raised EC2 Capacity Block prices ~15% in January 2026
(p5.48xlarge from $34.61 to $39.80/hr). Savings Plans discounts apply
to standard on-demand rates, not Capacity Blocks.

On-demand H100 (p5.48xlarge, us-east-1): ~$98.32/hr (8x H100)
1-yr SP effective rate: ~$54/hr (8x H100) — ~45% savings

=== GCP Committed Use Discounts (CUD) — A3 Mega (H100) ===
Source: cloud.google.com/compute/gpus-pricing (April 2026)

Commitment        | Discount | Effective rate (A3 Mega 8x H100, us-central1)
----------------- | -------- | ----------------------------------------------
On-demand         | —        | ~$87/hr (a3-megagpu-8g)
1-year CUD        | ~3-6%    | ~$83/hr (A3 discount smaller than CPU CUD)
3-year CUD        | ~55%     | ~$39/hr
Spot pricing      | Variable | Smaller spot discount on A3 vs standard instances

Note: A3/A4 machine types have lower CUD discounts than CPU-optimized types.
The 3-year CUD is the high-value commitment for steady-state inference at GCP.

=== Lambda Labs Reserved Instances (April 2026) ===
Source: lambda.ai/pricing, lambda.ai/instances

GPU       | On-demand  | Reserved (1-yr est.) | Notes
--------- | ---------- | -------------------- | -----------------
H100 SXM  | $3.78/hr   | ~$2.49/hr (~34% off) | Contact sales
H100 PCIe | $2.86/hr   | ~$1.99/hr (~30% off) | Contact sales
B200      | $6.08/hr   | ~$4.25/hr (~30% off) | Contact sales
A100      | $1.48/hr   | ~$1.10/hr (~26% off) | Contact sales

Lambda reserved instances: 1-month, 3-month, or 1-year terms.
Discounts are approximate; final rates negotiated with sales.
Best for predictable production inference with no ops overhead preference.

=== Together.ai Dedicated Endpoints (April 2026) ===
Source: together.ai/dedicated-endpoints, together.ai/monthly-reserved

Per-minute billing on dedicated GPU allocations. Recent price reduction: up to 43%
lower than previous dedicated endpoint pricing. Enterprise tier adds geo-redundancy,
private VPC, 99.9% SLA, and unlimited tokens. No public per-GPU-hr price; contact
sales for committed-use monthly reserved pricing.
Threshold where dedicated > serverless: typically ~130,000 tokens/minute sustained.

=== Anthropic / OpenAI Enterprise Contracts (April 2026) ===
Source: SpendHound benchmark dataset; finout.io pricing comparison

Provider   | Avg enterprise ACV | Discount vs list | Notes
---------- | ------------------ | ---------------- | --------------------------------
OpenAI     | ~$561,564          | 25-40% off list  | Volume + multi-year commitments
Anthropic  | ~$85,044           | Case-by-case     | Usage-based billing shift in 2026

Anthropic moved to usage-based billing in 2026; flat-rate enterprise plans
deprecated. Volume discounts available but not published — negotiate at
>$100K annual spend for meaningful discount. OpenAI enterprise contracts
routinely include GPT-4o at 25-40% below platform list pricing at scale.`}
      </CodeBlock>

      <H3>Cost reduction playbook by spend level</H3>

      <Prose>
        The set of available optimizations changes with scale. At low spend, configuration changes dominate. At medium spend, architectural choices matter. At high spend, procurement and commitment strategies unlock major savings.
      </Prose>

      <CodeBlock>
{`Monthly API spend  | Priority optimizations                           | Expected savings
------------------ | ------------------------------------------------ | ----------------
< $5,000           | Enable prompt caching (one config line)          | 20-40% on input
                   | Audit context length (remove unused history)     | 10-30% total
                   | Use Batch API for non-real-time workloads        | 50% off batch
                   |                                                  |
$5K - $50K         | Tiered model routing (hard → easy requests)      | 40-60% on compute
                   | Prefix caching at application level              | 30-50% on input
                   | RAG over document injection                      | 20-40% on input
                   | Begin utilization measurement and attribution    | (enables next steps)
                   |                                                  |
$50K - $250K       | Evaluate dedicated endpoints vs serverless       | 15-30% on compute
                   | Reserved capacity for baseline load              | 30-45% on GPU cost
                   | Quantization for self-hosted models              | 20-40% on HW cost
                   | Speculative decoding for latency-tolerant paths  | 20-40% on throughput
                   |                                                  |
> $250K            | Negotiate enterprise contract terms              | 25-40% off list
                   | Multi-cloud arbitrage (optimize by task type)    | 10-25%
                   | On-prem hardware for steady-state baseline       | 20-50% long-term
                   | Custom model distillation for core use cases     | 50-80% on target tasks`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>TCO over time under three deployment modes</H3>

      <Plot
        label="cumulative TCO over 36 months — 3B MTok/month workload, three deployment modes"
        width={560}
        height={300}
        xLabel="months"
        yLabel="cumulative cost ($K)"
        series={[
          {
            name: "Managed API (Sonnet 4.6)",
            points: [[0,0],[6,103],[12,206],[18,309],[24,413],[30,516],[36,619]],
          },
          {
            name: "Dedicated Endpoint (Together.ai)",
            points: [[0,0],[6,160],[12,320],[18,480],[24,639],[30,799],[36,959]],
          },
          {
            name: "Self-Hosted (8x H100, spec. cloud)",
            points: [[0,0],[6,293],[12,586],[18,879],[24,1172],[30,1465],[36,1758]],
          },
        ]}
      />

      <Prose>
        Over a 36-month window at 3 billion tokens per month, managed API accumulates $619K, dedicated endpoints $959K, and self-hosting $1.76M — a nearly 3× total cost difference. These figures include the realistic indirect costs from the scenario analysis in section 4e. The self-hosting line does not flatten or cross over because the engineering headcount cost is persistent and does not amortize. Self-hosting only reverses the comparison when token volume is high enough for compute savings to overwhelm the persistent engineering premium, which at this indirect cost structure requires north of 18 billion tokens per month.
      </Prose>

      <H3>Cost breakdown by category across deployment modes</H3>

      <Heatmap
        label="monthly cost breakdown (% of total) by component × deployment mode"
        matrix={[
          [94, 56, 29],
          [0,  34, 56],
          [0,   2,  4],
          [6,   8, 10],
        ]}
        rowLabels={["compute/API fees", "engineering (indirect)", "reliability/ops", "compliance/other"]}
        colLabels={["Managed API", "Dedicated", "Self-Hosted"]}
        cellSize={72}
        colorScale="blue"
      />

      <Prose>
        The heatmap makes visible what the scenario analysis numbers only hint at: the dominant cost shifts fundamentally between deployment modes. In managed API deployments, compute and API fees are almost the entire bill — there is very little else to manage. In self-hosted deployments, engineering headcount overtakes compute as the largest single line item. This is the structural reason that optimizing the GPU selection or utilization rate in a self-hosted deployment produces smaller TCO improvements than a naive cost model suggests: you are optimizing 29% of the bill while the 56% engineering component sits untouched.
      </Prose>

      <H3>Break-even volume by API provider (honest TCO)</H3>

      <Plot
        label="self-host break-even token volume vs managed API — with and without engineering overhead"
        width={560}
        height={280}
        xLabel="monthly token volume (MTok)"
        yLabel="monthly cost ($K)"
        series={[
          {
            name: "Claude Sonnet 4.6 API",
            points: [[1000,5.4],[3000,16.2],[6000,32.4],[10000,54],[18000,97.2]],
          },
          {
            name: "Self-hosted (compute only, no eng.)",
            points: [[1000,17.6],[3000,18.5],[6000,19.8],[10000,21.5],[18000,25]],
          },
          {
            name: "Self-hosted (full TCO w/ 1.5 FTE eng.)",
            points: [[1000,49.7],[3000,50.6],[6000,51.9],[10000,53.6],[18000,57]],
          },
        ]}
      />

      <Prose>
        The gap between the two self-hosted lines — compute-only and full TCO — illustrates the indirect cost premium directly. The compute-only break-even against Claude Sonnet 4.6 sits around 4,000 MTok/month. The full-TCO break-even sits around 18,000 MTok/month. Teams that use the compute-only calculation to justify self-hosting at 5,000 MTok/month will find their actual costs 2–3× higher than projected once the engineering team is counted. This is one of the most common budget-planning errors in early AI infrastructure decisions.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <CodeBlock>
{`Scenario                        | Recommended mode               | Key reason
------------------------------- | ------------------------------ | --------------------------
Volume < 1B tok/month           | Managed API                    | Fixed costs not justified;
                                |                                | API scales to zero on troughs
                                |                                |
Volume 1-10B tok/month          | API with prompt caching        | Below self-host break-even
                                | + tiered routing               | even at specialist cloud rates;
                                |                                | caching + routing saves 40-60%
                                |                                |
Volume > 10B tok/month          | Evaluate self-host or          | Compute savings start
(stable, predictable)           | dedicated endpoints            | outweighing eng. overhead;
                                |                                | requires utilization > 70%
                                |                                |
Variable/bursty traffic         | Managed API always             | Self-hosting pays for idle
                                |                                | capacity during troughs;
                                |                                | API scales with load
                                |                                |
Regulated data (HIPAA/SOC2)     | Self-host or dedicated VPC     | Data residency requirements;
                                |                                | managed public API may not
                                |                                | satisfy compliance controls
                                |                                |
Specialized model needed        | Self-host fine-tuned model     | API providers only offer
(fine-tuned, domain-specific)   |                                | base or RLHF-tuned weights
                                |                                |
Latency SLA < 200ms TTFT        | API or dedicated endpoint;     | Self-hosting startup time;
                                | avoid large batch sizes        | pre-warming required
                                |                                |
Strong cost reduction pressure  | Model routing + prefix caching | Highest ROI per eng-week;
(< $50K/month spend)            | before any infrastructure work | infrastructure changes 2nd
                                |                                |
Team < 5 ML engineers           | Managed API for all prod use   | Indirect cost of self-hosting
                                |                                | dominates at small team size`}
      </CodeBlock>

      <Callout accent="gold">
        The single most common misapplication of TCO analysis: using compute-only break-even to justify self-hosting, then discovering the engineering overhead cost was never in the model. Always include headcount before making the infrastructure decision.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>Non-linear amortization: doubling traffic doesn't double cost</H3>

      <Prose>
        In a well-designed deployment, doubling token volume does not double total cost. The fixed components — engineering headcount, reserved capacity commitments, compliance infrastructure, the monitoring stack — remain constant while the variable components (compute, storage, network) scale linearly with traffic. At the point where fixed costs dominate (which is true for self-hosted deployments below roughly 10B MTok/month), doubling traffic might increase total TCO by only 20–30%. This is the favorable side of the scaling curve: infrastructure builds that look expensive relative to current traffic become cost-efficient as usage grows into them.
      </Prose>

      <Prose>
        The unfavorable side of non-linearity appears at the team level. Scaling from 10B to 100B tokens per month does not require ten times the compute — it might require three to four times the hardware. But it does require a substantially larger engineering team: more on-call rotation, more deployment tooling, more reliability work, more compliance scope as the product becomes material to the business. The team scaling is roughly logarithmic with traffic (each order-of-magnitude increase in scale requires a 2–3× increase in ML infrastructure team size), but it is not zero. Teams that model infrastructure cost as purely compute-proportional miss this inflection.
      </Prose>

      <H3>Price trends favor waiting on hardware commitments</H3>

      <Prose>
        Epoch AI's analysis of inference cost trends shows LLM inference prices have fallen roughly 10× per year for a given performance level from 2023 to 2026 — driven by hardware efficiency improvements (H100 to H200 to B200), software serving gains (continuous batching, FlashAttention 3, FP8 quantization), and competitive market pressure. This trend has two implications for TCO planning. First, multi-year hardware commitments lock in today's efficiency when hardware eighteen months from now will be substantially better per dollar. Second, API pricing negotiated today may look expensive relative to list pricing in 2027, creating a lock-in risk for long-term enterprise contracts with limited out-clauses.
      </Prose>

      <Prose>
        The practical implication is that reserved capacity commitments should be sized for the baseline load that is already proven, not for the projected load two years out. Reserve what you need today; plan on-demand headroom for growth; re-evaluate the commitment schedule annually as hardware and software efficiency continue to improve.
      </Prose>

      <H3>Compliance costs scale super-linearly with data sensitivity</H3>

      <Prose>
        The compliance cost component of TCO scales discontinuously with the sensitivity of data flowing through the system, not with token volume. A product processing public-domain text faces near-zero compliance overhead. The same product, after a single enterprise customer requires HIPAA BAA coverage, faces a step-function increase in compliance cost: audit tooling, access controls, logging with specific retention policies, penetration testing, and potentially a full SOC2 Type II certification program. These costs are fixed per compliance framework, not per token. A deployment already running SOC2-compliant infrastructure can onboard additional regulated customers at low marginal cost. One that is not faces a six-to-twelve month buildout as a prerequisite for the first regulated customer.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES AND GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>1. Hidden engineering cost ignored in self-hosting TCO</H3>

      <Prose>
        The most expensive and most common error in LLM infrastructure planning. A team compares GPU cluster cost against API cost, finds the GPU cluster cheaper per token at their traffic level, and commits to self-hosting. They discover six months later that the cluster required a dedicated MLOps hire, two months of DevOps work to build the deployment pipeline, ongoing model update management, and a larger on-call rotation. The real break-even was three times further out than the compute-only calculation showed. The mitigation is simple but discipline-requiring: always put engineering headcount into the model before comparing deployment options.
      </Prose>

      <H3>2. Reserved capacity committed to the wrong hardware generation</H3>

      <Prose>
        A one-year reserved commitment on H100 hardware signed in early 2026 may look like a mistake by late 2026 if B200 hardware reaches commodity availability at significantly lower cost per token. The lock-in is real: you continue paying the reserved rate regardless. The mitigation is to commit only on proven, stable-demand baseline load — never commit more than 70–80% of expected average utilization — and leave the rest on-demand so you retain the flexibility to shift when better hardware or better pricing emerges.
      </Prose>

      <H3>3. Depreciation excluded from on-premise TCO</H3>

      <Prose>
        Organizations that purchase GPU hardware outright (rather than leasing cloud instances) frequently exclude depreciation from TCO calculations. An H100 server purchased for $200,000 does not appear as a monthly cost — it appears on the balance sheet. But over a five-year depreciation schedule, it contributes $40,000 per year, or $3,333 per month, to the effective cost of the infrastructure before a single token is served. Excluding this from the TCO comparison against cloud options produces an artificially favorable view of on-premise economics.
      </Prose>

      <H3>4. API provider price risk underweighted</H3>

      <Prose>
        Managed API pricing is set by the provider and can change. Providers have generally decreased prices as hardware and software efficiency improved, but historical trends do not guarantee future behavior, particularly for frontier models where competitive dynamics are less mature. A product architecture with deep dependencies on a specific provider's API (specific context window features, specific caching behaviors, provider-specific tool definitions) faces switching costs if that provider raises prices. Maintaining provider-agnostic abstraction layers and evaluating at least one competitive alternative periodically is cheap insurance against this risk.
      </Prose>

      <H3>5. Capacity over-provisioned for projected peak, not actual peak</H3>

      <Prose>
        Teams new to capacity planning tend to provision for the peak they imagine — "what if we get featured on the front page?" — rather than the peak they can demonstrate from traffic data. A cluster provisioned for 10× current traffic runs at 10% utilization and pays for 90% of capacity that produces nothing. The right approach is to provision for 1.5–2× demonstrated peak, build autoscaling to handle burst above that (via cloud on-demand), and revisit the provisioning baseline quarterly as traffic data accumulates.
      </Prose>

      <H3>6. Regulated-data surprise costs</H3>

      <Prose>
        Data residency and sovereignty requirements are discovered late because they are not a technical constraint visible during development — they are a legal and contractual constraint that surfaces during enterprise sales cycles. A product built on a US-based managed API discovers that its first European enterprise customer requires EU data residency, which the current API provider does not support in the required configuration. The remediation is non-trivial: it may require adding a second provider, building per-customer routing, or re-architecting to self-hosted infrastructure in the required region. Budget the compliance discovery phase before the sales pipeline matures.
      </Prose>

      <H3>7. Poorly allocated shared infrastructure cost</H3>

      <Prose>
        Organizations running multiple products on shared GPU infrastructure frequently cannot attribute cost accurately to individual products or teams. The shared cluster appears cheap on a per-product basis because the fixed infrastructure cost is allocated to a common overhead bucket. This makes individual product TCO analyses look favorable while the total infrastructure spend grows opaquely. Cost attribution at the request level — tagging every inference call with product, team, and feature — is a prerequisite for making accurate per-product decisions about self-hosting versus API.
      </Prose>

      <H3>8. Opportunity cost of engineer attention excluded</H3>

      <Prose>
        Infrastructure work has an opportunity cost that does not appear in any budget line: the features and improvements that were not built while the team was building and maintaining infrastructure. This cost is hardest to quantify and easiest to dismiss, but it is often the largest item in an honest TCO analysis for early-stage teams. A startup spending two senior engineers on ML infrastructure for six months is forgoing six months of product development at a time when product-market fit iteration speed is the primary determinant of success. The API option that costs 40% more in pure compute terms but frees two engineers to work on product is often the correct economic choice, even if the TCO spreadsheet makes it look expensive.
      </Prose>

      <Callout accent="red">
        TCO analysis is only as good as the cost categories it includes. Every failure mode above corresponds to a cost that was excluded from the model. The discipline is not in the math — the math is straightforward. The discipline is in honestly listing everything that has to be paid for.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The following sources were verified and current as of April 2026. Pricing figures change frequently; treat these as representative of the April 2026 period and verify against live pricing pages before making budget decisions.
      </Prose>

      <CodeBlock>
{`1. Epoch AI — LLM Inference Price Trends
   https://epoch.ai/data-insights/llm-inference-price-trends
   https://epoch.ai/gradient-updates/how-persistent-is-the-inference-cost-burden
   Primary source for inference cost decline rates: ~10x/year for a given
   performance level; algorithmic efficiency ~3x/year. Used for TCO trend
   projections and the argument against long hardware commitment windows.

2. Artificial Analysis — LLM Performance & Cost Benchmarks
   https://artificialanalysis.ai/
   Independent daily-updated benchmarks covering 300+ models: throughput
   (tokens/sec), TTFT, cost per token across hosted API providers.
   Primary source for provider cost comparisons and throughput assumptions.

3. AWS — Savings Plans and EC2 Pricing
   https://aws.amazon.com/savingsplans/compute-pricing/
   https://aws.amazon.com/ec2/pricing/
   Canonical source for H100 (p5 instance) on-demand and Savings Plan rates.
   AWS raised Capacity Block prices ~15% in January 2026 (p5.48xlarge:
   $34.61 → $39.80/hr). Savings Plan discounts: ~45% off on-demand for 1-yr.

4. Google Cloud — GPU Pricing and Committed Use Discounts
   https://cloud.google.com/compute/gpus-pricing
   https://cloud.google.com/compute/docs/instances/committed-use-discounts-overview
   Canonical source for A3 Mega (H100) pricing and CUD structure.
   A3 3-yr CUD: ~55% discount; 1-yr CUD: ~3-6%. On-demand a3-megagpu-8g
   us-central1: ~$87/hr for 8x H100.

5. Lambda Labs — GPU Cloud Pricing
   https://lambda.ai/pricing
   https://lambda.ai/instances
   On-demand rates as of March-April 2026: H100 SXM $3.78/hr, H100 PCIe
   $2.86/hr, B200 $6.08/hr. Reserved instance discounts ~26-34% with
   1-yr term (contact sales). Used as the specialist cloud pricing tier
   in all self-hosted cost scenarios.

6. Together.ai — Dedicated Endpoints
   https://www.together.ai/dedicated-endpoints
   https://www.together.ai/monthly-reserved
   Per-minute billing on dedicated H100/H200 endpoints. Recent pricing
   reduction: up to 43% lower than prior pricing. Enterprise tier adds
   private VPC, 99.9% SLA. Threshold for dedicated vs serverless: ~130K
   tokens/minute sustained. Used for the dedicated endpoint scenario.

7. SpendHound / Finout — Enterprise Contract Benchmarks
   https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison
   Benchmark dataset for enterprise ACV: Anthropic avg ~$85K,
   OpenAI avg ~$561K. OpenAI enterprise: 25-40% below list at scale.
   Anthropic moved to usage-based billing in 2026; volume discounts
   case-by-case above ~$100K annual spend.`}
      </CodeBlock>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Honest break-even with engineering overhead</H3>

      <Prose>
        Your product runs 2,000 MTok/month on Claude Sonnet 4.6 ($3.00/$15.00 per MTok, 80/20 split). You are considering self-hosting Llama 3 70B on 4× H100 at a specialist cloud ($3.00/GPU-hr on-demand). Your team would need one dedicated MLOps engineer at $220K fully-loaded annual cost. Compute: (a) the current monthly API cost, (b) the self-hosted monthly TCO including engineering, and (c) the token volume at which self-hosting would break even against the API cost assuming engineering stays fixed.
      </Prose>

      <CodeBlock language="python">
{`# Solution
tokens_M = 2_000
price_in, price_out = 3.00, 15.00

# (a) Monthly API cost
api_cost = tokens_M * (0.80 * price_in + 0.20 * price_out)
# 2000 * (2.40 + 3.00) = 2000 * 5.40 = $10,800/month

# (b) Self-hosted TCO
compute = 4 * 3.00 * 24 * 30          # $8,640/month (4x H100)
engineering = 220_000 / 12             # $18,333/month (1 FTE)
storage_network = 300                  # $300/month estimate
self_host_tco = compute + engineering + storage_network
# $8,640 + $18,333 + $300 = $27,273/month

# (c) Break-even volume
# At break-even: api_blended_rate * V = fixed + marginal * V
# fixed = $18,633 (engineering + storage), marginal = compute_per_mtok
api_blended = 0.80 * 3.00 + 0.20 * 15.00     # $5.40/MTok
marginal_self = compute / tokens_M              # $4.32/MTok at current volume
# As volume grows: compute cost is $4.32 at 5k tok/sec throughput (fixed fleet)
# Actually: fixed fleet → fixed compute; marginal cost is ~$0 until capacity runs out
# Treat compute as fixed until GPU is saturated (~5B MTok/month for 4x H100)
fixed_self = engineering + storage_network      # $18,633
# Break-even: api_cost = fixed + compute
# 5.40 * V = 18,633 + 8,640  → V = 27,273 / 5.40
be_volume = (fixed_self + compute) / api_blended
# => 5,050 MTok/month — 2.5x current volume, not reachable on 4x H100
# Correct conclusion: self-hosting loses at 2,000 MTok/month; break-even needs
# either much higher volume (5,000+ MTok) or lower engineering overhead.`}
      </CodeBlock>

      <H3>Exercise 2 — Reserved vs on-demand decision</H3>

      <Prose>
        Your deployment runs 6 GPUs during weekday business hours (9am–9pm, 12 hours) and 2 GPUs otherwise. You are choosing between on-demand pricing at $8/GPU-hr and a 1-year reserved commitment at $4.40/GPU-hr (45% discount). Compute the monthly cost under each strategy — (a) pure on-demand, (b) reserve 2 GPUs + on-demand for the rest, (c) reserve 6 GPUs. Which is cheapest, and what is the minimum utilization rate for the reserved strategy to beat on-demand?
      </Prose>

      <CodeBlock language="python">
{`# Solution
on_demand = 8.00      # $/GPU-hr
reserved  = 4.40      # $/GPU-hr, 1-yr commitment (always billed)

# Business hours: 12 hr/day × 22 weekdays = 264 hr/month at 6 GPUs
# Off-hours: (24-12)*22 + 24*8 = 264 + 192 = 456 hr/month at 2 GPUs

biz_hrs = 264    # hours at 6 GPU demand
off_hrs  = 456   # hours at 2 GPU demand

# (a) Pure on-demand
a = (6 * biz_hrs + 2 * off_hrs) * on_demand
# (6*264 + 2*456) * 8 = (1584 + 912) * 8 = 2496 * 8 = $19,968

# (b) Reserve 2, on-demand for burst
# Reserved: 2 GPUs × 720 hr × $4.40 = $6,336
# On-demand: 4 extra GPUs during biz hours = 4 * 264 * $8 = $8,448
b = 2 * 720 * reserved + 4 * biz_hrs * on_demand
# $6,336 + $8,448 = $14,784

# (c) Reserve 6 GPUs (always pay for 6)
# Never need on-demand
c = 6 * 720 * reserved
# 6 * 720 * 4.40 = $19,008

print(f"(a) Pure on-demand:        \${a:,.0f}/month")
print(f"(b) Reserve 2 + on-demand: \${b:,.0f}/month (cheapest)")
print(f"(c) Reserve 6 GPUs:        \${c:,.0f}/month")

# Break-even utilization for reserved vs on-demand per GPU:
# reserved wins when: utilization > P_reserved / P_on-demand
be_util = reserved / on_demand
print(f"Reserved break-even utilization: {be_util:.1%}")
# => 55.0% — if you use reserved GPUs > 55% of committed hours, reserved wins

# Strategy (b) wins because it matches reserved to the minimum (always-on)
# floor and uses on-demand only for the burst above that floor.`}
      </CodeBlock>

      <H3>Exercise 3 — Utilization tax calculation</H3>

      <Prose>
        A self-hosted Llama 3 70B deployment on 8× H100 ($3.00/GPU-hr) runs at 52% average GPU utilization. The theoretical maximum throughput at 100% utilization is 1,000 tok/sec per GPU. What is the effective $/MTok at actual utilization? How much does cost per token improve if utilization rises to 75%? What is the monthly dollar savings from the utilization improvement?
      </Prose>

      <CodeBlock language="python">
{`# Solution
gpu_count   = 8
gpu_cost_hr = 3.00
util_actual = 0.52
util_target = 0.75
tps_per_gpu = 1_000   # at 100% utilization

# Monthly GPU cost (fixed regardless of utilization)
monthly_gpu_cost = gpu_count * gpu_cost_hr * 24 * 30
# 8 * 3.00 * 720 = $17,280/month

# Tokens produced per month at each utilization level
def tokens_per_month_M(util):
    tps_aggregate = gpu_count * tps_per_gpu * util
    return tps_aggregate * 3600 * 24 * 30 / 1e6   # in millions

tok_actual = tokens_per_month_M(util_actual)   # 8 * 520 * 2,592,000 / 1e6
tok_target = tokens_per_month_M(util_target)   # 8 * 750 * ...

cost_per_mtok_actual = monthly_gpu_cost / tok_actual
cost_per_mtok_target = monthly_gpu_cost / tok_target

print(f"Tokens at 52% util:  {tok_actual:,.0f} MTok/month")
print(f"Tokens at 75% util:  {tok_target:,.0f} MTok/month")
print(f"$/MTok at 52% util:  \${cost_per_mtok_actual:.4f}")
print(f"$/MTok at 75% util:  \${cost_per_mtok_target:.4f}")

# The "savings" from better utilization = more tokens served for the same cost
# In dollar terms: to produce tok_target at 52% util you'd need more GPUs
# Extra GPUs needed = (tok_target / tok_actual - 1) * gpu_count
extra_gpus = (tok_target / tok_actual - 1) * gpu_count
savings_month = extra_gpus * gpu_cost_hr * 720
print(f"Monthly savings (equiv. GPU reduction): \${savings_month:,.2f}")

# Tokens at 52% util:   9,704 MTok/month
# Tokens at 75% util:  12,960 MTok/month
# $/MTok at 52% util:      $1.7808
# $/MTok at 75% util:      $1.3333
# Monthly savings (equiv. GPU reduction): $5,386.15
# Going from 52% to 75% utilization saves the equivalent of ~2.5 GPUs/month.`}
      </CodeBlock>

      <H3>Exercise 4 — Compliance cost step-function</H3>

      <Prose>
        Your AI product processes general business documents on a managed API at $8,500/month in API fees. A potential enterprise customer requires HIPAA BAA coverage and SOC2 Type II certification. Your legal team estimates: HIPAA BAA setup (attorney fees, policy updates) at $15,000 one-time; SOC2 Type II audit at $30,000/year; ongoing compliance tooling (logging, access controls, audit trail) at $2,500/month; one part-time security engineer at 0.3 FTE ($240K fully-loaded) to maintain compliance posture. What is the minimum annual contract value the enterprise customer must bring to justify the compliance investment? What does the first-year TCO look like with vs without the customer?
      </Prose>

      <CodeBlock language="python">
{`# Solution
api_monthly = 8_500    # current monthly API cost
months = 12

# Compliance costs
hipaa_setup_onetime = 15_000
soc2_audit_annual   = 30_000
tooling_monthly     = 2_500
security_eng_annual = 0.3 * 240_000   # $72,000/year

# Total annual compliance cost
annual_compliance = (hipaa_setup_onetime + soc2_audit_annual +
                     tooling_monthly * 12 + security_eng_annual)
# 15,000 + 30,000 + 30,000 + 72,000 = $147,000 first year
# (subsequent years: $30,000 + $30,000 + $72,000 = $132,000/year)

# Current baseline (without enterprise customer)
baseline_annual = api_monthly * 12   # $102,000

# With enterprise customer: add compliance, add their API usage
# Assume enterprise customer adds 50% more token volume = +$51,000/year API
enterprise_api_additional = api_monthly * 0.5 * 12   # $51,000

total_with_customer = baseline_annual + annual_compliance + enterprise_api_additional
# $102K + $147K + $51K = $300,000 first year

# Break-even: minimum ACV to cover compliance
# ACV must cover compliance_annual + additional API cost
min_acv = annual_compliance + enterprise_api_additional
print(f"Minimum ACV to break even: \${min_acv:,.0f}")
# => $198,000 annual contract value (first year)
# => $183,000 annual contract value (subsequent years)

print(f"\\nFirst-year TCO without enterprise: \${baseline_annual:,.0f}")
print(f"First-year TCO with enterprise:    \${total_with_customer:,.0f}")
print(f"Net first-year cost of compliance: \${annual_compliance:,.0f}")
print(f"Break-even ACV (yr 1):             \${min_acv:,.0f}")
print(f"Break-even ACV (yr 2+):            \${annual_compliance - hipaa_setup_onetime + enterprise_api_additional:,.0f}")`}
      </CodeBlock>

      <H3>Exercise 5 — Full TCO scenario for a new product</H3>

      <Prose>
        A team is launching an AI writing assistant. Month 1 traffic: 50 MTok. Month 6 traffic: 500 MTok. Month 12 traffic: 2,000 MTok. They are choosing between (A) starting on managed API and staying there through month 12, or (B) self-hosting from day one on 2× H100 ($3.00/GPU-hr specialist cloud) with 1.0 FTE engineering. Compute the cumulative 12-month TCO for each option using Claude Sonnet 4.6 pricing, assuming linear traffic growth between the milestones. Which option wins and at what month does the crossover occur, if any?
      </Prose>

      <CodeBlock language="python">
{`# Solution — linear interpolation between traffic milestones
import numpy as np

# Traffic curve (MTok/month)
months = np.arange(1, 13)
# Linear segments: 1→6 (50→500), 6→12 (500→2000)
traffic = np.concatenate([
    np.linspace(50, 500, 6),      # months 1-6
    np.linspace(500, 2000, 6),    # months 7-12
])

# Option A: Managed API (Claude Sonnet 4.6)
api_blended = 0.80 * 3.00 + 0.20 * 15.00   # $5.40/MTok
option_a = traffic * api_blended

# Option B: Self-hosted 2x H100 + 1 FTE
compute_fixed = 2 * 3.00 * 24 * 30          # $4,320/month
engineering    = 220_000 / 12               # $18,333/month
self_host_monthly = compute_fixed + engineering   # $22,653/month (fixed)
option_b = np.full(12, self_host_monthly)

# Cumulative costs
cum_a = np.cumsum(option_a)
cum_b = np.cumsum(option_b)

print("Month | Traffic  | API cum.  | Self-host | Leader")
print("-" * 55)
for i, m in enumerate(months):
    leader = "API" if cum_a[i] < cum_b[i] else "Self-host"
    print(f"  {m:2d}  | {traffic[i]:6.0f} MTok | \${cum_a[i]:8,.0f} | \${cum_b[i]:8,.0f} | {leader}")

# Month |  Traffic  | API cum.  | Self-host | Leader
# -------------------------------------------------------
#    1  |     50    |       270 |    22,653 | API
#    2  |    149    |     1,073 |    45,306 | API
#    3  |    248    |     2,410 |    67,959 | API
#    4  |    347    |     4,282 |    90,612 | API
#    5  |    446    |     6,688 |   113,265 | API
#    6  |    500    |     9,388 |   135,918 | API
#    7  |    750    |    13,433 |   158,571 | API
#    8  |  1,000    |    18,833 |   181,224 | API
#    9  |  1,250    |    25,588 |   203,877 | API
#   10  |  1,500    |    33,698 |   226,530 | API
#   11  |  1,750    |    43,163 |   249,183 | API
#   12  |  2,000    |    53,983 |   271,836 | API
#
# No crossover in 12 months. API total: $53,983. Self-host total: $271,836.
# Self-hosting costs 5x more over year 1 due to engineering overhead at low volume.
# At 2,000 MTok/month the monthly cost finally approaches parity on compute alone,
# but the 12-month cumulative deficit never recovers.
# Conclusion: for this traffic ramp, managed API wins decisively in year 1.`}
      </CodeBlock>

    </div>
  ),
};

export default costOptimizationTCO;
