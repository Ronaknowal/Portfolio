import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { Plot } from "../../components/viz";

const costOptimizationTCO = {
  title: "Cost Optimization & TCO Analysis",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        The previous topic on inference cost economics covered per-request pricing — the dollars-per-million-tokens equation and the levers that sit inside a single inference call. This one zooms out: what is the total cost of running an LLM product across its entire infrastructure, and which levers actually move the needle at the organization level? For any serious deployment, the bill has five or six distinct line items, and the ratios between them determine where optimization effort pays off. Treating inference compute as the only cost is the most common mistake in early-stage LLM budgeting, and it leads to optimizing the wrong things.
      </Prose>

      <H2>The TCO decomposition</H2>

      <Prose>
        A mature LLM infrastructure bill breaks down into a handful of recurring categories. The exact percentages shift with architecture choices, traffic patterns, and team size, but the rough shape is consistent across deployments that have reached scale.
      </Prose>

      <CodeBlock>
{`line item                           typical % of total   drivers
inference compute (GPU hours)       55-75%              traffic volume, model size, utilization
API costs (if using third-party)    varies              same, on commercial pricing
storage (weights, KV cache, logs)   2-8%                model count, log retention
network (ingress/egress)            3-10%               streaming, multi-region
observability (logs, metrics)       2-5%                eval volume, log retention
staff (mlops, ml infra)             varies              scale of the stack
data / eval / prompt engineering    varies              quality bar, product maturity`}
      </CodeBlock>

      <Prose>
        Inference compute dominates in every self-hosted deployment that reaches meaningful traffic. API costs substitute for that line item if you are on a managed provider, at commercial margins on top of the underlying hardware cost. Storage is easy to underestimate at early scale: a production deployment holding weights for multiple models, maintaining a KV cache across long sessions, and retaining logs for compliance and eval ends up with a surprisingly large persistent storage bill. Network costs surface most aggressively in streaming deployments — each token streamed to the client costs egress, and real-time streaming at high concurrency adds up. Observability and eval compute are rarely budgeted explicitly in early deployments but grow to 10-20% of serving cost as quality discipline matures. Staff and data costs are highly variable and context-dependent, but both tend to be undercounted in initial TCO models.
      </Prose>

      <H2>The 80/20 of optimization</H2>

      <Prose>
        In practice, a small number of high-leverage moves account for the majority of achievable TCO savings. Everything else — kernel tuning, custom hardware, advanced quantization schemes — matters at hyperscale but produces marginal returns for most deployments.
      </Prose>

      <Prose>
        <strong>Tiered model routing</strong> is consistently the highest-leverage intervention. Route easy requests to small models, hard requests to capable ones, and calibrate the routing threshold to your quality requirements. Typical savings are 40-70% on inference compute, because a large fraction of most production workloads consists of requests that a small model handles well. Routing is covered in depth in an earlier topic; the point here is that it usually produces the largest single reduction in the overall bill.
      </Prose>

      <Prose>
        <strong>Prefix caching</strong> produces large wins specifically on agent and chat workloads. Any workload where a significant fraction of every request is shared with prior requests — system prompts, tool definitions, policy documents, conversation history preambles — benefits from caching the KV state for that shared prefix. Typical savings are 20-40% on input cost for repetitive traffic. The effort required to implement it is low; it is largely a configuration change on the serving stack or an API-level feature flag.
      </Prose>

      <Prose>
        <strong>Context length discipline</strong> is the easiest lever and the most frequently neglected. Long-context models encourage prompt inflation. Summarize old conversation turns once they exceed a threshold. Retrieve relevant document chunks rather than injecting entire documents. Prune tool definitions to the subset that is actually usable in the current context. These practices typically reduce median input length by 30-70% with no quality loss and sometimes with quality gain. Shorter, focused contexts tend to produce better attention patterns on the relevant content.
      </Prose>

      <Prose>
        <strong>Reserved or spot capacity</strong> replaces on-demand pricing for predictable or deferrable workloads. Committing to baseline GPU capacity on a one- to three-year reservation typically yields 40-60% off on-demand pricing. Spot instances for batch work and latency-tolerant workloads offer 60-80% off. The combination of reserved baseline plus on-demand burst plus spot batch produces typical savings of 20-40% over pure on-demand across the aggregate bill.
      </Prose>

      <Prose>
        <strong>Quantization</strong> — FP8 or INT4 inference — reduces the memory footprint and compute cost of serving. Typical savings are 20-50% on compute capacity for equivalent throughput. Modern quantization at FP8 preserves quality well enough for most production workloads; INT4 requires more careful evaluation against a quality threshold but is viable for many tasks. Below these five levers, the marginal returns drop sharply. Kernel-level tuning and custom hardware matter at hyperscale but represent significant engineering investment for gains that are already partially captured by well-maintained open-source serving stacks.
      </Prose>

      <H2>Self-host vs API — the cross-over model</H2>

      <Prose>
        The self-hosting versus managed API decision recurs at every stage of an LLM product's growth. The economic structure is straightforward: self-hosting carries large fixed costs in GPU capacity and MLOps engineering, with low marginal cost per token once that infrastructure is in place. Managed APIs have near-zero fixed costs and higher marginal costs that include the provider's margin over raw hardware. The break-even point shifts with traffic volume, and below it the API wins; above it, self-hosting wins — assuming you can utilize the reserved capacity efficiently.
      </Prose>

      <Plot
        label="self-host vs api break-even (illustrative, 8b model class)"
        width={520}
        height={240}
        xLabel="requests per day (thousands)"
        yLabel="monthly cost ($k)"
        series={[
          { name: "api (pay-per-token)", points: [[1, 0.5], [10, 5], [50, 25], [100, 50], [500, 250]] },
          { name: "self-host (fixed + marginal)", points: [[1, 15], [10, 17], [50, 25], [100, 35], [500, 110]] },
        ]}
      />

      <Prose>
        The illustrative break-even for an 8B-class model in 2025 is around 50,000 requests per day at average token lengths. That threshold shifts in both directions depending on model size — for 70B-class models the fixed cost of self-hosting is higher but the per-token API premium is also higher, so the cross-over typically comes earlier in absolute terms. It also shifts with GPU pricing trends, your ability to fill reserved capacity, and whether the workload is steady enough to avoid paying for unused reserved instances. Below the break-even, the API is almost always the right answer when engineering cost is included in the comparison. Above it, the decision depends on whether your team has the MLOps capability to operate the self-hosted stack efficiently.
      </Prose>

      <H3>Reserved, on-demand, and spot</H3>

      <Prose>
        Cloud GPU capacity is available at three price points, and a well-structured production deployment uses all three. On-demand instances are billed by the hour with no commitment, at the highest per-unit price — they are the right choice for variable or unpredictable peak traffic. Reserved instances require committing to N GPUs for one to three years at 40-60% off on-demand pricing; they are the right choice for baseline load that is predictable and stable. Spot instances are reclaimed cloud inventory offered at 60-80% off on-demand but with the provider's right to reclaim them with minutes of notice — they are the right choice for batch inference, offline eval runs, and any workload that can checkpoint and resume.
      </Prose>

      <Prose>
        The standard production pattern is reserved capacity for the predictable baseline, on-demand for traffic spikes, and spot for all batch work. Compared to pure on-demand across all workloads, this approach saves 30-50% on the aggregate GPU bill. The main operational requirement is that the baseline load must be reliably predictable — committing to reserved capacity that ends up underutilized is worse than on-demand, because you pay for the reservation regardless.
      </Prose>

      <H2>Context-length discipline</H2>

      <Prose>
        This is the easiest unit-cost lever, and it is consistently underutilized. Long-context models create an engineering anti-pattern: the simplest implementation of any feature that needs context is to inject all of it. Entire conversation histories, complete documents, every tool definition in the system, the full policy manual. The model can technically accept it. The cost compounds linearly with every token injected, on every request, for the lifetime of the product.
      </Prose>

      <Prose>
        Good production hygiene looks different. Conversation history gets summarized once it exceeds a configurable turn count — the last N turns are kept verbatim, earlier turns as a compressed summary. Documents are chunked and the relevant chunks are retrieved at query time rather than injected wholesale. Tool definitions are pruned to the subset that is actually relevant to the current task rather than included exhaustively. These are not performance optimizations in the traditional sense; they are prompt discipline. The engineering effort is modest — one summarization call, one retrieval call, one filtering pass on the tool list. The savings are consistent: reducing median input length by 30-70% maps directly to a 30-70% reduction in input token cost, which at a typical output-to-input ratio of 1:4 by token count translates to a 20-50% reduction in total per-request cost. Quality is preserved and often improved, because shorter contexts that focus on the relevant information produce better attention on the parts that matter.
      </Prose>

      <H3>Where savings come from operationally</H3>

      <Prose>
        The sequence in which optimizations should be pursued is not intuitive. The temptation is to start with infrastructure — better hardware, custom kernels, deployment topology. The return on that work is real but modest until the higher-level problems are addressed. A practical ordering by effort-to-return ratio looks like this:
      </Prose>

      <Prose>
        First, instrument cost attribution. Measure cost per request and cost per user. You cannot optimize what you cannot attribute, and most teams running their first LLM products have no idea which endpoints or user segments are driving the bill. This step is pure observability work, not optimization, but it determines where all subsequent effort goes.
      </Prose>

      <Prose>
        Second, deploy prefix caching and tiered routing. Both are low-effort configuration changes relative to the returns they produce. Prefix caching is often a one-line change in serving configuration or an API parameter. Routing requires a classifier but the classifier can be lightweight.
      </Prose>

      <Prose>
        Third, audit prompt discipline. Walk through every production prompt template and ask what can be removed or retrieved lazily. This is sometimes the highest-return intervention of all, and it costs only engineering review time.
      </Prose>

      <Prose>
        Fourth, shift baseline capacity from on-demand to reserved and batch workloads to spot. This requires confidence in traffic forecasts but no changes to the model or serving stack.
      </Prose>

      <Prose>
        Only then: quantization, custom serving configurations, hardware negotiation. These are real gains but smaller per unit of engineering effort than the four preceding steps.
      </Prose>

      <Callout accent="gold">
        Most LLM cost problems are not infrastructure problems. They are prompt, routing, and caching problems — all of which are above the stack and cheaper to fix.
      </Callout>

      <H2>The non-obvious costs</H2>

      <Prose>
        Several cost categories rarely appear on the GPU bill but shape total cost of ownership meaningfully, especially at the quality and compliance levels that enterprise deployments require.
      </Prose>

      <Prose>
        <strong>Eval compute</strong> is the most consistently underestimated. Running automated evaluations on every model update or prompt change requires inference compute — you are running the model against an eval set, possibly multiple times with different configurations. A mature eval pipeline running on every deploy can consume 10-20% of serving compute. This is not waste; it is quality insurance. But it should be budgeted explicitly, and it should be included in the total cost model from the beginning rather than discovered after the eval infrastructure is already built.
      </Prose>

      <Prose>
        <strong>Data and labeling</strong> become significant costs if the deployment involves fine-tuning or self-training. Human annotation for preference data, task-specific fine-tuning datasets, and red-teaming exercises involve significant labor. For many fine-tuning projects, the human labeling cost exceeds the GPU cost of the training run itself. This is often invisible in infrastructure cost models because it falls under a different budget line.
      </Prose>

      <Prose>
        <strong>Oncall and debugging</strong> are real operational costs that scale with the complexity of the stack. LLM systems fail in subtle ways — prompt injection, latent context corruption, model behavior drift after a provider update, KV cache poisoning in long sessions. These failures require engineering time to diagnose and often have no obvious error signal. SRE capacity for an LLM product is a legitimate line item in the TCO model.
      </Prose>

      <Prose>
        <strong>Compliance costs</strong> scale sharply with enterprise customer requirements. Data residency requirements constrain where inference can run, often forcing single-region deployments that sacrifice utilization efficiency. SOC2 Type II and similar certifications require audit tooling, log retention, and access control infrastructure. For products targeting regulated industries — healthcare, financial services, legal — compliance infrastructure can represent a substantial fraction of engineering effort and ongoing operational cost.
      </Prose>

      <Prose>
        Cost optimization at the organization level is rarely about a single clever intervention. It is about enforcing discipline at six or seven layers simultaneously — routing, caching, context hygiene, capacity management, eval efficiency, compliance scoping — and maintaining enough observability to know where the next marginal dollar is best spent. The next topic — edge and on-premise deployment — is where several of these tradeoffs shift fundamentally, as the fixed-cost and marginal-cost structure of self-hosting moves to the extreme end of the spectrum.
      </Prose>
    </div>
  ),
};

export default costOptimizationTCO;
