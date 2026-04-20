import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { Plot } from "../../components/viz";

const queueingTheoryLLMServing = {
  title: "Queueing Theory for LLM Serving",
  readTime: "10 min",
  content: () => (
    <div>
      <Prose>
        LLM serving is a queueing system. Requests arrive at some rate, the GPU serves them at some rate, and if the two are misaligned you either idle expensive hardware or pile up queues that make every user wait. The job of capacity planning is to find the operating point where neither failure mode dominates. Classical queueing theory — the mathematics that predicts telephone-exchange waiting times, developed in the 1910s by A.K. Erlang — turns out to be the right starting tool for that job. It was built for Poisson call arrivals and exponential hold times, which are not quite the right model for token generation, but the qualitative predictions transfer with surprising fidelity. Not the whole answer, but a good starting one.
      </Prose>

      <H2>Little's Law — the one equation to remember</H2>

      <Prose>
        Little's Law states that the average number of items in a stable system equals the arrival rate times the average time each item spends in the system.
      </Prose>

      <MathBlock>{"L = \\lambda \\cdot W"}</MathBlock>

      <Prose>
        For an LLM endpoint: <Code>L</Code> is the average number of concurrent requests being processed (in queue or actively generating tokens), <Code>λ</Code> is the request arrival rate in requests per second, and <Code>W</Code> is the average end-to-end latency per request. The law holds unconditionally — it requires no assumptions about the distribution of arrival times, service times, or scheduling policy. It is a pure consequence of flow balance in any steady-state system. You cannot violate it; you can only wait for a system to stop being in steady state.
      </Prose>

      <Prose>
        The practical use is direct. If you want to serve 100 requests per second at 2-second average latency, Little's Law says you need capacity for <Code>100 × 2 = 200</Code> concurrent requests. If your GPU can only hold 50 concurrent KV caches before running out of memory, then one of three things must happen: the arrival rate drops, the latency rises, or requests are rejected. The equation doesn't care which — it will balance regardless. What queueing theory adds on top of Little's Law is predictions about how the system reaches that balance, how fast queues grow, and how far you can push utilization before latency starts to blow up.
      </Prose>

      <H2>M/M/1 — the simplest useful model</H2>

      <Prose>
        The M/M/1 queue is the textbook starting point: Poisson arrivals (the first M), exponentially distributed service times (the second M), and a single server (the 1). Utilization is defined as <Code>ρ = λ/μ</Code>, where <Code>μ</Code> is the average service rate. For the system to be stable, <Code>ρ</Code> must be strictly less than one — otherwise the queue grows without bound. The expected wait time for a request that enters the system is:
      </Prose>

      <MathBlock>{"W = \\frac{1}{\\mu - \\lambda} = \\frac{1}{\\mu(1 - \\rho)}"}</MathBlock>

      <Prose>
        The denominator <Code>μ(1 − ρ)</Code> is the key. At low utilization the denominator is close to <Code>μ</Code> and wait time is just the service time. As <Code>ρ</Code> approaches 1, the denominator collapses toward zero and wait time diverges to infinity. This is not a numerical artifact — it reflects the real behavior of any system where variance in arrivals or service times means that any brief overload generates a queue that then takes time to drain. The graph makes the shape visceral.
      </Prose>

      <Plot
        label="m/m/1 — latency vs utilization"
        width={520}
        height={260}
        xLabel="utilization (ρ)"
        yLabel="expected wait (1/μ units)"
        series={[
          { name: "M/M/1 expected wait", points: [[0.1, 0.11], [0.3, 0.43], [0.5, 1], [0.7, 2.33], [0.8, 4], [0.9, 9], [0.95, 19], [0.98, 49]] },
        ]}
      />

      <Prose>
        The curve is flat through roughly <Code>ρ = 0.5</Code>, then begins bending upward, then goes nearly vertical as <Code>ρ → 1</Code>. This shape is why capacity planners do not target 100% utilization. At 90% utilization, expected wait is already nine times the service time. At 95%, it is nineteen times. The headroom you leave — deliberately underloading the system — is not waste; it is what keeps the latency distribution from blowing up when arrival variance causes transient overload. Typical operating targets for LLM serving endpoints sit around <Code>ρ = 0.7</Code> to <Code>0.8</Code>, which keeps expected wait in the two-to-four times service-time range and leaves room for p99 latency to stay tolerable.
      </Prose>

      <H2>Why LLM serving doesn't fit M/M/1 cleanly</H2>

      <Prose>
        The M/M/1 model gets the shape right but misses several structural features of LLM workloads that matter for accurate planning. Three departures stand out.
      </Prose>

      <Prose>
        First, service time is bimodal rather than exponential. LLM inference has two distinct phases: prefill, which processes the input prompt in one parallel forward pass and costs O(prompt length × model size) in compute, and decode, which generates tokens autoregressively one at a time and costs O(response length × model size). A request with a 2,000-token prompt and a 50-token response has a service profile dominated by a single expensive prefill step. A request with a 200-token prompt and a 2,000-token response is dominated by a long decode sequence. These two shapes behave differently under batching, produce different memory pressure patterns, and have very different sensitivity to concurrency. Treating them as draws from a single exponential service distribution throws away most of the structure.
      </Prose>

      <Prose>
        Second, continuous batching makes the effective service rate <Code>μ</Code> a dynamic function of load rather than a constant. In a standard GPU decode pass, adding another request to an active batch costs relatively little — the GPU is already memory-bandwidth bound, and many additional decodes can share the same memory read for free. This means that as more requests are batched together, each individual request gets slightly slower (because KV cache reads are still finite) but the aggregate system throughput increases. The "<Code>μ</Code>" in Little's Law is not a fixed parameter; it shifts with the batch size, which shifts with the load. A model that assumes constant service rate will underestimate throughput at moderate load and overestimate it near capacity.
      </Prose>

      <Prose>
        Third, KV cache capacity is a hard admission-control constraint that no standard M/M model captures. When GPU memory is exhausted, the system cannot accept new requests regardless of how many GPUs there are or how efficient the scheduler is — new arrivals must either queue on CPU, wait for existing generations to complete and free cache slots, or be rejected outright. This creates a discrete cliff in the service envelope that classical queueing theory, which assumes unlimited buffer capacity by default, was not designed to represent.
      </Prose>

      <H3>Better models — M/G/c and processor sharing</H3>

      <Prose>
        For workloads that are prefill-dominated — long prompt, short response — the M/G/c queue is a closer fit. The G stands for general service time distribution, which lets you model the bimodal prefill/decode split more faithfully. The c means multiple servers, mapping to a fleet of GPUs. M/G/c doesn't have a closed-form expression as clean as M/M/1, but the Pollaczek-Khinchine formula gives mean wait time as a function of the service time mean and variance, and the qualitative behavior is similar: utilization is the dominant driver, with high variance in service time making tail latency worse at any fixed <Code>ρ</Code>.
      </Prose>

      <Prose>
        For decode-dominated workloads with continuous batching, processor-sharing queues are the right abstraction. In a processor-sharing model, all active requests share server capacity proportionally — exactly what continuous batching does during the decode phase when the GPU processes one token from each active generation per step. Processor-sharing queues are known to have worse average latency than first-come-first-served at the same utilization, but much better fairness: no single long request can monopolize the server and starve shorter ones. The math is uglier than M/M/1 but the qualitative predictions for capacity planning, tail latency, and admission thresholds stay in the same family.
      </Prose>

      <H2>Tail latency and p99</H2>

      <Prose>
        Average latency is not what users feel. They feel the tail — the slowest 1% of their requests, which in production is typically three to five times the mean. Under M/M/1, p99 latency grows faster than linearly as <Code>ρ → 1</Code>: when the mean is blowing up, the distribution is also widening, so the 99th percentile gets worse faster than the average does. For LLM serving, the tail is made worse by a specific mechanism: prefill blocking. When a new request with a long prompt arrives, its prefill step monopolizes GPU compute for the duration of that pass, delaying the next decode step for every other active request in the batch. A single 10,000-token prefill can add hundreds of milliseconds of latency to dozens of concurrently decoding requests. The mean barely moves; the p99 lurches.
      </Prose>

      <Prose>
        Modern LLM serving schedulers treat p99 as the primary objective, not an afterthought. Chunked prefill — splitting long prompt processing into fixed-size chunks interleaved with decode steps — bounds the maximum latency impact of any single arriving request. Priority queues in systems like SGLang allow shorter requests to preempt decode steps from longer ones, reducing tail latency at the cost of slightly lower throughput. These are not magic solutions; they trade one part of the latency distribution against another. But they are explicit moves against the p99 problem rather than optimizations of the mean that happen to improve p99 incidentally.
      </Prose>

      <H3>Capacity planning — practical recipe</H3>

      <Prose>
        Translating queueing theory into a GPU budget estimate requires combining Little's Law with two empirically measured quantities: how many concurrent requests a single GPU can hold in KV cache, and how many tokens per second that GPU can sustain at peak concurrency. The function below is a rough first-pass calculator that combines both constraints and then scales up by the utilization target to leave the headroom the M/M/1 curve demands.
      </Prose>

      <CodeBlock language="python">
{`def estimate_capacity(
    target_rps: float,            # requests per second
    target_latency_p50_sec: float,
    tokens_per_request: float,     # avg tokens generated per request
    model_decode_tps_per_gpu: float,  # avg decode tokens/sec at max concurrency
    concurrent_requests_per_gpu: int,  # KV-cache-limited
) -> dict:
    """Rough capacity planning for an LLM endpoint."""
    # Little's Law: L = lambda * W
    expected_concurrent = target_rps * target_latency_p50_sec

    # Per-GPU capacity in concurrent requests
    gpus_needed = max(
        expected_concurrent / concurrent_requests_per_gpu,
        (target_rps * tokens_per_request) / model_decode_tps_per_gpu,
    )
    return {
        "expected_concurrent": expected_concurrent,
        "gpus_needed_ceiling": int(gpus_needed + 0.5),
        "planned_utilization_target": 0.7,
        "actual_gpus_for_p99": int(gpus_needed / 0.7 + 0.5),
    }`}
      </CodeBlock>

      <Prose>
        The <Code>max</Code> in the middle is the key. Two independent constraints bind the GPU count: KV cache capacity (can the GPU hold enough concurrent states?) and throughput (can the GPU generate tokens fast enough?). Whichever constraint is tighter determines the floor. The final line divides by 0.7 — the planned utilization target — to get the number of GPUs you actually provision, as opposed to the minimum that would barely keep the queue stable. That 30% headroom is not conservatism for its own sake; it is what the M/M/1 curve requires to keep p99 latency from tracking more than three or four times the mean.
      </Prose>

      <H2>Queueing theory as a modeling language</H2>

      <Prose>
        The math is approximate. Real LLM serving has bimodal service times, dynamic batch sizes, hard memory constraints, and scheduling policies that none of the classical models capture precisely. But the approximations are useful in a specific way: they identify the shape of how a system breaks, and the shape is robust even when the numbers are off. Little's Law sets hard constraints that no scheduling trick can circumvent — if you want lower latency and higher throughput simultaneously, you need more hardware. The <Code>ρ → 1</Code> latency divergence correctly predicts that running GPUs at 98% utilization will produce catastrophic p99 latency in production, regardless of how efficient your scheduler is. Processor-sharing models correctly predict that continuous batching doesn't make latency linear in load — it shifts the failure mode from starvation to queue depth, but the failure is still there.
      </Prose>

      <Callout accent="gold">
        Queueing theory doesn't tell you exact latencies; it tells you the shape of how your system breaks. That shape is the thing a good SRE plans against.
      </Callout>

      <Prose>
        The value is the intuition transferred to new situations. When a new batching algorithm claims 2× throughput improvement, queueing theory immediately asks: does that improve <Code>μ</Code>, reduce variance in service time, or increase the KV cache capacity? Each change has a different effect on the latency distribution and on where the ρ-versus-latency curve bends. When a new memory management technique claims to increase concurrency by 40%, Little's Law says you need to know whether latency also drops or whether you are just accepting more in-flight requests at the same average wait — those are different outcomes despite the same throughput number.
      </Prose>

      <Prose>
        The next topics — speculative decoding, prefix caching, inference cost economics — each change the parameters of this model but not its underlying structure. Speculative decoding increases effective <Code>μ</Code> for short responses. Prefix caching reduces effective prefill cost and therefore service time variance. Inference cost economics are just the integral of the GPU-time implied by the queueing model over a billing period. Understand the queueing first, and the individual optimizations stop being a list of tricks and start making sense as moves in a well-defined optimization game.
      </Prose>
    </div>
  ),
};

export default queueingTheoryLLMServing;
