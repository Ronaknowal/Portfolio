import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";

const autoscalingGPU = {
  title: "Autoscaling & GPU Resource Management",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        Autoscaling an LLM fleet is harder than autoscaling a web-server farm, and the difficulty is not incidental. Three specific properties of GPU-based LLM serving combine to make the classical approach — watch CPU utilization, add instances when it gets high, remove them when it drops — fail in ways that are worth enumerating precisely before reaching for a fix.
      </Prose>

      <Prose>
        First, cold-start times are measured in minutes rather than seconds. Loading a 70B-parameter model from blob storage into GPU memory is not like pulling a Node process out of an image cache. It is tens of gigabytes of weights moving over interconnects at whatever bandwidth the storage tier permits, followed by initialization passes that touch every parameter. The median time from "trigger scaling event" to "instance is accepting requests" sits between two and ten minutes depending on infrastructure. Second, GPUs are expensive enough that idle capacity is a real problem, not a rounding error. An H100 running at zero utilization costs the same per hour as one running at a hundred percent. The margin on idle GPU time is negative. Third, LLM traffic patterns are burstier than typical web traffic in ways that compound the first two problems. Agent workloads emit large correlated bursts. A single viral moment can multiply traffic tenfold overnight. The gap between "steady state demand" and "peak demand" is often an order of magnitude, and it can close in minutes. Those three constraints together define the design space. The naive autoscaler optimized for web servers is wrong not because it was carelessly designed but because it was designed for a different problem.
      </Prose>

      <H2>Why the classical signals don't work</H2>

      <Prose>
        GPU utilization, as reported by <Code>nvidia-smi</Code> or the equivalent cloud metrics API, is the most intuitive candidate for an autoscaling signal. It is also one of the most misleading ones in this context. A vLLM worker can be fully loaded — KV cache nearly exhausted, all available GPU memory bandwidth consumed by decode passes, admission control about to start rejecting requests — while <Code>nvidia-smi</Code> reports something in the range of forty to fifty percent utilization. The metric measures whether the streaming multiprocessors have work scheduled, not whether they are the bottleneck. Decode-bound serving is memory-bandwidth-bound, not compute-bound. The SMs are spending most of their wall-clock time waiting for HBM reads to complete. That wait does not register as utilization. At that same forty-five percent reading, adding one more request may push KV cache occupancy past the eviction threshold and cause cascading latency spikes for everyone already in the batch. GPU utilization says "room to spare." The actual system says "full."
      </Prose>

      <Prose>
        CPU utilization is worse as an autoscaling signal, not better. On an LLM inference worker, the CPU is mostly idle. Tokenization, batch assembly, and scheduling are negligible compared to the GPU-side compute. A worker saturated at the GPU level, with a full KV cache and a growing request queue, may report two to five percent CPU utilization. An autoscaler watching CPU will conclude the fleet is lightly loaded and sit on its hands while users wait. Request-rate thresholds are marginally better but still miss the core pathology: a fleet receiving moderate request rate but serving long-context requests can be more overloaded than one receiving high request rate on short prompts, because load is denominated in KV blocks, not in requests per second.
      </Prose>

      <H2>The right signals</H2>

      <Prose>
        Three metrics predict when to scale in ways that the classical signals do not. Each corresponds to a different failure mode, and a composite trigger watching all three catches situations that any single signal would miss.
      </Prose>

      <Prose>
        KV cache utilization — the fraction of GPU memory currently allocated to active KV blocks — is the most direct measure of whether the serving system is approaching its capacity limit. At sustained utilization above eighty percent, admission control in most inference engines begins rejecting or queuing incoming requests. That is the correct moment to have already started a new instance, not the moment to start looking for one. Queue depth — the number of requests waiting for capacity — should stay near zero in healthy steady state. When it sustains above ten, the fleet is running behind demand and the gap compounds over time rather than clearing itself. Time-to-first-token at the ninety-ninth percentile is the user-visible expression of both of those problems. TTFT p99 crossing two seconds is a meaningful SLO breach on most products, and it almost always reflects one of the two underlying conditions. Scaling on TTFT p99 catches the cases where KV cache and queue depth look acceptable on aggregate but individual requests are being starved.
      </Prose>

      <CodeBlock language="python">
{`# Composite scaling signal — fire if ANY threshold breached
def should_scale_up(metrics):
    return (
        metrics.kv_cache_utilization_p95 > 0.75 or
        metrics.queue_depth_p95 > 5 or
        metrics.ttft_p99_ms > 2000
    )

def should_scale_down(metrics):
    return (
        metrics.kv_cache_utilization_p95 < 0.30 and
        metrics.queue_depth_p95 == 0 and
        metrics.ttft_p99_ms < 800 and
        metrics.instance_count > min_instances
    )`}
      </CodeBlock>

      <Prose>
        The asymmetry between scale-up and scale-down thresholds is deliberate and important. Scale-up fires on any single breach; scale-down requires all conditions to hold simultaneously. That asymmetry reflects the asymmetric cost of the two errors: over-provisioning by one instance for an hour costs money. Under-provisioning for a minute drops requests. The thresholds themselves — seventy-five percent KV utilization to scale up, thirty percent to scale down — are not universal constants but a starting point that most production stacks tune to their specific traffic patterns.
      </Prose>

      <H2>Cold start — the defining constraint</H2>

      <Prose>
        Loading a 70B-parameter model from blob storage to GPU memory is not a fast operation. In BF16, 70B parameters occupy roughly 140 GB. Even a modern storage tier with multi-gigabyte-per-second throughput to GPU memory requires a minimum of a few minutes for that transfer, and on configurations that route through slower network paths the number approaches ten. The GPU initialization passes that follow — loading the CUDA context, allocating the KV cache pool, running a warmup pass to JIT-compile the attention kernels — add additional time. A new instance triggered at the moment demand exceeds capacity will not be serving requests for several minutes after the trigger fires.
      </Prose>

      <Prose>
        That gap between trigger and availability shapes how the autoscaler has to behave. Four mitigations exist, each with different cost profiles. Predictive scaling uses historical demand patterns — time-of-day curves, day-of-week effects, known events — to pre-warm instances before the spike arrives. A fleet that starts a new instance fifteen minutes before the morning traffic ramp costs slightly more in pre-warming time than one that reacts exactly at threshold, but avoids the cold-start window entirely. Faster model loading addresses the root cause directly: shared weight pools keep model weights pre-staged in host memory or on fast NVMe, RDMA loading moves weights from storage to GPU without CPU involvement, and safetensors memory-mapped files allow demand-paged loading that makes the first request faster even if full loading takes time. Always-warm pools — keeping N instances running even when traffic does not require them — pay the idle GPU cost continuously in exchange for eliminating the cold-start penalty entirely. For latency-sensitive products where a ten-minute outage window during a traffic spike is unacceptable, the always-warm pool is often the right trade. The idle cost is quantifiable and predictable; the cost of dropped requests during a viral moment is not.
      </Prose>

      <H3>Spot and preemptible instances</H3>

      <Prose>
        GPU spot instances on major cloud providers are priced forty to seventy percent below on-demand rates, reflecting the provider's ability to reclaim them with minutes of notice when reserved capacity is needed. For batch workloads — offline inference, embedding generation, evaluation runs — spot instances are attractive. For user-facing serving, the calculus is harder. A reclaimed instance kills every in-flight request it is handling. Requests need to be detected as lost, re-routed to surviving instances, and re-executed from the beginning or from a checkpoint. That machinery is buildable but adds meaningful complexity to the request routing layer. The production pattern that balances cost and availability is a split fleet: baseline capacity on on-demand instances that cannot be reclaimed, burst capacity on spot that can be interrupted. The on-demand tier handles the traffic the product cannot afford to drop; the spot tier handles overflow where occasional request failures are tolerable or where the retry mechanism makes the failure invisible to the user. This requires the fleet to be explicitly designed for instance loss, with per-request timeouts, dead-instance detection, and re-routing implemented in the load balancer rather than assumed away.
      </Prose>

      <H3>Horizontal vs vertical scaling</H3>

      <Prose>
        For LLM serving, horizontal scaling — more instances of the same GPU configuration — is the default answer to increased demand, and it is right for the majority of throughput-oriented deployments. More instances means more KV cache memory pooled across a larger concurrent request set, more independent prefill capacity, and more decode bandwidth available in parallel. The economics favor horizontal: commodity GPU nodes are easier to provision than exotic large-memory configurations, and marginal throughput scales linearly with instances once the architecture fits on a single node. Vertical scaling matters in two specific situations that horizontal cannot address. The first is models that do not fit in a single GPU's memory and require tensor-parallel splits across multiple devices: the per-instance topology — how many GPUs per node, how wide the tensor-parallel group — determines whether the model is runnable at all, and vertical scaling here means choosing the right per-instance hardware size rather than adding more instances. The second is latency-sensitive serving where per-token decode speed matters more than aggregate throughput. A 70B model on two H100s with the weights split tensor-parallel has lower decode latency per request than the same model on one H100 because more GPU memory bandwidth is available per decode step. For most serving workloads the throughput-optimal configuration wins; for real-time voice interfaces or latency-SLO-bound products the per-token latency case for larger per-instance GPU counts is real.
      </Prose>

      <H2>Scheduling and bin packing</H2>

      <Prose>
        A fleet serving multiple model sizes does not have a simple one-to-one mapping between requests and instances. A 70B model requires two H100s in a typical tensor-parallel configuration; an 8B model runs comfortably on a fraction of one H100 with memory to spare. If the scheduler assigns one model per node without regard for remaining capacity, the 8B serving instances are running on hardware that is seventy or eighty percent idle from a memory perspective while 70B requests queue for their pair of dedicated nodes. Bin packing — placing multiple workloads on each physical GPU to improve utilization — recovers that waste. NVIDIA's Multi-Process Service allows multiple CUDA processes to share a single GPU, each with its own memory and compute allocation. Kubernetes with GPU partitioning enables coarser-grained sharing at the node level. Custom schedulers built around Triton or vLLM track KV cache occupancy per instance and route requests to the instances with the most headroom, effectively doing online bin packing without explicit partition assignment. The gap between a naive scheduler and a bin-packing one in a mixed-model fleet is typically a two to three times difference in hardware utilization at the same quality-of-service level. That gap translates directly to infrastructure cost, and closing it is most of what sophisticated cluster schedulers are selling.
      </Prose>

      <H3>Quota and isolation</H3>

      <Prose>
        A shared GPU fleet serving multiple customers or internal workloads runs into the same multi-tenancy problems as every shared compute layer, now with GPUs as the primary scarce resource and KV cache occupancy as the secondary one. Fair-share scheduling prevents one tenant from consuming the full fleet during off-peak hours in a way that leaves no headroom when others arrive. Per-tenant quotas set explicit ceilings so that a single large agent workload does not crowd out interactive users. Priority tiers — reserved capacity for latency-sensitive workloads, best-effort for batch — require the scheduler to preempt or defer lower-priority requests when higher-priority ones arrive. The implementation in most production stacks lives partly in the load balancer, partly in the inference engine's admission control, and partly in an external quota service that tokens against a central ledger. Rate limiting — covered separately in this section — is the user-facing surface of the same problem: a per-tenant KV cache quota expressed as a tokens-per-minute ceiling rather than a memory allocation.
      </Prose>

      <H2>The autoscaling feedback loop</H2>

      <Prose>
        A less obvious hazard in production autoscaling is the oscillation it creates when configured without dampening. The sequence is predictable in retrospect but easy to miss in the design phase: queue depth rises above threshold, autoscaler fires and adds instances, new capacity comes online and clears the queue, latency drops to well below the scale-down threshold, autoscaler removes instances, queue begins building again, repeat. The period of this oscillation is set by the cold-start time plus the scale-down cooldown, and if those are short the system hunts constantly between under- and over-provisioned states, spending GPU hours spinning up and down and delivering variable latency to users throughout. Four standard mitigations reduce oscillation amplitude. Cooldown periods — blocking all scaling actions for a fixed window after the last one — are the simplest and most common; fifteen to thirty minutes on scale-down is typical. Asymmetric thresholds, as in the code block above, make it harder to remove instances than to add them, which biases the system toward over-provisioning rather than under. Smoothing the input metrics with an exponentially weighted moving average prevents a single-burst spike from triggering a scale-up that persists for thirty minutes after the burst has passed. Explicit hysteresis — requiring sustained threshold breach for a minimum duration before acting — filters out transient spikes that would otherwise trigger expensive cold-start sequences for demand that has already passed.
      </Prose>

      <Callout accent="gold">
        The goal of autoscaling isn't to minimize GPU hours. It's to keep the fleet sized just-above demand most of the time. A minute of over-provisioning is cheaper than a minute of dropped requests.
      </Callout>

      <Prose>
        Autoscaling is where theory meets operations. The right metrics, the right speed of response, and the right buffer of warm capacity together determine whether your LLM product feels fast and available or laggy and flaky. The next topic — disaggregated prefill and decode — takes this further, splitting what an "instance" does into two specialized kinds, one optimized for the compute-bound prefill phase and one for the memory-bound decode phase. Once the two phases live on separate hardware, the autoscaling problem bifurcates as well: prefill capacity and decode capacity scale independently against their own resource signals, which is both more powerful and more complex to operate than the unified fleet model described here.
      </Prose>
    </div>
  ),
};

export default autoscalingGPU;
