import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const autoscalingGPU = {
  title: "Autoscaling & GPU Resource Management",
  readTime: "42 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        A GPU costs between two and ten dollars per hour depending on the cloud provider, instance type, and whether you are on on-demand or reserved pricing. An H100 SXM at on-demand rates runs around eight dollars per hour. An idle H100 costs exactly as much as a fully loaded one. That arithmetic — where utilization does not change cost — is the defining tension of every decision in LLM infrastructure. Web servers have the same property in a mild form, but for LLM serving the stakes are severe enough to force architectural choices that would be overkill in almost any other context.
      </Prose>

      <Prose>
        LLM traffic is not flat. It is spiky in ways that are simultaneously predictable and unpredictable. It follows daily cycles — American office hours, European morning rushes, Asian peak times — that are clear enough in historical data to model. But it also spikes on events that are invisible to the model: a viral post, a product launch, a news cycle, an agent orchestrator that suddenly decides to fan out fifty parallel calls. Those spike amplitudes are not bounded by anything the operator controls. A fleet sized for the daily peak handles the predictable load but fails on the event-driven burst. A fleet sized for the worst observed burst is idle eighty percent of the time at enormous cost.
      </Prose>

      <Prose>
        The operational answer is autoscaling: dynamically adjusting fleet size in response to measured demand. The implementation answer is where the difficulty lives. Classical autoscaling — watch CPU utilization, add instances when it crosses a threshold, remove them when it drops — does not transfer to LLM serving without modification that amounts to a near-complete redesign. The signals are wrong, the timescales are wrong, and the cost structure of errors is asymmetric in ways that classical autoscalers do not account for.
      </Prose>

      <Prose>
        The signal problem: GPU utilization as reported by <Code>nvidia-smi</Code> does not measure whether the GPU is the bottleneck. A vLLM worker can be at the edge of KV cache exhaustion — one more request away from starting to evict tokens and cause latency spikes — while the utilization metric reads forty-five percent. The GPU is spending most of its time waiting for high-bandwidth memory reads to complete during the decode phase. HBM latency does not register as SM utilization. CPU utilization is worse: on an inference node it hovers near two to five percent even when the GPU is saturated, because tokenization and scheduling are negligible. An autoscaler watching CPU concludes the fleet is lightly loaded and does nothing while users wait.
      </Prose>

      <Prose>
        The timescale problem: adding a new instance to a web-server fleet takes seconds. Adding a new instance to an LLM serving fleet takes minutes. In BF16 precision, a 70B-parameter model occupies roughly 140 gigabytes. Even a fast storage tier with multi-gigabyte-per-second throughput to GPU memory requires several minutes for that transfer. CUDA initialization and KV cache allocation add more. The median time from scale-up trigger to first served request for a 70B model sits between three and eight minutes depending on infrastructure. If the autoscaler triggers at the moment demand exceeds capacity, the new capacity will not arrive until demand may have already receded.
      </Prose>

      <Prose>
        The asymmetric cost problem: over-provisioning by one instance for an hour costs eight dollars and produces nothing for users. Under-provisioning for a minute drops requests, degrades tail latency, and, if the retry behavior is not carefully designed, generates a feedback loop where retries amplify overload. A system that fails for sixty seconds during a spike is often more visible to users than one that wastes twenty GPU-hours over a week. The autoscaler's cost function is asymmetric: the penalty for being under-provisioned at the wrong moment is far higher than the penalty for being slightly over-provisioned most of the time. Any design that treats the two errors as equal is wrong.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Before reaching for the math, it is worth being precise about what autoscaling is actually trying to do. It is not minimizing GPU-hours. It is keeping the fleet sized just above demand, with enough headroom that the queue never builds up and enough reactivity that a spike does not outrun capacity for long. Those two goals are in tension — headroom means idle GPUs; reactivity means scaling before you are sure you need to — and the resolution lives in choosing the right signals, the right thresholds, and the right lead time.
      </Prose>

      <H3>Why classical signals misfit LLM workloads</H3>

      <Prose>
        The mental model for classical autoscaling is a stateless web server. Requests arrive, are handled in roughly fixed time, and leave. CPU utilization tracks how much of the server's capacity is consumed because CPU is the bottleneck. The model fits because CPU time and request-handling time are proportional. In LLM serving, neither condition holds. The bottleneck is GPU HBM bandwidth during decode, not CPU. Request-handling time spans two phases — prefill and decode — with completely different resource profiles. And the system has a hard capacity constraint in KV cache that has no analogue in a stateless web server.
      </Prose>

      <Prose>
        The right signals for autoscaling LLM fleets are the ones that predict imminent failure before it arrives. Three metrics cover the space. KV cache utilization — the fraction of GPU HBM currently allocated to active KV blocks — is the most direct measure of whether the serving system is approaching its hard capacity limit. At sustained utilization above seventy-five to eighty percent, most inference engines are already throttling admission; the autoscaler needs to have started a new instance before that point. Queue depth measures how many requests are waiting behind the current batch. In a healthy fleet, queue depth is near zero most of the time. Sustained queue depth above five to ten indicates the fleet is falling behind and the gap compounds. Time-to-first-token at the ninety-ninth percentile is the user-visible expression of both conditions and the one that directly ties to SLA commitments.
      </Prose>

      <H3>Scale-up is bounded by model-load time; scale-down is fast</H3>

      <Prose>
        The asymmetry between scale-up and scale-down latency is the most important operational fact about LLM autoscaling. Scaling down is fast: stop admitting new requests to an instance, wait for in-flight requests to complete (which takes at most as long as the longest response being generated), and terminate. Total time for a graceful scale-down is bounded by the maximum response generation time, which is typically under a minute even for long contexts. Scaling up is slow: download model weights, load into GPU HBM, initialize CUDA context, allocate KV cache pools, run warmup passes. For a 70B model this is three to eight minutes on typical cloud storage. The fleet needs to react to demand that has not yet arrived.
      </Prose>

      <Prose>
        The practical implication is that scale-up decisions must lead demand. The autoscaler has to decide to add capacity based on a prediction about where demand will be in five minutes, not where it is now. Two strategies address this. Predictive scaling uses historical demand patterns — daily cycles, day-of-week effects, known events — to pre-warm instances on a schedule. Metric-based lead time uses faster-reacting metrics — queue depth trending up, TTFT percentiles rising — that precede the full saturation event by enough time to start a new instance before saturation arrives. In practice, a production autoscaler uses both: a predictive component that handles the known cycle and a reactive component that handles the surprises.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Scale-up decision rule</H3>

      <Prose>
        The basic reactive scale-up rule fires when any of the primary metrics crosses its threshold. The OR logic is deliberate: each metric catches a different failure mode, and waiting for all three to breach simultaneously would mean the system is already degraded in multiple dimensions before acting.
      </Prose>

      <MathBlock>{`\\text{scale\\_up} \\iff P99_{\\text{TTFT}} > \\text{SLO} \\; \\lor \\; Q_{\\text{depth}} > Q_{\\text{thresh}} \\; \\lor \\; \\text{KV}_{\\text{util}} > \\text{KV}_{\\text{thresh}}`}</MathBlock>

      <Prose>
        Scale-down uses AND logic — all conditions must hold simultaneously before removing capacity. This asymmetry reflects the asymmetric cost of the two errors described in Section 1.
      </Prose>

      <MathBlock>{`\\text{scale\\_down} \\iff P99_{\\text{TTFT}} < \\text{SLO}_{\\text{low}} \\; \\land \\; Q_{\\text{depth}} = 0 \\; \\land \\; \\text{KV}_{\\text{util}} < \\text{KV}_{\\text{low}} \\; \\land \\; t_{\\text{idle}} > T_{\\text{cooldown}}`}</MathBlock>

      <H3>Cold-start amortization</H3>

      <Prose>
        Every instance that is started incurs a fixed cold-start cost: the time during which the GPU is occupied loading the model but not serving requests. That dead time must be amortized over the useful life of the instance to evaluate whether starting it was worthwhile. Let <Code>t_load</Code> be the model load time in minutes and <Code>t_active</Code> be how long the instance serves requests before being stopped. The fraction of GPU-time wasted on loading is:
      </Prose>

      <MathBlock>{`\\text{overhead} = \\frac{t_{\\text{load}}}{t_{\\text{load}} + t_{\\text{active}}}`}</MathBlock>

      <Prose>
        For a 3-minute load time and 10-minute active window, overhead is 23% — nearly a quarter of GPU-time wasted. For a 30-minute active window, overhead falls to 9%. For a 60-minute window, it is 5%. The rule of thumb for interactive serving is to keep instances active for at least ten times the load time before stopping them. For a 3-minute load, that means a minimum 30-minute active window before scale-down is considered. This is the quantitative basis for the long cooldown periods in scale-down logic.
      </Prose>

      <H3>Predictive scaling with Holt-Winters</H3>

      <Prose>
        For workloads with daily or weekly cycles, Holt-Winters exponential smoothing provides a simple predictive model. It tracks three components: a level (the current baseline), a trend (the rate of change), and seasonal components (the cyclic pattern). The forecast at horizon <Code>h</Code> is:
      </Prose>

      <MathBlock>{`\\hat{y}_{t+h} = (L_t + h \\cdot B_t) \\cdot S_{t+h-m}`}</MathBlock>

      <Prose>
        where <Code>L_t</Code> is the smoothed level, <Code>B_t</Code> is the trend estimate, <Code>S</Code> is the seasonal factor for period <Code>m</Code> (typically 24 hours or 168 hours for weekly patterns), and the three smoothing parameters <Code>α</Code>, <Code>β</Code>, <Code>γ</Code> are fitted on historical data. The target instance count is then:
      </Prose>

      <MathBlock>{`N_{\\text{target}} = \\left\\lceil \\frac{\\hat{y}_{t+t_{\\text{load}}}}{\\mu \\cdot \\rho_{\\text{target}}} \\right\\rceil`}</MathBlock>

      <Prose>
        where <Code>μ</Code> is the per-instance request throughput and <Code>ρ_target</Code> is the target utilization (typically 0.65–0.75 for interactive serving). The forecast is evaluated at time <Code>t + t_load</Code>, not time <Code>t</Code>, because any instance started now will not be serving requests until after the cold-start window.
      </Prose>

      <H3>Cost and spot tradeoff</H3>

      <Prose>
        Total hourly GPU cost for a fleet is:
      </Prose>

      <MathBlock>{`\\text{Cost} = N_{\\text{od}} \\cdot c_{\\text{od}} + N_{\\text{spot}} \\cdot c_{\\text{spot}}`}</MathBlock>

      <Prose>
        where <Code>N_od</Code> is on-demand instances, <Code>c_od</Code> is on-demand price per hour, <Code>N_spot</Code> is spot instances, and <Code>c_spot</Code> is the spot price (typically 0.3–0.6× on-demand). Spot instances can be reclaimed with two to five minutes of notice on most cloud providers. A reclaimed instance kills all in-flight requests it holds. The expected cost of one spot interruption is the number of requests in flight at the time of interruption multiplied by the re-execution cost of each request (latency plus compute). For batch workloads, this cost is low; for interactive serving mid-conversation, the user experience impact makes it intolerable without a careful retry strategy.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <H3>4a — Simulating demand and comparing reactive vs predictive autoscalers</H3>

      <Prose>
        The starting point is a realistic demand curve. LLM traffic follows a daily sinusoidal pattern with multiplicative noise and occasional spikes. We simulate 48 hours at one-minute resolution and run two autoscalers — pure reactive and predictive — side by side to measure how each handles the same traffic.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
MINUTES = 48 * 60          # 48-hour simulation
t = np.arange(MINUTES)

# Daily sinusoidal demand: peak at 14:00, trough at 04:00
base_rps = 20 + 15 * np.sin(2 * np.pi * (t - 4 * 60) / (24 * 60))
# Multiplicative noise
noise = rng.lognormal(0, 0.15, MINUTES)
# Random spike events (3 spikes per 48 hours on average)
spikes = np.zeros(MINUTES)
spike_times = rng.choice(MINUTES, 3, replace=False)
for st in spike_times:
    spikes[st:st+30] += rng.uniform(15, 35)

demand_rps = np.maximum(0, base_rps * noise + spikes)

# ---------- reactive autoscaler ----------
# Adds instances when queue > threshold; removes after cooldown.
CAPACITY_PER_INSTANCE = 5.0    # RPS per GPU instance
LOAD_TIME_MINUTES = 5          # cold-start delay
SCALE_UP_QUEUE_THRESH = 3.0    # trigger if projected queue > 3
SCALE_DOWN_COOLDOWN = 30       # minutes before removing an instance
MIN_INSTANCES = 2

def reactive_autoscaler(demand, min_inst=MIN_INSTANCES):
    instances = min_inst
    pending_additions = []       # (ready_at_minute, count)
    last_scale_down = -SCALE_DOWN_COOLDOWN
    inst_history = []
    dropped = []

    for minute in range(MINUTES):
        # Bring online any instances whose load time has elapsed
        ready = [n for (t_ready, n) in pending_additions if t_ready <= minute]
        instances += sum(ready)
        pending_additions = [(tr, n) for (tr, n) in pending_additions if tr > minute]

        capacity = instances * CAPACITY_PER_INSTANCE
        current_demand = demand[minute]
        queue = max(0, current_demand - capacity)
        dropped.append(max(0, queue))

        # Scale up: if queue is building, add one instance (with cold-start delay)
        if queue > SCALE_UP_QUEUE_THRESH:
            pending_additions.append((minute + LOAD_TIME_MINUTES, 1))

        # Scale down: if capacity well above demand and cooldown expired
        if (instances > min_inst
                and capacity > 1.5 * current_demand
                and minute - last_scale_down >= SCALE_DOWN_COOLDOWN):
            instances = max(min_inst, instances - 1)
            last_scale_down = minute

        inst_history.append(instances + len(pending_additions))

    return np.array(inst_history), np.array(dropped)

# ---------- predictive autoscaler (Holt-Winters) ----------
def holt_winters_forecast(series, alpha=0.3, beta=0.05, gamma=0.2,
                           season_len=24*60, h=LOAD_TIME_MINUTES):
    """Single multiplicative Holt-Winters, returns forecast at horizon h."""
    # Initialize
    L = series[:season_len].mean()
    B = 0.0
    S = series[:season_len] / L
    forecasts = []
    for i in range(len(series)):
        s_idx = i % season_len
        if i > 0:
            L_prev, S_prev = L, S[s_idx]
            obs = series[i]
            L = alpha * (obs / S_prev) + (1 - alpha) * (L_prev + B)
            B = beta * (L - L_prev) + (1 - beta) * B
            S[s_idx] = gamma * (obs / L) + (1 - gamma) * S_prev
        fc_s_idx = (i + h) % season_len
        forecasts.append((L + h * B) * S[fc_s_idx])
    return np.maximum(0, np.array(forecasts))

def predictive_autoscaler(demand, min_inst=MIN_INSTANCES):
    forecast = holt_winters_forecast(demand)
    instances = min_inst
    last_scale_down = -SCALE_DOWN_COOLDOWN
    inst_history = []
    dropped = []

    for minute in range(MINUTES):
        target = max(min_inst,
                     int(np.ceil(forecast[minute] / (CAPACITY_PER_INSTANCE * 0.70))))
        # Scale up immediately (pre-warming means we've already done load)
        if target > instances:
            instances = target
        # Scale down with cooldown
        elif (target < instances
              and minute - last_scale_down >= SCALE_DOWN_COOLDOWN):
            instances = max(min_inst, instances - 1)
            last_scale_down = minute

        capacity = instances * CAPACITY_PER_INSTANCE
        dropped.append(max(0, demand[minute] - capacity))
        inst_history.append(instances)

    return np.array(inst_history), np.array(dropped)

react_inst, react_dropped = reactive_autoscaler(demand_rps)
pred_inst, pred_dropped = predictive_autoscaler(demand_rps)

print(f"Reactive  — avg instances: {react_inst.mean():.1f}, "
      f"total dropped RPS-minutes: {react_dropped.sum():.0f}")
print(f"Predictive— avg instances: {pred_inst.mean():.1f}, "
      f"total dropped RPS-minutes: {pred_dropped.sum():.0f}")`}
      </CodeBlock>

      <CodeBlock language="text">
{`Reactive  — avg instances: 6.3, total dropped RPS-minutes: 412
Predictive— avg instances: 7.1, total dropped RPS-minutes:  38`}
      </CodeBlock>

      <Prose>
        The predictive autoscaler drops eleven times fewer requests than the reactive one, at the cost of one extra instance on average (about eight dollars per hour extra at H100 on-demand rates). The reactive autoscaler always lags the demand by the cold-start window, which is when spikes inflict the most damage. The predictive autoscaler runs slightly hotter — more instances than strictly needed for much of the day — but absorbs spikes without dropping anything.
      </Prose>

      <H3>4b — Cold-start cost accounting</H3>

      <Prose>
        Quantifying the waste from cold-starts requires tracking the ratio of load time to total instance lifetime. The following traces every instance through its full lifecycle — start trigger, load completion, serving window, termination — and computes the utilization efficiency for each.
      </Prose>

      <CodeBlock language="python">
{`from dataclasses import dataclass, field
from typing import List

@dataclass
class InstanceLifecycle:
    triggered_at: int       # minute the scale-up was triggered
    online_at: int          # minute the instance became available
    terminated_at: int = 0  # minute the instance was stopped
    load_time: int = LOAD_TIME_MINUTES

    @property
    def total_lifetime(self):
        return self.terminated_at - self.triggered_at

    @property
    def useful_time(self):
        return self.terminated_at - self.online_at

    @property
    def waste_fraction(self):
        if self.total_lifetime == 0:
            return 1.0
        return self.load_time / self.total_lifetime

def simulate_instance_lifecycles(active_windows: List[tuple]) -> List[InstanceLifecycle]:
    """
    active_windows: list of (trigger_minute, terminate_minute) pairs
    Returns lifecycle stats for each instance.
    """
    lifecycles = []
    for trigger, terminate in active_windows:
        lc = InstanceLifecycle(
            triggered_at=trigger,
            online_at=trigger + LOAD_TIME_MINUTES,
            terminated_at=terminate,
        )
        lifecycles.append(lc)
    return lifecycles

# Example: one spike at minute 200 triggers 3 instances, each running for N minutes
example_windows = [
    (200, 200 + 10),   # too short: 5 min load + 5 min serving = 50% waste
    (200, 200 + 30),   # marginal: 5 min load + 25 min serving = 17% waste
    (200, 200 + 60),   # good:     5 min load + 55 min serving = 8% waste
    (200, 200 + 120),  # excellent: 5 min load + 115 min serving = 4% waste
]

lifecycles = simulate_instance_lifecycles(example_windows)
print(f"{'Active window (min)':<22} {'Waste fraction':>15} {'GPU-min wasted':>15}")
for lc, (_, end) in zip(lifecycles, example_windows):
    window = end - 200
    print(f"{window:<22} {lc.waste_fraction:>14.0%} {lc.load_time:>15}")`}
      </CodeBlock>

      <CodeBlock language="text">
{`Active window (min)    Waste fraction  GPU-min wasted
10                              50%               5
30                              17%               5
60                               8%               5
120                              4%               5`}
      </CodeBlock>

      <Prose>
        The cold-start cost is fixed at five GPU-minutes per instance regardless of how long the instance runs. What changes is how much useful work gets done against that fixed overhead. Running an instance for ten minutes to amortize five minutes of loading is operationally indefensible — you are spending half your GPU-time on loading. The minimum viable active window is roughly ten times the load time; below that, the economics of spin-up and spin-down do not close.
      </Prose>

      <H3>4c — Multi-metric autoscaling: combining queue depth, P99, and utilization</H3>

      <Prose>
        A single metric can be gamed by workload patterns that stress the dimension it does not measure. The composite signal uses all three primary metrics and fires on any single breach. The weights in the combination are not additive — it is an OR gate, not a weighted sum.
      </Prose>

      <CodeBlock language="python">
{`from dataclasses import dataclass

@dataclass
class FleetMetrics:
    kv_cache_util_p95: float   # fraction of KV cache allocated (0-1)
    queue_depth_p95: float     # 95th pct waiting requests
    ttft_p99_ms: float         # 99th pct time-to-first-token, milliseconds
    instance_count: int
    pending_instances: int     # spinning up, not yet available

# Thresholds — tune per workload
SCALE_UP_KV       = 0.75
SCALE_UP_QUEUE    = 5.0
SCALE_UP_TTFT_MS  = 2000.0

SCALE_DOWN_KV     = 0.30
SCALE_DOWN_QUEUE  = 0.0
SCALE_DOWN_TTFT_MS = 800.0
MIN_INSTANCES     = 2

def composite_scaling_decision(m: FleetMetrics) -> str:
    """
    Returns 'scale_up', 'scale_down', or 'hold'.
    OR logic for up; AND logic for down.
    """
    scale_up = (
        m.kv_cache_util_p95  > SCALE_UP_KV    or
        m.queue_depth_p95    > SCALE_UP_QUEUE  or
        m.ttft_p99_ms        > SCALE_UP_TTFT_MS
    )
    scale_down = (
        m.kv_cache_util_p95  < SCALE_DOWN_KV      and
        m.queue_depth_p95   <= SCALE_DOWN_QUEUE    and
        m.ttft_p99_ms        < SCALE_DOWN_TTFT_MS  and
        m.instance_count     > MIN_INSTANCES        and
        m.pending_instances  == 0   # don't remove while scaling up
    )
    if scale_up:
        return "scale_up"
    if scale_down:
        return "scale_down"
    return "hold"

# Scenario table — verify each signal independently triggers scale_up
scenarios = [
    FleetMetrics(kv_cache_util_p95=0.82, queue_depth_p95=1.0, ttft_p99_ms=400, instance_count=4, pending_instances=0),
    FleetMetrics(kv_cache_util_p95=0.40, queue_depth_p95=9.0, ttft_p99_ms=400, instance_count=4, pending_instances=0),
    FleetMetrics(kv_cache_util_p95=0.40, queue_depth_p95=1.0, ttft_p99_ms=2500, instance_count=4, pending_instances=0),
    FleetMetrics(kv_cache_util_p95=0.25, queue_depth_p95=0.0, ttft_p99_ms=600, instance_count=6, pending_instances=0),
]
labels = ["KV only breached", "Queue only breached", "TTFT only breached", "All clear -> scale_down"]
for label, s in zip(labels, scenarios):
    print(f"{label:<30} -> {composite_scaling_decision(s)}")`}
      </CodeBlock>

      <CodeBlock language="text">
{`KV only breached               -> scale_up
Queue only breached            -> scale_up
TTFT only breached             -> scale_up
All clear -> scale_down        -> scale_down`}
      </CodeBlock>

      <H3>4d — Mixed on-demand and spot fleet with interrupt handling</H3>

      <Prose>
        A split fleet keeps baseline capacity on on-demand instances that cannot be reclaimed, and uses spot instances for burst capacity. Spot interruptions arrive with two to five minutes notice. The interrupt handler must drain in-flight requests gracefully: stop accepting new requests the moment the notice arrives, allow current requests to complete within the notice window, and re-route any that cannot complete in time to surviving instances.
      </Prose>

      <CodeBlock language="python">
{`import heapq
from typing import Optional

@dataclass
class Instance:
    instance_id: str
    is_spot: bool
    started_at: float            # wall-clock minute
    in_flight: list = field(default_factory=list)
    draining: bool = False       # True when spot termination notice received
    termination_at: Optional[float] = None

def handle_spot_interruption(instance: Instance,
                              current_time: float,
                              notice_minutes: float = 2.0,
                              on_demand_pool: list = None) -> dict:
    """
    Called when spot termination notice arrives.
    Returns action plan: what to drain vs re-route.
    """
    instance.draining = True
    instance.termination_at = current_time + notice_minutes

    to_complete = []   # can finish within notice window
    to_reroute  = []   # need to be sent elsewhere

    for req in instance.in_flight:
        remaining_gen_time = req.get("remaining_tokens", 0) * req.get("ms_per_token", 20) / 1000
        if remaining_gen_time <= notice_minutes * 60:   # seconds
            to_complete.append(req)
        else:
            to_reroute.append(req)

    return {
        "drain": to_complete,
        "reroute": to_reroute,
        "terminate_at": instance.termination_at,
        "spot_id": instance.instance_id,
    }

# Fleet configuration
def build_split_fleet(od_count: int, spot_count: int) -> list:
    fleet = []
    for i in range(od_count):
        fleet.append(Instance(f"od-{i}", is_spot=False, started_at=0.0))
    for i in range(spot_count):
        fleet.append(Instance(f"spot-{i}", is_spot=True, started_at=0.0))
    return fleet

# Cost analysis
def fleet_hourly_cost(od_count: int, spot_count: int,
                      od_price: float = 8.0,
                      spot_price: float = 3.0) -> float:
    return od_count * od_price + spot_count * spot_price

# Compare: 8 on-demand vs 4 on-demand + 4 spot
print(f"8 on-demand:          \${fleet_hourly_cost(8, 0):.2f}/hr")
print(f"4 on-demand + 4 spot: \${fleet_hourly_cost(4, 4):.2f}/hr")
print(f"Savings:              \${fleet_hourly_cost(8,0) - fleet_hourly_cost(4,4):.2f}/hr")`}
      </CodeBlock>

      <CodeBlock language="text">
{`8 on-demand:          $64.00/hr
4 on-demand + 4 spot: $44.00/hr
Savings:              $20.00/hr`}
      </CodeBlock>

      <Prose>
        The split fleet saves twenty dollars per hour in this example — about fourteen thousand dollars per month for a moderately sized serving deployment. The cost is exposure to spot interruptions on the burst tier. For interactive workloads, the interrupt handler needs to re-route seamlessly; for async batch workloads, the request can simply be requeued when the interrupt arrives.
      </Prose>

      <H3>4e — HPA-style vs KEDA-style autoscaling</H3>

      <Prose>
        Kubernetes Horizontal Pod Autoscaler (HPA) and KEDA (Kubernetes Event-Driven Autoscaling) represent two different philosophies. HPA reacts to resource metrics at a fixed polling interval (default 15 seconds) and scales based on a ratio formula. KEDA reacts to event sources — including Prometheus metrics — and can scale to zero. The distinction matters for LLM serving because scale-to-zero is operationally useful for batch workloads but dangerous for interactive ones.
      </Prose>

      <CodeBlock language="yaml">
{`# HPA-style: scale on custom Prometheus metric (vllm:num_requests_waiting)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-serving
  minReplicas: 2
  maxReplicas: 20
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0    # scale up fast
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60              # add at most 2 pods/min
    scaleDown:
      stabilizationWindowSeconds: 1800  # 30-min cooldown before scale-down
      policies:
      - type: Pods
        value: 1
        periodSeconds: 300             # remove at most 1 pod per 5 min
  metrics:
  - type: External
    external:
      metric:
        name: vllm_num_requests_waiting
        selector:
          matchLabels:
            deployment: vllm-serving
      target:
        type: AverageValue
        averageValue: "5"  # target: ≤5 waiting requests per replica
---
# KEDA-style: ScaledObject targeting same Prometheus metric
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-keda-scaler
spec:
  scaleTargetRef:
    name: vllm-serving
  minReplicaCount: 2
  maxReplicaCount: 20
  cooldownPeriod: 1800     # 30 minutes — matches HPA scale-down stabilization
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      metricName: vllm_num_requests_waiting
      threshold: "5"
      query: >
        sum(vllm:num_requests_waiting{deployment="vllm-serving"})
        / count(vllm:num_requests_waiting{deployment="vllm-serving"})`}
      </CodeBlock>

      <Prose>
        The HPA approach requires a Custom Metrics API adapter (e.g., Prometheus Adapter) to surface the Prometheus metric as a Kubernetes metric. KEDA integrates Prometheus natively and adds the scale-to-zero capability. For LLM serving, the right choice is KEDA with <Code>minReplicaCount: 2</Code> — preserving scale-to-zero for cost control while ensuring a warm baseline. Both approaches need the long scale-down cooldown to avoid oscillation caused by the cold-start delay.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>Kubernetes HPA with custom metrics</H3>

      <Prose>
        In a standard Kubernetes deployment, HPA with CPU or memory targets is configured in minutes. Custom metrics — the kind needed for LLM autoscaling — require an additional component: a Custom Metrics API server that queries Prometheus and translates the results into the format the HPA controller expects. The Prometheus Adapter is the most common choice, configured via a rules file that maps Prometheus queries to Kubernetes metric names. The HPA spec then references these metric names using the <Code>external</Code> or <Code>pods</Code> metric type. Kubernetes 1.33 introduced configurable HPA tolerance (the deadband around the target below which no scaling action fires), which is particularly useful for LLM fleets because a small fluctuation in queue depth should not trigger a scale event that then takes five minutes to materialize.
      </Prose>

      <H3>KEDA</H3>

      <Prose>
        KEDA (version 2.19 as of 2026) is a CNCF project that adds event-driven scaling to any Kubernetes workload. It ships with over seventy built-in scalers including a Prometheus scaler, a Kafka scaler, an Azure Queue scaler, and an AWS SQS scaler. For LLM serving, the Prometheus scaler is the primary entry point: configure a <Code>ScaledObject</Code> pointing at the Prometheus endpoint, write a PromQL query that returns the signal (e.g., <Code>avg(vllm:gpu_cache_usage_perc)</Code>), and set a threshold. KEDA's PredictKube extension adds AI-based predictive autoscaling on top of the reactive KEDA scaler, fitting a time-series model on observed request patterns and issuing scale-up actions ahead of predicted demand. The vLLM production stack Helm chart (v0.1.9+) integrates KEDA directly; enabling autoscaling requires adding the observability stack (Prometheus + Grafana) to the cluster and setting a single values flag.
      </Prose>

      <H3>Knative Serving</H3>

      <Prose>
        Knative Serving provides a higher-level abstraction: instead of managing Deployments and HPAs directly, you define a Knative Service and configure its concurrency limits. Knative's built-in autoscaler, the Knative Pod Autoscaler (KPA), watches request concurrency in real time and scales replicas to keep the number of concurrent requests per pod near the configured target. KPA operates in two modes: stable mode uses a 60-second measurement window for smooth scaling, and panic mode activates when concurrency exceeds 200% of target within a 6-second window, triggering immediate aggressive scale-up. For LLM workloads, the right configuration is to set <Code>containerConcurrency</Code> to approximately the <Code>max_num_seqs</Code> configured in the inference engine, so that Knative's concurrency model maps directly onto the KV cache capacity model of the engine.
      </Prose>

      <H3>AWS SageMaker autoscaling</H3>

      <Prose>
        SageMaker Inference endpoints support Application Auto Scaling with target tracking policies. The predefined metric <Code>SageMakerVariantInvocationsPerInstance</Code> tracks requests per instance per minute. For LLM serving, custom CloudWatch metrics — GPU utilization per variant, model latency percentiles — are more appropriate. The scaling policy uses a JSON configuration specifying the target metric, target value, scale-in and scale-out cooldown periods, and the scaling dimension (number of instances behind the endpoint variant). SageMaker's async inference endpoints add a second trigger: the queue depth in S3 can drive scaling via a custom metric that counts pending requests.
      </Prose>

      <H3>Ray Serve autoscaling</H3>

      <Prose>
        Ray Serve (2.54+) treats autoscaling as a first-class feature of its deployment model. Setting <Code>num_replicas="auto"</Code> enables autoscaling with sensible defaults. The key configuration is <Code>target_ongoing_requests</Code>: the number of in-flight requests per replica that Serve targets. This is the Ray equivalent of the KV utilization target. For LLM serving, set this to roughly 60–70% of the engine's <Code>max_num_seqs</Code> to keep a utilization buffer. <Code>upscale_delay_s</Code> (default 30s) and <Code>downscale_delay_s</Code> (default 600s) control the reaction time; for LLM fleets with long cold-starts, increasing <Code>upscale_delay_s</Code> to 0 and <Code>downscale_delay_s</Code> to 1800 (30 minutes) is closer to the right production setting. Ray Serve's LLM-specific layer adds awareness of the vLLM engine scheduler, exposing queue depth and KV cache fill as native scaling signals.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Plot
        label="demand vs capacity — reactive and predictive autoscalers (48-hour simulation)"
        width={580}
        height={300}
        xLabel="hour of day (repeated over 48h)"
        yLabel="RPS / instance count × 5"
        series={[
          {
            name: "demand (RPS)",
            points: [
              [0,12],[1,10],[2,9],[3,8],[4,8],[5,9],[6,12],[7,17],[8,22],
              [9,27],[10,31],[11,33],[12,34],[13,35],[14,36],[15,35],[16,33],
              [17,31],[18,28],[19,25],[20,22],[21,19],[22,16],[23,14],
              [24,12],[25,10],[26,9],[27,8],[28,8],[29,9],[30,12],[31,17],
              [32,22],[33,38],[34,31],[35,33],[36,34],[37,35],[38,36],[39,35],
              [40,33],[41,31],[42,28],[43,25],[44,22],[45,19],[46,16],[47,14],
            ],
          },
          {
            name: "reactive capacity (RPS)",
            points: [
              [0,10],[1,10],[2,10],[3,10],[4,10],[5,10],[6,10],[7,15],[8,20],
              [9,25],[10,30],[11,30],[12,35],[13,35],[14,35],[15,35],[16,35],
              [17,30],[18,25],[19,25],[20,20],[21,20],[22,15],[23,15],
              [24,10],[25,10],[26,10],[27,10],[28,10],[29,10],[30,10],[31,15],
              [32,20],[33,25],[34,30],[35,30],[36,35],[37,35],[38,35],[39,35],
              [40,35],[41,30],[42,25],[43,25],[44,20],[45,20],[46,15],[47,15],
            ],
          },
          {
            name: "predictive capacity (RPS)",
            points: [
              [0,15],[1,12],[2,10],[3,10],[4,10],[5,12],[6,15],[7,20],[8,25],
              [9,30],[10,35],[11,35],[12,35],[13,35],[14,40],[15,40],[16,35],
              [17,35],[18,30],[19,25],[20,25],[21,20],[22,18],[23,15],
              [24,15],[25,12],[26,10],[27,10],[28,10],[29,12],[30,15],[31,20],
              [32,25],[33,40],[34,35],[35,35],[36,35],[37,35],[38,40],[39,40],
              [40,35],[41,35],[42,30],[43,25],[44,25],[45,20],[46,18],[47,15],
            ],
          },
        ]}
      />

      <Prose>
        The demand curve follows the expected daily sinusoid with a spike at hour 9 on the second day (the viral event). The reactive autoscaler lags the spike by the cold-start window — five minutes — and during that window demand exceeds capacity. The predictive autoscaler overshoots the baseline by one to two instances in exchange for never falling below demand.
      </Prose>

      <Plot
        label="cost vs dropped-request rate under different autoscaling policies"
        width={540}
        height={280}
        xLabel="avg instances (proxy for hourly cost)"
        yLabel="dropped RPS-minutes per 48 hours"
        series={[
          {
            name: "reactive autoscaler",
            points: [
              [4, 980], [5, 620], [6, 340], [7, 180], [8, 80], [9, 30], [10, 12],
            ],
          },
          {
            name: "predictive autoscaler",
            points: [
              [5, 120], [6, 55], [7, 22], [8, 8], [9, 3], [10, 0],
            ],
          },
          {
            name: "always-warm (over-provisioned)",
            points: [
              [12, 0], [14, 0], [16, 0],
            ],
          },
        ]}
      />

      <Prose>
        The Pareto frontier of cost versus dropped requests shows that the predictive autoscaler dominates the reactive one: at any given cost level, it drops fewer requests. The always-warm policy drops zero requests but at substantially higher cost. The crossover point — where predictive autoscaling becomes cheaper than always-warm — depends on how much traffic the workload carries during off-peak hours. For workloads with low overnight traffic, predictive autoscaling wins decisively.
      </Prose>

      <StepTrace
        label="one scale-up decision — from metric breach to new instance serving"
        steps={[
          {
            label: "metric threshold breach detected",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div style={{ color: colors.gold }}>t=0:00 — autoscaler polling cycle fires</div>
                <div>KV cache util (p95): 0.81  ← exceeds 0.75 threshold</div>
                <div>Queue depth (p95):   3.2   — below threshold</div>
                <div>TTFT p99:            1,840ms — below 2,000ms SLO</div>
                <div style={{ color: colors.gold }}>Decision: SCALE_UP (KV signal triggered)</div>
                <div>Current instances: 4 — target: 5</div>
              </div>
            ),
          },
          {
            label: "new pod scheduled on cluster",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div style={{ color: colors.gold }}>t=0:15 — Kubernetes scheduler places pod on GPU node</div>
                <div>Node: gpu-node-07 (H100 80GB, 0 other tenants)</div>
                <div>Pod status: Pending → ContainerCreating</div>
                <div>CUDA driver initialization: ~45s estimated</div>
                <div>Container image pull: cached (0s) — image pre-pulled</div>
              </div>
            ),
          },
          {
            label: "model weights loading from shared volume",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div style={{ color: colors.gold }}>t=1:05 — vLLM process starts, loading Llama-3-70B</div>
                <div>Source: NFS shared volume (ReadWriteMany mount)</div>
                <div>Model size: 140GB (BF16), transfer rate ~800MB/s</div>
                <div>Estimated load time: 175s ≈ 2m 55s</div>
                <div>KV cache pool: allocating 48GB for 512 max sequences</div>
                <div style={{ color: colors.textSecondary }}>Meanwhile: existing 4 instances absorbing traffic at KV util 0.84</div>
              </div>
            ),
          },
          {
            label: "warmup pass and readiness probe",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div style={{ color: colors.gold }}>t=4:00 — weights loaded, running warmup forward pass</div>
                <div>Warmup: 3 × 512-token batches to JIT-compile attention kernels</div>
                <div>Warmup duration: ~18s</div>
                <div>Readiness probe: HTTP GET /health → 200 OK</div>
                <div style={{ color: colors.green }}>t=4:20 — pod marked Ready, added to service endpoints</div>
                <div>Load balancer routing new requests to instance 5</div>
              </div>
            ),
          },
          {
            label: "fleet stabilized — new steady state",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div style={{ color: colors.gold }}>t=5:00 — metrics re-evaluated</div>
                <div>KV cache util (p95): 0.61  ← below all thresholds</div>
                <div>Queue depth (p95):   0.4   — near zero</div>
                <div>TTFT p99:            980ms — within SLO</div>
                <div style={{ color: colors.green }}>Decision: HOLD — scale-down blocked by 30-min cooldown</div>
                <div>Next scale-down evaluation: t=34:20</div>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        No single autoscaling strategy is optimal for all workloads. The right choice depends on traffic shape, latency requirements, cost constraints, and how much operational complexity the team can absorb.
      </Prose>

      <Heatmap
        label="autoscaling strategy by workload type and traffic pattern"
        matrix={[
          [1, 0, 0, 0],
          [1, 1, 0, 0],
          [0, 1, 1, 0],
          [0, 0, 1, 1],
          [0, 1, 0, 1],
        ]}
        rowLabels={[
          "Interactive chat (P99 < 2s SLO)",
          "API with daily traffic cycle",
          "Mixed interactive + batch",
          "Async batch / offline inference",
          "Cost-constrained burst capacity",
        ]}
        colLabels={["Always-warm pool", "Predictive (Holt-Winters)", "Reactive + cooldown", "Spot burst tier"]}
        cellSize={60}
        colorScale="green"
      />

      <Prose>
        Always-warm pools are right for interactive chat products where the cold-start window is unacceptable and the cost of dropped requests during a spike exceeds the cost of idle GPUs. Keep a minimum of N instances running at all times sized for the expected trough traffic, with autoscaling on top for peaks. The idle cost is real but bounded and predictable — it can be planned for in the infrastructure budget.
      </Prose>

      <Prose>
        Predictive autoscaling with Holt-Winters (or a comparable time-series model) is right for any workload with a discernible daily or weekly pattern — which is most production API traffic. The model learns the cycle from historical data and pre-warms capacity before the ramp arrives. The lead time is set to the cold-start window, so the fleet is ready when demand arrives rather than scrambling to catch up. The limitation is event-driven spikes that the model cannot predict; a reactive fallback handles those.
      </Prose>

      <Prose>
        Reactive autoscaling with long cooldown is the right baseline for stable workloads without strong daily patterns — enterprise integrations with consistent load profiles, internal tools, or any workload where demand is roughly flat. It is simpler to configure and requires no historical data. The asymmetric thresholds and long scale-down cooldown prevent oscillation.
      </Prose>

      <Prose>
        Spot burst tiers are right for batch workloads and async inference where occasional request failures are tolerable or invisible to end users. An agent pipeline running overnight document indexing can use a fully spot fleet with retry logic — a reclaimed instance restarts its work, and the overall pipeline completes a few minutes later. User-facing interactive serving should never put the entire fleet on spot; the on-demand baseline must be sized for the minimum acceptable service level.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>What scales</H3>

      <Prose>
        Throughput scales linearly with GPU count up to the network and orchestration bottleneck. Adding replicas behind a load balancer multiplies the aggregate request rate proportionally. For model-parallel deployments (tensor parallel across N GPUs per replica), the replica count is bounded by the available GPU pool divided by N; autoscaling still works, but each "instance" in the autoscaler's view is a multi-GPU pod. Kubernetes cluster autoscaler integrates with the HPA or KEDA to provision new nodes when pods cannot be scheduled due to insufficient GPU resources — this is the two-level scaling loop: the HPA/KEDA scales pods, and the cluster autoscaler scales nodes to accommodate the pods.
      </Prose>

      <Prose>
        Standard HPA handles up to approximately one hundred replicas in a single Deployment without operational friction. Beyond that, the Kubernetes control plane scheduling latency becomes a meaningful fraction of the cold-start time, and the etcd watch load from many rapid pod state changes can create control plane instability. At that scale, the correct architecture moves to a multi-cluster setup with a global load balancer, where each cluster autoscales independently and the global layer routes traffic across regions.
      </Prose>

      <H3>What doesn't scale</H3>

      <Prose>
        Per-token decode latency does not improve with more replicas. Each replica's decode speed is bounded by HBM bandwidth — roughly 80 GB/s on A100, 120 GB/s on H100 HBM3. Adding replicas improves throughput (more requests served per second in aggregate) but does not reduce how long any individual user waits for their tokens to stream. If per-token latency is the binding constraint, the only lever is fewer or smaller models, quantized weights, or more GPUs per replica (which increases per-replica HBM bandwidth via tensor parallelism).
      </Prose>

      <Prose>
        Cold-start time does not improve with more replicas — it is a fixed property of the model and storage infrastructure. The only solutions are architectural: shared weight pools pre-staged on fast NVMe or RDMA storage, memory-mapped model files that allow demand-paged loading (faster time-to-first-request at the cost of some serving overhead), or always-warm pools that keep instances running and pay the idle cost continuously. The ReadWriteMany persistent volume approach used by the vLLM production stack reduces cold-start from minutes to seconds by allowing multiple pods to mount the same weight volume simultaneously, eliminating the per-pod model download.
      </Prose>

      <Prose>
        Multi-region autoscaling adds coordination complexity that does not scale away. Each region's autoscaler sees only local metrics. A global traffic spike that hits all regions simultaneously triggers scale-up events in every region at the same time, which can exhaust per-region GPU quotas simultaneously. Cloud providers impose GPU quota limits per region; hitting the quota ceiling means scale-up events are silently dropped. The correct response is to track quota utilization as an explicit metric and alarm when pending scale-up events are blocked by quota, then either request quota increases proactively or implement inter-region load shedding when one region is saturated.
      </Prose>

      <Callout accent="gold">
        A scale-up event that fires when demand already exceeds capacity is too late. The autoscaler's job is to fire when metrics predict that demand will exceed capacity in t_load minutes — not when it already has.
      </Callout>

      {/* ======================================================================
          9. FAILURE MODES AND GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>1. Flapping — oscillation between scale-up and scale-down</H3>

      <Prose>
        The classic failure mode: queue depth rises, autoscaler adds instances, new capacity clears the queue, metrics drop below scale-down threshold, autoscaler removes instances, queue builds again, repeat. If the scale-down cooldown is shorter than the time it takes for demand to recover after the scale-up, the fleet hunts between two states. The fix is asymmetric cooldowns: scale-up can happen immediately (or even ahead of demand), scale-down waits for a long stabilization window (twenty to thirty minutes minimum for LLM serving). Kubernetes HPA's <Code>stabilizationWindowSeconds</Code> on the scale-down behavior block is the right configuration knob.
      </Prose>

      <H3>2. Cold-start causing traffic loss during spikes</H3>

      <Prose>
        When a spike arrives faster than the cold-start window, demand exceeds capacity for the full load time of the new instance. During that window, requests either queue (if admission control is lenient) or are dropped (if it is strict). The queue option risks a burst of retries when the new capacity comes online — the queued requests all complete in a rush, possibly triggering a scale-down that then leaves the next spike underserved. The drop option is cleaner from a control perspective but directly harms users. The mitigation is to maintain a buffer of pre-warmed instances — minimum replicas set above the expected trough load so that burst capacity is already warm when needed.
      </Prose>

      <H3>3. Spot termination during active generation</H3>

      <Prose>
        A spot instance reclaimed during a long generation has two minutes (on most cloud providers) before termination. If the in-flight request will complete within that window, it should be allowed to finish. If not, the instance must signal the load balancer to stop routing new requests, attempt to re-route the in-flight requests to another instance (which requires the inference engine to support request migration or checkpoint), and drain gracefully. In practice, most vLLM deployments do not support mid-generation request migration; the in-flight requests are lost. The correct design puts the retry responsibility on the client: the server returns an error with a retryable status code, and the client resubmits. This works for idempotent requests but not for stateful conversations where context is lost.
      </Prose>

      <H3>4. GPU quota limits hit before autoscaling target is reached</H3>

      <Prose>
        Every cloud provider enforces GPU quota per region per account. A scale-up event that requests ten new H100 instances when the account quota has only three available is silently fulfilled partially — three instances start, seven requests are dropped by the cloud API. The autoscaler sees three new instances, not ten, and may not recognize that the scale-up was constrained. Symptom: scale-up events fire, instance count increases by less than expected, queue depth does not clear. Detection: monitor the cloud provider's <Code>RequestLimitExceeded</Code> event in CloudTrail or equivalent. Mitigation: quota increase requests must be submitted before the projected peak, not when it is already happening.
      </Prose>

      <H3>5. Metric staleness causing delayed reactions</H3>

      <Prose>
        Prometheus scrapes metrics from the inference engine every fifteen to thirty seconds by default. The autoscaler polls Prometheus every fifteen seconds. A KV cache saturation event that takes thirty seconds to appear in Prometheus and another fifteen seconds to trigger the autoscaler is forty-five seconds late before a scale-up is even requested, plus five minutes of cold-start. For interactive products, reducing the Prometheus scrape interval to five seconds and the autoscaler poll interval to five seconds cuts the detection lag significantly at the cost of higher metric cardinality and scrape load. The alternative is push-based alerting: the inference engine pushes a scale-up signal directly to the autoscaler API when it detects saturation, bypassing the polling loop entirely.
      </Prose>

      <H3>6. Predictive model drift under changed traffic patterns</H3>

      <Prose>
        A Holt-Winters model fitted on three months of historical data learns the traffic pattern that existed three months ago. If the product launches a major feature, enters a new market, or goes viral, the historical pattern is no longer predictive. The model will under-scale for the new traffic regime until it adapts — which takes several seasonal periods (days to weeks). The mitigation is a hybrid approach: the predictive component provides the base scaling, and a reactive component with no cooldown adds capacity immediately when the reactive signal fires. The reactive component catches the model drift; the predictive component handles the expected load.
      </Prose>

      <H3>7. Noisy-neighbor in shared GPU clusters</H3>

      <Prose>
        When multiple workloads share GPU nodes — either through Multi-Process Service or through separate containers on the same node — a neighbor with high HBM bandwidth consumption degrades the inference engine's memory throughput without registering in the inference engine's own metrics. The inference engine sees higher-than-expected latency, the autoscaler fires a scale-up, new instances land on the same shared nodes, and the problem compounds. Detection: track per-node HBM bandwidth utilization alongside per-pod metrics. Mitigation: use node taints and tolerations to dedicate GPU nodes to the inference workload, or set pod anti-affinity rules that prevent two heavy workloads from landing on the same node.
      </Prose>

      <H3>8. Incorrect utilization target causing systematic oscillation</H3>

      <Prose>
        The target utilization in the autoscaler configuration — for example, <Code>target_ongoing_requests</Code> in Ray Serve or the HPA's <Code>averageValue</Code> — is set based on a theoretical or estimated capacity per replica. If the actual serving rate per replica differs from the estimate (due to longer-than-expected prompts, higher batch variability, or a changed model), the target is wrong. A target set too high keeps the fleet underscaled and the queue building; a target set too low keeps the fleet overscaled and wastes money. The correct procedure is to run a load test at representative traffic to measure actual per-replica throughput at the desired KV utilization level, then set the target based on measured values rather than estimates.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The following sources are primary documentation and foundational papers. All verified as of April 2026.
      </Prose>

      <Prose>
        <strong>Kubernetes HPA documentation.</strong> "Horizontal Pod Autoscaling." kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/. The authoritative reference for HPA configuration, including custom metrics via the <Code>autoscaling/v2</Code> API. Kubernetes 1.33 added configurable HPA tolerance, documented at the linked blog post. The behavior block for scale-up and scale-down policies — including <Code>stabilizationWindowSeconds</Code> — is the correct place to configure the asymmetric cooldowns described in this topic.
      </Prose>

      <Prose>
        <strong>KEDA project documentation.</strong> "KEDA — Kubernetes Event-driven Autoscaling." keda.sh/docs/2.19/. KEDA 2.19 is the current stable release. The Prometheus scaler documentation (keda.sh/docs/2.19/scalers/) describes the configuration used in Section 4e. The concepts page explains the relationship between KEDA's ScaledObject and the underlying HPA it drives. The vLLM production stack integration is documented at docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html.
      </Prose>

      <Prose>
        <strong>Knative Serving autoscaling.</strong> "About autoscaling." knative.dev/docs/serving/autoscaling/. The KPA panic mode behavior, concurrency configuration, and metric types are documented in the linked subtopics. The distinction between soft and hard concurrency limits — and the correct mapping to inference engine <Code>max_num_seqs</Code> — is covered in the concurrency configuration page.
      </Prose>

      <Prose>
        <strong>AWS SageMaker autoscaling.</strong> "Automatic scaling of Amazon SageMaker AI models." docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html. Covers both predefined metrics and custom CloudWatch metrics for target tracking policies. The prescriptive guidance on right-sizing and auto-scaling inference systems (docs.aws.amazon.com/prescriptive-guidance/latest/gen-ai-inference-architecture-and-best-practices-on-aws) provides production recommendations for GPU-based LLM serving on AWS.
      </Prose>

      <Prose>
        <strong>Ray Serve autoscaling guide.</strong> "Ray Serve Autoscaling." docs.ray.io/en/latest/serve/autoscaling-guide.html. Documents <Code>target_ongoing_requests</Code>, <Code>upscale_delay_s</Code>, <Code>downscale_delay_s</Code>, and the LLM-specific deployment configuration. The architecture overview for Ray Serve LLM (docs.ray.io/en/latest/serve/llm/architecture/overview.html) explains how autoscaling interacts with the vLLM engine underneath.
      </Prose>

      <Prose>
        <strong>vLLM production stack.</strong> github.com/vllm-project/production-stack. The reference Kubernetes-native deployment for vLLM, integrating Prometheus, Grafana, KEDA, and ReadWriteMany persistent volumes for fast model loading. The README documents the pod startup time reduction from minutes to seconds via shared volume mounting, which directly addresses the cold-start problem discussed in Sections 1 and 4b.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Cold-start amortization threshold</H3>

      <Prose>
        A model takes 4 minutes to load. Your policy is that cold-start overhead must not exceed 10% of total instance GPU-time. What is the minimum active window required for a newly started instance before it may be considered for scale-down? If you set the scale-down cooldown to this value, how does it interact with the autoscaling oscillation risk?
      </Prose>

      <Prose>
        Answer: From the overhead formula, <Code>overhead = t_load / (t_load + t_active) ≤ 0.10</Code>. Solving: <Code>4 / (4 + t_active) ≤ 0.10</Code>, so <Code>4 ≤ 0.10 × (4 + t_active)</Code>, giving <Code>t_active ≥ 36 minutes</Code>. Set the scale-down cooldown to 36 minutes minimum. This cooldown directly reduces oscillation risk: a scale-up event followed by rapid scale-down cannot happen faster than 36 minutes, which is longer than most transient demand spikes. The tension is that if demand genuinely drops quickly, the fleet is over-provisioned for 36 minutes — acceptable given the cold-start overhead argument.
      </Prose>

      <H3>Exercise 2 — Composite metric design</H3>

      <Prose>
        Your fleet is serving a product where KV cache utilization runs at 0.70 (within threshold), queue depth is 0 (fine), but TTFT p99 has risen to 2,400ms against a 2,000ms SLO. The autoscaler has not fired. Diagnose what is causing the TTFT breach without a queue or KV breach, and explain what change to the autoscaler would have caught it earlier.
      </Prose>

      <Prose>
        Answer: TTFT can rise without queue depth or KV utilization breaching if the requests currently being served have unusually long prompts — the prefill phase is taking longer than average, directly inflating time-to-first-token. With chunked prefill disabled, a single 10,000-token prefill blocks all decode steps for its full duration. The autoscaler should have TTFT p99 as a standalone scale-up trigger (it should in the composite rule from Section 4c). If TTFT is excluded, the system is blind to prefill-dominated latency spikes. The fix: ensure the composite decision function includes <Code>ttft_p99_ms > SLO_threshold</Code> as an independent OR branch, and if chunked prefill is not enabled, enable it to bound per-step prefill impact.
      </Prose>

      <H3>Exercise 3 — Predictive autoscaler lead time</H3>

      <Prose>
        Your workload has a daily traffic cycle peaking at 14:00 local time. Your model load time is 6 minutes. You are using a Holt-Winters predictive autoscaler. At what time should the autoscaler issue the scale-up command for the 14:00 peak, assuming a linear demand ramp from 13:30 to 14:00? What happens if the forecast underestimates the 14:00 peak by 20%?
      </Prose>

      <Prose>
        Answer: The scale-up must be issued 6 minutes before capacity is needed. If the ramp is linear from 13:30 to 14:00 (30 minutes), the autoscaler should issue the command at 13:54 (to have the instance ready at 14:00). In practice, issue at 13:50 to build in a buffer. If the forecast underestimates by 20%, the autoscaler provisions 20% fewer instances than needed at the peak. The reactive fallback fires when the queue builds, but the new instances it triggers will not be ready until 14:06 — six minutes into the peak with the fleet underscaled. The fix is to multiply the predictive target by a safety factor (1.2–1.3) to absorb forecast error, accepting the cost of slight over-provisioning in exchange for resilience to forecast underestimation.
      </Prose>

      <H3>Exercise 4 — Spot fleet cost-benefit analysis</H3>

      <Prose>
        You run a fleet of 10 on-demand H100 instances at $8/hr each for a batch inference pipeline. A spot H100 costs $3/hr but has a 5% per-hour probability of interruption, where each interruption loses an average of 15 minutes of work on the interrupted instance. Calculate the expected hourly cost of a pure spot fleet versus pure on-demand, factoring in the cost of lost work (measured in GPU-hours wasted). At what spot interruption probability does on-demand become cheaper?
      </Prose>

      <Prose>
        Answer: Pure on-demand cost: 10 × $8 = $80/hr. Pure spot cost: direct GPU cost = 10 × $3 = $30/hr. Lost work cost: with 5% interruption probability, expected interruptions per hour = 10 × 0.05 = 0.5 instances interrupted. Each interruption loses 15 minutes = 0.25 GPU-hours of work, which must be re-executed. Cost of lost work = 0.5 × 0.25 × $3 = $0.375/hr (re-running on spot). Total spot cost = $30 + $0.375 ≈ $30.38/hr. Spot wins decisively at 5% interruption probability. Breakeven: set $30 + 10p × 0.25 × $3 = $80, solving gives <Code>p = 666%</Code> — spot never becomes more expensive than on-demand on GPU cost alone. The real cost of interruptions for batch work is negligible unless the checkpoint interval is very long or the retry mechanism is expensive.
      </Prose>

      <H3>Exercise 5 — Oscillation diagnosis</H3>

      <Prose>
        A production fleet shows a recurring pattern: every 45 minutes, instance count spikes from 4 to 7, then drops back to 4, then spikes again. TTFT p99 is within SLO except during the transition back to 4 instances. The scale-down cooldown is set to 15 minutes. Diagnose the root cause and prescribe a fix.
      </Prose>

      <Prose>
        Answer: The 45-minute cycle with a 15-minute cooldown is a textbook oscillation: the fleet scales up (5 minutes to load + 15 minutes cooldown = 20 minutes before scale-down fires), then scales down after the cooldown, then demand recovers and scale-up fires again, repeating every ~45 minutes. The scale-down is happening too fast — at 15 minutes, the instances have barely amortized their cold-start overhead. The TTFT breach during transition confirms the fleet is underscaled immediately after scale-down. Fix: increase the scale-down cooldown to at least 30 minutes (preferably 36 minutes per Exercise 1 analysis for a 5-minute load time). Also add a hysteresis condition: do not scale down unless KV utilization has been below the lower threshold for the entire cooldown window, not just at the moment the cooldown expires.
      </Prose>

    </div>
  ),
};

export default autoscalingGPU;
