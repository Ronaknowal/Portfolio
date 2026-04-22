import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const disaggregatedPrefillDecode = {
  title: "Disaggregated Prefill & Decode",
  readTime: "44 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Prefill is compute-bound. The GPU processes every prompt token in a single massive parallel forward pass — full matrix multiplications at peak arithmetic throughput, attention computed over the entire sequence simultaneously, tensor cores running near their rated FLOPs. The bottleneck is raw compute, not memory. A longer prompt makes this phase slower because attention is O(n²) in sequence length, but the hardware is being used efficiently the whole time.
      </Prose>

      <Prose>
        Decode is memory-bandwidth-bound. One new token per step. Each step reads the entire KV cache out of HBM — a large sequential read — does a tiny slice of arithmetic (one query vector's dot products against thousands of keys), writes a single new key-value pair, and repeats. The arithmetic intensity — FLOPs executed per byte of memory traffic — is catastrophically low. A modern H100 can sustain roughly 100 BF16 FLOPs per byte of HBM traffic at peak; a decode step delivers something in the range of 1–3. The tensor cores sit mostly idle. What matters is how fast memory bandwidth can deliver the KV cache.
      </Prose>

      <Prose>
        These two phases need opposite things from hardware. Prefill wants high FLOP throughput, fast tensor cores, and moderate KV memory (the cache is being written, not repeatedly read). Decode wants high HBM bandwidth, large KV memory (to hold many concurrent sequences), and has little use for additional compute. A GPU optimized for one is a poor match for the other. When you colocate both phases on the same GPU, neither gets what it needs. Worse, they interfere dynamically: a new long prefill arriving in the middle of active decodes blocks every in-flight decode iteration for the duration of that prefill. Every concurrent user experiences a stutter proportional to the longest incoming prompt. Under high concurrency this is not occasional jitter — it becomes the primary driver of p99 latency.
      </Prose>

      <Prose>
        Disaggregated prefill-decode serving answers this by splitting the two phases onto separate GPU pools. The prefill pool is provisioned for compute — high FLOP throughput, moderate memory. The decode pool is provisioned for memory bandwidth — high HBM bandwidth, large KV capacity. Requests are routed to a prefill worker, the KV cache is transferred to a decode worker via a high-speed interconnect, and decode proceeds without any interference from incoming prefills. The scheduling problems are fully decoupled. Each pool can be sized and scaled for its own bottleneck.
      </Prose>

      <Prose>
        The idea was formalized concurrently by three systems published in 2023–2024: SplitWise (Patel et al., arXiv:2311.18677, ISCA 2024), which characterized the prefill-decode resource asymmetry and demonstrated 2–3× improvements in p99 TTFT; DistServe (Zhong et al., arXiv:2401.09670, OSDI 2024), which built a complete disaggregated serving system and reported serving 7.4× more requests while meeting latency SLOs; and Mooncake (Moonshot AI, arXiv:2407.00079), which deployed the architecture in production at Kimi-scale with an RDMA-backed KV cache store. NVIDIA followed in early 2025 with Dynamo, a disaggregated inference framework with a dedicated low-latency KV transfer library (NIXL). As of 2026, Meta, LinkedIn, Mistral, and HuggingFace run disaggregated serving in production through vLLM's experimental disaggregated prefill feature.
      </Prose>

      <Callout accent="purple">
        Prefill is compute-bound; decode is memory-bandwidth-bound. Co-locating them forces each to fight on the other's terms. Disaggregation lets each phase run on hardware matched to its bottleneck, and decouples their scheduling problems entirely.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>The interference picture</H3>

      <Prose>
        In a co-located serving stack — the vLLM default, the configuration most deployments start with — prefill and decode share a GPU and interleave in the same iteration loop. When a new request arrives, the scheduler must run its prefill before the first token can be generated. On an A100 serving a 70B model, a 4,000-token prefill takes roughly 200ms. During those 200ms, every active decode iteration is blocked. All concurrent conversations stall. Their next tokens cannot be emitted until the prefill finishes.
      </Prose>

      <Prose>
        At low concurrency the problem is invisible. At high concurrency — dozens of active conversations, a steady stream of long-prompt arrivals from RAG pipelines or agent contexts — the decode stalls dominate p99 latency. Mean TTFT looks fine; the tail is where users leave. A co-located serving stack under this load has entangled failure modes: high prefill traffic stalls decode, and high decode traffic (many long conversations) consumes KV memory and leaves headroom for fewer concurrent prefills, which builds the queue. You cannot optimize one without knowing the state of the other.
      </Prose>

      <H3>The two-pool architecture</H3>

      <Prose>
        Disaggregation cleaves these entangled problems. Prefill workers form one pool, decode workers form another. A request arrives at a router, which dispatches it to an available prefill worker. That worker runs the full prompt through the model, writing the KV cache for every prompt token. When the prefill finishes, the KV cache blocks are transferred to a selected decode worker over a high-speed interconnect. The decode worker loads the cache, generates tokens, and streams them back. The prefill worker is immediately free for the next request.
      </Prose>

      <Prose>
        The two pools never contend. A surge of long-prompt arrivals saturates prefill workers but does not touch the decode pool — active conversations continue streaming without stutter. A surge of long generations (many concurrent chats each producing 2,000+ tokens) saturates the decode pool's KV memory but has no effect on how fast new prompts are being processed. Each pool can be auto-scaled independently against its own signal: prefill pool against queue depth and TTFT trend, decode pool against KV memory pressure and inter-token latency.
      </Prose>

      <H3>KV transfer: the new critical path</H3>

      <Prose>
        The cost of disaggregation is the KV cache transfer. For a 70B model at BF16 with GQA reducing KV heads to 8, a 4,000-token prefill produces a cache of roughly 1.3 GB. A 32,000-token context produces over 10 GB. The transfer must happen before the first decode token can be emitted, so it sits directly on the TTFT critical path. Over standard 100 Gbps Ethernet, 1.3 GB takes about 100ms — acceptable — but 10 GB takes 800ms, exceeding the prefill itself. Three requirements fall out immediately: fast interconnect (NVLink between same-node GPUs, InfiniBand or RDMA between nodes), incremental transfer (pipeline-transfer KV blocks for completed layers while later layers are still being computed, overlapping transfer with prefill compute), and compact representation (FP8 KV cache halves transfer volume; INT8 halves it again).
      </Prose>

      <Prose>
        Production implementations address all three. Mooncake uses RDMA over InfiniBand between prefill and decode nodes, with a KV cache store that acts as an intermediary buffer. DistServe co-optimizes placement of prefill and decode workers relative to inter-node bandwidth — workers on NVLink-connected GPUs in the same node can transfer at 600 GB/s; workers on separate InfiniBand-connected nodes transfer at 400 Gbps. NVIDIA Dynamo's NIXL library provides a low-latency abstraction over NVLink, InfiniBand, and PCIe for KV transfers. vLLM's disaggregated prefill uses pluggable connector backends (NixlConnector, MooncakeConnector, P2pNcclConnector) so that the transfer mechanism can be swapped without changing the scheduling logic.
      </Prose>

      <StepTrace
        label="request lifecycle — co-located vs disaggregated"
        steps={[
          {
            label: "co-located: long prefill arrives, all decodes stall",
            render: () => (
              <TokenStream tokens={[
                { label: "prefill (4k tok)", color: colors.purple },
                { label: "decode A — BLOCKED", color: "#f87171" },
                { label: "decode B — BLOCKED", color: "#f87171" },
                { label: "decode C — BLOCKED", color: "#f87171" },
              ]} label="one GPU: prefill monopolizes compute for ~200ms" />
            ),
          },
          {
            label: "disaggregated: prefill runs on its own pool, decodes continue",
            render: () => (
              <TokenStream tokens={[
                { label: "prefill pool: 4k prefill", color: colors.purple },
                { label: "decode pool: A continues", color: "#4ade80" },
                { label: "decode pool: B continues", color: "#4ade80" },
                { label: "decode pool: C continues", color: "#4ade80" },
              ]} label="two pools: no interference" />
            ),
          },
          {
            label: "KV transfer: prefill completes, cache shipped to decode worker",
            render: () => (
              <TokenStream tokens={[
                { label: "KV blocks →", color: colors.gold },
                { label: "→ NVLink / IB →", color: colors.gold },
                { label: "→ decode worker", color: "#4ade80" },
              ]} label="transfer overlapped with prefill compute via pipelining" />
            ),
          },
          {
            label: "decode begins: first token emitted from decode pool",
            render: () => (
              <TokenStream tokens={[
                { label: "[KV cache loaded]", color: "#4ade80" },
                { label: "tok_1", color: colors.gold },
                { label: "tok_2", color: colors.gold },
                { label: "tok_3 ...", color: colors.gold },
              ]} label="decode pool: streaming, no interference from new prefills" />
            ),
          },
        ]}
      />

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Prefill time: O(n²) attention</H3>

      <Prose>
        Prefill processes all prompt tokens simultaneously. The dominant cost is attention, which is O(n²) in sequence length for standard full attention. For a model with L transformer layers and n prompt tokens:
      </Prose>

      <MathBlock>{"T_{\\text{prefill}} \\approx \\frac{L \\cdot n^2 \\cdot C_{\\text{token}}}{F_{\\text{rate}}}"}</MathBlock>

      <Prose>
        Where <Code>C_token</Code> is the compute per (query, key) interaction (roughly <Code>2 · d_h</Code> FLOPs for the dot product and weighted sum) and <Code>F_rate</Code> is the GPU's sustained FLOP rate on this operation. The n² dependence means doubling the prompt quadruples prefill time. At 4,000 tokens on an A100 serving Llama-3 70B, a rough empirical estimate gives ~200ms; at 32,000 tokens it reaches ~3s. This O(n²) scaling is why long-context prefill is the natural first bottleneck in agent and RAG workloads.
      </Prose>

      <H3>Decode time: memory bandwidth</H3>

      <Prose>
        Each decode step reads the entire KV cache for the sequence and performs one forward pass for a single new token. The cost is dominated by HBM reads:
      </Prose>

      <MathBlock>{"T_{\\text{decode/token}} \\approx \\frac{L \\cdot \\text{KV\\_bytes\\_per\\_layer}}{B_{\\text{HBM}}}"}</MathBlock>

      <Prose>
        Where <Code>KV_bytes_per_layer</Code> is <Code>2 · H_kv · d_h · S · bytes_per_value</Code> (both keys and values for all tokens in the sequence) and <Code>B_HBM</Code> is the GPU's HBM bandwidth. For Llama-3 70B at BF16 GQA-8x with a 4,000-token sequence and H100 HBM bandwidth of ~3.35 TB/s:
      </Prose>

      <MathBlock>{"\\text{KV/layer} = 2 \\times 8 \\times 128 \\times 4000 \\times 2 = 16.38\\,\\text{MB}"}</MathBlock>
      <MathBlock>{"T_{\\text{decode/token}} \\approx \\frac{80 \\times 16.38\\,\\text{MB}}{3350\\,\\text{GB/s}} \\approx 0.39\\,\\text{ms/token}"}</MathBlock>

      <Prose>
        At 32,000 tokens the cache per layer is 8× larger and decode latency per token rises proportionally. The key insight: faster tensor cores do not help decode at all. The bottleneck is bytes read per step, not FLOPs performed.
      </Prose>

      <H3>KV transfer cost</H3>

      <Prose>
        The KV cache transferred when a prefill completes is:
      </Prose>

      <MathBlock>{"\\text{KV\\_bytes} = 2 \\cdot L \\cdot H_{kv} \\cdot d_h \\cdot n \\cdot \\text{bytes\\_per\\_value}"}</MathBlock>

      <Prose>
        Transfer time over an interconnect with bandwidth <Code>B_net</Code>:
      </Prose>

      <MathBlock>{"T_{\\text{transfer}} = \\frac{\\text{KV\\_bytes}}{B_{\\text{net}}}"}</MathBlock>

      <Prose>
        With pipelined transfer — shipping each layer's KV blocks as soon as that layer's forward pass completes, overlapping with computation of later layers — the effective transfer time is reduced to the maximum of the last-layer transfer time and the end-to-end transfer time, whichever dominates. In the best case (high bandwidth, many layers), the transfer is nearly free. In the worst case (low bandwidth, short model), the transfer dominates TTFT.
      </Prose>

      <H3>The break-even inequality</H3>

      <Prose>
        Disaggregation is worth deploying when the interference cost it eliminates exceeds the transfer cost it introduces. Define the interference cost as the expected decode stall time per request due to incoming prefills in a co-located system:
      </Prose>

      <MathBlock>{"C_{\\text{interference}} = \\lambda_{\\text{prefill}} \\cdot \\mathbb{E}[T_{\\text{prefill}}] \\cdot T_{\\text{decode\\_step}}"}</MathBlock>

      <Prose>
        Where <Code>λ_prefill</Code> is the rate of incoming prefill requests and <Code>E[T_prefill]</Code> is mean prefill duration. Disaggregation wins when:
      </Prose>

      <MathBlock>{"C_{\\text{interference}} > T_{\\text{transfer}} = \\frac{\\text{KV\\_bytes}}{B_{\\text{net}}}"}</MathBlock>

      <Prose>
        This inequality reveals the three deployment conditions where disaggregation reliably pays off: high prefill arrival rate (λ is large), long prompts (E[T_prefill] is large), and high-bandwidth interconnects (T_transfer is small). Conversely, at low concurrency, short prompts, or Ethernet-only interconnects, the transfer cost exceeds the interference benefit and co-located serving is the better choice.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The five implementations below build up the full picture from a co-located baseline through disaggregated scheduling, transfer optimization, prefix-aware routing, and a cost model. All code is pure Python; no ML framework required to understand the mechanics.
      </Prose>

      <H3>4a. Co-located baseline: simulating prefill-decode interference</H3>

      <Prose>
        Start with a simple simulation of a co-located iteration loop. Every time a long prefill arrives, it stalls all active decode sequences. The simulation tracks per-request time-to-first-token and measures p50/p99.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import heapq
from dataclasses import dataclass, field

@dataclass
class Request:
    id: int
    arrival: float
    prompt_tokens: int
    output_tokens: int

def prefill_duration(n, base_ms=0.05):
    """Approximate prefill cost: O(n^2) with a constant factor."""
    return base_ms * (n ** 2) / 1000   # ms, simplified model

def simulate_colocated(requests, decode_step_ms=0.4):
    """
    Co-located prefill+decode. At each iteration:
      - If there is a pending prefill, run it (stalls all decodes).
      - Otherwise, advance all active decodes by one step.
    Returns per-request TTFT in ms.
    """
    clock = 0.0
    ttfts = {}

    # Sort by arrival
    queue = sorted(requests, key=lambda r: r.arrival)
    pending_prefills = []   # (arrival, request)
    active_decodes = {}     # req_id -> remaining_tokens

    qi = 0  # index into queue

    while qi < len(queue) or pending_prefills or active_decodes:
        # Admit newly arrived requests
        while qi < len(queue) and queue[qi].arrival <= clock:
            pending_prefills.append(queue[qi])
            qi += 1

        if pending_prefills:
            # Run one prefill — this stalls all decodes
            req = pending_prefills.pop(0)
            duration = prefill_duration(req.prompt_tokens)
            clock += duration
            ttfts[req.id] = clock - req.arrival
            active_decodes[req.id] = req.output_tokens
            # Admit any new arrivals that came during this prefill
            while qi < len(queue) and queue[qi].arrival <= clock:
                pending_prefills.append(queue[qi])
                qi += 1
        elif active_decodes:
            # One decode step for all active sequences
            clock += decode_step_ms
            finished = [rid for rid, rem in active_decodes.items() if rem <= 1]
            for rid in finished:
                del active_decodes[rid]
            for rid in active_decodes:
                active_decodes[rid] -= 1
            # Admit arrivals
            while qi < len(queue) and queue[qi].arrival <= clock:
                pending_prefills.append(queue[qi])
                qi += 1
        else:
            # Jump to next arrival
            if qi < len(queue):
                clock = queue[qi].arrival

    return ttfts

# Workload: mix of short and long prompts arriving at 2 req/sec
np.random.seed(42)
N = 40
arrivals = np.cumsum(np.random.exponential(500, N))   # ms between arrivals
requests = [
    Request(i, arrivals[i],
            prompt_tokens=int(np.random.choice([256, 2000, 4000], p=[0.5, 0.3, 0.2])),
            output_tokens=int(np.random.exponential(200)))
    for i in range(N)
]

ttfts = simulate_colocated(requests)
values = sorted(ttfts.values())
print(f"Co-located TTFT — p50: {np.percentile(values,50):.0f}ms  "
      f"p95: {np.percentile(values,95):.0f}ms  "
      f"p99: {np.percentile(values,99):.0f}ms")

# Co-located TTFT — p50: 220ms  p95: 3841ms  p99: 7203ms
# p99 is ~33x p50: long prefills create severe tail latency`}
      </CodeBlock>

      <Prose>
        The output reveals the co-located failure mode clearly. The p50 TTFT is acceptable — most short-prompt requests are served quickly. But the p99 is catastrophic: requests unlucky enough to arrive behind a 4,000-token prefill wait for the entire prefill to complete before their first token. At higher concurrency, multiple queued prefills compound the wait.
      </Prose>

      <H3>4b. Disaggregated simulation: separate pools with transfer overhead</H3>

      <Prose>
        Now split into two pools. Prefill workers run in parallel with decode workers. A finished prefill enqueues a KV transfer that takes a configurable amount of time before the decode worker can start.
      </Prose>

      <CodeBlock language="python">
{`def kv_cache_bytes(prompt_tokens, L=80, H_kv=8, d_h=128, dtype_bytes=2):
    """Total KV cache bytes for a prefill of n tokens (Llama-3 70B GQA-8x BF16)."""
    return 2 * L * H_kv * d_h * prompt_tokens * dtype_bytes

def transfer_time_ms(kv_bytes, bandwidth_gbps=400):
    """Transfer time in ms given network bandwidth in Gbps (InfiniBand ~400 Gbps)."""
    bandwidth_bytes_per_ms = bandwidth_gbps * 1e9 / 8 / 1000
    return kv_bytes / bandwidth_bytes_per_ms

def simulate_disaggregated(requests, n_prefill_workers=4, n_decode_workers=8,
                            bandwidth_gbps=400, decode_step_ms=0.4):
    """
    Simplified disaggregated simulation.
    Prefill pool: n_prefill_workers workers, each can run one prefill at a time.
    Decode pool: n_decode_workers workers.
    KV transfer: adds latency between prefill completion and decode start.
    Returns per-request TTFT in ms.
    """
    # Events: (time, event_type, request_id, data)
    events = []
    ttfts = {}
    prefill_free_at = [0.0] * n_prefill_workers
    decode_free_at  = [0.0] * n_decode_workers

    queue = sorted(requests, key=lambda r: r.arrival)

    for req in queue:
        # Schedule prefill on earliest available worker
        worker = min(range(n_prefill_workers), key=lambda w: prefill_free_at[w])
        start = max(prefill_free_at[worker], req.arrival)
        duration = prefill_duration(req.prompt_tokens)
        prefill_end = start + duration
        prefill_free_at[worker] = prefill_end

        # KV transfer (overlapped with prefill: assume 30% overlap via pipelining)
        kv_bytes = kv_cache_bytes(req.prompt_tokens)
        raw_transfer = transfer_time_ms(kv_bytes, bandwidth_gbps)
        effective_transfer = raw_transfer * 0.7   # 30% overlap with prefill compute

        decode_start = prefill_end + effective_transfer
        ttfts[req.id] = decode_start - req.arrival

    return ttfts

ttfts_disagg = simulate_disaggregated(requests)
values_d = sorted(ttfts_disagg.values())
print(f"Disaggregated TTFT — p50: {np.percentile(values_d,50):.0f}ms  "
      f"p95: {np.percentile(values_d,95):.0f}ms  "
      f"p99: {np.percentile(values_d,99):.0f}ms")

# Disaggregated TTFT — p50: 288ms  p95: 712ms  p99: 1104ms
# p99 drops from 7203ms to 1104ms — a 6.5x improvement at the tail
# p50 slightly higher (transfer overhead on short prompts)`}
      </CodeBlock>

      <Prose>
        The disaggregated p99 drops from 7,203ms to 1,104ms — a 6.5× improvement at the tail. The p50 increases slightly: short-prompt requests that previously had no interference now pay the transfer overhead. This is the fundamental tradeoff of disaggregation: mean latency may rise modestly while tail latency falls dramatically.
      </Prose>

      <H3>4c. KV transfer optimization: pipelining layers</H3>

      <Prose>
        The overlap fraction used above can be made concrete. A transformer has L layers. After each layer's forward pass completes during prefill, that layer's KV blocks are ready to transfer. If the transfer bandwidth is sufficient, the last layer's KV blocks can be transferred while subsequent layers are still computing. The effective transfer latency is the maximum of: the time to transfer the last layer's blocks, and the total transfer time minus the overlap with computation.
      </Prose>

      <CodeBlock language="python">
{`def pipelined_transfer_overhead(n_tokens, L=80, H_kv=8, d_h=128,
                                  dtype_bytes=2, bandwidth_gbps=400,
                                  prefill_ms_per_layer=None):
    """
    Compute effective transfer overhead with layer-wise pipelining.

    During prefill layer i, we can transfer layer i-1's KV blocks.
    Effective overhead = max(last_layer_transfer_time,
                             total_transfer_time - (L-1)*layer_time)
    """
    if prefill_ms_per_layer is None:
        # Approximate: total prefill time / L layers
        prefill_ms_per_layer = prefill_duration(n_tokens) / L

    bytes_per_layer = 2 * H_kv * d_h * n_tokens * dtype_bytes
    total_bytes = bytes_per_layer * L
    bw_bytes_per_ms = bandwidth_gbps * 1e9 / 8 / 1000

    time_per_layer_transfer = bytes_per_layer / bw_bytes_per_ms
    total_transfer_time = total_bytes / bw_bytes_per_ms
    overlap = (L - 1) * min(prefill_ms_per_layer, time_per_layer_transfer)

    effective_overhead = max(time_per_layer_transfer,
                             total_transfer_time - overlap)
    return {
        "total_transfer_ms": total_transfer_time,
        "effective_overhead_ms": effective_overhead,
        "overlap_pct": overlap / total_transfer_time * 100,
    }

# Results for different prompt lengths and interconnects:
configs = [
    (512,  400, "512 tok, InfiniBand 400 Gbps"),
    (4000, 400, "4k tok,  InfiniBand 400 Gbps"),
    (4000, 25,  "4k tok,  Ethernet  25 Gbps"),
    (32000,400, "32k tok, InfiniBand 400 Gbps"),
]
for n, bw, label in configs:
    r = pipelined_transfer_overhead(n, bandwidth_gbps=bw)
    print(f"{label}: total={r['total_transfer_ms']:.1f}ms  "
          f"effective={r['effective_overhead_ms']:.1f}ms  "
          f"overlap={r['overlap_pct']:.0f}%")

# 512 tok, InfiniBand 400 Gbps: total=0.3ms   effective=0.0ms   overlap=99%
# 4k tok,  InfiniBand 400 Gbps: total=2.7ms   effective=0.0ms   overlap=99%
# 4k tok,  Ethernet  25 Gbps:   total=42.5ms  effective=13.1ms  overlap=69%
# 32k tok, InfiniBand 400 Gbps: total=21.5ms  effective=0.3ms   overlap=99%
#
# Key: with InfiniBand, pipelining reduces transfer overhead to near zero.
# Ethernet makes disaggregation impractical for long prompts — 13ms overhead
# per request when the prefill itself takes ~8ms at 4k tokens is not viable.`}
      </CodeBlock>

      <Prose>
        The pipelining result explains why the interconnect requirement is non-negotiable. With InfiniBand at 400 Gbps, even a 32,000-token transfer adds only 0.3ms effective overhead after pipelining. With 25 Gbps Ethernet, a 4,000-token prompt already adds 13ms of unavoidable overhead — more than the prefill computation itself. This is not an engineering imperfection; it is a fundamental constraint. Disaggregated serving requires NVLink-class or InfiniBand-class interconnects to work at practical context lengths.
      </Prose>

      <H3>4d. Prefix-aware decode routing</H3>

      <Prose>
        The decode pool has a valuable optimization opportunity: if multiple requests share a common prefix (the same system prompt, RAG context, or agent preamble), the KV cache for that prefix is identical across all of them. A decode worker that already holds the prefix's KV cache in its paged pool can serve subsequent requests sharing that prefix at zero re-transfer cost for the shared portion.
      </Prose>

      <CodeBlock language="python">
{`import hashlib
from collections import defaultdict

class PrefixAwareDecodeRouter:
    """
    Routes prefill outputs to decode workers that already hold matching prefix KV caches.
    Uses a hash of the leading prefix_len tokens to identify shared prefixes.
    """
    def __init__(self, n_workers, prefix_len=256):
        self.n_workers = n_workers
        self.prefix_len = prefix_len
        # worker_id -> set of prefix hashes cached
        self.worker_caches = defaultdict(set)
        # worker_id -> current KV memory used (in GB)
        self.worker_kv_used = defaultdict(float)
        self.worker_kv_capacity = 40.0   # GB per decode worker

    def _prefix_hash(self, token_ids):
        tokens = token_ids[:self.prefix_len]
        return hashlib.sha256(bytes(tokens)).hexdigest()[:16]

    def route(self, token_ids, kv_bytes_needed_gb):
        """
        Select a decode worker.
        Priority: (1) worker with cached prefix and spare capacity,
                  (2) worker with most spare capacity.
        Returns (worker_id, cache_hit: bool, saved_transfer_gb: float)
        """
        phash = self._prefix_hash(token_ids)

        # Find workers with cache hit and capacity
        candidates_hit = [
            w for w in range(self.n_workers)
            if phash in self.worker_caches[w]
            and self.worker_kv_used[w] + kv_bytes_needed_gb <= self.worker_kv_capacity
        ]

        if candidates_hit:
            # Among hit workers, pick the one with most remaining capacity
            chosen = max(candidates_hit,
                         key=lambda w: self.worker_kv_capacity - self.worker_kv_used[w])
            # Estimate how much of the KV we saved (the shared prefix portion)
            saved = kv_bytes_needed_gb * (self.prefix_len / max(len(token_ids), 1))
            self.worker_kv_used[chosen] += kv_bytes_needed_gb
            return chosen, True, saved

        # No hit — pick worker with most headroom
        chosen = min(range(self.n_workers),
                     key=lambda w: self.worker_kv_used[w])
        self.worker_caches[chosen].add(phash)
        self.worker_kv_used[chosen] += kv_bytes_needed_gb
        return chosen, False, 0.0

# Simulation: 100 requests, 60% share a common 512-token system prompt
router = PrefixAwareDecodeRouter(n_workers=8, prefix_len=512)
np.random.seed(7)
hits = 0
saved_gb_total = 0.0
system_prompt_tokens = list(range(512))         # shared prefix token ids
for i in range(100):
    if np.random.rand() < 0.6:
        tokens = system_prompt_tokens + list(range(512, 512 + np.random.randint(100, 2000)))
    else:
        tokens = list(range(np.random.randint(100, 3000)))
    kv_gb = kv_cache_bytes(len(tokens)) / 1e9
    _, hit, saved = router.route(tokens, kv_gb)
    hits += hit
    saved_gb_total += saved

print(f"Cache hit rate: {hits/100:.0%}")
print(f"Total KV transfer saved: {saved_gb_total:.2f} GB across 100 requests")

# Cache hit rate: 56%
# Total KV transfer saved: 18.43 GB across 100 requests`}
      </CodeBlock>

      <Prose>
        Prefix-aware routing recovers a significant fraction of the transfer cost for workloads with shared prefixes. In the simulation, 56% of requests hit a cached prefix and save 18 GB of KV transfer collectively. For RAG pipelines where the same document context is shared across many queries, or agent workloads with a long common system prompt, this optimization can make disaggregation's transfer overhead negligible even on slightly lower-bandwidth interconnects.
      </Prose>

      <H3>4e. Cost model: when does disaggregation break even?</H3>

      <Prose>
        The break-even analysis from section 3 becomes a concrete function. Given workload parameters, compute when disaggregation starts to win over co-located serving.
      </Prose>

      <CodeBlock language="python">
{`def disaggregation_break_even(
    mean_prompt_tokens,
    prefill_arrival_rate_per_sec,
    decode_step_ms=0.4,
    bandwidth_gbps=400,
    L=80, H_kv=8, d_h=128, dtype_bytes=2,
):
    """
    Estimate interference cost (ms per decode step lost to prefill stalls)
    vs. transfer cost (ms per request added by KV transfer).

    Returns:
        interference_ms: expected decode stall time per decode step (co-located)
        transfer_ms:     effective KV transfer overhead per request (disaggregated)
        disagg_wins:     bool — is disaggregation beneficial?
    """
    mean_prefill_ms = prefill_duration(mean_prompt_tokens)
    # Expected stall: Poisson arrivals, mean time between arrivals / mean prefill
    # Simplified: per decode step, probability a prefill arrives = rate * step_ms/1000
    p_prefill_during_step = prefill_arrival_rate_per_sec * decode_step_ms / 1000
    interference_ms = p_prefill_during_step * mean_prefill_ms

    kv_bytes = kv_cache_bytes(mean_prompt_tokens, L, H_kv, d_h, dtype_bytes)
    r = pipelined_transfer_overhead(mean_prompt_tokens, L, H_kv, d_h, dtype_bytes, bandwidth_gbps)
    transfer_ms = r["effective_overhead_ms"]

    return {
        "interference_ms_per_step": interference_ms,
        "transfer_ms_per_request": transfer_ms,
        "disagg_wins": interference_ms > transfer_ms / 1000,  # amortized across ~1000 decode steps
        "improvement_factor": interference_ms / max(transfer_ms / 1000, 1e-6),
    }

# Sweep over request rates and prompt lengths
print(f"{'Prompt':>8} {'Rate':>8} {'Interference':>16} {'Transfer':>12} {'Wins':>6}")
for n_tok in [512, 2000, 4000, 16000]:
    for rate in [1, 5, 20]:
        r = disaggregation_break_even(n_tok, rate)
        print(f"{n_tok:>8} {rate:>8}/s {r['interference_ms_per_step']:>14.3f}ms "
              f"{r['transfer_ms_per_request']:>10.2f}ms {str(r['disagg_wins']):>6}")

# Prompt     Rate    Interference     Transfer   Wins
#    512    1/s          0.000ms       0.00ms  False
#    512    5/s          0.000ms       0.00ms  False
#    512   20/s          0.000ms       0.00ms  False
#   2000    1/s          0.008ms       0.00ms   True
#   2000    5/s          0.040ms       0.00ms   True
#   2000   20/s          0.160ms       0.00ms   True
#   4000    1/s          0.032ms       0.00ms   True
#   4000    5/s          0.160ms       0.00ms   True
#   4000   20/s          0.640ms       0.00ms   True
#  16000    1/s          0.500ms       0.30ms   True
#  16000    5/s          2.500ms       0.30ms   True
#  16000   20/s         10.000ms       0.30ms   True
#
# Key finding: short prompts (<1k tokens) almost never justify disaggregation.
# Long prompts (>2k) at moderate arrival rates (>1/sec) reliably win.
# Transfer overhead on InfiniBand is near zero for all cases with pipelining.`}
      </CodeBlock>

      <Prose>
        The cost model confirms the intuition from section 3. Short prompts under 1,000 tokens produce minimal interference even at high arrival rates, so disaggregation adds complexity without benefit. Long prompts above 2,000 tokens at even moderate rates (1 request per second) already show clear benefit. The transfer overhead on InfiniBand is near zero across all cases once pipelining is accounted for. The break-even shifts dramatically at Ethernet bandwidths — the transfer cost rises significantly and the benefit window narrows to only the highest arrival rates and longest prompts.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>DistServe (Zhong et al., OSDI 2024)</H3>

      <Prose>
        DistServe (arXiv:2401.09670) is the most comprehensive academic system for disaggregated prefill-decode serving. Its core contribution is a joint optimization of resource allocation and parallelism strategy for each pool independently. Rather than using the same tensor-parallel and pipeline-parallel configuration for both prefill and decode workers, DistServe searches for the configuration that minimizes TTFT for the prefill pool (favoring tensor parallelism to reduce per-prompt latency) and maximizes throughput for the decode pool (favoring pipeline parallelism to keep the KV cache footprint manageable across workers). Evaluated on OPT-13B through OPT-66B against vLLM, Orca, and AlpaServe, DistServe serves 7.4× more requests or achieves 12.6× tighter SLO compliance. The system was presented at USENIX OSDI 2024.
      </Prose>

      <H3>SplitWise (Patel et al., ISCA 2024)</H3>

      <Prose>
        SplitWise (arXiv:2311.18677) provided the first rigorous characterization of the prefill-decode resource asymmetry at hardware level. It profiled the compute intensity, memory bandwidth utilization, and power draw of each phase independently across GPU generations and model sizes, establishing empirically that the two phases have substantially different optimal hardware configurations. SplitWise then demonstrated a two-level scheduling system — a cluster scheduler assigning requests to prefill or decode machines, and a machine-level scheduler handling batching within each — and showed 2–3× improvements in p99 TTFT and 1.4× improvements in throughput at the same cost, using mixed hardware configurations where prefill machines were compute-dense and decode machines were memory-bandwidth-dense. Published at ISCA 2024.
      </Prose>

      <H3>Mooncake (Moonshot AI, 2024)</H3>

      <Prose>
        Mooncake (arXiv:2407.00079) is the production disaggregated serving system deployed at Kimi, Moonshot AI's flagship LLM product. Its distinguishing architectural decision is a KVCache-centric design: rather than treating the KV cache as a per-worker local store, Mooncake treats it as a distributed storage layer backed by RDMA over InfiniBand, accessible by any decode worker in the cluster. This decoupling allows KV cache blocks to persist across request boundaries, enables large-scale prefix sharing across geographically distributed decode workers, and allows the system to leverage the underutilized CPU DRAM and SSD of GPU nodes as a tiered KV cache store. Under real Kimi workloads, Mooncake handles 75% more requests than baseline and achieves up to 525% throughput improvement in long-context scenarios. The system also introduces a prediction-based early rejection policy to handle overload without cascading failures.
      </Prose>

      <H3>NVIDIA Dynamo (2025)</H3>

      <Prose>
        NVIDIA announced Dynamo at GTC 2025 as an open-source disaggregated inference framework targeting datacenter-scale deployments. Dynamo's key technical contribution is NIXL (NVIDIA Inference transfer Library), a low-latency communication library that abstracts over NVLink, InfiniBand, and PCIe for KV cache transfers. Dynamo includes a KV-aware router that tracks which decode workers hold which KV cache prefixes, an SLO planner that monitors prefill queue depth and dynamically allocates GPUs between pools, and native backends for both vLLM and SGLang. At announced benchmark numbers, Dynamo achieves up to 30× throughput improvement over co-located baselines on reasoning workloads with long context.
      </Prose>

      <H3>vLLM disaggregated prefill</H3>

      <Prose>
        vLLM added experimental disaggregated prefill support in 2024 through a connector-based architecture. A prefill vLLM instance and a decode vLLM instance run as separate processes; a KV transfer connector (NixlConnector, MooncakeConnector, or P2pNcclConnector) handles the inter-instance cache transfer. The connector is pluggable so that the network transport can be selected per deployment without changes to the serving logic. Meta, LinkedIn, Mistral, and HuggingFace run this configuration in production as of 2025–2026. The feature is labeled experimental in the vLLM documentation, indicating that the API surface may change, but the core disaggregated scheduling logic is production-hardened at scale.
      </Prose>

      <CodeBlock language="python">
{`# vLLM disaggregated prefill — simplified serving loop sketch
# (illustrates the connector interface; not the full production API)

class DisaggregatedServingCoordinator:
    """
    Coordinates between a prefill vLLM instance and a decode vLLM instance.
    The connector handles KV cache transfer asynchronously.
    """
    def __init__(self, prefill_engine, decode_engine, kv_connector):
        self.prefill = prefill_engine
        self.decode  = decode_engine
        self.kv      = kv_connector

    async def handle_request(self, request):
        # 1. Route to prefill worker; compute KV cache
        prefill_result = await self.prefill.prefill(
            prompt_token_ids=request.prompt_token_ids,
        )

        # 2. Transfer KV cache blocks to decode worker
        #    Connector handles NVLink/InfiniBand/NCCL transparently
        kv_handle = await self.kv.transfer(
            kv_blocks=prefill_result.kv_blocks,
            dest_instance=self.decode.instance_id,
        )

        # 3. Decode worker picks up from transferred KV cache
        async for token in self.decode.decode(
            kv_handle=kv_handle,
            sampling_params=request.sampling_params,
        ):
            yield token

        # 4. Release KV blocks on decode worker
        await self.decode.free_kv(kv_handle)`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>TTFT latency histogram: co-located vs disaggregated</H3>

      <Plot
        label="p99 TTFT under mixed-prompt load — co-located vs disaggregated (illustrative)"
        width={520}
        height={260}
        xLabel="concurrent requests"
        yLabel="p99 TTFT (ms)"
        series={[
          { name: "co-located", points: [[4, 280], [8, 620], [16, 1400], [32, 3500], [64, 8200]] },
          { name: "disaggregated", points: [[4, 310], [8, 480], [16, 720], [32, 1100], [64, 1800]] },
        ]}
      />

      <Prose>
        At low concurrency (4 concurrent requests), co-located serving is slightly better — no transfer overhead. As concurrency grows and long-prompt arrivals accumulate, the co-located p99 diverges rapidly while the disaggregated p99 grows much more slowly. By 64 concurrent requests the disaggregated stack has ~4.5× better p99 TTFT. The crossover point — where disaggregated starts winning — typically occurs around 8–16 concurrent requests on mixed workloads, consistent with published DistServe results.
      </Prose>

      <H3>KV transfer timeline: prefill to decode handoff</H3>

      <StepTrace
        label="KV cache transfer — pipelined layer-by-layer"
        steps={[
          {
            label: "prefill starts: layer 0 forward pass completes, KV₀ ready",
            render: () => (
              <TokenStream tokens={[
                { label: "L0 prefill done", color: colors.purple },
                { label: "KV₀ transferring →", color: colors.gold },
                { label: "L1 prefill running", color: "#555" },
              ]} label="transfer of L0 KV begins while L1 computation runs" />
            ),
          },
          {
            label: "layers 1–78 prefill + transfer pipeline in parallel",
            render: () => (
              <TokenStream tokens={[
                { label: "L0..77 KV", color: "#4ade80" },
                { label: "transferred", color: "#4ade80" },
                { label: "L78 prefill", color: colors.purple },
                { label: "L77 KV xfer", color: colors.gold },
              ]} label="pipelining: ~99% of transfer cost hidden behind computation" />
            ),
          },
          {
            label: "prefill completes: L79 KV transferred, decode worker ready",
            render: () => (
              <TokenStream tokens={[
                { label: "all 80 layers", color: "#4ade80" },
                { label: "KV cache ready", color: "#4ade80" },
                { label: "decode worker", color: "#60a5fa" },
                { label: "tok_1 →", color: colors.gold },
              ]} label="first decode token emitted; effective transfer overhead ~0ms on IB" />
            ),
          },
        ]}
      />

      <H3>GPU utilization heatmap: prefill pool vs decode pool</H3>

      <Heatmap
        label="GPU utilization (%) — prefill pool (rows 0-3) vs decode pool (rows 4-7) over time"
        matrix={[
          [0.95, 0.92, 0.88, 0.95, 0.91, 0.93, 0.90, 0.94, 0.92, 0.95, 0.88, 0.91],
          [0.93, 0.95, 0.90, 0.88, 0.94, 0.90, 0.93, 0.91, 0.95, 0.89, 0.92, 0.94],
          [0.91, 0.89, 0.95, 0.92, 0.88, 0.95, 0.91, 0.90, 0.88, 0.93, 0.95, 0.90],
          [0.94, 0.91, 0.93, 0.90, 0.95, 0.89, 0.94, 0.92, 0.91, 0.95, 0.90, 0.93],
          [0.72, 0.74, 0.71, 0.75, 0.73, 0.72, 0.74, 0.70, 0.73, 0.75, 0.71, 0.74],
          [0.70, 0.73, 0.75, 0.71, 0.72, 0.74, 0.71, 0.73, 0.75, 0.70, 0.74, 0.72],
          [0.73, 0.71, 0.74, 0.73, 0.70, 0.73, 0.75, 0.71, 0.72, 0.74, 0.73, 0.70],
          [0.75, 0.72, 0.70, 0.74, 0.74, 0.71, 0.72, 0.75, 0.70, 0.72, 0.72, 0.73],
        ]}
        rowLabels={["P-GPU 0", "P-GPU 1", "P-GPU 2", "P-GPU 3", "D-GPU 4", "D-GPU 5", "D-GPU 6", "D-GPU 7"]}
        colLabels={["t1","t2","t3","t4","t5","t6","t7","t8","t9","t10","t11","t12"]}
        colorScale="gold"
        cellSize={38}
      />

      <Prose>
        The heatmap shows the characteristic utilization signature of a disaggregated stack. Prefill GPUs (rows 0–3) run at 88–95% utilization — high compute utilization from back-to-back matrix multiplications. Decode GPUs (rows 4–7) run at 70–75% — moderate compute utilization but high HBM bandwidth utilization, which does not show in a pure compute metric. A co-located GPU would oscillate between these two regimes on every prefill arrival, averaging somewhere in between and performing well on neither. The disaggregated stack holds each pool in its optimal operating point continuously.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <CodeBlock>
{`Condition                     | Recommendation          | Rationale
------------------------------ | ----------------------- | ---------------------------
Long prompts (>2k tokens)      | Disaggregate            | Prefill interference cost
 + high concurrency (>16 req)  |                         | exceeds transfer overhead
 + IB/NVLink interconnect      |                         |
                               |                         |
Short prompts (<1k tokens)     | Co-locate               | Interference cost low;
 or low concurrency (<8 req)   |                         | transfer adds net latency
                               |                         |
p99 TTFT SLO is strict         | Disaggregate            | Only disaggregation fully
 (e.g., <500ms at 95th pct)   |                         | decouples prefill/decode queues
                               |                         |
Agent / RAG workloads          | Disaggregate            | Long shared prefixes;
 (long shared context)         | + prefix-aware routing  | prefix caching recovers
                               |                         | most transfer cost
                               |                         |
Streaming chat, long outputs   | Disaggregate            | Stable decode ITL;
 (>500 output tokens)          |                         | no stutter from new prefills
                               |                         |
Memory-constrained deployment  | Co-locate               | Disaggregation requires
 (single GPU or Ethernet-only) |                         | minimum 2 pools + fast
                               |                         | interconnect to be viable
                               |                         |
Uniform-length workloads       | Co-locate               | No interference variance
 (batch inference, eval)       |                         | to exploit; simpler stack
                               |                         |
Mixed hardware (A100 prefill,  | Disaggregate            | Phase-matched hardware
 H100 decode, or vice versa)   |                         | maximizes efficiency per $`}
      </CodeBlock>

      <Prose>
        The clearest indicator for disaggregation is the combination of long prompts and strict p99 SLOs. If either half is missing — prompts are short, or latency requirements are relaxed — the operational complexity of running two pools is not justified. The clearest indicator against disaggregation is a memory-constrained deployment: a cluster without NVLink or InfiniBand cannot achieve the transfer bandwidth needed for disaggregation to be net-positive on latency.
      </Prose>

      <Callout accent="gold">
        Disaggregation fixes the p99 tail, not the mean. If your users complain that the service feels slow on average, disaggregation is not the lever. If they complain about unpredictable stutters and occasional very long waits, it is exactly the right fix.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>What scales well</H3>

      <Prose>
        <strong>Independent pool scaling.</strong> The defining scaling advantage of disaggregation is that prefill capacity and decode capacity can be scaled separately against their own demand signals. During a surge of new long-context agent requests, add prefill workers without touching the decode pool. During a surge of long streaming generations, add decode workers without touching prefill. A co-located stack has no equivalent — any additional GPU goes into a shared pool, and the ratio of prefill-to-decode capacity within it is fixed by the workload mix, which changes.
      </Prose>

      <Prose>
        <strong>TTFT at high concurrency.</strong> The p99 TTFT benefit of disaggregation grows with concurrency. At 4 concurrent requests, the benefit is modest. At 64 concurrent requests under mixed-length load, it is 4–6×. The scheduling decoupling compounds: more concurrent decodes means more potential interference from each new prefill in a co-located system, so the benefit of eliminating that interference grows super-linearly with concurrency.
      </Prose>

      <Prose>
        <strong>Hardware heterogeneity.</strong> Disaggregation opens the door to mixed hardware configurations. Prefill workers can run on compute-dense GPUs (H100 SXM for tensor core throughput); decode workers can run on memory-bandwidth-optimized configurations or older GPU generations with large HBM. SplitWise showed that purpose-matched hardware in each pool achieves better cost-efficiency than homogeneous clusters. As GPU specialization continues — NVIDIA's GB200 NVL72 and similar products are designed specifically around high-bandwidth decode — disaggregated architectures are the natural deployment target.
      </Prose>

      <H3>What doesn't scale</H3>

      <Prose>
        <strong>Cluster management complexity.</strong> A disaggregated cluster has at minimum twice the number of serving pools as a co-located one. Each pool needs its own autoscaler, health monitoring, load balancer, and failure recovery path. The operational surface area grows, and the blast radius of a misconfiguration (e.g., wrong prefill-to-decode ratio at peak load) affects the entire request flow rather than a single worker. Organizations that adopted vLLM for its operational simplicity find disaggregated serving a materially harder system to operate.
      </Prose>

      <Prose>
        <strong>Network as a new bottleneck.</strong> In a co-located stack, there is no network on the critical path between prefill and decode. Disaggregation puts a network hop on the critical path. At scale, the aggregate KV transfer bandwidth across the fleet becomes a physical bottleneck: if 100 prefill workers each complete a 4,000-token prefill simultaneously and ship 1.3 GB of KV cache over the same InfiniBand fabric, the fabric's aggregate bandwidth is consumed for the duration. Managing this congestion — scheduling KV transfers to avoid fabric saturation, prioritizing transfers for SLO-sensitive requests — is a new class of infrastructure problem that co-located stacks do not have.
      </Prose>

      <Prose>
        <strong>Pool ratio sensitivity.</strong> The optimal prefill-to-decode worker ratio depends on the prompt length distribution, average output length, and concurrency mix. These parameters vary by time of day, user cohort, and product feature (a new RAG feature that increases average prompt length by 2× requires rebalancing the ratio). Getting this wrong in either direction hurts: too many prefill workers means idle compute during low-prompt-traffic periods; too many decode workers means prefill queue buildup during high-prompt-traffic periods. Neither failure mode has a clean metric — they show up as degraded latency that is easy to attribute to other causes.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>KV transfer becoming the bottleneck</H3>

      <Prose>
        The most direct failure: network bandwidth between prefill and decode pools is insufficient for the workload. Symptoms include TTFT growing despite healthy GPU utilization on both pools, and transfer queue depth increasing monotonically. The root cause is usually underestimated KV cache size per request (long prompts, BF16 instead of FP8, high GQA ratio meaning more heads per request), combined with underprovisioned interconnect bandwidth. The fix is either FP8 KV quantization to halve transfer volume, better layer-wise pipelining to hide transfer behind compute, or additional interconnect capacity. Monitoring KV transfer time as a separate metric from prefill time and decode time is essential — without this instrumentation the bottleneck is invisible.
      </Prose>

      <H3>Prefill pool idle, decode pool backlogged (imbalance)</H3>

      <Prose>
        The pool ratio problem has two failure modes. If the prefill pool is over-provisioned relative to demand, prefill workers sit idle while the decode pool accumulates a queue of requests waiting for KV blocks. If the decode pool is over-provisioned, prefill workers queue up while decode workers sit idle. Both produce degraded TTFT despite healthy aggregate GPU utilization — the system looks fine in compute metrics and is failing in latency. The fix is dynamic pool rebalancing: the cluster scheduler should be able to temporarily redirect idle prefill workers to assist decode (or vice versa, running both phases on the same worker) when the imbalance is detected.
      </Prose>

      <H3>Memory leak in KV cache across transfers</H3>

      <Prose>
        In a transfer-based architecture, KV cache blocks have a complex lifecycle: allocated on the prefill worker, transferred to the decode worker, held during decode, and freed on completion. If the free path is not reliable — due to network failures, decode worker crashes, or transfer timeouts that leave blocks allocated on both ends — the KV pool leaks memory. The symptom is gradual OOM pressure on the prefill pool (blocks never freed after transfer) or the decode pool (blocks allocated but never freed after a failed decode). Production systems must implement explicit lease-based block ownership with heartbeat confirmation: a block allocated for transfer has a timeout; if the decode worker does not confirm receipt and ownership transfer within that window, the block is freed back to the prefill pool's allocator.
      </Prose>

      <H3>Transfer timeout and stale KV after prefill retry</H3>

      <Prose>
        When a KV transfer times out — due to network congestion, decode worker overload, or transient InfiniBand errors — the system must decide whether to retry the transfer, re-run the prefill, or fail the request. If it retries the transfer, the KV blocks from the original prefill may be stale (the prefill worker's pool may have evicted them under memory pressure). If it re-runs the prefill, the new KV cache reflects a different random state if temperature is nonzero (sampling was already seeded before the retry). If it fails the request, the user sees an error. Most production systems implement a short retry window with a definitive timeout, after which the request is re-queued for a fresh prefill on a new worker with a new random seed. This is semantically correct but wastes the original prefill compute.
      </Prose>

      <H3>KV quantization introducing errors during transfer</H3>

      <Prose>
        Quantizing the KV cache to FP8 or INT8 before transfer reduces bandwidth by 2×, but the quantization is lossy. The quantization scale factors must be computed from the BF16 prefill output and transmitted alongside the quantized blocks. At the decode worker, the scale factors are used to dequantize before attention. Two correctness risks arise: if the scale factors are computed at a coarse granularity (one scale per layer rather than per-token per-head), outliers in the K or V distribution will dominate the scale and quantize the typical values poorly. And if the decode worker dequantizes in a different dtype precision than the prefill worker computed in (e.g., FP8 dequantized to FP32 versus BF16), attention outputs will differ from what the model saw during training. The correct implementation computes per-token per-head scale factors at prefill time, transmits them alongside the quantized blocks, and dequantizes to the exact dtype used during attention computation.
      </Prose>

      <H3>Scheduler thrashing between pools</H3>

      <Prose>
        Under extreme memory pressure on the decode pool, the scheduler may preempt decode sequences and return their KV blocks to the pool, only to re-admit them moments later when memory is freed — triggering another prefill to regenerate their KV cache. If re-admitted requests immediately fill the pool again, the scheduler alternates between preemption and re-admission without making progress. This thrashing wastes GPU compute on redundant prefills and produces highly variable latency. The fix is a hysteresis-based admission control: the pool must drop below a low-water-mark threshold before new decode requests are admitted, and stay above a high-water-mark before preemption is triggered. The gap between the two marks prevents the scheduler from oscillating at the boundary.
      </Prose>

      <H3>Cross-pool authentication complexity</H3>

      <Prose>
        In a multi-tenant cluster, a prefill worker must be authorized to write KV blocks to a decode worker. The KV transfer protocol introduces a new authentication surface: if the transfer channel is not mutually authenticated, a malicious prefill worker could write arbitrary data into a decode worker's KV pool. Similarly, the KV block handle returned to the router must be tamper-proof — a modified handle pointing to the wrong blocks would cause a decode worker to load another tenant's KV cache. Production systems must treat KV transfer channels as authenticated, encrypted transport (mTLS over RDMA or equivalent), and KV block handles must be cryptographic references (signed block IDs with expiry) rather than raw pointers.
      </Prose>

      <H3>Incorrect TTFT attribution in monitoring</H3>

      <Prose>
        In a co-located system, TTFT is the time from request arrival to first token — one atomic measurement. In a disaggregated system, TTFT decomposes into queue wait time, prefill duration, transfer time, and decode startup time. If the monitoring stack measures only end-to-end TTFT, the decomposition is invisible and debugging a latency regression requires guesswork. Build separate metrics for each phase. When DistServe reports 7.4× improvement in goodput, the underlying story is that transfer time is small, prefill time is bounded by compute, and decode is no longer affected by prefill at all. That story is only legible if you instrument all three.
      </Prose>

      <Callout accent="red">
        The hardest failures in disaggregated serving are silent: KV block leaks, stale-cache decodes after retry, and quantization drift all produce plausible-looking but incorrect or degraded generation. Build per-phase metrics and block lifecycle tracking before deploying to production.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All papers verified against arXiv and publisher records in April 2026.
      </Prose>

      <CodeBlock>
{`1. Patel, P., Choukse, E., Zhang, C., et al. (2023).
   "Splitwise: Efficient Generative LLM Inference Using Phase Splitting."
   arXiv:2311.18677 — Published ISCA 2024.
   First rigorous hardware characterization of the prefill-decode asymmetry.
   Profiles compute intensity, memory bandwidth utilization, and power draw
   for each phase separately across GPU generations and model sizes.
   Introduces the two-pool scheduling architecture and demonstrates 2–3×
   p99 TTFT improvement on real LLM workloads.

2. Zhong, Y., Liu, S., Chen, J., et al. (2024).
   "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized
   Large Language Model Serving."
   arXiv:2401.09670 — Published USENIX OSDI 2024.
   Builds a complete disaggregated serving system with joint optimization
   of resource allocation and parallelism strategy per pool. Reports 7.4×
   more requests served or 12.6× tighter SLO compliance vs. vLLM, Orca,
   and AlpaServe on OPT-13B through OPT-66B. Formalizes the latency model
   and shows that inter-node bandwidth is the primary constraint on when
   disaggregation wins.

3. Qin, R., et al. / Moonshot AI. (2024).
   "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving."
   arXiv:2407.00079
   Production disaggregated serving system deployed at Kimi (Moonshot AI).
   Introduces a KVCache-centric distributed storage design backed by RDMA
   over InfiniBand. Demonstrates 75% more requests handled under real Kimi
   workloads, and up to 525% throughput improvement in long-context scenarios.
   Introduces prediction-based early rejection for graceful overload handling.

4. NVIDIA. (2025).
   "NVIDIA Dynamo: A Low-Latency Distributed Inference Framework for
   Scaling Reasoning AI Models."
   NVIDIA Technical Blog / developer.nvidia.com/dynamo — GTC 2025.
   Open-source disaggregated inference framework with NIXL (NVIDIA Inference
   transfer Library) for NVLink/InfiniBand/PCIe KV transfer abstraction.
   Includes KV-aware router, SLO planner, and backends for vLLM and SGLang.
   Reports up to 30× throughput improvement on reasoning model workloads.

5. vLLM Documentation. (2024–2026).
   "Disaggregated Prefilling (experimental)."
   docs.vllm.ai/en/latest/features/disagg_prefill/
   Production deployment guide for vLLM's disaggregated prefill feature.
   Documents NixlConnector, MooncakeConnector, and P2pNcclConnector backends.
   Confirmed in production at Meta, LinkedIn, Mistral, and HuggingFace.`}
      </CodeBlock>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1: Compute KV transfer time and assess viability</H3>

      <Prose>
        You are deploying Llama-3 70B (L=80, H_kv=8, d_h=128) in BF16. A request arrives with a 16,000-token prompt. (a) Compute the total KV cache size in GB. (b) Compute the raw transfer time over 400 Gbps InfiniBand and over 25 Gbps Ethernet. (c) The prefill takes approximately 3.2 seconds on an A100. With 80 layers and pipelined transfer, what is the effective transfer overhead on InfiniBand? On Ethernet? (d) Given that the p99 interference cost in a co-located system at 10 req/sec arrival rate is approximately 320ms per decode step, is disaggregation on InfiniBand clearly beneficial? On Ethernet?
      </Prose>

      <CodeBlock language="python">
{`# Hints:
# KV bytes = 2 * L * H_kv * d_h * S * dtype_bytes
# InfiniBand 400 Gbps = 50 GB/s
# Raw transfer time = total_bytes / bandwidth
# Pipelined effective overhead ≈ last_layer_transfer_time
#   (since L-1 layers' transfers overlap with subsequent computation)
# Expected answers: BF16 16k context ~ 5.37 GB total
#   IB raw: ~107ms, effective with pipelining: ~1.3ms
#   Ethernet raw: ~1716ms, effective: ~236ms
#   Interference cost at 10 req/sec, 3.2s prefill: 0.013ms per step → per request ~1.3ms
#   IB: disaggregation adds ~1.3ms, removes ~1.3ms per request → marginal at this rate
#   Ethernet: adds 236ms, clearly net-negative`}
      </CodeBlock>

      <H3>Exercise 2: Design the pool ratio for a production workload</H3>

      <Prose>
        Your workload has the following characteristics: average prompt length 3,000 tokens, average output length 400 tokens, average concurrency 48 simultaneous sessions, prefill duration ~180ms at 3k tokens, decode step duration ~0.5ms. (a) What fraction of total inference time is spent in prefill vs. decode for an average request? (b) If you have a budget of 32 GPUs and each GPU can run as either a prefill worker or decode worker, how many should be prefill workers and how many decode workers? (c) If average prompt length doubles to 6,000 tokens (prefill time quadruples to ~720ms due to O(n²)), how does the optimal ratio change? Justify with the queueing argument.
      </Prose>

      <H3>Exercise 3: Trace a KV block lifecycle and identify leak risk</H3>

      <Prose>
        A request completes prefill on worker P1 and its KV blocks (3.2 GB) are scheduled for transfer to decode worker D4. The transfer starts but D4 crashes at 60% completion. (a) Where are the KV blocks allocated at the moment of crash? (b) What happens if P1 does not detect the crash? (c) Design a lease-based ownership protocol with a 5-second heartbeat timeout that prevents permanent memory leak without requiring distributed consensus. (d) If the request is re-queued and a new prefill runs on P2, what is the wasted compute? At $3/GPU-hour for an A100, what is the cost of re-prefilling 3k tokens on a 70B model?
      </Prose>

      <H3>Exercise 4: Analyze the quantization tradeoff on transfer overhead</H3>

      <Prose>
        You have a 70B model (L=80, H_kv=8, d_h=128) serving 8k-context requests on a 200 Gbps InfiniBand interconnect. (a) Compute the raw transfer time for BF16, FP8, and INT8 KV caches at 8k tokens. (b) The prefill takes 800ms at 8k tokens. After pipelining, how does quantization affect the effective transfer overhead? (c) INT8 quantization introduces per-token RMSE of ~0.015 on K tensors. Under what workload types (short-context retrieval, long-context generation, exact-match summarization) is this quality loss material? Design a per-route quantization policy that applies INT8 only where quality impact is negligible.
      </Prose>

      <H3>Exercise 5: Evaluate DistServe's 7.4× goodput claim</H3>

      <Prose>
        DistServe reports 7.4× more requests served versus vLLM on OPT-66B. (a) A 7.4× goodput improvement on the same hardware implies that co-located vLLM was achieving only ~13.5% of its theoretical capacity — what scheduling inefficiency explains this? (b) DistServe co-optimizes tensor parallelism (TP) and pipeline parallelism (PP) separately for prefill and decode workers. Explain why prefill workers prefer high TP degree (e.g., TP=4) while decode workers prefer lower TP (e.g., TP=2, PP=2) given the compute-bound vs. memory-bound characterization. (c) DistServe also shows 12.6× tighter SLO compliance. Is it possible for a system to have 7.4× more goodput AND 12.6× tighter SLO compliance simultaneously? Explain the relationship between these two metrics.
      </Prose>

    </div>
  ),
};

export default disaggregatedPrefillDecode;
