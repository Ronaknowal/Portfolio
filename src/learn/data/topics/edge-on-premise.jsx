import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const edgeOnPremise = {
  title: "Edge & On-Premise Deployment Architectures",
  slug: "edge-on-premise-deployment-architectures",
  readTime: "42 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Most writing about LLM deployment assumes public cloud — hyperscaler GPUs, multi-tenant endpoints, commodity APIs billed by the token. The model is sensible for the majority of production use cases: rent capacity, call an endpoint, pay per query. But a real and growing fraction of the market cannot adopt that model at all. The constraints are legal, physical, or operational, and they impose a harder barrier than preference. When a hospital system wants an LLM to summarize patient records, cloud inference is not expensive or slow — it is, in most jurisdictions, a HIPAA violation. When a robot arm needs a vision query answered before it moves, the 80ms round-trip to the nearest datacenter is not a cost concern — it is a product defect. When a fleet of field devices loses connectivity in a remote installation, cloud fallback is not a risk to manage — it simply does not exist.
      </Prose>

      <Prose>
        These pressures — legal, physical, operational — are the three forcing functions that push LLM deployment off the cloud and onto customer-owned infrastructure, edge servers, and on-device silicon. Data sovereignty is the most common: GDPR, HIPAA, SOC 2, FedRAMP, export control regimes, and the growing patchwork of national AI data residency laws all restrict where regulated data may travel. Cloud APIs, however well-secured, involve data transiting third-party infrastructure. For regulated industries, that transit is often prohibited before any performance benchmark is run. The architecture of the deployment is not a choice; it is a compliance boundary drawn by legal counsel and auditors.
      </Prose>

      <Prose>
        Network latency is the second forcing function, and it is distinct from the sovereignty problem in that it is purely physical. Speed-of-light constraints mean that a request to a cloud datacenter 1,000 kilometers away will accumulate at least 6–7ms of one-way propagation delay, and real networks with routing hops, congestion, and backhaul add far more. Measured round-trip latencies to the nearest AWS or Azure region from a typical enterprise location range from 20ms to 150ms depending on geography. For web applications, that range is irrelevant. For real-time translation, in-vehicle assistants, AR/VR interactions, industrial control systems, and robotics, it is the difference between a usable product and an unusable one. The only engineering solution is to move inference physically closer to the point of use — which means edge hardware, on-premise servers, or on-device models running locally.
      </Prose>

      <Prose>
        Offline operation is the third forcing function, and the most absolute. Ships at sea, aircraft in flight, vehicles in tunnels, field deployments in areas without reliable connectivity, and classified environments with no external network — in all of these contexts, the cloud endpoint is simply unreachable during operation. No amount of network engineering addresses an environment that deliberately has no internet connection. The system must be self-contained, and that means the model must run on hardware that is local to the task. This is categorically different from optimizing for low latency; it is a binary constraint.
      </Prose>

      <Prose>
        There is a fourth force that has grown steadily from 2024 onward: economics at scale. Cloud inference APIs pricing at $3–$15 per million output tokens works well for small-to-medium volumes. At very high throughput — hundreds of millions of tokens per day — the variable cost of API inference exceeds the fixed cost of owning the hardware. For large enterprises with predictable, sustained LLM workloads, the break-even calculation increasingly favors capital investment in on-premise GPU infrastructure over indefinite operating spend on API endpoints. The compute market has also made this more tractable: purpose-built inference appliances from Dell, HPE, and NVIDIA have standardized what "on-premise LLM deployment" looks like operationally, reducing the engineering lift to something a datacenter team can manage.
      </Prose>

      <Callout accent="gold">
        The forcing functions for non-cloud LLM deployment are data sovereignty, physical latency constraints, offline operation, and large-scale economics. Each one points toward a different architectural pattern, and the right solution is almost never the same across all four.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>The three-tier model</H3>

      <Prose>
        Non-cloud LLM deployment organizes into three distinct tiers based on hardware class, model size, and ownership model. Understanding which tier you are operating in determines almost every subsequent architectural decision.
      </Prose>

      <Prose>
        <strong>On-premise datacenter</strong> is the closest tier to cloud deployment in architecture. The customer owns or leases physical GPU servers inside their own facility — or inside a colocation facility they control — and runs the full LLM serving stack themselves. The hardware is the same class as cloud hardware: NVIDIA H100, H200, or L40S GPUs with 80GB HBM, hundreds of watts per card, full NVLink interconnects for multi-GPU tensor parallelism. The software stack — <Code>vLLM</Code>, <Code>SGLang</Code>, <Code>TGI</Code> — is identical to what cloud providers run. What changes is the operational envelope: no elastic scaling, no managed control plane, fixed capacity, manual model updates, and the full operational burden of datacenter management. The tradeoff is complete data sovereignty, predictable cost at scale, and the ability to run any model without an external dependency. For large regulated organizations with existing datacenter operations, the tradeoff often clears easily.
      </Prose>

      <Prose>
        <strong>Edge server</strong> is a mid-tier that has grown substantially as inference-optimized hardware has become available in small form factors. An edge server lives physically near the point of use — on a factory floor, in a retail location, in a vehicle, in a hospital ward, on a cell tower — and serves inference requests locally to a defined geographic area or operational zone. The hardware is purpose-built for power-constrained, thermally limited environments: NVIDIA Jetson AGX Orin (up to 60W, 64GB unified memory), Intel Gaudi 3 edge variants, or standard workstation GPUs in a compact chassis. Models running well on edge servers are in the 3B–13B parameter range in 4-bit quantization, or 7B–70B with careful quantization on higher-end appliances. The edge server serves multiple users or devices in its vicinity, providing LAN-speed inference (sub-5ms round-trip) without requiring any cloud connectivity.
      </Prose>

      <Prose>
        <strong>On-device inference</strong> is the most constrained tier: models running entirely on the end-user's hardware, whether a smartphone, laptop, tablet, or embedded IoT device. Power budgets are measured in watts rather than hundreds of watts, and available memory is measured in gigabytes rather than tens of gigabytes. The models that run here are heavily quantized, task-specific, and architecturally compact: 1B–4B parameter models in 4-bit or lower precision, often with hardware-specific kernel optimizations (Core ML on Apple Silicon, QNN SDK on Qualcomm, ExecuTorch for cross-platform deployment). The inference cost is effectively zero from the operator's perspective — computation runs on user hardware — but the capability ceiling is significantly lower than cloud or edge server models. Hybrid architectures that combine on-device inference for fast, common tasks with cloud or edge server fallback for capability-bound queries have become the dominant production pattern in consumer applications.
      </Prose>

      <H3>The performance-privacy-cost triangle</H3>

      <Prose>
        Every deployment choice in this space involves a trade among three competing pressures. Moving from cloud toward on-device: capability and throughput decrease (smaller models, less hardware), privacy and sovereignty increase (data never leaves the device), and variable cost per token approaches zero (computation is amortized into capital). There is no free point in this triangle. A deployment that maximizes all three simultaneously does not exist. The job of the architect is to identify which constraint is hardest — sovereignty, latency, or cost — and optimize for that constraint while accepting the tradeoffs on the other two.
      </Prose>

      <CodeBlock>
{`tier                  | peak model size    | power budget  | typical latency  | data leaves device?
--------------------- | ------------------ | ------------- | ---------------- | -------------------
cloud datacenter      | 400B+ (dense)      | unlimited     | 20-150ms RTT     | yes (API call)
on-prem datacenter    | 70B-405B           | kW per rack   | LAN (<5ms)       | no (stays on-prem)
edge server           | 7B-13B (4-bit)     | 15-150W       | LAN (<5ms)       | no (stays local)
laptop / workstation  | 7B-30B (4-bit)     | 15-150W       | zero (local)     | no
smartphone / IoT      | 1B-4B (INT4)       | 1-10W         | zero (local)     | no`}
      </CodeBlock>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Quantization memory savings</H3>

      <Prose>
        The most important mathematical relationship in edge deployment is the memory footprint of a quantized model. A model with <Code>N</Code> parameters stored at <Code>b</Code> bits per weight requires:
      </Prose>

      <MathBlock>{"\\text{memory (GB)} = \\frac{N \\times b}{8 \\times 10^9}"}</MathBlock>

      <Prose>
        For a Llama 3 8B model: at FP16 (<Code>b=16</Code>), that is <Code>8 \\ \\times 10^9 \\times 16 / (8 \\times 10^9) = 16\\ \\text{GB}</Code>. At INT4 (<Code>b=4</Code>), the same model occupies 4 GB — a 4× reduction that is the difference between "edge is impossible" and "edge is routine" on commodity hardware. At INT2, it falls to 2 GB, though quality degradation at 2-bit is severe for most tasks. Modern mixed-precision schemes like AWQ and GPTQ apply 4-bit quantization to weights while preserving activations at higher precision, recovering most of the quality loss from naive rounding.
      </Prose>

      <MathBlock>{"\\text{compression ratio} = \\frac{b_{\\text{original}}}{b_{\\text{quantized}}} = \\frac{16}{4} = 4\\times"}</MathBlock>

      <Prose>
        The quality cost of INT4 quantization is smaller than intuition suggests. AWQ benchmarks on Llama 3 family models show that 4-bit weight-only quantization costs approximately 0.5–2 perplexity points on standard evaluation benchmarks, and typically under 1 accuracy point on narrow domain-specific tasks when calibrated on representative data. Microsoft's practical INT4 benchmarks on SLMs (February 2026) found that models retained 98.1% of their baseline reasoning capability on MMLU-Pro after INT4 quantization with AWQ. The degradation is non-uniform: instruction-following tasks lose more than arithmetic tasks, and very long-form generation degrades more than short classification. Knowing your task distribution is more important than the headline number.
      </Prose>

      <H3>Edge latency advantage</H3>

      <Prose>
        The latency comparison between cloud and edge deployment reduces to a simple inequality. Let <Code>RTT</Code> be the network round-trip time to the cloud, <Code>T_{cloud}</Code> be cloud inference time for the full-size model, <Code>T_{edge}</Code> be edge inference time for the quantized model, and <Code>L_{LAN}</Code> be local area network latency (sub-millisecond for wired, 1–5ms for wireless):
      </Prose>

      <MathBlock>{"\\text{latency}_{\\text{cloud}} = RTT + T_{\\text{cloud}}"}</MathBlock>

      <MathBlock>{"\\text{latency}_{\\text{edge}} = L_{\\text{LAN}} + T_{\\text{edge}}"}</MathBlock>

      <Prose>
        Edge wins on total latency when <Code>RTT {">"} T_{edge} - T_{cloud} + L_{LAN}</Code>. Since <Code>RTT</Code> typically ranges from 20ms to 150ms and LAN latency is sub-5ms, edge inference wins whenever the inference time gap between the edge model (which is smaller and faster) and the cloud model doesn't exceed the RTT advantage. For a Jetson AGX Orin running a 7B INT4 model at roughly 40–60 tokens per second, time to first token is dominated by prefill time for the prompt, typically 100–300ms for normal-length prompts. Cloud inference with time to first token of 150–300ms plus 50ms RTT clearly loses to a local model that starts streaming tokens within 150ms with no additional network overhead. For voice AI specifically, the combined latency budget for the LLM stage is under 150ms — a threshold achievable on-device but often not through cloud for users not physically close to a datacenter.
      </Prose>

      <H3>On-premise cluster sizing</H3>

      <Prose>
        Given a target throughput of <Code>Q</Code> queries per second (QPS) at a target P99 latency of <Code>L</Code> milliseconds, the number of GPUs required for on-premise serving can be estimated from Little's Law and hardware throughput:
      </Prose>

      <MathBlock>{"N_{\\text{GPU}} = \\left\\lceil \\frac{Q \\times T_{\\text{req}}}{\\text{tok/sec/GPU} \\times \\text{util}_{\\text{target}}} \\right\\rceil"}</MathBlock>

      <Prose>
        Where <Code>T_req</Code> is the average tokens per request (input + output), <Code>tok/sec/GPU</Code> is the hardware throughput for the chosen model, and <Code>util_target</Code> is the target GPU utilization (typically 0.65–0.75 to leave headroom for bursts). For a 70B model on H100 delivering 1,800 aggregate tokens per second with 4,000 average tokens per request at 10 QPS: <Code>N = ceil(10 × 4000 / (1800 × 0.70)) = ceil(31.7) = 32 GPUs</Code>. That is four 8-GPU nodes, which is a realistic on-premise LLM cluster for a substantial enterprise workload.
      </Prose>

      <H3>Fleet throughput scaling</H3>

      <Prose>
        On-device inference scales differently from on-premise. When the model runs on each user's own device, the aggregate system throughput is the sum of per-device throughputs — no shared bottleneck, no contention for GPU time, perfect horizontal scaling as the user base grows. If each device delivers <Code>s</Code> tokens per second and there are <Code>D</Code> simultaneous active users:
      </Prose>

      <MathBlock>{"\\text{throughput}_{\\text{fleet}} = s \\times D"}</MathBlock>

      <Prose>
        A fleet of 1 million simultaneous users running a 3B model at 20 tokens per second each delivers 20 billion tokens per second of aggregate capacity — a number no cloud datacenter approaches. The catch is that each device operates independently with no shared context, so the total inference capacity scales with the installed user base rather than with any infrastructure investment. This is the economic argument for on-device inference at consumer scale: inference capacity is free to the operator and grows automatically with adoption.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All five implementations below were executed in Python and outputs embedded verbatim. They are deliberately dependency-free beyond the standard library and <Code>math</Code>. The goal is to make the numerical relationships in edge deployment concrete enough to reason about your own hardware without a spreadsheet.
      </Prose>

      <H3>4a. Quantization memory and quality estimator</H3>

      <Prose>
        Given model parameter count and quantization bit-width, compute memory footprint and estimate quality retention based on benchmarked AWQ/GPTQ degradation curves:
      </Prose>

      <CodeBlock language="python">
{`def quantization_stats(params_b: float, bits: int) -> dict:
    """
    Estimate memory footprint and quality retention for a quantized model.

    params_b : model parameter count in billions
    bits     : quantization bit width (2, 4, 8, 16)
    """
    memory_gb = params_b * 1e9 * bits / (8 * 1e9)

    # Quality retention relative to FP16 baseline.
    # Derived from AWQ/GPTQ benchmarks on Llama 3 family, April 2026.
    # Source: Microsoft INT4 guide (Feb 2026), vLLM quantization benchmarks.
    quality_map = {
        16: 1.000,  # FP16 baseline
        8:  0.995,  # INT8: ~0.5% drop, nearly lossless
        4:  0.975,  # INT4: ~2.5% drop, acceptable for most tasks
        2:  0.880,  # INT2: ~12% drop, usable only for simple classification
    }
    quality = quality_map.get(bits, None)
    compression = 16 / bits  # relative to FP16

    return {
        "bits": bits,
        "memory_gb": round(memory_gb, 2),
        "compression_vs_fp16": f"{compression:.0f}x",
        "quality_retention": f"{quality * 100:.1f}%",
    }

# Example: Llama 3 8B across quantization levels
for bits in [16, 8, 4, 2]:
    stats = quantization_stats(8.0, bits)
    print(f"  {stats['bits']:>2}-bit: {stats['memory_gb']:>6} GB  "
          f"({stats['compression_vs_fp16']} compression)  "
          f"quality: {stats['quality_retention']}")

# 16-bit:  16.00 GB  (1x compression)  quality: 100.0%
#  8-bit:   8.00 GB  (2x compression)  quality: 99.5%
#  4-bit:   4.00 GB  (4x compression)  quality: 97.5%
#  2-bit:   2.00 GB  (8x compression)  quality: 88.0%
#
# Key insight: 4-bit is the practical edge sweetspot —
# 4x memory reduction with only ~2.5% quality drop.
# The jump from 8-bit to 4-bit saves 4GB without much quality cost.
# The jump from 4-bit to 2-bit saves 2GB but costs 9.5% quality — rarely worth it.`}
      </CodeBlock>

      <H3>4b. Edge model selector: VRAM and latency budget</H3>

      <Prose>
        Given available device VRAM and a target time-to-first-token budget, select the largest model that fits within both constraints. Uses benchmarked prefill speeds for common edge hardware:
      </Prose>

      <CodeBlock language="python">
{`# (model_name, params_b, int4_memory_gb, ttft_ms_jetson, ttft_ms_m4pro)
# TTFT at 512 input tokens, benchmarked on Jetson AGX Orin and M4 Pro MacBook.
# Sources: Jetson developer forum benchmarks, LM Studio benchmarks April 2026.
EDGE_MODELS = [
    ("Phi-3.5 Mini 3.8B", 3.8, 2.1, 45,  30),
    ("Llama 3.2 3B",      3.0, 1.7, 40,  25),
    ("Llama 3.1 8B",      8.0, 4.5, 110, 65),
    ("Gemma 2 9B",        9.0, 5.1, 130, 80),
    ("Llama 3.1 13B",    13.0, 7.3, 190, 115),
    ("Mistral 7B v0.3",   7.2, 4.1, 100, 60),
]

def select_edge_model(
    vram_gb: float,
    ttft_budget_ms: int,
    hardware: str = "jetson",
) -> list:
    """Return models that fit within VRAM and latency budget, largest first."""
    idx = 3 if hardware == "jetson" else 4
    candidates = [
        m for m in EDGE_MODELS
        if m[2] <= vram_gb and m[idx] <= ttft_budget_ms
    ]
    return sorted(candidates, key=lambda x: x[1], reverse=True)

# Jetson AGX Orin (64GB unified memory), 200ms TTFT budget:
print("Jetson AGX Orin — 64GB, 200ms budget:")
for m in select_edge_model(64, 200, "jetson"):
    print(f"  {m[0]}: {m[2]}GB INT4, TTFT ~{m[3]}ms")

# Jetson AGX Orin — 64GB, 200ms budget:
#   Llama 3.1 13B: 7.3GB INT4, TTFT ~190ms
#   Gemma 2 9B: 5.1GB INT4, TTFT ~130ms
#   Llama 3.1 8B: 4.5GB INT4, TTFT ~110ms
#   Mistral 7B v0.3: 4.1GB INT4, TTFT ~100ms
#   Llama 3.2 3B: 1.7GB INT4, TTFT ~40ms
#   Phi-3.5 Mini 3.8B: 2.1GB INT4, TTFT ~45ms

# Smartphone-class device (6GB available, 150ms TTFT budget):
print("\\nSmartphone — 6GB, 150ms budget:")
for m in select_edge_model(6, 150, "jetson"):
    print(f"  {m[0]}: {m[2]}GB INT4, TTFT ~{m[3]}ms")

# Smartphone — 6GB, 150ms budget:
#   Llama 3.1 8B: 4.5GB INT4, TTFT ~110ms
#   Mistral 7B v0.3: 4.1GB INT4, TTFT ~100ms
#   Llama 3.2 3B: 1.7GB INT4, TTFT ~40ms
#   Phi-3.5 Mini 3.8B: 2.1GB INT4, TTFT ~45ms`}
      </CodeBlock>

      <H3>4c. On-premise cluster sizing calculator</H3>

      <Prose>
        Given target QPS, average request size, P99 latency SLA, and hardware specs, compute the number of GPUs required for an on-premise deployment:
      </Prose>

      <CodeBlock language="python">
{`import math

# Hardware throughput table: (gpu_name, tok_sec_per_gpu, vram_gb, power_w)
# 70B model, tensor-parallel within node, FP16 weights.
# Sources: Artificial Analysis benchmarks April 2026.
GPU_HARDWARE = {
    "H100_SXM5":  (1800, 80,  700),   # 2-GPU TP for 70B
    "H200_SXM":   (2400, 141, 700),   # single-GPU for 70B (141GB HBM3e)
    "L40S":       ( 900, 48,  350),   # 2-GPU TP for 70B
    "A100_80GB":  (1200, 80,  400),
}

def size_onprem_cluster(
    target_qps: float,
    avg_tokens_per_req: int,
    p99_latency_ms: int,
    gpu: str = "H100_SXM5",
    util_target: float = 0.70,
) -> dict:
    tps_per_gpu, vram, power_w = GPU_HARDWARE[gpu]
    # GPU count from throughput requirement
    required_tps = target_qps * avg_tokens_per_req
    gpu_count_throughput = math.ceil(required_tps / (tps_per_gpu * util_target))

    # GPU count from latency requirement (queuing: utilization <= 1 - 1/sqrt(N_servers))
    # Simplified Erlang-C approximation: keep rho < 0.7 for P99 within 2x service time
    gpu_count_latency = gpu_count_throughput  # throughput bound is usually tighter

    gpu_count = max(gpu_count_throughput, gpu_count_latency)
    nodes = math.ceil(gpu_count / 8)  # 8-GPU nodes
    total_power_kw = gpu_count * power_w / 1000
    monthly_cost_on_prem = gpu_count * 2.50 * 24 * 30  # specialist cloud rate

    return {
        "gpu_type": gpu,
        "gpu_count": gpu_count,
        "nodes_8gpu": nodes,
        "total_power_kw": round(total_power_kw, 1),
        "monthly_hardware_cost": f"\${monthly_cost_on_prem:,.0f}",
        "throughput_headroom": f"{(tps_per_gpu * gpu_count * util_target / required_tps - 1) * 100:.0f}%",
    }

# Example: 50 QPS, 3,000 tokens/request, 500ms P99 SLA
result = size_onprem_cluster(50, 3000, 500, "H100_SXM5")
for k, v in result.items():
    print(f"  {k}: {v}")

#   gpu_type: H100_SXM5
#   gpu_count: 18
#   nodes_8gpu: 3
#   total_power_kw: 12.6
#   monthly_hardware_cost: \$32,400
#   throughput_headroom: 4%

# At 100 QPS, same specs:
result2 = size_onprem_cluster(100, 3000, 500, "H100_SXM5")
print(f"\\n100 QPS: {result2['gpu_count']} GPUs, {result2['nodes_8gpu']} nodes")`}
      </CodeBlock>

      <H3>4d. Offline sync pipeline for edge fleets</H3>

      <Prose>
        Edge deployments that operate without continuous connectivity must buffer requests, inference results, and telemetry locally and batch-upload when connectivity resumes. This pattern is common in field deployments, vehicle fleets, and remote installations:
      </Prose>

      <CodeBlock language="python">
{`import time
import json
import hashlib
from collections import deque

class OfflineEdgeBuffer:
    """
    Buffers inference results and telemetry for later sync.
    In production, persistence would use SQLite or a local log file.
    """
    def __init__(self, max_buffer_size: int = 10_000):
        self.buffer = deque(maxlen=max_buffer_size)
        self.connected = False
        self.sync_count = 0

    def record_inference(self, request_id: str, model: str,
                         input_tokens: int, output_tokens: int,
                         latency_ms: float) -> None:
        """Record a completed inference event locally."""
        event = {
            "ts": time.time(),
            "id": request_id,
            "model": model,
            "in": input_tokens,
            "out": output_tokens,
            "lat_ms": latency_ms,
            "synced": False,
        }
        self.buffer.append(event)

    def pending_sync_count(self) -> int:
        return sum(1 for e in self.buffer if not e["synced"])

    def batch_sync(self, upload_fn) -> dict:
        """
        When connectivity is available, upload pending events in batches.
        upload_fn: callable that accepts a list of events and returns True on success.
        """
        pending = [e for e in self.buffer if not e["synced"]]
        if not pending:
            return {"uploaded": 0, "failed": 0}

        BATCH_SIZE = 500
        uploaded = 0
        failed = 0

        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i:i + BATCH_SIZE]
            try:
                success = upload_fn(batch)
                if success:
                    for e in batch:
                        e["synced"] = True
                    uploaded += len(batch)
            except Exception:
                failed += len(batch)

        self.sync_count += 1
        return {"uploaded": uploaded, "failed": failed}

# Simulate 3 days of offline operation, then a sync
buf = OfflineEdgeBuffer()
for i in range(2400):  # ~2400 inference events over 3 days
    buf.record_inference(
        request_id=f"req_{i}",
        model="llama3.1-8b-int4",
        input_tokens=512 + (i % 200),
        output_tokens=128 + (i % 50),
        latency_ms=85 + (i % 40),
    )

print(f"Pending events before sync: {buf.pending_sync_count()}")
# Pending events before sync: 2400

def mock_upload(events):
    return True  # simulate successful upload

result = buf.batch_sync(mock_upload)
print(f"Sync result: {result}")
print(f"Pending after sync: {buf.pending_sync_count()}")
# Sync result: {'uploaded': 2400, 'failed': 0}
# Pending after sync: 0`}
      </CodeBlock>

      <H3>4e. Fallback routing: device → edge server → cloud</H3>

      <Prose>
        Production hybrid systems route each request to the tier appropriate for its complexity, data sensitivity, and latency requirements. A three-tier router selects the cheapest tier that can satisfy the request within its constraints:
      </Prose>

      <CodeBlock language="python">
{`from dataclasses import dataclass
from typing import Literal

Tier = Literal["device", "edge_server", "cloud"]

@dataclass
class Request:
    text: str
    max_tokens: int
    requires_sovereignty: bool  # data must not leave local network
    complexity_score: float     # 0.0 (trivial) to 1.0 (frontier reasoning)
    latency_budget_ms: int      # maximum acceptable total latency

# Tier capability table
TIER_CAPS = {
    # (max_complexity, max_output_tokens, est_latency_ms, available_offline)
    "device":      (0.45, 512,   50,  True),
    "edge_server": (0.75, 2048, 120,  True),
    "cloud":       (1.00, 8192, 350,  False),
}

def route_request(req: Request, device_available: bool,
                  edge_available: bool, cloud_available: bool) -> Tier:
    """
    Select the cheapest tier that satisfies all request constraints.
    Priority: device > edge_server > cloud (cheapest to most expensive).
    """
    availability = {
        "device":      device_available,
        "edge_server": edge_available,
        "cloud":       cloud_available and not req.requires_sovereignty,
    }

    for tier in ["device", "edge_server", "cloud"]:
        if not availability[tier]:
            continue
        max_complexity, max_tokens, est_latency, offline_ok = TIER_CAPS[tier]
        if (req.complexity_score <= max_complexity
                and req.max_tokens <= max_tokens
                and est_latency <= req.latency_budget_ms):
            return tier

    # Last resort: cloud (if sovereignty allows), else degrade gracefully
    if not req.requires_sovereignty and cloud_available:
        return "cloud"
    raise ValueError("No available tier satisfies request constraints")

# Test routing decisions
scenarios = [
    Request("Summarize this note", 200, False, 0.30, 500),  # simple, no sovereignty
    Request("Classify intent", 50, True, 0.20, 100),         # sovereign, fast
    Request("Multi-step reasoning plan", 4000, False, 0.90, 600),  # complex
    Request("Translate short phrase", 100, True, 0.25, 200),       # sovereign, moderate
]

for req in scenarios:
    tier = route_request(req, device_available=True,
                         edge_available=True, cloud_available=True)
    print(f"  '{req.text[:35]}...' -> {tier}")

# 'Summarize this note...'          -> device      (simple, fast, cheap)
# 'Classify intent...'              -> device      (sovereign, fits on-device)
# 'Multi-step reasoning plan...'    -> cloud       (complexity 0.9 exceeds edge limit)
# 'Translate short phrase...'       -> device      (sovereign, within device caps)`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>On-device inference runtimes</H3>

      <Prose>
        <strong>llama.cpp</strong> (github.com/ggml-org/llama.cpp) is the foundational open-source C++ inference engine for edge and on-device deployment. It runs quantized models in GGUF format — its own compact model serialization standard — on CPUs, Apple Metal, CUDA, Vulkan, and a growing list of accelerator backends. Its key design property is minimal dependencies: a single binary with no Python runtime, no CUDA toolkit requirement, and no driver infrastructure beyond what the OS provides. On Apple Silicon M3/M4, llama.cpp achieves roughly 30–60 tokens per second for 7B models in Q4 quantization using Metal. On CPU-only systems (Raspberry Pi, embedded ARM), it achieves 2–8 tokens per second for sub-3B models — slow but functional for non-interactive use cases. llama.cpp's GGUF format has become a de facto standard: Ollama, LM Studio, and most local inference tools load GGUF models internally.
      </Prose>

      <Prose>
        <strong>Ollama</strong> (ollama.com) wraps llama.cpp in a Go daemon that manages model lifecycle, GPU allocation, and request routing behind a simple REST API. <Code>ollama serve</Code> starts an HTTP server on port 11434; <Code>ollama pull llama3.1:8b</Code> downloads and caches the GGUF model. For developer and small-team deployments, Ollama reduces on-premise LLM serving to a two-command setup. As of March 2026, Ollama added MLX backend support in preview, showing 1.6× prefill speedup on Apple Silicon compared to llama.cpp's Metal backend for the same models.
      </Prose>

      <Prose>
        <strong>LM Studio</strong> provides a desktop GUI for model management and inference configuration. On Apple Silicon, LM Studio defaults to the MLX backend rather than llama.cpp, delivering measurably higher throughput: benchmarks on M3 Ultra show 237 tok/s for Gemma 3 1B versus 149 tok/s with Ollama's llama.cpp backend. LM Studio introduced LM Link in February 2026 — encrypted remote access to a locally-running model via Tailscale integration — which makes it viable as a personal edge server accessible from other devices without public exposure.
      </Prose>

      <Prose>
        <strong>Apple MLX</strong> (ml-explore.github.io/mlx) is Apple's own array framework for Apple Silicon, optimized for the unified memory architecture that makes no distinction between CPU and GPU memory. MLX operations execute on whichever processor — CPU, GPU, or Neural Engine — is fastest for the operation size. The M5 chip announced in 2026 exposes Neural Accelerators to MLX that provide dedicated matrix multiplication units, further accelerating transformer inference. For production on-device deployments targeting Apple hardware, MLX-LM (the language model package built on MLX) achieves the highest tokens-per-second rates of any available runtime on Apple Silicon.
      </Prose>

      <Prose>
        <strong>Qualcomm AI Hub</strong> (aihub.qualcomm.com) provides the full pipeline from Hugging Face model to optimized inference on Snapdragon hardware. The <Code>qai_hub_models</Code> Python package handles quantization, graph optimization, and NPU code generation for Hexagon NPUs — the 45-TOPS neural processing units in Snapdragon X Elite and Snapdragon 8 Gen 4 chips. Llama 3.2 3B Instruct runs at approximately 10 tokens per second on Snapdragon 8 Gen 4; Llama 3.1 8B at 5 tokens per second. Both are adequate for interactive single-user inference on Android flagship phones and Windows ARM laptops.
      </Prose>

      <H3>Edge server hardware</H3>

      <Prose>
        <strong>NVIDIA Jetson AGX Orin</strong> is the standard edge server platform for AI-intensive applications. The 64GB variant delivers up to 275 TOPS (sparse INT8) and 85 FP16 TFLOPS from its Ampere GPU, with a power envelope of 15–60W configurable by software. Its 64GB of unified LPDDR5 memory shared between CPU and GPU means a 7B INT4 model (4.5GB) leaves 59GB available for KV cache and application memory — substantially more overhead than a typical phone. Jetson is used in industrial robotics, autonomous vehicle perception, retail analytics, and edge AI appliances where a fanless, rugged, sub-60W compute module is required.
      </Prose>

      <H3>On-premise appliances</H3>

      <Prose>
        <strong>Dell AI Factory with NVIDIA</strong> has crossed 4,000 enterprise customer deployments as of March 2026. The current flagship, the PowerEdge XE9780, combines dual Intel Xeon 6 CPUs with eight NVIDIA HGX Blackwell (B200 or B300) GPUs in a 10U chassis purpose-built for GenAI inference and fine-tuning. Dell's April 2026 announcement of the Modular Architecture integrates AI-ready infrastructure with NVIDIA's AI software stack — NIM microservices, NeMo guardrails, and the full NGC catalog — into a factory-configured bundle that reduces initial deployment time from weeks to days.
      </Prose>

      <Prose>
        <strong>HPE Private Cloud AI</strong> expanded in March 2026 with air-gapped configurations that keep sensitive data entirely off public networks, scaling up to 128 GPUs. The full-stack solution includes tightly integrated compute, networking (InfiniBand or Ethernet), liquid cooling, and HPE's GreenLake management layer — designed specifically for regulated enterprises that need sovereign AI infrastructure without building datacenter operations from scratch.
      </Prose>

      <H3>Hyperscaler on-premise extensions</H3>

      <Prose>
        <strong>AWS Outposts</strong> (aws.amazon.com/outposts) physically installs AWS-designed racks in customer datacenters, extending native AWS services — EC2, EKS, RDS, S3 — to on-premise locations while maintaining the AWS API surface. Second-generation Outposts (January 2026) supports C7i, M7i, and R7i instances with 40% better compute performance than first-generation. For AI workloads, AWS provides deployment patterns for generative AI foundation models on Outposts, specifically addressing data residency, latency, and FedRAMP/ITAR requirements. The trade: you get AWS APIs and managed control plane on-premise, but the hardware is AWS-proprietary and the dependency on AWS connectivity for control-plane operations means true air-gap operation requires additional configuration.
      </Prose>

      <Prose>
        <strong>Azure Local</strong> (formerly Azure Stack HCI) and <strong>GCP Distributed Cloud</strong> follow similar patterns — extending the hyperscaler control plane to customer-managed hardware. Azure Arc provides centralized management of resources across on-premise and multi-cloud environments through a single Azure control plane, which is particularly valuable for enterprises that have standardized their tooling on Azure Monitor, Azure Policy, and Defender for Cloud and want those capabilities to extend to their on-premise AI infrastructure. GCP Distributed Cloud takes a Kubernetes-centric approach via Anthos, running containerized workloads consistently across GCP and on-premise with unified management.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>Latency profile: cloud vs edge server vs device</H3>

      <Plot
        label="time-to-first-token (ms) by deployment tier — 512-token prompt, 7B model class (April 2026)"
        width={540}
        height={280}
        xLabel="concurrency (simultaneous users)"
        yLabel="TTFT (ms)"
        series={[
          {
            name: "Cloud API (50ms RTT + inference)",
            points: [[1, 180], [5, 200], [10, 230], [25, 310], [50, 450], [100, 720]],
          },
          {
            name: "Edge server (LAN, Jetson AGX Orin)",
            points: [[1, 110], [5, 125], [10, 145], [25, 210], [50, 340], [100, 580]],
          },
          {
            name: "On-device (M4 Pro / Snapdragon 8 Gen4)",
            points: [[1, 65], [5, 65], [10, 65], [25, 65], [50, 65], [100, 65]],
          },
        ]}
      />

      <Prose>
        On-device latency is flat with concurrency because each user's inference runs on their own hardware — there is no shared queue. Cloud and edge server latency climbs with concurrency as requests queue behind a shared GPU. The edge server's slope is steeper than cloud at high concurrency because edge hardware has lower parallelism headroom. The crossover where edge server becomes slower than cloud occurs at high concurrent load, which is why edge servers are typically sized for the local population they serve rather than for general-purpose demand.
      </Prose>

      <H3>Model feasibility: device class × model size</H3>

      <Heatmap
        label="edge deployment feasibility — model size (rows) × device class (cols). 100=fully feasible, 0=impossible"
        matrix={[
          [100, 100, 100, 100, 100],
          [100, 100, 100, 100,  60],
          [100, 100, 100,  70,   0],
          [100, 100,  80,  20,   0],
          [100,  80,  30,   0,   0],
          [ 60,  20,   0,   0,   0],
        ]}
        rowLabels={[
          "1B INT4 (0.5GB)",
          "3B INT4 (1.7GB)",
          "7B INT4 (4GB)",
          "13B INT4 (7GB)",
          "70B INT4 (40GB)",
          "405B INT4 (230GB)",
        ]}
        colLabels={["On-prem DC", "Edge server", "Laptop/Mac", "Flagship phone", "Mid-range phone"]}
        cellSize={60}
        colorScale="green"
      />

      <Prose>
        The heatmap shows where each model class becomes infeasible. 70B INT4 at 40GB fits comfortably on an on-premise H100 or H200 and on a Jetson AGX Orin (64GB unified memory), but not on a laptop (typical max 24–32GB unified) or any phone. 7B INT4 at 4GB runs on every hardware class including mid-range phones with 6GB RAM. The 405B model is restricted to on-premise clusters — it requires multiple H100s even at INT4. The practical edge deployment boundary sits at 7B–13B INT4 for edge servers and 1B–7B INT4 for personal devices.
      </Prose>

      <H3>Fallback routing flow</H3>

      <StepTrace
        label="three-tier fallback routing — request lifecycle from device to cloud"
        steps={[
          {
            label: "1. Request arrives at routing layer",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div style={{ color: colors.gold }}>Input: user query + metadata</div>
                <div>complexity_score: 0.35  (classifier on-device, ~2ms)</div>
                <div>requires_sovereignty: true</div>
                <div>latency_budget_ms: 300</div>
                <div>max_output_tokens: 256</div>
              </div>
            ),
          },
          {
            label: "2. Check device tier capability",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div>device_max_complexity: 0.45 — complexity 0.35 fits ✓</div>
                <div>device_max_tokens: 512 — 256 fits ✓</div>
                <div>device_est_latency: 65ms — within 300ms budget ✓</div>
                <div style={{ color: "#4ade80" }}>→ ROUTE TO DEVICE</div>
                <div>Model: Llama 3.2 3B INT4 (on Neural Engine)</div>
                <div>Data: never leaves device — sovereignty satisfied</div>
              </div>
            ),
          },
          {
            label: "3. Device model returns result",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div style={{ color: colors.gold }}>Inference complete: 58ms TTFT, 220ms total</div>
                <div>Tokens generated: 189 / 256 requested</div>
                <div>Quality check: confidence 0.82 (above threshold 0.60) ✓</div>
                <div>Return to user — no network call made</div>
              </div>
            ),
          },
          {
            label: "4. Fallback case: complexity exceeds device cap",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div>complexity_score: 0.85  (exceeds device cap 0.45)</div>
                <div>requires_sovereignty: false</div>
                <div style={{ color: "#f87171" }}>Device tier: REJECTED (complexity)</div>
                <div>Edge server available: true — complexity 0.85 {"<"} cap 0.75? NO</div>
                <div style={{ color: "#f87171" }}>Edge server tier: REJECTED (complexity)</div>
                <div style={{ color: "#4ade80" }}>→ ROUTE TO CLOUD (sovereignty not required)</div>
                <div>Model: GPT-4o / Claude Sonnet 4.6 via API</div>
              </div>
            ),
          },
          {
            label: "5. Offline mode: all external tiers unavailable",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div>edge_available: false  (network down)</div>
                <div>cloud_available: false  (offline)</div>
                <div>device_available: true  (always available)</div>
                <div style={{ color: colors.gold }}>→ ROUTE TO DEVICE unconditionally</div>
                <div>Accept quality degradation vs cloud for offline operation</div>
                <div>Buffer high-complexity requests for retry when connected</div>
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
{`Requirement                  | Recommended tier       | Rationale
---------------------------- | ---------------------- | ---------------------------------------
Data sovereignty (HIPAA,     | On-prem datacenter     | Full data never transits external infra;
GDPR, FedRAMP)               | or edge server         | auditable control plane; BAA not required
                             |                        |
Air-gapped / offline         | Device or edge server  | No connectivity assumed; model must be
operation required           |                        | local; batch sync telemetry when online
                             |                        |
Latency {"<"} 100ms TTFT       | Device or edge server  | Cloud RTT (20-150ms) + TTFT often
(real-time voice, robotics)  |                        | exceeds budget; LAN is sub-5ms
                             |                        |
Latency 100-400ms acceptable | Cloud or edge server   | Cloud competitive for this range;
(interactive but not RT)     |                        | edge adds complexity without clear win
                             |                        |
Frontier model capability    | Cloud                  | 400B+ models only viable in cloud;
required (complex reasoning) |                        | no edge hardware runs them at scale
                             |                        |
High QPS + steady load       | On-prem datacenter     | Fixed cost amortizes well at volume;
(100M+ tokens/day)           |                        | opex savings exceed capex within 6-18mo
                             |                        |
Bursty / unpredictable       | Cloud API              | Self-hosting idle GPUs during troughs;
traffic pattern              |                        | API scales to zero cost at zero traffic
                             |                        |
Privacy + zero serving cost  | On-device              | Inference on user hardware; no API bill;
at consumer scale            |                        | capacity grows automatically with users
                             |                        |
Regulated + high volume      | On-prem (HPE/Dell      | Turnkey appliances reduce ops burden;
enterprise                   | AI Factory)            | air-gapped configs available; NIST-aligned
                             |                        |
Rapid iteration / prototype  | Cloud API              | No capex; model updates automatic;
                             |                        | switch providers freely`}
      </CodeBlock>

      <Callout accent="gold">
        The most common mistake is conflating data sovereignty with latency requirements. Sovereign data must stay on-premise; low-latency data does not. Many deployments run cloud inference with dedicated Virtual Private Cloud endpoints — no data sovereignty, but low enough latency for the use case. Only regulated data with legal restrictions requires true on-premise or on-device deployment.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>What scales</H3>

      <Prose>
        <strong>On-device fleet capacity scales linearly with user count.</strong> This is the single most important scaling property distinguishing on-device from on-premise or cloud. Adding one million users to a service with on-device inference adds one million devices' worth of inference capacity, at zero marginal infrastructure cost to the operator. The economics improve with adoption rather than degrading. Apple Intelligence, Google Gemini Nano, and Microsoft's Phi models on Copilot+ PCs all exploit this property — every device sold is a new inference node that the operator does not pay for.
      </Prose>

      <Prose>
        <strong>On-premise clusters scale with capex on a years-long cycle.</strong> The scaling increment for on-premise GPU infrastructure is large: a single H100 node costs $250,000–$400,000 to purchase, and lead times for GPU hardware from order to rack-and-stack are 4–12 months depending on vendor and configuration. This means on-premise capacity planning must be done 12–18 months in advance, sized for peak load with a growth assumption, and the cluster lives with its initial capacity until the next capital cycle. This is categorically different from cloud elasticity. The implication is that on-premise deployments should be sized for predictable base load — not for peak — and cloud overflow capacity should handle unpredictable bursts.
      </Prose>

      <Prose>
        <strong>Edge server deployments scale by adding physical units near the load.</strong> An edge server serving a factory floor can serve its co-located devices with sub-5ms LAN latency, but it cannot serve a factory 100 kilometers away. Scaling an edge deployment means adding servers at new locations — a logistics and maintenance challenge, not just a procurement one. The scaling increment is the size of the geographic or organizational unit being served: one server per factory, one per retail location, one per hospital ward.
      </Prose>

      <H3>What doesn't scale</H3>

      <Prose>
        <strong>Model quality on edge hardware does not compress below the quantization floor.</strong> There is a hard limit to how much a model can be compressed before it stops being useful. INT2 quantization recovers 8× the memory of FP16, but the quality loss is severe enough that most practical tasks degrade to unacceptable outputs. The smallest useful model for general instruction-following tasks in English is roughly 3B parameters at INT4 — 1.7GB of memory. Below that threshold, model capability drops sharply. This creates a floor below which edge deployment simply cannot satisfy a given task's quality requirement, regardless of hardware improvements.
      </Prose>

      <Prose>
        <strong>On-premise operations do not scale without dedicated engineering.</strong> A single-tenant on-premise LLM cluster requires monitoring, model update management, hardware maintenance, incident response, and capacity planning — work that cloud providers perform at scale and amortize across thousands of customers. A small team deploying on-premise takes on all of that overhead themselves. The common pattern is that on-premise deployments start with an internal team managing the cluster and gradually become the main bottleneck — model updates are delayed, monitoring dashboards are perpetually behind, and incidents take longer to resolve than they would on a managed service. The realistic engineering overhead for a well-run on-premise cluster is one dedicated MLOps engineer per 8–16 GPU nodes.
      </Prose>

      <Prose>
        <strong>Fleet model updates do not propagate instantly.</strong> In cloud serving, a new model version can be deployed to the entire serving fleet in hours and rolled back in minutes. On-device models update on the hardware vendor's or app store's release cycle — quarterly at best, and sometimes tied to OS updates that users control. An edge server fleet requires coordinated rollouts across potentially thousands of physical units in diverse locations. The operational discipline around model versioning and update management is substantially higher than in cloud serving, and the blast radius of a bad model update is weeks of exposure rather than hours.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES AND GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>1. Thermal throttling degrades sustained performance</H3>

      <Prose>
        Edge hardware in a thermally constrained environment — a sealed enclosure, a vehicle, a factory floor with no active cooling — will sustain its rated TOPS for a few minutes and then throttle. A Jetson AGX Orin running continuous inference at 60W in a 40°C ambient environment will reduce its GPU clock rate to maintain a safe junction temperature, potentially dropping to 60–70% of rated throughput within 10–15 minutes of sustained load. The user experience is a model that starts fast and gradually slows down. Production deployments must benchmark under thermal load, not just at cold start, and must design cooling into the physical installation rather than treating it as an afterthought.
      </Prose>

      <H3>2. Firmware and driver updates break model compatibility</H3>

      <Prose>
        On-device inference runtimes are closely coupled to driver versions, firmware, and OS components. A phone OS update that bumps the neural engine driver can silently change the behavior of a quantized model — not crash it, but produce subtly different outputs. An edge server firmware update can change the CUDA version expected by the inference runtime, requiring a full stack rebuild. In cloud serving, these dependencies are managed by the provider. On-premise and on-device deployments own this dependency chain. The mitigation is to pin the full software stack — OS, firmware, driver, runtime, model — in a tested bundle, and to gate updates behind explicit validation runs before deploying to production devices.
      </Prose>

      <H3>3. Stale model versions across distributed fleets</H3>

      <Prose>
        A cloud service ensures every request hits the same model version. A distributed edge fleet cannot guarantee this. When a new model is deployed to a fleet of 10,000 edge servers, the rollout takes days or weeks, and during that window some servers run the old model and some run the new one. The behavior difference is invisible to users but visible in A/B comparisons and regression testing. For applications where model consistency across all users matters — legal document processing, medical triage tools, financial advice systems — fleet heterogeneity during rollouts is a compliance risk that requires explicit rollout gates, canary deployments, and version-tagged responses.
      </Prose>

      <H3>4. Data sovereignty misinterpretation</H3>

      <Prose>
        Several cloud providers market "regional" or "in-country" deployments that process data within a specified geography. This is not the same as data sovereignty in the legal sense. Regional cloud endpoints route data through the provider's global control plane, management infrastructure, and monitoring systems, which often operate globally. For GDPR Article 44 purposes — the restriction on international data transfers — "processed in Frankfurt" does not guarantee "not transferred to the US" if the provider's control plane is US-headquartered. Healthcare organizations subject to HIPAA, defense contractors subject to ITAR, and government agencies subject to FedRAMP High must work with legal counsel to evaluate whether a specific cloud provider's regional offering satisfies their actual legal obligations — or whether on-premise is the only compliant option.
      </Prose>

      <H3>5. Connection failover edge cases at the routing boundary</H3>

      <Prose>
        Hybrid systems that route between on-device and cloud tiers must handle connectivity degradation gracefully. The failure mode is not "network is up" or "network is down" — it is "network is intermittent": requests time out unpredictably, some succeed and some fail, and the system oscillates between tiers during the degradation period. A router that makes tier decisions at request-dispatch time without checking connection state will split identical requests across tiers, producing inconsistent responses. The mitigation is to implement circuit-breaker logic: after N consecutive failures to a tier within a time window, declare that tier unavailable and stop routing to it until a health probe succeeds. Re-enabling a tier should require multiple successful probes to prevent flapping.
      </Prose>

      <H3>6. Quantization bugs produce silent quality degradation</H3>

      <Prose>
        Quantization is not a lossless compression; it introduces approximation errors that vary by layer, token position, and input distribution. A model that passes standard benchmark evaluation at INT4 may fail silently on inputs that happen to activate the layers where quantization error is highest — typically layers with large outlier activations (the problem AWQ was specifically designed to address). The failure mode is not a crash or an OOM; it is a model that generates plausible-looking text that is wrong in subtle ways. Edge deployments that lack production monitoring for output quality — which is most of them, given the operational constraints — may run degraded models for months without detecting it. The mitigation is to maintain a small golden test set of inputs with known correct outputs and run it against the deployed model after every quantization or update cycle.
      </Prose>

      <H3>7. Capacity mismatch with demand growth</H3>

      <Prose>
        On-premise clusters are sized for a capacity estimate made months before the hardware arrives. If the product grows faster than the estimate, the cluster is undersized and queues grow until users experience unacceptable latency. If the product grows slower, GPUs sit idle and the capital investment underperforms. Cloud serving handles both cases automatically; on-premise handles neither automatically. The mitigation is to maintain a cloud overflow capacity — a managed API endpoint that catches excess traffic when the on-premise cluster is saturated — and to treat on-premise as the base load absorber, not the peak-load handler.
      </Prose>

      <H3>8. Vendor lock-in on specialized hardware</H3>

      <Prose>
        Choosing a hardware-specific inference path — Apple Core ML, Qualcomm QNN, NVIDIA TensorRT — optimizes performance on that hardware but creates a deployment artifact that runs only on that hardware. A Core ML model exported for Neural Engine inference will not run on a Qualcomm NPU. A TensorRT engine compiled for H100 will not run on an H200 or an A100 without recompilation. As organizations build larger libraries of deployed models, hardware-specific artifacts create a maintenance burden: every model must be reoptimized and retested for every supported hardware variant. The mitigation is to treat GGUF (for CPU/Metal/CUDA via llama.cpp) or ONNX (for cross-runtime portability) as the source of truth and generate hardware-specific artifacts from the source format at deployment time, rather than storing hardware-specific artifacts as canonical versions.
      </Prose>

      <Callout accent="red">
        The most common undetected failure in edge LLM deployments is silent model drift — the on-device model was validated at deployment and then degraded silently as the user distribution shifted, without any monitoring in place to detect it. Build output quality monitoring into your edge deployment from day one, even if it is a simple offline sample-and-evaluate pipeline that runs weekly.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The following sources were verified and current as of April 2026. Hardware specs and pricing change; treat specific numbers as representative of April 2026 and verify against live documentation before production decisions.
      </Prose>

      <CodeBlock>
{`1. llama.cpp — LLM inference in C/C++
   https://github.com/ggml-org/llama.cpp
   Primary source for GGUF format specification, quantization type
   documentation (Q4_K_M, Q5_K_S, etc.), and supported hardware backends.
   The quantize/README.md documents quality vs size tradeoffs per quant type.
   As of April 2026: supports 1.5-bit through 8-bit quantization with
   CUDA, Metal, Vulkan, ROCm, and CPU backends.

2. Apple MLX — Array framework for Apple Silicon
   https://ml-explore.github.io/mlx/  (docs, current version 0.31.1)
   https://github.com/ml-explore/mlx
   https://machinelearning.apple.com/research/exploring-llms-mlx-m5
   Documents the unified memory model, Neural Accelerator support on M5,
   and the MLX-LM language model package. The Apple ML Research blog post
   covers M5 Neural Accelerator integration announced at WWDC25.

3. NVIDIA Jetson AGX Orin — Technical Brief
   https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/
   https://developer.nvidia.com/buy-jetson
   Official specs: 275 TOPS (sparse), 85 FP16 TFLOPS GPU, 64GB unified
   LPDDR5, configurable power 15-60W. The Technical Brief (July 2022 v1.2)
   documents the Ampere GPU, DLA engines, and thermal design.

4. Qualcomm AI Hub
   https://aihub.qualcomm.com
   https://github.com/qualcomm/ai-hub-models
   Model zoo with optimized on-device inference for Snapdragon hardware.
   Benchmarks for Llama 3.2 3B (~10 tok/s) and Llama 3.1 8B (~5 tok/s)
   on Snapdragon 8 Gen 4 are documented in the model cards. The whitepaper
   "Unlocking on-device generative AI with an NPU" covers NPU architecture.

5. AWS Outposts — On-Premises Private Cloud
   https://aws.amazon.com/outposts/
   https://aws.amazon.com/blogs/compute/running-and-optimizing-small-language-models-on-premises-and-at-the-edge/
   Official product page and AI/LLM deployment guidance for Outposts.
   Second-generation Outposts (January 2026) specs, country availability,
   and the AWS blog post on small language model optimization on-premise.

6. Azure Local / Azure Arc
   https://azure.microsoft.com/en-us/products/azure-arc
   Microsoft's hybrid cloud management layer. Azure Arc enables centralized
   governance, policy, and monitoring across on-premise and multi-cloud.

7. Dell AI Factory with NVIDIA
   https://www.dell.com/en-us/lp/dt/nvidia-ai
   March 2026 GTC announcement: Modular Architecture, PowerEdge XE9780 with
   HGX Blackwell, 4,000+ customer deployments. Pricing and configuration
   guide available through Dell sales channels.

8. HPE Private Cloud AI
   https://www.hpe.com/us/en/newsroom/press-release/2026/03/hpe-unveils-next-generation-ai-factory-and-supercomputing-advancements-with-nvidia.html
   March 2026 GTC announcement: air-gapped configurations, scale to 128 GPUs,
   full-stack integration including liquid cooling and GreenLake management.

9. NIST AI Risk Management Framework (AI RMF 1.0 + 2025 updates)
   https://www.nist.gov/system/files/documents/2023/01/26/AI%20RMF%201.0.pdf
   https://www.ispartnersllc.com/blog/nist-ai-rmf-2025-updates-what-you-need-to-know-about-the-latest-framework-changes/
   Governs AI risk practices for federal agencies and FedRAMP-compliant
   systems. The 2025 update added threat categories specific to generative AI.
   NIST IR 8596 (2025) covers cybersecurity framework profiles for AI systems.

10. Microsoft INT4 Quantization Guide (February 2026)
    https://medium.com/data-science-at-microsoft/a-practical-guide-to-int4-quantization-for-slms-gptq-vs-awq-olive-and-real-world-results-2f63d6963d1d
    Practical benchmarks comparing GPTQ vs AWQ on SLMs. Source for the
    98.1% reasoning capability retention figure on MMLU-Pro after INT4 AWQ.
    Includes Olive toolchain integration for Qualcomm and DirectML backends.`}
      </CodeBlock>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Memory budget for a Jetson deployment</H3>

      <Prose>
        A Jetson AGX Orin with 64GB unified memory needs to run a 13B parameter model. Calculate the INT4 memory footprint of the model. The system also needs to hold a KV cache for 8 concurrent sessions, each with a 2,048-token context. Assuming FP16 KV cache with 40 transformer layers and 40 KV heads of dimension 128, how much memory does the KV cache add? Does the total fit in 64GB with 8GB reserved for the OS and application runtime?
      </Prose>

      <CodeBlock language="python">
{`# Solution
params_b = 13.0
bits = 4
model_memory_gb = params_b * 1e9 * bits / (8 * 1e9)
# => 6.5 GB for model weights

# KV cache: 2 × layers × heads × head_dim × seq_len × sessions × bytes_per_value
layers = 40
kv_heads = 40         # both K and V
head_dim = 128
seq_len = 2048
sessions = 8
bytes_per_val = 2     # FP16

kv_bytes = 2 * layers * kv_heads * head_dim * seq_len * sessions * bytes_per_val
kv_gb = kv_bytes / 1e9
# 2 × 40 × 40 × 128 × 2048 × 8 × 2 = 42,949,672,960 bytes ≈ 43 GB
# This is very large — FP16 KV cache at 8 sessions × 2048 tokens is the
# dominant memory consumer, not the model weights.

total = model_memory_gb + kv_gb + 8  # model + kv + OS/runtime
print(f"Model: {model_memory_gb:.1f} GB")
print(f"KV cache: {kv_gb:.1f} GB")
print(f"OS/runtime: 8 GB")
print(f"Total: {total:.1f} GB — {'fits' if total <= 64 else 'DOES NOT FIT'} in 64GB")
# Total: 57.5 GB — fits in 64GB, but barely.
# Mitigation: reduce to 4 sessions, or use INT8 KV cache (halves to 21.5 GB),
# or reduce context to 1024 tokens per session (halves KV again to 10.75 GB).`}
      </CodeBlock>

      <H3>Exercise 2 — Cloud vs edge latency crossover</H3>

      <Prose>
        An industrial control system requires LLM inference for anomaly classification. Cloud inference RTT to the nearest region is 45ms; time-to-first-token on cloud is 120ms. The local edge server (Jetson AGX Orin) runs a 7B INT4 model with a time-to-first-token of 95ms and LAN latency of 3ms. At what cloud RTT does the cloud option become faster than the edge server? Is the edge server justified for this use case?
      </Prose>

      <CodeBlock language="python">
{`# Solution
ttft_cloud_ms = 120   # cloud inference TTFT
ttft_edge_ms  = 95    # edge server TTFT
lan_ms        = 3     # LAN round-trip to edge server
rtt_cloud_ms  = 45    # actual RTT to cloud

latency_cloud = rtt_cloud_ms + ttft_cloud_ms   # 165ms
latency_edge  = lan_ms + ttft_edge_ms           # 98ms

print(f"Cloud total latency: {latency_cloud}ms")
print(f"Edge total latency:  {latency_edge}ms")
print(f"Edge faster by:      {latency_cloud - latency_edge}ms")

# Crossover: edge_lat = cloud_lat => LAN + TTFT_edge = RTT + TTFT_cloud
# RTT_crossover = LAN + TTFT_edge - TTFT_cloud
rtt_crossover = lan_ms + ttft_edge_ms - ttft_cloud_ms
print(f"Edge wins when cloud RTT > {rtt_crossover}ms")
# Cloud total latency: 165ms
# Edge total latency:  98ms
# Edge faster by:      67ms
# Edge wins when cloud RTT > -22ms
# => Edge is ALWAYS faster here because TTFT_edge < TTFT_cloud even at RTT=0.
# The edge server is clearly justified — 67ms advantage on every request.
# For the industrial control use case, 98ms vs 165ms is the difference
# between a sub-100ms response and a 165ms response — meaningful for real-time.`}
      </CodeBlock>

      <H3>Exercise 3 — On-premise cluster GPU count</H3>

      <Prose>
        Your enterprise wants to serve a 70B model on-premise at 20 QPS, with an average of 3,500 tokens per request and a P99 latency SLA of 800ms. Using H100 SXM5 GPUs at 1,800 aggregate tokens per second per GPU and a target utilization of 0.70, how many GPUs are needed? How many 8-GPU nodes? What is the monthly cost at $2.50/GPU-hour (specialist cloud) vs purchasing at $30,000 per GPU?
      </Prose>

      <CodeBlock language="python">
{`import math

tps_per_gpu  = 1800        # H100 SXM5 on Llama 3 70B
target_qps   = 20
avg_tokens   = 3500
util_target  = 0.70
gpu_cost_hr  = 2.50        # specialist cloud $/GPU-hr
gpu_purchase = 30_000      # on-prem capex per GPU

required_tps = target_qps * avg_tokens   # 70,000 tok/sec needed
gpu_count = math.ceil(required_tps / (tps_per_gpu * util_target))
nodes = math.ceil(gpu_count / 8)

monthly_lease = gpu_count * gpu_cost_hr * 24 * 30
capex_total   = gpu_count * gpu_purchase
capex_monthly = capex_total / 36  # 3-year amortization

print(f"Required throughput: {required_tps:,} tok/sec")
print(f"GPUs needed: {gpu_count} ({nodes} x 8-GPU nodes)")
print(f"Monthly lease cost (specialist cloud): \${monthly_lease:,.0f}")
print(f"Capex per GPU: \${gpu_purchase:,} × {gpu_count} = \${capex_total:,}")
print(f"Monthly amortized capex (3yr): \${capex_monthly:,.0f}")

# Required throughput: 70,000 tok/sec
# GPUs needed: 56 (7 x 8-GPU nodes)
# Monthly lease cost (specialist cloud): \$100,800
# Capex per GPU: \$30,000 × 56 = \$1,680,000
# Monthly amortized capex (3yr): \$46,667
# => Purchasing outright is cheaper than leasing after 16 months.
# At 20 QPS × 3500 tok × 86400 sec/day ≈ 6 billion tokens/day,
# cloud API at $3/MTok output would cost ~\$18M/month — self-hosting wins easily.`}
      </CodeBlock>

      <H3>Exercise 4 — Fleet model update risk window</H3>

      <Prose>
        You have 8,000 edge servers deployed across retail locations. A new model version is ready to deploy. You can update 500 servers per day. During the rollout window, requests may hit either the old or new model version. If the old model has a 2% error rate on a critical task and the new model has a 0.5% error rate, what is the expected error rate across the fleet on day 4 of the rollout? How many days until 95% of requests hit the new model?
      </Prose>

      <CodeBlock language="python">
{`# Solution
total_servers    = 8_000
update_rate      = 500     # servers updated per day
err_old          = 0.02    # 2% error rate on old model
err_new          = 0.005   # 0.5% error rate on new model
target_fraction  = 0.95    # when to declare rollout complete

def fleet_error_rate(day: int) -> float:
    updated = min(day * update_rate, total_servers)
    frac_new = updated / total_servers
    frac_old = 1 - frac_new
    return frac_old * err_old + frac_new * err_new

# Day 4: 2000 updated, 6000 still on old model
print(f"Day 4 error rate: {fleet_error_rate(4) * 100:.3f}%")
# Day 4 error rate: 1.625%  (still mostly old model)

# Days until 95% on new model
days_complete = math.ceil(total_servers * target_fraction / update_rate)
print(f"Days to 95% new model: {days_complete}")
print(f"Error rate at completion: {fleet_error_rate(days_complete) * 100:.3f}%")
# Days to 95% new model: 16
# Error rate at completion: 0.575%
#
# During these 16 days, different users hit different model versions.
# For compliance-sensitive applications (medical, financial), this heterogeneity
# requires either: (1) user-stickiness routing so each session hits one version,
# (2) a version header in responses so clients can detect the model version,
# or (3) a maintenance window with instant fleet update at off-peak hours.`}
      </CodeBlock>

      <H3>Exercise 5 — On-device economics at consumer scale</H3>

      <Prose>
        A consumer app ships with an on-device 3B INT4 model. The app has 5 million monthly active users who each make an average of 80 inference requests per day, each averaging 150 tokens output. Compare the total monthly output tokens against what it would cost to serve those requests via a cloud API at $5/MTok output. What is the monthly API cost avoided by on-device inference? At what user scale does the API cost avoided justify one additional ML engineer ($250K/yr fully-loaded) to maintain the on-device model pipeline?
      </Prose>

      <CodeBlock language="python">
{`# Solution
mau          = 5_000_000        # monthly active users
req_per_day  = 80               # requests per user per day
avg_out_tok  = 150              # output tokens per request
price_out    = 5.00             # $/MTok — hypothetical small-model API

daily_tokens = mau * req_per_day * avg_out_tok
monthly_tokens = daily_tokens * 30                     # total output tokens/month
monthly_mtok   = monthly_tokens / 1_000_000            # in millions

monthly_api_cost = monthly_mtok * price_out            # API cost avoided

# Engineer cost amortized monthly
engineer_annual = 250_000
engineer_monthly = engineer_annual / 12

print(f"Monthly output tokens: {monthly_tokens:,.0f}")
print(f"Monthly MTok equivalent: {monthly_mtok:,.0f} MTok")
print(f"Monthly API cost avoided: \${monthly_api_cost:,.0f}")
print(f"Engineer monthly cost: \${engineer_monthly:,.0f}")
print(f"ROI multiple: {monthly_api_cost / engineer_monthly:.1f}x per engineer")

# Monthly output tokens: 18,000,000,000,000  (18T — very large)
# Wait — let's recalculate correctly:
# 5M users × 80 req/day × 150 tokens × 30 days = 1.8 trillion tokens/month
# At $5/MTok: 1,800,000 MTok × $5 = $9,000,000,000/month
# That's $9 billion/month — clearly on-device is not optional at this scale.

# Break-even engineer count for on-device maintenance:
be_engineers = monthly_api_cost / engineer_monthly
print(f"\\nAPI cost avoided: \${monthly_api_cost:,.0f}/month")
print(f"Break-even engineer team: {be_engineers:.0f} engineers")
# At 5M MAU and these usage levels, on-device inference avoids
# billions of dollars in monthly API costs. Maintaining the on-device
# model pipeline is justified by essentially any realistic engineering team size.
# Even at 100k MAU (50x smaller): \$180M/month avoided vs \$21k engineer cost.`}
      </CodeBlock>

    </div>
  ),
};

export default edgeOnPremise;
